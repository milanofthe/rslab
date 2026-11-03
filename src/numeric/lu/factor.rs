//! The numeric LU factorization: scaling, the permuted input, the
//! left-looking driver over the assembly forest, and the emit of each
//! finished panel into `L` and `U^T`.

use super::delay::NodePivots;
use super::factors::LuNumeric;
use super::node::{lu_ll_factor_node, LuPools};
use super::solver::LuSymbolic;
use super::structure::LuStructure;
use crate::numeric::supernodal::{Input, InputProgram};

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::settings::{SolverSettings, ZeroPivotAction};
use crate::numeric::supernodal::panel::{PanelArena, PanelFactor, PanelOut, PanelStorage};
use crate::numeric::supernodal::{emit_refcount_offsets, Cells, LlSchedule};
use crate::refine::PivotFloor;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::AtomicUsize;

/// Fold the MC64 row matching back into the factors: pivot rows, the pivot
/// floors and the row scaling are reported in `A`'s row indices, as the
/// solves expect.
fn finish_matching<T: Scalar>(fac: &mut LuNumeric<T>, lusym: &LuSymbolic) {
    if let Some(m) = &lusym.matching {
        for e in fac.perm_row.iter_mut() {
            *e = m.row_of[*e];
        }
        for f in fac.floors.iter_mut() {
            f.row = m.row_of[f.row];
        }
        fac.d_row = m.r.clone();
    }
}

pub(super) struct LlEmit<T> {
    /// Number of consumers (ancestors that pull) still to come; freed at 0.
    pub(super) refcount: Vec<AtomicUsize>,
    /// First elimination position of each supernode without delayed pivots
    /// (symbolic prefix sum of ncol).
    pub(super) e_offset: Vec<usize>,
    /// The factors' buffers: every supernode factors `L` straight into its
    /// slot of `l_arena`; `U^T` is gathered into `u_arena` at the emit.
    pub(super) l_arena: PanelArena<T>,
    pub(super) u_arena: PanelArena<T>,
    pub(super) panels: Cells<(PanelOut, PanelOut)>,
    /// What each factored supernode leaves (its pivots, those it delays).
    pub(super) pivots: Cells<NodePivots>,
    /// Whether pivots may be delayed (`delay`): the positions are then
    /// known, and the supernodes emitted, only once the whole forest is
    /// factored.
    pub(super) delays: bool,
    /// The values the fronts that take delayed pivots may add over their
    /// slots, all together; what they added so far; whether one would have
    /// passed it (the factorization then stops).
    pub(super) allowance: usize,
    pub(super) growth: AtomicUsize,
    /// The values of the slots, the reference of the growth.
    pub(super) slot_values: usize,
    pub(super) overflowed: std::sync::atomic::AtomicBool,
    /// `e_of_g[g]` = elimination position of COLUMN g; `row_pos_of_g[g]` =
    /// position whose PIVOT ROW is g. Written by [`place`](Self::place)
    /// (disjoint g), read in `emit_and_free`, after the join barrier or the
    /// join chain that makes the consumers' writes visible.
    pub(super) e_of_g: Cells<usize>,
    pub(super) row_pos_of_g: Cells<usize>,
    /// The pivot floors, `(supernode, pivot of the supernode, what the floor
    /// added to the pivot)`, each node's pushed once it is factored.
    pub(super) floors: std::sync::Mutex<Vec<(usize, usize, T)>>,
}

impl<T: Scalar> LlEmit<T> {
    /// The emit state, the arenas in `storage` (the buffers of an earlier
    /// factor of this analysis) when given.
    fn new(
        sym: &SymbolicFactorization,
        sched: &LlSchedule,
        st: &LuStructure,
        delay_growth: f64,
        storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
    ) -> Self {
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let ns = sym.supernodes.len();
        let delays = delay_growth > 0.0;
        let (l_slots, u_slots) = lu_slots(sym, st, delays);
        let allowance = delay_allowance(&l_slots, &u_slots, delay_growth);
        let slot_values = l_slots.iter().chain(&u_slots).map(|&(v, _)| v).sum();
        let (l_arena, u_arena) = match storage {
            _ if delays => {
                // Slots zeroed as they are claimed, so a grown front's never
                // is (the buffers of an earlier factor would be resident).
                drop(storage);
                (
                    PanelArena::lazy(l_slots.into_iter(), allowance),
                    PanelArena::lazy(u_slots.into_iter(), allowance),
                )
            }
            Some((l, u)) => (
                PanelArena::reuse(l, l_slots.into_iter()),
                PanelArena::reuse(u, u_slots.into_iter()),
            ),
            None => (
                PanelArena::new(l_slots.into_iter()),
                PanelArena::new(u_slots.into_iter()),
            ),
        };
        LlEmit {
            refcount,
            e_offset,
            l_arena,
            u_arena,
            panels: Cells::new_default(ns),
            pivots: Cells::new_default(ns),
            delays,
            allowance,
            growth: AtomicUsize::new(0),
            slot_values,
            overflowed: Default::default(),
            e_of_g: Cells::new(n, usize::MAX),
            row_pos_of_g: Cells::new(n, usize::MAX),
            floors: Default::default(),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }
    #[inline]
    unsafe fn rg(&self, g: usize) -> usize {
        *self.row_pos_of_g.get(g)
    }

    /// Give supernode `s`'s pivots the elimination positions from `eoff` on:
    /// the position maps of their columns and rows.
    ///
    /// # Safety
    /// `s` is factored, its record visible, and nobody else places it.
    pub(super) unsafe fn place(&self, s: usize, eoff: usize) {
        let pv = self.pivots.get(s);
        for p in 0..pv.npiv {
            let (gc, gr) = (pv.cols[p] as usize, pv.rows[p] as usize);
            self.e_of_g.set(gc, eoff + p);
            self.row_pos_of_g.set(gr, eoff + p);
        }
    }
}

/// An arena slot: `(values, rows)`.
pub(super) type Slot = (usize, usize);

/// The arena slots `(values, rows)` of the `L` and `U^T` panels of every
/// supernode: its panels of the structure `st`, and with delayed pivots
/// room in the row lists for the pivots it may pass on.
pub(super) fn lu_slots(
    sym: &SymbolicFactorization,
    st: &LuStructure,
    delays: bool,
) -> (Vec<Slot>, Vec<Slot>) {
    use crate::numeric::supernodal::panel::slot_sizes;
    let ns = sym.supernodes.len();
    let slot = |s: usize, nrows: usize| {
        let w = sym.supernodes[s].ncol;
        let (v, m) = slot_sizes(w, nrows - w);
        (v, if delays && w > 0 { m + w } else { m })
    };
    (
        (0..ns).map(|s| slot(s, st.rows_l(s).len())).collect(),
        (0..ns).map(|s| slot(s, st.cols_u(s).len())).collect(),
    )
}

/// The values delayed pivots may add to the factor over its slots:
/// `delay_growth` of the slots' values.
pub(super) fn delay_allowance(l_slots: &[Slot], u_slots: &[Slot], delay_growth: f64) -> usize {
    let values: usize = l_slots.iter().chain(u_slots).map(|&(v, _)| v).sum();
    if delay_growth > 0.0 {
        (delay_growth * values as f64).ceil() as usize
    } else {
        0
    }
}

/// Emit supernode `k` once its last updater is done (or, with delayed pivots,
/// once the forest is): the left-looking panel `lbuf` (`L` strictly below the
/// diagonal, `U`'s diagonal block on and above it) becomes the `L` panel as
/// it is, its pivot columns, and `U^T`'s panel, whose off-block rows the
/// factorization wrote in place (`U12`), gets its diagonal block and the
/// rows of the delayed columns from the upper triangle. Both get their rows
/// past the pivots in elimination order.
fn emit_and_free<T: Scalar>(
    k: usize,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    st: &LuStructure,
) {
    // SAFETY (the slot and record accesses below): the owner of supernode
    // `k` emits it exactly once, after its last updater has read the slots
    // (refcount zero) or the forest is done; nobody reads them afterwards.
    let pv = unsafe { emit.pivots.get(k) };
    let (npiv, ol, ou) = pv.off(st, k, sym.supernodes[k].ncol);
    if npiv == 0 {
        return;
    }
    let (nrow_l, nrow_u, nfs) = (npiv + ol.len(), npiv + ou.len(), pv.rows.len());
    // `U^T`'s diagonal block and delayed columns from the upper triangle of
    // the `L` slot.
    {
        let lbuf: &[T] = unsafe { emit.l_arena.slot(k) };
        let ut = unsafe { emit.u_arena.slot_mut(k) };
        for p in 0..npiv {
            for i in p..nfs {
                ut[p * nrow_u + i] = lbuf[i * nrow_l + p];
            }
        }
    }
    // The rows past the pivots in elimination indices, into the arenas' row slots.
    let l_rows = unsafe { emit.l_arena.rows_mut(k) };
    for (r, &g) in l_rows.iter_mut().zip(ol) {
        *r = unsafe { emit.rg(g as usize) } as u32;
    }
    let u_rows = unsafe { emit.u_arena.rows_mut(k) };
    for (r, &g) in u_rows.iter_mut().zip(ou) {
        *r = unsafe { emit.eg(g as usize) } as u32;
    }
    let l_out = unsafe { emit.l_arena.finish_slot(k, npiv, ol.len(), None) };
    let u_out = unsafe { emit.u_arena.finish_slot(k, npiv, ou.len(), None) };
    unsafe { emit.panels.set(k, (l_out, u_out)) };
}

/// Supernodal left-looking LU into an `LuNumeric`. `inp` is the equilibrated permuted matrix; `d_row`/`d_col`
/// the equilibration carried into the result; `delay_growth` the growth of
/// the factor delayed pivots may cause (`delay`). `None` when they would
/// have caused more.
#[allow(clippy::too_many_arguments)]
fn factor_lu_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    st: &LuStructure,
    inp: Input<T>,
    d_row: &[f64],
    d_col: &[f64],
    perturb_floor: Option<f64>,
    delay_growth: f64,
    kt: KernelTuning,
    storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
    pools: &LuPools<T>,
) -> Result<Option<LuNumeric<T>>, RslabError> {
    let n = sym.n;
    let ns = sym.supernodes.len();
    let emit = LlEmit::<T>::new(sym, sched, st, delay_growth, storage);
    let factor_node =
        |s: usize| lu_ll_factor_node(s, sym, inp, sched, st, pools, &emit, perturb_floor, kt);
    let emit_free = |k: usize| {
        if !emit.delays {
            emit_and_free(k, &emit, sym, st)
        }
    };
    let forest =
        crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free);
    if emit.overflowed.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(None);
    }
    forest?;
    // SAFETY (the record reads below): the forest is factored, every record
    // written and visible after its join.
    let npiv = |s: usize| unsafe { emit.pivots.get(s) }.npiv;
    let mut e_offset = Vec::with_capacity(ns);
    let mut at = 0usize;
    for s in 0..ns {
        e_offset.push(at);
        at += npiv(s);
    }
    debug_assert_eq!(at, n, "every column pivoted once");
    if emit.delays {
        // The positions follow from the pivots every supernode took; then
        // every supernode is emitted.
        for (s, &eoff) in e_offset.iter().enumerate() {
            // SAFETY: each supernode placed once, after the forest.
            unsafe { emit.place(s, eoff) };
        }
        crate::parallel::for_each(ns, 1, |k| emit_and_free(k, &emit, sym, st));
    }
    let delayed = (0..ns)
        .map(|s| unsafe { emit.pivots.get(s) }.passed())
        .sum();
    let forced = (0..ns).map(|s| unsafe { emit.pivots.get(s) }.forced).sum();
    let growth = emit.growth.load(std::sync::atomic::Ordering::Acquire) as f64
        / emit.slot_values.max(1) as f64;
    let kept: Vec<bool> = (0..ns).map(|s| npiv(s) > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let mut perm = vec![0usize; n];
    let mut perm_row = vec![0usize; n];
    for (s, &eoff) in e_offset.iter().enumerate() {
        let pv = unsafe { emit.pivots.get(s) };
        for p in 0..pv.npiv {
            perm[eoff + p] = sym.perm[pv.cols[p] as usize];
            perm_row[eoff + p] = sym.perm[pv.rows[p] as usize];
        }
    }
    let widths: Vec<usize> = (0..ns).map(npiv).collect();
    let LlEmit {
        l_arena,
        u_arena,
        panels,
        floors,
        ..
    } = emit;
    let (l, zeros_l) = l_arena.finish(n, widths.iter().copied(), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).0)
    });
    let (ut, zeros_u) = u_arena.finish(n, widths.iter().copied(), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).1)
    });
    // The factor is of `A_hat + Delta`, `A_hat = D_r A D_c` (`B`'s rows under
    // a matching) and `Delta` diagonal in elimination order: of `A + E` with
    // `E = D_r^-1 Delta D_c^-1` at each position's pivot row and column.
    let mut floors = floors.into_inner().unwrap_or_else(|e| e.into_inner());
    floors.sort_unstable_by_key(|&(s, k, _)| e_offset[s] + k);
    let floors = floors
        .into_iter()
        .map(|(s, k, delta)| {
            let e = e_offset[s] + k;
            let (row, col) = (perm_row[e], perm[e]);
            PivotFloor {
                row,
                col,
                delta: delta * T::from_real((d_row[row] * d_col[col]).recip()),
            }
        })
        .collect();

    Ok(Some(LuNumeric {
        l,
        ut,
        perm,
        perm_row,
        d_row: d_row.to_vec(),
        d_col: d_col.to_vec(),
        supernode_parent,
        floors,
        n_zeros: zeros_l + zeros_u,
        delayed,
        forced,
        delay_growth: growth,
        delays_dropped: false,
    }))
}

/// PARDISO phases 2-3 for the general path: numeric LU reusing a [`LuSymbolic`].
/// `a` must share the analyzed pattern (`n`, `nnz`). `storage` holds the panel
/// buffers of an earlier factor of the same analysis, filled again in place.
#[allow(clippy::needless_range_loop)] // CSC column loops index col_ptr + scaling
pub(crate) fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
    storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
    pools: &LuPools<T>,
) -> Result<LuNumeric<T>, RslabError> {
    a.validate()?;
    let n = lusym.n;
    if a.n != n || a.row_idx.len() != lusym.nnz {
        return Err(RslabError::InvalidInput(
            "factor_general_lu_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    if n == 0 {
        return Ok(LuNumeric {
            l: PanelFactor::empty(),
            ut: PanelFactor::empty(),
            perm: Vec::new(),
            perm_row: Vec::new(),
            d_row: Vec::new(),
            d_col: Vec::new(),
            supernode_parent: Vec::new(),
            floors: Vec::new(),
            n_zeros: 0,
            delayed: 0,
            forced: 0,
            delay_growth: 0.0,
            delays_dropped: false,
        });
    }

    let perturb_floor: Option<f64> = match opts.pivoting.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    // The assembly-tree levels are no longer needed: the driver is a
    // work-stealing tree recursion, not a level-synchronous sweep.
    let (sym, _by_level) = lusym
        .symb
        .sym_and_levels()
        .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;

    // Two-sided equilibration A_hat = D_r A D_c (`opts.scaling`: the one-pass
    // default, the iterated Ruiz form or none; `scaling::twosided`). Tames the
    // dynamic range (MoM near-field saddles span ten orders) so the LU factor,
    // its threshold test and its static pivot floor work on unit-scale rows and
    // columns; the solve undoes it transparently. Computed from the original
    // (unpermuted) A. Under the MC64 row matching the matrix factored is its
    // row-permuted form `B` (row `i` of `B` is row `row_of[i]` of `A`) with the
    // matching's scalings, which already put every row and column maximum at
    // one: the equilibration setting has nothing left to do there.
    let (d_row, d_col): (Vec<f64>, Vec<f64>) = match &lusym.matching {
        Some(m) => ((0..n).map(|i| m.r[m.row_of[i]]).collect(), m.c.clone()),
        None => crate::logging::timed(
            || format!("lu: equilibration {:?}", opts.scaling),
            || {
                // on the factorization's workers (its sweeps run in parallel)
                let (r, c, steps) = opts.threads.run(
                    0,
                    |cap| cap,
                    || {
                        crate::scaling::equilibrate_general(
                            a,
                            &opts.scaling,
                            opts.pivoting.threshold,
                        )
                    },
                );
                if crate::logging::enabled(crate::logging::LogLevel::Debug) {
                    crate::logging::debug(&format!("lu: equilibration in {steps} steps"));
                }
                (r, c)
            },
        ),
    };

    // `A`'s row `r` is `B`'s row `b_row[r]`, scaled by `d_row[b_row[r]]`.
    let b_row: Option<Vec<usize>> = lusym.matching.as_ref().map(|m| m.row_map());
    let prog = lusym.input.get_or_init(|| {
        crate::logging::timed(
            || "lu: input program".into(),
            || InputProgram::general(&a.col_ptr, &a.row_idx, b_row.as_deref(), sym),
        )
    });
    let weight = |r: usize, j: usize| d_row[b_row.as_ref().map_or(r, |b| b[r])] * d_col[j];
    let vals = prog.values(&a.col_ptr, &a.row_idx, &a.values, Some(&weight));
    drop(b_row);
    let inp = Input::new(prog, &vals);
    // Worker stack sized to the assembly-tree depth (overflow-safe on deep chain
    // trees), shared by both LU paths.
    let stack = crate::numeric::settings::stack_for_depth(
        crate::numeric::settings::supernode_tree_depth(sym),
    );

    // Run in a scoped pool of `opts.threads` so concurrent solves don't
    // oversubscribe.
    let mut fac = opts.threads.run(
        stack,
        |cap| crate::numeric::supernodal::analysis::recommend_threads_for_sym(&lusym.symb, cap),
        || {
            let sched = lusym
                .symb
                .ll_schedule()
                .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;
            let growth = opts.pivoting.delay_growth;
            let run = |growth: f64, storage| {
                let st = lusym.structure_for(growth);
                let kt = opts.kernel();
                factor_lu_left_looking(
                    sym,
                    sched,
                    st,
                    inp,
                    &d_row,
                    &d_col,
                    perturb_floor,
                    growth,
                    kt,
                    storage,
                    pools,
                )
            };
            match run(growth, storage)? {
                Some(fac) => Ok(fac),
                None => {
                    // The delayed pivots would have grown the factor past
                    // their allowance: the factor without them, every pivot
                    // in its own front.
                    crate::logging::warn(&format!(
                        "lu: delayed pivots exceed pivoting.delay_growth = {growth}; factored \
                         without delays"
                    ));
                    let mut fac = run(0.0, None)?.ok_or_else(|| {
                        RslabError::InvalidInput("internal: no delays, no allowance".to_string())
                    })?;
                    fac.delays_dropped = true;
                    Ok(fac)
                }
            }
        },
    )?;
    finish_matching(&mut fac, lusym);
    Ok(fac)
}
