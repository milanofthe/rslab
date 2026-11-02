//! The numeric LU factorization: scaling, the permuted input, the
//! left-looking driver over the assembly forest, and the emit of each
//! finished panel into `L` and `U^T`.

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
    /// First elimination position of each supernode (symbolic prefix sum of ncol).
    pub(super) e_offset: Vec<usize>,
    /// The factors' buffers: every supernode factors `L` straight into its
    /// slot of `l_arena`; `U^T` is gathered into `u_arena` at the emit.
    pub(super) l_arena: PanelArena<T>,
    pub(super) u_arena: PanelArena<T>,
    pub(super) panels: Cells<(PanelOut, PanelOut)>,
    /// `e_of_g[g]` = elimination position of COLUMN g; `row_pos_of_g[g]` =
    /// position whose PIVOT ROW is g. Written in-node (disjoint g), read after the
    /// join barrier (and in `emit_and_free`, where the join chain makes consumer
    /// writes visible).
    pub(super) e_of_g: Cells<usize>,
    pub(super) row_pos_of_g: Cells<usize>,
    pub(super) perm: Cells<usize>,
    pub(super) perm_row: Cells<usize>,
    /// The pivot floors, `(elimination position, what the floor added to
    /// the pivot)`, each node's pushed once it is factored.
    pub(super) floors: std::sync::Mutex<Vec<(usize, T)>>,
}

impl<T: Scalar> LlEmit<T> {
    /// The emit state, the arenas in `storage` (the buffers of an earlier
    /// factor of this analysis) when given.
    fn new(
        sym: &SymbolicFactorization,
        sched: &LlSchedule,
        st: &LuStructure,
        storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
    ) -> Self {
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let ns = sym.supernodes.len();
        let ncol = |s: usize| sym.supernodes[s].ncol;
        let l_sizes = (0..ns).map(|s| (ncol(s), st.rows_l(s).len() - ncol(s)));
        let u_sizes = (0..ns).map(|s| (ncol(s), st.cols_u(s).len() - ncol(s)));
        let (l_arena, u_arena) = match storage {
            Some((l, u)) => (PanelArena::reuse(l, l_sizes), PanelArena::reuse(u, u_sizes)),
            None => (PanelArena::new(l_sizes), PanelArena::new(u_sizes)),
        };
        LlEmit {
            refcount,
            e_offset,
            l_arena,
            u_arena,
            panels: Cells::new_default(sym.supernodes.len()),
            e_of_g: Cells::new(n, usize::MAX),
            row_pos_of_g: Cells::new(n, usize::MAX),
            perm: Cells::new(n, 0),
            perm_row: Cells::new(n, 0),
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
}

/// Emit supernode `k` once its last updater is done: the left-looking panel
/// `lbuf` (`L` strictly below the diagonal, `U`'s diagonal block on and above
/// it) becomes the `L` panel as it is, and `U^T`'s panel, whose off-block rows
/// the factorization wrote in place (`U12`), gets its diagonal block from the
/// upper triangle. Both get their off-block rows in elimination order.
fn emit_and_free<T: Scalar>(
    k: usize,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    st: &LuStructure,
) {
    let snode = &sym.supernodes[k];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let (rows_l, cols_u) = (st.rows_l(k), st.cols_u(k));
    let (nrow_l, nrow_u) = (rows_l.len(), cols_u.len());
    // SAFETY (the slot accesses below): the owner of supernode `k` emits it
    // exactly once, after its last updater has read the slots (refcount
    // zero); nobody reads them afterwards.
    let eoff = emit.e_offset[k];
    debug_assert!(
        (0..ncol).all(|p| unsafe { emit.eg(first + p) } == eoff + p),
        "the diagonal block is in elimination order"
    );
    // `U^T`'s diagonal block from the upper triangle of the `L` slot.
    {
        let lbuf: &[T] = unsafe { emit.l_arena.slot(k) };
        let ut = unsafe { emit.u_arena.slot_mut(k) };
        debug_assert_eq!(ut.len(), nrow_u * ncol);
        for p in 0..ncol {
            for i in p..ncol {
                ut[p * nrow_u + i] = lbuf[i * nrow_l + p];
            }
        }
    }
    // The off-block rows in elimination indices, into the arenas' row slots.
    let l_rows = unsafe { emit.l_arena.rows_mut(k) };
    for (r, i) in l_rows.iter_mut().zip(ncol..nrow_l) {
        *r = unsafe { emit.rg(rows_l[i] as usize) } as u32;
    }
    let u_rows = unsafe { emit.u_arena.rows_mut(k) };
    for (r, &g) in u_rows.iter_mut().zip(&cols_u[ncol..nrow_u]) {
        *r = unsafe { emit.eg(g as usize) } as u32;
    }
    let l_out = unsafe { emit.l_arena.finish_slot(k, ncol, None) };
    let u_out = unsafe { emit.u_arena.finish_slot(k, ncol, None) };
    unsafe { emit.panels.set(k, (l_out, u_out)) };
}

/// Supernodal left-looking LU into an `LuNumeric`. `inp` is the equilibrated permuted matrix; `d_row`/`d_col`
/// the equilibration carried into the result.
#[allow(clippy::too_many_arguments)]
fn factor_lu_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    st: &LuStructure,
    inp: Input<T>,
    d_row: &[f64],
    d_col: &[f64],
    perturb_floor: Option<f64>,
    kt: KernelTuning,
    storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
    pools: &LuPools<T>,
) -> Result<LuNumeric<T>, RslabError> {
    let n = sym.n;
    let emit = LlEmit::<T>::new(sym, sched, st, storage);
    let factor_node =
        |s: usize| lu_ll_factor_node(s, sym, inp, sched, st, pools, &emit, perturb_floor, kt);
    let emit_free = |k: usize| emit_and_free(k, &emit, sym, st);
    crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free)?;
    let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let LlEmit {
        l_arena,
        u_arena,
        panels,
        floors,
        ..
    } = emit;
    let (l, zeros_l) = l_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).0)
    });
    let (ut, zeros_u) = u_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).1)
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm.get(e) }).collect();
    let perm_row: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm_row.get(e) }).collect();
    // The factor is of `A_hat + Delta`, `A_hat = D_r A D_c` (`B`'s rows under
    // a matching) and `Delta` diagonal in elimination order: of `A + E` with
    // `E = D_r^-1 Delta D_c^-1` at each position's pivot row and column.
    let mut floors = floors.into_inner().unwrap_or_else(|e| e.into_inner());
    floors.sort_unstable_by_key(|&(e, _)| e);
    let floors = floors
        .into_iter()
        .map(|(e, delta)| {
            let (row, col) = (perm_row[e], perm[e]);
            PivotFloor {
                row,
                col,
                delta: delta * T::from_real((d_row[row] * d_col[col]).recip()),
            }
        })
        .collect();

    Ok(LuNumeric {
        l,
        ut,
        perm,
        perm_row,
        d_row: d_row.to_vec(),
        d_col: d_col.to_vec(),
        supernode_parent,
        floors,
        n_zeros: zeros_l + zeros_u,
    })
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
            factor_lu_left_looking(
                sym,
                lusym.symb.ll_schedule().ok_or_else(|| {
                    RslabError::InvalidInput("internal: empty symbolic".to_string())
                })?,
                &lusym.structure,
                inp,
                &d_row,
                &d_col,
                perturb_floor,
                opts.kernel(),
                storage,
                pools,
            )
        },
    )?;
    finish_matching(&mut fac, lusym);
    Ok(fac)
}
