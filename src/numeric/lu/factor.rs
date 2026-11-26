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
use crate::numeric::supernodal::panel::{
    PanelArena, PanelFactor, PanelOut, PanelScratch, PanelStorage,
};
use crate::numeric::supernodal::{emit_refcount_offsets, Cells, LlSchedule, ScratchPool};
use crate::refine::PivotFloor;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::AtomicUsize;

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
    pub(super) refcount: Vec<AtomicUsize>,
    pub(super) e_offset: Vec<usize>,
    pub(super) l_arena: PanelArena<T>,
    pub(super) u_arena: PanelArena<T>,
    pub(super) panels: Cells<(PanelOut, PanelOut)>,
    pub(super) pivots: Cells<NodePivots<T>>,
    pub(super) delays: bool,
    pub(super) overflowed: std::sync::atomic::AtomicBool,
    pub(super) e_of_g: Cells<usize>,
    pub(super) row_pos_of_g: Cells<usize>,
    pub(super) floors: std::sync::Mutex<Vec<(usize, usize, T)>>,
}

impl<T: Scalar> LlEmit<T> {
    fn new(
        sym: &SymbolicFactorization,
        sched: &LlSchedule,
        st: &LuStructure,
        delay_growth: f64,
        storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
        pivots: Cells<NodePivots<T>>,
    ) -> Self {
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let ns = sym.supernodes.len();
        let delays = delay_growth > 0.0;
        let (l_slots, u_slots) = lu_slots(sym, st);
        let allowance = delay_allowance(&l_slots, &u_slots, delay_growth);
        let (l_arena, u_arena) = match storage {
            // https://github.com/milanofthe/rslab/issues/288
            _ if delays => {
                drop(storage);
                let room = |v: &[Slot]| v.iter().map(|&(v, _)| v).sum::<usize>() + allowance;
                (
                    PanelArena::lazy(ns, room(&l_slots)),
                    PanelArena::lazy(ns, room(&u_slots)),
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
            pivots: if pivots.len() == ns {
                pivots
            } else {
                Cells::new_default(ns)
            },
            delays,
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

    pub(super) unsafe fn place(&self, s: usize, eoff: usize) {
        let pv = self.pivots.get(s);
        for p in 0..pv.npiv {
            let (gc, gr) = (pv.cols[p] as usize, pv.rows[p] as usize);
            debug_assert!(self.eg(gc) == usize::MAX && self.rg(gr) == usize::MAX);
            self.e_of_g.set(gc, eoff + p);
            self.row_pos_of_g.set(gr, eoff + p);
        }
    }
}

pub(super) type Slot = (usize, usize);

pub(super) fn lu_slots(sym: &SymbolicFactorization, st: &LuStructure) -> (Vec<Slot>, Vec<Slot>) {
    use crate::numeric::supernodal::panel::slot_sizes;
    let ns = sym.supernodes.len();
    let slot = |s: usize, nrows: usize| {
        let w = sym.supernodes[s].ncol;
        slot_sizes(w, nrows - w)
    };
    (
        (0..ns).map(|s| slot(s, st.rows_l(s).len())).collect(),
        (0..ns).map(|s| slot(s, st.cols_u(s).len())).collect(),
    )
}

pub(super) fn delay_growth(opts: &SolverSettings) -> f64 {
    if opts.pivoting.delayed {
        crate::constants::LU_DELAY_ALLOWANCE
    } else {
        0.0
    }
}

pub(super) fn delay_allowance(l_slots: &[Slot], u_slots: &[Slot], delay_growth: f64) -> usize {
    let values: usize = l_slots.iter().chain(u_slots).map(|&(v, _)| v).sum();
    if delay_growth > 0.0 {
        (delay_growth * values as f64).ceil() as usize
    } else {
        0
    }
}

fn emit_and_free<T: Scalar>(
    k: usize,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    st: &LuStructure,
    finish: &ScratchPool<PanelScratch<T>>,
) {
    // SAFETY: the owner of supernode `k` emits it exactly once, after its
    // last updater has read the slots (refcount zero) or after the forest is
    // done, and nobody reads them afterwards.
    let pv = unsafe { emit.pivots.get(k) };
    let (npiv, ol, ou) = pv.off(st, k, sym.supernodes[k].ncol);
    if npiv == 0 {
        return;
    }
    {
        use crate::numeric::supernodal::panel::Layout;
        let (ll, lu) = (Layout::new(npiv, ol.len()), Layout::new(npiv, ou.len()));
        let lbuf: &[T] = unsafe { emit.l_arena.slot(k) };
        let ut = unsafe { emit.u_arena.slot_mut(k) };
        for p in 0..npiv {
            ut[lu.at(p, p)] = lbuf[ll.at(p, p)];
        }
    }
    let l_rows = unsafe { emit.l_arena.rows_mut(k) };
    for (r, &g) in l_rows.iter_mut().zip(ol) {
        *r = unsafe { emit.rg(g as usize) } as u32;
    }
    let u_rows = unsafe { emit.u_arena.rows_mut(k) };
    for (r, &g) in u_rows.iter_mut().zip(ou) {
        *r = unsafe { emit.eg(g as usize) } as u32;
    }
    let l_out = unsafe { emit.l_arena.finish_slot(k, npiv, ol.len(), None, finish) };
    let u_out = unsafe { emit.u_arena.finish_slot(k, npiv, ou.len(), None, finish) };
    unsafe { emit.panels.set(k, (l_out, u_out)) };
}

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
    let mut records = pools.pivots.take();
    let emit = LlEmit::<T>::new(
        sym,
        sched,
        st,
        delay_growth,
        storage,
        std::mem::take(&mut *records),
    );
    let factor_node =
        |s: usize| lu_ll_factor_node(s, sym, inp, sched, st, pools, &emit, perturb_floor, kt);
    let emit_free = |k: usize| {
        if !emit.delays {
            emit_and_free(k, &emit, sym, st, &pools.finish)
        }
    };
    let forest =
        crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free);
    if emit.overflowed.load(std::sync::atomic::Ordering::Acquire) {
        return Ok(None);
    }
    forest?;
    // SAFETY: the forest is factored, and every record is written and
    // visible after its join.
    let npiv = |s: usize| unsafe { emit.pivots.get(s) }.npiv;
    let mut e_offset = Vec::with_capacity(ns);
    let mut at = 0usize;
    for s in 0..ns {
        e_offset.push(at);
        at += npiv(s);
    }
    debug_assert_eq!(at, n, "every column pivoted once");
    if emit.delays {
        for (s, &eoff) in e_offset.iter().enumerate() {
            // SAFETY: each supernode placed once, after the forest.
            unsafe { emit.place(s, eoff) };
        }
        crate::parallel::ranges::<crate::parallel::Rayon>(
            ns,
            crate::parallel::Work::ns(f64::INFINITY),
            |r| r.for_each(|k| emit_and_free(k, &emit, sym, st, &pools.finish)),
        );
    }
    let delayed = (0..ns)
        .map(|s| unsafe { emit.pivots.get(s) }.passed())
        .sum();
    let forced = (0..ns).map(|s| unsafe { emit.pivots.get(s) }.forced).sum();
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
        pivots,
        ..
    } = emit;
    *records = pivots;
    let (l, zeros_l) = l_arena.finish(n, widths.iter().copied(), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).0)
    });
    let (ut, zeros_u) = u_arena.finish(n, widths.iter().copied(), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).1)
    });
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
        delays_dropped: false,
    }))
}

pub(crate) fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
    delay_growth: f64,
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

    let (sym, _by_level) = lusym
        .symb
        .sym_and_levels()
        .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;

    let (d_row, d_col): (Vec<f64>, Vec<f64>) = match &lusym.matching {
        Some(m) => ((0..n).map(|i| m.r[m.row_of[i]]).collect(), m.c.clone()),
        None => {
            let _s = rsprobe::scope!("scale");
            // https://github.com/milanofthe/rslab/issues/288
            let (r, c, steps) = opts.threads.run(
                0,
                |cap| cap,
                || crate::scaling::equilibrate_general(a, &opts.scaling, opts.pivoting.threshold),
            );
            rsprobe::add!("steps", steps);
            (r, c)
        }
    };

    let prog = lusym.input.get_or_init(|| {
        let _s = rsprobe::scope!(DEBUG, "input program");
        let b_row = lusym.matching.as_ref().map(|m| m.row_map());
        InputProgram::general(&a.col_ptr, &a.row_idx, b_row.as_deref(), sym)
    });
    let (d_row_elim, d_col_elim): (Vec<f64>, Vec<f64>) =
        sym.perm.iter().map(|&i| (d_row[i], d_col[i])).unzip();
    let stack = crate::numeric::settings::stack_for_depth(
        crate::numeric::settings::supernode_tree_depth(sym),
    );

    let mut fac = opts.threads.run(
        stack,
        |cap| crate::numeric::supernodal::analysis::recommend_threads_for_sym(&lusym.symb, cap),
        || {
            let inp = Input::new(prog, &a.values, Some((&d_row_elim, &d_col_elim)));
            let sched = lusym
                .symb
                .ll_schedule()
                .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;
            let growth = delay_growth;
            let run = |growth: f64, storage| {
                let st = &lusym.structure;
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
                    rsprobe::warn!(
                        "delayed pivots exceed their allowance ({growth}); factored without \
                         delays"
                    );
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
