//! The numeric LDL^T factorization: the permuted input, the left-looking
//! driver over the assembly forest, and the emit of each finished panel.

use super::bunch_kaufman::BkPools;
use super::node::ll_factor_node;
use super::pivots::LdltPivots;
use crate::numeric::supernodal::analysis::{recommend_threads_for_sym, SupernodalAnalysis};

use crate::error::RslabError;
use crate::inertia::Inertia;
use crate::numeric::settings::{
    stack_for_depth, supernode_tree_depth, SolverSettings, ZeroPivotAction,
};
use crate::numeric::supernodal::panel::{PanelArena, PanelFactor, PanelOut, PanelStorage};
use crate::numeric::supernodal::{emit_refcount_offsets, Cells, Input, InputProgram, LlSchedule};
use crate::refine::PivotFloor;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct LdltNumeric<T> {
    /// `L` in panel form, in elimination order.
    pub factor: PanelFactor<T>,
    /// `D`, the permutation and the outcome.
    pub pivots: LdltPivots<T>,
    /// Structural panel slots holding an exact zero (cancellation); the
    /// stored nonzeros are `factor.nnz() - n_zeros`.
    pub n_zeros: usize,
}

pub(crate) fn factor_numeric<T: Scalar>(
    symb: &SupernodalAnalysis,
    a: &CscMatrix<T>,
    scale: Option<&[f64]>,
    opts: &SolverSettings,
    storage: Option<PanelStorage<T>>,
    pools: &BkPools<T>,
) -> Result<LdltNumeric<T>, RslabError> {
    let n = symb.n;
    if a.n != n || a.row_idx.len() != symb.nnz {
        return Err(RslabError::InvalidInput(
            "factor_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    let inner = match &symb.inner {
        None => {
            return Ok(LdltNumeric {
                factor: PanelFactor::empty(),
                pivots: LdltPivots {
                    n: 0,
                    d_diag: Vec::new(),
                    d_subdiag: Vec::new(),
                    two_by_two: Vec::new(),
                    perm: Vec::new(),
                    supernode_parent: Vec::new(),
                    floors: Vec::new(),
                    inertia: Inertia::new(0, 0, 0),
                },
                n_zeros: 0,
            });
        }
        Some(i) => i,
    };
    let sym = &inner.sym;
    let stack = stack_for_depth(supernode_tree_depth(sym));

    let prog = inner.input.get_or_init(|| {
        let _s = rsprobe::scope!(DEBUG, "input program");
        InputProgram::symmetric(&a.col_ptr, &a.row_idx, &sym.perm_inv)
    });
    let scale_elim: Option<Vec<f64>> = scale.map(|s| sym.perm.iter().map(|&i| s[i]).collect());

    let sched = inner.schedule();
    opts.threads.run(
        stack,
        |cap| recommend_threads_for_sym(symb, cap),
        || {
            let inp = Input::new(prog, &a.values, scale_elim.as_deref().map(|s| (s, s)));
            factor_left_looking(sym, sched, inp, scale, opts, storage, pools)
        },
    )
}

pub(super) struct LlEmitLdlt<T> {
    pub(super) refcount: Vec<AtomicUsize>,
    pub(super) e_offset: Vec<usize>,
    pub(super) arena: PanelArena<T>,
    pub(super) panels: Cells<PanelOut>,
    pub(super) e_of_g: Cells<usize>,
    pub(super) perm: Cells<usize>,
    pub(super) d_diag: Cells<T>,
    pub(super) d_subdiag: Cells<T>,
    pub(super) two_by_two: Cells<bool>,
    pub(super) inertia_pos: AtomicUsize,
    pub(super) inertia_neg: AtomicUsize,
    pub(super) inertia_zero: AtomicUsize,
    pub(super) floors: std::sync::Mutex<Vec<(usize, T)>>,
}

impl<T: Scalar> LlEmitLdlt<T> {
    fn new(
        sym: &SymbolicFactorization,
        sched: &LlSchedule,
        storage: Option<PanelStorage<T>>,
    ) -> Self {
        let nsuper = sym.supernodes.len();
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let shapes = (0..nsuper).map(|s| {
            let w = sym.supernodes[s].ncol;
            crate::numeric::supernodal::panel::slot_sizes(w, sched.rows(s).len() - w)
        });
        let arena = match storage {
            Some(v) => PanelArena::reuse(v, shapes),
            None => PanelArena::new(shapes),
        };
        LlEmitLdlt {
            refcount,
            e_offset,
            arena,
            panels: Cells::new_default(nsuper),
            e_of_g: Cells::new(n, usize::MAX),
            perm: Cells::new(n, 0),
            d_diag: Cells::new(n, T::zero()),
            d_subdiag: Cells::new(n, T::zero()),
            two_by_two: Cells::new(n, false),
            inertia_pos: AtomicUsize::new(0),
            inertia_neg: AtomicUsize::new(0),
            inertia_zero: AtomicUsize::new(0),
            floors: Default::default(),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }

    #[inline]
    pub(super) unsafe fn d_of(&self, k: usize, ncol: usize) -> (&[T], &[T], &[bool]) {
        let r = self.e_offset[k]..self.e_offset[k] + ncol;
        (
            self.d_diag.slice(r.clone()),
            self.d_subdiag.slice(r.clone()),
            self.two_by_two.slice(r),
        )
    }
}

fn ldlt_emit_and_free<T: Scalar>(
    k: usize,
    emit: &LlEmitLdlt<T>,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    pools: &BkPools<T>,
) {
    let ncol = sym.supernodes[k].ncol;
    let nrow = sched.rows(k).len();
    debug_assert_eq!(
        emit.arena.slot_len(k),
        crate::numeric::supernodal::panel::Layout::new(ncol, nrow - ncol).len()
    );
    // SAFETY: the owner of supernode `k` emits it exactly once, after its
    // last updater has read the panel (refcount zero), and nobody reads it
    // afterwards.
    let rows = unsafe { emit.arena.rows_mut(k) };
    for (r, &g) in rows.iter_mut().zip(&sched.rows(k)[ncol..nrow]) {
        *r = unsafe { emit.eg(g as usize) } as u32;
    }
    let (_, _, two) = unsafe { emit.d_of(k, ncol) };
    let out = unsafe {
        emit.arena
            .finish_slot(k, ncol, nrow - ncol, Some(two), &pools.finish)
    };
    unsafe { emit.panels.set(k, out) };
}

fn static_pivot_floor<T: Scalar>(
    values: impl Iterator<Item = T>,
    opts: &SolverSettings,
) -> Option<f64> {
    match opts.pivoting.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = values.map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    }
}

fn factor_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    inp: Input<T>,
    scale: Option<&[f64]>,
    opts: &SolverSettings,
    storage: Option<PanelStorage<T>>,
    pools: &BkPools<T>,
) -> Result<LdltNumeric<T>, RslabError> {
    let n = sym.n;
    let perturb_floor = static_pivot_floor(inp.values(), opts);

    let emit = LlEmitLdlt::<T>::new(sym, sched, storage);
    let kt = opts.kernel();
    let factor_node =
        |s: usize| ll_factor_node(s, sym, inp, sched, pools, &emit, perturb_floor, kt);
    let emit_free = |k: usize| ldlt_emit_and_free(k, &emit, sym, sched, pools);
    // https://github.com/milanofthe/rslab/issues/288
    crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free)?;
    let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let LlEmitLdlt {
        arena,
        panels,
        perm,
        d_diag,
        d_subdiag,
        two_by_two,
        inertia_pos,
        inertia_neg,
        inertia_zero,
        floors,
        ..
    } = emit;
    let (factor, n_zeros) = arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(panels.get_mut(s))
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *perm.get(e) }).collect();
    let d_diag: Vec<T> = (0..n).map(|e| unsafe { *d_diag.get(e) }).collect();
    let d_subdiag: Vec<T> = (0..n).map(|e| unsafe { *d_subdiag.get(e) }).collect();
    let two_by_two: Vec<bool> = (0..n).map(|e| unsafe { *two_by_two.get(e) }).collect();
    let mut floors = floors.into_inner().unwrap_or_else(|e| e.into_inner());
    floors.sort_unstable_by_key(|&(e, _)| e);
    let floors = floors
        .into_iter()
        .map(|(e, delta)| {
            let i = perm[e];
            let s = scale.map_or(1.0, |s| s[i]);
            PivotFloor {
                row: i,
                col: i,
                delta: delta * T::from_real((s * s).recip()),
            }
        })
        .collect();
    let inertia = Inertia::new(
        inertia_pos.load(Ordering::Relaxed),
        inertia_neg.load(Ordering::Relaxed),
        inertia_zero.load(Ordering::Relaxed),
    );

    Ok(LdltNumeric {
        factor,
        pivots: LdltPivots {
            n,
            d_diag,
            d_subdiag,
            two_by_two,
            perm,
            supernode_parent,
            floors,
            inertia,
        },
        n_zeros,
    })
}
