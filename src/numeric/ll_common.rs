//! Shared scaffolding of the supernodal left-looking drivers - the pieces that
//! are identical between the LDLᵀ and the LU twin: the per-supernode slot
//! store, the raw panel pointer used for disjoint-row parallel writes, and the
//! heuristic `tuned` driver (ND bakeoff + calibrated worker count).

use crate::diagnostics::MemoryEstimate;
use crate::error::RslabError;
use crate::numeric::multifrontal_ldlt::{FactorMethod, SolverSettings};
use crate::symbolic::OrderingMethod;

/// One `UnsafeCell` payload per supernode, written exactly once by the
/// supernode's owner and read only by nodes that are (transitively) its
/// assembly-tree ancestors - the single-writer-before-readers discipline the
/// left-looking schedule guarantees. `free` resets a slot once its last
/// consumer is done.
pub(crate) struct SlotStore<P> {
    slots: Vec<std::cell::UnsafeCell<P>>,
}

// SAFETY: single-writer-before-readers, disjoint indices (see the type doc).
unsafe impl<P: Send> Sync for SlotStore<P> {}

impl<P: Default> SlotStore<P> {
    pub fn new(nsuper: usize) -> Self {
        SlotStore {
            slots: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(P::default()))
                .collect(),
        }
    }

    /// SAFETY: `k` must be a fully-factored descendant of the current node
    /// (its owner's write happened-before this read).
    pub unsafe fn get(&self, k: usize) -> &P {
        &*self.slots[k].get()
    }

    /// SAFETY: only the owner of supernode `s` calls this, exactly once.
    pub unsafe fn set(&self, s: usize, p: P) {
        *self.slots[s].get() = p;
    }

    /// Release `k`'s payload once it has been compacted.
    /// SAFETY: `k`'s last consumer is done - no other thread reads this slot.
    pub unsafe fn free(&self, k: usize) {
        *self.slots[k].get() = P::default();
    }
}

/// Raw base pointer of a panel buffer, smuggled across rayon workers so each
/// task can write its own **disjoint row range** of a column-major panel. Safe
/// only because callers partition the rows so no two tasks touch the same cell.
#[derive(Clone, Copy)]
pub(crate) struct PanelPtr<T>(pub *mut T);
// SAFETY: the pointer is only dereferenced on disjoint, caller-partitioned cells.
unsafe impl<T> Send for PanelPtr<T> {}
unsafe impl<T> Sync for PanelPtr<T> {}
impl<T> PanelPtr<T> {
    /// Extract the raw pointer. Taking `self` by value forces a closure to
    /// capture the whole (Send+Sync) wrapper rather than disjoint-capturing the
    /// bare field.
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// Minimum predicted factor flops for the exact nested-dissection bakeoff in
/// `tuned` (below this the analysis cost cannot amortize).
pub(crate) const ND_BAKEOFF_MIN_FLOPS: u64 = 5_000_000_000;
/// Small systems never enter the bakeoff regardless of predicted flops -
/// dense-ish small matrices can post huge flops without an ND story.
pub(crate) const ND_BAKEOFF_MIN_N: usize = 10_000;
/// Adopt ND only on a clear predicted win, not a coin flip.
pub(crate) const ND_BAKEOFF_ADOPT_RATIO: f64 = 0.75;

/// The deterministic heuristic settings pick shared by `LdltSolver::tuned` and
/// `LuSolver::tuned`: default settings, an exact ND bakeoff on large systems,
/// and (feature `tuning`, when the one-time install diagnosis has run) the
/// calibrated cost-model worker count.
pub(crate) fn tuned<A: ?Sized, S>(
    a: &A,
    n: usize,
    analyze: impl Fn(&A) -> Result<S, RslabError>,
    analyze_with: impl Fn(&A, &SolverSettings) -> Result<S, RslabError>,
    estimate: impl Fn(&S) -> MemoryEstimate,
    exact_nnz: impl Fn(&S) -> usize,
) -> Result<(S, SolverSettings), RslabError> {
    let sym = analyze(a)?;
    let s = SolverSettings::default();
    #[allow(unused_mut)]
    let (sym, mut s) =
        if n >= ND_BAKEOFF_MIN_N && estimate(&sym).factor_flops >= ND_BAKEOFF_MIN_FLOPS {
            nd_bakeoff(a, sym, s, &analyze_with, &estimate, &exact_nnz)?
        } else {
            (sym, s)
        };
    // Install-diagnosed worker count: only when a calibration cache exists
    // (written once by `tuning::install_diagnose`); never measures here.
    #[cfg(feature = "tuning")]
    if let Some((cores, calib)) = crate::tuning::cached_calibration() {
        let est = estimate(&sym);
        let t = crate::tuning::recommend_threads_cost_model(&est, &calib, 0, cores);
        s.threads = crate::numeric::multifrontal_ldlt::Threads::Fixed(t);
    }
    Ok((sym, s))
}

/// Re-analyze with [`OrderingMethod::MetisND`] and keep whichever ordering the
/// *exact* symbolic quantities favour: ND is adopted only on a clear
/// predicted-flops win with no regression in exact fill or in the
/// method-relevant transient peak, so the pick is Pareto-safe. Deterministic -
/// both candidates are measured on this matrix, nothing is modeled.
pub(crate) fn nd_bakeoff<A: ?Sized, S>(
    a: &A,
    sym: S,
    s: SolverSettings,
    analyze_with: &impl Fn(&A, &SolverSettings) -> Result<S, RslabError>,
    estimate: &impl Fn(&S) -> MemoryEstimate,
    exact_nnz: &impl Fn(&S) -> usize,
) -> Result<(S, SolverSettings), RslabError> {
    if s.ordering == OrderingMethod::MetisND {
        return Ok((sym, s));
    }
    let mut s_nd = s.clone();
    s_nd.ordering = OrderingMethod::MetisND;
    let sym_nd = match analyze_with(a, &s_nd) {
        Ok(x) => x,
        Err(_) => return Ok((sym, s)), // ND analysis failed -> keep the pick
    };
    let est = estimate(&sym);
    let est_nd = estimate(&sym_nd);
    let peak = |e: &MemoryEstimate| match s.method {
        FactorMethod::Multifrontal => e.mf_transient_peak_bytes,
        _ => e.panel_live_peak_bytes,
    };
    let flops_win = (est_nd.factor_flops as f64) < est.factor_flops as f64 * ND_BAKEOFF_ADOPT_RATIO;
    let fill_ok = exact_nnz(&sym_nd) <= exact_nnz(&sym);
    let mem_ok = peak(&est_nd) <= peak(&est);
    if flops_win && fill_ok && mem_ok {
        Ok((sym_nd, s_nd))
    } else {
        Ok((sym, s))
    }
}

/// A fixed-size array of independently written cells: each index is written by
/// exactly one owner (disjoint indices) and read only after a happens-before
/// barrier (subtree join / refcount Acquire-Release). Centralizes the
/// `Vec<UnsafeCell<V>>` pattern of the left-looking emit state.
pub(crate) struct Cells<V>(Vec<std::cell::UnsafeCell<V>>);

// SAFETY: disjoint-index writes; cross-thread visibility is the caller's
// barrier (see the type doc).
unsafe impl<V: Send> Sync for Cells<V> {}

impl<V: Default> Cells<V> {
    /// `n` default-initialized cells (for payloads without a cheap `Clone`).
    pub fn new_default(n: usize) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(V::default()))
                .collect(),
        )
    }
}

impl<V: Clone> Cells<V> {
    pub fn new(n: usize, init: V) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(init.clone()))
                .collect(),
        )
    }
}

impl<V> Cells<V> {
    /// SAFETY: `i` is this caller's exclusively owned index.
    #[inline]
    pub unsafe fn set(&self, i: usize, v: V) {
        *self.0[i].get() = v;
    }
    /// SAFETY: the write to `i` happened-before this read.
    #[inline]
    pub unsafe fn get(&self, i: usize) -> &V {
        &*self.0[i].get()
    }
    /// SAFETY: as [`set`](Self::set) - exclusive owner, e.g. for in-place take.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn get_mut(&self, i: usize) -> &mut V {
        &mut *self.0[i].get()
    }
}

/// Consumer refcounts (free a panel when its count reaches zero) and the
/// per-supernode first elimination position - the shared head of both
/// left-looking emit states.
pub(crate) fn emit_refcount_offsets(
    sym: &crate::symbolic::SymbolicFactorization,
    update_list: &[Vec<usize>],
) -> (Vec<std::sync::atomic::AtomicUsize>, Vec<usize>) {
    use std::sync::atomic::AtomicUsize;
    let nsuper = sym.supernodes.len();
    let mut refcount: Vec<AtomicUsize> = (0..nsuper).map(|_| AtomicUsize::new(0)).collect();
    for ul in update_list {
        for &k in ul {
            *refcount[k].get_mut() += 1;
        }
    }
    let mut e_offset = vec![0usize; nsuper];
    let mut acc = 0usize;
    for (s, snode) in sym.supernodes.iter().enumerate() {
        e_offset[s] = acc;
        acc += snode.ncol;
    }
    (refcount, e_offset)
}

/// Cached scatter program for the numeric phase's permuted matrix (the KLU
/// pattern applied to the supernodal twins): the permuted CSC *structure* is a
/// pure function of the analyzed pattern and the fill-reducing permutation, so
/// it is built once per symbolic analysis; every (re)factorization then only
/// scatters the new values through `pos` - one linear pass, no counting sort,
/// no per-column sorting.
pub(crate) struct PermScatter {
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    /// `pos[k]` = slot of original entry `k` in the permuted values array.
    pub pos: Vec<usize>,
}

impl PermScatter {
    /// Build for the symmetric lower-triangle fold `Pᵀ A P` (LDLᵀ path):
    /// original entry `(i, j)` lands at permuted `(max(gi, gj), min(gi, gj))`
    /// with `g = perm_inv[·]`. Columns come out row-sorted.
    pub fn build_lower(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(
            n,
            a_col_ptr,
            a_row_idx,
            |gi, gj| {
                if gi >= gj {
                    (gi, gj)
                } else {
                    (gj, gi)
                }
            },
            perm_inv,
        )
    }

    /// Build for the full (unfolded) permutation `Pᵀ A P` (LU path).
    pub fn build_full(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(n, a_col_ptr, a_row_idx, |gi, gj| (gi, gj), perm_inv)
    }

    /// Build for the transpose of the full permutation, `(Pᵀ A P)ᵀ` (the LU
    /// path's `a_perm_t`): entry `(i, j)` lands at `(gj, gi)`.
    pub fn build_full_transposed(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(n, a_col_ptr, a_row_idx, |gi, gj| (gj, gi), perm_inv)
    }

    fn build_with(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        target: impl Fn(usize, usize) -> (usize, usize),
        perm_inv: &[usize],
    ) -> Self {
        let nnz = a_row_idx.len();
        // Pass 1: count entries per target column.
        let mut col_ptr = vec![0usize; n + 1];
        for (j, &gj) in perm_inv.iter().enumerate() {
            for k in a_col_ptr[j]..a_col_ptr[j + 1] {
                let (_, c) = target(perm_inv[a_row_idx[k]], gj);
                col_ptr[c + 1] += 1;
            }
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
        }
        // Pass 2: place (row, original-entry) pairs per column.
        let mut cursor = col_ptr[..n].to_vec();
        let mut pairs: Vec<(usize, usize)> = vec![(0, 0); nnz];
        for (j, &gj) in perm_inv.iter().enumerate() {
            for k in a_col_ptr[j]..a_col_ptr[j + 1] {
                let (r, c) = target(perm_inv[a_row_idx[k]], gj);
                let p = cursor[c];
                cursor[c] += 1;
                pairs[p] = (r, k);
            }
        }
        // Pass 3: sort each column by row (once, at build time) and freeze the
        // structure + position map.
        let mut row_idx = vec![0usize; nnz];
        let mut pos = vec![0usize; nnz];
        for c in 0..n {
            let (s, e) = (col_ptr[c], col_ptr[c + 1]);
            pairs[s..e].sort_unstable_by_key(|&(r, _)| r);
            for (p, &(r, k)) in (s..e).zip(pairs[s..e].iter()) {
                row_idx[p] = r;
                pos[k] = p;
            }
        }
        PermScatter {
            col_ptr,
            row_idx,
            pos,
        }
    }

    /// Scatter a fresh value set through the frozen structure: one linear pass.
    /// `scale(k)` maps the original entry's value (e.g. equilibration); pass
    /// the identity for the plain permutation.
    pub fn scatter<T: crate::scalar::Scalar>(
        &self,
        values: &[T],
        mut map: impl FnMut(usize, T) -> T,
    ) -> Vec<T> {
        debug_assert_eq!(values.len(), self.pos.len());
        let mut out = vec![T::zero(); values.len()];
        for (k, &p) in self.pos.iter().enumerate() {
            out[p] = map(k, values[k]);
        }
        out
    }
}
