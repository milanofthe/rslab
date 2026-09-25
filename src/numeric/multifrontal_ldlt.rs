//! Sparse LDL^T factorization over any [`Scalar`] field: the real (`f64`)
//! and the complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) case, on
//! the value-agnostic symbolic analysis (ordering, elimination tree,
//! supernode amalgamation), with a supernodal left-looking kernel.
//!
//! ## Pivoting scope
//!
//! * Pivoting is restricted to the **fully-summed block** of each supernode:
//!   dense Bunch-Kaufman with 1x1 and 2x2 pivots, so an indefinite block (a KKT
//!   saddle, a circuit's zero-diagonal source row next to its node) factors
//!   whenever the pair sits in one supernode, which the amalgamation makes the
//!   common case (a 45k-node power grid: 1690 2x2 pivots, no failure). There
//!   is no delayed pivoting: a fully-summed block that is singular in exact
//!   mode surfaces as [`RslabError::NumericallyRankDeficient`], and the
//!   static-pivot mode ([`ZeroPivotAction`], the `preconditioner` settings)
//!   lifts the pivot to the floor instead and reports it in `n_perturbed`.
//! * The global factor `L` is kept in supernodal panel form
//!   ([`PanelFactor`]): each supernode's dense panel, once its last consumer
//!   is done, is finished in place (off-block rows into elimination order,
//!   the 2x2 couplings cleared, `drop_tol` applied) and becomes the stored
//!   factor, so the memory peak is the resident panels themselves (see the
//!   a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate)).
//!
//! The result is an [`LdltNumeric`] in factorization order: the panels plus
//! `D`, the permutation and the outcome. [`LdltNumeric::into_factors`]
//! materializes the compressed-column [`LdltFactors`] for the generic
//! [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt).

use crate::dense::ldlt_generic::{bk_alpha, swap_sym_lower_bounded, LdltFactors};
use crate::error::RslabError;
use crate::inertia::Inertia;
use crate::numeric::supernodal::panel::{finish_panel, PanelArena, PanelFactor, PanelOut};
use crate::scalar::Scalar;

/// Scale-invariant singularity floor for a 2x2 Bunch-Kaufman pivot: a block
/// whose `|det|` falls below `GROWTH_EPS * scale^2` (scale = the largest block
/// entry magnitude) is numerically singular - rejected in exact mode and lifted
/// in static-pivot mode. Bounds the element growth `1/|det|` can otherwise
/// inject into the trailing update.
const GROWTH_EPS: f64 = 1e-14;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::{symbolic_factorize_with_method, SupernodeParams, SymbolicFactorization};
use rayon::prelude::*;

use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::settings::{
    in_scoped_pool, stack_for_depth, supernode_tree_depth, ReorderMode, SolverSettings,
    ZeroPivotAction,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Static-pivot perturbation, the complex-symmetric analogue of rslab's f64
/// `perturb_to_floor` (`dense::factor`): lift a pivot whose magnitude is below
/// `abs_floor` up to that floor, preserving phase. For `T = f64` this reduces
/// to `sign(d)*max(|d|, abs_floor)`, matching the real kernel.
#[inline]
pub(crate) fn perturb_pivot<T: Scalar>(d: T, abs_floor: f64) -> T {
    let mag = d.magnitude();
    if mag >= abs_floor {
        d
    } else if mag == 0.0 {
        T::from_real(abs_floor)
    } else {
        d * T::from_real(abs_floor / mag)
    }
}

/// Column-tile width for [`lower_tile_gemm`]. Wide enough that each tile's
/// GEMM stays BLAS-3-efficient, narrow enough that the wasted
/// above-diagonal strip per tile (`< TILE/2` rows) is negligible.
const SCHUR_TILE: usize = 256;

/// Grow `buf` to at least `len` entries without clearing what it holds: the
/// cmod callers overwrite the prefix they read (the D-apply loops fill `vc` and
/// `vd_buf`, and `lower_tile_gemm` writes `u_buf` with `read_dst = false`), so
/// zeroing it before every update only cost bandwidth.
fn grow_scratch<T: Scalar>(buf: &mut Vec<T>, len: usize) {
    if buf.len() < len {
        buf.resize(len, T::zero());
    }
}

/// Symmetric trailing-update GEMM computed **only on and below the tile
/// diagonal**: `TMP[:, j] = G * L21^T[:, j]` for rows `>= tile start`. The
/// consumers (the front Schur subtraction and the left-looking panel
/// subtraction) read only entries with `row >= col`, so the full `m x ncols`
/// product wastes up to half the flops (exactly half for the square front
/// Schur, approaching half for wide root panels where `ncols ~ m`). Tiling
/// the columns and starting each tile's rows at its own diagonal keeps the
/// per-element summation deterministic while cutting the waste to
/// `< SCHUR_TILE/2` rows per tile.
///
/// Layouts: `tmp` is `m x ncols` column-major (column stride `m`, row
/// stride 1); `lhs` is `m x k` with column stride `lhs_cs` (row stride 1);
/// `rhs` is read as `k x ncols` with strides `(rhs_cs = 1, rhs_rs)` -
/// element `(kk, j)` at `rhs[j + kk*rhs_rs]`. Each tile's GEMM goes
/// rayon-parallel at/above the `par_cdiv` flop bar.
///
/// SAFETY: the three buffers must be pairwise-disjoint allocations sized
/// for the strides passed (`tmp` >= `m*ncols`; `lhs` rows `[0, m)` x cols
/// `[0, k)` under `lhs_cs`; `rhs` valid at `j + kk*rhs_rs` for `j < ncols`,
/// `kk < k`).
#[allow(clippy::too_many_arguments)]
unsafe fn lower_tile_gemm<T: Scalar>(
    tmp: &mut [T],
    m: usize,
    ncols: usize,
    k: usize,
    lhs: *const T,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    par_cdiv: usize,
) {
    debug_assert!(ncols <= m);
    debug_assert!(tmp.len() >= m * ncols);
    let mut c0 = 0usize;
    while c0 < ncols {
        let tw = SCHUR_TILE.min(ncols - c0);
        let mrows = m - c0;
        let par = if (mrows as u128) * (tw as u128) * (k as u128) >= par_cdiv as u128 {
            gemm::Parallelism::Rayon(0)
        } else {
            gemm::Parallelism::None
        };
        // Dst tile = columns [c0, c0+tw) rows [c0, m) of `tmp`; lhs = rows
        // [c0, m); rhs = columns [c0, c0+tw).
        crate::dense::gemm_backend::gemm(
            mrows,
            tw,
            k,
            tmp.as_mut_ptr().add(c0 * m + c0),
            m as isize,
            1,
            false,
            lhs.add(c0),
            lhs_cs,
            1,
            rhs.add(c0),
            1,
            rhs_rs,
            T::zero(),
            T::one(),
            false,
            false,
            false,
            par,
        );
        c0 += tw;
    }
}

use crate::numeric::supernodal::PanelPtr as LdltPanelPtr;
use crate::numeric::supernodal::{emit_refcount_offsets, Cells, LlSchedule, PermScatter};

/// Apply a factored Bunch-Kaufman panel's transform sequence to rows
/// `[r0, r1)` of the column-major `panel` (stride `nrow`), for pivot steps
/// `[kb, ke)`. Bit-identical to the corresponding rows of the full-height
/// panel factorization: per 1x1 step the in-panel updates use the **final**
/// column-`k` multipliers (`w_j*d^-1` is exactly the stored `L(j,k)`), then
/// the column is scaled by `d^-1`; per 2x2 step the multiplier pair is
/// rebuilt from the (already perturbed) stored `D` block with the same
/// expressions and order. Deep rows are never pivot candidates, so each
/// caller's row range is independent - the lever that lifts the dominant
/// `O((nrow-ke)*pw^2)` panel work off the serial getf2 path onto all idle
/// workers (ports the LU twin's `apply_panel_trailing` to Bunch-Kaufman).
///
/// `deep_swaps[k - kb]` records the pivot interchange partner of step `k`
/// (`usize::MAX` when the step did not interchange): getf2 bounds its swaps
/// to the panel rows, so the deep-row segments of each interchange are
/// replayed here, immediately before the step's transform - the original
/// full-height order, row by row.
///
/// `mult_snap` holds the in-panel multipliers **as of each step's time**
/// (`mult_snap[(k - kb)*nb + (j - kb)]` is step `k`'s coefficient for
/// in-panel row `j`). Reading them from the final panel would be wrong:
/// later symmetric interchanges permute the rows of earlier multiplier
/// columns (unlike LU, where produced pivot rows never move again).
///
/// SAFETY: `[r0, r1)` must be this caller's exclusive rows and within the
/// buffer; columns `[kb, ke)` must be in bounds under stride `nrow`.
#[allow(clippy::too_many_arguments)]
unsafe fn apply_bk_panel_trailing<T: Scalar>(
    base: *mut T,
    nrow: usize,
    kb: usize,
    ke: usize,
    d_diag: &[T],
    d_subdiag: &[T],
    two_by_two: &[bool],
    deep_swaps: &[usize],
    mult_snap: &[T],
    nb: usize,
    r0: usize,
    r1: usize,
) {
    // Blocked form of the per-pivot sweep. Pivots are taken in sub-blocks of
    // `TRAILING_SB` columns: inside a sub-block a pivot's rank-1 (rank-2)
    // update reaches only the sub-block's remaining columns (scalar loops),
    // while its contribution to the columns beyond the sub-block is deferred
    // and applied as one GEMM `B[:, ke2..ke] -= W * M` per sub-block, where
    // `W` holds the pivot columns before their `D^{-1}` scaling and `M` the
    // multipliers of `mult_snap`. A pivot swap that reaches beyond the
    // sub-block first flushes the pending pivots (the swapped-in column must
    // carry every earlier update, as it does in the sequential sweep). Rows
    // are independent, so any row range gives the same values.
    let deep = r1.saturating_sub(r0);
    if deep == 0 || ke <= kb {
        return;
    }
    // One spare column: a 2x2 pivot that starts on a sub-block's last
    // column extends the sub-block by one, the pair is never split.
    let mut w: Vec<T> = vec![T::zero(); deep * (TRAILING_SB + 1)];
    let mut kb2 = kb;
    while kb2 < ke {
        let mut ke2 = (kb2 + TRAILING_SB).min(ke);
        if ke2 < ke && two_by_two[ke2 - 1] {
            ke2 += 1;
        }
        // Pending pivots [pend0, k) whose deferred update has not been applied.
        let mut pend0 = kb2;
        let mut k = kb2;
        while k < ke2 {
            let kp = deep_swaps[k - kb];
            if kp != usize::MAX {
                if kp >= ke2 {
                    flush_trailing(
                        base, nrow, kb, ke, kb2, ke2, pend0, k, &w, deep, mult_snap, nb, r0,
                    );
                    pend0 = k;
                }
                let src = if two_by_two[k] { k + 1 } else { k };
                let ca = base.add(src * nrow);
                let cb = base.add(kp * nrow);
                for i in r0..r1 {
                    core::ptr::swap(ca.add(i), cb.add(i));
                }
            }
            if two_by_two[k] {
                let (d11, d21, d22) = (d_diag[k], d_subdiag[k], d_diag[k + 1]);
                let det = d11 * d22 - d21 * d21;
                let detinv = det.recip();
                let colk = base.add(k * nrow);
                let colk1 = base.add((k + 1) * nrow);
                for j in (k + 2)..ke2 {
                    let l1j = mult_snap[(k - kb) * nb + (j - kb)];
                    let l2j = mult_snap[(k + 1 - kb) * nb + (j - kb)];
                    let colj = base.add(j * nrow);
                    for i in r0..r1 {
                        *colj.add(i) = *colj.add(i) - *colk.add(i) * l1j - *colk1.add(i) * l2j;
                    }
                }
                let (head, tail) = w.split_at_mut((k + 1 - kb2) * deep);
                let wk = &mut head[(k - kb2) * deep..];
                let wk1 = &mut tail[..deep];
                for i in r0..r1 {
                    let wik = *colk.add(i);
                    let wik1 = *colk1.add(i);
                    wk[i - r0] = wik;
                    wk1[i - r0] = wik1;
                    *colk.add(i) = (d22 * wik - d21 * wik1) * detinv;
                    *colk1.add(i) = (d11 * wik1 - d21 * wik) * detinv;
                }
                k += 2;
            } else {
                let dinv = d_diag[k].recip();
                let colk = base.add(k * nrow);
                for j in (k + 1)..ke2 {
                    let wj_dinv = mult_snap[(k - kb) * nb + (j - kb)];
                    if wj_dinv != T::zero() {
                        let colj = base.add(j * nrow);
                        for i in r0..r1 {
                            *colj.add(i) = *colj.add(i) - *colk.add(i) * wj_dinv;
                        }
                    }
                }
                let wk = &mut w[(k - kb2) * deep..(k - kb2 + 1) * deep];
                for i in r0..r1 {
                    let v = *colk.add(i);
                    wk[i - r0] = v;
                    *colk.add(i) = v * dinv;
                }
                k += 1;
            }
        }
        flush_trailing(
            base, nrow, kb, ke, kb2, ke2, pend0, ke2, &w, deep, mult_snap, nb, r0,
        );
        kb2 = ke2;
    }
}

/// Columns per sub-block of the blocked trailing sweep (the `k` of its
/// GEMMs). 16 measured best: 32 doubles the scalar within-block work,
/// 8 halves the GEMM efficiency.
const TRAILING_SB: usize = 16;

/// Apply the deferred updates of pivots `[p0, p1)` of the sub-block
/// `[kb2, ke2)` (their unscaled columns in `w`, indexed from `kb2`) to the
/// columns `[ke2, ke)` of rows `r0..r0 + deep`:
/// `B[:, ke2..ke] -= W[:, p0..p1] * M[p0..p1, ke2..ke]`.
///
/// # Safety
/// `base` is the panel with leading dimension `nrow`; the row range is
/// this task's own.
#[allow(clippy::too_many_arguments)]
unsafe fn flush_trailing<T: Scalar>(
    base: *mut T,
    nrow: usize,
    kb: usize,
    ke: usize,
    kb2: usize,
    ke2: usize,
    p0: usize,
    p1: usize,
    w: &[T],
    deep: usize,
    mult_snap: &[T],
    nb: usize,
    r0: usize,
) {
    let npend = p1 - p0;
    let ncols = ke - ke2;
    if npend == 0 || ncols == 0 {
        return;
    }
    let lhs = w.as_ptr().add((p0 - kb2) * deep);
    // M rows p0..p1, columns ke2..ke: row-major with stride `nb`.
    let rhs = mult_snap.as_ptr().add((p0 - kb) * nb + (ke2 - kb));
    let dst = base.add(ke2 * nrow + r0);
    // The direct kernel, not the backend entry: these products are skinny
    // (k = TRAILING_SB), where the complex split's plane copies cost more
    // than they save (measured: 1.84 s against 1.98 s single-core).
    gemm::gemm(
        deep,
        ncols,
        npend,
        dst,
        nrow as isize,
        1,
        true,
        lhs,
        deep as isize,
        1,
        rhs,
        1,
        nb as isize,
        T::one(),
        T::zero() - T::one(),
        false,
        false,
        false,
        gemm::Parallelism::None,
    );
}

/// Factor a sparse symmetric matrix `A` as `P^T A P = L D L^T` with
/// Bunch-Kaufman pivoting. Works for `T = f64` and `T = Complex<f64>`
/// (complex symmetric, `A = A^T`).
///
/// Returns an [`LdltFactors`] in factorization order; solve with
/// [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt).
pub fn factor_sparse_ldlt<T: Scalar>(a: &CscMatrix<T>) -> Result<LdltFactors<T>, RslabError> {
    factor_sparse_ldlt_with(a, &SolverSettings::default())
}

/// Like [`factor_sparse_ldlt`] but with explicit [`SolverSettings`] -
/// notably static-pivoting (preconditioner) mode via `on_zero_pivot`.
///
/// Convenience wrapper: runs [`analyze`] then [`factor_numeric`]. For the
/// PARDISO-style *analyze once, factor many* workflow - FEM Newton steps or a
/// frequency sweep that reuse one sparsity pattern - call them separately and
/// keep the [`MultifrontalSymbolic`] across factorizations.
pub fn factor_sparse_ldlt_with<T: Scalar>(
    a: &CscMatrix<T>,
    opts: &SolverSettings,
) -> Result<LdltFactors<T>, RslabError> {
    let symb = analyze(a.n, &a.col_ptr, &a.row_idx)?;
    factor_numeric(&symb, a, None, opts).map(LdltNumeric::into_factors)
}

/// Reusable symbolic analysis (fill-reducing ordering + assembly-tree levels)
/// for a fixed sparsity pattern. Value-independent: build once with [`analyze`]
/// and pass to [`factor_numeric`] for each set of numeric values sharing the
/// pattern - the PARDISO phase-1 analysis.
pub struct MultifrontalSymbolic {
    inner: Option<SymbolicInner>,
    n: usize,
    nnz: usize,
}

impl MultifrontalSymbolic {
    /// The fill-reducing ordering the analysis settled on (`perm[k]` the column that became
    /// column `k`); empty for `n = 0`.
    pub fn permutation(&self) -> &[usize] {
        self.inner.as_ref().map_or(&[], |i| &i.sym.perm[..])
    }
}

struct SymbolicInner {
    sym: SymbolicFactorization,
    /// Assembly-tree levels: `by_level[l]` are the supernodes at level `l`, all
    /// mutually independent (factored concurrently by the rayon driver).
    by_level: Vec<Vec<usize>>,
    /// Lazily built scatter program for `P^T A P` (lower fold): the permuted
    /// structure is fixed per pattern, so every (re)factorization reduces to
    /// one linear values scatter. See [`crate::numeric::supernodal::PermScatter`].
    lower_scatter: std::sync::OnceLock<crate::numeric::supernodal::PermScatter>,
    /// Lazily built left-looking schedule (row structures + updater lists),
    /// pattern-only and shared by the numeric drivers and the estimators.
    ll_schedule: std::sync::OnceLock<LlSchedule>,
}

impl MultifrontalSymbolic {
    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Internal accessor for the unsymmetric LU driver: the symbolic
    /// factorization and the precomputed assembly-tree levels. `None` for the
    /// empty (`n == 0`) analysis.
    pub(crate) fn sym_and_levels(&self) -> Option<(&SymbolicFactorization, &[Vec<usize>])> {
        self.inner.as_ref().map(|i| (&i.sym, i.by_level.as_slice()))
    }

    /// The cached pattern-only left-looking schedule (row structures + updater
    /// lists), built on first use and shared by the numeric drivers and the
    /// a-priori estimators. `None` for the empty analysis.
    pub(crate) fn ll_schedule(&self) -> Option<&LlSchedule> {
        self.inner
            .as_ref()
            .map(|i| i.ll_schedule.get_or_init(|| LlSchedule::build(&i.sym)))
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)`: the number of
    /// eliminated columns and the full front height. The raw material for
    /// factorization-cost diagnostics - front-size distribution (small vs dense
    /// fronts -> BLAS-2 vs BLAS-3 efficiency) and a factor-flop estimate.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        match &self.inner {
            Some(i) => i.sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect(),
            None => Vec::new(),
        }
    }

    pub fn n_supernodes(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.sym.supernodes.len())
    }

    /// Rows of the largest front after amalgamation.
    pub fn max_front(&self) -> usize {
        self.inner
            .as_ref()
            .and_then(|i| i.sym.supernodes.iter().map(|s| s.nrow).max())
            .unwrap_or(0)
    }

    /// The decisions the analysis took on its own (what `Auto` resolved to),
    /// for the [`Diagnostics`](crate::Diagnostics) of every factorization
    /// reusing it. `requested` is the ordering the caller asked for.
    pub fn decisions(
        &self,
        requested: crate::symbolic::OrderingMethod,
    ) -> crate::diagnostics::Decisions {
        let mut d = crate::diagnostics::Decisions {
            ordering_requested: format!("{requested:?}"),
            n_supernodes: self.n_supernodes(),
            max_front: self.max_front(),
            tree_levels: self.n_levels(),
            ..Default::default()
        };
        match &self.inner {
            Some(i) => {
                d.ordering_used = format!("{:?}", i.sym.resolved_method);
                d.preprocess = format!("{:?}", i.sym.resolved_preprocess);
                d.amalgamation = format!("{:?}", i.sym.resolved_amalgamation);
            }
            None => d.ordering_used = d.ordering_requested.clone(),
        }
        d
    }

    /// Number of assembly-tree levels (the level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.by_level.len())
    }

    /// Supernode count per assembly-tree level, leaves first. `level_widths()[l]`
    /// is the number of mutually independent fronts at level `l` - the available
    /// tree-parallelism at that depth. Wide near the leaves, narrowing to (often)
    /// a single chain at the root; the shape that decides whether tree-parallelism
    /// alone saturates the cores or the top fronts need node-parallelism.
    pub fn level_widths(&self) -> Vec<usize> {
        self.inner
            .as_ref()
            .map_or_else(Vec::new, |i| i.by_level.iter().map(|lv| lv.len()).collect())
    }
}

/// PARDISO phase 1: analyze a sparsity pattern (`n`, CSC `col_ptr`/`row_idx`,
/// lower triangle). The result is value-independent and reusable across many
/// [`factor_numeric`] calls that share the pattern.
pub fn analyze(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Result<MultifrontalSymbolic, RslabError> {
    analyze_with(n, col_ptr, row_idx, &SolverSettings::default())
}

/// [`analyze`] with explicit composable [`SolverSettings`] (child-reordering
/// strategy). Reuse the result across many `factor` calls that share the pattern.
pub fn analyze_with(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<MultifrontalSymbolic, RslabError> {
    // The symbolic build (ordering, elimination tree, supernode amalgamation,
    // postorder) can recurse to O(n) on pathological patterns - dense/random
    // graphs where nested dissection finds no good separators - and would overflow
    // the caller's stack. Run it in a scoped pool whose workers have a stack sized
    // to the problem (committed on demand), the same robustness mechanism the
    // factorization uses. Shallow analyses get the floor stack at negligible cost.
    //
    // The pool is sized to the settings' thread budget (the same
    // solver-in-the-loop contract the factorization honours), so every parallel
    // step inside the analysis - notably the ND seed ensemble - respects the
    // configured worker count instead of grabbing all cores.
    in_scoped_pool(opts.resolved_threads(), stack_for_depth(n), || {
        analyze_with_inner(n, col_ptr, row_idx, opts)
    })
}

fn analyze_with_inner(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<MultifrontalSymbolic, RslabError> {
    let nnz = row_idx.len();
    if n == 0 {
        return Ok(MultifrontalSymbolic {
            inner: None,
            n: 0,
            nnz,
        });
    }
    // Symbolic analysis on the structure only; feed a unit-valued f64 pattern.
    let pattern = CscMatrix::<f64> {
        n,
        col_ptr: col_ptr.to_vec(),
        row_idx: row_idx.to_vec(),
        values: vec![1.0; nnz],
    };
    // Disable LdltCompress: it transforms the pattern via a quotient-graph
    // compression beyond a plain permutation, so `sym.perm` would no longer be
    // consistent with the `A_perm` built in `factor_numeric`.
    // Relaxed/fill-tolerant amalgamation - a standard sparse-direct technique
    // (PARDISO/MUMPS apply it to every matrix): when fundamental supernodes are
    // narrow the Schur-update GEMMs are low-rank and memory-bound, so trade a
    // little explicit-zero fill for wider, higher-rank dense fronts. The width is
    // a sweet spot: too narrow -> memory-bound BLAS-2; too wide -> flops wasted on
    // explicit zeros. `<=256-wide, <=64 extra rows/merge` measured best across the
    // EM FEM / MoM matrices (~ -15...-25 % factor time vs the previous 512/128). The lever is
    // workload-agnostic; it rides the general `SupernodeParams.relax` knob and is
    // gated to `n >= RELAX_MIN_N` inside `find_supernodes`.
    let snode_params = SupernodeParams {
        // `preprocess: None` is a correctness requirement, not a tuning knob:
        // LdltCompress rewrites the pattern beyond a permutation, breaking the
        // `sym.perm` <-> `A_perm` consistency `factor_numeric` relies on. The
        // tunable amalgamation knobs (`nemin`, `relax`) ride the composable
        // `SolverSettings`; everything else stays at the tuned default.
        preprocess: crate::symbolic::supernode::OrderingPreprocess::None,
        nemin: opts.nemin,
        relax: opts.relax,
        given_perm: opts.permutation.clone(),
        ..SupernodeParams::default()
    };
    let mut sym = symbolic_factorize_with_method(&pattern, &snode_params, opts.ordering)?;

    // Liu (1986) contribution-stack minimization. Reorder each supernode's
    // children so the live contribution-block stack peak is minimized during
    // factorization. This is a pure **scheduling hint**: supernode IDs, the
    // e-numbering and the factor are unchanged (the global emit walks IDs, not
    // children, and trailing rows are sorted), so it is correctness-, fill- and
    // throughput-neutral - it only shrinks the transient CB-stack that drives
    // factorization peak RSS.
    //
    // Each node leaves a contribution block of size `cb = (nrow-ncol)^2` for its
    // parent and needs `peak` working-stack to factor its subtree. Processing
    // children in order, the stack while doing child `i` is `sum_{j<i} cb_j +
    // peak_i`; Liu's theorem minimizes `max_i(sum_{j<i} cb_j + peak_i)` by ordering
    // children by `(peak - cb)` descending. Supernodes are in postorder, so a
    // single forward sweep has every child's `(peak, cb)` ready.
    //
    // **Hybrid Liu**: reordering is only applied where the contribution stack is
    // actually large (`sum children cb >= LIU_MIN_STACK`) - the upper/mid tree,
    // which is a handful of nodes carrying the spike. The vast majority of small
    // leaf nodes keep their natural order, whose rayon spawn pattern parallelizes
    // better. This keeps almost all of Liu's memory win while shedding most of
    // its throughput cost (the memory-optimal child order is not the
    // parallel-load-optimal one). `peak[s]` is always computed against the order
    // actually used, so the propagation stays exact.
    let nsuper = sym.supernodes.len();
    if opts.reorder == ReorderMode::HybridLiu {
        // ~64 MB of `Complex<f64>` contribution blocks: below this the reorder
        // saves little memory but can still disturb leaf parallelism.
        const LIU_MIN_STACK: f64 = 4_000_000.0;
        let mut cb = vec![0.0f64; nsuper];
        let mut peak = vec![0.0f64; nsuper];
        for s in 0..nsuper {
            let cn = (sym.supernodes[s].nrow - sym.supernodes[s].ncol) as f64;
            cb[s] = cn * cn;
            let mut kids = std::mem::take(&mut sym.supernodes[s].children);
            let stack_total: f64 = kids.iter().map(|&c| cb[c]).sum();
            if stack_total >= LIU_MIN_STACK {
                kids.sort_by(|&a, &b| {
                    (peak[b] - cb[b])
                        .partial_cmp(&(peak[a] - cb[a]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut acc = 0.0f64; // sum cb of already-processed children
            let mut pk = 0.0f64;
            for &ch in &kids {
                pk = pk.max(acc + peak[ch]);
                acc += cb[ch];
            }
            // Assembly step: all children CBs live at once (acc), then this
            // node's own CB remains.
            peak[s] = pk.max(acc).max(cb[s]);
            sym.supernodes[s].children = kids;
        }
    }

    // Assembly-tree levels: level(s) = 1 + max(level(children)); same-level
    // supernodes are mutually independent.
    let mut level = vec![0usize; nsuper];
    let mut max_level = 0usize;
    for s in 0..nsuper {
        let mut lv = 0usize;
        for &ch in &sym.supernodes[s].children {
            lv = lv.max(level[ch] + 1);
        }
        level[s] = lv;
        max_level = max_level.max(lv);
    }
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (s, &lv) in level.iter().enumerate() {
        by_level[lv].push(s);
    }

    Ok(MultifrontalSymbolic {
        inner: Some(SymbolicInner {
            sym,
            by_level,
            lower_scatter: std::sync::OnceLock::new(),
            ll_schedule: std::sync::OnceLock::new(),
        }),
        n,
        nnz,
    })
}

/// PARDISO phases 2-3: numeric factorization reusing a [`MultifrontalSymbolic`].
/// `a` must carry the same sparsity pattern (`n`, `nnz`) the analysis was built
/// from. Honours static pivoting and incomplete-factor dropping via `opts`.
/// Realize a [`Threads::Auto`] policy from a symbolic analysis: compute the three
/// predictive features (factor-flops, max front height, max tree width) and apply
/// the [`recommend_threads_from`](crate::analysis::recommend_threads_from) policy,
/// capped at `max_cores`. Value-independent, so it is the same for every scalar.
pub(crate) fn recommend_threads_for_sym(symb: &MultifrontalSymbolic, max_cores: usize) -> usize {
    let fd = symb.front_dims();
    let flops: u64 = fd
        .iter()
        .map(|&(nc, nr)| (nr as u64) * (nr as u64) * (nc as u64))
        .sum();
    let front_nrow_max = fd.iter().map(|&(_, nr)| nr).max().unwrap_or(0);
    let tree_width_max = symb.level_widths().into_iter().max().unwrap_or(0);
    crate::analysis::recommend_threads_from(flops, front_nrow_max, tree_width_max, max_cores)
}

/// The numeric result of a sparse LDL^T factorization: the unit lower factor
/// `L` in supernodal panel form (the storage the solves run on, written by
/// the drivers without a copy) plus the block diagonal `D`, the pivot
/// permutation and the numeric outcome. [`into_factors`](Self::into_factors)
/// materializes the compressed-column [`LdltFactors`] for the reference
/// solves.
#[derive(Clone, Debug)]
pub struct LdltNumeric<T> {
    /// `L` in panel form, in elimination order.
    pub factor: PanelFactor<T>,
    /// Diagonal of the block-diagonal `D`, length `n`.
    pub d_diag: Vec<T>,
    /// Sub-diagonal of `D` (the `(k+1, k)` entry of a 2x2 block at `k`).
    pub d_subdiag: Vec<T>,
    /// `true` at the first column of each 2x2 pivot block.
    pub two_by_two: Vec<bool>,
    /// `perm[e]` is the original index eliminated at position `e`.
    pub perm: Vec<usize>,
    /// Supernode tree over the factor's supernodes (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// Pivots perturbed by the static regularization.
    pub n_perturbed: usize,
    /// Structural panel slots holding an exact zero (cancellation or
    /// `drop_tol`); the stored nonzeros are `factor.nnz() - n_zeros`.
    pub n_zeros: usize,
    /// Inertia of the factored matrix.
    pub inertia: Inertia,
}

impl<T: Scalar> LdltNumeric<T> {
    /// Dimension.
    pub fn n(&self) -> usize {
        self.factor.n
    }

    /// The compressed-column form for the reference solves (copies the factor).
    pub fn into_factors(self) -> LdltFactors<T> {
        let (l_col_ptr, l_row_idx, l_values) = self.factor.to_csc(true);
        let supernode_ptr: Vec<usize> = self.factor.sn_col.iter().map(|&c| c as usize).collect();
        LdltFactors {
            n: self.factor.n,
            l_col_ptr,
            l_row_idx,
            l_values,
            d_diag: self.d_diag,
            d_subdiag: self.d_subdiag,
            two_by_two: self.two_by_two,
            perm: self.perm,
            supernode_ptr,
            supernode_parent: self.supernode_parent,
            n_perturbed: self.n_perturbed,
            inertia: self.inertia,
        }
    }

    /// Split into the panel factor and an [`LdltFactors`] shell carrying `D`,
    /// the permutation and the outcome with empty CSC arrays: the solver keeps
    /// the shell for the diagonal solves and hands the panels to its plan.
    pub(crate) fn into_parts(self) -> (PanelFactor<T>, LdltFactors<T>) {
        let supernode_ptr: Vec<usize> = self.factor.sn_col.iter().map(|&c| c as usize).collect();
        let shell = LdltFactors {
            n: self.factor.n,
            l_col_ptr: Vec::new(),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            d_diag: self.d_diag,
            d_subdiag: self.d_subdiag,
            two_by_two: self.two_by_two,
            perm: self.perm,
            supernode_ptr,
            supernode_parent: self.supernode_parent,
            n_perturbed: self.n_perturbed,
            inertia: self.inertia,
        };
        (self.factor, shell)
    }
}

pub fn factor_numeric<T: Scalar>(
    symb: &MultifrontalSymbolic,
    a: &CscMatrix<T>,
    scale: Option<&[f64]>,
    opts: &SolverSettings,
) -> Result<LdltNumeric<T>, RslabError> {
    a.validate()?;
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
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                two_by_two: Vec::new(),
                perm: Vec::new(),
                supernode_parent: Vec::new(),
                n_perturbed: 0,
                n_zeros: 0,
                inertia: Inertia::new(0, 0, 0),
            });
        }
        Some(i) => i,
    };
    let sym = &inner.sym;
    // Worker stack sized to the assembly-tree depth so the recursive tree
    // factorization never overflows on deep chain trees (banded / 1D + low nemin).
    let stack = stack_for_depth(supernode_tree_depth(sym));

    // A_perm = P^T A P (lower fold) through the cached scatter program: the
    // structure is frozen on the first factorization of this pattern; every
    // later (re)factorization pays one linear values pass only.
    let scatter = inner
        .lower_scatter
        .get_or_init(|| PermScatter::build_lower(n, &a.col_ptr, &a.row_idx, &sym.perm_inv));
    let a_perm = CscMatrix {
        n,
        col_ptr: scatter.col_ptr.clone(),
        row_idx: scatter.row_idx.clone(),
        values: scatter.scatter(a, scale),
    };

    // Run in a scoped pool of `opts.threads` so concurrent solves don't
    // oversubscribe.
    let sched = inner.ll_schedule.get_or_init(|| LlSchedule::build(sym));
    opts.threads.run(
        stack,
        |cap| recommend_threads_for_sym(symb, cap),
        || factor_left_looking(sym, sched, a_perm, opts),
    )
}

/// One factored supernode's left-looking payload: the dense panel, the
/// Bunch-Kaufman D (diagonal + sub-diagonal + 2x2 flags, pivoted order), and
/// the within-panel pivot permutation (identity on the off-diagonal rows).
struct LdltSlot<T> {
    d: Vec<T>,
    dsub: Vec<T>,
    two: Vec<bool>,
    lperm: Vec<usize>,
}
impl<T> Default for LdltSlot<T> {
    fn default() -> Self {
        LdltSlot {
            d: Vec::new(),
            dsub: Vec::new(),
            two: Vec::new(),
            lperm: Vec::new(),
        }
    }
}
type LlStore<T> = crate::numeric::supernodal::SlotStore<LdltSlot<T>>;

/// Compact (CSC-fragment) form of one supernode's L factor, produced the moment
/// its last consumer pulls from it so the dense panel can be freed during
/// factorization. Row indices are already final elimination positions.
struct LlEmitLdlt<T> {
    refcount: Vec<AtomicUsize>,
    e_offset: Vec<usize>,
    /// The factor's buffer: every supernode factors into its own slot.
    arena: PanelArena<T>,
    panels: Cells<PanelOut>,
    e_of_g: Cells<usize>,
    perm: Cells<usize>,
    d_diag: Cells<T>,
    d_subdiag: Cells<T>,
    two_by_two: Cells<bool>,
    // Inertia accumulated across supernodes (block-aware).
    inertia_pos: AtomicUsize,
    inertia_neg: AtomicUsize,
    inertia_zero: AtomicUsize,
}

impl<T: Scalar> LlEmitLdlt<T> {
    fn new(sym: &SymbolicFactorization, sched: &LlSchedule) -> Self {
        let nsuper = sym.supernodes.len();
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let arena =
            PanelArena::new((0..nsuper).map(|s| sched.rows(s).len() * sym.supernodes[s].ncol));
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
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }
}

/// Compact supernode `k`'s L factor and free its dense panel + D/lperm. Called the
/// instant `k`'s last consumer pulled from it. Mirrors the per-supernode body of
/// the legacy L emit (unit diagonal, skip the 2x2 `d21` coupling row).
fn ldlt_emit_and_free<T: Scalar>(
    k: usize,
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    drop_tol: Option<f64>,
) {
    let ncol = sym.supernodes[k].ncol;
    let nrow = sched.rows(k).len();
    // SAFETY: the owner of supernode `k` emits it exactly once, after its last
    // updater has read the slot (refcount zero); nobody reads it afterwards.
    let slot = unsafe { store.take(k) };
    let panel = unsafe { emit.arena.slot_mut(k) };
    let (lperm, t2) = (&slot.lperm, &slot.two);
    debug_assert_eq!(panel.len(), nrow * ncol);
    debug_assert!(
        (0..ncol)
            .all(|p| unsafe { emit.eg(sched.rows(k)[lperm[p]] as usize) } == emit.e_offset[k] + p),
        "the diagonal block is in elimination order"
    );
    let e_rows: Vec<u32> = (ncol..nrow)
        .map(|i| unsafe { emit.eg(sched.rows(k)[lperm[i]] as usize) } as u32)
        .collect();
    let out = finish_panel(panel, ncol, e_rows, Some(&t2[..ncol]), drop_tol);
    unsafe { emit.panels.set(k, out) };
    if ldlt_no_free() {
        // The debugging hold: keep an (emptied) shell in place.
        unsafe { store.set(k, LdltSlot::default()) };
    }
}

static LDLT_NO_FREE_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn ldlt_no_free() -> bool {
    *LDLT_NO_FREE_FLAG.get_or_init(|| {
        std::env::var("RLA_NO_FREE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Factor one supernode's panel: assemble `A`, apply every descendant's `cmod`
/// update (BLAS-3 with scalar fallback), then `cdiv` (partial 1x1 LDL^T). Reads
/// only already-factored descendant panels from `store`, so sibling subtrees run
/// concurrently. Writes the factored panel + diagonal into `store`.
#[allow(clippy::too_many_arguments)]
fn ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    sched: &LlSchedule,
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    let ll_gemm_gate = kt.scalar_gate;
    let ll_gemm_par = kt.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(s).len();
    let n = sym.n;
    // SAFETY: this task owns supernode `s`; nobody reads the slot before it
    // is published by `store.set` at the end of the cdiv.
    let panel: &mut [T] = unsafe { emit.arena.slot_mut(s) };
    debug_assert_eq!(panel.len(), nrow * ncol);

    // Global-to-local rows (narrow entries halve the table's random-access
    // footprint); restored when the node returns.
    let gloc = crate::numeric::supernodal::Gloc::new(n, sched.rows(s));
    // Assemble A's lower-triangle columns of this supernode.
    for p in 0..ncol {
        let c = first + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let li = gloc[a_perm.row_idx[k]] as usize;
            panel[li + p * nrow] = panel[li + p * nrow] + a_perm.values[k];
        }
    }
    let plan = crate::numeric::supernodal::CmodPlan::new(sym, sched, s, false, ll_gemm_par);
    let (spans, tile_w, tiled) = (&plan.spans, plan.tile_w, plan.tiled);
    let seq_gemm_par = if plan.forks { ll_gemm_par } else { usize::MAX };
    if tiled {
        let gloc_ref = &gloc;
        let spans_ref = spans;
        panel
            .par_chunks_mut(nrow * tile_w)
            .enumerate()
            .for_each(|(ti, tile)| {
                let c0 = ti * tile_w;
                let c1 = (c0 + tile_w).min(ncol);
                let mut vd_buf: Vec<T> = Vec::new();
                let mut u_buf: Vec<T> = Vec::new();
                for &(kk, p0, p1) in spans_ref {
                    let nck = sym.supernodes[kk].ncol;
                    let nrk = sched.rows(kk).len();
                    let ok = &sched.rows(kk)[nck..];
                    let nok = ok.len();
                    // Updater columns landing in this slab.
                    let q0 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c0);
                    let q1 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c1);
                    let npk = q1 - q0;
                    if npk == 0 {
                        continue;
                    }
                    // SAFETY: `kk` is a factored descendant of `s`, its cells
                    // are written and never mutated again.
                    let slot = unsafe { store.get(kk) };
                    let pk: &[T] = unsafe { emit.arena.slot(kk) };
                    let (dk, dsub_k, two_k) = (&slot.d, &slot.dsub, &slot.two);
                    // G = (kk's block rows q0..q1) * D, column-major npk x nck.
                    grow_scratch(&mut vd_buf, npk * nck);
                    let mut ck = 0;
                    while ck < nck {
                        if two_k[ck] {
                            let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                            for i in 0..npk {
                                let a = pk[(nck + q0 + i) + ck * nrk];
                                let b = pk[(nck + q0 + i) + (ck + 1) * nrk];
                                vd_buf[i + ck * npk] = d11 * a + d21 * b;
                                vd_buf[i + (ck + 1) * npk] = d21 * a + d22 * b;
                            }
                            ck += 2;
                        } else {
                            let dkc = dk[ck];
                            for i in 0..npk {
                                vd_buf[i + ck * npk] = pk[(nck + q0 + i) + ck * nrk] * dkc;
                            }
                            ck += 1;
                        }
                    }
                    let mrows = nok - q0;
                    grow_scratch(&mut u_buf, mrows * npk);
                    // Serial per slab - the parallelism is across slabs.
                    // SAFETY: lhs (read), rhs (read), dst (write) pairwise
                    // disjoint; strides in bounds.
                    unsafe {
                        lower_tile_gemm(
                            &mut u_buf,
                            mrows,
                            npk,
                            nck,
                            pk.as_ptr().add(nck + q0),
                            nrk as isize,
                            vd_buf.as_ptr(),
                            npk as isize,
                            usize::MAX,
                        )
                    };
                    for c in 0..npk {
                        let tcol = ok[q0 + c] as usize - first;
                        let ucol = &u_buf[c * mrows..c * mrows + mrows];
                        let dst_col = &mut tile[(tcol - c0) * nrow..(tcol - c0 + 1) * nrow];
                        for r in (q0 + c)..nok {
                            let dst = gloc_ref[ok[r] as usize] as usize;
                            dst_col[dst] = dst_col[dst] - ucol[r - q0];
                        }
                    }
                }
            });
    }

    // Sequential per-update cmod (small nodes / small total update work).
    let mut vc: Vec<T> = Vec::new();
    let mut vd_buf: Vec<T> = Vec::new();
    let mut u_buf: Vec<T> = Vec::new();
    for &(kk, p0, p1) in spans.iter().filter(|_| !tiled) {
        let nck = sym.supernodes[kk].ncol;
        let nrk = sched.rows(kk).len();
        let ok = &sched.rows(kk)[nck..];
        let nok = ok.len();
        // SAFETY: `kk` is a factored descendant of `s` (its update reaches `s`),
        // so its panel/dval cells are written and never mutated again.
        let slot = unsafe { store.get(kk) };
        let pk: &[T] = unsafe { emit.arena.slot(kk) };
        let dk = &slot.d;
        // Bunch-Kaufman block structure of `kk`'s D (pivoted column order). The
        // cmod `L*D*L^T` is invariant under `kk`'s internal column permutation, so
        // only the block-diagonal `D`-apply has to honor the 2x2 blocks.
        let (dsub_k, two_k) = (&slot.dsub, &slot.two);
        let npk = p1 - p0;
        // Gate on the REAL work (rows >= p0); the scalar path already
        // iterates from the target block, so small tails route there.
        if (nok - p0) * npk * nck < ll_gemm_gate {
            grow_scratch(&mut vc, nck);
            for c_idx in p0..p1 {
                let tcol = ok[c_idx] as usize - first;
                // vc = D * (column `c_idx` of kk's off-diagonal block), with D
                // block-diagonal (1x1 and complex-symmetric 2x2 blocks).
                let mut ck = 0;
                while ck < nck {
                    let a = pk[(nck + c_idx) + ck * nrk];
                    if two_k[ck] {
                        let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                        let b = pk[(nck + c_idx) + (ck + 1) * nrk];
                        vc[ck] = d11 * a + d21 * b;
                        vc[ck + 1] = d21 * a + d22 * b;
                        ck += 2;
                    } else {
                        vc[ck] = dk[ck] * a;
                        ck += 1;
                    }
                }
                for r_idx in c_idx..nok {
                    let trow = gloc[ok[r_idx] as usize] as usize;
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + pk[(nck + r_idx) + ck * nrk] * vc[ck];
                    }
                    panel[trow + tcol * nrow] = panel[trow + tcol * nrow] - acc;
                }
            }
        } else {
            grow_scratch(&mut vd_buf, npk * nck);
            // G = (kk's in-panel off-diagonal block) * D, stored column-major as
            // `vd_buf[c + ck*npk]`. D is block-diagonal (1x1 and 2x2 blocks); a
            // 2x2 block mixes its two columns. GEMM below is unchanged.
            let mut ck = 0;
            while ck < nck {
                if two_k[ck] {
                    let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                    for i in 0..npk {
                        let a = pk[(nck + p0 + i) + ck * nrk];
                        let b = pk[(nck + p0 + i) + (ck + 1) * nrk];
                        vd_buf[i + ck * npk] = d11 * a + d21 * b;
                        vd_buf[i + (ck + 1) * npk] = d21 * a + d22 * b;
                    }
                    ck += 2;
                } else {
                    let dkc = dk[ck];
                    for i in 0..npk {
                        vd_buf[i + ck * npk] = pk[(nck + p0 + i) + ck * nrk] * dkc;
                    }
                    ck += 1;
                }
            }
            // Only rows >= p0 land in (or below) the target block: computing
            // the full `nok`-tall product and discarding rows `< p0` in the
            // write-back wasted `p0*npk*nck` flops per update - large for
            // updates into high supernodes, where most of the updater's
            // off-diagonal rows lie above the target. Mirror the LU twin:
            // offset the lhs by `p0` and compute `mrows = nok - p0` rows.
            // The write-back below also reads only rows `>= c` per column
            // (the symmetric lower part), so the product is computed
            // tile-wise from each tile's diagonal downward - the same
            // `lower_tile_gemm` that serves the panel Schur updates. For
            // updates into the topmost supernodes (`mrows ~ npk`) the full
            // rectangle wasted another ~half of the flops.
            let mrows = nok - p0;
            grow_scratch(&mut u_buf, mrows * npk);
            // SAFETY: lhs (`pk` off-diag block from row p0, read), rhs
            // (`vd_buf`, read), dst (`u_buf`, write) are pairwise-disjoint;
            // strides in bounds.
            unsafe {
                lower_tile_gemm(
                    &mut u_buf,
                    mrows,
                    npk,
                    nck,
                    pk.as_ptr().add(nck + p0),
                    nrk as isize,
                    vd_buf.as_ptr(),
                    npk as isize,
                    seq_gemm_par,
                )
            };
            for c in 0..npk {
                let tcol = ok[p0 + c] as usize - first;
                let ucol = &u_buf[c * mrows..c * mrows + mrows];
                for r in (p0 + c)..nok {
                    let dst = gloc[ok[r] as usize] as usize + tcol * nrow;
                    panel[dst] = panel[dst] - ucol[r - p0];
                }
            }
        }
    }
    ll_cdiv_emit(
        s,
        sym,
        sched,
        store,
        emit,
        perturb_floor,
        n_perturbed,
        kt,
        panel,
    )
}

/// One blocked Bunch-Kaufman panel step of the left-looking cdiv over the
/// fully-summed columns `[kb, ke)`: the in-panel getf2 (rows `< ke`) plus the
/// row-parallel deep replay. Touches ONLY panel columns `[kb, ke)` (their full
/// `nrow` height), which is what makes the cdiv panel lookahead sound: the
/// step for panel `p+1` may run concurrently with the wide part of panel
/// `p`'s deferred Schur update (columns `>= ke2`), the two column ranges are
/// disjoint. Returns the number of perturbed pivots.
#[allow(clippy::too_many_arguments)]
fn ll_bk_panel_step<T: Scalar>(
    panel: &mut [T],
    nrow: usize,
    kb: usize,
    ke: usize,
    nb: usize,
    alpha: f64,
    perturb_floor: Option<f64>,
    ll_cdiv_par: usize,
    d: &mut [T],
    d_subdiag: &mut [T],
    two_by_two: &mut [bool],
    lperm: &mut [usize],
    l1: &mut [T],
    l2: &mut [T],
    deep_swaps: &mut [usize],
    mult_snap: &mut [T],
) -> Result<usize, RslabError> {
    let mut perturbed = 0usize;
    // getf2: unblocked Bunch-Kaufman over the panel columns [kb, ke), with
    // EVERYTHING bounded to the panel rows `< ke`: pivot candidates,
    // rank-1/rank-2 updates, interchanges. The deep rows `[ke, nrow)` -
    // the dominant `O((nrow-ke)*pw^2)` share on tall panels - are lifted
    // off this serial path into the parallel `apply_bk_panel_trailing`
    // below (bit-identical replay; ports the LU twin's lever).
    for ds in deep_swaps.iter_mut() {
        *ds = usize::MAX;
    }
    let mut k = kb;
    while k < ke {
        let absakk = panel[k + k * nrow].magnitude();
        // colmax over the in-panel candidate rows (k+1)..ke.
        let mut colmax_sq = 0.0;
        let mut imax = k;
        for i in (k + 1)..ke {
            let m = panel[k * nrow + i].magnitude_sq();
            if m > colmax_sq {
                colmax_sq = m;
                imax = i;
            }
        }
        let colmax = colmax_sq.sqrt();

        let kstep;
        let kp;
        if absakk.max(colmax) == 0.0 {
            if perturb_floor.is_none() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            kstep = 1;
            kp = k;
        } else if absakk >= alpha * colmax {
            kstep = 1;
            kp = k;
        } else {
            // rowmax in row `imax`, restricted to the panel.
            let mut rowmax_sq = 0.0;
            for j in k..imax {
                let m = panel[j * nrow + imax].magnitude_sq();
                if m > rowmax_sq {
                    rowmax_sq = m;
                }
            }
            for i in (imax + 1)..ke {
                let m = panel[imax * nrow + i].magnitude_sq();
                if m > rowmax_sq {
                    rowmax_sq = m;
                }
            }
            let rowmax = rowmax_sq.sqrt();
            if absakk >= alpha * colmax * (colmax / rowmax) {
                kstep = 1;
                kp = k;
            } else if panel[imax * nrow + imax].magnitude() >= alpha * rowmax {
                kstep = 1;
                kp = imax;
            } else {
                kstep = 2;
                kp = imax;
            }
        }

        if kstep == 1 {
            if kp != k {
                swap_sym_lower_bounded(panel, nrow, k, kp, ke);
                lperm.swap(k, kp);
                deep_swaps[k - kb] = kp;
            }
            let mut dk = panel[k + k * nrow];
            match perturb_floor {
                Some(floor) if dk.magnitude() < floor => {
                    dk = perturb_pivot(dk, floor);
                    panel[k + k * nrow] = dk;
                    perturbed += 1;
                }
                None if dk == T::zero() => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            d[k] = dk;
            let dinv = dk.recip();
            // Update the in-panel trailing columns (k+1)..ke over the
            // panel rows, then scale column k's panel rows (deep rows
            // replayed in the parallel apply).
            for j in (k + 1)..ke {
                let wj_dinv = panel[k * nrow + j] * dinv;
                mult_snap[(k - kb) * nb + (j - kb)] = wj_dinv;
                if wj_dinv != T::zero() {
                    for i in j..ke {
                        panel[j * nrow + i] = panel[j * nrow + i] - panel[k * nrow + i] * wj_dinv;
                    }
                }
            }
            for i in (k + 1)..ke {
                panel[k * nrow + i] = panel[k * nrow + i] * dinv;
            }
            k += 1;
        } else {
            if kp != k + 1 {
                swap_sym_lower_bounded(panel, nrow, k + 1, kp, ke);
                lperm.swap(k + 1, kp);
                deep_swaps[k - kb] = kp;
            }
            let mut d11 = panel[k + k * nrow];
            let d21 = panel[k * nrow + (k + 1)];
            let mut d22 = panel[(k + 1) + (k + 1) * nrow];
            let mut det = d11 * d22 - d21 * d21;
            let scale = d11.magnitude().max(d22.magnitude()).max(d21.magnitude());
            let growth_floor = GROWTH_EPS * scale * scale;
            match perturb_floor {
                Some(floor) => {
                    let fl = (floor * floor).max(growth_floor);
                    if det.magnitude() < fl {
                        let lift = floor.max(scale * GROWTH_EPS.sqrt());
                        d11 = d11 + T::from_real(lift);
                        d22 = d22 + T::from_real(lift);
                        det = d11 * d22 - d21 * d21;
                        if det.magnitude() < fl {
                            det = det + T::from_real(fl);
                        }
                        perturbed += 1;
                    }
                }
                None if det.magnitude() <= growth_floor => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            let detinv = det.recip();
            d[k] = d11;
            d_subdiag[k] = d21;
            d[k + 1] = d22;
            two_by_two[k] = true;
            for i in (k + 2)..ke {
                let wik = panel[k * nrow + i];
                let wik1 = panel[(k + 1) * nrow + i];
                l1[i] = (d22 * wik - d21 * wik1) * detinv;
                l2[i] = (d11 * wik1 - d21 * wik) * detinv;
                mult_snap[(k - kb) * nb + (i - kb)] = l1[i];
                mult_snap[(k + 1 - kb) * nb + (i - kb)] = l2[i];
            }
            for j in (k + 2)..ke {
                let l1j = l1[j];
                let l2j = l2[j];
                for i in j..ke {
                    panel[j * nrow + i] = panel[j * nrow + i]
                        - panel[k * nrow + i] * l1j
                        - panel[(k + 1) * nrow + i] * l2j;
                }
            }
            for i in (k + 2)..ke {
                panel[k * nrow + i] = l1[i];
                panel[(k + 1) * nrow + i] = l2[i];
            }
            k += 2;
        }
    }
    // Deep rows [ke, nrow): replay this panel's interchanges + pivot
    // transforms row-parallel (bit-identical to the old full-height
    // getf2 - same per-row op sequence). This is the dominant panel
    // work on tall supernodes; it now runs on all idle workers instead
    // of the serial getf2 path.
    if nrow > ke {
        let deep = nrow - ke;
        let pw = ke - kb;
        let par = deep * pw * pw >= ll_cdiv_par;
        if par {
            let pp = LdltPanelPtr(panel.as_mut_ptr());
            let nthreads = rayon::current_num_threads().max(1);
            let cs = deep.div_ceil(nthreads).max(1);
            let ranges: Vec<(usize, usize)> = (0..nthreads)
                .map(|c| {
                    let r0 = ke + c * cs;
                    (r0.min(nrow), (r0 + cs).min(nrow))
                })
                .filter(|(a, b)| a < b)
                .collect();
            ranges.par_iter().for_each(|&(r0, r1)| {
                // SAFETY: disjoint row chunk; see `apply_bk_panel_trailing`.
                unsafe {
                    apply_bk_panel_trailing(
                        pp.get(),
                        nrow,
                        kb,
                        ke,
                        d,
                        d_subdiag,
                        two_by_two,
                        deep_swaps,
                        mult_snap,
                        nb,
                        r0,
                        r1,
                    )
                };
            });
        } else {
            // Row chunks that keep a chunk of every panel column in L1 across
            // the whole pivot sweep (the sweep streams each column once per
            // pivot; unchunked, a deep panel is re-read from L2 `pw` times).
            // Rows are independent, so every row's arithmetic is unchanged:
            // bit-identical to one call over all rows.
            // SAFETY: single task over all deep rows.
            unsafe {
                apply_bk_panel_trailing(
                    panel.as_mut_ptr(),
                    nrow,
                    kb,
                    ke,
                    d,
                    d_subdiag,
                    two_by_two,
                    deep_swaps,
                    mult_snap,
                    nb,
                    ke,
                    nrow,
                )
            };
        }
    }

    Ok(perturbed)
}

/// cdiv + store + emit for supernode `s` on an already fully cmod-updated
/// `panel` - the tail of [`ll_factor_node`], extracted so the spine
/// pipeline executor (issue #20) can drive assembly/cmod itself and reuse
/// the identical factor kernel. Takes `panel` and the global->local scratch
/// `gloc` by value (`gloc` is returned to the thread-local scratch slot on
/// every exit path).
#[allow(clippy::too_many_arguments)]
fn ll_cdiv_emit<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
    panel: &mut [T],
) -> Result<(), RslabError> {
    let snode = &sym.supernodes[s];
    let ncol = snode.ncol;
    let nrow = sched.rows(s).len();
    // cdiv: partial **blocked** Bunch-Kaufman LDL^T (1x1 and 2x2 pivots), the
    // rectangular `nrow x ncol` analogue of `factor_front`'s panel kernel. The
    // fully-summed columns are factored in panels of width `NB` with pivoting
    // **bounded to the panel** (candidate rows `(k+1)..ke`), then each panel's
    // trailing update - the remaining panel columns `[ke, ncol)` over all rows
    // `[ke, nrow)` - is deferred to one SIMD GEMM (the BLAS-3 bulk, replacing the
    // scalar rank-1/rank-2 sweeps that dominated wide separators). Unlike
    // `factor_front` there is **no `A22` block** (the panel has no columns beyond
    // `ncol`; that Schur update is the ancestors' `cmod`), so the trailing region
    // is the rectangular `(nrow-ke) x (ncol-ke)` lower part. Pivoting stays inside
    // `0..ncol`, so the off-diagonal rows `[ncol, nrow)` keep their identity and
    // `s`'s contribution to ancestors is unaffected by this internal permutation.
    //
    // Adaptive panel width: wide separators get double-width panels - the
    // deferred Schur GEMM's inner dimension is `nb`, and k = 64 is too thin
    // to reach peak on root-class panels (measured ~79 Gflop/s-eq). The
    // extra serial getf2 work is O(nb^3) per panel - negligible against the
    // GEMM gain at this size. The global nb sweep said 128 loses overall
    // because SMALL panels pay; widening only above `ncol >= 512` (a pure
    // function of the node, thread-count independent) keeps them at default.
    let nb = if ncol >= 512 {
        kt.panel_nb.max(128)
    } else {
        kt.panel_nb
    };
    // Same join-steal guard as cmod: a small node must not fork inside its
    // cdiv (deep-row apply / deferred Schur GEMM) - the blocked join steals
    // foreign subtree work and stalls this node's dependents. Total cdiv
    // work ~ nrow*ncol^2 (panel + trailing updates).
    let ll_cdiv_par = if nrow * ncol * ncol >= 100_000_000 {
        kt.par_cdiv
    } else {
        usize::MAX
    };
    let alpha = bk_alpha();
    let mut d = vec![T::zero(); ncol];
    let mut d_subdiag = vec![T::zero(); ncol];
    let mut two_by_two = vec![false; ncol];
    let mut lperm: Vec<usize> = (0..nrow).collect();
    // 2x2 multiplier scratch (reused; only `[k+2, nrow)` is ever read each step).
    let mut l1 = vec![T::zero(); nrow];
    let mut l2 = vec![T::zero(); nrow];
    // Per-panel deferred-GEMM scratch (reused across panels).
    let mut l21buf: Vec<T> = Vec::new();
    let mut gbuf: Vec<T> = Vec::new();
    let mut tmp: Vec<T> = Vec::new();
    // Per-step pivot-interchange partners of the current panel (`usize::MAX`
    // = no interchange), consumed by the deep-row replay.
    let mut deep_swaps = vec![usize::MAX; nb];
    // Time-of-step in-panel multipliers (`nb x nb`, column = step), consumed
    // by the deep-row replay (later interchanges permute the final panel's
    // multiplier rows, so the finals cannot be read back).
    let mut mult_snap = vec![T::zero(); nb * nb];
    let mut local_perturbed = 0usize;
    // Panel-lookahead state: a second scratch set for the joined next-panel
    // step, the wide-Schur staging buffer, and the high-water mark of columns
    // already factored ahead by the lookahead join.
    let mut l1b = vec![T::zero(); nrow];
    let mut l2b = vec![T::zero(); nrow];
    let mut deep_swaps_b = vec![usize::MAX; nb];
    let mut mult_snap_b = vec![T::zero(); nb * nb];
    let mut tmp_w: Vec<T> = Vec::new();
    let mut done_through = 0usize;
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + nb).min(ncol);
        if kb >= done_through {
            let r = ll_bk_panel_step(
                panel,
                nrow,
                kb,
                ke,
                nb,
                alpha,
                perturb_floor,
                ll_cdiv_par,
                &mut d,
                &mut d_subdiag,
                &mut two_by_two,
                &mut lperm,
                &mut l1,
                &mut l2,
                &mut deep_swaps,
                &mut mult_snap,
            );
            match r {
                Ok(np) => local_perturbed += np,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        // Deferred panel trailing update: panel[ke.., ke..ncol] -= L21*D*R^T, where
        // L21 = panel rows [ke,nrow) x panel cols [kb,ke) (mtxpw), G = L21*D (block-
        // diagonal D), and R = the first `cw` rows of L21 (the rows that are
        // themselves remaining panel columns [ke,ncol)). The result `tmp` is the
        // rectangular `mt x cw` Schur block; only its lower part is written back.
        let pw = ke - kb;
        let cw = ncol - ke; // remaining fully-summed columns to update
        let mt = nrow - ke; // trailing rows (left-factor height)
        if pw > 0 && cw > 0 && mt > 0 {
            l21buf.clear();
            l21buf.resize(mt * pw, T::zero());
            for cc in 0..pw {
                let c = kb + cc;
                for rr in 0..mt {
                    l21buf[rr + cc * mt] = panel[(ke + rr) + c * nrow];
                }
            }
            gbuf.clear();
            gbuf.resize(mt * pw, T::zero());
            let mut cc = 0;
            while cc < pw {
                let c = kb + cc;
                if two_by_two[c] {
                    let (d11, d21, d22) = (d[c], d_subdiag[c], d[c + 1]);
                    for rr in 0..mt {
                        let a = l21buf[rr + cc * mt];
                        let b = l21buf[rr + (cc + 1) * mt];
                        gbuf[rr + cc * mt] = a * d11 + b * d21;
                        gbuf[rr + (cc + 1) * mt] = a * d21 + b * d22;
                    }
                    cc += 2;
                } else {
                    let dc = d[c];
                    for rr in 0..mt {
                        gbuf[rr + cc * mt] = l21buf[rr + cc * mt] * dc;
                    }
                    cc += 1;
                }
            }
            // Panel lookahead: split this panel's Schur into the NARROW part
            // (the next panel's columns [ke, ke2)) and the WIDE rest
            // ([ke2, ncol)), then factor the next panel concurrently with the
            // wide GEMM - the two touch disjoint column ranges. The gate is a
            // pure function of the node shape (never of thread count or the
            // racy chain state), so the GEMM split, and therefore the bits,
            // are deterministic per matrix; `ll_thread_determinism` holds.
            let ke2 = (ke + nb).min(ncol);
            let cw_n = ke2 - ke;
            let wide = cw - cw_n;
            let look = kt.use_gemm_schur && wide > 0 && mt * wide * pw >= kt.par_cdiv;
            if look {
                // Narrow Schur into the next panel's columns.
                tmp.clear();
                tmp.resize(mt * cw_n, T::zero());
                // SAFETY: `tmp`, `gbuf`, `l21buf` are distinct allocations
                // sized for the (mt, cw_n, pw) strides.
                unsafe {
                    lower_tile_gemm(
                        &mut tmp,
                        mt,
                        cw_n,
                        pw,
                        gbuf.as_ptr(),
                        mt as isize,
                        l21buf.as_ptr(),
                        mt as isize,
                        ll_cdiv_par,
                    )
                };
                for cc2 in 0..cw_n {
                    let c = ke + cc2;
                    for rr in cc2..mt {
                        let dst = (ke + rr) + c * nrow;
                        panel[dst] = panel[dst] - tmp[rr + cc2 * mt];
                    }
                }
                // Join: next panel's getf2 + deep replay (columns [ke, ke2))
                // alongside the wide Schur (columns [ke2, ncol)).
                tmp_w.clear();
                tmp_w.resize(mt * wide, T::zero());
                let (left, right) = panel.split_at_mut(ke2 * nrow);
                let (gbuf_ref, l21_ref, tw_ref) = (&gbuf, &l21buf, &mut tmp_w);
                let (step_res, ()) = rayon::join(
                    || {
                        ll_bk_panel_step(
                            left,
                            nrow,
                            ke,
                            ke2,
                            nb,
                            alpha,
                            perturb_floor,
                            ll_cdiv_par,
                            &mut d,
                            &mut d_subdiag,
                            &mut two_by_two,
                            &mut lperm,
                            &mut l1b,
                            &mut l2b,
                            &mut deep_swaps_b,
                            &mut mult_snap_b,
                        )
                    },
                    || {
                        // SAFETY: distinct allocations; the rhs offset selects
                        // the wide columns' R rows (row = column index).
                        unsafe {
                            lower_tile_gemm(
                                tw_ref,
                                mt,
                                wide,
                                pw,
                                gbuf_ref.as_ptr(),
                                mt as isize,
                                l21_ref.as_ptr().add(cw_n),
                                mt as isize,
                                ll_cdiv_par,
                            )
                        };
                        for cc2 in cw_n..cw {
                            let c = ke + cc2;
                            let col = &mut right[(c - ke2) * nrow..(c - ke2 + 1) * nrow];
                            let tcol = &tw_ref[(cc2 - cw_n) * mt..(cc2 - cw_n + 1) * mt];
                            for rr in cc2..mt {
                                col[ke + rr] = col[ke + rr] - tcol[rr];
                            }
                        }
                    },
                );
                match step_res {
                    Ok(np) => local_perturbed += np,
                    Err(e) => {
                        return Err(e);
                    }
                }
                done_through = ke2;
            } else {
                tmp.clear();
                tmp.resize(mt * cw, T::zero());
                if kt.use_gemm_schur {
                    // The write-back below reads only `rr >= cc2`, so compute the
                    // rectangular product tile-by-tile from each tile's diagonal
                    // downward. Matters most at the tree root where `cw ~ mt`
                    // (nearly-square panel) and the full product wasted ~half its
                    // flops; for tall separator panels (`mt >> cw`) the saving is
                    // small but never negative.
                    // SAFETY: `tmp`, `gbuf`, `l21buf` are distinct allocations sized
                    // for the (mt, cw, pw) strides.
                    unsafe {
                        lower_tile_gemm(
                            &mut tmp,
                            mt,
                            cw,
                            pw,
                            gbuf.as_ptr(),
                            mt as isize,
                            l21buf.as_ptr(),
                            mt as isize,
                            ll_cdiv_par,
                        )
                    };
                } else {
                    for cc2 in 0..cw {
                        for rr in 0..mt {
                            let mut acc = T::zero();
                            for kk2 in 0..pw {
                                acc = acc + gbuf[rr + kk2 * mt] * l21buf[cc2 + kk2 * mt];
                            }
                            tmp[rr + cc2 * mt] = acc;
                        }
                    }
                }
                // Subtract the lower part: column c = ke+cc2 gets rows r = ke+rr, rr >= cc2.
                for cc2 in 0..cw {
                    let c = ke + cc2;
                    for rr in cc2..mt {
                        let dst = (ke + rr) + c * nrow;
                        panel[dst] = panel[dst] - tmp[rr + cc2 * mt];
                    }
                }
            }
        }
        kb = ke;
    }
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    // Populate the O(n) emit maps + inertia for `s` (block-aware over its 1x1/2x2
    // Bunch-Kaufman D), mirroring the legacy pass-1 emit. The `e`-numbering is one
    // position per column, so `e_offset[s] + p` is column `p`'s elimination index.
    let eoff = emit.e_offset[s];
    let (mut ipos, mut ineg, mut izero) = (0usize, 0usize, 0usize);
    let mut pp = 0;
    while pp < ncol {
        let g = sched.rows(s)[lperm[pp]] as usize;
        let e = eoff + pp;
        // SAFETY: each global index / position is written by exactly one supernode.
        unsafe {
            emit.e_of_g.set(g, e);
            emit.perm.set(e, sym.perm[g]);
            emit.d_diag.set(e, d[pp]);
        }
        if two_by_two[pp] {
            let g2 = sched.rows(s)[lperm[pp + 1]] as usize;
            unsafe {
                emit.e_of_g.set(g2, e + 1);
                emit.perm.set(e + 1, sym.perm[g2]);
                emit.d_diag.set(e + 1, d[pp + 1]);
                emit.d_subdiag.set(e, d_subdiag[pp]);
                emit.two_by_two.set(e, true);
            }
            let det_r = (d[pp] * d[pp + 1] - d_subdiag[pp] * d_subdiag[pp]).real();
            let tr_r = (d[pp] + d[pp + 1]).real();
            if det_r < 0.0 {
                ipos += 1;
                ineg += 1;
            } else if det_r > 0.0 {
                if tr_r >= 0.0 {
                    ipos += 2;
                } else {
                    ineg += 2;
                }
            } else {
                izero += 1;
                if tr_r >= 0.0 {
                    ipos += 1;
                } else {
                    ineg += 1;
                }
            }
            pp += 2;
        } else {
            let r = d[pp].real();
            if r > 0.0 {
                ipos += 1;
            } else if r < 0.0 {
                ineg += 1;
            } else {
                izero += 1;
            }
            pp += 1;
        }
    }
    emit.inertia_pos.fetch_add(ipos, Ordering::Relaxed);
    emit.inertia_neg.fetch_add(ineg, Ordering::Relaxed);
    emit.inertia_zero.fetch_add(izero, Ordering::Relaxed);
    // SAFETY: this thread owns supernode `s` and writes its cell exactly once.
    unsafe {
        store.set(
            s,
            LdltSlot {
                d,
                dsub: d_subdiag,
                two: two_by_two,
                lperm,
            },
        )
    };
    Ok(())
}

/// Static-pivot floor (absolute), translated from rslab's ZeroPivotAction.
/// `PerturbToEps { abs_floor }` is taken as given (rslab convention: an
/// absolute floor, typically `eps_rel * ||A||inf`); `Fail` disables
/// perturbation. `a_perm` holds the values being factored.
fn static_pivot_floor<T: Scalar>(a_perm: &CscMatrix<T>, opts: &SolverSettings) -> Option<f64> {
    match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a_perm
                .values
                .iter()
                .map(|v| v.magnitude())
                .fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    }
}

/// Supernodal **left-looking** LDL^T with **Bunch-Kaufman 1x1/2x2 pivoting**. Each
/// supernode's dense panel is assembled from `A`, updated by every previously
/// factored descendant (`cmod`: pull the descendant's contribution columns that
/// land in this panel, applying its block-diagonal `D`), then factored in place
/// (`cdiv`: partial Bunch-Kaufman, no trailing update). Pivoting is bounded to
/// each panel's fully-summed block, so the off-diagonal rows keep their identity
/// and the descendant->ancestor `cmod` is unaffected by a panel's internal
/// permutation. There is **no contribution-block stack and no extract copy-out**
/// (the panels are the factor), so the transient is just the factor itself (the
/// PARDISO memory profile), including indefinite (zero-/tiny-diagonal) systems
/// via the 2x2 blocks.
fn factor_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    a_perm: CscMatrix<T>,
    opts: &SolverSettings,
) -> Result<LdltNumeric<T>, RslabError> {
    let n = sym.n;
    let perturb_floor = static_pivot_floor(&a_perm, opts);

    let nsuper = sym.supernodes.len();
    // Factor in parallel over the assembly forest: sibling subtrees concurrently,
    // each node after its subtree (whose panels are its only updaters). Panels are
    // written once and read only by ancestors -> no synchronization needed beyond
    // the recursion structure (see `LlStore`).
    let store = LlStore::<T>::new(nsuper);
    let emit = LlEmitLdlt::<T>::new(sym, sched);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let kt = opts.kernel();
    let factor_node = |s: usize| {
        ll_factor_node(
            s,
            sym,
            &a_perm,
            sched,
            &store,
            &emit,
            perturb_floor,
            &n_perturbed_atomic,
            kt,
        )
    };
    let emit_free = |k: usize| ldlt_emit_and_free(k, &store, &emit, sym, sched, opts.drop_tol);
    crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free)?;
    drop(store); // panels moved into the emit cells; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
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
        ..
    } = emit;
    let (factor, n_zeros) = arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(panels.get_mut(s))
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *perm.get(e) }).collect();
    let d_diag: Vec<T> = (0..n).map(|e| unsafe { *d_diag.get(e) }).collect();
    let d_subdiag: Vec<T> = (0..n).map(|e| unsafe { *d_subdiag.get(e) }).collect();
    let two_by_two: Vec<bool> = (0..n).map(|e| unsafe { *two_by_two.get(e) }).collect();
    let inertia = Inertia::new(
        inertia_pos.load(Ordering::Relaxed),
        inertia_neg.load(Ordering::Relaxed),
        inertia_zero.load(Ordering::Relaxed),
    );

    Ok(LdltNumeric {
        factor,
        d_diag,
        d_subdiag,
        two_by_two,
        perm,
        supernode_parent,
        n_perturbed,
        n_zeros,
        inertia,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::ldlt_generic::solve_ldlt;
    use crate::symbolic::OrderingMethod;
    use num_complex::Complex;

    /// A tridiagonal (chain) matrix with no amalgamation (`nemin = 1`) builds an
    /// assembly tree as deep as the matrix; the recursive tree factorization must
    /// not overflow the worker stack on either path. Regression for the
    /// `STATUS_STACK_OVERFLOW` the auto-tuning sweep hit on banded + `nemin = 1`.
    #[test]
    fn deep_chain_tree_does_not_overflow_stack() {
        let n = 20_000usize;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(4.0f64);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let s = SolverSettings::default().with_nemin(1).with_threads(0);
        let f = factor_sparse_ldlt_with(&a, &s).expect("deep chain factors without overflow");
        assert_eq!(f.n, n);
    }

    #[test]
    fn rcm_and_autorace_orderings_factor_and_solve() {
        // A 2D-grid SPD system must factor and solve correctly under the new RCM
        // ordering and under AutoRace (which now includes RCM as a candidate).
        let m = 14;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        for ord in [OrderingMethod::Rcm, OrderingMethod::AutoRace] {
            let opts = SolverSettings::default().with_ordering(ord);
            let symb = analyze_with(a.n, &a.col_ptr, &a.row_idx, &opts).unwrap();
            let f = factor_numeric(&symb, &a, None, &opts)
                .unwrap()
                .into_factors();
            let x = solve_ldlt(&f, &b).unwrap();
            assert!(
                residual_inf(&a, &x, &b) < 1e-9,
                "ordering {ord:?} residual {}",
                residual_inf(&a, &x, &b)
            );
        }
    }

    fn residual_inf<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    /// 1D Laplacian-style SPD tridiagonal of size n (diag 2+something, off -1).
    fn tridiag_spd_f64(n: usize) -> CscMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(4.0);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(-1.0);
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn f64_sparse_tridiag_residual() {
        let a = tridiag_spd_f64(20);
        let b: Vec<f64> = (0..20).map(|i| (i as f64) - 9.5).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-10);
    }

    /// 2D 5-point grid (mxm), lower triangle, complex-symmetric, diagonally
    /// dominant. Branching assembly tree -> exercises multi-child `cmod`.
    fn grid2d_lower<T: Scalar>(m: usize, diag: T, off: T) -> CscMatrix<T> {
        let n = m * m;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |r: usize, c: usize| r * m + c;
        let mut push = |i: usize, j: usize, v: T| {
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            rows.push(hi);
            cols.push(lo);
            vals.push(v);
        };
        for r in 0..m {
            for c in 0..m {
                let p = idx(r, c);
                push(p, p, diag);
                if c + 1 < m {
                    push(p, idx(r, c + 1), off);
                }
                if r + 1 < m {
                    push(p, idx(r + 1, c), off);
                }
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn left_looking_tridiagonal_solves() {
        // Chain assembly tree (tridiagonal): exercises the basic left-looking
        // cmod/cdiv.
        let a = tridiag_spd_f64(50);
        let b: Vec<f64> = (0..50).map(|i| (i % 7) as f64 - 3.0).collect();
        let ll = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(residual_inf(&a, &xl, &b) < 1e-9, "left-looking residual");
    }

    #[test]
    fn left_looking_2d_grid_solves() {
        // Branching assembly tree -> multi-child cmod and deeper update lists.
        let a = grid2d_lower::<f64>(12, 8.0, -1.0);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        let ll = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "left-looking grid residual"
        );
    }

    #[test]
    fn left_looking_complex_symmetric_type_agnostic() {
        // The left-looking path is generic over `Scalar`: complex-symmetric here.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let a = grid2d_lower::<Complex<f64>>(10, c(8.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
        let ll = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "complex left-looking residual"
        );
    }

    #[test]
    fn left_looking_indefinite_2x2_inertia() {
        // [[0,1],[1,0]] (eigenvalues +/-1) forces a single 2x2 Bunch-Kaufman block.
        // The left-looking path must take that 2x2 (zero diagonal -> no 1x1 pivot)
        // and report inertia (1+, 1-).
        let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 0], &[0.0, 1.0]).unwrap();
        let ll = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        assert!(ll.two_by_two.iter().any(|&t| t), "expected a 2x2 block");
        assert_eq!(
            (ll.inertia.positive, ll.inertia.negative, ll.inertia.zero),
            (1, 1, 0)
        );
        let b = [1.0_f64, -2.0];
        let x = solve_ldlt(&ll, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-12, "2x2 residual");
    }

    #[test]
    fn left_looking_indefinite_2d_grid_solves() {
        // 2D 5-point grid with a *small* diagonal (0.5 << 2*|off|): far from
        // diagonally dominant -> genuinely indefinite, so Bunch-Kaufman must take
        // many 2x2 pivots across several supernodes and still give a true solve -
        // the exact indefinite EM-FEM case the 2x2 pivoting is for.
        let a = grid2d_lower::<f64>(10, 0.5, -1.0);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let ll = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        assert!(
            ll.two_by_two.iter().filter(|&&t| t).count() > 0,
            "indefinite system should use 2x2 pivots"
        );
        assert!(
            ll.inertia.negative > 0 && ll.inertia.positive + ll.inertia.negative == n,
            "indefinite, nonsingular inertia"
        );
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "left-looking indefinite residual"
        );
    }

    #[test]
    fn left_looking_indefinite_complex_symmetric() {
        // Complex-symmetric indefinite grid: the 2x2 path is type-agnostic. The
        // 2x2 blocks here are complex-symmetric (not Hermitian), exercising the
        // generic det/detinv arithmetic.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let a = grid2d_lower::<Complex<f64>>(9, c(0.4, 0.3), c(-1.0, 0.1));
        let n = a.n;
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
        let ll = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        assert!(
            ll.two_by_two.iter().filter(|&&t| t).count() > 0,
            "indefinite system should use 2x2 pivots"
        );
        assert_eq!(
            ll.inertia.positive + ll.inertia.negative + ll.inertia.zero,
            n,
            "inertia covers every pivot"
        );
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "complex left-looking indefinite residual"
        );
    }

    #[test]
    fn f64_dense_front_blocked_multi_panel() {
        // A fully dense symmetric matrix is one front of width n=100 > NB(64),
        // so factoring it exercises the blocked **multi-panel** Bunch-Kaufman
        // path (which the small n<=50 tests never reach). Diagonally dominant SPD.
        let n = 100;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            for i in j..n {
                rows.push(i);
                cols.push(j);
                vals.push(if i == j {
                    n as f64 + 1.0
                } else {
                    ((i + 2 * j) % 5) as f64 - 2.0
                });
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn complex_dense_front_blocked_multi_panel() {
        // Dense complex-symmetric, one front of width 90 > NB -> multi-panel.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 90;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            for i in j..n {
                rows.push(i);
                cols.push(j);
                vals.push(if i == j {
                    c(n as f64, 1.0)
                } else {
                    c(((i + 3 * j) % 5) as f64 - 2.0, 0.2)
                });
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b = vec![c(1.0, 0.5); n];
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-9);
    }

    #[test]
    fn f64_sparse_2d_grid_residual() {
        // 2D 5-point Laplacian on a 5x5 grid (n=25), SPD.
        let m = 5;
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let idx = |r: usize, c: usize| r * m + c;
        for r in 0..m {
            for c in 0..m {
                let p = idx(r, c);
                rows.push(p);
                cols.push(p);
                vals.push(4.0);
                // lower-triangle neighbors only
                if c + 1 < m {
                    let q = idx(r, c + 1);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(-1.0);
                }
                if r + 1 < m {
                    let q = idx(r + 1, c);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(-1.0);
                }
            }
        }
        let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn complex_sparse_tridiag_residual() {
        // Complex-symmetric Helmholtz-style tridiagonal: diagonal (4 + 0.5i),
        // off-diagonal (-1 + 0.1i). Complex symmetric (A = A^T), diagonally
        // dominant so the fully-summed blocks stay nonsingular.
        let c = |re, im| Complex::new(re, im);
        let n = 16;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(c(4.0, 0.5));
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(c(-1.0, 0.1));
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 7.5, 1.0 - i as f64)).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-10,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn complex_sparse_large_grid_parallel() {
        // 12x12 complex-symmetric grid (n=144): a deep, bushy assembly tree
        // that genuinely exercises multiple parallel levels in the rayon driver.
        let c = |re, im| Complex::new(re, im);
        let m = 12;
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let idx = |r: usize, cc: usize| r * m + cc;
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 0.5));
                if cc + 1 < m {
                    let q = idx(r, cc + 1);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.1));
                }
                if r + 1 < m {
                    let q = idx(r + 1, cc);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.1));
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 11) as f64 - 5.0, 1.0)).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn perturb_rescues_singular_complex() {
        // Structurally singular complex-symmetric system: index 1 is fully
        // decoupled with a zero diagonal (zero row/column). Exact mode must
        // fail; static-pivoting (preconditioner) mode must succeed, report a
        // perturbation, and produce a finite, solvable factor of `A + E`.
        let c = |re, im| Complex::new(re, im);
        let n = 3;
        let rows = vec![0, 2, 1];
        let cols = vec![0, 0, 1];
        let vals = vec![c(2.0, 1.0), c(-1.0, 0.3), c(0.0, 0.0)];
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();

        assert!(
            factor_sparse_ldlt(&a).is_err(),
            "exact mode should reject the singular pivot"
        );

        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
            ..Default::default()
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert!(
            f.n_perturbed >= 1,
            "expected >=1 perturbation, got {}",
            f.n_perturbed
        );
        let b = vec![c(1.0, 0.0); n];
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            x.iter().all(|v| v.norm().is_finite()),
            "factor must stay finite"
        );
    }

    #[test]
    fn exact_mode_never_perturbs_well_conditioned() {
        // A diagonally dominant complex-symmetric grid factors exactly with no
        // perturbation - the static-pivot path must not trigger spuriously.
        let a = {
            let c = |re, im| Complex::new(re, im);
            let n = 16;
            let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
            for j in 0..n {
                r.push(j);
                cc.push(j);
                v.push(c(4.0, 0.5));
                if j + 1 < n {
                    r.push(j + 1);
                    cc.push(j);
                    v.push(c(-1.0, 0.1));
                }
            }
            CscMatrix::<Complex<f64>>::from_triplets(n, &r, &cc, &v).unwrap()
        };
        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
            ..Default::default()
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert_eq!(
            f.n_perturbed, 0,
            "well-conditioned matrix needs no perturbation"
        );
    }

    #[test]
    fn complex_sparse_2d_grid_residual() {
        // 2D complex-symmetric grid: diagonal (4 + i), neighbor (-1 + 0.2i).
        let c = |re, im| Complex::new(re, im);
        let m = 5;
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let idx = |r: usize, cc: usize| r * m + cc;
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 1.0));
                if cc + 1 < m {
                    let q = idx(r, cc + 1);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.2));
                }
                if r + 1 < m {
                    let q = idx(r + 1, cc);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.2));
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }
}
