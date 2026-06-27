//! Generic multifrontal sparse LDLᵀ factorization over any [`Scalar`] field.
//!
//! This drives a full sparse symmetric-indefinite solve for both the real
//! (`f64`) and complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) paths by
//! reusing the existing **value-agnostic** symbolic analysis (ordering,
//! elimination tree, supernode amalgamation) and applying the generic dense
//! Bunch-Kaufman kernel from [`crate::dense::ldlt_generic`] front-by-front.
//!
//! This is the single, data-type-generic symmetric multifrontal driver (the
//! former f64-dedicated driver has been removed). It is rayon-parallel with a
//! `gemm` BLAS-3 Schur update and relaxed amalgamation; delayed pivoting and
//! the remaining feral robustness features are being ported in.
//!
//! ## Current pivoting scope
//!
//! * Pivoting is restricted to the **fully-summed block** of each front (no
//!   delayed pivoting). This produces a valid factorization whenever each
//!   fully-summed block is nonsingular; pathological indefinite cases that
//!   would require delaying a pivot to the parent are out of scope for now and
//!   surface as [`FeralError::NumericallyRankDeficient`].
//! * The reassembled factor is held as a dense `n×n` global `L`. This is
//!   `O(n²)` memory and is a correctness-first choice; a sparse-CSC global `L`
//!   with a supernodal triangular solve is a later optimization.
//!
//! The result is returned as an [`LdltFactors`] in factorization order, so the
//! generic [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt) handles the
//! triangular/diagonal solves and permutation directly.

use crate::dense::ldlt_generic::{bk_alpha, swap_sym_lower, LdltFactors};
use crate::error::FeralError;
use crate::inertia::Inertia;
use crate::scalar::Scalar;

/// Scale-invariant singularity floor for a 2×2 Bunch-Kaufman pivot: a block
/// whose `|det|` falls below `GROWTH_EPS · scale²` (scale = the largest block
/// entry magnitude) is numerically singular — rejected in exact mode and lifted
/// in static-pivot mode. Bounds the element growth `1/|det|` can otherwise
/// inject into the trailing update.
const GROWTH_EPS: f64 = 1e-14;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::{symbolic_factorize, SupernodeParams, SymbolicFactorization};
use rayon::prelude::*;

/// Action to take when a near-zero pivot is encountered during factorization.
///
/// This is the static-pivoting policy knob shared by the symmetric LDLᵀ and the
/// unsymmetric LU paths (via [`FactorOptions`] and the LU options).
#[derive(Debug, Clone)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot at face value (zero the column, count as a zero in
    /// the inertia signature, flag for iterative refinement). The perturbation
    /// magnitude is unbounded — use only when downstream code tolerates sign
    /// loss in the perturbed positions and re-checks inertia.
    ForceAccept,
    /// Return [`FeralError::NumericallyRankDeficient`].
    Fail,
    /// Replace the tiny pivot with `sign(d) · max(|d|, abs_floor)`, keeping the
    /// column live (LAPACK / MA57-style static pivoting). The factor satisfies
    /// `L·D·Lᵀ = A + Δ` for the produced `L`, `D`; `Δ` is bounded in the worst
    /// case by `‖A[:,k]‖² / abs_floor`, so drive iterative refinement against
    /// the unperturbed `A` for tight tolerances. A typical recipe is
    /// `abs_floor = eps_rel · ‖A‖∞` with `eps_rel ∈ [1e-12, 1e-8]`.
    PerturbToEps { abs_floor: f64 },
}
use std::sync::atomic::{AtomicBool, Ordering};

/// Diagnostic toggle for the contribution-block Schur update kernel. When
/// `true` (default) the deferred update `CB = A22 − L21·D·L21ᵀ` runs as a
/// single SIMD GEMM ([`gemm`]); when `false` the identical update runs as a
/// scalar triple loop over the same `l21`/`g`/`cb` buffers. Both paths produce
/// the same factor — this exists only to A/B the kernel, mirroring feral's
/// `FORCE_SCALAR_FRONTAL`.
pub(crate) static USE_GEMM_SCHUR: AtomicBool = AtomicBool::new(true);

/// Set the contribution-block kernel: `true` = SIMD GEMM, `false` = scalar.
/// Process-wide; intended for benchmarks and tests, not the solve path.
pub fn set_use_gemm_schur(on: bool) {
    USE_GEMM_SCHUR.store(on, Ordering::Relaxed);
}

/// Liu (1986) contribution-stack minimization, applied at analysis time by
/// reordering each supernode's children. When `true` (default) it shrinks the
/// transient contribution-block stack — the dominant factorization peak-RSS
/// driver on large fronts — at a modest factor-throughput cost (the
/// memory-optimal child order is not always the parallel-load-optimal one).
/// Set `false` to recover maximum throughput when memory is not the constraint.
pub(crate) static USE_LIU_REORDER: AtomicBool = AtomicBool::new(true);

/// Toggle Liu child-reordering (memory-light vs max-throughput). Process-wide.
pub fn set_use_liu_reorder(on: bool) {
    USE_LIU_REORDER.store(on, Ordering::Relaxed);
}

/// Options controlling the generic multifrontal factorization. Defaults give an
/// **exact** complete factorization that fails on rank deficiency. Relaxing
/// them turns the factorization into a robust, memory-light **preconditioner**.
#[derive(Debug, Clone)]
pub struct FactorOptions {
    /// Near-zero pivot policy. Reuses feral's [`ZeroPivotAction`]: `Fail`
    /// (exact, default) returns [`FeralError::NumericallyRankDeficient`] on a
    /// singular pivot; `PerturbToEps { abs_floor }` is robust static pivoting —
    /// a pivot below `abs_floor` is lifted to that floor (the
    /// complex-symmetric analogue of feral's f64 `perturb_to_floor`), so the
    /// factorization never fails and produces `L D Lᵀ = A + E` for small `E`.
    /// That is exactly the never-fail behaviour a preconditioner needs.
    pub on_zero_pivot: ZeroPivotAction,
    /// Threshold dropping for incomplete factorization. When `Some(tau)`, fill
    /// entries of `L` with magnitude below `tau` (relative to the column) are
    /// discarded, trading factor accuracy for memory. `None` = complete
    /// factorization. (Wired in a later stage.)
    pub drop_tol: Option<f64>,
}

impl Default for FactorOptions {
    fn default() -> Self {
        Self {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: None,
        }
    }
}

impl FactorOptions {
    /// Exact, complete factorization (the default): fail on a singular pivot,
    /// no fill dropping. Use for a direct solve where accuracy is required.
    pub fn exact() -> Self {
        Self::default()
    }

    /// Robust never-fail **preconditioner** mode: static pivoting replaces any
    /// pivot below `abs_floor` (typically `eps_rel·‖A‖`) so the factorization
    /// always succeeds. Compose with [`with_drop_tol`](Self::with_drop_tol) for
    /// an incomplete preconditioner.
    pub fn preconditioner(abs_floor: f64) -> Self {
        Self {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor },
            drop_tol: None,
        }
    }

    /// Builder: enable incomplete-factor threshold dropping (`|fill| < tau` is
    /// discarded, relative to the column/row).
    pub fn with_drop_tol(mut self, tau: f64) -> Self {
        self.drop_tol = Some(tau);
        self
    }

    /// Builder: set the near-zero pivot policy.
    pub fn with_pivot(mut self, policy: ZeroPivotAction) -> Self {
        self.on_zero_pivot = policy;
        self
    }
}

/// Static-pivot perturbation, the complex-symmetric analogue of feral's f64
/// `perturb_to_floor` (`dense::factor`): lift a pivot whose magnitude is below
/// `abs_floor` up to that floor, preserving phase. For `T = f64` this reduces
/// to `sign(d)·max(|d|, abs_floor)`, matching the real kernel.
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

/// Per-front partial-factorization output, in within-front pivot order.
struct FrontFactors<T> {
    /// Total front size (eliminated + contribution rows).
    nrow: usize,
    /// Number of eliminated (fully-summed) columns.
    nelim: usize,
    /// Pivot position → local row index (length `nrow`). Identity on the
    /// contribution rows `[nelim, nrow)`, which are never interchanged.
    perm: Vec<usize>,
    /// Unit lower `L` of the front, `nrow × nelim` column-major in pivot order.
    l: Vec<T>,
    /// `D` block diagonal, length `nelim`.
    d_diag: Vec<T>,
    /// `D` sub-diagonal, length `nelim`.
    d_subdiag: Vec<T>,
    /// `true` at the first column of each 2×2 block, length `nelim`.
    two_by_two: Vec<bool>,
    /// Number of pivots statically perturbed in this front.
    n_perturbed: usize,
    /// Inertia (signs of `D`) over this front's eliminated pivots. Exact for a
    /// real symmetric matrix; advisory (pivot real-part signs) for complex.
    inertia: Inertia,
}

/// Partially factor the first `ncol` (fully-summed) columns of a dense
/// lower-triangle front `f` (`nrow × nrow`, column-major) with Bunch-Kaufman
/// pivoting restricted to the fully-summed block. The entire trailing front is
/// updated; the trailing `[ncol, nrow)` block is returned as the contribution
/// block (`cnrow × cnrow` column-major lower triangle).
fn factor_front<T: Scalar>(
    f: &mut [T],
    nrow: usize,
    ncol: usize,
    perturb_floor: Option<f64>,
) -> Result<(FrontFactors<T>, Vec<T>), FeralError> {
    let n = nrow; // column stride
    let alpha = bk_alpha();
    let one = T::one();

    let mut perm: Vec<usize> = (0..nrow).collect();
    let mut d_diag = vec![T::zero(); ncol];
    let mut d_subdiag = vec![T::zero(); ncol];
    let mut two_by_two = vec![false; ncol];
    let mut n_perturbed = 0usize;
    let mut inertia = Inertia::new(0, 0, 0);
    // Reusable 2×2-pivot multiplier scratch, hoisted out of the pivot loop so an
    // indefinite front with many 2×2 blocks does not allocate per pivot. Only
    // entries `[k+2, n)` are ever written/read each step, so stale values left
    // below are never observed.
    let mut l1 = vec![T::zero(); nrow];
    let mut l2 = vec![T::zero(); nrow];
    // Per-panel trailing-GEMM scratch (reused across panels).
    let mut l21buf: Vec<T> = Vec::new();
    let mut gbuf: Vec<T> = Vec::new();
    let mut tmp: Vec<T> = Vec::new();

    // Blocked Bunch-Kaufman: factor the fully-summed columns in panels of width
    // `NB` with pivoting **bounded to the panel**, deferring each panel's
    // trailing Schur update to one SIMD GEMM (the BLAS-3 bulk, replacing the
    // scalar BLAS-2 column sweeps that dominated large fronts). The last column
    // of a panel has no in-panel candidate below it, so it is always a 1×1 step
    // — a 2×2 block can never straddle a panel boundary.
    const NB: usize = 64;
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + NB).min(ncol);
        let mut k = kb;
        while k < ke {
            let absakk = f[k * n + k].magnitude();

            // colmax restricted to the in-panel rows (k+1)..ke.
            let mut colmax_sq = 0.0;
            let mut imax = k;
            for i in (k + 1)..ke {
                let m = f[k * n + i].magnitude_sq();
                if m > colmax_sq {
                    colmax_sq = m;
                    imax = i;
                }
            }
            let colmax = colmax_sq.sqrt();

            let kstep;
            let kp;
            if absakk.max(colmax) == 0.0 {
                // Fully zero pivot column. Exact mode fails; static-pivot mode
                // takes a 1×1 step and lets the perturbation below lift the zero
                // diagonal up to the floor.
                if perturb_floor.is_none() {
                    return Err(FeralError::NumericallyRankDeficient);
                }
                kstep = 1;
                kp = k;
            } else if absakk >= alpha * colmax {
                kstep = 1;
                kp = k;
            } else {
                // rowmax in row imax, restricted to the fully-summed block (squared
                // domain, single final sqrt).
                let mut rowmax_sq = 0.0;
                for j in k..imax {
                    let m = f[j * n + imax].magnitude_sq();
                    if m > rowmax_sq {
                        rowmax_sq = m;
                    }
                }
                for i in (imax + 1)..ke {
                    let m = f[imax * n + i].magnitude_sq();
                    if m > rowmax_sq {
                        rowmax_sq = m;
                    }
                }
                let rowmax = rowmax_sq.sqrt();
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kstep = 1;
                    kp = k;
                } else if f[imax * n + imax].magnitude() >= alpha * rowmax {
                    kstep = 1;
                    kp = imax;
                } else {
                    kstep = 2;
                    kp = imax;
                }
            }

            if kstep == 1 {
                if kp != k {
                    swap_sym_lower(f, n, k, kp);
                    perm.swap(k, kp);
                }
                let mut d = f[k * n + k];
                match perturb_floor {
                    Some(floor) if d.magnitude() < floor => {
                        d = perturb_pivot(d, floor);
                        f[k * n + k] = d;
                        n_perturbed += 1;
                    }
                    None if d == T::zero() => return Err(FeralError::NumericallyRankDeficient),
                    _ => {}
                }
                d_diag[k] = d;
                // Inertia: sign of the 1×1 pivot (real part).
                let r = d.real();
                if r > 0.0 {
                    inertia.positive += 1;
                } else if r < 0.0 {
                    inertia.negative += 1;
                } else {
                    inertia.zero += 1;
                }
                let dinv = d.recip();
                // Update only the in-panel trailing columns `(k+1)..ke` (across all
                // rows, so the panel's L21 multiplier rows are formed). The columns
                // beyond `ke` are deferred to this panel's trailing GEMM.
                for j in (k + 1)..ke {
                    let wj_dinv = f[k * n + j] * dinv;
                    if wj_dinv != T::zero() {
                        for i in j..n {
                            f[j * n + i] = f[j * n + i] - f[k * n + i] * wj_dinv;
                        }
                    }
                }
                for i in (k + 1)..n {
                    f[k * n + i] = f[k * n + i] * dinv;
                }
                k += 1;
            } else {
                if kp != k + 1 {
                    swap_sym_lower(f, n, k + 1, kp);
                    perm.swap(k + 1, kp);
                }
                let mut d11 = f[k * n + k];
                let d21 = f[k * n + (k + 1)];
                let mut d22 = f[(k + 1) * n + (k + 1)];
                let mut det = d11 * d22 - d21 * d21;
                // Scale-invariant singularity / growth guard: a 2×2 whose `|det|`
                // is below `GROWTH_EPS · scale²` would inject `1/|det|` growth into
                // the trailing update. `scale` is the largest block-entry magnitude.
                let scale = d11.magnitude().max(d22.magnitude()).max(d21.magnitude());
                let growth_floor = GROWTH_EPS * scale * scale;
                // Static-pivot the 2×2 when its determinant is near-singular. The
                // real kernel (feral's `perturb_2x2_to_floor`) shifts the small
                // eigenvalue; for complex-symmetric blocks the eigenvalues are
                // complex, so we shift both diagonals by the floor (lifting |det|)
                // and, as a last resort, nudge det itself — enough to keep the
                // preconditioner factor live.
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
                            n_perturbed += 1;
                        }
                    }
                    None if det.magnitude() <= growth_floor => {
                        return Err(FeralError::NumericallyRankDeficient)
                    }
                    _ => {}
                }
                let detinv = det.recip();
                d_diag[k] = d11;
                d_subdiag[k] = d21;
                d_diag[k + 1] = d22;
                two_by_two[k] = true;
                // Inertia of the 2×2 block from det / trace (real parts): det<0 →
                // one +, one −; det>0 → two of sign(trace); det≈0 → one 0, one
                // sign(trace).
                let det_r = det.real();
                let tr_r = (d11 + d22).real();
                if det_r < 0.0 {
                    inertia.positive += 1;
                    inertia.negative += 1;
                } else if det_r > 0.0 {
                    if tr_r >= 0.0 {
                        inertia.positive += 2;
                    } else {
                        inertia.negative += 2;
                    }
                } else {
                    inertia.zero += 1;
                    if tr_r >= 0.0 {
                        inertia.positive += 1;
                    } else {
                        inertia.negative += 1;
                    }
                }

                for i in (k + 2)..n {
                    let wik = f[k * n + i];
                    let wik1 = f[(k + 1) * n + i];
                    l1[i] = (d22 * wik - d21 * wik1) * detinv;
                    l2[i] = (d11 * wik1 - d21 * wik) * detinv;
                }
                for j in (k + 2)..ke {
                    let l1j = l1[j];
                    let l2j = l2[j];
                    for i in j..n {
                        f[j * n + i] = f[j * n + i] - f[k * n + i] * l1j - f[(k + 1) * n + i] * l2j;
                    }
                }
                for i in (k + 2)..n {
                    f[k * n + i] = l1[i];
                    f[(k + 1) * n + i] = l2[i];
                }
                k += 2;
            }
        }

        // Deferred panel trailing update: f[ke.., ke..] −= L21·D·L21ᵀ. Build the
        // panel's L21 (trailing rows × panel cols) and G = L21·D (block-diagonal
        // D), GEMM into a temp, then subtract its lower triangle into `f`.
        let pw = ke - kb;
        let mt = n - ke;
        if mt > 0 && pw > 0 {
            l21buf.clear();
            l21buf.resize(mt * pw, T::zero());
            for (cc, c) in (kb..ke).enumerate() {
                for (rr, r) in (ke..n).enumerate() {
                    l21buf[cc * mt + rr] = f[c * n + r];
                }
            }
            gbuf.clear();
            gbuf.resize(mt * pw, T::zero());
            let mut c = kb;
            while c < ke {
                let cc = c - kb;
                if two_by_two[c] {
                    let (d11, d21, d22) = (d_diag[c], d_subdiag[c], d_diag[c + 1]);
                    for rr in 0..mt {
                        let a = l21buf[cc * mt + rr];
                        let b = l21buf[(cc + 1) * mt + rr];
                        gbuf[cc * mt + rr] = a * d11 + b * d21;
                        gbuf[(cc + 1) * mt + rr] = a * d21 + b * d22;
                    }
                    c += 2;
                } else {
                    let d = d_diag[c];
                    for rr in 0..mt {
                        gbuf[cc * mt + rr] = l21buf[cc * mt + rr] * d;
                    }
                    c += 1;
                }
            }
            tmp.clear();
            tmp.resize(mt * mt, T::zero());
            let par = if (mt as u128) * (mt as u128) * (pw as u128) >= 8_000_000 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            if USE_GEMM_SCHUR.load(Ordering::Relaxed) {
                // SAFETY: `tmp`, `gbuf`, `l21buf` are three distinct,
                // non-overlapping allocations sized for (m,n,k)=(mt,mt,pw) under
                // the strides passed; `T` is `f64`/`Complex<f64>` (gemm-supported).
                unsafe {
                    gemm::gemm(
                        mt,
                        mt,
                        pw,
                        tmp.as_mut_ptr(),
                        mt as isize,
                        1,
                        false,
                        gbuf.as_ptr(),
                        mt as isize,
                        1,
                        l21buf.as_ptr(),
                        1,
                        mt as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        par,
                    );
                }
            } else {
                for jj in 0..mt {
                    for ii in jj..mt {
                        let mut acc = T::zero();
                        for cc in 0..pw {
                            acc = acc + gbuf[cc * mt + ii] * l21buf[cc * mt + jj];
                        }
                        tmp[jj * mt + ii] = acc;
                    }
                }
            }
            for jj in 0..mt {
                let cj = ke + jj;
                for ii in jj..mt {
                    let ri = ke + ii;
                    f[cj * n + ri] = f[cj * n + ri] - tmp[jj * mt + ii];
                }
            }
        }
        kb = ke;
    }

    // Extract the front's L (nrow × ncol, pivot order).
    let mut l = vec![T::zero(); nrow * ncol];
    let mut c = 0;
    while c < ncol {
        if two_by_two[c] {
            l[c * nrow + c] = one;
            l[(c + 1) * nrow + (c + 1)] = one;
            for i in (c + 2)..nrow {
                l[c * nrow + i] = f[c * nrow + i];
                l[(c + 1) * nrow + i] = f[(c + 1) * nrow + i];
            }
            c += 2;
        } else {
            l[c * nrow + c] = one;
            for i in (c + 1)..nrow {
                l[c * nrow + i] = f[c * nrow + i];
            }
            c += 1;
        }
    }

    // Contribution block CB = A22 − L21·D·L21ᵀ. The per-panel trailing GEMMs
    // above already applied the whole Schur update into `f`'s trailing
    // `[ncol, nrow)²` lower triangle, so extract it directly (mirrored to both
    // triangles for the parent's extend-add).
    let cnrow = nrow - ncol;
    let mut cb = vec![T::zero(); cnrow * cnrow];
    for j in 0..cnrow {
        for i in j..cnrow {
            let v = f[(ncol + j) * n + (ncol + i)];
            cb[j * cnrow + i] = v;
            cb[i * cnrow + j] = v;
        }
    }

    Ok((
        FrontFactors {
            nrow,
            nelim: ncol,
            perm,
            l,
            d_diag,
            d_subdiag,
            two_by_two,
            n_perturbed,
            inertia,
        },
        cb,
    ))
}

/// Reassembled per-front factor, retained for the global pass.
struct NodeFactor<T> {
    front: FrontFactors<T>,
    row_indices: Vec<usize>,
    /// This front's contribution block (`cnrow × cnrow` column-major lower
    /// triangle), consumed by the parent's extend-add. Kept on the node (rather
    /// than a separate take-able slot) so independent subtrees factor in
    /// parallel without a shared mutable contribution pool.
    contrib: Vec<T>,
}

thread_local! {
    /// Per-worker global→front-local index scratch (`usize`, scalar-independent),
    /// reused across every front a thread factors and held at the all-`usize::MAX`
    /// invariant between uses. Replaces the old `map_init` workspace now that the
    /// driver is a work-stealing tree recursion rather than a level `par_iter`.
    static GLOC_SCRATCH: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A supernode's own factor plus the flat `(supernode-id, factor)` list for the
/// rest of its subtree — the return shape of [`factor_subtree`].
type SubtreeFactors<T> = (NodeFactor<T>, Vec<(usize, NodeFactor<T>)>);

/// Factor one supernode's front: build its row structure, assemble the original
/// (permuted) entries and the children's contribution blocks, then partially
/// factor the fully-summed columns. Reads only already-computed children, so
/// supernodes on the same assembly-tree level run concurrently.
fn factor_one_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    child_refs: &[&NodeFactor<T>],
    perturb_floor: Option<f64>,
) -> Result<NodeFactor<T>, FeralError> {
    let snode = &sym.supernodes[s];
    let n = sym.n;
    let ncol = snode.ncol;
    let own_last = snode.first_col + ncol;

    // Front row structure: own columns ++ sorted trailing rows (from the
    // permuted pattern of the own columns plus the children contribution rows).
    let mut trailing: Vec<usize> = Vec::new();
    for j in snode.first_col..own_last {
        for k in sym.permuted_pattern.col_ptr[j]..sym.permuted_pattern.col_ptr[j + 1] {
            let r = sym.permuted_pattern.row_idx[k];
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    for child in child_refs {
        for &r in &child.row_indices[child.front.nelim..] {
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    trailing.sort_unstable();
    trailing.dedup();
    let mut ri = Vec::with_capacity(ncol + trailing.len());
    ri.extend(snode.first_col..own_last);
    ri.extend(trailing);
    let nrow = ri.len();

    // Front buffer (transient — the unavoidable nrow² zeroing dominates, so a
    // per-front allocation adds only negligible malloc over a pooled one).
    let mut fbuf: Vec<T> = vec![T::zero(); nrow * nrow];
    let f = &mut fbuf[..];

    // Take the thread-local global→local scratch (held at all-`usize::MAX`) for
    // the assembly; returned before `factor_front` so the front GEMM's
    // work-stealing tasks can never re-enter the borrow.
    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, usize::MAX);
    }
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li;
    }

    // Scatter original entries of the eliminated columns.
    for p in 0..ncol {
        let c = snode.first_col + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let g = a_perm.row_idx[k];
            let lr = gloc[g];
            debug_assert!(lr != usize::MAX, "original entry outside front");
            let (hi, lo) = if lr >= p { (lr, p) } else { (p, lr) };
            f[lo * nrow + hi] = f[lo * nrow + hi] + a_perm.values[k];
        }
    }

    // Extend-add each child's contribution block.
    for child in child_refs {
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        let cb = &child.contrib;
        for j in 0..cn {
            let lj = gloc[crows[j]];
            for i in j..cn {
                let li = gloc[crows[i]];
                let (hi, lo) = if li >= lj { (li, lj) } else { (lj, li) };
                f[lo * nrow + hi] = f[lo * nrow + hi] + cb[j * cn + i];
            }
        }
    }

    // Restore the all-`usize::MAX` invariant and return the scratch to the
    // thread-local before `factor_front` (which spawns work-stealing GEMM tasks).
    for &g in &ri {
        gloc[g] = usize::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);

    let (front, contrib) = factor_front(f, nrow, ncol, perturb_floor)?;
    Ok(NodeFactor {
        front,
        row_indices: ri,
        contrib,
    })
}

/// Recursively factor the assembly subtree rooted at supernode `s` with a
/// work-stealing tree schedule: the children's subtrees factor concurrently and
/// this node only after they finish. Independent subtrees fill idle threads and
/// the per-front GEMM shares the same rayon pool — no level-barrier stall. See
/// the unsymmetric twin in [`crate::numeric::multifrontal_lu`].
fn factor_subtree<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    perturb_floor: Option<f64>,
) -> Result<SubtreeFactors<T>, FeralError> {
    let children = &sym.supernodes[s].children;
    let mut outs: Vec<SubtreeFactors<T>> = children
        .par_iter()
        .map(|&ch| factor_subtree(ch, sym, a_perm, perturb_floor))
        .collect::<Result<Vec<_>, _>>()?;
    let nf = {
        let child_refs: Vec<&NodeFactor<T>> = outs.iter().map(|(own, _)| own).collect();
        factor_one_node(s, sym, a_perm, &child_refs, perturb_floor)?
    };
    // Free the children's contribution blocks NOW: they have been extend-added
    // into this front and are never read again (the global emit uses only the
    // L/D factor). Retaining the whole `Σ cnrow²` CB stack to the end was the
    // dominant transient-memory cost (OOMs on large fronts); dropping each CB as
    // its parent consumes it keeps only the active contribution frontier live.
    for (own, _) in outs.iter_mut() {
        own.contrib = Vec::new();
    }
    let mut subtree = Vec::new();
    for (i, (own, rest)) in outs.into_iter().enumerate() {
        subtree.push((children[i], own));
        subtree.extend(rest);
    }
    Ok((nf, subtree))
}

/// Factor a sparse symmetric matrix `A` as `Pᵀ A P = L D Lᵀ` via generic
/// multifrontal Bunch-Kaufman. Works for `T = f64` and `T = Complex<f64>`
/// (complex symmetric, `A = Aᵀ`).
///
/// Returns an [`LdltFactors`] in factorization order; solve with
/// [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt).
pub fn factor_sparse_ldlt<T: Scalar>(a: &CscMatrix<T>) -> Result<LdltFactors<T>, FeralError> {
    factor_sparse_ldlt_with(a, &FactorOptions::default())
}

/// Like [`factor_sparse_ldlt`] but with explicit [`FactorOptions`] —
/// notably static-pivoting (preconditioner) mode via `on_zero_pivot`.
///
/// Convenience wrapper: runs [`analyze`] then [`factor_numeric`]. For the
/// PARDISO-style *analyze once, factor many* workflow — FEM Newton steps or a
/// frequency sweep that reuse one sparsity pattern — call them separately and
/// keep the [`MultifrontalSymbolic`] across factorizations.
pub fn factor_sparse_ldlt_with<T: Scalar>(
    a: &CscMatrix<T>,
    opts: &FactorOptions,
) -> Result<LdltFactors<T>, FeralError> {
    let symb = analyze(a.n, &a.col_ptr, &a.row_idx)?;
    factor_numeric(&symb, a, opts)
}

/// Reusable symbolic analysis (fill-reducing ordering + assembly-tree levels)
/// for a fixed sparsity pattern. Value-independent: build once with [`analyze`]
/// and pass to [`factor_numeric`] for each set of numeric values sharing the
/// pattern — the PARDISO phase-1 analysis.
pub struct MultifrontalSymbolic {
    inner: Option<SymbolicInner>,
    n: usize,
    nnz: usize,
}

struct SymbolicInner {
    sym: SymbolicFactorization,
    /// Assembly-tree levels: `by_level[l]` are the supernodes at level `l`, all
    /// mutually independent (factored concurrently by the rayon driver).
    by_level: Vec<Vec<usize>>,
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

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)`: the number of
    /// eliminated columns and the full front height. The raw material for
    /// factorization-cost diagnostics — front-size distribution (small vs dense
    /// fronts → BLAS-2 vs BLAS-3 efficiency) and a factor-flop estimate.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        match &self.inner {
            Some(i) => i.sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect(),
            None => Vec::new(),
        }
    }

    /// Number of assembly-tree levels (the level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.by_level.len())
    }
}

/// PARDISO phase 1: analyze a sparsity pattern (`n`, CSC `col_ptr`/`row_idx`,
/// lower triangle). The result is value-independent and reusable across many
/// [`factor_numeric`] calls that share the pattern.
pub fn analyze(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Result<MultifrontalSymbolic, FeralError> {
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
    // Relaxed/fill-tolerant amalgamation — a standard sparse-direct technique
    // (PARDISO/MUMPS apply it to every matrix): when fundamental supernodes are
    // narrow the Schur-update GEMMs are low-rank and memory-bound, so trade a
    // little explicit-zero fill for wider, higher-rank dense fronts. These
    // defaults (≤512-wide, ≤128 extra rows/merge) are tuned to roughly double
    // factor throughput on the EM FEM / MoM matrices RLA targets, but the lever
    // is workload-agnostic; it is exposed as the general `SupernodeParams.relax`
    // knob and gated to `n >= RELAX_MIN_N` inside `find_supernodes`.
    let snode_params = SupernodeParams {
        preprocess: crate::symbolic::supernode::OrderingPreprocess::None,
        relax: Some(crate::symbolic::supernode::RelaxAmalgamation {
            max_width: 512,
            max_extra_rows: 128,
        }),
        ..SupernodeParams::default()
    };
    let mut sym = symbolic_factorize(&pattern, &snode_params)?;

    // Liu (1986) contribution-stack minimization. Reorder each supernode's
    // children so the live contribution-block stack peak is minimized during
    // factorization. This is a pure **scheduling hint**: supernode IDs, the
    // e-numbering and the factor are unchanged (the global emit walks IDs, not
    // children, and trailing rows are sorted), so it is correctness-, fill- and
    // throughput-neutral — it only shrinks the transient CB-stack that drives
    // factorization peak RSS.
    //
    // Each node leaves a contribution block of size `cb = (nrow−ncol)²` for its
    // parent and needs `peak` working-stack to factor its subtree. Processing
    // children in order, the stack while doing child `i` is `Σ_{j<i} cb_j +
    // peak_i`; Liu's theorem minimizes `maxᵢ(Σ_{j<i} cb_j + peak_i)` by ordering
    // children by `(peak − cb)` descending. Supernodes are in postorder, so a
    // single forward sweep has every child's `(peak, cb)` ready.
    //
    // **Hybrid Liu**: reordering is only applied where the contribution stack is
    // actually large (`Σ children cb ≥ LIU_MIN_STACK`) — the upper/mid tree,
    // which is a handful of nodes carrying the spike. The vast majority of small
    // leaf nodes keep their natural order, whose rayon spawn pattern parallelizes
    // better. This keeps almost all of Liu's memory win while shedding most of
    // its throughput cost (the memory-optimal child order is not the
    // parallel-load-optimal one). `peak[s]` is always computed against the order
    // actually used, so the propagation stays exact.
    let nsuper = sym.supernodes.len();
    if USE_LIU_REORDER.load(Ordering::Relaxed) {
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
            let mut acc = 0.0f64; // Σ cb of already-processed children
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
        inner: Some(SymbolicInner { sym, by_level }),
        n,
        nnz,
    })
}

/// PARDISO phases 2–3: numeric factorization reusing a [`MultifrontalSymbolic`].
/// `a` must carry the same sparsity pattern (`n`, `nnz`) the analysis was built
/// from. Honours static pivoting and incomplete-factor dropping via `opts`.
pub fn factor_numeric<T: Scalar>(
    symb: &MultifrontalSymbolic,
    a: &CscMatrix<T>,
    opts: &FactorOptions,
) -> Result<LdltFactors<T>, FeralError> {
    a.validate()?;
    let n = symb.n;
    if a.n != n || a.row_idx.len() != symb.nnz {
        return Err(FeralError::InvalidInput(
            "factor_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    let inner = match &symb.inner {
        None => {
            return Ok(LdltFactors {
                n: 0,
                l_col_ptr: vec![0],
                l_row_idx: Vec::new(),
                l_values: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                two_by_two: Vec::new(),
                perm: Vec::new(),
                n_perturbed: 0,
                inertia: Inertia::new(0, 0, 0),
            });
        }
        Some(i) => i,
    };
    let sym = &inner.sym;

    // Static-pivot floor (absolute), translated from feral's ZeroPivotAction.
    // `PerturbToEps { abs_floor }` is taken as given (feral convention: an
    // absolute floor, typically `eps_rel · ‖A‖∞`); `Fail` disables perturbation.
    let perturb_floor: Option<f64> = match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    // 2. Permuted matrix A_perm = Pᵀ A P in permuted (new) numbering, lower
    //    triangle. `perm_inv` is old→new.
    let nnz = a.row_idx.len();
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let gi = sym.perm_inv[i];
            let gj = sym.perm_inv[j];
            let (r, c) = if gi >= gj { (gi, gj) } else { (gj, gi) };
            rows.push(r);
            cols.push(c);
            vals.push(a.values[k]);
        }
    }
    let a_perm = CscMatrix::<T>::from_triplets(n, &rows, &cols, &vals)?;

    // 3. Multifrontal numeric factorization with a work-stealing schedule over
    //    the assembly tree: each subtree factors independently (children before
    //    parent), filling idle threads without a level barrier, and the per-front
    //    GEMM shares the same rayon pool. The precomputed `by_level` is no longer
    //    consulted here (it remains available via `MultifrontalSymbolic::n_levels`).
    let nsuper = sym.supernodes.len();

    // Roots of the assembly forest: supernodes that are no node's child.
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    let roots: Vec<usize> = (0..nsuper).filter(|&s| !is_child[s]).collect();
    let root_outs: Vec<SubtreeFactors<T>> = roots
        .par_iter()
        .map(|&r| factor_subtree(r, sym, &a_perm, perturb_floor))
        .collect::<Result<Vec<_>, _>>()?;
    // Scatter the subtree factors into `node_results` (by supernode id) for the
    // global emit pass, which still walks supernodes in postorder.
    let mut node_results: Vec<Option<NodeFactor<T>>> = (0..nsuper).map(|_| None).collect();
    for (i, (own, subtree)) in root_outs.into_iter().enumerate() {
        node_results[roots[i]] = Some(own);
        for (s, nf) in subtree {
            node_results[s] = Some(nf);
        }
    }

    // Collect the factored nodes in supernode (= elimination) order.
    let mut nodes: Vec<&NodeFactor<T>> = Vec::with_capacity(nsuper);
    for node_opt in &node_results {
        match node_opt {
            Some(nd) => nodes.push(nd),
            None => {
                return Err(FeralError::InvalidInput(
                    "internal: unfactored supernode".to_string(),
                ))
            }
        }
    }

    // 4a. Assign factorization order e and gather D in e-order.
    let mut e_of_g = vec![usize::MAX; n];
    let mut perm = vec![0usize; n];
    let mut d_diag = vec![T::zero(); n];
    let mut d_subdiag = vec![T::zero(); n];
    let mut two_by_two = vec![false; n];
    let mut e = 0usize;
    for node in &nodes {
        let ff = &node.front;
        for j in 0..ff.nelim {
            let g = node.row_indices[ff.perm[j]];
            e_of_g[g] = e;
            perm[e] = sym.perm[g];
            d_diag[e] = ff.d_diag[j];
            d_subdiag[e] = ff.d_subdiag[j];
            two_by_two[e] = ff.two_by_two[j];
            e += 1;
        }
    }
    debug_assert_eq!(e, n, "every index eliminated exactly once");

    // 4b. Emit each front's L columns into the global CSC L, in e-order. A
    //     supernode's eliminated columns form a contiguous increasing e-range,
    //     so iterating nodes then `j` yields columns in ascending CSC order;
    //     rows within a column are sorted.
    let one = T::one();
    let mut l_col_ptr = Vec::with_capacity(n + 1);
    l_col_ptr.push(0);
    let mut l_row_idx: Vec<usize> = Vec::new();
    let mut l_values: Vec<T> = Vec::new();
    let mut col: Vec<(usize, T)> = Vec::new();
    for node in &nodes {
        let ff = &node.front;
        let nrow = ff.nrow;
        for j in 0..ff.nelim {
            col.clear();
            let diag_e = e_of_g[node.row_indices[ff.perm[j]]];
            col.push((diag_e, one));
            for i in (j + 1)..nrow {
                let v = ff.l[j * nrow + i];
                if v != T::zero() {
                    let row_e = e_of_g[node.row_indices[ff.perm[i]]];
                    col.push((row_e, v));
                }
            }
            // Incomplete factorization: drop sub-threshold fill (relative to the
            // column's largest multiplier), keeping the unit diagonal. Shrinks
            // nnz(L) and the apply cost — an approximate factor for use as a
            // preconditioner. `None` keeps the factor complete.
            if let Some(tau) = opts.drop_tol {
                let colmax = col
                    .iter()
                    .filter(|&&(r, _)| r != diag_e)
                    .map(|&(_, v)| v.magnitude())
                    .fold(0.0, f64::max);
                let thresh = tau * colmax;
                col.retain(|&(r, v)| r == diag_e || v.magnitude() >= thresh);
            }
            col.sort_unstable_by_key(|&(r, _)| r);
            for &(r, v) in &col {
                l_row_idx.push(r);
                l_values.push(v);
            }
            l_col_ptr.push(l_row_idx.len());
        }
    }

    let n_perturbed: usize = nodes.iter().map(|nd| nd.front.n_perturbed).sum();
    // Inertia is additive over the assembly tree: sum the per-front signatures.
    let mut inertia = Inertia::new(0, 0, 0);
    for nd in &nodes {
        inertia.positive += nd.front.inertia.positive;
        inertia.negative += nd.front.inertia.negative;
        inertia.zero += nd.front.inertia.zero;
    }

    Ok(LdltFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        d_diag,
        d_subdiag,
        two_by_two,
        perm,
        n_perturbed,
        inertia,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::ldlt_generic::solve_ldlt;
    use num_complex::Complex;

    fn residual_inf<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    /// 1D Laplacian-style SPD tridiagonal of size n (diag 2+something, off −1).
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

    #[test]
    fn f64_dense_front_blocked_multi_panel() {
        // A fully dense symmetric matrix is one front of width n=100 > NB(64),
        // so factoring it exercises the blocked **multi-panel** Bunch-Kaufman
        // path (which the small n≤50 tests never reach). Diagonally dominant SPD.
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
        // Dense complex-symmetric, one front of width 90 > NB → multi-panel.
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
        // 2D 5-point Laplacian on a 5×5 grid (n=25), SPD.
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
        // off-diagonal (−1 + 0.1i). Complex symmetric (A = Aᵀ), diagonally
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
        // 12×12 complex-symmetric grid (n=144): a deep, bushy assembly tree
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

        let opts = FactorOptions {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert!(
            f.n_perturbed >= 1,
            "expected ≥1 perturbation, got {}",
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
        // perturbation — the static-pivot path must not trigger spuriously.
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
        let opts = FactorOptions {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert_eq!(
            f.n_perturbed, 0,
            "well-conditioned matrix needs no perturbation"
        );
    }

    #[test]
    fn complex_sparse_2d_grid_residual() {
        // 2D complex-symmetric grid: diagonal (4 + i), neighbor (−1 + 0.2i).
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
