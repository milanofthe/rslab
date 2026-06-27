//! Generic multifrontal sparse LDLᵀ factorization over any [`Scalar`] field.
//!
//! This drives a full sparse symmetric-indefinite solve for both the real
//! (`f64`) and complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) paths by
//! reusing the existing **value-agnostic** symbolic analysis (ordering,
//! elimination tree, supernode amalgamation) and applying the generic dense
//! Bunch-Kaufman kernel from [`crate::dense::ldlt_generic`] front-by-front.
//!
//! As with the dense kernel, this does **not** touch the heavily optimized f64
//! multifrontal driver in [`crate::numeric::factorize`] (blocked, SIMD,
//! delayed pivoting, inertia, rayon). That stays the f64 performance path; this
//! is the shared generic reference and the complex-symmetric driver.
//!
//! ## Correctness-first scope
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

use crate::dense::factor::ZeroPivotAction;
use crate::dense::ldlt_generic::{bk_alpha, swap_sym_lower, LdltFactors};
use crate::error::FeralError;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::{symbolic_factorize, SupernodeParams, SymbolicFactorization};
use rayon::prelude::*;
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

/// Options controlling the generic multifrontal factorization. Defaults give an
/// **exact** complete factorization that fails on rank deficiency. Relaxing
/// them turns the factorization into a robust, memory-light **preconditioner**.
#[derive(Debug, Clone)]
pub struct GenericFactorOptions {
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

impl Default for GenericFactorOptions {
    fn default() -> Self {
        Self {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: None,
        }
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

    let mut k = 0;
    while k < ncol {
        let absakk = f[k * n + k].magnitude();

        // colmax restricted to the fully-summed block rows (k+1)..ncol.
        let mut colmax = 0.0;
        let mut imax = k;
        for i in (k + 1)..ncol {
            let m = f[k * n + i].magnitude();
            if m > colmax {
                colmax = m;
                imax = i;
            }
        }

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
            // rowmax in row imax, restricted to the fully-summed block.
            let mut rowmax = 0.0;
            for j in k..imax {
                let m = f[j * n + imax].magnitude();
                if m > rowmax {
                    rowmax = m;
                }
            }
            for i in (imax + 1)..ncol {
                let m = f[imax * n + i].magnitude();
                if m > rowmax {
                    rowmax = m;
                }
            }
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
            let dinv = d.recip();
            // Update only the fully-summed trailing columns `(k+1)..ncol`
            // (across all rows, so the L21 multiplier rows are formed), then
            // store multipliers. The contribution block `[ncol,nrow)²` is left
            // untouched and updated once, after the panel, by a single GEMM.
            for j in (k + 1)..ncol {
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
            // Static-pivot the 2×2 when its determinant is near-singular. The
            // real kernel (feral's `perturb_2x2_to_floor`) shifts the small
            // eigenvalue; for complex-symmetric blocks the eigenvalues are
            // complex, so we shift both diagonals by the floor (lifting |det|)
            // and, as a last resort, nudge det itself — enough to keep the
            // preconditioner factor live.
            match perturb_floor {
                Some(floor) if det.magnitude() < floor * floor => {
                    d11 = d11 + T::from_real(floor);
                    d22 = d22 + T::from_real(floor);
                    det = d11 * d22 - d21 * d21;
                    if det.magnitude() < floor * floor {
                        det = det + T::from_real(floor * floor);
                    }
                    n_perturbed += 1;
                }
                None if det == T::zero() => return Err(FeralError::NumericallyRankDeficient),
                _ => {}
            }
            let detinv = det.recip();
            d_diag[k] = d11;
            d_subdiag[k] = d21;
            d_diag[k + 1] = d22;
            two_by_two[k] = true;

            let mut l1 = vec![T::zero(); nrow];
            let mut l2 = vec![T::zero(); nrow];
            for i in (k + 2)..n {
                let wik = f[k * n + i];
                let wik1 = f[(k + 1) * n + i];
                l1[i] = (d22 * wik - d21 * wik1) * detinv;
                l2[i] = (d11 * wik1 - d21 * wik) * detinv;
            }
            for j in (k + 2)..ncol {
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

    // Contribution block: the trailing [ncol, nrow) cells were deliberately
    // NOT touched by the panel elimination above (the update loops stop at
    // `ncol`), so they still hold the original A22. Apply the whole deferred
    // symmetric Schur update CB = A22 − L21·D·L21ᵀ as one SIMD GEMM. This is
    // the FLOP-dominant step; gemm gives AVX2/AVX-512 complex/real kernels.
    let cnrow = nrow - ncol;
    let mut cb = vec![T::zero(); cnrow * cnrow];
    // Seed cb with A22, mirrored to both triangles (A is symmetric; no conj).
    for j in 0..cnrow {
        for i in j..cnrow {
            let v = f[(ncol + j) * n + (ncol + i)];
            cb[j * cnrow + i] = v;
            cb[i * cnrow + j] = v;
        }
    }
    if cnrow > 0 && ncol > 0 {
        // L21 (cnrow × ncol, column-major): the off-block multiplier rows.
        let mut l21 = vec![T::zero(); cnrow * ncol];
        for c in 0..ncol {
            for r in 0..cnrow {
                l21[c * cnrow + r] = f[c * n + (ncol + r)];
            }
        }
        // G = L21·D (cnrow × ncol, column-major); D is block-diagonal.
        let mut g = vec![T::zero(); cnrow * ncol];
        let mut c = 0;
        while c < ncol {
            if two_by_two[c] {
                let (d11, d21, d22) = (d_diag[c], d_subdiag[c], d_diag[c + 1]);
                for r in 0..cnrow {
                    let a = l21[c * cnrow + r];
                    let b = l21[(c + 1) * cnrow + r];
                    g[c * cnrow + r] = a * d11 + b * d21;
                    g[(c + 1) * cnrow + r] = a * d21 + b * d22;
                }
                c += 2;
            } else {
                let d = d_diag[c];
                for r in 0..cnrow {
                    g[c * cnrow + r] = l21[c * cnrow + r] * d;
                }
                c += 1;
            }
        }
        // cb ← cb − G·L21ᵀ. Column-major: cb and G have col-stride `cnrow`,
        // row-stride 1; L21ᵀ (ncol × cnrow) reads L21 with strides swapped.
        if USE_GEMM_SCHUR.load(Ordering::Relaxed) {
            // SAFETY: `cb`, `g`, `l21` are three distinct, non-overlapping
            // allocations, each sized for the (m,n,k) = (cnrow, cnrow, ncol)
            // access pattern under the strides passed. The only `Scalar` impls
            // are `f64` and `Complex<f64>`, both supported gemm element types,
            // so the runtime element-type dispatch cannot hit the
            // unsupported-type panic.
            unsafe {
                gemm::gemm(
                    cnrow,
                    cnrow,
                    ncol,
                    cb.as_mut_ptr(),
                    cnrow as isize,
                    1,
                    true,
                    g.as_ptr(),
                    cnrow as isize,
                    1,
                    l21.as_ptr(),
                    1,
                    cnrow as isize,
                    T::one(),
                    -T::one(),
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
        } else {
            // Scalar reference: same data, same result, no SIMD. cb[i,j] -=
            // Σ_c g[i,c]·l21[j,c] (= (G·L21ᵀ)[i,j]); lower triangle only, then
            // mirror, since CB is symmetric.
            for j in 0..cnrow {
                for i in j..cnrow {
                    let mut acc = cb[j * cnrow + i];
                    for c in 0..ncol {
                        acc = acc - g[c * cnrow + i] * l21[c * cnrow + j];
                    }
                    cb[j * cnrow + i] = acc;
                    cb[i * cnrow + j] = acc;
                }
            }
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

/// Borrow a child node's result, erroring if it is somehow not yet computed
/// (cannot happen: children occupy strictly lower assembly-tree levels).
fn child_ref<T: Scalar>(
    node_results: &[Option<NodeFactor<T>>],
    ch: usize,
) -> Result<&NodeFactor<T>, FeralError> {
    node_results
        .get(ch)
        .and_then(|o| o.as_ref())
        .ok_or_else(|| FeralError::InvalidInput("internal: missing child node".to_string()))
}

/// Factor one supernode's front: build its row structure, assemble the original
/// (permuted) entries and the children's contribution blocks, then partially
/// factor the fully-summed columns. Reads only already-computed children, so
/// supernodes on the same assembly-tree level run concurrently.
fn factor_one_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    node_results: &[Option<NodeFactor<T>>],
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
    for &ch in &snode.children {
        let child = child_ref(node_results, ch)?;
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

    let mut gloc = vec![usize::MAX; n]; // permuted-global → front-local
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li;
    }

    let mut f = vec![T::zero(); nrow * nrow];

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
    for &ch in &snode.children {
        let child = child_ref(node_results, ch)?;
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

    let (front, contrib) = factor_front(&mut f, nrow, ncol, perturb_floor)?;
    Ok(NodeFactor {
        front,
        row_indices: ri,
        contrib,
    })
}

/// Factor a sparse symmetric matrix `A` as `Pᵀ A P = L D Lᵀ` via generic
/// multifrontal Bunch-Kaufman. Works for `T = f64` and `T = Complex<f64>`
/// (complex symmetric, `A = Aᵀ`).
///
/// Returns an [`LdltFactors`] in factorization order; solve with
/// [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt).
pub fn factor_sparse_ldlt<T: Scalar>(a: &CscMatrix<T>) -> Result<LdltFactors<T>, FeralError> {
    factor_sparse_ldlt_with(a, &GenericFactorOptions::default())
}

/// Like [`factor_sparse_ldlt`] but with explicit [`GenericFactorOptions`] —
/// notably static-pivoting (preconditioner) mode via `on_zero_pivot`.
///
/// Convenience wrapper: runs [`analyze`] then [`factor_numeric`]. For the
/// PARDISO-style *analyze once, factor many* workflow — FEM Newton steps or a
/// frequency sweep that reuse one sparsity pattern — call them separately and
/// keep the [`GenericSymbolic`] across factorizations.
pub fn factor_sparse_ldlt_with<T: Scalar>(
    a: &CscMatrix<T>,
    opts: &GenericFactorOptions,
) -> Result<LdltFactors<T>, FeralError> {
    let symb = analyze(a.n, &a.col_ptr, &a.row_idx)?;
    factor_numeric(&symb, a, opts)
}

/// Reusable symbolic analysis (fill-reducing ordering + assembly-tree levels)
/// for a fixed sparsity pattern. Value-independent: build once with [`analyze`]
/// and pass to [`factor_numeric`] for each set of numeric values sharing the
/// pattern — the PARDISO phase-1 analysis.
pub struct GenericSymbolic {
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

impl GenericSymbolic {
    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Internal accessor for the unsymmetric LU driver: the symbolic
    /// factorization and the precomputed assembly-tree levels. `None` for the
    /// empty (`n == 0`) analysis.
    pub(crate) fn sym_and_levels(&self) -> Option<(&SymbolicFactorization, &[Vec<usize>])> {
        self.inner
            .as_ref()
            .map(|i| (&i.sym, i.by_level.as_slice()))
    }
}

/// PARDISO phase 1: analyze a sparsity pattern (`n`, CSC `col_ptr`/`row_idx`,
/// lower triangle). The result is value-independent and reusable across many
/// [`factor_numeric`] calls that share the pattern.
pub fn analyze(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Result<GenericSymbolic, FeralError> {
    let nnz = row_idx.len();
    if n == 0 {
        return Ok(GenericSymbolic { inner: None, n: 0, nnz });
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
    let snode_params = SupernodeParams {
        preprocess: crate::symbolic::supernode::OrderingPreprocess::None,
        ..SupernodeParams::default()
    };
    let sym = symbolic_factorize(&pattern, &snode_params)?;

    // Assembly-tree levels: level(s) = 1 + max(level(children)); same-level
    // supernodes are mutually independent.
    let nsuper = sym.supernodes.len();
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

    Ok(GenericSymbolic {
        inner: Some(SymbolicInner { sym, by_level }),
        n,
        nnz,
    })
}

/// PARDISO phases 2–3: numeric factorization reusing a [`GenericSymbolic`].
/// `a` must carry the same sparsity pattern (`n`, `nnz`) the analysis was built
/// from. Honours static pivoting and incomplete-factor dropping via `opts`.
pub fn factor_numeric<T: Scalar>(
    symb: &GenericSymbolic,
    a: &CscMatrix<T>,
    opts: &GenericFactorOptions,
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

    // 3. Multifrontal numeric factorization, parallelized level-by-level over
    //    the assembly tree (levels precomputed in `analyze`). Same-level
    //    supernodes are mutually independent and factored concurrently; each
    //    reads only its children's contribution blocks from lower levels.
    let by_level = &inner.by_level;
    let nsuper = sym.supernodes.len();

    let mut node_results: Vec<Option<NodeFactor<T>>> = (0..nsuper).map(|_| None).collect();
    for level_nodes in by_level {
        let computed: Vec<(usize, NodeFactor<T>)> = level_nodes
            .par_iter()
            .map(|&s| {
                factor_one_node(s, sym, &a_perm, &node_results, perturb_floor).map(|nf| (s, nf))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (s, nf) in computed {
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

        let opts = GenericFactorOptions {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert!(f.n_perturbed >= 1, "expected ≥1 perturbation, got {}", f.n_perturbed);
        let b = vec![c(1.0, 0.0); n];
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(x.iter().all(|v| v.norm().is_finite()), "factor must stay finite");
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
        let opts = GenericFactorOptions {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert_eq!(f.n_perturbed, 0, "well-conditioned matrix needs no perturbation");
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
