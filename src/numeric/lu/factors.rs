//! The LU factor types: the supernodal panel form of a numeric factorization
//! and the compressed-column `L`/`U` with their triangular solves.

use crate::error::RslabError;
use crate::numeric::supernodal::panel::PanelFactor;
use crate::scalar::{fmadd, Scalar};
use crate::sparse::general::GeneralCsc;
use rayon::prelude::*;

/// Stored unsymmetric LU factors, in factorization order. Solve with
/// [`solve_lu`]. The factored system is `P^T A P = L U`, `perm[e]` mapping
/// factorization position `e` to the original index.
pub struct LuFactors<T> {
    pub n: usize,
    /// `L` in CSC (unit lower, explicit unit diagonal).
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_values: Vec<T>,
    /// `U` in CSR (upper; the diagonal entry carries the pivot).
    pub u_row_ptr: Vec<usize>,
    pub u_col_idx: Vec<usize>,
    pub u_values: Vec<T>,
    /// Column permutation: factorization position -> original column index
    /// (`P^T A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position -> original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `A_hat = diag(d_row)*A*diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
    /// Column partition of the factor into supernodes (the fronts): columns
    /// `supernode_ptr[s]..supernode_ptr[s + 1]` of `L` and rows of `U` share
    /// one structure. Length `ns + 1`; empty when unknown.
    pub supernode_ptr: Vec<usize>,
    /// Parent of every supernode in the assembly tree (`usize::MAX` for a
    /// root); empty when unknown.
    pub supernode_parent: Vec<usize>,
    /// Number of statically perturbed pivots.
    pub n_perturbed: usize,
    /// Thread policy the **solve phase** should honour: resolved from
    /// the factorization's [`SolverSettings::threads`](crate::SolverSettings::threads) so an iterative solve using
    /// this factor as a preconditioner runs its parallel orthogonalization in a
    /// pool of the **same** width - factor and solve share one concurrency budget
    /// instead of the solve silently fanning out over the global pool.
    /// [`Threads::Ambient`](crate::Threads::Ambient) means "use the caller's
    /// current pool" (the solver-in-the-loop path); otherwise a concrete
    /// [`Threads::Fixed`](crate::Threads::Fixed) worker count.
    pub solve_threads: crate::numeric::settings::Threads,
}

/// The numeric result of a sparse LU factorization: the unit lower `L` and
/// the transposed upper factor `U^T` (its diagonal in the panel) in
/// supernodal panel form (the storage the solves run on, written by the
/// drivers without a copy), the permutations, the scalings and the outcome.
/// [`into_factors`](Self::into_factors) materializes the compressed
/// [`LuFactors`] (`L` as CSC, `U` as CSR) for the reference solves.
#[derive(Clone, Debug)]
pub struct LuNumeric<T> {
    /// `L` in panel form, in elimination order (unit diagonal implicit).
    pub l: PanelFactor<T>,
    /// `U^T` in panel form: column `c` of the panel is row `c` of `U`, the
    /// diagonal of `U` at the panel's diagonal.
    pub ut: PanelFactor<T>,
    /// `perm[e]` is the original column eliminated at position `e`.
    pub perm: Vec<usize>,
    /// `perm_row[e]` is the original row that became pivot row `e`.
    pub perm_row: Vec<usize>,
    /// Row scaling applied before the factorization.
    pub d_row: Vec<f64>,
    /// Column scaling applied before the factorization.
    pub d_col: Vec<f64>,
    /// Supernode tree over the factor's supernodes (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// Pivots perturbed by the static regularization.
    pub n_perturbed: usize,
    /// Structural panel slots holding an exact zero (the symmetrized
    /// pattern, cancellation or `drop_tol`); the stored nonzeros are
    /// `l.nnz() + ut.nnz() - n_zeros`.
    pub n_zeros: usize,
    /// Thread policy the solves inherit.
    pub solve_threads: crate::numeric::settings::Threads,
}

impl<T: Scalar> LuNumeric<T> {
    /// Dimension.
    pub fn n(&self) -> usize {
        self.l.n
    }

    /// Stored fill `nnz(L) + nnz(U)`: the structural panel entries minus the
    /// slots holding an exact zero.
    pub fn factor_nnz(&self) -> usize {
        self.l.nnz() + self.ut.nnz() - self.n_zeros
    }

    /// Bytes of the two panel factors.
    pub fn bytes(&self) -> usize {
        self.l.bytes() + self.ut.bytes()
    }

    fn shell(&self) -> LuFactors<T> {
        LuFactors {
            n: self.l.n,
            l_col_ptr: Vec::new(),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            u_row_ptr: Vec::new(),
            u_col_idx: Vec::new(),
            u_values: Vec::new(),
            perm: self.perm.clone(),
            perm_row: self.perm_row.clone(),
            d_row: self.d_row.clone(),
            d_col: self.d_col.clone(),
            supernode_ptr: self.l.sn_col.iter().map(|&c| c as usize).collect(),
            supernode_parent: self.supernode_parent.clone(),
            n_perturbed: self.n_perturbed,
            solve_threads: self.solve_threads,
        }
    }

    /// The compressed form for the reference solves (copies both factors).
    pub fn into_factors(self) -> LuFactors<T> {
        let mut f = self.shell();
        let (l_col_ptr, l_row_idx, l_values) = self.l.to_csc(true);
        let (u_row_ptr, u_col_idx, u_values) = self.ut.to_csc(false);
        f.l_col_ptr = l_col_ptr;
        f.l_row_idx = l_row_idx;
        f.l_values = l_values;
        f.u_row_ptr = u_row_ptr;
        f.u_col_idx = u_col_idx;
        f.u_values = u_values;
        f
    }

    /// Split into the two panel factors and an [`LuFactors`] shell (empty CSC
    /// arrays) carrying the permutations and scalings for the solver.
    pub(crate) fn into_parts(self) -> (PanelFactor<T>, PanelFactor<T>, LuFactors<T>) {
        let shell = self.shell();
        (self.l, self.ut, shell)
    }
}

impl<T: Scalar> LuFactors<T> {
    /// Stored fill: `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len() + self.u_values.len()
    }
}

/// Solve `A x = b` from an unsymmetric LU factorization (`P^T A P = L U`).
#[allow(clippy::needless_range_loop)] // CSC/CSR solves index col_ptr/row_ptr + scaling
pub fn solve_lu<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // y_hat = P_row * (D_r b): row-equilibrate then row-permute the RHS.
    let mut y: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm_row[e];
            b[orig] * T::from_real(f.d_row[orig])
        })
        .collect();
    // Forward solve L y = y_hat (CSC, unit diagonal). Column-oriented: once y[e] is
    // final, eliminate it from the rows below. Axpys via `fmadd` (FMA on
    // native builds; see `scalar::fmadd`). The explicit unit diagonal is a
    // column's FIRST entry (rows sorted, lower triangular in elimination
    // numbering), so the sweep starts at `col_ptr[e] + 1` instead of
    // branching on `i != e` at every nonzero.
    for e in 0..n {
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e, "unit diagonal must lead its column");
        let nye = T::zero() - y[e];
        for k in (s + 1)..ee {
            let i = f.l_row_idx[k];
            y[i] = fmadd(f.l_values[k], nye, y[i]);
        }
    }
    // Backward solve U x = y (CSR by row). The pivot is a row's FIRST entry
    // (columns sorted, upper triangular), replacing the per-nonzero
    // diagonal-search branch. Deliberately mul+sub, NOT `fmadd`: the
    // accumulator is a latency-bound serial chain (see `solve_ldlt`'s
    // backward-sweep note).
    let mut x = vec![T::zero(); n];
    for e in (0..n).rev() {
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e, "pivot must lead its row");
        let diag = f.u_values[s];
        let mut acc = y[e];
        for k in (s + 1)..ee {
            acc = acc - f.u_values[k] * x[f.u_col_idx[k]];
        }
        x[e] = acc * diag.recip();
    }
    // Undo the column permutation and apply the column equilibration:
    // x_orig[perm[e]] = D_c[perm[e]] * x_hat[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm[e];
        out[orig] = x[e] * T::from_real(f.d_col[orig]);
    }
    Ok(out)
}

/// Solve `A^T * x = b` against the stored factorization of `A`. The factor chain is `A^-1 = D_c P_c (LU)^-1 P_r^T D_r` (see
/// [`solve_lu`]), so `A^-T = D_r P_r L^-T U^-T P_c^T D_c`: column-equilibrate
/// and column-permute the RHS, forward-solve `U^T` (lower triangular; `U` is
/// CSR with the pivot leading each row, so once `z[e]` is final it
/// eliminates from the trailing columns of row `e`, scatter form),
/// backward-solve `L^T` (unit upper; dot column `e` of `L` against the
/// already-solved tail), then row-permute and row-equilibrate the result.
///
/// This is the plain TRANSPOSE, not the conjugate transpose - complex
/// callers wanting `A^-H b` conjugate the RHS and the result around this
/// call (`A^-H b = conj(A^-T conj(b))`), as the condition estimator does.
#[allow(clippy::needless_range_loop)] // scatter-form sweeps write w[e] and w[u_col_idx[k]]
pub fn solve_lu_transpose<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // w_hat = P_c^T (D_c b): column-equilibrate then column-permute the RHS.
    let mut w: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm[e];
            b[orig] * T::from_real(f.d_col[orig])
        })
        .collect();
    // Forward solve U^T z = w_hat, in place in `w`.
    for e in 0..n {
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e, "pivot must lead its row");
        let ze = w[e] * f.u_values[s].recip();
        w[e] = ze;
        if ze != T::zero() {
            let nze = T::zero() - ze;
            for k in (s + 1)..ee {
                let j = f.u_col_idx[k];
                w[j] = fmadd(f.u_values[k], nze, w[j]);
            }
        }
    }
    // Backward solve L^T v = z, in place in `w` (unit diagonal leads each
    // column, so the dot starts at `col_ptr[e] + 1`).
    for e in (0..n).rev() {
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e, "unit diagonal must lead its column");
        let mut acc = w[e];
        for k in (s + 1)..ee {
            acc = acc - f.l_values[k] * w[f.l_row_idx[k]];
        }
        w[e] = acc;
    }
    // x_orig[perm_row[e]] = D_r[perm_row[e]] * v[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm_row[e];
        out[orig] = w[e] * T::from_real(f.d_row[orig]);
    }
    Ok(out)
}

/// Solve `A * X = B` for `nrhs` right-hand sides at once. `b` and the returned
/// `x` are **row-major** `n x nrhs` buffers (`b[i*nrhs + c]` is RHS `c` at row
/// `i`). The `L`/`U` structure is traversed once and each value applied to all
/// `nrhs` columns - faster than `nrhs` separate [`solve_lu`] calls.
/// Below this RHS count / work size the LU block solve runs serially (the
/// parallel gather/scatter overhead only amortizes for wide multi-RHS).
const PAR_SOLVE_MIN_RHS: usize = 8;
const PAR_SOLVE_MIN_WORK: usize = 1 << 18;

/// Solve `A X = B` for `nrhs` right-hand sides. Row-major `n x nrhs` layout, as
/// [`solve_ldlt_many`](crate::solve_ldlt_many). For a wide RHS the columns are
/// split into per-thread chunks (each RHS independent); the result is
/// **bit-identical** to the serial block solve.
pub fn solve_lu_many<T: Scalar>(
    f: &LuFactors<T>,
    b: &[T],
    nrhs: usize,
) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if nrhs == 0 || b.len() != n * nrhs {
        return Err(RslabError::DimensionMismatch {
            expected: n * nrhs,
            got: b.len(),
        });
    }
    let nthreads = rayon::current_num_threads().max(1);
    if nrhs < PAR_SOLVE_MIN_RHS || n * nrhs < PAR_SOLVE_MIN_WORK || nthreads < 2 {
        return solve_lu_block(f, b, nrhs);
    }
    let nchunks = nthreads.min(nrhs);
    let chunk = nrhs.div_ceil(nchunks);
    let ranges: Vec<(usize, usize)> = (0..nchunks)
        .map(|t| (t * chunk, ((t + 1) * chunk).min(nrhs)))
        .filter(|&(a, e)| a < e)
        .collect();
    let parts: Result<Vec<(usize, usize, Vec<T>)>, RslabError> = ranges
        .par_iter()
        .map(|&(c0, c1)| {
            let w = c1 - c0;
            let mut sub = vec![T::zero(); n * w];
            for i in 0..n {
                let ib = i * nrhs;
                let sb = i * w;
                sub[sb..sb + w].copy_from_slice(&b[ib + c0..ib + c1]);
            }
            let xs = solve_lu_block(f, &sub, w)?;
            Ok((c0, c1, xs))
        })
        .collect();
    let parts = parts?;
    let mut x = vec![T::zero(); n * nrhs];
    for (c0, c1, xs) in parts {
        let w = c1 - c0;
        for i in 0..n {
            let ib = i * nrhs;
            let sb = i * w;
            x[ib + c0..ib + c1].copy_from_slice(&xs[sb..sb + w]);
        }
    }
    Ok(x)
}

/// Serial block solve over `nrhs` right-hand sides; fanned over column chunks by
/// the parallel [`solve_lu_many`].
fn solve_lu_block<T: Scalar>(f: &LuFactors<T>, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    // Y_hat = P_row * (D_r B): row-equilibrate then row-permute each RHS block.
    let mut y = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm_row[e];
        let s = T::from_real(f.d_row[orig]);
        let (eb, ob) = (e * nrhs, orig * nrhs);
        for c in 0..nrhs {
            y[eb + c] = b[ob + c] * s;
        }
    }
    // Reusable single-row scratch. Hoisting the row that is reused across an inner
    // sweep into a **local** buffer breaks the apparent aliasing of `y[..]` with
    // itself, so the `nrhs`-wide AXPY kernels below operate on non-aliasing,
    // contiguous slices the compiler can vectorize (and the hoisted row is loaded
    // once per outer step, not once per nonzero).
    let mut row = vec![T::zero(); nrhs];
    // Forward solve L Y = Y_hat (CSC, unit diagonal). `y[e]` (the column's source row)
    // is read by every nonzero of column `e` and is not written in this sweep.
    for e in 0..n {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e);
        for k in (s + 1)..ee {
            let i = f.l_row_idx[k];
            let nlval = T::zero() - f.l_values[k];
            let ib = i * nrhs;
            let tgt = &mut y[ib..ib + nrhs];
            for c in 0..nrhs {
                tgt[c] = fmadd(nlval, row[c], tgt[c]);
            }
        }
    }
    // Backward solve U X = Y (CSR by row), in place in `y`. Accumulate row `e`'s
    // update in the local buffer (the off-diagonal sources `y[c_col]`, `c_col > e`,
    // are already solved and not touched here), then scale and write it back.
    // The pivot leads its row (sorted columns, upper triangular).
    for e in (0..n).rev() {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e);
        let diag = f.u_values[s];
        for k in (s + 1)..ee {
            let nuval = T::zero() - f.u_values[k];
            let cb = f.u_col_idx[k] * nrhs;
            let src = &y[cb..cb + nrhs];
            for c in 0..nrhs {
                row[c] = fmadd(nuval, src[c], row[c]);
            }
        }
        let dinv = diag.recip();
        for c in 0..nrhs {
            y[eb + c] = row[c] * dinv;
        }
    }
    // Undo column permutation + column equilibration: out[perm[e]] = D_c * x_hat[e].
    let mut out = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm[e];
        let s = T::from_real(f.d_col[orig]);
        let (ob, eb) = (orig * nrhs, e * nrhs);
        for c in 0..nrhs {
            out[ob + c] = y[eb + c] * s;
        }
    }
    Ok(out)
}

/// Solve `A x = b` with iterative refinement against the original matrix `a`.
/// Each step computes the residual `r = b - A x` and applies the correction
/// `x <- x + (LU)^-1 r`, stopping once `||r||inf` stops improving or `max_iter` is
/// reached. This recovers the accuracy a static / within-block-pivoted factor
/// loses on ill-conditioned matrices, at the cost of a few extra solves.
pub fn solve_lu_refined<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    max_iter: usize,
) -> Result<Vec<T>, RslabError> {
    Ok(solve_lu_refined_with(f, a, b, &crate::refine::RefinePolicy::steps(max_iter))?.0)
}

/// Iterative refinement of an `LU` solve under an explicit
/// [`RefinePolicy`](crate::refine::RefinePolicy), reporting the achieved
/// backward error.
pub fn solve_lu_refined_with<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    policy: &crate::refine::RefinePolicy,
) -> Result<(Vec<T>, crate::refine::RefineOutcome), RslabError> {
    let n = f.n;
    if a.n != n || b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let mut x = solve_lu(f, b)?;
    let outcome = crate::refine::refine_in_place(a, b, &mut x, policy, |r| solve_lu(f, r))?;
    Ok((x, outcome))
}
