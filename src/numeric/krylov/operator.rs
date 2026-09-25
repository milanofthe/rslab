//! The operator and preconditioner interfaces of the Krylov solvers, with
//! their implementations for the sparse matrices and every factor type.

use super::util::*;

use crate::error::RslabError;
use crate::numeric::ldlt::LdltSolver;
use crate::numeric::settings::{SolverSettings, Threads};
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;

/// A linear operator `A`: applies `y = A x`. The Krylov solvers depend only on
/// this trait, so the operator may be an explicit sparse matrix
/// ([`CscMatrix`] symmetric / [`GeneralCsc`] general) **or matrix-free** - e.g.
/// a fast multipole (FMM/MLFMA) MoM operator the caller implements. RLA then
/// only factors the sparse near-field as the [`Preconditioner`].
pub trait LinearOperator<T: Scalar> {
    /// The system dimension.
    fn n(&self) -> usize;
    /// Write `y <- A x`. `x` and `y` have length `n`.
    fn apply(&self, x: &[T], y: &mut [T]);
    /// Block apply: `Y[:,c] <- A X[:,c]` for `c in 0..s`, with `X`,`Y` **column-
    /// major** `nxs` (RHS `c` is the contiguous slice `[c*n, (c+1)*n)`). The
    /// default loops the single-vector [`apply`](Self::apply); explicit-matrix
    /// operators override it with an amortized block matvec (each matrix entry
    /// loaded once for all `s` columns - the BLAS-3 arithmetic intensity that
    /// makes a multi-RHS solve pay over `s` separate ones).
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        let n = self.n();
        for c in 0..s {
            self.apply(&x[c * n..c * n + n], &mut y[c * n..c * n + n]);
        }
    }
}

impl<T: Scalar> LinearOperator<T> for CscMatrix<T> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.symv(x, y);
    }
    /// Amortized block symv: each lower-triangle entry `(i,j,v)` is loaded once
    /// and scattered to all `s` columns (`y[:,c] += v*x[j,c]`, and symmetrically
    /// `y[j,c] += v*x[i,c]` off the diagonal) - the BLAS-3 reuse a multi-RHS
    /// solve buys over `s` separate `symv`s.
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        let n = self.n;
        for v in y.iter_mut() {
            *v = T::zero();
        }
        for j in 0..n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let v = self.values[k];
                if i != j {
                    for c in 0..s {
                        let cb = c * n;
                        y[cb + i] = y[cb + i] + v * x[cb + j];
                        y[cb + j] = y[cb + j] + v * x[cb + i];
                    }
                } else {
                    for c in 0..s {
                        let cb = c * n;
                        y[cb + i] = y[cb + i] + v * x[cb + j];
                    }
                }
            }
        }
    }
}

impl<T: Scalar> LinearOperator<T> for GeneralCsc<T> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.matvec(x, y);
    }
    /// Amortized block matvec: each entry `(i,j,v)` is loaded once and applied to
    /// all `s` columns (`y[i,c] += v*x[j,c]`).
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        let n = self.n;
        for v in y.iter_mut() {
            *v = T::zero();
        }
        for j in 0..n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let v = self.values[k];
                for c in 0..s {
                    let cb = c * n;
                    y[cb + i] = y[cb + i] + v * x[cb + j];
                }
            }
        }
    }
}

/// A preconditioner `M ~ A`: applies `z = M^-1 r`. Implemented by a factored
/// [`LdltSolver`](crate::numeric::ldlt::LdltSolver)
/// and by [`NoPreconditioner`] (the unpreconditioned baseline).
pub trait Preconditioner<T: Scalar> {
    /// Write `z <- M^-1 r`. `r` and `z` have length `n`.
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError>;
    /// Block apply: `Z[:,c] <- M^-1 R[:,c]` for `c in 0..s`, with `R`,`Z` **column-
    /// major** `nxs`. The default loops [`apply`](Self::apply); a factored solver
    /// overrides it with a block triangular solve (`solve_many`) that loads each
    /// `L`/`D`/`U` value once for all `s` columns.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        for c in 0..s {
            self.apply(&r[c * n..c * n + n], &mut z[c * n..c * n + n])?;
        }
        Ok(())
    }
    /// Thread policy the **solve phase** should honour. A factored
    /// preconditioner returns the resolved [`Threads`] budget it was built with, so
    /// [`gmres_block`](super::gmres_block)'s parallel orthogonalization runs in a pool of the **same**
    /// width - factor and solve share one concurrency budget instead of the solve
    /// silently fanning out over the global pool (the embedded / solver-in-the-loop
    /// design point). The default [`Threads::Ambient`] means "use the caller's
    /// current pool" - the behaviour for [`NoPreconditioner`] and any preconditioner
    /// that carries no factorization budget.
    fn solve_threads(&self) -> Threads {
        Threads::Ambient
    }
}

/// The identity preconditioner `M = I` (`z = r`): unpreconditioned iteration,
/// the baseline against which a real preconditioner's iteration count is read.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPreconditioner;

impl<T: Scalar> Preconditioner<T> for NoPreconditioner {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

/// A factored RLA solver is a preconditioner: `M^-1 r` is one forward/back
/// substitution against the stored `LDL^T` factor.
impl<T: Scalar> Preconditioner<T> for LdltSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via [`solve_many`](LdltSolver::solve_many): one block
    /// triangular solve loads each `L`/`D` value once for all `s` columns.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        apply_block_via_rowmajor(r, z, s, n, |b, s| self.solve_many(b, s))
    }
}

/// A memory-halved preconditioner: factor `A` (supplied in `Complex<f64>`) in
/// `Complex<f32>` and apply it inside an `f64` Krylov iteration. The stored
/// factor occupies **half the bytes** and its triangular solves run in single
/// precision (gemm `c32` during the factor); because the outer COCG/COCR still
/// iterates in `f64`, the *solution* keeps full `f64` accuracy. This is the
/// standard mixed-precision setup for large 3D EM FEM / MOM preconditioning.
pub struct LowPrecisionPreconditioner {
    inner: LdltSolver<Complex<f32>>,
}

impl LowPrecisionPreconditioner {
    /// Down-cast `A` to `Complex<f32>` and factor it (static-pivoting honoured
    /// via `opts`, e.g. `ZeroPivotAction::PerturbToEps`).
    pub fn factor(a: &CscMatrix<Complex<f64>>, opts: &SolverSettings) -> Result<Self, RslabError> {
        let a32 = CscMatrix::<Complex<f32>> {
            n: a.n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            values: a
                .values
                .iter()
                .map(|v| Complex::new(v.re as f32, v.im as f32))
                .collect(),
        };
        Ok(Self {
            inner: LdltSolver::factor_with(&a32, opts)?,
        })
    }

    /// Stored factor fill (nnz of `L`); each entry is a single-precision
    /// `Complex<f32>` (8 bytes vs 16 for `Complex<f64>`).
    pub fn factor_nnz(&self) -> usize {
        self.inner.factor_nnz()
    }

    /// Number of statically perturbed pivots (see [`LdltSolver::n_perturbed`]).
    pub fn n_perturbed(&self) -> usize {
        self.inner.n_perturbed()
    }
}

impl Preconditioner<Complex<f64>> for LowPrecisionPreconditioner {
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), RslabError> {
        let r32: Vec<Complex<f32>> = r
            .iter()
            .map(|v| Complex::new(v.re as f32, v.im as f32))
            .collect();
        let z32 = self.inner.solve(&r32)?;
        for (zi, v) in z.iter_mut().zip(z32) {
            *zi = Complex::new(v.re as f64, v.im as f64);
        }
        Ok(())
    }
}

/// Memory-halved **unsymmetric** preconditioner: factor the general matrix `A`
/// (given in `Complex<f64>`) in `Complex<f32>` LU and apply it inside an `f64`
/// GMRES iteration. The `Complex<f32>` factor uses half the bytes (and gemm
/// `c32`); the outer GMRES keeps full `f64` accuracy. The unsymmetric analogue
/// of [`LowPrecisionPreconditioner`], for MoM/FEM general systems.
pub struct LowPrecisionLu {
    inner: crate::numeric::lu::LuFactors<Complex<f32>>,
}

impl LowPrecisionLu {
    /// Down-cast `A` to `Complex<f32>` and LU-factor it (options honoured -
    /// static pivoting and/or incomplete dropping for a preconditioner).
    pub fn factor(a: &GeneralCsc<Complex<f64>>, opts: &SolverSettings) -> Result<Self, RslabError> {
        let a32 = GeneralCsc::<Complex<f32>> {
            n: a.n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            values: a
                .values
                .iter()
                .map(|v| Complex::new(v.re as f32, v.im as f32))
                .collect(),
        };
        Ok(Self {
            inner: crate::numeric::lu::factor_general_lu(&a32, opts)?,
        })
    }

    /// Stored fill `nnz(L)+nnz(U)`, in single-precision entries.
    pub fn factor_nnz(&self) -> usize {
        crate::numeric::lu::LuFactors::factor_nnz(&self.inner)
    }

    /// Number of statically perturbed pivots.
    pub fn n_perturbed(&self) -> usize {
        self.inner.n_perturbed
    }
}

impl Preconditioner<Complex<f64>> for LowPrecisionLu {
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), RslabError> {
        let r32: Vec<Complex<f32>> = r
            .iter()
            .map(|v| Complex::new(v.re as f32, v.im as f32))
            .collect();
        let z32 = crate::numeric::lu::solve_lu(&self.inner, &r32)?;
        for (zi, v) in z.iter_mut().zip(z32) {
            *zi = Complex::new(v.re as f64, v.im as f64);
        }
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.inner.solve_threads
    }
}

/// A factorization usable as both a **direct solver** and a [`Preconditioner`].
/// Implemented by the symmetric [`LdltSolver`] and the general
/// [`LuFactors`](crate::numeric::lu::LuFactors), so a caller's
/// solver loop can hold `&dyn Factorization` and swap symmetric/general,
/// exact/incomplete, or `f64`/`f32` factors freely.
pub trait Factorization<T: Scalar>: Preconditioner<T> {
    /// Solve `A x = b` directly from the stored factor.
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError>;
    /// Stored fill (factor nonzeros) - the memory metric.
    fn factor_nnz(&self) -> usize;
    /// Number of statically perturbed pivots (0 for an exact factor).
    fn n_perturbed(&self) -> usize;
}

impl<T: Scalar> Factorization<T> for LdltSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        LdltSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        LdltSolver::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        LdltSolver::n_perturbed(self)
    }
}

impl<T: Scalar> Preconditioner<T> for crate::numeric::lu::LuFactors<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = crate::numeric::lu::solve_lu(self, r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.solve_threads
    }
    /// Block apply via `solve_lu_many` (one block triangular solve over all `s`
    /// columns).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        apply_block_via_rowmajor(r, z, s, n, |b, s| {
            crate::numeric::lu::solve_lu_many(self, b, s)
        })
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::lu::LuFactors<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        crate::numeric::lu::solve_lu(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::lu::LuFactors::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        self.n_perturbed
    }
}

/// The high-level [`LuSolver`](crate::numeric::lu::LuSolver) is a
/// preconditioner / factorization too - the unsymmetric twin of the
/// [`LdltSolver`] impls, so solver-in-the-loop code can be generic over either.
impl<T: Scalar> Preconditioner<T> for crate::numeric::lu::LuSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.solve_thread_policy()
    }
    /// Block apply via [`LuSolver::solve_many`](crate::numeric::lu::LuSolver::solve_many).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        apply_block_via_rowmajor(r, z, s, n, |b, s| self.solve_many(b, s))
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::lu::LuSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        crate::numeric::lu::LuSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::lu::LuSolver::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        crate::numeric::lu::LuSolver::n_perturbed(self)
    }
}

/// The KLU path composes with the iterative stack exactly like the
/// supernodal solvers: an exact (or sweep-refactored) `M^-1 = (LU)^-1` for
/// [`gmres`](super::gmres)/[`gmres_block`](super::gmres_block). Sequential by design, so [`solve_threads`]
/// pins the orthogonalization pool to one worker.
///
/// [`solve_threads`]: Preconditioner::solve_threads
impl<T: Scalar> Preconditioner<T> for crate::numeric::klu::KluSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.solve_thread_policy()
    }
    /// Block apply via [`KluSolver::solve_many`](crate::KluSolver::solve_many).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        apply_block_via_rowmajor(r, z, s, n, |b, s| self.solve_many(b, s))
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::klu::KluSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        crate::numeric::klu::KluSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::klu::KluSolver::factor_nnz(self)
    }
    /// KLU never perturbs pivots: a vanishing pivot is a hard
    /// [`RslabError::SingularBasis`] at factor time instead.
    fn n_perturbed(&self) -> usize {
        0
    }
}
