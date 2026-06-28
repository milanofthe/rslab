//! Krylov iteration for complex-symmetric systems, preconditioned by an RLA
//! factorization.
//!
//! The target use is **3D EM FEM / MOM**: large complex-symmetric `A = Aᵀ`
//! (PARDISO `mtype 6`) systems solved iteratively, with a robust, memory-light
//! RLA factorization (static-pivoted, optionally `f32` / incomplete) as the
//! preconditioner. The iterative method of choice for `A = Aᵀ` is **COCG**
//! (Conjugate Orthogonal Conjugate Gradient, van der Vorst & Melissen 1990):
//! structurally CG, but every inner product is the *unconjugated* bilinear
//! form `xᵀy = Σ xᵢyᵢ` — the correct geometry for a complex-symmetric (not
//! Hermitian) operator. For `T = f64` it reduces exactly to preconditioned CG.
//!
//! The [`Preconditioner`] trait decouples the iteration from the factorization
//! precision: an `f64` factor applies directly; an `f32` factor (memory-halved)
//! down-/up-casts inside `apply`, while the iteration itself always runs in the
//! working precision `T`.

use crate::error::FeralError;
use crate::numeric::multifrontal_ldlt::FactorOptions;
use crate::numeric::sparse_solver::LdltSolver;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;

/// A linear operator `A`: applies `y = A x`. The Krylov solvers depend only on
/// this trait, so the operator may be an explicit sparse matrix
/// ([`CscMatrix`] symmetric / [`GeneralCsc`] general) **or matrix-free** — e.g.
/// a fast multipole (FMM/MLFMA) MoM operator the caller implements. RLA then
/// only factors the sparse near-field as the [`Preconditioner`].
pub trait LinearOperator<T: Scalar> {
    /// The system dimension.
    fn n(&self) -> usize;
    /// Write `y ← A x`. `x` and `y` have length `n`.
    fn apply(&self, x: &[T], y: &mut [T]);
    /// Block apply: `Y[:,c] ← A X[:,c]` for `c in 0..s`, with `X`,`Y` **column-
    /// major** `n×s` (RHS `c` is the contiguous slice `[c·n, (c+1)·n)`). The
    /// default loops the single-vector [`apply`](Self::apply); explicit-matrix
    /// operators override it with an amortized block matvec (each matrix entry
    /// loaded once for all `s` columns — the BLAS-3 arithmetic intensity that
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
    /// and scattered to all `s` columns (`y[:,c] += v·x[j,c]`, and symmetrically
    /// `y[j,c] += v·x[i,c]` off the diagonal) — the BLAS-3 reuse a multi-RHS
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
    /// all `s` columns (`y[i,c] += v·x[j,c]`).
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

/// A preconditioner `M ≈ A`: applies `z = M⁻¹ r`. Implemented by a factored
/// [`LdltSolver`](crate::numeric::sparse_solver::LdltSolver)
/// and by [`NoPreconditioner`] (the unpreconditioned baseline).
pub trait Preconditioner<T: Scalar> {
    /// Write `z ← M⁻¹ r`. `r` and `z` have length `n`.
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError>;
    /// Block apply: `Z[:,c] ← M⁻¹ R[:,c]` for `c in 0..s`, with `R`,`Z` **column-
    /// major** `n×s`. The default loops [`apply`](Self::apply); a factored solver
    /// overrides it with a block triangular solve (`solve_many`) that loads each
    /// `L`/`D`/`U` value once for all `s` columns.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), FeralError> {
        for c in 0..s {
            self.apply(&r[c * n..c * n + n], &mut z[c * n..c * n + n])?;
        }
        Ok(())
    }
}

/// The identity preconditioner `M = I` (`z = r`): unpreconditioned iteration,
/// the baseline against which a real preconditioner's iteration count is read.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPreconditioner;

impl<T: Scalar> Preconditioner<T> for NoPreconditioner {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

/// A factored RLA solver is a preconditioner: `M⁻¹ r` is one forward/back
/// substitution against the stored `LDLᵀ` factor.
impl<T: Scalar> Preconditioner<T> for LdltSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via [`solve_many`](LdltSolver::solve_many): one block
    /// triangular solve loads each `L`/`D` value once for all `s` columns. The
    /// Krylov block is column-major; `solve_many` is row-major, so it is
    /// transposed in/out (`O(n·s)`, cheap against the solve).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), FeralError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = self.solve_many(&rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
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
    pub fn factor(a: &CscMatrix<Complex<f64>>, opts: &FactorOptions) -> Result<Self, FeralError> {
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
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), FeralError> {
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
    inner: crate::numeric::multifrontal_lu::LuFactors<Complex<f32>>,
}

impl LowPrecisionLu {
    /// Down-cast `A` to `Complex<f32>` and LU-factor it (options honoured —
    /// static pivoting and/or incomplete dropping for a preconditioner).
    pub fn factor(a: &GeneralCsc<Complex<f64>>, opts: &FactorOptions) -> Result<Self, FeralError> {
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
            inner: crate::numeric::multifrontal_lu::factor_general_lu(&a32, opts)?,
        })
    }

    /// Stored fill `nnz(L)+nnz(U)`, in single-precision entries.
    pub fn factor_nnz(&self) -> usize {
        crate::numeric::multifrontal_lu::LuFactors::factor_nnz(&self.inner)
    }

    /// Number of statically perturbed pivots.
    pub fn n_perturbed(&self) -> usize {
        self.inner.n_perturbed
    }
}

impl Preconditioner<Complex<f64>> for LowPrecisionLu {
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), FeralError> {
        let r32: Vec<Complex<f32>> = r
            .iter()
            .map(|v| Complex::new(v.re as f32, v.im as f32))
            .collect();
        let z32 = crate::numeric::multifrontal_lu::solve_lu(&self.inner, &r32)?;
        for (zi, v) in z.iter_mut().zip(z32) {
            *zi = Complex::new(v.re as f64, v.im as f64);
        }
        Ok(())
    }
}

/// Unconjugated bilinear inner product `xᵀy = Σ xᵢyᵢ` (no complex conjugation —
/// the defining choice of the COCG geometry for `A = Aᵀ`).
#[inline]
fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi * yi;
    }
    s
}

/// Euclidean norm `‖x‖₂ = √Σ|xᵢ|²` (genuine modulus — used only for the stopping
/// test, never inside the Krylov recurrences).
#[inline]
fn norm2<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.magnitude_sq()).sum::<f64>().sqrt()
}

/// Outcome of a Krylov solve.
#[derive(Debug, Clone)]
pub struct KrylovResult<T> {
    /// The computed solution.
    pub x: Vec<T>,
    /// Number of iterations actually performed.
    pub iters: usize,
    /// `true` if `‖b − Ax‖ / ‖b‖ ≤ tol` was reached within `max_iter`.
    pub converged: bool,
    /// Final relative residual `‖b − Ax‖ / ‖b‖`.
    pub final_res: f64,
}

/// Preconditioned COCG for a complex-symmetric `A = Aᵀ` stored as a lower-
/// triangle [`CscMatrix`] (multiplied via [`CscMatrix::symv`]).
///
/// Solves `A x = b` to relative residual `tol` (or `max_iter` iterations).
/// `precond` supplies `M⁻¹`; pass [`NoPreconditioner`] for the unpreconditioned
/// baseline. Starts from `x₀ = 0`.
///
/// Breakdown (a zero bilinear denominator `pᵀAp` or `rᵀz`, possible for an
/// indefinite complex-symmetric operator) stops the iteration and returns the
/// best iterate with `converged = false`.
pub fn cocg<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
) -> Result<KrylovResult<T>, FeralError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }

    let mut x = vec![T::zero(); n];
    // r₀ = b − A x₀ = b (x₀ = 0).
    let mut r = b.to_vec();
    let bnorm = norm2(b);
    if bnorm == 0.0 {
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
        });
    }

    let mut z = vec![T::zero(); n];
    precond.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut rho = dotu(&r, &z); // rᵀz, unconjugated
    let mut q = vec![T::zero(); n];

    let mut final_res = norm2(&r) / bnorm;
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iter {
        op.apply(&p, &mut q); // q = A p
        let pq = dotu(&p, &q);
        if pq == T::zero() {
            break; // breakdown
        }
        let alpha = rho * pq.recip();
        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * q[i];
        }
        iters += 1;
        final_res = norm2(&r) / bnorm;
        if final_res <= tol {
            converged = true;
            break;
        }
        precond.apply(&r, &mut z)?;
        let rho_new = dotu(&r, &z);
        if rho == T::zero() {
            break; // breakdown
        }
        let beta = rho_new * rho.recip();
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rho = rho_new;
    }

    Ok(KrylovResult {
        x,
        iters,
        converged,
        final_res,
    })
}

/// Preconditioned COCR (Conjugate Orthogonal Conjugate Residual, Sogabe &
/// Zhang 2007) for complex-symmetric `A = Aᵀ`. The CR-family analogue of
/// [`cocg`]: it minimises a residual-like quantity and is typically **smoother
/// and more robust on strongly indefinite** operators (high-frequency 3D
/// Helmholtz) where COCG's residual can oscillate or break down.
///
/// Same interface and conventions as [`cocg`]. Costs one matrix–vector product
/// and one preconditioner apply per iteration. Reduces to preconditioned CR
/// for `T = f64`.
pub fn cocr<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
) -> Result<KrylovResult<T>, FeralError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }

    let mut x = vec![T::zero(); n];
    let mut r = b.to_vec(); // r₀ = b (x₀ = 0)
    let bnorm = norm2(b);
    if bnorm == 0.0 {
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
        });
    }

    let mut z = vec![T::zero(); n]; // z = M⁻¹ r
    precond.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut ap = vec![T::zero(); n];
    op.apply(&p, &mut ap); // A p
    let mut az = ap.clone(); // A z (= A p at init since p = z)
    let mut gamma = dotu(&z, &az); // zᵀ A z
    let mut w = vec![T::zero(); n]; // M⁻¹ A p
    let mut aw = vec![T::zero(); n]; // A w

    let mut final_res = norm2(&r) / bnorm;
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iter {
        precond.apply(&ap, &mut w)?; // w = M⁻¹ A p
        op.apply(&w, &mut aw); // A w
        let denom = dotu(&ap, &w); // (A p)ᵀ M⁻¹ (A p)
        if denom == T::zero() {
            break;
        }
        let alpha = gamma * denom.recip();
        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * ap[i];
            z[i] = z[i] - alpha * w[i];
            az[i] = az[i] - alpha * aw[i];
        }
        iters += 1;
        final_res = norm2(&r) / bnorm;
        if final_res <= tol {
            converged = true;
            break;
        }
        let gamma_new = dotu(&z, &az);
        if gamma == T::zero() {
            break;
        }
        let beta = gamma_new * gamma.recip();
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
            ap[i] = az[i] + beta * ap[i];
        }
        gamma = gamma_new;
    }

    Ok(KrylovResult {
        x,
        iters,
        converged,
        final_res,
    })
}

/// Conjugated (Hermitian) inner product `⟨x, y⟩ = Σ conj(xᵢ)·yᵢ` — the geometry
/// GMRES orthogonalises in (distinct from COCG's unconjugated form).
#[inline]
fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi.conj() * yi;
    }
    s
}

/// Complex Givens rotation `(c, s)` that zeroes `g` against `f`: with the
/// rotation `[[conj(c), conj(s)], [-s, c]]`, `conj(c)·f + conj(s)·g = r` (real)
/// and `-s·f + c·g = 0`, `|c|²+|s|² = 1`.
#[inline]
fn givens<T: Scalar>(f: T, g: T) -> (T, T) {
    if g == T::zero() {
        return (T::one(), T::zero());
    }
    if f == T::zero() {
        return (T::zero(), T::one());
    }
    let r = (f.magnitude_sq() + g.magnitude_sq()).sqrt();
    let inv = T::from_real(1.0 / r);
    (f * inv, g * inv)
}

/// Right-preconditioned restarted **GMRES(`restart`)** for a general
/// (unsymmetric) operator — the natural Krylov method for unsymmetric MoM/FEM
/// systems where COCG/COCR do not apply. `op` may be matrix-free; `precond`
/// supplies `M⁻¹` (e.g. an RLA [`LuFactors`](crate::numeric::multifrontal_lu::LuFactors)
/// near-field factor). Solves `A x = b` from `x₀ = 0`.
pub fn gmres<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<KrylovResult<T>, FeralError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(FeralError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // DGKS reorthogonalization threshold: redo the projection only when a single
    // MGS pass cancelled more than this fraction of the vector's length.
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    let bnorm = norm2(b);
    let mut x = vec![T::zero(); n];
    if bnorm == 0.0 {
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
        });
    }

    let mut total = 0usize;
    let mut z = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    let mut ax = vec![T::zero(); n];
    // Arnoldi basis as one **flat** `n × (m+1)` buffer (column `i` is
    // `v[i*n .. (i+1)*n]`) — contiguous, no per-iteration vector allocation, and
    // cache-friendly for the Gram–Schmidt sweeps.
    let mut v = vec![T::zero(); n * (m + 1)];

    while total < max_iter {
        op.apply(&x, &mut ax);
        let r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta = norm2(&r);
        if beta / bnorm <= tol {
            break;
        }
        // Column 0 of the basis = r / ‖r‖. Hessenberg H, Givens, and LS RHS g.
        let inv_beta = T::from_real(1.0 / beta);
        for k in 0..n {
            v[k] = r[k] * inv_beta;
        }
        let mut h = vec![vec![T::zero(); m]; m + 1];
        let mut cs = vec![T::zero(); m];
        let mut sn = vec![T::zero(); m];
        let mut g = vec![T::zero(); m + 1];
        g[0] = T::from_real(beta);
        let mut jdim = 0usize;
        let mut converged_inner = false;
        for j in 0..m {
            if total >= max_iter {
                break;
            }
            // Right preconditioning: w = A · M⁻¹ · v[j].
            precond.apply(&v[j * n..j * n + n], &mut z)?;
            op.apply(&z, &mut w);
            // Modified Gram–Schmidt against the existing basis, with **conditional**
            // reorthogonalization (DGKS): the second pass — essential on
            // ill-conditioned operators (MoM near-field) where a single MGS pass
            // loses orthogonality and the Hessenberg residual estimate drifts from
            // the true residual — runs only when the projection cancelled most of
            // the vector (‖w‖ dropped below `η·‖w₀‖`, η = 1/√2). Well-conditioned
            // cycles skip it, halving the orthogonalization cost.
            let wnorm0 = norm2(&w);
            for i in 0..=j {
                let hij = dotc(&v[i * n..i * n + n], &w);
                h[i][j] = hij;
                for k in 0..n {
                    w[k] = w[k] - hij * v[i * n + k];
                }
            }
            let mut hn = norm2(&w);
            if hn < REORTH_ETA * wnorm0 {
                for i in 0..=j {
                    let s = dotc(&v[i * n..i * n + n], &w);
                    h[i][j] = h[i][j] + s;
                    for k in 0..n {
                        w[k] = w[k] - s * v[i * n + k];
                    }
                }
                hn = norm2(&w);
            }
            h[j + 1][j] = T::from_real(hn);
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                for k in 0..n {
                    v[(j + 1) * n + k] = w[k] * inv;
                }
            } else {
                // Invariant subspace: zero the (reused) basis column so a later
                // sweep never reads stale data from a previous restart cycle.
                for k in 0..n {
                    v[(j + 1) * n + k] = T::zero();
                }
            }
            // Apply previous rotations to the new Hessenberg column.
            for i in 0..j {
                let temp = cs[i].conj() * h[i][j] + sn[i].conj() * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }
            // New rotation zeroing h[j+1][j]; apply to H and the LS RHS g.
            let (c, s) = givens(h[j][j], h[j + 1][j]);
            cs[j] = c;
            sn[j] = s;
            h[j][j] = c.conj() * h[j][j] + s.conj() * h[j + 1][j];
            h[j + 1][j] = T::zero();
            let g_next = -s * g[j];
            g[j] = c.conj() * g[j];
            g[j + 1] = g_next;
            total += 1;
            jdim = j + 1;
            if g[j + 1].magnitude() / bnorm <= tol {
                converged_inner = true;
                break;
            }
        }
        // Back-substitute the upper-triangular H for y, then x += M⁻¹·(V·y).
        let mut y = vec![T::zero(); jdim];
        for i in (0..jdim).rev() {
            let mut s = g[i];
            for k in (i + 1)..jdim {
                s = s - h[i][k] * y[k];
            }
            y[i] = s * h[i][i].recip();
        }
        let mut vy = vec![T::zero(); n];
        for i in 0..jdim {
            let yi = y[i];
            for k in 0..n {
                vy[k] = vy[k] + v[i * n + k] * yi;
            }
        }
        precond.apply(&vy, &mut z)?;
        for k in 0..n {
            x[k] = x[k] + z[k];
        }
        // NOTE: do **not** stop on the inner (Hessenberg LS) residual estimate
        // alone. On ill-conditioned operators (MoM near-field) the estimate can
        // dip below `tol` while the *true* residual ‖b−Ax‖ is orders larger
        // (loss of Arnoldi orthogonality). Restart and let the outer loop's
        // top-of-cycle TRUE-residual check decide convergence. `converged_inner`
        // only governs early-exit of the inner Arnoldi sweep, not termination.
        let _ = converged_inner;
    }

    op.apply(&x, &mut ax);
    let r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
    let final_res = norm2(&r) / bnorm;
    Ok(KrylovResult {
        x,
        iters: total,
        converged: final_res <= tol,
        final_res,
    })
}

/// Outcome of a block (multi-RHS) Krylov solve.
#[derive(Debug, Clone)]
pub struct BlockKrylovResult<T> {
    /// Solutions, column-major `n×s` (RHS `c` is the slice `[c·n, (c+1)·n)`).
    pub x: Vec<T>,
    /// Block iterations performed (the RHS advance in lockstep).
    pub iters: usize,
    /// `true` iff **every** RHS reached `tol`.
    pub converged: bool,
    /// Per-RHS final relative residual `‖b_c − A x_c‖ / ‖b_c‖`.
    pub final_res: Vec<f64>,
}

/// Right-preconditioned restarted **block GMRES** for `s` right-hand sides `b`
/// (column-major `n×s`). The `s` systems advance in lockstep so the two expensive
/// operations — the operator matvec and the preconditioner solve — are issued
/// once per step as **block** applies ([`LinearOperator::apply_block`] /
/// [`Preconditioner::apply_block`]), reaching BLAS-3 arithmetic intensity (each
/// factor / matrix value touched once for all `s` columns). Each RHS keeps its
/// own Arnoldi basis / Hessenberg / Givens, so a column converges identically to
/// the single-RHS [`gmres`]; the systems share only the batched operator and
/// preconditioner calls. Solves `A X = B` from `X₀ = 0`.
///
/// **Deflation:** a RHS whose true residual reaches `tol` drops out of the block,
/// so the batched applies shrink to the active width as columns converge — the
/// fast-converging RHS are never dragged along by the slowest one.
///
/// This is the MoM/FEM many-excitations path: factor (or `f32`-factor) once, then
/// drive all right-hand sides through one block iteration.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gmres_block<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<BlockKrylovResult<T>, FeralError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if s == 0 || b.len() != n * s {
        return Err(FeralError::DimensionMismatch {
            expected: n * s,
            got: b.len(),
        });
    }
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    let mut x = vec![T::zero(); n * s];
    let bnorm: Vec<f64> = (0..s).map(|c| norm2(&b[c * n..c * n + n])).collect();

    // Scratch sized for the **full** width `s` and reused; each restart cycle uses
    // only the first `sa` columns, where `sa` is the count of still-active RHS.
    // The basis stride is therefore `sa` (recomputed per cycle): block `j`, active
    // column `a` lives at `vbas[(j*sa + a)*n ..]`, so block `j` (the `n×sa` input
    // to one block apply) is the contiguous prefix `vbas[j*sa*n .. (j+1)*sa*n]`.
    let mut vbas = vec![T::zero(); n * s * (m + 1)];
    let mut zblk = vec![T::zero(); n * s]; // M⁻¹ · (block j)
    let mut wblk = vec![T::zero(); n * s]; // A · zblk
    let mut axblk = vec![T::zero(); n * s];
    let mut vyblk = vec![T::zero(); n * s];
    let mut xc = vec![T::zero(); n * s]; // compact live-RHS solutions for the residual matvec

    // Per-active-position Arnoldi state (indexed `0..sa`, reset each cycle).
    let mut h: Vec<Vec<Vec<T>>> = (0..s).map(|_| vec![vec![T::zero(); m]; m + 1]).collect();
    let mut cs: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m]).collect();
    let mut sn: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m]).collect();
    let mut g: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m + 1]).collect();
    let mut jdim = vec![0usize; s];
    let mut converged = vec![false; s];
    let mut final_res = vec![0.0f64; s];
    let mut total = 0usize;
    for c in 0..s {
        if bnorm[c] == 0.0 {
            converged[c] = true;
        }
    }

    while total < max_iter {
        // **Deflation:** gather the not-yet-converged ("live") RHS into a compact
        // block, recompute their true residual with one block matvec, and keep
        // only those still above `tol` as the active set for this cycle. Converged
        // columns never re-enter the (expensive) inner block applies again.
        let live: Vec<usize> = (0..s).filter(|&c| !converged[c]).collect();
        if live.is_empty() {
            break;
        }
        let lw = live.len();
        for (a, &c) in live.iter().enumerate() {
            xc[a * n..a * n + n].copy_from_slice(&x[c * n..c * n + n]);
        }
        op.apply_block(&xc[..lw * n], &mut axblk[..lw * n], lw);
        // Active set = live RHS whose true residual still exceeds `tol`.
        let mut act: Vec<usize> = Vec::new();
        for (a, &c) in live.iter().enumerate() {
            let cb = c * n;
            let ab = a * n;
            let mut rn = 0.0;
            for i in 0..n {
                rn += (b[cb + i] - axblk[ab + i]).magnitude_sq();
            }
            let beta = rn.sqrt();
            final_res[c] = beta / bnorm[c];
            if final_res[c] <= tol {
                converged[c] = true;
            } else {
                // Initialize active position `act.len()` from this residual.
                let ap = act.len();
                let inv = T::from_real(1.0 / beta);
                for i in 0..n {
                    vbas[ap * n + i] = (b[cb + i] - axblk[ab + i]) * inv; // block 0, col ap
                }
                for row in h[ap].iter_mut() {
                    for e in row.iter_mut() {
                        *e = T::zero();
                    }
                }
                for e in g[ap].iter_mut() {
                    *e = T::zero();
                }
                g[ap][0] = T::from_real(beta);
                jdim[ap] = 0;
                act.push(c);
            }
        }
        let sa = act.len();
        if sa == 0 {
            break;
        }

        let mut inner_done = vec![false; sa];
        for j in 0..m {
            if total >= max_iter {
                break;
            }
            let jblock = j * sa * n;
            // Batched right preconditioning + operator apply over the active block.
            precond.apply_block(&vbas[jblock..jblock + sa * n], &mut zblk[..sa * n], sa, n)?;
            op.apply_block(&zblk[..sa * n], &mut wblk[..sa * n], sa);
            let mut all_done = true;
            for ap in 0..sa {
                if inner_done[ap] {
                    continue;
                }
                all_done = false;
                let wb = ap * n;
                // Modified Gram–Schmidt against this RHS's own basis, DGKS reorth.
                let wnorm0 = norm2(&wblk[wb..wb + n]);
                for i in 0..=j {
                    let vb = (i * sa + ap) * n;
                    let hij = dotc(&vbas[vb..vb + n], &wblk[wb..wb + n]);
                    h[ap][i][j] = hij;
                    for k in 0..n {
                        wblk[wb + k] = wblk[wb + k] - hij * vbas[vb + k];
                    }
                }
                let mut hn = norm2(&wblk[wb..wb + n]);
                if hn < REORTH_ETA * wnorm0 {
                    for i in 0..=j {
                        let vb = (i * sa + ap) * n;
                        let ss = dotc(&vbas[vb..vb + n], &wblk[wb..wb + n]);
                        h[ap][i][j] = h[ap][i][j] + ss;
                        for k in 0..n {
                            wblk[wb + k] = wblk[wb + k] - ss * vbas[vb + k];
                        }
                    }
                    hn = norm2(&wblk[wb..wb + n]);
                }
                h[ap][j + 1][j] = T::from_real(hn);
                let v1 = ((j + 1) * sa + ap) * n;
                if hn > 0.0 {
                    let inv = T::from_real(1.0 / hn);
                    for k in 0..n {
                        vbas[v1 + k] = wblk[wb + k] * inv;
                    }
                } else {
                    for k in 0..n {
                        vbas[v1 + k] = T::zero();
                    }
                }
                // Previous rotations, then a new one to zero h[j+1][j]; update g.
                for i in 0..j {
                    let temp = cs[ap][i].conj() * h[ap][i][j] + sn[ap][i].conj() * h[ap][i + 1][j];
                    h[ap][i + 1][j] = -sn[ap][i] * h[ap][i][j] + cs[ap][i] * h[ap][i + 1][j];
                    h[ap][i][j] = temp;
                }
                let (cj, sj) = givens(h[ap][j][j], h[ap][j + 1][j]);
                cs[ap][j] = cj;
                sn[ap][j] = sj;
                h[ap][j][j] = cj.conj() * h[ap][j][j] + sj.conj() * h[ap][j + 1][j];
                h[ap][j + 1][j] = T::zero();
                let g_next = -sj * g[ap][j];
                g[ap][j] = cj.conj() * g[ap][j];
                g[ap][j + 1] = g_next;
                jdim[ap] = j + 1;
                if g[ap][j + 1].magnitude() / bnorm[act[ap]] <= tol {
                    inner_done[ap] = true;
                }
            }
            total += 1;
            if all_done {
                break;
            }
        }

        // x_c += M⁻¹ (V_a y_a): back-substitute each active RHS, build the compact
        // VY block, one batched preconditioner apply, then scatter to global `x`.
        for e in vyblk[..sa * n].iter_mut() {
            *e = T::zero();
        }
        for ap in 0..sa {
            let jd = jdim[ap];
            if jd == 0 {
                continue;
            }
            let mut y = vec![T::zero(); jd];
            for i in (0..jd).rev() {
                let mut acc = g[ap][i];
                for k in (i + 1)..jd {
                    acc = acc - h[ap][i][k] * y[k];
                }
                y[i] = acc * h[ap][i][i].recip();
            }
            let vyb = ap * n;
            for i in 0..jd {
                let yi = y[i];
                let vb = (i * sa + ap) * n;
                for k in 0..n {
                    vyblk[vyb + k] = vyblk[vyb + k] + vbas[vb + k] * yi;
                }
            }
        }
        precond.apply_block(&vyblk[..sa * n], &mut zblk[..sa * n], sa, n)?;
        for ap in 0..sa {
            let c = act[ap];
            for i in 0..n {
                x[c * n + i] = x[c * n + i] + zblk[ap * n + i];
            }
        }
    }

    // Final true residual per RHS.
    op.apply_block(&x, &mut axblk, s);
    let mut all_conv = true;
    for c in 0..s {
        if bnorm[c] == 0.0 {
            final_res[c] = 0.0;
            continue;
        }
        let cb = c * n;
        let mut rn = 0.0;
        for i in 0..n {
            rn += (b[cb + i] - axblk[cb + i]).magnitude_sq();
        }
        final_res[c] = rn.sqrt() / bnorm[c];
        if final_res[c] > tol {
            all_conv = false;
        }
    }
    Ok(BlockKrylovResult {
        x,
        iters: total,
        converged: all_conv,
        final_res,
    })
}

/// Adapter: a closure block-matvec `op(x, y, s)` (`Y ← A·X`, column-major `n×s`) as a
/// [`LinearOperator`] for the matrix-free call path. `FnMut` (the Arnoldi issues applies
/// sequentially) so the operator's own scratch lives in the closure capture — no struct, no
/// interior-mutability dance at the call site. The `RefCell` is borrowed for one apply at a time.
struct FnOp<F> {
    f: std::cell::RefCell<F>,
    n: usize,
}
impl<T: Scalar, F: FnMut(&[T], &mut [T], usize)> LinearOperator<T> for FnOp<F> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        (self.f.borrow_mut())(x, y, 1)
    }
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        (self.f.borrow_mut())(x, y, s)
    }
}

/// Adapter: a closure block-preconditioner `pc(r, z, s)` (`Z ← M⁻¹·R`) as a [`Preconditioner`].
struct FnPc<G> {
    f: std::cell::RefCell<G>,
}
impl<T: Scalar, G: FnMut(&[T], &mut [T], usize) -> Result<(), FeralError>> Preconditioner<T> for FnPc<G> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError> {
        (self.f.borrow_mut())(r, z, 1)
    }
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, _n: usize) -> Result<(), FeralError> {
        (self.f.borrow_mut())(r, z, s)
    }
}

/// Closure entry point for [`gmres_block`]: pass the block matvec and block preconditioner as
/// `FnMut` closures plus the dimension `n` — the natural form for a **matrix-free** MoM/FEM
/// operator that captures its own assembly data + scratch, with no `LinearOperator`/`Preconditioner`
/// boilerplate. For an unpreconditioned solve pass `|r, z, _| { z.copy_from_slice(r); Ok(()) }`.
#[allow(clippy::too_many_arguments)]
pub fn gmres_block_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    s: usize,
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<BlockKrylovResult<T>, FeralError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), FeralError>,
{
    let op = FnOp { f: std::cell::RefCell::new(op), n };
    let pc = FnPc { f: std::cell::RefCell::new(precond) };
    gmres_block(&op, b, s, &pc, tol, max_iter, restart)
}

/// Closure entry point for [`gmres`] (single RHS) — see [`gmres_block_fn`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<KrylovResult<T>, FeralError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), FeralError>,
{
    let op = FnOp { f: std::cell::RefCell::new(op), n };
    let pc = FnPc { f: std::cell::RefCell::new(precond) };
    gmres(&op, b, &pc, tol, max_iter, restart)
}

/// A factorization usable as both a **direct solver** and a [`Preconditioner`].
/// Implemented by the symmetric [`LdltSolver`] and the general
/// [`LuFactors`](crate::numeric::multifrontal_lu::LuFactors), so a caller's
/// solver loop can hold `&dyn Factorization` and swap symmetric/general,
/// exact/incomplete, or `f64`/`f32` factors freely.
pub trait Factorization<T: Scalar>: Preconditioner<T> {
    /// Solve `A x = b` directly from the stored factor.
    fn solve(&self, b: &[T]) -> Result<Vec<T>, FeralError>;
    /// Stored fill (factor nonzeros) — the memory metric.
    fn factor_nnz(&self) -> usize;
    /// Number of statically perturbed pivots (0 for an exact factor).
    fn n_perturbed(&self) -> usize;
}

impl<T: Scalar> Factorization<T> for LdltSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, FeralError> {
        LdltSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        LdltSolver::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        LdltSolver::n_perturbed(self)
    }
}

impl<T: Scalar> Preconditioner<T> for crate::numeric::multifrontal_lu::LuFactors<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError> {
        let x = crate::numeric::multifrontal_lu::solve_lu(self, r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via `solve_lu_many` (one block triangular solve over all `s`
    /// columns). Column-major Krylov block ↔ row-major `solve_lu_many` transpose.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), FeralError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = crate::numeric::multifrontal_lu::solve_lu_many(self, &rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::multifrontal_lu::LuFactors<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, FeralError> {
        crate::numeric::multifrontal_lu::solve_lu(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::multifrontal_lu::LuFactors::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        self.n_perturbed
    }
}

/// The high-level [`LuSolver`](crate::numeric::multifrontal_lu::LuSolver) is a
/// preconditioner / factorization too — the unsymmetric twin of the
/// [`LdltSolver`] impls, so solver-in-the-loop code can be generic over either.
impl<T: Scalar> Preconditioner<T> for crate::numeric::multifrontal_lu::LuSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via [`LuSolver::solve_many`](crate::numeric::multifrontal_lu::LuSolver::solve_many).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), FeralError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = self.solve_many(&rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::multifrontal_lu::LuSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, FeralError> {
        crate::numeric::multifrontal_lu::LuSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::multifrontal_lu::LuSolver::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        crate::numeric::multifrontal_lu::LuSolver::n_perturbed(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::multifrontal_ldlt::{FactorOptions, ZeroPivotAction};
    use num_complex::Complex;

    type C = Complex<f64>;

    /// 2D complex-symmetric Helmholtz-style grid (lower triangle).
    fn grid(m: usize, diag: C, off: C) -> CscMatrix<C> {
        let n = m * m;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                c.push(p);
                v.push(diag);
                if b + 1 < m {
                    let (hi, lo) = (idx(a, b + 1), p);
                    r.push(hi);
                    c.push(lo);
                    v.push(off);
                }
                if a + 1 < m {
                    let (hi, lo) = (idx(a + 1, b), p);
                    r.push(hi);
                    c.push(lo);
                    v.push(off);
                }
            }
        }
        CscMatrix::<C>::from_triplets(n, &r, &c, &v).unwrap()
    }

    #[test]
    fn cocg_unpreconditioned_solves_complex_symmetric() {
        let c = |re, im| Complex::new(re, im);
        let a = grid(8, c(4.0, 0.5), c(-1.0, 0.1));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let res = cocg(&a, &b, &NoPreconditioner, 1e-10, 2000).unwrap();
        assert!(res.converged, "COCG should converge, res={}", res.final_res);
        // Verify against the actual residual.
        let mut ax = vec![C::default(); n];
        a.symv(&res.x, &mut ax);
        let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-7, "residual {}", r);
    }

    #[test]
    fn cocr_solves_complex_symmetric_pre_and_unpre() {
        let c = |re, im| Complex::new(re, im);
        let a = grid(10, c(4.0, 0.5), c(-1.0, 0.1));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        // Unpreconditioned COCR converges to the true solution.
        let un = cocr(&a, &b, &NoPreconditioner, 1e-10, 3000).unwrap();
        assert!(un.converged, "COCR res={}", un.final_res);
        let mut ax = vec![C::default(); n];
        a.symv(&un.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-7, "COCR residual {}", res);

        // RLA-preconditioned COCR collapses to a handful of iterations.
        let m = LdltSolver::factor(&a).unwrap();
        let pre = cocr(&a, &b, &m, 1e-10, 3000).unwrap();
        assert!(pre.converged && pre.iters <= 3, "iters {}", pre.iters);
    }

    #[test]
    fn gmres_solves_unsymmetric_with_lu_preconditioner() {
        use crate::numeric::multifrontal_lu::factor_general_lu;
        use crate::sparse::general::GeneralCsc;
        // Genuinely unsymmetric complex 2D grid (right ≠ left couplings).
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        // Unpreconditioned GMRES converges (well-conditioned).
        let un = gmres(&a, &b, &NoPreconditioner, 1e-10, 2000, 40).unwrap();
        assert!(un.converged, "GMRES res={}", un.final_res);

        // LU factor as preconditioner → 1–2 iterations.
        let lu = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let pre = gmres(&a, &b, &lu, 1e-10, 200, 40).unwrap();
        assert!(pre.converged, "preconditioned GMRES res={}", pre.final_res);
        assert!(
            pre.iters <= 3,
            "LU-preconditioned GMRES iters {}",
            pre.iters
        );
        // Verify the true residual.
        let mut y = vec![C::default(); n];
        a.matvec(&pre.x, &mut y);
        let res = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-8, "residual {}", res);
    }

    /// Build the genuinely unsymmetric complex grid used by the GMRES tests.
    fn unsym_grid(m: usize) -> crate::sparse::general::GeneralCsc<C> {
        use crate::sparse::general::GeneralCsc;
        let c = |re, im| Complex::new(re, im);
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
    }

    #[test]
    fn gmres_block_single_rhs_matches_scalar_gmres() {
        // s = 1 block GMRES reduces to the single-RHS path (same Arnoldi, same
        // Givens, default block apply = one single apply): same solution to the
        // requested tolerance, same iteration count up to a ±1 boundary effect
        // (the true residual can straddle `tol` by an FP rounding difference
        // between the two summation orders).
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(8);
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let lu = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let single = gmres(&a, &b, &lu, 1e-10, 200, 40).unwrap();
        let blk = gmres_block(&a, &b, 1, &lu, 1e-10, 200, 40).unwrap();
        assert!(blk.converged);
        assert!(
            (blk.iters as i64 - single.iters as i64).abs() <= 1,
            "block(s=1) iters {} vs single {}",
            blk.iters,
            single.iters
        );
        let diff = (0..n).map(|i| (blk.x[i] - single.x[i]).norm()).fold(0.0, f64::max);
        assert!(diff < 1e-7, "block(s=1) solution differs by {diff}");
    }

    #[test]
    fn gmres_block_multi_rhs_solves_each_column() {
        // Several distinct right-hand sides solved in one block iteration; every
        // column must reach its own system's true residual.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(10);
        let n = a.n;
        let s = 5;
        // Column-major n×s block: RHS `k` is shifted/scaled so columns differ.
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        let lu = factor_general_lu(&a, &FactorOptions::default()).unwrap();
        let res = gmres_block(&a, &bblk, s, &lu, 1e-10, 200, 40).unwrap();
        assert!(res.converged, "block GMRES must converge; res={:?}", res.final_res);
        // True residual per column.
        for k in 0..s {
            let mut y = vec![C::default(); n];
            a.matvec(&res.x[k * n..k * n + n], &mut y);
            let r = (0..n).map(|i| (y[i] - bblk[k * n + i]).norm()).fold(0.0, f64::max);
            assert!(r < 1e-8, "column {k} residual {r}");
        }
        // Each column must equal the single-RHS solve of that column.
        for k in 0..s {
            let single = gmres(&a, &bblk[k * n..k * n + n], &lu, 1e-10, 200, 40).unwrap();
            let diff = (0..n)
                .map(|i| (res.x[k * n + i] - single.x[i]).norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-9, "column {k} differs from single solve by {diff}");
        }
    }

    #[test]
    fn f32_lu_preconditioner_keeps_f64_accuracy_in_gmres() {
        use crate::sparse::general::GeneralCsc;
        let c = |re, im| Complex::new(re, im);
        let m = 10;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(20.0, 2.0)); // strongly diagonally dominant → f32 factor is accurate
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    vv.push(c(-1.0, 0.2));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    vv.push(c(-1.5, 0.3));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();
        // The f32 LU factor is a half-memory preconditioner accurate to ~1e-6
        // (the f32 apply floor). f64 GMRES reaches that floor in a couple of
        // iterations — ample for a MoM/FEM Krylov tolerance, at half the factor
        // memory. (A residual below ~1e-6 is not attainable with an f32 apply;
        // use the f64 factor for tighter tolerances.)
        let pc = LowPrecisionLu::factor(&a, &FactorOptions::default()).unwrap();
        assert!(pc.factor_nnz() > 0);
        let res = gmres(&a, &b, &pc, 1e-6, 200, 50).unwrap();
        assert!(res.converged, "mixed-precision GMRES res={}", res.final_res);
        assert!(res.iters <= 6, "iters {}", res.iters);
        let mut y = vec![C::default(); n];
        a.matvec(&res.x, &mut y);
        let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-5, "residual {}", r);
    }

    #[test]
    fn cocr_handles_indefinite_helmholtz() {
        // Indefinite complex-symmetric: high-frequency 2D Helmholtz with a
        // negative-real diagonal shift (`diag = -1 + 0.3i`). A robust
        // preconditioner (static-pivoted RLA) plus COCR must still converge.
        let c = |re, im| Complex::new(re, im);
        let a = grid(10, c(-1.0, 0.3), c(1.0, 0.05));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c(1.0, (i % 3) as f64 - 1.0)).collect();
        let opts = FactorOptions {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
            drop_tol: None,
            ..Default::default()
        };
        let m = LdltSolver::factor_with(&a, &opts).unwrap();
        let pre = cocr(&a, &b, &m, 1e-9, 500).unwrap();
        assert!(
            pre.converged,
            "indefinite COCR res={} iters={}",
            pre.final_res, pre.iters
        );
        let mut ax = vec![C::default(); n];
        a.symv(&pre.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-6, "residual {}", res);
    }

    #[test]
    fn incomplete_factor_reduces_fill_and_still_preconditions() {
        // Threshold dropping shrinks the factor (less memory) at the cost of a
        // weaker preconditioner — but COCG must still converge to the true
        // f64 solution. Demonstrates the memory ↔ iteration tradeoff.
        let c = |re, im| Complex::new(re, im);
        let a = grid(16, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let full = LdltSolver::factor(&a).unwrap();
        let opts = FactorOptions {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
            ..Default::default()
        };
        let inc = LdltSolver::factor_with(&a, &opts).unwrap();

        assert!(
            inc.factor_nnz() < full.factor_nnz(),
            "dropping should reduce fill: incomplete {} vs complete {}",
            inc.factor_nnz(),
            full.factor_nnz()
        );

        let rf = cocg(&a, &b, &full, 1e-10, 1000).unwrap();
        let ri = cocg(&a, &b, &inc, 1e-10, 1000).unwrap();
        assert!(ri.converged, "incomplete-preconditioned COCG must converge");
        assert!(
            ri.iters >= rf.iters,
            "incomplete factor should need ≥ complete-factor iterations"
        );
        let mut ax = vec![C::default(); n];
        a.symv(&ri.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-8, "residual {}", res);
    }

    #[test]
    fn f32_preconditioner_keeps_f64_accuracy() {
        // Factor the preconditioner in Complex<f32> (half memory) but iterate in
        // f64: the solution must still reach f64-level residual, and the f32
        // factor — though approximate — keeps the iteration count tiny.
        let c = |re, im| Complex::new(re, im);
        let a = grid(14, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let m = LowPrecisionPreconditioner::factor(&a, &FactorOptions::default()).unwrap();
        let res = cocg(&a, &b, &m, 1e-10, 500).unwrap();
        assert!(res.converged, "mixed-precision COCG res={}", res.final_res);
        // A few iterations suffice; the f32 factor is a strong preconditioner.
        assert!(res.iters <= 12, "f32-preconditioned iters {}", res.iters);
        // Full f64 accuracy recovered despite the single-precision factor.
        let mut ax = vec![C::default(); n];
        a.symv(&res.x, &mut ax);
        let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-8, "mixed-precision residual {}", r);
        assert!(m.factor_nnz() > 0);
    }

    #[test]
    fn rla_preconditioner_collapses_iteration_count() {
        // A complete RLA factorization is ≈ A⁻¹, so preconditioned COCG must
        // converge in a handful of iterations — vastly fewer than without.
        let c = |re, im| Complex::new(re, im);
        let a = grid(12, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let unpre = cocg(&a, &b, &NoPreconditioner, 1e-10, 5000).unwrap();
        let m = LdltSolver::factor(&a).unwrap();
        let pre = cocg(&a, &b, &m, 1e-10, 5000).unwrap();

        assert!(pre.converged && unpre.converged);
        assert!(
            pre.iters <= 3,
            "complete-factor preconditioner should need ≤3 iters, got {}",
            pre.iters
        );
        assert!(
            pre.iters * 5 < unpre.iters,
            "preconditioner should cut iterations sharply: {} vs {}",
            pre.iters,
            unpre.iters
        );
    }
}
