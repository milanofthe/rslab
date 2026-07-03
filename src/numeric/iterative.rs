//! Krylov iteration for complex-symmetric systems, preconditioned by an RLA
//! factorization.
//!
//! The target use is **3D EM FEM / MOM**: large complex-symmetric `A = Aᵀ`
//! (PARDISO `mtype 6`) systems solved iteratively, with a robust, memory-light
//! RLA factorization (static-pivoted, optionally `f32` / incomplete) as the
//! preconditioner. The iterative method of choice for `A = Aᵀ` is **COCG**
//! (Conjugate Orthogonal Conjugate Gradient, van der Vorst & Melissen 1990):
//! structurally CG, but every inner product is the *unconjugated* bilinear
//! form `xᵀy = Σ xᵢyᵢ` - the correct geometry for a complex-symmetric (not
//! Hermitian) operator. For `T = f64` it reduces exactly to preconditioned CG.
//!
//! The [`Preconditioner`] trait decouples the iteration from the factorization
//! precision: an `f64` factor applies directly; an `f32` factor (memory-halved)
//! down-/up-casts inside `apply`, while the iteration itself always runs in the
//! working precision `T`.
//!
//! ## Orthogonalization (GMRES paths)
//!
//! The two GMRES paths orthogonalize the Arnoldi basis by **different**, both
//! backward-stable, schemes (issue #8):
//!
//! - **Single-RHS [`gmres`]:** *modified* Gram-Schmidt (each projection updates
//!   `w` before the next is taken) with a conditional DGKS second pass, triggered
//!   when one MGS sweep cancels more than `1/√2` of `‖w‖`.
//! - **Block [`gmres_block`]:** *classical* Gram-Schmidt with a conditional second
//!   pass = **CGS2**, batched over the whole panel for BLAS-3 arithmetic
//!   intensity; the DGKS second pass is decided and applied **per column**.
//!
//! Consequently a **block solve with `s = 1` is *not* bit-identical to the single-
//! RHS [`gmres`]**: MGS and CGS accumulate the projections in a different order, so
//! the two can differ by a rounding ULP per step and hence by up to ±1 iteration
//! at a residual that straddles `tol`. Both converge to the same solution within
//! the requested tolerance. This is a deliberate design point (CGS2's panel-wide
//! reductions are what make the multi-RHS path fast and thread-count deterministic),
//! not a bug; see the `gmres_block_single_rhs_matches_scalar_gmres` test.

use crate::error::RslabError;
use crate::numeric::multifrontal_ldlt::SolverSettings;
use crate::numeric::sparse_solver::LdltSolver;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;
use rayon::prelude::*;

/// A linear operator `A`: applies `y = A x`. The Krylov solvers depend only on
/// this trait, so the operator may be an explicit sparse matrix
/// ([`CscMatrix`] symmetric / [`GeneralCsc`] general) **or matrix-free** - e.g.
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
    /// and scattered to all `s` columns (`y[:,c] += v·x[j,c]`, and symmetrically
    /// `y[j,c] += v·x[i,c]` off the diagonal) - the BLAS-3 reuse a multi-RHS
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
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError>;
    /// Block apply: `Z[:,c] ← M⁻¹ R[:,c]` for `c in 0..s`, with `R`,`Z` **column-
    /// major** `n×s`. The default loops [`apply`](Self::apply); a factored solver
    /// overrides it with a block triangular solve (`solve_many`) that loads each
    /// `L`/`D`/`U` value once for all `s` columns.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
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
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

/// A factored RLA solver is a preconditioner: `M⁻¹ r` is one forward/back
/// substitution against the stored `LDLᵀ` factor.
impl<T: Scalar> Preconditioner<T> for LdltSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via [`solve_many`](LdltSolver::solve_many): one block
    /// triangular solve loads each `L`/`D` value once for all `s` columns. The
    /// Krylov block is column-major; `solve_many` is row-major, so it is
    /// transposed in/out (`O(n·s)`, cheap against the solve).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
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
    inner: crate::numeric::multifrontal_lu::LuFactors<Complex<f32>>,
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
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), RslabError> {
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

/// Unconjugated bilinear inner product `xᵀy = Σ xᵢyᵢ` (no complex conjugation -
/// the defining choice of the COCG geometry for `A = Aᵀ`).
#[inline]
fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi * yi;
    }
    s
}

/// Euclidean norm `‖x‖₂ = √Σ|xᵢ|²` (genuine modulus - used only for the stopping
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
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
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
/// Same interface and conventions as [`cocg`]. Costs one matrix-vector product
/// and one preconditioner apply per iteration. Reduces to preconditioned CR
/// for `T = f64`.
pub fn cocr<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
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

/// Conjugated (Hermitian) inner product `⟨x, y⟩ = Σ conj(xᵢ)·yᵢ` - the geometry
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

/// Largest leading dimension `d ≤ jdim` whose Hessenberg diagonals are all above a
/// relative breakdown threshold `eps · max_i |h[i][i]|`. The upper-triangular
/// solve for `y` back-substitutes with `1/h[i][i]`; after Givens the diagonal is
/// `√(|f|²+|g|²)` and normally nonzero, but under exact stagnation or a rank-
/// deficient Hessenberg (hard / indefinite / singular operators) some `h[i][i]`
/// can be `0`, so an unguarded `recip()` would emit `Inf`/`NaN` into `x` and the
/// residual. Truncating the solve to this well-conditioned prefix instead yields a
/// **deterministic breakdown**: the leading Krylov block is solved, the degenerate
/// tail is dropped, and the outer true-residual check reports non-convergence. In
/// the well-conditioned case every diagonal clears the threshold and `jdim` is
/// returned unchanged, so the normal path is unaffected.
#[inline]
#[allow(clippy::needless_range_loop)]
fn well_conditioned_dim<T: Scalar>(h: &[Vec<T>], jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i][i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i][i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

/// Fixed row-block size for the block-orthogonalization reductions below. A
/// compile-time constant (never thread-count dependent), so the chunked sums are
/// **bit-identical across thread counts** - preserving the block solve's
/// determinism guarantee while spreading the reduction over all cores.
const ORTHO_CHUNK: usize = 2048;

/// Column-wise projection of a panel `W` (`n×sa`) onto each column's **own**
/// Arnoldi basis: `proj[i*sa + ap] = ⟨V_i[:,ap], W[:,ap]⟩` for block `i` in
/// `0..blocks` and active column `ap` in `0..sa`. The basis is blocks-major -
/// block `i` is the contiguous slice `vbas[i*sa*n .. (i+1)*sa*n]`, its column `ap`
/// at offset `+ap*n`; `W` column `ap` is `w[ap*n .. ap*n+n]`.
///
/// This is the classical (block) Gram-Schmidt projection: **all** projections are
/// taken against the same `W`, so the `blocks·sa` inner products are independent
/// and computed as one panel sweep instead of the `O(blocks·sa)` sequential,
/// latency-bound BLAS-1 reductions of modified Gram-Schmidt. The reduction is a
/// fixed row-chunk sum folded in chunk order → deterministic regardless of the
/// thread count.
/// `scratch` is a caller-owned reduction buffer of length `≥ nchunks · width`
/// (`nchunks = ⌈n/ORTHO_CHUNK⌉`, `width = blocks·sa`), reused across steps so the
/// hot loop allocates nothing. Each chunk writes its `width` partial sums into its
/// own slice; the slices are then folded in chunk order.
#[inline]
fn block_project<T: Scalar>(
    vbas: &[T],
    w: &[T],
    blocks: usize,
    sa: usize,
    n: usize,
    proj: &mut [T],
    scratch: &mut [T],
) {
    let width = blocks * sa;
    for p in proj[..width].iter_mut() {
        *p = T::zero();
    }
    if width == 0 || n == 0 {
        return;
    }
    let nchunks = n.div_ceil(ORTHO_CHUNK);
    let part = &mut scratch[..nchunks * width];
    part.par_chunks_mut(width).enumerate().for_each(|(ci, out)| {
        let r0 = ci * ORTHO_CHUNK;
        let r1 = (r0 + ORTHO_CHUNK).min(n);
        for i in 0..blocks {
            for ap in 0..sa {
                let vb = (i * sa + ap) * n;
                let wb = ap * n;
                let mut sdot = T::zero();
                for k in r0..r1 {
                    sdot = sdot + vbas[vb + k].conj() * w[wb + k];
                }
                out[i * sa + ap] = sdot;
            }
        }
    });
    // Fold partials in chunk order: the summation order is fixed by the chunk
    // layout, so the result does not depend on how many threads ran.
    for ci in 0..nchunks {
        let base = ci * width;
        for t in 0..width {
            proj[t] = proj[t] + part[base + t];
        }
    }
}

/// Subtract the projected components from the panel in place, per column:
/// `W[:,ap] -= Σ_i proj[i*sa+ap] · V_i[:,ap]`, accumulated in block order (`i`
/// ascending) at every element. Parallel over fixed row-chunks within each column
/// → the element-wise order is fixed, so the update is deterministic.
#[inline]
fn block_subtract<T: Scalar>(vbas: &[T], w: &mut [T], blocks: usize, sa: usize, n: usize, proj: &[T]) {
    for ap in 0..sa {
        let wcol = &mut w[ap * n..ap * n + n];
        wcol.par_chunks_mut(ORTHO_CHUNK).enumerate().for_each(|(ci, wc)| {
            let r0 = ci * ORTHO_CHUNK;
            for i in 0..blocks {
                let hij = proj[i * sa + ap];
                if hij == T::zero() {
                    continue;
                }
                let vb = (i * sa + ap) * n + r0;
                for k in 0..wc.len() {
                    wc[k] = wc[k] - hij * vbas[vb + k];
                }
            }
        });
    }
}

/// **Flexible** right-preconditioned restarted **GMRES(`restart`)** (FGMRES,
/// Saad 1993) for a general (unsymmetric) operator - the natural Krylov method
/// for unsymmetric MoM/FEM systems where COCG/COCR do not apply. `op` may be
/// matrix-free; `precond` supplies `M⁻¹` (e.g. an RLA
/// [`LuFactors`](crate::numeric::multifrontal_lu::LuFactors) near-field factor).
/// Solves `A x = b` from the optional initial guess `x0` (default `x₀ = 0`).
///
/// **Warm start (issue #5):** pass `x0 = Some(prev)` to seed the iteration from a
/// previous, related solution - on a sequence of slowly varying systems this
/// often cuts the iteration count substantially. Convergence is still measured
/// relative to ‖b‖.
///
/// **Flexible variant (issue #7):** the preconditioned Arnoldi vectors
/// `z_j = M⁻¹ v_j` (already formed to build `w = A z_j`) are *kept* as a second
/// basis `Z = [z_0 … z_{m-1}]`, and the restart update is `x += Z y` directly.
/// This (a) removes the one extra `M⁻¹` solve per cycle that plain right-
/// preconditioned GMRES spends rebuilding `M⁻¹(V y)`, and (b) makes the method
/// flexible: `M` may **vary** between steps (an inner Krylov solve, or a
/// preconditioner strengthened across iterations). Cost: one extra `n·(m+1)`
/// basis of storage.
pub fn gmres<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // DGKS reorthogonalization threshold: redo the projection only when a single
    // MGS pass cancelled more than this fraction of the vector's length.
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    let bnorm = norm2(b);
    // Warm start (issue #5): seed `x` from the caller's initial guess `x0`; the
    // per-cycle true residual `r = b − A x` then measures progress from that
    // guess. Convergence is still relative to ‖b‖. Absent `x0`, `x₀ = 0`.
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n {
                return Err(RslabError::DimensionMismatch { expected: n, got: g.len() });
            }
            g.to_vec()
        }
        None => vec![T::zero(); n],
    };
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = T::zero());
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
        });
    }

    let mut total = 0usize;
    let mut w = vec![T::zero(); n];
    let mut ax = vec![T::zero(); n];
    // Arnoldi basis as one **flat** `n × (m+1)` buffer (column `i` is
    // `v[i*n .. (i+1)*n]`) - contiguous, no per-iteration vector allocation, and
    // cache-friendly for the Gram-Schmidt sweeps.
    let mut v = vec![T::zero(); n * (m + 1)];
    // FGMRES preconditioned basis `Z = [z_0 … z_{m-1}]`, `z_j = M⁻¹ v_j` (issue
    // #7): stored as it is computed so the restart update is `x += Z y` with no
    // second preconditioner solve, and so a *variable* `M` is honoured exactly.
    let mut zb = vec![T::zero(); n * m];

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
            // Flexible right preconditioning: z_j = M⁻¹ v[j] (stored into the Z
            // basis for the restart update), then w = A z_j.
            precond.apply(&v[j * n..j * n + n], &mut zb[j * n..j * n + n])?;
            op.apply(&zb[j * n..j * n + n], &mut w);
            // Modified Gram-Schmidt against the existing basis, with **conditional**
            // reorthogonalization (DGKS): the second pass - essential on
            // ill-conditioned operators (MoM near-field) where a single MGS pass
            // loses orthogonality and the Hessenberg residual estimate drifts from
            // the true residual - runs only when the projection cancelled most of
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
        // Back-substitute the upper-triangular H for y, then (FGMRES) x += Z·y
        // directly from the stored preconditioned basis - no second `M⁻¹` solve.
        // Guard the solve against a (near-)singular Hessenberg diagonal: on
        // breakdown, solve only the well-conditioned leading block so a stagnated
        // / rank-deficient cycle degrades to a truncated update instead of
        // dividing by ~0.
        let jd = well_conditioned_dim(&h, jdim);
        let mut y = vec![T::zero(); jd];
        for i in (0..jd).rev() {
            let mut s = g[i];
            for k in (i + 1)..jd {
                s = s - h[i][k] * y[k];
            }
            y[i] = s * h[i][i].recip();
        }
        for i in 0..jd {
            let yi = y[i];
            for k in 0..n {
                x[k] = x[k] + zb[i * n + k] * yi;
            }
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

/// Form and add one converged column's solution contribution to the global `x`
/// **mid-cycle**, so the column can be compacted out of the active panel: back-
/// substitute `y` from that column's frozen Hessenberg/Givens state (`h`,`g`,`jd`
/// rows), build `V_ap · y` from its Arnoldi basis at the *current* stride `sa`,
/// apply the preconditioner once (`x_c += M⁻¹·(V_ap·y)`). This is the block
/// analogue of the single-RHS restart update, issued for a single column the
/// instant its Hessenberg estimate reaches `tol`, so the batched applies can then
/// shrink to the still-active width.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn finalize_block_column<T, M>(
    precond: &M,
    vbas: &[T],
    h: &[Vec<Vec<T>>],
    g: &[Vec<T>],
    jd: usize,
    ap: usize,
    sa: usize,
    n: usize,
    c: usize,
    x: &mut [T],
) -> Result<(), RslabError>
where
    T: Scalar,
    M: Preconditioner<T> + ?Sized,
{
    // Guard against a (near-)singular Hessenberg diagonal: solve only the well-
    // conditioned leading block (deterministic breakdown, no NaN into `x`).
    let jd = well_conditioned_dim(&h[ap], jd);
    if jd == 0 {
        return Ok(());
    }
    let mut y = vec![T::zero(); jd];
    for i in (0..jd).rev() {
        let mut acc = g[ap][i];
        for k in (i + 1)..jd {
            acc = acc - h[ap][i][k] * y[k];
        }
        y[i] = acc * h[ap][i][i].recip();
    }
    let mut vy = vec![T::zero(); n];
    for i in 0..jd {
        let yi = y[i];
        let vb = (i * sa + ap) * n;
        for k in 0..n {
            vy[k] = vy[k] + vbas[vb + k] * yi;
        }
    }
    let mut z = vec![T::zero(); n];
    precond.apply(&vy, &mut z)?;
    let cb = c * n;
    for k in 0..n {
        x[cb + k] = x[cb + k] + z[k];
    }
    Ok(())
}

/// Right-preconditioned restarted **block GMRES** for `s` right-hand sides `b`
/// (column-major `n×s`). The `s` systems advance in lockstep so the two expensive
/// operations - the operator matvec and the preconditioner solve - are issued
/// once per step as **block** applies ([`LinearOperator::apply_block`] /
/// [`Preconditioner::apply_block`]), reaching BLAS-3 arithmetic intensity (each
/// factor / matrix value touched once for all `s` columns). Each RHS keeps its
/// own Arnoldi basis / Hessenberg / Givens, so a column converges identically to
/// the single-RHS [`gmres`]; the systems share only the batched operator and
/// preconditioner calls. Solves `A X = B` from the optional initial guess `x0`
/// (column-major `n×s`, default `X₀ = 0`).
///
/// **Warm start (issue #5):** `x0 = Some(prev)` seeds every column from a related
/// previous solution; on a slowly varying sequence this cuts the block iteration
/// count. Each column's convergence is still relative to its own ‖B[:,c]‖.
///
/// **Deflation:** a RHS whose true residual reaches `tol` drops out of the block,
/// so the batched applies shrink to the active width as columns converge - the
/// fast-converging RHS are never dragged along by the slowest one.
///
/// This is the MoM/FEM many-excitations path: factor (or `f32`-factor) once, then
/// drive all right-hand sides through one block iteration.
///
/// **Orthogonalization (issue #8):** the panel is orthogonalized by **block CGS2**
/// (classical Gram-Schmidt with a conditional, now *per-column*, second pass), not
/// the MGS+DGKS of the single-RHS [`gmres`]. The two summation orders differ, so a
/// block solve with `s = 1` is **not** bit-identical to [`gmres`] and may differ by
/// up to ±1 iteration - both still converge to `tol`. See the module-level
/// "Orthogonalization" note for the rationale.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gmres_block<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if s == 0 || b.len() != n * s {
        return Err(RslabError::DimensionMismatch {
            expected: n * s,
            got: b.len(),
        });
    }
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    // Warm start (issue #5): seed every column from `x0` (column-major `n×s`).
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n * s {
                return Err(RslabError::DimensionMismatch { expected: n * s, got: g.len() });
            }
            g.to_vec()
        }
        None => vec![T::zero(); n * s],
    };
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
    // Block Gram-Schmidt projection panels: `proj[i*sa + ap]` for blocks `0..=j`,
    // columns `0..sa`. One classical (block) projection pass, plus a **conditional**
    // reorthogonalization pass (block DGKS) taken only when a column loses
    // orthogonality - the backward-stable, single-thread-cheap analogue of the
    // old per-RHS MGS+DGKS, now batched over the whole panel.
    let mut proj1 = vec![T::zero(); m * s];
    let mut proj2 = vec![T::zero(); m * s];
    let mut wnorm0 = vec![0.0f64; s]; // panel column norms before ortho (DGKS reorth test)
    let mut reorth_col = vec![false; s]; // per-column DGKS second-pass flags (issue #8)
    // Reduction scratch for `block_project`: `nchunks · (m·s)`, reused every step
    // so the orthogonalization allocates nothing in the hot loop.
    let mut proj_scratch = vec![T::zero(); n.div_ceil(ORTHO_CHUNK) * m * s];

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
            // Exact solution of `A x = 0` is `0`; discard any warm-start seed here.
            x[c * n..c * n + n].iter_mut().for_each(|v| *v = T::zero());
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
        let mut sa = act.len();
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
            // **Block Gram-Schmidt** of the whole `sa`-column panel `W` against each
            // column's own basis `V_0..V_j`: one classical projection pass
            // (project → subtract), then a **conditional** reorthogonalization pass
            // (block DGKS) taken only when a column's norm collapses - the
            // backward-stable, single-thread-cheap analogue of the old per-RHS
            // MGS+DGKS. Both passes are panel-wide, high-arithmetic-intensity sweeps
            // parallelized over the vector dimension, replacing the `O(j·sa)`
            // sequential BLAS-1 inner products. The reorth decision is taken from
            // serial column norms, so it is identical across thread counts (the
            // whole solve stays bit-identical regardless of parallelism). Converged
            // columns are still swept (kept in the panel for batching) but their
            // Hessenberg/Givens state is frozen below.
            let blocks = j + 1;
            for ap in 0..sa {
                wnorm0[ap] = norm2(&wblk[ap * n..ap * n + n]);
            }
            block_project(&vbas, &wblk, blocks, sa, n, &mut proj1, &mut proj_scratch);
            block_subtract(&vbas, &mut wblk, blocks, sa, n, &proj1);
            // **Per-column** DGKS second pass (issue #8): decide the reorth *per
            // column* from its own norm collapse, not panel-globally. Frozen
            // (converged-this-cycle) columns are excluded (they are compacted out at
            // each step boundary, so a stale/collapsed frozen column can never
            // trigger a reorth). Only columns that actually lost orthogonality get
            // the second projection subtracted; the rest keep their pass-1 result
            // bit-for-bit - a single ill-conditioned column no longer imposes the
            // arithmetic second orthogonalization on the well-conditioned columns.
            let mut any_reorth = false;
            for ap in 0..sa {
                let need = !inner_done[ap] && norm2(&wblk[ap * n..ap * n + n]) < REORTH_ETA * wnorm0[ap];
                reorth_col[ap] = need;
                any_reorth |= need;
            }
            if any_reorth {
                block_project(&vbas, &wblk, blocks, sa, n, &mut proj2, &mut proj_scratch);
                // Zero the second-pass projection for columns that do not need it, so
                // `block_subtract` skips them (its `hij == 0` guard) and their `w`
                // and Hessenberg entries stay exactly at the pass-1 values.
                for ap in 0..sa {
                    if !reorth_col[ap] {
                        for i in 0..blocks {
                            proj2[i * sa + ap] = T::zero();
                        }
                    }
                }
                block_subtract(&vbas, &mut wblk, blocks, sa, n, &proj2);
            }
            for ap in 0..sa {
                if inner_done[ap] {
                    continue;
                }
                let wb = ap * n;
                // Hessenberg column: projection pass, plus the per-column reorth
                // correction (zero for columns that did not reorth, so `hij == proj1`).
                for i in 0..=j {
                    let hij = if any_reorth { proj1[i * sa + ap] + proj2[i * sa + ap] } else { proj1[i * sa + ap] };
                    h[ap][i][j] = hij;
                }
                let hn = norm2(&wblk[wb..wb + n]);
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
            // **Within-cycle deflation.** Columns whose Hessenberg estimate reached
            // `tol` this step are (a) finalized now - their solution contribution is
            // formed from the basis at the *current* stride and added to `x` - and
            // (b) compacted out of the active panel, remapping the per-column
            // Arnoldi state and rewriting the basis at the narrower stride `sa`.
            // The next step's batched preconditioner / operator applies therefore
            // shrink to the still-active width, exactly as promised: a fast RHS is
            // no longer dragged along by the slowest one until the next restart.
            if inner_done.iter().any(|&d| d) {
                for ap in 0..sa {
                    if inner_done[ap] {
                        finalize_block_column(precond, &vbas, &h, &g, jdim[ap], ap, sa, n, act[ap], &mut x)?;
                    }
                }
                let survivors: Vec<usize> = (0..sa).filter(|&ap| !inner_done[ap]).collect();
                let sa_new = survivors.len();
                // Basis columns `0..=j+1` are populated for the survivors. Compact
                // to the new stride blocks-outer / survivors-inner so the write
                // offset is monotonically increasing and never clobbers an unread
                // source (each column's new offset is `<=` its old offset).
                let blocks_built = j + 2;
                for i in 0..blocks_built {
                    for (ap_new, &ap_old) in survivors.iter().enumerate() {
                        let src = (i * sa + ap_old) * n;
                        let dst = (i * sa_new + ap_new) * n;
                        if src != dst {
                            vbas.copy_within(src..src + n, dst);
                        }
                    }
                }
                // Remap the per-column Arnoldi state (Vec swaps are O(1) pointer
                // moves; the frozen columns' now-stale slots are never read again -
                // the next cycle re-initializes positions `0..sa`).
                for (ap_new, &ap_old) in survivors.iter().enumerate() {
                    if ap_new != ap_old {
                        h.swap(ap_new, ap_old);
                        cs.swap(ap_new, ap_old);
                        sn.swap(ap_new, ap_old);
                        g.swap(ap_new, ap_old);
                        jdim[ap_new] = jdim[ap_old];
                        act[ap_new] = act[ap_old];
                    }
                }
                sa = sa_new;
                act.truncate(sa);
                inner_done = vec![false; sa];
                if sa == 0 {
                    break;
                }
            }
        }

        // x_c += M⁻¹ (V_a y_a): back-substitute each still-active RHS, build the
        // compact VY block, one batched preconditioner apply, then scatter to
        // global `x`. Columns that deflated mid-cycle were already finalized
        // individually above, so `sa == 0` here means the whole panel converged
        // within the cycle and there is nothing left to batch.
        if sa > 0 {
            for e in vyblk[..sa * n].iter_mut() {
                *e = T::zero();
            }
            for ap in 0..sa {
                // Guard the per-column solve against a (near-)singular Hessenberg
                // diagonal: truncate to the well-conditioned leading block so a
                // rank-deficient column breaks down deterministically instead of
                // dividing by ~0 and polluting the batched applies with NaN.
                let jd = well_conditioned_dim(&h[ap], jdim[ap]);
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
/// sequentially) so the operator's own scratch lives in the closure capture - no struct, no
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
impl<T: Scalar, G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>> Preconditioner<T> for FnPc<G> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, 1)
    }
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, _n: usize) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, s)
    }
}

/// Closure entry point for [`gmres_block`]: pass the block matvec and block preconditioner as
/// `FnMut` closures plus the dimension `n` - the natural form for a **matrix-free** MoM/FEM
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
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp { f: std::cell::RefCell::new(op), n };
    let pc = FnPc { f: std::cell::RefCell::new(precond) };
    gmres_block(&op, b, s, &pc, tol, max_iter, restart, None)
}

/// Closure entry point for [`gmres`] (single RHS) - see [`gmres_block_fn`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp { f: std::cell::RefCell::new(op), n };
    let pc = FnPc { f: std::cell::RefCell::new(precond) };
    gmres(&op, b, &pc, tol, max_iter, restart, None)
}

/// A factorization usable as both a **direct solver** and a [`Preconditioner`].
/// Implemented by the symmetric [`LdltSolver`] and the general
/// [`LuFactors`](crate::numeric::multifrontal_lu::LuFactors), so a caller's
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

impl<T: Scalar> Preconditioner<T> for crate::numeric::multifrontal_lu::LuFactors<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = crate::numeric::multifrontal_lu::solve_lu(self, r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via `solve_lu_many` (one block triangular solve over all `s`
    /// columns). Column-major Krylov block ↔ row-major `solve_lu_many` transpose.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
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
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
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
/// preconditioner / factorization too - the unsymmetric twin of the
/// [`LdltSolver`] impls, so solver-in-the-loop code can be generic over either.
impl<T: Scalar> Preconditioner<T> for crate::numeric::multifrontal_lu::LuSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via [`LuSolver::solve_many`](crate::numeric::multifrontal_lu::LuSolver::solve_many).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
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
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
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
    use crate::numeric::multifrontal_ldlt::{SolverSettings, ZeroPivotAction};
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
        let un = gmres(&a, &b, &NoPreconditioner, 1e-10, 2000, 40, None).unwrap();
        assert!(un.converged, "GMRES res={}", un.final_res);

        // LU factor as preconditioner → 1-2 iterations.
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let pre = gmres(&a, &b, &lu, 1e-10, 200, 40, None).unwrap();
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

    #[test]
    fn gmres_singular_operator_breaks_down_without_nan() {
        // Rank-deficient operator (issue #11): A = diag(1, 1, 0) is singular and
        // `b = (1,1,1)` has a component in the null space (e₂), so GMRES cannot
        // drive the residual to zero - it stagnates. The Krylov subspace is
        // A-invariant with a *singular* restriction (eigenvalue 0), so the upper-
        // triangular Hessenberg factor acquires a ~0 diagonal. The unguarded
        // back-substitution would divide by it and emit NaN/Inf into `x` and the
        // reported residual; the guard must instead truncate to the well-
        // conditioned block, giving a deterministic breakdown: a finite iterate and
        // a truthful non-convergence report.
        use crate::sparse::general::GeneralCsc;
        let c = |re: f64, im: f64| Complex::new(re, im);
        // Only the (0,0) and (1,1) entries; row/col 2 is all-zero → A e₂ = 0.
        let a = GeneralCsc::<C>::from_triplets(3, &[0, 1], &[0, 1], &[c(1.0, 0.0), c(1.0, 0.0)]).unwrap();
        let b = vec![c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
        let res = gmres(&a, &b, &NoPreconditioner, 1e-12, 50, 10, None).unwrap();
        // No NaN/Inf reached the solution or the residual.
        assert!(res.x.iter().all(|z| z.re.is_finite() && z.im.is_finite()), "solution has NaN/Inf: {:?}", res.x);
        assert!(res.final_res.is_finite(), "residual is NaN/Inf: {}", res.final_res);
        // Deterministic breakdown: reported as non-converged with a sane residual
        // (the singular direction pins the relative residual near 1/√3 ≈ 0.577 - it
        // is bounded well below the blow-up an unguarded divide would produce).
        assert!(!res.converged, "singular system must not report convergence");
        assert!(res.final_res > 1e-12 && res.final_res <= 1.0, "residual not in the sane breakdown range: {}", res.final_res);
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

    /// Wraps a preconditioner and counts every scalar `apply` (the `M⁻¹` solves).
    struct CountingPc<'a, M: ?Sized> {
        inner: &'a M,
        applies: std::sync::atomic::AtomicUsize,
    }
    impl<T: Scalar, M: Preconditioner<T> + ?Sized> Preconditioner<T> for CountingPc<'_, M> {
        fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
            self.applies.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.apply(r, z)
        }
    }

    #[test]
    fn fgmres_saves_one_precond_apply_per_restart_cycle() {
        // FGMRES (issue #7): the preconditioned basis `Z` is kept, so the restart
        // update `x += Z y` costs **no** extra `M⁻¹` solve. The loop applies `M⁻¹`
        // exactly once per inner iteration and never at the restart, so over a
        // multi-cycle solve the total preconditioner-apply count equals the total
        // iteration count. Plain right-preconditioned GMRES would spend one extra
        // `M⁻¹` per cycle (rebuilding `M⁻¹(V y)`), i.e. `iters + n_cycles`.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(10); // n = 100
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        // Weak (heavily incomplete) factor so the solve needs several restart
        // cycles at a short restart length - exercising the per-cycle update path.
        let mut opts = SolverSettings::default();
        opts.drop_tol = Some(8e-1);
        let lu = factor_general_lu(&a, &opts).unwrap();
        let restart = 5;
        let counting = CountingPc { inner: &lu, applies: std::sync::atomic::AtomicUsize::new(0) };
        let res = gmres(&a, &b, &counting, 1e-10, 2000, restart, None).unwrap();
        assert!(res.converged, "FGMRES must converge, res={}", res.final_res);
        let applies = counting.applies.load(std::sync::atomic::Ordering::Relaxed);
        // Multiple restart cycles actually occurred (proves the saving is nonzero).
        assert!(res.iters > restart, "expected multiple cycles, iters={}", res.iters);
        // FGMRES: exactly one apply per inner iteration, none at restart.
        assert_eq!(
            applies, res.iters,
            "FGMRES precond applies {} must equal iters {} (no per-cycle extra solve)",
            applies, res.iters
        );
    }

    #[test]
    fn gmres_warm_start_cuts_total_iterations_on_related_sequence() {
        // Warm start (issue #5): a sequence of related systems `A x = b_k` with a
        // slowly rotating right-hand side. Cold-starting every solve from 0 pays
        // the full iteration count each time; seeding each solve with the previous
        // solution (which is close, because the RHS barely moved) collapses the
        // per-solve count. Total iterations must drop by a clear margin.
        let a = unsym_grid(12); // n = 144
        let n = a.n;
        let c = |re: f64, im: f64| Complex::new(re, im);
        // Two fixed directions; b_k interpolates between them by a small angle step.
        let b0: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let b1: Vec<C> = (0..n).map(|i| c(((i * 3) % 7) as f64 - 3.0, ((i % 3) as f64) - 1.0)).collect();
        let steps = 10;
        let (tol, maxit, restart) = (1e-8, 4000, 60);

        let bk = |k: usize| -> Vec<C> {
            let th = 5e-5 * k as f64; // slow rotation
            let (ct, st) = (th.cos(), th.sin());
            (0..n).map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0)).collect()
        };

        // Cold: every solve from x0 = 0.
        let mut cold_total = 0usize;
        for k in 0..steps {
            let r = gmres(&a, &bk(k), &NoPreconditioner, tol, maxit, restart, None).unwrap();
            assert!(r.converged, "cold solve {k} did not converge");
            cold_total += r.iters;
        }

        // Warm: first solve from 0, each subsequent seeded with the previous x.
        let mut warm_total = 0usize;
        let mut prev: Option<Vec<C>> = None;
        for k in 0..steps {
            let r = gmres(&a, &bk(k), &NoPreconditioner, tol, maxit, restart, prev.as_deref()).unwrap();
            assert!(r.converged, "warm solve {k} did not converge");
            warm_total += r.iters;
            prev = Some(r.x);
        }

        // A meaningful reduction (well beyond noise): warm start must cut the total
        // iteration count by at least 30% over the sequence.
        assert!(
            (warm_total as f64) < 0.7 * (cold_total as f64),
            "warm start did not help enough: cold={cold_total}, warm={warm_total}"
        );
    }

    #[test]
    fn gmres_block_single_rhs_matches_scalar_gmres() {
        // s = 1 block GMRES reduces to the single-RHS path (same Arnoldi, same
        // Givens, default block apply = one single apply): same solution to the
        // requested tolerance, same iteration count up to a ±1 boundary effect.
        // The paths are NOT bit-identical - block uses CGS2, single uses MGS+DGKS,
        // so the projections sum in a different order and the true residual can
        // straddle `tol` by a rounding ULP. This is documented as a design point in
        // the module-level "Orthogonalization" note (issue #8), not a defect.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(8);
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let single = gmres(&a, &b, &lu, 1e-10, 200, 40, None).unwrap();
        let blk = gmres_block(&a, &b, 1, &lu, 1e-10, 200, 40, None).unwrap();
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
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let res = gmres_block(&a, &bblk, s, &lu, 1e-10, 200, 40, None).unwrap();
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
            let single = gmres(&a, &bblk[k * n..k * n + n], &lu, 1e-10, 200, 40, None).unwrap();
            let diff = (0..n)
                .map(|i| (res.x[k * n + i] - single.x[i]).norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-9, "column {k} differs from single solve by {diff}");
        }
    }

    #[test]
    fn gmres_block_within_cycle_deflation_shrinks_applies() {
        // Different-convergence-rate regime (issue #4): a diagonal operator with
        // distinct eigenvalues, unpreconditioned, with RHS `k` supported on `k+1`
        // distinct eigenvalues. GMRES on such a RHS converges in exactly `k+1`
        // steps, so the columns finish at staggered steps within a *single* cycle.
        // The fix must (i) still solve every column exactly like single-RHS GMRES
        // and (ii) actually shrink the batched operator applies as columns deflate
        // mid-cycle - not only at restart. A counting operator records the width
        // of every `apply_block`, and we assert the panel narrows.
        use std::sync::Mutex;

        struct CountingOp<'a> {
            inner: &'a GeneralCsc<C>,
            widths: Mutex<Vec<usize>>,
        }
        impl LinearOperator<C> for CountingOp<'_> {
            fn n(&self) -> usize {
                self.inner.n()
            }
            fn apply(&self, x: &[C], y: &mut [C]) {
                self.widths.lock().unwrap().push(1);
                self.inner.apply(x, y);
            }
            fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
                self.widths.lock().unwrap().push(s);
                self.inner.apply_block(x, y, s);
            }
        }

        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 8;
        let s = 4;
        // Diagonal operator with distinct entries → the minimal polynomial degree
        // of a RHS equals the number of distinct diagonal entries in its support.
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rr.push(i);
            cc.push(i);
            vv.push(c(2.0 + i as f64, 0.5 + 0.1 * i as f64));
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        // RHS `k` = sum of the first `k+1` unit vectors → converges in `k+1` steps.
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..=k {
                bblk[k * n + i] = c(1.0, 0.0);
            }
        }

        let op = CountingOp {
            inner: &a,
            widths: Mutex::new(Vec::new()),
        };
        let res = gmres_block(&op, &bblk, s, &NoPreconditioner, 1e-12, 200, 40, None).unwrap();
        assert!(res.converged, "block GMRES must converge; res={:?}", res.final_res);

        // (i) Every column matches the single-RHS GMRES solve of that column.
        for k in 0..s {
            let single = gmres(&a, &bblk[k * n..k * n + n], &NoPreconditioner, 1e-12, 200, 40, None).unwrap();
            let diff = (0..n)
                .map(|i| (res.x[k * n + i] - single.x[i]).norm())
                .fold(0.0, f64::max);
            assert!(diff < 1e-9, "column {k} differs from single solve by {diff}");
        }

        // (ii) The batched applies actually shrank mid-cycle. Without within-cycle
        // deflation every `apply_block` would run at full width `s`; with it, later
        // steps run narrower. Assert a full-width apply happened (the first step),
        // that some apply narrowed below `s`, that the panel drained to width 1,
        // and that the total column-applies fell below the full-width bound.
        let widths = op.widths.into_inner().unwrap();
        let ncalls = widths.len();
        let total_cols: usize = widths.iter().sum();
        assert_eq!(*widths.iter().max().unwrap(), s, "the first cycle must open at full width");
        assert!(*widths.iter().min().unwrap() < s, "no apply narrowed: deflation did not shrink the panel");
        assert!(widths.contains(&1), "the panel must drain to a single active column");
        assert!(
            total_cols < s * ncalls,
            "total column-applies {total_cols} not below the full-width bound {}",
            s * ncalls
        );
    }

    #[test]
    fn gmres_block_bcgs2_bit_identical_across_thread_counts() {
        // The block-CGS2 orthogonalization reduces over fixed row-chunks folded in
        // chunk order, so the whole block solve is **bit-identical regardless of
        // the thread count** - the determinism guarantee. Solve the same block in
        // a 1-thread and an 8-thread rayon pool and require exact equality.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        // Wide enough that a chunked reduction actually spans several chunks.
        let a = unsym_grid(60);
        let n = a.n;
        let s = 5;
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let solve = || gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
        let x1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(solve);
        let x8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap().install(solve);
        assert_eq!(x1.iters, x8.iters, "iteration count must not depend on threads");
        assert!(x1.x == x8.x, "block solve must be bit-identical across thread counts");
    }

    #[test]
    fn with_threads_caps_block_gmres_pool_and_keeps_result() {
        // `with_threads(p)` runs the block solve in a scoped pool of exactly `p`
        // workers (the embedded / solver-in-the-loop cap) and produces the same
        // result as the unbounded solve.
        use crate::numeric::multifrontal_ldlt::with_threads;
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(30);
        let n = a.n;
        let s = 4;
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let seen = with_threads(3, || rayon::current_num_threads());
        assert_eq!(seen, 3, "with_threads must cap the pool to the requested width");
        let capped = with_threads(3, || gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap());
        let plain = gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
        assert!(capped.converged);
        assert!(capped.x == plain.x, "capped-pool solve must match the unbounded solve bit-for-bit");
    }

    #[test]
    fn ambient_threads_factor_matches_default_and_runs_on_shared_pool() {
        // `Threads::Ambient` factors on the current pool (no new spawn) - the
        // re-factor-in-loop path. Inside a `with_threads(2)` pool the factor must be
        // bit-identical to the normal (scoped-pool) factor: the numeric result is
        // independent of the thread policy.
        use crate::numeric::multifrontal_ldlt::{with_threads, Threads};
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(24);
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let lu_default = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let opts_amb = SolverSettings::default().with_thread_policy(Threads::Ambient);
        let lu_amb = with_threads(2, || {
            assert_eq!(rayon::current_num_threads(), 2);
            factor_general_lu(&a, &opts_amb).unwrap()
        });
        let x_def = gmres_block(&a, &b, 1, &lu_default, 1e-10, 200, 40, None).unwrap();
        let x_amb = gmres_block(&a, &b, 1, &lu_amb, 1e-10, 200, 40, None).unwrap();
        assert!(x_def.x == x_amb.x, "ambient-pool factor must be bit-identical to the default factor");
    }

    #[test]
    fn default_thread_policy_caps_at_four() {
        // The pareto-optimal embedded default: predict per matrix, never exceed 4.
        use crate::numeric::multifrontal_ldlt::Threads;
        assert_eq!(SolverSettings::default().threads, Threads::Auto { max: 4 });
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
        // iterations - ample for a MoM/FEM Krylov tolerance, at half the factor
        // memory. (A residual below ~1e-6 is not attainable with an f32 apply;
        // use the f64 factor for tighter tolerances.)
        let pc = LowPrecisionLu::factor(&a, &SolverSettings::default()).unwrap();
        assert!(pc.factor_nnz() > 0);
        let res = gmres(&a, &b, &pc, 1e-6, 200, 50, None).unwrap();
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
        let opts = SolverSettings {
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
        // weaker preconditioner - but COCG must still converge to the true
        // f64 solution. Demonstrates the memory ↔ iteration tradeoff.
        let c = |re, im| Complex::new(re, im);
        let a = grid(16, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let full = LdltSolver::factor(&a).unwrap();
        let opts = SolverSettings {
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
        // factor - though approximate - keeps the iteration count tiny.
        let c = |re, im| Complex::new(re, im);
        let a = grid(14, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let m = LowPrecisionPreconditioner::factor(&a, &SolverSettings::default()).unwrap();
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
        // converge in a handful of iterations - vastly fewer than without.
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
