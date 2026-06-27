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
use crate::numeric::multifrontal_generic::GenericFactorOptions;
use crate::numeric::sparse_solver::SparseSymmetricLdlt;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use num_complex::Complex;

/// A preconditioner `M ≈ A`: applies `z = M⁻¹ r`. Implemented by a factored
/// [`SparseSymmetricLdlt`](crate::numeric::sparse_solver::SparseSymmetricLdlt)
/// and by [`NoPreconditioner`] (the unpreconditioned baseline).
pub trait Preconditioner<T: Scalar> {
    /// Write `z ← M⁻¹ r`. `r` and `z` have length `n`.
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError>;
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
impl<T: Scalar> Preconditioner<T> for SparseSymmetricLdlt<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), FeralError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
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
    inner: SparseSymmetricLdlt<Complex<f32>>,
}

impl LowPrecisionPreconditioner {
    /// Down-cast `A` to `Complex<f32>` and factor it (static-pivoting honoured
    /// via `opts`, e.g. `ZeroPivotAction::PerturbToEps`).
    pub fn factor(
        a: &CscMatrix<Complex<f64>>,
        opts: &GenericFactorOptions,
    ) -> Result<Self, FeralError> {
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
            inner: SparseSymmetricLdlt::factor_with(&a32, opts)?,
        })
    }

    /// Stored factor fill (nnz of `L`); each entry is a single-precision
    /// `Complex<f32>` (8 bytes vs 16 for `Complex<f64>`).
    pub fn factor_nnz(&self) -> usize {
        self.inner.factor_nnz()
    }

    /// Number of statically perturbed pivots (see [`SparseSymmetricLdlt::n_perturbed`]).
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
pub fn cocg<T, M>(
    a: &CscMatrix<T>,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
) -> Result<KrylovResult<T>, FeralError>
where
    T: Scalar,
    M: Preconditioner<T> + ?Sized,
{
    let n = a.n;
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
        a.symv(&p, &mut q); // q = A p
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
pub fn cocr<T, M>(
    a: &CscMatrix<T>,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
) -> Result<KrylovResult<T>, FeralError>
where
    T: Scalar,
    M: Preconditioner<T> + ?Sized,
{
    let n = a.n;
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
    a.symv(&p, &mut ap); // A p
    let mut az = ap.clone(); // A z (= A p at init since p = z)
    let mut gamma = dotu(&z, &az); // zᵀ A z
    let mut w = vec![T::zero(); n]; // M⁻¹ A p
    let mut aw = vec![T::zero(); n]; // A w

    let mut final_res = norm2(&r) / bnorm;
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iter {
        precond.apply(&ap, &mut w)?; // w = M⁻¹ A p
        a.symv(&w, &mut aw); // A w
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::factor::ZeroPivotAction;
    use crate::numeric::multifrontal_generic::GenericFactorOptions;
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
        let m = SparseSymmetricLdlt::factor(&a).unwrap();
        let pre = cocr(&a, &b, &m, 1e-10, 3000).unwrap();
        assert!(pre.converged && pre.iters <= 3, "iters {}", pre.iters);
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
        let opts = GenericFactorOptions {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
            drop_tol: None,
        };
        let m = SparseSymmetricLdlt::factor_with(&a, &opts).unwrap();
        let pre = cocr(&a, &b, &m, 1e-9, 500).unwrap();
        assert!(pre.converged, "indefinite COCR res={} iters={}", pre.final_res, pre.iters);
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

        let full = SparseSymmetricLdlt::factor(&a).unwrap();
        let opts = GenericFactorOptions {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
        };
        let inc = SparseSymmetricLdlt::factor_with(&a, &opts).unwrap();

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

        let m = LowPrecisionPreconditioner::factor(&a, &GenericFactorOptions::default()).unwrap();
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
        let m = SparseSymmetricLdlt::factor(&a).unwrap();
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
