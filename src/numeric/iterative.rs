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
use crate::numeric::sparse_solver::SparseSymmetricLdlt;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;

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

#[cfg(test)]
mod tests {
    use super::*;
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
