//! High-level generic sparse symmetric direct solver.
//!
//! [`SparseSymmetricLdlt`] wraps the generic multifrontal factorization
//! ([`crate::numeric::multifrontal_generic`]) with symmetric equilibration and
//! a convenient factor-once / solve-many interface. It works for both `f64`
//! (real symmetric) and `Complex<f64>` (complex symmetric, PARDISO `mtype 6`).
//!
//! ## Equilibration
//!
//! Before factoring, the matrix is symmetrically scaled `Â = D A D` with a
//! **real** diagonal `D = diag(s)`, `s_i = 1/√rᵢ`, where `rᵢ = maxⱼ |Aᵢⱼ|` is
//! the row magnitude. This one-pass infinity-norm equilibration improves
//! conditioning and, because it uses off-diagonal magnitudes, tolerates a zero
//! diagonal (common in complex-symmetric and saddle-point systems). Solving
//! `A x = b` becomes: factor `Â`, then `x = D · (Â⁻¹ · (D b))`.

use crate::dense::ldlt_generic::{solve_ldlt, LdltFactors};
use crate::error::FeralError;
use crate::numeric::multifrontal_generic::factor_sparse_ldlt;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;

/// A factored sparse symmetric matrix, ready to solve against many right-hand
/// sides. Generic over the scalar field `T` (`f64` or `Complex<f64>`).
pub struct SparseSymmetricLdlt<T> {
    /// Factors of the equilibrated matrix `Â = D A D`, in factorization order.
    factors: LdltFactors<T>,
    /// Real symmetric equilibration diagonal `s` (`D = diag(s)`).
    scale: Vec<f64>,
}

impl<T: Scalar> SparseSymmetricLdlt<T> {
    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Equilibrate and factor `A` as `Â = D A D = Pᵀ L D_bk Lᵀ P`.
    pub fn factor(a: &CscMatrix<T>) -> Result<Self, FeralError> {
        a.validate()?;
        let n = a.n;

        // Row magnitudes rᵢ = maxⱼ |Aᵢⱼ| over the symmetric matrix (lower
        // triangle stored, so each off-diagonal updates both endpoints).
        let mut row_max = vec![0.0f64; n];
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                let m = a.values[k].magnitude();
                if m > row_max[i] {
                    row_max[i] = m;
                }
                if i != j && m > row_max[j] {
                    row_max[j] = m;
                }
            }
        }
        // sᵢ = 1/√rᵢ; an all-zero row (rᵢ = 0) is left unscaled and will surface
        // as a singular pivot during factorization.
        let scale: Vec<f64> = row_max
            .iter()
            .map(|&r| if r > 0.0 { 1.0 / r.sqrt() } else { 1.0 })
            .collect();

        // Scaled values Âᵢⱼ = sᵢ · Aᵢⱼ · sⱼ (structure unchanged). Built in CSC
        // order so it lines up with `a.col_ptr`/`a.row_idx`.
        let mut scaled_values = Vec::with_capacity(a.values.len());
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                scaled_values.push(a.values[k] * T::from_real(scale[i] * scale[j]));
            }
        }
        let scaled = CscMatrix::<T> {
            n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            values: scaled_values,
        };

        let factors = factor_sparse_ldlt(&scaled)?;
        Ok(Self { factors, scale })
    }

    /// Solve `A · x = rhs` using the stored factors.
    pub fn solve(&self, rhs: &[T]) -> Result<Vec<T>, FeralError> {
        let n = self.factors.n;
        if rhs.len() != n {
            return Err(FeralError::DimensionMismatch {
                expected: n,
                got: rhs.len(),
            });
        }
        // b̂ = D b
        let b_hat: Vec<T> = rhs
            .iter()
            .zip(&self.scale)
            .map(|(&r, &s)| r * T::from_real(s))
            .collect();
        // ẑ = Â⁻¹ b̂
        let mut x = solve_ldlt(&self.factors, &b_hat)?;
        // x = D ẑ
        for (xi, &s) in x.iter_mut().zip(&self.scale) {
            *xi = *xi * T::from_real(s);
        }
        Ok(x)
    }

    /// Solve `A · x = rhs` with iterative refinement against the original
    /// matrix `a` (which must be the matrix this was factored from). Each step
    /// computes the residual `r = rhs − A x` and applies the correction
    /// `x ← x + A⁻¹ r`, stopping once `‖r‖∞` stops improving or `max_iter` is
    /// reached. This recovers accuracy lost to the within-fully-summed-block
    /// pivoting on harder indefinite systems, at the cost of a few extra solves.
    pub fn solve_refined(
        &self,
        a: &CscMatrix<T>,
        rhs: &[T],
        max_iter: usize,
    ) -> Result<Vec<T>, FeralError> {
        let n = self.factors.n;
        if a.n != n {
            return Err(FeralError::DimensionMismatch {
                expected: n,
                got: a.n,
            });
        }
        let mut x = self.solve(rhs)?;
        let mut ax = vec![T::zero(); n];
        let mut best_x = x.clone();
        let mut best_res = f64::INFINITY;
        // `max_iter` correction steps, plus the initial residual evaluation.
        for _ in 0..=max_iter {
            a.symv(&x, &mut ax);
            let r: Vec<T> = rhs.iter().zip(&ax).map(|(&b, &axi)| b - axi).collect();
            let res = r.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            // Track the best iterate — refinement can be non-monotone on very
            // ill-conditioned systems.
            if res < best_res {
                best_res = res;
                best_x.clone_from(&x);
            }
            if res == 0.0 {
                break;
            }
            let dx = self.solve(&r)?;
            for (xi, &d) in x.iter_mut().zip(&dx) {
                *xi = *xi + d;
            }
        }
        Ok(best_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn residual_inf<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn f64_badly_scaled_diagonal() {
        // Diagonal entries spanning ~10 orders of magnitude. Equilibration
        // should keep the solve accurate on the original system.
        let n = 12;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(10.0_f64.powi(j as i32 - 6)); // 1e-6 .. 1e5
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let solver = SparseSymmetricLdlt::factor(&a).unwrap();
        let x = solver.solve(&b).unwrap();
        // Relative residual (the absolute one is dominated by the 1e5 row).
        let mut ax = vec![0.0; n];
        a.symv(&x, &mut ax);
        let rel = (0..n)
            .map(|i| (ax[i] - b[i]).abs() / b[i].abs().max(1.0))
            .fold(0.0, f64::max);
        assert!(rel < 1e-10, "relative residual {}", rel);
    }

    #[test]
    fn complex_grid_solve() {
        let c = |re, im| Complex::new(re, im);
        let m = 6;
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
                    vals.push(c(-1.0, 0.3));
                }
                if r + 1 < m {
                    let q = idx(r + 1, cc);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.3));
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let solver = SparseSymmetricLdlt::factor(&a).unwrap();

        // Solve against two different right-hand sides with the one factor.
        for shift in [0.0, 1.0] {
            let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 10.0 + shift, 1.0)).collect();
            let x = solver.solve(&b).unwrap();
            assert!(
                residual_inf(&a, &x, &b) < 1e-9,
                "residual {}",
                residual_inf(&a, &x, &b)
            );
        }
    }

    #[test]
    fn refined_solve_is_no_worse_than_plain() {
        // Complex-symmetric tridiagonal; refinement must not increase the
        // residual and should reach near machine precision.
        let c = |re, im| Complex::new(re, im);
        let n = 30;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(c(3.0, 0.4));
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(c(-1.0, 0.2));
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 15.0, 2.0)).collect();
        let solver = SparseSymmetricLdlt::factor(&a).unwrap();

        let x_plain = solver.solve(&b).unwrap();
        let x_ref = solver.solve_refined(&a, &b, 3).unwrap();
        let r_plain = residual_inf(&a, &x_plain, &b);
        let r_ref = residual_inf(&a, &x_ref, &b);
        assert!(r_ref <= r_plain.max(1e-300), "refined {} vs plain {}", r_ref, r_plain);
        assert!(r_ref < 1e-12, "refined residual {}", r_ref);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 3.0]).unwrap();
        let solver = SparseSymmetricLdlt::factor(&a).unwrap();
        assert!(matches!(
            solver.solve(&[1.0, 2.0, 3.0]),
            Err(FeralError::DimensionMismatch { .. })
        ));
    }
}
