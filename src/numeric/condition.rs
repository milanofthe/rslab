//! One-norm condition estimation for the unsymmetric LU factor (feral #94
//! port, generalized over the [`Scalar`] field), plus the classical
//! element-growth diagnostic (feral #93 analogue).
//!
//! The estimator is the Hager (1984) / Higham (1988) DLACON-style power
//! iteration on the unit 1-ball: each iteration is one `A⁻¹` solve and one
//! `A⁻ᴴ` solve through the stored factor (the LU is unsymmetric, so the
//! symmetric `A⁻ᵀ = A⁻¹` shortcut does not apply), capped at 5 iterations
//! with the alternating-sign Higham refinement as a floor. The result is a
//! **lower bound** on `‖A⁻¹‖₁` (usually within a small factor);
//! `κ₁ ≈ ‖A‖₁ · ‖A⁻¹‖₁` follows with [`GeneralCsc::one_norm`].
//!
//! This is the solver-in-the-loop enabler for "ill-conditioned system →
//! refine / re-solve with perturbation" decisions inside one engine,
//! without a dense inverse or an SVD.

use crate::error::RslabError;
use crate::numeric::multifrontal_lu::{solve_lu, solve_lu_transpose, LuSolver};
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

/// The two solves one Hager-Higham iteration needs, in place on `rhs`.
pub trait ConditionOperator<T: Scalar> {
    fn dim(&self) -> usize;
    /// `rhs ← A⁻¹ · rhs`.
    fn apply_inverse(&self, rhs: &mut [T]) -> Result<(), RslabError>;
    /// `rhs ← A⁻ᴴ · rhs` (conjugate transpose; the plain transpose for real
    /// fields).
    fn apply_inverse_adjoint(&self, rhs: &mut [T]) -> Result<(), RslabError>;
}

/// Iteration cap (LAPACK `dlacon` uses 5; the estimate almost always
/// converges in 2-3).
const HAGER_MAX_ITER: usize = 5;

/// Estimate `‖A⁻¹‖₁` by the Hager-Higham power iteration on the unit
/// 1-ball, driving all solves through `op`. Returns a lower bound; `0.0`
/// for an empty operator.
///
/// Field-generic: the real algorithm's `ξ = sign(y)` becomes the unit phase
/// `ξᵢ = yᵢ/|yᵢ|` (LAPACK `zlacon`'s generalization) and the local-max test
/// compares `‖z‖∞` against `Re⟨x, z⟩`, which reduces to the classical
/// `zᵀx` for real scalars.
pub fn hager_higham_inverse_norm_1<T: Scalar, O: ConditionOperator<T> + ?Sized>(
    op: &O,
) -> Result<f64, RslabError> {
    let n = op.dim();
    if n == 0 {
        return Ok(0.0);
    }
    let one_norm = |v: &[T]| v.iter().map(|x| x.magnitude()).sum::<f64>();

    let mut x: Vec<T> = vec![T::from_real(1.0 / n as f64); n];
    let mut est = 0.0f64;
    let mut last_j = usize::MAX;
    for iter in 0..HAGER_MAX_ITER {
        let mut y = x.clone();
        op.apply_inverse(&mut y)?; // y = A⁻¹ x
        let e = one_norm(&y);
        if iter > 0 && e <= est {
            break; // estimate stopped growing
        }
        est = e;
        // ξ = phase(y): sign for real fields, y/|y| for complex (1 at 0).
        let mut z: Vec<T> = y
            .iter()
            .map(|&v| {
                let m = v.magnitude();
                if m == 0.0 {
                    T::one()
                } else {
                    v * T::from_real(1.0 / m)
                }
            })
            .collect();
        // z = A⁻ᴴ ξ.
        op.apply_inverse_adjoint(&mut z)?;
        // Local max on the unit 1-ball: ‖z‖∞ ≤ Re⟨x, z⟩ means no vertex
        // improves on the current x.
        let zx: f64 = z
            .iter()
            .zip(&x)
            .map(|(&zi, &xi)| (zi.conj() * xi).real())
            .sum();
        let (mut jmax, mut zmax) = (0usize, -1.0f64);
        for (j, &zj) in z.iter().enumerate() {
            let m = zj.magnitude();
            if m > zmax {
                zmax = m;
                jmax = j;
            }
        }
        if zmax <= zx || jmax == last_j {
            break;
        }
        last_j = jmax;
        x.iter_mut().for_each(|v| *v = T::zero());
        x[jmax] = T::one();
    }

    // Higham's alternating-sign refinement: a deliberately awkward RHS that
    // catches the cases the power iteration underestimates; take the max.
    let denom = (n.max(2) - 1) as f64;
    let mut b: Vec<T> = (0..n)
        .map(|i| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            T::from_real(sign * (1.0 + i as f64 / denom))
        })
        .collect();
    op.apply_inverse(&mut b)?;
    let alt = 2.0 * one_norm(&b) / (3.0 * n as f64);
    Ok(est.max(alt))
}

/// [`ConditionOperator`] over a stored [`LuSolver`] factor: `A⁻¹` via
/// [`solve_lu`], `A⁻ᴴ` via conjugating around [`solve_lu_transpose`]
/// (`A⁻ᴴ b = conj(A⁻ᵀ conj(b))`; the conjugations are the identity for
/// real fields).
struct LuConditionOp<'a, T: Scalar> {
    solver: &'a LuSolver<T>,
}

impl<T: Scalar> ConditionOperator<T> for LuConditionOp<'_, T> {
    fn dim(&self) -> usize {
        self.solver.factors().n
    }
    fn apply_inverse(&self, rhs: &mut [T]) -> Result<(), RslabError> {
        let x = solve_lu(self.solver.factors(), rhs)?;
        rhs.copy_from_slice(&x);
        Ok(())
    }
    fn apply_inverse_adjoint(&self, rhs: &mut [T]) -> Result<(), RslabError> {
        rhs.iter_mut().for_each(|v| *v = v.conj());
        let x = solve_lu_transpose(self.solver.factors(), rhs)?;
        for (r, xi) in rhs.iter_mut().zip(&x) {
            *r = xi.conj();
        }
        Ok(())
    }
}

impl<T: Scalar> LuSolver<T> {
    /// Hager-Higham one-norm condition estimate `κ₁ ≈ ‖A‖₁ · ‖A⁻¹‖₁` for
    /// the factored system (feral #94 port). `a` must be the matrix this
    /// solver was factored from (the factor does not retain it; it supplies
    /// `‖A‖₁`). The estimate is a **lower bound** on the true `κ₁`, usually
    /// within a small factor; ≤ 5 solve/transpose-solve pairs, no dense
    /// work. Use it to route "ill-conditioned → refine or re-factor with
    /// tighter pivoting" decisions.
    pub fn condest_1(&self, a: &GeneralCsc<T>) -> Result<f64, RslabError> {
        let n = self.factors().n;
        if a.n != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: a.n,
            });
        }
        let op = LuConditionOp { solver: self };
        Ok(a.one_norm() * hager_higham_inverse_norm_1(&op)?)
    }

    /// Classical element-growth factor `ρ = max|uᵢⱼ| / max|âᵢⱼ|` of the
    /// stored factor, where `Â = D_r A D_c` is the equilibrated matrix that
    /// was actually factored (feral #93 analogue; with unit `L`, `U` carries
    /// the growth). `a` must be the matrix this solver was factored from.
    /// Large growth (≫ 1) flags pivot instability that the static-pivoting
    /// policy could not contain - the standard companion signal to
    /// [`condest_1`](Self::condest_1). Returns `0.0` for an empty/zero
    /// matrix.
    pub fn element_growth(&self, a: &GeneralCsc<T>) -> Result<f64, RslabError> {
        let f = self.factors();
        if a.n != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: a.n,
            });
        }
        let mut max_a = 0.0f64;
        for j in 0..a.n {
            let dc = f.d_col[j];
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let m = a.values[k].magnitude() * f.d_row[a.row_idx[k]] * dc;
                if m > max_a {
                    max_a = m;
                }
            }
        }
        if max_a == 0.0 {
            return Ok(0.0);
        }
        let max_u = f
            .u_values
            .iter()
            .map(|v| v.magnitude())
            .fold(0.0f64, f64::max);
        Ok(max_u / max_a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::multifrontal_ldlt::SolverSettings;
    use num_complex::Complex;

    fn lu(a: &GeneralCsc<f64>) -> LuSolver<f64> {
        LuSolver::factor(a, &SolverSettings::default()).unwrap()
    }

    /// `‖·‖₁` hand oracle: `[[1,2],[3,4]]` has column sums 4 and 6.
    #[test]
    fn one_norm_hand_value() {
        let a = GeneralCsc::from_triplets(2, &[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 3.0, 2.0, 4.0])
            .unwrap();
        assert_eq!(a.one_norm(), 6.0);
    }

    /// Transpose solve: `Aᵀ x = b` must hold against an explicitly built
    /// `Aᵀ` matvec, on an unsymmetric matrix with nontrivial equilibration
    /// and pivoting.
    #[test]
    fn transpose_solve_residual() {
        let k = 12;
        let n = k * k;
        let idx = |x: usize, y: usize| y * k + x;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y);
                r.push(p);
                c.push(p);
                v.push(4.0 + (p % 3) as f64);
                if x + 1 < k {
                    r.push(idx(x + 1, y));
                    c.push(p);
                    v.push(-1.5);
                    r.push(p);
                    c.push(idx(x + 1, y));
                    v.push(-0.5);
                }
                if y + 1 < k {
                    r.push(idx(x, y + 1));
                    c.push(p);
                    v.push(-1.25);
                    r.push(p);
                    c.push(idx(x, y + 1));
                    v.push(-0.75);
                }
            }
        }
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let at = GeneralCsc::from_triplets(n, &c, &r, &v).unwrap();
        let s = lu(&a);
        let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
        let x = solve_lu_transpose(s.factors(), &b).unwrap();
        let mut atx = vec![0.0; n];
        at.matvec(&x, &mut atx);
        let res = b
            .iter()
            .zip(&atx)
            .map(|(bi, ai)| (bi - ai).abs())
            .fold(0.0, f64::max);
        assert!(res < 1e-10, "transpose residual {res}");
    }

    #[test]
    fn condest_identity_is_one() {
        let n = 5;
        let idx: Vec<usize> = (0..n).collect();
        let ones = vec![1.0; n];
        let a = GeneralCsc::from_triplets(n, &idx, &idx, &ones).unwrap();
        let s = lu(&a);
        let k1 = s.condest_1(&a).unwrap();
        assert!((k1 - 1.0).abs() < 1e-12, "identity κ₁ = {k1}");
    }

    #[test]
    fn condest_diagonal_exact_within_2x() {
        let a = GeneralCsc::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1e3, 1e6]).unwrap();
        let s = lu(&a);
        let k1 = s.condest_1(&a).unwrap();
        // κ₁(diag(1,1e3,1e6)) = 1e6 exactly; the estimator is a lower bound.
        assert!(k1 <= 1e6 * (1.0 + 1e-9), "over-estimate: {k1}");
        assert!(k1 >= 0.5e6, "lower bound too loose: {k1}");
    }

    /// Feral's hand oracle: `B = [[1,2],[3,4]]`, `‖B‖₁ = 6`,
    /// `B⁻¹ = [[-2,1],[1.5,-0.5]]`, `‖B⁻¹‖₁ = 3.5`, so `κ₁ = 21`; the
    /// estimate must land in `[10.5, 21·(1+1e-6)]`.
    #[test]
    fn condest_2x2_hand_oracle() {
        let a = GeneralCsc::from_triplets(2, &[0, 1, 0, 1], &[0, 0, 1, 1], &[1.0, 3.0, 2.0, 4.0])
            .unwrap();
        let s = lu(&a);
        let k1 = s.condest_1(&a).unwrap();
        assert!((10.5..=21.0 * (1.0 + 1e-6)).contains(&k1), "κ₁ = {k1}");
    }

    #[test]
    fn condest_complex_diagonal() {
        type C = Complex<f64>;
        let vals = vec![C::new(1.0, 0.0), C::new(0.0, 10.0), C::new(0.1, 0.0)];
        let a = GeneralCsc::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &vals).unwrap();
        let s = LuSolver::<C>::factor(&a, &SolverSettings::default()).unwrap();
        let k1 = s.condest_1(&a).unwrap();
        // κ₁ = ‖A‖₁·‖A⁻¹‖₁ = 10 · 10 = 100 exactly (diagonal).
        assert!(k1 <= 100.0 * (1.0 + 1e-9), "over-estimate: {k1}");
        assert!(k1 >= 50.0, "lower bound too loose: {k1}");
    }

    /// Growth on a diagonally dominant system stays modest; the getter must
    /// also reject a dimension mismatch.
    #[test]
    fn element_growth_sane() {
        let a = GeneralCsc::from_triplets(
            3,
            &[0, 1, 2, 1, 0],
            &[0, 1, 2, 0, 1],
            &[4.0, 5.0, 6.0, -1.0, -2.0],
        )
        .unwrap();
        let s = lu(&a);
        let g = s.element_growth(&a).unwrap();
        assert!(g > 0.0 && g < 100.0, "growth {g}");
        let wrong = GeneralCsc::from_triplets(2, &[0, 1], &[0, 1], &[1.0, 1.0]).unwrap();
        assert!(s.element_growth(&wrong).is_err());
        assert!(s.condest_1(&wrong).is_err());
    }
}
