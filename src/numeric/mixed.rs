//! Mixed-precision direct solve (issue #18): factor in **single** precision,
//! refine to **double** against the original matrix.
//!
//! A single-precision factor halves memory and bandwidth - the factorization
//! runs faster and the (memory-bound) triangular solves ~2x - and
//! multi-precision iterative-refinement theory (Carson & Higham) says when
//! the double-precision solution is certifiably recovered: plain IR contracts
//! while `kappa(A) · eps_f32 << 1`; preconditioned GMRES (the low-precision
//! factor as the preconditioner, GMRES-IR) extends the reach by several
//! orders of magnitude. This module implements that ladder explicitly:
//!
//! 1. factor `cast_lo(A)` in the low precision (heuristic settings pick);
//! 2. plain iterative refinement: residuals in the HIGH precision against
//!    the original `A`, corrections through the low-precision factor;
//! 3. on stagnation, escalate to GMRES-IR (same factor, same operator,
//!    warm-started at the best iterate);
//! 4. report what happened ([`MixedInfo`]): achieved normwise backward
//!    error, iteration counts, and whether the certificate (backward error
//!    at the double-precision roundoff level) holds - the caller decides
//!    whether to fall back to a full double-precision factor.
//!
//! The pre-existing [`LowPrecisionPreconditioner`](crate::numeric::iterative)
//! types are the "bring your own Krylov" building blocks; this module is the
//! self-contained factor-and-solve driver with the certificate.
//!
//! Deterministic and bit-identical across thread counts (both precision
//! domains inherit the house guarantee).

use std::marker::PhantomData;

use num_complex::Complex;

use crate::error::RslabError;
use crate::numeric::iterative::{gmres, LinearOperator, Preconditioner};
use crate::numeric::multifrontal_ldlt::SolverSettings;
use crate::numeric::multifrontal_lu::LuSolver;
use crate::numeric::sparse_solver::LdltSolver;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

/// A high-precision scalar whose [`Scalar::Lo`] partner is itself a full
/// [`Scalar`] (factorizable): `f64 -> f32`, `c64 -> c32`. The cast pair
/// lives on [`Scalar`] (`demote`/`promote`); this marker adds the
/// factorability bound and the certificate roundoff level.
pub trait MixedScalar: Scalar<Lo: Scalar> {
    /// Unit roundoff of the HIGH precision (the certificate level).
    const EPS_HI: f64;
}

impl MixedScalar for f64 {
    const EPS_HI: f64 = f64::EPSILON;
}

impl MixedScalar for Complex<f64> {
    const EPS_HI: f64 = f64::EPSILON;
}

/// Cast a symmetric CSC matrix to the low precision (same pattern).
fn cast_csc_lo<T: MixedScalar>(a: &CscMatrix<T>) -> CscMatrix<T::Lo> {
    CscMatrix {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a.values.iter().map(|&v| v.demote()).collect(),
    }
}

/// Cast a general CSC matrix to the low precision (same pattern).
fn cast_general_lo<T: MixedScalar>(a: &GeneralCsc<T>) -> GeneralCsc<T::Lo> {
    GeneralCsc {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a.values.iter().map(|&v| v.demote()).collect(),
    }
}

/// Outcome of a mixed-precision solve: what the refinement achieved and how.
#[derive(Debug, Clone, Copy)]
pub struct MixedInfo {
    /// Plain iterative-refinement steps taken.
    pub ir_iters: usize,
    /// GMRES iterations of the escalation stage (0 = not needed).
    pub gmres_iters: usize,
    /// Final normwise backward error `‖r‖∞ / (‖A‖∞·‖x‖∞ + ‖b‖∞)`.
    pub backward_error: f64,
    /// The certificate: backward error at the double-precision roundoff
    /// level (`<= CERT_FACTOR · eps_f64`). When `false` the caller should
    /// fall back to a full double-precision factorization.
    pub certified: bool,
}

/// Backward-error certificate threshold, as a multiple of the high-precision
/// unit roundoff. A flat modest multiple keeps the promise honest without
/// failing benign systems on the constant.
const CERT_FACTOR: f64 = 64.0;
/// Plain-IR budget before escalating to GMRES-IR.
const IR_MAX: usize = 8;
/// Contraction guard: an IR step must shrink the residual by at least this
/// factor, else it has stagnated (kappa too large for plain IR).
const IR_CONTRACT: f64 = 0.5;
/// GMRES-IR budget (restart length and total iterations).
const GMRES_RESTART: usize = 30;
const GMRES_MAX: usize = 120;

/// Low-precision-solve preconditioner adapter: `z = cast_hi(F⁻¹ cast_lo(r))`.
enum LoFactor<'a, T: MixedScalar> {
    Ldlt(&'a LdltSolver<T::Lo>),
    Lu(&'a LuSolver<T::Lo>),
}

struct LoPrecond<'a, T: MixedScalar> {
    f: LoFactor<'a, T>,
    _t: PhantomData<T>,
}

impl<T: MixedScalar> LoPrecond<'_, T> {
    fn solve_up(&self, r: &[T]) -> Result<Vec<T>, RslabError> {
        let rl: Vec<T::Lo> = r.iter().map(|&v| v.demote()).collect();
        let zl = match &self.f {
            LoFactor::Ldlt(s) => s.solve(&rl)?,
            LoFactor::Lu(s) => s.solve(&rl)?,
        };
        Ok(zl.into_iter().map(T::promote).collect())
    }
}

impl<T: MixedScalar> Preconditioner<T> for LoPrecond<'_, T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let zz = self.solve_up(r)?;
        z.copy_from_slice(&zz);
        Ok(())
    }
}

/// The refinement ladder shared by both mixed solvers: plain IR (cheap per
/// step) with a stagnation guard, then GMRES-IR warm-started at the best
/// iterate. `x0` is the cast-up low-precision solve of `b`.
fn refine_ladder<T: MixedScalar, A: LinearOperator<T> + ?Sized>(
    op: &A,
    pre: &LoPrecond<'_, T>,
    b: &[T],
    anorm_inf: f64,
    target: f64,
    mut x: Vec<T>,
) -> Result<(Vec<T>, MixedInfo), RslabError> {
    let n = op.n();
    let bnorm = b.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
    let mut ax = vec![T::zero(); n];
    let berr = |x: &[T], r: &[T]| {
        let xn = x.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        let rn = r.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        rn / (anorm_inf * xn + bnorm).max(f64::MIN_POSITIVE)
    };

    let mut ir_iters = 0usize;
    let mut best_x = x.clone();
    let mut best_be = f64::INFINITY;
    let mut prev_rn = f64::INFINITY;
    loop {
        op.apply(&x, &mut ax);
        let r: Vec<T> = b.iter().zip(&ax).map(|(&bi, &ai)| bi - ai).collect();
        let be = berr(&x, &r);
        if be < best_be {
            best_be = be;
            best_x.clone_from(&x);
        }
        if be <= target {
            return Ok((
                best_x,
                MixedInfo {
                    ir_iters,
                    gmres_iters: 0,
                    backward_error: best_be,
                    certified: true,
                },
            ));
        }
        let rn = r.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        if ir_iters >= IR_MAX || (ir_iters > 0 && rn > prev_rn * IR_CONTRACT) {
            break; // budget spent or stagnated -> escalate
        }
        prev_rn = rn;
        let dx = pre.solve_up(&r)?;
        for (xi, d) in x.iter_mut().zip(dx) {
            *xi = *xi + d;
        }
        ir_iters += 1;
    }

    // GMRES-IR: low-precision factor as preconditioner, high-precision
    // operator, warm start at the best iterate so far. `gmres` measures
    // convergence relative to ‖b‖, which upper-bounds the backward error.
    let res = gmres(
        op,
        b,
        pre,
        target,
        GMRES_MAX,
        GMRES_RESTART,
        Some(best_x.as_slice()),
    )?;
    op.apply(&res.x, &mut ax);
    let r: Vec<T> = b.iter().zip(&ax).map(|(&bi, &ai)| bi - ai).collect();
    let be2 = berr(&res.x, &r);
    let (xf, bef) = if be2 < best_be {
        (res.x, be2)
    } else {
        (best_x, best_be)
    };
    Ok((
        xf,
        MixedInfo {
            ir_iters,
            gmres_iters: res.iters,
            backward_error: bef,
            certified: bef <= target,
        },
    ))
}

/// Mixed-precision symmetric solver: single-precision LDLᵀ factor of
/// `cast(A)`, solves refined to double precision against the original `A`.
pub struct MixedLdltSolver<T: MixedScalar> {
    lo: LdltSolver<T::Lo>,
    anorm_inf: f64,
    n: usize,
}

impl<T: MixedScalar> MixedLdltSolver<T> {
    /// Factor `cast_lo(A)` with the heuristic settings pick
    /// ([`LdltSolver::tuned`] on the low-precision matrix).
    pub fn factor(a: &CscMatrix<T>) -> Result<Self, RslabError> {
        let alo = cast_csc_lo(a);
        let (sym, s) = LdltSolver::<T::Lo>::tuned(&alo)?;
        Ok(Self {
            lo: sym.factor(&alo, &s)?,
            anorm_inf: sym_norm_inf(a),
            n: a.n,
        })
    }

    /// Factor with explicit settings (pivot policy, threads, ...).
    pub fn factor_with(a: &CscMatrix<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        let alo = cast_csc_lo(a);
        Ok(Self {
            lo: LdltSolver::<T::Lo>::factor_with(&alo, opts)?,
            anorm_inf: sym_norm_inf(a),
            n: a.n,
        })
    }

    /// Solve `A x = b` to double precision via the IR → GMRES-IR ladder
    /// against the original `a` (which must be the matrix this was factored
    /// from). Check [`MixedInfo::certified`]; when `false`, fall back to a
    /// double-precision factorization.
    pub fn solve(&self, a: &CscMatrix<T>, b: &[T]) -> Result<(Vec<T>, MixedInfo), RslabError> {
        self.solve_to(a, b, CERT_FACTOR * T::EPS_HI)
    }

    /// [`solve`](Self::solve) with an explicit backward-error target - e.g.
    /// `1e-8` when the solution feeds an outer Krylov loop anyway
    /// (preconditioner-grade), so the ladder stops early instead of
    /// spending its full GMRES budget chasing the eps-level certificate.
    /// `certified` in the returned info refers to the given target.
    pub fn solve_to(
        &self,
        a: &CscMatrix<T>,
        b: &[T],
        target_backward_error: f64,
    ) -> Result<(Vec<T>, MixedInfo), RslabError> {
        if a.n != self.n || b.len() != self.n {
            return Err(RslabError::DimensionMismatch {
                expected: self.n,
                got: if a.n != self.n { a.n } else { b.len() },
            });
        }
        let pre = LoPrecond::<T> {
            f: LoFactor::Ldlt(&self.lo),
            _t: PhantomData,
        };
        let x0 = pre.solve_up(b)?;
        refine_ladder(a, &pre, b, self.anorm_inf, target_backward_error, x0)
    }

    /// Stored factor nonzeros (each a LOW-precision entry - half the bytes).
    pub fn factor_nnz(&self) -> usize {
        self.lo.factor_nnz()
    }
}

/// Mixed-precision unsymmetric solver: single-precision LU factor of
/// `cast(A)`, solves refined to double precision against the original `A`.
pub struct MixedLuSolver<T: MixedScalar> {
    lo: LuSolver<T::Lo>,
    anorm_inf: f64,
    n: usize,
}

impl<T: MixedScalar> MixedLuSolver<T> {
    /// Factor `cast_lo(A)` with the heuristic settings pick
    /// ([`LuSolver::tuned`] on the low-precision matrix).
    pub fn factor(a: &GeneralCsc<T>) -> Result<Self, RslabError> {
        let alo = cast_general_lo(a);
        let (sym, s) = LuSolver::<T::Lo>::tuned(&alo)?;
        Ok(Self {
            lo: sym.factor(&alo, &s)?,
            anorm_inf: gen_norm_inf(a),
            n: a.n,
        })
    }

    /// Factor with explicit settings (pivot policy, threads, ...).
    pub fn factor_with(a: &GeneralCsc<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        let alo = cast_general_lo(a);
        Ok(Self {
            lo: LuSolver::<T::Lo>::factor(&alo, opts)?,
            anorm_inf: gen_norm_inf(a),
            n: a.n,
        })
    }

    /// Solve `A x = b` to double precision (see [`MixedLdltSolver::solve`]).
    pub fn solve(&self, a: &GeneralCsc<T>, b: &[T]) -> Result<(Vec<T>, MixedInfo), RslabError> {
        self.solve_to(a, b, CERT_FACTOR * T::EPS_HI)
    }

    /// [`solve`](Self::solve) with an explicit backward-error target (see
    /// [`MixedLdltSolver::solve_to`]).
    pub fn solve_to(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        target_backward_error: f64,
    ) -> Result<(Vec<T>, MixedInfo), RslabError> {
        if a.n != self.n || b.len() != self.n {
            return Err(RslabError::DimensionMismatch {
                expected: self.n,
                got: if a.n != self.n { a.n } else { b.len() },
            });
        }
        let pre = LoPrecond::<T> {
            f: LoFactor::Lu(&self.lo),
            _t: PhantomData,
        };
        let x0 = pre.solve_up(b)?;
        refine_ladder(a, &pre, b, self.anorm_inf, target_backward_error, x0)
    }

    /// Stored factor nonzeros (each a LOW-precision entry - half the bytes).
    pub fn factor_nnz(&self) -> usize {
        self.lo.factor_nnz()
    }
}

/// `‖A‖∞` of a symmetric lower-triangle CSC (both triangles counted).
fn sym_norm_inf<T: Scalar>(a: &CscMatrix<T>) -> f64 {
    let mut row_sum = vec![0.0f64; a.n];
    for c in 0..a.n {
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[k];
            let m = a.values[k].magnitude();
            row_sum[r] += m;
            if r != c {
                row_sum[c] += m;
            }
        }
    }
    row_sum.into_iter().fold(0.0, f64::max)
}

/// `‖A‖∞` of a general CSC.
fn gen_norm_inf<T: Scalar>(a: &GeneralCsc<T>) -> f64 {
    let mut row_sum = vec![0.0f64; a.n];
    for c in 0..a.n {
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            row_sum[a.row_idx[k]] += a.values[k].magnitude();
        }
    }
    row_sum.into_iter().fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2D 5-point Laplacian (SPD, lower triangle), kappa ~ O(k²) - plain IR
    /// territory.
    fn lap2d(k: usize) -> CscMatrix<f64> {
        let n = k * k;
        let idx = |x: usize, y: usize| y * k + x;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y);
                r.push(p);
                c.push(p);
                v.push(4.0);
                if x + 1 < k {
                    r.push(idx(x + 1, y));
                    c.push(p);
                    v.push(-1.0);
                }
                if y + 1 < k {
                    r.push(idx(x, y + 1));
                    c.push(p);
                    v.push(-1.0);
                }
            }
        }
        CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
    }

    #[test]
    fn mixed_ldlt_certifies_on_laplacian() {
        let a = lap2d(40);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| ((i % 13) as f64) - 6.0).collect();
        let m = MixedLdltSolver::<f64>::factor(&a).unwrap();
        let (x, info) = m.solve(&a, &b).unwrap();
        assert!(
            info.certified,
            "laplacian must certify (be={:.2e}, ir={}, gmres={})",
            info.backward_error, info.ir_iters, info.gmres_iters
        );
        // Cross-check against the double-precision direct solve.
        let xd = LdltSolver::<f64>::factor(&a).unwrap().solve(&b).unwrap();
        let dmax = x
            .iter()
            .zip(&xd)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let xmax = xd.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            dmax <= 1e-10 * xmax.max(1.0),
            "mixed and double solutions must agree (dmax={dmax:.2e})"
        );
    }

    #[test]
    fn mixed_lu_certifies_on_unsymmetric() {
        // Unsymmetric convection-ish variant of the Laplacian.
        let k = 30;
        let n = k * k;
        let idx = |x: usize, y: usize| y * k + x;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y);
                r.push(p);
                c.push(p);
                v.push(4.0);
                if x + 1 < k {
                    r.push(idx(x + 1, y));
                    c.push(p);
                    v.push(-1.3); // downwind
                    r.push(p);
                    c.push(idx(x + 1, y));
                    v.push(-0.7); // upwind
                }
                if y + 1 < k {
                    r.push(idx(x, y + 1));
                    c.push(p);
                    v.push(-1.2);
                    r.push(p);
                    c.push(idx(x, y + 1));
                    v.push(-0.8);
                }
            }
        }
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
        let m = MixedLuSolver::<f64>::factor(&a).unwrap();
        let (x, info) = m.solve(&a, &b).unwrap();
        assert!(
            info.certified,
            "convection system must certify (be={:.2e}, ir={}, gmres={})",
            info.backward_error, info.ir_iters, info.gmres_iters
        );
        let mut ax = vec![0.0; n];
        a.matvec(&x, &mut ax);
        let res = b
            .iter()
            .zip(&ax)
            .map(|(bi, ai)| (bi - ai).abs())
            .fold(0.0, f64::max);
        assert!(res <= 1e-11, "true residual small (res={res:.2e})");
    }

    /// The certificate must honestly report failure when the ladder cannot
    /// reach a backward-stable solution within budget. NOTE the semantics:
    /// the certificate is *backward* stability - the same guarantee class a
    /// double-precision direct factorization gives. Near-singular systems
    /// with huge solutions can be backward-stable at tiny normwise backward
    /// error while the forward error is large; that is a conditioning
    /// property, not a certificate failure (a first version of this test
    /// expected `!certified` on a kappa ~ 4e16 pair system and learned the
    /// ladder legitimately certifies it at be ~ 1e-31). To force an honest
    /// failure, cripple the low-precision factor itself: drop-tolerance
    /// 0.9 discards almost all fill, IR diverges, and the GMRES-IR budget
    /// (120 iterations) is far too small for an n=8100 Laplacian with a
    /// near-useless preconditioner.
    #[test]
    fn mixed_reports_uncertified_when_ladder_stalls() {
        let a = lap2d(90);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        let opts = SolverSettings::preconditioner(1e-12).with_drop_tol(0.9);
        let m = MixedLdltSolver::<f64>::factor_with(&a, &opts).unwrap();
        let (_, info) = m.solve(&a, &b).unwrap();
        assert!(
            !info.certified,
            "a crippled factor must NOT certify (be={:.2e}, ir={}, gmres={})",
            info.backward_error, info.ir_iters, info.gmres_iters
        );
        // Consistency: the flag must agree with the reported number.
        assert_eq!(
            info.certified,
            info.backward_error <= 64.0 * f64::EPSILON,
            "certificate flag must be consistent with the backward error"
        );
    }
}
