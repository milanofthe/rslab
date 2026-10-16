//! Small dense complex linear algebra with checked inputs: the projected
//! matrices of Krylov methods (the Ritz values and vectors of a
//! [`BlockArnoldi`](crate::BlockArnoldi), a reduced model's resolvent). Dense
//! `O(d^3)` routines on row-major `Complex<f64>` matrices of up to a few hundred
//! rows, independent of the sparse solvers.

use crate::error::RslabError;
use crate::numeric::dense_eig::{lu, normalize, qr_budget, schur_form, triangular_eigenvector};
use num_complex::Complex;

type C = Complex<f64>;

fn check(len: usize, expected: usize) -> Result<(), RslabError> {
    if len == expected {
        Ok(())
    } else {
        Err(RslabError::DimensionMismatch { expected, got: len })
    }
}

/// `X = A^-1 B` by partial-pivot LU: `a` is `d x d` row-major, `b` is
/// `d x nrhs` row-major, so is the result. `NumericallyRankDeficient` where a
/// pivot falls below `DENSE_SINGULAR_TOL` times `||A||inf` (at least 1).
///
/// ```
/// use num_complex::Complex;
/// let a = [Complex::new(2.0, 0.0), Complex::new(0.0, 0.0),
///          Complex::new(0.0, 0.0), Complex::new(4.0, 0.0)];
/// let x = rslab::dense::lu_solve(&a, 2, &[Complex::new(2.0, 0.0), Complex::new(2.0, 0.0)], 1)
///     .unwrap();
/// assert_eq!(x, vec![Complex::new(1.0, 0.0), Complex::new(0.5, 0.0)]);
/// ```
pub fn lu_solve(a: &[C], d: usize, b: &[C], nrhs: usize) -> Result<Vec<C>, RslabError> {
    check(a.len(), d * d)?;
    check(b.len(), d * nrhs)?;
    lu(a, d, b, nrhs).ok_or(RslabError::NumericallyRankDeficient)
}

/// The complex Schur form `A = Q T Q^H` of a general dense matrix: `T` upper
/// triangular with the eigenvalues on its diagonal, `Q` unitary. Householder
/// Hessenberg reduction and the shifted QR iteration (Wilkinson shifts, LAPACK
/// `zlahqr`'s exceptional shifts, deflation at the bottom) with the transform
/// accumulated.
#[derive(Debug, Clone)]
pub struct Schur {
    d: usize,
    t: Vec<C>,
    q: Vec<C>,
}

impl Schur {
    /// The Schur form of `a` (`d x d` row-major). `NotConverged` where the QR
    /// iteration does not converge within `QR_ITER_PER_DIM` steps per dimension.
    pub fn new(a: &[C], d: usize) -> Result<Schur, RslabError> {
        check(a.len(), d * d)?;
        Self::with_budget(a, d, qr_budget(d))
    }

    fn with_budget(a: &[C], d: usize, budget: usize) -> Result<Schur, RslabError> {
        let (t, q) = schur_form(a, d, budget)
            .map_err(|_| RslabError::NotConverged { iterations: budget })?;
        Ok(Schur { d, t, q })
    }

    /// The dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// `T`, `d x d` row-major, zero below the diagonal.
    pub fn t(&self) -> &[C] {
        &self.t
    }

    /// `Q`, `d x d` row-major, unitary.
    pub fn q(&self) -> &[C] {
        &self.q
    }

    /// The eigenvalues, the diagonal of `T`.
    pub fn eigenvalues(&self) -> Vec<C> {
        (0..self.d).map(|i| self.t[i * self.d + i]).collect()
    }

    /// The eigenvalue `T_kk`'s right eigenvector `x` (`A x = lambda x`, unit
    /// norm) and left eigenvector `l` as a row (`l^T A = lambda l^T`), scaled to
    /// `l^T x = 1`: `x l^T` is the eigenvalue's spectral projector, `|x| |l|` its
    /// condition number. Back and forward substitution on `T` (no inverse of the
    /// eigenvector matrix, which is ill-conditioned where eigenvalues cluster);
    /// a repeated eigenvalue's denominators are floored at its own precision, so a
    /// defective one gives a large condition number, not a division by zero.
    /// `None` where the substitution overflows even with rescaling, or `k` is not
    /// below the dimension.
    pub fn eigenvectors(&self, k: usize) -> Option<(Vec<C>, Vec<C>)> {
        let d = self.d;
        if k >= d {
            return None;
        }
        let z = triangular_eigenvector(&self.t, d, k, false)?;
        let u = triangular_eigenvector(&self.t, d, k, true)?;
        // x = Q z, l^T = u^T Q^H
        let mut x: Vec<C> = (0..d)
            .map(|i| (0..=k).map(|j| self.q[i * d + j] * z[j]).sum())
            .collect();
        let mut l: Vec<C> = (0..d)
            .map(|i| (k..d).map(|j| u[j] * self.q[i * d + j].conj()).sum())
            .collect();
        normalize(&mut x);
        let lx: C = l.iter().zip(&x).map(|(a, b)| a * b).sum();
        if lx.norm() == 0.0 || !(lx.re.is_finite() && lx.im.is_finite()) {
            return None;
        }
        l.iter_mut().for_each(|v| *v /= lx);
        Some((x, l))
    }

    /// `X = (alpha I + beta A)^-1 B` (`b` `d x nrhs` row-major, so is the
    /// result) through the Schur form: `Q (alpha I + beta T)^-1 Q^H B`, one
    /// triangular solve, `O(d^2)` per column. `NumericallyRankDeficient` where
    /// `alpha + beta lambda_k` falls below `DENSE_SINGULAR_TOL` times
    /// `|alpha| + |beta| max |T_ij|`.
    pub fn solve_shifted(
        &self,
        alpha: C,
        beta: C,
        b: &[C],
        nrhs: usize,
    ) -> Result<Vec<C>, RslabError> {
        let d = self.d;
        check(b.len(), d * nrhs)?;
        let zero = C::new(0.0, 0.0);
        let tmax = self.t.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
        let thresh = crate::constants::DENSE_SINGULAR_TOL * (alpha.norm() + beta.norm() * tmax);
        // Y = Q^H B
        let mut y = vec![zero; d * nrhs];
        for i in 0..d {
            for c in 0..nrhs {
                y[i * nrhs + c] = (0..d)
                    .map(|r| self.q[r * d + i].conj() * b[r * nrhs + c])
                    .sum();
            }
        }
        for c in 0..nrhs {
            for i in (0..d).rev() {
                let mut s = y[i * nrhs + c];
                for j in (i + 1)..d {
                    s -= beta * self.t[i * d + j] * y[j * nrhs + c];
                }
                let den = alpha + beta * self.t[i * d + i];
                if den.norm().is_nan() || den.norm() <= thresh {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                y[i * nrhs + c] = s / den;
            }
        }
        Ok((0..d * nrhs)
            .map(|q| {
                let (i, c) = (q / nrhs, q % nrhs);
                (0..d).map(|j| self.q[i * d + j] * y[j * nrhs + c]).sum()
            })
            .collect())
    }
}

/// The eigenvalues of a small dense matrix with their right eigenvectors
/// ([`eigen`]).
#[derive(Debug, Clone)]
pub struct Eigen {
    /// The eigenvalues, in the order of the Schur form's diagonal.
    pub values: Vec<C>,
    /// The right eigenvectors, `d x d` row-major: column `k` belongs to
    /// `values[k]`, unit Euclidean norm.
    pub vectors: Vec<C>,
}

/// The eigenvalues and right eigenvectors of a general dense complex matrix `a`
/// (`d x d` row-major), `A x_k = lambda_k x_k`, from its [`Schur`] form
/// ([`Schur::eigenvectors`]). `NotConverged` as [`Schur::new`];
/// `NumericallyRankDeficient` where an eigenvector overflows.
pub fn eigen(a: &[C], d: usize) -> Result<Eigen, RslabError> {
    let s = Schur::new(a, d)?;
    let mut vectors = vec![C::new(0.0, 0.0); d * d];
    for k in 0..d {
        let (x, _) = s
            .eigenvectors(k)
            .ok_or(RslabError::NumericallyRankDeficient)?;
        for i in 0..d {
            vectors[i * d + k] = x[i];
        }
    }
    Ok(Eigen {
        values: s.eigenvalues(),
        vectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::dense_eig::eigenvalues;

    fn random(d: usize, seed: u64) -> Vec<C> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..d * d).map(|_| C::new(next(), next())).collect()
    }

    fn matvec(a: &[C], d: usize, x: &[C]) -> Vec<C> {
        (0..d)
            .map(|i| (0..d).map(|j| a[i * d + j] * x[j]).sum())
            .collect()
    }

    fn norm(v: &[C]) -> f64 {
        v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
    }

    /// `U J U^H` for the unitary `U` of a random matrix's Schur form.
    fn similar(j: &[C], d: usize, seed: u64) -> Vec<C> {
        let u = Schur::new(&random(d, seed), d).unwrap().q().to_vec();
        (0..d * d)
            .map(|q| {
                let (r, c) = (q / d, q % d);
                (0..d)
                    .map(|k| {
                        (0..d)
                            .map(|l| u[r * d + k] * j[k * d + l] * u[c * d + l].conj())
                            .sum::<C>()
                    })
                    .sum()
            })
            .collect()
    }

    fn sorted(mut v: Vec<C>) -> Vec<C> {
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
        v
    }

    /// The Schur form reproduces the matrix (`Q T Q^H = A`), `Q` is unitary and `T`
    /// upper triangular with the eigenvalues of the best-effort routine.
    #[test]
    fn schur_reproduces_the_matrix() {
        for (d, seed) in [(1, 1), (2, 2), (7, 3), (40, 4), (120, 5)] {
            let a = random(d, seed);
            let s = Schur::new(&a, d).expect("converges");
            let (t, q) = (s.t(), s.q());
            let anorm = a.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
            for i in 0..d {
                for j in 0..i {
                    assert_eq!(t[i * d + j], C::new(0.0, 0.0), "T below the diagonal");
                }
                for j in 0..d {
                    let qtq: C = (0..d)
                        .map(|k| {
                            (0..d)
                                .map(|l| q[i * d + k] * t[k * d + l] * q[j * d + l].conj())
                                .sum::<C>()
                        })
                        .sum();
                    assert!((qtq - a[i * d + j]).norm() < 1e-12 * anorm * d as f64);
                    let qhq: C = (0..d).map(|k| q[k * d + i].conj() * q[k * d + j]).sum();
                    let id = if i == j { 1.0 } else { 0.0 };
                    assert!((qhq - id).norm() < 1e-12 * d as f64);
                }
            }
            for (g, w) in sorted(s.eigenvalues())
                .iter()
                .zip(sorted(eigenvalues(&a, d)))
            {
                assert!((g - w).norm() < 1e-9 * anorm.max(1.0));
            }
        }
    }

    /// Every right and left eigenpair satisfies its eigenproblem to the working
    /// precision, `l^T x = 1`, and a matrix with a known spectrum gives it.
    #[test]
    fn eigenpairs_satisfy_the_eigenproblem() {
        for (d, seed) in [(3, 11), (30, 12), (90, 13)] {
            let a = random(d, seed);
            let s = Schur::new(&a, d).unwrap();
            let anorm = a.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
            for (k, lam) in s.eigenvalues().into_iter().enumerate() {
                let (x, l) = s.eigenvectors(k).expect("finite");
                assert!((norm(&x) - 1.0).abs() < 1e-12);
                let r: Vec<C> = matvec(&a, d, &x)
                    .iter()
                    .zip(&x)
                    .map(|(u, v)| u - lam * v)
                    .collect();
                assert!(
                    norm(&r) < 1e-11 * anorm * d as f64,
                    "right {k}: {:e}",
                    norm(&r)
                );
                // l^T A = lam l^T
                let la: Vec<C> = (0..d)
                    .map(|j| (0..d).map(|i| l[i] * a[i * d + j]).sum::<C>() - lam * l[j])
                    .collect();
                assert!(norm(&la) < 1e-10 * anorm * d as f64 * norm(&l), "left {k}");
                let lx: C = l.iter().zip(&x).map(|(u, v)| u * v).sum();
                assert!((lx - 1.0).norm() < 1e-12);
            }
            let e = eigen(&a, d).unwrap();
            assert_eq!(e.values, s.eigenvalues());
        }
        // a known spectrum: W diag W^-1
        let d = 12;
        let w = random(d, 21);
        let lam: Vec<C> = (0..d).map(|k| C::new(k as f64, -0.1 * k as f64)).collect();
        let mut id = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            id[i * d + i] = C::new(1.0, 0.0);
        }
        let winv = lu_solve(&w, d, &id, d).unwrap();
        let a: Vec<C> = (0..d * d)
            .map(|q| {
                let (i, j) = (q / d, q % d);
                (0..d)
                    .map(|k| w[i * d + k] * lam[k] * winv[k * d + j])
                    .sum()
            })
            .collect();
        for (g, w) in sorted(eigen(&a, d).unwrap().values).iter().zip(sorted(lam)) {
            assert!((g - w).norm() < 1e-9, "{g} against {w}");
        }
    }

    /// A defective matrix (a Jordan block under a unitary similarity): the
    /// eigenvalue to the accuracy a Jordan block allows (`eps^(1/d)`), every
    /// vector finite and along the one eigenvector `U e_0`; a Jordan block of 30
    /// (whose back substitution would overflow without rescaling) stays finite.
    #[test]
    fn a_defective_matrix_gives_finite_vectors() {
        let lam = C::new(2.0, 1.0);
        for d in [5usize, 30] {
            let mut j = vec![C::new(0.0, 0.0); d * d];
            for i in 0..d {
                j[i * d + i] = lam;
                if i + 1 < d {
                    j[i * d + i + 1] = C::new(1.0, 0.0);
                }
            }
            let a = if d == 5 {
                similar(&j, d, 31)
            } else {
                j.clone()
            };
            let s = Schur::new(&a, d).unwrap();
            let e = eigen(&a, d).unwrap();
            let v0 = if d == 5 {
                Schur::new(&random(d, 31), d)
                    .unwrap()
                    .q()
                    .iter()
                    .step_by(d)
                    .copied()
                    .collect::<Vec<_>>()
            } else {
                (0..d)
                    .map(|i| C::new(if i == 0 { 1.0 } else { 0.0 }, 0.0))
                    .collect()
            };
            for k in 0..d {
                assert!(
                    (s.eigenvalues()[k] - lam).norm() < f64::EPSILON.powf(1.0 / d as f64) * 10.0
                );
                let x: Vec<C> = (0..d).map(|i| e.vectors[i * d + k]).collect();
                assert!(x.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
                let along: C = v0.iter().zip(&x).map(|(u, v)| u.conj() * v).sum();
                assert!(
                    along.norm() > 1.0 - 1e-3,
                    "d {d} vector {k}: {}",
                    along.norm()
                );
            }
        }
    }

    /// Clustered eigenvalues (three within 1e-10) under a similarity: every pair
    /// is still an eigenpair to the working precision.
    #[test]
    fn clustered_eigenvalues_give_eigenpairs() {
        let d = 6;
        let vals = [1.0, 1.0 + 1e-10, 1.0 + 2e-10, 3.0, 4.0, -2.0];
        let mut j = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            j[i * d + i] = C::new(vals[i], 0.0);
        }
        let a = similar(&j, d, 41);
        let s = Schur::new(&a, d).unwrap();
        for (k, lam) in s.eigenvalues().into_iter().enumerate() {
            let (x, _) = s.eigenvectors(k).unwrap();
            let r: Vec<C> = matvec(&a, d, &x)
                .iter()
                .zip(&x)
                .map(|(u, v)| u - lam * v)
                .collect();
            assert!(norm(&r) < 1e-12, "{k}: {:e}", norm(&r));
        }
    }

    /// The cyclic permutation (eigenvalues the roots of unity) does not move
    /// under QR with Wilkinson shifts alone; the exceptional shifts get it
    /// moving. A budget spent before convergence is `NotConverged`, and wrong
    /// lengths are `DimensionMismatch`.
    #[test]
    fn exceptional_shifts_and_errors() {
        let d = 8;
        let mut a = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            a[((i + 1) % d) * d + i] = C::new(1.0, 0.0);
        }
        let s = Schur::new(&a, d).expect("the exceptional shifts converge");
        for lam in s.eigenvalues() {
            assert!((lam.norm() - 1.0).abs() < 1e-10, "{lam}");
        }
        assert!(matches!(
            Schur::with_budget(&random(6, 3), 6, 0),
            Err(RslabError::NotConverged { iterations: 0 })
        ));
        assert!(matches!(
            Schur::new(&[C::new(1.0, 0.0); 3], 2),
            Err(RslabError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));
        assert!(matches!(
            lu_solve(&[C::new(0.0, 0.0); 4], 2, &[C::new(1.0, 0.0); 2], 1),
            Err(RslabError::NumericallyRankDeficient)
        ));
    }

    /// `(alpha I + beta A)^-1 B` through the Schur form agrees with the LU solve.
    #[test]
    fn the_shifted_solve_agrees_with_lu() {
        let d = 25;
        let a = random(d, 51);
        let b = random(d, 52)[..d * 3].to_vec();
        let (alpha, beta) = (C::new(1.0, 0.3), C::new(-0.4, 0.2));
        let s = Schur::new(&a, d).unwrap();
        let x = s.solve_shifted(alpha, beta, &b, 3).unwrap();
        let m: Vec<C> = (0..d * d)
            .map(|q| {
                let id = if q / d == q % d {
                    alpha
                } else {
                    C::new(0.0, 0.0)
                };
                id + beta * a[q]
            })
            .collect();
        let want = lu_solve(&m, d, &b, 3).unwrap();
        for (g, w) in x.iter().zip(&want) {
            assert!((g - w).norm() < 1e-11, "{g} against {w}");
        }
    }
}
