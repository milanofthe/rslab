//! Small **dense** complex linear algebra: the projected matrices of Krylov
//! methods (GCRO-DR's harmonic-Ritz extraction, the Ritz values of a caller's
//! [`BlockArnoldi`](crate::BlockArnoldi)). These routines operate on small
//! matrices (a restart length, a few hundred at most), never in the Krylov hot
//! loop, so they are written for clarity and robustness, not peak speed, and
//! everything runs in `Complex<f64>` regardless of the outer solve's scalar field.
//!
//! * [`lu_solve`]: a partial-pivot complex **LU solve** (`A X = B`).
//! * [`schur`] and [`eigen`]: the complex **Schur decomposition** `A = Q T Q^H`
//!   (Householder Hessenberg reduction, Wilkinson-shifted Givens QR with the
//!   transform accumulated) and from it the eigenvalues with their right
//!   eigenvectors (back substitution on `T`). Both report a QR that does not
//!   converge within its budget as `None`.
//! * `eigenvalues`: the eigenvalues alone, best effort, for GCRO-DR.
//!
//! **Why approximate is safe for GCRO-DR.** The harmonic-Ritz vectors only seed
//! the recycle subspace `U`, which *accelerates* convergence; the outer GMRES
//! true-residual test is authoritative, so a defective / inaccurate `U` can only
//! make the solve slower, never wrong. Accordingly `eigenvalues` and
//! `harmonic_ritz_smallest` are guarded (capped iterations, singular-pivot
//! fallbacks) and return "best effort" rather than erroring. A caller whose
//! result is the spectrum itself takes [`eigen`], which says when it failed.

use crate::constants::{DENSE_SINGULAR_TOL, INVERSE_ITERATION_SHIFT, QR_ITER_PER_DIM, TINY};
use num_complex::Complex;

type C = Complex<f64>;

/// Solve the dense complex system `A X = B` by partial-pivot LU. `a` is `dxd`
/// row-major (consumed as a working copy), `b` is `dxnrhs` row-major. Returns the
/// solution `X` (`dxnrhs` row-major), or `None` if `A` is numerically singular
/// (a pivot below `tol*||A||inf`), so the caller can skip the recycle update rather
/// than divide by ~0.
pub fn lu_solve(a: &[C], d: usize, b: &[C], nrhs: usize) -> Option<Vec<C>> {
    if d == 0 {
        return Some(Vec::new());
    }
    let mut lu = a.to_vec();
    let mut x = b.to_vec();
    // Scale for the singularity threshold: max row-sum norm of A.
    let mut anorm = 0.0f64;
    for i in 0..d {
        let mut s = 0.0;
        for j in 0..d {
            s += lu[i * d + j].norm();
        }
        anorm = anorm.max(s);
    }
    let thresh = DENSE_SINGULAR_TOL * anorm.max(1.0);
    for col in 0..d {
        // Partial pivot: largest magnitude in the active column.
        let mut piv = col;
        let mut best = lu[col * d + col].norm();
        for i in (col + 1)..d {
            let v = lu[i * d + col].norm();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best <= thresh {
            return None;
        }
        if piv != col {
            for j in 0..d {
                lu.swap(col * d + j, piv * d + j);
            }
            for j in 0..nrhs {
                x.swap(col * nrhs + j, piv * nrhs + j);
            }
        }
        let diag = lu[col * d + col];
        for i in (col + 1)..d {
            let f = lu[i * d + col] / diag;
            lu[i * d + col] = f;
            for j in (col + 1)..d {
                let v = lu[col * d + j];
                lu[i * d + j] -= f * v;
            }
            for j in 0..nrhs {
                let v = x[col * nrhs + j];
                x[i * nrhs + j] -= f * v;
            }
        }
    }
    // Back substitution.
    for j in 0..nrhs {
        for i in (0..d).rev() {
            let mut s = x[i * nrhs + j];
            for kk in (i + 1)..d {
                s -= lu[i * d + kk] * x[kk * nrhs + j];
            }
            x[i * nrhs + j] = s / lu[i * d + i];
        }
    }
    Some(x)
}

/// Reduce a general dense complex matrix `a` (`dxd`, row-major, modified in place)
/// to upper **Hessenberg** form by Householder similarity transforms. With `q`
/// (`dxd` row-major) the transform is accumulated into it from the right: a `q`
/// that starts as the identity ends as `Q` with `A = Q H Q^H`.
#[allow(clippy::needless_range_loop)]
fn to_hessenberg(a: &mut [C], d: usize, mut q: Option<&mut [C]>) {
    // A matrix up to 2x2 is Hessenberg already.
    if d <= 2 {
        return;
    }
    for col in 0..(d - 2) {
        // Householder vector zeroing a[col+2..d, col] against a[col+1, col].
        let mut alpha = 0.0f64;
        for i in (col + 1)..d {
            alpha += a[i * d + col].norm_sqr();
        }
        alpha = alpha.sqrt();
        if alpha == 0.0 {
            continue;
        }
        let x0 = a[(col + 1) * d + col];
        let phase = if x0.norm() == 0.0 {
            C::new(1.0, 0.0)
        } else {
            x0 / x0.norm()
        };
        let beta = -phase * alpha;
        // v = x - beta e_1, stored in a working vector indexed col+1..d.
        let mut v = vec![C::new(0.0, 0.0); d];
        for i in (col + 1)..d {
            v[i] = a[i * d + col];
        }
        v[col + 1] -= beta;
        let mut vnorm = 0.0f64;
        for i in (col + 1)..d {
            vnorm += v[i].norm_sqr();
        }
        if vnorm == 0.0 {
            continue;
        }
        // H = I - 2 v v^H / (v^H v). Apply on the left: A <- H A.
        for j in 0..d {
            let mut s = C::new(0.0, 0.0);
            for i in (col + 1)..d {
                s += v[i].conj() * a[i * d + j];
            }
            let f = (s * 2.0) / vnorm;
            for i in (col + 1)..d {
                a[i * d + j] -= f * v[i];
            }
        }
        // Apply on the right: A <- A H, and Q <- Q H.
        reflect_right(a, d, &v, col + 1, vnorm);
        if let Some(q) = q.as_deref_mut() {
            reflect_right(q, d, &v, col + 1, vnorm);
        }
    }
}

/// `M <- M (I - 2 v v^H / vnorm)` for the `dxd` row-major `M`, `v` nonzero from
/// index `from` on.
fn reflect_right(m: &mut [C], d: usize, v: &[C], from: usize, vnorm: f64) {
    for i in 0..d {
        let mut s = C::new(0.0, 0.0);
        for j in from..d {
            s += m[i * d + j] * v[j];
        }
        let f = (s * 2.0) / vnorm;
        for j in from..d {
            m[i * d + j] -= f * v[j].conj();
        }
    }
}

/// One shifted Givens QR step on the active leading `pxp` block of the upper
/// Hessenberg matrix `h` (`dxd` row-major): `H <- R*Q + muI` where `H - muI = Q*R`.
/// Without `q` the rotations are applied to rows/cols `0..p` only (eigenvalues
/// only). With `q` (`dxd` row-major) the left rotations also reach the columns
/// right of the active block, so the deflated part stays the Schur form, and
/// the transform is accumulated into `q` from the right.
fn qr_step(h: &mut [C], d: usize, p: usize, mu: C, mut q: Option<&mut [C]>) {
    let cols = if q.is_some() { d } else { p };
    // Shift.
    for i in 0..p {
        h[i * d + i] -= mu;
    }
    // Left rotations forming R; remember them for the right application.
    let mut cs = vec![C::new(0.0, 0.0); p];
    let mut sn = vec![C::new(0.0, 0.0); p];
    for i in 0..(p - 1) {
        let a = h[i * d + i];
        let b = h[(i + 1) * d + i];
        let r = (a.norm_sqr() + b.norm_sqr()).sqrt();
        if r == 0.0 {
            cs[i] = C::new(1.0, 0.0);
            sn[i] = C::new(0.0, 0.0);
            continue;
        }
        let c = a / r;
        let s = b / r;
        cs[i] = c;
        sn[i] = s;
        // Rows i, i+1 across columns i..cols: [[c_bar, s_bar],[-s, c]].
        for j in i..cols {
            let hij = h[i * d + j];
            let hi1j = h[(i + 1) * d + j];
            h[i * d + j] = c.conj() * hij + s.conj() * hi1j;
            h[(i + 1) * d + j] = -s * hij + c * hi1j;
        }
    }
    // Right rotations: H <- R*Q, Q = G_0^H G_1^H ... so apply the transpose from the
    // right to column pairs (i, i+1) across rows 0..=min(i+2, p)-1.
    for i in 0..(p - 1) {
        rotate_columns(h, d, i, (i + 2).min(p), cs[i], sn[i]);
        if let Some(q) = q.as_deref_mut() {
            rotate_columns(q, d, i, d, cs[i], sn[i]);
        }
    }
    // Undo the shift.
    for i in 0..p {
        h[i * d + i] += mu;
    }
}

/// Columns `i, i+1` of the `dxd` row-major `m` across rows `0..rows` times the
/// transposed rotation `(c, s)` of [`qr_step`].
fn rotate_columns(m: &mut [C], d: usize, i: usize, rows: usize, c: C, s: C) {
    for row in 0..rows {
        let hi = m[row * d + i];
        let hi1 = m[row * d + i + 1];
        m[row * d + i] = c * hi + s * hi1;
        m[row * d + i + 1] = -s.conj() * hi + c.conj() * hi1;
    }
}

/// Wilkinson shift: the eigenvalue of the trailing `2x2` block of the active
/// leading `pxp` submatrix that is closer to the corner entry `h[p-1,p-1]`.
fn wilkinson_shift(h: &[C], d: usize, p: usize) -> C {
    let a = h[(p - 2) * d + (p - 2)];
    let b = h[(p - 2) * d + (p - 1)];
    let c = h[(p - 1) * d + (p - 2)];
    let dd = h[(p - 1) * d + (p - 1)];
    let tr = a + dd;
    let disc = ((a - dd) * (a - dd) + b * c * 4.0).sqrt(); // lit: quadratic formula
    let l1 = (tr + disc) / 2.0;
    let l2 = (tr - disc) / 2.0;
    if (l1 - dd).norm() <= (l2 - dd).norm() {
        l1
    } else {
        l2
    }
}

/// Eigenvalues of a general dense complex matrix `a` (`dxd`, row-major).
/// Hessenberg reduction followed by shifted-QR with bottom-right deflation.
/// Iterations are capped; any non-converged tail is reported as its current
/// diagonal (safe - see the module note on why approximate spectra are fine).
pub fn eigenvalues(a: &[C], d: usize) -> Vec<C> {
    let mut h = a.to_vec();
    to_hessenberg(&mut h, d, None);
    let mut eigs = vec![C::new(0.0, 0.0); d];
    let mut p = d;
    let mut budget = QR_ITER_PER_DIM * d + QR_ITER_PER_DIM;
    while p > 0 {
        if p == 1 {
            eigs[0] = h[0];
            break;
        }
        // Deflate a negligible bottom subdiagonal.
        let sub = h[(p - 1) * d + (p - 2)].norm();
        let dg = h[(p - 2) * d + (p - 2)].norm() + h[(p - 1) * d + (p - 1)].norm();
        if sub <= f64::EPSILON * dg.max(TINY) {
            eigs[p - 1] = h[(p - 1) * d + (p - 1)];
            p -= 1;
            continue;
        }
        if budget == 0 {
            // Give up: report the remaining diagonal as-is.
            for i in 0..p {
                eigs[i] = h[i * d + i];
            }
            break;
        }
        budget -= 1;
        let mu = wilkinson_shift(&h, d, p);
        qr_step(&mut h, d, p, mu, None);
    }
    eigs
}

/// The complex Schur decomposition `A = Q T Q^H` of a general dense matrix `a`
/// (`dxd`, row-major): `T` upper triangular with the eigenvalues on its diagonal,
/// `Q` unitary, both `dxd` row-major. The QR iteration of `eigenvalues` with the
/// transform accumulated; `None` where it does not converge within the same
/// budget (`QR_ITER_PER_DIM` per dimension), rather than a partial result.
pub fn schur(a: &[C], d: usize) -> Option<(Vec<C>, Vec<C>)> {
    let zero = C::new(0.0, 0.0);
    let mut t = a.to_vec();
    let mut q = vec![zero; d * d];
    for i in 0..d {
        q[i * d + i] = C::new(1.0, 0.0);
    }
    to_hessenberg(&mut t, d, Some(&mut q));
    // the reflectors leave rounding below the subdiagonal: Hessenberg exactly
    for i in 0..d {
        for j in 0..i.saturating_sub(1) {
            t[i * d + j] = zero;
        }
    }
    let mut p = d;
    let mut budget = QR_ITER_PER_DIM * d + QR_ITER_PER_DIM;
    while p > 1 {
        let sub = t[(p - 1) * d + (p - 2)].norm();
        let dg = t[(p - 2) * d + (p - 2)].norm() + t[(p - 1) * d + (p - 1)].norm();
        if sub <= f64::EPSILON * dg.max(TINY) {
            t[(p - 1) * d + (p - 2)] = zero;
            p -= 1;
            continue;
        }
        if budget == 0 {
            return None;
        }
        budget -= 1;
        let mu = wilkinson_shift(&t, d, p);
        qr_step(&mut t, d, p, mu, Some(&mut q));
    }
    Some((t, q))
}

/// The eigenvalues of a small dense matrix with their right eigenvectors
/// ([`eigen`]).
#[derive(Debug, Clone)]
pub struct Eigen {
    /// The eigenvalues, in the order of the Schur form's diagonal.
    pub values: Vec<C>,
    /// The right eigenvectors, `dxd` row-major: column `k` belongs to
    /// `values[k]`, unit Euclidean norm.
    pub vectors: Vec<C>,
}

/// The eigenvalues and right eigenvectors of a general dense complex matrix `a`
/// (`dxd`, row-major), `A x_k = lambda_k x_k`: the Schur form ([`schur`]) and per
/// eigenvalue the back substitution `(T - lambda_k I) y = 0`, `y_k = 1`, then
/// `x_k = Q y` normalized. A denominator `T_ii - lambda_k` below the working
/// precision of `T` (`f64::EPSILON` times its largest entry: a repeated
/// eigenvalue) is replaced by that floor, as LAPACK's `ztrevc` does, so a
/// defective matrix gives nearly parallel vectors instead of a division by zero.
/// `None` where the Schur form does not converge.
pub fn eigen(a: &[C], d: usize) -> Option<Eigen> {
    let zero = C::new(0.0, 0.0);
    let (t, q) = schur(a, d)?;
    let tmax = t.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
    let floor = f64::EPSILON * tmax.max(TINY);
    let mut vectors = vec![zero; d * d];
    let mut y = vec![zero; d];
    for k in 0..d {
        let lam = t[k * d + k];
        y.iter_mut().for_each(|v| *v = zero);
        y[k] = C::new(1.0, 0.0);
        for i in (0..k).rev() {
            let mut s = zero;
            for j in (i + 1)..=k {
                s += t[i * d + j] * y[j];
            }
            let mut den = t[i * d + i] - lam;
            if den.norm() < floor {
                den = C::new(floor, 0.0);
            }
            y[i] = -s / den;
        }
        let mut x: Vec<C> = (0..d)
            .map(|i| (0..=k).map(|j| q[i * d + j] * y[j]).sum())
            .collect();
        normalize(&mut x);
        for i in 0..d {
            vectors[i * d + k] = x[i];
        }
    }
    let values = (0..d).map(|i| t[i * d + i]).collect();
    Some(Eigen { values, vectors })
}

/// One-vector inverse iteration for the eigenvector of `a` (`dxd` row-major)
/// belonging to the (accurate) eigenvalue `mu`: solve `(A - (mu+eps)I) w = w`
/// twice from a fixed seed, returning the unit-norm result. Returns `None` if the
/// shifted solve is singular even after perturbation.
fn inverse_iteration(a: &[C], d: usize, mu: C) -> Option<Vec<C>> {
    // Perturb the shift slightly off the exact eigenvalue so (A - muI) is solvable.
    let scale = {
        let mut s = 0.0f64;
        for v in a.iter() {
            s = s.max(v.norm());
        }
        s.max(1.0)
    };
    let eps = C::new(
        INVERSE_ITERATION_SHIFT * scale,
        INVERSE_ITERATION_SHIFT * scale,
    );
    let mut shifted = a.to_vec();
    let shift = mu + eps;
    for i in 0..d {
        shifted[i * d + i] -= shift;
    }
    // Seed: normalized all-ones (deterministic).
    let mut w = vec![C::new(1.0, 0.0); d];
    normalize(&mut w);
    for _ in 0..2 {
        let sol = lu_solve(&shifted, d, &w, 1)?;
        w = sol;
        normalize(&mut w);
    }
    Some(w)
}

/// Scale a complex vector to unit Euclidean norm (no-op on a zero vector).
fn normalize(w: &mut [C]) {
    let mut nrm = 0.0f64;
    for v in w.iter() {
        nrm += v.norm_sqr();
    }
    nrm = nrm.sqrt();
    if nrm > 0.0 {
        let inv = 1.0 / nrm;
        for v in w.iter_mut() {
            *v *= inv;
        }
    }
}

/// The `k` harmonic-Ritz pairs of **smallest magnitude** for the GCRO-DR
/// generalized eigenproblem `M_1 g = theta M_2 g` (both `dxd`, row-major). Returns
/// `(theta_i, g_i)` sorted by ascending `|theta|`, at most `k` of them. The standard-form
/// matrix `T = M_2^-1 M_1` is formed by an LU solve; its eigenvalues are the
/// harmonic-Ritz values and its eigenvectors (via inverse iteration) the
/// coefficient vectors `g_i`. On a singular `M_2` the empty list is returned (the
/// caller then keeps its previous recycle subspace).
pub fn harmonic_ritz_smallest(m1: &[C], m2: &[C], d: usize, k: usize) -> Vec<(C, Vec<C>)> {
    if d == 0 || k == 0 {
        return Vec::new();
    }
    // T = M_2^-1 M_1 (solve M_2 T = M_1).
    let t = match lu_solve(m2, d, m1, d) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let eigs = eigenvalues(&t, d);
    // Sort eigenvalue indices by ascending magnitude.
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| {
        eigs[i]
            .norm()
            .partial_cmp(&eigs[j].norm())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let take = k.min(d);
    let mut out = Vec::with_capacity(take);
    for &idx in order.iter().take(take) {
        let theta = eigs[idx];
        if let Some(g) = inverse_iteration(&t, d, theta) {
            if g.iter().all(|z| z.re.is_finite() && z.im.is_finite()) {
                out.push((theta, g));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_set(mut got: Vec<C>, mut want: Vec<C>, tol: f64) {
        // Match each wanted eigenvalue to the nearest computed one.
        assert_eq!(got.len(), want.len());
        want.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        got.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).norm() < tol, "eig {g} vs {w}");
        }
    }

    /// `A x = A_dense x` for a row-major `dxd` matrix.
    fn matvec(a: &[C], d: usize, x: &[C]) -> Vec<C> {
        (0..d)
            .map(|i| (0..d).map(|j| a[i * d + j] * x[j]).sum())
            .collect()
    }

    /// A deterministic dense complex test matrix of dimension `d`.
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

    /// The Schur form reproduces the matrix (`Q T Q^H = A`), `Q` is unitary and `T`
    /// upper triangular with the eigenvalues of [`eigenvalues`] on its diagonal.
    #[test]
    fn schur_reproduces_the_matrix() {
        for (d, seed) in [(1, 1), (2, 2), (7, 3), (40, 4), (120, 5)] {
            let a = random(d, seed);
            let (t, q) = schur(&a, d).expect("converges");
            let anorm = a.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
            for i in 0..d {
                for j in 0..i {
                    assert_eq!(t[i * d + j], C::new(0.0, 0.0), "T below the diagonal");
                }
                for j in 0..d {
                    // (Q T Q^H)_ij and (Q^H Q)_ij
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
            let diag: Vec<C> = (0..d).map(|i| t[i * d + i]).collect();
            approx_set(diag, eigenvalues(&a, d), 1e-9 * anorm.max(1.0));
        }
    }

    /// Every eigenpair satisfies `A x = lambda x` to the working precision, the
    /// vectors are unit, and a matrix with known eigenvalues (`W diag W^-1`) gives
    /// them.
    #[test]
    fn eigen_pairs_satisfy_the_eigenproblem() {
        for (d, seed) in [(3, 11), (30, 12), (90, 13)] {
            let a = random(d, seed);
            let e = eigen(&a, d).expect("converges");
            let anorm = a.iter().map(|v| v.norm()).fold(0.0f64, f64::max);
            for k in 0..d {
                let x: Vec<C> = (0..d).map(|i| e.vectors[i * d + k]).collect();
                let nx = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                assert!((nx - 1.0).abs() < 1e-12);
                let ax = matvec(&a, d, &x);
                let r = ax
                    .iter()
                    .zip(&x)
                    .map(|(u, v)| (u - e.values[k] * v).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(
                    r < 1e-11 * anorm * d as f64,
                    "d {d} pair {k}: residual {r:e}"
                );
            }
        }
        // known spectrum
        let d = 12;
        let w = random(d, 21);
        let lam: Vec<C> = (0..d).map(|k| C::new(k as f64, -0.1 * k as f64)).collect();
        let winv = {
            let mut id = vec![C::new(0.0, 0.0); d * d];
            for i in 0..d {
                id[i * d + i] = C::new(1.0, 0.0);
            }
            lu_solve(&w, d, &id, d).expect("invertible")
        };
        let a: Vec<C> = (0..d * d)
            .map(|q| {
                let (i, j) = (q / d, q % d);
                (0..d)
                    .map(|k| w[i * d + k] * lam[k] * winv[k * d + j])
                    .sum()
            })
            .collect();
        let e = eigen(&a, d).expect("converges");
        approx_set(e.values, lam, 1e-9);
    }

    /// A Jordan block (one defective eigenvalue) gives finite vectors along its
    /// one eigenvector, not a division by zero.
    #[test]
    fn eigen_of_a_defective_matrix_is_finite() {
        let d = 4;
        let mut a = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            a[i * d + i] = C::new(2.0, 1.0);
            if i + 1 < d {
                a[i * d + i + 1] = C::new(1.0, 0.0);
            }
        }
        let e = eigen(&a, d).expect("converges");
        for k in 0..d {
            assert!((e.values[k] - C::new(2.0, 1.0)).norm() < 1e-12);
            assert!(
                e.vectors[k].norm() > 1.0 - 1e-6,
                "along e_1: {}",
                e.vectors[k]
            );
            assert!(e
                .vectors
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite()));
        }
    }

    #[test]
    fn eig_diagonal() {
        let d = 4;
        let mut a = vec![C::new(0.0, 0.0); d * d];
        let vals = [
            C::new(1.0, 0.0),
            C::new(2.0, 1.0),
            C::new(-3.0, 0.5),
            C::new(0.2, -2.0),
        ];
        for i in 0..d {
            a[i * d + i] = vals[i];
        }
        let e = eigenvalues(&a, d);
        approx_set(e, vals.to_vec(), 1e-10);
    }

    #[test]
    fn eig_upper_triangular() {
        // Eigenvalues of an upper-triangular matrix are its diagonal.
        let d = 3;
        let a = vec![
            C::new(2.0, 0.0),
            C::new(1.0, 1.0),
            C::new(3.0, -1.0),
            C::new(0.0, 0.0),
            C::new(-1.0, 0.5),
            C::new(0.4, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(5.0, 2.0),
        ];
        let e = eigenvalues(&a, d);
        approx_set(
            e,
            vec![C::new(2.0, 0.0), C::new(-1.0, 0.5), C::new(5.0, 2.0)],
            1e-9,
        );
    }

    #[test]
    fn eig_general_known() {
        // A = [[0,1],[-2,-3]] has eigenvalues -1 and -2.
        let d = 2;
        let a = vec![
            C::new(0.0, 0.0),
            C::new(1.0, 0.0),
            C::new(-2.0, 0.0),
            C::new(-3.0, 0.0),
        ];
        let e = eigenvalues(&a, d);
        approx_set(e, vec![C::new(-1.0, 0.0), C::new(-2.0, 0.0)], 1e-10);
    }

    #[test]
    fn eig_companion_5x5() {
        // Companion matrix of (x-1)(x-2)(x-3)(x-4)(x-5): eigenvalues 1..5.
        // p(x) = x^5 -15x^4 +85x^3 -225x^2 +274x -120.
        let d = 5;
        let mut a = vec![C::new(0.0, 0.0); d * d];
        for i in 1..d {
            a[i * d + (i - 1)] = C::new(1.0, 0.0);
        }
        let coeffs = [-120.0, 274.0, -225.0, 85.0, -15.0]; // c0..c4
        for (i, &cf) in coeffs.iter().enumerate() {
            a[i * d + (d - 1)] = C::new(-cf, 0.0);
        }
        let e = eigenvalues(&a, d);
        approx_set(e, (1..=5).map(|v| C::new(v as f64, 0.0)).collect(), 1e-6);
    }

    #[test]
    fn lu_solve_identity_and_general() {
        let d = 3;
        let a = vec![
            C::new(2.0, 0.0),
            C::new(0.0, 1.0),
            C::new(1.0, 0.0),
            C::new(1.0, 0.0),
            C::new(3.0, 0.0),
            C::new(0.0, -1.0),
            C::new(0.0, 0.0),
            C::new(1.0, 0.0),
            C::new(4.0, 0.0),
        ];
        // Known x, build b = A x, recover x.
        let x = [C::new(1.0, 1.0), C::new(-2.0, 0.5), C::new(0.3, -0.7)];
        let mut b = vec![C::new(0.0, 0.0); d];
        for i in 0..d {
            for j in 0..d {
                b[i] += a[i * d + j] * x[j];
            }
        }
        let sol = lu_solve(&a, d, &b, 1).unwrap();
        for i in 0..d {
            assert!(
                (sol[i] - x[i]).norm() < 1e-12,
                "x[{i}] {} vs {}",
                sol[i],
                x[i]
            );
        }
    }

    #[test]
    fn eig_random_8x8_matches_numpy() {
        let d = 8;
        let a = vec![
            C::new(1.6243453636632417, 0.48851814653749703),
            C::new(-0.6117564136500754, -0.07557171302105573),
            C::new(-0.5281717522634557, 1.131629387451427),
            C::new(-1.0729686221561705, 1.5198168164221988),
            C::new(0.8654076293246785, 2.1855754065331614),
            C::new(-2.3015386968802827, -1.3964963354881377),
            C::new(1.74481176421648, -1.4441138054295894),
            C::new(-0.7612069008951028, -0.5044658629464512),
            C::new(0.31903909605709857, 0.16003706944783047),
            C::new(-0.2493703754774101, 0.8761689211162249),
            C::new(1.462107937044974, 0.31563494724160523),
            C::new(-2.060140709497654, -2.022201215824003),
            C::new(-0.3224172040135075, -0.3062040126283718),
            C::new(-0.38405435466841564, 0.8279746426072462),
            C::new(1.1337694423354374, 0.2300947353643834),
            C::new(-1.0998912673140309, 0.7620111803120247),
            C::new(-0.17242820755043575, -0.22232814261035927),
            C::new(-0.8778584179213718, -0.20075806892999745),
            C::new(0.04221374671559283, 0.1865613909882843),
            C::new(0.5828152137158222, 0.4100516472082563),
            C::new(-1.1006191772129212, 0.19829972012676975),
            C::new(1.1447237098396141, 0.11900864580745882),
            C::new(0.9015907205927955, -0.6706622862890306),
            C::new(0.5024943389018682, 0.3775637863209194),
            C::new(0.9008559492644118, 0.12182127099143693),
            C::new(-0.6837278591743331, 1.1294839079119197),
            C::new(-0.12289022551864817, 1.198917879901507),
            C::new(-0.9357694342590688, 0.18515641748394385),
            C::new(-0.2678880796260159, -0.3752849500901142),
            C::new(0.530355466738186, -0.6387304074542224),
            C::new(-0.691660751725309, 0.4234943540641129),
            C::new(-0.39675352685597737, 0.07734006834855942),
            C::new(-0.6871727001195994, -0.3438536755710756),
            C::new(-0.8452056414987196, 0.04359685683424694),
            C::new(-0.671246130836819, -0.6200008439481293),
            C::new(-0.01266459891890136, 0.6980320340722189),
            C::new(-1.1173103486352778, -0.4471285647859982),
            C::new(0.23441569781709215, 1.2245077048054989),
            C::new(1.6598021771098705, 0.4034916417908),
            C::new(0.7420441605773356, 0.593578523237067),
            C::new(-0.19183555236161492, -1.0949118457410418),
            C::new(-0.8876289640848363, 0.1693824330586681),
            C::new(-0.7471582937508376, 0.7405564510962748),
            C::new(1.6924546010277466, -0.9537006018079346),
            C::new(0.05080775477602897, -0.26621850600362207),
            C::new(-0.6369956465693534, 0.03261454669335856),
            C::new(0.19091548466746602, -1.3731173202467557),
            C::new(2.100255136478842, 0.31515939204229176),
            C::new(0.12015895248162915, 0.8461606475850334),
            C::new(0.6172031097074192, -0.8595159408319863),
            C::new(0.3001703199558275, 0.35054597866410736),
            C::new(-0.35224984649351865, -1.3122834112374318),
            C::new(-1.1425181980221402, -0.038695509266051115),
            C::new(-0.3493427224128775, -1.6157723547032947),
            C::new(-0.2088942333747781, 1.121417708235664),
            C::new(0.5866231911821976, 0.4089005379368278),
            C::new(0.8389834138745049, -0.024616955875778355),
            C::new(0.9311020813035573, -0.7751616191691596),
            C::new(0.2855873252542588, 1.2737559301587766),
            C::new(0.8851411642707281, 1.9671017492547347),
            C::new(-0.7543979409966528, -1.857981864446752),
            C::new(1.2528681552332879, 1.2361640304528203),
            C::new(0.5129298204180088, 1.6276507531489064),
            C::new(-0.29809283510271567, 0.3380116965744758),
        ];
        let mut e = eigenvalues(&a, d);
        e.sort_by(|x, y| x.norm().partial_cmp(&y.norm()).unwrap());
        let want = [
            (-3.0246896865f64, 0.0101947612f64),
            (-0.5247920659, -0.5118133206),
            (0.2920100885, 0.9969847067),
            (1.5346484662, -0.9970700478),
            (1.9131085994, 0.5558594025),
            (-1.9552701000, 2.6632188664),
            (2.0362020110, 2.5655692069),
            (-2.0510910757, -2.5016233124),
        ];
        let mut w: Vec<(f64, f64)> = want.to_vec();
        w.sort_by(|x, y| {
            (x.0 * x.0 + x.1 * x.1)
                .partial_cmp(&(y.0 * y.0 + y.1 * y.1))
                .unwrap()
        });
        for (g, wv) in e.iter().zip(w.iter()) {
            let dd = ((g.re - wv.0).powi(2) + (g.im - wv.1).powi(2)).sqrt();
            assert!(dd < 1e-6, "eig {} vs ({},{}) dist={}", g, wv.0, wv.1, dd);
        }
    }

    #[test]
    fn harmonic_ritz_recovers_smallest() {
        // M2 = I, M1 = diag(1,2,3,4): generalized problem reduces to standard,
        // smallest |theta| = 1 with eigenvector e_0.
        let d = 4;
        let mut m1 = vec![C::new(0.0, 0.0); d * d];
        let mut m2 = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            m1[i * d + i] = C::new((i + 1) as f64, 0.0);
            m2[i * d + i] = C::new(1.0, 0.0);
        }
        let pairs = harmonic_ritz_smallest(&m1, &m2, d, 2);
        assert_eq!(pairs.len(), 2);
        assert!(
            (pairs[0].0 - C::new(1.0, 0.0)).norm() < 1e-8,
            "theta0 {}",
            pairs[0].0
        );
        assert!(
            (pairs[1].0 - C::new(2.0, 0.0)).norm() < 1e-8,
            "theta1 {}",
            pairs[1].0
        );
        // Eigenvector for theta=1 aligns with e_0.
        let g = &pairs[0].1;
        let dom = g
            .iter()
            .map(|z| z.norm())
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(dom, 0, "dominant component should be e_0");
    }
}
