//! Small **dense** complex linear algebra: the projected matrices of Krylov
//! methods (GCRO-DR's harmonic-Ritz extraction, the Ritz values of a
//! [`BlockArnoldi`](crate::BlockArnoldi)). These routines operate on small
//! matrices (a restart length, a few hundred at most), never in the Krylov hot
//! loop, so they are written for clarity and robustness, not peak speed, and
//! everything runs in `Complex<f64>` regardless of the outer solve's scalar
//! field.
//!
//! * `lu`: a partial-pivot complex LU solve (`A X = B`).
//! * `qr_iterate`: the Hessenberg QR iteration (Wilkinson shifts, LAPACK's
//!   exceptional shifts, deflation at the bottom), optionally accumulating the
//!   transform: the eigenvalues of `eigenvalues`, the Schur form of
//!   [`Schur`](crate::dense::Schur).
//! * the Schur form's eigenvectors: back and forward substitution on `T` with a
//!   per-eigenvalue floor and overflow rescaling.
//!
//! The public, checked entry points are [`crate::dense`]'s. `eigenvalues` and
//! `harmonic_ritz_smallest` serve GCRO-DR and are best effort: the
//! harmonic-Ritz vectors only seed the recycle subspace, which *accelerates*
//! convergence, and the outer GMRES true-residual test is authoritative, so a
//! defective or inaccurate subspace can only make the solve slower, never wrong.

use crate::constants::{
    DENSE_SINGULAR_TOL, DISCRIMINANT_FACTOR, INVERSE_ITERATION_SHIFT, QR_EXCEPTIONAL_PERIOD,
    QR_EXCEPTIONAL_SHIFT, QR_ITER_PER_DIM, TINY,
};
use num_complex::Complex;

type C = Complex<f64>;

pub(crate) fn lu(a: &[C], d: usize, b: &[C], nrhs: usize) -> Option<Vec<C>> {
    if d == 0 {
        return Some(Vec::new());
    }
    let mut lu = a.to_vec();
    let mut x = b.to_vec();
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

#[allow(clippy::needless_range_loop)]
fn to_hessenberg(a: &mut [C], d: usize, mut q: Option<&mut [C]>) {
    if d <= 2 {
        return;
    }
    for col in 0..(d - 2) {
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
        reflect_right(a, d, &v, col + 1, vnorm);
        if let Some(q) = q.as_deref_mut() {
            reflect_right(q, d, &v, col + 1, vnorm);
        }
    }
}

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

fn qr_step(h: &mut [C], d: usize, p: usize, mu: C, mut q: Option<&mut [C]>) {
    let cols = if q.is_some() { d } else { p };
    for i in 0..p {
        h[i * d + i] -= mu;
    }
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
        for j in i..cols {
            let hij = h[i * d + j];
            let hi1j = h[(i + 1) * d + j];
            h[i * d + j] = c.conj() * hij + s.conj() * hi1j;
            h[(i + 1) * d + j] = -s * hij + c * hi1j;
        }
    }
    for i in 0..(p - 1) {
        rotate_columns(h, d, i, (i + 2).min(p), cs[i], sn[i]);
        if let Some(q) = q.as_deref_mut() {
            rotate_columns(q, d, i, d, cs[i], sn[i]);
        }
    }
    for i in 0..p {
        h[i * d + i] += mu;
    }
}

fn rotate_columns(m: &mut [C], d: usize, i: usize, rows: usize, c: C, s: C) {
    for row in 0..rows {
        let hi = m[row * d + i];
        let hi1 = m[row * d + i + 1];
        m[row * d + i] = c * hi + s * hi1;
        m[row * d + i + 1] = -s.conj() * hi + c.conj() * hi1;
    }
}

fn wilkinson_shift(h: &[C], d: usize, p: usize) -> C {
    let a = h[(p - 2) * d + (p - 2)];
    let b = h[(p - 2) * d + (p - 1)];
    let c = h[(p - 1) * d + (p - 2)];
    let dd = h[(p - 1) * d + (p - 1)];
    let tr = a + dd;
    let disc = ((a - dd) * (a - dd) + b * c * DISCRIMINANT_FACTOR).sqrt();
    let l1 = (tr + disc) / 2.0;
    let l2 = (tr - disc) / 2.0;
    if (l1 - dd).norm() <= (l2 - dd).norm() {
        l1
    } else {
        l2
    }
}

fn qr_iterate(h: &mut [C], d: usize, mut q: Option<&mut [C]>, budget: usize) -> usize {
    let mut p = d;
    let mut left = budget;
    let mut since = 0usize;
    while p > 1 {
        let sub = h[(p - 1) * d + (p - 2)].norm();
        let dg = h[(p - 2) * d + (p - 2)].norm() + h[(p - 1) * d + (p - 1)].norm();
        if sub <= f64::EPSILON * dg.max(TINY) {
            if q.is_some() {
                h[(p - 1) * d + (p - 2)] = C::new(0.0, 0.0);
            }
            p -= 1;
            since = 0;
            continue;
        }
        if left == 0 {
            return p;
        }
        left -= 1;
        since += 1;
        let mu = if since.is_multiple_of(QR_EXCEPTIONAL_PERIOD) {
            if !(since / QR_EXCEPTIONAL_PERIOD).is_multiple_of(2) {
                h[0] + QR_EXCEPTIONAL_SHIFT * h[d].re.abs()
            } else {
                h[(p - 1) * d + (p - 1)] + QR_EXCEPTIONAL_SHIFT * h[(p - 1) * d + (p - 2)].re.abs()
            }
        } else {
            wilkinson_shift(h, d, p)
        };
        qr_step(h, d, p, mu, q.as_deref_mut());
    }
    0
}

pub(crate) fn qr_budget(d: usize) -> usize {
    QR_ITER_PER_DIM * d + QR_ITER_PER_DIM
}

/// Eigenvalues of a general dense complex matrix `a` (`dxd`, row-major), best
/// effort: Hessenberg reduction and the QR iteration of `qr_iterate`; a block
/// that does not deflate within the budget is reported as its current diagonal
/// (safe for GCRO-DR, see the module note).
pub fn eigenvalues(a: &[C], d: usize) -> Vec<C> {
    let mut h = a.to_vec();
    to_hessenberg(&mut h, d, None);
    qr_iterate(&mut h, d, None, qr_budget(d));
    (0..d).map(|i| h[i * d + i]).collect()
}

pub(crate) fn schur_form(a: &[C], d: usize, budget: usize) -> Result<(Vec<C>, Vec<C>), usize> {
    let zero = C::new(0.0, 0.0);
    let mut t = a.to_vec();
    let mut q = vec![zero; d * d];
    for i in 0..d {
        q[i * d + i] = C::new(1.0, 0.0);
    }
    to_hessenberg(&mut t, d, Some(&mut q));
    for i in 0..d {
        for j in 0..i.saturating_sub(1) {
            t[i * d + j] = zero;
        }
    }
    match qr_iterate(&mut t, d, Some(&mut q), budget) {
        0 => Ok((t, q)),
        p => Err(p),
    }
}

// https://github.com/milanofthe/rslab/issues/290
pub(crate) fn triangular_eigenvector(t: &[C], d: usize, k: usize, left: bool) -> Option<Vec<C>> {
    let zero = C::new(0.0, 0.0);
    let lam = t[k * d + k];
    let floor = (f64::EPSILON * lam.norm()).max(TINY);
    let big = f64::MAX.sqrt();
    let mut y = vec![zero; d];
    y[k] = C::new(1.0, 0.0);
    let order: Vec<usize> = if left {
        ((k + 1)..d).collect()
    } else {
        (0..k).rev().collect()
    };
    for i in order {
        let s: C = if left {
            (k..i).map(|l| y[l] * t[l * d + i]).sum()
        } else {
            ((i + 1)..=k).map(|j| t[i * d + j] * y[j]).sum()
        };
        let mut den = t[i * d + i] - lam;
        if den.norm() < floor {
            den = C::new(floor, 0.0);
        }
        y[i] = -s / den;
        if !(y[i].re.is_finite() && y[i].im.is_finite()) {
            return None;
        }
        if y[i].norm() > big {
            let inv = 1.0 / y[i].norm();
            y.iter_mut().for_each(|v| *v *= inv);
        }
    }
    Some(y)
}

fn inverse_iteration(a: &[C], d: usize, mu: C) -> Option<Vec<C>> {
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
    let mut w = vec![C::new(1.0, 0.0); d];
    normalize(&mut w);
    for _ in 0..2 {
        let sol = lu(&shifted, d, &w, 1)?;
        w = sol;
        normalize(&mut w);
    }
    Some(w)
}

pub(crate) fn normalize(w: &mut [C]) {
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
    let t = match lu(m2, d, m1, d) {
        Some(t) => t,
        None => return Vec::new(),
    };
    let eigs = eigenvalues(&t, d);
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
    use crate::test_constants::{solve_backward_error, UC64};

    fn approx_set(mut got: Vec<C>, mut want: Vec<C>, tol: f64) {
        assert_eq!(got.len(), want.len());
        want.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        got.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).norm() < tol, "eig {g} vs {w}");
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
    fn eig_companion_of_roots_one_to_five() {
        let d = 5;
        let mut a = vec![C::new(0.0, 0.0); d * d];
        for i in 1..d {
            a[i * d + (i - 1)] = C::new(1.0, 0.0);
        }
        let ascending_coeffs = [-120.0, 274.0, -225.0, 85.0, -15.0];
        for (i, &cf) in ascending_coeffs.iter().enumerate() {
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
        let x = [C::new(1.0, 1.0), C::new(-2.0, 0.5), C::new(0.3, -0.7)];
        let mut b = vec![C::new(0.0, 0.0); d];
        for i in 0..d {
            for j in 0..d {
                b[i] += a[i * d + j] * x[j];
            }
        }
        let sol = lu(&a, d, &b, 1).unwrap();
        let norm1 = |m: &[C], cols: usize| {
            (0..cols)
                .map(|j| {
                    (0..m.len() / cols)
                        .map(|i| m[i * cols + j].norm())
                        .sum::<f64>()
                })
                .fold(0.0, f64::max)
        };
        let r: Vec<C> = (0..d)
            .map(|i| b[i] - (0..d).map(|j| a[i * d + j] * sol[j]).sum::<C>())
            .collect();
        let omega = norm1(&r, 1) / (norm1(&a, d) * norm1(&sol, 1) + norm1(&b, 1));
        assert!(
            omega <= solve_backward_error(d, UC64),
            "{sol:?} vs {x:?}: backward error {omega:e}"
        );
    }

    #[test]
    fn harmonic_ritz_recovers_smallest() {
        let d = 4;
        let mut m1 = vec![C::new(0.0, 0.0); d * d];
        let mut m2 = vec![C::new(0.0, 0.0); d * d];
        for i in 0..d {
            m1[i * d + i] = C::new((i + 1) as f64, 0.0);
            m2[i * d + i] = C::new(1.0, 0.0);
        }
        let pairs = harmonic_ritz_smallest(&m1, &m2, d, 2);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, C::new(1.0, 0.0));
        assert_eq!(pairs[1].0, C::new(2.0, 0.0));
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
