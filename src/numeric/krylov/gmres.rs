//! Restarted GMRES with modified Gram-Schmidt and a conditional second pass.

use super::util::*;
use super::*;
use crate::memory::grow;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// **Flexible** right-preconditioned restarted **GMRES(`restart`)** (FGMRES,
/// Saad 1993) for a general (unsymmetric) operator - the natural Krylov method
/// for unsymmetric MoM/FEM systems where COCG/COCR do not apply. `op` may be
/// matrix-free; `precond` supplies `M^-1` (e.g. an
/// [`LuSolver`](crate::LuSolver) near-field factor).
/// Solves `A x = b` from the optional initial guess `x0` (default `x_0 = 0`).
///
/// **Warm start:** pass `x0 = Some(prev)` to seed the iteration from a
/// previous, related solution - on a sequence of slowly varying systems this
/// often cuts the iteration count substantially. Convergence is still measured
/// relative to ||b||.
///
/// **Flexible variant:** the preconditioned Arnoldi vectors
/// `z_j = M^-1 v_j` (already formed to build `w = A z_j`) are *kept* as a second
/// basis `Z = [z_0 ... z_{m-1}]`, and the restart update is `x += Z y` directly.
/// This (a) removes the one extra `M^-1` solve per cycle that plain right-
/// preconditioned GMRES spends rebuilding `M^-1(V y)`, and (b) makes the method
/// flexible: `M` may **vary** between steps (an inner Krylov solve, or a
/// preconditioner strengthened across iterations). Cost: one extra `n*(m+1)`
/// basis of storage.
pub fn gmres<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    settings: &KrylovSettings,
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
    let (tol, max_iter, reorth_eta) = (settings.tol, settings.max_iter, settings.reorth_eta);
    let flexible_bases = 2;
    let m = settings.restart_for(n, 1, std::mem::size_of::<T>(), flexible_bases);
    let bnorm = norm2(b);
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n {
                return Err(RslabError::DimensionMismatch {
                    expected: n,
                    got: g.len(),
                });
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
            stop: StopReason::Converged,
        });
    }

    let mut total = 0usize;
    let mut w = vec![T::zero(); n];
    let mut ax = vec![T::zero(); n];
    // https://github.com/milanofthe/rslab/issues/287
    let mut v = basis(n * (m + 1));
    let mut zb = basis(n * m);
    let mut h = vec![T::zero(); (m + 1) * m];
    let mut cs = vec![T::zero(); m];
    let mut sn = vec![T::zero(); m];
    let mut g = vec![T::zero(); m + 1];
    let mut y = vec![T::zero(); m];

    let mut final_res;
    // https://github.com/milanofthe/rslab/issues/287
    loop {
        op.apply(&x, &mut ax);
        let r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta = norm2(&r);
        final_res = beta / bnorm;
        if final_res <= tol || total >= max_iter {
            break;
        }
        let inv_beta = T::from_real(1.0 / beta);
        grow(&mut v, n);
        for k in 0..n {
            v[k] = r[k] * inv_beta;
        }
        for e in h.iter_mut() {
            *e = T::zero();
        }
        for e in cs.iter_mut() {
            *e = T::zero();
        }
        for e in sn.iter_mut() {
            *e = T::zero();
        }
        for e in g.iter_mut() {
            *e = T::zero();
        }
        g[0] = T::from_real(beta);
        let mut jdim = 0usize;
        for j in 0..m {
            if total >= max_iter {
                break;
            }
            grow(&mut zb, (j + 1) * n);
            precond.apply(&v[j * n..j * n + n], &mut zb[j * n..j * n + n])?;
            op.apply(&zb[j * n..j * n + n], &mut w);
            let wnorm0 = norm2(&w);
            for i in 0..=j {
                let hij = dotc(&v[i * n..i * n + n], &w);
                h[i * m + j] = hij;
                for k in 0..n {
                    w[k] = w[k] - hij * v[i * n + k];
                }
            }
            let mut hn = norm2(&w);
            // https://github.com/milanofthe/rslab/issues/287
            if hn < reorth_eta * wnorm0 {
                for i in 0..=j {
                    let s = dotc(&v[i * n..i * n + n], &w);
                    h[i * m + j] = h[i * m + j] + s;
                    for k in 0..n {
                        w[k] = w[k] - s * v[i * n + k];
                    }
                }
                hn = norm2(&w);
            }
            h[(j + 1) * m + j] = T::from_real(hn);
            grow(&mut v, (j + 2) * n);
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                for k in 0..n {
                    v[(j + 1) * n + k] = w[k] * inv;
                }
            } else {
                for k in 0..n {
                    v[(j + 1) * n + k] = T::zero();
                }
            }
            for i in 0..j {
                let temp = cs[i].conj() * h[i * m + j] + sn[i].conj() * h[(i + 1) * m + j];
                h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
                h[i * m + j] = temp;
            }
            let (c, s) = givens(h[j * m + j], h[(j + 1) * m + j]);
            cs[j] = c;
            sn[j] = s;
            h[j * m + j] = c.conj() * h[j * m + j] + s.conj() * h[(j + 1) * m + j];
            h[(j + 1) * m + j] = T::zero();
            let g_next = -s * g[j];
            g[j] = c.conj() * g[j];
            g[j + 1] = g_next;
            total += 1;
            jdim = j + 1;
            if g[j + 1].magnitude() / bnorm <= tol {
                break;
            }
        }
        let jd = well_conditioned_dim(&h, m, jdim);
        for i in (0..jd).rev() {
            let mut s = g[i];
            for k in (i + 1)..jd {
                s = s - h[i * m + k] * y[k];
            }
            y[i] = s * h[i * m + i].recip();
        }
        for i in 0..jd {
            let yi = y[i];
            for k in 0..n {
                x[k] = x[k] + zb[i * n + k] * yi;
            }
        }
    }

    Ok(KrylovResult {
        x,
        iters: total,
        converged: final_res <= tol,
        final_res,
        stop: if final_res <= tol {
            StopReason::Converged
        } else {
            StopReason::MaxIter
        },
    })
}
