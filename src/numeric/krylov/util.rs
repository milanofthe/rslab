//! Arithmetic shared by the Krylov solvers: the bilinear and sesquilinear
//! products, Givens rotations, the least-squares rank guard and the batched
//! Gram-Schmidt passes of the block solvers.

use crate::constants::DELAYED_UPDATE_PANEL_VECTORS;
use crate::parallel::{Disjoint, Rayon};
use crate::scalar::Scalar;
use crate::stored::{axpy_as, dotc_as, Rows, Stored};

// https://github.com/milanofthe/rslab/issues/287
pub(super) fn basis<T: Scalar>(cap: usize) -> Vec<T> {
    Vec::with_capacity(cap)
}

#[inline]
pub(super) fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi * yi;
    }
    s
}

#[inline]
pub(super) fn norm2<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.magnitude_sq()).sum::<f64>().sqrt()
}

#[inline]
pub(super) fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    dotc_as(x, y, |v| v)
}

#[inline]
pub(super) fn axpy<T: Scalar>(x: &[T], a: T, out: &mut [T]) {
    axpy_as(x, a, out, |v| v)
}

#[inline]
pub(super) fn givens<T: Scalar>(f: T, g: T) -> (T, T) {
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

// https://github.com/milanofthe/rslab/issues/287
#[inline]
pub(super) fn well_conditioned_dim<T: Scalar>(h: &[T], stride: usize, jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i * stride + i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i * stride + i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

static ORTHO_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::SCAN_NS_PER_ENTRY);

pub(super) fn column_chunks<T: Scalar>(
    cols: usize,
    n: usize,
    chunk: usize,
    count: usize,
    f: impl Fn(usize, usize, usize) + Sync + Send,
) {
    let nchunks = n.div_ceil(chunk);
    let work = crate::parallel::Work::of(count as f64, ORTHO_RATE.of::<T>());
    crate::parallel::ranges::<Rayon>(cols * nchunks, work, |r| {
        for p in r {
            let (a, ci) = (p / nchunks, p % nchunks);
            f(a, ci * chunk, ((ci + 1) * chunk).min(n));
        }
    });
}

fn fold<T: Scalar>(part: &[T], width: usize, out: &mut [T]) {
    let nchunks = part.len() / width;
    let work = crate::parallel::Work::of((nchunks * width) as f64, ORTHO_RATE.of::<T>());
    crate::parallel::chunks_mut::<Rayon, _>(&mut out[..width], 1, work, |t0, o| {
        for (t, o) in (t0..).zip(o) {
            *o = (0..nchunks).fold(T::zero(), |e, ci| e + part[ci * width + t]);
        }
    });
}

// https://doi.org/10.1016/j.parco.2022.102940
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn delayed_project<'v, T: Scalar>(
    layers: &(dyn Fn(usize) -> &'v Stored<T> + Sync),
    off: &[usize],
    u: &[T],
    w: &[T],
    n: usize,
    chunk: usize,
    proj: &mut [T],
    scratch: &mut [T],
) {
    let width = off[off.len() - 1];
    debug_assert!(off
        .windows(2)
        .all(|span| span[1] >= span[0] + 2 && (span[1] - span[0]) % 2 == 0));
    if width == 0 || n == 0 {
        proj[..width].iter_mut().for_each(|p| *p = T::zero());
        return;
    }
    let nchunks = n.div_ceil(chunk);
    let part = &mut scratch[..nchunks * width];
    let work = crate::parallel::Work::of((n * width) as f64, ORTHO_RATE.of::<T>());
    crate::parallel::chunks_mut::<Rayon, _>(part, width, work, |c0, outs| {
        for (ci, out) in (c0..).zip(outs.chunks_mut(width)) {
            let (r0, r1) = (ci * chunk, ((ci + 1) * chunk).min(n));
            for a in 0..off.len() - 1 {
                let v = layers(a);
                let k = (off[a + 1] - off[a]) / 2 - 1;
                let (ua, wa) = (&u[a * n + r0..a * n + r1], &w[a * n + r0..a * n + r1]);
                let o = &mut out[off[a]..off[a + 1]];
                for i in 0..k {
                    (o[i], o[k + i]) = v.dotc2(i * n + r0, ua, wa);
                }
                (o[2 * k], o[2 * k + 1]) = (dotc(ua, ua), dotc(ua, wa));
            }
        }
    });
    fold(part, width, proj);
}

pub(super) struct Delayed<T: Scalar> {
    pub k: usize,
    pub inv_u: f64,
    pub inv_nrm: f64,
    pub h: T,
    pub v: Rows<T>,
    pub z: Option<Rows<T>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn delayed_update<T: Scalar>(
    cols: &[Delayed<T>],
    off: &[usize],
    proj: &[T],
    (u, z, w): (&mut [T], &mut [T], &mut [T]),
    n: usize,
    chunk: usize,
    norms: &mut [f64],
    scratch: &mut [f64],
) {
    let sa = cols.len();
    let nchunks = n.div_ceil(chunk);
    let (up, zp, wp) = (Disjoint::new(u), Disjoint::new(z), Disjoint::new(w));
    let sp = Disjoint::new(&mut scratch[..sa * nchunks]);
    let count = cols
        .iter()
        .map(|c| c.k + DELAYED_UPDATE_PANEL_VECTORS)
        .sum::<usize>()
        * n;
    column_chunks::<T>(sa, n, chunk, count, |a, r0, r1| {
        let d = &cols[a];
        let k = d.k;
        let s = &proj[off[a]..off[a] + k];
        let q = &proj[off[a] + k..off[a] + 2 * k];
        let at = a * n + r0..a * n + r1;
        // SAFETY: piece (a, chunk) alone holds these rows of column `a` (and of its layers
        // `k` and `k + 1`, the layers before it only read).
        let (ua, wa) = unsafe { (up.range(at.clone()), wp.range(at.clone())) };
        for i in 0..k {
            unsafe { d.v.axpy2(i * n + r0, (-s[i], &mut *ua), (-q[i], &mut *wa)) };
        }
        let (su, iu) = (T::from_real(d.inv_u * d.inv_nrm), T::from_real(d.inv_u));
        let mut nn = 0.0;
        for (x, y) in ua.iter_mut().zip(wa.iter_mut()) {
            *x = *x * su;
            *y = *y * iu - d.h * *x;
            nn += y.magnitude_sq();
        }
        unsafe {
            d.v.write(k * n + r0, ua);
            d.v.write((k + 1) * n + r0, wa);
            if let Some(zr) = d.z {
                let za = zp.range(at);
                za.iter_mut().for_each(|e| *e = *e * iu);
                zr.write(k * n + r0, za);
            }
            let c = a * nchunks + r0 / chunk;
            sp.range(c..c + 1)[0] = nn;
        }
    });
    for a in 0..sa {
        norms[a] = scratch[a * nchunks..(a + 1) * nchunks]
            .iter()
            .sum::<f64>()
            .sqrt();
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn panel_project<T: Scalar>(
    v: &[T],
    cols: usize,
    w: &[T],
    q: usize,
    n: usize,
    chunk: usize,
    proj: &mut [T],
    scratch: &mut [T],
) {
    let width = cols * q;
    if width == 0 || n == 0 {
        proj[..width].iter_mut().for_each(|p| *p = T::zero());
        return;
    }
    let nchunks = n.div_ceil(chunk);
    let part = &mut scratch[..nchunks * width];
    let sums = |ci: usize, out: &mut [T]| {
        let r0 = ci * chunk;
        let r1 = (r0 + chunk).min(n);
        for j in 0..cols {
            let vj = &v[j * n + r0..j * n + r1];
            for c in 0..q {
                let wc = &w[c * n + r0..c * n + r1];
                let mut s = T::zero();
                for (a, b) in vj.iter().zip(wc) {
                    s = s + a.conj() * *b;
                }
                out[j * q + c] = s;
            }
        }
    };
    if nchunks == 1 {
        sums(0, part);
    } else {
        let work = crate::parallel::Work::of((n * width) as f64, ORTHO_RATE.of::<T>());
        crate::parallel::chunks_mut::<Rayon, _>(part, width, work, |c0, outs| {
            for (ci, out) in (c0..).zip(outs.chunks_mut(width)) {
                sums(ci, out);
            }
        });
    }
    fold(part, width, proj);
}

#[inline]
pub(super) fn panel_subtract<T: Scalar>(
    v: &[T],
    cols: usize,
    w: &mut [T],
    q: usize,
    n: usize,
    chunk: usize,
    coef: &[T],
) {
    for c in 0..q {
        let wc = &mut w[c * n..(c + 1) * n];
        let update = |ci: usize, part: &mut [T]| {
            let r0 = ci * chunk;
            for j in 0..cols {
                let h = coef[j * q + c];
                if h == T::zero() {
                    continue;
                }
                let vj = &v[j * n + r0..j * n + r0 + part.len()];
                for (x, &y) in part.iter_mut().zip(vj) {
                    *x = *x - h * y;
                }
            }
        };
        if n <= chunk {
            update(0, wc);
        } else {
            let work = crate::parallel::Work::of((n * cols) as f64, ORTHO_RATE.of::<T>());
            crate::parallel::chunks_mut::<Rayon, _>(wc, chunk, work, |c0, parts| {
                for (ci, part) in (c0..).zip(parts.chunks_mut(chunk)) {
                    update(ci, part);
                }
            });
        }
    }
}
