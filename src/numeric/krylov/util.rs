//! Arithmetic shared by the Krylov solvers: the bilinear and sesquilinear
//! products, Givens rotations, the least-squares rank guard and the batched
//! Gram-Schmidt passes of the block solvers.

use crate::parallel::SendPtr;
use crate::scalar::Scalar;
use crate::stored::{axpy_as, dotc_as, Rows, Stored};

/// A Krylov basis of up to `cap` scalars: the capacity reserved (address space alone), its
/// length grown by [`grow`] as the iteration writes it, so a solve that stops after `k` steps
/// holds `k` basis vectors rather than the restart's `m`. `vec![T::zero(); cap]` writes every
/// scalar up front and makes the whole budgeted basis resident (a 79 k unknown, 23 column MoM
/// block solve: about 8 GB of its 26 GB peak, where 89 steps used 2.6 GB).
pub(super) fn basis<T: Scalar>(cap: usize) -> Vec<T> {
    Vec::with_capacity(cap)
}

/// Grow the basis `v` ([`basis`]) to at least `len` scalars, zero filled; inside the reserved
/// capacity nothing moves.
#[inline]
pub(super) fn grow<T: Scalar>(v: &mut Vec<T>, len: usize) {
    if v.len() < len {
        v.resize(len, T::zero());
    }
}

/// Unconjugated bilinear inner product `x^T y = sum x_i y_i` (no complex conjugation -
/// the defining choice of the COCG geometry for `A = A^T`).
#[inline]
pub(super) fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi * yi;
    }
    s
}

/// Euclidean norm `||x||_2 = sqrt(sum |x_i|^2)` (genuine modulus - used only for the stopping
/// test, never inside the Krylov recurrences).
#[inline]
pub(super) fn norm2<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.magnitude_sq()).sum::<f64>().sqrt()
}

/// Conjugated (Hermitian) inner product `<x, y> = sum conj(x_i)*y_i` - the geometry
/// GMRES orthogonalises in (distinct from COCG's unconjugated form).
#[inline]
pub(super) fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    dotc_as(x, y, |v| v)
}

/// `out += a x`, entry by entry.
#[inline]
pub(super) fn axpy<T: Scalar>(x: &[T], a: T, out: &mut [T]) {
    axpy_as(x, a, out, |v| v)
}

/// Complex Givens rotation `(c, s)` that zeroes `g` against `f`: with the
/// rotation `[[conj(c), conj(s)], [-s, c]]`, `conj(c)*f + conj(s)*g = r` (real)
/// and `-s*f + c*g = 0`, `|c|^2+|s|^2 = 1`.
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

/// Largest leading dimension `d <= jdim` whose Hessenberg diagonals are all above a
/// relative breakdown threshold `eps * max_i |h[i][i]|`. The upper-triangular
/// solve for `y` back-substitutes with `1/h[i][i]`; after Givens the diagonal is
/// `sqrt(|f|^2+|g|^2)` and normally nonzero, but under exact stagnation or a rank-
/// deficient Hessenberg (hard / indefinite / singular operators) some `h[i][i]`
/// can be `0`, so an unguarded `recip()` would emit `Inf`/`NaN` into `x` and the
/// residual. Truncating the solve to this well-conditioned prefix instead yields a
/// **deterministic breakdown**: the leading Krylov block is solved, the degenerate
/// tail is dropped, and the outer true-residual check reports non-convergence. In
/// the well-conditioned case every diagonal clears the threshold and `jdim` is
/// returned unchanged, so the normal path is unaffected.
///
/// The Hessenberg matrix is one row-major `(m+1) x m` buffer, diagonal entry `i`
/// at `h[i*stride + i]`.
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

/// What a multiply-add of the orthogonalization costs a thread.
static ORTHO_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::SCAN_NS_PER_ENTRY);

/// `f(a, r0, r1)` for the rows `r0..r1` of every row chunk of `chunk` rows of every one of
/// `cols` columns of `n` rows, as one parallel section of `count` multiply-adds.
pub(super) fn column_chunks<T: Scalar>(
    cols: usize,
    n: usize,
    chunk: usize,
    count: usize,
    f: impl Fn(usize, usize, usize) + Sync + Send,
) {
    let nchunks = n.div_ceil(chunk);
    let work = crate::parallel::Work::of(count as f64, ORTHO_RATE.of::<T>());
    crate::parallel::for_each(cols * nchunks, work, |p| {
        let (a, ci) = (p / nchunks, p % nchunks);
        f(a, ci * chunk, ((ci + 1) * chunk).min(n));
    });
}

/// `out[t] = sum_ci part[ci * width + t]` over the `part.len() / width` row chunks in chunk
/// order: the partial sums of a reduction folded, in parallel over `t`, so the result does not
/// depend on the thread count.
fn fold<T: Scalar>(part: &[T], width: usize, out: &mut [T]) {
    let nchunks = part.len() / width;
    let work = crate::parallel::Work::of((nchunks * width) as f64, ORTHO_RATE.of::<T>());
    let op = SendPtr::of(&mut out[..width]);
    crate::parallel::for_each(width, work, |t| {
        let sum = (0..nchunks).fold(T::zero(), |e, ci| e + part[ci * width + t]);
        // SAFETY: item `t` alone writes `out[t]`.
        unsafe { op.part(t, 1)[0] = sum };
    });
}

/// The one pass over the bases of a delayed CGS2 step (Bielich, Langou, Thomas, Swirydowicz,
/// Yamazaki, Boman 2022): panel column `a` holds its pending vector `u` (in `u`) and the
/// operator's image of it (in `w`), its basis `layers(a)` the `k` finished vectors `V` before
/// `u`, `2 k + 2 = off[a+1] - off[a]`. Into `proj[off[a]..off[a+1]]`: `V^H u` (the delayed
/// second pass of `u`), `V^H w` (the first pass of `w`), `u^H u` and `u^H w`; each basis
/// vector read once for both.
///
/// The products of all columns are independent and summed over row chunks in one section,
/// each chunk into its own slice of `scratch` (`>= ceil(n / chunk) * off[last]`), folded in
/// chunk order: the result does not depend on the thread count.
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
    if width == 0 || n == 0 {
        proj[..width].iter_mut().for_each(|p| *p = T::zero());
        return;
    }
    let nchunks = n.div_ceil(chunk);
    let part = &mut scratch[..nchunks * width];
    let work = crate::parallel::Work::of((n * width) as f64, ORTHO_RATE.of::<T>());
    crate::parallel::chunks_mut(part, width, work, |ci, out| {
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
    });
    fold(part, width, proj);
}

/// What the update of a delayed CGS2 step ([`delayed_update`]) does to one panel column:
/// with `s = proj[off..off+k]` and `q = proj[off+k..off+2k]` of [`delayed_project`], its
/// pending vector `u` becomes the next basis vector `v = (u - V s) inv_u inv_nrm` (layer `k`
/// of `v`), `w` becomes `(w - V q) inv_u - h v` (layer `k + 1`), and the preconditioned
/// vector `z` (scaled by `inv_u`) goes to layer `k` of `z` where kept.
pub(super) struct Delayed<T: Scalar> {
    pub k: usize,
    pub inv_u: f64,
    pub inv_nrm: f64,
    pub h: T,
    pub v: Rows<T>,
    pub z: Option<Rows<T>>,
}

/// The one update pass of a delayed CGS2 step over the bases, every panel column `a` as
/// `cols[a]` says ([`Delayed`]), each basis vector read once for `u` and `w`; the written
/// layers stored in the same pass. `norms[a]` the norm of the new `w`, summed over row
/// chunks in chunk order (`scratch >= cols.len() * ceil(n / chunk)`).
///
/// # Safety
/// Every column's `v` and `z` ([`Stored::rows`]) reach their layers `k + 1` and `k`, and no
/// two columns share them.
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
    let (up, zp, wp) = (SendPtr::of(u), SendPtr::of(z), SendPtr::of(w));
    let sp = SendPtr::of(&mut scratch[..sa * nchunks]);
    // the basis read once and the three vectors of the panel (`u`, `w`, `z`) per column
    let count = cols.iter().map(|c| c.k + 3).sum::<usize>() * n; // lit: three panel vectors
    column_chunks::<T>(sa, n, chunk, count, |a, r0, r1| {
        let d = &cols[a];
        let (k, len) = (d.k, r1 - r0);
        let s = &proj[off[a]..off[a] + k];
        let q = &proj[off[a] + k..off[a] + 2 * k];
        // SAFETY: piece (a, chunk) alone holds these rows of column `a` (and of its layers
        // `k` and `k + 1`, the layers before it only read).
        let (ua, wa) = unsafe { (up.part(a * n + r0, len), wp.part(a * n + r0, len)) };
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
                let za = zp.part(a * n + r0, len);
                za.iter_mut().for_each(|e| *e = *e * iu);
                zr.write(k * n + r0, za);
            }
            sp.part(a * nchunks + r0 / chunk, 1)[0] = nn;
        }
    });
    for a in 0..sa {
        norms[a] = scratch[a * nchunks..(a + 1) * nchunks]
            .iter()
            .sum::<f64>()
            .sqrt();
    }
}

/// The panel product `proj[j * q + c] = <V_j, W_c>` of the first `cols` columns
/// of the basis `v` (column-major, `n` rows) and the `q` columns of the panel
/// `w`: the basis read once for the whole panel. Row chunks of `chunk` rows are
/// summed in parallel into `scratch` (`>= ceil(n / chunk) * cols * q`) and folded
/// in chunk order, so the result does not depend on the thread count.
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
    // one chunk: no parallel section (nothing to spread, nothing to allocate)
    if nchunks == 1 {
        sums(0, part);
    } else {
        let work = crate::parallel::Work::of((n * width) as f64, ORTHO_RATE.of::<T>());
        crate::parallel::chunks_mut(part, width, work, sums);
    }
    fold(part, width, proj);
}

/// `W_c -= sum_j coef[j * q + c] V_j` over the first `cols` basis columns, in
/// parallel over row chunks, the sum in column order at every entry (thread-count
/// independent).
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
            crate::parallel::chunks_mut(wc, chunk, work, update);
        }
    }
}
