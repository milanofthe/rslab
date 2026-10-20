//! Arithmetic shared by the Krylov solvers: the bilinear and sesquilinear
//! products, Givens rotations, the least-squares rank guard and the batched
//! Gram-Schmidt passes of the block solver.

use crate::numeric::settings::Threads;
use crate::scalar::Scalar;

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
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi.conj() * yi;
    }
    s
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
#[inline]
#[allow(clippy::needless_range_loop)]
pub(super) fn well_conditioned_dim<T: Scalar>(h: &[Vec<T>], jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i][i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i][i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

/// [`well_conditioned_dim`] for a **flat** row-major Hessenberg buffer:
/// the single-RHS [`gmres`] stores `H` as one `(m+1)xm` `Vec<T>` (diagonal entry
/// `i` at `h[i*stride + i]`) rather than a `Vec<Vec<T>>`, so this reads the same
/// breakdown-guarded leading dimension off the flat layout. Identical logic.
#[inline]
pub(super) fn well_conditioned_dim_flat<T: Scalar>(h: &[T], stride: usize, jdim: usize) -> usize {
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

/// The scoped rayon pool for the block-GMRES orthogonalization reductions,
/// from the preconditioner's [`Threads`] policy. `Ambient`
/// returns `None` - the reductions then run on the caller's current pool (the
/// solver-in-the-loop path, where the caller has already installed one bounded
/// rayon pool). Any concrete policy takes the calling thread's pool of that
/// width (the one the factorizations and solves of that width run in, see
/// [`scoped_pool`](crate::numeric::settings::scoped_pool)) **once** per
/// solve; the four `block_project` / `block_subtract` calls per step enter
/// it, so factor and solve share one concurrency budget instead of the solve
/// fanning out over the global pool. The chunk-order reduction fold is
/// thread-count independent, so the pool never perturbs the
/// bit-identical-across-thread-counts guarantee.
pub(super) fn solve_thread_pool(policy: Threads) -> Option<std::sync::Arc<rayon::ThreadPool>> {
    match policy {
        // A lent pool serves every section (see `crate::parallel`).
        _ if crate::parallel::lent().is_some() => None,
        Threads::Ambient => None,
        // `Fixed(k)` (the resolved factor budget): a `k`-worker pool (`0` = all
        // cores). `Auto` never reaches here - factors resolve it to `Fixed` up
        // front - but the `|cap| cap` fallback keeps this total.
        p => crate::numeric::settings::scoped_pool(p.resolve(|cap| cap), 0),
    }
}

/// Run an orthogonalization reduction `f` in the solve pool if there is one,
/// else on the current pool. The pool is entered as the factorizations enter
/// theirs ([`enter`](crate::numeric::settings::enter)): a worker of another
/// pool waits for it without taking its own pool's work. Confined to closures
/// capturing only Krylov *data* (basis / panel slices) - never the operator
/// or preconditioner - so it never imposes a `Send`/`Sync` bound on the
/// matrix-free (`FnOperator`/`FnPreconditioner`) call path.
#[inline]
pub(super) fn ortho_in_pool<R: Send>(
    pool: &Option<std::sync::Arc<rayon::ThreadPool>>,
    f: impl FnOnce() -> R + Send,
) -> R {
    match pool {
        Some(p) => crate::numeric::settings::enter(p, f),
        None => f(),
    }
}

/// Column-wise projection of a panel `W` (`nxsa`) onto each column's **own**
/// Arnoldi basis: `proj[i*sa + ap] = <V_i[:,ap], W[:,ap]>` for block `i` in
/// `0..blocks` and active column `ap` in `0..sa`. The basis is blocks-major -
/// block `i` is the contiguous slice `vbas[i*sa*n .. (i+1)*sa*n]`, its column `ap`
/// at offset `+ap*n`; `W` column `ap` is `w[ap*n .. ap*n+n]`.
///
/// This is the classical (block) Gram-Schmidt projection: **all** projections are
/// taken against the same `W`, so the `blocks*sa` inner products are independent
/// and computed as one panel sweep instead of the `O(blocks*sa)` sequential,
/// latency-bound BLAS-1 reductions of modified Gram-Schmidt. The reduction is a
/// fixed row-chunk sum folded in chunk order -> deterministic regardless of the
/// thread count.
/// `scratch` is a caller-owned reduction buffer of length `>= nchunks * width`
/// (`nchunks = ceil(n / chunk)`, `width = blocks*sa`), reused across steps so the
/// hot loop allocates nothing. Each chunk writes its `width` partial sums into its
/// own slice; the slices are then folded in chunk order.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn block_project<T: Scalar>(
    vbas: &[T],
    w: &[T],
    blocks: usize,
    sa: usize,
    n: usize,
    chunk: usize,
    proj: &mut [T],
    scratch: &mut [T],
) {
    let width = blocks * sa;
    for p in proj[..width].iter_mut() {
        *p = T::zero();
    }
    if width == 0 || n == 0 {
        return;
    }
    let nchunks = n.div_ceil(chunk);
    let part = &mut scratch[..nchunks * width];
    crate::parallel::chunks_mut(part, width, |ci, out| {
        let r0 = ci * chunk;
        let r1 = (r0 + chunk).min(n);
        for i in 0..blocks {
            for ap in 0..sa {
                let vb = (i * sa + ap) * n;
                let wb = ap * n;
                let mut sdot = T::zero();
                for k in r0..r1 {
                    sdot = sdot + vbas[vb + k].conj() * w[wb + k];
                }
                out[i * sa + ap] = sdot;
            }
        }
    });
    // Fold partials in chunk order: the summation order is fixed by the chunk
    // layout, so the result does not depend on how many threads ran.
    for ci in 0..nchunks {
        let base = ci * width;
        for t in 0..width {
            proj[t] = proj[t] + part[base + t];
        }
    }
}

/// Subtract the projected components from the panel in place, per column:
/// `W[:,ap] -= sum_i proj[i*sa+ap] * V_i[:,ap]`, accumulated in block order (`i`
/// ascending) at every element. Parallel over fixed row-chunks within each column
/// -> the element-wise order is fixed, so the update is deterministic.
#[inline]
pub(super) fn block_subtract<T: Scalar>(
    vbas: &[T],
    w: &mut [T],
    blocks: usize,
    sa: usize,
    n: usize,
    chunk: usize,
    proj: &[T],
) {
    for ap in 0..sa {
        let wcol = &mut w[ap * n..ap * n + n];
        crate::parallel::chunks_mut(wcol, chunk, |ci, wc| {
            let r0 = ci * chunk;
            for i in 0..blocks {
                let hij = proj[i * sa + ap];
                if hij == T::zero() {
                    continue;
                }
                let vb = (i * sa + ap) * n + r0;
                for k in 0..wc.len() {
                    wc[k] = wc[k] - hij * vbas[vb + k];
                }
            }
        });
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
    for p in proj[..width].iter_mut() {
        *p = T::zero();
    }
    if width == 0 || n == 0 {
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
        crate::parallel::chunks_mut(part, width, sums);
    }
    for ci in 0..nchunks {
        let base = ci * width;
        for t in 0..width {
            proj[t] = proj[t] + part[base + t];
        }
    }
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
            crate::parallel::chunks_mut(wc, chunk, update);
        }
    }
}
