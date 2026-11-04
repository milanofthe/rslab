//! The symmetric trailing-update GEMM of the LDL^T kernel, computed only on and
//! below the diagonal.

use crate::scalar::Scalar;

/// Grow `buf` to at least `len` entries without clearing what it holds: the
/// cmod callers overwrite the prefix they read (the D-apply loops fill `vc` and
/// `vd_buf`, and `lower_tile_gemm` writes `u_buf` with `read_dst = false`), so
/// zeroing it before every update only cost bandwidth.
pub(super) fn grow_scratch<T: Scalar>(buf: &mut Vec<T>, len: usize) {
    if buf.len() < len {
        buf.resize(len, T::zero());
    }
}

/// The largest [`split_plane_entries`](crate::dense::gemm_backend::split_plane_entries)
/// of a [`lower_tile_gemm`] call with these arguments, one per tile.
pub(super) fn lower_tile_planes(
    m: usize,
    ncols: usize,
    k: usize,
    par_cdiv: usize,
    ks: &crate::KernelSettings,
) -> usize {
    let mut worst = 0;
    let mut c0 = 0;
    while c0 < ncols {
        let tw = ks.schur_tile.max(1).min(ncols - c0);
        let mrows = m - c0;
        let par = (mrows as u128) * (tw as u128) * (k as u128) >= par_cdiv as u128;
        worst = worst.max(crate::dense::gemm_backend::split_plane_entries(
            mrows, tw, k, par, ks,
        ));
        c0 += tw;
    }
    worst
}

/// Symmetric trailing-update GEMM computed **only on and below the tile
/// diagonal**: `TMP[:, j] = G * L21^T[:, j]` for rows `>= tile start`. The
/// consumers (the front Schur subtraction and the left-looking panel
/// subtraction) read only entries with `row >= col`, so the full `m x ncols`
/// product wastes up to half the flops (exactly half for the square front
/// Schur, approaching half for wide root panels where `ncols ~ m`). Tiling
/// the columns and starting each tile's rows at its own diagonal keeps the
/// per-element summation deterministic while cutting the waste to
/// `< schur_tile / 2` rows per tile ([`KernelSettings::schur_tile`](crate::KernelSettings::schur_tile)).
///
/// Layouts: `tmp` is `m x ncols` column-major (column stride `m`, row
/// stride 1); `lhs` is `m x k` with column stride `lhs_cs` (row stride 1);
/// `rhs` is read as `k x ncols` with strides `(rhs_cs = 1, rhs_rs)` -
/// element `(kk, j)` at `rhs[j + kk*rhs_rs]`. Each tile's GEMM goes
/// rayon-parallel at/above the `par_cdiv` flop bar.
///
/// SAFETY: the three buffers must be pairwise-disjoint allocations sized
/// for the strides passed (`tmp` >= `m*ncols`; `lhs` rows `[0, m)` x cols
/// `[0, k)` under `lhs_cs`; `rhs` valid at `j + kk*rhs_rs` for `j < ncols`,
/// `kk < k`).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn lower_tile_gemm<T: Scalar>(
    tmp: &mut [T],
    m: usize,
    ncols: usize,
    k: usize,
    lhs: *const T,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    par_cdiv: usize,
    ks: &crate::KernelSettings,
) {
    debug_assert!(ncols <= m);
    debug_assert!(tmp.len() >= m * ncols);
    let mut c0 = 0usize;
    while c0 < ncols {
        let tw = ks.schur_tile.max(1).min(ncols - c0);
        let mrows = m - c0;
        let par = if (mrows as u128) * (tw as u128) * (k as u128) >= par_cdiv as u128 {
            gemm::Parallelism::Rayon(0)
        } else {
            gemm::Parallelism::None
        };
        // Dst tile = columns [c0, c0+tw) rows [c0, m) of `tmp`; lhs = rows
        // [c0, m); rhs = columns [c0, c0+tw).
        crate::dense::gemm_backend::gemm(
            mrows,
            tw,
            k,
            tmp.as_mut_ptr().add(c0 * m + c0),
            m as isize,
            1,
            false,
            lhs.add(c0),
            lhs_cs,
            1,
            rhs.add(c0),
            1,
            rhs_rs,
            T::zero(),
            T::one(),
            false,
            false,
            false,
            crate::dense::gemm_backend::GemmMode::new(par, ks),
        );
        c0 += tw;
    }
}

/// The trailing update of one panel step on and below the diagonal, straight
/// into the panel: `B[r, c] -= G[r, :] R[c, :]^T` for the trailing columns
/// `c` in `[c0, c1)` and the rows `r >= c`, indices relative to the first
/// trailing row and column of the panel. `G` and `R` are the `mt x pw`
/// column-major blocks of the step (leading dimension `mt`); `dst` is the
/// panel entry at the first trailing row of column `c0`, column stride
/// `nrow`.
///
/// The columns go in tiles of `tw`, the tiled form of the `cmod` update, and
/// the rows below a tile's diagonal block in blocks of `rb`: each block is
/// one GEMM into the panel, the diagonal block one GEMM into a buffer of
/// `pool` whose lower triangle is then subtracted, so the panel's upper
/// triangle is never written. With `par` the pieces run on the pool, each a
/// sequential GEMM; a piece computes the same bits either way.
///
/// # Safety
/// `dst` with stride `nrow` must address the `mt` trailing rows of columns
/// `[c0, c1)`, which no one else touches meanwhile; `g` and `r` hold
/// `mt * pw` entries.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn schur_tiles<T: Scalar>(
    dst: *mut T,
    nrow: usize,
    mt: usize,
    pw: usize,
    g: &[T],
    r: &[T],
    (c0, c1): (usize, usize),
    (tw, rb): (usize, usize),
    par: bool,
    pool: &crate::numeric::supernodal::ScratchPool<Vec<T>>,
    ks: &crate::KernelSettings,
) {
    use crate::dense::gemm_backend::{gemm, GemmMode};
    debug_assert!(g.len() >= mt * pw && r.len() >= mt * pw && c1 <= mt);
    let (tw, rb) = (tw.max(1), rb.max(1));
    let tiles = (c1 - c0).div_ceil(tw);
    // Pieces of tile `t`: its diagonal block, then its row blocks below.
    let pieces = |t: usize| 1 + (mt - (c0 + (t + 1) * tw).min(c1)).div_ceil(rb);
    let mut first = Vec::with_capacity(tiles + 1);
    first.push(0usize);
    for t in 0..tiles {
        first.push(first[t] + pieces(t));
    }
    let base = crate::numeric::supernodal::PanelPtr(dst);
    let mode = GemmMode::new(gemm::Parallelism::None, ks);
    let piece = |q: usize| {
        let t = first.partition_point(|&f| f <= q) - 1;
        let a = c0 + t * tw;
        let b = (a + tw).min(c1);
        let w = b - a;
        // SAFETY: column `a` of the caller's range and rows of its trailing
        // block; the operands are the caller's blocks, disjoint from the panel.
        unsafe {
            let col = base.get().add((a - c0) * nrow);
            if q == first[t] {
                let mut diag = pool.take();
                diag.clear();
                diag.resize(w * w, T::zero());
                gemm(
                    w,
                    w,
                    pw,
                    diag.as_mut_ptr(),
                    w as isize,
                    1,
                    false,
                    g.as_ptr().add(a),
                    mt as isize,
                    1,
                    r.as_ptr().add(a),
                    1,
                    mt as isize,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    mode,
                );
                for j in 0..w {
                    for i in j..w {
                        let p = col.add(a + i + j * nrow);
                        *p = *p - diag[i + j * w];
                    }
                }
            } else {
                let r0 = b + (q - first[t] - 1) * rb;
                let r1 = (r0 + rb).min(mt);
                gemm(
                    r1 - r0,
                    w,
                    pw,
                    col.add(r0),
                    nrow as isize,
                    1,
                    true,
                    g.as_ptr().add(r0),
                    mt as isize,
                    1,
                    r.as_ptr().add(a),
                    1,
                    mt as isize,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    mode,
                );
            }
        }
    };
    let total = first[tiles];
    if par && total > 1 {
        crate::parallel::for_each(total, 1, piece);
    } else {
        (0..total).for_each(piece);
    }
}

/// The largest split planes ([`split_plane_entries`](crate::dense::gemm_backend::split_plane_entries))
/// a [`schur_tiles`] call over the columns `[c0, c1)` takes on one worker,
/// and its piece count.
pub(super) fn schur_tile_planes(
    mt: usize,
    (c0, c1): (usize, usize),
    pw: usize,
    (tw, rb): (usize, usize),
    ks: &crate::KernelSettings,
) -> (usize, usize) {
    use crate::dense::gemm_backend::split_plane_entries;
    let (tw, rb) = (tw.max(1), rb.max(1));
    let (mut worst, mut pieces) = (0, 0);
    let mut a = c0;
    while a < c1 {
        let b = (a + tw).min(c1);
        worst = worst
            .max(split_plane_entries(b - a, b - a, pw, false, ks))
            .max(split_plane_entries(rb.min(mt - b), b - a, pw, false, ks));
        pieces += 1 + (mt - b).div_ceil(rb);
        a = b;
    }
    (worst, pieces)
}
