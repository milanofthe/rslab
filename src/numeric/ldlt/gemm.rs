//! The symmetric trailing-update GEMM of the LDL^T kernel, computed only on and
//! below the diagonal.

use crate::scalar::Scalar;

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

// https://github.com/milanofthe/rslab/issues/288
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
