//! The one GEMM entry of the numeric kernels.
//!
//! Every trailing update goes through [`gemm`], which has the signature and
//! semantics of the `gemm` crate (`dst := alpha * dst + beta * lhs * rhs`,
//! arbitrary strides): the pure-Rust SIMD kernels, the same on every
//! platform, so a factorization is bit-identical wherever it runs. One entry
//! point keeps the kernels' calling convention in one place.

use crate::scalar::Scalar;

/// See the module docs; the arguments are those of `gemm::gemm`.
///
/// # Safety
/// The pointers and strides must describe valid, non-overlapping matrices
/// of the given sizes (the contract of `gemm::gemm`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn gemm<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
    conj_dst: bool,
    conj_lhs: bool,
    conj_rhs: bool,
    parallelism: gemm::Parallelism,
) {
    gemm::gemm(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        read_dst,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        alpha,
        beta,
        conj_dst,
        conj_lhs,
        conj_rhs,
        parallelism,
    )
}
