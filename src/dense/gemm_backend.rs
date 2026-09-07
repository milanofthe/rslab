//! The one GEMM entry of the numeric kernels.
//!
//! Every trailing update goes through [`gemm`], which has the signature and
//! semantics of the `gemm` crate (`dst := alpha * dst + beta * lhs * rhs`,
//! arbitrary strides). By default that is where it goes: the pure-Rust SIMD
//! kernels, bit-identical everywhere. With the `accelerate` feature on Apple
//! platforms the call is routed to `cblas_dgemm` / `cblas_zgemm` of the
//! Accelerate framework when the strides describe a column- or row-major
//! operand (which every kernel of this crate produces), so the update runs on
//! the AMX matrix units: several times the single-core throughput of NEON.
//! Results then differ from the pure-Rust build in the last bits (a different
//! summation order), which is why the feature is opt-in.

use crate::scalar::Scalar;

/// See the module docs; the arguments are those of `gemm::gemm`.
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
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    {
        accelerate::single_threaded_blas();
        if !conj_dst
            && !conj_lhs
            && !conj_rhs
            && T::accelerate_gemm(
                m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs,
                alpha, beta,
            )
        {
            return;
        }
    }
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

/// The Accelerate BLAS binding (Apple platforms only).
#[cfg(all(feature = "accelerate", target_vendor = "apple"))]
pub mod accelerate {
    use std::ffi::c_void;

    pub const COL_MAJOR: i32 = 102;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        pub fn cblas_dgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );
        pub fn cblas_zgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *const c_void,
            ldb: i32,
            beta: *const c_void,
            c: *mut c_void,
            ldc: i32,
        );
    }

    /// rslab schedules its own parallelism over the elimination tree, so the
    /// BLAS must not add threads of its own underneath (they oversubscribe
    /// the cores: 129 ms against 118 ms on a 98k FEM factorization with 8
    /// workers). Accelerate reads `VECLIB_MAXIMUM_THREADS` on first use;
    /// this sets it to 1 once, unless the user set it explicitly.
    pub fn single_threaded_blas() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            if std::env::var_os("VECLIB_MAXIMUM_THREADS").is_none() {
                std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
            }
        });
    }

    /// Column-major operand description of a strided matrix: `(trans, ld)`
    /// or `None` when neither stride is 1 (not expressible for BLAS).
    /// `rows x cols` is the logical shape of the operand as used.
    pub fn operand(cs: isize, rs: isize, rows: usize, cols: usize) -> Option<(i32, i32)> {
        if rs == 1 && cs >= rows as isize {
            Some((NO_TRANS, cs as i32))
        } else if cs == 1 && rs >= cols as isize {
            Some((TRANS, rs as i32))
        } else {
            None
        }
    }

    /// Whether the sizes fit the 32-bit BLAS interface.
    pub fn fits(m: usize, n: usize, k: usize) -> bool {
        m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize
    }
}
