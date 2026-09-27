//! The system CBLAS behind the large dense products (cfg `rslab_blas`, set
//! by `build.rs`): Accelerate on macOS, or the library `RSLAB_BLAS_LIB`
//! names elsewhere (MKL, OpenBLAS).
//!
//! [`gemm`] takes a product that reaches
//! [`KernelSettings::blas_min_flops`](crate::KernelSettings::blas_min_flops)
//! when its layout maps onto CBLAS (unit stride along one side of every
//! operand, no conjugated destination, no conjugate without transpose) and
//! the library's threading can be held to the calling thread. Each call runs
//! single-threaded on its worker. The product is cut into blocks of
//! [`blas_par_block`](crate::KernelSettings::blas_par_block) columns (or rows,
//! whichever side is longer), which rayon spreads when the kernel marks the
//! product parallel; the cut does not depend on the thread count, and with
//! it neither does the result.

use super::gemm_backend::GemmMode;
use crate::scalar::Scalar;
use rayon::prelude::*;

/// CBLAS enumerators (ABI constants).
pub(crate) const COL_MAJOR: i32 = 102;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;
const CONJ_TRANS: i32 = 113;

/// One operand as the kernels pass it: element `(i, j)` at `p + i rs + j cs`.
#[derive(Clone, Copy)]
struct Operand<T> {
    p: *const T,
    cs: isize,
    rs: isize,
    conj: bool,
}

impl<T> Operand<T> {
    fn transposed(self) -> Self {
        Self {
            cs: self.rs,
            rs: self.cs,
            ..self
        }
    }

    /// CBLAS transpose flag and leading dimension of this `rows x cols`
    /// operand, if its layout maps.
    fn cblas(self, rows: usize, cols: usize) -> Option<(i32, i32)> {
        if self.rs == 1 && !self.conj && (cols <= 1 || self.cs >= rows as isize) {
            Some((NO_TRANS, ld(self.cs.max(rows as isize))?))
        } else if self.cs == 1 && (rows <= 1 || self.rs >= cols as isize) {
            let flag = if self.conj { CONJ_TRANS } else { TRANS };
            Some((flag, ld(self.rs.max(cols as isize))?))
        } else {
            None
        }
    }

    /// The operand from element `(i, j)` on.
    ///
    /// # Safety
    /// `(i, j)` must lie inside the operand.
    unsafe fn from(self, i: usize, j: usize) -> Self {
        Self {
            p: self.p.offset(i as isize * self.rs + j as isize * self.cs),
            ..self
        }
    }
}

fn ld(x: isize) -> Option<i32> {
    i32::try_from(x.max(1)).ok()
}

/// Shares the raw pointers of one product with the rayon workers; the
/// blocks they write are disjoint.
struct Shared<T>(T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T: Copy> Shared<T> {
    // A method, so a closure captures the whole wrapper rather than its
    // (non-`Sync`) fields.
    fn get(&self) -> T {
        self.0
    }
}

/// The product of [`gemm_backend::gemm`](super::gemm_backend::gemm) on the
/// system BLAS; `false` (nothing written) leaves it to the portable kernels.
///
/// # Safety
/// As `gemm::gemm`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm<T: Scalar>(
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
    mode: GemmMode,
) -> bool {
    if !T::HAS_CBLAS || conj_dst || m == 0 || n == 0 || k == 0 {
        return false;
    }
    if (m as u128) * (n as u128) * (k as u128) < mode.blas_min_flops as u128
        || i32::try_from(m.max(n).max(k)).is_err()
        || !threading::available()
    {
        return false;
    }
    let a = Operand {
        p: lhs,
        cs: lhs_cs,
        rs: lhs_rs,
        conj: conj_lhs,
    };
    let b = Operand {
        p: rhs,
        cs: rhs_cs,
        rs: rhs_rs,
        conj: conj_rhs,
    };
    // The destination column-major as it is, or row-major as C^T = B^T A^T.
    let (m, n, a, b, ldc) = if dst_rs == 1 && (n <= 1 || dst_cs >= m as isize) {
        (m, n, a, b, ld(dst_cs.max(m as isize)))
    } else if dst_cs == 1 && (m <= 1 || dst_rs >= n as isize) {
        (
            n,
            m,
            b.transposed(),
            a.transposed(),
            ld(dst_rs.max(n as isize)),
        )
    } else {
        return false;
    };
    let Some(ldc) = ldc else { return false };
    if a.cblas(m, k).is_none() || b.cblas(k, n).is_none() {
        return false;
    }
    let c_beta = if read_dst { alpha } else { T::zero() };
    let block = mode.blas_par_block.max(1);
    let by_columns = n >= m;
    let len = if by_columns { n } else { m };
    let shared = Shared((a, b, dst));
    let run = |t: usize| {
        let (a, b, dst) = shared.get();
        let (s, e) = (t * block, ((t + 1) * block).min(len));
        let (mm, nn, a, b, c) = if by_columns {
            (m, e - s, a, b.from(0, s), dst.add(s * ldc as usize))
        } else {
            (e - s, n, a.from(s, 0), b, dst.add(s))
        };
        let (Some((ta, lda)), Some((tb, ldb))) = (a.cblas(mm, k), b.cblas(k, nn)) else {
            unreachable!("a block keeps the layout of its product")
        };
        threading::single(|| {
            T::cblas_gemm(
                ta, tb, mm as i32, nn as i32, k as i32, beta, a.p, lda, b.p, ldb, c_beta, c, ldc,
            )
        });
    };
    let blocks = len.div_ceil(block);
    if blocks > 1 && matches!(mode.parallelism, gemm::Parallelism::Rayon(_)) {
        (0..blocks).into_par_iter().for_each(run);
    } else {
        (0..blocks).for_each(run);
    }
    true
}

/// Whether the system BLAS is in use: linked, and its threading held to the
/// calling thread.
pub(crate) fn available() -> bool {
    threading::available()
}

/// Holds the library's threading to the calling thread for one call.
mod threading {
    use std::sync::OnceLock;

    #[derive(Clone, Copy)]
    enum Control {
        /// Accelerate (macOS 15): per-thread, `1` is single-threaded.
        Accelerate {
            get: unsafe extern "C" fn() -> u32,
            set: unsafe extern "C" fn(u32) -> i32,
        },
        /// MKL: the per-thread count, which returns the previous one (`0`
        /// stands for the global count).
        Mkl(unsafe extern "C" fn(i32) -> i32),
        /// OpenBLAS has a process-wide count only; it is set to one once.
        Global,
    }

    fn control() -> Option<Control> {
        static CONTROL: OnceLock<Option<Control>> = OnceLock::new();
        *CONTROL.get_or_init(|| {
            // SAFETY: each symbol is transmuted to the signature its library
            // documents, and only when it resolved.
            unsafe {
                let (get, set) = (symbol(b"BLASGetThreading\0"), symbol(b"BLASSetThreading\0"));
                if !get.is_null() && !set.is_null() {
                    return Some(Control::Accelerate {
                        get: std::mem::transmute::<*mut (), unsafe extern "C" fn() -> u32>(get),
                        set: std::mem::transmute::<*mut (), unsafe extern "C" fn(u32) -> i32>(set),
                    });
                }
                // The mixed-case name is MKL's C entry on every platform; the
                // lower-case one is the Fortran entry (by reference) on Windows.
                let mkl = symbol(b"MKL_Set_Num_Threads_Local\0");
                if !mkl.is_null() {
                    return Some(Control::Mkl(std::mem::transmute::<
                        *mut (),
                        unsafe extern "C" fn(i32) -> i32,
                    >(mkl)));
                }
                let openblas = symbol(b"openblas_set_num_threads\0");
                if !openblas.is_null() {
                    std::mem::transmute::<*mut (), unsafe extern "C" fn(i32)>(openblas)(1);
                    return Some(Control::Global);
                }
                None
            }
        })
    }

    pub(super) fn available() -> bool {
        control().is_some()
    }

    /// Run `f` with the library single-threaded on this thread.
    ///
    /// # Safety
    /// `f` must be sound to call.
    pub(super) unsafe fn single<R>(f: impl FnOnce() -> R) -> R {
        match control() {
            Some(Control::Accelerate { get, set }) => {
                let previous = get();
                set(1);
                let out = f();
                set(previous);
                out
            }
            Some(Control::Mkl(set)) => {
                let previous = set(1);
                let out = f();
                set(previous);
                out
            }
            _ => f(),
        }
    }

    /// Address of `name` (null-terminated) in an image already loaded into
    /// the process, or null.
    #[cfg(unix)]
    unsafe fn symbol(name: &[u8]) -> *mut () {
        #[cfg(target_vendor = "apple")]
        const RTLD_DEFAULT: *mut () = -2isize as *mut ();
        #[cfg(not(target_vendor = "apple"))]
        const RTLD_DEFAULT: *mut () = std::ptr::null_mut();
        extern "C" {
            fn dlsym(handle: *mut (), symbol: *const u8) -> *mut ();
        }
        dlsym(RTLD_DEFAULT, name.as_ptr())
    }

    /// Windows has no `RTLD_DEFAULT`: each loaded module is asked in turn.
    #[cfg(windows)]
    unsafe fn symbol(name: &[u8]) -> *mut () {
        extern "system" {
            fn GetCurrentProcess() -> *mut ();
            fn K32EnumProcessModules(
                process: *mut (),
                modules: *mut *mut (),
                bytes: u32,
                needed: *mut u32,
            ) -> i32;
            fn GetProcAddress(module: *mut (), name: *const u8) -> *mut ();
        }
        let mut modules = [std::ptr::null_mut(); 1024];
        let mut needed = 0u32;
        if K32EnumProcessModules(
            GetCurrentProcess(),
            modules.as_mut_ptr(),
            std::mem::size_of_val(&modules) as u32,
            &mut needed,
        ) == 0
        {
            return std::ptr::null_mut();
        }
        let count = (needed as usize / std::mem::size_of::<*mut ()>()).min(modules.len());
        modules[..count]
            .iter()
            .map(|&module| GetProcAddress(module, name.as_ptr()))
            .find(|symbol| !symbol.is_null())
            .unwrap_or(std::ptr::null_mut())
    }
}

/// The CBLAS entries, resolved in the library `build.rs` links.
#[allow(clippy::too_many_arguments)]
pub(crate) mod ffi {
    use std::ffi::c_void;
    extern "C" {
        pub fn cblas_sgemm(
            layout: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        pub fn cblas_dgemm(
            layout: i32,
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
        pub fn cblas_cgemm(
            layout: i32,
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
        pub fn cblas_zgemm(
            layout: i32,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelSettings;
    use num_complex::Complex64;

    /// Every layout the kernels pass, against the portable product.
    #[test]
    fn matches_the_portable_kernels() {
        if !available() {
            return;
        }
        let ks = KernelSettings {
            blas_min_flops: 0,
            blas_par_block: 7,
            ..KernelSettings::default()
        };
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 2_000_000) as f64 / 1e6 - 1.0
        };
        let (m, n, k) = (23, 19, 11);
        let lhs: Vec<Complex64> = (0..m * k).map(|_| Complex64::new(next(), next())).collect();
        let rhs: Vec<Complex64> = (0..k * n).map(|_| Complex64::new(next(), next())).collect();
        let dst0: Vec<Complex64> = (0..m * n).map(|_| Complex64::new(next(), next())).collect();
        let (alpha, beta) = (Complex64::new(0.5, -1.0), Complex64::new(-1.0, 0.25));
        for (lcs, lrs) in [(m as isize, 1), (1, k as isize)] {
            for (dcs, drs) in [(m as isize, 1), (1, n as isize)] {
                for conj_lhs in [false, true] {
                    for par in [gemm::Parallelism::None, gemm::Parallelism::Rayon(0)] {
                        let mode = GemmMode::new(par, &ks);
                        let (mut want, mut got) = (dst0.clone(), dst0.clone());
                        unsafe {
                            gemm::gemm(
                                m,
                                n,
                                k,
                                want.as_mut_ptr(),
                                dcs,
                                drs,
                                true,
                                lhs.as_ptr(),
                                lcs,
                                lrs,
                                rhs.as_ptr(),
                                k as isize,
                                1,
                                alpha,
                                beta,
                                false,
                                conj_lhs,
                                false,
                                gemm::Parallelism::None,
                            );
                            let taken = super::gemm(
                                m,
                                n,
                                k,
                                got.as_mut_ptr(),
                                dcs,
                                drs,
                                true,
                                lhs.as_ptr(),
                                lcs,
                                lrs,
                                rhs.as_ptr(),
                                k as isize,
                                1,
                                alpha,
                                beta,
                                false,
                                conj_lhs,
                                false,
                                mode,
                            );
                            // A conjugate without transpose has no CBLAS form.
                            assert!(taken || conj_lhs);
                            if !taken {
                                continue;
                            }
                        }
                        for (w, g) in want.iter().zip(&got) {
                            assert!((w - g).norm() < 1e-12, "{w} vs {g}");
                        }
                    }
                }
            }
        }
    }
}
