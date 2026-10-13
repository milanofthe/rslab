//! The numbers of the Python bindings. The solver's own (including the
//! Krylov defaults the bindings expose as `KRYLOV_TOL` and
//! `KRYLOV_MAX_ITER`) are `rslab::constants`.

/// Fields of a `KrylovResult` when unpacked as a tuple: `x`, `converged`,
/// `iters`, `final_res`, `stop` (fields).
/// Used by: `PyKrylovResult::__len__`, `PyKrylovResult::__getitem__`.
/// Why: the documented 5-tuple of the Krylov functions.
/// Contract: format (the Python API).
pub const KRYLOV_RESULT_FIELDS: usize = 5;

/// Bytes per index of the CSC arrays the bindings hold for a factor
/// (`usize` column pointers and row indices), as the memory plan counts
/// them (bytes).
/// Used by: `common::memory_plan_dict`.
/// Why: 64-bit indices on the platforms the wheels are built for.
/// Contract: estimate.
pub const INDEX_BYTES: u64 = 8;

/// Itemsizes of the four scalar fields, which name them in the dtype
/// dispatch (bytes).
/// Used by: `common::scalar_kind`, `with_dtype_name!`.
/// Why: the types' sizes; `complex64` and `float64` share 8 bytes and are
/// told apart by the complex flag.
/// Contract: format.
pub const F64_BYTES: usize = std::mem::size_of::<f64>();

/// See [`F64_BYTES`].
pub const F32_BYTES: usize = std::mem::size_of::<f32>();

/// See [`F64_BYTES`] (`complex128`).
pub const C64_BYTES: usize = std::mem::size_of::<num_complex::Complex<f64>>();

/// See [`F64_BYTES`] (`complex64`).
pub const C32_BYTES: usize = std::mem::size_of::<num_complex::Complex<f32>>();
