//! The Krylov solvers. Defaults of [`KrylovSettings`](crate::KrylovSettings),
//! and of the Python bindings' Krylov functions.

/// Target of the relative residual `||b - A x|| / ||b||`, per column (1).
/// Used by: `KrylovSettings::tol`, the Python `gmres`, `gmres_block`,
/// `cocg`, `cocr` and their factor methods.
/// Why: tuned default, well above the roundoff of a double iteration.
/// Contract: results.
pub const KRYLOV_TOL: f64 = 1e-8;

/// Iteration budget per solve (iterations).
/// Used by: `KrylovSettings::max_iter`, the Python Krylov functions as
/// `maxit`.
/// Why: tuned default; bounds an iteration that stagnates.
/// Contract: results.
pub const KRYLOV_MAX_ITER: usize = 400;

/// Memory cap of the Krylov bases for the default restart length (bytes).
/// Used by: `KrylovSettings::basis_budget_bytes`.
/// Why: bounds the bases at 1 GiB, a share of a workstation's memory.
/// Contract: results (the restart length on large systems).
pub const KRYLOV_BASIS_BUDGET_BYTES: usize = 1 << 30;

/// Shortest default GMRES restart length (vectors).
/// Used by: `KrylovSettings::restart_min`.
/// Why: tuned default.
/// Contract: results.
pub const KRYLOV_RESTART_MIN: usize = 20;

/// Longest default GMRES restart length (vectors).
/// Used by: `KrylovSettings::restart_max`.
/// Why: tuned default.
/// Contract: results.
pub const KRYLOV_RESTART_MAX: usize = 80;

/// A new basis vector is orthogonalized a second time when the first pass
/// keeps less than this fraction of its norm (1).
/// Used by: `KrylovSettings::reorth_eta`.
/// Why: the DGKS criterion `1/sqrt(2)`.
/// Contract: results.
pub const KRYLOV_REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Rows per chunk of the parallel orthogonalization sums of block GMRES
/// (rows). The chunks are summed in a fixed order.
/// Used by: `KrylovSettings::ortho_chunk`.
/// Why: tuned default.
/// Contract: results (the summation order; not the thread count).
pub const KRYLOV_ORTHO_CHUNK: usize = 2048;
