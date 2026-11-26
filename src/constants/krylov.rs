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

/// Panel vectors that the update of a delayed CGS2 step passes over per
/// column besides its basis: the pending vector `u`, the operator's image
/// `w` and the preconditioned vector `z` (vectors).
/// Used by: the work estimate of `delayed_update` in block GMRES.
/// Why: counts the vectors that the update reads and writes.
/// Contract: timing.
pub const DELAYED_UPDATE_PANEL_VECTORS: usize = 3;

/// Recycle bases that GCRO-DR holds beside its Krylov bases: `U`, `P = M^-1 U`
/// and `C = A P`, each `n x k` (vectors per recycle dimension).
/// Used by: `gmres_recycled`, which takes them out of the basis budget before
/// it sizes the default restart.
/// Why: the three bases of the method.
/// Contract: results (the restart length on large systems).
pub const RECYCLE_BASES: usize = 3;

/// Panels of `n x s` scalars that a block GMRES allocates in
/// `BlockGmres::new`: the solutions `x` and `out`, the `panel`, its
/// preconditioned image `z` and the operator's image `w` (panels).
/// Used by: `block_bytes`, the memory a block GMRES is predicted to hold.
/// Why: counts the panels of `BlockGmres::new`.
/// Contract: estimate.
pub const BLOCK_GMRES_PANELS: usize = 5;

/// Index vectors of one entry per right-hand side that a block GMRES holds:
/// `jd`, `off`, `chk` and `closing` (vectors).
/// Used by: `block_bytes`, the memory a block GMRES is predicted to hold.
/// Why: counts the index vectors of `BlockGmres::new`.
/// Contract: estimate.
pub const BLOCK_GMRES_INDEX_VECTORS: usize = 4;
