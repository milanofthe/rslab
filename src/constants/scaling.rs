//! Equilibration and matching: the scalings of the factor paths.

/// Upper bound on the argument to `exp` in the MC64 scaling factors (1):
/// `ln(f64::MAX) ~ 709.78`, so `exp` of at most this stays finite.
/// Used by: `scaling::mc64`.
/// Why: the largest integer below the overflow bound.
/// Contract: results (clamps extreme scalings).
pub const LOG_HUGE: f64 = 709.0;

/// Sweeps of the iterative infinity-norm equilibration at most (sweeps).
/// Used by: `scaling::infnorm::compute_infnorm`.
/// Why: most matrices converge in 2 to 4 sweeps; a few pathological ones
/// need all 10.
/// Contract: results (the scaling).
pub const INFNORM_MAX_ITER: usize = 10;

/// The equilibration stops once every row and column norm is within this of
/// 1 (1).
/// Used by: `scaling::infnorm::compute_infnorm`.
/// Why: tuned default; the factorization needs the scale, not the digits.
/// Contract: results.
pub const INFNORM_TOL: f64 = 1e-8;

/// The greedy initialization of the Hungarian matching skips columns with
/// more than `m / HUNGARIAN_DENSE_COL_DIV` entries (1) ...
/// Used by: `scaling::hungarian::hungarian_init_heuristic`.
/// Why: SPRAL's guard against assigning on dense columns.
/// Contract: results (the initial matching, and so ties of the final one).
pub const HUNGARIAN_DENSE_COL_DIV: usize = 10;

/// ... on matrices of more than this many rows (rows).
/// Used by: `scaling::hungarian::hungarian_init_heuristic`.
/// Why: on small matrices every column is "dense" by that measure.
/// Contract: results.
pub const HUNGARIAN_DENSE_MIN_ROWS: usize = 50;
