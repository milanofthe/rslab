//! Equilibration and matching: the scalings of the factor paths.

/// Upper bound on the argument to `exp` in the MC64 scaling factors (1):
/// `ln(f64::MAX) ~ 709.78`, so `exp` of at most this stays finite.
/// Used by: `scaling::mc64`.
/// Why: the largest integer below the overflow bound.
/// Contract: results (clamps extreme scalings).
pub const LOG_HUGE: f64 = 709.0;
