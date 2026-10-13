//! Pivoting, matching and refinement of the factor paths, and the routing
//! of the automatic path. Defaults of
//! [`PivotSettings`](crate::PivotSettings),
//! [`MatchingSettings`](crate::MatchingSettings),
//! [`RefinePolicy`](crate::RefinePolicy) and
//! [`AutoSettings`](crate::AutoSettings).

/// Threshold partial pivoting of the LU path, `u` (1): the diagonal stays
/// the pivot unless it falls below `u * |colmax|` in its fully-summed block.
/// Used by: `PivotSettings::threshold`.
/// Why: the customary sparse LU compromise between stability and keeping
/// the analyzed structure (MA48 and UMFPACK use 0.1).
/// Contract: results.
pub const LU_PIVOT_THRESHOLD: f64 = 0.1;

/// A diagonal entry counts as missing below this fraction of its column's
/// largest magnitude; the LU row matching runs when any is missing (1).
/// Used by: `MatchingSettings::negligible_diagonal`.
/// Why: tuned default.
/// Contract: results (whether the matching runs).
pub const MATCHING_NEGLIGIBLE_DIAGONAL: f64 = 1e-10;

/// Most refinement steps of a refined solve (steps).
/// Used by: `RefinePolicy::default`.
/// Why: two steps is what a well-conditioned system needs to reach the
/// roundoff floor.
/// Contract: results (of refined solves).
pub const REFINE_MAX_STEPS: usize = 2;

/// Backward error at which refinement stops (1).
/// Used by: `RefinePolicy::default`.
/// Why: a few units of double roundoff, below which the measure itself is
/// noise.
/// Contract: results (of refined solves).
pub const REFINE_TARGET: f64 = 4.0 * f64::EPSILON;

/// The flops per factor entry, as KLU's analysis predicts them, from which
/// the automatic path takes a pattern as supernodal work (flops per entry).
/// Used by: `AutoSettings::supernodal_flops_per_entry`.
/// Why: below it KLU's refactor is faster on every matrix class measured,
/// above it the supernodal kernels win by a growing factor.
/// Contract: results (the path chosen).
pub const AUTO_SUPERNODAL_FLOPS_PER_ENTRY: f64 = 25.0;

/// Tolerance of the value-symmetry test that selects LDL^T on a
/// structurally symmetric supernodal pattern (1).
/// Used by: `AutoSettings::symmetry_tol`.
/// Why: exact symmetry; LDL^T of a nearly symmetric matrix is opt-in.
/// Contract: results (the path chosen).
pub const AUTO_SYMMETRY_TOL: f64 = 0.0;
