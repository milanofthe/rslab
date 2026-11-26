//! The numbers the minimum-degree orderings (AMD, AMF) are defined by,
//! shared by `rslab-amd` and `rslab-amf` so the two cannot drift apart, and
//! the resolution of the progress every ordering reports.
//!
//! Each constant states its unit, what reads it, why it has its value, and
//! its contract: **results** (changing it changes the permutation).

/// Dense-row threshold multiplier (1): a variable with initial degree above
/// `min(max(16, floor(DENSE_ALPHA * sqrt(n))), n)` is deferred
/// to the end of the ordering.
/// Used by: `AmdOptions`, `AmfOptions` and `WorkspaceOptions` defaults.
/// Why: SuiteSparse AMD's default (Davis 1996, section 5).
/// Contract: results.
pub const DENSE_ALPHA: f64 = 10.0;

/// Floor of the dense-row threshold, applied before the `min(n)` cap so the
/// threshold stays at most `n` (degree).
/// Used by: `Workspace::new_with_n_buckets` (AMD and AMF).
/// Why: SuiteSparse AMD's definition (faer `amd.rs:173-179`).
/// Contract: results.
pub const DENSE_DEGREE_MIN: usize = 16;

/// The quotient-graph workspace holds the off-diagonal entries plus
/// `1 / ELBOW_DIV` of them as elbow room, plus `n` (1).
/// Used by: `Workspace::new_with_n_buckets`.
/// Why: SuiteSparse AMD's `iwlen = nzaat + nzaat/5 + n` (faer
/// `amd.rs:921-924`); the room decides when garbage collection runs.
/// Contract: results (garbage collection reorders the lists).
pub const ELBOW_DIV: usize = 5;

/// Scores above `n` share degree buckets in steps of `n / BUCKET_STEP_DIV`
/// (1).
/// Used by: `quotient_graph::algo::amf_bucket_of`.
/// Why: tuned default; fill scores above `n` need only coarse buckets.
/// Contract: results (ties between equal-bucket candidates).
pub const BUCKET_STEP_DIV: usize = 8;

/// Steps a progress is reported in (percent of its units: vertices of an
/// ordering, panel entries of a factorization, columns of an analysis part).
/// Used by: `progress::Counter` (AMD, AMF, nested dissection, at most one
/// callback call per step) and rslab's `progress::Steps` (the factorization
/// and the parts of the analysis, at most one event per step).
/// Why: the count of calls and events does not grow with the problem; a
/// hundredth is finer than a reader follows.
/// Contract: output.
pub const PROGRESS_STEPS: u64 = 100;

/// Most hops of the George-Liu pseudo-peripheral node search of RCM
/// (hops).
/// Used by: `rcm::pseudo_peripheral`.
/// Why: George-Liu terminates in a handful of hops; the cap is for safety.
/// Contract: results (only where the search would run longer).
pub const RCM_PERIPHERAL_MAX_HOPS: usize = 16;

/// Denominator of the closed form of the sum of squares,
/// `0^2 + 1^2 + ... + (f - 1)^2 = (f - 1) f (2 f - 1) / 6` (1).
/// Used by: the flop counters of `quotient_graph::algo` (AMD and AMF).
/// Why: the closed form of the sum.
/// Contract: output (the flop counts, not the permutation).
pub const SUM_OF_SQUARES_DIV: f64 = 6.0;
