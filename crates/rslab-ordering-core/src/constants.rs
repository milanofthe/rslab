//! The numbers the minimum-degree orderings (AMD, AMF) are defined by,
//! shared by `rslab-amd` and `rslab-amf` so the two cannot drift apart.
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
/// (1). The AMF bucket of `algo` and the minimum-fill metric must use the
/// same step.
/// Used by: `quotient_graph::algo::amf_bucket_of`, `MinFill::bucket`.
/// Why: tuned default; fill scores above `n` need only coarse buckets.
/// Contract: results (ties between equal-bucket candidates).
pub const BUCKET_STEP_DIV: usize = 8;

/// Most hops of the George-Liu pseudo-peripheral node search of RCM
/// (hops).
/// Used by: `rcm::pseudo_peripheral`.
/// Why: George-Liu terminates in a handful of hops; the cap is for safety.
/// Contract: results (only where the search would run longer).
pub const RCM_PERIPHERAL_MAX_HOPS: usize = 16;
