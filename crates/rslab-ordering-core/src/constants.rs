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
