//! Every number nested dissection is tuned or defined by: the defaults of
//! [`MetisOptions`](crate::MetisOptions) and the fixed values of the
//! coarsening, the refinement and the recursion.
//!
//! Each constant states its unit, what reads it, why it has its value, and
//! its contract: **results** (changing it changes the permutation) or
//! **timing** (the permutation is unchanged).

/// Default seed of the deterministic RNG (1).
/// Used by: `MetisOptions::seed`.
/// Why: any fixed seed makes the runs reproducible.
/// Contract: results.
pub const SEED: u64 = 1;

/// Initial-bisection trials at the coarsest level (trials).
/// Used by: `MetisOptions::niparts`.
/// Why: METIS 5.2.0 default.
/// Contract: results.
pub const NIPARTS: u32 = 7;

/// Coarsening stops below this many vertices (vertices).
/// Used by: `MetisOptions::coarsen_floor`.
/// Why: METIS 5.2.0 default.
/// Contract: results.
pub const COARSEN_FLOOR: u32 = 120;

/// Subproblems of at most this many vertices are ordered by AMD instead of
/// further dissection (vertices).
/// Used by: `MetisOptions::nd_to_amd_switch`.
/// Why: METIS 5.2.0 default.
/// Contract: results.
pub const ND_TO_AMD_SWITCH: u32 = 200;

/// Reduction ratio below which heavy-edge matching falls back to 2-hop
/// matching (1).
/// Used by: `MetisOptions::two_hop_ratio_threshold`.
/// Why: METIS 5.2.0 default.
/// Contract: results.
pub const TWO_HOP_RATIO_THRESHOLD: f64 = 0.85;

/// Maximum partition imbalance as a fractional deviation (1).
/// Used by: `MetisOptions::max_imbalance`.
/// Why: METIS 5.2.0's `ufactor` 200, a 1.20 load balance tolerance.
/// Contract: results.
pub const MAX_IMBALANCE: f64 = 0.20;

/// FM passes at each uncoarsening level (passes).
/// Used by: `MetisOptions::fm_passes`.
/// Why: METIS 5.2.0 default.
/// Contract: results.
pub const FM_PASSES: u32 = 10;

/// A separator refinement pass stops after this many consecutive moves
/// without a better separator (moves).
/// Used by: `MetisOptions::move_limit`.
/// Why: limits below about 50 000 cost fill on grids (+37 % nnz(L) on a
/// 40^3 grid at `min(5 * separator, 400)`), so the default effectively
/// never binds.
/// Contract: results.
pub const MOVE_LIMIT: usize = 1 << 20;

/// A refinement pass also stops when the separator has grown to this
/// multiple of the best one seen (1).
/// Used by: `MetisOptions::max_overshoot`.
/// Why: tuned default.
/// Contract: results.
pub const MAX_OVERSHOOT: f64 = 4.0;

/// Subproblems with at least this many vertices order their two sides in
/// parallel (vertices).
/// Used by: `MetisOptions::parallel_min_vertices`.
/// Why: tuned default.
/// Contract: timing.
pub const PAR_MIN_VERTICES: usize = 4096;

/// Levels with at least this many adjacency entries contract in parallel
/// blocks (entries).
/// Used by: `MetisOptions::parallel_min_edges`.
/// Why: tuned default.
/// Contract: timing.
pub const PAR_MIN_EDGES: usize = 200_000;
