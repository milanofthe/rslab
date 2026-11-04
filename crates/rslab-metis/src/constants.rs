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
/// Used by: `MetisOptions::parallel_min_vertices` (where unset).
/// Why: tuned default.
/// Machine: a standalone default; rslab sets its own from what a section
/// costs on its pool.
/// Contract: timing.
pub const PAR_MIN_VERTICES: usize = 4096;

/// Levels with at least this many adjacency entries contract in parallel
/// blocks (entries).
/// Used by: `MetisOptions::parallel_min_edges` (where unset).
/// Why: tuned default.
/// Machine: a standalone default; rslab sets its own from what a section
/// costs on its pool.
/// Contract: timing.
pub const PAR_MIN_EDGES: usize = 200_000;

/// Blocks per worker of a parallel coarse-graph contraction (blocks per
/// worker).
/// Used by: `coarsen::contract`.
/// Why: the coarse vertices carry uneven work; several blocks per worker
/// balance it.
/// Contract: timing (the blocks are concatenated in order, the coarse graph
/// is the serial one).
pub const FINE_PIECES_PER_WORKER: usize = 8;

/// The 2-hop matching fallback runs only on graphs of at least this many
/// vertices (vertices).
/// Used by: `coarsen::coarsen_level`.
/// Why: tiny graphs need no fallback.
/// Contract: results.
pub const TWO_HOP_MIN_VERTICES: i32 = 4;

/// Coarsening stops when a level keeps more than this share of the previous
/// level's vertices (1). Not `OrderingSettings::compress_max_ratio`.
/// Used by: `coarsen::coarsen`.
/// Why: a level that removes under 5 % of the vertices has stalled.
/// Contract: results.
pub const COARSEN_STALL_RATIO: f64 = 0.95;

/// An FM bisection pass stops after this many consecutive moves without
/// improvement (moves).
/// Used by: `fm_refine::refine_bisection`.
/// Why: tuned default; bounds a pass that has stopped improving.
/// Contract: results.
pub const FM_NO_IMPROVE_MAX: u32 = 50;

/// Offset of the child seeds: child `c` of a subproblem draws its seed from
/// `CHILD_SEED_OFFSET + c` (1).
/// Used by: `node_nd::nd_subproblem`.
/// Why: any offset keeps the children's streams apart from the parent's.
/// Contract: results.
pub const CHILD_SEED_OFFSET: u64 = 7;

/// A separator whose larger side holds at least this share of the vertices
/// is useless; the subproblem is ordered by AMD instead (1).
/// Used by: `node_nd::nd_subproblem`.
/// Why: a lopsided split costs a dissection level for almost no
/// independence.
/// Contract: results.
pub const UNBALANCED_SPLIT_FRACTION: f64 = 0.9;

/// The label classes of a node separation, `[A, B, SEP]` (classes).
/// Used by: `sep_refine::class_weights`.
/// Why: two sides and the separator.
/// Contract: results.
pub const LABEL_CLASSES: usize = 3;
