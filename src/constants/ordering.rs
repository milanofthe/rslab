//! The analysis: the fill-reducing ordering, its race, and the
//! amalgamation of columns into supernodes. Defaults of
//! [`OrderingSettings`](crate::OrderingSettings),
//! [`RaceSettings`](crate::RaceSettings),
//! [`AmalgamationSettings`](crate::AmalgamationSettings) and
//! [`RelaxAmalgamation`](crate::RelaxAmalgamation).

/// Order the graph of indistinguishable-vertex groups only when the groups
/// shrink it to at most this share of its vertices (1).
/// Used by: `OrderingSettings::compress_max_ratio`.
/// Why: tuned default; a compression that removes few vertices does not
/// repay building the compressed graph.
/// Contract: results (the ordering runs on another graph).
pub const COMPRESS_MAX_RATIO: f64 = 0.95;

/// Nested dissection joins the ordering race only above this many unknowns.
/// Used by: `RaceSettings::nd_min_n`.
/// Why: tuned default; a dissection must amortize over the factorization.
/// Contract: results (which orderings compete).
pub const RACE_ND_MIN_N: usize = 10_000;

/// ... and only when the best cheap candidate predicts at least this factor
/// time, in flops per worker or on the longest elimination chain (flops).
/// Used by: `RaceSettings::nd_min_work`.
/// Why: tuned default; a dissection must amortize over the factorization.
/// Contract: results.
pub const RACE_ND_MIN_WORK: u64 = 1_250_000_000;

/// Workers the race's time prediction assumes (workers).
/// Used by: `RaceSettings::assumed_workers`.
/// Why: the default worker cap, [`DEFAULT_MAX_THREADS`](super::DEFAULT_MAX_THREADS).
/// Contract: results (moves the nested-dissection gate).
pub const RACE_ASSUMED_WORKERS: usize = 4;

/// Patterns with at least this many entries (lower triangle) start the
/// dissection at once, speculatively, where more than one worker is
/// available (nonzeros).
/// Used by: `RaceSettings::eager_nd_min_nnz`.
/// Why: on smaller patterns a discarded speculative dissection outlasts the
/// cheap stage (it doubled the race on a 128k power grid).
/// Contract: timing (the race decides as without the speculation).
pub const RACE_EAGER_ND_MIN_NNZ: usize = 2_000_000;

/// Seeds of the nested-dissection ensemble, `nd.seed` upwards (seeds).
/// Used by: `RaceSettings::ensemble_size`.
/// Why: tuned default.
/// Contract: results (with the ensemble on).
pub const RACE_ENSEMBLE_SIZE: usize = 3;

/// The ensemble runs from this predicted flop count, the sum of squared
/// column counts (flops).
/// Used by: `RaceSettings::ensemble_min_flops`.
/// Why: tuned default; only heavy factorizations repay several
/// dissections.
/// Contract: results (with the ensemble on).
pub const RACE_ENSEMBLE_MIN_FLOPS: u64 = 50_000_000_000;

/// Supernodes narrower than this merge into their parent when the parent is
/// narrower too (columns).
/// Used by: `AmalgamationSettings::nemin`.
/// Why: the MA57 default of its `nemin`; narrower fronts do not reach GEMM
/// rates.
/// Contract: results (the supernode partition).
pub const NEMIN: usize = 16;

/// Relaxed amalgamation applies from this many unknowns.
/// Used by: `AmalgamationSettings::relax_min_n`.
/// Why: tuned default.
/// Contract: results (with relaxed amalgamation on).
pub const RELAX_MIN_N: usize = 1024;

/// `Auto` amalgamation treats a tree as path-like, and does not renumber,
/// when fewer than this share of its internal nodes have several children
/// (1).
/// Used by: `AmalgamationSettings::path_like_fraction`.
/// Why: tuned default; renumbering pays only where there are siblings to
/// bring together.
/// Contract: results.
pub const PATH_LIKE_FRACTION: f64 = 0.05;

/// The root supernode cap applies from this many unknowns.
/// Used by: `AmalgamationSettings::root_cap_min_n`.
/// Why: see [`ROOT_CAP_FRACTION`].
/// Contract: results.
pub const ROOT_CAP_MIN_N: usize = 1024;

/// Merges into a root supernode stop at this share of the unknowns (1).
/// Used by: `AmalgamationSettings::root_cap_fraction`.
/// Why: a wide top-level Schur complement (interior-point KKT systems)
/// otherwise grows one dense root block.
/// Contract: results.
pub const ROOT_CAP_FRACTION: f64 = 0.05;

/// ... and at most at this many columns (columns).
/// Used by: `AmalgamationSettings::root_cap_max`.
/// Why: tuned default; bounds the dense root block on large systems.
/// Contract: results.
pub const ROOT_CAP_MAX: usize = 2048;

/// Widest front a relaxed merge may produce (columns).
/// Used by: `RelaxAmalgamation::default`.
/// Why: tuned default (relaxed amalgamation is off by default).
/// Contract: results (with relaxed amalgamation on).
pub const RELAX_MAX_WIDTH: usize = 256;

/// Most explicit-zero rows one relaxed merge may add (rows).
/// Used by: `RelaxAmalgamation::default`.
/// Why: tuned default (relaxed amalgamation is off by default).
/// Contract: results (with relaxed amalgamation on).
pub const RELAX_MAX_EXTRA_ROWS: usize = 64;
