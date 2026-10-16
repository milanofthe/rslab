//! The supernodal triangular solves.

/// Rows per register tile of the panel products of the solves (rows).
/// Used by: `supernodal::solve::product_tile`, `supernodal::solve::tri_tile`.
/// Why: a tile of four rows by up to four right-hand sides stays in registers.
/// Contract: results (the tile fixes the summation order).
pub const TILE_ROWS: usize = 4;

/// Independent leaf subtrees the tree is cut into for the parallel sweeps
/// (subtrees).
/// Used by: `SolveSettings::leaf_subtrees`.
/// Why: tuned default; many more than workers, for balance.
/// Contract: timing (the solves are thread-count independent).
pub const SOLVE_LEAF_SUBTREES: usize = 128;

/// Column block of a large ancestor's sweep (columns): the block's triangle
/// runs its right-hand-side tiles side by side, its update of the rows below
/// in row chunks of [`SOLVE_PRODUCT_ROWS`].
/// Used by: `SolveSettings::block`.
/// Why: tuned default.
/// Contract: rounding (the blocking orders the sums).
pub const SOLVE_BLOCK: usize = 512;

/// Columns per dot task of a large ancestor's backward sweep (columns).
/// Used by: `SolveSettings::ancestor_chunk`.
/// Why: tuned default.
/// Contract: rounding.
pub const SOLVE_ANCESTOR_CHUNK: usize = 32;

/// Panel entries from which an ancestor node uses the blocked sweep with
/// parallel sections inside the node (entries).
/// Used by: `SolveSettings::apex_min_work`.
/// Why: the ancestors run as a task tree, and a node on its critical path
/// that sweeps alone holds the rest up. Measured on a 32 k near-field LU
/// (36.6 M factor entries, 13 right-hand sides, 8 cores): 83 / 57 / 57 ms on
/// 4 / 8 / 16 threads, against 117 / 67 / 64 ms at `1 << 18` and
/// 86 / 61 / 57 ms at `0`. For a node of at most [`SOLVE_BLOCK`] columns
/// the blocked sweep is the plain one's arithmetic.
/// Contract: rounding.
pub const SOLVE_APEX_MIN_ENTRIES: usize = 1 << 14;

/// Rows per task of a large ancestor's forward update of the rows below a
/// column block (rows): fixed, so no row's arithmetic depends on the thread
/// count, and a multiple of [`TILE_ROWS`].
/// Used by: `supernodal::solve::SolvePlan::apex_forward`.
/// Why: 128 rows of a 512-column block are 64 k multiply-adds per
/// right-hand side, well above the cost of starting a task.
/// Contract: timing (each row takes its terms in the same order in any
/// chunk).
pub const SOLVE_PRODUCT_ROWS: usize = 128;

/// Work of a triangular sweep from which it runs on the thread pool, in
/// factor panel entries times right-hand sides, the latter counted up to
/// four (entries x rhs). Not the unit of the KLU path's
/// [`KLU_PAR_MIN_FMADDS`](super::KLU_PAR_MIN_FMADDS).
/// Used by: `SolveSettings::par_min_work`.
/// Why: about where handing a sweep to the pool starts to pay; a smaller
/// one costs several times the solve there at a few thousand unknowns.
/// Contract: timing (the same bits either way).
pub const SOLVE_PAR_MIN_ENTRY_RHS: usize = 1 << 20;

/// Right-hand sides per register tile of the solve kernels, and how many of
/// them count toward the parallel gate
/// [`SOLVE_PAR_MIN_ENTRY_RHS`] (right-hand sides).
/// Used by: `supernodal::solve` (`bwd_node_block`, `panel_product`,
/// `dot4_block`, `tri_rows`, `tri_backward`, `SolvePlan::fit`).
/// Why: a tile of [`TILE_ROWS`] rows by four right-hand sides stays in
/// registers; the kernels are instantiated for tiles of 1 to 4.
/// Contract: rounding (each right-hand side is summed alike in any tile;
/// the tile width only groups them).
pub const SOLVE_RHS_TILE: usize = 4;

/// Independent accumulators of the solve's dot products, summed as
/// `(s0 + s1) + (s2 + s3)` at the end (accumulators).
/// Used by: `supernodal::solve::dot4`, `supernodal::solve::dot4_tile`.
/// Why: four lanes hide the FMA latency; the fixed lane count and fold
/// tree make the sums independent of the thread count.
/// Contract: results (the summation order; the fold tree assumes four).
pub const FOLD_LANES: usize = 4;
