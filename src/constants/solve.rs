//! The supernodal triangular solves.

/// Rows per register tile of the panel products of the solves (rows).
/// Used by: `supernodal::solve::tri_tile`.
/// Why: a tile of four rows by up to four right-hand sides stays in registers.
/// Contract: results (the tile fixes the summation order).
pub const TILE_ROWS: usize = 4;

/// Independent leaf subtrees the tree is cut into for the parallel sweeps
/// (subtrees).
/// Used by: `SolveSettings::leaf_subtrees`.
/// Why: tuned default; many more than workers, for balance.
/// Contract: timing (the solves are thread-count independent).
pub const SOLVE_LEAF_SUBTREES: usize = 128;

/// Right-hand sides per register tile of the solve kernels, and how many of
/// them count toward a sweep's work for the parallel gate (right-hand
/// sides).
/// Used by: `supernodal::solve` (`tri_rows`, `SolvePlan::fit`).
/// Why: a tile of [`TILE_ROWS`] rows by four right-hand sides stays in
/// registers; the kernels are instantiated for tiles of 1 to 4.
/// Contract: rounding (each right-hand side is summed alike in any tile;
/// the tile width only groups them).
pub const SOLVE_RHS_TILE: usize = 4;

/// Columns per block of a node's diagonal triangle in the sweeps through
/// GEMMs (columns).
/// Used by: `supernodal::solve::SolvePlan::fwd_node_gemm`, `bwd_node_gemm`.
/// Why: each block's triangle runs on the register tiles and the rows below it
/// in the block take one product; 64 keeps the triangles a small share and
/// the products wide enough for the SIMD GEMM.
/// Contract: rounding.
pub const SOLVE_DIAG_BLOCK: usize = 64;

/// Rows per task of a node's product of its rows below the diagonal block
/// with the right-hand sides, in the multi right-hand side forward sweep
/// (rows).
/// Used by: `supernodal::solve::SolvePlan::fwd_node_gemm`.
/// Why: 256 rows of a 900-column node are about 40 M flops for 23
/// right-hand sides, far above a task's start.
/// Contract: timing.
pub const SOLVE_GEMM_ROWS: usize = 256;

/// Work of a node's product below its diagonal block from which it runs in
/// row chunks on the pool, in entries times right-hand sides.
/// Used by: `supernodal::solve::SolvePlan::fwd_node_gemm`.
/// Why: about a millisecond of one core; below it the product is short
/// enough to stay with the task that runs the node.
/// Contract: timing.
pub const SOLVE_GEMM_PAR_MIN: usize = 1 << 22;

/// Columns per task of an ancestor's take of the leaf subtrees' terms in the
/// forward sweep (columns).
/// Used by: `supernodal::solve::SolvePlan::top_forward_flow`.
/// Why: 32 columns of 23 right-hand sides from 128 subtrees are about
/// 100 k additions, well above a task's start.
/// Contract: timing (each column sums its sources in the same order).
pub const SOLVE_PULL_COLS: usize = 32;

/// Additions of an ancestor's take of the subtrees' terms (sources times
/// columns times right-hand sides) from which it runs on the pool.
/// Used by: `supernodal::solve::SolvePlan::top_forward_flow`.
/// Why: the wide nodes near the root take the terms of every leaf subtree on
/// the critical path (a 1067-column root of a near-field LU spent 13 ms
/// there on one thread); below this it is under a millisecond.
/// Contract: timing.
pub const SOLVE_PULL_PAR_MIN: usize = 1 << 20;
