//! The supernodal triangular solves.

/// Rows per register tile of the panel products of the solves (rows).
/// Used by: `supernodal::solve::tri_tile`.
/// Why: a tile of four rows by up to four right-hand sides stays in registers.
/// Contract: results (the tile fixes the summation order).
pub const TILE_ROWS: usize = 4;

/// Independent leaf subtrees the tree is cut into for the parallel sweeps
/// (subtrees).
/// Used by: `SolveSettings::leaf_subtrees`.
/// Why: many more than workers, for balance of the subtrees' phase. Fewer,
/// larger leaf subtrees shrink the accumulators the subtrees fill and the
/// ancestors take (tia130a near-field LU: 319 k slots at 128, 160 k at 32).
/// Near-field LUs, 16 right-hand sides, EPYC-Milan (6 cores / 12 threads),
/// median of 21 (`examples/lu_solve_timing.rs` with `RSLAB_DELAYED=1`,
/// `RSLAB_LEAF`), ms:
///
/// | leaf subtrees               | 32    | 64    | 128   |
/// |-----------------------------|-------|-------|-------|
/// | tia130a 1 GHz, 12 threads   | 27.6  | 28.4  | 30.2  |
/// | tia130a 1 GHz, 1 thread     | 106.3 | 107.2 | 109.6 |
/// | tia130a 170 GHz, 12 threads | 27.6  | 27.9  | 29.4  |
/// | momstack 1 GHz, 12 threads  | 4.40  | 4.98  | 5.15  |
/// | pa180a 1 GHz, 12 threads    | 93.1  | 100.4 | 99.3  |
///
/// (pa180a varies by a tenth from run to run.) Not measured on other matrix
/// families or on more threads, so the default stays.
/// Contract: rounding (the cut groups the subtrees' terms; every cut solves
/// alike on any thread count).
pub const SOLVE_LEAF_SUBTREES: usize = 128;

/// Work of a triangular sweep from which it starts on the thread pool,
/// before its ways are measured, in factor panel entries times right-hand
/// sides, the latter counted up to four (entries x rhs). Not the unit of the
/// KLU path's [`KLU_PAR_MIN_FMADDS`](super::KLU_PAR_MIN_FMADDS).
/// Used by: `supernodal::solve::SolvePlan::fit`.
/// Why: about where handing a sweep to the pool starts to pay; a smaller
/// one costs several times the solve there at a few thousand unknowns.
/// Machine: a prior, replaced by what the sweeps measure.
/// Contract: timing (the same bits either way).
pub const SOLVE_PAR_MIN_ENTRY_RHS: usize = 1 << 20;

/// Right-hand sides per register tile of the solve kernels, and how many of
/// them count toward a sweep's work where it starts on the pool (right-hand
/// sides).
/// Used by: `supernodal::solve` (`tri_rows`, `tri_up`, `SolvePlan::fit`).
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

/// Work of a row chunk of a node's products in the sweeps, in panel entries
/// times right-hand sides: a product of more is cut into row chunks, which
/// an ancestor runs on the pool.
/// Used by: `SolveSettings::gemm_task` (`supernodal::solve::SolvePlan::node_chunks`).
/// Why: the ancestors' chain to the root is the critical path of the sweeps,
/// and its nodes are mostly narrow and deep (tia130a near-field LU, 15.8 k
/// unknowns, 16 right-hand sides: of the forward chain's 5.0 M entries 0.9 M
/// in triangles, the rest below them in nodes such as 77 x 1601 or
/// 743 x 1278). Whole products of up to 4 M entries times right-hand sides
/// and triangles on one thread capped both sweeps of the ancestors at 2.7 to
/// 2.9 times their one-thread time on 12 threads. At the 0.22 ns an entry
/// and right-hand side the ancestors took there, a chunk is about 30 us,
/// several times a task's start. With these chunks the ancestors' sweeps run
/// 3.2 (forward) and 3.9 (backward) times faster on 12 threads than on one,
/// the solve 38.6 -> 32.4 ms (EPYC-Milan, 6 cores / 12 threads, median of 21,
/// `examples/lu_solve_timing.rs` with `RSLAB_DELAYED=1`). Swept with
/// `RSLAB_GEMM_TASK` on 12 threads (leaf subtrees 128), ms:
///
/// | entries x rhs | 2^14 | 2^15 | 2^16 | 2^17 | 2^18 | 2^19 |
/// |---------------|------|------|------|------|------|------|
/// | 4 rhs         | 15.6 | 17.0 | 16.1 | 17.6 | 18.4 | 19.0 |
/// | 16 rhs        | 32.1 | 32.1 | 30.0 | 29.6 | 31.8 | 30.8 |
///
/// Within the run-to-run spread from 2^14 to 2^17 at 16 right-hand sides
/// (2^21: 43.6 before the products below a node ran in one section); 4 favour
/// smaller chunks by up to a tenth, not reproduced on 6 threads.
/// Contract: timing (a row's sums are the same in a chunk as in the whole
/// product; see `node_chunks`).
pub const SOLVE_GEMM_TASK: usize = 1 << 17;

/// Rows a row chunk of a node's product for one right-hand side starts at a
/// multiple of (rows).
/// Used by: `supernodal::solve::chunk_align` (`SolvePlan::node_chunks`, `chunk_rows`).
/// Why: for one right-hand side the GEMM runs a matrix-vector product whose
/// column runs the rows in SIMD lanes from the product's first row and the rows
/// past its last full lane one by one, with other rounding; a chunk starting at
/// a multiple of the widest lane (16: `f32` on AVX-512) leaves every row in its
/// lane of the whole product. Chunks cut at arbitrary rows gave the forward
/// sweep of tia130a's near-field LU other bits on 2 threads than on 1.
/// Contract: results (the solution is the same on any thread count).
pub const SOLVE_GEMV_ROWS: usize = 16;

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

/// Columns of a sparse right-hand side per block of the reach-limited
/// forward sweeps of [`LdltSolver::port_matrix`](crate::LdltSolver::port_matrix)
/// (right-hand sides).
/// Used by: `supernodal::solve::SolvePlan::port_matrix`.
/// Why: a block sweeps the union of its columns' reaches once; columns
/// grouped by the subtree they start in share most of it, and a wider block
/// applies every panel entry read to more right-hand sides until the union
/// outgrows one column's reach. Port matrix of 200 six-unknown ports
/// (`examples/port_matrix_timing.rs`, median of 5), seconds:
///
/// | width                    | 1     | 2     | 4     | 8     | 16    | 32    | 64    |
/// |--------------------------|-------|-------|-------|-------|-------|-------|-------|
/// | helmholtz 400^2, 1 thr   | 0.231 | 0.096 | 0.070 | 0.069 | 0.072 | 0.084 | 0.135 |
/// | helmholtz 400^2, 8 thr   | 0.045 | 0.030 | 0.014 | 0.016 | 0.021 | 0.023 | 0.046 |
/// | curl-curl 20^3, 1 thr    | 1.046 | 0.662 | 0.535 | 0.581 | 0.558 | 0.578 | 0.649 |
/// | curl-curl 20^3, 8 thr    | 0.246 | 0.132 | 0.125 | 0.112 | 0.133 | 0.128 | 0.235 |
///
/// Flat from 4 to 16; 8 is within 15% of the best in every row.
/// Contract: timing (`Z` has the same bits for every thread count).
pub const SOLVE_PORT_BLOCK: usize = 8;
