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

/// Work of a triangular sweep from which it runs on the thread pool, in
/// factor panel entries times right-hand
/// sides, the latter counted up to four (entries x rhs). Not the unit of the
/// KLU path's [`KLU_PAR_MIN_FMADDS`](super::KLU_PAR_MIN_FMADDS).
/// Used by: `supernodal::solve::SolvePlan::fit`.
/// Why: about where handing a sweep to the pool starts to pay; a smaller
/// one costs several times the solve there at a few thousand unknowns.
/// Machine: an estimate, not measured at run time.
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

/// Rows a row chunk of a node's product for one right-hand side starts at a
/// multiple of (rows).
/// Used by: `supernodal::solve::chunk_align` (`SolvePlan::cut`, `chunk_rows`).
/// Why: for one right-hand side the GEMM runs a matrix-vector product whose
/// column runs the rows in SIMD lanes from the product's first row and the rows
/// past its last full lane one by one, with other rounding; a chunk starting at
/// a multiple of the widest lane (16: `f32` on AVX-512) leaves every row in its
/// lane of the whole product. Chunks cut at arbitrary rows gave the forward
/// sweep of tia130a's near-field LU other bits on 2 threads than on 1.
/// Contract: results (the solution is the same on any thread count).
pub const SOLVE_GEMV_ROWS: usize = 16;

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
