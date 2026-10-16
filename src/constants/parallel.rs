//! Threads and parallel granularity: worker pools, their stacks, and the
//! gates from which a loop runs on the pool. Every value here changes time
//! only; the results do not depend on the thread count.

/// Stack of the thread a foreign rayon worker enters a pool through; it only
/// waits (bytes).
/// Used by: `numeric::settings::enter`.
/// Why: the thread runs no solver code, so a small stack suffices.
/// Contract: timing.
pub const HOP_STACK: usize = 256 * 1024;

/// Scoped pools a thread keeps, at most this many; the least recently used
/// goes (pools).
/// Used by: `numeric::settings::scoped_pool`.
/// Why: a few worker counts and stack sizes in turn (factor, solve) stay
/// warm without holding threads for every combination ever used.
/// Contract: timing.
pub const POOLS_KEPT: usize = 4;

/// Native stack one level of the recursive tree factorization takes, as
/// budgeted for the worker stack (bytes per supernode-tree level).
/// Used by: `numeric::settings::stack_for_depth`.
/// Why: about 6.7 KB measured per frame, with headroom.
/// Contract: timing (a too small stack overflows, it never changes bits).
pub const STACK_PER_LEVEL: usize = 32 * 1024;

/// Smallest worker stack of a factorization (bytes).
/// Used by: `numeric::settings::stack_for_depth`.
/// Why: at least the rayon default; covers a tree depth of about 500.
/// Contract: timing.
pub const STACK_MIN: usize = 16 * 1024 * 1024;

/// Largest worker stack of a factorization (bytes): 8 GiB on 64-bit
/// targets (a tree depth of about 256k), 1 GiB on 32-bit ones (wasm32),
/// where the 64-bit value would overflow `usize` at const evaluation.
/// Used by: `numeric::settings::stack_for_depth`.
/// Why: address space only; a stack is committed as the recursion descends.
/// Contract: timing.
pub const STACK_MAX: usize = if usize::BITS >= 64 { 8 << 30 } else { 1 << 30 };

/// Updaters of a supernode from which their release after its
/// factorization runs on the pool (updaters).
/// Used by: `supernodal::forest::ll_forest`.
/// Why: the wide top-of-tree frees are disjoint and worth spreading; a
/// short list is cheaper on the calling thread.
/// Contract: timing.
pub const FREE_PAR_MIN_UPDATERS: usize = 64;

/// Entries from which the AMD input pattern is transposed in parallel
/// blocks (nonzeros).
/// Used by: `ordering::amd`.
/// Why: below it the parallel section costs more than the transpose.
/// Contract: timing.
pub const AMD_TRANSPOSE_PAR_MIN_NNZ: usize = 1 << 16;

/// Default cap of the predicted worker count, `Threads::Auto { max }`
/// (workers).
/// Used by: `Threads::default`.
/// Why: the strong-scaling efficiency knee is at about 4 to 6 threads, so 4
/// is the best throughput per core and leaves room for concurrent solves
/// (solver-in-the-loop). `Auto` still predicts fewer where more would
/// regress. The ordering race assumes the same count
/// ([`RACE_ASSUMED_WORKERS`](super::RACE_ASSUMED_WORKERS)).
/// Contract: timing.
pub const DEFAULT_MAX_THREADS: usize = 4;

/// Pieces a parallel loop is cut into per worker (pieces per worker).
/// Used by: `parallel::piece_len` (every `crate::parallel` loop), the
/// column blocks of a GEMM on a lent pool (`dense::gemm_backend::gemm`),
/// the row chunks of the blocked ancestor sweep
/// (`supernodal::solve::SolvePlan::apex_forward`).
/// Why: a few pieces per worker balance uneven pieces while keeping the
/// per-piece overhead small.
/// Contract: timing.
pub const PIECES_PER_WORKER: usize = 4;

/// Pieces per worker of the loops whose pieces are blocks of a pattern with
/// uneven columns (pieces per worker).
/// Used by: `ordering::amd::permute_pattern`,
/// `symbolic::supervariables::Supervariables::compress`.
/// Why: finer than [`PIECES_PER_WORKER`]: the work per column varies more.
/// Contract: timing (the blocks are concatenated in order).
pub const FINE_PIECES_PER_WORKER: usize = 8;

/// Fewest columns per block of a GEMM split on a lent pool (columns).
/// Used by: `dense::gemm_backend::gemm`.
/// Why: narrower blocks lose the GEMM kernel's rate.
/// Contract: timing (the per-entry sums do not depend on the blocking).
pub const GEMM_PIECE_MIN_COLS: usize = 16;

/// Fewest columns per piece of the LU analysis's symmetrization sort
/// (columns).
/// Used by: `lu::solver::symmetrized_lower_pattern`.
/// Why: a column's sort is short; pieces must carry many.
/// Contract: timing.
pub const SYMMETRIZE_PIECE_MIN_COLS: usize = 256;

/// Fewest columns per piece of the LU analysis's diagonal check (columns).
/// Used by: `lu::solver::diagonal_needs_matching`.
/// Why: one column's check is a few comparisons.
/// Contract: timing.
pub const DIAGONAL_CHECK_PIECE_MIN_COLS: usize = 1024;

/// Fewest entries per piece of the MC64 logarithm pass (entries).
/// Used by: `scaling::mc64::compute_matching_general`.
/// Why: one entry is one logarithm.
/// Contract: timing.
pub const MC64_LOG_PIECE_MIN_ENTRIES: usize = 4096;

/// Fewest columns per piece of the MC64 column-maximum pass (columns).
/// Used by: `scaling::mc64::compute_matching_general`.
/// Why: a column is a short scan.
/// Contract: timing.
pub const MC64_COLMAX_PIECE_MIN_COLS: usize = 256;

/// Fewest runs per piece of the supervariable detection (runs).
/// Used by: `symbolic::supervariables::Supervariables::of`.
/// Why: a run of equal keys is a short comparison.
/// Contract: timing.
pub const SUPERVARIABLE_PIECE_MIN_RUNS: usize = 64;

/// Entries per chunk of the parallel zero fill of a new factor (entries).
/// Used by: `supernodal::panel::PanelArena::new_in`.
/// Why: the factor is often hundreds of MB (41 ms serial on a 465 MB
/// factor); 64k entries per chunk keep the scheduling overhead negligible.
/// Contract: timing.
pub const ZERO_FILL_CHUNK: usize = 1 << 16;

/// A spinning KLU replay worker yields its thread once every this many
/// spins plus one, a power of two minus one used as a mask (spins).
/// Used by: `KluSolver::refactor` (pipelined replay wait).
/// Why: rarely enough not to slow a short wait, often enough not to starve
/// the worker it waits for on an oversubscribed machine.
/// Contract: timing.
pub const SPIN_YIELD_MASK: u32 = 0x3FF;

/// The structural thread recommendation keeps at most two workers on
/// patterns whose largest front has fewer rows than this ... (rows).
/// Used by: `supernodal::analysis::recommend_threads_from`.
/// Why: thin fronts offer no node parallelism to exploit.
/// Contract: timing.
pub const THIN_FRONT_MAX_ROWS: usize = 512;

/// ... and whose widest tree level has fewer nodes than this (nodes).
/// Used by: `supernodal::analysis::recommend_threads_from`.
/// Why: a narrow, path-like tree offers no tree parallelism either.
/// Contract: timing.
pub const NARROW_TREE_MAX_WIDTH: usize = 128;

/// Workers the structural recommendation keeps on such thin, narrow
/// patterns (workers).
/// Used by: `supernodal::analysis::recommend_threads_from`.
/// Why: more only oversubscribe.
/// Contract: timing.
pub const THIN_PATTERN_THREADS: usize = 2;

/// Below this predicted factor work the recommendation keeps at most
/// [`SMALL_WORK_THREADS`] workers (flops).
/// Used by: `supernodal::analysis::recommend_threads_from`.
/// Why: parallel scheduling overhead dominates tiny factorizations.
/// Contract: timing.
pub const SMALL_WORK_MAX_FLOPS: u64 = 300_000_000;

/// Workers the recommendation keeps on small work (workers).
/// Used by: `supernodal::analysis::recommend_threads_from`.
/// Why: see [`SMALL_WORK_MAX_FLOPS`].
/// Contract: timing.
pub const SMALL_WORK_THREADS: usize = 4;
