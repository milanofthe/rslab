//! Threads and parallel granularity: worker pools, their stacks, and the
//! gates from which a loop runs on the pool. Every value here changes time
//! only; the results do not depend on the thread count.

/// Stack of the thread a foreign rayon worker enters a pool through; it only
/// waits (bytes).
/// Used by: `numeric::settings::HopShared::start`.
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
/// Used by: `parallel::piece_len` (every `crate::parallel` loop, per
/// thread it runs on), the column blocks of a GEMM on a lent pool
/// (`dense::gemm_backend::gemm`), the row chunks of the blocked ancestor
/// sweep (`supernodal::solve::SolvePlan::apex_forward`), the pieces of a
/// KLU refactor's levels and blocks (`KluSolver::refactor`).
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

/// A section goes wide only if that saves at least this fraction of its
/// serial time (1).
/// Used by: `parallel::Dispatch::width`.
/// Why: margin for the scatter of the measured costs and the learned
/// rates, so a section near break-even stays serial instead of flipping
/// between the two.
/// Contract: timing.
pub const GAIN: f64 = 0.2;

/// What a section costs a lent pool that measures nothing
/// (`parallel::Dispatch::assumed`): this, plus [`ASSUMED_HELPER_NS`] per
/// thread beyond the calling one (ns).
/// Used by: `parallel::Pool::dispatch` (its default).
/// Why: the wake and hand-off of a desktop CPU's pool; a pool that knows
/// better says so through `Pool::dispatch`.
/// Machine: a prior for pools that do not measure themselves.
/// Contract: timing.
pub const ASSUMED_SECTION_NS: f64 = 2_000.0;

/// See [`ASSUMED_SECTION_NS`] (ns per helper thread).
/// Used by: `parallel::Dispatch::assumed`.
/// Why: each helper is one more hand-off and one more cache to fill.
/// Machine: a prior, as [`ASSUMED_SECTION_NS`].
/// Contract: timing.
pub const ASSUMED_HELPER_NS: f64 = 1_000.0;

/// Sections timed per thread count when a pool's section cost is measured
/// (`parallel::Dispatch::measure`, the median taken), after as many to wake
/// the workers (sections).
/// Used by: `parallel::Dispatch::measure`.
/// Why: enough for a stable median, about a millisecond per thread count.
/// Contract: timing.
pub const CALIBRATION_ROUNDS: usize = 64;

/// How long each piece of a calibration section is busy (ns).
/// Used by: `parallel::Dispatch::measure`.
/// Why: long enough for an awake pool's threads to take their pieces before
/// the caller has run them all (a work-stealing pool runs a piece nobody
/// took on the caller, and instant pieces would make every section look
/// free), and about the least work a section is ever worth.
/// Contract: timing.
pub const CALIBRATION_PIECE_NS: u64 = 5_000;

/// Cache lines each piece of a calibration section writes (lines).
/// Used by: `parallel::Dispatch::measure`.
/// Why: the calibration pays the hand-over of a piece's output a real
/// section pays, not only the queue's.
/// Contract: timing.
pub const CALIBRATION_LINES: usize = 8;

/// A thread measures one in this many of the runs a
/// [`Rate`](crate::parallel::Rate) could learn from (runs).
/// Used by: `parallel::Rate::sample`.
/// Why: two clock reads per sampled run, too few to show.
/// Contract: timing.
pub const RATE_SAMPLE_EVERY: u32 = 16;

/// A [`Rate`](crate::parallel::Rate) learns only from runs it estimates at
/// this many ns or more (ns).
/// Used by: `parallel::Rate::sample`.
/// Why: two clock reads do not show against it, and the rate is updated
/// rarely enough that the threads do not contend for it.
/// Contract: timing.
pub const RATE_MIN_NS: f64 = 2_000.0;

/// The weight of a new measurement in a rate's running mean (1).
/// Used by: `parallel::Rate::learn`.
/// Why: a few runs move it, one outlier does not.
/// Contract: timing.
pub const RATE_WEIGHT: f32 = 0.25;

/// What a complex multiply-add costs against a real one, the factor
/// between the real and the complex prior of every rate (1).
/// Used by: `parallel::ScalarRates`.
/// Why: four real multiply-adds make one complex one.
/// Contract: timing.
pub const COMPLEX_OPS: f32 = 4.0;

/// ns per multiply-add of a dense product (`m n k` of a GEMM), before any
/// is measured (ns per multiply-add).
/// Used by: `dense::gemm_backend::gemm`, `numeric::supernodal::node`.
/// Why: a SIMD kernel's rate with its packing.
/// Machine: a prior, replaced by what the sequential products measure.
/// Contract: timing.
pub const GEMM_NS_PER_FLOP: f32 = 0.05;

/// ns per multiply-add of the scalar loops of the dense kernels (a panel's
/// deep rows, a triangular solve's columns, a slab's updates), before any is
/// measured (ns per multiply-add).
/// Used by: `numeric::ldlt`, `numeric::lu` (their row and column sweeps).
/// Why: a scalar loop over strided columns.
/// Machine: a prior, replaced by what the runs measure.
/// Contract: timing.
pub const LOOP_NS_PER_FLOP: f32 = 0.5;

/// ns per panel entry and right-hand side of a supernodal triangular sweep,
/// before any is measured (ns per entry).
/// Used by: `supernodal::solve`.
/// Why: one multiply-add on a streamed panel entry.
/// Machine: a prior, replaced by what the sweeps measure.
/// Contract: timing.
pub const SOLVE_NS_PER_ENTRY: f32 = 0.5;

/// ns per entry of a pass over a pattern or a vector (a scan, a copy, a
/// transpose, a fill, a comparison), before any is measured (ns per entry).
/// Used by: the analysis passes (`ordering::amd`, `scaling::mc64`,
/// `lu::solver`, `symbolic::supervariables`), the zero fill of a factor,
/// the Krylov reductions.
/// Why: a streamed entry and a random access.
/// Machine: a prior, replaced by what the passes measure.
/// Contract: timing.
pub const SCAN_NS_PER_ENTRY: f32 = 1.0;

/// ns per entry of a pass that sorts, hashes or takes a logarithm per
/// entry, before any is measured (ns per entry).
/// Used by: `scaling::mc64`, `lu::solver`, `symbolic::supervariables`.
/// Why: a comparison sort's few passes, a hash, a `ln`.
/// Machine: a prior, replaced by what the passes measure.
/// Contract: timing.
pub const SORT_NS_PER_ENTRY: f32 = 10.0;

/// ns per updater released after a supernode's factorization, before any
/// is measured (ns per updater).
/// Used by: `supernodal::forest::ll_forest`.
/// Why: a shared counter, and for the last consumer a panel freed.
/// Machine: a prior, replaced by what the releases measure.
/// Contract: timing.
pub const FREE_NS_PER_UPDATER: f32 = 100.0;

/// ns per entry of the symmetrized pattern a fill-reducing ordering takes,
/// before any is measured (ns per entry).
/// Used by: `symbolic::race` (whether the candidates run side by side),
/// `symbolic::ordering_graph` (what it learns from).
/// Why: a minimum-degree ordering's quotient-graph updates.
/// Machine: a prior, replaced by what the orderings measure.
/// Contract: timing.
pub const ORDER_NS_PER_ENTRY: f32 = 50.0;

/// ns per vertex of a nested-dissection subproblem, from which its two
/// sides are ordered side by side (ns per vertex).
/// Used by: `symbolic::ordering_graph` (`MetisOptions::parallel_min_vertices`
/// left unset).
/// Why: a multilevel bisection with refinement and the recursion below it.
/// Machine: a prior; the section cost it is set against is measured.
/// Contract: timing.
pub const ND_NS_PER_VERTEX: f32 = 1_000.0;

/// ns per adjacency entry of a coarse-graph contraction, from which a level
/// contracts in parallel blocks (ns per entry).
/// Used by: `symbolic::ordering_graph` (`MetisOptions::parallel_min_edges`
/// left unset).
/// Why: a hash-merge of the matched vertices' adjacencies.
/// Machine: a prior; the section cost it is set against is measured.
/// Contract: timing.
pub const CONTRACT_NS_PER_ENTRY: f32 = 10.0;

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
