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
