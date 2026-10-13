//! Memory accounting: what scratch is kept, and the fixed terms of the
//! memory plan.

/// The most heap one scratch object, and one worker's buffer of split
/// planes, keep between the kernels that borrow them; larger ones are freed
/// when given back (bytes).
/// Used by: `supernodal::store`, `dense::gemm_backend`, `memory::worker_bytes`,
/// `memory::pooled_bytes`.
/// Why: small factorizations (Newton loops up to some thousand unknowns)
/// keep all their scratch and refactor without allocating; on large ones a
/// node that needs more allocates its own, which its flops dwarf, and what
/// the pools and threads hold on to stays bounded by the worker count.
/// Contract: timing (memory only).
pub const SCRATCH_KEEP: u64 = 256 << 10;

/// Heap every factorization allocates besides its data: the diagnostics and
/// log records, the thread pool's registry and the per-call small vectors
/// (bytes).
/// Used by: the memory plans of the LDL^T, LU and KLU paths.
/// Why: a flat allowance; these allocations do not grow with the matrix.
/// Contract: estimate.
pub const BOOKKEEPING: u64 = 512 << 10;

/// Nodes one worker can have in flight: a worker waiting inside a parallel
/// product of its node runs other ready work meanwhile (rayon's work
/// stealing), which can be another node (nodes).
/// Used by: `memory::worker_bytes`, `memory::pooled_bytes`, the heaviest
/// node set of the memory plan.
/// Why: one node of its own plus one stolen while it waits.
/// Contract: estimate.
pub const NODES_PER_WORKER: usize = 2;

/// The packing slab the GEMM crate keeps per worker (bytes).
/// Used by: `memory::worker_bytes`.
/// Why: an allowance for the `gemm` crate's thread-local packing buffer.
/// Contract: estimate.
pub const GEMM_SLAB: usize = 1 << 20;
