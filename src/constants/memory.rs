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

/// Bytes of a narrow index: a `u32` row, column or slot (`Li`, `Ki`), a
/// global-to-local map entry (bytes).
/// Used by: the memory plans of the three paths, `memory::worker_bytes`,
/// the panel and solve plan sizes.
/// Why: `size_of::<u32>()`.
/// Contract: estimate.
pub const IDX32_BYTES: usize = 4;

/// Bytes of a word: a `usize` pointer or offset, a `u64`, an `f64`
/// scaling factor (bytes).
/// Used by: the memory plans of the three paths.
/// Why: `size_of::<usize>()` on the 64-bit targets the model describes.
/// Contract: estimate.
pub const WORD_BYTES: usize = 8;

/// Bytes of a `Vec` header: pointer, capacity and length (bytes).
/// Used by: the memory plans (per-node row lists), `solve::layout_size`.
/// Why: three words.
/// Contract: estimate.
pub const VEC_HEADER_BYTES: usize = 24;

/// The first capacity of a `Vec` of small elements filled by pushes
/// (entries).
/// Used by: `memory::pushed`, the KLU block buffers' growth model.
/// Why: the standard library's minimum non-zero capacity for elements of
/// at most 1 KiB.
/// Contract: estimate.
pub const VEC_MIN_CAP: usize = 4;

/// The input program a factorization builds once per analysis keeps, per
/// entry, its index (in the column or the row part) and its position (bytes
/// per entry).
/// Used by: `LdltSymbolic::memory_plan`, `LuSymbolic::memory_plan`.
/// Why: two `u32` per entry.
/// Contract: estimate.
pub const INPUT_PROGRAM_BYTES_PER_ENTRY: usize = 2 * IDX32_BYTES;

/// The peak of building that input program, per entry (bytes per entry).
/// Used by: `LdltSymbolic::memory_plan`, `LuSymbolic::memory_plan`.
/// Why: the build's temporaries as the model counts them.
/// Contract: estimate.
pub const INPUT_BUILD_BYTES_PER_ENTRY: usize = 28;

/// ... and per column, LDL^T (bytes per column).
/// Used by: `LdltSymbolic::memory_plan`.
/// Why: as for [`INPUT_BUILD_BYTES_PER_ENTRY`].
/// Contract: estimate.
pub const LDLT_INPUT_BUILD_BYTES_PER_COL: usize = 32;

/// ... and per column, LU (bytes per column).
/// Used by: `LuSymbolic::memory_plan`.
/// Why: as for [`INPUT_BUILD_BYTES_PER_ENTRY`].
/// Contract: estimate.
pub const LU_INPUT_BUILD_BYTES_PER_COL: usize = 48;

/// Per supernode while an LDL^T forest is factored: the emit cells
/// (refcount, offset, panel record, the pivots by column) and the forest's
/// own lists (bytes per supernode).
/// Used by: `LdltSymbolic::memory_plan`.
/// Why: the cells' sizes as the model counts them.
/// Contract: estimate.
pub const LDLT_NODE_CELL_BYTES: usize = 8 + 8 + 40 + 96 + 8 + 8 + 8;

/// Per supernode while an LU forest is factored: the emit cells (refcount,
/// offset, panel records, positions and permutations) and the forest's
/// lists (bytes per supernode).
/// Used by: `LuSymbolic::memory_plan`.
/// Why: the cells' sizes as the model counts them.
/// Contract: estimate.
pub const LU_NODE_CELL_BYTES: usize = 8 + 8 + 80 + 24 + 8 + 8 + 8;

/// Per column while an LU forest is factored: the positions and
/// permutations of the emit cells (bytes per column).
/// Used by: `LuSymbolic::memory_plan`.
/// Why: four words.
/// Contract: estimate.
pub const LU_NODE_CELL_BYTES_PER_COL: usize = 4 * WORD_BYTES;

/// Per column of an LU factor: both permutations and both scalings (bytes
/// per column).
/// Used by: `LuSymbolic::memory_plan`.
/// Why: four words.
/// Contract: estimate.
pub const LU_PERM_SCALE_BYTES_PER_COL: usize = 4 * WORD_BYTES;

/// Scratch pools of an LDL^T factorization: the panel kernel's, the
/// deep-row replay's, the `cmod`'s and the arena's (pools).
/// Used by: `LdltSymbolic::memory_plan`.
/// Why: one per pooled scratch type.
/// Contract: estimate.
pub const LDLT_SCRATCH_POOLS: usize = 4;

/// Scratch pools of an LU factorization: the node kernel's, the tiled
/// `cmod`'s and the arena's (pools).
/// Used by: `LuSymbolic::memory_plan`.
/// Why: one per pooled scratch type.
/// Contract: estimate.
pub const LU_SCRATCH_POOLS: usize = 3;

/// Per supernode while a solve plan is built: `sn_of`, the tree (parent,
/// children), work, heap, flags and slots (bytes per supernode).
/// Used by: `supernodal::solve::layout_size`.
/// Why: the build's arrays as the model counts them.
/// Contract: estimate.
pub const SOLVE_BUILD_NODE_BYTES: usize = 4 + 24 + 4 + 8 + 16 + 1 + 4 + 4 + 4;

/// Arrays of a KLU factor: `L`, `U` and the off-block entries `F`, each
/// with its column pointers (arrays).
/// Used by: `KluSymbolic::symbolic_fill_uncached`, `KluSymbolic::memory_plan`.
/// Why: the factor's structure.
/// Contract: estimate.
pub const KLU_FACTOR_ARRAYS: usize = 3;

/// The KLU drivers reserve `L` at this many times the input entries
/// (times).
/// Used by: `klu::factor` (`BlockOut::reset`, the sequential driver), and
/// the memory plan's model of them.
/// Why: tuned default; a block that outgrows it doubles the capacity.
/// Contract: timing (capacity only).
pub const KLU_L_RESERVE: usize = 4;

/// ... `U` at this many times (times).
/// Used by: as [`KLU_L_RESERVE`].
/// Why: as [`KLU_L_RESERVE`].
/// Contract: timing.
pub const KLU_U_RESERVE: usize = 2;

/// ... and the off-block entries at the input entries divided by this (1).
/// Used by: as [`KLU_L_RESERVE`].
/// Why: off-block entries are a small share on circuit matrices.
/// Contract: timing.
pub const KLU_F_RESERVE_DIV: usize = 4;

/// Per column of the largest block, a KLU worker's DFS scratch besides one
/// value (bytes per column).
/// Used by: `KluSymbolic::memory_plan`.
/// Why: the scratch arrays as the model counts them.
/// Contract: estimate.
pub const KLU_DFS_BYTES_PER_COL: u64 = 8 + 4 + 4 + 8 + 4 + 4 + 8;

/// Per BTF block, what the parallel driver adds besides the block's own
/// buffers (bytes per block).
/// Used by: `KluSymbolic::memory_plan`.
/// Why: an allowance for the per-block records.
/// Contract: estimate.
pub const KLU_PAR_BLOCK_BYTES: u64 = 64;
