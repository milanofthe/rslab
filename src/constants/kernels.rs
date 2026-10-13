//! The dense kernels of the supernodal LDL^T and LU factorizations.

/// Scale-invariant singularity floor for a 2x2 Bunch-Kaufman pivot (1): a
/// block whose `|det|` falls below `GROWTH_EPS * scale^2` (scale = the
/// largest block entry magnitude) is numerically singular, rejected in exact
/// mode and lifted in static-pivot mode. Bounds the element growth `1/|det|`
/// can otherwise inject into the trailing update.
/// Used by: `ldlt::bunch_kaufman` (2x2 pivot test and static-pivot lift).
/// Why: about 45 units of double roundoff.
/// Contract: results.
pub const GROWTH_EPS: f64 = 1e-14;

/// Bunch-Kaufman panel width of the LDL^T kernels (columns).
/// Used by: `KernelSettings::panel_nb`.
/// Why: tuned default; a global sweep found wider panels lose on small
/// nodes.
/// Contract: results (the pivot search window).
pub const PANEL_NB: usize = 64;

/// Narrowest Bunch-Kaufman panel the kernels accept; a smaller
/// `panel_nb` is raised to it (columns).
/// Used by: `SolverSettings::kernel`.
/// Why: guards the blocking against degenerate widths.
/// Contract: results (for settings below it).
pub const PANEL_NB_MIN: usize = 8;

/// Below this an update runs as a scalar loop instead of a SIMD GEMM
/// (flops).
/// Used by: `KernelSettings::scalar_gate`.
/// Why: tuned default; a GEMM call's setup outweighs tiny products.
/// Contract: rounding (scalar mul-add vs FMA GEMM).
pub const SCALAR_GATE: usize = 4096;

/// From this an update GEMM runs rayon-parallel (flops).
/// Used by: `KernelSettings::par_gemm`.
/// Why: tuned default.
/// Contract: rounding (selects the tiled update path).
pub const PAR_GEMM_MIN_FLOPS: usize = 1_000_000;

/// From this a panel's trailing update runs rayon-parallel (flops).
/// Used by: `KernelSettings::par_cdiv`.
/// Why: tuned default.
/// Contract: rounding.
pub const PAR_CDIV_MIN_FLOPS: usize = 8_000_000;

/// From this the updates of one supernode fork across workers (flops). A
/// small node that forks pays the join-steal latency: its waiting thread
/// steals other work, often a whole sibling subtree, and the node's chain
/// stalls.
/// Used by: `KernelSettings::fork_min_flops`.
/// Why: tuned default; a forking small node stalls its chain (74 ms of
/// cmod at 0.03 Gflop measured on a 1046x170 node).
/// Contract: rounding (selects the tiled update path).
pub const FORK_MIN_FLOPS: usize = 100_000_000;

/// Column tile of the lower-triangular Schur GEMM (columns).
/// Used by: `KernelSettings::schur_tile`.
/// Why: tuned default.
/// Contract: rounding.
pub const SCHUR_TILE: usize = 256;

/// Columns per sub-block of the deferred trailing sweep in the
/// Bunch-Kaufman panel, the rank of its GEMMs (columns).
/// Used by: `KernelSettings::trailing_block`.
/// Why: 16 measured best; 32 doubles the scalar within-block work, 8
/// halves the GEMM efficiency.
/// Contract: rounding.
pub const TRAILING_BLOCK: usize = 16;

/// A complex product runs as real products on split planes when its flops
/// `m n k` are at least this many times its plane copies `m k + k n + m n`
/// (1); thin products lose on the copies.
/// Used by: `KernelSettings::complex_split_min_ratio`.
/// Why: tuned default.
/// Contract: rounding.
pub const COMPLEX_SPLIT_MIN_RATIO: usize = 64;

/// Tile edge of the complex split, which bounds its scratch (entries).
/// Used by: `KernelSettings::complex_split_tile`.
/// Why: tuned default.
/// Contract: rounding.
pub const COMPLEX_SPLIT_TILE: usize = 256;
