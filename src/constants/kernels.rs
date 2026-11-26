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

/// From this an update GEMM takes the parallel path, on as many threads as
/// its work is worth (flops).
/// Used by: `KernelSettings::par_gemm`.
/// Why: tuned default.
/// Contract: rounding (selects the tiled update path).
pub const PAR_GEMM_MIN_FLOPS: usize = 1_000_000;

/// From this a panel's trailing update takes the parallel path, on as many
/// threads as its work is worth (flops).
/// Used by: `KernelSettings::par_cdiv`.
/// Why: tuned default.
/// Contract: rounding.
pub const PAR_CDIV_MIN_FLOPS: usize = 8_000_000;

/// From this the updates of one supernode take the forking path, on as many
/// threads as their work is worth (flops). A small node that forks pays
/// the join-steal latency: its waiting thread steals other work, often a
/// whole sibling subtree, and the node's chain stalls.
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

/// Rows of the off-block part of an LU pivot column brought up to date at a
/// time from the panel's earlier pivots (rows).
/// Used by: `numeric::lu::node::deep_update`.
/// Why: a run of complex doubles stays in the L1 cache across the panel's
/// up to 32 columns.
/// Contract: timing (each row takes the same operations in any run).
pub const LU_DEEP_ROWS: usize = 256;

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

/// Nodes with at least this many columns take wide cdiv panels of at least
/// [`WIDE_PANEL_NB`] columns, LDL^T and LU alike (columns).
/// Used by: `KernelSettings::wide_panel_min_ncol`.
/// Why: the deferred Schur GEMM's inner dimension is the panel width, and
/// 64 is too thin to reach peak on root-class panels (about 79 Gflop/s
/// measured); a global wider width loses because small panels pay, so only
/// nodes from this width widen. A function of the node alone, never of the
/// thread count.
/// Contract: results (the pivot window of the wide nodes).
pub const WIDE_PANEL_MIN_NCOL: usize = 512;

/// Least cdiv panel width of a wide node (columns).
/// Used by: `KernelSettings::wide_panel_nb`.
/// Why: see [`WIDE_PANEL_MIN_NCOL`]; the extra serial panel work is
/// `O(nb^3)`, negligible against the GEMM gain at this size.
/// Contract: results.
pub const WIDE_PANEL_NB: usize = 128;

/// cdiv panel width of the LU path below [`WIDE_PANEL_MIN_NCOL`] (columns).
/// Used by: `KernelSettings::lu_panel_nb`.
/// Why: swept 32/48/64/96 on the MoM fronts, 32 is best for typical panels.
/// Contract: results (the pivot window).
pub const LU_PANEL_NB: usize = 32;

/// Rows per task of the parallel `U12` triangular solve in the LU cdiv
/// (rows).
/// Used by: `KernelSettings::trsm_rows`.
/// Why: tuned default; the rows are independent.
/// Contract: timing (each row is solved alike in any blocking).
pub const TRSM_ROWS: usize = 256;

/// Real planes per complex operand of a split product: the real part, the
/// imaginary part and their sum, the three products of the 3M method
/// (planes).
/// Used by: `dense::gemm_backend::complex_gemm_3m`,
/// `dense::gemm_backend::split_plane_entries`.
/// Why: the method's definition.
/// Contract: definition.
pub const SPLIT_PLANES: usize = 3;

/// Row-length buffers of the LDL^T panel kernel: the two pivot columns and
/// their lookahead twins (buffers).
/// Used by: `ldlt::bunch_kaufman::ll_cdiv_scratch`.
/// Why: a 2x2 pivot reads two columns, the lookahead holds the next two.
/// Contract: estimate.
pub const BK_ROW_BUFFERS: usize = 4;

/// Columns per block of the diagonal part of a supernode's panel, the one
/// layout the factorizations write and the solves read
/// (`supernodal::panel::Layout`): each block holds its columns' rows from
/// its first column to the panel's width, the off-block rows of all columns
/// form one rectangle after the blocks (columns).
/// Used by: `supernodal::panel::Layout`, the sweeps' triangles in column
/// blocks (`supernodal::solve`), the tiles of the trailing updates.
/// Why: each block's triangle runs on the register tiles of the sweeps and
/// the rows below it take one product; 64 keeps the triangles a small share
/// (the unused upper part of a block's own square) and the products wide
/// enough for the SIMD GEMM.
/// Contract: rounding.
pub const PANEL_BLOCK: usize = 64;

/// The 4 of the discriminant `fro^2 - 4 det^2` of the quadratic
/// `sigma^4 - fro sigma^2 + det^2 = 0` whose roots are the squared singular
/// values of a 2x2 pivot block of `D`, `fro` its squared Frobenius norm and
/// `det` the magnitude of its determinant.
/// Used by: `ldlt::solver` (the smallest pivot in the diagnostics).
/// Why: the discriminant `b^2 - 4ac` of a quadratic.
/// Contract: estimate.
pub const SINGULAR_VALUE_DISCRIMINANT: f64 = 4.0;
