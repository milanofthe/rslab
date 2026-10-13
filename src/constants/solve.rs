//! The supernodal triangular solves.

/// Rows per register tile of the panel products of the solves (rows).
/// Used by: `supernodal::solve::product_tile`, `supernodal::solve::tri_tile`.
/// Why: a tile of four rows by up to four right-hand sides stays in registers.
/// Contract: results (the tile fixes the summation order).
pub const TILE_ROWS: usize = 4;

/// Independent leaf subtrees the tree is cut into for the parallel sweeps
/// (subtrees).
/// Used by: `SolveSettings::leaf_subtrees`.
/// Why: tuned default; many more than workers, for balance.
/// Contract: timing (the solves are thread-count independent).
pub const SOLVE_LEAF_SUBTREES: usize = 128;

/// Column block of the ancestor sweeps (columns).
/// Used by: `SolveSettings::block`.
/// Why: tuned default.
/// Contract: rounding (the blocking orders the sums).
pub const SOLVE_BLOCK: usize = 512;

/// Columns per product or dot task of the ancestor sweeps (columns).
/// Used by: `SolveSettings::ancestor_chunk`.
/// Why: tuned default.
/// Contract: rounding.
pub const SOLVE_ANCESTOR_CHUNK: usize = 32;

/// Panel entries from which an ancestor node uses the blocked sweep
/// (entries).
/// Used by: `SolveSettings::apex_min_work`.
/// Why: tuned default.
/// Contract: rounding.
pub const SOLVE_APEX_MIN_ENTRIES: usize = 1 << 18;

/// Work of a triangular sweep from which it runs on the thread pool, in
/// factor panel entries times right-hand sides, the latter counted up to
/// four (entries x rhs). Not the unit of the KLU path's
/// [`KLU_PAR_MIN_FMADDS`](super::KLU_PAR_MIN_FMADDS).
/// Used by: `SolveSettings::par_min_work`.
/// Why: about where handing a sweep to the pool starts to pay; a smaller
/// one costs several times the solve there at a few thousand unknowns.
/// Contract: timing (the same bits either way).
pub const SOLVE_PAR_MIN_ENTRY_RHS: usize = 1 << 20;
