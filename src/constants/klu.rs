//! The KLU path: the work of the parallel factorization and refactorization.

/// Replay work of a column beyond its fmadds (its scatter, pivot and
/// emitted `L` column) (fmadds).
/// Used by: `klu::factor::compute_replay_plan`, `KluSymbolic::memory_plan`.
/// Why: the fixed per-column cost of the replay loop relative to one fmadd.
/// Contract: timing (the work a refactor decides its parallel replay on,
/// same bits).
pub const COL_WORK: u64 = 32;

/// Columns one worker claims at a time in the pipelined replay (columns).
/// Used by: `KluSolver::refactor` (pipelined replay).
/// Why: a column waits on the `L` columns it reads, so a longer run only
/// serializes columns that could overlap.
/// Contract: timing.
pub const PIPE_RUN: usize = 1;

/// ns per unit of replay work (an fmadd, [`COL_WORK`] a column) of a real
/// refactorization, before any is measured (ns per fmadd).
/// Used by: `KluSolver::refactor`.
/// Why: a scattered multiply-add with its index load.
/// Machine: a prior, replaced by what the replays measure.
/// Contract: timing.
pub const KLU_REPLAY_NS_PER_WORK: f32 = 0.5;

/// ns a pipelined replay spends per column and per `L` column a column
/// reads beyond the replay itself (the claim, the wait for the column it
/// reads, the line it moves between cores), before any is measured (ns per
/// hand-off).
/// Used by: `KluSolver::refactor` (whether a block replays pipelined).
/// Why: about a cache line moved between cores, where the column read is
/// not done yet; it is what makes light columns slower pipelined than
/// sequentially.
/// Machine: a prior, replaced by what the pipelined replays measure.
/// Contract: timing.
pub const KLU_HANDOFF_NS: f32 = 50.0;

/// ns per input entry of a block's first factorization (the reach, the
/// pivot search, the fill), before any is measured (ns per entry).
/// Used by: `KluSymbolic::factor` (whether its blocks factor side by side).
/// Why: a few fmadds of fill and a depth-first reach per entry.
/// Machine: a prior, replaced by what the factorizations measure.
/// Contract: timing.
pub const KLU_FACTOR_NS_PER_ENTRY: f32 = 25.0;

/// KLU's diagonal-preference pivot threshold (1): the diagonal is the pivot
/// while `|a_jj| >= KLU_PIVOT_THRESHOLD * max_i |a_ij|`. Not the LU path's
/// [`LU_PIVOT_THRESHOLD`](super::LU_PIVOT_THRESHOLD): KLU keeps the
/// BTF/AMD-chosen diagonal unless it is numerically tiny.
/// Used by: `KluSettings::pivot_threshold`.
/// Why: SuiteSparse KLU's default.
/// Contract: results.
pub const KLU_PIVOT_THRESHOLD: f64 = 1e-3;
