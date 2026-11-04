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

/// Mean replay work of a block's columns from which a refactor starts it
/// pipelined rather than level by level, before any of its ways is
/// measured (fmadds per column): a pipelined column costs a claim and a
/// wait per `L` column it reads, which lighter columns do not repay, while a
/// level costs a parallel section, which a deep elimination DAG of heavy
/// columns pays far too often.
/// Used by: `KluSolver::refactor` (the ways a block starts from).
/// Why: on the SuiteSparse circuits and SANE's c7552 stage matrix at 12
/// workers the columns carry either at most about 200 (c7552, circuit_4,
/// scircuit: level by level 1.5 to 2x the sequential replay, pipelined
/// slower than it) or at least about 1700 (rajat15, the ASICs, onetone2,
/// twotone: pipelined 3 to 7x, level by level barely faster than
/// sequential); 512 separates the two classes.
/// Machine: a prior, replaced by what the replays measure.
/// Contract: timing.
pub const PIPE_COL_WORK: u64 = 512;

/// Replay work from which a refactor starts a block side by side, before
/// any of its ways is measured, where the user set no floor (fmadds plus
/// [`COL_WORK`] a column).
/// Used by: `KluSolver::refactor` (the ways a block starts from).
/// Why: below it starting the parallel section costs more than the overlap
/// gains (the default floor of 1.13).
/// Machine: a prior, replaced by what the replays measure.
/// Contract: timing.
pub const KLU_PAR_MIN_FMADDS: u64 = 200_000;

/// Mean level width of a block's elimination DAG from which a refactor
/// starts it side by side, before any of its ways is measured, where the
/// user set no ratio (columns).
/// Used by: `KluSolver::refactor` (the ways a block starts from).
/// Why: at least two units of work side by side.
/// Machine: a prior, replaced by what the replays measure.
/// Contract: timing.
pub const KLU_PAR_MIN_RATIO: f64 = 2.0;

/// ns per unit of replay work (an fmadd, [`COL_WORK`] a column) of a real
/// refactorization, before any is measured (ns per fmadd).
/// Used by: `KluSolver::refactor`.
/// Why: a scattered multiply-add with its index load.
/// Machine: a prior, replaced by what the replays measure.
/// Contract: timing.
pub const KLU_REPLAY_NS_PER_WORK: f32 = 0.5;

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
