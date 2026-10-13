//! The KLU path: replay planning of the parallel refactorization.

/// Replay work of a column beyond its fmadds (its scatter, pivot and
/// emitted `L` column) (fmadds).
/// Used by: `klu::factor::compute_replay_plan`, `KluSymbolic::memory_plan`.
/// Why: the fixed per-column cost of the replay loop relative to one fmadd.
/// Contract: timing (decides parallel vs sequential replay, same bits).
pub const COL_WORK: u64 = 32;

/// Mean replay work of a block's columns from which it replays pipelined
/// rather than level by level (fmadds per column): a pipelined column costs
/// a claim and a wait per `L` column it reads, which lighter columns do not
/// repay, while a level costs a parallel section, which a deep elimination
/// DAG of heavy columns pays far too often.
/// Used by: `klu::factor::replay_choice`.
/// Why: on the SuiteSparse circuits and SANE's c7552 stage matrix at 12
/// workers the columns carry either at most about 200 (c7552, circuit_4,
/// scircuit: level by level 1.5 to 2x the sequential replay, pipelined
/// slower than it) or at least about 1700 (rajat15, the ASICs, onetone2,
/// twotone: pipelined 3 to 7x, level by level barely faster than
/// sequential); 512 separates the two classes.
/// Contract: timing.
pub const PIPE_COL_WORK: u64 = 512;

/// Columns one worker claims at a time in the pipelined replay (columns).
/// Used by: `KluSolver::refactor` (pipelined replay).
/// Why: a column waits on the `L` columns it reads, so a longer run only
/// serializes columns that could overlap.
/// Contract: timing.
pub const PIPE_RUN: usize = 1;

/// Fewest columns per piece of a level in the level-by-level replay; a
/// level of fewer than two pieces runs on the calling thread (columns).
/// Used by: `KluSolver::refactor` (level replay).
/// Why: below it a parallel section costs more than the columns.
/// Contract: timing.
pub const LEVEL_PIECE: usize = 8;
