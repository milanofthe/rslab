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

/// KLU's diagonal-preference pivot threshold (1): the diagonal is the pivot
/// while `|a_jj| >= KLU_PIVOT_THRESHOLD * max_i |a_ij|`. Not the LU path's
/// [`LU_PIVOT_THRESHOLD`](super::LU_PIVOT_THRESHOLD): KLU keeps the
/// BTF/AMD-chosen diagonal unless it is numerically tiny.
/// Used by: `KluSettings::pivot_threshold`.
/// Why: SuiteSparse KLU's default.
/// Contract: results.
pub const KLU_PIVOT_THRESHOLD: f64 = 1e-3;

/// Nonzeros from which `KluParallel::Auto` factors blocks in parallel
/// (nonzeros).
/// Used by: `KluSettings::par_min_nnz`.
/// Why: tuned default.
/// Contract: timing (bit-identical in every mode).
pub const KLU_PAR_MIN_NNZ: usize = 8_000;

/// Replay work a unit of parallel refactorization must carry, in fmadds
/// plus [`COL_WORK`] a column (fmadds). Not the unit of the solves'
/// [`SOLVE_PAR_MIN_ENTRY_RHS`](super::SOLVE_PAR_MIN_ENTRY_RHS).
/// Used by: `KluSettings::par_min_work`.
/// Why: tuned default; below it starting the parallel section costs more
/// than the overlap gains.
/// Contract: timing.
pub const KLU_PAR_MIN_FMADDS: u64 = 200_000;

/// Simultaneous work the structure must offer for a parallel refactor:
/// across blocks `sum work / max block work`, inside a block the mean level
/// width of its elimination DAG (1).
/// Used by: `KluSettings::par_min_ratio`.
/// Why: at least two units of work side by side.
/// Contract: timing.
pub const KLU_PAR_MIN_RATIO: f64 = 2.0;
