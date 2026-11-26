//! The KLU path: the work of the parallel factorization and refactorization.

/// Replay work of a column beyond its fmadds (its scatter, pivot and
/// emitted `L` column) (fmadds).
/// Used by: `klu::factor::compute_replay_plan`, `KluSymbolic::memory_plan`.
/// Why: the fixed per-column cost of the replay loop relative to one fmadd.
/// Contract: timing (the work a refactor decides its parallel replay on,
/// same bits).
pub const COL_WORK: u64 = 32;

/// Columns of a supernode the dense kernel of the KLU refactorization and
/// solve applies in one pass over the gathered rows (columns): the kernel's
/// unroll, part of its structure, not a tuned gate.
/// Used by: `klu::supernode::apply`, `klu::supernode::dense_pays`.
/// Why: the gathered rows are loaded and stored once per pass; four
/// multiply-adds a row keep the pass bound by the `L` values it reads.
/// Contract: timing (the order of the additions into a row, last bits).
pub const KLU_SN_PASS: usize = 4;

/// Memory accesses of one `L` entry in the column-by-column replay of a
/// supernode (accesses): its row index, its `L` value, the read and the
/// write of the work vector entry.
/// Used by: `klu::supernode::dense_pays`, which charges `4 (t + w s)` for a
/// supernode of `w` columns over `s` shared rows with `t = w (w - 1) / 2`
/// entries in its triangle.
/// Why: the count of the accesses of the replay loop.
/// Contract: timing (the order of the additions into a row, last bits).
pub const KLU_SN_COLUMN_ACCESSES: usize = 4;

/// Memory accesses of one triangle entry in the dense kernel of a
/// supernode (accesses): its `L` value, the read and the write of the work
/// vector entry. A shared-row entry costs only its `L` value.
/// Used by: `klu::supernode::dense_pays`.
/// Why: the count of the accesses of the triangle loop of
/// `klu::supernode::apply`.
/// Contract: timing (the order of the additions into a row, last bits).
pub const KLU_SN_TRIANGLE_ACCESSES: usize = 3;

/// Memory accesses of one shared row of a supernode in one pass of the
/// dense kernel (accesses): the read and the write of `tmp`. A supernode of
/// `w` columns takes `w / P + w % P` passes with `P` = [`KLU_SN_PASS`].
/// Used by: `klu::supernode::dense_pays`.
/// Why: the count of the accesses of a pass of `klu::supernode::apply`.
/// Contract: timing (the order of the additions into a row, last bits).
pub const KLU_SN_PASS_ACCESSES: usize = 2;

/// Memory accesses of one shared row of a supernode for its gather and its
/// scatter in the dense kernel (accesses): the gather reads the index and
/// the work vector entry and writes `tmp`, the scatter reads the index and
/// `tmp`, writes the work vector entry and clears `tmp`.
/// Used by: `klu::supernode::dense_pays`. With the other access counts a
/// supernode of four columns or more is always applied densely, and one of
/// two or three columns only without shared rows.
/// Why: the count of the accesses of the gather and the scatter of
/// `klu::supernode::apply`.
/// Contract: timing (the order of the additions into a row, last bits).
pub const KLU_SN_GATHER_SCATTER_ACCESSES: usize = 7;

/// Columns one worker claims at a time in the pipelined replay (columns).
/// Used by: `KluSolver::refactor` (pipelined replay).
/// Why: a column waits on the `L` columns it reads, so a longer run only
/// serializes columns that could overlap.
/// Contract: timing.
pub const PIPE_RUN: usize = 1;

/// Mean replay work of a block's columns from which a refactor starts it
/// pipelined rather than level by level (fmadds per column): a pipelined column costs a claim and a
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
/// Machine: an estimate, not measured at run time.
/// Contract: timing.
pub const PIPE_COL_WORK: u64 = 512;

/// Replay work from which a refactor starts a block side by side, where
/// the user set no floor (fmadds plus
/// [`COL_WORK`] a column).
/// Used by: `KluSolver::refactor` (the ways a block starts from).
/// Why: below it starting the parallel section costs more than the overlap
/// gains (the default floor of 1.13).
/// Machine: an estimate, not measured at run time.
/// Contract: timing.
pub const KLU_PAR_MIN_FMADDS: u64 = 200_000;

/// Mean level width of a block's elimination DAG from which a refactor
/// starts it side by side, where the
/// user set no ratio (columns).
/// Used by: `KluSolver::refactor` (the ways a block starts from).
/// Why: at least two units of work side by side.
/// Machine: an estimate, not measured at run time.
/// Contract: timing.
pub const KLU_PAR_MIN_RATIO: f64 = 2.0;

/// ns per unit of replay work (an fmadd, [`COL_WORK`] a column) of a real
/// refactorization (ns per fmadd).
/// Used by: `KluSolver::refactor`.
/// Why: a scattered multiply-add with its index load.
/// Machine: an estimate, not measured at run time.
/// Contract: timing.
pub const KLU_REPLAY_NS_PER_WORK: f32 = 0.5;

/// ns per input entry of a block's first factorization (the reach, the
/// pivot search, the fill), in ns per entry.
/// Used by: `KluSymbolic::factor` (whether its blocks factor side by side).
/// Why: a few fmadds of fill and a depth-first reach per entry.
/// Machine: an estimate, not measured at run time.
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
