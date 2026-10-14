//! The options of the KLU path.

use super::*;
use crate::constants::KLU_PIVOT_THRESHOLD;

/// Options for the KLU path. Defaults follow SuiteSparse KLU: threshold
/// partial pivoting with strong diagonal preference (`pivot_threshold =
/// 1e-3`), row-max scaling on, BTF on.
#[derive(Debug, Clone)]
pub struct KluSettings {
    /// Threshold for diagonal preference: the diagonal entry is taken as the
    /// pivot when `|a_jj| >= pivot_threshold * max_i |a_ij|` over the
    /// eligible column. `1.0` is plain partial pivoting; small values keep
    /// the BTF/AMD-chosen diagonal (less fill) unless it is numerically tiny.
    pub pivot_threshold: f64,
    /// Divide every row by its max-magnitude entry before factoring (and
    /// scale RHS/solution accordingly). Cheap and markedly more robust on
    /// badly row-equilibrated inputs.
    pub row_scaling: bool,
    /// Permute to block upper triangular form first. Disable only for
    /// experiments; without BTF the whole matrix is one block, structural
    /// singularity surfaces as a numeric zero pivot, and the diagonal
    /// preference loses its zero-free guarantee.
    pub btf: bool,
    /// Parallel execution of factor and refactor over the (independent) BTF
    /// diagonal blocks, and of a refactor inside a block (level by level or
    /// pipelined), on the current workers ([`crate::parallel`]).
    /// **Bit-identical to sequential in every mode**: each column is
    /// replayed and each block factored sequentially by construction, so the
    /// result does not depend on scheduling or thread count. The default
    /// `Auto` goes side by side, and on as many threads, as the work is
    /// worth on the pool it runs on: the work (input entries for a factor,
    /// the exact replay work of the pivot-final pattern for a refactor) at
    /// the rates measured on earlier runs, against what a section costs on
    /// the pool ([`crate::parallel::Dispatch`]), decided anew by every
    /// refactor with the pool it is given. Real circuits are often one
    /// giant irreducible block plus thousands of singletons, where
    /// distributing blocks cannot help, and small systems refactor faster
    /// on one thread than on any pool. Run inside a bounded rayon pool to
    /// cap it for solver-in-the-loop use, or force `Off` for strictly
    /// sequential execution.
    pub parallel: KluParallel,
    /// Maximum-product row matching (MC64) as the transversal of the block
    /// triangular form: the matched, largest-product entries become the
    /// diagonal, so the diagonal-preference pivoting rarely has to leave
    /// it. The structural transversal only guarantees a zero-free
    /// diagonal; on the ibmpg1 power grid it leaves 14k of 45k columns to
    /// off-diagonal pivots and the fill at seven times the symbolic
    /// estimate. Analysis-time (value dependent); a `refactor` keeps the
    /// matching. Default `true`; needs `btf`.
    pub matching: bool,
    /// A fixed structural gate for the first factor in place of the time
    /// gate of [`KluParallel::Auto`]: blocks factor side by side from this
    /// many input nonzeros on, where no block holds more than half the
    /// columns. Default `None`: as the work is worth on the pool.
    pub par_min_nnz: Option<usize>,
    /// A fixed structural gate for the refactor in place of the time gate of
    /// [`KluParallel::Auto`] (with [`par_min_ratio`](Self::par_min_ratio)):
    /// the replay work, in fmadds and 32 a column, a block must carry to
    /// replay its columns side by side, and all blocks together to replay
    /// side by side (not the panel entries of
    /// [`SolveSettings::par_min_work`](crate::SolveSettings::par_min_work)).
    /// Where it passes, the replay goes as wide as the pool. Default
    /// `None`: as the work is worth on the pool.
    pub par_min_work: Option<u64>,
    /// With [`par_min_work`](Self::par_min_work), the simultaneous work the
    /// structure must offer for a parallel refactor: across blocks `sum
    /// work / max block work`, inside a block the mean level width of its
    /// elimination DAG. Default `None`: as the work is worth on the pool.
    pub par_min_ratio: Option<f64>,
    /// Caller-owned cancellation flag for the numeric phase, read and never
    /// written, polled at block boundaries and inside the pipelined refactor at
    /// column boundaries. The flag armed for a factorization is carried into
    /// the factors, so a later [`refactor`](KluSolver::refactor) on them
    /// observes the same flag. **Default `None`**.
    pub interrupt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Parallel per-block execution policy for the KLU path
/// (see [`KluSettings::parallel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KluParallel {
    /// Side by side, and on as many threads, as the work is worth on the
    /// pool it runs on (or as the structural gates
    /// [`par_min_nnz`](KluSettings::par_min_nnz),
    /// [`par_min_work`](KluSettings::par_min_work) and
    /// [`par_min_ratio`](KluSettings::par_min_ratio) decide, where set).
    #[default]
    Auto,
    /// Always parallel where the structure allows it, on every thread of
    /// the pool (still bit-identical).
    On,
    /// Strictly sequential.
    Off,
}

impl Default for KluSettings {
    fn default() -> Self {
        Self {
            pivot_threshold: KLU_PIVOT_THRESHOLD,
            row_scaling: true,
            btf: true,
            parallel: KluParallel::Auto,
            matching: true,
            par_min_nnz: None,
            par_min_work: None,
            par_min_ratio: None,
            interrupt: None,
        }
    }
}

impl KluSettings {
    /// Builder: arm the numeric phase with a caller-owned cancellation flag.
    pub fn with_interrupt(mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.interrupt = Some(flag);
        self
    }

    /// Poll the caller's flag; `Ok(())` when unarmed or clear.
    #[inline]
    pub(crate) fn interrupted(&self) -> Result<(), RslabError> {
        interrupt_check(self.interrupt.as_deref())
    }

    /// Set the diagonal-preference threshold (see
    /// [`pivot_threshold`](Self::pivot_threshold)). `1.0` is plain partial
    /// pivoting.
    pub fn with_pivot_threshold(mut self, u: f64) -> Self {
        self.pivot_threshold = u;
        self
    }

    /// Composable toggle for row-max scaling
    /// (see [`row_scaling`](Self::row_scaling)).
    pub fn with_row_scaling(mut self, on: bool) -> Self {
        self.row_scaling = on;
        self
    }

    /// Enable or disable the MC64 row matching (see
    /// [`KluSettings::matching`]).
    pub fn with_matching(mut self, on: bool) -> Self {
        self.matching = on;
        self
    }

    /// Composable toggle for the BTF permutation (see [`btf`](Self::btf)).
    pub fn with_btf(mut self, on: bool) -> Self {
        self.btf = on;
        self
    }

    /// Composable setter for the parallel per-block policy
    /// (see [`parallel`](Self::parallel)).
    pub fn with_parallel(mut self, p: KluParallel) -> Self {
        self.parallel = p;
        self
    }
}
