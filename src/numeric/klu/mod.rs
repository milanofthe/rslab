//! KLU-style sparse LU: BTF + per-block left-looking Gilbert-Peierls.
//!
//! The third direct path next to the supernodal LDL^T and LU, built for
//! circuit-shaped matrices: extremely sparse, unsymmetric, near-triangularizable,
//! with diagonal blocks far too small for supernodal/BLAS-3 kernels to pay off.
//! Algorithmic reference: SuiteSparse KLU (Davis & Palamadai Natarajan); this is
//! an independent pure-Rust implementation, no FFI.
//!
//! Pipeline:
//!
//! 1. **Analyze** ([`KluSymbolic::analyze`]): maximum transversal + Tarjan SCC
//!    ([`crate::ordering::btf`]) permute the matrix to block *upper* triangular
//!    form with a zero-free diagonal (structural singularity is detected here),
//!    then AMD orders each irreducible diagonal block on its symmetrized
//!    pattern.
//! 2. **Factor** ([`KluSymbolic::factor`]): each diagonal block is factored by
//!    a left-looking Gilbert-Peierls LU, per-column depth-first reach on the
//!    growing L pattern, so the numeric work is proportional to the flop count,
//!    with threshold partial pivoting that prefers the (structurally nonzero)
//!    diagonal. Off-block entries are not factored; they only enter the block
//!    back-substitution. Optional row-max scaling equilibrates the rows first.
//! 3. **Refactor** ([`KluSolver::refactor`]): numeric-only re-factorization for
//!    a matrix with the *same* pattern (frequency sweeps, Newton steps): the
//!    stored pattern and pivot sequence are replayed with no symbolic work and
//!    no pivot search. A changed pattern is detected and rejected; a pivot that
//!    became zero under the frozen pivot order fails cleanly so the caller can
//!    re-[`factor`](KluSymbolic::factor) with pivoting.
//!
//! Each block factors sequentially, and independent blocks may factor in
//! parallel ([`KluParallel`]); the results are bit-identical across runs and
//! thread counts.

use crate::constants::{IDX32_BYTES, KLU_L_RESERVE, KLU_U_RESERVE, VEC_MIN_CAP, WORD_BYTES};
use crate::error::RslabError;
use crate::numeric::supernodal::PanelPtr;
use crate::ordering::btf;
use crate::scalar::{fmadd, Scalar};
use crate::sparse::general::GeneralCsc;

mod factor;
mod refactor;
mod settings;
mod solver;
mod symbolic;
#[cfg(test)]
mod tests;

use factor::{diagnostics_flops, factor_csc, factor_impl, row_scale_inv_into};
pub use settings::{KluParallel, KluSettings};

const UNSET: usize = usize::MAX;

/// Narrow index type for the numeric factor's row-index streams and the
/// factor-time DFS state. The Gilbert-Peierls kernel is memory-bound on
/// index chasing; 32-bit indices halve that traffic (`n < 2^32 - 1` is
/// enforced at analyze time, far beyond this path's design point).
type Ki = u32;
const KI_UNSET: Ki = Ki::MAX;
/// Tag bit in the refactor scatter program: the entry lands in an F slot
/// (off-block value) rather than the elimination work vector.
const KI_FBIT: Ki = 1 << (Ki::BITS - 1);

/// Symbolic analysis for the KLU path: the BTF block structure plus the
/// per-block fill-reducing ordering. Analyze once, then factor any number of
/// matrices sharing the pattern.
#[derive(Debug, Clone)]
pub struct KluSymbolic {
    n: usize,
    nnz: usize,
    /// Pre-pivot row permutation (new-to-old): BTF matching then SCC order then
    /// per-block AMD. Partial pivoting at factor time refines this within
    /// each block.
    pre_row_perm: Vec<usize>,
    /// Column permutation (new-to-old); never changed by pivoting.
    col_perm: Vec<usize>,
    /// Diagonal-block boundaries; see [`crate::ordering::btf::BtfForm`].
    block_ptr: Vec<usize>,
    /// The analyzed pattern in the pre-pivot permuted space (column `k` is
    /// original column `col_perm[k]`, rows are pre-pivot positions). Kept so
    /// the a-priori estimators run without the matrix, like
    /// [`LuSymbolic`](crate::LuSymbolic)'s stored symbolic structure.
    pat_col_ptr: Vec<usize>,
    pat_row_idx: Vec<usize>,
    /// The factor the analysis predicts, computed with the ordering (see
    /// [`KluSymbolic::estimated_factor_nnz`]).
    predicted: KluEstimate,
    /// Lazily computed, cached symbolic fill (the estimator pass costs about
    /// as much as a numeric factor, so the phased `factor` must not pay it
    /// again on every call).
    fill: std::sync::OnceLock<KluFill>,
}

/// The KLU factor as the analysis predicts it a priori, from the blocks'
/// symmetrized Cholesky structure (see [`KluSymbolic::estimated_factor_nnz`]).
#[derive(Debug, Clone, Copy, Default)]
struct KluEstimate {
    factor_nnz: u64,
    flops: u64,
}

/// Exact symbolic fill of the KLU factor under the diagonal-pivoting
/// assumption (the default expectation: BTF guarantees a structurally nonzero
/// diagonal and `pivot_threshold` strongly prefers it). Threshold pivoting at factor
/// time can shift individual counts, not their order of magnitude.
#[derive(Debug, Clone)]
struct KluFill {
    l_nnz: u64,
    u_nnz: u64,
    f_nnz: u64,
    /// Gilbert-Peierls flop count (multiply-subtract pairs + divisions).
    flops: u64,
    /// The factor's buffers as the drivers grow them, each as `(values,
    /// other bytes)`: the sequential driver's appended arrays and its reused
    /// block buffer, and the parallel driver's per-block buffers summed.
    /// Independent of the scalar type, which prices the values.
    seq_arrays: (u64, u64),
    seq_block: (u64, u64),
    par_blocks: (u64, u64),
    /// Per block its replay work and the levels of its elimination DAG, as
    /// the refactorization's plan reads them (`factor::planned`).
    replay: Vec<(u64, usize)>,
}

/// Entry capacities of one block's output buffer after the driver factored
/// a block of `bn` columns, `annz` input entries and `l`, `u`, `f` factor
/// and off-block entries into it: mirrors `BlockOut::reset` (its reserves,
/// skipped for a singleton) and the kernel's pushes.
#[derive(Debug, Clone, Copy, Default)]
struct BlockCaps {
    l: usize,
    u: usize,
    f: usize,
    diag: usize,
    prog: usize,
    fin: usize,
    colptr: [usize; 3],
}

impl BlockCaps {
    fn fill(&mut self, bn: usize, annz: usize, l: usize, u: usize, f: usize) {
        use crate::memory::grown;
        let push = |cap: usize, len: usize| {
            let mut c = cap;
            while c < len {
                c = (2 * c).max(VEC_MIN_CAP);
            }
            c
        };
        let reserve = if bn == 1 { 0 } else { annz };
        self.fin = grown(self.fin, bn);
        self.l = push(grown(self.l, KLU_L_RESERVE * reserve), l);
        self.u = push(grown(self.u, KLU_U_RESERVE * reserve), u);
        self.f = push(self.f, f);
        self.diag = push(grown(self.diag, bn), bn);
        self.prog = push(grown(self.prog, reserve), annz - f);
        for c in &mut self.colptr {
            *c = push(*c, bn + 1);
        }
    }

    /// `(values, other bytes)`: `L`, `U` and off-block entries (a value and
    /// a narrow index each, off-block entries also their input position) and
    /// the diagonal; the scatter pairs, final positions and column pointers.
    fn bytes(&self) -> (u64, u64) {
        let vals = self.l + self.u + self.f + self.diag;
        let other = IDX32_BYTES * (self.l + self.u + 2 * self.f)
            + WORD_BYTES * self.prog
            + IDX32_BYTES * self.fin
            + WORD_BYTES * self.colptr.iter().sum::<usize>();
        (vals as u64, other as u64)
    }
}

/// The numeric KLU factorization: `P A Q = L U` per diagonal block plus the
/// off-block entries, with row scaling folded in.
#[derive(Debug, Clone)]
struct KluFactors<T> {
    /// The flag armed for the factorization, carried so `refactor` polls it too.
    interrupt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    n: usize,
    nnz_a: usize,
    block_ptr: Vec<usize>,
    /// Final row permutation (new-to-old), pivoting included.
    row_perm: Vec<usize>,
    /// Inverse: original row -> final position.
    pinv: Vec<usize>,
    col_perm: Vec<usize>,
    /// Per-original-row reciprocal scale factor (all 1 when scaling is off).
    rs_inv: Vec<f64>,
    scaled: bool,
    /// Whether the first factor's blocks were worth factoring side by side
    /// (decided on the blocks' input entries, and run so once the rate has
    /// had its probes; a refactor decides on the exact replay work in
    /// `plan` instead). Read by tests only.
    #[cfg_attr(not(test), allow(dead_code))]
    parallel: bool,
    /// L: strictly-below-diagonal entries per column, unit diagonal implicit.
    /// Row indices are final positions within the column's block (narrow
    /// [`Ki`] indices: the solve/refactor loops are index-bound).
    l_colptr: Vec<usize>,
    l_rowidx: Vec<Ki>,
    l_val: Vec<T>,
    /// U: strictly-above-diagonal within-block entries per column, stored in
    /// elimination (topological) order, the refactor replay order.
    u_colptr: Vec<usize>,
    u_rowidx: Vec<Ki>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    /// Off-block entries (rows in earlier blocks, final positions), per
    /// column in the input's storage order. Not factored; applied in the
    /// block back-substitution.
    f_colptr: Vec<usize>,
    f_rowidx: Vec<Ki>,
    f_val: Vec<T>,
    /// Refactor scatter program, aligned with the input's storage order.
    /// For entry `k` of the pattern-frozen matrix: `scatter_expect[k]` is
    /// the final row position recorded at factor time (`pinv[row]`), the
    /// branch-free pattern check; `scatter_target[k]` encodes the value's
    /// destination: F slot `i` as `KI_FBIT | i`, else work-vector position.
    scatter_expect: Vec<Ki>,
    scatter_target: Vec<Ki>,
    /// The work of the refactorization's replay, from the pivot-final
    /// pattern, which every refactor decides its parallel replay on with the
    /// pool it runs on ([`ReplayPlan`]).
    plan: ReplayPlan,
    /// The next refactor's row scaling, swapped with `rs_inv` once it
    /// succeeds, and the sequential replay's work vector: kept so a
    /// refactor allocates nothing.
    rs_next: Vec<f64>,
    replay_x: Vec<T>,
    /// The parallel replays' work vectors, one per worker, all-zero.
    replay_xs: Vec<Vec<T>>,
    /// Which columns the current pipelined replay has done.
    ready: Ready,
    /// How the last refactor decided to replay each block of `plan.blocks`
    /// (it replayed them sequentially instead while the replay rate had its
    /// probes, and when it explored).
    modes: Vec<Mode>,
}

/// How a refactor replays a block of the [`ReplayPlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Column after column on the calling thread.
    Serial,
    /// Level by level, each level on as many threads as it is worth.
    Levels,
    /// Pipelined on this many threads.
    Pipelined(usize),
}

/// The work of a refactorization's replay, in fmadds plus
/// [`COL_WORK`](crate::constants::COL_WORK) a column, as a refactor decides
/// on it: per block that can replay its columns side by side its
/// elimination DAG ([`BlockPlan`]), and the total and largest block work
/// for replaying the blocks side by side. Empty under
/// [`KluParallel::Off`].
#[derive(Debug, Clone, Default)]
struct ReplayPlan {
    /// The policy and the user's structural floors the factorization was
    /// made under ([`KluSettings::parallel`], [`KluSettings::par_min_work`],
    /// [`KluSettings::par_min_ratio`]).
    policy: KluParallel,
    min_work: Option<u64>,
    min_ratio: Option<f64>,
    /// The blocks of two columns or more whose DAG has two levels or more.
    blocks: Vec<BlockPlan>,
    /// The replay work of all blocks, and of the largest.
    total: u64,
    largest: u64,
    /// Every planned block replays so, whatever it is worth (tests).
    #[cfg(test)]
    forced: Option<Mode>,
}

/// A block's elimination DAG as its replay runs side by side: level by
/// level (a section per level of two columns or more), or pipelined (every
/// column taken in order by the next free thread, waiting for each `L`
/// column it reads just before it reads it).
#[derive(Debug, Clone)]
struct BlockPlan {
    /// The block.
    b: usize,
    /// Its replay work.
    work: u64,
    /// What a pipelined replay hands between threads: its columns and the
    /// `L` columns they read.
    handoffs: u64,
    /// The most work of a level of two columns or more (none below it).
    widest: u64,
    /// Level `l` is the columns `cols[ptr[l]..ptr[l + 1]]`, ascending, each
    /// reading only `L` columns of earlier levels; `level_work[l]` their
    /// replay work.
    ptr: Vec<u32>,
    cols: Vec<Ki>,
    level_work: Vec<u64>,
}

/// Per column the number of the refactor that last replayed it: a column
/// of the current one is done when it carries its number, so nothing is
/// cleared between refactors.
#[derive(Debug, Default)]
struct Ready {
    epoch: u32,
    done: Vec<std::sync::atomic::AtomicU32>,
}

impl Clone for Ready {
    /// A fresh one: no column of a refactor yet to come is done.
    fn clone(&self) -> Self {
        Ready::default()
    }
}

impl Ready {
    /// The next refactor's number, over `n` columns.
    fn next(&mut self, n: usize) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.done.clear();
            self.epoch = 1;
        }
        self.done
            .resize_with(n, || std::sync::atomic::AtomicU32::new(0));
    }
}

/// KLU solver handle: factor (or analyze+factor), then solve / refactor.
#[derive(Debug, Clone)]
pub struct KluSolver<T> {
    factors: KluFactors<T>,
    diagnostics: crate::diagnostics::Diagnostics,
    /// Solve-phase accumulators (every `solve*` call records into them).
    solves: crate::diagnostics::SolveCounter,
    /// The works of its own `solve*` calls.
    works: crate::numeric::direct::WorkPool<T>,
    /// The solves are sequential (the determinism guarantee), so a Krylov
    /// solve preconditioned by this factor orthogonalizes on one worker.
    solve_threads: crate::numeric::settings::Threads,
}

fn pattern_mismatch() -> RslabError {
    RslabError::InvalidInput(
        "klu: matrix pattern does not match the symbolic analysis / stored factorization"
            .to_string(),
    )
}

/// Poll a caller-owned cancellation flag. One `Option` branch when unarmed.
#[inline]
pub(crate) fn interrupt_check(
    flag: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(), RslabError> {
    match flag {
        Some(f) if f.load(std::sync::atomic::Ordering::Relaxed) => Err(RslabError::Interrupted),
        _ => Ok(()),
    }
}
