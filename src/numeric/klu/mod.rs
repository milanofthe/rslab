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
mod supernode;
mod symbolic;
#[cfg(test)]
mod tests;

use factor::{factor_csc, factor_impl, row_scale_inv_into};
pub use settings::{KluParallel, KluSettings};

const UNSET: usize = usize::MAX;

// https://github.com/milanofthe/rslab/issues/290
type Ki = u32;
const KI_UNSET: Ki = Ki::MAX;
const KI_FBIT: Ki = 1 << (Ki::BITS - 1);

/// Symbolic analysis for the KLU path: the BTF block structure plus the
/// per-block fill-reducing ordering. Analyze once, then factor any number of
/// matrices sharing the pattern.
#[derive(Debug, Clone)]
pub struct KluSymbolic {
    n: usize,
    nnz: usize,
    pre_row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    block_ptr: Vec<usize>,
    pat_col_ptr: Vec<usize>,
    pat_row_idx: Vec<usize>,
    predicted: KluEstimate,
    fill: std::sync::OnceLock<KluFill>,
}

#[derive(Debug, Clone, Copy, Default)]
struct KluEstimate {
    factor_nnz: u64,
    flops: u64,
}

#[derive(Debug, Clone)]
struct KluFill {
    l_nnz: u64,
    u_nnz: u64,
    f_nnz: u64,
    flops: u64,
    seq_arrays: (u64, u64),
    seq_block: (u64, u64),
    par_blocks: (u64, u64),
    replay: Vec<(u64, usize)>,
}

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

    fn bytes(&self) -> (u64, u64) {
        let vals = self.l + self.u + self.f + self.diag;
        let other = IDX32_BYTES * (self.l + self.u + 2 * self.f)
            + WORD_BYTES * self.prog
            + IDX32_BYTES * self.fin
            + WORD_BYTES * self.colptr.iter().sum::<usize>();
        (vals as u64, other as u64)
    }
}

#[derive(Debug, Clone)]
struct KluFactors<T> {
    interrupt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    threads: crate::numeric::settings::Threads,
    n: usize,
    nnz_a: usize,
    block_ptr: Vec<usize>,
    row_perm: Vec<usize>,
    pinv: Vec<usize>,
    col_perm: Vec<usize>,
    rs_inv: Vec<f64>,
    scaled: bool,
    l_colptr: Vec<usize>,
    l_rowidx: Vec<Ki>,
    l_val: Vec<T>,
    u_colptr: Vec<usize>,
    u_rowidx: Vec<Ki>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    sn_last: Vec<Ki>,
    u_dense: Vec<bool>,
    l_widest: usize,
    f_colptr: Vec<usize>,
    f_rowidx: Vec<Ki>,
    f_val: Vec<T>,
    scatter_expect: Vec<Ki>,
    scatter_target: Vec<Ki>,
    plan: ReplayPlan,
    rs_next: Vec<f64>,
    replay_x: Vec<T>,
    replay_xs: Vec<Vec<T>>,
    ready: Ready,
    modes: Vec<Mode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Serial,
    Levels,
    Pipelined(usize),
}

#[derive(Debug, Clone, Default)]
struct ReplayPlan {
    policy: KluParallel,
    min_work: Option<u64>,
    min_ratio: Option<f64>,
    blocks: Vec<BlockPlan>,
    total: u64,
    largest: u64,
    #[cfg(test)]
    forced: Option<Mode>,
}

#[derive(Debug, Clone)]
struct BlockPlan {
    block: usize,
    work: u64,
    widest: u64,
    level_ptr: Vec<u32>,
    level_cols: Vec<Ki>,
    level_work: Vec<u64>,
}

// https://github.com/milanofthe/rslab/issues/290
#[derive(Debug, Default)]
struct Ready {
    epoch: u32,
    done: Vec<std::sync::atomic::AtomicU32>,
}

impl Clone for Ready {
    fn clone(&self) -> Self {
        Ready::default()
    }
}

impl Ready {
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
    solves: crate::diagnostics::SolveCounter,
    works: crate::numeric::direct::WorkPool<T>,
    solve_threads: crate::numeric::settings::Threads,
}

fn pattern_mismatch() -> RslabError {
    RslabError::InvalidInput(
        "klu: matrix pattern does not match the symbolic analysis / stored factorization"
            .to_string(),
    )
}

#[inline]
pub(crate) fn interrupt_check(
    flag: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(), RslabError> {
    match flag {
        Some(f) if f.load(std::sync::atomic::Ordering::Relaxed) => Err(RslabError::Interrupted),
        _ => Ok(()),
    }
}
