//! The supernodal analysis shared by the LDL^T and LU paths: fill-reducing
//! ordering, elimination tree, supernodes and assembly-tree levels, computed
//! once per pattern, with the lazily built permutation program and
//! left-looking schedule every factorization of the pattern reuses.

use crate::error::RslabError;
use crate::numeric::settings::{in_scoped_pool, stack_for_depth, ReorderMode, SolverSettings};
use crate::numeric::supernodal::LlSchedule;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::{symbolic_factorize_with_method, SupernodeParams, SymbolicFactorization};

/// Reusable symbolic analysis (fill-reducing ordering + assembly-tree levels)
/// for a fixed sparsity pattern. Value-independent: build once with [`analyze`]
/// and pass to [`factor_numeric`] for each set of numeric values sharing the
/// pattern - the PARDISO phase-1 analysis.
pub struct SupernodalAnalysis {
    pub(crate) inner: Option<SymbolicInner>,
    pub(crate) n: usize,
    pub(crate) nnz: usize,
}

impl SupernodalAnalysis {
    /// The fill-reducing ordering the analysis settled on (`perm[k]` the column that became
    /// column `k`); empty for `n = 0`.
    pub fn permutation(&self) -> &[usize] {
        self.inner.as_ref().map_or(&[], |i| &i.sym.perm[..])
    }
}

pub(crate) struct SymbolicInner {
    pub(crate) sym: SymbolicFactorization,
    /// Assembly-tree levels: `by_level[l]` are the supernodes at level `l`, all
    /// mutually independent (factored concurrently by the rayon driver).
    pub(crate) by_level: Vec<Vec<usize>>,
    /// Lazily built scatter program for `P^T A P` (lower fold): the permuted
    /// structure is fixed per pattern, so every (re)factorization reduces to
    /// one linear values scatter. See [`crate::numeric::supernodal::PermScatter`].
    pub(crate) lower_scatter: std::sync::OnceLock<crate::numeric::supernodal::PermScatter>,
    /// Lazily built left-looking schedule (row structures + updater lists),
    /// pattern-only and shared by the numeric drivers and the estimators.
    pub(crate) ll_schedule: std::sync::OnceLock<LlSchedule>,
}

impl SupernodalAnalysis {
    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Internal accessor for the unsymmetric LU driver: the symbolic
    /// factorization and the precomputed assembly-tree levels. `None` for the
    /// empty (`n == 0`) analysis.
    pub(crate) fn sym_and_levels(&self) -> Option<(&SymbolicFactorization, &[Vec<usize>])> {
        self.inner.as_ref().map(|i| (&i.sym, i.by_level.as_slice()))
    }

    /// The cached pattern-only left-looking schedule (row structures + updater
    /// lists), built on first use and shared by the numeric drivers and the
    /// a-priori estimators. `None` for the empty analysis.
    pub(crate) fn ll_schedule(&self) -> Option<&LlSchedule> {
        self.inner
            .as_ref()
            .map(|i| i.ll_schedule.get_or_init(|| LlSchedule::build(&i.sym)))
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)`: the number of
    /// eliminated columns and the full front height. The raw material for
    /// factorization-cost diagnostics - front-size distribution (small vs dense
    /// fronts -> BLAS-2 vs BLAS-3 efficiency) and a factor-flop estimate.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        match &self.inner {
            Some(i) => i.sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect(),
            None => Vec::new(),
        }
    }

    pub fn n_supernodes(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.sym.supernodes.len())
    }

    /// Rows of the largest front after amalgamation.
    pub fn max_front(&self) -> usize {
        self.inner
            .as_ref()
            .and_then(|i| i.sym.supernodes.iter().map(|s| s.nrow).max())
            .unwrap_or(0)
    }

    /// The decisions the analysis took on its own (what `Auto` resolved to),
    /// for the [`Diagnostics`](crate::Diagnostics) of every factorization
    /// reusing it. `requested` is the ordering the caller asked for.
    pub fn decisions(
        &self,
        requested: crate::symbolic::OrderingMethod,
    ) -> crate::diagnostics::Decisions {
        let mut d = crate::diagnostics::Decisions {
            ordering_requested: format!("{requested:?}"),
            n_supernodes: self.n_supernodes(),
            max_front: self.max_front(),
            tree_levels: self.n_levels(),
            ..Default::default()
        };
        match &self.inner {
            Some(i) => {
                d.ordering_used = format!("{:?}", i.sym.resolved_method);
                d.preprocess = format!("{:?}", i.sym.resolved_preprocess);
                d.amalgamation = format!("{:?}", i.sym.resolved_amalgamation);
            }
            None => d.ordering_used = d.ordering_requested.clone(),
        }
        d
    }

    /// Number of assembly-tree levels (the level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.by_level.len())
    }

    /// Supernode count per assembly-tree level, leaves first. `level_widths()[l]`
    /// is the number of mutually independent fronts at level `l` - the available
    /// tree-parallelism at that depth. Wide near the leaves, narrowing to (often)
    /// a single chain at the root; the shape that decides whether tree-parallelism
    /// alone saturates the cores or the top fronts need node-parallelism.
    pub fn level_widths(&self) -> Vec<usize> {
        self.inner
            .as_ref()
            .map_or_else(Vec::new, |i| i.by_level.iter().map(|lv| lv.len()).collect())
    }
}

/// PARDISO phase 1: analyze a sparsity pattern (`n`, CSC `col_ptr`/`row_idx`,
/// lower triangle). The result is value-independent and reusable across many
/// [`factor_numeric`] calls that share the pattern.
pub fn analyze(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Result<SupernodalAnalysis, RslabError> {
    analyze_with(n, col_ptr, row_idx, &SolverSettings::default())
}

/// [`analyze`] with explicit composable [`SolverSettings`] (child-reordering
/// strategy). Reuse the result across many `factor` calls that share the pattern.
pub fn analyze_with(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<SupernodalAnalysis, RslabError> {
    // The symbolic build (ordering, elimination tree, supernode amalgamation,
    // postorder) can recurse to O(n) on pathological patterns - dense/random
    // graphs where nested dissection finds no good separators - and would overflow
    // the caller's stack. Run it in a scoped pool whose workers have a stack sized
    // to the problem (committed on demand), the same robustness mechanism the
    // factorization uses. Shallow analyses get the floor stack at negligible cost.
    //
    // The pool is sized to the settings' thread budget (the same
    // solver-in-the-loop contract the factorization honours), so every parallel
    // step inside the analysis - notably the ND seed ensemble - respects the
    // configured worker count instead of grabbing all cores.
    in_scoped_pool(opts.resolved_threads(), stack_for_depth(n), || {
        analyze_with_inner(n, col_ptr, row_idx, opts)
    })
}

fn analyze_with_inner(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<SupernodalAnalysis, RslabError> {
    let nnz = row_idx.len();
    if n == 0 {
        return Ok(SupernodalAnalysis {
            inner: None,
            n: 0,
            nnz,
        });
    }
    // Symbolic analysis on the structure only; feed a unit-valued f64 pattern.
    let pattern = CscMatrix::<f64> {
        n,
        col_ptr: col_ptr.to_vec(),
        row_idx: row_idx.to_vec(),
        values: vec![1.0; nnz],
    };
    // Disable LdltCompress: it transforms the pattern via a quotient-graph
    // compression beyond a plain permutation, so `sym.perm` would no longer be
    // consistent with the `A_perm` built in `factor_numeric`.
    // Relaxed/fill-tolerant amalgamation - a standard sparse-direct technique
    // (PARDISO/MUMPS apply it to every matrix): when fundamental supernodes are
    // narrow the Schur-update GEMMs are low-rank and memory-bound, so trade a
    // little explicit-zero fill for wider, higher-rank dense fronts. The width is
    // a sweet spot: too narrow -> memory-bound BLAS-2; too wide -> flops wasted on
    // explicit zeros. `<=256-wide, <=64 extra rows/merge` measured best across the
    // EM FEM / MoM matrices (~ -15...-25 % factor time vs the previous 512/128). The lever is
    // workload-agnostic; it rides the general `SupernodeParams.relax` knob and is
    // gated to `n >= RELAX_MIN_N` inside `find_supernodes`.
    let snode_params = SupernodeParams {
        // `preprocess: None` is a correctness requirement, not a tuning knob:
        // LdltCompress rewrites the pattern beyond a permutation, breaking the
        // `sym.perm` <-> `A_perm` consistency `factor_numeric` relies on. The
        // tunable amalgamation knobs (`nemin`, `relax`) ride the composable
        // `SolverSettings`; everything else stays at the tuned default.
        preprocess: crate::symbolic::supernode::OrderingPreprocess::None,
        nemin: opts.nemin,
        relax: opts.relax,
        given_perm: opts.permutation.clone(),
        ..SupernodeParams::default()
    };
    let mut sym = symbolic_factorize_with_method(&pattern, &snode_params, opts.ordering)?;

    // Liu (1986) contribution-stack minimization. Reorder each supernode's
    // children so the live contribution-block stack peak is minimized during
    // factorization. This is a pure **scheduling hint**: supernode IDs, the
    // e-numbering and the factor are unchanged (the global emit walks IDs, not
    // children, and trailing rows are sorted), so it is correctness-, fill- and
    // throughput-neutral - it only shrinks the transient CB-stack that drives
    // factorization peak RSS.
    //
    // Each node leaves a contribution block of size `cb = (nrow-ncol)^2` for its
    // parent and needs `peak` working-stack to factor its subtree. Processing
    // children in order, the stack while doing child `i` is `sum_{j<i} cb_j +
    // peak_i`; Liu's theorem minimizes `max_i(sum_{j<i} cb_j + peak_i)` by ordering
    // children by `(peak - cb)` descending. Supernodes are in postorder, so a
    // single forward sweep has every child's `(peak, cb)` ready.
    //
    // **Hybrid Liu**: reordering is only applied where the contribution stack is
    // actually large (`sum children cb >= LIU_MIN_STACK`) - the upper/mid tree,
    // which is a handful of nodes carrying the spike. The vast majority of small
    // leaf nodes keep their natural order, whose rayon spawn pattern parallelizes
    // better. This keeps almost all of Liu's memory win while shedding most of
    // its throughput cost (the memory-optimal child order is not the
    // parallel-load-optimal one). `peak[s]` is always computed against the order
    // actually used, so the propagation stays exact.
    let nsuper = sym.supernodes.len();
    if opts.reorder == ReorderMode::HybridLiu {
        // ~64 MB of `Complex<f64>` contribution blocks: below this the reorder
        // saves little memory but can still disturb leaf parallelism.
        const LIU_MIN_STACK: f64 = 4_000_000.0;
        let mut cb = vec![0.0f64; nsuper];
        let mut peak = vec![0.0f64; nsuper];
        for s in 0..nsuper {
            let cn = (sym.supernodes[s].nrow - sym.supernodes[s].ncol) as f64;
            cb[s] = cn * cn;
            let mut kids = std::mem::take(&mut sym.supernodes[s].children);
            let stack_total: f64 = kids.iter().map(|&c| cb[c]).sum();
            if stack_total >= LIU_MIN_STACK {
                kids.sort_by(|&a, &b| {
                    (peak[b] - cb[b])
                        .partial_cmp(&(peak[a] - cb[a]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut acc = 0.0f64; // sum cb of already-processed children
            let mut pk = 0.0f64;
            for &ch in &kids {
                pk = pk.max(acc + peak[ch]);
                acc += cb[ch];
            }
            // Assembly step: all children CBs live at once (acc), then this
            // node's own CB remains.
            peak[s] = pk.max(acc).max(cb[s]);
            sym.supernodes[s].children = kids;
        }
    }

    // Assembly-tree levels: level(s) = 1 + max(level(children)); same-level
    // supernodes are mutually independent.
    let mut level = vec![0usize; nsuper];
    let mut max_level = 0usize;
    for s in 0..nsuper {
        let mut lv = 0usize;
        for &ch in &sym.supernodes[s].children {
            lv = lv.max(level[ch] + 1);
        }
        level[s] = lv;
        max_level = max_level.max(lv);
    }
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (s, &lv) in level.iter().enumerate() {
        by_level[lv].push(s);
    }

    Ok(SupernodalAnalysis {
        inner: Some(SymbolicInner {
            sym,
            by_level,
            lower_scatter: std::sync::OnceLock::new(),
            ll_schedule: std::sync::OnceLock::new(),
        }),
        n,
        nnz,
    })
}

/// PARDISO phases 2-3: numeric factorization reusing a [`SupernodalAnalysis`].
/// `a` must carry the same sparsity pattern (`n`, `nnz`) the analysis was built
/// from. Honours static pivoting and incomplete-factor dropping via `opts`.
/// Realize a [`Threads::Auto`] policy from a symbolic analysis: compute the three
/// predictive features (factor-flops, max front height, max tree width) and apply
/// the [`recommend_threads_from`](crate::analysis::recommend_threads_from) policy,
/// capped at `max_cores`. Value-independent, so it is the same for every scalar.
pub(crate) fn recommend_threads_for_sym(symb: &SupernodalAnalysis, max_cores: usize) -> usize {
    let fd = symb.front_dims();
    let flops: u64 = fd
        .iter()
        .map(|&(nc, nr)| (nr as u64) * (nr as u64) * (nc as u64))
        .sum();
    let front_nrow_max = fd.iter().map(|&(_, nr)| nr).max().unwrap_or(0);
    let tree_width_max = symb.level_widths().into_iter().max().unwrap_or(0);
    crate::analysis::recommend_threads_from(flops, front_nrow_max, tree_width_max, max_cores)
}
