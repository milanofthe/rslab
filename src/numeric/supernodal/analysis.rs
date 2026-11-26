//! The supernodal analysis shared by the LDL^T and LU paths: fill-reducing
//! ordering, elimination tree, supernodes and assembly-tree levels, computed
//! once per pattern, with the lazily built permutation program and
//! left-looking schedule every factorization of the pattern reuses.

use crate::constants::{
    NARROW_TREE_MAX_WIDTH, SMALL_WORK_MAX_FLOPS, SMALL_WORK_THREADS, THIN_FRONT_MAX_ROWS,
    THIN_PATTERN_THREADS,
};
use crate::error::RslabError;
use crate::numeric::settings::{stack_for_depth, SolverSettings};
use crate::numeric::supernodal::LlSchedule;
use crate::sparse::csc::CscPattern;
use crate::symbolic::SymbolicFactorization;

pub(crate) struct SupernodalAnalysis {
    pub(crate) inner: Option<SymbolicInner>,
    pub(crate) n: usize,
    pub(crate) nnz: usize,
}

impl SupernodalAnalysis {
    pub(crate) fn heap_bytes(&self) -> u64 {
        self.inner.as_ref().map_or(0, |i| {
            i.sym.heap_bytes()
                + crate::memory::nested_bytes(&i.by_level)
                + i.input.get().map_or(0, |p| p.heap_bytes())
                + i.ll_schedule.heap_bytes()
        })
    }

    /// The fill-reducing ordering the analysis settled on (`perm[k]` the column that became
    /// column `k`); empty for `n = 0`.
    pub fn permutation(&self) -> &[usize] {
        self.inner.as_ref().map_or(&[], |i| &i.sym.perm[..])
    }
}

pub(crate) struct SymbolicInner {
    pub(crate) sym: SymbolicFactorization,
    pub(crate) by_level: Vec<Vec<usize>>,
    pub(crate) input: std::sync::OnceLock<crate::numeric::supernodal::InputProgram>,
    pub(crate) ll_schedule: LlSchedule,
}

impl SymbolicInner {
    pub(crate) fn schedule(&self) -> &LlSchedule {
        &self.ll_schedule
    }
}

impl SupernodalAnalysis {
    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn sym_and_levels(&self) -> Option<(&SymbolicFactorization, &[Vec<usize>])> {
        self.inner.as_ref().map(|i| (&i.sym, i.by_level.as_slice()))
    }

    pub(crate) fn ll_schedule(&self) -> Option<&LlSchedule> {
        self.inner.as_ref().map(SymbolicInner::schedule)
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

    pub(crate) fn probe(&self, nnz: usize, requested: crate::symbolic::OrderingMethod) {
        rsprobe::add!("nonzeros", nnz);
        rsprobe::add!("supernodes", self.n_supernodes());
        rsprobe::debug!("{}", self.describe(nnz, requested));
    }

    fn describe(&self, nnz: usize, requested: crate::symbolic::OrderingMethod) -> String {
        let d = self.decisions(requested);
        let asked = match d.ordering_used == d.ordering_requested {
            true => String::new(),
            false => format!(" (requested {})", d.ordering_requested),
        };
        format!(
            "n={} nnz(A)={nnz} ordering={}{asked} supernodes={} max_front={} levels={}",
            self.n, d.ordering_used, d.n_supernodes, d.max_front, d.tree_levels
        )
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

pub(crate) fn analyze_with(
    pattern: CscPattern<i32>,
    nnz: usize,
    opts: &SolverSettings,
    amd: Option<&[usize]>,
) -> Result<SupernodalAnalysis, RslabError> {
    let n = pattern.n;
    opts.threads.run(
        stack_for_depth(n),
        |cap| cap,
        || analyze_with_inner(pattern, nnz, opts, amd),
    )
}

fn analyze_with_inner(
    pattern: CscPattern<i32>,
    nnz: usize,
    opts: &SolverSettings,
    amd: Option<&[usize]>,
) -> Result<SupernodalAnalysis, RslabError> {
    let n = pattern.n;
    if n == 0 {
        return Ok(SupernodalAnalysis {
            inner: None,
            n: 0,
            nnz,
        });
    }
    let mut sym =
        crate::symbolic::analyze_pattern_from(pattern, &opts.ordering, &opts.amalgamation, amd)?;
    let ll_schedule = {
        let _s = rsprobe::scope!(DEBUG, "schedule");
        LlSchedule::build(&sym)
    };
    sym.pattern = CscPattern {
        n,
        col_ptr: Vec::new(),
        row_idx: Vec::new(),
    };

    let nsuper = sym.supernodes.len();
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
            input: std::sync::OnceLock::new(),
            ll_schedule,
        }),
        n,
        nnz,
    })
}

pub(crate) fn recommend_threads_for_sym(symb: &SupernodalAnalysis, max_cores: usize) -> usize {
    let fd = symb.front_dims();
    let flops: u64 = fd
        .iter()
        .map(|&(nc, nr)| (nr as u64) * (nr as u64) * (nc as u64))
        .sum();
    let front_nrow_max = fd.iter().map(|&(_, nr)| nr).max().unwrap_or(0);
    let tree_width_max = symb.level_widths().into_iter().max().unwrap_or(0);
    recommend_threads_from(flops, front_nrow_max, tree_width_max, max_cores)
}

pub(crate) fn auto_threads(
    symb: &SupernodalAnalysis,
    estimate: &crate::diagnostics::MemoryEstimate,
    cap: usize,
) -> usize {
    #[cfg(feature = "tuning")]
    if let Some((cores, calib)) = crate::tuning::cached_calibration() {
        return crate::tuning::recommend_threads_cost_model(estimate, &calib, cap, cores);
    }
    let _ = estimate;
    recommend_threads_for_sym(symb, cap)
}

fn recommend_threads_from(
    factor_flops: u64,
    front_nrow_max: usize,
    tree_width_max: usize,
    max_cores: usize,
) -> usize {
    let cores = max_cores.max(1);
    let thin_fronts = front_nrow_max < THIN_FRONT_MAX_ROWS;
    let narrow_tree = tree_width_max < NARROW_TREE_MAX_WIDTH;
    if thin_fronts && narrow_tree {
        return cores.min(THIN_PATTERN_THREADS);
    }
    let small_work = factor_flops < SMALL_WORK_MAX_FLOPS;
    if small_work {
        return cores.min(SMALL_WORK_THREADS);
    }
    cores
}
