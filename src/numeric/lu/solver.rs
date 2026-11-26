//! The public LU interface: the reusable analysis on the symmetrized pattern
//! (with the optional MC64 row matching), the factor handle and its solves.

use super::factor::factor_general_lu_numeric;
use super::factors::LuPivots;
use super::structure::LuStructure;
use crate::constants::{
    IDX32_BYTES, INPUT_PROGRAM_BYTES_PER_ENTRY, LU_DELAY_ALLOWANCE, LU_INPUT_BUILD_BYTES_PER_COL,
    LU_INPUT_SCALING_BYTES_PER_COL, LU_NODE_CELL_BYTES, LU_NODE_CELL_BYTES_PER_COL,
    LU_PERM_SCALE_BYTES_PER_COL, LU_SCRATCH_POOLS, VEC_HEADER_BYTES, WORD_BYTES,
};

static SYMMETRIZE_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::SORT_NS_PER_ENTRY);

static DIAGONAL_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::SCAN_NS_PER_ENTRY);

use crate::error::RslabError;
use crate::numeric::settings::SolverSettings;
use crate::numeric::supernodal::analysis::analyze_with;
use crate::numeric::supernodal::panel::{PanelFactor, PanelStorage};
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

// https://github.com/milanofthe/rslab/issues/288
fn symmetrized_lower_pattern<T: Scalar>(
    a: &GeneralCsc<T>,
    row_map: Option<&[usize]>,
) -> (Vec<usize>, Vec<usize>) {
    let n = a.n;
    let row = |k: usize| row_map.map_or(a.row_idx[k], |m| m[a.row_idx[k]]);
    let mut start = vec![0usize; n + 1];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            start[row(k).min(j) + 1] += 1;
        }
    }
    for j in 0..n {
        start[j + 1] += start[j];
    }
    let mut scattered = vec![0usize; start[n]];
    let mut cursor = start[..n].to_vec();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = row(k);
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            scattered[cursor[lo]] = hi;
            cursor[lo] += 1;
        }
    }
    let mut buckets: Vec<&mut [usize]> = Vec::with_capacity(n);
    let mut rest: &mut [usize] = &mut scattered;
    for j in 0..n {
        let (head, tail) = rest.split_at_mut(start[j + 1] - start[j]);
        buckets.push(head);
        rest = tail;
    }
    let sorts = crate::parallel::Work::of(start[n] as f64, &SYMMETRIZE_RATE);
    let kept: Vec<usize> = crate::parallel::map_owned(buckets, sorts, |seg| {
        seg.sort_unstable();
        let mut len = 0;
        for p in 0..seg.len() {
            if len == 0 || seg[p] != seg[len - 1] {
                seg[len] = seg[p];
                len += 1;
            }
        }
        len
    });
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut len = 0;
    for j in 0..n {
        scattered.copy_within(start[j]..start[j] + kept[j], len);
        len += kept[j];
        col_ptr.push(len);
    }
    scattered.truncate(len);
    scattered.shrink_to_fit();
    (col_ptr, scattered)
}

// https://github.com/milanofthe/rslab/issues/288
fn diagonal_needs_matching<T: Scalar>(a: &GeneralCsc<T>, negligible: f64) -> bool {
    let scan = crate::parallel::Work::of(a.nnz() as f64, &DIAGONAL_RATE);
    crate::parallel::any(a.n, scan, |j| {
        let (rows, vals) = (
            &a.row_idx[a.col_ptr[j]..a.col_ptr[j + 1]],
            &a.values[a.col_ptr[j]..a.col_ptr[j + 1]],
        );
        let top = vals.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        let diag = rows.binary_search(&j).map_or(0.0, |p| vals[p].magnitude());
        diag.is_nan() || diag <= negligible * top
    })
}

pub(super) struct LuMatching {
    pub(super) row_of: Vec<usize>,
    pub(super) r: Vec<f64>,
    pub(super) c: Vec<f64>,
}

impl LuMatching {
    pub(super) fn row_map(&self) -> Vec<usize> {
        let mut map = vec![0usize; self.row_of.len()];
        for (i, &r) in self.row_of.iter().enumerate() {
            map[r] = i;
        }
        map
    }
}

pub struct LuSymbolic {
    pub(super) symb: crate::numeric::supernodal::analysis::SupernodalAnalysis,
    pub(super) n: usize,
    pub(super) nnz: usize,
    pub(super) matching: Option<LuMatching>,
    pub(super) requested_ordering: crate::symbolic::OrderingMethod,
    pub(super) est_cache: crate::diagnostics::EstimateCache,
    pub(super) input: std::sync::OnceLock<crate::numeric::supernodal::InputProgram>,
    pub(super) structure: LuStructure,
}

impl LuSymbolic {
    /// Heap bytes this analysis holds: the ordering, the supernodes, the
    /// schedule, the exact structures and, once a factorization has built
    /// it, the input program every later factorization reuses.
    /// The analysis stays alive while it is factored, so these bytes are part
    /// of every factorization's footprint.
    pub fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        self.symb.heap_bytes()
            + self.structure.heap_bytes()
            + self.input.get().map_or(0, |p| p.heap_bytes())
            + self.matching.as_ref().map_or(0, |m| {
                vec_bytes(&m.row_of) + vec_bytes(&m.r) + vec_bytes(&m.c)
            })
    }

    /// The fill-reducing column ordering of the analysis (`perm[k]` the column of the
    /// (row-matched) matrix that became column `k`): pass it to
    /// [`SolverSettings::with_permutation`] to analyse a nearby pattern without a new
    /// ordering.
    pub fn permutation(&self) -> &[usize] {
        self.symb.permutation()
    }

    /// PARDISO phase 1: analyze the symmetrized pattern `A union A^T` of `a`
    /// (values ignored, so any matrix with the target pattern works), with
    /// the row matching where the diagonal needs it. Reuse the result across
    /// many [`factor`](Self::factor) calls that share the pattern.
    pub fn analyze<T: Scalar>(
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSymbolic, RslabError> {
        Self::analyze_from(a, opts, None)
    }

    pub(crate) fn analyze_from<T: Scalar>(
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
        amd: Option<&[usize]>,
    ) -> Result<LuSymbolic, RslabError> {
        a.validate()?;
        let n = a.n;
        let nnz = a.row_idx.len();
        if n == 0 {
            return Ok(LuSymbolic {
                symb: analyze_with(crate::symbolic::full_pattern(0, &[0], &[])?, 0, opts, None)?,
                n: 0,
                nnz: 0,
                matching: None,
                requested_ordering: opts.ordering.method,
                est_cache: Default::default(),
                input: std::sync::OnceLock::new(),
                structure: LuStructure::empty(),
            });
        }
        let _s = rsprobe::scope!("analyze");
        let matching = if opts.pivoting.delayed
            || (opts.matching.enabled
                && diagonal_needs_matching(a, opts.matching.negligible_diagonal))
        {
            let cache = {
                let _s = rsprobe::scope!(DEBUG, "matching");
                opts.threads.run(
                    0,
                    |cap| cap,
                    || crate::scaling::mc64::compute_matching_general(a),
                )?
            };
            if cache.n_matched == n {
                let (r, c) = crate::scaling::mc64::unsymmetric_scaling(&cache);
                Some(LuMatching {
                    row_of: cache.perm,
                    r,
                    c,
                })
            } else {
                rsprobe::warn!(
                    "structurally rank-deficient ({} of {n} columns matched); row matching skipped",
                    cache.n_matched
                );
                None
            }
        } else {
            None
        };
        let (pattern, lower_nnz) = {
            let _s = rsprobe::scope!(DEBUG, "symmetrize");
            let row_map = matching.as_ref().map(LuMatching::row_map);
            let (col_ptr, row_idx) = opts.threads.run(
                0,
                |cap| cap,
                || symmetrized_lower_pattern(a, row_map.as_deref()),
            );
            (
                crate::symbolic::full_pattern(n, &col_ptr, &row_idx)?,
                row_idx.len(),
            )
        };
        let symb = analyze_with(pattern, lower_nnz, opts, amd)?;
        let structure = match (symb.sym_and_levels(), symb.ll_schedule()) {
            (Some((sym, levels)), Some(sched)) => {
                let row_map = matching.as_ref().map(LuMatching::row_map);
                let _s = rsprobe::scope!(DEBUG, "structure");
                opts.threads.run(
                    0,
                    |cap| cap,
                    || {
                        LuStructure::build(
                            &a.col_ptr,
                            &a.row_idx,
                            row_map.as_deref(),
                            sym,
                            levels,
                            sched,
                        )
                    },
                )
            }
            _ => LuStructure::empty(),
        };
        symb.probe(nnz, opts.ordering.method);
        Ok(LuSymbolic {
            symb,
            n,
            nnz,
            matching,
            requested_ordering: opts.ordering.method,
            est_cache: Default::default(),
            input: std::sync::OnceLock::new(),
            structure,
        })
    }

    /// Whether the analysis carries an MC64 row matching.
    pub fn has_matching(&self) -> bool {
        self.matching.is_some()
    }

    /// PARDISO **phases 2-3**: equilibrate and LU-factor `a`, reusing this
    /// analysis, into a ready-to-solve [`LuSolver`]. `a` must share the analyzed
    /// pattern. The unsymmetric twin of [`LdltSymbolic::factor`].
    ///
    /// [`LdltSymbolic::factor`]: crate::numeric::ldlt::LdltSymbolic::factor
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSolver<T>, RslabError> {
        let _s = rsprobe::scope!("factor");
        let pools = super::node::LuPools::new();
        let (l, ut, factors, nnz, diagnostics, opts) = self.numeric(a, opts, None, &pools)?;
        let (plan_l, plan_u) = {
            let _s = rsprobe::scope!("solve layout");
            let plan = crate::numeric::supernodal::solve::SolvePlan::from_panels;
            (
                plan(l, &factors.supernode_parent, true, opts.solve),
                plan(ut, &factors.supernode_parent, false, opts.solve),
            )
        };
        Ok(LuSolver {
            solve_threads: opts.threads,
            factors,
            plan_l,
            plan_u,
            nnz,
            diagnostics,
            solves: Default::default(),
            works: Default::default(),
            factored: true,
            pools: super::node::LuPools::new(),
        })
    }

    /// Factor `a` again into `lu`, a factor of this analysis (the next
    /// Newton step): the numeric factorization of [`factor`](Self::factor),
    /// its panels written into `lu`'s buffers and `lu`'s solve schedule kept
    /// where pivoting left the rows unchanged. The same bits as a fresh
    /// [`factor`](Self::factor). After an error `lu` holds no factor and
    /// refuses to solve until a refactorization succeeds.
    pub fn refactor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
        lu: &mut LuSolver<T>,
    ) -> Result<(), RslabError> {
        let _s = rsprobe::scope!("refactor");
        lu.factored = false;
        let storage = (lu.plan_l.take_storage(), lu.plan_u.take_storage());
        let (l, ut, factors, nnz, diagnostics, opts) =
            self.numeric(a, opts, Some(storage), &lu.pools)?;
        {
            let _s = rsprobe::scope!("solve layout");
            lu.plan_l
                .refill(l, &factors.supernode_parent, true, opts.solve);
            lu.plan_u
                .refill(ut, &factors.supernode_parent, false, opts.solve);
        }
        lu.factors = factors;
        lu.nnz = nnz;
        lu.diagnostics = diagnostics;
        lu.solve_threads = opts.threads;
        lu.factored = true;
        Ok(())
    }

    // https://github.com/milanofthe/rslab/issues/288
    #[allow(clippy::type_complexity)]
    fn numeric<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
        storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
        pools: &super::node::LuPools<T>,
    ) -> Result<
        (
            PanelFactor<T>,
            PanelFactor<T>,
            LuPivots<T>,
            usize,
            crate::diagnostics::Diagnostics,
            SolverSettings,
        ),
        RslabError,
    > {
        let estimate = self.estimate_memory::<T>();
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::supernodal::analysis::auto_threads(&self.symb, &estimate, cap)
        });
        let opts = opts.pinned(resolved_threads);
        let mut warnings = opts.ignored_on(crate::numeric::settings::FactorPath::Lu);
        let growth = super::factor::delay_growth(&opts);
        let numeric = factor_general_lu_numeric(self, a, &opts, growth, storage, pools)?;
        let nnz = numeric.factor_nnz() as u64;
        let factor_bytes = numeric.bytes() as u64;
        let (delayed, forced) = (numeric.delayed, numeric.forced);
        if numeric.delays_dropped {
            warnings.push(format!(
                "delayed pivots exceed their allowance (LU_DELAY_ALLOWANCE = {LU_DELAY_ALLOWANCE}); \
                 factored without delays"
            ));
        }
        let (l, ut, factors) = numeric.into_parts();
        let mut decisions = self.symb.decisions(self.requested_ordering);
        decisions.scaling = match (&self.matching, &opts.scaling) {
            (Some(_), _) => "Mc64RowMatching",
            (None, crate::scaling::ScalingStrategy::InfNorm) => "TwoSidedRuiz",
            (None, crate::scaling::ScalingStrategy::Identity) => "Identity",
            (None, _) => "TwoSidedRowCol",
        }
        .to_string();
        decisions.method = "LeftLooking".to_string();
        let diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            n: self.n,
            nnz_a: self.nnz as u64,
            factor_nnz: nnz,
            factor_bytes,
            estimate: Some(estimate),
            decisions,
            numeric: crate::diagnostics::NumericReport {
                perturbed: factors.floors.len(),
                two_by_two: None,
                inertia: None,
                min_pivot: None,
                delayed: Some(delayed),
                forced: Some(forced),
            },
            warnings,
            ..Default::default()
        };
        diagnostics.probe();
        Ok((l, ut, factors, nnz as usize, diagnostics, opts))
    }

    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)` of the symmetrized
    /// pattern - for factorization-cost diagnostics (front-size distribution and
    /// a factor-flop estimate). See `SupernodalAnalysis::front_dims`.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See `SupernodalAnalysis::level_widths`.
    pub fn level_widths(&self) -> Vec<usize> {
        self.symb.level_widths()
    }

    /// **A-priori** peak-memory estimate for factoring a matrix of scalar type `T`
    /// with this analysis - computed purely from the symbolic structure, *before*
    /// any numeric work, so a scheduler can fail-fast or pick an approximation when
    /// the estimate exceeds the memory budget. Deterministic and reproducible.
    /// Exact symbolic factor fill (the compact L+U value count summed over
    /// supernodes), the reliable memory-backstop metric. Unlike
    /// [`MemoryEstimate::factor_nnz`](crate::diagnostics::MemoryEstimate::factor_nnz),
    /// a dense-panel upper bound that overshoots the real fill ~6-7x
    /// non-uniformly across orderings, this tracks the actually-stored fill.
    pub fn symbolic_factor_nnz(&self) -> usize {
        let Some((sym, _)) = self.symb.sym_and_levels() else {
            return 0;
        };
        let Some(sched) = self.symb.ll_schedule() else {
            return 0;
        };
        (0..sym.supernodes.len())
            .map(|s| {
                let nc = sym.supernodes[s].ncol;
                let cnrow = sched.rows(s).len().saturating_sub(nc);
                let l = nc * (nc + 1) / 2 + cnrow * nc;
                let u = nc * (nc + 1) / 2 + nc * cnrow;
                l + u
            })
            .sum()
    }

    // https://github.com/milanofthe/rslab/issues/288
    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        self.est_cache.get::<T>(|| self.estimate_memory_for::<T>())
    }

    fn estimate_memory_for<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        let Some((sym, _)) = self.symb.sym_and_levels() else {
            return crate::diagnostics::MemoryEstimate {
                value_bytes,
                ..Default::default()
            };
        };
        let st = &self.structure;
        let nsuper = sym.supernodes.len();
        let panels: u64 = (0..nsuper)
            .map(|s| ((st.rows_l(s).len() + st.cols_u(s).len()) * sym.supernodes[s].ncol) as u64)
            .sum();
        let settings = SolverSettings {
            threads: crate::numeric::settings::Threads::Fixed(0),
            ..SolverSettings::default()
        };
        let plan = self.memory_plan::<T>(&settings, 1);
        crate::diagnostics::MemoryEstimate {
            value_bytes,
            factor_nnz: panels,
            factor_bytes: plan.factor_bytes,
            panels_all_bytes: panels * value_bytes as u64,
            panel_live_peak_bytes: panels * value_bytes as u64,
            transient_peak_bytes: plan.peak_bytes(),
            factor_flops: (0..nsuper)
                .map(|s| {
                    let nc = sym.supernodes[s].ncol as u64;
                    (st.rows_l(s).len() * st.cols_u(s).len()) as u64 * nc
                })
                .sum(),
            critical_path_flops: 0,
            max_tree_width: 0,
        }
    }

    /// The heap a factorization of scalar type `T` under `opts` needs, and a
    /// solve of `nrhs` right-hand sides after it, predicted from this
    /// analysis before any numeric work: for a preflight check
    /// ([`MemoryPlan::fits_in`](crate::MemoryPlan::fits_in)) and for
    /// scheduling factorizations side by side. The kernels' scratch grows
    /// with the worker count, so plan with the threads the factorization
    /// will run on.
    pub fn memory_plan<T: Scalar>(&self, opts: &SolverSettings, nrhs: usize) -> crate::MemoryPlan {
        let threads = opts.resolved_threads();
        let mut plan = crate::MemoryPlan {
            threads,
            nrhs,
            ..Default::default()
        };
        let (Some((sym, _)), Some(sched)) = (self.symb.sym_and_levels(), self.symb.ll_schedule())
        else {
            return plan;
        };
        let st = &self.structure;
        plan.analysis_bytes = self.heap_bytes();
        let vb = std::mem::size_of::<T>();
        let k = opts.kernel().k;
        let (n, nnz, ns) = (sym.n, self.nnz, sym.supernodes.len());
        let w = |s: usize| sym.supernodes[s].ncol;
        let widths: Vec<usize> = (0..ns).map(w).collect();
        let delay_growth = super::factor::delay_growth(opts);
        let (l_slots, u_slots) = super::factor::lu_slots(sym, st);
        let allowance = super::factor::delay_allowance(&l_slots, &u_slots, delay_growth);
        let panels: usize = l_slots
            .iter()
            .chain(&u_slots)
            .map(|&(v, _)| v)
            .sum::<usize>()
            + allowance;
        let rows: usize = l_slots.iter().chain(&u_slots).map(|&(_, m)| m).sum();
        let passing: usize = if delay_growth > 0.0 {
            (0..ns)
                .map(|s| st.rows_l(s).len() + st.cols_u(s).len())
                .sum()
        } else {
            0
        };
        let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
        let parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
        use crate::numeric::supernodal::solve::layout_size;
        let cores = crate::numeric::settings::all_cores();
        let layout_l = layout_size(
            &parent,
            &widths,
            &|s| &st.rows_l(s)[w(s)..],
            &opts.solve,
            cores,
        );
        let layout_u = layout_size(
            &parent,
            &widths,
            &|s| &st.cols_u(s)[w(s)..],
            &opts.solve,
            cores,
        );
        let (growth, growth_build) = if self.input.get().is_some() {
            (0, 0)
        } else {
            (
                INPUT_PROGRAM_BYTES_PER_ENTRY * nnz + 2 * WORD_BYTES * (n + 1),
                INPUT_PROGRAM_BYTES_PER_ENTRY * nnz + LU_INPUT_BUILD_BYTES_PER_COL * n,
            )
        };
        let factor_with = |panels: usize| {
            vb * (panels + n)
                + (layout_l.held + layout_u.held) as usize
                + LU_PERM_SCALE_BYTES_PER_COL * n
                + WORD_BYTES * ns
        };
        let factor = factor_with(panels);
        let cells =
            ns * LU_NODE_CELL_BYTES + LU_NODE_CELL_BYTES_PER_COL * n + IDX32_BYTES * passing;
        let slots =
            WORD_BYTES * (0..ns).map(|s| st.rows_l(s).len()).sum::<usize>() + IDX32_BYTES * rows;
        let scratch: Vec<(u64, usize, usize)> = (0..ns)
            .map(|s| super::node::lu_node_scratch::<T>(s, sym, sched, st, &k, threads))
            .collect();
        let mut planes: Vec<(usize, usize)> = scratch.iter().map(|&(_, p, c)| (p, c)).collect();
        let (workers, kept) = crate::memory::worker_bytes::<T>(n, 2, threads, &mut planes);
        let largest = scratch.iter().map(|&(b, _, _)| b).max().unwrap_or(0);
        let pooled = crate::memory::pooled_bytes(LU_SCRATCH_POOLS, threads, largest);
        let nodes = crate::memory::concurrent_peak(sym, threads, |s| scratch[s].0);
        let during = (LU_INPUT_SCALING_BYTES_PER_COL * n + vb * panels + cells + slots) as u64
            + workers
            + pooled
            + nodes;
        let after = factor_with(panels) as u64
            + (IDX32_BYTES * rows + 2 * VEC_HEADER_BYTES * ns) as u64
            + layout_l.build.max(layout_u.build);
        plan.analysis_growth_bytes = growth as u64;
        plan.factor_bytes = factor as u64;
        plan.kept_bytes = kept + pooled;
        plan.factor_peak_bytes = (growth_build as u64).max(growth as u64 + during.max(after))
            + crate::constants::BOOKKEEPING;
        plan.solve_bytes = (nrhs * vb) as u64
            * (2 * n as u64 + layout_l.solve_per_rhs.max(layout_u.solve_per_rhs));
        plan
    }
}

/// A factored unsymmetric LU solver, ready to solve against many right-hand
/// sides, the unsymmetric twin of [`LdltSolver`](crate::numeric::ldlt::LdltSolver).
/// Build via [`LuSymbolic::factor`] (analyze once, factor many) or the one-shot
/// [`LuSolver::factor`].
#[derive(Clone)]
pub struct LuSolver<T> {
    factors: LuPivots<T>,
    pub(crate) plan_l: crate::numeric::supernodal::solve::SolvePlan<T>,
    plan_u: crate::numeric::supernodal::solve::SolvePlan<T>,
    nnz: usize,
    diagnostics: crate::diagnostics::Diagnostics,
    solves: crate::diagnostics::SolveCounter,
    works: crate::numeric::direct::WorkPool<T>,
    solve_threads: crate::numeric::settings::Threads,
    factored: bool,
    pools: super::node::LuPools<T>,
}

impl<T: Scalar> LuSolver<T> {
    /// Heap bytes this factor holds: its values in solve layout, the pivots,
    /// the permutations and the solve schedule. `L` and `U^T` each have their own.
    pub fn heap_bytes(&self) -> u64 {
        self.plan_l.heap_bytes() + self.plan_u.heap_bytes() + self.factors.heap_bytes()
    }

    /// One-shot analyze + equilibrate + factor of a general matrix `A`.
    pub fn factor(a: &GeneralCsc<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        LuSymbolic::analyze(a, opts)?.factor(a, opts)
    }

    /// Per-call diagnostics: measured factor time, fill, thread budget, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate).
    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)), the solve-phase accumulators
    /// included. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        let u_diag_inv = self.plan_u.diag_inv();
        d.numeric.min_pivot = (self.factored && !u_diag_inv.is_empty())
            .then(|| 1.0 / u_diag_inv.iter().map(|v| v.magnitude()).fold(0.0, f64::max));
        d
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.nnz
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }
}

impl<T: Scalar> crate::numeric::direct::SolveCore<T> for LuSolver<T> {
    const NAME: &'static str = "lu";

    fn dim(&self) -> usize {
        self.factors.n
    }

    fn counter(&self) -> &crate::diagnostics::SolveCounter {
        &self.solves
    }
    fn works(&self) -> &crate::numeric::direct::WorkPool<T> {
        &self.works
    }

    fn perturbations(&self) -> &[crate::PivotFloor<T>] {
        &self.factors.floors
    }

    fn log_det_raw(&self) -> Result<crate::numeric::direct::LogDet<T>, RslabError> {
        if !self.factored {
            return Err(RslabError::InvalidInput(
                "the last refactorization failed; refactor before solving".to_string(),
            ));
        }
        let f = &self.factors;
        let mut d = crate::numeric::direct::LogDet::one();
        for &r in self.plan_u.diag_inv() {
            d.div(r);
        }
        d.permutation(&f.perm_row);
        d.permutation(&f.perm);
        for &s in f.d_row.iter().chain(&f.d_col) {
            d.div_scale(s);
        }
        Ok(d)
    }

    fn solve_raw_into(
        &self,
        b: &[T],
        nrhs: usize,
        transpose: bool,
        x: &mut [T],
        work: &mut crate::SolveWork<T>,
    ) -> Result<(), RslabError> {
        if !self.factored {
            return Err(RslabError::InvalidInput(
                "the last refactorization failed; refactor before solving".to_string(),
            ));
        }
        let f = &self.factors;
        let n = f.n;
        let (gather, g_scale, scatter, s_scale) = if transpose {
            (&f.perm, &f.d_col, &f.perm_row, &f.d_row)
        } else {
            (&f.perm_row, &f.d_row, &f.perm, &f.d_col)
        };
        let y = &mut work.y;
        let (first, last) = if transpose {
            (&self.plan_u, &self.plan_l)
        } else {
            (&self.plan_l, &self.plan_u)
        };
        let threads = self.solve_threads;
        first.sweep(&mut work.plan, nrhs, threads, |pw| {
            let pool = pw.on_pool();
            crate::numeric::direct::gather_scaled(b, nrhs, gather, g_scale, y, pool);
            first.forward_block(nrhs, y, pw);
        });
        last.sweep(&mut work.plan, nrhs, threads, |pw| {
            last.backward_block(nrhs, y, pw);
            let pool = pw.on_pool();
            crate::numeric::direct::scatter_scaled(y, nrhs, scatter, s_scale, x, pool);
        });
        debug_assert_eq!(scatter.len(), n);
        Ok(())
    }
}

crate::numeric::direct::direct_solver!(LuSolver);
