//! The public LU interface: the reusable analysis on the symmetrized pattern
//! (with the optional MC64 row matching), the factor handle and its solves.

use super::factor::factor_general_lu_numeric;
use super::factors::{LuFactors, LuNumeric};

use crate::error::RslabError;
use crate::numeric::settings::SolverSettings;
use crate::numeric::supernodal::analysis::analyze_with;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use std::sync::Mutex;

/// Lower triangle of the symmetrized pattern `A union A^T` as CSC `(col_ptr,
/// row_idx)`. The symmetric analysis needs a structurally symmetric pattern so
/// the elimination tree carries fill for both `L` and `U`.
fn symmetrized_lower_pattern<T: Scalar>(a: &GeneralCsc<T>) -> (Vec<usize>, Vec<usize>) {
    let n = a.n;
    // Counting-scatter (no `BTreeSet`: no per-element heap allocation, no
    // pointer-chasing). Each entry contributes a lower-triangle pair `(hi, lo)`
    // to bucket `lo`; buckets are then sorted + deduped into CSC.
    let mut counts = vec![0usize; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let lo = if i < j { i } else { j };
            counts[lo] += 1;
        }
    }
    let mut start = vec![0usize; n + 1];
    for j in 0..n {
        start[j + 1] = start[j] + counts[j];
    }
    let total = start[n];
    let mut scattered = vec![0usize; total];
    let mut cursor = start[..n].to_vec();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            scattered[cursor[lo]] = hi;
            cursor[lo] += 1;
        }
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::with_capacity(total);
    for j in 0..n {
        let seg = &mut scattered[start[j]..start[j + 1]];
        seg.sort_unstable();
        let mut last = usize::MAX;
        for &i in seg.iter() {
            if i != last {
                row_idx.push(i);
                last = i;
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx)
}

/// Reusable symbolic analysis for the unsymmetric LU path - the symmetrized
/// pattern `A union A^T` analyzed once. Pass to [`factor_general_lu_numeric`] for
/// each value-set that shares the pattern (frequency sweep / Newton).
/// The MC64 row matching the analysis was done under: the factored matrix
/// is `B = diag(r) P A diag(c)` with `B` row `i` = `A` row `row_of[i]`.
pub(super) struct LuMatching {
    pub(super) row_of: Vec<usize>,
    /// `A`-row scaling.
    pub(super) r: Vec<f64>,
    pub(super) c: Vec<f64>,
}

pub struct LuSymbolic {
    pub(super) symb: crate::numeric::supernodal::analysis::SupernodalAnalysis,
    pub(super) n: usize,
    pub(super) nnz: usize,
    pub(super) matching: Option<LuMatching>,
    /// Wall time of the analysis and the ordering it was asked for, carried
    /// into the diagnostics of every factorization reusing it.
    pub(super) analyze_ms: f64,
    pub(super) requested_ordering: crate::symbolic::OrderingMethod,
    /// [`estimate_memory`](Self::estimate_memory) results, keyed by scalar
    /// size (the estimate depends on `T` only through `size_of::<T>()`, and
    /// rebuilding the supernode row structures per call is expensive).
    pub(super) est_cache: Mutex<Vec<(usize, crate::diagnostics::MemoryEstimate)>>,
    /// The split permuted input's program ([`InputProgram::general`](crate::numeric::supernodal::InputProgram)), built at the first
    /// factorization: every (re)factorization reduces to one linear values
    /// scatter (row matching and equilibration applied on the way).
    pub(super) input: std::sync::OnceLock<crate::numeric::supernodal::InputProgram>,
}

impl LuSymbolic {
    /// The fill-reducing column ordering of the analysis (`perm[k]` the column of the
    /// (row-matched) matrix that became column `k`): pass it to
    /// [`SolverSettings::with_permutation`] to analyse a nearby pattern without a new
    /// ordering.
    pub fn permutation(&self) -> &[usize] {
        self.symb.permutation()
    }

    /// PARDISO **phase 1**: analyze the symmetrized pattern `A union A^T` of `a`
    /// (values ignored, so any matrix with the target pattern works). Reuse the
    /// result across many [`factor`](Self::factor) calls that share the pattern
    /// - the unsymmetric twin of [`LdltSymbolic::analyze`].
    ///
    /// [`LdltSymbolic::analyze`]: crate::numeric::ldlt::LdltSymbolic::analyze
    pub fn analyze<T: Scalar>(a: &GeneralCsc<T>) -> Result<LuSymbolic, RslabError> {
        Self::analyze_with(a, &SolverSettings::default())
    }

    /// [`analyze`](Self::analyze) with explicit composable [`SolverSettings`]
    /// (child-reordering strategy).
    pub fn analyze_with<T: Scalar>(
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSymbolic, RslabError> {
        a.validate()?;
        let n = a.n;
        let nnz = a.row_idx.len();
        if n == 0 {
            return Ok(LuSymbolic {
                symb: analyze_with(0, &[0], &[], opts)?,
                n: 0,
                nnz: 0,
                matching: None,
                analyze_ms: 0.0,
                requested_ordering: opts.ordering,
                est_cache: Mutex::new(Vec::new()),
                input: std::sync::OnceLock::new(),
            });
        }
        let t = crate::clock::Instant::now();
        // MC64 row matching: analyze the row-permuted matrix `B` whose
        // diagonal carries the matched entries.
        let matching = if opts.lu_matching {
            let cache = crate::scaling::mc64::compute_matching_general(a)?;
            if cache.n_matched == n {
                let (r, c) = crate::scaling::mc64::unsymmetric_scaling(&cache);
                // `cache.perm[j]` is the row matched to column `j`: it becomes
                // row `j` of `B`.
                Some(LuMatching {
                    row_of: cache.perm,
                    r,
                    c,
                })
            } else {
                crate::logging::warn(&format!(
                    "lu analyze: structurally rank-deficient ({} of {n} columns matched); row matching skipped",
                    cache.n_matched
                ));
                None
            }
        } else {
            None
        };
        let (col_ptr, row_idx) = match &matching {
            Some(m) => symmetrized_lower_pattern(&Self::row_permuted(a, m)),
            None => symmetrized_lower_pattern(a),
        };
        let symb = analyze_with(n, &col_ptr, &row_idx, opts)?;
        let analyze_ms = t.elapsed().as_secs_f64() * 1e3;
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            let d = symb.decisions(opts.ordering);
            crate::logging::info(&format!(
                "lu analyze: n={n} nnz(A)={nnz} ordering={}{} supernodes={} max_front={} \
                 levels={} {analyze_ms:.1} ms",
                d.ordering_used,
                if d.ordering_used != d.ordering_requested {
                    format!(" (requested {})", d.ordering_requested)
                } else {
                    String::new()
                },
                d.n_supernodes,
                d.max_front,
                d.tree_levels
            ));
        }
        Ok(LuSymbolic {
            symb,
            n,
            nnz,
            matching,
            analyze_ms,
            requested_ordering: opts.ordering,
            est_cache: Mutex::new(Vec::new()),
            input: std::sync::OnceLock::new(),
        })
    }

    /// `A` with its rows permuted by the matching (values untouched; the
    /// scaling is applied in the numeric phase).
    fn row_permuted<T: Scalar>(a: &GeneralCsc<T>, m: &LuMatching) -> GeneralCsc<T> {
        let n = a.n;
        let mut b_row_of_a = vec![0usize; n];
        for (b, &ar) in m.row_of.iter().enumerate() {
            b_row_of_a[ar] = b;
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::with_capacity(a.row_idx.len());
        let mut values = Vec::with_capacity(a.values.len());
        col_ptr.push(0);
        let mut col: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            col.clear();
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                col.push((b_row_of_a[a.row_idx[k]], a.values[k]));
            }
            col.sort_unstable_by_key(|e| e.0);
            for &(r, v) in &col {
                row_idx.push(r);
                values.push(v);
            }
            col_ptr.push(row_idx.len());
        }
        GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        }
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
        let estimate = self.estimate_memory::<T>();
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::supernodal::analysis::recommend_threads_for_sym(&self.symb, cap)
        });
        let warnings = opts.ignored_on(crate::numeric::settings::FactorPath::Lu);
        for w in &warnings {
            crate::logging::warn(&format!("lu settings: {w}"));
        }
        let t = crate::clock::Instant::now();
        let numeric = factor_general_lu_numeric(self, a, opts)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz = numeric.factor_nnz() as u64;
        let factor_bytes = numeric.bytes() as u64;
        let (l, ut, factors) = numeric.into_parts();
        let mut decisions = self.symb.decisions(self.requested_ordering);
        decisions.scaling = if self.matching.is_some() {
            "Mc64RowMatching".to_string()
        } else {
            "TwoSidedRowCol".to_string()
        };
        decisions.method = "LeftLooking".to_string();
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            n: self.n,
            nnz_a: self.nnz as u64,
            factor_nnz: nnz,
            estimate: Some(estimate),
            decisions,
            numeric: crate::diagnostics::NumericReport {
                perturbed: factors.n_perturbed,
                two_by_two: None,
                inertia: None,
            },
            warnings,
            ..Default::default()
        };
        // Bytes per stored entry: the scalar value plus its usize index.
        diagnostics.push("analyze", self.analyze_ms, 0, 0);
        diagnostics.push("factor", factor_ms, estimate.factor_flops, factor_bytes);
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            crate::logging::info(&format!("lu factor: {}", diagnostics.summary()));
        }
        // Solve layout: supernodal panels of `L` and `U^T` plus the tree
        // schedule; the CSC arrays are released so the factor is held once.
        let t = crate::clock::Instant::now();
        let plan_l = crate::numeric::supernodal::solve::SolvePlan::from_panels(
            l,
            &factors.supernode_parent,
            true,
        );
        let plan_u = crate::numeric::supernodal::solve::SolvePlan::from_panels(
            ut,
            &factors.supernode_parent,
            false,
        );
        diagnostics.push(
            "solve-layout",
            t.elapsed().as_secs_f64() * 1e3,
            0,
            (plan_l.bytes() + plan_u.bytes()) as u64,
        );
        Ok(LuSolver {
            factors,
            plan_l,
            plan_u,
            nnz: nnz as usize,
            diagnostics,
            solves: Default::default(),
        })
    }

    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)` of the symmetrized
    /// pattern - for factorization-cost diagnostics (front-size distribution and
    /// a factor-flop estimate). See [`SupernodalAnalysis::front_dims`](crate::SupernodalAnalysis::front_dims).
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See [`SupernodalAnalysis::level_widths`](crate::SupernodalAnalysis::level_widths).
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
                // L: diagonal lower-triangle + off-diagonal rows; U: upper-tri + U12.
                let l = nc * (nc + 1) / 2 + cnrow * nc;
                let u = nc * (nc + 1) / 2 + nc * cnrow;
                l + u
            })
            .sum()
    }

    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        // Cache per scalar size: the estimate is a pure function of the
        // structure and `size_of::<T>()`, and `tuned` + phased `factor` ask
        // for it repeatedly.
        if let Ok(cache) = self.est_cache.lock() {
            if let Some(&(_, est)) = cache.iter().find(|&&(vb, _)| vb == value_bytes) {
                return est;
            }
        }
        let est = self.estimate_memory_for(value_bytes);
        if let Ok(mut cache) = self.est_cache.lock() {
            if !cache.iter().any(|&(vb, _)| vb == value_bytes) {
                cache.push((value_bytes, est));
            }
        }
        est
    }

    /// The uncached estimate body, a pure function of the symbolic structure
    /// and the scalar size.
    fn estimate_memory_for(&self, value_bytes: usize) -> crate::diagnostics::MemoryEstimate {
        let Some((sym, _levels)) = self.symb.sym_and_levels() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                false,
            );
        };
        let nsuper = sym.supernodes.len();
        let Some(sched) = self.symb.ll_schedule() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                false,
            );
        };
        // The `L` and `U^T` panels of a supernode, `(w + m) x w` each; they are
        // the stored factor (no compact copy, see `PanelFactor`).
        let panel_bytes = |s: usize| -> u64 {
            let nc = sym.supernodes[s].ncol;
            let nr = sched.rows(s).len();
            (2 * nr * nc * value_bytes) as u64
        };
        let compact_bytes = panel_bytes;
        // The split permuted input: its values (per factorization) and structure
        // (per analysis, `u32` indices and `usize` positions).
        let input_bytes = (self.nnz * (value_bytes + 12)) as u64;
        let mut est = crate::diagnostics::estimate_left_looking(
            nsuper,
            &panel_bytes,
            &compact_bytes,
            &|s| sched.updaters(s),
            value_bytes,
            input_bytes,
            true,
        );
        est.factor_flops = (0..nsuper)
            .map(|s| {
                let (nc, nr) = (sym.supernodes[s].ncol as u64, sched.rows(s).len() as u64);
                nr * nr * nc
            })
            .sum();
        est
    }
}

/// A factored unsymmetric LU solver, ready to solve against many right-hand
/// sides - the high-level, equilibrated counterpart of the raw [`LuFactors`]
/// (and the unsymmetric twin of [`LdltSolver`](crate::numeric::ldlt::LdltSolver)).
/// Build via [`LuSymbolic::factor`] (analyze once, factor many) or the one-shot
/// [`LuSolver::factor`].
pub struct LuSolver<T> {
    factors: LuFactors<T>,
    /// `L` and `U^T` (the panels, their only storage) with the tree schedule
    /// of [`crate::numeric::supernodal::solve`]; `factors` carries the
    /// permutations, scalings and counters with empty CSC arrays.
    plan_l: crate::numeric::supernodal::solve::SolvePlan<T>,
    plan_u: crate::numeric::supernodal::solve::SolvePlan<T>,
    nnz: usize,
    diagnostics: crate::diagnostics::Diagnostics,
    /// Solve-phase accumulators (every `solve*` call records into them).
    solves: crate::diagnostics::SolveCounter,
}

impl<T: Scalar> LuSolver<T> {
    /// Thread policy the solve phase should honour (issue #9): the resolved
    /// [`Threads`](crate::Threads) budget the factorization used, carried on the
    /// stored [`LuFactors`]. An iterative solve using this factor as a
    /// preconditioner runs its parallel orthogonalization in a pool of this width.
    pub fn solve_thread_policy(&self) -> crate::numeric::settings::Threads {
        self.factors.solve_threads
    }

    /// One-shot analyze + equilibrate + factor of a general matrix `A`.
    pub fn factor(a: &GeneralCsc<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        // Through the symbolic object, so the diagnostics are filled the same
        // way as on the analyze-once path (the former direct call returned
        // an empty `Diagnostics`).
        LuSymbolic::analyze_with(a, opts)?.factor(a, opts)
    }

    /// The **heuristic** settings pick for `a` - the model-free default, the
    /// unsymmetric counterpart of [`LdltSolver::tuned`](crate::LdltSolver::tuned):
    /// analysis with the adaptive ordering heuristic, the proven default kernel
    /// configuration, and (on large systems) the exact nested-dissection bakeoff.
    pub fn tuned(a: &GeneralCsc<T>) -> Result<(LuSymbolic, SolverSettings), RslabError> {
        Self::tuned_with(a, &SolverSettings::default())
    }

    /// [`tuned`](Self::tuned) on top of the caller's settings: the analysis
    /// knobs (`nemin`, `relax`, `reorder`, `lu_matching`, ...) come from
    /// `base`, the ordering is the heuristic race, the thread count the
    /// calibrated pick.
    pub fn tuned_with(
        a: &GeneralCsc<T>,
        base: &SolverSettings,
    ) -> Result<(LuSymbolic, SolverSettings), RslabError> {
        crate::numeric::settings::tuned(a, base, LuSymbolic::analyze_with, |sym: &LuSymbolic| {
            sym.estimate_memory::<T>()
        })
    }

    /// Per-call diagnostics: measured factor time, fill, thread budget, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate). Populated by
    /// the phased [`LuSymbolic::factor`]; empty for the one-shot
    /// [`factor`](Self::factor).
    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)), the solve-phase accumulators
    /// included. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        d
    }

    fn record_solve(&self, rhs: usize, t: crate::clock::Instant, refine_steps: usize) {
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.solves.record(rhs, ms, refine_steps);
        if crate::logging::enabled(crate::logging::LogLevel::Debug) {
            crate::logging::debug(&format!(
                "lu solve: n={} rhs={rhs} refine_steps={refine_steps} {ms:.3} ms",
                self.factors.n
            ));
        }
    }

    /// Solve `A x = b` using the stored factors.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let t = crate::clock::Instant::now();
        let x = self.solve_inner(b, 1)?;
        self.record_solve(1, t, 0);
        Ok(x)
    }

    /// `x = A^{-1} b` on `nrhs` row-major right-hand sides: row scaling and
    /// permutation, the supernodal `L` and `U` sweeps, column permutation
    /// and scaling.
    fn solve_inner(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        let n = f.n;
        if nrhs == 0 || b.len() != n * nrhs {
            return Err(RslabError::DimensionMismatch {
                expected: n * nrhs,
                got: b.len(),
            });
        }
        let mut y = vec![T::zero(); n * nrhs];
        for e in 0..n {
            let orig = f.perm_row[e];
            let sr = T::from_real(f.d_row[orig]);
            let src = &b[orig * nrhs..(orig + 1) * nrhs];
            let dst = &mut y[e * nrhs..(e + 1) * nrhs];
            for c in 0..nrhs {
                dst[c] = src[c] * sr;
            }
        }
        self.plan_l.forward(nrhs, &mut y);
        self.plan_u.backward(nrhs, &mut y);
        let mut out = vec![T::zero(); n * nrhs];
        for e in 0..n {
            let orig = f.perm[e];
            let sc = T::from_real(f.d_col[orig]);
            let src = &y[e * nrhs..(e + 1) * nrhs];
            let dst = &mut out[orig * nrhs..(orig + 1) * nrhs];
            for c in 0..nrhs {
                dst[c] = src[c] * sc;
            }
        }
        Ok(out)
    }

    /// Solve `A * X = B` for `nrhs` right-hand sides at once. `b` and the
    /// returned `x` are **row-major** `n x nrhs` buffers (`b[i*nrhs + c]` is RHS
    /// `c` at row `i`). Faster than `nrhs` separate [`solve`](Self::solve) calls.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let t = crate::clock::Instant::now();
        let x = self.solve_inner(b, nrhs)?;
        self.record_solve(nrhs, t, 0);
        Ok(x)
    }

    /// Solve `A x = b` with iterative refinement against the original matrix `a`
    /// (which must be the matrix this was factored from) - recovers accuracy on
    /// hard systems where the static-pivoted factor alone is insufficient.
    pub fn solve_refined(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        max_iter: usize,
    ) -> Result<Vec<T>, RslabError> {
        Ok(self
            .solve_refined_with(a, b, &crate::refine::RefinePolicy::steps(max_iter))?
            .0)
    }

    /// Iterative refinement under an explicit
    /// [`RefinePolicy`](crate::refine::RefinePolicy), reporting the achieved
    /// backward error.
    pub fn solve_refined_with(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<(Vec<T>, crate::refine::RefineOutcome), RslabError> {
        let t = crate::clock::Instant::now();
        let mut x = self.solve_inner(b, 1)?;
        let outcome = self.refine_into(a, b, &mut x, policy)?;
        self.record_solve(1, t, outcome.steps);
        Ok((x, outcome))
    }

    /// Refine an existing iterate in place, allocating nothing for the
    /// solution.
    pub fn refine_into(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        x: &mut [T],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<crate::refine::RefineOutcome, RslabError> {
        let n = self.factors.n;
        if a.n != n || b.len() != n || x.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: a.n,
            });
        }
        crate::refine::refine_in_place(a, b, x, policy, |r| self.solve_inner(r, 1))
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.nnz
    }

    /// Number of statically perturbed pivots (preconditioner mode).
    pub fn n_perturbed(&self) -> usize {
        self.factors.n_perturbed
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Borrow the underlying raw factors (CSC `L` / CSR `U`, permutations,
    /// equilibration), e.g. to use as a [`Preconditioner`](crate::Preconditioner).
    /// The factor's permutations, scalings and counters. The CSC arrays of
    /// `L` and `U` are empty here: the values live in the panels of the solve
    /// plans (use [`factor_general_lu`] for a factor with CSC arrays).
    pub fn factors(&self) -> &LuFactors<T> {
        &self.factors
    }
}

/// Factor a general (unsymmetric) sparse matrix `A` as `P^T A P = L U` via
/// generic multifrontal LU with partial pivoting. `a` holds the **full** matrix
/// (both triangles). Convenience wrapper over [`LuSymbolic::analyze`] +
/// [`factor_general_lu_numeric`]; for *analyze once, factor many* keep the
/// [`LuSymbolic`] across calls. Solve with [`solve_lu`](crate::solve_lu) / [`solve_lu_refined`](crate::solve_lu_refined).
pub fn factor_general_lu<T: Scalar>(
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
) -> Result<LuFactors<T>, RslabError> {
    // The analysis honours the caller's symbolic settings (ordering, child reordering):
    // `analyze` alone took the defaults and silently ignored `opts.ordering`.
    factor_general_lu_numeric(&LuSymbolic::analyze_with(a, opts)?, a, opts)
        .map(LuNumeric::into_factors)
}

// Supernodal left-looking LU
//
// The unsymmetric twin of the left-looking LDL^T path: each supernode keeps two
// dense panels - `lbuf` (its columns: diagonal block + L21, full height) and
// U12 (its rows' trailing-column part, in the `U^T` panel) - assembled from `A` and
// updated by every factored descendant. The contribution of descendant `k` is
// the rank-`ncol_k` outer product `-L_k[Ok,:]*U_k[:,Ok]`; the part landing in
// `s` splits into two GEMMs: `-L_k[Ok,:]*U_k[:,Pk]` into `lbuf` (columns of `s`)
// and `-L_k[Pk,:]*U_k[:,trailing]` into U12 (rows of `s`). Then the panel
// is factored in place (`cdiv`) with **no trailing/CB update** - there is no
// contribution-block stack and no per-front extract copy-out, the PARDISO
// transient profile. 1x1 static pivoting (no row interchange), as in the
// multifrontal v1; matches the equilibrated preconditioner use case.
// ===========================================================================
