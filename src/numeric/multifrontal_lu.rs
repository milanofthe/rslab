//! Generic **unsymmetric** sparse LU factorization over any [`Scalar`] field -
//! the general (non-symmetric) complex path, complementing the symmetric LDL^T
//! path in [`crate::numeric::multifrontal_ldlt`].
//!
//! It targets matrices whose *values* are unsymmetric (e.g. MoM A-EFIE
//! near-field saddle preconditioners, where the symmetric and antisymmetric
//! parts are comparable) but reuses the full symmetric machinery: the
//! fill-reducing ordering, supernodes and assembly tree
//! ([`analyze`](crate::numeric::multifrontal_ldlt::analyze)) and the SIMD
//! `gemm` Schur kernel. Only the dense panel kernel changes - an unsymmetric
//! LU producing separate `L` and `U` - and the analysis runs on the
//! **symmetrized pattern** `A union A^T` so the elimination structure carries
//! fill for both factors.
//!
//! ## Pivoting
//!
//! * **Threshold partial pivoting** (UMFPACK-style, `THRESH = 0.1`), bounded to
//!   each panel's fully-summed block: the diagonal is kept unless it falls below
//!   `THRESH * |colmax|`, in which case the column max is brought up. Sub-floor
//!   pivots are perturbed in preconditioner mode ([`ZeroPivotAction::PerturbToEps`])
//!   or rejected in exact mode. Pivoting stays cheap on the equilibrated,
//!   unit-diagonal MoM matrices while guarding the genuinely ill-scaled columns.
//! * The factors `L` (unit lower) and `U^T` (the pivots on its diagonal) are
//!   kept in supernodal panel form ([`PanelFactor`], see [`LuNumeric`]), in
//!   factorization order: each supernode's finished panels become the stored
//!   factor, and [`LuNumeric::into_factors`] materializes the sparse `L` (CSC)
//!   and `U` (CSR) of [`LuFactors`] on demand. The factorization is the
//!   supernodal **left-looking** kernel (low transient, no contribution-block
//!   stack).

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::multifrontal_ldlt::{analyze_with, perturb_pivot};
use crate::numeric::panel_factor::{finish_panel, PanelArena, PanelFactor, PanelOut};
use crate::numeric::settings::{SolverSettings, ZeroPivotAction};
use crate::scalar::{fmadd, Scalar};
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Stored unsymmetric LU factors, in factorization order. Solve with
/// [`solve_lu`]. The factored system is `P^T A P = L U`, `perm[e]` mapping
/// factorization position `e` to the original index.
pub struct LuFactors<T> {
    pub n: usize,
    /// `L` in CSC (unit lower, explicit unit diagonal).
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_values: Vec<T>,
    /// `U` in CSR (upper; the diagonal entry carries the pivot).
    pub u_row_ptr: Vec<usize>,
    pub u_col_idx: Vec<usize>,
    pub u_values: Vec<T>,
    /// Column permutation: factorization position -> original column index
    /// (`P^T A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position -> original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `A_hat = diag(d_row)*A*diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
    /// Column partition of the factor into supernodes (the fronts): columns
    /// `supernode_ptr[s]..supernode_ptr[s + 1]` of `L` and rows of `U` share
    /// one structure. Length `ns + 1`; empty when unknown.
    pub supernode_ptr: Vec<usize>,
    /// Parent of every supernode in the assembly tree (`usize::MAX` for a
    /// root); empty when unknown.
    pub supernode_parent: Vec<usize>,
    /// Number of statically perturbed pivots.
    pub n_perturbed: usize,
    /// Thread policy the **solve phase** should honour (issue #9): resolved from
    /// the factorization's [`SolverSettings::threads`] so an iterative solve using
    /// this factor as a preconditioner runs its parallel orthogonalization in a
    /// pool of the **same** width - factor and solve share one concurrency budget
    /// instead of the solve silently fanning out over the global pool.
    /// [`Threads::Ambient`](crate::Threads::Ambient) means "use the caller's
    /// current pool" (the solver-in-the-loop path); otherwise a concrete
    /// [`Threads::Fixed`](crate::Threads::Fixed) worker count.
    pub solve_threads: crate::numeric::settings::Threads,
}

/// The numeric result of a sparse LU factorization: the unit lower `L` and
/// the transposed upper factor `U^T` (its diagonal in the panel) in
/// supernodal panel form (the storage the solves run on, written by the
/// drivers without a copy), the permutations, the scalings and the outcome.
/// [`into_factors`](Self::into_factors) materializes the compressed
/// [`LuFactors`] (`L` as CSC, `U` as CSR) for the reference solves.
#[derive(Clone, Debug)]
pub struct LuNumeric<T> {
    /// `L` in panel form, in elimination order (unit diagonal implicit).
    pub l: PanelFactor<T>,
    /// `U^T` in panel form: column `c` of the panel is row `c` of `U`, the
    /// diagonal of `U` at the panel's diagonal.
    pub ut: PanelFactor<T>,
    /// `perm[e]` is the original column eliminated at position `e`.
    pub perm: Vec<usize>,
    /// `perm_row[e]` is the original row that became pivot row `e`.
    pub perm_row: Vec<usize>,
    /// Row scaling applied before the factorization.
    pub d_row: Vec<f64>,
    /// Column scaling applied before the factorization.
    pub d_col: Vec<f64>,
    /// Supernode tree over the factor's supernodes (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// Pivots perturbed by the static regularization.
    pub n_perturbed: usize,
    /// Structural panel slots holding an exact zero (the symmetrized
    /// pattern, cancellation or `drop_tol`); the stored nonzeros are
    /// `l.nnz() + ut.nnz() - n_zeros`.
    pub n_zeros: usize,
    /// Thread policy the solves inherit.
    pub solve_threads: crate::numeric::settings::Threads,
}

impl<T: Scalar> LuNumeric<T> {
    /// Dimension.
    pub fn n(&self) -> usize {
        self.l.n
    }

    /// Stored fill `nnz(L) + nnz(U)`: the structural panel entries minus the
    /// slots holding an exact zero.
    pub fn factor_nnz(&self) -> usize {
        self.l.nnz() + self.ut.nnz() - self.n_zeros
    }

    /// Bytes of the two panel factors.
    pub fn bytes(&self) -> usize {
        self.l.bytes() + self.ut.bytes()
    }

    fn shell(&self) -> LuFactors<T> {
        LuFactors {
            n: self.l.n,
            l_col_ptr: Vec::new(),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            u_row_ptr: Vec::new(),
            u_col_idx: Vec::new(),
            u_values: Vec::new(),
            perm: self.perm.clone(),
            perm_row: self.perm_row.clone(),
            d_row: self.d_row.clone(),
            d_col: self.d_col.clone(),
            supernode_ptr: self.l.sn_col.iter().map(|&c| c as usize).collect(),
            supernode_parent: self.supernode_parent.clone(),
            n_perturbed: self.n_perturbed,
            solve_threads: self.solve_threads,
        }
    }

    /// The compressed form for the reference solves (copies both factors).
    pub fn into_factors(self) -> LuFactors<T> {
        let mut f = self.shell();
        let (l_col_ptr, l_row_idx, l_values) = self.l.to_csc(true);
        let (u_row_ptr, u_col_idx, u_values) = self.ut.to_csc(false);
        f.l_col_ptr = l_col_ptr;
        f.l_row_idx = l_row_idx;
        f.l_values = l_values;
        f.u_row_ptr = u_row_ptr;
        f.u_col_idx = u_col_idx;
        f.u_values = u_values;
        f
    }

    /// Split into the two panel factors and an [`LuFactors`] shell (empty CSC
    /// arrays) carrying the permutations and scalings for the solver.
    pub(crate) fn into_parts(self) -> (PanelFactor<T>, PanelFactor<T>, LuFactors<T>) {
        let shell = self.shell();
        (self.l, self.ut, shell)
    }
}

impl<T: Scalar> LuFactors<T> {
    /// Stored fill: `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len() + self.u_values.len()
    }
}

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

/// Fold the MC64 row matching back into the factors: pivot rows and the row
/// scaling are reported in `A`'s row indices, as the solves expect.
fn finish_matching<T: Scalar>(fac: &mut LuNumeric<T>, lusym: &LuSymbolic) {
    if let Some(m) = &lusym.matching {
        for e in fac.perm_row.iter_mut() {
            *e = m.row_of[*e];
        }
        fac.d_row = m.r.clone();
    }
}

/// Reusable symbolic analysis for the unsymmetric LU path - the symmetrized
/// pattern `A union A^T` analyzed once. Pass to [`factor_general_lu_numeric`] for
/// each value-set that shares the pattern (frequency sweep / Newton).
/// The MC64 row matching the analysis was done under: the factored matrix
/// is `B = diag(r) P A diag(c)` with `B` row `i` = `A` row `row_of[i]`.
struct LuMatching {
    row_of: Vec<usize>,
    /// `A`-row scaling.
    r: Vec<f64>,
    c: Vec<f64>,
}

/// The permuted input of the numeric phase, split the way the supernodes read
/// it: entry `(i, j)` of `P^T B P` (`B` the matrix factored, `A` or its
/// row-matched form) goes to the columns of `j`'s supernode if row `i` lies in
/// its diagonal block or below, else to the `U12` rows of `i`'s supernode (`j`
/// then lies past it). Every entry is read once, so the values are one array
/// of `nnz`: the column part `[..split]` (by column: `col_ptr`, `row_idx`),
/// then the row part (by row: `row_ptr`, `col_idx`, offsets from `split`).
/// The structure is fixed per analysis; a factorization scatters its values
/// through `pos` (entry `k` of `A` to slot `pos[k]`), row matching included.
struct LuScatter {
    col_ptr: Vec<usize>,
    row_idx: Vec<Li>,
    row_ptr: Vec<usize>,
    col_idx: Vec<Li>,
    pos: Vec<usize>,
}

impl LuScatter {
    /// `b_row[r]`: the row of `B` that row `r` of `A` becomes (`None`: `B = A`).
    fn build(
        a: &GeneralCsc<impl Scalar>,
        b_row: Option<&[usize]>,
        sym: &SymbolicFactorization,
    ) -> Self {
        let n = a.n;
        let mut first = vec![0usize; n];
        for sn in &sym.supernodes {
            first[sn.first_col..sn.first_col + sn.ncol].fill(sn.first_col);
        }
        let at = |j: usize, k: usize| {
            let r = a.row_idx[k];
            (sym.perm_inv[b_row.map_or(r, |b| b[r])], sym.perm_inv[j])
        };
        // Count per target column / row, place `(index, entry)` pairs, sort each
        // column / row by index (the assembly walks them in that order).
        let (mut col_ptr, mut row_ptr) = (vec![0usize; n + 1], vec![0usize; n + 1]);
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let (gi, gj) = at(j, k);
                if gi >= first[gj] {
                    col_ptr[gj + 1] += 1;
                } else {
                    row_ptr[gi + 1] += 1;
                }
            }
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
            row_ptr[c + 1] += row_ptr[c];
        }
        let split = col_ptr[n];
        let mut pairs: Vec<(Li, usize)> = vec![(0, 0); a.row_idx.len()];
        let (mut cc, mut rc) = (col_ptr[..n].to_vec(), row_ptr[..n].to_vec());
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let (gi, gj) = at(j, k);
                if gi >= first[gj] {
                    pairs[cc[gj]] = (gi as Li, k);
                    cc[gj] += 1;
                } else {
                    pairs[split + rc[gi]] = (gj as Li, k);
                    rc[gi] += 1;
                }
            }
        }
        drop((first, cc, rc));
        let mut pos = vec![0usize; pairs.len()];
        let mut idx = |ptr: &[usize], base: usize| -> Vec<Li> {
            let mut out = vec![0 as Li; ptr[n]];
            for c in 0..n {
                let run = &mut pairs[base + ptr[c]..base + ptr[c + 1]];
                run.sort_unstable_by_key(|&(g, _)| g);
                for (p, &(g, k)) in (ptr[c]..ptr[c + 1]).zip(run.iter()) {
                    out[p] = g;
                    pos[k] = base + p;
                }
            }
            out
        };
        let row_idx = idx(&col_ptr, 0);
        let col_idx = idx(&row_ptr, split);
        LuScatter {
            col_ptr,
            row_idx,
            row_ptr,
            col_idx,
            pos,
        }
    }
}

/// One factorization's permuted input: the [`LuScatter`] structure and its values.
#[derive(Clone, Copy)]
struct LuInput<'a, T> {
    sc: &'a LuScatter,
    vals: &'a [T],
}

impl<'a, T: Scalar> LuInput<'a, T> {
    /// Rows and values of column `c` in its supernode's columns (row `>=` the
    /// supernode's first column).
    #[inline]
    fn col(self, c: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let r = self.sc.col_ptr[c]..self.sc.col_ptr[c + 1];
        let (idx, vals) = (&self.sc.row_idx[r.clone()], &self.vals[r]);
        idx.iter().zip(vals).map(|(&g, &v)| (g as usize, v))
    }

    /// Columns and values of row `r` in its supernode's `U12` (column past the
    /// supernode).
    #[inline]
    fn row(self, r: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let split = self.sc.col_ptr[self.sc.col_ptr.len() - 1];
        let k = split + self.sc.row_ptr[r]..split + self.sc.row_ptr[r + 1];
        let idx = &self.sc.col_idx[self.sc.row_ptr[r]..self.sc.row_ptr[r + 1]];
        idx.iter()
            .zip(&self.vals[k])
            .map(|(&g, &v)| (g as usize, v))
    }
}

pub struct LuSymbolic {
    symb: crate::numeric::multifrontal_ldlt::MultifrontalSymbolic,
    n: usize,
    nnz: usize,
    matching: Option<LuMatching>,
    /// Wall time of the analysis and the ordering it was asked for, carried
    /// into the diagnostics of every factorization reusing it.
    analyze_ms: f64,
    requested_ordering: crate::symbolic::OrderingMethod,
    /// [`estimate_memory`](Self::estimate_memory) results, keyed by scalar
    /// size (the estimate depends on `T` only through `size_of::<T>()`, and
    /// rebuilding the supernode row structures per call is expensive).
    est_cache: Mutex<Vec<(usize, crate::diagnostics::MemoryEstimate)>>,
    /// The split permuted input's structure ([`LuScatter`]), built at the first
    /// factorization: every (re)factorization reduces to one linear values
    /// scatter (row matching and equilibration applied on the way).
    scatter: std::sync::OnceLock<LuScatter>,
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
    /// [`LdltSymbolic::analyze`]: crate::numeric::sparse_solver::LdltSymbolic::analyze
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
                scatter: std::sync::OnceLock::new(),
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
            scatter: std::sync::OnceLock::new(),
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
    /// [`LdltSymbolic::factor`]: crate::numeric::sparse_solver::LdltSymbolic::factor
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSolver<T>, RslabError> {
        let estimate = self.estimate_memory::<T>();
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&self.symb, cap)
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
        let plan_l = crate::numeric::supernodal_solve::SolvePlan::from_panels(
            l,
            &factors.supernode_parent,
            true,
        );
        let plan_u = crate::numeric::supernodal_solve::SolvePlan::from_panels(
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
    /// a factor-flop estimate). See [`MultifrontalSymbolic::front_dims`](crate::MultifrontalSymbolic::front_dims).
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See [`MultifrontalSymbolic::level_widths`](crate::MultifrontalSymbolic::level_widths).
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
/// (and the unsymmetric twin of [`LdltSolver`](crate::numeric::sparse_solver::LdltSolver)).
/// Build via [`LuSymbolic::factor`] (analyze once, factor many) or the one-shot
/// [`LuSolver::factor`].
pub struct LuSolver<T> {
    factors: LuFactors<T>,
    /// `L` and `U^T` (the panels, their only storage) with the tree schedule
    /// of [`crate::numeric::supernodal_solve`]; `factors` carries the
    /// permutations, scalings and counters with empty CSC arrays.
    plan_l: crate::numeric::supernodal_solve::SolvePlan<T>,
    plan_u: crate::numeric::supernodal_solve::SolvePlan<T>,
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
        crate::numeric::ll_common::tuned(a, base, LuSymbolic::analyze_with, |sym: &LuSymbolic| {
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
/// [`LuSymbolic`] across calls. Solve with [`solve_lu`] / [`solve_lu_refined`].
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

/// One factored supernode's left-looking payload besides its arena slots: the
/// within-front row permutation from partial pivoting (`rperm[i]` is the
/// row-structure index physically at panel position `i`; identity on the
/// trailing rows, only read by the emit).
type LuLlStore = crate::numeric::ll_common::SlotStore<Vec<usize>>;

use crate::numeric::ll_common::{emit_refcount_offsets, Cells, Li, LlSchedule, PanelPtr};

/// Apply a factored NB-wide panel transform (column scale by `pinv`, within-panel
/// rank-1 against the stored `U11`) to rows `[r0, r1)` of a column-major buffer
/// based at `base` with column stride `nrow`. Bit-identical to the corresponding
/// rows of a full-height `getf2`. Used for the deep trailing rows, which are never
/// pivot candidates, so each caller's row range is independent.
///
/// SAFETY: `[r0, r1)` must be this caller's exclusive rows and within the buffer;
/// columns `[kb, kb+pw)` must be in bounds under stride `nrow`.
#[inline]
unsafe fn apply_panel_trailing<T: Scalar>(
    base: *mut T,
    nrow: usize,
    kb: usize,
    pw: usize,
    pinv_blk: &[T],
    r0: usize,
    r1: usize,
) {
    // `kk` indexes pinv_blk and drives the column arithmetic (`k`, `j`) and inner
    // range - not a plain slice walk.
    #[allow(clippy::needless_range_loop)]
    for kk in 0..pw {
        let k = kb + kk;
        let pinv_k = pinv_blk[kk];
        let colk = base.add(k * nrow);
        for i in r0..r1 {
            *colk.add(i) = *colk.add(i) * pinv_k;
        }
        for jj in (kk + 1)..pw {
            let j = kb + jj;
            let ukj = *base.add(j * nrow + k);
            if ukj != T::zero() {
                let colj = base.add(j * nrow);
                for i in r0..r1 {
                    *colj.add(i) = *colj.add(i) - *colk.add(i) * ukj;
                }
            }
        }
    }
}

struct LlEmit<T> {
    /// Number of consumers (ancestors that pull) still to come; freed at 0.
    refcount: Vec<AtomicUsize>,
    /// First elimination position of each supernode (symbolic prefix sum of ncol).
    e_offset: Vec<usize>,
    /// The factors' buffers: every supernode factors `L` straight into its
    /// slot of `l_arena`; `U^T` is gathered into `u_arena` at the emit.
    l_arena: PanelArena<T>,
    u_arena: PanelArena<T>,
    panels: Cells<(PanelOut, PanelOut)>,
    /// `e_of_g[g]` = elimination position of COLUMN g; `row_pos_of_g[g]` =
    /// position whose PIVOT ROW is g. Written in-node (disjoint g), read after the
    /// join barrier (and in `emit_and_free`, where the join chain makes consumer
    /// writes visible).
    e_of_g: Cells<usize>,
    row_pos_of_g: Cells<usize>,
    perm: Cells<usize>,
    perm_row: Cells<usize>,
}

impl<T: Scalar> LlEmit<T> {
    fn new(sym: &SymbolicFactorization, sched: &LlSchedule) -> Self {
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let sizes =
            || (0..sym.supernodes.len()).map(|s| sched.rows(s).len() * sym.supernodes[s].ncol);
        LlEmit {
            refcount,
            e_offset,
            l_arena: PanelArena::new(sizes()),
            u_arena: PanelArena::new(sizes()),
            panels: Cells::new_default(sym.supernodes.len()),
            e_of_g: Cells::new(n, usize::MAX),
            row_pos_of_g: Cells::new(n, usize::MAX),
            perm: Cells::new(n, 0),
            perm_row: Cells::new(n, 0),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }
    #[inline]
    unsafe fn rg(&self, g: usize) -> usize {
        *self.row_pos_of_g.get(g)
    }
}

/// Emit supernode `k` once its last updater is done: the left-looking panel
/// `lbuf` (`L` strictly below the diagonal, `U`'s diagonal block on and above
/// it) becomes the `L` panel as it is, and `U^T`'s panel, whose off-block rows
/// the factorization wrote in place (`U12`), gets its diagonal block from the
/// upper triangle. Both get their off-block rows in elimination order.
fn emit_and_free<T: Scalar>(
    k: usize,
    store: &LuLlStore,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    drop_tol: Option<f64>,
) {
    let snode = &sym.supernodes[k];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(k).len();
    let cnrow = nrow - ncol;
    // SAFETY: the owner of supernode `k` emits it exactly once, after its last
    // updater has read the slots (refcount zero); nobody reads them afterwards.
    let rperm = unsafe { store.take(k) };
    let eoff = emit.e_offset[k];
    debug_assert!(
        (0..ncol).all(|p| unsafe { emit.eg(first + p) } == eoff + p)
            && (0..ncol).all(|i| unsafe { emit.rg(sched.rows(k)[rperm[i]] as usize) } == eoff + i),
        "the diagonal block is in elimination order"
    );
    // `U^T`'s diagonal block from the upper triangle of the `L` slot.
    {
        let lbuf: &[T] = unsafe { emit.l_arena.slot(k) };
        let ut = unsafe { emit.u_arena.slot_mut(k) };
        debug_assert_eq!(ut.len(), nrow * ncol);
        for p in 0..ncol {
            for i in p..ncol {
                ut[p * nrow + i] = lbuf[i * nrow + p];
            }
        }
    }
    let l_rows: Vec<u32> = (ncol..nrow)
        .map(|i| unsafe { emit.rg(sched.rows(k)[rperm[i]] as usize) } as u32)
        .collect();
    let u_rows: Vec<u32> = (0..cnrow)
        .map(|t| unsafe { emit.eg(sched.rows(k)[ncol + t] as usize) } as u32)
        .collect();
    let l_out = finish_panel(
        unsafe { emit.l_arena.slot_mut(k) },
        ncol,
        l_rows,
        None,
        drop_tol,
    );
    let u_out = finish_panel(
        unsafe { emit.u_arena.slot_mut(k) },
        ncol,
        u_rows,
        None,
        drop_tol,
    );
    unsafe { emit.panels.set(k, (l_out, u_out)) };
}

#[allow(clippy::too_many_arguments)]
fn lu_ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    inp: LuInput<T>,
    sched: &LlSchedule,
    store: &LuLlStore,
    emit: &LlEmit<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    let ll_gemm_gate = kt.scalar_gate;
    let ll_gemm_par = kt.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(s).len();
    let cnrow = nrow - ncol;
    let n = sym.n;
    // `lbuf`: nrowxncol (columns of s, full height). `ut`: the `U^T` panel, nrowxncol
    // as well; the factorization writes its off-block rows, `U12[p, t]` at
    // `ut[p * nrow + ncol + t]` (row `p` of `U` is column `p` of the panel), and the
    // emit fills the diagonal block.
    // SAFETY: this task owns supernode `s`; nobody reads the slots before they
    // are published by `store.set` at the end of the node.
    let lbuf: &mut [T] = unsafe { emit.l_arena.slot_mut(s) };
    let ut: &mut [T] = unsafe { emit.u_arena.slot_mut(s) };
    debug_assert_eq!(lbuf.len(), nrow * ncol);

    let gloc = crate::numeric::ll_common::Gloc::new(n, sched.rows(s));
    // Assemble columns of s (full) into lbuf, and the U12 rows into ut.
    for p in 0..ncol {
        let c = first + p;
        for (g, v) in inp.col(c) {
            let li = gloc[g];
            if li != Li::MAX {
                let li = li as usize;
                lbuf[p * nrow + li] = lbuf[p * nrow + li] + v;
            }
        }
        for (g, v) in inp.row(c) {
            let lc = gloc[g];
            if lc != Li::MAX {
                let lc = lc as usize;
                ut[p * nrow + lc] = ut[p * nrow + lc] + v;
            }
        }
    }
    // cmod from every factored descendant. NOTE: cmod-aggregation (K-stacking many
    // descendant updates into one fat GEMM) was measured and rejected - across MoM
    // topologies 91-95 % of cmod flop already runs as large parallel GEMMs, and the
    // only aggregation reaching those dominant updates carries an 11-15x zero-pad
    // blowup (each top-of-tree descendant touches a small, distinct row/col subset
    // of the large target). The `RLA_CMOD_DIST` histogram below documents this.
    let plan = crate::numeric::ll_common::CmodPlan::new(sym, sched, s, true, ll_gemm_par);
    let (spans, forks, tile_w, tiled) = (&plan.spans, plan.forks, plan.tile_w, plan.tiled);
    let tile_u = (cnrow.max(1) / 16).clamp(32, 256);

    // Column-tiled parallel cmod: disjoint `&mut` slabs of the target
    // buffers; per slab every updater's contribution in updater order with a
    // serial GEMM. One fan-out per node instead of one per update; the slab
    // stays cache-hot across the updaters; slab widths are pure functions of
    // the node (never of the thread count). Every entry lives in exactly one
    // slab and receives its contributions in the same updater order. The LU
    // node has TWO target buffers, so the tiling runs as two phases: `lbuf`
    // slabs (L/U11 updates), then `U12` slabs of `ut` (runs of its columns).
    if tiled {
        let gloc_ref = &gloc;
        let spans_ref = spans;
        lbuf.par_chunks_mut(nrow * tile_w)
            .enumerate()
            .for_each(|(ti, slab)| {
                let c0 = ti * tile_w;
                let c1 = (c0 + tile_w).min(ncol);
                let mut lupd: Vec<T> = Vec::new();
                for &(kk, p0, p1) in spans_ref {
                    let nck = sym.supernodes[kk].ncol;
                    let nrk = sched.rows(kk).len();
                    let ok = &sched.rows(kk)[nck..];
                    let nok = ok.len();
                    let q0 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c0);
                    let q1 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c1);
                    let npk = q1 - q0;
                    if npk == 0 {
                        continue;
                    }
                    // SAFETY: `kk` is a factored descendant of `s`.
                    let lk: &[T] = unsafe { emit.l_arena.slot(kk) };
                    let uk: &[T] = unsafe { emit.u_arena.slot(kk) };
                    let mrows = nok - p0;
                    lupd.clear();
                    lupd.resize(mrows * npk, T::zero());
                    // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                    unsafe {
                        crate::dense::gemm_backend::gemm(
                            mrows,
                            npk,
                            nck,
                            lupd.as_mut_ptr(),
                            mrows as isize,
                            1,
                            false,
                            lk.as_ptr().add(nck + p0),
                            nrk as isize,
                            1,
                            uk.as_ptr().add(nck + q0),
                            1,
                            nrk as isize,
                            T::zero(),
                            T::one(),
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                    for jj in 0..npk {
                        let cbase = (ok[q0 + jj] as usize - first - c0) * nrow;
                        let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                        for i in 0..mrows {
                            let dst = cbase + gloc_ref[ok[p0 + i] as usize] as usize;
                            slab[dst] = slab[dst] - ucol[i];
                        }
                    }
                }
            });
        if cnrow > 0 {
            let rs_s = sched.rows(s);
            // A slab is the run `[u0, u1)` of U12's columns: rows `ncol + u0..ncol + u1`
            // of every `ut` column, disjoint between slabs.
            let up = PanelPtr(ut.as_mut_ptr());
            (0..cnrow.div_ceil(tile_u)).into_par_iter().for_each(|ti| {
                let u0 = ti * tile_u;
                let u1 = (u0 + tile_u).min(cnrow);
                let g0 = rs_s[ncol + u0];
                let g1 = if ncol + u1 < rs_s.len() {
                    rs_s[ncol + u1]
                } else {
                    Li::MAX
                };
                let mut uupd: Vec<T> = Vec::new();
                for &(kk, p0, p1) in spans_ref {
                    let nck = sym.supernodes[kk].ncol;
                    let nrk = sched.rows(kk).len();
                    let ok = &sched.rows(kk)[nck..];
                    let nok = ok.len();
                    let t0 = p1 + ok[p1..nok].partition_point(|&g| g < g0);
                    let t1 = p1 + ok[p1..nok].partition_point(|&g| g < g1);
                    let ntr = t1 - t0;
                    let npk = p1 - p0;
                    if ntr == 0 || npk == 0 {
                        continue;
                    }
                    // SAFETY: `kk` is a factored descendant of `s`.
                    let lk: &[T] = unsafe { emit.l_arena.slot(kk) };
                    let uk: &[T] = unsafe { emit.u_arena.slot(kk) };
                    uupd.clear();
                    uupd.resize(npk * ntr, T::zero());
                    // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                    unsafe {
                        crate::dense::gemm_backend::gemm(
                            npk,
                            ntr,
                            nck,
                            uupd.as_mut_ptr(),
                            npk as isize,
                            1,
                            false,
                            lk.as_ptr().add(nck + p0),
                            nrk as isize,
                            1,
                            uk.as_ptr().add(nck + t0),
                            1,
                            nrk as isize,
                            T::zero(),
                            T::one(),
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                    for jj in 0..ntr {
                        let lt = gloc_ref[ok[t0 + jj] as usize] as usize;
                        let ucol = &uupd[jj * npk..jj * npk + npk];
                        for i in 0..npk {
                            // SAFETY: row `lt` lies in this slab's run.
                            unsafe {
                                let d = up.get().add((ok[p0 + i] as usize - first) * nrow + lt);
                                *d = *d - ucol[i];
                            }
                        }
                    }
                }
            });
        }
    }

    // Sequential per-update cmod (small nodes / narrow panels).
    let mut lupd: Vec<T> = Vec::new();
    let mut uupd: Vec<T> = Vec::new();
    for &(kk, p0, p1) in spans.iter().filter(|_| !tiled) {
        let nck = sym.supernodes[kk].ncol;
        let nrk = sched.rows(kk).len();
        let ok = &sched.rows(kk)[nck..];
        let nok = ok.len();
        // SAFETY: `kk` is a factored descendant of `s`.
        let lk: &[T] = unsafe { emit.l_arena.slot(kk) };
        let uk: &[T] = unsafe { emit.u_arena.slot(kk) };
        let npk = p1 - p0;
        let mrows = nok - p0; // rows used by the L update (Ok subset sched.rows(s) from here)
        let ntrail = nok - p1;
        if mrows * npk * nck < ll_gemm_gate {
            // Scalar path.
            for jj in 0..npk {
                let tcol = ok[p0 + jj] as usize - first;
                for i in 0..mrows {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + lk[(nck + p0 + i) + ck * nrk] * uk[ck * nrk + nck + p0 + jj];
                    }
                    let trow = gloc[ok[p0 + i] as usize] as usize;
                    lbuf[tcol * nrow + trow] = lbuf[tcol * nrow + trow] - acc;
                }
            }
            for jj in 0..ntrail {
                let tu = gloc[ok[p1 + jj] as usize] as usize - ncol;
                for i in 0..npk {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + lk[(nck + p0 + i) + ck * nrk] * uk[ck * nrk + nck + p1 + jj];
                    }
                    let urow = ok[p0 + i] as usize - first;
                    ut[urow * nrow + ncol + tu] = ut[urow * nrow + ncol + tu] - acc;
                }
            }
        } else {
            // `forks` folds in the join-steal guard and the global serial
            // switch: a small node never forks here.
            let par = if forks && mrows * npk * nck >= ll_gemm_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // L update: Lupd(mrowsxnpk) = L_k[Ok>=p0,:] * U_k[:,Pk].
            lupd.clear();
            lupd.resize(mrows * npk, T::zero());
            // SAFETY: lhs (lk off-diag rows), rhs (uk Pk cols), dst (lupd) are
            // disjoint; strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    mrows,
                    npk,
                    nck,
                    lupd.as_mut_ptr(),
                    mrows as isize,
                    1,
                    false,
                    lk.as_ptr().add(nck + p0),
                    nrk as isize,
                    1,
                    uk.as_ptr().add(nck + p0),
                    1,
                    nrk as isize,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
            for jj in 0..npk {
                let cbase = (ok[p0 + jj] as usize - first) * nrow;
                let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                for i in 0..mrows {
                    let dst = cbase + gloc[ok[p0 + i] as usize] as usize;
                    lbuf[dst] = lbuf[dst] - ucol[i];
                }
            }
            // U update: Uupd(npkxntrail) = L_k[Pk,:] * U_k[:,trailing].
            if ntrail > 0 {
                uupd.clear();
                uupd.resize(npk * ntrail, T::zero());
                // SAFETY: as above; rhs is the trailing U columns of `uk`.
                unsafe {
                    crate::dense::gemm_backend::gemm(
                        npk,
                        ntrail,
                        nck,
                        uupd.as_mut_ptr(),
                        npk as isize,
                        1,
                        false,
                        lk.as_ptr().add(nck + p0),
                        nrk as isize,
                        1,
                        uk.as_ptr().add(nck + p1),
                        1,
                        nrk as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        par,
                    );
                }
                for jj in 0..ntrail {
                    let lt = gloc[ok[p1 + jj] as usize] as usize;
                    let ucol = &uupd[jj * npk..jj * npk + npk];
                    for i in 0..npk {
                        let dst = (ok[p0 + i] as usize - first) * nrow + lt;
                        ut[dst] = ut[dst] - ucol[i];
                    }
                }
            }
        }
    }
    // cdiv: in-place **blocked** panel LU (1x1 static pivoting), no trailing/CB
    // update. Mirrors the multifrontal `lu_front` getrf - unblocked `getf2` over
    // an NB-wide panel, then the dominant trailing update as a single SIMD GEMM
    // (rank-NB) - but restricted to the panel: the trailing is the remaining
    // panel columns (`lbuf`) plus the `U12` rows (in `ut`), with no `A22`/CB. This
    // routes the `O(ncol^2*nrow)` cdiv work (the measured 77 % of the left-looking
    // factor) through BLAS-3 instead of scalar rank-1 sweeps.
    // Panel width. Swept 32/48/64/96 on the MoM fronts: 32 optimal for typical
    // panels - but root-class WIDE panels want a fatter deferred-GEMM inner
    // dimension (k = nb), the same lever as the LDLT twin's adaptive nb. Pure
    // function of `ncol`, never of the thread count.
    let nb_cdiv = if ncol >= 512 { 128 } else { 32 };
    // Join-steal guard (see the cmod fork gate above): a small node must not
    // fork inside its cdiv either.
    let ll_cdiv_par = if nrow * ncol * ncol >= 100_000_000 {
        kt.par_cdiv
    } else {
        usize::MAX
    };
    let mut local_perturbed = 0usize;
    // Restricted partial pivoting: row interchanges within the fully-summed block
    // `[0, ncol)` only (the standard sparse-direct choice). `rperm[i]` is the
    // row-structure index physically at position `i`; the trailing rows are never
    // interchanged, so the contribution rows `Ok` ancestors pull are unaffected
    // and `cmod` needs no permutation awareness.
    let mut rperm: Vec<usize> = (0..nrow).collect();
    // Pivot reciprocals of the current panel, reused by the parallel trailing apply.
    let mut pinv_blk: Vec<T> = vec![T::zero(); nb_cdiv];
    let mut kb = 0;
    while kb < ncol {
        kt.interrupted()?;
        let ke = (kb + nb_cdiv).min(ncol);
        // getf2: factor columns [kb, ke) over the **fully-summed rows [k+1, ncol)**
        // only - the deep trailing rows [ncol, nrow) (never pivot candidates) are
        // lifted off this serial path into the parallel apply below.
        for k in kb..ke {
            // **Threshold** partial pivoting (UMFPACK-style): keep the diagonal
            // pivot unless it is below `THRESH` of the largest candidate in the
            // fully-summed block - so a well-scaled/equilibrated matrix never
            // interchanges (no fill or accuracy cost) while small/zero diagonals
            // still get a stable pivot. `THRESH^2` compared on squared magnitudes.
            // `THRESH = kt.pivot_u` (tunable, default 0.1); `u = 1` recovers full
            // partial pivoting, `u = 0` keeps the diagonal unless it is exactly zero.
            let thresh_sq = kt.pivot_u * kt.pivot_u;
            // Static pivoting fast path (`u == 0`): keep the natural pivot order and
            // skip the argmax search entirely - the "skip pivot search" speed lever
            // for fixed-pattern value sequences (solver-in-the-loop: reuse a good
            // order across a frequency sweep / time-stepping). The search result is
            // never consumed when `u == 0` (the threshold test `diag_sq < 0` can
            // never fire), so skipping it is behaviour-identical, only faster. A
            // sub-floor / zero diagonal is still caught below by the pivot policy.
            if thresh_sq > 0.0 {
                let mut p = k;
                let mut best = lbuf[k * nrow + k].magnitude_sq();
                for i in (k + 1)..ncol {
                    let m = lbuf[k * nrow + i].magnitude_sq();
                    if m > best {
                        best = m;
                        p = i;
                    }
                }
                let diag_sq = lbuf[k * nrow + k].magnitude_sq();
                if p != k && diag_sq < thresh_sq * best {
                    for c in 0..ncol {
                        lbuf.swap(c * nrow + k, c * nrow + p);
                    }
                    for t in ncol..nrow {
                        ut.swap(k * nrow + t, p * nrow + t);
                    }
                    rperm.swap(k, p);
                }
            }
            let mut piv = lbuf[k * nrow + k];
            match perturb_floor {
                Some(floor) if piv.magnitude() < floor => {
                    piv = perturb_pivot(piv, floor);
                    local_perturbed += 1;
                }
                None if piv == T::zero() => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            lbuf[k * nrow + k] = piv;
            let pinv = piv.recip();
            pinv_blk[k - kb] = pinv;
            for i in (k + 1)..ncol {
                lbuf[k * nrow + i] = lbuf[k * nrow + i] * pinv;
            }
            for j in (k + 1)..ke {
                let u_kj = lbuf[j * nrow + k];
                if u_kj != T::zero() {
                    for i in (k + 1)..ncol {
                        lbuf[j * nrow + i] = lbuf[j * nrow + i] - lbuf[k * nrow + i] * u_kj;
                    }
                }
            }
        }
        let pw = ke - kb;
        // Trailing-row L21 panel [ncol, nrow): apply the just-computed panel
        // transform (scale by `pinv_blk`, within-panel rank-1 against `U11`) to the
        // deep rows, parallel over **disjoint** row chunks. Bit-identical to the
        // full-height getf2 - same per-row op sequence - but the dominant `cnrow`
        // work now runs on all idle workers instead of the serial panel path.
        if cnrow > 0 {
            let par = cnrow * pw * pw >= ll_cdiv_par;
            if par {
                let pp = PanelPtr(lbuf.as_mut_ptr());
                let nthreads = rayon::current_num_threads().max(1);
                let cs = (nrow - ncol).div_ceil(nthreads).max(1);
                let ranges: Vec<(usize, usize)> = (0..nthreads)
                    .map(|c| {
                        let r0 = ncol + c * cs;
                        (r0.min(nrow), (r0 + cs).min(nrow))
                    })
                    .filter(|(a, b)| a < b)
                    .collect();
                // Capture the whole `pp` (Send+Sync) - destructure inside so Rust
                // does not disjoint-capture the bare `*mut T`.
                ranges.par_iter().for_each(|&(r0, r1)| {
                    // SAFETY: disjoint row chunk; see `apply_panel_trailing`.
                    unsafe { apply_panel_trailing(pp.get(), nrow, kb, pw, &pinv_blk, r0, r1) };
                });
            } else {
                // SAFETY: single-threaded over all trailing rows.
                unsafe {
                    apply_panel_trailing(lbuf.as_mut_ptr(), nrow, kb, pw, &pinv_blk, ncol, nrow)
                };
            }
        }
        // TRSM: U = L11^-1 * (trailing panel columns of lbuf) and the U12 rows.
        // Each trailing column is an independent forward substitution reading
        // only the finished panel columns [kb, ke), so the block parallelizes
        // over disjoint column chunks - bit-identical per-column op order.
        // Profiled at 22% of cdiv CPU when serial (MoM fronts).
        if (ncol - ke) * pw * pw >= ll_cdiv_par {
            let (head, tail) = lbuf.split_at_mut(ke * nrow);
            tail.par_chunks_mut(nrow).for_each(|col| {
                for r in (kb + 1)..ke {
                    let mut acc = col[r];
                    for i in kb..r {
                        acc = acc - head[i * nrow + r] * col[i];
                    }
                    col[r] = acc;
                }
            });
        } else {
            for j in ke..ncol {
                for r in (kb + 1)..ke {
                    let mut acc = lbuf[j * nrow + r];
                    for i in kb..r {
                        acc = acc - lbuf[i * nrow + r] * lbuf[j * nrow + i];
                    }
                    lbuf[j * nrow + r] = acc;
                }
            }
        }
        // U12 rows (the `cnrow` contribution columns of U): the forward substitution
        // over the panel rows, `x_r -= L[r, i] x_i` for `i` ascending, on whole rows
        // of U12 (the contiguous runs `ut[r * nrow + ncol..(r + 1) * nrow]`), so every
        // entry sees the operations of its own column's substitution in order;
        // parallel over disjoint runs of the columns.
        let trsm_u = |t0: usize, t1: usize, u: PanelPtr<T>, lref: &[T]| {
            for r in (kb + 1)..ke {
                for i in kb..r {
                    let l = lref[i * nrow + r];
                    // SAFETY: rows `r != i` of `ut`, the caller's columns `[t0, t1)`.
                    unsafe {
                        let (xr, xi) = (u.get().add(r * nrow), u.get().add(i * nrow));
                        for t in t0..t1 {
                            *xr.add(t) = *xr.add(t) - l * *xi.add(t);
                        }
                    }
                }
            }
        };
        let up = PanelPtr(ut.as_mut_ptr());
        if cnrow * pw * pw >= ll_cdiv_par {
            let lref: &[T] = lbuf;
            (ncol..nrow)
                .into_par_iter()
                .step_by(256)
                .for_each(|t0| trsm_u(t0, (t0 + 256).min(nrow), up, lref));
        } else {
            trsm_u(ncol, nrow, up, lbuf);
        }
        // GEMM: lbuf[ke.., ke..ncol] -= L21[ke.., kb..ke] * U[kb..ke, ke..ncol].
        let mt = nrow - ke;
        let nt = ncol - ke;
        if mt > 0 && nt > 0 {
            let par = if (mt * nt * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let base = lbuf.as_mut_ptr();
            // SAFETY: the three sub-blocks of `lbuf` are disjoint; strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    mt,
                    nt,
                    pw,
                    base.add(ke * nrow + ke),
                    nrow as isize,
                    1,
                    true,
                    base.add(kb * nrow + ke),
                    nrow as isize,
                    1,
                    base.add(ke * nrow + kb),
                    nrow as isize,
                    1,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
        }
        // GEMM: U12[ke..ncol, :] -= L[ke..ncol, kb..ke] * U12[kb..ke, :].
        if cnrow > 0 && nt > 0 {
            let par = if (nt * cnrow * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let lptr = lbuf.as_ptr();
            let uptr = ut.as_mut_ptr();
            // SAFETY: dst (U12 rows `ke..ncol`, `ut` columns) is disjoint from the
            // read sub-blocks of `lbuf` and U12 (rows `kb..ke`); strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    nt,
                    cnrow,
                    pw,
                    uptr.add(ke * nrow + ncol),
                    1,
                    nrow as isize,
                    true,
                    lptr.add(kb * nrow + ke),
                    nrow as isize,
                    1,
                    uptr.add(kb * nrow + ncol),
                    1,
                    nrow as isize,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
        }
        kb = ke;
    }
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    // Populate the O(n) index maps for `s` from its (final) `rperm` and the
    // symbolic elimination offset - consumed by `emit_and_free` and the assembly.
    // Writes target disjoint global indices; visibility via the subtree join.
    let eoff = emit.e_offset[s];
    for (p, &rp) in rperm[..ncol].iter().enumerate() {
        let g_col = first + p;
        let g_row = sched.rows(s)[rp] as usize;
        // SAFETY: each global index is written by exactly one supernode.
        unsafe {
            emit.e_of_g.set(g_col, eoff + p);
            emit.row_pos_of_g.set(g_row, eoff + p);
            emit.perm.set(eoff + p, sym.perm[g_col]);
            emit.perm_row.set(eoff + p, sym.perm[g_row]);
        }
    }
    // SAFETY: this thread owns `s`, writes its cells exactly once.
    unsafe { store.set(s, rperm) };
    Ok(())
}

/// Supernodal left-looking LU producing the same [`LuFactors`] as the
/// multifrontal path. `inp` is the equilibrated permuted matrix; `d_row`/`d_col`
/// the equilibration carried into the result.
#[allow(clippy::too_many_arguments)]
fn factor_lu_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    inp: LuInput<T>,
    d_row: &[f64],
    d_col: &[f64],
    perturb_floor: Option<f64>,
    drop_tol: Option<f64>,
    kt: KernelTuning,
) -> Result<LuNumeric<T>, RslabError> {
    let n = sym.n;
    let nsuper = sym.supernodes.len();
    let store = LuLlStore::new(nsuper);
    let emit = LlEmit::<T>::new(sym, sched);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let factor_node = |s: usize| {
        lu_ll_factor_node(
            s,
            sym,
            inp,
            sched,
            &store,
            &emit,
            perturb_floor,
            &n_perturbed_atomic,
            kt,
        )
    };
    let emit_free = |k: usize| emit_and_free(k, &store, &emit, sym, sched, drop_tol);
    crate::numeric::ll_common::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free)?;
    drop(store); // panels moved into the emit cells; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
    let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let LlEmit {
        l_arena,
        u_arena,
        panels,
        ..
    } = emit;
    let (l, zeros_l) = l_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).0)
    });
    let (ut, zeros_u) = u_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).1)
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm.get(e) }).collect();
    let perm_row: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm_row.get(e) }).collect();

    Ok(LuNumeric {
        l,
        ut,
        perm,
        perm_row,
        d_row: d_row.to_vec(),
        d_col: d_col.to_vec(),
        supernode_parent,
        n_perturbed,
        n_zeros: zeros_l + zeros_u,
        solve_threads: crate::numeric::settings::Threads::Ambient,
    })
}

/// PARDISO phases 2-3 for the general path: numeric LU reusing a [`LuSymbolic`].
/// `a` must share the analyzed pattern (`n`, `nnz`).
#[allow(clippy::needless_range_loop)] // CSC column loops index col_ptr + scaling
pub fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
) -> Result<LuNumeric<T>, RslabError> {
    a.validate()?;
    let n = lusym.n;
    if a.n != n || a.row_idx.len() != lusym.nnz {
        return Err(RslabError::InvalidInput(
            "factor_general_lu_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    if n == 0 {
        return Ok(LuNumeric {
            l: PanelFactor::empty(),
            ut: PanelFactor::empty(),
            perm: Vec::new(),
            perm_row: Vec::new(),
            d_row: Vec::new(),
            d_col: Vec::new(),
            supernode_parent: Vec::new(),
            n_perturbed: 0,
            n_zeros: 0,
            solve_threads: crate::numeric::settings::Threads::Ambient,
        });
    }

    // Resolve the solve-phase thread policy (issue #9): `Ambient` stays ambient
    // (caller-installed pool); every other policy is pinned to the concrete worker
    // count the factorization itself used, so a preconditioned iterative solve
    // orthogonalizes in a pool of exactly that width.
    let solve_policy = match opts.threads {
        crate::numeric::settings::Threads::Ambient => crate::numeric::settings::Threads::Ambient,
        p => crate::numeric::settings::Threads::Fixed(p.resolve(|cap| {
            crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&lusym.symb, cap)
        })),
    };

    let perturb_floor: Option<f64> = match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    // The assembly-tree levels are no longer needed: the driver is a
    // work-stealing tree recursion, not a level-synchronous sweep.
    let (sym, _by_level) = lusym
        .symb
        .sym_and_levels()
        .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;

    // Two-sided equilibration A_hat = D_r A D_c with d_r[i] = 1/sqrt(max_j |A_ij|),
    // d_c[j] = 1/sqrt(max_i |A_ij|). Tames the dynamic range (these MoM near-field
    // matrices span ~6 orders) so the LU factor - and any incomplete drop -
    // stays well-scaled; the solve undoes it transparently. Computed from the
    // original (unpermuted) A.
    // The matrix the pipeline factors: `A` itself, or its MC64 row-permuted
    // form `B` (row `i` of `B` is row `row_of[i]` of `A`) with the matching's
    // scalings; otherwise the max-norm equilibration.
    let (d_row, d_col): (Vec<f64>, Vec<f64>) = match &lusym.matching {
        Some(m) => ((0..n).map(|i| m.r[m.row_of[i]]).collect(), m.c.clone()),
        None => {
            let mut rmax = vec![0.0f64; n];
            let mut cmax = vec![0.0f64; n];
            for j in 0..n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    let i = a.row_idx[k];
                    let m = a.values[k].magnitude();
                    if m > rmax[i] {
                        rmax[i] = m;
                    }
                    if m > cmax[j] {
                        cmax[j] = m;
                    }
                }
            }
            (
                rmax.iter()
                    .map(|&r| crate::scaling::inv_sqrt_scale_guarded(r))
                    .collect(),
                cmax.iter()
                    .map(|&c| crate::scaling::inv_sqrt_scale_guarded(c))
                    .collect(),
            )
        }
    };

    // `A`'s row `r` is `B`'s row `b_row[r]`, scaled by `d_row[b_row[r]]`.
    let b_row: Option<Vec<usize>> = lusym.matching.as_ref().map(|m| {
        let mut b = vec![0usize; n];
        for (i, &r) in m.row_of.iter().enumerate() {
            b[r] = i;
        }
        b
    });
    let sc = lusym
        .scatter
        .get_or_init(|| LuScatter::build(a, b_row.as_deref(), sym));
    let mut vals = vec![T::zero(); a.row_idx.len()];
    for j in 0..n {
        let dc = d_col[j];
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let r = a.row_idx[k];
            let dr = d_row[b_row.as_ref().map_or(r, |b| b[r])];
            vals[sc.pos[k]] = a.values[k] * T::from_real(dr * dc);
        }
    }
    drop(b_row);
    let inp = LuInput { sc, vals: &vals };
    // Worker stack sized to the assembly-tree depth (overflow-safe on deep chain
    // trees), shared by both LU paths.
    let stack = crate::numeric::settings::stack_for_depth(
        crate::numeric::settings::supernode_tree_depth(sym),
    );

    // Run in a scoped pool of `opts.threads` so concurrent solves don't
    // oversubscribe.
    let mut fac = opts.threads.run(
        stack,
        |cap| crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&lusym.symb, cap),
        || {
            factor_lu_left_looking(
                sym,
                lusym.symb.ll_schedule().ok_or_else(|| {
                    RslabError::InvalidInput("internal: empty symbolic".to_string())
                })?,
                inp,
                &d_row,
                &d_col,
                perturb_floor,
                opts.drop_tol,
                opts.kernel(),
            )
        },
    )?;
    fac.solve_threads = solve_policy;
    finish_matching(&mut fac, lusym);
    Ok(fac)
}

/// Solve `A x = b` from an unsymmetric LU factorization (`P^T A P = L U`).
#[allow(clippy::needless_range_loop)] // CSC/CSR solves index col_ptr/row_ptr + scaling
pub fn solve_lu<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // y_hat = P_row * (D_r b): row-equilibrate then row-permute the RHS.
    let mut y: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm_row[e];
            b[orig] * T::from_real(f.d_row[orig])
        })
        .collect();
    // Forward solve L y = y_hat (CSC, unit diagonal). Column-oriented: once y[e] is
    // final, eliminate it from the rows below. Axpys via `fmadd` (FMA on
    // native builds; see `scalar::fmadd`). The explicit unit diagonal is a
    // column's FIRST entry (rows sorted, lower triangular in elimination
    // numbering), so the sweep starts at `col_ptr[e] + 1` instead of
    // branching on `i != e` at every nonzero.
    for e in 0..n {
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e, "unit diagonal must lead its column");
        let nye = T::zero() - y[e];
        for k in (s + 1)..ee {
            let i = f.l_row_idx[k];
            y[i] = fmadd(f.l_values[k], nye, y[i]);
        }
    }
    // Backward solve U x = y (CSR by row). The pivot is a row's FIRST entry
    // (columns sorted, upper triangular), replacing the per-nonzero
    // diagonal-search branch. Deliberately mul+sub, NOT `fmadd`: the
    // accumulator is a latency-bound serial chain (see `solve_ldlt`'s
    // backward-sweep note).
    let mut x = vec![T::zero(); n];
    for e in (0..n).rev() {
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e, "pivot must lead its row");
        let diag = f.u_values[s];
        let mut acc = y[e];
        for k in (s + 1)..ee {
            acc = acc - f.u_values[k] * x[f.u_col_idx[k]];
        }
        x[e] = acc * diag.recip();
    }
    // Undo the column permutation and apply the column equilibration:
    // x_orig[perm[e]] = D_c[perm[e]] * x_hat[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm[e];
        out[orig] = x[e] * T::from_real(f.d_col[orig]);
    }
    Ok(out)
}

/// Solve `A^T * x = b` against the stored factorization of `A` (feral #94
/// enabler). The factor chain is `A^-1 = D_c P_c (LU)^-1 P_r^T D_r` (see
/// [`solve_lu`]), so `A^-T = D_r P_r L^-T U^-T P_c^T D_c`: column-equilibrate
/// and column-permute the RHS, forward-solve `U^T` (lower triangular; `U` is
/// CSR with the pivot leading each row, so once `z[e]` is final it
/// eliminates from the trailing columns of row `e`, scatter form),
/// backward-solve `L^T` (unit upper; dot column `e` of `L` against the
/// already-solved tail), then row-permute and row-equilibrate the result.
///
/// This is the plain TRANSPOSE, not the conjugate transpose - complex
/// callers wanting `A^-H b` conjugate the RHS and the result around this
/// call (`A^-H b = conj(A^-T conj(b))`), as the condition estimator does.
#[allow(clippy::needless_range_loop)] // scatter-form sweeps write w[e] and w[u_col_idx[k]]
pub fn solve_lu_transpose<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // w_hat = P_c^T (D_c b): column-equilibrate then column-permute the RHS.
    let mut w: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm[e];
            b[orig] * T::from_real(f.d_col[orig])
        })
        .collect();
    // Forward solve U^T z = w_hat, in place in `w`.
    for e in 0..n {
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e, "pivot must lead its row");
        let ze = w[e] * f.u_values[s].recip();
        w[e] = ze;
        if ze != T::zero() {
            let nze = T::zero() - ze;
            for k in (s + 1)..ee {
                let j = f.u_col_idx[k];
                w[j] = fmadd(f.u_values[k], nze, w[j]);
            }
        }
    }
    // Backward solve L^T v = z, in place in `w` (unit diagonal leads each
    // column, so the dot starts at `col_ptr[e] + 1`).
    for e in (0..n).rev() {
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e, "unit diagonal must lead its column");
        let mut acc = w[e];
        for k in (s + 1)..ee {
            acc = acc - f.l_values[k] * w[f.l_row_idx[k]];
        }
        w[e] = acc;
    }
    // x_orig[perm_row[e]] = D_r[perm_row[e]] * v[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm_row[e];
        out[orig] = w[e] * T::from_real(f.d_row[orig]);
    }
    Ok(out)
}

/// Solve `A * X = B` for `nrhs` right-hand sides at once. `b` and the returned
/// `x` are **row-major** `n x nrhs` buffers (`b[i*nrhs + c]` is RHS `c` at row
/// `i`). The `L`/`U` structure is traversed once and each value applied to all
/// `nrhs` columns - faster than `nrhs` separate [`solve_lu`] calls.
/// Below this RHS count / work size the LU block solve runs serially (the
/// parallel gather/scatter overhead only amortizes for wide multi-RHS).
const PAR_SOLVE_MIN_RHS: usize = 8;
const PAR_SOLVE_MIN_WORK: usize = 1 << 18;

/// Solve `A X = B` for `nrhs` right-hand sides. Row-major `n x nrhs` layout, as
/// [`solve_ldlt_many`](crate::solve_ldlt_many). For a wide RHS the columns are
/// split into per-thread chunks (each RHS independent); the result is
/// **bit-identical** to the serial block solve.
pub fn solve_lu_many<T: Scalar>(
    f: &LuFactors<T>,
    b: &[T],
    nrhs: usize,
) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if nrhs == 0 || b.len() != n * nrhs {
        return Err(RslabError::DimensionMismatch {
            expected: n * nrhs,
            got: b.len(),
        });
    }
    let nthreads = rayon::current_num_threads().max(1);
    if nrhs < PAR_SOLVE_MIN_RHS || n * nrhs < PAR_SOLVE_MIN_WORK || nthreads < 2 {
        return solve_lu_block(f, b, nrhs);
    }
    let nchunks = nthreads.min(nrhs);
    let chunk = nrhs.div_ceil(nchunks);
    let ranges: Vec<(usize, usize)> = (0..nchunks)
        .map(|t| (t * chunk, ((t + 1) * chunk).min(nrhs)))
        .filter(|&(a, e)| a < e)
        .collect();
    let parts: Result<Vec<(usize, usize, Vec<T>)>, RslabError> = ranges
        .par_iter()
        .map(|&(c0, c1)| {
            let w = c1 - c0;
            let mut sub = vec![T::zero(); n * w];
            for i in 0..n {
                let ib = i * nrhs;
                let sb = i * w;
                sub[sb..sb + w].copy_from_slice(&b[ib + c0..ib + c1]);
            }
            let xs = solve_lu_block(f, &sub, w)?;
            Ok((c0, c1, xs))
        })
        .collect();
    let parts = parts?;
    let mut x = vec![T::zero(); n * nrhs];
    for (c0, c1, xs) in parts {
        let w = c1 - c0;
        for i in 0..n {
            let ib = i * nrhs;
            let sb = i * w;
            x[ib + c0..ib + c1].copy_from_slice(&xs[sb..sb + w]);
        }
    }
    Ok(x)
}

/// Serial block solve over `nrhs` right-hand sides; fanned over column chunks by
/// the parallel [`solve_lu_many`].
fn solve_lu_block<T: Scalar>(f: &LuFactors<T>, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    // Y_hat = P_row * (D_r B): row-equilibrate then row-permute each RHS block.
    let mut y = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm_row[e];
        let s = T::from_real(f.d_row[orig]);
        let (eb, ob) = (e * nrhs, orig * nrhs);
        for c in 0..nrhs {
            y[eb + c] = b[ob + c] * s;
        }
    }
    // Reusable single-row scratch. Hoisting the row that is reused across an inner
    // sweep into a **local** buffer breaks the apparent aliasing of `y[..]` with
    // itself, so the `nrhs`-wide AXPY kernels below operate on non-aliasing,
    // contiguous slices the compiler can vectorize (and the hoisted row is loaded
    // once per outer step, not once per nonzero).
    let mut row = vec![T::zero(); nrhs];
    // Forward solve L Y = Y_hat (CSC, unit diagonal). `y[e]` (the column's source row)
    // is read by every nonzero of column `e` and is not written in this sweep.
    for e in 0..n {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e);
        for k in (s + 1)..ee {
            let i = f.l_row_idx[k];
            let nlval = T::zero() - f.l_values[k];
            let ib = i * nrhs;
            let tgt = &mut y[ib..ib + nrhs];
            for c in 0..nrhs {
                tgt[c] = fmadd(nlval, row[c], tgt[c]);
            }
        }
    }
    // Backward solve U X = Y (CSR by row), in place in `y`. Accumulate row `e`'s
    // update in the local buffer (the off-diagonal sources `y[c_col]`, `c_col > e`,
    // are already solved and not touched here), then scale and write it back.
    // The pivot leads its row (sorted columns, upper triangular).
    for e in (0..n).rev() {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e);
        let diag = f.u_values[s];
        for k in (s + 1)..ee {
            let nuval = T::zero() - f.u_values[k];
            let cb = f.u_col_idx[k] * nrhs;
            let src = &y[cb..cb + nrhs];
            for c in 0..nrhs {
                row[c] = fmadd(nuval, src[c], row[c]);
            }
        }
        let dinv = diag.recip();
        for c in 0..nrhs {
            y[eb + c] = row[c] * dinv;
        }
    }
    // Undo column permutation + column equilibration: out[perm[e]] = D_c * x_hat[e].
    let mut out = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm[e];
        let s = T::from_real(f.d_col[orig]);
        let (ob, eb) = (orig * nrhs, e * nrhs);
        for c in 0..nrhs {
            out[ob + c] = y[eb + c] * s;
        }
    }
    Ok(out)
}

/// Solve `A x = b` with iterative refinement against the original matrix `a`.
/// Each step computes the residual `r = b - A x` and applies the correction
/// `x <- x + (LU)^-1 r`, stopping once `||r||inf` stops improving or `max_iter` is
/// reached. This recovers the accuracy a static / within-block-pivoted factor
/// loses on ill-conditioned matrices, at the cost of a few extra solves.
pub fn solve_lu_refined<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    max_iter: usize,
) -> Result<Vec<T>, RslabError> {
    Ok(solve_lu_refined_with(f, a, b, &crate::refine::RefinePolicy::steps(max_iter))?.0)
}

/// Iterative refinement of an `LU` solve under an explicit
/// [`RefinePolicy`](crate::refine::RefinePolicy), reporting the achieved
/// backward error.
pub fn solve_lu_refined_with<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    policy: &crate::refine::RefinePolicy,
) -> Result<(Vec<T>, crate::refine::RefineOutcome), RslabError> {
    let n = f.n;
    if a.n != n || b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let mut x = solve_lu(f, b)?;
    let outcome = crate::refine::refine_in_place(a, b, &mut x, policy, |r| solve_lu(f, r))?;
    Ok((x, outcome))
}

#[cfg(test)]
mod tests {

    /// A badly scaled, row-scrambled unsymmetric system: without the MC64
    /// row matching the front-restricted pivoting finds no usable pivot or
    /// loses digits; with it (the default) the componentwise backward error
    /// is roundoff.
    #[test]
    fn lu_matching_bounds_pivot_growth() {
        let m = 40usize;
        let n = m * m;
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            let rs = |i: usize| 10f64.powi(((i * 7919) % 13) as i32 - 6);
            let mut push = |i: usize, v: f64| cols[j].push(((i + 17) % n, v * rs(i)));
            push(j, 4.0);
            if x > 0 {
                push(j - 1, -1.4);
            }
            if x + 1 < m {
                push(j + 1, -0.6);
            }
            if y > 0 {
                push(j - m, -1.0);
            }
            if y + 1 < m {
                push(j + m, -1.0);
            }
        }
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for c in &mut cols {
            c.sort_by_key(|e| e.0);
            for &(r, v) in c.iter() {
                row_idx.push(r);
                values.push(v);
            }
            col_ptr.push(row_idx.len());
        }
        let a = GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        };
        let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
        // Componentwise backward error `max_i |r_i| / (|A||x| + |b|)_i`: the
        // rows span twelve decades, so a normwise residual would only
        // measure the largest rows.
        let omega = |x: &[f64]| {
            let mut r = b.clone();
            let mut d: Vec<f64> = b.iter().map(|v| v.abs()).collect();
            for j in 0..n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    r[a.row_idx[k]] -= a.values[k] * x[j];
                    d[a.row_idx[k]] += a.values[k].abs() * x[j].abs();
                }
            }
            r.iter()
                .zip(&d)
                .map(|(ri, di)| if *di > 0.0 { ri.abs() / di } else { 0.0 })
                .fold(0.0, f64::max)
        };
        let opts = SolverSettings::default().with_threads(1);
        let s = LuSolver::factor(&a, &opts).unwrap();
        assert_eq!(s.diagnostics().decisions.scaling, "Mc64RowMatching");
        let x = s.solve(&b).unwrap();
        assert!(
            omega(&x) < 1e-12,
            "backward error with matching {}",
            omega(&x)
        );
        // Without the matching the shifted rows leave the fully-summed
        // blocks without a usable pivot.
        let s0 = LuSolver::factor(&a, &opts.with_lu_matching(false));
        assert!(s0.is_err() || s0.unwrap().diagnostics().decisions.scaling == "TwoSidedRowCol");
    }
    use super::*;
    use num_complex::Complex;

    fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        let mut y = vec![T::zero(); a.n];
        a.matvec(x, &mut y);
        (0..a.n)
            .map(|i| (y[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn f64_unsymmetric_tridiag() {
        // Unsymmetric real tridiagonal (full storage): diag 4, sub -1, super -2.
        let n = 20;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(4.0);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
                r.push(i);
                c.push(i + 1);
                v.push(-2.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 9.5).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-10, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn lu_solve_many_matches_single() {
        let n = 12;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(5.0_f64);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
                r.push(i);
                c.push(i + 1);
                v.push(-2.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let solver = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
        let nrhs = 5;
        let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
        let x = solver.solve_many(&b, nrhs).unwrap();
        for col in 0..nrhs {
            let bc: Vec<f64> = (0..n).map(|i| b[i * nrhs + col]).collect();
            let xc = solver.solve(&bc).unwrap();
            for i in 0..n {
                assert!(
                    (x[i * nrhs + col] - xc[i]).abs() < 1e-10,
                    "rhs {col} row {i}"
                );
            }
        }
    }

    /// Every column of a block solve is BITWISE the single-column solve, whatever the
    /// block width: a non-flexible Krylov method applies the solve to blocks in its Arnoldi
    /// steps and to one column in its update, and any difference breaks the Arnoldi
    /// relation (a single-precision factor made it 1e-3 on a saddle system). Covers the
    /// leaf subtrees and the ancestor levels (a 2D grid) and the apex nodes with rows below
    /// them (a 3D grid). Under FMA the fused complex multiply-add is not symmetric in its
    /// factors, so the operand order of every single-column branch matters.
    #[test]
    fn lu_block_solve_is_bitwise_the_single_solve() {
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let c = |re: f64, im: f64| Complex::new(re, im);
        let grid = |m: usize, rnd: &mut dyn FnMut() -> f64| {
            let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
            for a in 0..m {
                for b in 0..m {
                    let i = a * m + b;
                    r.push(i);
                    cc.push(i);
                    v.push(c(4.5 + rnd(), rnd()));
                    for j in [(a + 1 < m).then(|| i + m), (b + 1 < m).then(|| i + 1)]
                        .into_iter()
                        .flatten()
                    {
                        r.push(i);
                        cc.push(j);
                        v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                        r.push(j);
                        cc.push(i);
                        v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                    }
                }
            }
            (m * m, r, cc, v)
        };
        // a 3D grid: its top separators are large enough for the apex sweep, with rows
        // below them (the off-block product)
        let grid3 = |m: usize, rnd: &mut dyn FnMut() -> f64| {
            let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
            let id = |a: usize, b: usize, e: usize| (a * m + b) * m + e;
            for a in 0..m {
                for b in 0..m {
                    for e in 0..m {
                        let i = id(a, b, e);
                        r.push(i);
                        cc.push(i);
                        v.push(c(6.5 + rnd(), rnd()));
                        for j in [
                            (a + 1 < m).then(|| id(a + 1, b, e)),
                            (b + 1 < m).then(|| id(a, b + 1, e)),
                            (e + 1 < m).then(|| id(a, b, e + 1)),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            r.push(i);
                            cc.push(j);
                            v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                            r.push(j);
                            cc.push(i);
                            v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                        }
                    }
                }
            }
            (m * m * m, r, cc, v)
        };
        for (n, r, cc, v) in [grid(60, &mut rnd), grid3(24, &mut rnd)] {
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &r, &cc, &v).unwrap();
            let solver = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
            for nrhs in [2usize, 3, 5] {
                let b: Vec<Complex<f64>> = (0..n * nrhs).map(|_| c(rnd(), rnd())).collect();
                let x = solver.solve_many(&b, nrhs).unwrap();
                for col in 0..nrhs {
                    let bc: Vec<Complex<f64>> = (0..n).map(|i| b[i * nrhs + col]).collect();
                    let xc = solver.solve(&bc).unwrap();
                    for i in 0..n {
                        assert!(
                            x[i * nrhs + col] == xc[i],
                            "n {n} nrhs {nrhs} rhs {col} row {i}"
                        );
                    }
                }
            }
        }
    }

    /// An analysis on a given ordering: its own ordering reproduces the factor, a
    /// neighbouring pattern takes it with the elimination tree and counts recomputed, and a
    /// non-permutation is refused.
    #[test]
    fn analysis_on_a_given_ordering() {
        let m = 12;
        let n = m * m;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let i = a * m + b;
                r.push(i);
                c.push(i);
                v.push(4.5 + 0.01 * i as f64);
                for j in [(a + 1 < m).then(|| i + m), (b + 1 < m).then(|| i + 1)]
                    .into_iter()
                    .flatten()
                {
                    r.push(i);
                    c.push(j);
                    v.push(-1.0);
                    r.push(j);
                    c.push(i);
                    v.push(-1.2);
                }
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let o = SolverSettings::default().with_lu_matching(false);
        let s0 = LuSymbolic::analyze_with(&a, &o).unwrap();
        let f0 = s0.factor(&a, &o).unwrap();
        let o1 = o.clone().with_permutation(s0.permutation().into());
        let s1 = LuSymbolic::analyze_with(&a, &o1).unwrap();
        assert_eq!(s1.permutation(), s0.permutation());
        let f1 = s1.factor(&a, &o1).unwrap();
        assert_eq!(f1.factor_nnz(), f0.factor_nnz());
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        assert_eq!(f1.solve(&b).unwrap(), f0.solve(&b).unwrap());
        // a neighbouring pattern (one more coupling) on the same ordering
        let (mut r2, mut c2, mut v2) = (r.clone(), c.clone(), v.clone());
        r2.extend([0, n - 1]);
        c2.extend([n - 1, 0]);
        v2.extend([-0.1, -0.1]);
        let a2 = GeneralCsc::<f64>::from_triplets(n, &r2, &c2, &v2).unwrap();
        let s2 = LuSymbolic::analyze_with(&a2, &o1).unwrap();
        let x = s2.factor(&a2, &o1).unwrap().solve(&b).unwrap();
        let mut ax = vec![0.0f64; n];
        for col in 0..n {
            for k in a2.col_ptr[col]..a2.col_ptr[col + 1] {
                ax[a2.row_idx[k]] += a2.values[k] * x[col];
            }
        }
        let res = ax
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(res < 1e-12, "residual on the reused ordering {res:.1e}");
        // not a permutation
        let bad: Vec<usize> = (0..n).map(|i| i / 2).collect();
        assert!(LuSymbolic::analyze_with(&a, &o.clone().with_permutation(bad.into())).is_err());
    }

    #[test]
    fn pivoting_triggered_small_diagonal() {
        // Small diagonal, large off-diagonals -> partial pivoting fires on
        // (nearly) every column. Well-conditioned overall, so the solve must
        // still hit a tiny residual: this isolates the pivoting/perm logic
        // (correctness) from numerical stability.
        let c = |re, im| Complex::new(re, im);
        let m = 6;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(0.3, 0.05)); // small diagonal
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3)); // large off-diagonal
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(1.5, -0.2));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(1.8, 0.1));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(2.2, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn lu_left_looking_pivoting_small_diagonal() {
        // Small diagonal, large off-diagonals -> restricted partial pivoting must
        // fire on (nearly) every column. The left-looking path (1x1 static) would
        // eliminate on the tiny pivots and lose accuracy; with pivoting it must
        // match the multifrontal and hit a tiny residual.
        let c = |re, im| Complex::new(re, im);
        let m = 6;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(0.05, 0.01)); // tiny diagonal -> threshold pivoting fires
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(1.5, -0.2));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(1.8, 0.1));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(2.2, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let ll = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let mf = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let xl = solve_lu(&ll, &b).unwrap();
        let xm = solve_lu(&mf, &b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); n];
        a.matvec(&xl, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-9, "left-looking pivoting residual {res}");
        let diff = (0..n).map(|i| (xl[i] - xm[i]).norm()).fold(0.0, f64::max);
        assert!(diff < 1e-9, "left-looking vs multifrontal differ {diff}");
    }

    #[test]
    fn lu_pivot_u_knob_wired_and_solves() {
        // The tunable threshold `u` governs the left-looking LU pivot test. On a
        // well-scaled, diagonally-dominant grid the pivot never needs to move, so
        // every `u in [0, 1]` must solve to a tiny residual (the knob changes the
        // factor path but not correctness here). Verifies the field is threaded
        // end-to-end (SolverSettings -> KernelTuning -> kernel) and clamps.
        let c = |re, im| Complex::new(re, im);
        let m = 7;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(12.0, 1.0)); // dominant diagonal -> no interchange needed
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-1.3, -0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.1, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.9, 0.15));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        for u in [0.0f64, 0.1, 0.5, 1.0] {
            let s = SolverSettings::default().with_pivot_u(u);
            let f = factor_general_lu(&a, &s).unwrap();
            let x = solve_lu(&f, &b).unwrap();
            let mut ax = vec![Complex::new(0.0, 0.0); n];
            a.matvec(&x, &mut ax);
            let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
            assert!(res < 1e-9, "pivot_u={u} residual {res}");
        }
        // Out-of-range values clamp into [0, 1].
        assert_eq!(SolverSettings::default().with_pivot_u(5.0).pivot_u, 1.0);
        assert_eq!(SolverSettings::default().with_pivot_u(-2.0).pivot_u, 0.0);
    }

    #[test]
    fn static_pivot_reuse_across_value_sweep() {
        // Solver-in-the-loop: analyze the pattern once, then factor a *sweep* of
        // value sets that share it with static pivoting (`pivot_u = 0`, no pivot
        // search per column). On a diagonally-dominant family each static factor
        // solves accurately, and iterative refinement against the original matrix
        // recovers full accuracy - the frequency-sweep / time-stepping use case.
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut rr, mut cc) = (Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                }
            }
        }
        let template =
            GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
                .unwrap();
        let analysis = LuSymbolic::analyze(&template).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 0.7)).collect();
        let static_opts = SolverSettings::default().with_pivot_u(0.0);
        for shift in [0.0, 1.5, -0.8, 3.0] {
            let vv: Vec<Complex<f64>> = rr
                .iter()
                .zip(&cc)
                .map(|(&i, &j)| {
                    if i == j {
                        c(9.0 + shift, 1.0)
                    } else {
                        c(-1.0, 0.2)
                    }
                })
                .collect();
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
            // Reuse the one analysis; static factor (no pivot search).
            let f = factor_general_lu_numeric(&analysis, &a, &static_opts)
                .map(LuNumeric::into_factors)
                .unwrap();
            let x = solve_lu_refined(&f, &a, &b, 2).unwrap();
            let mut ax = vec![Complex::new(0.0, 0.0); n];
            a.matvec(&x, &mut ax);
            let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
            assert!(res < 1e-9, "static reuse shift={shift} residual {res}");
        }
    }

    #[test]
    fn complex_unsymmetric_2d_grid() {
        // 2D 5-point grid with unsymmetric neighbor couplings (right != left).
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2)); // p,q
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1)); // q,p (different!)
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn lu_left_looking_2d_grid_solves() {
        // Unsymmetric, diagonally dominant complex 2D grid.
        let c = |re, im| Complex::new(re, im);
        let m = 14;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(16.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
        let sym = LuSymbolic::analyze(&a).unwrap();
        let ll = sym.factor(&a, &SolverSettings::default()).unwrap();
        let xl = ll.solve(&b).unwrap();
        let mut am = vec![Complex::new(0.0, 0.0); n];
        a.matvec(&xl, &mut am);
        let res = (0..n).map(|i| (am[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-8, "left-looking LU residual {res}");
    }

    #[test]
    fn complex_f32_lu_solves() {
        // The Complex<f32> LU path (used by the mixed-precision preconditioner).
        let c = |re: f32, im: f32| num_complex::Complex::<f32>::new(re, im);
        let m = 10;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(20.0, 2.0));
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    vv.push(c(-1.0, 0.2));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    vv.push(c(-1.5, 0.3));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<num_complex::Complex<f32>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<num_complex::Complex<f32>> =
            (0..n).map(|i| c((i % 5) as f32 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        let r = resid(&a, &x, &b);
        assert!(r < 1e-3, "f32 LU residual {}", r);
    }

    #[test]
    fn phased_general_lu_analyze_once_factor_many() {
        // PARDISO workflow for the unsymmetric path: analyze the pattern once,
        // factor several value sets that share it - each must match the
        // one-shot factor's solve. The frequency-sweep / Newton use case.
        let c = |re, im| Complex::new(re, im);
        let m = 7;
        let n = m * m;
        let (mut rr, mut cc) = (Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                }
            }
        }
        // Template (values irrelevant) -> analyze once.
        let template =
            GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
                .unwrap();
        let analysis = LuSymbolic::analyze(&template).unwrap();
        assert_eq!(analysis.n(), n);

        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 1.0)).collect();
        for shift in [0.0, 2.5, -1.0] {
            let vv: Vec<Complex<f64>> = rr
                .iter()
                .zip(&cc)
                .map(|(&i, &j)| {
                    if i == j {
                        c(8.0 + shift, 1.0)
                    } else {
                        c(-1.0, 0.2)
                    }
                })
                .collect();
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
            let phased = factor_general_lu_numeric(&analysis, &a, &SolverSettings::default())
                .map(LuNumeric::into_factors)
                .unwrap();
            let one_shot = factor_general_lu(&a, &SolverSettings::default()).unwrap();
            let xp = solve_lu(&phased, &b).unwrap();
            let xo = solve_lu(&one_shot, &b).unwrap();
            for (p, o) in xp.iter().zip(&xo) {
                assert!((p - o).norm() < 1e-10);
            }
            assert!(resid(&a, &xp, &b) < 1e-8);
        }
    }

    #[test]
    fn incomplete_lu_reduces_fill_and_still_solves() {
        use crate::numeric::settings::ZeroPivotAction;
        // Unsymmetric grid: incomplete LU (drop_tol) must shrink nnz(L+U) yet
        // still drive iterative refinement to a small residual - the MoM
        // sparse-preconditioner configuration.
        let c = |re, im| Complex::new(re, im);
        let m = 14;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        let full = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
            ..Default::default()
        };
        let inc = factor_general_lu(&a, &opts).unwrap();
        assert!(
            inc.factor_nnz() < full.factor_nnz(),
            "ILU should reduce fill: {} vs {}",
            inc.factor_nnz(),
            full.factor_nnz()
        );
        // The incomplete factor + a few refinement steps still solves accurately.
        let x = solve_lu_refined(&inc, &a, &b, 10).unwrap();
        assert!(resid(&a, &x, &b) < 1e-6, "residual {}", resid(&a, &x, &b));
    }
}
