//! The automatic path: one fixed pattern, many value sets, each factored by
//! the path its work calls for, in place.
//!
//! [`AutoSymbolic`] holds a [`CscLayout`] and its analysis, shared by every
//! [`AutoSolver`] over it (one per solve loop or sweep worker). The analysis
//! routes the pattern by the work KLU's analysis predicts for it: a pattern
//! whose flops per factor entry stay below
//! [`AutoSettings::supernodal_flops_per_entry`] is sparse work and goes to
//! [`KluSolver`], whose numeric-only refactor replays the frozen pivots; a
//! pattern above it (large fronts, as in a power grid or a mesh) goes to the
//! supernodal paths, whose blocked kernels turn that work into dense
//! products:
//!
//! * [`LdltSolver`] when the pattern is structurally symmetric and the
//!   values of the factorization at hand are symmetric (checked per
//!   factorization, `O(nnz)`), at half the work of an LU;
//! * [`LuSolver`] otherwise, and whenever the LDL^T factorization fails.
//!
//! Every factorization after a path's first refactors into that path's
//! factor in place, and a value set equal to the last factored one keeps its
//! factor without any numeric work. The solves are those of the three
//! solvers ([`solve_into`](AutoSolver::solve_into) in lent work allocates
//! nothing), and an [`AutoSolver`] is a [`Factorization`](crate::Factorization)
//! and a [`Preconditioner`](crate::Preconditioner) like them.

use std::sync::{Arc, OnceLock};

use crate::error::RslabError;
use crate::numeric::direct::{SolveCore, SolveWork};
use crate::numeric::klu::{KluSettings, KluSolver, KluSymbolic};
use crate::numeric::ldlt::{LdltSolver, LdltSymbolic};
use crate::numeric::lu::{LuSolver, LuSymbolic};
use crate::numeric::settings::{SolverSettings, Threads};
use crate::pencil::Pencil;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use crate::sparse::layout::CscLayout;

/// The settings of the automatic path: each path's own, and the routing
/// threshold between them.
#[derive(Debug, Clone)]
pub struct AutoSettings {
    /// The KLU path's analysis and factorization.
    pub klu: KluSettings,
    /// The supernodal LDL^T and LU paths' analyses and factorizations.
    pub supernodal: SolverSettings,
    /// The flops per factor entry, as KLU's analysis predicts them, from
    /// which a pattern is supernodal work. Default `25`: below it KLU's
    /// refactor is faster on every matrix class measured, above it the
    /// supernodal kernels win by a growing factor.
    pub supernodal_flops_per_entry: f64,
    /// The tolerance of the value-symmetry test that selects LDL^T on a
    /// structurally symmetric supernodal pattern: `|a_ij - a_ji| <= tol *
    /// max|a|`. Default `0`, exact symmetry; above it LDL^T factors the
    /// lower triangle of a matrix symmetric only within `tol`.
    pub symmetry_tol: f64,
}

impl Default for AutoSettings {
    fn default() -> Self {
        Self {
            klu: KluSettings::default(),
            supernodal: SolverSettings::default(),
            supernodal_flops_per_entry: crate::constants::AUTO_SUPERNODAL_FLOPS_PER_ENTRY,
            symmetry_tol: crate::constants::AUTO_SYMMETRY_TOL,
        }
    }
}

impl AutoSettings {
    pub fn with_klu(mut self, klu: KluSettings) -> Self {
        self.klu = klu;
        self
    }

    pub fn with_supernodal(mut self, supernodal: SolverSettings) -> Self {
        self.supernodal = supernodal;
        self
    }

    pub fn with_supernodal_flops_per_entry(mut self, flops: f64) -> Self {
        self.supernodal_flops_per_entry = flops;
        self
    }

    pub fn with_symmetry_tol(mut self, tol: f64) -> Self {
        self.symmetry_tol = tol;
        self
    }
}

/// The path a factorization of the automatic path took.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AutoBackend {
    Klu,
    Ldlt,
    Lu,
}

struct LdltCandidate {
    tpair: Vec<usize>,
    lower_from: Vec<usize>,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    sym: LdltSymbolic,
}

impl LdltCandidate {
    fn build(layout: &CscLayout, opts: &SolverSettings, amd: &[usize]) -> Option<Self> {
        let tpair = layout.transpose_slots()?;
        let (n, cp, ri) = (layout.n(), layout.col_ptr(), layout.row_idx());
        let mut col_ptr = Vec::with_capacity(n + 1);
        col_ptr.push(0);
        let (mut row_idx, mut lower_from) = (Vec::new(), Vec::new());
        for c in 0..n {
            for (k, &r) in (cp[c]..cp[c + 1]).zip(&ri[cp[c]..cp[c + 1]]) {
                if r >= c {
                    row_idx.push(r);
                    lower_from.push(k);
                }
            }
            col_ptr.push(row_idx.len());
        }
        let skeleton = CscMatrix::<f64> {
            n,
            col_ptr: col_ptr.clone(),
            row_idx: row_idx.clone(),
            values: vec![1.0; row_idx.len()],
        };
        let sym = LdltSymbolic::analyze_from(&skeleton, opts, Some(amd)).ok()?;
        Some(Self {
            tpair,
            lower_from,
            col_ptr,
            row_idx,
            sym,
        })
    }

    fn equals_transpose<T: Scalar>(&self, vals: &[T], tol: f64) -> bool {
        if tol == 0.0 {
            return (vals.iter().zip(&self.tpair)).all(|(&v, &t)| v == vals[t]);
        }
        let bound = tol * vals.iter().fold(0.0f64, |m, v| m.max(v.magnitude()));
        (vals.iter().zip(&self.tpair)).all(|(&v, &t)| (v - vals[t]).magnitude() <= bound)
    }

    fn lower<T: Scalar>(&self, vals: &[T], lower: &mut Option<CscMatrix<T>>) {
        let m = lower.get_or_insert_with(|| CscMatrix {
            n: self.col_ptr.len() - 1,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values: vec![T::zero(); self.row_idx.len()],
        });
        for (v, &k) in m.values.iter_mut().zip(&self.lower_from) {
            *v = vals[k];
        }
    }

    fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.tpair)
            + vec_bytes(&self.lower_from)
            + vec_bytes(&self.col_ptr)
            + vec_bytes(&self.row_idx)
            + self.sym.heap_bytes()
    }
}

enum Route {
    Klu(Box<KluSymbolic>),
    Supernodal {
        ldlt: Option<Box<LdltCandidate>>,
        /// Made when a value set first needs it (eagerly by
        /// [`AutoSymbolic::analyze`]).
        lu: OnceLock<Box<LuSymbolic>>,
        /// The routing KLU analysis' ordering (minimum degree within its
        /// blocks): the minimum-degree candidate of the supernodal analyses'
        /// ordering race, which therefore does not order the pattern so
        /// again.
        amd: Vec<usize>,
    },
}

/// A fixed pattern ([`CscLayout`]) with the analysis of the path its work
/// selected, shared by every [`AutoSolver`] over it.
///
/// The analysis reads values (KLU's and LU's row matchings), so it is made
/// on a value set: [`analyze`](Self::analyze) takes a reference one, and a
/// pattern from [`new`](Self::new) is analyzed on the values of its first
/// factorization. Concurrent solvers over one pattern should share an
/// analyzed one, so the result does not depend on which factors first.
pub struct AutoSymbolic {
    layout: CscLayout,
    route: OnceLock<Route>,
}

impl AutoSymbolic {
    /// The pattern of `layout`, analyzed at its first factorization. Fails
    /// with [`RslabError::StructurallySingular`] if no value set can make it
    /// regular.
    pub fn new(layout: CscLayout) -> Result<Self, RslabError> {
        if layout.structural_rank() < layout.n() {
            return Err(RslabError::StructurallySingular);
        }
        Ok(Self {
            layout,
            route: OnceLock::new(),
        })
    }

    /// The pattern of `layout`, analyzed now on `values` (one per slot):
    /// every factorization is then independent of the order solvers factor
    /// in.
    pub fn analyze<T: Scalar>(
        layout: CscLayout,
        values: &[T],
        settings: &AutoSettings,
    ) -> Result<Self, RslabError> {
        let sym = Self::new(layout)?;
        let a = sym.layout.matrix(values.to_vec());
        let route = sym.route(&a, settings, true)?;
        let _ = sym.route.set(route);
        Ok(sym)
    }

    pub fn layout(&self) -> &CscLayout {
        &self.layout
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.layout.n()
    }

    /// Whether the pattern routed to the supernodal paths; `None` before its
    /// analysis.
    pub fn is_supernodal(&self) -> Option<bool> {
        self.route
            .get()
            .map(|r| matches!(r, Route::Supernodal { .. }))
    }

    /// Heap bytes the layout and the analyses made so far hold.
    pub fn heap_bytes(&self) -> u64 {
        self.layout.heap_bytes()
            + match self.route.get() {
                None => 0,
                Some(Route::Klu(k)) => k.heap_bytes(),
                Some(Route::Supernodal { ldlt, lu, amd }) => {
                    ldlt.as_ref().map_or(0, |c| c.heap_bytes())
                        + lu.get().map_or(0, |s| s.heap_bytes())
                        + crate::memory::vec_bytes(amd)
                }
            }
    }

    /// The heap a solver over this pattern needs (see
    /// [`MemoryPlan`](crate::MemoryPlan)), for `nrhs` right-hand sides per
    /// solve; `None` before the analysis. On the supernodal paths a solver
    /// keeps the LDL^T and the LU factor side by side once value sets of both
    /// kinds came, so their footprints add (the LU's where it is analyzed),
    /// and the peaks are the larger one's.
    pub fn memory_plan<T: Scalar>(
        &self,
        settings: &AutoSettings,
        nrhs: usize,
    ) -> Option<crate::MemoryPlan> {
        match self.route.get()? {
            Route::Klu(k) => Some(k.memory_plan::<T>(&settings.klu, nrhs)),
            Route::Supernodal { ldlt, lu, .. } => {
                let opts = &settings.supernodal;
                let a = ldlt.as_ref().map(|c| c.sym.memory_plan::<T>(opts, nrhs));
                let b = lu.get().map(|s| s.memory_plan::<T>(opts, nrhs));
                match (a, b) {
                    (Some(a), Some(b)) => Some(crate::MemoryPlan {
                        threads: a.threads.max(b.threads),
                        nrhs,
                        analysis_bytes: a.analysis_bytes + b.analysis_bytes,
                        analysis_growth_bytes: a.analysis_growth_bytes + b.analysis_growth_bytes,
                        factor_bytes: a.factor_bytes + b.factor_bytes,
                        factor_peak_bytes: a.factor_peak_bytes.max(b.factor_peak_bytes),
                        solve_bytes: a.solve_bytes.max(b.solve_bytes),
                        kept_bytes: a.kept_bytes + b.kept_bytes,
                    }),
                    (a, b) => a.or(b),
                }
            }
        }
    }

    /// A solver over this pattern, for one solve loop or one sweep worker.
    pub fn factorizer<T: Scalar>(&self) -> AutoSolver<'_, T> {
        AutoSolver::over(Analysis::Borrowed(self))
    }

    fn route<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        settings: &AutoSettings,
        eager: bool,
    ) -> Result<Route, RslabError> {
        let klu = KluSymbolic::analyze(a, &settings.klu)?;
        let per_entry = klu.estimated_flops() as f64 / klu.estimated_factor_nnz().max(1) as f64;
        let supernodal = per_entry >= settings.supernodal_flops_per_entry;
        rsprobe::debug!(
            "auto: n={} nnz={} flops/entry={per_entry:.1} -> {}",
            a.n,
            a.nnz(),
            if supernodal { "supernodal" } else { "klu" }
        );
        if !supernodal {
            return Ok(Route::Klu(Box::new(klu)));
        }
        let amd = klu.into_ordering();
        let opts = &settings.supernodal;
        let lu = OnceLock::new();
        if eager {
            let _ = lu.set(Box::new(LuSymbolic::analyze_from(a, opts, Some(&amd))?));
        }
        Ok(Route::Supernodal {
            ldlt: LdltCandidate::build(&self.layout, opts, &amd).map(Box::new),
            lu,
            amd,
        })
    }
}

#[derive(Clone)]
enum Analysis<'s> {
    Borrowed(&'s AutoSymbolic),
    Held(Arc<AutoSymbolic>),
}

impl Analysis<'_> {
    fn get(&self) -> &AutoSymbolic {
        match self {
            Analysis::Borrowed(s) => s,
            Analysis::Held(s) => s,
        }
    }
}

fn in_place<S>(
    slot: &mut Option<S>,
    refactor: impl FnOnce(&mut S) -> Result<(), RslabError>,
    factor: impl FnOnce() -> Result<S, RslabError>,
) -> Result<(), RslabError> {
    match slot {
        Some(s) => refactor(s),
        None => {
            *slot = Some(factor()?);
            Ok(())
        }
    }
}

/// A factorization on the automatic path: the values to factor (one per
/// slot of the layout), each path's factor, kept while another path serves
/// a value set so that its next factorization is again in place, and the
/// path the last successful factorization took.
///
/// [`AutoSymbolic::factorizer`] makes one over a shared analysis for a
/// solve loop ([`assemble`](Self::assemble) or
/// [`values_mut`](Self::values_mut), then [`refactor`](Self::refactor);
/// or a member of a [`Pencil`] with [`factor_member`](Self::factor_member));
/// [`AutoSolver::factor`] analyzes and factors one matrix, like the
/// one-shot forms of the other solvers. The solves are theirs: `solve`,
/// `solve_many`, `solve_transpose`, the `_into` forms in lent
/// [`SolveWork`], and the refined ones.
#[derive(Clone)]
pub struct AutoSolver<'s, T> {
    sym: Analysis<'s>,
    csc: GeneralCsc<T>,
    lower: Option<CscMatrix<T>>,
    klu: Option<KluSolver<T>>,
    ldlt: Option<LdltSolver<T>>,
    lu: Option<LuSolver<T>>,
    active: Option<AutoBackend>,
    prev: Vec<T>,
    member: Option<(u64, u64, T)>,
    member_extra: Vec<T>,
    solve_threads: Threads,
    solves: crate::diagnostics::SolveCounter,
    works: crate::numeric::direct::WorkPool<T>,
    pinned: bool,
    own: Option<KluSolver<T>>,
}

impl<T: Scalar> AutoSolver<'static, T> {
    /// A solver holding a share of `sym`, so it outlives any borrow (the
    /// [`factorizer`](AutoSymbolic::factorizer) of a held analysis).
    pub fn new(sym: Arc<AutoSymbolic>) -> Self {
        AutoSolver::over(Analysis::Held(sym))
    }

    /// Analyze and factor `a` on the path its work selects, the solver holding
    /// its analysis: the one-shot form.
    pub fn factor(a: &GeneralCsc<T>, settings: &AutoSettings) -> Result<Self, RslabError> {
        let sym = AutoSymbolic::new(CscLayout::of(a)?)?;
        let mut s = AutoSolver::new(Arc::new(sym));
        s.values_mut().copy_from_slice(&a.values);
        s.refactor(settings)?;
        Ok(s)
    }
}

impl<'s, T: Scalar> AutoSolver<'s, T> {
    fn over(sym: Analysis<'s>) -> Self {
        let layout = sym.get().layout();
        let csc = layout.matrix(vec![T::zero(); layout.nnz()]);
        Self {
            sym,
            csc,
            lower: None,
            klu: None,
            ldlt: None,
            lu: None,
            active: None,
            prev: Vec::new(),
            member: None,
            member_extra: Vec::new(),
            solve_threads: Threads::Ambient,
            solves: Default::default(),
            works: Default::default(),
            pinned: false,
            own: None,
        }
    }

    /// The factorization this solver holds as the origin of independent
    /// points (the frequencies of a sweep): a copy that factors every value
    /// set from it. The fork replays the origin's pivot order; where a pivot
    /// vanishes it factors that value set afresh without adopting the new
    /// order. So what a fork makes of a value set depends on the value set
    /// and the origin alone, never on what the fork factored before: forks
    /// of one origin factor the points in any grouping, on any threads, to
    /// the same bits. A fork of an origin without a factor factors every
    /// value set afresh. (The supernodal paths factor every value set as a
    /// fresh factorization would, a fork or not.)
    pub fn fork(&self) -> Self {
        let mut f = self.clone();
        f.pinned = true;
        if f.own.take().is_some() {
            f.active = None;
            f.prev.clear();
            f.member = None;
        }
        f
    }

    fn klu_factor(&self) -> Option<&KluSolver<T>> {
        self.own.as_ref().or(self.klu.as_ref())
    }

    /// The values the next [`refactor`](Self::refactor) factors, one per
    /// slot of the layout.
    pub fn values_mut(&mut self) -> &mut [T] {
        self.member = None;
        &mut self.csc.values
    }

    /// The matrix the next (or the last) factorization factors.
    pub fn matrix(&self) -> &GeneralCsc<T> {
        &self.csc
    }

    /// Load the entry values `entries` (the layout's entry order,
    /// duplicates summed), see [`CscLayout::scatter`].
    pub fn assemble(&mut self, entries: &[T]) {
        self.member = None;
        let layout = self.sym.get().layout();
        layout.scatter(entries, &mut self.csc.values);
    }

    /// Factor member `(alpha, extra)` of `pencil` (whose layout this
    /// analysis is over, see [`Pencil::fill`]) in place, as
    /// [`refactor`](Self::refactor) does. The factor is kept, with no fill
    /// and no comparison of values, when the pencil, its generation (see
    /// [`Pencil::update`]), `alpha` and `extra` equal those of the last
    /// successful factorization. `Ok(true)`: numeric work ran. After an
    /// error no factor is held.
    pub fn factor_member(
        &mut self,
        pencil: &Pencil,
        alpha: T,
        extra: &[T],
        settings: &AutoSettings,
    ) -> Result<bool, RslabError> {
        let (id, generation) = pencil.key();
        if self.active.is_some()
            && self.member == Some((id, generation, alpha))
            && self.member_extra == extra
        {
            return Ok(false);
        }
        if pencil.layout().nnz() != self.csc.values.len() || extra.len() != pencil.extras() {
            return Err(RslabError::DimensionMismatch {
                expected: self.csc.values.len(),
                got: pencil.layout().nnz(),
            });
        }
        pencil.fill(alpha, extra, &mut self.csc.values);
        self.factor_now(settings)?;
        self.member = Some((id, generation, alpha));
        self.member_extra.clear();
        self.member_extra.extend_from_slice(extra);
        Ok(true)
    }

    /// The path of the last successful factorization; `None` without one.
    pub fn backend(&self) -> Option<AutoBackend> {
        self.active
    }

    /// Factor the loaded values, in place of the last factor of the path
    /// they take. A value set equal to the last factored one keeps its
    /// factor; after an error no factor is held.
    pub fn refactor(&mut self, settings: &AutoSettings) -> Result<AutoBackend, RslabError> {
        if let Some(b) = self.active {
            if self.prev == self.csc.values {
                return Ok(b);
            }
        }
        let b = self.factor_now(settings)?;
        self.prev.extend_from_slice(&self.csc.values);
        Ok(b)
    }

    fn factor_now(&mut self, settings: &AutoSettings) -> Result<AutoBackend, RslabError> {
        self.active = None;
        self.prev.clear();
        self.member = None;
        let b = self.factor_values(settings)?;
        self.active = Some(b);
        self.solve_threads = match b {
            AutoBackend::Klu => self
                .klu_factor()
                .map(crate::Preconditioner::<T>::solve_threads),
            AutoBackend::Ldlt => self
                .ldlt
                .as_ref()
                .map(crate::Preconditioner::<T>::solve_threads),
            AutoBackend::Lu => self
                .lu
                .as_ref()
                .map(crate::Preconditioner::<T>::solve_threads),
        }
        .unwrap_or(Threads::Ambient);
        Ok(b)
    }

    fn factor_values(&mut self, settings: &AutoSettings) -> Result<AutoBackend, RslabError> {
        let sym = self.sym.get();
        let route = match sym.route.get() {
            Some(r) => r,
            None => {
                let r = sym.route(&self.csc, settings, false)?;
                sym.route.get_or_init(|| r)
            }
        };
        let csc = &self.csc;
        match route {
            Route::Klu(ks) => {
                if let Some(s) = &mut self.klu {
                    if s.refactor(csc).is_ok() {
                        self.own = None;
                        return Ok(AutoBackend::Klu);
                    }
                }
                if self.pinned {
                    self.own = None;
                    self.own = Some(ks.factor(csc, &settings.klu)?);
                    return Ok(AutoBackend::Klu);
                }
                self.klu = None;
                self.klu = Some(ks.factor(csc, &settings.klu)?);
                Ok(AutoBackend::Klu)
            }
            Route::Supernodal { ldlt, lu, amd } => {
                let opts = &settings.supernodal;
                if let Some(cand) = ldlt {
                    if cand.equals_transpose(&csc.values, settings.symmetry_tol) {
                        cand.lower(&csc.values, &mut self.lower);
                        if let Some(lower) = &self.lower {
                            let done = in_place(
                                &mut self.ldlt,
                                |s| cand.sym.refactor(lower, opts, s),
                                || cand.sym.factor(lower, opts),
                            );
                            if done.is_ok() {
                                return Ok(AutoBackend::Ldlt);
                            }
                        }
                    }
                }
                let ls = match lu.get() {
                    Some(s) => s,
                    None => {
                        let s = LuSymbolic::analyze_from(csc, opts, Some(amd))?;
                        lu.get_or_init(|| Box::new(s))
                    }
                };
                in_place(
                    &mut self.lu,
                    |s| ls.refactor(csc, opts, s),
                    || ls.factor(csc, opts),
                )?;
                Ok(AutoBackend::Lu)
            }
        }
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.csc.n
    }

    /// Stored entries of the active factor; `0` without one.
    pub fn factor_nnz(&self) -> usize {
        match self.active {
            Some(AutoBackend::Klu) => self.klu_factor().map_or(0, |s| s.factor_nnz()),
            Some(AutoBackend::Ldlt) => self.ldlt.as_ref().map_or(0, |s| s.factor_nnz()),
            Some(AutoBackend::Lu) => self.lu.as_ref().map_or(0, |s| s.factor_nnz()),
            None => 0,
        }
    }

    /// The active factor's [`Diagnostics`](crate::Diagnostics) with this
    /// solver's solve accounts; empty without a factor.
    pub fn diagnostics(&self) -> crate::Diagnostics {
        let mut d = match self.active {
            Some(AutoBackend::Klu) => self.klu_factor().map(|s| s.diagnostics()),
            Some(AutoBackend::Ldlt) => self.ldlt.as_ref().map(|s| s.diagnostics()),
            Some(AutoBackend::Lu) => self.lu.as_ref().map(|s| s.diagnostics()),
            None => None,
        }
        .unwrap_or_default();
        d.solves = self.solves.snapshot();
        d
    }

    /// Heap bytes this solver holds: its values, every factor it keeps, and
    /// an analysis no other holds.
    pub fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        let own = match &self.sym {
            Analysis::Held(s) if Arc::strong_count(s) == 1 => s.heap_bytes(),
            _ => 0,
        };
        own + vec_bytes(&self.csc.col_ptr)
            + vec_bytes(&self.csc.row_idx)
            + vec_bytes(&self.csc.values)
            + vec_bytes(&self.prev)
            + vec_bytes(&self.member_extra)
            + self.lower.as_ref().map_or(0, |m| {
                vec_bytes(&m.col_ptr) + vec_bytes(&m.row_idx) + vec_bytes(&m.values)
            })
            + self.klu.as_ref().map_or(0, |s| s.heap_bytes())
            + self.own.as_ref().map_or(0, |s| s.heap_bytes())
            + self.ldlt.as_ref().map_or(0, |s| s.heap_bytes())
            + self.lu.as_ref().map_or(0, |s| s.heap_bytes())
    }
}

impl<T: Scalar> SolveCore<T> for AutoSolver<'_, T> {
    const NAME: &'static str = "auto";

    fn dim(&self) -> usize {
        self.csc.n
    }

    fn solve_raw_into(
        &self,
        b: &[T],
        nrhs: usize,
        transpose: bool,
        x: &mut [T],
        work: &mut SolveWork<T>,
    ) -> Result<(), RslabError> {
        match self.active {
            Some(AutoBackend::Klu) => self
                .klu_factor()
                .map(|s| s.solve_raw_into(b, nrhs, transpose, x, work)),
            Some(AutoBackend::Ldlt) => self
                .ldlt
                .as_ref()
                .map(|s| s.solve_raw_into(b, nrhs, transpose, x, work)),
            Some(AutoBackend::Lu) => self
                .lu
                .as_ref()
                .map(|s| s.solve_raw_into(b, nrhs, transpose, x, work)),
            None => None,
        }
        .unwrap_or_else(|| {
            Err(RslabError::InvalidInput(
                "auto: no successful factorization to solve with".to_string(),
            ))
        })
    }

    fn counter(&self) -> &crate::diagnostics::SolveCounter {
        &self.solves
    }
    fn works(&self) -> &crate::numeric::direct::WorkPool<T> {
        &self.works
    }

    fn perturbations(&self) -> &[crate::PivotFloor<T>] {
        match self.active {
            Some(AutoBackend::Klu) => self.klu_factor().map_or(&[], |s| s.perturbations()),
            Some(AutoBackend::Ldlt) => self.ldlt.as_ref().map_or(&[], |s| s.perturbations()),
            Some(AutoBackend::Lu) => self.lu.as_ref().map_or(&[], |s| s.perturbations()),
            None => &[],
        }
    }

    fn log_det_raw(&self) -> Result<crate::numeric::direct::LogDet<T>, RslabError> {
        match self.active {
            Some(AutoBackend::Klu) => self.klu_factor().map(|s| s.log_det_raw()),
            Some(AutoBackend::Ldlt) => self.ldlt.as_ref().map(|s| s.log_det_raw()),
            Some(AutoBackend::Lu) => self.lu.as_ref().map(|s| s.log_det_raw()),
            None => None,
        }
        .unwrap_or_else(|| {
            Err(RslabError::InvalidInput(
                "auto: no successful factorization to solve with".to_string(),
            ))
        })
    }
}

crate::numeric::direct::direct_solver!(AutoSolver<'s>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_constants::{solve_backward_error, U64, UC64};
    use crate::test_grid::grid;
    use num_complex::Complex;

    fn layout(n: usize, rows: &[usize], cols: &[usize]) -> AutoSymbolic {
        AutoSymbolic::new(CscLayout::new(n, rows, cols).unwrap()).unwrap()
    }

    fn close(x: &[f64], want: &[f64]) -> bool {
        x.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12)
    }

    #[test]
    fn sums_duplicates_and_solves_every_way() {
        let sym = layout(2, &[0, 0, 0, 1], &[0, 0, 1, 1]);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&[1.5, 0.5, 1.0, 3.0]);
        assert_eq!(
            f.refactor(&AutoSettings::default()).unwrap(),
            AutoBackend::Klu
        );
        assert!(close(&f.solve(&[5.0, 6.0]).unwrap(), &[1.5, 2.0]));
        assert!(close(&f.solve_transpose(&[2.0, 7.0]).unwrap(), &[1.0, 2.0]));
        let mut x = [0.0; 4];
        let mut w = SolveWork::new();
        f.solve_many_into(&[5.0, 6.0, 2.0, 3.0], 2, &mut x, &mut w)
            .unwrap();
        assert!(close(&x, &[1.5, 2.0, 0.5, 1.0]));
        assert_eq!(f.diagnostics().solves.rhs, 4);
        assert!(f.heap_bytes() > 0 && f.factor_nnz() >= 3);
    }

    #[test]
    fn forks_factor_a_value_set_whatever_they_factored_before() {
        let sym = layout(2, &[0, 0, 1, 1], &[0, 1, 0, 1]);
        let s = AutoSettings::default();
        let mut origin = sym.factorizer::<f64>();
        origin.assemble(&[4.0, 1.0, 1.0, 3.0]);
        origin.refactor(&s).unwrap();
        let point = [2.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        let mut a = origin.fork();
        a.assemble(&[0.0, 1.0, 1.0, 3.0]);
        a.refactor(&s).unwrap();
        assert!(a.own.is_some(), "the vanished pivot factored afresh");
        assert!(close(&a.solve(&[1.0, 3.0]).unwrap(), &[0.0, 1.0]));
        a.assemble(&point);
        a.refactor(&s).unwrap();
        let mut fresh = origin.fork();
        fresh.assemble(&point);
        fresh.refactor(&s).unwrap();
        let (xa, xf) = (a.solve(&b).unwrap(), fresh.solve(&b).unwrap());
        let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&xa), bits(&xf));
        assert!(close(&xa, &[0.2, 0.6]));
    }

    #[test]
    fn forks_of_an_origin_without_a_factor_factor_every_value_set_afresh() {
        let sym = layout(2, &[0, 0, 1, 1], &[0, 1, 0, 1]);
        let s = AutoSettings::default();
        let mut origin = sym.factorizer::<f64>();
        origin.assemble(&[0.0, 0.0, 0.0, 0.0]);
        assert!(origin.refactor(&s).is_err());
        let (first, point) = ([4.0, 1.0, 1.0, 3.0], [1e-5, 1.0, 1.0, 3.0]);
        let b = [1.0, 2.0];
        let mut a = origin.fork();
        a.assemble(&first);
        a.refactor(&s).unwrap();
        a.assemble(&point);
        a.refactor(&s).unwrap();
        let mut fresh = origin.fork();
        fresh.assemble(&point);
        fresh.refactor(&s).unwrap();
        let mut own = sym.factorizer::<f64>();
        own.assemble(&point);
        own.refactor(&s).unwrap();
        let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let xo = bits(&own.solve(&b).unwrap());
        assert_eq!(bits(&a.solve(&b).unwrap()), xo);
        assert_eq!(bits(&fresh.solve(&b).unwrap()), xo);
    }

    #[test]
    fn tracks_new_values_and_reports_singular() {
        let sym = layout(2, &[0, 0, 1, 1], &[0, 1, 0, 1]);
        let s = AutoSettings::default();
        let mut f = sym.factorizer::<f64>();
        f.assemble(&[2.0, 1.0, 0.0, 3.0]);
        f.refactor(&s).unwrap();
        assert!(close(&f.solve(&[5.0, 6.0]).unwrap(), &[1.5, 2.0]));
        f.assemble(&[4.0, 2.0, 0.0, 6.0]);
        f.refactor(&s).unwrap();
        assert!(close(&f.solve(&[10.0, 12.0]).unwrap(), &[1.5, 2.0]));
        f.assemble(&[1.0, 2.0, 2.0, 4.0]);
        assert!(f.refactor(&s).is_err());
        assert_eq!(f.backend(), None);
        assert!(f.solve(&[1.0, 1.0]).is_err());
        f.assemble(&[1.0, 2.0, 2.0, 5.0]);
        f.refactor(&s).unwrap();
    }

    #[test]
    fn structurally_singular_pattern_is_rejected() {
        let l = CscLayout::new(2, &[0, 1], &[0, 0]).unwrap();
        assert!(matches!(
            AutoSymbolic::new(l),
            Err(RslabError::StructurallySingular)
        ));
    }

    fn laplacian(rows: &[usize], cols: &[usize]) -> Vec<f64> {
        (rows.iter().zip(cols))
            .map(|(&r, &c)| if r == c { 4.0 } else { -1.0 })
            .collect()
    }

    fn row_sums<T: Scalar>(n: usize, rows: &[usize], vals: &[T]) -> Vec<T> {
        let mut b = vec![T::zero(); n];
        for (&r, &v) in rows.iter().zip(vals) {
            b[r] = b[r] + v;
        }
        b
    }

    fn abs_row_sums<T: Scalar>(n: usize, rows: &[usize], vals: &[T]) -> Vec<f64> {
        let mut s = vec![0.0f64; n];
        for (&r, v) in rows.iter().zip(vals) {
            s[r] += v.magnitude();
        }
        s
    }

    fn backward_error<T: Scalar>(
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
        x: &[T],
        b: &[T],
    ) -> f64 {
        let mut r = b.to_vec();
        for ((&i, &j), &v) in rows.iter().zip(cols).zip(vals) {
            r[i] = r[i] - v * x[j];
        }
        let inf = |v: &[T]| v.iter().map(|e| e.magnitude()).fold(0.0, f64::max);
        let a_norm = abs_row_sums(b.len(), rows, vals)
            .into_iter()
            .fold(0.0, f64::max);
        inf(&r) / (a_norm * inf(x) + inf(b))
    }

    #[test]
    fn sparse_work_takes_klu_at_any_size() {
        let n = 20_000;
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            if i + 1 < n {
                rows.extend([i, i + 1]);
                cols.extend([i + 1, i]);
            }
        }
        let sym = layout(n, &rows, &cols);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&laplacian(&rows, &cols));
        assert_eq!(
            f.refactor(&AutoSettings::default()).unwrap(),
            AutoBackend::Klu
        );
        assert_eq!(sym.is_supernodal(), Some(false));
    }

    #[test]
    fn dense_work_takes_ldlt_then_lu_then_ldlt_in_place() {
        let (rows, cols) = grid(130);
        let n = 130 * 130;
        let s = AutoSettings::default();
        let sym = layout(n, &rows, &cols);
        let vals = laplacian(&rows, &cols);
        let b = row_sums(n, &rows, &vals);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&vals);
        assert_eq!(f.refactor(&s).unwrap(), AutoBackend::Ldlt);
        let x = f.solve(&b).unwrap();
        assert!(close(&x, &vec![1.0; n]));

        let mut asym = vals.clone();
        asym[1] = -0.5;
        f.assemble(&asym);
        assert_eq!(f.refactor(&s).unwrap(), AutoBackend::Lu);
        let x2 = f.solve(&row_sums(n, &rows, &asym)).unwrap();
        assert!(close(&x2, &vec![1.0; n]));
        let col_sums = row_sums(n, &cols, &asym);
        assert!(close(&f.solve_transpose(&col_sums).unwrap(), &vec![1.0; n]));
        let plan = sym.memory_plan::<f64>(&s, 1).unwrap();
        assert!(plan.factor_bytes > 0 && plan.analysis_bytes > 0);

        f.assemble(&vals);
        assert_eq!(f.refactor(&s).unwrap(), AutoBackend::Ldlt);
        let xr = f.solve(&b).unwrap();
        assert!(x.iter().zip(&xr).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn one_shot_owns_its_analysis_and_matches_the_shared_one() {
        let (rows, cols) = grid(130);
        let n = 130 * 130;
        let vals = laplacian(&rows, &cols);
        let a = GeneralCsc::from_triplets(n, &rows, &cols, &vals).unwrap();
        let s = AutoSettings::default();
        let one = AutoSolver::factor(&a, &s).unwrap();
        let sym = layout(n, &rows, &cols);
        let mut shared = sym.factorizer::<f64>();
        shared.assemble(&vals);
        shared.refactor(&s).unwrap();
        assert_eq!(one.backend(), shared.backend());
        let b = row_sums(n, &rows, &vals);
        let (x1, x2) = (one.solve(&b).unwrap(), shared.solve(&b).unwrap());
        assert!(x1.iter().zip(&x2).all(|(a, b)| a.to_bits() == b.to_bits()));
        let f: &dyn crate::Factorization<f64> = &one;
        assert_eq!((f.n(), f.n_perturbed()), (n, 0));
    }

    #[test]
    fn symmetry_within_tolerance_takes_ldlt() {
        let (rows, cols) = grid(130);
        let n = 130 * 130;
        let mut vals = laplacian(&rows, &cols);
        let skew = 1e-14;
        vals[1] *= 1.0 + skew;
        let sym = Arc::new(layout(n, &rows, &cols));
        let mut exact = AutoSolver::<f64>::new(sym.clone());
        exact.assemble(&vals);
        assert_eq!(
            exact.refactor(&AutoSettings::default()).unwrap(),
            AutoBackend::Lu
        );
        let mut within = AutoSolver::<f64>::new(sym);
        within.assemble(&vals);
        let s = AutoSettings::default().with_symmetry_tol(1e-12);
        assert_eq!(within.refactor(&s).unwrap(), AutoBackend::Ldlt);
        let b = row_sums(n, &rows, &vals);
        let (x1, x2) = (exact.solve(&b).unwrap(), within.solve(&b).unwrap());
        let a_norm = abs_row_sums(n, &rows, &vals)
            .into_iter()
            .fold(0.0, f64::max);
        let omega = solve_backward_error(n, U64);
        let symmetrized = omega + vals[1].abs() * skew / a_norm;
        let (e1, e2) = (
            backward_error(&rows, &cols, &vals, &x1, &b),
            backward_error(&rows, &cols, &vals, &x2, &b),
        );
        assert!(e1 <= omega, "exact symmetry: backward error {e1:e}");
        assert!(
            e2 <= symmetrized,
            "within the tolerance: backward error {e2:e}"
        );
    }

    #[test]
    fn unchanged_values_keep_the_factor() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let flag = Arc::new(AtomicBool::new(false));
        let s =
            AutoSettings::default().with_klu(KluSettings::default().with_interrupt(flag.clone()));
        let sym = layout(2, &[0, 1], &[0, 1]);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&[2.0, 4.0]);
        f.refactor(&s).unwrap();
        flag.store(true, Ordering::Relaxed);
        f.assemble(&[2.0, 4.0]);
        assert_eq!(f.refactor(&s).unwrap(), AutoBackend::Klu);
        assert!(close(&f.solve(&[2.0, 4.0]).unwrap(), &[1.0, 1.0]));
        f.assemble(&[2.0, 5.0]);
        assert!(f.refactor(&s).is_err());
    }

    #[test]
    fn complex_symmetric_dense_work_takes_ldlt() {
        let (rows, cols) = grid(130);
        let n = 130 * 130;
        let c = |re, im| Complex::new(re, im);
        let vals: Vec<Complex<f64>> = (rows.iter().zip(&cols))
            .map(|(&r, &k)| if r == k { c(4.0, 0.5) } else { c(-1.0, 0.1) })
            .collect();
        let layout = CscLayout::new(n, &rows, &cols).unwrap();
        let mut ref_vals = vec![c(0.0, 0.0); layout.nnz()];
        layout.scatter(&vals, &mut ref_vals);
        let sym = AutoSymbolic::analyze(layout, &ref_vals, &AutoSettings::default()).unwrap();
        assert_eq!(sym.is_supernodal(), Some(true));
        let mut f = sym.factorizer::<Complex<f64>>();
        f.values_mut().copy_from_slice(&ref_vals);
        assert_eq!(
            f.refactor(&AutoSettings::default()).unwrap(),
            AutoBackend::Ldlt
        );
        let b = row_sums(n, &rows, &vals);
        let x = f.solve(&b).unwrap();
        let omega = backward_error(&rows, &cols, &vals, &x, &b);
        assert!(
            omega <= solve_backward_error(n, UC64),
            "backward error {omega:e}"
        );
    }
}
