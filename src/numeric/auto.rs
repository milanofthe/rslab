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
//! factor in place, and a value set bitwise equal to the last factored one
//! keeps its factor without any numeric work. The solves run in work the
//! solver keeps, so a warm loop allocates nothing.

use std::sync::OnceLock;

use crate::error::RslabError;
use crate::numeric::direct::SolveWork;
use crate::numeric::klu::{KluSettings, KluSolver, KluSymbolic};
use crate::numeric::ldlt::{LdltSolver, LdltSymbolic};
use crate::numeric::lu::{LuSolver, LuSymbolic};
use crate::numeric::settings::SolverSettings;
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
}

impl Default for AutoSettings {
    fn default() -> Self {
        Self {
            klu: KluSettings::default(),
            supernodal: SolverSettings::default(),
            supernodal_flops_per_entry: 25.0,
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
}

/// The path a factorization of the automatic path took.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AutoBackend {
    Klu,
    Ldlt,
    Lu,
}

/// The LDL^T side structure of a structurally symmetric pattern: the
/// transposed slot of each slot for the value-symmetry test, and the lower
/// triangle with its analysis.
struct LdltCandidate {
    tpair: Vec<usize>,
    /// Lower-triangle entry -> the full slot it reads its value from.
    lower_from: Vec<usize>,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    sym: LdltSymbolic,
}

impl LdltCandidate {
    /// `None` if the pattern is structurally unsymmetric or its lower
    /// triangle fails to analyze.
    fn build(layout: &CscLayout, opts: &SolverSettings) -> Option<Self> {
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
        let sym = LdltSymbolic::analyze(&skeleton, opts).ok()?;
        Some(Self {
            tpair,
            lower_from,
            col_ptr,
            row_idx,
            sym,
        })
    }

    /// Whether the slot values are symmetric (`a_ij == a_ji`; the transpose
    /// for complex fields, not the adjoint).
    fn symmetric<T: Scalar>(&self, vals: &[T]) -> bool {
        (vals.iter().zip(&self.tpair)).all(|(&v, &t)| v == vals[t])
    }

    /// The lower triangle of the slot values `vals`, into `lower`.
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
}

/// The analysis of the path the pattern's work selected. Boxed: the analyses
/// are large and of very different sizes.
enum Route {
    Klu(Box<KluSymbolic>),
    Supernodal {
        ldlt: Option<Box<LdltCandidate>>,
        /// Made when a value set first needs it (eagerly by
        /// [`AutoSymbolic::analyze`]).
        lu: OnceLock<Box<LuSymbolic>>,
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

    /// Whether the pattern routed to the supernodal paths; `None` before its
    /// analysis.
    pub fn is_supernodal(&self) -> Option<bool> {
        self.route
            .get()
            .map(|r| matches!(r, Route::Supernodal { .. }))
    }

    /// A solver over this pattern, for one solve loop or one sweep worker.
    pub fn factorizer<T: Scalar>(&self) -> AutoSolver<'_, T> {
        AutoSolver {
            sym: self,
            csc: self.layout.matrix(vec![T::zero(); self.layout.nnz()]),
            lower: None,
            klu: None,
            ldlt: None,
            lu: None,
            active: None,
            prev: Vec::new(),
            work: SolveWork::new(),
        }
    }

    /// The analysis of `a`'s pattern in the path its work selects.
    fn route<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        settings: &AutoSettings,
        eager: bool,
    ) -> Result<Route, RslabError> {
        let klu = KluSymbolic::analyze(a, &settings.klu)?;
        let per_entry = klu.estimated_flops() as f64 / klu.estimated_factor_nnz().max(1) as f64;
        let supernodal = per_entry >= settings.supernodal_flops_per_entry;
        crate::logging::info(&format!(
            "auto: n={} nnz={} flops/entry={per_entry:.1} -> {}",
            a.n,
            a.nnz(),
            if supernodal { "supernodal" } else { "klu" }
        ));
        if !supernodal {
            return Ok(Route::Klu(Box::new(klu)));
        }
        let lu = OnceLock::new();
        if eager {
            let _ = lu.set(Box::new(LuSymbolic::analyze(a, &settings.supernodal)?));
        }
        Ok(Route::Supernodal {
            ldlt: LdltCandidate::build(&self.layout, &settings.supernodal).map(Box::new),
            lu,
        })
    }
}

/// Factor into `slot` in place when it holds a factor of the same analysis,
/// fresh otherwise.
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

/// The numeric state over an [`AutoSymbolic`] for one solve loop: the
/// values to factor (one per slot of the layout), each path's factor, kept
/// while another path serves a value set so that its next factorization is
/// again in place, and the solves' work.
pub struct AutoSolver<'s, T> {
    sym: &'s AutoSymbolic,
    csc: GeneralCsc<T>,
    lower: Option<CscMatrix<T>>,
    klu: Option<KluSolver<T>>,
    ldlt: Option<LdltSolver<T>>,
    lu: Option<LuSolver<T>>,
    /// The path holding the factor of the last successful factorization.
    active: Option<AutoBackend>,
    /// The values of the last successful factorization; empty without one.
    prev: Vec<T>,
    work: SolveWork<T>,
}

impl<T: Scalar> AutoSolver<'_, T> {
    /// The values the next [`factor`](Self::factor) factors, one per slot.
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.csc.values
    }

    /// The matrix the next (or the last) [`factor`](Self::factor) factors.
    pub fn matrix(&self) -> &GeneralCsc<T> {
        &self.csc
    }

    /// Load the entry values `entries` (the layout's entry order,
    /// duplicates summed), see [`CscLayout::scatter`].
    pub fn assemble(&mut self, entries: &[T]) {
        self.sym.layout.scatter(entries, &mut self.csc.values);
    }

    /// Load scaled runs of entry values, see [`CscLayout::scatter_scaled`].
    pub fn assemble_scaled(&mut self, parts: &[(&[T], T)], rest: T) {
        (self.sym.layout).scatter_scaled(parts, rest, &mut self.csc.values);
    }

    /// The path of the last successful factorization; `None` without one.
    pub fn backend(&self) -> Option<AutoBackend> {
        self.active
    }

    /// Factor the loaded values. A value set bitwise equal to the last
    /// factored one keeps its factor; after an error no factor is held.
    pub fn factor(&mut self, settings: &AutoSettings) -> Result<AutoBackend, RslabError> {
        if let Some(b) = self.active {
            // NaN never compares equal, so a poisoned value set re-factors.
            if self.prev == self.csc.values {
                return Ok(b);
            }
        }
        self.active = None;
        self.prev.clear();
        let b = self.factor_values(settings)?;
        self.active = Some(b);
        self.prev.extend_from_slice(&self.csc.values);
        Ok(b)
    }

    fn factor_values(&mut self, settings: &AutoSettings) -> Result<AutoBackend, RslabError> {
        let sym = self.sym;
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
                // A replay that meets a vanished pivot factors afresh.
                if let Some(s) = &mut self.klu {
                    if s.refactor(csc).is_ok() {
                        return Ok(AutoBackend::Klu);
                    }
                }
                self.klu = None;
                self.klu = Some(ks.factor(csc, &settings.klu)?);
                Ok(AutoBackend::Klu)
            }
            Route::Supernodal { ldlt, lu } => {
                let opts = &settings.supernodal;
                if let Some(cand) = ldlt {
                    if cand.symmetric(&csc.values) {
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
                        let s = LuSymbolic::analyze(csc, opts)?;
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

    /// Solve `A x = b` on the last successful factorization, in the work the
    /// solver keeps: no allocation once it has served a solve.
    pub fn solve_into(&mut self, b: &[T], x: &mut [T]) -> Result<(), RslabError> {
        self.solve_with(b, x, false)
    }

    /// Solve `A^T x = b` (the plain transpose) on the last successful
    /// factorization, see [`solve_into`](Self::solve_into).
    pub fn solve_transpose_into(&mut self, b: &[T], x: &mut [T]) -> Result<(), RslabError> {
        self.solve_with(b, x, true)
    }

    /// [`solve_into`](Self::solve_into) into a fresh vector.
    pub fn solve(&mut self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let mut x = vec![T::zero(); b.len()];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    fn solve_with(&mut self, b: &[T], x: &mut [T], transpose: bool) -> Result<(), RslabError> {
        let w = &mut self.work;
        macro_rules! go {
            ($s:expr) => {
                match &$s {
                    Some(s) if transpose => s.solve_transpose_into(b, x, w),
                    Some(s) => s.solve_into(b, x, w),
                    None => Err(not_factored()),
                }
            };
        }
        match self.active {
            Some(AutoBackend::Klu) => go!(self.klu),
            Some(AutoBackend::Ldlt) => go!(self.ldlt),
            Some(AutoBackend::Lu) => go!(self.lu),
            None => Err(not_factored()),
        }
    }
}

fn not_factored() -> RslabError {
    RslabError::InvalidInput("auto: no successful factorization to solve with".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn layout(n: usize, rows: &[usize], cols: &[usize]) -> AutoSymbolic {
        AutoSymbolic::new(CscLayout::new(n, rows, cols).unwrap()).unwrap()
    }

    fn close(x: &[f64], want: &[f64]) -> bool {
        x.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12)
    }

    #[test]
    fn sums_duplicates_and_solves_both_ways() {
        // A = [[2, 1], [0, 3]] with (0, 0) given as 1.5 + 0.5.
        let sym = layout(2, &[0, 0, 0, 1], &[0, 0, 1, 1]);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&[1.5, 0.5, 1.0, 3.0]);
        assert_eq!(
            f.factor(&AutoSettings::default()).unwrap(),
            AutoBackend::Klu
        );
        assert!(close(&f.solve(&[5.0, 6.0]).unwrap(), &[1.5, 2.0]));
        let mut xt = [0.0; 2];
        f.solve_transpose_into(&[2.0, 7.0], &mut xt).unwrap();
        assert!(close(&xt, &[1.0, 2.0]));
    }

    #[test]
    fn tracks_new_values_and_reports_singular() {
        let sym = layout(2, &[0, 0, 1, 1], &[0, 1, 0, 1]);
        let s = AutoSettings::default();
        let mut f = sym.factorizer::<f64>();
        f.assemble(&[2.0, 1.0, 0.0, 3.0]);
        f.factor(&s).unwrap();
        assert!(close(&f.solve(&[5.0, 6.0]).unwrap(), &[1.5, 2.0]));
        f.assemble(&[4.0, 2.0, 0.0, 6.0]);
        f.factor(&s).unwrap();
        assert!(close(&f.solve(&[10.0, 12.0]).unwrap(), &[1.5, 2.0]));
        f.assemble(&[1.0, 2.0, 2.0, 4.0]);
        assert!(f.factor(&s).is_err());
        assert_eq!(f.backend(), None);
        assert!(f.solve(&[1.0, 1.0]).is_err());
        f.assemble(&[1.0, 2.0, 2.0, 5.0]);
        f.factor(&s).unwrap();
    }

    #[test]
    fn structurally_singular_pattern_is_rejected() {
        let l = CscLayout::new(2, &[0, 1], &[0, 0]).unwrap();
        assert!(matches!(
            AutoSymbolic::new(l),
            Err(RslabError::StructurallySingular)
        ));
    }

    /// The `(rows, cols)` of an `m x m` grid Laplacian's pattern.
    fn grid(m: usize) -> (Vec<usize>, Vec<usize>) {
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for i in 0..m * m {
            rows.push(i);
            cols.push(i);
            for j in [i + 1, i + m] {
                if j < m * m && (j != i + 1 || j % m != 0) {
                    rows.extend([i, j]);
                    cols.extend([j, i]);
                }
            }
        }
        (rows, cols)
    }

    fn laplacian(rows: &[usize], cols: &[usize]) -> Vec<f64> {
        (rows.iter().zip(cols))
            .map(|(&r, &c)| if r == c { 4.0 } else { -1.0 })
            .collect()
    }

    /// `A 1`, the row sums of the entries.
    fn row_sums<T: Scalar>(n: usize, rows: &[usize], vals: &[T]) -> Vec<T> {
        let mut b = vec![T::zero(); n];
        for (&r, &v) in rows.iter().zip(vals) {
            b[r] = b[r] + v;
        }
        b
    }

    #[test]
    fn sparse_work_takes_klu_at_any_size() {
        // A 20k chain: large, but no dense work.
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
            f.factor(&AutoSettings::default()).unwrap(),
            AutoBackend::Klu
        );
        assert_eq!(sym.is_supernodal(), Some(false));
    }

    #[test]
    fn dense_work_takes_ldlt_then_lu_then_ldlt_in_place() {
        // A 130 x 130 grid (~70 flops per factor entry): supernodal.
        let (rows, cols) = grid(130);
        let n = 130 * 130;
        let s = AutoSettings::default();
        let sym = layout(n, &rows, &cols);
        let vals = laplacian(&rows, &cols);
        let b = row_sums(n, &rows, &vals);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&vals);
        assert_eq!(f.factor(&s).unwrap(), AutoBackend::Ldlt);
        let x = f.solve(&b).unwrap();
        assert!(close(&x, &vec![1.0; n]));

        // One off-diagonal broken from its mirror: the LU takes it.
        let mut asym = vals.clone();
        asym[1] = -0.5;
        f.assemble(&asym);
        assert_eq!(f.factor(&s).unwrap(), AutoBackend::Lu);
        let x2 = f.solve(&row_sums(n, &rows, &asym)).unwrap();
        assert!(close(&x2, &vec![1.0; n]));

        // Symmetric again: LDL^T in place, the same bits as at first.
        f.assemble(&vals);
        assert_eq!(f.factor(&s).unwrap(), AutoBackend::Ldlt);
        let xr = f.solve(&b).unwrap();
        assert!(x.iter().zip(&xr).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn unchanged_values_keep_the_factor() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        // The factor keeps the interrupt flag: once raised, any numeric work
        // fails, so only the identity path can succeed.
        let flag = Arc::new(AtomicBool::new(false));
        let s =
            AutoSettings::default().with_klu(KluSettings::default().with_interrupt(flag.clone()));
        let sym = layout(2, &[0, 1], &[0, 1]);
        let mut f = sym.factorizer::<f64>();
        f.assemble(&[2.0, 4.0]);
        f.factor(&s).unwrap();
        flag.store(true, Ordering::Relaxed);
        f.assemble(&[2.0, 4.0]);
        assert_eq!(f.factor(&s).unwrap(), AutoBackend::Klu);
        assert!(close(&f.solve(&[2.0, 4.0]).unwrap(), &[1.0, 1.0]));
        f.assemble(&[2.0, 5.0]);
        assert!(f.factor(&s).is_err());
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
            f.factor(&AutoSettings::default()).unwrap(),
            AutoBackend::Ldlt
        );
        let x = f.solve(&row_sums(n, &rows, &vals)).unwrap();
        assert!(x.iter().all(|v| (v - c(1.0, 0.0)).norm() < 1e-10));
    }
}
