//! Every factor path fills its `Diagnostics` (stages, decisions, numeric
//! outcome, solve accumulators), a setting the chosen path does not read is
//! reported instead of silently ignored, and the log sink receives the
//! factorization records.

use std::sync::{Arc, Mutex};

use rslab::logging::{self, LogLevel, LogSink};
use rslab::{
    CscMatrix, GeneralCsc, KluSettings, KluSolver, LdltSolver, LuSolver, OrderingMethod,
    ScalingStrategy, SolverSettings,
};

/// 2D 5-point grid Laplacian (k x k), SPD-shifted; lower triangle for LDLT.
fn grid_lower(k: usize) -> CscMatrix<f64> {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(4.5);
            for (di, dj) in [(1usize, 0usize), (0, 1)] {
                let (ii, jj) = (i + di, j + dj);
                if ii < k && jj < k {
                    let q = ii * k + jj;
                    rows.push(q.max(p));
                    cols.push(q.min(p));
                    vals.push(-1.0);
                }
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

/// The same grid as a full unsymmetric matrix (a small skew on the couplings).
fn grid_full(k: usize) -> GeneralCsc<f64> {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(4.5);
            for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (ii, jj) = (i as i64 + di, j as i64 + dj);
                if ii >= 0 && jj >= 0 && (ii as usize) < k && (jj as usize) < k {
                    rows.push(p);
                    cols.push(ii as usize * k + jj as usize);
                    vals.push(if di + dj > 0 { -1.0 } else { -0.9 });
                }
            }
        }
    }
    GeneralCsc::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn ldlt_diagnostics_are_filled() {
    let a = grid_lower(30);
    let f = LdltSolver::factor_with(&a, &SolverSettings::exact()).unwrap();
    let b = vec![1.0; a.n];
    let _ = f.solve(&b).unwrap();
    let _ = f.solve_many(&vec![1.0; 2 * a.n], 2).unwrap();
    let d = f.diagnostics();
    assert_eq!(d.n, a.n);
    assert_eq!(d.nnz_a as usize, a.row_idx.len());
    assert!(d.factor_nnz >= d.nnz_a && d.fill_ratio() >= 1.0);
    for stage in ["analyze", "scale", "factor"] {
        assert!(d.stage_ms(stage).is_some(), "stage {stage} missing: {d}");
    }
    assert_eq!(d.decisions.method, "LeftLooking");
    assert!(!d.decisions.ordering_used.is_empty());
    assert!(d.decisions.n_supernodes > 0 && d.decisions.max_front > 0);
    assert_eq!(d.numeric.inertia, Some((a.n, 0, 0)), "SPD grid");
    assert_eq!(d.numeric.two_by_two, Some(0));
    assert_eq!(d.solves.calls, 2);
    assert_eq!(d.solves.rhs, 3);
    assert!(d.warnings.is_empty(), "{:?}", d.warnings);
    assert!(d.summary().contains("nnz(L)="));
}

#[test]
fn lu_diagnostics_are_filled_on_both_entry_points() {
    let a = grid_full(30);
    let opts = SolverSettings::exact().with_ordering(OrderingMethod::Amd);
    let direct = LuSolver::factor(&a, &opts).unwrap();
    let d = direct.diagnostics();
    assert_eq!(d.n, a.n);
    assert!(d.stage_ms("analyze").is_some() && d.stage_ms("factor").is_some());
    assert_eq!(d.decisions.ordering_requested, "Amd");
    assert_eq!(d.decisions.ordering_used, "Amd");
    assert_eq!(d.numeric.two_by_two, None);
    let _ = direct.solve(&vec![1.0; a.n]).unwrap();
    assert_eq!(direct.diagnostics().solves.calls, 1);
}

#[test]
fn klu_diagnostics_are_filled() {
    let a = grid_full(20);
    let f = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let _ = f.solve(&vec![1.0; a.n]).unwrap();
    let d = f.diagnostics();
    assert_eq!(d.n, a.n);
    assert_eq!(d.decisions.method, "Klu");
    assert!(d.stage_ms("klu-factor").is_some());
    assert_eq!(d.solves.calls, 1);
}

#[test]
fn settings_ignored_by_a_path_are_reported() {
    // pivot_u belongs to the left-looking LU: the LDL^T path reports it.
    let ldlt = SolverSettings::exact().with_pivot_u(0.5);
    let w = ldlt.ignored_on(rslab::FactorPath::Ldlt);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("pivot_u"));
    assert!(ldlt.ignored_on(rslab::FactorPath::Lu).is_empty());
    // scaling belongs to the symmetric path: the LU path reports it.
    let lu = SolverSettings::exact().with_scaling(ScalingStrategy::Identity);
    let w = lu.ignored_on(rslab::FactorPath::Lu);
    assert_eq!(w.len(), 1, "{w:?}");
    assert!(w[0].contains("scaling"));
    assert!(lu.ignored_on(rslab::FactorPath::Ldlt).is_empty());
    // defaults are honoured everywhere
    assert!(SolverSettings::default()
        .ignored_on(rslab::FactorPath::Lu)
        .is_empty());
    // and the warning rides on the factorization's diagnostics
    let a = grid_lower(12);
    let f = LdltSolver::factor_with(&a, &ldlt).unwrap();
    assert_eq!(f.diagnostics().warnings.len(), 1);
}

struct Capture(Arc<Mutex<Vec<(LogLevel, String)>>>);
impl LogSink for Capture {
    fn emit(&self, level: LogLevel, msg: &str) {
        self.0.lock().unwrap().push((level, msg.to_string()));
    }
}

#[test]
fn factorizations_log_through_the_sink() {
    let got = Arc::new(Mutex::new(Vec::new()));
    logging::set_sink(Box::new(Capture(got.clone())));
    logging::set_level(LogLevel::Info);
    let a = grid_lower(12);
    let f = LdltSolver::factor_with(&a, &SolverSettings::exact().with_pivot_u(0.5)).unwrap();
    let _ = f.solve(&vec![1.0; a.n]).unwrap();
    let records = got.lock().unwrap().clone();
    logging::reset_sink();
    logging::set_level(LogLevel::Warning);
    let text: Vec<&str> = records.iter().map(|(_, m)| m.as_str()).collect();
    assert!(text.iter().any(|m| m.contains("ldlt analyze:")), "{text:?}");
    assert!(text.iter().any(|m| m.contains("ldlt factor:")), "{text:?}");
    assert!(
        records
            .iter()
            .any(|(l, m)| *l == LogLevel::Warning && m.contains("pivot_u")),
        "{text:?}"
    );
    // solves log at Debug only
    assert!(!text.iter().any(|m| m.contains("ldlt solve:")), "{text:?}");
}
