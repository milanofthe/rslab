//! The refinement contract: what "converged" means, what is reported, and that
//! the in-place entry points agree with the allocating ones.

use rslab::prelude::*;
use rslab::{BackwardError, RefinePolicy};

/// Diagonal system whose last equation is scaled down by `1e-12`. Its solution
/// is `(1, 1, 1)` in every row, but an error in the third component is invisible
/// to a normwise measure and obvious to a componentwise one: the row's own
/// scale, not the matrix norm, is what that component is measured against.
fn tiny_row_system() -> (CscMatrix<f64>, Vec<f64>) {
    let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1e-12]).unwrap();
    let b = vec![1.0, 1.0, 1e-12];
    (a, b)
}

#[test]
fn normwise_certifies_what_componentwise_rejects() {
    let (a, b) = tiny_row_system();
    let f = LdltSolver::factor(&a).unwrap();
    // Measure only: no correction step, so this tests the criterion rather than
    // the solver.
    let measure_only = |m| RefinePolicy {
        max_steps: 0,
        target: 4.0 * f64::EPSILON,
        measure: m,
    };

    let mut x = vec![1.0, 1.0, 1.0 + 1e-4];
    let norm = f
        .refine_into(&a, &b, &mut x, &measure_only(BackwardError::Normwise))
        .unwrap();
    let mut x = vec![1.0, 1.0, 1.0 + 1e-4];
    let comp = f
        .refine_into(&a, &b, &mut x, &measure_only(BackwardError::Componentwise))
        .unwrap();

    assert!(
        norm.certified,
        "normwise should not see the error: omega {:.3e}",
        norm.omega
    );
    assert!(
        !comp.certified && comp.omega > 1e-6,
        "componentwise should reject it: omega {:.3e}",
        comp.omega
    );
}

/// The default policy stops at the roundoff floor instead of spending its
/// budget, and says so.
#[test]
fn default_policy_stops_at_the_target() {
    let n = 60;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(4.0);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0);
        }
    }
    let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let f = LdltSolver::factor(&a).unwrap();

    let policy = RefinePolicy {
        max_steps: 8,
        ..Default::default()
    };
    let (x, outcome) = f.solve_refined_with(&a, &b, &policy).unwrap();
    assert!(outcome.certified, "omega {:.3e}", outcome.omega);
    assert!(
        outcome.steps < policy.max_steps,
        "spent the whole budget ({} steps) on a well-conditioned system",
        outcome.steps
    );
    let mut ax = vec![0.0; n];
    a.symv(&x, &mut ax);
    let res = ax
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(res < 1e-10, "residual {res:.3e}");
}

/// In-place refinement and the allocating entry point produce the same bits,
/// and the legacy fixed-step call is the `steps` policy.
#[test]
fn entry_points_agree_bitwise() {
    let n = 40;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(3.0 + (i % 5) as f64);
        if i + 2 < n {
            rows.push(i + 2);
            cols.push(i);
            vals.push(-1.0);
        }
    }
    let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
    let f = LdltSolver::factor(&a).unwrap();

    let policy = RefinePolicy::steps(2);
    let (x_alloc, _) = f.solve_refined_with(&a, &b, &policy).unwrap();
    let mut x_into = f.solve(&b).unwrap();
    f.refine_into(&a, &b, &mut x_into, &policy).unwrap();
    let x_legacy = f.solve_refined(&a, &b, 2).unwrap();

    for i in 0..n {
        assert_eq!(x_alloc[i].to_bits(), x_into[i].to_bits(), "row {i}");
        assert_eq!(x_alloc[i].to_bits(), x_legacy[i].to_bits(), "row {i}");
    }
}

/// The same contract on the unsymmetric and circuit paths.
#[test]
fn lu_and_klu_report_the_same_way() {
    let n = 30;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(5.0);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0);
            rows.push(i);
            cols.push(i + 1);
            vals.push(-2.0);
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let (_, out) = lu
        .solve_refined_with(&a, &b, &RefinePolicy::default())
        .unwrap();
    assert!(out.certified, "lu omega {:.3e}", out.omega);

    let klu = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let (_, out) = klu
        .solve_refined_with(&a, &b, &RefinePolicy::default())
        .unwrap();
    assert!(out.certified, "klu omega {:.3e}", out.omega);
}
