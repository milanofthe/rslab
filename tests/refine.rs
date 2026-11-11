//! The refinement contract: what "converged" means, what is reported, and that
//! the in-place entry points agree with the allocating ones.

use num_complex::Complex64 as Z;
use rslab::prelude::*;
use rslab::{
    BackwardError, MixedPrecision, PivotFloor, RefineOperator, RefinePolicy, RefineSystem,
};

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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    // Measure only: no correction step, so this tests the criterion rather than
    // the solver.
    let measure_only = |m| RefinePolicy {
        max_steps: 0,
        target: 4.0 * f64::EPSILON,
        measure: m,
        ..Default::default()
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();

    let policy = RefinePolicy {
        max_steps: 8,
        ..Default::default()
    };
    let (x, outcome) = f.solve_refined(&a, &b, &policy).unwrap();
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();

    let policy = RefinePolicy::steps(2);
    let (x_alloc, _) = f.solve_refined(&a, &b, &policy).unwrap();
    let mut x_into = f.solve(&b).unwrap();
    f.refine_into(&a, &b, &mut x_into, &policy).unwrap();

    for i in 0..n {
        assert_eq!(x_alloc[i].to_bits(), x_into[i].to_bits(), "row {i}");
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
    let (_, out) = lu.solve_refined(&a, &b, &RefinePolicy::default()).unwrap();
    assert!(out.certified, "lu omega {:.3e}", out.omega);

    let klu = KluSolver::factor(&a, &KluSettings::default()).unwrap();
    let (_, out) = klu.solve_refined(&a, &b, &RefinePolicy::default()).unwrap();
    assert!(out.certified, "klu omega {:.3e}", out.omega);
}

// ---------------------------------------------------------------------------
// Static pivoting: the factor is of `A + E`, `E` its reported pivot floors.
// ---------------------------------------------------------------------------

/// The floor every test below factors with: on the equilibrated matrix
/// (entries of order one) it lifts the pivots the repeated constraints
/// leave at roundoff, and is large enough that `E` dominates the error of
/// a refinement against `A`.
const FLOOR: f64 = 1e-2;

/// `|b - (A + E) x| / |b|` below which a system counts as solved to roundoff.
const ROUNDOFF: f64 = 1e-12;

/// A saddle point `[K B; B^T 0]` whose constraints repeat: every third
/// column of `B` copies the one before it, so `A` is singular and static
/// pivoting has to lift pivots. `K` is a complex grid operator on `m x m`
/// nodes, symmetric when `symmetric` (then only the lower triangle is
/// returned, the `LdltSolver` input). Triplets `(n, rows, cols, values)`.
fn floored_saddle(symmetric: bool) -> (usize, Vec<usize>, Vec<usize>, Vec<Z>) {
    let (m, n2) = (10usize, 24usize);
    let n1 = m * m;
    let (left, right) = if symmetric {
        (Z::new(-1.0, 0.05), Z::new(-1.0, 0.05))
    } else {
        (Z::new(-1.2, 0.1), Z::new(-0.8, 0.0))
    };
    let mut t: Vec<(usize, usize, Z)> = Vec::new();
    for j in 0..n1 {
        let (x, y) = (j % m, j / m);
        t.push((j, j, Z::new(4.0, 0.3)));
        if x > 0 {
            t.push((j - 1, j, left));
        }
        if x + 1 < m {
            t.push((j + 1, j, right));
        }
        if y > 0 {
            t.push((j - m, j, left));
        }
        if y + 1 < m {
            t.push((j + m, j, right));
        }
    }
    for c in 0..n2 {
        let i = if c % 3 == 2 { 4 * (c - 1) } else { 4 * c };
        let (j, v) = (n1 + c, Z::new(1.0, 0.0));
        t.extend([(i, j, v), (i + 1, j, -v), (j, i, v), (j, i + 1, -v)]);
    }
    if symmetric {
        t.retain(|&(r, c, _)| r >= c);
    }
    let n = n1 + n2;
    let (rows, cols, vals) = (
        t.iter().map(|e| e.0).collect(),
        t.iter().map(|e| e.1).collect(),
        t.iter().map(|e| e.2).collect(),
    );
    (n, rows, cols, vals)
}

fn rhs(n: usize) -> Vec<Z> {
    (0..n)
        .map(|i| Z::new(((i * 31) % 17) as f64 - 8.0, (i % 5) as f64))
        .collect()
}

fn norm(v: &[Z]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// `|b - (A + E) x| / |b|`, `E` the floors.
fn floored_residual<A: RefineOperator<Z>>(
    a: &A,
    floors: &[PivotFloor<Z>],
    x: &[Z],
    b: &[Z],
) -> f64 {
    let mut r = vec![Z::default(); b.len()];
    a.apply(x, &mut r);
    for f in floors {
        r[f.row] += f.delta * x[f.col];
    }
    let r: Vec<Z> = b.iter().zip(&r).map(|(p, q)| p - q).collect();
    norm(&r) / norm(b)
}

/// The floors are the matrix the factor is of: a plain solve solves `A + E`
/// to roundoff, with and without the row matching (whose permutation and
/// scaling the reported entries undo), where it leaves a large residual
/// against `A`.
#[test]
fn lu_perturbations_are_the_factored_matrix() {
    let (n, r, c, v) = floored_saddle(false);
    let a = GeneralCsc::<Z>::from_triplets(n, &r, &c, &v).unwrap();
    let b = rhs(n);
    for matching in [false, true] {
        let opts = SolverSettings::preconditioner(FLOOR)
            .with_matching(matching)
            .with_threads(1);
        let s = LuSolver::factor(&a, &opts).unwrap();
        let floors = s.perturbations();
        assert!(!floors.is_empty(), "matching {matching}: no floor");
        assert_eq!(floors.len(), s.n_perturbed());
        let mut rows: Vec<usize> = floors.iter().map(|f| f.row).collect();
        let mut cols: Vec<usize> = floors.iter().map(|f| f.col).collect();
        rows.sort_unstable();
        rows.dedup();
        cols.sort_unstable();
        cols.dedup();
        assert_eq!((rows.len(), cols.len()), (floors.len(), floors.len()));
        let x = s.solve(&b).unwrap();
        let (be, be_a) = (
            floored_residual(&a, floors, &x, &b),
            floored_residual(&a, &[], &x, &b),
        );
        eprintln!(
            "lu matching {matching}: {} floors, |b - A x| {be_a:.1e}, |b - (A + E) x| {be:.1e}",
            floors.len()
        );
        assert!(
            be < ROUNDOFF,
            "matching {matching}: |b - (A + E) x|/|b| = {be:.1e}"
        );
        assert!(
            be_a > 1e3 * be,
            "matching {matching}: E does not show against A"
        );
    }
}

/// The symmetric path the same way: the floors sit on the diagonal.
#[test]
fn ldlt_perturbations_are_the_factored_matrix() {
    let (n, r, c, v) = floored_saddle(true);
    let a = CscMatrix::<Z>::from_triplets(n, &r, &c, &v).unwrap();
    let b = rhs(n);
    let s = LdltSolver::factor(&a, &SolverSettings::preconditioner(FLOOR)).unwrap();
    let floors = s.perturbations();
    assert!(!floors.is_empty(), "no floor");
    assert_eq!(floors.len(), s.n_perturbed());
    assert!(floors.iter().all(|f| f.row == f.col));
    let x = s.solve(&b).unwrap();
    let be = floored_residual(&a, floors, &x, &b);
    eprintln!("ldlt: {} floors, |b - (A + E) x| {be:.1e}", floors.len());
    assert!(be < ROUNDOFF, "|b - (A + E) x|/|b| = {be:.1e}");
}

/// Refinement against `A` cannot converge on a singular `A` (the correction
/// `(A + E)^-1 E` has eigenvalue one on its null space) and keeps the first
/// iterate; against `A + E` it certifies in a step or two, on both paths.
#[test]
fn refinement_against_the_factored_matrix_converges() {
    let policy = RefinePolicy {
        max_steps: 8,
        ..Default::default()
    };
    let factored = policy.with_system(RefineSystem::Factored);
    let (n, r, c, v) = floored_saddle(false);
    let a = GeneralCsc::<Z>::from_triplets(n, &r, &c, &v).unwrap();
    let b = rhs(n);
    let lu = LuSolver::factor(&a, &SolverSettings::preconditioner(FLOOR)).unwrap();
    let (_, given) = lu.solve_refined(&a, &b, &policy).unwrap();
    let (x, own) = lu.solve_refined(&a, &b, &factored).unwrap();
    eprintln!("lu: against A {given:?}, against A + E {own:?}");
    assert!(!given.certified, "A is singular: {given:?}");
    assert!(own.certified && own.steps <= 2, "{own:?}");
    assert!(floored_residual(&a, lu.perturbations(), &x, &b) < ROUNDOFF);

    let (n, r, c, v) = floored_saddle(true);
    let a = CscMatrix::<Z>::from_triplets(n, &r, &c, &v).unwrap();
    let b = rhs(n);
    let ldlt = LdltSolver::factor(&a, &SolverSettings::preconditioner(FLOOR)).unwrap();
    let (_, given) = ldlt.solve_refined(&a, &b, &policy).unwrap();
    let (_, own) = ldlt.solve_refined(&a, &b, &factored).unwrap();
    eprintln!("ldlt: against A {given:?}, against A + E {own:?}");
    assert!(!given.certified, "A is singular: {given:?}");
    assert!(own.certified && own.steps <= 2, "{own:?}");
}

/// The mixed-precision use: a single-precision factor with floors, refined
/// in double precision against `A + E`, reaches double accuracy for the
/// perturbed system (with floors that were not the factor's it would
/// stagnate at their error); its floors are the low factor's, promoted.
/// Against `A` it does not converge.
#[test]
fn mixed_precision_refines_against_the_factored_matrix() {
    let (n, r, c, v) = floored_saddle(false);
    let a = GeneralCsc::<Z>::from_triplets(n, &r, &c, &v).unwrap();
    let b = rhs(n);
    let opts = SolverSettings::preconditioner(FLOOR);
    let sym = LuSymbolic::analyze(&a, &opts).unwrap();
    let m = MixedPrecision::new(sym.factor(&a.demoted(), &opts).unwrap());
    let floors = Factorization::<Z>::perturbations(&m);
    assert_eq!(floors.len(), m.inner().n_perturbed());
    assert!(!floors.is_empty());
    let policy = RefinePolicy {
        max_steps: 20,
        ..Default::default()
    };
    let (_, given) = m.solve_refined(&a, &b, &policy).unwrap();
    let factored = policy.with_system(RefineSystem::Factored);
    let (x, own) = m.solve_refined(&a, &b, &factored).unwrap();
    eprintln!(
        "mixed: against A {given:?}, against A + E {own:?}, |b - (A + E) x| {:.1e}",
        floored_residual(&a, &floors, &x, &b)
    );
    assert!(!given.certified, "A is singular: {given:?}");
    assert!(own.certified, "{own:?}");
}
