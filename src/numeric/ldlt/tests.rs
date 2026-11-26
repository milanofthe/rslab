use super::*;
use crate::numeric::settings::SolverSettings;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::OrderingMethod;
use crate::test_constants::{solve_backward_error, U64, UC64};
use num_complex::Complex;

#[test]
fn deep_chain_tree_does_not_overflow_stack() {
    let n = 20_000usize;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(4.0f64);
        if i + 1 < n {
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0);
        }
    }
    let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
    for threads in [0, 1] {
        let s = SolverSettings::default()
            .with_nemin(1)
            .with_threads(threads);
        let f = LdltSolver::factor(&a, &s).expect("deep chain factors without overflow");
        assert_eq!(f.n(), n);
    }
}

#[test]
fn rcm_and_autorace_orderings_factor_and_solve() {
    let m = 14;
    let n = m * m;
    let idx = |a: usize, b: usize| a * m + b;
    let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            r.push(p);
            cc.push(p);
            v.push(6.0_f64);
            if b + 1 < m {
                r.push(idx(a, b + 1));
                cc.push(p);
                v.push(-1.0);
            }
            if a + 1 < m {
                r.push(idx(a + 1, b));
                cc.push(p);
                v.push(-1.0);
            }
        }
    }
    let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    for ord in [OrderingMethod::Rcm, OrderingMethod::Auto] {
        let opts = SolverSettings::default().with_ordering(ord);
        let x = LdltSolver::factor(&a, &opts).unwrap().solve(&b).unwrap();
        assert!(
            backward_error(&a, &x, &b) <= solve_backward_error(a.n, U64),
            "ordering {ord:?} residual {}",
            backward_error(&a, &x, &b)
        );
    }
}

fn two_by_two<T: Scalar>(f: &LdltSolver<T>) -> usize {
    f.diagnostics().numeric.two_by_two.unwrap_or(0)
}

pub(super) fn backward_error<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::zero(); a.n];
    a.symv(x, &mut ax);
    let mut rows = vec![0.0f64; a.n];
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            rows[i] += a.values[k].magnitude();
            if i != j {
                rows[j] += a.values[k].magnitude();
            }
        }
    }
    let inf = |v: &[T]| v.iter().map(|e| e.magnitude()).fold(0.0, f64::max);
    let r = (0..a.n)
        .map(|i| (ax[i] - b[i]).magnitude())
        .fold(0.0, f64::max);
    r / (rows.iter().fold(0.0f64, |m, &v| m.max(v)) * inf(x) + inf(b))
}

fn tridiag_spd_f64(n: usize) -> CscMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(4.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn f64_sparse_tridiag_residual() {
    let a = tridiag_spd_f64(20);
    let b: Vec<f64> = (0..20).map(|i| (i as f64) - 9.5).collect();
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(backward_error(&a, &x, &b) <= solve_backward_error(a.n, U64));
}

fn grid2d_lower<T: Scalar>(m: usize, diag: T, off: T) -> CscMatrix<T> {
    let n = m * m;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |r: usize, c: usize| r * m + c;
    let mut push = |i: usize, j: usize, v: T| {
        let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
        rows.push(hi);
        cols.push(lo);
        vals.push(v);
    };
    for r in 0..m {
        for c in 0..m {
            let p = idx(r, c);
            push(p, p, diag);
            if c + 1 < m {
                push(p, idx(r, c + 1), off);
            }
            if r + 1 < m {
                push(p, idx(r + 1, c), off);
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn indefinite_2x2_inertia() {
    let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 0], &[0.0, 1.0]).unwrap();
    let ll = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    assert!(two_by_two(&ll) > 0, "expected a 2x2 block");
    assert_eq!(
        (
            ll.inertia().positive,
            ll.inertia().negative,
            ll.inertia().zero
        ),
        (1, 1, 0)
    );
    let b = [1.0_f64, -2.0];
    let x = ll.solve(&b).unwrap();
    assert!(
        backward_error(&a, &x, &b) <= solve_backward_error(a.n, U64),
        "2x2 backward error"
    );
}

#[test]
fn indefinite_2d_grid_solves() {
    let a = grid2d_lower::<f64>(10, 0.5, -1.0);
    let n = a.n;
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let ll = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    assert!(
        two_by_two(&ll) > 0,
        "indefinite system should use 2x2 pivots"
    );
    assert!(
        ll.inertia().negative > 0 && ll.inertia().positive + ll.inertia().negative == n,
        "indefinite, nonsingular inertia"
    );
    let xl = ll.solve(&b).unwrap();
    assert!(
        backward_error(&a, &xl, &b) <= solve_backward_error(n, U64),
        "left-looking indefinite residual"
    );
}

#[test]
fn indefinite_complex_symmetric() {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let a = grid2d_lower::<Complex<f64>>(9, c(0.4, 0.3), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
    let ll = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    assert!(
        two_by_two(&ll) > 0,
        "indefinite system should use 2x2 pivots"
    );
    assert_eq!(
        ll.inertia().positive + ll.inertia().negative + ll.inertia().zero,
        n,
        "inertia covers every pivot"
    );
    let xl = ll.solve(&b).unwrap();
    assert!(
        backward_error(&a, &xl, &b) <= solve_backward_error(n, UC64),
        "complex left-looking indefinite residual"
    );
}

#[test]
fn f64_dense_front_blocked_multi_panel() {
    let n = 100;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        for i in j..n {
            rows.push(i);
            cols.push(j);
            vals.push(if i == j {
                n as f64 + 1.0
            } else {
                ((i + 2 * j) % 5) as f64 - 2.0
            });
        }
    }
    let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        backward_error(&a, &x, &b) <= solve_backward_error(a.n, U64),
        "residual {}",
        backward_error(&a, &x, &b)
    );
}

#[test]
fn complex_dense_front_blocked_multi_panel() {
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 90;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        for i in j..n {
            rows.push(i);
            cols.push(j);
            vals.push(if i == j {
                c(n as f64, 1.0)
            } else {
                c(((i + 3 * j) % 5) as f64 - 2.0, 0.2)
            });
        }
    }
    let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b = vec![c(1.0, 0.5); n];
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(backward_error(&a, &x, &b) <= solve_backward_error(a.n, UC64));
}

#[test]
fn f64_sparse_2d_grid_residual() {
    let m = 5;
    let n = m * m;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let idx = |r: usize, c: usize| r * m + c;
    for r in 0..m {
        for c in 0..m {
            let p = idx(r, c);
            rows.push(p);
            cols.push(p);
            vals.push(4.0);
            if c + 1 < m {
                let q = idx(r, c + 1);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(-1.0);
            }
            if r + 1 < m {
                let q = idx(r + 1, c);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(-1.0);
            }
        }
    }
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        backward_error(&a, &x, &b) <= solve_backward_error(a.n, U64),
        "residual {}",
        backward_error(&a, &x, &b)
    );
}

#[test]
fn complex_sparse_tridiag_residual() {
    let c = |re, im| Complex::new(re, im);
    let n = 16;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(c(4.0, 0.5));
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(c(-1.0, 0.1));
        }
    }
    let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 7.5, 1.0 - i as f64)).collect();
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        backward_error(&a, &x, &b) <= solve_backward_error(n, UC64),
        "residual {}",
        backward_error(&a, &x, &b)
    );
}

#[test]
fn complex_sparse_large_grid_parallel() {
    let c = |re, im| Complex::new(re, im);
    let m = 12;
    let n = m * m;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let idx = |r: usize, cc: usize| r * m + cc;
    for r in 0..m {
        for cc in 0..m {
            let p = idx(r, cc);
            rows.push(p);
            cols.push(p);
            vals.push(c(4.0, 0.5));
            if cc + 1 < m {
                let q = idx(r, cc + 1);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(c(-1.0, 0.1));
            }
            if r + 1 < m {
                let q = idx(r + 1, cc);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(c(-1.0, 0.1));
            }
        }
    }
    let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 11) as f64 - 5.0, 1.0)).collect();
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        backward_error(&a, &x, &b) <= solve_backward_error(a.n, UC64),
        "residual {}",
        backward_error(&a, &x, &b)
    );
}

#[test]
fn perturb_rescues_singular_complex() {
    let c = |re, im| Complex::new(re, im);
    let n = 3;
    let rows = vec![0, 2, 1];
    let cols = vec![0, 0, 1];
    let vals = vec![c(2.0, 1.0), c(-1.0, 0.3), c(0.0, 0.0)];
    let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();

    assert!(
        LdltSolver::factor(&a, &SolverSettings::default()).is_err(),
        "exact mode should reject the singular pivot"
    );

    let opts = SolverSettings::preconditioner(1e-8);
    let f = LdltSolver::factor(&a, &opts).unwrap();
    assert!(
        f.n_perturbed() >= 1,
        "expected >=1 perturbation, got {}",
        f.n_perturbed()
    );
    let b = vec![c(1.0, 0.0); n];
    let x = f.solve(&b).unwrap();
    assert!(
        x.iter().all(|v| v.norm().is_finite()),
        "factor must stay finite"
    );
}

#[test]
fn exact_mode_never_perturbs_well_conditioned() {
    let a = {
        let c = |re, im| Complex::new(re, im);
        let n = 16;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            r.push(j);
            cc.push(j);
            v.push(c(4.0, 0.5));
            if j + 1 < n {
                r.push(j + 1);
                cc.push(j);
                v.push(c(-1.0, 0.1));
            }
        }
        CscMatrix::<Complex<f64>>::from_triplets(n, &r, &cc, &v).unwrap()
    };
    let opts = SolverSettings::preconditioner(1e-8);
    let f = LdltSolver::factor(&a, &opts).unwrap();
    assert_eq!(
        f.n_perturbed(),
        0,
        "well-conditioned matrix needs no perturbation"
    );
}

#[test]
fn complex_sparse_2d_grid_residual() {
    let c = |re, im| Complex::new(re, im);
    let m = 5;
    let n = m * m;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let idx = |r: usize, cc: usize| r * m + cc;
    for r in 0..m {
        for cc in 0..m {
            let p = idx(r, cc);
            rows.push(p);
            cols.push(p);
            vals.push(c(4.0, 1.0));
            if cc + 1 < m {
                let q = idx(r, cc + 1);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(c(-1.0, 0.2));
            }
            if r + 1 < m {
                let q = idx(r + 1, cc);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(c(-1.0, 0.2));
            }
        }
    }
    let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        backward_error(&a, &x, &b) <= solve_backward_error(a.n, UC64),
        "residual {}",
        backward_error(&a, &x, &b)
    );
}
