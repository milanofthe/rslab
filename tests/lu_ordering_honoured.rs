//! `LuSolver::factor(a, opts)` must analyze with the caller's settings: the ordering
//! method in `opts` decides the fill. (Regression: the one-shot factor path called the
//! default analysis and silently factored every matrix with AMD, whatever `opts` said.)
use num_complex::Complex;
use rslab::{GeneralCsc, LuSolver, OrderingMethod, SolverSettings};

/// 2D 5-point grid Laplacian (k x k), shifted to be unsymmetric-safe, as GeneralCsc.
fn grid(k: usize) -> GeneralCsc<Complex<f64>> {
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(Complex::new(4.0, 0.1));
            for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (ii, jj) = (i as i64 + di, j as i64 + dj);
                if ii >= 0 && jj >= 0 && (ii as usize) < k && (jj as usize) < k {
                    rows.push(p);
                    cols.push(ii as usize * k + jj as usize);
                    vals.push(Complex::new(-1.0, 0.0));
                }
            }
        }
    }
    GeneralCsc::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn factor_honours_the_ordering_in_the_settings() {
    let a = grid(60);
    let fill = |m: OrderingMethod| {
        LuSolver::factor(&a, &SolverSettings::exact().with_ordering(m))
            .unwrap()
            .factor_nnz()
    };
    let (amd, nd, rcm) = (
        fill(OrderingMethod::Amd),
        fill(OrderingMethod::MetisND),
        fill(OrderingMethod::Rcm),
    );
    // RCM is a band ordering: on a 2D grid its fill is far above minimum-degree /
    // nested dissection. Identical fills mean the setting was ignored.
    assert!(
        rcm > amd * 3 / 2 && rcm > nd * 3 / 2,
        "orderings must change the fill: amd {amd} nd {nd} rcm {rcm}"
    );
    assert_ne!(amd, rcm);
}

#[test]
fn ldlt_factor_with_honours_the_ordering_in_the_settings() {
    use rslab::{CscMatrix, LdltSolver};
    // Lower triangle of the same grid (CscMatrix stores the lower triangle only).
    let k = 60usize;
    let n = k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            rows.push(p);
            cols.push(p);
            vals.push(4.0f64);
            for (di, dj) in [(1usize, 0usize), (0, 1)] {
                let (ii, jj) = (i + di, j + dj);
                if ii < k && jj < k {
                    rows.push(ii * k + jj);
                    cols.push(p);
                    vals.push(-1.0);
                }
            }
        }
    }
    let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
    let fill = |m: OrderingMethod| {
        LdltSolver::factor_with(&a, &SolverSettings::exact().with_ordering(m))
            .unwrap()
            .factor_nnz()
    };
    let (amd, rcm) = (fill(OrderingMethod::Amd), fill(OrderingMethod::Rcm));
    assert!(
        rcm > amd * 3 / 2,
        "orderings must change the fill: amd {amd} rcm {rcm}"
    );
}
