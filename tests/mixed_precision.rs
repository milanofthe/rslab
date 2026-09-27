//! A factor of the demoted matrix preconditions a full-precision iteration:
//! the analysis is shared across precisions, the block apply solves all
//! columns at once, and refinement against the full-precision matrix
//! recovers its accuracy.

use num_complex::Complex64;
use rslab::{
    gmres_block, BackwardError, CscMatrix, Demote, Factorization, GeneralCsc, KrylovSettings,
    LdltSymbolic, LinearOperator, LuSymbolic, MixedPrecision, Preconditioner, RefinePolicy, Scalar,
    SolverSettings,
};

/// 2D grid (k x k) with a complex shift, full (`sym = false`, a small skew on
/// the couplings) or the lower triangle of the symmetric one.
fn grid<T: Scalar>(
    k: usize,
    sym: bool,
    diag: T,
    off: T,
) -> (usize, Vec<usize>, Vec<usize>, Vec<T>) {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..k {
        for j in 0..k {
            let p = i * k + j;
            r.push(p);
            c.push(p);
            v.push(diag);
            for (di, dj) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (ii, jj) = (i as i64 + di, j as i64 + dj);
                if ii < 0 || jj < 0 || ii as usize >= k || jj as usize >= k {
                    continue;
                }
                let q = ii as usize * k + jj as usize;
                if !sym {
                    r.push(p);
                    c.push(q);
                    v.push(if di + dj > 0 {
                        off
                    } else {
                        off * T::from_real(0.9)
                    });
                } else if q > p {
                    r.push(q);
                    c.push(p);
                    v.push(off);
                }
            }
        }
    }
    (k * k, r, c, v)
}

fn residual<T: Scalar>(a: &dyn LinearOperator<T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::zero(); b.len()];
    a.apply(x, &mut ax);
    let r: f64 = ax
        .iter()
        .zip(b)
        .map(|(&p, &q)| (p - q).magnitude_sq())
        .sum();
    let nb: f64 = b.iter().map(|v| v.magnitude_sq()).sum();
    (r / nb).sqrt()
}

#[test]
fn block_gmres_with_a_single_precision_lu_reaches_double_accuracy() {
    let c = Complex64::new;
    let (n, r, cc, v) = grid(40, false, c(4.0, 0.5), c(-1.0, 0.1));
    let a = GeneralCsc::from_triplets(n, &r, &cc, &v).unwrap();
    let opts = SolverSettings::default();
    // One analysis for both precisions.
    let sym = LuSymbolic::analyze(&a, &opts).unwrap();
    let m = MixedPrecision::new(sym.factor(&a.demoted(), &opts).unwrap());
    let nrhs = 3;
    let b: Vec<Complex64> = (0..n * nrhs)
        .map(|i| c((i % 7) as f64 - 3.0, 0.5))
        .collect();

    // The block apply solves the columns together and equals the column loop.
    let mut z = vec![Complex64::default(); n * nrhs];
    m.apply_block(&b, &mut z, nrhs, n).unwrap();
    for col in 0..nrhs {
        let zc = Factorization::solve(&m, &b[col * n..(col + 1) * n]).unwrap();
        assert_eq!(&z[col * n..(col + 1) * n], &zc[..]);
    }

    let settings = KrylovSettings::default().with_tol(1e-10);
    let res = gmres_block(&a, &b, nrhs, &m, &settings, None, None).unwrap();
    assert!(res.converged);
    for col in 0..nrhs {
        let (x, bc) = (&res.x[col * n..(col + 1) * n], &b[col * n..(col + 1) * n]);
        assert!(residual(&a, x, bc) < 1e-9, "column {col}");
    }
}

#[test]
fn refinement_against_the_double_matrix_recovers_double_accuracy() {
    let policy = RefinePolicy {
        max_steps: 10,
        target: 1e-14,
        measure: BackwardError::Normwise,
    };
    // Real symmetric: LDL^T in f32 under f64 refinement.
    let (n, r, c, v) = grid(40, true, 4.5f64, -1.0);
    let a = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
    let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
    let m = MixedPrecision::new(
        sym.factor(&a.demoted(), &SolverSettings::default())
            .unwrap(),
    );
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    let once = Factorization::solve(&m, &b).unwrap();
    let (x, outcome) = m.solve_refined(&a, &b, &policy).unwrap();
    assert!(residual(&a, &once, &b) > 1e-9, "one single-precision solve");
    assert!(residual(&a, &x, &b) < 1e-13, "refined: {outcome:?}");
    assert_eq!(f64::promote(1.5f64.demote()), 1.5);
}
