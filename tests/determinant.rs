//! The determinant, the inverse and the dense constructors of the direct
//! solvers against a dense LU with partial pivoting.

pub mod common;

use common::constants::{exp_of_logs, gamma, log_det_error, solve_backward_error, U64, UC64};
use num_complex::Complex;
use rslab::*;

type C = Complex<f64>;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

trait Field: Scalar {
    const U: f64;
    fn make(v: C) -> Self;
}
impl Field for f64 {
    const U: f64 = U64;
    fn make(v: C) -> Self {
        v.re
    }
}
impl Field for C {
    const U: f64 = UC64;
    fn make(v: C) -> Self {
        v
    }
}

fn dense<T: Field>(n: usize, seed: u64, scale: f64, symmetric: bool) -> Vec<T> {
    let mut g = Lcg(seed);
    let mut a = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            if symmetric && j > i {
                continue;
            }
            let v = T::make(C::new(g.next(), g.next()) * scale);
            a[i * n + j] = v;
            if symmetric {
                a[j * n + i] = v;
            }
        }
    }
    a
}

fn to_c<T: Scalar>(v: T) -> C {
    C::new(v.real(), v.imag())
}

fn reference_log_det<T: Scalar>(n: usize, a: &[T]) -> (C, f64) {
    let mut m: Vec<C> = a.iter().map(|&v| to_c(v)).collect();
    let (mut phase, mut log_abs) = (C::new(1.0, 0.0), 0.0);
    for k in 0..n {
        let p = (k..n)
            .max_by(|&x, &y| m[x * n + k].norm().total_cmp(&m[y * n + k].norm()))
            .unwrap();
        if p != k {
            for j in 0..n {
                m.swap(k * n + j, p * n + j);
            }
            phase = -phase;
        }
        let piv = m[k * n + k];
        phase *= piv / piv.norm();
        log_abs += piv.norm().ln();
        for i in k + 1..n {
            let l = m[i * n + k] / piv;
            for j in k + 1..n {
                let u = m[k * n + j];
                m[i * n + j] -= l * u;
            }
        }
    }
    (phase, log_abs)
}

// https://github.com/milanofthe/rslab/issues/294
fn check_log_det<T: Scalar>(what: &str, n: usize, got: (T, f64), want: (C, f64)) {
    let (tol_per_unknown, tol_floor) = (2e-14, 1e-12);
    let tol = (tol_per_unknown * n as f64).max(tol_floor);
    let phase = to_c(got.0);
    assert!(
        (got.1 - want.1).abs() <= tol,
        "{what}: log|det| {} vs {}",
        got.1,
        want.1
    );
    assert!(
        (phase - want.0).norm() <= tol,
        "{what}: phase {phase} vs {}",
        want.0
    );
}

const SIZES: [usize; 6] = [1, 2, 3, 10, 100, 300];

fn log_det_cases<T: Field>() {
    for (symmetric, seed) in [(false, 11), (true, 31)] {
        for (s, &n) in SIZES.iter().enumerate() {
            let a = dense::<T>(n, seed + s as u64, 1.0, symmetric);
            let want = reference_log_det(n, &a);
            if symmetric {
                let m = CscMatrix::from_dense_row_major(n, &a).unwrap();
                let f = LdltSolver::factor(&m, &SolverSettings::default()).unwrap();
                check_log_det(&format!("ldlt n={n}"), n, f.log_det().unwrap(), want);
            }
            let g = GeneralCsc::from_dense_row_major(n, &a).unwrap();
            let lu = LuSolver::factor(&g, &SolverSettings::default()).unwrap();
            let what = if symmetric { "lu sym" } else { "lu" };
            check_log_det(&format!("{what} n={n}"), n, lu.log_det().unwrap(), want);
            if !symmetric {
                let klu = KluSolver::factor(&g, &KluSettings::default()).unwrap();
                check_log_det(&format!("klu n={n}"), n, klu.log_det().unwrap(), want);
            }
        }
    }
}

#[test]
fn log_det_matches_a_dense_lu_real() {
    log_det_cases::<f64>();
}

#[test]
fn log_det_matches_a_dense_lu_complex() {
    log_det_cases::<C>();
}

#[test]
fn log_det_survives_where_det_under_and_overflows() {
    let n = 300;
    for scale in [1e-3, 1e3] {
        let a = dense::<f64>(n, 7, scale, false);
        let want = reference_log_det(n, &a);
        let g = GeneralCsc::from_dense_row_major(n, &a).unwrap();
        let lu = LuSolver::factor(&g, &SolverSettings::default()).unwrap();
        check_log_det(&format!("lu scale={scale}"), n, lu.log_det().unwrap(), want);
        let det = lu.det().unwrap();
        assert!(
            if scale < 1.0 {
                det == 0.0
            } else {
                det.is_infinite()
            },
            "det of scale {scale} is {det}"
        );
        let s = dense::<f64>(n, 8, scale, true);
        let f = LdltSolver::factor(
            &CscMatrix::from_dense_row_major(n, &s).unwrap(),
            &SolverSettings::default(),
        )
        .unwrap();
        check_log_det(
            &format!("ldlt scale={scale}"),
            n,
            f.log_det().unwrap(),
            reference_log_det(n, &s),
        );
    }
}

#[test]
fn det_sign_of_a_real_symmetric_matrix_is_its_inertia() {
    for (s, &n) in SIZES.iter().enumerate() {
        let a = dense::<f64>(n, 51 + s as u64, 1.0, true);
        let f = LdltSolver::factor(
            &CscMatrix::from_dense_row_major(n, &a).unwrap(),
            &SolverSettings::default(),
        )
        .unwrap();
        let (phase, _) = f.log_det().unwrap();
        let sign = if f.inertia().negative % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        assert_eq!(
            phase,
            sign,
            "n={n}: phase {phase}, inertia {:?}",
            f.inertia()
        );
    }
}

#[test]
fn small_determinants_are_exact_in_form() {
    let within = |err: f64, det: f64, n: usize, a_norm: f64, inverse_sum: f64, u: f64| {
        let log_err = log_det_error(solve_backward_error(n, u), a_norm, inverse_sum);
        err <= det.abs() * (log_err + exp_of_logs(det.abs().ln()) + u)
    };
    let g = GeneralCsc::<f64>::from_dense_row_major(2, &[2.0, 1.0, 1.0, 3.0]).unwrap();
    let d = LuSolver::factor(&g, &SolverSettings::default())
        .unwrap()
        .det()
        .unwrap();
    assert!(
        within((d - 5.0).abs(), 5.0, 2, 4.0, 7.0 / 5.0, U64),
        "det [[2, 1], [1, 3]] is {d}, not 5"
    );
    let p = GeneralCsc::<f64>::from_dense_row_major(2, &[0.0, 1.0, 1.0, 0.0]).unwrap();
    let d = KluSolver::factor(&p, &KluSettings::default())
        .unwrap()
        .det()
        .unwrap();
    assert!(
        within((d + 1.0).abs(), 1.0, 2, 1.0, 2.0, U64),
        "det of the row swap [[0, 1], [1, 0]] is {d}, not -1"
    );
    let z = C::new(-3.0, 4.0);
    let g = GeneralCsc::from_dense_row_major(1, &[z]).unwrap();
    let d = LuSolver::factor(&g, &SolverSettings::default())
        .unwrap()
        .det()
        .unwrap();
    assert!(
        within(
            (d - z).norm(),
            z.norm(),
            1,
            z.norm(),
            z.norm().recip(),
            UC64
        ),
        "det of the complex 1x1 matrix [{z}] is {d}"
    );
}

#[test]
fn a_perturbed_factor_refuses_its_determinant() {
    let singular = [1.0, 2.0, 2.0, 4.0];
    let m = CscMatrix::<f64>::from_dense_row_major(2, &singular).unwrap();
    let f = LdltSolver::factor(&m, &SolverSettings::preconditioner(1e-8)).unwrap();
    assert!(
        f.n_perturbed() > 0,
        "the floor lifts no pivot of the singular matrix"
    );
    assert!(matches!(f.log_det(), Err(RslabError::InvalidInput(_))));
}

fn inverse_error<T: Scalar>(n: usize, a: &[T], x: &[T]) -> (f64, f64) {
    let mut err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = T::zero();
            for k in 0..n {
                s = s + a[i * n + k] * x[j * n + k];
            }
            let e = if i == j { s - T::one() } else { s };
            err = err.max(e.magnitude());
        }
    }
    let norm_inf = (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].magnitude()).sum::<f64>())
        .fold(0.0, f64::max);
    let x_max = x.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
    (err, norm_inf * x_max)
}

fn inverse_cases<T: Field>() {
    for (s, &n) in SIZES.iter().enumerate() {
        let omega = solve_backward_error(n, T::U);
        let per_scale = omega + gamma(n, T::U);
        let a = dense::<T>(n, 71 + s as u64, 1.0, false);
        let g = GeneralCsc::from_dense_row_major(n, &a).unwrap();
        for (name, x) in [
            (
                "lu",
                LuSolver::factor(&g, &SolverSettings::default())
                    .unwrap()
                    .inverse()
                    .unwrap(),
            ),
            (
                "klu",
                KluSolver::factor(&g, &KluSettings::default())
                    .unwrap()
                    .inverse()
                    .unwrap(),
            ),
        ] {
            let (err, scale) = inverse_error(n, &a, &x);
            assert!(
                err <= per_scale * scale + omega,
                "{name} n={n}: {err:e} (scale {scale:e})"
            );
        }
        let a = dense::<T>(n, 91 + s as u64, 1.0, true);
        let m = CscMatrix::from_dense_row_major(n, &a).unwrap();
        let x = LdltSolver::factor(&m, &SolverSettings::default())
            .unwrap()
            .inverse()
            .unwrap();
        let (err, scale) = inverse_error(n, &a, &x);
        assert!(
            err <= per_scale * scale + omega,
            "ldlt n={n}: {err:e} (scale {scale:e})"
        );
    }
}

#[test]
fn inverse_is_accurate_to_the_condition() {
    inverse_cases::<f64>();
    inverse_cases::<C>();
}

#[test]
fn dense_constructors_give_the_sparse_factor() {
    let n = 40;
    let mut a = dense::<C>(n, 5, 1.0, false);
    let mut s = dense::<C>(n, 6, 1.0, true);
    let mut g = Lcg(9);
    for i in 0..n {
        for j in 0..n {
            if i != j && g.next() < 0.0 {
                a[i * n + j] = C::new(0.0, 0.0);
                if j < i {
                    s[i * n + j] = C::new(0.0, 0.0);
                    s[j * n + i] = C::new(0.0, 0.0);
                }
            }
        }
    }
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let (mut rs, mut cs, mut vs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            if a[i * n + j] != C::new(0.0, 0.0) {
                r.push(i);
                c.push(j);
                v.push(a[i * n + j]);
            }
            if j <= i && s[i * n + j] != C::new(0.0, 0.0) {
                rs.push(i);
                cs.push(j);
                vs.push(s[i * n + j]);
            }
        }
    }
    let gd = GeneralCsc::from_dense_row_major(n, &a).unwrap();
    let gt = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
    assert_eq!(
        gd.nnz(),
        r.len(),
        "the dense constructor keeps an exact zero in the pattern"
    );
    let b: Vec<C> = (0..n).map(|i| C::new(i as f64, 1.0)).collect();
    let opts = SolverSettings::default();
    assert_eq!(
        LuSolver::factor(&gd, &opts).unwrap().solve(&b).unwrap(),
        LuSolver::factor(&gt, &opts).unwrap().solve(&b).unwrap(),
        "LU of the dense constructor differs from LU of the same triplets"
    );
    let md = CscMatrix::from_dense_row_major(n, &s).unwrap();
    let mt = CscMatrix::from_triplets(n, &rs, &cs, &vs).unwrap();
    assert_eq!(
        md.values.len(),
        rs.len(),
        "the symmetric dense constructor keeps an exact zero in the pattern"
    );
    assert_eq!(
        LdltSolver::factor(&md, &opts).unwrap().solve(&b).unwrap(),
        LdltSolver::factor(&mt, &opts).unwrap().solve(&b).unwrap(),
        "LDLT of the dense constructor differs from LDLT of the same triplets"
    );
    assert!(GeneralCsc::<f64>::from_dense_row_major(3, &[1.0; 8]).is_err());
}
