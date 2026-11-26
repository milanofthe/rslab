//! The matrix family `A0 + alpha*A1 + E` on one layout ([`Pencil`]): a
//! member's solves against the dense solve of the member's matrix,
//! assembled here from the entries, for real and complex `alpha`; the
//! unchanged-member memo of `AutoSolver::factor_member`; the products of
//! [`EntryRows`] against dense; the same bits on one and several threads.
//!
//! A solve is held to the forward error a stable solve can have: the
//! normwise backward error bound `GROWTH * gamma_m` (as in
//! `thread_determinism`) times the condition number `||A||inf ||A^-1||inf`
//! of the dense reference, `A^-1` the dense solve of the identity.

use num_complex::Complex64 as C;
use rslab::{AutoSettings, AutoSymbolic, EntryRows, Pencil, Scalar};

pub mod common;

use common::constants::{gamma, DIAGONALLY_DOMINANT_GROWTH, U64};

struct Family {
    n: usize,
    a0: (Vec<usize>, Vec<usize>, Vec<f64>),
    a1: (Vec<usize>, Vec<usize>, Vec<f64>),
    extra: (Vec<usize>, Vec<usize>),
}

fn family() -> Family {
    let n = 6;
    let mut a0 = (vec![], vec![], vec![]);
    for i in 0..n {
        a0.0.push(i);
        a0.1.push(i);
        a0.2.push(8.0 + i as f64);
    }
    for (r, c, v) in [(0, 1, 1.0), (2, 0, -0.5), (0, 1, 0.25), (4, 5, 2.0)] {
        a0.0.push(r);
        a0.1.push(c);
        a0.2.push(v);
    }
    let a1 = (
        vec![1, 3, 0, 1, 5, 3],
        vec![1, 2, 1, 1, 0, 3],
        vec![3.0, 0.5, -1.0, 1.5, 0.75, 2.0],
    );
    let extra = (vec![3, 1], vec![2, 4]);
    Family { n, a0, a1, extra }
}

fn pencil(f: &Family) -> Pencil {
    let mut p = Pencil::new(
        f.n,
        (&f.a0.0, &f.a0.1),
        (&f.a1.0, &f.a1.1),
        (&f.extra.0, &f.extra.1),
    )
    .unwrap();
    p.update(|a0, a1| {
        a0.copy_from_slice(&f.a0.2);
        a1.copy_from_slice(&f.a1.2);
    });
    p
}

fn dense(f: &Family, alpha: C, extra: &[C]) -> Vec<C> {
    let n = f.n;
    let mut a = vec![C::new(0.0, 0.0); n * n];
    for k in 0..f.a0.2.len() {
        a[f.a0.0[k] * n + f.a0.1[k]] += f.a0.2[k];
    }
    for k in 0..f.a1.2.len() {
        a[f.a1.0[k] * n + f.a1.1[k]] += alpha * f.a1.2[k];
    }
    for k in 0..extra.len() {
        a[f.extra.0[k] * n + f.extra.1[k]] += extra[k];
    }
    a
}

fn transposed(a: &[C], n: usize) -> Vec<C> {
    (0..n * n).map(|k| a[(k % n) * n + k / n]).collect()
}

fn inf_norm(a: &[C], n: usize) -> f64 {
    (0..n)
        .map(|i| a[i * n..(i + 1) * n].iter().map(|v| v.norm()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn check(a: &[C], n: usize, b: &[C], x: &[C], complex: bool, what: &str) {
    let want = rslab::dense::lu_solve(a, n, b, 1).unwrap();
    let mut eye = vec![C::new(0.0, 0.0); n * n];
    (0..n).for_each(|i| eye[i * n + i] = C::new(1.0, 0.0));
    let inv = rslab::dense::lu_solve(a, n, &eye, n).unwrap();
    let kappa = inf_norm(a, n) * inf_norm(&inv, n);
    let m = if complex { 2 * n } else { n };
    let bound = kappa * DIAGONALLY_DOMINANT_GROWTH * gamma(m, U64);
    let inf = |v: &[C]| v.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let diff: Vec<C> = x.iter().zip(&want).map(|(p, q)| p - q).collect();
    let err = inf(&diff) / inf(&want);
    assert!(err <= bound, "{what}: forward error {err:e} > {bound:e}");
}

#[test]
fn real_members_solve_as_dense() {
    let f = family();
    let p = pencil(&f);
    let sym = AutoSymbolic::new(p.layout().clone()).unwrap();
    let mut s = sym.factorizer::<f64>();
    let settings = AutoSettings::default();
    let b: Vec<f64> = (0..f.n).map(|i| 1.0 + 0.5 * i as f64).collect();
    let extra = [0.75, -1.25];
    for alpha in [0.0, 0.5, 3.0, -0.25] {
        assert!(s.factor_member(&p, alpha, &extra, &settings).unwrap());
        let ex: Vec<C> = extra.iter().map(|&v| C::new(v, 0.0)).collect();
        let a = dense(&f, C::new(alpha, 0.0), &ex);
        let bc: Vec<C> = b.iter().map(|&v| C::new(v, 0.0)).collect();
        let to_c = |x: Vec<f64>| x.into_iter().map(|v| C::new(v, 0.0)).collect::<Vec<C>>();
        let x = to_c(s.solve(&b).unwrap());
        check(&a, f.n, &bc, &x, false, &format!("alpha {alpha}"));
        let xt = to_c(s.solve_transpose(&b).unwrap());
        let at = transposed(&a, f.n);
        check(
            &at,
            f.n,
            &bc,
            &xt,
            false,
            &format!("alpha {alpha}, transposed"),
        );
    }
}

#[test]
fn complex_members_solve_as_dense() {
    let f = family();
    let p = pencil(&f);
    let sym = AutoSymbolic::new(p.layout().clone()).unwrap();
    let mut s = sym.factorizer::<C>();
    let settings = AutoSettings::default();
    let b: Vec<C> = (0..f.n).map(|i| C::new(1.0, 0.5 * i as f64)).collect();
    for w in [0.0, 1.0, 7.5, 1e3] {
        let alpha = C::new(0.0, w);
        let extra = [C::from_polar(0.5, -w * 1e-3), C::from_polar(1.5, -w * 2e-3)];
        assert!(s.factor_member(&p, alpha, &extra, &settings).unwrap());
        let a = dense(&f, alpha, &extra);
        check(&a, f.n, &b, &s.solve(&b).unwrap(), true, &format!("w {w}"));
        let at = transposed(&a, f.n);
        let xt = s.solve_transpose(&b).unwrap();
        check(&at, f.n, &b, &xt, true, &format!("w {w}, transposed"));
    }
}

#[test]
fn unchanged_member_keeps_its_factor() {
    let f = family();
    let mut p = pencil(&f);
    let sym = AutoSymbolic::new(p.layout().clone()).unwrap();
    let mut s = sym.factorizer::<f64>();
    let st = AutoSettings::default();
    let extra = [0.75, -1.25];
    assert!(s.factor_member(&p, 2.0, &extra, &st).unwrap());
    assert!(!s.factor_member(&p, 2.0, &extra, &st).unwrap(), "unchanged");
    p.update(|_, _| {});
    assert!(
        s.factor_member(&p, 2.0, &extra, &st).unwrap(),
        "after update"
    );
    assert!(!s.factor_member(&p, 2.0, &extra, &st).unwrap());
    assert!(s.factor_member(&p, 2.5, &extra, &st).unwrap(), "new alpha");
    assert!(
        s.factor_member(&p, 2.5, &[0.75, -1.0], &st).unwrap(),
        "new extra"
    );
    assert!(!s.factor_member(&p, 2.5, &[0.75, -1.0], &st).unwrap());
    let q = pencil(&f);
    assert!(
        s.factor_member(&q, 2.5, &[0.75, -1.0], &st).unwrap(),
        "another pencil of the same terms is another family"
    );
    let _ = s.values_mut();
    assert!(
        s.factor_member(&q, 2.5, &[0.75, -1.0], &st).unwrap(),
        "values written by hand are taken for the member's"
    );
    let mut fork = s.fork();
    assert!(
        !fork.factor_member(&q, 2.5, &[0.75, -1.0], &st).unwrap(),
        "a fork of the factor does not hold the member"
    );
    let mut zero = Pencil::new(1, (&[0], &[0]), (&[], &[]), (&[], &[])).unwrap();
    zero.update(|a0, _| a0[0] = 0.0);
    let zsym = AutoSymbolic::new(zero.layout().clone()).unwrap();
    let mut z = zsym.factorizer::<f64>();
    assert!(z.factor_member(&zero, 0.0, &[], &st).is_err());
    assert!(
        z.factor_member(&zero, 0.0, &[], &st).is_err(),
        "a failed factorization holds a member"
    );
}

#[test]
fn entry_rows_products_match_dense() {
    let f = family();
    let (rows, cols, vals) = &f.a1;
    let e = EntryRows::new(f.n, rows, cols).unwrap();
    let x: Vec<C> = (0..f.n)
        .map(|i| C::new(1.0 - 0.3 * i as f64, 0.2 * i as f64))
        .collect();
    let mut want = vec![C::new(0.0, 0.0); f.n];
    let mut abs_want = vec![0.0; f.n];
    for k in 0..vals.len() {
        want[rows[k]] += vals[k] * x[cols[k]];
        abs_want[rows[k]] += vals[k].abs() * x[cols[k]].norm();
    }
    let mut y = vec![C::new(0.0, 0.0); f.n];
    e.mul_into(vals, &x, &mut y);
    let xr: Vec<f64> = x.iter().map(|v| v.re).collect();
    let mut yr = vec![0.0; f.n];
    e.mul_into(vals, &xr, &mut yr);
    let xa: Vec<f64> = x.iter().map(|v| v.norm()).collect();
    let mut ya = vec![0.0; f.n];
    e.abs_mul_into(vals, &xa, &mut ya);
    let u = f64::EPSILON;
    for r in 0..f.n {
        let terms: f64 = (0..vals.len())
            .filter(|&k| rows[k] == r)
            .map(|k| vals[k].abs() * x[cols[k]].norm())
            .sum();
        let products_in_row_at_most = vals.len() as f64;
        let tol = 4.0 * products_in_row_at_most * u * terms;
        let want_r: f64 = (0..vals.len())
            .filter(|&k| rows[k] == r)
            .map(|k| vals[k] * xr[cols[k]])
            .sum();
        assert!((y[r] - want[r]).norm() <= tol, "row {r}");
        assert!((yr[r] - want_r).abs() <= tol, "real row {r}");
        assert!((ya[r] - abs_want[r]).abs() <= tol, "abs row {r}");
    }
    assert!(EntryRows::new(2, &[0, 2], &[0, 0]).is_err());
    assert!(EntryRows::new(2, &[0], &[0, 1]).is_err());
}

fn large(n: usize, deg: usize) -> Pencil {
    let mut t: [(Vec<usize>, Vec<usize>); 2] = Default::default();
    for (s, (rows, cols)) in t.iter_mut().enumerate() {
        for r in 0..n {
            for j in 0..deg {
                rows.push(r);
                cols.push((r * (s + 1) + j * j * 7919) % n);
            }
        }
    }
    let mut p = Pencil::new(n, (&t[0].0, &t[0].1), (&t[1].0, &t[1].1), (&[], &[])).unwrap();
    p.update(|a0, a1| {
        (a0.iter_mut().enumerate()).for_each(|(k, v)| *v = 1.0 / (1.0 + k as f64));
        (a1.iter_mut().enumerate()).for_each(|(k, v)| *v = (k as f64).sin());
    });
    p
}

#[test]
fn same_bits_on_one_and_several_threads() {
    let p = large(200_000, 5);
    let n = p.layout().n();
    let x: Vec<C> = (0..n)
        .map(|i| C::new((i as f64).cos(), 1.0 / (1.0 + i as f64)))
        .collect();
    let xr: Vec<f64> = x.iter().map(|v| v.re).collect();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut fr = vec![0.0; p.layout().nnz()];
            p.fill(0.75, &[], &mut fr);
            let mut fc = vec![C::new(0.0, 0.0); p.layout().nnz()];
            p.fill(C::new(0.0, 3.0), &[], &mut fc);
            let mut y = vec![C::new(0.0, 0.0); n];
            p.a1_rows().mul_into(p.a1(), &x, &mut y);
            let mut ya = vec![0.0; n];
            p.a0_rows().abs_mul_into(p.a0(), &xr, &mut ya);
            (fr, fc, y, ya)
        })
    };
    let one = run(1);
    for threads in [2, 4] {
        assert!(run(threads) == one, "{threads} threads");
    }
}

#[test]
fn fill_is_the_scatter_of_the_member() {
    let f = family();
    let p = pencil(&f);
    let alpha = C::new(0.5, -2.0);
    let extra = [C::new(1.0, 1.0), C::new(-0.5, 0.25)];
    let mut entries: Vec<C> = f.a0.2.iter().map(|&v| C::from_real(v)).collect();
    entries.extend(f.a1.2.iter().map(|&v| alpha * C::from_real(v)));
    entries.extend_from_slice(&extra);
    let mut want = vec![C::new(0.0, 0.0); p.layout().nnz()];
    p.layout().scatter(&entries, &mut want);
    let mut got = vec![C::new(0.0, 0.0); p.layout().nnz()];
    p.fill(alpha, &extra, &mut got);
    assert_eq!(
        got, want,
        "the fill is not the scatter of A0, alpha A1 and E in the layout's order"
    );
}
