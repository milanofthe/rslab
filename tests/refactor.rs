//! Refactoring a supernodal LU in place: the bits of a fresh factorization,
//! and a failed refactorization leaves a solver that refuses to solve until
//! the next one succeeds.

use rslab::prelude::*;
use rslab::{GeneralCsc, LuSymbolic, SolveWork};

/// A 5-point stencil on an `m` by `m` grid with an unsymmetric coupling,
/// its values moved by `shift`.
fn grid(m: usize, shift: f64) -> GeneralCsc<f64> {
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(4.5 + (i % 3) as f64 * 0.1 + shift);
        let right = (i % m + 1 < m).then_some(i + 1);
        let down = (i / m + 1 < m).then_some(i + m);
        for j in right.into_iter().chain(down) {
            r.push(j);
            c.push(i);
            v.push(-1.0 - shift * (j % 3) as f64);
            r.push(i);
            c.push(j);
            v.push(-0.8 + shift);
        }
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

#[test]
fn a_refactored_lu_solves_as_a_fresh_one() {
    let opts = SolverSettings::default();
    for m in [5, 30] {
        let (a, next) = (grid(m, 0.0), grid(m, 0.7));
        let n = a.n;
        let sym = LuSymbolic::analyze(&a, &opts).unwrap();
        let mut lu = sym.factor(&a, &opts).unwrap();
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
        let (mut x, mut work) = (vec![0.0; n], SolveWork::new());
        for values in [&next, &a, &next] {
            sym.refactor(values, &opts, &mut lu).unwrap();
            lu.solve_into(&b, &mut x, &mut work).unwrap();
            let fresh = sym.factor(values, &opts).unwrap();
            assert_eq!(x, fresh.solve(&b).unwrap(), "n = {n}");
            assert_eq!(lu.factor_nnz(), fresh.factor_nnz());
        }
    }
}

#[test]
fn a_failed_refactor_refuses_to_solve_until_the_next_succeeds() {
    let opts = SolverSettings::default();
    let a = grid(6, 0.0);
    let n = a.n;
    let sym = LuSymbolic::analyze(&a, &opts).unwrap();
    let mut lu = sym.factor(&a, &opts).unwrap();
    let mut singular = a.clone();
    singular.values.iter_mut().for_each(|v| *v = 0.0);
    assert!(sym.refactor(&singular, &opts, &mut lu).is_err());
    let b = vec![1.0; n];
    assert!(lu.solve(&b).is_err());
    sym.refactor(&a, &opts, &mut lu).unwrap();
    assert_eq!(
        lu.solve(&b).unwrap(),
        sym.factor(&a, &opts).unwrap().solve(&b).unwrap()
    );
}
