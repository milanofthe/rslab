//! The Newton loop's contract: once a factorization exists and a
//! `SolveWork` has served one solve, refactoring (KLU) and solving into the
//! caller's buffers (all three direct solvers, one right-hand side plain and
//! transposed; KLU also several) allocate nothing on the calling thread and
//! give the bits of the allocating entry points. The supernodal solvers' block
//! solves of several right-hand sides take their nodes' products as GEMMs,
//! whose packing buffers the `gemm` crate allocates per product. The pool's
//! path is held to the bits of the calling thread's: its tasks allocate on
//! the workers they run on.
//!
//! The counts are the `allocs` of an rsprobe scope around the runs, in a
//! context of the test's own (`rsprobe::Alloc`, feature `rsprobe/alloc`), at
//! INFO, the default threshold: rslab's scopes and measures are active.

use rslab::prelude::*;
use rslab::{GeneralCsc, KluParallel, KluSettings, KluSolver, LdltSymbolic, LuSymbolic, SolveWork};
use rsprobe::{Context, Level};

#[global_allocator]
static ALLOC: rsprobe::Alloc = rsprobe::Alloc::new();

fn allocs(threshold: Level, f: impl FnMut()) -> u64 {
    counted(threshold, 2, 100, f)
}

fn counted(threshold: Level, warm: usize, runs: usize, mut f: impl FnMut()) -> u64 {
    let ctx = Context::builder().threshold(threshold).build();
    let _on = ctx.enter();
    (0..warm).for_each(|_| f());
    {
        let _s = rsprobe::scope!(ERROR, "runs");
        (0..runs).for_each(|_| f());
    }
    let r = ctx.record();
    let node = r.find(&["runs"]).expect("the runs' scope");
    r.nodes[node as usize].get("allocs")
}

fn five_point_grid(m: usize, lower_only: bool) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(4.5 + (i % 3) as f64 * 0.1);
        let right = (i % m + 1 < m).then_some(i + 1);
        let down = (i / m + 1 < m).then_some(i + m);
        for j in right.into_iter().chain(down) {
            r.push(j);
            c.push(i);
            v.push(-1.0);
            if !lower_only {
                r.push(i);
                c.push(j);
                v.push(-0.8);
            }
        }
    }
    (n, r, c, v)
}

macro_rules! check_solves {
    ($what:expr, $s:expr, $n:expr, $level:expr, $limit:expr, $many_free:expr) => {{
        let (s, n) = (&$s, $n);
        let nrhs = 3;
        let b: Vec<f64> = (0..n * nrhs).map(|i| 1.0 + (i % 7) as f64).collect();
        let mut x = vec![0.0; n * nrhs];
        let mut work = SolveWork::new();
        let one = allocs($level, || {
            s.solve_into(&b[..n], &mut x[..n], &mut work).unwrap()
        });
        assert!(one <= $limit, "{}: solve_into allocates {one}", $what);
        assert_eq!(
            x[..n],
            s.solve(&b[..n]).unwrap()[..],
            "{}: solve bits",
            $what
        );
        let tr = allocs($level, || {
            s.solve_transpose_into(&b[..n], &mut x[..n], &mut work)
                .unwrap()
        });
        assert!(
            tr <= $limit,
            "{}: solve_transpose_into allocates {tr}",
            $what
        );
        assert_eq!(
            x[..n],
            s.solve_transpose(&b[..n]).unwrap()[..],
            "{}: transpose bits",
            $what
        );
        let many = allocs($level, || {
            s.solve_many_into(&b, nrhs, &mut x, &mut work).unwrap()
        });
        assert!(
            !$many_free || many <= $limit,
            "{}: solve_many_into allocates {many}",
            $what
        );
        assert_eq!(
            x,
            s.solve_many(&b, nrhs).unwrap(),
            "{}: solve_many bits",
            $what
        );
    }};
}

#[test]
fn a_warm_newton_loop_allocates_nothing() {
    let mut opts = SolverSettings::default();
    opts.solve.par_min_work = Some(usize::MAX);
    for m in [6, 40] {
        let (n, r, c, v) = five_point_grid(m, false);
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let settings = KluSettings::default().with_parallel(KluParallel::Off);
        let mut klu = KluSolver::factor(&a, &settings).unwrap();
        let refactor = allocs(Level::INFO, || klu.refactor(&a).unwrap());
        assert_eq!(refactor, 0, "KLU refactor allocates (n = {n})");
        check_solves!(format!("KLU n={n}"), klu, n, Level::INFO, 0, true);

        let lu = LuSymbolic::analyze(&a, &opts)
            .unwrap()
            .factor(&a, &opts)
            .unwrap();
        check_solves!(format!("LU n={n}"), lu, n, Level::INFO, 0, false);

        let (n, r, c, v) = five_point_grid(m, true);
        let s = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
        let ldlt = LdltSymbolic::analyze(&s, &opts)
            .unwrap()
            .factor(&s, &opts)
            .unwrap();
        check_solves!(format!("LDLT n={n}"), ldlt, n, Level::INFO, 0, false);

        let mut par = opts.clone();
        par.solve.par_min_work = Some(0);
        let lu_par = LuSymbolic::analyze(&a, &par)
            .unwrap()
            .factor(&a, &par)
            .unwrap();
        let ldlt_par = LdltSymbolic::analyze(&s, &par)
            .unwrap()
            .factor(&s, &par)
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            check_solves!(
                format!("LU n={n}, pool"),
                lu_par,
                n,
                Level::INFO,
                u64::MAX,
                false
            );
            check_solves!(
                format!("LDLT n={n}, pool"),
                ldlt_par,
                n,
                Level::INFO,
                u64::MAX,
                false
            );
        });
        let b: Vec<f64> = (0..n).map(|i| 1.0 - (i % 5) as f64).collect();
        assert_eq!(
            lu_par.solve(&b).unwrap(),
            lu.solve(&b).unwrap(),
            "LU n={n}: the pool's path differs from the calling thread's bits"
        );
        assert_eq!(
            ldlt_par.solve(&b).unwrap(),
            ldlt.solve(&b).unwrap(),
            "LDLT n={n}: the pool's path differs from the calling thread's bits"
        );
    }
}

#[test]
fn a_warm_supernodal_refactor_allocates_the_same_at_every_size() {
    let opts = SolverSettings {
        threads: rslab::Threads::Fixed(1),
        ..SolverSettings::default()
    };
    let count = |m: usize| {
        let (n, r, c, v) = five_point_grid(m, false);
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let sym = LuSymbolic::analyze(&a, &opts).unwrap();
        let mut lu = sym.factor(&a, &opts).unwrap();
        let lu_count = allocs(Level::INFO, || sym.refactor(&a, &opts, &mut lu).unwrap());
        let (n, r, c, v) = five_point_grid(m, true);
        let s = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
        let dsym = LdltSymbolic::analyze(&s, &opts).unwrap();
        let mut ldlt = dsym.factor(&s, &opts).unwrap();
        let ldlt_count = allocs(Level::INFO, || dsym.refactor(&s, &opts, &mut ldlt).unwrap());
        (lu_count, ldlt_count)
    };
    let (smallest_side, larger_side) = (6, 12);
    let (lu, ldlt) = count(smallest_side);
    assert_eq!(
        count(larger_side),
        (lu, ldlt),
        "per 100 refactors, n = 144 against n = 36"
    );
}

#[test]
fn block_arnoldi_steps_allocate_nothing() {
    use num_complex::Complex;
    use rslab::{BlockArnoldi, FnOperator, KrylovSettings};
    let n = 64;
    let a: Vec<Complex<f64>> = (0..n * n)
        .map(|q| Complex::new(((q * 7 + 3) % 11) as f64, ((q * 5 + 1) % 13) as f64))
        .collect();
    let op = FnOperator::new(n, |x: &[Complex<f64>], y: &mut [Complex<f64>], s: usize| {
        for c in 0..s {
            for i in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for j in 0..n {
                    acc += a[i * n + j] * x[c * n + j];
                }
                y[c * n + i] = acc;
            }
        }
    });
    let r: Vec<Complex<f64>> = (0..2 * n)
        .map(|i| Complex::new(1.0 + i as f64, 0.5))
        .collect();
    let mut ar = BlockArnoldi::new(&r, n, 2, 24, &KrylovSettings::default()).unwrap();
    let steps = counted(Level::INFO, 1, 20, || {
        ar.step(&op).unwrap();
    });
    assert_eq!(steps, 0, "allocations in 20 steps");
}
