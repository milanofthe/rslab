//! The Newton loop's contract: once a factorization exists and a
//! `SolveWork` has served one solve, refactoring (KLU) and solving into the
//! caller's buffers (all three direct solvers, one or several right-hand
//! sides, plain and transposed) allocate nothing on the calling thread and
//! give the bits of the allocating entry points. A factor large enough for
//! the thread pool adds only what the pool's own work queues allocate, now
//! and then.
//!
//! One test in this binary, so no other test allocates alongside.

use rslab::prelude::*;
use rslab::{GeneralCsc, KluParallel, KluSettings, KluSolver, LdltSymbolic, LuSymbolic, SolveWork};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;
static POOLED: AtomicUsize = AtomicUsize::new(0);
thread_local!(static MINE: Cell<usize> = const { Cell::new(0) });
thread_local!(static IN_POOL: Cell<bool> = const { Cell::new(false) });

impl Counting {
    fn count() {
        MINE.with(|c| c.set(c.get() + 1));
        if IN_POOL.with(Cell::get) {
            POOLED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        Counting::count();
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        Counting::count();
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// Allocations on the calling thread: the idle workers of the global pool
/// tidy their queues in the background, on their own threads.
fn mine() -> usize {
    MINE.with(Cell::get)
}

/// Allocations on the threads of the test's pool, for a solve spread over
/// it.
fn pooled() -> usize {
    POOLED.load(Ordering::Relaxed)
}

/// At most this many allocations over a hundred solves on the pool: its
/// work-stealing queues grow and retire buffers now and then.
const POOL: usize = 5;

/// Allocations of `f` by `count` after two warm-up runs, over a hundred
/// runs.
fn allocs(count: fn() -> usize, mut f: impl FnMut()) -> usize {
    f();
    f();
    let before = count();
    for _ in 0..100 {
        f();
    }
    count() - before
}

/// A 5-point stencil on an `m` by `m` grid, as triplets: symmetric (lower
/// triangle only) or with an unsymmetric coupling.
fn grid(m: usize, lower_only: bool) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
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

/// Every `*_into` entry point of one solver: at most `$limit` allocations
/// over a hundred warm solves, and the bits of the allocating twin.
macro_rules! check_solves {
    ($what:expr, $s:expr, $n:expr, $count:expr, $limit:expr) => {{
        let (s, n) = (&$s, $n);
        let nrhs = 3;
        let b: Vec<f64> = (0..n * nrhs).map(|i| 1.0 + (i % 7) as f64).collect();
        let mut x = vec![0.0; n * nrhs];
        let mut work = SolveWork::new();
        let one = allocs($count, || {
            s.solve_into(&b[..n], &mut x[..n], &mut work).unwrap()
        });
        assert!(one <= $limit, "{}: solve_into allocates", $what);
        assert_eq!(
            x[..n],
            s.solve(&b[..n]).unwrap()[..],
            "{}: solve bits",
            $what
        );
        let tr = allocs($count, || {
            s.solve_transpose_into(&b[..n], &mut x[..n], &mut work)
                .unwrap()
        });
        assert!(tr <= $limit, "{}: solve_transpose_into allocates", $what);
        assert_eq!(
            x[..n],
            s.solve_transpose(&b[..n]).unwrap()[..],
            "{}: transpose bits",
            $what
        );
        let many = allocs($count, || {
            s.solve_many_into(&b, nrhs, &mut x, &mut work).unwrap()
        });
        assert!(many <= $limit, "{}: solve_many_into allocates", $what);
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
    let opts = SolverSettings::default();
    for m in [6, 40] {
        // KLU: the refactor and the solves.
        let (n, r, c, v) = grid(m, false);
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let settings = KluSettings::default().with_parallel(KluParallel::Off);
        let mut klu = KluSolver::factor(&a, &settings).unwrap();
        let refactor = allocs(mine, || klu.refactor(&a).unwrap());
        assert_eq!(refactor, 0, "KLU refactor allocates (n = {n})");
        check_solves!(format!("KLU n={n}"), klu, n, mine, 0);

        // Supernodal LU on the same pattern, its analysis kept.
        let lu = LuSymbolic::analyze(&a, &opts)
            .unwrap()
            .factor(&a, &opts)
            .unwrap();
        check_solves!(format!("LU n={n}"), lu, n, mine, 0);

        // Supernodal LDL^T on the symmetric stencil.
        let (n, r, c, v) = grid(m, true);
        let s = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
        let ldlt = LdltSymbolic::analyze(&s, &opts)
            .unwrap()
            .factor(&s, &opts)
            .unwrap();
        check_solves!(format!("LDLT n={n}"), ldlt, n, mine, 0);

        // The pool's path, forced, entered from a worker of the pool: only
        // the pool's own allocations, and the bits of the calling thread's
        // path.
        let mut par = opts.clone();
        par.solve.par_min_work = 0;
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
            .start_handler(|_| IN_POOL.with(|p| p.set(true)))
            .build()
            .unwrap();
        pool.install(|| {
            check_solves!(format!("LU n={n}, pool"), lu_par, n, pooled, POOL);
            check_solves!(format!("LDLT n={n}, pool"), ldlt_par, n, pooled, POOL);
        });
        let b: Vec<f64> = (0..n).map(|i| 1.0 - (i % 5) as f64).collect();
        assert_eq!(lu_par.solve(&b).unwrap(), lu.solve(&b).unwrap());
        assert_eq!(ldlt_par.solve(&b).unwrap(), ldlt.solve(&b).unwrap());
    }
}
