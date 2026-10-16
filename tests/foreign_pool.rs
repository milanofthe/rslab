//! Factorizations, solves and Krylov solves from inside another rayon pool:
//! a parallel sweep whose every task calls rslab under a worker policy of its
//! own must not nest the sweep's tasks into each other.
//!
//! A worker of a rayon pool that waits (in `join` or `scope` on its own pool,
//! or in a cross-pool `install`) takes its own pool's other jobs meanwhile; a
//! task of the sweep taken that way runs on the stack of the task that waits,
//! and a sweep of many tasks then nests without bound. Each test counts, per
//! worker, the tasks in progress and fails at the first task that starts
//! while another one is in progress on its worker: a nesting is caught where
//! it happens, on every run, not only where it overflows a stack.

use rslab::{
    gmres_block, AutoBackend, AutoSettings, AutoSymbolic, CscLayout, GeneralCsc, KrylovSettings,
    LuSolver, SolverSettings, Threads,
};
use std::cell::Cell;
use std::sync::Mutex;

thread_local!(static IN_PROGRESS: Cell<usize> = const { Cell::new(0) });

/// A task of the sweep in progress on this worker, counted until it ends
/// (or unwinds).
struct InProgress;

impl InProgress {
    /// The task `k` started; fails if another one is in progress here.
    fn start(k: usize) -> Self {
        let others = IN_PROGRESS.with(|d| d.replace(d.get() + 1));
        let task = InProgress;
        assert_eq!(
            others, 0,
            "task {k} started inside another task of the sweep"
        );
        task
    }
}

impl Drop for InProgress {
    fn drop(&mut self) {
        IN_PROGRESS.with(|d| d.set(d.get() - 1));
    }
}

/// The `(rows, cols)` of an `m x m` grid Laplacian's pattern.
fn grid(m: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut rows, mut cols) = (Vec::new(), Vec::new());
    for i in 0..m * m {
        rows.push(i);
        cols.push(i);
        for j in [i + 1, i + m] {
            if j < m * m && (j != i + 1 || j % m != 0) {
                rows.extend([i, j]);
                cols.extend([j, i]);
            }
        }
    }
    (rows, cols)
}

/// The entries of the grid Laplacian plus `shift` on the diagonal, in the
/// pattern's order.
fn laplacian(rows: &[usize], cols: &[usize], shift: f64) -> Vec<f64> {
    (rows.iter().zip(cols))
        .map(|(&r, &c)| if r == c { 4.0 + shift } else { -1.0 })
        .collect()
}

/// Workers of the sweeps' pool.
const WORKERS: usize = 4;

/// `tasks` tasks of a sweep on a foreign pool, each with the state `init`
/// makes per worker; fails at the first task that starts on a worker with
/// another task in progress. Every task is a job of its own, spawned up
/// front: the tasks wait in the queue of the worker that spawned them, and a
/// worker done with one looks for the next in the other workers' queues, so
/// a section of rslab run on the sweep's pool has its pieces taken while
/// tasks still wait, and its worker, waiting for them, takes a task.
fn sweep<S: Send>(tasks: usize, init: impl Fn() -> S + Sync, task: impl Fn(&mut S, usize) + Sync) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(WORKERS)
        .build()
        .unwrap();
    let states: Vec<Mutex<Option<S>>> = (0..WORKERS).map(|_| Mutex::new(None)).collect();
    let (states, init, task) = (&states, &init, &task);
    pool.install(|| {
        rayon::scope(|s| {
            for k in 0..tasks {
                s.spawn(move |_| {
                    let _task = InProgress::start(k);
                    let worker = rayon::current_thread_index().unwrap();
                    let mut state = states[worker].lock().unwrap();
                    task(state.get_or_insert_with(init), k);
                });
            }
        })
    });
}

/// A sweep of supernodal factorizations under `solver`, each followed by
/// `solves` solves.
fn factor_sweep(solver: SolverSettings, solves: usize) {
    let m = 32;
    let n = m * m;
    let (rows, cols) = grid(m);
    let layout = CscLayout::new(n, &rows, &cols).unwrap();
    let mut vals = vec![0.0; layout.nnz()];
    layout.scatter(&laplacian(&rows, &cols, 0.0), &mut vals);
    let settings = AutoSettings::default()
        .with_supernodal_flops_per_entry(0.0)
        .with_supernodal(solver);
    let sym = AutoSymbolic::analyze(layout, &vals, &settings).unwrap();
    sweep(
        1000,
        || sym.factorizer::<f64>(),
        |f, k| {
            f.values_mut().copy_from_slice(&vals);
            f.values_mut()[0] = 4.0 + k as f64 * 1e-6;
            assert_eq!(f.refactor(&settings).unwrap(), AutoBackend::Ldlt);
            for _ in 0..solves {
                assert!(f.solve(&vec![1.0; n]).unwrap()[0].is_finite());
            }
        },
    );
}

/// One worker per factorization, as a sweep of many points runs them: the
/// factorization in a pool of its own, the solve on the calling thread.
#[test]
fn a_sweep_of_serial_factorizations_does_not_nest() {
    factor_sweep(SolverSettings::default().with_threads(Threads::Fixed(1)), 1);
}

/// Two workers per factorization and every solve on the pool, several solves
/// per factorization: they run in a pool of their own, the sweep's worker
/// waiting for it without taking work.
#[test]
fn a_sweep_of_parallel_factorizations_and_solves_does_not_nest() {
    let mut solver = SolverSettings::default().with_threads(Threads::Fixed(2));
    solver.solve.par_min_work = 0;
    factor_sweep(solver, 8);
}

/// Block GMRES preconditioned by a factor of two workers: its
/// orthogonalization runs in a pool of the factor's width, entered without
/// taking the sweep's work.
#[test]
fn a_sweep_of_krylov_solves_does_not_nest() {
    let m = 30;
    let n = m * m;
    let (rows, cols) = grid(m);
    let a = GeneralCsc::from_triplets(n, &rows, &cols, &laplacian(&rows, &cols, 0.0)).unwrap();
    let near = GeneralCsc::from_triplets(n, &rows, &cols, &laplacian(&rows, &cols, 0.5)).unwrap();
    let lu = LuSolver::factor(&near, &SolverSettings::default().with_threads(2)).unwrap();
    let settings = KrylovSettings::default().with_tol(1e-10);
    sweep(
        400,
        || (),
        |_, k| {
            let b: Vec<f64> = (0..n).map(|i| ((i + k) % 7) as f64 - 3.0).collect();
            let r = gmres_block(&a, &b, 1, &lu, &settings, None, None).unwrap();
            assert!(r.x.iter().all(|v| v.is_finite()));
        },
    );
}
