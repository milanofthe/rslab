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
#[path = "common/grid.rs"]
mod grid;
use grid::grid;
use std::cell::Cell;
use std::sync::Mutex;

thread_local!(static IN_PROGRESS: Cell<usize> = const { Cell::new(0) });

struct InProgress;

impl InProgress {
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

fn laplacian(rows: &[usize], cols: &[usize], shift: f64) -> Vec<f64> {
    (rows.iter().zip(cols))
        .map(|(&r, &c)| if r == c { 4.0 + shift } else { -1.0 })
        .collect()
}

const WORKERS: usize = 4;

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

#[test]
fn a_sweep_of_serial_factorizations_does_not_nest() {
    factor_sweep(SolverSettings::default().with_threads(Threads::Fixed(1)), 1);
}

#[test]
fn a_sweep_of_parallel_factorizations_and_solves_does_not_nest() {
    let mut solver = SolverSettings::default().with_threads(Threads::Fixed(2));
    solver.solve.par_min_work = Some(0);
    factor_sweep(solver, 8);
}

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
