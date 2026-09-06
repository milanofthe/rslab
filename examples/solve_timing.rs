//! Time the LDL^T solves (single and blocked right-hand sides) on a 3D grid.
//! `cargo run --release --example solve_timing -- [m] [nrhs]`.
use rslab::{CscMatrix, LdltSolver, OrderingMethod, SolverSettings};
use std::time::Instant;

fn grid3d(m: usize) -> CscMatrix<f64> {
    let n = m * m * m;
    let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        let (x, y, z) = (j % m, (j / m) % m, j / (m * m));
        row_idx.push(j);
        values.push(6.0);
        if x + 1 < m {
            row_idx.push(j + 1);
            values.push(-1.0);
        }
        if y + 1 < m {
            row_idx.push(j + m);
            values.push(-1.0);
        }
        if z + 1 < m {
            row_idx.push(j + m * m);
            values.push(-1.0);
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n,
        col_ptr,
        row_idx,
        values,
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

fn main() {
    if std::env::var("QOS").is_ok() {
        let n: usize = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .spawn_handler(|t| {
                std::thread::Builder::new().spawn(move || {
                    #[cfg(target_os = "macos")]
                    unsafe {
                        pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE
                    }
                    t.run()
                })?;
                Ok(())
            })
            .build_global()
            .unwrap();
    }
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let nrhs: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reps: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let a = grid3d(m);
    let n = a.n;
    let t = Instant::now();
    let s = LdltSolver::factor_with(
        &a,
        &SolverSettings::default().with_ordering(OrderingMethod::MetisND),
    )
    .unwrap();
    println!(
        "n={n} nnz(L)={} factor {:.2} s  threads={}",
        s.factor_nnz(),
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let _x = s.solve(&b).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "solve   {:>8.2} ms  {:>6.2} MDOF/s  {:>6.0} Mnnz/s",
        best * 1e3,
        n as f64 / best / 1e6,
        s.factor_nnz() as f64 / best / 1e6
    );
    let bb: Vec<f64> = (0..n * nrhs).map(|k| (k % 11) as f64 - 5.0).collect();
    let mut best = f64::INFINITY;
    for _ in 0..reps.div_ceil(2) {
        let t = Instant::now();
        let _x = s.solve_many(&bb, nrhs).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!(
        "solve{nrhs:<3} {:>8.2} ms  {:>6.2} MDOF/s  {:>6.0} Mnnz/s",
        best * 1e3,
        (n * nrhs) as f64 / best / 1e6,
        (s.factor_nnz() * nrhs) as f64 / best / 1e6
    );
}
