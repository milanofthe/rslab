//! Isolated multi-RHS **solve throughput**: the cost that dominates a converged
//! GMRES (≈30 triangular solves per RHS at 50k DOFs). Times `niter` repeats of a
//! block solve `solve_lu_many(s)` against `niter` repeats of `s` separate
//! `solve_lu` calls, on a real MoM LU factor. This isolates the factor-load
//! amortization (each `L`/`U` value touched once for all `s` columns) from the
//! Krylov orthogonalization / convergence noise of the full GMRES bench.
//!
//! Run: `cargo bench --bench solve_many`  (`RLA_BLOCK_S`, `RLA_NITER`).

use std::time::Instant;

use num_complex::Complex;
use rslab::prelude::*;
use rslab::{factor_general_lu, solve_lu, solve_lu_many, SolverSettings};

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
type C = Complex<f64>;

fn run(path: &std::path::Path, s: usize, niter: usize) {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(mtx) = parse_mtx_complex_general(&contents, &name) else {
        return;
    };
    drop(contents);
    let Ok(a) = mtx.to_general_csc() else {
        return;
    };
    let n = a.n;
    let lu = factor_general_lu(&a, &SolverSettings::preconditioner(1e-10)).unwrap();
    let fill = lu.factor_nnz();

    // Row-major n×s block of right-hand sides.
    let brow: Vec<C> = (0..n * s)
        .map(|t| Complex::new((t % 7) as f64 - 3.0, (t % 5) as f64 - 2.0))
        .collect();
    // Column-major copies for the per-RHS loop.
    let bcol: Vec<Vec<C>> = (0..s)
        .map(|c| (0..n).map(|i| brow[i * s + c]).collect())
        .collect();

    // s separate single-RHS solves, niter times.
    let t = Instant::now();
    let mut sink = C::default();
    for _ in 0..niter {
        for col in &bcol {
            let x = solve_lu(&lu, col).unwrap();
            sink += x[0];
        }
    }
    let loop_ms = t.elapsed().as_secs_f64() * 1e3;

    // One block solve, niter times.
    let t = Instant::now();
    for _ in 0..niter {
        let x = solve_lu_many(&lu, &brow, s).unwrap();
        sink += x[0];
    }
    let blk_ms = t.elapsed().as_secs_f64() * 1e3;

    std::hint::black_box(sink);
    println!(
        "{name:30} n={n:6} fill={fill:9} s={s:3} niter={niter:3}  loop {loop_ms:8.1} ms   block {blk_ms:8.1} ms   speedup {:4.2}x",
        loop_ms / blk_ms,
    );
}

fn main() {
    let s: usize = std::env::var("RLA_BLOCK_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let niter: usize = std::env::var("RLA_NITER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let mut files: Vec<_> = match std::fs::read_dir(DIR) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect(),
        Err(e) => {
            println!("cannot read {DIR}: {e}");
            return;
        }
    };
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    let filter = std::env::var("RLA_DIAG_FILTER").unwrap_or_default();
    println!(
        "Multi-RHS solve throughput: s separate solves vs one block solve  [s={s} niter={niter}]\n"
    );
    for f in &files {
        if filter.is_empty() || f.file_name().unwrap().to_string_lossy().contains(&filter) {
            run(f, s, niter);
        }
    }
}
