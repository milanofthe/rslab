//! Multi-RHS payoff: solving `s` right-hand sides of a real complex MoM system
//! via **one block GMRES** vs **`s` separate GMRES** runs, both right-
//! preconditioned by the same RLA LU factor. The block path batches the operator
//! matvec and the preconditioner triangular solve across all `s` columns (BLAS-3
//! reuse: each matrix / factor value touched once for all RHS), so it should beat
//! the loop on wall time while producing the same per-column solution.
//!
//! Run: `cargo bench --bench block_gmres` (set `RLA_BLOCK_S=32` to vary the count).

use std::time::Instant;

use num_complex::Complex;
use rla::prelude::*;
use rla::{factor_general_lu, gmres, gmres_block, FactorOptions};

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
type C = Complex<f64>;

fn run(path: &std::path::Path, s: usize) {
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
    // s distinct right-hand sides, column-major n×s.
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = Complex::new(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }

    let lu = factor_general_lu(&a, &FactorOptions::preconditioner(1e-10)).unwrap();
    let (tol, maxit, restart) = (1e-8, 200, 50);

    // s separate single-RHS GMRES runs.
    let t = Instant::now();
    let mut xloop = vec![C::default(); n * s];
    let mut loop_iters = 0;
    for k in 0..s {
        let r = gmres(&a, &bblk[k * n..k * n + n], &lu, tol, maxit, restart).unwrap();
        xloop[k * n..k * n + n].copy_from_slice(&r.x);
        loop_iters = loop_iters.max(r.iters);
    }
    let loop_ms = t.elapsed().as_secs_f64() * 1e3;

    // One block GMRES over all s RHS.
    let t = Instant::now();
    let blk = gmres_block(&a, &bblk, s, &lu, tol, maxit, restart).unwrap();
    let blk_ms = t.elapsed().as_secs_f64() * 1e3;

    // Max per-column solution difference (must agree to the solve tolerance).
    let diff = (0..n * s).map(|i| (blk.x[i] - xloop[i]).norm()).fold(0.0, f64::max);

    println!(
        "{name:30} n={n:6} s={s:3}  loop {loop_ms:8.1} ms (iters {loop_iters})   block {blk_ms:8.1} ms (iters {})   speedup {:4.2}x   conv {}  diff {:.1e}",
        blk.iters,
        loop_ms / blk_ms,
        blk.converged,
        diff,
    );
}

fn main() {
    let s: usize = std::env::var("RLA_BLOCK_S").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
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
    println!("Multi-RHS GMRES: s separate runs vs one block run  [s={s}]\n");
    for f in files.iter().take(4) {
        run(f, s);
    }
}
