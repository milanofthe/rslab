//! **Incomplete-factor preconditioner + GMRES** trade-off (the `pc` path).
//!
//! An exact sparse factor is a one-shot direct solve; dropping fill below a
//! relative threshold ``drop_tol`` turns it into a memory-light ILU-style
//! preconditioner, and GMRES corrects the inexact factor back to the true
//! solution. Bigger ``drop_tol`` -> less fill (memory) but more GMRES iterations;
//! total wall time has a sweet spot in between. This bench sweeps ``drop_tol`` on
//! one convection-diffusion system and records fill, iterations, and time so the
//! trade-off is a single figure.
//!
//! Run: `cargo bench --features matgen --bench precond_gmres`
//!   env: `RLA_DIM=180` grid side (n = DIM^2), `RLA_JSON=<path>` to emit JSONL.

use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::fem::{convection_diffusion, Flow};
use rslab::{gmres, LuSolver, SolverSettings};

type C = Complex<f64>;

fn emit(fields: &str) {
    let Ok(path) = std::env::var("RLA_JSON") else { return };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{{{fields}}}");
    }
}

fn main() {
    let dim: usize = std::env::var("RLA_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(180);
    let a = convection_diffusion::<C>(&[dim, dim], 0.02, Flow::Rotating, true);
    let n = a.n;
    let bytes = std::mem::size_of::<C>();
    // One fixed right-hand side.
    let b: Vec<C> = (0..n).map(|i| Complex::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0)).collect();
    let (tol, maxit, restart) = (1e-8, 500, 100);

    // drop_tol = 0.0 is the exact factor (one GMRES step); the rest are ILU
    // preconditioners of increasing aggressiveness.
    let taus = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1];
    println!(
        "Preconditioner + GMRES trade-off  [n={n}  conv-diff rotating upwind]\n\
         drop_tol   fill(MB)   fac(ms)   gmres(ms)  total(ms)  iters   res"
    );
    for &tau in &taus {
        let mut opts = SolverSettings::preconditioner(1e-10);
        if tau > 0.0 {
            opts = opts.with_drop_tol(tau);
        }
        let t = Instant::now();
        let Ok(f) = LuSolver::<C>::factor(&a, &opts) else {
            eprintln!("factor failed at drop_tol={tau}");
            continue;
        };
        let fac = t.elapsed().as_secs_f64() * 1e3;
        let fill = f.factor_nnz();
        let t = Instant::now();
        let Ok(kr) = gmres(&a, &b, &f, tol, maxit, restart, None) else {
            eprintln!("gmres failed at drop_tol={tau}");
            continue;
        };
        let slv = t.elapsed().as_secs_f64() * 1e3;
        let fill_mb = (fill * bytes) as f64 / (1024.0 * 1024.0);
        println!(
            "{tau:8.0e}   {fill_mb:7.1}   {fac:7.1}   {slv:8.1}   {:8.1}   {:5}   {:.1e}",
            fac + slv,
            kr.iters,
            kr.final_res
        );
        emit(&format!(
            "\"n\":{n},\"drop_tol\":{tau:e},\"fill\":{fill},\"fill_mb\":{fill_mb:.4},\
             \"fac_ms\":{fac:.3},\"slv_ms\":{slv:.3},\"total_ms\":{:.3},\"iters\":{},\"res\":{:e}",
            fac + slv,
            kr.iters,
            kr.final_res
        ));
    }
}
