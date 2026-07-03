//! **Adaptive GMRES restart under a memory budget** (issue #12).
//!
//! The single-RHS FGMRES basis is the `V`+`Z` pair, `2 * n * (restart+1)` scalars,
//! allocated up front - so `restart` sets the memory floor regardless of how few
//! iterations actually run. The Python binding therefore caps an unspecified
//! `restart` so the basis stays under a 1 GiB budget (an explicit `restart` is always
//! honoured). This bench measures the memory <-> iteration trade-off the policy
//! navigates: for a preconditioned convection-diffusion system it sweeps fixed
//! restart lengths (fewer restart columns => less basis memory but more iterations)
//! across problem sizes, and records the memory each choice implies alongside the
//! restart the adaptive policy would pick. The plot overlays the analytic policy
//! curve, which rides at the max restart until the basis would exceed the budget and
//! then declines to hold memory flat.
//!
//! Run: `cargo bench --features matgen --bench adaptive_restart`
//!   env: RLA_DROPTOL (default 0.05), RLA_JSON=<path> to emit JSONL.

use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::fem::{convection_diffusion, Flow};
use rslab::{factor_general_lu, gmres, SolverSettings};

type C = Complex<f64>;

// Mirror of the Python binding's `adaptive_restart` (bases = 2 for single-RHS FGMRES).
const BUDGET_BYTES: usize = 1 << 30; // 1 GiB
const RESTART_MIN: usize = 20;
const RESTART_MAX: usize = 80;
fn adaptive_restart(n: usize, scalar_bytes: usize, bases: usize) -> usize {
    let per_layer = n.saturating_mul(scalar_bytes).saturating_mul(bases);
    if per_layer == 0 {
        return RESTART_MAX;
    }
    let cap = (BUDGET_BYTES / per_layer).saturating_sub(1);
    cap.clamp(RESTART_MIN, RESTART_MAX)
}

fn emit(fields: &str) {
    let Ok(path) = std::env::var("RLA_JSON") else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{{{fields}}}");
    }
}

fn main() {
    let droptol: f64 = std::env::var("RLA_DROPTOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let bytes = std::mem::size_of::<C>();
    let tol = 1e-8;
    let dims = [80usize, 130, 180];
    let restarts = [10usize, 20, 40, 80];

    println!(
        "Adaptive GMRES restart under a 1 GiB basis budget  [drop_tol={droptol:e}]\n\
         n        restart   basis(MB)   iters   wct(ms)     adaptive"
    );

    for &d in &dims {
        let a = convection_diffusion::<C>(&[d, d], 0.02, Flow::Rotating, true);
        let n = a.n;
        let b: Vec<C> = (0..n)
            .map(|i| Complex::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0))
            .collect();
        let opts = SolverSettings::preconditioner(1e-10).with_drop_tol(droptol);
        let Ok(lu) = factor_general_lu(&a, &opts) else {
            eprintln!("factor failed at n={n}");
            continue;
        };
        let adaptive = adaptive_restart(n, bytes, 2);
        for &restart in &restarts {
            let basis_mb = (2 * n * (restart + 1) * bytes) as f64 / (1024.0 * 1024.0);
            let t = Instant::now();
            let Ok(res) = gmres(&a, &b, &lu, tol, 100_000, restart, None) else {
                eprintln!("gmres failed at n={n} restart={restart}");
                continue;
            };
            let wct = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "{n:6}   {restart:5}     {basis_mb:8.2}   {:5}   {wct:8.1}    {adaptive:5}",
                res.iters
            );
            emit(&format!(
                "\"n\":{n},\"restart\":{restart},\"basis_mb\":{basis_mb:.4},\"iters\":{},\
                 \"wct_ms\":{wct:.3},\"adaptive\":{adaptive},\"res\":{:e}",
                res.iters, res.final_res
            ));
        }
    }
}
