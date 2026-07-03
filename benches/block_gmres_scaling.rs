//! **Strong-scaling** of block GMRES orthogonalization (issue #3, item 1).
//!
//! The block operator matvec and the block preconditioner solve were already
//! BLAS-3 and parallel; the orthogonalization was the serial, latency-bound piece
//! left behind (per-RHS MGS + DGKS). Block-CGS2 turns it into panel-wide sweeps
//! parallelized over the vector dimension. This bench isolates that: one factor,
//! then the **same** block GMRES solve timed in rayon pools of 1..P threads, so
//! the reported speedup/efficiency is the orthogonalization + matvec scaling.
//!
//! Self-contained (no external matrices): a complex convection--diffusion grid
//! (rotating flow, upwind) gives an unsymmetric operator; a drop-tolerance
//! incomplete LU keeps it in the iterative regime where per-iteration cost - and
//! thus orthogonalization - dominates.
//!
//! Run: `cargo bench --bench block_gmres_scaling`
//!   env: `RLA_DIM=180` grid side (n = DIM²), `RLA_BLOCK_S=5` RHS count,
//!        `RLA_DROPTOL=0.01` preconditioner drop tolerance.
//!   mode: `RLA_RHS_SWEEP=1` fixes the thread count (`RLA_THREADS`, default = all
//!         cores) and sweeps the RHS count `s` instead, reporting **time per RHS** -
//!         the "I already have the factor, drive many RHS" question: does adding
//!         RHS stay cheap (BLAS-3 reuse + parallel ortho) or grow linearly.

use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::fem::{convection_diffusion, Flow};
use rslab::{factor_general_lu, gmres_block, SolverSettings};

type C = Complex<f64>;

fn build_rhs(n: usize, s: usize) -> Vec<C> {
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] =
                Complex::new(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    bblk
}

/// Append one JSONL record to the file named by `RLA_JSON`, tagged with the
/// `RLA_VARIANT` label (e.g. `bcgs2` / `mgs`), so the plot script can overlay the
/// current build against the committed reference. No-op if `RLA_JSON` is unset.
fn emit(mode: &str, variant: &str, fields: &str) {
    let Ok(path) = std::env::var("RLA_JSON") else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(
            f,
            "{{\"mode\":\"{mode}\",\"variant\":\"{variant}\",{fields}}}"
        );
    }
}

fn main() {
    let dim: usize = std::env::var("RLA_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(180);
    let s: usize = std::env::var("RLA_BLOCK_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let droptol: f64 = std::env::var("RLA_DROPTOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    let variant = std::env::var("RLA_VARIANT").unwrap_or_else(|_| "bcgs2".into());

    let a = convection_diffusion::<C>(&[dim, dim], 0.01, Flow::Rotating, true);
    let n = a.n;

    let mut opts = SolverSettings::preconditioner(1e-10);
    if droptol > 0.0 {
        opts = opts.with_drop_tol(droptol);
    }
    let lu = factor_general_lu(&a, &opts).unwrap();
    let (tol, maxit, restart) = (1e-6, 400, 80);

    let max_p = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    // --- Grid mode: per-RHS cost over the FULL thread ladder × RHS count, so the
    // thread scaling is visible at every RHS size (not just 1-vs-max). ---
    if std::env::var("RLA_GRID").is_ok() {
        let threads = [1usize, 2, 4, 6, 8, 12]
            .iter()
            .copied()
            .filter(|&p| p <= max_p)
            .collect::<Vec<_>>();
        let s_list = [1usize, 4, 16];
        // Prebuild RHS + pools once.
        let rhs: Vec<Vec<C>> = s_list.iter().map(|&s| build_rhs(n, s)).collect();
        let pools: Vec<_> = threads
            .iter()
            .map(|&p| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(p)
                    .build()
                    .unwrap()
            })
            .collect();
        print!("Block GMRES per-RHS(ms) over threads × s  [n={n}  drop_tol={droptol}]\nthreads");
        for &s in &s_list {
            print!("      s={s:<2}       ");
        }
        println!();
        // per-RHS at 1 thread for each s, to report speedup down each column.
        let mut base = vec![0.0f64; s_list.len()];
        for (ti, &p) in threads.iter().enumerate() {
            print!("{p:5}  ");
            for (si, &s) in s_list.iter().enumerate() {
                let bblk = &rhs[si];
                let solve = || gmres_block(&a, bblk, s, &lu, tol, maxit, restart, None).unwrap();
                let _ = pools[ti].install(solve); // warm up
                let mut ms = f64::INFINITY;
                for _ in 0..2 {
                    let t = Instant::now();
                    let _ = pools[ti].install(solve);
                    ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
                }
                let tpr = ms / s as f64;
                if ti == 0 {
                    base[si] = tpr;
                }
                let sp = base[si] / tpr;
                print!("  {tpr:7.1} ({sp:4.2}x)");
                emit(
                    "grid",
                    &variant,
                    &format!("\"n\":{n},\"s\":{s},\"threads\":{p},\"per_rhs_ms\":{tpr:.3},\"speedup\":{sp:.4}"),
                );
            }
            println!();
        }
        return;
    }

    // --- RHS-sweep mode: fix threads, grow s, report per-RHS cost ---
    if std::env::var("RLA_RHS_SWEEP").is_ok() {
        let p: usize = std::env::var("RLA_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(max_p);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(p)
            .build()
            .unwrap();
        println!(
            "Block GMRES RHS scaling  [n={n}  threads={p}  drop_tol={droptol}]\n\
              s    time(ms)   time/RHS(ms)   iters   res"
        );
        for s in [1usize, 2, 4, 8, 16, 32] {
            let bblk = build_rhs(n, s);
            let solve = || gmres_block(&a, &bblk, s, &lu, tol, maxit, restart, None).unwrap();
            let mut res = pool.install(solve); // warm up
            let mut ms = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                res = pool.install(solve);
                ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let maxres = res.final_res.iter().copied().fold(0.0, f64::max);
            let tpr = ms / s as f64;
            println!(
                "{s:3}   {ms:9.1}    {tpr:9.1}     {:5}   {:.1e}",
                res.iters, maxres
            );
            emit(
                "rhs",
                &variant,
                &format!("\"n\":{n},\"s\":{s},\"threads\":{p},\"total_ms\":{ms:.3},\"per_rhs_ms\":{tpr:.3},\"iters\":{}", res.iters),
            );
        }
        return;
    }

    let bblk = build_rhs(n, s);
    let plan: Vec<usize> = [1usize, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24]
        .iter()
        .copied()
        .filter(|&p| p <= max_p)
        .collect();

    // Peak memory is dominated by the block Krylov basis `n·s·(m+1)`; the BCGS2
    // scratch (`proj1/proj2` = m·s, reduction scratch = ⌈n/ORTHO_CHUNK⌉·m·s) is a
    // rounding error next to it. Report both so the split is explicit.
    let bytes = std::mem::size_of::<C>();
    let basis_mb = (n * s * (restart + 1) * bytes) as f64 / (1024.0 * 1024.0);
    let scratch_mb = ((restart * s * 2) + n.div_ceil(2048) * restart * s) * bytes;
    println!(
        "Block GMRES strong scaling  [n={n}  s={s}  drop_tol={droptol}  cores<= {max_p}]\n\
         basis {basis_mb:.0} MB   BCGS2 scratch {:.0} KB   (scratch/basis = {:.3}%)\n\
         threads    time(ms)   speedup   efficiency   iters   res",
        scratch_mb as f64 / 1024.0,
        scratch_mb as f64 / (basis_mb * 1024.0 * 1024.0) * 100.0
    );
    let mut t1 = 0.0f64;
    for (idx, &p) in plan.iter().enumerate() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(p)
            .build()
            .unwrap();
        let solve = || gmres_block(&a, &bblk, s, &lu, tol, maxit, restart, None).unwrap();
        let mut res = pool.install(solve); // warm up (page-in, branch predictors)
                                           // Best of several timed runs: single wall-clock samples are noisy on a
                                           // loaded machine, and the minimum is the least-contended estimate.
        let reps = 3;
        let mut ms = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            res = pool.install(solve);
            ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
        }
        if idx == 0 {
            t1 = ms;
        }
        let speedup = t1 / ms;
        let eff = speedup / p as f64;
        let maxres = res.final_res.iter().copied().fold(0.0, f64::max);
        println!(
            "{p:5}    {ms:9.1}   {speedup:6.2}x   {:8.0}%   {:5}   {:.1e}",
            eff * 100.0,
            res.iters,
            maxres
        );
    }
}
