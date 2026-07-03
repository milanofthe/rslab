//! **GCRO-DR Krylov subspace recycling** on a sequence of related solves (issue #5).
//!
//! The benchmark-scale mirror of `gmres_recycled_cross_solve_beats_warm_and_cold`
//! (src/numeric/iterative.rs): a *stagnation spectrum* (a tight cluster of tiny
//! eigenvalues below a spread bulk) is exactly the regime where restarted FGMRES
//! keeps re-discovering and discarding the near-invariant subspace at every restart.
//! We build a slowly varying operator (base spectrum + a per-step diagonal
//! perturbation) and a rotating right-hand side, then solve the sequence three ways:
//!
//!   * **cold**   - every solve from `x0 = 0`, plain FGMRES(m);
//!   * **warm**   - each solve seeded with the previous solution;
//!   * **gcrodr(k)** - warm start *and* a `Recycle` handle carried across the
//!     sequence, deflating a `k`-dim harmonic-Ritz subspace within *and* across solves.
//!
//! Step 0 (empty handle) isolates the *within-solve* deflation; the cumulative
//! totals show the *cross-solve* recycling win. Sweeps `k`.
//!
//! Run: `cargo bench --bench recycle_study`
//!   env: RLA_N (spectrum size, default 20000), RLA_NSMALL (cluster size, default 8),
//!        RLA_STEPS (sequence length, default 8), RLA_RESTART (default 20),
//!        RLA_JSON=<path> to emit JSONL.

use std::time::Instant;

use num_complex::Complex;
use rslab::{gmres, gmres_recycled, GeneralCsc, NoPreconditioner, Recycle};

type C = Complex<f64>;

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

/// `n_small` tiny eigenvalues near the origin below a spread cluster on `[1,11]`
/// (mirrors the unit-test `stagnation_spectrum`): the tight low cluster is what
/// makes restarted GMRES stagnate.
fn stagnation_spectrum(n: usize, n_small: usize) -> Vec<C> {
    (0..n)
        .map(|i| {
            if i < n_small {
                Complex::new(0.01 + 0.004 * i as f64, 0.002 * i as f64)
            } else {
                let t = (i - n_small) as f64 / (n - n_small) as f64;
                Complex::new(1.0 + 10.0 * t, 0.3 * (i as f64).sin())
            }
        })
        .collect()
}

/// Complex diagonal operator with a prescribed spectrum.
fn diag_op(eigs: &[C]) -> GeneralCsc<C> {
    let n = eigs.len();
    let idx: Vec<usize> = (0..n).collect();
    GeneralCsc::<C>::from_triplets(n, &idx, &idx, eigs).unwrap()
}

fn main() {
    let n: usize = std::env::var("RLA_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20000);
    let n_small: usize = std::env::var("RLA_NSMALL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let steps: usize = std::env::var("RLA_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let restart: usize = std::env::var("RLA_RESTART")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let (tol, maxit) = (1e-9, 200_000);
    let ks = [5usize, 10, 20];

    let base = stagnation_spectrum(n, n_small);
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.15 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(0.4 * (i as f64).sin(), 1.0)).collect();

    // Slowly varying operator: base spectrum + eps_k on each diagonal entry.
    let ak = |kk: usize| -> GeneralCsc<C> {
        let eps = 2e-3 * kk as f64;
        let eigs: Vec<C> = base
            .iter()
            .enumerate()
            .map(|(i, &e)| e + c(eps * (1.0 + 0.05 * i as f64), 0.0))
            .collect();
        diag_op(&eigs)
    };
    let bk = |kk: usize| -> Vec<C> {
        let th = 0.02 * kk as f64;
        let (ct, st) = (th.cos(), th.sin());
        (0..n)
            .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
            .collect()
    };

    println!(
        "GCRO-DR recycling  [n={n} cluster={n_small} steps={steps} restart={restart}]\n\
         method       k   step   iters   cum_iters   ms       cum_ms"
    );

    // --- Cold: every solve from x0 = 0. ---
    let mut cum_iters = 0usize;
    let mut cum_ms = 0.0f64;
    for kk in 0..steps {
        let a = ak(kk);
        let t = Instant::now();
        let r = gmres(&a, &bk(kk), &NoPreconditioner, tol, maxit, restart, None).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        cum_iters += r.iters;
        cum_ms += ms;
        println!(
            "cold         -   {kk:4}   {:5}   {cum_iters:9}   {ms:7.1}  {cum_ms:8.1}",
            r.iters
        );
        emit(&format!(
            "\"method\":\"cold\",\"k\":0,\"n\":{n},\"step\":{kk},\"iters\":{},\
             \"cum_iters\":{cum_iters},\"ms\":{ms:.3},\"cum_ms\":{cum_ms:.3},\"res\":{:e}",
            r.iters, r.final_res
        ));
    }

    // --- Warm: seed each solve with the previous solution. ---
    cum_iters = 0;
    cum_ms = 0.0;
    let mut prev: Option<Vec<C>> = None;
    for kk in 0..steps {
        let a = ak(kk);
        let t = Instant::now();
        let r = gmres(
            &a,
            &bk(kk),
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            prev.as_deref(),
        )
        .unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        cum_iters += r.iters;
        cum_ms += ms;
        println!(
            "warm         -   {kk:4}   {:5}   {cum_iters:9}   {ms:7.1}  {cum_ms:8.1}",
            r.iters
        );
        emit(&format!(
            "\"method\":\"warm\",\"k\":0,\"n\":{n},\"step\":{kk},\"iters\":{},\
             \"cum_iters\":{cum_iters},\"ms\":{ms:.3},\"cum_ms\":{cum_ms:.3},\"res\":{:e}",
            r.iters, r.final_res
        ));
        prev = Some(r.x);
    }

    // --- GCRO-DR(k): warm start + a recycle handle carried across the sequence. ---
    for &k in &ks {
        cum_iters = 0;
        cum_ms = 0.0;
        let mut rec = Recycle::<C>::new(k);
        let mut prevr: Option<Vec<C>> = None;
        for kk in 0..steps {
            let a = ak(kk);
            let t = Instant::now();
            let r = gmres_recycled(
                &a,
                &bk(kk),
                &NoPreconditioner,
                tol,
                maxit,
                restart,
                prevr.as_deref(),
                &mut rec,
            )
            .unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            cum_iters += r.iters;
            cum_ms += ms;
            println!(
                "gcrodr(k={k:2}) {k:3}   {kk:4}   {:5}   {cum_iters:9}   {ms:7.1}  {cum_ms:8.1}",
                r.iters
            );
            emit(&format!(
                "\"method\":\"gcrodr\",\"k\":{k},\"n\":{n},\"step\":{kk},\"iters\":{},\
                 \"cum_iters\":{cum_iters},\"ms\":{ms:.3},\"cum_ms\":{cum_ms:.3},\"res\":{:e}",
                r.iters, r.final_res
            ));
            prevr = Some(r.x);
        }
    }
}
