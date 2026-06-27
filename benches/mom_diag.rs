//! Factorization-cost diagnostic for the real MoM matrices — the data that
//! decides where the PARDISO performance gap lives.
//!
//! For each `*.mtx` in `rapidmom/precond_matrices` it reports, per matrix:
//!   * analyze (symbolic) vs numeric-factor wall time — the phase split,
//!   * factor fill `nnz(L)+nnz(U)` and the growth ratio over `nnz(A)`,
//!   * an estimated factorization flop count,
//!   * a front-size histogram with the **flop share per bucket** — the key
//!     signal: if most flops sit in small fronts the kernel is BLAS-2-bound and
//!     amalgamation (not a faster kernel) is the lever.
//!
//! Compare fill against PARDISO `iparm[17]` and flops against `iparm[18]`
//! (printed by the companion `pardiso_mom.py`).
//!
//! Run: `cargo bench --bench mom_diag`.

use std::time::Instant;

use rla::prelude::*;
use rla::{analyze_general, factor_general_lu_numeric};
use num_complex::Complex;

type C = Complex<f64>;

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";

/// Estimated factorization flops to eliminate `ncol` pivots from an `nrow`-tall
/// front: the dominant trailing rank-1 updates, ~8 real flops per complex
/// multiply-add. `Σ_{k=0}^{ncol-1} 8·(nrow-1-k)²`.
fn front_flops(ncol: usize, nrow: usize) -> f64 {
    let mut f = 0.0f64;
    for k in 0..ncol {
        let m = (nrow - 1 - k) as f64;
        f += 8.0 * m * m;
    }
    f
}

// Front-height buckets (BLAS-2 territory at the small end, BLAS-3 at the large).
const BUCKETS: [usize; 6] = [8, 32, 128, 512, 2048, usize::MAX];
const BUCKET_LABEL: [&str; 6] = ["≤8", "≤32", "≤128", "≤512", "≤2048", ">2048"];

fn bucket_of(nrow: usize) -> usize {
    BUCKETS.iter().position(|&b| nrow <= b).unwrap_or(5)
}

fn diag_file(path: &std::path::Path) {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("{name}: read error {e}");
            return;
        }
    };
    let mtx = match parse_mtx_complex_general(&contents, &name) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}: parse error {e}");
            return;
        }
    };
    drop(contents);
    let a = match mtx.to_general_csc() {
        Ok(a) => a,
        Err(e) => {
            println!("{name}: build error {e}");
            return;
        }
    };
    let n = a.n;
    let nnz_a = a.nnz();

    // Phase split: analyze (symbolic) vs numeric factor.
    let t = Instant::now();
    let sym = match analyze_general(&a) {
        Ok(s) => s,
        Err(e) => {
            println!("{name}: analyze error {e:?}");
            return;
        }
    };
    let analyze_ms = t.elapsed().as_secs_f64() * 1e3;

    let opts = FactorOptions::preconditioner(1e-10);
    let t = Instant::now();
    let f = match factor_general_lu_numeric(&sym, &a, &opts) {
        Ok(f) => f,
        Err(e) => {
            println!("{name}: factor error {e:?}");
            return;
        }
    };
    let factor_ms = t.elapsed().as_secs_f64() * 1e3;

    let fill = f.factor_nnz();
    let dims = sym.front_dims();

    // Per-bucket front count + flop share.
    let mut bucket_cnt = [0usize; 6];
    let mut bucket_flop = [0.0f64; 6];
    let mut total_flop = 0.0f64;
    let mut max_nrow = 0usize;
    let mut sum_ncol = 0usize;
    let mut max_ncol = 0usize;
    // Flop-weighted mean front width = effective Schur-GEMM rank — the metric
    // that governs compute-bound vs memory-bound throughput.
    let mut fw_ncol_num = 0.0f64;
    for &(ncol, nrow) in &dims {
        let fl = front_flops(ncol, nrow);
        let b = bucket_of(nrow);
        bucket_cnt[b] += 1;
        bucket_flop[b] += fl;
        total_flop += fl;
        max_nrow = max_nrow.max(nrow);
        sum_ncol += ncol;
        max_ncol = max_ncol.max(ncol);
        fw_ncol_num += fl * ncol as f64;
    }
    let fw_ncol = if total_flop > 0.0 { fw_ncol_num / total_flop } else { 0.0 };

    println!("=== {name}  n={n}  nnz(A)={nnz_a} ===");
    println!(
        "  phases:  analyze {analyze_ms:8.1} ms   factor {factor_ms:8.1} ms   levels {}",
        sym.n_levels()
    );
    println!(
        "  fill:    nnz(L+U)={fill:>10}   growth {:.1}× over nnz(A)   ({:.0} MB f64-complex)",
        fill as f64 / nnz_a as f64,
        fill as f64 * 16.0 / 1e6,
    );
    println!(
        "  flops:   est {:.2} Gflop   {:.1} Gflop/s   fronts={}   max nrow={max_nrow}",
        total_flop / 1e9,
        total_flop / 1e9 / (factor_ms / 1e3),
        dims.len(),
    );
    println!(
        "  width:   mean ncol={:.1}   flop-weighted ncol={fw_ncol:.0}   max ncol={max_ncol}   (GEMM rank)",
        sum_ncol as f64 / dims.len() as f64,
    );
    println!("  front-size histogram (flop share):");
    for b in 0..6 {
        if bucket_cnt[b] == 0 {
            continue;
        }
        let share = if total_flop > 0.0 {
            100.0 * bucket_flop[b] / total_flop
        } else {
            0.0
        };
        let bar = "#".repeat((share / 2.5).round() as usize);
        println!(
            "    nrow {:>6}: {:>7} fronts  {:5.1}% flops  {bar}",
            BUCKET_LABEL[b], bucket_cnt[b], share,
        );
    }
    println!();
}

fn main() {
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
    // Optional substring filter for fast single-matrix iteration during the
    // amalgamation sweep.
    let filter = std::env::var("RLA_DIAG_FILTER").unwrap_or_default();
    println!("MoM factorization-cost diagnostic (RLA unsymmetric LU)\n");
    for f in &files {
        if filter.is_empty() || f.file_name().unwrap().to_string_lossy().contains(&filter) {
            diag_file(f);
        }
    }
}
