//! Node-parallelism profiler: factor a few big-front matrices at 1 vs N threads
//! and report the wall-clock speedup plus (with `RLA_PROFILE=1`) the serial-panel
//! (getf2 Bunch-Kaufman) vs deferred-GEMM (Schur) CPU-ms split. The split bounds
//! the achievable speedup (Amdahl): if the serial panel is a large fraction, no
//! amount of GEMM parallelism helps; if it is small but the wall-clock still
//! doesn't scale, the trailing GEMM itself is the bottleneck.
//!
//! Run: `RLA_PROFILE=1 cargo bench --bench front_profile --features matgen`

use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::{stencil, structured};
use rslab::{CscMatrix, FactorMethod, FactorOptions, LdltSymbolic};

type C = Complex<f64>;

fn factor_ms(sym: &LdltSymbolic, a: &CscMatrix<C>, threads: usize) -> f64 {
    let opts = FactorOptions::default()
        .with_method(FactorMethod::LeftLooking)
        .with_threads(threads);
    let t = Instant::now();
    let f = sym.factor(a, &opts).expect("factor");
    let ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(f.factor_nnz());
    ms
}

fn run(name: &str, a: CscMatrix<C>) {
    let sym = LdltSymbolic::analyze(&a).expect("analyze");
    let nrow_max = sym.front_dims().iter().map(|&(_, r)| r).max().unwrap_or(0);
    eprintln!("\n=== {name}  n={}  nnz={}  front_nrow_max={}", a.n, a.values.len(), nrow_max);
    let _ = factor_ms(&sym, &a, 1); // warm up
    let mut t1 = 0.0;
    for &t in &[1usize, 2, 4, 8, 12] {
        // best of two to reduce noise
        let ms = factor_ms(&sym, &a, t).min(factor_ms(&sym, &a, t));
        if t == 1 {
            t1 = ms;
            eprintln!("  threads={:>2}  {:>9.1} ms", t, ms);
        } else {
            eprintln!("  threads={:>2}  {:>9.1} ms  speedup {:.2}x", t, ms, t1 / ms);
        }
    }
}

fn main() {
    let c = |re, im| Complex::new(re, im);
    // Big-front, high flop-concentration (3D): the node-parallelism regime.
    run("poisson3d_40", stencil::laplacian::<C>(&[40, 40, 40], &stencil::StencilOpts::default()));
    run("helmholtz3d_30", stencil::helmholtz(&[30, 30, 30], c(2.0, 0.1), &stencil::StencilOpts::default()));
    // Wide-tree, low concentration (2D): tree-parallel regime, for contrast.
    run("poisson2d_360", stencil::laplacian::<C>(&[360, 360], &stencil::StencilOpts::default()));
    // Thin (banded): should not scale at all.
    run("banded_40000", structured::banded::<C>(40000, 40, 1.0, 1));
}
