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
use rslab::{set_panel_nb, CscMatrix, FactorMethod, FactorOptions, LdltSymbolic};

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
    // NB sensitivity: for each panel width report the t1 work split (via
    // RLA_PROFILE) and the 1->12 thread scaling.
    for &nb in &[32usize, 64, 96, 128] {
        set_panel_nb(nb);
        let _ = factor_ms(&sym, &a, 1); // warm up + emit the t1 getf2/schur split
        let t1 = factor_ms(&sym, &a, 1).min(factor_ms(&sym, &a, 1));
        let t12 = factor_ms(&sym, &a, 12).min(factor_ms(&sym, &a, 12));
        eprintln!(
            "  NB={:>3}  t1={:>9.1}ms  t12={:>9.1}ms  speedup {:.2}x",
            nb, t1, t12, t1 / t12
        );
    }
    set_panel_nb(64);
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
