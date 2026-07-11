//! Factor-throughput probe on the PARDISO-reference case (helmholtz 40^3,
//! complex, the h2h matrix where the factor gap is ~8.8x): ordering flop
//! comparison, a config sweep, and warm best-of-3 timings in ONE process
//! (bench_suite is a cold single shot - too noisy for kernel A/B). Run with
//! `RLA_PROFILE=1` for the [RLA_LDLT_LL] / [RLA_LDLT_CMOD_DIST] phase splits.
//! `cargo run --release --features matgen --bin factor_probe_helmholtz [threads...]`
//! See dev/research/factor-throughput-2026-07.md for the measurement log.

#[cfg(not(feature = "matgen"))]
fn main() {
    eprintln!("build with --features matgen");
}

#[cfg(feature = "matgen")]
use std::time::Instant;

#[cfg(feature = "matgen")]
use num_complex::Complex;
#[cfg(feature = "matgen")]
use rslab::matgen::stencil::{helmholtz, StencilOpts};
#[cfg(feature = "matgen")]
use rslab::{LdltSymbolic, SolverSettings};

#[cfg(feature = "matgen")]
fn main() {
    let a = helmholtz(
        &[40, 40, 40],
        Complex::new(0.02, 0.01),
        &StencilOpts::default(),
    );
    eprintln!("n={} nnz={}", a.n, a.nnz());
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let threads = if args.is_empty() { vec![8usize] } else { args };

    let sym = LdltSymbolic::analyze(&a).unwrap();
    let est = sym.estimate_memory::<Complex<f64>>();
    eprintln!(
        "symbolic AUTO(AMF): fill {} geom-flops {:.3e} crit-path {:.3e} tree-width {}",
        sym.symbolic_factor_nnz(),
        est.factor_flops as f64,
        est.critical_path_flops as f64,
        est.max_tree_width
    );
    for (name, ord) in [
        ("MetisND", rslab::OrderingMethod::MetisND),
        ("Amd", rslab::OrderingMethod::Amd),
        ("Rcm", rslab::OrderingMethod::Rcm),
    ] {
        let o = SolverSettings::default().with_ordering(ord);
        match LdltSymbolic::analyze_with(&a, &o) {
            Ok(s2) => {
                let e2 = s2.estimate_memory::<Complex<f64>>();
                eprintln!(
                    "symbolic {name}: fill {} geom-flops {:.3e} crit-path {:.3e} tree-width {}",
                    s2.symbolic_factor_nnz(),
                    e2.factor_flops as f64,
                    e2.critical_path_flops as f64,
                    e2.max_tree_width
                );
            }
            Err(e) => eprintln!("symbolic {name}: FAILED {e}"),
        }
    }

    let t8 = *threads.first().unwrap_or(&8);
    let gt = |sg: usize, pg: usize, pc: usize| rslab::GemmThresholds {
        scalar_gate: sg,
        par_gemm: pg,
        par_cdiv: pc,
    };
    let configs: Vec<(&str, SolverSettings)> = vec![
        ("default        ", SolverSettings::default()),
        (
            "nb32           ",
            SolverSettings::default().with_panel_nb(32),
        ),
        (
            "nb128          ",
            SolverSettings::default().with_panel_nb(128),
        ),
        (
            "par_gemm 2e5   ",
            SolverSettings::default().with_gemm_thresholds(gt(4096, 200_000, 8_000_000)),
        ),
        (
            "par_cdiv 2e6   ",
            SolverSettings::default().with_gemm_thresholds(gt(4096, 1_000_000, 2_000_000)),
        ),
        (
            "nb32+both thr  ",
            SolverSettings::default()
                .with_panel_nb(32)
                .with_gemm_thresholds(gt(4096, 200_000, 2_000_000)),
        ),
        (
            "MetisND        ",
            SolverSettings::default().with_ordering(rslab::OrderingMethod::MetisND),
        ),
        (
            "Multifrontal   ",
            SolverSettings::default().with_method(rslab::FactorMethod::Multifrontal),
        ),
        (
            "MF + MetisND   ",
            SolverSettings::default()
                .with_method(rslab::FactorMethod::Multifrontal)
                .with_ordering(rslab::OrderingMethod::MetisND),
        ),
    ];
    for (name, base) in configs {
        let opts = base.clone().with_threads(t8);
        // analyze-time knobs may differ (ordering) - use matching analysis.
        let symc = LdltSymbolic::analyze_with(&a, &opts).unwrap();
        let estc = symc.estimate_memory::<Complex<f64>>();
        let _ = symc.factor(&a, &opts).unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let f = symc.factor(&a, &opts).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            best = best.min(ms);
            std::hint::black_box(f.factor_nnz());
        }
        let gf = estc.factor_flops as f64 / (best / 1e3) / 1e9;
        eprintln!("cfg {name} @{t8}  best {best:8.1} ms   {gf:7.2} geom-Gflop/s");
    }
}
