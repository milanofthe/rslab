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

    // The heuristic default pick (`LdltSolver::tuned` = what `factor()` runs):
    // ordering via the exact ND bakeoff, threads from the cached calibration if
    // the install diagnosis has run.
    let t0 = Instant::now();
    let (sym_h, s_h) = rslab::LdltSolver::<Complex<f64>>::tuned(&a).unwrap();
    let ana_ms = t0.elapsed().as_secs_f64() * 1e3;
    let est_h = sym_h.estimate_memory::<Complex<f64>>();
    let _ = sym_h.factor(&a, &s_h).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let f = sym_h.factor(&a, &s_h).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(f.factor_nnz());
    }
    eprintln!(
        "heuristic tuned(): {:?} threads {:?}  ana {ana_ms:.0} ms  fac best {best:.1} ms  fill {}  flops {:.3e}",
        s_h.ordering,
        s_h.threads,
        sym_h.symbolic_factor_nnz(),
        est_h.factor_flops as f64
    );

    // Mixed precision (issue #18): c32 factor + certified IR to c64. The
    // numeric-factor timing excludes analysis (same protocol as the lines
    // above): analyze the cast matrix once, warm best-of-3 on sym.factor.
    let a32 = rslab::CscMatrix::<num_complex::Complex<f32>> {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a
            .values
            .iter()
            .map(|v| num_complex::Complex::new(v.re as f32, v.im as f32))
            .collect(),
    };
    let (sym32, s32) = rslab::LdltSolver::<num_complex::Complex<f32>>::tuned(&a32).unwrap();
    let _ = sym32.factor(&a32, &s32).unwrap();
    let mut best_mf = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        let f = sym32.factor(&a32, &s32).unwrap();
        best_mf = best_mf.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(f.factor_nnz());
    }
    let m = rslab::MixedLdltSolver::<Complex<f64>>::factor(&a).unwrap();
    let b: Vec<Complex<f64>> = (0..a.n)
        .map(|i| Complex::new(((i % 13) as f64) - 6.0, ((i % 7) as f64) - 3.0))
        .collect();
    let t0 = Instant::now();
    let (x, info) = m.solve(&a, &b).unwrap();
    let slv_ms = t0.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&x);
    // Reference c64 solve time on the heuristic factor.
    let f_h = sym_h.factor(&a, &s_h).unwrap();
    let t0 = Instant::now();
    let xd = f_h.solve(&b).unwrap();
    let slv64_ms = t0.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&xd);
    eprintln!(
        "mixed c32+IR: fac best {best_mf:.1} ms  solve {slv_ms:.1} ms (c64 solve {slv64_ms:.1} ms)  ir {} gmres {} be {:.1e} certified {}",
        info.ir_iters, info.gmres_iters, info.backward_error, info.certified
    );
}
