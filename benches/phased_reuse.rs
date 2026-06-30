//! Phased-reuse benchmark: the PARDISO-style "analyze once, factor many" workflow
//! that a frequency sweep / Newton iteration exercises (same sparsity pattern,
//! changing values). Measures the analyze phase (fill-reducing ordering +
//! symbolic, value-independent) and the numeric factor separately per matrix, so
//! `phased_reuse.py` can show the amortized cost over K factorizations:
//! analyze-once (`analyze + K*factor`) vs analyze-each (`K*(analyze + factor)`).
//!
//! Helmholtz 3D matrices stand in for an EM/FEM frequency sweep: one pattern,
//! many `k²`. Correctness of reuse is asserted (the analysis factors a second
//! value set and still solves).
//!
//! Run: `cargo bench --bench phased_reuse --features matgen`
#![allow(clippy::needless_range_loop)]

use std::io::Write;
use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::stencil;
use rslab::{CscMatrix, SolverSettings, LdltSymbolic};

type C = Complex<f64>;

fn best_of<F: FnMut() -> f64>(reps: usize, mut f: F) -> f64 {
    (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min)
}

fn main() {
    let out_path =
        std::env::var("RLA_BENCH_OUT").unwrap_or_else(|_| "benches/bench_out/phased_reuse.jsonl".to_string());
    if let Some(d) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let mut out = std::fs::File::create(&out_path).expect("open out");
    let opts = SolverSettings::default();
    let so = stencil::StencilOpts::default();

    // A spread of fill profiles: the analyze/factor ratio (hence the reuse win)
    // grows as the factor gets cheaper relative to the value-independent analysis.
    // Helmholtz 3D (high fill, factor-dominated) is the EM frequency-sweep case;
    // 2D Poisson and banded (low fill) show where reuse pays most.
    let mut cases: Vec<(String, CscMatrix<C>, CscMatrix<C>)> = Vec::new();
    for k in [20usize, 28, 36] {
        cases.push((
            format!("helmholtz3d_{}", k * k * k),
            stencil::helmholtz(&[k, k, k], Complex::new(2.0, 0.1), &so),
            stencil::helmholtz(&[k, k, k], Complex::new(3.5, 0.2), &so),
        ));
    }
    for m in [200usize, 360] {
        cases.push((
            format!("helmholtz2d_{}", m * m),
            stencil::helmholtz(&[m, m], Complex::new(0.5, 0.05), &so),
            stencil::helmholtz(&[m, m], Complex::new(0.9, 0.1), &so),
        ));
    }
    for n in [40000usize, 80000] {
        cases.push((
            format!("banded_{n}"),
            rslab::matgen::structured::banded::<C>(n, 24, 1.0, 1),
            rslab::matgen::structured::banded::<C>(n, 24, 1.3, 2),
        ));
    }

    for (name, a, a2) in &cases {
        let n = a.n;
        let nnz = a.values.len();

        let analyze_ms = best_of(3, || {
            let t = Instant::now();
            let s = LdltSymbolic::analyze(a).expect("analyze");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(s.n());
            ms
        });
        // Reuse one analysis for the factor timing (the phased path).
        let sym = LdltSymbolic::analyze(a).expect("analyze");
        let factor_ms = best_of(3, || {
            let t = Instant::now();
            let f = sym.factor(a, &opts).expect("factor");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(f.factor_nnz());
            ms
        });

        // Reuse correctness: factor a *second* value set on the same analysis.
        let f2 = sym.factor(a2, &opts).expect("factor a2");
        let b: Vec<C> = (0..n).map(|i| Complex::new(i as f64 % 7.0 - 3.0, 1.0)).collect();
        let x = f2.solve(&b).expect("solve");
        let mut ax = vec![C::new(0.0, 0.0); n];
        a2.symv(&x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt()
            / b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(res < 1e-8, "reuse solve residual {res:.1e}");

        let rec = format!(
            "{{\"name\":\"{name}\",\"n\":{n},\"nnz\":{nnz},\"analyze_ms\":{analyze_ms:.3},\"factor_ms\":{factor_ms:.3}}}"
        );
        writeln!(out, "{rec}").expect("write");
        eprintln!(
            "[phased] {name:<16} n={n:>7} nnz={nnz:>9}  analyze {analyze_ms:>8.2}ms  factor {factor_ms:>9.2}ms  (analyze {:.0}% of one solve)",
            100.0 * analyze_ms / (analyze_ms + factor_ms)
        );
    }
    eprintln!("[phased] wrote {out_path}");
}
