//! Auto-tuning sweep harness: build a diverse matrix corpus, and for each matrix
//! measure factor time / fill / memory / residual across a grid of solver knobs
//! (fill-ordering, amalgamation `nemin`, the GEMM parallelism thresholds, thread
//! count). Emits one JSONL record per `(matrix, param-combo)` -
//! `{matrix, n, nnz, dtype, features{...}, params{...}, metrics{...}}` - the
//! dataset that drives the data-driven scheduling fixes and the parameter
//! predictor.
//!
//! `features` is the matrix's canonical structural fingerprint (under the default
//! analysis), so the ML framing is clean: features fixed per matrix, knobs varied,
//! outcomes measured.
//!
//! Env:
//! * `RLA_SWEEP_OUT`   output JSONL path (default `benches/bench_out/sweep.jsonl`)
//! * `RLA_SWEEP_SMOKE=1`  tiny corpus + tiny grid (CI / sanity; the default here)
//! * `RLA_SWEEP_SCALE=f`  multiply generated dimensions by `f` (default 1.0)
//!
//! Run: `cargo bench --bench sweep --features matgen`
#![allow(clippy::needless_range_loop)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::{random, stencil, structured};
use rslab::{
    set_gemm_thresholds, AnalyzeOptions, CscMatrix, FactorMethod, FactorOptions, GemmThresholds,
    GeneralCsc, LdltSymbolic, LuSymbolic, OrderingMethod, StructuralFeatures,
};

type C = Complex<f64>;

// ---- live-bytes counting allocator (measured factor peak) --------------------
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ON: AtomicBool = AtomicBool::new(false);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() && ON.load(Ordering::Relaxed) {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if ON.load(Ordering::Relaxed) {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        }
        System.dealloc(p, l);
    }
}
#[global_allocator]
static A: Counting = Counting;

fn meter_reset() {
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    ON.store(true, Ordering::Relaxed);
}
fn meter_peak_mb() -> f64 {
    ON.store(false, Ordering::Relaxed);
    PEAK.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
}

enum Mat {
    Sym(CscMatrix<C>),
    Unsym(GeneralCsc<C>),
}

struct Entry {
    name: String,
    mat: Mat,
}

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

/// Diverse, size-bounded corpus spanning the Structure x Symmetry x Cond x Density
/// axes. Scaled by `RLA_SWEEP_SCALE`; trimmed under `RLA_SWEEP_SMOKE`.
fn corpus() -> Vec<Entry> {
    let scale: f64 = std::env::var("RLA_SWEEP_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let smoke = std::env::var("RLA_SWEEP_SMOKE").is_ok();
    let g = |x: usize| ((x as f64 * scale).round() as usize).max(4);
    let mut e: Vec<Entry> = Vec::new();
    let mut sym = |name: String, a: CscMatrix<C>| e.push(Entry { name, mat: Mat::Sym(a) });

    // 2D Poisson (well-conditioned, banded, diagonally dominant).
    for s in if smoke { vec![40] } else { vec![40, 70, 100] } {
        let m = g(s);
        sym(
            format!("poisson2d_{}", m * m),
            stencil::laplacian::<C>(&[m, m], &stencil::StencilOpts::default()),
        );
    }
    if !smoke {
        // Anisotropic 2D (harder; long thin elimination fronts).
        for s in [60usize, 90] {
            let m = g(s);
            let opts = stencil::StencilOpts {
                aniso: [1.0, 100.0, 1.0],
                ..stencil::StencilOpts::default()
            };
            sym(format!("aniso2d_{}", m * m), stencil::laplacian::<C>(&[m, m], &opts));
        }
        // Jumping-coefficient 2D (ill-conditioned heterogeneous media).
        for s in [60usize, 90] {
            let m = g(s);
            let opts = stencil::StencilOpts {
                jump_contrast: 1e4,
                shift: 1e-2,
                ..stencil::StencilOpts::default()
            };
            sym(format!("jump2d_{}", m * m), stencil::laplacian::<C>(&[m, m], &opts));
        }
        // 3D Poisson (dense fronts near the root - the top-of-tree regime).
        for k in [10usize, 16] {
            let kk = g(k);
            sym(
                format!("poisson3d_{}", kk * kk * kk),
                stencil::laplacian::<C>(&[kk, kk, kk], &stencil::StencilOpts::default()),
            );
        }
        // 3D complex Helmholtz (EM-FEM, indefinite-ish, complex-symmetric).
        for k in [10usize, 14] {
            let kk = g(k);
            sym(
                format!("helmholtz3d_{}", kk * kk * kk),
                stencil::helmholtz(&[kk, kk, kk], c(2.0, 0.1), &stencil::StencilOpts::default()),
            );
        }
        // Banded (narrow vs wide).
        for (n, bw) in [(4000usize, 8usize), (6000, 40)] {
            sym(format!("banded_{}_{}", g(n), bw), structured::banded::<C>(g(n), bw, 1.0, 1));
        }
        // Arrow / bordered (one dense border block - high degree-CV).
        sym("arrow_4000".into(), structured::arrow::<C>(g(4000), 24, 1e-2, 1));
        // Random SPD (irregular pattern).
        for (n, d) in [(3000usize, 14usize)] {
            sym(format!("rand_spd_{}", g(n)), random::random_spd::<C>(g(n), d, 1.0, 1));
        }
    }

    // Unsymmetric: BEM/MoM-like and random.
    let bem_sizes = if smoke { vec![] } else { vec![1500usize, 3000] };
    for n in bem_sizes {
        e.push(Entry {
            name: format!("bem_{}", g(n)),
            mat: Mat::Unsym(rslab::matgen::bem::kernel(g(n), &rslab::matgen::bem::BemOpts::default())),
        });
    }
    if !smoke {
        e.push(Entry {
            name: "rand_unsym_3000".into(),
            mat: Mat::Unsym(random::random_unsym::<C>(g(3000), 14, 2.0, 1)),
        });
    }
    e
}

/// One point in the knob grid.
#[derive(Clone, Copy)]
struct Param {
    ordering: OrderingMethod,
    nemin: usize,
    par_cdiv: usize,
    threads: usize,
}

fn grid() -> Vec<Param> {
    let smoke = std::env::var("RLA_SWEEP_SMOKE").is_ok();
    let orderings = [OrderingMethod::Amd, OrderingMethod::MetisND];
    let nemins = if smoke { vec![16usize] } else { vec![16usize, 48] };
    let cdivs = if smoke {
        vec![8_000_000usize]
    } else {
        vec![8_000_000usize, 2_000_000]
    };
    let threads = if smoke { vec![2usize] } else { vec![2usize, 0] };
    let mut v = Vec::new();
    for &ordering in &orderings {
        for &nemin in &nemins {
            for &par_cdiv in &cdivs {
                for &t in &threads {
                    v.push(Param { ordering, nemin, par_cdiv, threads: t });
                }
            }
        }
    }
    v
}

fn residual_sym(a: &CscMatrix<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![C::new(0.0, 0.0); a.n];
    a.symv(x, &mut ax);
    let num: f64 = (0..a.n).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den.max(1e-300)
}
fn residual_unsym(a: &GeneralCsc<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![C::new(0.0, 0.0); a.n];
    a.matvec(x, &mut ax);
    let num: f64 = (0..a.n).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den.max(1e-300)
}

fn ordering_name(o: OrderingMethod) -> &'static str {
    match o {
        OrderingMethod::Amd => "amd",
        OrderingMethod::Amf => "amf",
        OrderingMethod::MetisND => "metis",
        OrderingMethod::ScotchND => "scotch",
        OrderingMethod::KahipND => "kahip",
        OrderingMethod::Auto => "auto",
        OrderingMethod::AutoRace => "auto_race",
    }
}

fn main() {
    let out_path = std::env::var("RLA_SWEEP_OUT")
        .unwrap_or_else(|_| "benches/bench_out/sweep.jsonl".to_string());
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut out = std::fs::File::create(&out_path).expect("open sweep out");

    let corpus = corpus();
    let grid = grid();
    eprintln!(
        "[sweep] {} matrices x {} param-combos = {} factorizations -> {}",
        corpus.len(),
        grid.len(),
        corpus.len() * grid.len(),
        out_path
    );

    for entry in &corpus {
        // Canonical fingerprint under the default analysis.
        let (feat, n, nnz, dtype) = match &entry.mat {
            Mat::Sym(a) => {
                let sym = LdltSymbolic::analyze(a).expect("analyze sym");
                (StructuralFeatures::from_symmetric(a, &sym), a.n, a.values.len(), "complex128")
            }
            Mat::Unsym(a) => {
                let sym = LuSymbolic::analyze(a).expect("analyze unsym");
                (StructuralFeatures::from_general(a, &sym), a.n, a.values.len(), "complex128")
            }
        };
        let feat_json = serde_json::to_value(&feat).expect("feat json");
        eprintln!("[sweep] {} n={} nnz={}", entry.name, n, nnz);

        for p in &grid {
            set_gemm_thresholds(GemmThresholds {
                par_cdiv: p.par_cdiv,
                ..GemmThresholds::default()
            });
            let aopts = AnalyzeOptions::default()
                .with_ordering(p.ordering)
                .with_nemin(p.nemin);
            let fopts = FactorOptions::default()
                .with_method(FactorMethod::LeftLooking)
                .with_threads(p.threads);

            let (fac_ms, fill, peak_mb, est_mb, res) = match &entry.mat {
                Mat::Sym(a) => {
                    let sym = match LdltSymbolic::analyze_with(a, &aopts) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let est = sym.estimate_memory::<C>().transient_peak_bytes as f64 / 1048576.0;
                    let b: Vec<C> = (0..a.n).map(|i| c(i as f64 % 7.0 - 3.0, 1.0)).collect();
                    meter_reset();
                    let t = Instant::now();
                    let f = match sym.factor(a, &fopts) {
                        Ok(f) => f,
                        Err(_) => {
                            meter_peak_mb();
                            continue;
                        }
                    };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let peak = meter_peak_mb();
                    let x = f.solve(&b).unwrap_or_default();
                    let res = if x.len() == a.n { residual_sym(a, &x, &b) } else { f64::NAN };
                    (ms, f.factor_nnz(), peak, est, res)
                }
                Mat::Unsym(a) => {
                    let sym = match LuSymbolic::analyze_with(a, &aopts) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let est = sym.estimate_memory::<C>().transient_peak_bytes as f64 / 1048576.0;
                    let b: Vec<C> = (0..a.n).map(|i| c(i as f64 % 7.0 - 3.0, 1.0)).collect();
                    meter_reset();
                    let t = Instant::now();
                    let f = match sym.factor(a, &fopts) {
                        Ok(f) => f,
                        Err(_) => {
                            meter_peak_mb();
                            continue;
                        }
                    };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let peak = meter_peak_mb();
                    let x = f.solve(&b).unwrap_or_default();
                    let res = if x.len() == a.n { residual_unsym(a, &x, &b) } else { f64::NAN };
                    (ms, f.factor_nnz(), peak, est, res)
                }
            };

            let rec = serde_json::json!({
                "matrix": entry.name,
                "n": n,
                "nnz": nnz,
                "dtype": dtype,
                "features": feat_json,
                "params": {
                    "ordering": ordering_name(p.ordering),
                    "nemin": p.nemin,
                    "par_cdiv": p.par_cdiv,
                    "threads": p.threads,
                    "method": "left_looking",
                },
                "metrics": {
                    "factor_ms": fac_ms,
                    "factor_nnz": fill,
                    "peak_mb": peak_mb,
                    "est_transient_mb": est_mb,
                    "residual": res,
                },
            });
            writeln!(out, "{}", rec).expect("write rec");
        }
    }
    // Restore defaults for any in-process reuse.
    set_gemm_thresholds(GemmThresholds::default());
    eprintln!("[sweep] done -> {}", out_path);
}
