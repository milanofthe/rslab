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
//! Resource discipline: every matrix is gated by the **a-priori memory estimate**
//! (skip if the estimated transient peak exceeds `RLA_SWEEP_MEM_CAP_MB`, so the
//! run never OOMs), and matrices above `RLA_SWEEP_GRID_FLOP_CAP` run only the
//! baseline combo (the full grid is 16x the cost) so the wall-clock stays bounded.
//!
//! Env:
//! * `RLA_SWEEP_OUT`            output JSONL (default `benches/bench_out/sweep.jsonl`)
//! * `RLA_SWEEP_SMOKE=1`        tiny corpus + tiny grid (sanity)
//! * `RLA_SWEEP_SCALE=f`        multiply generated dimensions by `f` (default 1.0)
//! * `RLA_SWEEP_MEM_CAP_MB=n`   skip matrices whose est. transient peak exceeds this (default 40000)
//! * `RLA_SWEEP_GRID_FLOP_CAP=x` above this est. factor-flops, run only the baseline combo (default 2e10)
//! * `RLA_SWEEP_SUITESPARSE=1`  also fetch the SuiteSparse list (needs `--features matgen-download`)
//!
//! Run (generated only):  `cargo bench --bench sweep --features matgen`
//! Run (with SuiteSparse): `RLA_SWEEP_SUITESPARSE=1 cargo bench --bench sweep --features matgen-download`
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

/// Diverse generated corpus spanning the Structure x Symmetry x Cond x Density
/// axes with a size ladder. The memory gate in `main` trims anything too large
/// for the budget, so the ladder can be generous.
fn corpus() -> Vec<Entry> {
    let scale: f64 = std::env::var("RLA_SWEEP_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let smoke = std::env::var("RLA_SWEEP_SMOKE").is_ok();
    let g = |x: usize| ((x as f64 * scale).round() as usize).max(4);
    let mut e: Vec<Entry> = Vec::new();
    let mut sym = |name: String, a: CscMatrix<C>| e.push(Entry { name, mat: Mat::Sym(a) });

    if smoke {
        let m = g(40);
        sym(
            format!("poisson2d_{}", m * m),
            stencil::laplacian::<C>(&[m, m], &stencil::StencilOpts::default()),
        );
        return e;
    }

    // 2D Poisson (well-conditioned, banded, diagonally dominant).
    for s in [50usize, 100, 180, 280] {
        let m = g(s);
        sym(
            format!("poisson2d_{}", m * m),
            stencil::laplacian::<C>(&[m, m], &stencil::StencilOpts::default()),
        );
    }
    // Anisotropic 2D (long thin fronts).
    for s in [80usize, 160, 240] {
        let m = g(s);
        let opts = stencil::StencilOpts { aniso: [1.0, 100.0, 1.0], ..Default::default() };
        sym(format!("aniso2d_{}", m * m), stencil::laplacian::<C>(&[m, m], &opts));
    }
    // Jumping-coefficient 2D (ill-conditioned heterogeneous media).
    for s in [80usize, 160, 240] {
        let m = g(s);
        let opts =
            stencil::StencilOpts { jump_contrast: 1e4, shift: 1e-2, ..Default::default() };
        sym(format!("jump2d_{}", m * m), stencil::laplacian::<C>(&[m, m], &opts));
    }
    // 3D Poisson (dense fronts near the root - the top-of-tree regime).
    for k in [12usize, 20, 30, 40] {
        let kk = g(k);
        sym(
            format!("poisson3d_{}", kk * kk * kk),
            stencil::laplacian::<C>(&[kk, kk, kk], &stencil::StencilOpts::default()),
        );
    }
    // 3D complex Helmholtz (EM-FEM, complex-symmetric, indefinite-ish).
    for k in [12usize, 20, 28] {
        let kk = g(k);
        sym(
            format!("helmholtz3d_{}", kk * kk * kk),
            stencil::helmholtz(&[kk, kk, kk], c(2.0, 0.1), &stencil::StencilOpts::default()),
        );
    }
    // Banded (narrow vs wide).
    for (n, bw) in [(8000usize, 8usize), (20000, 16), (40000, 40)] {
        sym(format!("banded_{}_{}", g(n), bw), structured::banded::<C>(g(n), bw, 1.0, 1));
    }
    // Arrow / bordered (dense border block - high degree-CV).
    for (n, b) in [(8000usize, 24usize), (20000, 48)] {
        sym(format!("arrow_{}_{}", g(n), b), structured::arrow::<C>(g(n), b, 1e-2, 1));
    }
    // Random SPD (irregular pattern).
    for (n, d) in [(5000usize, 14usize), (15000, 20)] {
        sym(format!("rand_spd_{}", g(n)), random::random_spd::<C>(g(n), d, 1.0, 1));
    }

    // Unsymmetric: BEM/MoM-like (dense-ish) and random.
    for n in [1500usize, 3000, 6000] {
        e.push(Entry {
            name: format!("bem_{}", g(n)),
            mat: Mat::Unsym(rslab::matgen::bem::kernel(g(n), &rslab::matgen::bem::BemOpts::default())),
        });
    }
    for (n, d) in [(5000usize, 14usize), (15000, 18)] {
        e.push(Entry {
            name: format!("rand_unsym_{}", g(n)),
            mat: Mat::Unsym(random::random_unsym::<C>(g(n), d, 2.0, 1)),
        });
    }
    e
}

/// SuiteSparse matrices spanning domains (structural, CFD, circuit, QM, KKT, EM).
/// `(group, name)`. Fetched + cached on demand; gated by the memory cap like the
/// generated ones.
#[cfg(feature = "matgen-download")]
fn suitesparse_list() -> &'static [(&'static str, &'static str)] {
    &[
        ("HB", "bcsstk14"),
        ("HB", "bcsstk18"),
        ("HB", "bcsstk24"),
        ("HB", "bcsstk28"),
        ("Boeing", "bcsstk39"),
        ("Boeing", "msc10848"),
        ("Boeing", "crystk03"),
        ("Williams", "cant"),
        ("GHS_psdef", "wathen100"),
        ("GHS_psdef", "wathen120"),
        ("Nasa", "nasa2910"),
        ("Nasa", "nasasrb"),
        ("Rothberg", "cfd1"),
        ("Rothberg", "cfd2"),
        ("DNVS", "ship_001"),
        ("FIDAP", "ex11"),
        ("FIDAP", "ex40"),
        ("Bai", "qc2534"),
        ("GHS_indef", "cont-300"),
        ("GHS_indef", "bratu3d"),
        ("Schenk_ISEI", "barrier2-1"),
        ("Um", "2cubes_sphere"),
        ("Schmid", "thermal1"),
        ("Botonakis", "thermomech_dM"),
    ]
}

#[cfg(feature = "matgen-download")]
fn suitesparse_entries() -> Vec<Entry> {
    use rslab::{read_mtx_any, MtxLoaded};
    let mut out = Vec::new();
    for &(group, name) in suitesparse_list() {
        match rslab::matgen::download::fetch(group, name) {
            Ok(path) => match read_mtx_any(&path) {
                Ok(MtxLoaded::Symmetric(a)) => {
                    out.push(Entry { name: name.to_string(), mat: Mat::Sym(a) })
                }
                Ok(MtxLoaded::General(a)) => {
                    out.push(Entry { name: name.to_string(), mat: Mat::Unsym(a) })
                }
                Err(err) => eprintln!("[sweep] skip {name}: parse {err}"),
            },
            Err(err) => eprintln!("[sweep] skip {name}: fetch {err}"),
        }
    }
    out
}

#[cfg(not(feature = "matgen-download"))]
fn suitesparse_entries() -> Vec<Entry> {
    Vec::new()
}

/// One point in the knob grid.
#[derive(Clone, Copy)]
struct Param {
    ordering: OrderingMethod,
    nemin: usize,
    par_cdiv: usize,
    threads: usize,
}

const BASELINE: Param = Param {
    ordering: OrderingMethod::Amd,
    nemin: 16,
    par_cdiv: 8_000_000,
    threads: 0,
};

fn grid() -> Vec<Param> {
    let smoke = std::env::var("RLA_SWEEP_SMOKE").is_ok();
    if smoke {
        return vec![BASELINE, Param { nemin: 48, ..BASELINE }];
    }
    // Focused ordering comparison at the *production-default* factor knobs
    // (nemin 16, par_cdiv 8M, threads 2): does per-matrix ordering beat `Auto`?
    if std::env::var("RLA_SWEEP_ORDERINGS_ONLY").is_ok() {
        return [
            OrderingMethod::Auto,
            OrderingMethod::Amd,
            OrderingMethod::Amf,
            OrderingMethod::MetisND,
        ]
        .iter()
        .map(|&ordering| Param { ordering, nemin: 16, par_cdiv: 8_000_000, threads: 2 })
        .collect();
    }
    let orderings = [OrderingMethod::Amd, OrderingMethod::MetisND];
    let nemins = [16usize, 48];
    let cdivs = [8_000_000usize, 2_000_000];
    let threads = [2usize, 0];
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

/// Canonical analysis summary: structural features + the a-priori memory/flops
/// used for the resource gates. `None` if the matrix fails to analyze.
fn canonical(mat: &Mat) -> Option<(StructuralFeatures, f64, u64)> {
    match mat {
        Mat::Sym(a) => {
            let sym = LdltSymbolic::analyze(a).ok()?;
            let est = sym.estimate_memory::<C>();
            Some((
                StructuralFeatures::from_symmetric(a, &sym),
                est.transient_peak_bytes as f64 / 1048576.0,
                est.factor_flops,
            ))
        }
        Mat::Unsym(a) => {
            let sym = LuSymbolic::analyze(a).ok()?;
            let est = sym.estimate_memory::<C>();
            Some((
                StructuralFeatures::from_general(a, &sym),
                est.transient_peak_bytes as f64 / 1048576.0,
                est.factor_flops,
            ))
        }
    }
}

fn main() {
    let out_path = std::env::var("RLA_SWEEP_OUT")
        .unwrap_or_else(|_| "benches/bench_out/sweep.jsonl".to_string());
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut out = std::fs::File::create(&out_path).expect("open sweep out");

    let mem_cap_mb: f64 = std::env::var("RLA_SWEEP_MEM_CAP_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40_000.0);
    let grid_flop_cap: f64 = std::env::var("RLA_SWEEP_GRID_FLOP_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2e10);

    let mut corpus = corpus();
    if std::env::var("RLA_SWEEP_SUITESPARSE").is_ok() {
        corpus.extend(suitesparse_entries());
    }
    let full_grid = grid();
    eprintln!(
        "[sweep] {} matrices, up to {} combos, mem_cap={:.0} MB, grid_flop_cap={:.0e} -> {}",
        corpus.len(),
        full_grid.len(),
        mem_cap_mb,
        grid_flop_cap,
        out_path
    );

    let mut n_records = 0usize;
    let mut n_skipped_mem = 0usize;
    for entry in &corpus {
        let Some((feat, est_mb, flops)) = canonical(&entry.mat) else {
            eprintln!("[sweep] skip {}: analyze failed", entry.name);
            continue;
        };
        let (n, nnz) = match &entry.mat {
            Mat::Sym(a) => (a.n, a.values.len()),
            Mat::Unsym(a) => (a.n, a.values.len()),
        };
        if est_mb > mem_cap_mb {
            eprintln!("[sweep] skip {} n={}: est {:.0} MB > cap {:.0} MB", entry.name, n, est_mb, mem_cap_mb);
            n_skipped_mem += 1;
            continue;
        }
        // Bound compute: the full 16x grid only for matrices below the flop cap.
        let combos: &[Param] = if flops as f64 > grid_flop_cap {
            std::slice::from_ref(&BASELINE)
        } else {
            &full_grid
        };
        let feat_json = serde_json::to_value(&feat).expect("feat json");
        eprintln!(
            "[sweep] {} n={} nnz={} est={:.0}MB flops={:.1e} combos={}",
            entry.name, n, nnz, est_mb, flops as f64, combos.len()
        );

        for p in combos {
            set_gemm_thresholds(GemmThresholds { par_cdiv: p.par_cdiv, ..GemmThresholds::default() });
            let aopts = AnalyzeOptions::default().with_ordering(p.ordering).with_nemin(p.nemin);
            let fopts = FactorOptions::default()
                .with_method(FactorMethod::LeftLooking)
                .with_threads(p.threads);

            let (fac_ms, fill, peak_mb, res) = match &entry.mat {
                Mat::Sym(a) => {
                    let Ok(sym) = LdltSymbolic::analyze_with(a, &aopts) else { continue };
                    let b: Vec<C> = (0..a.n).map(|i| c(i as f64 % 7.0 - 3.0, 1.0)).collect();
                    meter_reset();
                    let t = Instant::now();
                    let Ok(f) = sym.factor(a, &fopts) else { meter_peak_mb(); continue };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let peak = meter_peak_mb();
                    let x = f.solve(&b).unwrap_or_default();
                    let res = if x.len() == a.n { residual_sym(a, &x, &b) } else { f64::NAN };
                    (ms, f.factor_nnz(), peak, res)
                }
                Mat::Unsym(a) => {
                    let Ok(sym) = LuSymbolic::analyze_with(a, &aopts) else { continue };
                    let b: Vec<C> = (0..a.n).map(|i| c(i as f64 % 7.0 - 3.0, 1.0)).collect();
                    meter_reset();
                    let t = Instant::now();
                    let Ok(f) = sym.factor(a, &fopts) else { meter_peak_mb(); continue };
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    let peak = meter_peak_mb();
                    let x = f.solve(&b).unwrap_or_default();
                    let res = if x.len() == a.n { residual_unsym(a, &x, &b) } else { f64::NAN };
                    (ms, f.factor_nnz(), peak, res)
                }
            };

            let rec = serde_json::json!({
                "matrix": entry.name, "n": n, "nnz": nnz, "dtype": "complex128",
                "features": feat_json,
                "params": {
                    "ordering": ordering_name(p.ordering), "nemin": p.nemin,
                    "par_cdiv": p.par_cdiv, "threads": p.threads, "method": "left_looking",
                },
                "metrics": {
                    "factor_ms": fac_ms, "factor_nnz": fill, "peak_mb": peak_mb,
                    "est_transient_mb": est_mb, "residual": res,
                },
            });
            writeln!(out, "{}", rec).expect("write rec");
            n_records += 1;
        }
    }
    set_gemm_thresholds(GemmThresholds::default());
    eprintln!(
        "[sweep] done: {} records, {} skipped (mem cap) -> {}",
        n_records, n_skipped_mem, out_path
    );
}
