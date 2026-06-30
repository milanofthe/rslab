//! Auto-tuning sweep harness: build a diverse matrix corpus, and for each matrix
//! measure the two performance metrics - **peak memory** and **factor speed** (+
//! residual for validity) - across a grid spanning *every* tunable
//! [`SolverSettings`] knob: fill-ordering, amalgamation `nemin` + relaxed
//! `max_width`, factor `method` (left-looking vs multifrontal), the kernel panel
//! width + GEMM thresholds (`scalar_gate`/`par_gemm`/`par_cdiv`) + Schur kernel,
//! and the worker count. The grid is one-factor-at-a-time over each knob's menu
//! (main effects) plus seeded random joint samples (interactions). Emits one JSONL
//! record per `(matrix, param-combo)` -
//! `{matrix, n, nnz, dtype, features{...}, params{...}, metrics{...}}` - the
//! dataset that trains the parameter predictor (features -> best knobs).
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
//! * `RLA_SWEEP_RANDOM=k`        random joint knob samples per matrix, on top of OFAT (default 16)
//! * `RLA_SWEEP_THREADS_ONLY=1`  sweep only the worker-count ladder (the thread-scaling dataset)
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
    CscMatrix, FactorMethod, GemmThresholds, GeneralCsc, LdltSymbolic, LuSymbolic, OrderingMethod,
    RelaxAmalgamation, SolverSettings, StructuralFeatures,
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
            // Saturating: an allocation made *before* `meter_reset` (uncounted) but
            // freed inside the metered window must not underflow `LIVE` (which would
            // wrap to ~2^64 and poison `PEAK`). Clamp at 0 - the peak then reflects
            // the net factor transient.
            let _ = LIVE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(l.size()))
            });
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

    // 2D Poisson (well-conditioned, banded, diagonally dominant) - fine ladder.
    for s in [50usize, 90, 140, 200, 280, 360] {
        let m = g(s);
        sym(
            format!("poisson2d_{}", m * m),
            stencil::laplacian::<C>(&[m, m], &stencil::StencilOpts::default()),
        );
    }
    // Anisotropic 2D (long thin fronts), two contrast levels.
    for &aniso in &[100.0, 1000.0] {
        for s in [100usize, 180, 260] {
            let m = g(s);
            let opts = stencil::StencilOpts { aniso: [1.0, aniso, 1.0], ..Default::default() };
            sym(
                format!("aniso2d_{}_{:.0}", m * m, aniso),
                stencil::laplacian::<C>(&[m, m], &opts),
            );
        }
    }
    // Jumping-coefficient 2D (ill-conditioned heterogeneous media).
    for s in [100usize, 180, 260] {
        let m = g(s);
        let opts =
            stencil::StencilOpts { jump_contrast: 1e4, shift: 1e-2, ..Default::default() };
        sym(format!("jump2d_{}", m * m), stencil::laplacian::<C>(&[m, m], &opts));
    }
    // 3D Poisson (dense fronts near the root - the top-of-tree regime).
    for k in [12usize, 18, 24, 32, 42] {
        let kk = g(k);
        sym(
            format!("poisson3d_{}", kk * kk * kk),
            stencil::laplacian::<C>(&[kk, kk, kk], &stencil::StencilOpts::default()),
        );
    }
    // 3D complex Helmholtz (EM-FEM, complex-symmetric, indefinite-ish), two k.
    for &(kr, ki) in &[(2.0, 0.1), (5.0, 0.3)] {
        for k in [12usize, 18, 26] {
            let kk = g(k);
            sym(
                format!("helmholtz3d_{}_{:.0}", kk * kk * kk, kr),
                stencil::helmholtz(&[kk, kk, kk], c(kr, ki), &stencil::StencilOpts::default()),
            );
        }
    }
    // Banded (narrow vs wide) - locality ladder.
    for (n, bw) in [(8000usize, 8usize), (20000, 16), (40000, 40), (80000, 24)] {
        sym(format!("banded_{}_{}", g(n), bw), structured::banded::<C>(g(n), bw, 1.0, 1));
    }
    // Arrow / bordered (dense border block - high degree-CV).
    for (n, b) in [(8000usize, 24usize), (20000, 48), (40000, 32)] {
        sym(format!("arrow_{}_{}", g(n), b), structured::arrow::<C>(g(n), b, 1e-2, 1));
    }
    // Random SPD (irregular pattern) - density ladder.
    for (n, d) in [(5000usize, 10usize), (10000, 16), (20000, 22)] {
        sym(format!("rand_spd_{}_{}", g(n), d), random::random_spd::<C>(g(n), d, 1.0, 1));
    }
    // Spectral (exactly-conditioned), SPD and indefinite.
    for &(kappa, indef) in &[(1e6, false), (1e8, false), (1e6, true)] {
        let n = g(4000);
        sym(
            format!("spectral_{}_{:.0e}_{}", n, kappa, indef as u8),
            random::spectral::<C>(n, kappa, indef, 1),
        );
    }

    // Unsymmetric: BEM/MoM-like (dense-ish) and random (density ladder).
    for n in [1500usize, 3000, 6000, 10000] {
        e.push(Entry {
            name: format!("bem_{}", g(n)),
            mat: Mat::Unsym(rslab::matgen::bem::kernel(g(n), &rslab::matgen::bem::BemOpts::default())),
        });
    }
    for (n, d) in [(5000usize, 10usize), (10000, 14), (20000, 18)] {
        e.push(Entry {
            name: format!("rand_unsym_{}_{}", g(n), d),
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
        // --- structural FEM ---
        ("HB", "bcsstk16"),
        ("HB", "bcsstk17"),
        ("HB", "bcsstk25"),
        ("HB", "bcsstk38"),
        ("Boeing", "ct20stif"),
        ("Boeing", "msc23052"),
        ("Boeing", "pwtk"),
        ("Cylshell", "s3rmt3m3"),
        ("Cylshell", "s3rmq4m1"),
        ("Cylshell", "s3dkt3m2"),
        ("DNVS", "shipsec1"),
        ("DNVS", "ship_003"),
        ("Nasa", "nasa1824"),
        ("Nasa", "nasa4704"),
        ("Simon", "raefsky4"),
        // --- CFD / fluid (unsymmetric) ---
        ("Simon", "raefsky3"),
        ("Simon", "raefsky2"),
        ("Simon", "venkat01"),
        ("FIDAP", "ex19"),
        ("FIDAP", "ex35"),
        // --- circuit / semiconductor (unsymmetric) ---
        ("Hamm", "scircuit"),
        ("Hamm", "memplus"),
        ("Bomhof", "circuit_3"),
        // --- thermal / PDE ---
        ("Botonakis", "FEM_3D_thermal1"),
        ("Wissgott", "parabolic_fem"),
        // --- optimization / KKT (symmetric indefinite) ---
        ("GHS_indef", "cont-201"),
        ("GHS_indef", "stokes64"),
        ("GHS_indef", "dixmaanl"),
        ("GHS_indef", "boyd1"),
        // --- acoustics / model reduction ---
        ("Cunningham", "qa8fm"),
        ("Oberwolfach", "gyro"),
        ("Oberwolfach", "t2dah_e"),
        ("PARSEC", "Si5H12"),
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

/// One point in the knob grid - every tunable [`SolverSettings`] knob the sweep
/// varies (analysis ordering/nemin/relax, factor method/threads, and the kernel
/// GEMM thresholds + panel width + Schur kernel). `relax_width = 0` means
/// relaxed amalgamation off. The recorded outcomes are peak memory + factor speed
/// (the two performance metrics) plus the residual for validity filtering.
#[derive(Clone, Copy, PartialEq)]
struct Param {
    ordering: OrderingMethod,
    nemin: usize,
    relax_width: usize,
    panel_nb: usize,
    scalar_gate: usize,
    par_gemm: usize,
    par_cdiv: usize,
    use_gemm_schur: bool,
    method: FactorMethod,
    threads: usize,
}

/// Production defaults (the historically-tuned config) - the OFAT centre point.
const BASELINE: Param = Param {
    ordering: OrderingMethod::Auto,
    nemin: 16,
    relax_width: 256,
    panel_nb: 64,
    scalar_gate: 4096,
    par_gemm: 1_000_000,
    par_cdiv: 8_000_000,
    use_gemm_schur: true,
    method: FactorMethod::LeftLooking,
    threads: 0,
};

// Per-knob value menus, swept one-factor-at-a-time around `BASELINE`.
const M_ORDERING: [OrderingMethod; 3] =
    [OrderingMethod::Auto, OrderingMethod::Amd, OrderingMethod::MetisND];
const M_NEMIN: [usize; 4] = [1, 16, 48, 128];
const M_RELAX: [usize; 4] = [0, 128, 256, 512];
const M_PANEL_NB: [usize; 4] = [32, 64, 96, 128];
const M_SCALAR_GATE: [usize; 3] = [1024, 4096, 16384];
const M_PAR_GEMM: [usize; 3] = [250_000, 1_000_000, 4_000_000];
const M_PAR_CDIV: [usize; 3] = [2_000_000, 8_000_000, 32_000_000];
const M_SCHUR: [bool; 2] = [true, false];
const M_METHOD: [FactorMethod; 2] = [FactorMethod::LeftLooking, FactorMethod::Multifrontal];

/// Tiny reproducible PRNG (splitmix-style LCG) for the random joint samples, so
/// the sweep dataset is deterministic without pulling a `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next_u64() >> 33) as usize % xs.len()]
    }
}

fn threads_mode() -> bool {
    std::env::var("RLA_SWEEP_THREADS_ONLY").is_ok()
}

/// Thread-scaling ladder at production-default knobs: vary only the worker count
/// to trace the per-matrix speedup curve. Capped at the 12 physical cores -
/// beyond them hyperthreading gives compute-bound BLAS-3 little and only adds
/// noise. Reduced for the heaviest matrices to bound wall-clock.
fn thread_ladder(flops: u64) -> Vec<Param> {
    let ladder: &[usize] = if flops as f64 > 5e10 { &[1, 4, 8, 12] } else { &[1, 2, 4, 6, 8, 12] };
    ladder.iter().map(|&threads| Param { threads, ..BASELINE }).collect()
}

/// Analyze + factor `mat` under `s`, metering the peak; returns
/// `(factor_ms, factor_nnz, peak_mb, residual)` or `None` if analyze/factor fails.
fn measure_one(mat: &Mat, s: &SolverSettings) -> Option<(f64, usize, f64, f64)> {
    match mat {
        Mat::Sym(a) => {
            let sym = LdltSymbolic::analyze_with(a, s).ok()?;
            let b: Vec<C> = (0..a.n).map(|i| c(i as f64 % 7.0 - 3.0, 1.0)).collect();
            meter_reset();
            let t = Instant::now();
            let Ok(f) = sym.factor(a, s) else {
                meter_peak_mb();
                return None;
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let peak = meter_peak_mb();
            let x = f.solve(&b).unwrap_or_default();
            let res = if x.len() == a.n { residual_sym(a, &x, &b) } else { f64::NAN };
            Some((ms, f.factor_nnz(), peak, res))
        }
        Mat::Unsym(a) => {
            let sym = LuSymbolic::analyze_with(a, s).ok()?;
            let b: Vec<C> = (0..a.n).map(|i| c(i as f64 % 7.0 - 3.0, 1.0)).collect();
            meter_reset();
            let t = Instant::now();
            let Ok(f) = sym.factor(a, s) else {
                meter_peak_mb();
                return None;
            };
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let peak = meter_peak_mb();
            let x = f.solve(&b).unwrap_or_default();
            let res = if x.len() == a.n { residual_unsym(a, &x, &b) } else { f64::NAN };
            Some((ms, f.factor_nnz(), peak, res))
        }
    }
}

/// Knob grid: one-factor-at-a-time over every knob's menu (main effects) plus
/// `RLA_SWEEP_RANDOM` seeded random joint samples (interactions), deduplicated.
/// Threads stay at the baseline here - the worker count is its own sweep
/// (`RLA_SWEEP_THREADS_ONLY`), so the algorithmic knobs are isolated.
fn grid() -> Vec<Param> {
    let smoke = std::env::var("RLA_SWEEP_SMOKE").is_ok();
    if smoke {
        return vec![BASELINE, Param { nemin: 48, ..BASELINE }, Param { method: FactorMethod::Multifrontal, ..BASELINE }];
    }
    if std::env::var("RLA_SWEEP_ORDERINGS_ONLY").is_ok() {
        return [OrderingMethod::Auto, OrderingMethod::Amd, OrderingMethod::Amf, OrderingMethod::MetisND]
            .iter()
            .map(|&ordering| Param { ordering, ..BASELINE })
            .collect();
    }
    let mut v: Vec<Param> = vec![BASELINE];
    // OFAT: vary each knob over its menu with the rest at baseline (main effects).
    for &x in &M_ORDERING { v.push(Param { ordering: x, ..BASELINE }); }
    for &x in &M_NEMIN { v.push(Param { nemin: x, ..BASELINE }); }
    for &x in &M_RELAX { v.push(Param { relax_width: x, ..BASELINE }); }
    for &x in &M_PANEL_NB { v.push(Param { panel_nb: x, ..BASELINE }); }
    for &x in &M_SCALAR_GATE { v.push(Param { scalar_gate: x, ..BASELINE }); }
    for &x in &M_PAR_GEMM { v.push(Param { par_gemm: x, ..BASELINE }); }
    for &x in &M_PAR_CDIV { v.push(Param { par_cdiv: x, ..BASELINE }); }
    for &x in &M_SCHUR { v.push(Param { use_gemm_schur: x, ..BASELINE }); }
    for &x in &M_METHOD { v.push(Param { method: x, ..BASELINE }); }
    // Random joint samples (seeded) for knob interactions the OFAT axes miss.
    let n_random: usize = std::env::var("RLA_SWEEP_RANDOM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let mut rng = Lcg(0x00C0_FFEE_1234_5678);
    for _ in 0..n_random {
        v.push(Param {
            ordering: rng.pick(&M_ORDERING),
            nemin: rng.pick(&M_NEMIN),
            relax_width: rng.pick(&M_RELAX),
            panel_nb: rng.pick(&M_PANEL_NB),
            scalar_gate: rng.pick(&M_SCALAR_GATE),
            par_gemm: rng.pick(&M_PAR_GEMM),
            par_cdiv: rng.pick(&M_PAR_CDIV),
            use_gemm_schur: rng.pick(&M_SCHUR),
            method: rng.pick(&M_METHOD),
            threads: 0,
        });
    }
    // Dedup (OFAT re-emits the baseline value of each knob).
    let mut uniq: Vec<Param> = Vec::new();
    for p in v {
        if !uniq.contains(&p) {
            uniq.push(p);
        }
    }
    uniq
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

fn method_name(m: FactorMethod) -> &'static str {
    match m {
        FactorMethod::LeftLooking => "left_looking",
        FactorMethod::Multifrontal => "multifrontal",
    }
}

/// Canonical analysis summary: structural features + the a-priori per-path
/// memory estimates (left-looking, multifrontal) + flops, used for the resource
/// gates. `None` if the matrix fails to analyze.
fn canonical(mat: &Mat) -> Option<(StructuralFeatures, f64, f64, f64, f64, u64)> {
    let mb = |b: u64| b as f64 / 1048576.0;
    match mat {
        Mat::Sym(a) => {
            let sym = LdltSymbolic::analyze(a).ok()?;
            let est = sym.estimate_memory::<C>();
            Some((
                StructuralFeatures::from_symmetric(a, &sym),
                mb(est.transient_peak_bytes),
                mb(est.mf_transient_peak_bytes),
                mb(est.panel_live_peak_bytes),
                est.factor_nnz as f64,
                est.factor_flops,
            ))
        }
        Mat::Unsym(a) => {
            let sym = LuSymbolic::analyze(a).ok()?;
            let est = sym.estimate_memory::<C>();
            Some((
                StructuralFeatures::from_general(a, &sym),
                mb(est.transient_peak_bytes),
                mb(est.mf_transient_peak_bytes),
                mb(est.panel_live_peak_bytes),
                est.factor_nnz as f64,
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
    // Optional name filter (substring) to sweep a subset, e.g. `RLA_SWEEP_ONLY=banded`.
    if let Ok(only) = std::env::var("RLA_SWEEP_ONLY") {
        corpus.retain(|e| e.name.contains(&only));
    }
    let full_grid = grid();
    // Autotune mode: instead of the knob grid, measure `default` vs the auto-tuner's
    // pick at three Pareto weights per matrix - the end-to-end tuned-vs-default bench.
    let autotune = std::env::var("RLA_SWEEP_AUTOTUNE").is_ok();
    eprintln!(
        "[sweep] {} matrices, mode={}, mem_cap={:.0} MB, grid_flop_cap={:.0e} -> {}",
        corpus.len(),
        if autotune { "autotune" } else { "grid" },
        mem_cap_mb,
        grid_flop_cap,
        out_path
    );

    let mut n_records = 0usize;
    let mut n_skipped_mem = 0usize;
    for entry in &corpus {
        let Some((feat, ll_mb, mf_mb, floor_mb, def_fill, flops)) = canonical(&entry.mat) else {
            eprintln!("[sweep] skip {}: analyze failed", entry.name);
            continue;
        };
        let (n, nnz) = match &entry.mat {
            Mat::Sym(a) => (a.n, a.values.len()),
            Mat::Unsym(a) => (a.n, a.values.len()),
        };
        // Matrix-level gate: skip only if even the cheaper path exceeds the cap
        // (the per-combo gate below then drops just the MF combos when MF alone is
        // over budget).
        if ll_mb.min(mf_mb) > mem_cap_mb {
            eprintln!("[sweep] skip {} n={}: est {:.0} MB > cap {:.0} MB", entry.name, n, ll_mb.min(mf_mb), mem_cap_mb);
            n_skipped_mem += 1;
            continue;
        }
        // Thread-scaling mode runs the ladder (bypassing the flop-gate, since
        // scaling matters most on the large matrices); otherwise the full grid is
        // bounded to small matrices and giants fall back to the baseline combo.
        let ladder = if threads_mode() { Some(thread_ladder(flops)) } else { None };
        let combos: &[Param] = match &ladder {
            Some(l) => l.as_slice(),
            None if flops as f64 > grid_flop_cap => std::slice::from_ref(&BASELINE),
            None => &full_grid,
        };
        let feat_json = serde_json::to_value(&feat).expect("feat json");
        eprintln!(
            "[sweep] {} n={} nnz={} est_ll={:.0}MB est_mf={:.0}MB flops={:.1e}",
            entry.name, n, nnz, ll_mb, mf_mb, flops as f64
        );

        // Autotune mode: default vs the tuner's pick at balanced/speed/memory weights.
        if autotune {
            if flops as f64 > grid_flop_cap {
                continue; // skip the heaviest matrices (bounded wall-clock)
            }
            // Guarded + a-priori-vetoed recommendation (the real default-path logic):
            // pass the MF / LL-floor ratio (the floor is the reliable LL reference)
            // so memory-pathological multifrontal picks are vetoed deterministically.
            let mf_ll = if floor_mb > 0.0 { mf_mb / floor_mb } else { 1.0 };
            let d = SolverSettings::default();
            let mb = |b: u64| b as f64 / 1048576.0;
            // Full default-path logic: guarded+vetoed recommendation, then the exact
            // a-priori memory backstop - re-analyze the pick and fall back to the
            // default if its estimated peak exceeds the default's (never more memory).
            let pick = |w: f64| {
                let s = rslab::recommend_settings_vetoed(&feat, w, mf_ll);
                let same = (s.reorder, s.ordering, s.nemin, s.relax)
                    == (d.reorder, d.ordering, d.nemin, d.relax);
                let est_of = |e: rslab::MemoryEstimate| {
                    (mb(e.mf_transient_peak_bytes), mb(e.panel_live_peak_bytes), e.factor_nnz as f64)
                };
                let (mf, flr, fill) = if same {
                    (mf_mb, floor_mb, def_fill)
                } else {
                    match &entry.mat {
                        Mat::Sym(a) => LdltSymbolic::analyze_with(a, &s)
                            .map(|sy| est_of(sy.estimate_memory::<C>()))
                            .unwrap_or((mf_mb, floor_mb, def_fill)),
                        Mat::Unsym(a) => LuSymbolic::analyze_with(a, &s)
                            .map(|sy| est_of(sy.estimate_memory::<C>()))
                            .unwrap_or((mf_mb, floor_mb, def_fill)),
                    }
                };
                // Backstop (never more memory): exact fill must not grow, and the
                // realistic floor stays under the default's - MF vs the LL floor, LL
                // floor-vs-floor (consistent bias). Else fall back to the default.
                let fill_ok = fill <= def_fill * 1.02;
                let ok = fill_ok
                    && if s.method == FactorMethod::Multifrontal { mf <= floor_mb } else { flr <= floor_mb };
                if ok { s } else { d.clone() }
            };
            let configs = [
                ("default", SolverSettings::default()),
                ("tuned_balanced", pick(0.7)),
                ("tuned_speed", pick(1.0)),
                ("tuned_memory", pick(0.0)),
            ];
            for (label, s) in &configs {
                let est = if s.method == FactorMethod::Multifrontal { mf_mb } else { ll_mb };
                if est > mem_cap_mb {
                    continue;
                }
                let Some((fac_ms, fill, peak_mb, res)) = measure_one(&entry.mat, s) else { continue };
                let relax_w = s.relax.map_or(0, |r| r.max_width);
                let rec = serde_json::json!({
                    "matrix": entry.name, "n": n, "nnz": nnz, "flops": flops, "dtype": "complex128",
                    "config": label, "features": feat_json,
                    "params": {
                        "ordering": ordering_name(s.ordering), "nemin": s.nemin,
                        "relax_width": relax_w, "panel_nb": s.panel_nb,
                        "scalar_gate": s.scalar_gate, "par_gemm": s.par_gemm, "par_cdiv": s.par_cdiv,
                        "use_gemm_schur": s.use_gemm_schur, "method": method_name(s.method),
                    },
                    "metrics": {
                        "factor_ms": fac_ms, "factor_nnz": fill, "peak_mb": peak_mb, "residual": res,
                    },
                });
                writeln!(out, "{}", rec).expect("write rec");
                n_records += 1;
            }
            continue;
        }

        for p in combos {
            // Per-combo memory gate: drop a combo whose path's a-priori peak is
            // over the cap (so a passing matrix never OOMs on its MF combos).
            let combo_est = if p.method == FactorMethod::Multifrontal { mf_mb } else { ll_mb };
            if combo_est > mem_cap_mb {
                continue;
            }
            // One unified settings object drives both phases: analyze reads the
            // ordering/nemin/relax subset, factor reads method/threads + the kernel
            // knobs (panel_nb, GEMM thresholds, Schur) - per-call, no global state.
            let relax = (p.relax_width > 0)
                .then_some(RelaxAmalgamation { max_width: p.relax_width, max_extra_rows: 64 });
            let s = SolverSettings::default()
                .with_ordering(p.ordering)
                .with_nemin(p.nemin)
                .with_relax(relax)
                .with_panel_nb(p.panel_nb)
                .with_gemm_thresholds(GemmThresholds {
                    scalar_gate: p.scalar_gate,
                    par_gemm: p.par_gemm,
                    par_cdiv: p.par_cdiv,
                })
                .with_use_gemm_schur(p.use_gemm_schur)
                .with_method(p.method)
                .with_threads(p.threads);

            let Some((fac_ms, fill, peak_mb, res)) = measure_one(&entry.mat, &s) else { continue };

            let rec = serde_json::json!({
                "matrix": entry.name, "n": n, "nnz": nnz, "dtype": "complex128",
                "features": feat_json,
                "params": {
                    "ordering": ordering_name(p.ordering), "nemin": p.nemin,
                    "relax_width": p.relax_width, "panel_nb": p.panel_nb,
                    "scalar_gate": p.scalar_gate, "par_gemm": p.par_gemm, "par_cdiv": p.par_cdiv,
                    "use_gemm_schur": p.use_gemm_schur, "method": method_name(p.method),
                    "threads": p.threads,
                },
                "metrics": {
                    "factor_ms": fac_ms, "factor_nnz": fill, "peak_mb": peak_mb,
                    "est_transient_mb": combo_est, "residual": res,
                },
            });
            writeln!(out, "{}", rec).expect("write rec");
            n_records += 1;
        }
    }
    eprintln!(
        "[sweep] done: {} records, {} skipped (mem cap) -> {}",
        n_records, n_skipped_mem, out_path
    );
}
