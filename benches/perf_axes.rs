//! Focused performance benchmark for the issue-#2 axes on real SuiteSparse
//! matrices (uses the same download cache as `validate_axes`). Reports factor
//! time + peak transient memory (a live-bytes counting allocator) for:
//!
//! * factor **method**: left-looking (default) vs multifrontal vs right-looking,
//! * **thread scaling** of the multifrontal path (front + tree parallelism),
//! * **wide multi-RHS solve**: parallel `solve_many` vs per-column serial solves,
//! * **32-bit compression**: stored index footprint full vs `u32`.
//!
//! Run:  `cargo bench --bench perf_axes --features matgen-download`

use num_complex::Complex;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

type C = Complex<f64>;

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
            let _ = LIVE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| Some(v.saturating_sub(l.size())));
        }
        System.dealloc(p, l);
    }
}
#[global_allocator]
static A: Counting = Counting;
fn meter_reset() { LIVE.store(0, Ordering::Relaxed); PEAK.store(0, Ordering::Relaxed); ON.store(true, Ordering::Relaxed); }
fn meter_peak_mb() -> f64 { ON.store(false, Ordering::Relaxed); PEAK.load(Ordering::Relaxed) as f64 / 1e6 }

#[cfg(feature = "matgen-download")]
const MATS: &[(&str, &str)] = &[
    ("Boeing", "bcsstk36"), ("HB", "bcsstk38"), ("Cylshell", "s3dkt3m2"),
    ("Nasa", "nasasrb"), ("Boeing", "ct20stif"), ("GHS_psdef", "cfd1"),
    ("Cunningham", "qa8fm"), ("Simon", "raefsky4"), ("GHS_psdef", "oilpan"),
    ("Oberwolfach", "gyro_k"), ("Boeing", "pwtk"), ("GHS_psdef", "bmwcra_1"),
];

#[cfg(feature = "matgen-download")]
fn main() {
    use rslab::{
        solve_ldlt, solve_ldlt_many, CompressedLdltFactors, CscMatrix, FactorMethod, LdltSymbolic,
        SolverSettings,
    };

    let threads: usize = std::env::var("RLA_PERF_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let cap_mb: f64 = std::env::var("RLA_PERF_MEM_CAP_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(6000.0);

    // best-of-`reps` factor time (ms) + peak MB under settings `s`.
    fn timed(sym: &LdltSymbolic, a: &CscMatrix<C>, s: &SolverSettings, reps: usize) -> Option<(f64, f64)> {
        let mut best = f64::INFINITY;
        let mut peak = 0.0;
        for _ in 0..reps {
            meter_reset();
            let t = Instant::now();
            let f = sym.factor(a, s).ok()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            peak = meter_peak_mb();
            std::hint::black_box(&f);
            best = best.min(ms);
        }
        Some((best, peak))
    }

    println!("{:<14} {:>8} {:>10} | {:>18} | {:>18} | {:>18} | {:>14} | {:>16}",
        "matrix", "n", "nnz", "LL ms / MB", "MF ms / MB", "RL ms / MB", "MF 1t/8t", "solve many/1x");
    println!("{}", "-".repeat(130));

    for &(group, name) in MATS {
        let Ok(path) = rslab::matgen::download::fetch(group, name) else { eprintln!("skip {name}: fetch"); continue; };
        let a = match rslab::read_mtx_any(&path) {
            Ok(rslab::MtxLoaded::Symmetric(a)) => a,
            _ => { eprintln!("skip {name}: not symmetric"); continue; }
        };
        if a.n == 0 { continue; }
        let Ok(sym) = LdltSymbolic::analyze(&a) else { continue; };
        let est = sym.estimate_memory::<C>();
        if (est.transient_peak_bytes.max(est.mf_transient_peak_bytes) as f64 / 1e6) > cap_mb {
            eprintln!("skip {name}: over mem cap"); continue;
        }
        let base = SolverSettings::default().with_threads(threads);
        let ll = timed(&sym, &a, &base.clone().with_method(FactorMethod::LeftLooking), 3);
        let mf = timed(&sym, &a, &base.clone().with_method(FactorMethod::Multifrontal), 3);
        let rl = timed(&sym, &a, &base.clone().with_method(FactorMethod::RightLooking), 3);
        // Thread scaling of the multifrontal path.
        let mf1 = timed(&sym, &a, &SolverSettings::default().with_threads(1).with_method(FactorMethod::Multifrontal), 2);
        let scaling = match (mf1, mf) {
            (Some((t1, _)), Some((t8, _))) if t8 > 0.0 => format!("{:.2}x", t1 / t8),
            _ => "-".into(),
        };
        // Wide multi-RHS solve: parallel block vs per-column serial. Uses the raw
        // factor (public LdltFactors) so the solve kernels can be called directly.
        let solve_str = rslab::factor_sparse_ldlt_with(&a, &base).ok().map(|f| {
            let nrhs = 256usize;
            let b: Vec<C> = (0..a.n * nrhs).map(|k| C::new((k % 11) as f64 - 5.0, 0.2)).collect();
            let t = Instant::now();
            let _ = std::hint::black_box(solve_ldlt_many(&f, &b, nrhs));
            let par = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            for c in 0..nrhs {
                let bc: Vec<C> = (0..a.n).map(|i| b[i * nrhs + c]).collect();
                let _ = std::hint::black_box(solve_ldlt(&f, &bc));
            }
            let ser = t.elapsed().as_secs_f64() * 1e3;
            format!("{:.2}x", ser / par.max(1e-9))
        }).unwrap_or_else(|| "-".into());
        // Compression memory (raw factor).
        let comp_str = rslab::factor_sparse_ldlt_with(&a, &base).ok().and_then(|f| {
            let full = 8 * (f.l_col_ptr.len() + f.l_row_idx.len() + f.perm.len());
            CompressedLdltFactors::from_factors(f)
                .map(|c| format!("{:.0}->{:.0}KB", full as f64 / 1024.0, c.index_bytes() as f64 / 1024.0))
        }).unwrap_or_else(|| "-".into());

        let fmt = |o: Option<(f64, f64)>| o.map(|(m, mb)| format!("{m:>8.1} /{mb:>6.0}")).unwrap_or_else(|| "     -    ".into());
        println!("{:<14} {:>8} {:>10} | {:>18} | {:>18} | {:>18} | {:>14} | {:>16}",
            name, a.n, a.row_idx.len(), fmt(ll), fmt(mf), fmt(rl), scaling, solve_str);
        eprintln!("  {name}: index {comp_str}");
    }
    println!("\nLL=left-looking (default), MF=multifrontal, RL=right-looking. ms=best-of-3 factor, MB=peak transient.");
    println!("MF 1t/8t = multifrontal thread scaling. solve many/1x = parallel wide-RHS (256) speedup vs per-column serial.");
}

#[cfg(not(feature = "matgen-download"))]
fn main() { eprintln!("perf_axes requires --features matgen-download"); }
