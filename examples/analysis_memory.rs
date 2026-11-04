//! Heap peak and wall time of the LU analysis by stage, on the near-field
//! pattern of a planar MoM system (`matgen::bem::near_field`), analysed as
//! rapidmom does: nested dissection, no row matching, zero `f32` values.
//!
//! A counting allocator tracks the live heap; every debug record of the
//! analysis (one per finished stage) prints the peak since the previous one
//! and the live heap, both above the input matrix, so the peak of a stage
//! is the line that closes it. The timed reps run with the records off.
//! The fingerprint (permutation, fronts, levels, exact `L` + `U` size)
//! must not change under a memory change.
//!
//! `cargo run --release --features matgen --example analysis_memory -- <n> <per_col> [reps] [threads] [ordering]`
use rslab::logging::{LogLevel, LogSink};
use rslab::{GeneralCsc, LuSymbolic, OrderingMethod, SolverSettings, Threads};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

impl Counting {
    fn grow(by: usize) {
        let live = LIVE.fetch_add(by, Relaxed) + by;
        PEAK.fetch_max(live, Relaxed);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        Counting::grow(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        Counting::grow(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        if size >= l.size() {
            Counting::grow(size - l.size());
        } else {
            LIVE.fetch_sub(l.size() - size, Relaxed);
        }
        System.realloc(p, l, size)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const GB: f64 = (1u64 << 30) as f64;

/// The live heap before the analysis: the input matrix.
static BASE: AtomicUsize = AtomicUsize::new(0);

/// Restart the peak at the live heap; the peak and live heap above the
/// input before the restart, in GB.
fn mark() -> (f64, f64) {
    let base = BASE.load(Relaxed);
    let live = LIVE.load(Relaxed);
    let peak = PEAK.swap(live, Relaxed);
    let above = |b: usize| b.saturating_sub(base) as f64 / GB;
    (above(peak), above(live))
}

struct StageSink(Instant);

impl LogSink for StageSink {
    fn emit(&self, _: LogLevel, msg: &str) {
        let (peak, live) = mark();
        println!(
            "{:8.2} s  peak {peak:6.2} GB  live {live:6.2} GB  {msg}",
            self.0.elapsed().as_secs_f64()
        );
    }
}

/// `VmHWM` / `VmRSS` of the process in GB (Linux; 0 elsewhere).
fn rss(key: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb * 1024.0 / GB)
}

/// FNV-1a over the words of the analysis's result.
fn fingerprint(s: &LuSymbolic) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    s.permutation().iter().for_each(|&p| mix(p as u64));
    for (nc, nr) in s.front_dims() {
        mix(nc as u64);
        mix(nr as u64);
    }
    s.level_widths().iter().for_each(|&w| mix(w as u64));
    mix(s.estimate_memory::<num_complex::Complex<f64>>().factor_nnz);
    h
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let n: usize = arg(1).and_then(|s| s.parse().ok()).unwrap_or(60_000);
    let per_col: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(900);
    let reps: usize = arg(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let threads: usize = arg(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let ordering = match arg(5).as_deref() {
        Some("auto") => OrderingMethod::Auto,
        Some("amd") => OrderingMethod::Amd,
        _ => OrderingMethod::MetisND,
    };
    let t = Instant::now();
    let a: GeneralCsc<f32> = rslab::matgen::bem::near_field(n, per_col, 0.2, 1);
    let nnz = a.row_idx.len();
    println!(
        "pattern n={n} nnz={nnz} ({:.0} per column) in {:.1} s, input {:.2} GB",
        nnz as f64 / n as f64,
        t.elapsed().as_secs_f64(),
        LIVE.load(Relaxed) as f64 / GB
    );
    let opts = SolverSettings::exact()
        .with_matching(false)
        .with_ordering(ordering)
        .with_threads(Threads::Fixed(threads));

    BASE.store(LIVE.load(Relaxed), Relaxed);
    mark();
    rslab::logging::set_sink(Box::new(StageSink(Instant::now())));
    rslab::logging::set_level(LogLevel::Debug);
    let t = Instant::now();
    let s = LuSymbolic::analyze(&a, &opts).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    rslab::logging::set_level(LogLevel::Warning);
    let (_, held) = mark();
    println!(
        "analysis {ms:.0} ms (records on), holds {held:.2} GB (heap_bytes {:.2} GB), \
         VmHWM {:.2} GB, VmRSS {:.2} GB, fingerprint {:016x}",
        s.heap_bytes() as f64 / GB,
        rss("VmHWM"),
        rss("VmRSS"),
        fingerprint(&s)
    );
    drop(s);

    // The whole analysis' peak, with the records off.
    BASE.store(LIVE.load(Relaxed), Relaxed);
    mark();
    let mut times = Vec::new();
    for r in 0..reps {
        let t = Instant::now();
        let s = LuSymbolic::analyze(&a, &opts).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
        let (peak, _) = mark();
        println!(
            "rep {r}: {:.0} ms, peak {peak:.2} GB above the input, fingerprint {:016x}",
            times[r],
            fingerprint(&s)
        );
    }
    times.sort_by(f64::total_cmp);
    if let Some(&best) = times.first() {
        println!("best {best:.0} ms, median {:.0} ms", times[times.len() / 2]);
    }
}
