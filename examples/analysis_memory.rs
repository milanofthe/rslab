//! Heap peak and wall time of the LU analysis by stage, on the near-field
//! pattern of a planar MoM system (`matgen::bem::near_field`), analysed as
//! a MoM solver does: nested dissection, no row matching, zero `f32` values.
//!
//! A counting allocator tracks the live heap and its peak above the input
//! matrix, behind `rsprobe::Alloc`: built with `--features rsprobe/alloc`,
//! the first analysis records its stages at DEBUG with the bytes allocated
//! and freed in each, written to the path in `RSPROBE_RECORD` for
//! `python -m rsprobe report`. The timed reps run without a context.
//! The fingerprint (permutation, fronts, levels, exact `L` + `U` size)
//! must not change under a memory change.
//!
//! `cargo run --release --features matgen,rsprobe/alloc --example analysis_memory -- <n> <per_col> [reps] [threads] [ordering]`
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
static COUNTING: rsprobe::Alloc<Counting> = rsprobe::Alloc::wrap(Counting);

const GB: f64 = (1u64 << 30) as f64;

static INPUT_HEAP: AtomicUsize = AtomicUsize::new(0);

fn restart_peak() -> (f64, f64) {
    let base = INPUT_HEAP.load(Relaxed);
    let live = LIVE.load(Relaxed);
    let peak = PEAK.swap(live, Relaxed);
    let above = |b: usize| b.saturating_sub(base) as f64 / GB;
    (above(peak), above(live))
}

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

    INPUT_HEAP.store(LIVE.load(Relaxed), Relaxed);
    restart_peak();
    let ctx = rsprobe::Context::builder()
        .threshold(rsprobe::Level::DEBUG)
        .build();
    ctx.param("n", n);
    ctx.param("nnz", nnz);
    ctx.param("threads", threads);
    let t = Instant::now();
    let s = {
        let _on = ctx.enter();
        LuSymbolic::analyze(&a, &opts).unwrap()
    };
    let ms = t.elapsed().as_secs_f64() * 1e3;
    if let Some(path) = std::env::var_os("RSPROBE_RECORD") {
        std::fs::write(path, ctx.record().to_string()).expect("writing the record");
    }
    let (peak, held) = restart_peak();
    println!(
        "analysis {ms:.0} ms (recorded), peak {peak:.2} GB, holds {held:.2} GB (heap_bytes {:.2} GB), \
         VmHWM {:.2} GB, VmRSS {:.2} GB, fingerprint {:016x}",
        s.heap_bytes() as f64 / GB,
        rss("VmHWM"),
        rss("VmRSS"),
        fingerprint(&s)
    );
    drop(s);

    INPUT_HEAP.store(LIVE.load(Relaxed), Relaxed);
    restart_peak();
    let mut times = Vec::new();
    for r in 0..reps {
        let t = Instant::now();
        let s = LuSymbolic::analyze(&a, &opts).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
        let (peak, _) = restart_peak();
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
