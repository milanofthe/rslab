//! Transient-memory comparison of the two LDLᵀ factor methods on a
//! complex-symmetric 3D grid: multifrontal (CB stack + per-front extract) vs
//! supernodal left-looking (panels only — no CB stack, no extract). Reports the
//! peak working-set sampled *during* each factorization and the factor time.
//!
//! The left-looking path is currently serial + scalar, so it is slower; this
//! bench isolates the **memory transient**, the reason it exists.
//!
//! Run: `cargo bench --bench leftlook_mem`.

use std::time::Instant;

use num_complex::Complex;
use rla::prelude::*;
use rla::{factor_sparse_ldlt_with, FactorMethod, FactorOptions};

type C = Complex<f64>;

#[cfg(windows)]
fn cur_ws_mb() -> f64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        pfc: u32,
        peak_ws: usize,
        ws: usize,
        q1: usize,
        q2: usize,
        q3: usize,
        q4: usize,
        pf: usize,
        peak_pf: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(p: isize, c: *mut Pmc, cb: u32) -> i32;
    }
    // SAFETY: POD output buffer of the documented PROCESS_MEMORY_COUNTERS size.
    unsafe {
        let mut pmc: Pmc = std::mem::zeroed();
        pmc.cb = std::mem::size_of::<Pmc>() as u32;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
            pmc.ws as f64 / 1e6
        } else {
            0.0
        }
    }
}
#[cfg(not(windows))]
fn cur_ws_mb() -> f64 {
    0.0
}

struct Sampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak: std::sync::Arc<std::sync::atomic::AtomicU64>,
    h: Option<std::thread::JoinHandle<()>>,
}
impl Sampler {
    fn start() -> Self {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let (s, p) = (stop.clone(), peak.clone());
        let h = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                p.fetch_max(cur_ws_mb() as u64, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        });
        Sampler {
            stop,
            peak,
            h: Some(h),
        }
    }
    fn stop(mut self) -> f64 {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
        self.peak.load(Ordering::Relaxed) as f64
    }
}

/// Complex-symmetric 3D 7-point grid (k³), lower triangle.
fn grid3d(k: usize, diag: C, off: C) -> CscMatrix<C> {
    let n = k * k * k;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut push = |p: usize, q: usize, v: C| {
        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
        rows.push(hi);
        cols.push(lo);
        vals.push(v);
    };
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                push(p, p, diag);
                if x + 1 < k {
                    push(p, idx(x + 1, y, z), off);
                }
                if y + 1 < k {
                    push(p, idx(x, y + 1, z), off);
                }
                if z + 1 < k {
                    push(p, idx(x, y, z + 1), off);
                }
            }
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

fn run(k: usize) {
    let a = grid3d(k, Complex::new(6.0, 1.0), Complex::new(-1.0, 0.1));
    let n = a.n;
    let base = cur_ws_mb();
    for (label, method) in [
        ("multifrontal ", FactorMethod::Multifrontal),
        ("left-looking ", FactorMethod::LeftLooking),
    ] {
        let opts = FactorOptions::default().with_method(method);
        let sampler = Sampler::start();
        let t = Instant::now();
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let peak = sampler.stop();
        println!(
            "  k={k:2} n={n:6}  {label}  factor {ms:8.1} ms   peak-WS {peak:6.0} MB  (transient +{:.0})   nnz(L)={}",
            peak - base,
            f.l_values.len(),
        );
    }
    println!();
}

fn main() {
    println!("LDLᵀ transient-memory: multifrontal (CB stack + extract) vs left-looking (panels only)\n");
    for &k in &[18usize, 24, 30] {
        run(k);
    }
}
