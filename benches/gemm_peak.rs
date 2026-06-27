//! Raw `gemm`-crate complex throughput ceiling — the reference for how much of
//! the per-front kernel gap is the crate vs how we drive it (panel rank, serial
//! getf2). Times a single large `C -= A·B` complex GEMM at a few shapes.
//!
//! Run: `cargo bench --bench gemm_peak`.

use num_complex::Complex;
use std::time::Instant;

type C = Complex<f64>;

fn bench(m: usize, n: usize, k: usize, par: gemm::Parallelism, label: &str) {
    let a = vec![C::new(1.0, 0.5); m * k];
    let b = vec![C::new(0.7, -0.3); k * n];
    let mut c = vec![C::new(0.0, 0.0); m * n];
    // warm-up
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            m as isize,
            1,
            true,
            a.as_ptr(),
            m as isize,
            1,
            b.as_ptr(),
            k as isize,
            1,
            C::new(1.0, 0.0),
            C::new(-1.0, 0.0),
            false,
            false,
            false,
            par,
        );
    }
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                c.as_mut_ptr(),
                m as isize,
                1,
                true,
                a.as_ptr(),
                m as isize,
                1,
                b.as_ptr(),
                k as isize,
                1,
                C::new(1.0, 0.0),
                C::new(-1.0, 0.0),
                false,
                false,
                false,
                par,
            );
        }
    }
    let s = t.elapsed().as_secs_f64() / reps as f64;
    // complex multiply-add ≈ 8 real flops
    let gflops = 8.0 * m as f64 * n as f64 * k as f64 / s / 1e9;
    println!(
        "  {label:28} {m}x{n}x{k}  {:.2} ms  {gflops:.1} Gflop/s",
        s * 1e3
    );
}

fn main() {
    println!("Raw gemm-crate complex (c64) throughput:\n");
    println!(" -- parallel (Rayon) --");
    for &k in &[64usize, 128, 256, 512] {
        bench(
            2000,
            2000,
            k,
            gemm::Parallelism::Rayon(0),
            &format!("rank-{k}"),
        );
    }
    bench(2000, 2000, 2000, gemm::Parallelism::Rayon(0), "square");
    println!(" -- single-thread --");
    bench(2000, 2000, 64, gemm::Parallelism::None, "rank-64 1T");
    bench(2000, 2000, 512, gemm::Parallelism::None, "rank-512 1T");
}
