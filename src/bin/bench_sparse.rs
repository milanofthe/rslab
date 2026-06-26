//! Ad-hoc performance baseline for the generic sparse solver.
//!
//! Builds 2D 5-point-Laplacian-style symmetric matrices of growing size and
//! times factor + solve for both `f64` and `Complex<f64>`. Run with:
//! `cargo run --release --bin bench_sparse`.
//!
//! This measures the *correctness-first* generic path (unblocked fronts, no
//! SIMD, within-block pivoting); it is a baseline to track, not an optimized
//! number. Compare against PARDISO / feral's f64 driver separately.

use std::time::Instant;

use feral::sparse::csc::CscMatrix;
use feral::FeralError;
use feral::SparseSymmetricLdlt;
use num_complex::Complex;

/// Build a 2D 5-point grid (m×m, n=m²) with the given diagonal and neighbor
/// values. Lower triangle only.
fn grid<T: Copy>(m: usize, diag: T, off: T) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let idx = |r: usize, c: usize| r * m + c;
    for r in 0..m {
        for c in 0..m {
            let p = idx(r, c);
            rows.push(p);
            cols.push(p);
            vals.push(diag);
            if c + 1 < m {
                let q = idx(r, c + 1);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(off);
            }
            if r + 1 < m {
                let q = idx(r + 1, c);
                let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                rows.push(hi);
                cols.push(lo);
                vals.push(off);
            }
        }
    }
    (rows, cols, vals)
}

fn bench_one<T>(label: &str, m: usize, diag: T, off: T, b_fn: impl Fn(usize) -> T)
where
    T: feral::Scalar,
{
    let n = m * m;
    let (rows, cols, vals) = grid(m, diag, off);
    let a = match CscMatrix::<T>::from_triplets(n, &rows, &cols, &vals) {
        Ok(a) => a,
        Err(e) => {
            println!("{label} m={m} build error: {e}");
            return;
        }
    };
    let nnz = a.nnz();
    let b: Vec<T> = (0..n).map(&b_fn).collect();

    let t0 = Instant::now();
    let solver = match SparseSymmetricLdlt::factor(&a) {
        Ok(s) => s,
        Err(e) => {
            println!("{label} m={m} factor error: {e}");
            return;
        }
    };
    let factor_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let x = match solver.solve(&b) {
        Ok(x) => x,
        Err(e) => {
            println!("{label} m={m} solve error: {e}");
            return;
        }
    };
    let solve_ms = t1.elapsed().as_secs_f64() * 1e3;

    // Residual ‖Ax − b‖∞.
    let mut ax = vec![T::zero(); n];
    a.symv(&x, &mut ax);
    let res = (0..n)
        .map(|i| (ax[i] - b[i]).magnitude())
        .fold(0.0, f64::max);

    println!(
        "{label:8} n={n:6} nnz={nnz:8}  factor={factor_ms:9.2} ms  solve={solve_ms:8.3} ms  res={res:.1e}"
    );
}

fn main() -> Result<(), FeralError> {
    println!("RLA generic sparse solver — baseline (release)\n");
    let sizes = [20usize, 40, 60, 80, 100];

    println!("== f64 (diag 4, off -1) ==");
    for &m in &sizes {
        bench_one::<f64>("f64", m, 4.0, -1.0, |i| (i % 7) as f64 - 3.0);
    }

    println!("\n== Complex<f64> (diag 4+i, off -1+0.1i) ==");
    let c = |re, im| Complex::new(re, im);
    for &m in &sizes {
        bench_one::<Complex<f64>>("complex", m, c(4.0, 1.0), c(-1.0, 0.1), move |i| {
            c((i % 7) as f64 - 3.0, 1.0)
        });
    }

    Ok(())
}
