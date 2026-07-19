//! Left-looking factorization bit-identity across thread counts and across
//! repeated runs at the same thread count.
//!
//! Regression test for the racy tiled-cmod mode pick: the `chain_phase`
//! signal (nodes currently in flight, a timing artifact) must never select
//! between the tiled and the sequential cmod path, because the sequential
//! path routes sub-`scalar_gate` updates through the scalar kernel (plain
//! mul+add) while the tiled path runs everything through FMA GEMM
//! micro-kernels - different rounding, different bits. The 3D grid below
//! places its separator nodes squarely in the once-racy dispatch zone
//! (`cmod_flops` between `par_gemm` and the deterministic fork gate), where
//! the drift was measured at up to ~1800 last-ulp entries between 1 and 8
//! threads and a few hundred entries run-to-run at 8 threads.

use num_complex::Complex;
use rslab::{factor_sparse_ldlt_with, CscMatrix, GeneralCsc, LuSolver, SolverSettings};

/// 3D 7-point Laplacian (k³ grid, SPD, lower triangle).
fn grid3d(k: usize) -> CscMatrix<f64> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                r.push(p);
                c.push(p);
                v.push(6.0);
                let mut nb = |q: usize| {
                    let (hi, lo) = if p >= q { (p, q) } else { (q, p) };
                    r.push(hi);
                    c.push(lo);
                    v.push(-1.0);
                };
                if x + 1 < k {
                    nb(idx(x + 1, y, z));
                }
                if y + 1 < k {
                    nb(idx(x, y + 1, z));
                }
                if z + 1 < k {
                    nb(idx(x, y, z + 1));
                }
            }
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

/// 3D convection-diffusion grid (unsymmetric, general CSC).
fn grid3d_conv(k: usize) -> GeneralCsc<f64> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                r.push(p);
                c.push(p);
                v.push(6.0);
                let mut nb = |q: usize, down: f64, up: f64| {
                    r.push(q);
                    c.push(p);
                    v.push(down);
                    r.push(p);
                    c.push(q);
                    v.push(up);
                };
                if x + 1 < k {
                    nb(idx(x + 1, y, z), -1.3, -0.7);
                }
                if y + 1 < k {
                    nb(idx(x, y + 1, z), -1.2, -0.8);
                }
                if z + 1 < k {
                    nb(idx(x, y, z + 1), -1.1, -0.9);
                }
            }
        }
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

fn bits_f64(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn bits_c64(x: &[Complex<f64>]) -> Vec<(u64, u64)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

/// Grid edge sizes: release runs the sizes the original drift was measured
/// at; debug (the CI profile) shrinks the f64 grids so the numeric-heavy
/// factors fit the runner budget. Every size below was verified to
/// reproduce the original racy-dispatch drift in its profile (the shrunk
/// debug sizes included), so the test discriminates on both.
const K_LDLT: usize = if cfg!(debug_assertions) { 16 } else { 24 };
const K_LDLT_C: usize = 16;
const K_LU: usize = if cfg!(debug_assertions) { 14 } else { 22 };

#[test]
fn ll_ldlt_bit_identical_across_threads_and_runs() {
    let a = grid3d(K_LDLT);
    let with = |t: usize| SolverSettings::default().with_threads(t);
    let f1 = factor_sparse_ldlt_with(&a, &with(1)).unwrap();
    let f8 = factor_sparse_ldlt_with(&a, &with(8)).unwrap();
    assert_eq!(
        bits_f64(&f1.l_values),
        bits_f64(&f8.l_values),
        "LL LDLT L differs between 1 and 8 threads"
    );
    assert_eq!(bits_f64(&f1.d_diag), bits_f64(&f8.d_diag));
    for _ in 0..3 {
        let fr = factor_sparse_ldlt_with(&a, &with(8)).unwrap();
        assert_eq!(
            bits_f64(&f8.l_values),
            bits_f64(&fr.l_values),
            "LL LDLT L differs run-to-run at 8 threads"
        );
    }
}

#[test]
fn ll_ldlt_complex_bit_identical_across_threads() {
    // Complex-typed variant of the same grid: the complex kernels dispatch
    // through the same racy-prone gates.
    let ar = grid3d(K_LDLT_C);
    let a = CscMatrix::<Complex<f64>> {
        n: ar.n,
        col_ptr: ar.col_ptr.clone(),
        row_idx: ar.row_idx.clone(),
        values: ar
            .values
            .iter()
            .map(|&v| Complex::new(v, 0.1 * v))
            .collect(),
    };
    let with = |t: usize| SolverSettings::default().with_threads(t);
    let f1 = factor_sparse_ldlt_with(&a, &with(1)).unwrap();
    let f8 = factor_sparse_ldlt_with(&a, &with(8)).unwrap();
    assert_eq!(
        bits_c64(&f1.l_values),
        bits_c64(&f8.l_values),
        "LL complex LDLT L differs between 1 and 8 threads"
    );
}

#[test]
fn ll_lu_bit_identical_across_threads_and_runs() {
    let a = grid3d_conv(K_LU);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let solve = |t: usize| -> Vec<f64> {
        let s = SolverSettings::default().with_threads(t);
        LuSolver::<f64>::factor(&a, &s).unwrap().solve(&b).unwrap()
    };
    let x1 = solve(1);
    let x8 = solve(8);
    assert_eq!(
        bits_f64(&x1),
        bits_f64(&x8),
        "LL LU solution differs between 1 and 8 threads"
    );
    for _ in 0..3 {
        let xr = solve(8);
        assert_eq!(
            bits_f64(&x8),
            bits_f64(&xr),
            "LL LU solution differs run-to-run at 8 threads"
        );
    }
}
