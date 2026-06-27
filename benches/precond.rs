//! Preconditioner memory ↔ convergence tradeoff for 3D complex-symmetric
//! systems — the 3D EM FEM / MOM use case.
//!
//! For one 3D 7-point complex-symmetric grid we factor several preconditioner
//! configurations and drive COCG with each, reporting factor fill (memory),
//! factor time, and the COCG iteration count + residual. This makes the
//! memory-reduction levers measurable:
//!   * f64 complete      — the exact reference factor.
//!   * f64 incomplete(τ) — threshold dropping; less fill, more iterations.
//!   * f32 complete      — half the bytes per entry (mixed precision).
//!   * f32 + incomplete  — the aggressive low-memory preconditioner.
//!
//! Run: `cargo bench --bench precond`.

use std::time::Instant;

use feral::sparse::csc::CscMatrix;
use feral::{
    cocg, GenericFactorOptions, LowPrecisionPreconditioner, NoPreconditioner, Preconditioner,
    SparseSymmetricLdlt, ZeroPivotAction,
};
use num_complex::Complex;

type C = Complex<f64>;

/// 3D 7-point grid (k×k×k), complex-symmetric, lower triangle.
fn grid3d(k: usize, diag: C, off: C) -> CscMatrix<C> {
    let n = k * k * k;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut push = |p: usize, q: usize, val: C| {
        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
        r.push(hi);
        c.push(lo);
        v.push(val);
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
    CscMatrix::<C>::from_triplets(n, &r, &c, &v).unwrap()
}

fn residual(a: &CscMatrix<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![C::default(); a.n];
    a.symv(x, &mut ax);
    (0..a.n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max)
}

/// Report one preconditioner: fill, bytes, factor time, COCG iters, residual.
fn report<M: Preconditioner<C>>(
    label: &str,
    a: &CscMatrix<C>,
    b: &[C],
    factor_ms: f64,
    nnz: usize,
    entry_bytes: usize,
    m: &M,
) {
    let t = Instant::now();
    let res = cocg(a, b, m, 1e-8, 2000).unwrap();
    let cocg_ms = t.elapsed().as_secs_f64() * 1e3;
    // L storage ≈ nnz · (value bytes + 8-byte row index).
    let mb = nnz as f64 * (entry_bytes + 8) as f64 / 1e6;
    println!(
        "{label:22} fill={nnz:9}  mem={mb:7.2} MB  factor={factor_ms:8.2} ms  \
         iters={:4}  cocg={cocg_ms:7.2} ms  res={:.1e}{}",
        res.iters,
        residual(a, &res.x, b),
        if res.converged { "" } else { "  (NO CONV)" },
    );
}

fn main() {
    let k = 24;
    let a = grid3d(k, Complex::new(6.0, 1.0), Complex::new(-1.0, 0.1));
    let n = a.n;
    let b: Vec<C> = (0..n)
        .map(|i| Complex::new((i % 7) as f64 - 3.0, 0.5))
        .collect();
    println!(
        "3D complex-symmetric preconditioning — grid {k}³ = {n} unknowns, nnz(A,lower)={}\n",
        a.values.len()
    );

    // Baseline: unpreconditioned COCG.
    let t = Instant::now();
    let r0 = cocg(&a, &b, &NoPreconditioner, 1e-8, 20000).unwrap();
    println!(
        "{:22} fill={:9}  mem={:7.2} MB  factor={:8.2} ms  iters={:4}  cocg={:7.2} ms  res={:.1e}{}\n",
        "unpreconditioned",
        0,
        0.0,
        0.0,
        r0.iters,
        t.elapsed().as_secs_f64() * 1e3,
        residual(&a, &r0.x, &b),
        if r0.converged { "" } else { "  (NO CONV)" },
    );

    let fail = ZeroPivotAction::Fail;
    let configs: [(&str, GenericFactorOptions); 3] = [
        ("f64 complete", GenericFactorOptions { on_zero_pivot: fail.clone(), drop_tol: None }),
        ("f64 incomplete τ=1e-2", GenericFactorOptions { on_zero_pivot: fail.clone(), drop_tol: Some(1e-2) }),
        ("f64 incomplete τ=5e-2", GenericFactorOptions { on_zero_pivot: fail.clone(), drop_tol: Some(5e-2) }),
    ];
    for (label, opts) in &configs {
        let t = Instant::now();
        let m = SparseSymmetricLdlt::factor_with(&a, opts).unwrap();
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        report(label, &a, &b, factor_ms, m.factor_nnz(), 16, &m);
    }

    // Mixed precision: Complex<f32> factor (8 bytes/entry).
    let f32_configs: [(&str, GenericFactorOptions); 2] = [
        ("f32 complete", GenericFactorOptions { on_zero_pivot: fail.clone(), drop_tol: None }),
        ("f32 incomplete τ=5e-2", GenericFactorOptions { on_zero_pivot: fail.clone(), drop_tol: Some(5e-2) }),
    ];
    for (label, opts) in &f32_configs {
        let t = Instant::now();
        let m = LowPrecisionPreconditioner::factor(&a, opts).unwrap();
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        report(label, &a, &b, factor_ms, m.factor_nnz(), 8, &m);
    }
}
