//! Complex-symmetric head-to-head: RLA vs faer.
//!
//! RLA factors `A = LDLᵀ` exploiting `A = Aᵀ` (PARDISO mtype 6), storing only
//! the lower triangle and a single factor `L`. faer has no complex-*symmetric*
//! sparse path (its sparse Cholesky/BK is Hermitian), so the fair analogue is
//! faer's sparse **LU** on the *full* matrix — which cannot exploit symmetry
//! and stores both `L` and `U`. This bench reports factor time, solve time,
//! residual, and factor fill so the symmetry advantage is visible.
//!
//! Run: `cargo bench --bench vs_faer` (or `cargo run --release --bench vs_faer`).

use std::time::Instant;

use feral::sparse::csc::CscMatrix;
use feral::SparseSymmetricLdlt;
use num_complex::Complex;

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat};

type C = Complex<f64>;

/// 2D 5-point grid (m×m, n=m²), complex-symmetric. Lower-triangle triplets.
fn grid_lower(m: usize, diag: C, off: C) -> (usize, Vec<usize>, Vec<usize>, Vec<C>) {
    let n = m * m;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
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
    (n, rows, cols, vals)
}

fn residual_inf(a: &CscMatrix<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![Complex::new(0.0, 0.0); a.n];
    a.symv(x, &mut ax);
    (0..a.n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max)
}

fn main() {
    // Fair comparison: RLA's multifrontal driver is rayon-parallel by default,
    // so let faer use all cores too (its default is sequential).
    faer::set_global_parallelism(faer::Par::rayon(0));
    println!("Complex-symmetric direct solve: RLA LDLᵀ vs faer sparse LU (both parallel)\n");
    println!(
        "{:>6} {:>9} {:>9}  {:>9} {:>9} {:>8}  {:>9} {:>9} {:>8}  {:>7}",
        "n", "nnzA/2", "nnzA", "RLA_fac", "RLA_slv", "RLA_res", "faer_fac", "faer_slv", "faer_res", "fac×"
    );

    let diag = Complex::new(4.0, 1.0);
    let off = Complex::new(-1.0, 0.1);
    for &m in &[40usize, 60, 80, 100, 140, 180] {
        let (n, rows, cols, vals) = grid_lower(m, diag, off);
        let nnz_lower = vals.len();
        let b: Vec<C> = (0..n)
            .map(|i| Complex::new((i % 7) as f64 - 3.0, 1.0))
            .collect();

        // ---- RLA: lower triangle, LDLᵀ ----
        let a = CscMatrix::<C>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let t = Instant::now();
        let solver = SparseSymmetricLdlt::factor(&a).unwrap();
        let rla_fac = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let x = solver.solve(&b).unwrap();
        let rla_slv = t.elapsed().as_secs_f64() * 1e3;
        let rla_res = residual_inf(&a, &x, &b);
        let rla_nnz = solver.factor_nnz();

        // ---- faer: full matrix, sparse LU ----
        let mut trip: Vec<Triplet<usize, usize, c64>> = Vec::with_capacity(2 * nnz_lower);
        for k in 0..nnz_lower {
            let (i, j, v) = (rows[k], cols[k], vals[k]);
            trip.push(Triplet::new(i, j, v));
            if i != j {
                trip.push(Triplet::new(j, i, v));
            }
        }
        let nnz_full = trip.len();
        let fa = SparseColMat::<usize, c64>::try_new_from_triplets(n, n, &trip).unwrap();
        let t = Instant::now();
        let lu = fa.sp_lu().unwrap();
        let faer_fac = t.elapsed().as_secs_f64() * 1e3;
        let mut xb = Mat::<c64>::from_fn(n, 1, |i, _| b[i]);
        let t = Instant::now();
        lu.solve_in_place(&mut xb);
        let faer_slv = t.elapsed().as_secs_f64() * 1e3;
        let xf: Vec<C> = (0..n).map(|i| xb[(i, 0)]).collect();
        let faer_res = residual_inf(&a, &xf, &b);

        println!(
            "{:>6} {:>9} {:>9}  {:>9.2} {:>9.3} {:>8.0e}  {:>9.2} {:>9.3} {:>8.0e}  {:>6.2}x   nnzL(RLA)={}",
            n,
            nnz_lower,
            nnz_full,
            rla_fac,
            rla_slv,
            rla_res,
            faer_fac,
            faer_slv,
            faer_res,
            faer_fac / rla_fac,
            rla_nnz,
        );
    }
}
