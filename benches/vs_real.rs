//! RLA vs faer on real MoM near-field preconditioner matrices.
//!
//! Reads every `*.mtx` in `rapidmom/precond_matrices` (complex `general`),
//! checks how symmetric each matrix is, then factors and solves it with both
//! RLA (complex-symmetric LDLᵀ on the lower triangle, static-pivoted) and faer
//! (sparse LU on the full matrix), reporting factor time, fill, solve time, and
//! the true residual against the full matrix.
//!
//! Run: `cargo bench --bench vs_real`.

use std::collections::HashMap;
use std::time::Instant;

use num_complex::Complex;
use rla::sparse::general::GeneralCsc;
use rla::{
    factor_general_lu, parse_mtx_complex_general, solve_lu, solve_lu_refined, FactorOptions,
    ZeroPivotAction,
};

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat};

type C = Complex<f64>;

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";

/// COO matrix–vector product `y = A x` over the full (general) entry list.
fn matvec(entries: &[(usize, usize, C)], x: &[C], y: &mut [C]) {
    for v in y.iter_mut() {
        *v = C::default();
    }
    for &(i, j, a) in entries {
        y[i] += a * x[j];
    }
}

fn resid(entries: &[(usize, usize, C)], x: &[C], b: &[C]) -> f64 {
    let mut y = vec![C::default(); b.len()];
    matvec(entries, x, &mut y);
    (0..b.len())
        .map(|i| (y[i] - b[i]).norm())
        .fold(0.0, f64::max)
}

fn bench_file(path: &std::path::Path) {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("{name}: read error {e}");
            return;
        }
    };
    let mtx = match parse_mtx_complex_general(&contents, &name) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}: parse error {e}");
            return;
        }
    };
    drop(contents);
    let n = mtx.n;
    let entries = &mtx.entries;
    let nnz = entries.len();

    // Symmetry check: max |A_ij - A_ji| relative to max |A|.
    let mut map: HashMap<(usize, usize), C> = HashMap::with_capacity(nnz);
    let mut amax = 0.0f64;
    for &(i, j, v) in entries {
        map.insert((i, j), v);
        amax = amax.max(v.norm());
    }
    let mut asym = 0.0f64;
    for &(i, j, v) in entries {
        if i != j {
            let t = map.get(&(j, i)).copied().unwrap_or(C::default());
            asym = asym.max((v - t).norm());
        }
    }
    drop(map);
    let rel_asym = if amax > 0.0 { asym / amax } else { 0.0 };

    let b: Vec<C> = vec![C::new(1.0, 0.0); n];

    // ---- RLA: general (unsymmetric) complex multifrontal LU ----
    let (rr, cc, vv): (Vec<usize>, Vec<usize>, Vec<C>) = {
        let mut rr = Vec::with_capacity(nnz);
        let mut cc = Vec::with_capacity(nnz);
        let mut vv = Vec::with_capacity(nnz);
        for &(i, j, v) in entries {
            rr.push(i);
            cc.push(j);
            vv.push(v);
        }
        (rr, cc, vv)
    };
    let g = match GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv) {
        Ok(g) => g,
        Err(e) => {
            println!("{name}: RLA build error {e}");
            return;
        }
    };
    let opts = FactorOptions {
        on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 },
        drop_tol: None,
    };
    let t = Instant::now();
    let rla = match factor_general_lu(&g, &opts) {
        Ok(s) => s,
        Err(e) => {
            println!("{name}: RLA factor error {e}");
            return;
        }
    };
    let rla_fac = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let xr = solve_lu(&rla, &b).unwrap_or_default();
    let rla_slv = t.elapsed().as_secs_f64() * 1e3;
    let rla_res = resid(entries, &xr, &b);
    // With a few iterative-refinement steps against the original matrix.
    let t = Instant::now();
    let xr_ref = solve_lu_refined(&rla, &g, &b, 5).unwrap_or_default();
    let rla_ref_ms = t.elapsed().as_secs_f64() * 1e3;
    let rla_ref_res = resid(entries, &xr_ref, &b);

    // ---- faer: full matrix, sparse LU ----
    let mut trip: Vec<Triplet<usize, usize, c64>> = Vec::with_capacity(nnz);
    for &(i, j, v) in entries {
        trip.push(Triplet::new(i, j, v));
    }
    let (faer_fac, faer_slv, faer_res, faer_ok) =
        match SparseColMat::<usize, c64>::try_new_from_triplets(n, n, &trip) {
            Ok(fa) => {
                let t = Instant::now();
                match fa.sp_lu() {
                    Ok(lu) => {
                        let ff = t.elapsed().as_secs_f64() * 1e3;
                        let mut xb = Mat::<c64>::from_fn(n, 1, |i, _| b[i]);
                        let t = Instant::now();
                        lu.solve_in_place(&mut xb);
                        let fs = t.elapsed().as_secs_f64() * 1e3;
                        let xf: Vec<C> = (0..n).map(|i| xb[(i, 0)]).collect();
                        (ff, fs, resid(entries, &xf, &b), true)
                    }
                    Err(_) => (0.0, 0.0, f64::INFINITY, false),
                }
            }
            Err(_) => (0.0, 0.0, f64::INFINITY, false),
        };

    println!("=== {name} ===");
    println!("  n={n}  nnz(full)={nnz}  rel.asymmetry={rel_asym:.2e}  |A|max={amax:.2e}");
    // Factor memory ≈ nnz(L+U) · (16-byte Complex<f64> value + 8-byte index).
    let rla_mb = rla.factor_nnz() as f64 * 24.0 / 1e6;
    println!(
        "  RLA-LU: factor={rla_fac:8.1} ms  solve={rla_slv:7.2} ms  fill={:9}  mem={rla_mb:7.1} MB  perturbed={}  res={rla_res:.2e}",
        rla.factor_nnz(),
        rla.n_perturbed,
    );
    println!("  RLA-LU+refine(5): {rla_ref_ms:7.2} ms  res={rla_ref_res:.2e}");
    if faer_ok {
        println!("  faer : factor={faer_fac:8.1} ms  solve={faer_slv:7.2} ms  res={faer_res:.2e}");
        if faer_fac > 0.0 {
            println!(
                "  speedup factor (faer/RLA): {:.2}x",
                faer_fac / rla_fac.max(1e-9)
            );
        }
    } else {
        println!("  faer : FAILED");
    }
    println!();
}

fn main() {
    faer::set_global_parallelism(faer::Par::rayon(0));
    let mut files: Vec<_> = match std::fs::read_dir(DIR) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect(),
        Err(e) => {
            println!("cannot read {DIR}: {e}");
            return;
        }
    };
    // Smallest files first.
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    println!("RLA vs faer on MoM near-field preconditioner matrices\n");
    for f in &files {
        bench_file(f);
    }
}
