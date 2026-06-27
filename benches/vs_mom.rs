//! Head-to-head on the **real complex MoM matrices**: RLA unsymmetric LU vs
//! faer sparse LU, factoring the same `precond_matrices/*.mtx` and solving
//! `A x = b`. Reports factor/solve time, factor fill and the true residual for
//! each, plus the RLA factor-time speedup over faer.
//!
//! RLA solver options are selected via the composable API and toggled by env so
//! the same run can be swept:
//!   * `RLA_EAGER=1`   → MemoryMode::Eager (default is LowMemory)
//!   * `RLA_BLR_CB=1`  → BlrMode::ContributionBlocks (approximate, refine)
//!   * `RLA_NO_LIU=1`  → ReorderMode::Off
//!
//! Run: `cargo bench --bench vs_mom` (optionally `RLA_DIAG_FILTER=spiral`).

use std::time::Instant;

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat};
use num_complex::Complex;
use rla::prelude::*;
use rla::{
    AnalyzeOptions, BlrMode, FactorMethod, FactorOptions, LuSymbolic, MemoryMode, ReorderMode,
};

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
type C = Complex<f64>;

fn rel_resid(a: &rla::GeneralCsc<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![Complex::new(0.0, 0.0); a.n];
    a.matvec(x, &mut ax);
    let num: f64 = (0..a.n).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den
}

fn run(path: &std::path::Path) {
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
    let a = match mtx.to_general_csc() {
        Ok(a) => a,
        Err(e) => {
            println!("{name}: build error {e}");
            return;
        }
    };
    let n = a.n;
    let nnz = a.nnz();
    let b: Vec<C> = (0..n)
        .map(|i| Complex::new((i % 7) as f64 - 3.0, ((i % 5) as f64 - 2.0) * 0.5))
        .collect();

    // ---- RLA unsymmetric LU (composable options via env) ----
    let reorder = if std::env::var("RLA_NO_LIU").is_ok() {
        ReorderMode::Off
    } else {
        ReorderMode::HybridLiu
    };
    let mut opts = FactorOptions::preconditioner(1e-10);
    if std::env::var("RLA_EAGER").is_ok() {
        opts = opts.with_memory(MemoryMode::Eager);
    }
    if std::env::var("RLA_BLR_CB").is_ok() {
        opts = opts.with_blr(BlrMode::contribution_blocks(1e-4));
    }
    if std::env::var("RLA_LEFTLOOKING").is_ok() {
        opts = opts.with_method(FactorMethod::LeftLooking);
    }
    let t = Instant::now();
    let sym = LuSymbolic::analyze_with(&a, &AnalyzeOptions::default().with_reorder(reorder)).unwrap();
    let rla_ana = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let rla = sym.factor(&a, &opts).unwrap();
    let rla_fac = t.elapsed().as_secs_f64() * 1e3;
    let rla_fill = rla.factor_nnz();
    let t = Instant::now();
    let xr = rla.solve(&b).unwrap();
    let rla_slv = t.elapsed().as_secs_f64() * 1e3;
    let rla_res = rel_resid(&a, &xr, &b);

    // ---- faer sparse LU on the same (full, general) matrix ----
    let mut trip: Vec<Triplet<usize, usize, c64>> = Vec::with_capacity(nnz);
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let v = a.values[k];
            trip.push(Triplet::new(i, j, c64::new(v.re, v.im)));
        }
    }
    let fa = match SparseColMat::<usize, c64>::try_new_from_triplets(n, n, &trip) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}: faer build error {e:?}");
            return;
        }
    };
    let t = Instant::now();
    let faer_res_lu = fa.sp_lu();
    let faer_fac = t.elapsed().as_secs_f64() * 1e3;
    let (faer_slv, faer_res, faer_ok) = match faer_res_lu {
        Ok(lu) => {
            let mut xb = Mat::<c64>::from_fn(n, 1, |i, _| c64::new(b[i].re, b[i].im));
            let t = Instant::now();
            lu.solve_in_place(&mut xb);
            let slv = t.elapsed().as_secs_f64() * 1e3;
            let xf: Vec<C> = (0..n).map(|i| Complex::new(xb[(i, 0)].re, xb[(i, 0)].im)).collect();
            (slv, rel_resid(&a, &xf, &b), true)
        }
        Err(_) => (0.0, f64::NAN, false),
    };

    let speedup = if rla_fac > 0.0 { faer_fac / rla_fac } else { 0.0 };
    println!(
        "{name:30} n={n:6} nnz={nnz:8}  RLA[ana{rla_ana:6.0} fac{rla_fac:7.0} slv{rla_slv:6.1} res{rla_res:.0e} fill{rla_fill:9}]  \
         faer[{}fac{faer_fac:8.0} slv{faer_slv:7.1} res{faer_res:.0e}]  fac×{speedup:5.2}",
        if faer_ok { "" } else { "FAILED " },
    );
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
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    let filter = std::env::var("RLA_DIAG_FILTER").unwrap_or_default();
    let mode = format!(
        "memory={}  blr={}  reorder={}",
        if std::env::var("RLA_EAGER").is_ok() { "Eager" } else { "LowMemory" },
        if std::env::var("RLA_BLR_CB").is_ok() { "CB" } else { "Off" },
        if std::env::var("RLA_NO_LIU").is_ok() { "Off" } else { "HybridLiu" },
    );
    println!("MoM direct solve: RLA unsymmetric LU vs faer sparse LU  [RLA {mode}]\n");
    for f in &files {
        if filter.is_empty() || f.file_name().unwrap().to_string_lossy().contains(&filter) {
            run(f);
        }
    }
}
