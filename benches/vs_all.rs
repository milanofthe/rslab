//! Head-to-head **exact** direct solve on the complex MoM matrices — RLA's two
//! factorization paths against the field:
//!   * **RLA-LL**  — supernodal left-looking LU (the default path)
//!   * **RLA-MF**  — multifrontal LU (assembly-tree of dense fronts)
//!   * **faer**    — faer sparse LU (pure-Rust competitor)
//!   * **PARDISO** — Intel MKL PARDISO, mtype=13 (complex unsymmetric), loaded at
//!                   runtime via `mkl_rt` (bench-only FFI; the solver lib stays
//!                   100% pure Rust). Skipped with a note if MKL is absent.
//!
//! All four do a **complete** (exact) factorization, so factor time / fill / true
//! residual are apples-to-apples. SuperLU (scipy) runs from the companion
//! `benches/superlu_mom.py` on the same matrices and `b` and is reported beside
//! this table.
//!
//! Run: `cargo bench --bench vs_all` (optionally `RLA_DIAG_FILTER=spiral`).

use std::ffi::c_void;
use std::time::Instant;

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat};
use num_complex::Complex;
use rla::prelude::*;
use rla::{AnalyzeOptions, FactorMethod, FactorOptions, LuSymbolic, ReorderMode};

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
type C = Complex<f64>;

fn rel_resid(a: &rla::GeneralCsc<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![Complex::new(0.0, 0.0); a.n];
    a.matvec(x, &mut ax);
    let num: f64 = (0..a.n).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den
}

/// CSC → 0-based CSR (transpose of storage; same matrix). PARDISO wants CSR with
/// ascending column indices per row — produced here by scanning columns in order.
fn build_full_csr(a: &rla::GeneralCsc<C>) -> (Vec<i32>, Vec<i32>, Vec<C>) {
    let n = a.n;
    let nnz = a.values.len();
    let mut ia = vec![0i32; n + 1];
    for k in 0..nnz {
        ia[a.row_idx[k] + 1] += 1;
    }
    for i in 0..n {
        ia[i + 1] += ia[i];
    }
    let mut ja = vec![0i32; nnz];
    let mut va = vec![Complex::new(0.0, 0.0); nnz];
    let mut next: Vec<i32> = ia.clone();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let r = a.row_idx[k];
            let dst = next[r] as usize;
            ja[dst] = j as i32;
            va[dst] = a.values[k];
            next[r] += 1;
        }
    }
    (ia, ja, va)
}

// ----------------------- minimal MKL PARDISO FFI ---------------------------
type PardisoFn = unsafe extern "C" fn(
    *mut i64,
    *const i32,
    *const i32,
    *const i32,
    *const i32,
    *const i32,
    *const c_void,
    *const i32,
    *const i32,
    *mut i32,
    *const i32,
    *mut i32,
    *const i32,
    *mut c_void,
    *mut c_void,
    *mut i32,
);

struct Pardiso {
    _lib: libloading::Library,
    f: PardisoFn,
    pt: [i64; 64],
    iparm: [i32; 64],
    mtype: i32,
}

impl Pardiso {
    fn try_new() -> Option<Self> {
        let lib = unsafe {
            libloading::Library::new("mkl_rt.2.dll")
                .or_else(|_| libloading::Library::new("mkl_rt.dll"))
                .or_else(|_| libloading::Library::new("libmkl_rt.so"))
                .ok()?
        };
        let f: PardisoFn = unsafe {
            let sym: libloading::Symbol<PardisoFn> =
                lib.get(b"pardiso").or_else(|_| lib.get(b"pardiso_")).ok()?;
            *sym
        };
        let mut iparm = [0i32; 64];
        let nthreads = std::thread::available_parallelism().map(|p| p.get() as i32).unwrap_or(4);
        iparm[0] = 1; // non-default iparm
        iparm[1] = 3; // parallel (OpenMP) nested dissection fill-reduction
        iparm[2] = nthreads; // OpenMP threads
        iparm[7] = 2; // iterative refinement (≤2 steps) — PARDISO's default accuracy
        iparm[9] = 13; // pivot perturbation 1e-13
        iparm[10] = 1; // scaling (recommended for unsymmetric)
        iparm[12] = 1; // weighted matching (recommended for unsymmetric)
        iparm[17] = -1; // report nnz in factors via iparm[17]
        iparm[34] = 1; // 0-based indexing
        Some(Pardiso { _lib: lib, f, pt: [0i64; 64], iparm, mtype: 13 })
    }

    fn call(&mut self, phase: i32, n: i32, ia: &[i32], ja: &[i32], a: &[C], b: &mut [C], x: &mut [C]) -> i32 {
        let (maxfct, mnum, nrhs, msglvl) = (1i32, 1i32, 1i32, 0i32);
        let mut perm = vec![0i32; n.max(1) as usize];
        let mut error = 0i32;
        unsafe {
            (self.f)(
                self.pt.as_mut_ptr(),
                &maxfct,
                &mnum,
                &self.mtype,
                &phase,
                &n,
                a.as_ptr() as *const c_void,
                ia.as_ptr(),
                ja.as_ptr(),
                perm.as_mut_ptr(),
                &nrhs,
                self.iparm.as_mut_ptr(),
                &msglvl,
                b.as_mut_ptr() as *mut c_void,
                x.as_mut_ptr() as *mut c_void,
                &mut error,
            );
        }
        error
    }
}

impl Drop for Pardiso {
    fn drop(&mut self) {
        let (mut dum_b, mut dum_x) = (vec![], vec![]);
        let _ = self.call(-1, 0, &[0], &[0], &[], &mut dum_b, &mut dum_x);
    }
}

fn run(path: &std::path::Path, pardiso_ok: bool) {
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

    println!("\n{name}  n={n}  nnz={nnz}");

    // ---- RLA exact, both paths ----
    let t = Instant::now();
    let sym =
        LuSymbolic::analyze_with(&a, &AnalyzeOptions::default().with_reorder(ReorderMode::HybridLiu))
            .unwrap();
    let ana = t.elapsed().as_secs_f64() * 1e3;
    for (label, method) in [("RLA-LL ", FactorMethod::LeftLooking), ("RLA-MF ", FactorMethod::Multifrontal)] {
        let opts = FactorOptions::default().with_method(method);
        let t = Instant::now();
        match sym.factor(&a, &opts) {
            Ok(f) => {
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b).unwrap();
                let slv = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "  {label}  ana {ana:7.0}  fac {fac:8.1}  slv {slv:6.1}  res {:.1e}  fill {}",
                    rel_resid(&a, &x, &b),
                    f.factor_nnz()
                );
            }
            Err(e) => println!("  {label}  ana {ana:7.0}  FACTOR FAILED: {e:?}"),
        }
    }

    // ---- faer sparse LU ----
    {
        let mut trip: Vec<Triplet<usize, usize, c64>> = Vec::with_capacity(nnz);
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                trip.push(Triplet::new(a.row_idx[k], j, c64::new(a.values[k].re, a.values[k].im)));
            }
        }
        match SparseColMat::<usize, c64>::try_new_from_triplets(n, n, &trip) {
            Ok(fa) => {
                let t = Instant::now();
                match fa.sp_lu() {
                    Ok(lu) => {
                        let fac = t.elapsed().as_secs_f64() * 1e3;
                        let mut xb = Mat::<c64>::from_fn(n, 1, |i, _| c64::new(b[i].re, b[i].im));
                        let t = Instant::now();
                        lu.solve_in_place(&mut xb);
                        let slv = t.elapsed().as_secs_f64() * 1e3;
                        let xf: Vec<C> = (0..n).map(|i| Complex::new(xb[(i, 0)].re, xb[(i, 0)].im)).collect();
                        println!(
                            "  faer     ana       —  fac {fac:8.1}  slv {slv:6.1}  res {:.1e}",
                            rel_resid(&a, &xf, &b)
                        );
                    }
                    Err(_) => println!("  faer     FACTOR FAILED"),
                }
            }
            Err(e) => println!("  faer     build error {e:?}"),
        }
    }

    // ---- MKL PARDISO mtype=13 ----
    if pardiso_ok {
        if let Some(mut ps) = Pardiso::try_new() {
            let (ia, ja, va) = build_full_csr(&a);
            let ni = n as i32;
            let (mut dum_b, mut dum_x) = (vec![Complex::new(0.0, 0.0); n], vec![Complex::new(0.0, 0.0); n]);
            let t = Instant::now();
            let e1 = ps.call(11, ni, &ia, &ja, &va, &mut dum_b, &mut dum_x);
            let ana_p = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let e2 = ps.call(22, ni, &ia, &ja, &va, &mut dum_b, &mut dum_x);
            let fac = t.elapsed().as_secs_f64() * 1e3;
            if e1 != 0 || e2 != 0 {
                println!("  PARDISO  analyze/factor error {e1}/{e2}");
            } else {
                let mut bb = b.clone();
                let mut x = vec![Complex::new(0.0, 0.0); n];
                let t = Instant::now();
                let e3 = ps.call(33, ni, &ia, &ja, &va, &mut bb, &mut x);
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let fill = ps.iparm[17]; // nnz in factors (×1000 if negative-flagged off)
                if e3 != 0 {
                    println!("  PARDISO  solve error {e3}");
                } else {
                    println!(
                        "  PARDISO  ana {ana_p:7.0}  fac {fac:8.1}  slv {slv:6.1}  res {:.1e}  fill {fill}",
                        rel_resid(&a, &x, &b)
                    );
                }
            }
        }
    }
}

fn main() {
    faer::set_global_parallelism(faer::Par::rayon(0));
    let pardiso_ok = Pardiso::try_new().is_some();
    if !pardiso_ok {
        println!("(PARDISO: mkl_rt not found — skipping; install MKL or add it to PATH)");
    }
    let mut files: Vec<_> = match std::fs::read_dir(DIR) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "mtx")).collect(),
        Err(e) => {
            println!("cannot read {DIR}: {e}");
            return;
        }
    };
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    let filter = std::env::var("RLA_DIAG_FILTER").unwrap_or_default();
    println!("Exact direct solve: RLA-LL / RLA-MF / faer / PARDISO (mtype=13)  [SuperLU via superlu_mom.py]");
    for f in &files {
        if filter.is_empty() || f.file_name().unwrap().to_string_lossy().contains(&filter) {
            run(f, pardiso_ok);
        }
    }
}
