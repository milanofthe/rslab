//! Benchmark engine for the scaling / memory / thread study. One invocation runs
//! one `(family, metric)` over a list of sizes at the process's thread count and
//! appends JSONL records `{solver,family,n,nnz,threads,fac_ms,slv_ms,ana_ms,
//! mem_mb,fill,res}` to `RLA_BENCH_OUT`. The Python driver `benches/run_bench.py`
//! orchestrates the size/thread sweeps and produces the matplotlib plots.
//!
//! Solvers: RLA left-looking (`ll`) and multifrontal (`mf`), `faer` sparse LU,
//! and MKL `pardiso` (mtype 6 for the symmetric family, 13 for the unsymmetric).
//! Symmetric matrices ⇒ RLA LDLᵀ; unsymmetric ⇒ RLA LU. Memory is the live-bytes
//! peak (Rust solvers) or the working-set transient (PARDISO/MKL).
//!
//! Env: `RLA_BENCH_FAMILY=sym|unsym`, `RLA_BENCH_SIZES=8000,27000,...`,
//! `RLA_BENCH_MEM=1` (memory pass, else time), `RLA_BENCH_SOLVERS=ll,mf,faer,pardiso`,
//! `RLA_BENCH_OUT=path.jsonl`, `RAYON_NUM_THREADS=N` (threads, also drives faer/MKL).
//!
//! Run via the driver; standalone: `cargo bench --bench bench_suite --features matgen`.
// CSC/COO conversion loops use the index as a value (col_ptr[c], next[c]).
#![allow(clippy::needless_range_loop)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat as FaerMat};
use num_complex::Complex;
use rslab::matgen::{bem, stencil};
use rslab::{
    gmres, parse_mtx_complex_general, CscMatrix, FactorMethod, FactorOptions, GeneralCsc,
    LdltSymbolic, LuSymbolic,
};
#[cfg(feature = "matgen-download")]
use rslab::{read_mtx_any, MtxLoaded};

type C = Complex<f64>;

// ---- live-bytes counting allocator (memory pass only) ------------------------
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static COUNTING_ON: AtomicBool = AtomicBool::new(false);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() && COUNTING_ON.load(Ordering::Relaxed) {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if COUNTING_ON.load(Ordering::Relaxed) {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        }
        System.dealloc(p, l);
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

fn live_peak<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let r = f();
    (r, PEAK.load(Ordering::Relaxed).saturating_sub(before) as f64 / 1e6)
}

// ---- MKL PARDISO FFI (mtype 6 symmetric / 13 unsymmetric) --------------------
type PardisoFn = unsafe extern "C" fn(
    *mut i64, *const i32, *const i32, *const i32, *const i32, *const i32, *const c_void,
    *const i32, *const i32, *mut i32, *const i32, *mut i32, *const i32, *mut c_void, *mut c_void,
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
    fn try_new(mtype: i32, threads: i32) -> Option<Self> {
        let lib = unsafe {
            libloading::Library::new("mkl_rt.2.dll")
                .or_else(|_| libloading::Library::new("mkl_rt.dll"))
                .or_else(|_| libloading::Library::new("libmkl_rt.so"))
                .ok()?
        };
        let f: PardisoFn = unsafe {
            let s: libloading::Symbol<PardisoFn> =
                lib.get(b"pardiso").or_else(|_| lib.get(b"pardiso_")).ok()?;
            *s
        };
        let mut iparm = [0i32; 64];
        iparm[0] = 1;
        iparm[1] = 3;
        iparm[2] = threads;
        iparm[7] = 2;
        iparm[9] = 13;
        iparm[10] = 1;
        iparm[12] = 1;
        iparm[17] = -1;
        iparm[34] = 1;
        Some(Pardiso { _lib: lib, f, pt: [0i64; 64], iparm, mtype })
    }
    #[allow(clippy::too_many_arguments)]
    fn call(&mut self, phase: i32, n: i32, ia: &[i32], ja: &[i32], a: &[C], b: &mut [C], x: &mut [C]) -> i32 {
        let (maxfct, mnum, nrhs, msglvl) = (1i32, 1i32, 1i32, 0i32);
        let mut perm = vec![0i32; n.max(1) as usize];
        let mut err = 0i32;
        unsafe {
            (self.f)(
                self.pt.as_mut_ptr(), &maxfct, &mnum, &self.mtype, &phase, &n,
                a.as_ptr() as *const c_void, ia.as_ptr(), ja.as_ptr(), perm.as_mut_ptr(), &nrhs,
                self.iparm.as_mut_ptr(), &msglvl, b.as_mut_ptr() as *mut c_void,
                x.as_mut_ptr() as *mut c_void, &mut err,
            );
        }
        err
    }
}
impl Drop for Pardiso {
    fn drop(&mut self) {
        let (mut db, mut dx) = (vec![], vec![]);
        let _ = self.call(-1, 0, &[0], &[0], &[], &mut db, &mut dx);
    }
}

// ---- CSC/CSR helpers ---------------------------------------------------------
/// Full CSR (0-based) of an unsymmetric `GeneralCsc` - for PARDISO mtype 13.
fn full_csr(a: &GeneralCsc<C>) -> (Vec<i32>, Vec<i32>, Vec<C>) {
    let n = a.n;
    let mut ia = vec![0i32; n + 1];
    for &r in &a.row_idx {
        ia[r + 1] += 1;
    }
    for i in 0..n {
        ia[i + 1] += ia[i];
    }
    let mut ja = vec![0i32; a.values.len()];
    let mut va = vec![Complex::new(0.0, 0.0); a.values.len()];
    let mut next = ia.clone();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let r = a.row_idx[k];
            let d = next[r] as usize;
            ja[d] = j as i32;
            va[d] = a.values[k];
            next[r] += 1;
        }
    }
    (ia, ja, va)
}

/// Upper-triangle CSR (0-based) from a lower-triangle symmetric `CscMatrix` - for
/// PARDISO mtype 6. A lower entry `(r,c)` (r≥c) maps to the upper CSR cell `(c,r)`.
fn upper_csr(a: &CscMatrix<C>) -> (Vec<i32>, Vec<i32>, Vec<C>) {
    let n = a.n;
    // row of the upper-CSR = original column c; count per c.
    let mut ia = vec![0i32; n + 1];
    for c in 0..n {
        ia[c + 1] += (a.col_ptr[c + 1] - a.col_ptr[c]) as i32;
    }
    for i in 0..n {
        ia[i + 1] += ia[i];
    }
    let mut ja = vec![0i32; a.values.len()];
    let mut va = vec![Complex::new(0.0, 0.0); a.values.len()];
    let mut next = ia.clone();
    for c in 0..n {
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[k]; // r ≥ c
            let d = next[c] as usize;
            ja[d] = r as i32; // upper: col = r ≥ row = c
            va[d] = a.values[k];
            next[c] += 1;
        }
    }
    (ia, ja, va)
}

/// Full unsymmetric `GeneralCsc` from a lower-triangle symmetric `CscMatrix`
/// (`A = Aᵀ`, complex-symmetric, no conjugate) - for faer's LU.
fn sym_to_full(a: &CscMatrix<C>) -> GeneralCsc<C> {
    let n = a.n;
    let mut rows = Vec::with_capacity(a.values.len() * 2);
    let mut cols = Vec::with_capacity(a.values.len() * 2);
    let mut vals = Vec::with_capacity(a.values.len() * 2);
    for c in 0..n {
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[k];
            let v = a.values[k];
            rows.push(r);
            cols.push(c);
            vals.push(v);
            if r != c {
                rows.push(c);
                cols.push(r);
                vals.push(v);
            }
        }
    }
    GeneralCsc::from_triplets(n, &rows, &cols, &vals).expect("sym_to_full")
}

fn rhs(n: usize) -> Vec<C> {
    (0..n).map(|i| Complex::new((i % 5) as f64 - 2.0, (i % 3) as f64 - 1.0)).collect()
}
fn rel(ax: &[C], b: &[C]) -> f64 {
    let num: f64 = (0..b.len()).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den.max(1e-300)
}

fn cube(target: usize) -> [usize; 3] {
    let k = (target as f64).cbrt().round().max(2.0) as usize;
    [k, k, k]
}

#[allow(clippy::too_many_arguments)]
fn emit(out: &mut dyn Write, solver: &str, family: &str, name: &str, n: usize, nnz: usize, threads: i32, mem: bool, ana: f64, fac: f64, slv: f64, memmb: f64, fill: usize, res: f64) {
    let _ = writeln!(
        out,
        "{{\"solver\":\"{solver}\",\"family\":\"{family}\",\"name\":\"{name}\",\"n\":{n},\"nnz\":{nnz},\"threads\":{threads},\"metric\":\"{}\",\"ana_ms\":{ana:.3},\"fac_ms\":{fac:.3},\"slv_ms\":{slv:.3},\"mem_mb\":{memmb:.1},\"fill\":{fill},\"res\":{res:.3e}}}",
        if mem { "mem" } else { "time" }
    );
}

/// A test system: symmetric (→ LDLᵀ / PARDISO mtype 6) or unsymmetric (→ LU /
/// mtype 13). faer always factors the full matrix as LU.
enum Mat {
    Sym(CscMatrix<C>),
    Unsym(GeneralCsc<C>),
}
impl Mat {
    fn n(&self) -> usize {
        match self {
            Mat::Sym(a) => a.n,
            Mat::Unsym(a) => a.n,
        }
    }
    fn nnz(&self) -> usize {
        match self {
            Mat::Sym(a) => a.values.len(),
            Mat::Unsym(a) => a.values.len(),
        }
    }
    fn resid(&self, x: &[C], ax: &mut [C]) {
        match self {
            Mat::Sym(a) => a.symv(x, ax),
            Mat::Unsym(a) => a.matvec(x, ax),
        }
    }
}

fn build_faer(a: &GeneralCsc<C>) -> Option<SparseColMat<usize, c64>> {
    let mut trip: Vec<Triplet<usize, usize, c64>> = Vec::with_capacity(a.values.len());
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            trip.push(Triplet::new(a.row_idx[k], j, c64::new(a.values[k].re, a.values[k].im)));
        }
    }
    SparseColMat::try_new_from_triplets(a.n, a.n, &trip).ok()
}

#[allow(clippy::too_many_arguments)]
fn run_matrix(out: &mut dyn Write, family: &str, name: &str, mat: &Mat, threads: i32, mem: bool, solvers: &[String], opts: &FactorOptions) {
    let (n, nnz) = (mat.n(), mat.nnz());
    let b = rhs(n);
    let has = |s: &str| solvers.iter().any(|x| x == s);

    // A-priori memory estimate (validate against the measured live peak).
    if let Mat::Unsym(a) = mat {
        if let Ok(sym) = LuSymbolic::analyze(a) {
            eprintln!("[bench] {name} a-priori {}", sym.estimate_memory::<C>());
        }
    }

    // --- RLA left-looking / multifrontal ---
    for (tag, method) in [("ll", FactorMethod::LeftLooking), ("mf", FactorMethod::Multifrontal)] {
        if !has(tag) {
            continue;
        }
        let o = opts.clone().with_method(method);
        // Error-tolerant: a singular / numerically hard corpus matrix must skip
        // (with a note) rather than panic and abort the whole sweep.
        macro_rules! skip_err {
            ($what:expr, $r:expr) => {
                match $r {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[{tag}] {name}: {} failed: {e:?}", $what);
                        continue;
                    }
                }
            };
        }
        match mat {
            Mat::Sym(a) => {
                let t = Instant::now();
                let sym = skip_err!("analyze", LdltSymbolic::analyze(a));
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &o));
                let f = skip_err!("factor", fr);
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = skip_err!("solve", f.solve(&b));
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.symv(&x, &mut ax);
                emit(out, tag, family, name, n, nnz, threads, mem, ana, fac, slv, mm, f.factor_nnz(), rel(&ax, &b));
            }
            Mat::Unsym(a) => {
                let t = Instant::now();
                let sym = skip_err!("analyze", LuSymbolic::analyze(a));
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &o));
                let f = skip_err!("factor", fr);
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = skip_err!("solve", f.solve(&b));
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.matvec(&x, &mut ax);
                emit(out, tag, family, name, n, nnz, threads, mem, ana, fac, slv, mm, f.factor_nnz(), rel(&ax, &b));
            }
        }
    }

    // --- faer (full LU) --- memory-gated: faer factors the FULL matrix as a complex
    // LU (several x the memory of RSLAB's structure-exploiting factor), so skip it
    // when RSLAB's *a-priori* transient estimate already implies faer would blow the
    // budget (dogfooding the estimator) or above a generous n cap. RSLAB + PARDISO
    // still run. Tunable via RLA_BENCH_FAER_MAX / RLA_BENCH_FAER_EST_MB.
    let faer_max: usize = std::env::var("RLA_BENCH_FAER_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000);
    let faer_est_mb: f64 = std::env::var("RLA_BENCH_FAER_EST_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000.0);
    let rslab_est_mb = match mat {
        Mat::Sym(a) => LdltSymbolic::analyze(a).ok().map(|s| s.estimate_memory::<C>().transient_peak_bytes),
        Mat::Unsym(a) => LuSymbolic::analyze(a).ok().map(|s| s.estimate_memory::<C>().transient_peak_bytes),
    }
    .map(|b| b as f64 / 1e6)
    .unwrap_or(0.0);
    let faer_ok = n <= faer_max && rslab_est_mb < faer_est_mb;
    if has("faer") && !faer_ok {
        eprintln!("[faer] skip {name} (n={n}, RSLAB est {rslab_est_mb:.0} MB ⇒ faer over budget)");
    }
    if has("faer") && faer_ok {
        let fa = match mat {
            Mat::Sym(a) => build_faer(&sym_to_full(a)),
            Mat::Unsym(a) => build_faer(a),
        };
        if let Some(fa) = fa {
            let t = Instant::now();
            let (lu, mm) = live_peak(|| fa.sp_lu());
            let fac = t.elapsed().as_secs_f64() * 1e3;
            if let Ok(lu) = lu {
                let mut xb = FaerMat::<c64>::from_fn(n, 1, |i, _| c64::new(b[i].re, b[i].im));
                let t = Instant::now();
                lu.solve_in_place(&mut xb);
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let x: Vec<C> = (0..n).map(|i| Complex::new(xb[(i, 0)].re, xb[(i, 0)].im)).collect();
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                mat.resid(&x, &mut ax);
                emit(out, "faer", family, name, n, nnz, threads, mem, 0.0, fac, slv, mm, 0, rel(&ax, &b));
            }
        }
    }

    // --- MKL PARDISO (mtype 6 symmetric / 13 unsymmetric) ---
    if has("pardiso") {
        let (mtype, ia, ja, va) = match mat {
            Mat::Sym(a) => {
                let (ia, ja, va) = upper_csr(a);
                (6, ia, ja, va)
            }
            Mat::Unsym(a) => {
                let (ia, ja, va) = full_csr(a);
                (13, ia, ja, va)
            }
        };
        if let Some(mut ps) = Pardiso::try_new(mtype, threads) {
            let ni = n as i32;
            let (mut db, mut dx) = (vec![Complex::new(0.0, 0.0); n], vec![Complex::new(0.0, 0.0); n]);
            let t = Instant::now();
            let e1 = ps.call(11, ni, &ia, &ja, &va, &mut db, &mut dx);
            let ana = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let e2 = ps.call(22, ni, &ia, &ja, &va, &mut db, &mut dx);
            let fac = t.elapsed().as_secs_f64() * 1e3;
            if e1 == 0 && e2 == 0 {
                let mut bb = b.clone();
                let mut x = vec![Complex::new(0.0, 0.0); n];
                let t = Instant::now();
                let e3 = ps.call(33, ni, &ia, &ja, &va, &mut bb, &mut x);
                let slv = t.elapsed().as_secs_f64() * 1e3;
                if e3 == 0 {
                    // MKL self-reported peak memory (KB): max(iparm(15), iparm(16)+iparm(17)),
                    // 0-based indices 14/15/16 - the analogue of the live-bytes peak.
                    let peak_kb = ps.iparm[14].max(ps.iparm[15] + ps.iparm[16]);
                    let mm = peak_kb.max(0) as f64 / 1024.0;
                    let fill = ps.iparm[17].max(0) as usize;
                    let mut ax = vec![Complex::new(0.0, 0.0); n];
                    mat.resid(&x, &mut ax);
                    emit(out, "pardiso", family, name, n, nnz, threads, mem, ana, fac, slv, mm, fill, rel(&ax, &b));
                }
            } else {
                eprintln!("[bench] PARDISO error {e1}/{e2}");
            }
        } else {
            eprintln!("[bench] PARDISO unavailable (mkl_rt not found)");
        }
    }

    // --- RSLAB preconditioner mode (static pivoting, never-fail) + GMRES refinement.
    // Factors a perturbed Â (never rank-deficient), then refines A x = b with GMRES
    // preconditioned by that factor - so the indefinite / hard matrices where exact
    // factorization fails are still solved. `slv_ms` here is the Krylov refinement cost.
    if has("pc") {
        // floor 1e-4: regularizes the indefinite zero/tiny pivots enough for a
        // well-conditioned preconditioner (GMRES then corrects back to A); a
        // well-conditioned matrix has O(1) equilibrated pivots, so none are perturbed
        // and pc reduces to the exact factor (1 GMRES step).
        let envf = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let floor = envf("RLA_BENCH_PC_FLOOR", 1e-4);
        let maxit = envf("RLA_BENCH_PC_MAXIT", 200.0) as usize;
        let restart = envf("RLA_BENCH_PC_RESTART", 100.0) as usize;
        let pc_opts = FactorOptions::preconditioner(floor).with_threads(threads.max(1) as usize);
        let outcome = match mat {
            Mat::Sym(a) => LdltSymbolic::analyze(a).and_then(|sym| {
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &pc_opts));
                let f = fr?;
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let kr = gmres(a, &b, &f, 1e-10, maxit, restart)?;
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.symv(&kr.x, &mut ax);
                Ok((fac, slv, mm, f.factor_nnz(), rel(&ax, &b), kr.iters, kr.converged))
            }),
            Mat::Unsym(a) => LuSymbolic::analyze(a).and_then(|sym| {
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &pc_opts));
                let f = fr?;
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let kr = gmres(a, &b, &f, 1e-10, maxit, restart)?;
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.matvec(&kr.x, &mut ax);
                Ok((fac, slv, mm, f.factor_nnz(), rel(&ax, &b), kr.iters, kr.converged))
            }),
        };
        match outcome {
            Ok((fac, slv, mm, fill, res, iters, conv)) => {
                eprintln!("[pc] {name}: {iters} GMRES iters, converged={conv}, res={res:.1e}");
                emit(out, "pc", family, name, n, nnz, threads, mem, 0.0, fac, slv, mm, fill, res);
            }
            Err(e) => eprintln!("[pc] {name}: factor/refine failed: {e:?}"),
        }
    }
}

/// Build the matrices for a family. `sym` = 3D Helmholtz (sparse stencil); `unsym`
/// = BEM/MoM near-field kernel with a **density-matched** cutoff (≈120 nnz/row at
/// any `n`, like real MoM); `real` = the on-disk `precond_matrices` (smallest first).
fn build_family(family: &str, sizes: &[usize]) -> Vec<(String, Mat)> {
    match family {
        "sym" => sizes
            .iter()
            .map(|&sz| {
                let a = stencil::helmholtz(&cube(sz), Complex::new(0.05, 0.02), &stencil::StencilOpts::default());
                (format!("helmholtz_{}", a.n), Mat::Sym(a))
            })
            .collect(),
        "unsym" => sizes
            .iter()
            .map(|&sz| {
                // cutoff ∝ 1/√n keeps ≈`deg` neighbours per row independent of n -
                // a realistic near-field (constant degree under mesh refinement),
                // unlike a fixed cutoff whose density grows with n.
                let deg = 120.0;
                let cutoff = (2.0 * (deg / sz as f64).sqrt()).min(1.2);
                let a = bem::kernel(sz, &bem::BemOpts { cutoff, ..Default::default() });
                (format!("mom_{}", a.n), Mat::Unsym(a))
            })
            .collect(),
        "corpus" => build_corpus(),
        "real" => {
            let dir = std::env::var("RLA_BENCH_REAL_DIR")
                .unwrap_or_else(|_| r"C:\Repositories\rapidmom\precond_matrices".into());
            let count: usize =
                std::env::var("RLA_BENCH_REAL_N").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
            let mut files: Vec<_> = std::fs::read_dir(&dir)
                .map(|rd| {
                    rd.filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
                        .collect()
                })
                .unwrap_or_default();
            files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
            files
                .into_iter()
                .take(count)
                .filter_map(|p| {
                    let stem = p.file_stem()?.to_string_lossy().to_string();
                    let contents = std::fs::read_to_string(&p).ok()?;
                    let a = parse_mtx_complex_general(&contents, &stem).ok()?.to_general_csc().ok()?;
                    Some((stem, Mat::Unsym(a)))
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

/// SuiteSparse validation corpus: download each `group/name`, auto-detect its type,
/// and route symmetric → LDLᵀ, general → LU. Curated to be diverse (SPD / CFD /
/// circuit / complex), non-singular, and memory-safe (n ≤ ~30k, run sequentially).
/// Override with `RLA_BENCH_CORPUS` as a comma-separated `group/name` list. A
/// matrix that fails to download/parse is skipped with a note.
#[cfg(feature = "matgen-download")]
fn build_corpus() -> Vec<(String, Mat)> {
    // Default set spanning the axes; all comfortably < a few GB factored.
    // Diverse, non-singular, ~1k-100k. (cryg10000 is a near-singular eigenvalue test
    // matrix - no solver gets a small Ax=b residual - so it is omitted.) A name that
    // fails to download is skipped with a note, so the list can be generous.
    const DEFAULT: &str = "\
        HB/bcsstk27,HB/bcsstk14,HB/bcsstk16,HB/bcsstk17,HB/bcsstk18,HB/bcsstk25,\
        Cylshell/s3rmt3m3,Boeing/msc10848,Boeing/crystk03,Boeing/bcsstk39,\
        GHS_psdef/wathen100,GHS_psdef/wathen120,GHS_psdef/oilpan,GHS_psdef/s3dkt3m2,\
        Williams/pdb1HYS,Williams/cant,Nasa/nasasrb,Rothberg/cfd1,Schmid/thermal1,\
        Um/2cubes_sphere,\
        GHS_indef/stokes64,GHS_indef/bratu3d,GHS_indef/copter2,GHS_indef/dixmaanl,\
        GHS_indef/cont-201,\
        HB/sherman5,HB/sherman3,FIDAP/ex11,Hamm/memplus,Simon/raefsky3,Wang/wang3,\
        Bai/af23560,Mallya/lhr34,Goodwin/rim,\
        Bai/qc2534,Bai/mhd4800b";
    let list = std::env::var("RLA_BENCH_CORPUS").unwrap_or_else(|_| DEFAULT.into());
    let mut mats: Vec<(String, Mat)> = list
        .split(',')
        .filter_map(|gn| {
            let gn = gn.trim();
            let (group, name) = gn.split_once('/')?;
            let path = match rslab::matgen::download::fetch(group, name) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[corpus] fetch {gn} failed: {e}");
                    return None;
                }
            };
            match read_mtx_any(&path) {
                Ok(MtxLoaded::Symmetric(a)) => Some((name.to_string(), Mat::Sym(a))),
                Ok(MtxLoaded::General(a)) => Some((name.to_string(), Mat::Unsym(a))),
                Err(e) => {
                    eprintln!("[corpus] read {gn} failed: {e:?}");
                    None
                }
            }
        })
        .collect();
    mats.sort_by_key(|(_, m)| m.n());
    mats
}

#[cfg(not(feature = "matgen-download"))]
fn build_corpus() -> Vec<(String, Mat)> {
    eprintln!("[corpus] needs --features matgen-download");
    Vec::new()
}

fn main() {
    let family = std::env::var("RLA_BENCH_FAMILY").unwrap_or_else(|_| "sym".into());
    let sizes: Vec<usize> = std::env::var("RLA_BENCH_SIZES")
        .unwrap_or_else(|_| "8000,27000,64000".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mem = std::env::var("RLA_BENCH_MEM").map(|v| v == "1").unwrap_or(false);
    let solvers: Vec<String> = std::env::var("RLA_BENCH_SOLVERS")
        .unwrap_or_else(|_| "ll,mf,faer,pardiso".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let threads: i32 = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|p| p.get() as i32).unwrap_or(1));
    let out_path = std::env::var("RLA_BENCH_OUT").unwrap_or_else(|_| "bench_results.jsonl".into());
    if mem {
        COUNTING_ON.store(true, Ordering::Relaxed);
    }
    faer::set_global_parallelism(faer::Par::rayon(threads.max(1) as usize));
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open RLA_BENCH_OUT");
    // RLA now runs in a scoped pool of `opts.threads`; drive it from the sweep var.
    let opts = FactorOptions::default().with_threads(threads.max(1) as usize);

    // Estimate-only mode (`RLA_BENCH_ESTIMATE=1`): emit the a-priori memory-estimate
    // breakdown per matrix and skip all factoring (instant - no numeric work).
    let estimate_only = std::env::var("RLA_BENCH_ESTIMATE").map(|v| v == "1").unwrap_or(false);

    for (name, mat) in build_family(&family, &sizes) {
        if estimate_only {
            let e = match &mat {
                Mat::Sym(a) => LdltSymbolic::analyze(a).ok().map(|s| s.estimate_memory::<C>()),
                Mat::Unsym(a) => LuSymbolic::analyze(a).ok().map(|s| s.estimate_memory::<C>()),
            };
            if let Some(e) = e {
                let mb = |b: u64| b as f64 / 1e6;
                let scratch = mb(e.transient_peak_bytes) - mb(e.panels_all_bytes) - mb(e.factor_bytes);
                let _ = writeln!(
                    out,
                    "{{\"name\":\"{name}\",\"n\":{},\"panels_mb\":{:.1},\"factor_mb\":{:.1},\"scratch_mb\":{:.1},\"transient_mb\":{:.1},\"freed_floor_mb\":{:.1}}}",
                    mat.n(), mb(e.panels_all_bytes), mb(e.factor_bytes), scratch.max(0.0),
                    mb(e.transient_peak_bytes), mb(e.panel_live_peak_bytes),
                );
            }
            continue;
        }
        eprintln!("[bench] family={family} name={name} n={} threads={threads} metric={}", mat.n(), if mem { "mem" } else { "time" });
        run_matrix(&mut out, &family, &name, &mat, threads, mem, &solvers, &opts);
    }
}
