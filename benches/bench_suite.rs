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

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat};
use num_complex::Complex;
use rla::matgen::{bem, stencil};
use rla::{CscMatrix, FactorMethod, FactorOptions, GeneralCsc, LdltSymbolic, LuSymbolic};

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

// ---- Windows working-set sampler (for PARDISO/MKL, which bypasses LIVE) -------
#[cfg(windows)]
fn cur_ws_mb() -> f64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        a: u32,
        peak_ws: usize,
        ws: usize,
        b: [usize; 6],
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(p: isize, c: *mut Pmc, cb: u32) -> i32;
    }
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

fn ws_sampled<R>(f: impl FnOnce() -> R) -> (R, f64) {
    use std::sync::Arc;
    let before = cur_ws_mb();
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicUsize::new(before as usize));
    let (s, p) = (stop.clone(), peak.clone());
    let h = std::thread::spawn(move || {
        while !s.load(Ordering::Relaxed) {
            p.fetch_max(cur_ws_mb() as usize, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
    });
    let r = f();
    stop.store(true, Ordering::Relaxed);
    let _ = h.join();
    (r, (peak.load(Ordering::Relaxed) as f64 - before).max(0.0))
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
/// Full CSR (0-based) of an unsymmetric `GeneralCsc` — for PARDISO mtype 13.
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

/// Upper-triangle CSR (0-based) from a lower-triangle symmetric `CscMatrix` — for
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
/// (`A = Aᵀ`, complex-symmetric, no conjugate) — for faer's LU.
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
fn emit(out: &mut dyn Write, solver: &str, family: &str, n: usize, nnz: usize, threads: i32, mem: bool, ana: f64, fac: f64, slv: f64, memmb: f64, fill: usize, res: f64) {
    let _ = writeln!(
        out,
        "{{\"solver\":\"{solver}\",\"family\":\"{family}\",\"n\":{n},\"nnz\":{nnz},\"threads\":{threads},\"metric\":\"{}\",\"ana_ms\":{ana:.3},\"fac_ms\":{fac:.3},\"slv_ms\":{slv:.3},\"mem_mb\":{memmb:.1},\"fill\":{fill},\"res\":{res:.3e}}}",
        if mem { "mem" } else { "time" }
    );
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
    let has = |s: &str| solvers.iter().any(|x| x == s);

    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open RLA_BENCH_OUT");

    for &sz in &sizes {
        let opts = FactorOptions::default();
        eprintln!("[bench] family={family} n≈{sz} threads={threads} metric={}", if mem { "mem" } else { "time" });
        if family == "sym" {
            // Complex-symmetric Helmholtz (mild shift ⇒ near-SPD, robust scaling).
            let a: CscMatrix<C> = stencil::helmholtz(&cube(sz), Complex::new(0.05, 0.02), &stencil::StencilOpts::default());
            let n = a.n;
            let nnz = a.values.len();
            let b = rhs(n);
            for method in [("ll", FactorMethod::LeftLooking), ("mf", FactorMethod::Multifrontal)] {
                if !has(method.0) {
                    continue;
                }
                let t = Instant::now();
                let sym = LdltSymbolic::analyze(&a).unwrap();
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let o = opts.clone().with_method(method.1);
                let t = Instant::now();
                let (f, mm) = live_peak(|| sym.factor(&a, &o).unwrap());
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b).unwrap();
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.symv(&x, &mut ax);
                emit(&mut out, method.0, &family, n, nnz, threads, mem, ana, fac, slv, mm, f.factor_nnz(), rel(&ax, &b));
            }
            if has("faer") {
                let fa = build_faer(&sym_to_full(&a));
                run_faer(&mut out, &family, n, nnz, threads, mem, fa, &a, &b, true);
            }
            if has("pardiso") {
                let (ia, ja, va) = upper_csr(&a);
                run_pardiso(&mut out, &family, n, nnz, threads, mem, 6, &ia, &ja, &va, ResidA::Sym(&a), &b);
            }
        } else {
            // Unsymmetric near-field BEM/MoM kernel.
            let bopts = bem::BemOpts { cutoff: 0.35, ..Default::default() };
            let a: GeneralCsc<C> = bem::kernel(sz, &bopts);
            let n = a.n;
            let nnz = a.values.len();
            let b = rhs(n);
            for method in [("ll", FactorMethod::LeftLooking), ("mf", FactorMethod::Multifrontal)] {
                if !has(method.0) {
                    continue;
                }
                let t = Instant::now();
                let sym = LuSymbolic::analyze(&a).unwrap();
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let o = opts.clone().with_method(method.1);
                let t = Instant::now();
                let (f, mm) = live_peak(|| sym.factor(&a, &o).unwrap());
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b).unwrap();
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.matvec(&x, &mut ax);
                emit(&mut out, method.0, &family, n, nnz, threads, mem, ana, fac, slv, mm, f.factor_nnz(), rel(&ax, &b));
            }
            if has("faer") {
                let fa = build_faer(&a);
                run_faer(&mut out, &family, n, nnz, threads, mem, fa, &a, &b, false);
            }
            if has("pardiso") {
                let (ia, ja, va) = full_csr(&a);
                run_pardiso(&mut out, &family, n, nnz, threads, mem, 13, &ia, &ja, &va, ResidA::Gen(&a), &b);
            }
        }
    }
}

enum ResidA<'a> {
    Sym(&'a CscMatrix<C>),
    Gen(&'a GeneralCsc<C>),
}
impl ResidA<'_> {
    fn apply(&self, x: &[C], ax: &mut [C]) {
        match self {
            ResidA::Sym(a) => a.symv(x, ax),
            ResidA::Gen(a) => a.matvec(x, ax),
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
fn run_faer(out: &mut dyn Write, family: &str, n: usize, nnz: usize, threads: i32, mem: bool, fa: Option<SparseColMat<usize, c64>>, ra: impl FaerResid, b: &[C], _sym: bool) {
    let Some(fa) = fa else { return };
    let t = Instant::now();
    let (lu, mm) = live_peak(|| fa.sp_lu());
    let fac = t.elapsed().as_secs_f64() * 1e3;
    let Ok(lu) = lu else { return };
    let mut xb = Mat::<c64>::from_fn(n, 1, |i, _| c64::new(b[i].re, b[i].im));
    let t = Instant::now();
    lu.solve_in_place(&mut xb);
    let slv = t.elapsed().as_secs_f64() * 1e3;
    let x: Vec<C> = (0..n).map(|i| Complex::new(xb[(i, 0)].re, xb[(i, 0)].im)).collect();
    let mut ax = vec![Complex::new(0.0, 0.0); n];
    ra.apply(&x, &mut ax);
    emit(out, "faer", family, n, nnz, threads, mem, 0.0, fac, slv, mm, 0, rel(&ax, b));
}

// helper traits so faer's residual matrix can be either Sym or Gen
trait FaerResid {
    fn apply(&self, x: &[C], ax: &mut [C]);
}
impl FaerResid for &CscMatrix<C> {
    fn apply(&self, x: &[C], ax: &mut [C]) {
        self.symv(x, ax)
    }
}
impl FaerResid for &GeneralCsc<C> {
    fn apply(&self, x: &[C], ax: &mut [C]) {
        self.matvec(x, ax)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pardiso(out: &mut dyn Write, family: &str, n: usize, nnz: usize, threads: i32, mem: bool, mtype: i32, ia: &[i32], ja: &[i32], va: &[C], ra: ResidA, b: &[C]) {
    let Some(mut ps) = Pardiso::try_new(mtype, threads) else {
        eprintln!("[bench] PARDISO unavailable (mkl_rt not found)");
        return;
    };
    let ni = n as i32;
    let (mut db, mut dx) = (vec![Complex::new(0.0, 0.0); n], vec![Complex::new(0.0, 0.0); n]);
    let t = Instant::now();
    let e1 = ps.call(11, ni, ia, ja, va, &mut db, &mut dx);
    let ana = t.elapsed().as_secs_f64() * 1e3;
    // PARDISO bypasses the Rust allocator; measure its factor by working set.
    let t = Instant::now();
    let (e2, mm) = ws_sampled(|| ps.call(22, ni, ia, ja, va, &mut db, &mut dx));
    let fac = t.elapsed().as_secs_f64() * 1e3;
    if e1 != 0 || e2 != 0 {
        eprintln!("[bench] PARDISO error {e1}/{e2}");
        return;
    }
    let mut bb = b.to_vec();
    let mut x = vec![Complex::new(0.0, 0.0); n];
    let t = Instant::now();
    let e3 = ps.call(33, ni, ia, ja, va, &mut bb, &mut x);
    let slv = t.elapsed().as_secs_f64() * 1e3;
    if e3 != 0 {
        return;
    }
    let fill = ps.iparm[17].max(0) as usize;
    let mut ax = vec![Complex::new(0.0, 0.0); n];
    ra.apply(&x, &mut ax);
    emit(out, "pardiso", family, n, nnz, threads, mem, ana, fac, slv, mm, fill, rel(&ax, b));
}
