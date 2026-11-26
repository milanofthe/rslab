//! Benchmark engine for the scaling / memory / thread study. One invocation runs
//! one `(family, metric)` over a list of sizes at the process's thread count and
//! appends JSONL records `{solver,family,n,nnz,threads,fac_ms,slv_ms,ana_ms,
//! mem_mb,fill,res}` to `RLA_BENCH_OUT`. The Python driver `benches/run_bench.py`
//! orchestrates the size/thread sweeps and produces the matplotlib plots.
//!
//! Solvers: RLA left-looking (`ll`) and multifrontal (`mf`), `faer` sparse LU,
//! MKL `pardiso` (mtype 6 for the symmetric family, 13 for the unsymmetric), and
//! Apple `accel`erate Sparse Solvers (macOS 15.5+: LDLT for real-symmetric, the
//! new sparse LU otherwise). Symmetric matrices => RLA LDL^T; unsymmetric => RLA LU.
//! Memory is the live-bytes peak (Rust solvers and Accelerate, whose allocations
//! run through instrumented callbacks) or the working-set transient (PARDISO/MKL).
//!
//! Env: `RLA_BENCH_FAMILY=sym|unsym`, `RLA_BENCH_SIZES=8000,27000,...`,
//! `RLA_BENCH_MEM=1` (memory pass, else time), `RLA_BENCH_SOLVERS=ll,mf,faer,pardiso,accel`,
//! `RLA_BENCH_OUT=path.jsonl`, `RAYON_NUM_THREADS=N` (threads, also drives faer/MKL).
//!
//! Run via the driver; standalone: `cargo bench --bench bench_suite --features matgen`.
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
use rslab::matgen::{bem, fem, stencil};
use rslab::{
    gmres, parse_mtx_complex_general, CscMatrix, GeneralCsc, KluSettings, KluSymbolic,
    LdltSymbolic, LuSymbolic, SolverSettings,
};
#[cfg(feature = "matgen-download")]
use rslab::{read_mtx_any, MtxLoaded};

type C = Complex<f64>;

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
    (
        r,
        PEAK.load(Ordering::Relaxed).saturating_sub(before) as f64 / 1e6,
    )
}

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
        Some(Pardiso {
            _lib: lib,
            f,
            pt: [0i64; 64],
            iparm,
            mtype,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn call(
        &mut self,
        phase: i32,
        n: i32,
        ia: &[i32],
        ja: &[i32],
        a: &[C],
        b: &mut [C],
        x: &mut [C],
    ) -> i32 {
        let (maxfct, mnum, nrhs, msglvl) = (1i32, 1i32, 1i32, 0i32);
        let mut perm = vec![0i32; n.max(1) as usize];
        let mut err = 0i32;
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
                &mut err,
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

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct AccelResult {
    ana_s: f64,
    fac_s: f64,
    slv_s: f64,
    peak_bytes: i64,
    factor_bytes: i64,
    workspace_bytes: i64,
    sym_status: i32,
    num_status: i32,
    chosen_order: i32,
    _pad: i32,
}

#[allow(clippy::type_complexity)]
type AccelFn = unsafe extern "C" fn(
    n: i32,
    col_ptr: *const std::os::raw::c_long,
    row_idx: *const i32,
    vals: *const C,
    symmetric: i32,
    fact_type: i32,
    order: i32,
    pivot_tol: f64,
    max_bytes: i64,
    b: *const C,
    x: *mut C,
    out: *mut AccelResult,
) -> i32;

const ACCEL_ORDER_BAKEOFF: i32 = -2;

struct Accel {
    _lib: libloading::Library,
    f: AccelFn,
}

impl Accel {
    fn try_new() -> Option<Self> {
        if !cfg!(target_os = "macos") {
            eprintln!("[accel] skipped: Apple Accelerate is macOS-only");
            return None;
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let src = root.join("benches/accel_shim.c");
        let dylib = root.join("target/accel_shim.dylib");
        let mtime = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let stale = match (mtime(&src), mtime(&dylib)) {
            (Some(s), Some(d)) => s > d,
            _ => true,
        };
        if stale {
            let st = std::process::Command::new("cc")
                .args([
                    "-O2",
                    "-std=c11",
                    "-mmacosx-version-min=15.5",
                    "-dynamiclib",
                ])
                .args(["-framework", "Accelerate"])
                .arg(&src)
                .arg("-o")
                .arg(&dylib)
                .status();
            if !st.map(|s| s.success()).unwrap_or(false) {
                eprintln!("[accel] shim compile failed (needs Xcode CLT + macOS 15.5 SDK)");
                return None;
            }
        }
        let lib = unsafe { libloading::Library::new(&dylib).ok()? };
        let f: AccelFn = unsafe {
            let s: libloading::Symbol<AccelFn> = lib.get(b"accel_sparse_solve_complex").ok()?;
            *s
        };
        Some(Accel { _lib: lib, f })
    }

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        vals: &[C],
        kind: i32,
        fact: i32,
        order: i32,
        pivot: f64,
        max_bytes: i64,
        b: &[C],
        x: &mut [C],
    ) -> (i32, AccelResult) {
        let cp: Vec<std::os::raw::c_long> =
            col_ptr.iter().map(|&v| v as std::os::raw::c_long).collect();
        let ri: Vec<i32> = row_idx.iter().map(|&v| v as i32).collect();
        let mut r = AccelResult::default();
        let st = unsafe {
            (self.f)(
                n as i32,
                cp.as_ptr(),
                ri.as_ptr(),
                vals.as_ptr(),
                kind,
                fact,
                order,
                pivot,
                max_bytes,
                b.as_ptr(),
                x.as_mut_ptr(),
                &mut r,
            )
        };
        (st, r)
    }
}

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

fn upper_csr(a: &CscMatrix<C>) -> (Vec<i32>, Vec<i32>, Vec<C>) {
    let n = a.n;
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
            let r = a.row_idx[k];
            debug_assert!(r >= c);
            let d = next[c] as usize;
            ja[d] = r as i32;
            va[d] = a.values[k];
            next[c] += 1;
        }
    }
    (ia, ja, va)
}

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
    (0..n)
        .map(|i| Complex::new((i % 5) as f64 - 2.0, (i % 3) as f64 - 1.0))
        .collect()
}
fn rel(ax: &[C], b: &[C]) -> f64 {
    let num: f64 = (0..b.len())
        .map(|i| (ax[i] - b[i]).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den.max(1e-300)
}

fn cube(target: usize) -> [usize; 3] {
    let k = (target as f64).cbrt().round().max(2.0) as usize;
    [k, k, k]
}

#[allow(clippy::too_many_arguments)]
fn emit(
    out: &mut dyn Write,
    solver: &str,
    family: &str,
    name: &str,
    n: usize,
    nnz: usize,
    threads: i32,
    mem: bool,
    ana: f64,
    fac: f64,
    slv: f64,
    memmb: f64,
    fill: usize,
    res: f64,
) {
    let _ = writeln!(
        out,
        "{{\"solver\":\"{solver}\",\"family\":\"{family}\",\"name\":\"{name}\",\"n\":{n},\"nnz\":{nnz},\"threads\":{threads},\"metric\":\"{}\",\"ana_ms\":{ana:.3},\"fac_ms\":{fac:.3},\"slv_ms\":{slv:.3},\"mem_mb\":{memmb:.1},\"fill\":{fill},\"res\":{res:.3e}}}",
        if mem { "mem" } else { "time" }
    );
}

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

#[path = "../tests/common/xorshift.rs"]
mod xorshift;
use xorshift::Rng;

fn circuit_matrix(n: usize, n_stages: usize, seed: u64) -> GeneralCsc<C> {
    let mut rng = Rng(seed | 1);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut colsum = vec![0.0f64; n];
    let stage = n / n_stages;
    for j in 0..n {
        let s = (j / stage).min(n_stages - 1);
        let (lo, hi) = (
            s * stage,
            if s == n_stages - 1 {
                n
            } else {
                (s + 1) * stage
            },
        );
        let span = hi - lo;
        for &t in &[1usize, 2, 17] {
            let i = lo + (j - lo + t) % span;
            if i != j {
                let g = 0.5 + rng.next_f64().abs();
                r.push(i);
                c.push(j);
                v.push(Complex::new(-g, 0.0));
                colsum[j] += g;
                let g2 = 0.5 + rng.next_f64().abs();
                r.push(j);
                c.push(i);
                v.push(Complex::new(-g2, 0.0));
                colsum[i] += g2;
            }
        }
        if s > 0 && (j - lo) % 3 == 0 {
            let i = lo - stage + (j - lo) % stage;
            let g = 0.2 + 0.3 * rng.next_f64().abs();
            r.push(i);
            c.push(j);
            v.push(Complex::new(-g, 0.0));
            colsum[j] += g;
        }
    }
    for (j, &cs) in colsum.iter().enumerate() {
        let margin = 0.1 + 0.1 * rng.next_f64().abs();
        let omega_c = 0.2 + 0.4 * rng.next_f64().abs();
        r.push(j);
        c.push(j);
        v.push(Complex::new(cs + margin, omega_c));
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).expect("circuit_matrix")
}

fn build_faer(a: &GeneralCsc<C>) -> Option<SparseColMat<usize, c64>> {
    let mut trip: Vec<Triplet<usize, usize, c64>> = Vec::with_capacity(a.values.len());
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            trip.push(Triplet::new(
                a.row_idx[k],
                j,
                c64::new(a.values[k].re, a.values[k].im),
            ));
        }
    }
    SparseColMat::try_new_from_triplets(a.n, a.n, &trip).ok()
}

#[allow(clippy::too_many_arguments)]
fn run_matrix(
    out: &mut dyn Write,
    family: &str,
    name: &str,
    mat: &Mat,
    threads: i32,
    mem: bool,
    solvers: &[String],
    opts: &SolverSettings,
) {
    let (n, nnz) = (mat.n(), mat.nnz());
    let b = rhs(n);
    let has = |s: &str| solvers.iter().any(|x| x == s);

    if let Mat::Unsym(a) = mat {
        if let Ok(sym) = LuSymbolic::analyze(a, &SolverSettings::default()) {
            eprintln!("[bench] {name} a-priori {}", sym.estimate_memory::<C>());
        }
    }

    for (tag, o) in [("default", SolverSettings::default()), ("ll", opts.clone())] {
        if !has(tag) {
            continue;
        }
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
                let sym = skip_err!(
                    "analyze",
                    LdltSymbolic::analyze(a, &SolverSettings::default())
                );
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
                emit(
                    out,
                    tag,
                    family,
                    name,
                    n,
                    nnz,
                    threads,
                    mem,
                    ana,
                    fac,
                    slv,
                    mm,
                    f.factor_nnz(),
                    rel(&ax, &b),
                );
            }
            Mat::Unsym(a) => {
                let t = Instant::now();
                let sym = skip_err!(
                    "analyze",
                    LuSymbolic::analyze(a, &SolverSettings::default())
                );
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
                emit(
                    out,
                    tag,
                    family,
                    name,
                    n,
                    nnz,
                    threads,
                    mem,
                    ana,
                    fac,
                    slv,
                    mm,
                    f.factor_nnz(),
                    rel(&ax, &b),
                );
            }
        }
    }

    if has("auto") {
        let res: Result<(), rslab::RslabError> = (|| {
            match mat {
                Mat::Sym(a) => {
                    let t = Instant::now();
                    let s = rslab::SolverSettings::default();
                    let sym = rslab::LdltSymbolic::analyze(a, &s)?;
                    let ana = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    let (fr, mm) = live_peak(|| sym.factor(a, &s));
                    let f = fr?;
                    let fac = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    let x = f.solve(&b)?;
                    let slv = t.elapsed().as_secs_f64() * 1e3;
                    let mut ax = vec![Complex::new(0.0, 0.0); n];
                    a.symv(&x, &mut ax);
                    emit(
                        out,
                        "auto",
                        family,
                        name,
                        n,
                        nnz,
                        threads,
                        mem,
                        ana,
                        fac,
                        slv,
                        mm,
                        f.factor_nnz(),
                        rel(&ax, &b),
                    );
                }
                Mat::Unsym(a) => {
                    let t = Instant::now();
                    let s = rslab::SolverSettings::default();
                    let sym = rslab::LuSymbolic::analyze(a, &s)?;
                    let ana = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    let (fr, mm) = live_peak(|| sym.factor(a, &s));
                    let f = fr?;
                    let fac = t.elapsed().as_secs_f64() * 1e3;
                    let t = Instant::now();
                    let x = f.solve(&b)?;
                    let slv = t.elapsed().as_secs_f64() * 1e3;
                    let mut ax = vec![Complex::new(0.0, 0.0); n];
                    a.matvec(&x, &mut ax);
                    emit(
                        out,
                        "auto",
                        family,
                        name,
                        n,
                        nnz,
                        threads,
                        mem,
                        ana,
                        fac,
                        slv,
                        mm,
                        f.factor_nnz(),
                        rel(&ax, &b),
                    );
                }
            }
            Ok(())
        })();
        if let Err(e) = res {
            eprintln!("[auto] {name}: failed: {e:?}");
        }
    }

    // https://github.com/milanofthe/rslab/issues/294
    let faer_max: usize = std::env::var("RLA_BENCH_FAER_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000);
    let faer_est_mb: f64 = std::env::var("RLA_BENCH_FAER_EST_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000.0);
    let rslab_est_mb = match mat {
        Mat::Sym(a) => LdltSymbolic::analyze(a, &SolverSettings::default())
            .ok()
            .map(|s| s.estimate_memory::<C>().transient_peak_bytes),
        Mat::Unsym(a) => LuSymbolic::analyze(a, &SolverSettings::default())
            .ok()
            .map(|s| s.estimate_memory::<C>().transient_peak_bytes),
    }
    .map(|b| b as f64 / 1e6)
    .unwrap_or(0.0);
    let faer_ok = n <= faer_max && rslab_est_mb < faer_est_mb;
    if has("faer") && !faer_ok {
        eprintln!("[faer] skip {name} (n={n}, RSLAB est {rslab_est_mb:.0} MB => faer over budget)");
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
                let x: Vec<C> = (0..n)
                    .map(|i| Complex::new(xb[(i, 0)].re, xb[(i, 0)].im))
                    .collect();
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                mat.resid(&x, &mut ax);
                emit(
                    out,
                    "faer",
                    family,
                    name,
                    n,
                    nnz,
                    threads,
                    mem,
                    0.0,
                    fac,
                    slv,
                    mm,
                    0,
                    rel(&ax, &b),
                );
            }
        }
    }

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
            let (mut db, mut dx) = (
                vec![Complex::new(0.0, 0.0); n],
                vec![Complex::new(0.0, 0.0); n],
            );
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
                    // https://github.com/milanofthe/rslab/issues/294
                    let (symbolic_peak_kb, symbolic_permanent_kb, numeric_kb) =
                        (ps.iparm[14], ps.iparm[15], ps.iparm[16]);
                    let peak_kb = symbolic_peak_kb.max(symbolic_permanent_kb + numeric_kb);
                    let mm = peak_kb.max(0) as f64 / 1024.0;
                    let fill = ps.iparm[17].max(0) as usize;
                    let mut ax = vec![Complex::new(0.0, 0.0); n];
                    mat.resid(&x, &mut ax);
                    emit(
                        out,
                        "pardiso",
                        family,
                        name,
                        n,
                        nnz,
                        threads,
                        mem,
                        ana,
                        fac,
                        slv,
                        mm,
                        fill,
                        rel(&ax, &b),
                    );
                }
            } else {
                eprintln!("[bench] PARDISO error {e1}/{e2}");
            }
        } else {
            eprintln!("[bench] PARDISO unavailable (mkl_rt not found)");
        }
    }

    if has("klu") {
        if let Mat::Unsym(a) = mat {
            let ks = KluSettings::default().with_threads(opts.threads);
            let outcome = (|| -> Result<_, rslab::RslabError> {
                let t = Instant::now();
                let sym = KluSymbolic::analyze(a, &ks)?;
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &ks));
                let f = fr?;
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b)?;
                let slv = t.elapsed().as_secs_f64() * 1e3;
                Ok((ana, fac, slv, mm, f.factor_nnz(), x))
            })();
            match outcome {
                Ok((ana, fac, slv, mm, fill, x)) => {
                    let mut ax = vec![Complex::new(0.0, 0.0); n];
                    a.matvec(&x, &mut ax);
                    emit(
                        out,
                        "klu",
                        family,
                        name,
                        n,
                        nnz,
                        threads,
                        mem,
                        ana,
                        fac,
                        slv,
                        mm,
                        fill,
                        rel(&ax, &b),
                    );
                }
                Err(e) => eprintln!("[klu] {name}: failed: {e:?}"),
            }
        }
    }

    // https://github.com/milanofthe/rslab/issues/294
    if has("accel") {
        if let Some(ac) = Accel::try_new() {
            // https://github.com/milanofthe/rslab/issues/294
            let order: i32 = match std::env::var("RLA_BENCH_ACCEL_ORDER").ok().as_deref() {
                Some("default") => -1,
                Some("amd") => 2,
                Some("metis") => 3,
                Some("colamd") => 4,
                _ => ACCEL_ORDER_BAKEOFF,
            };
            let pivot: f64 = std::env::var("RLA_BENCH_ACCEL_PIVOT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(-1.0);
            // https://github.com/milanofthe/rslab/issues/294
            let gate_bytes: i64 = (std::env::var("RLA_BENCH_ACCEL_EST_MB")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(9000.0)
                * 1e6) as i64;
            let mut x = vec![Complex::new(0.0, 0.0); n];
            let (variant, st, r) = match mat {
                Mat::Sym(a) if a.values.iter().all(|v| v.im == 0.0) => {
                    let (st, r) = ac.solve(
                        n, &a.col_ptr, &a.row_idx, &a.values, 2, 0, order, pivot, gate_bytes, &b,
                        &mut x,
                    );
                    if st == 0 || st == -100 {
                        ("chol", st, r)
                    } else {
                        let (st, r) = ac.solve(
                            n, &a.col_ptr, &a.row_idx, &a.values, 2, 1, order, pivot, gate_bytes,
                            &b, &mut x,
                        );
                        ("ldlt", st, r)
                    }
                }
                Mat::Sym(a) => {
                    let fa = sym_to_full(a);
                    let (st, r) = ac.solve(
                        n,
                        &fa.col_ptr,
                        &fa.row_idx,
                        &fa.values,
                        0,
                        80,
                        order,
                        pivot,
                        gate_bytes,
                        &b,
                        &mut x,
                    );
                    ("lu-full", st, r)
                }
                Mat::Unsym(a) => {
                    let (st, r) = ac.solve(
                        n, &a.col_ptr, &a.row_idx, &a.values, 0, 80, order, pivot, gate_bytes, &b,
                        &mut x,
                    );
                    ("lu", st, r)
                }
            };
            let ord_name = |o: i32| match o {
                2 => "amd",
                3 => "metis",
                4 => "colamd",
                _ => "default",
            };
            if st == -100 {
                eprintln!(
                    "[accel] skip {name}: {variant} a-priori {:.0} MB factor + {:.0} MB workspace over {:.0} MB gate",
                    r.factor_bytes as f64 / 1e6,
                    r.workspace_bytes as f64 / 1e6,
                    gate_bytes as f64 / 1e6
                );
            } else if st == 0 {
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                mat.resid(&x, &mut ax);
                eprintln!(
                    "[accel] {name}: {variant} ({}), a-priori factor {:.0} MB + workspace {:.0} MB",
                    ord_name(r.chosen_order),
                    r.factor_bytes as f64 / 1e6,
                    r.workspace_bytes as f64 / 1e6
                );
                emit(
                    out,
                    "accel",
                    family,
                    name,
                    n,
                    nnz,
                    threads,
                    mem,
                    r.ana_s * 1e3,
                    r.fac_s * 1e3,
                    r.slv_s * 1e3,
                    r.peak_bytes as f64 / 1e6,
                    0,
                    rel(&ax, &b),
                );
            } else {
                eprintln!(
                    "[accel] {name}: {variant} failed (status {st}, sym {}, num {})",
                    r.sym_status, r.num_status
                );
            }
        }
    }

    // https://github.com/milanofthe/rslab/issues/294
    if has("pc") {
        let envf = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let floor = envf("RLA_BENCH_PC_FLOOR", 1e-4);
        let maxit = envf("RLA_BENCH_PC_MAXIT", 200.0) as usize;
        let restart = envf("RLA_BENCH_PC_RESTART", 100.0) as usize;
        let pc_opts = SolverSettings::preconditioner(floor).with_threads(threads.max(1) as usize);
        let outcome = match mat {
            Mat::Sym(a) => LdltSymbolic::analyze(a, &SolverSettings::default()).and_then(|sym| {
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &pc_opts));
                let f = fr?;
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let kr = gmres(
                    a,
                    &b,
                    &f,
                    &rslab::KrylovSettings::default()
                        .with_tol(1e-10)
                        .with_max_iter(maxit)
                        .with_restart(restart),
                    None,
                )?;
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.symv(&kr.x, &mut ax);
                Ok((
                    fac,
                    slv,
                    mm,
                    f.factor_nnz(),
                    rel(&ax, &b),
                    kr.iters,
                    kr.converged,
                ))
            }),
            Mat::Unsym(a) => LuSymbolic::analyze(a, &SolverSettings::default()).and_then(|sym| {
                let t = Instant::now();
                let (fr, mm) = live_peak(|| sym.factor(a, &pc_opts));
                let f = fr?;
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let kr = gmres(
                    a,
                    &b,
                    &f,
                    &rslab::KrylovSettings::default()
                        .with_tol(1e-10)
                        .with_max_iter(maxit)
                        .with_restart(restart),
                    None,
                )?;
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.matvec(&kr.x, &mut ax);
                Ok((
                    fac,
                    slv,
                    mm,
                    f.factor_nnz(),
                    rel(&ax, &b),
                    kr.iters,
                    kr.converged,
                ))
            }),
        };
        match outcome {
            Ok((fac, slv, mm, fill, res, iters, conv)) => {
                eprintln!("[pc] {name}: {iters} GMRES iters, converged={conv}, res={res:.1e}");
                emit(
                    out, "pc", family, name, n, nnz, threads, mem, 0.0, fac, slv, mm, fill, res,
                );
            }
            Err(e) => eprintln!("[pc] {name}: factor/refine failed: {e:?}"),
        }
    }
}

// https://github.com/milanofthe/rslab/issues/294
fn build_family(family: &str, sizes: &[usize]) -> Vec<(String, Mat)> {
    match family {
        "sym" => sizes
            .iter()
            .enumerate()
            .map(|(i, &sz)| {
                let kc = ((sz as f64 / 3.0).cbrt().round() as usize).max(4);
                let ks = ((sz as f64 / 3.0).sqrt().round() as usize).max(4);
                let helm = |re, im| {
                    let m = stencil::helmholtz(
                        &cube(sz),
                        Complex::new(re, im),
                        &stencil::StencilOpts::default(),
                    );
                    (format!("helmholtz_{}", m.n), Mat::Sym(m))
                };
                let cc = |om, si| {
                    let m = fem::curl_curl(&[kc, kc, kc], om, si);
                    (format!("curlcurl_{}", m.n), Mat::Sym(m))
                };
                let sp = |beta| {
                    let m = fem::saddle_point::<C>(&[ks, ks], beta);
                    (format!("saddle_{}", m.n), Mat::Sym(m))
                };
                match i % 8 {
                    0 => helm(0.02, 0.01),
                    1 => helm(0.05, 0.02),
                    2 => helm(0.12, 0.06),
                    3 => cc(1.5, 0.05),
                    4 => cc(3.0, 0.1),
                    5 => cc(6.0, 0.3),
                    6 => sp(0.1),
                    _ => sp(0.01),
                }
            })
            .collect(),
        "unsym" => sizes
            .iter()
            .enumerate()
            .map(|(i, &sz)| {
                let kc = ((sz as f64).cbrt().round() as usize).max(4);
                let ks = ((sz as f64).sqrt().round() as usize).max(4);
                let cd3 = |eps| {
                    let m = fem::convection_diffusion::<C>(
                        &[kc, kc, kc],
                        eps,
                        fem::Flow::Rotating,
                        true,
                    );
                    (format!("convdiff3d_{}", m.n), Mat::Unsym(m))
                };
                let cd2 = |eps, flow, up| {
                    let m = fem::convection_diffusion::<C>(&[ks, ks], eps, flow, up);
                    (format!("convdiff2d_{}", m.n), Mat::Unsym(m))
                };
                let mom = |deg: f64| {
                    let cutoff = (2.0 * (deg / sz as f64).sqrt()).min(1.2);
                    let m = bem::kernel(
                        sz,
                        &bem::BemOpts {
                            cutoff,
                            ..Default::default()
                        },
                    );
                    (format!("mom_{}", m.n), Mat::Unsym(m))
                };
                match i % 8 {
                    0 => cd3(0.05),
                    1 => cd3(5e-3),
                    2 => cd2(0.05, fem::Flow::Diagonal, false),
                    3 => cd2(5e-3, fem::Flow::Rotating, true),
                    4 => cd2(1e-3, fem::Flow::Diagonal, true),
                    5 => cd2(1e-2, fem::Flow::Rotating, false),
                    6 => mom(80.0),
                    _ => mom(160.0),
                }
            })
            .collect(),
        "circuit" => sizes
            .iter()
            .map(|&sz| {
                let stages = ((8.0 * (sz as f64 / 2000.0).powf(0.43)).round() as usize).max(2);
                let m = circuit_matrix(sz, stages, 0xC0FFEE ^ sz as u64);
                (format!("circuit_{}", m.n), Mat::Unsym(m))
            })
            .collect(),
        "corpus" => build_corpus(),
        "real" => {
            let dir = std::env::var("RLA_BENCH_REAL_DIR")
                .expect("the real suite reads its .mtx files from RLA_BENCH_REAL_DIR");
            let count: usize = std::env::var("RLA_BENCH_REAL_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6);
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
                    let a = parse_mtx_complex_general(&contents, &stem)
                        .ok()?
                        .to_general_csc()
                        .ok()?;
                    Some((stem, Mat::Unsym(a)))
                })
                .collect()
        }
        _ => Vec::new(),
    }
}

// https://github.com/milanofthe/rslab/issues/294
#[cfg(feature = "matgen-download")]
fn build_corpus() -> Vec<(String, Mat)> {
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
    let mem = std::env::var("RLA_BENCH_MEM")
        .map(|v| v == "1")
        .unwrap_or(false);
    let solvers: Vec<String> = std::env::var("RLA_BENCH_SOLVERS")
        .unwrap_or_else(|_| "ll,mf,faer,pardiso".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let threads: i32 = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get() as i32)
                .unwrap_or(1)
        });
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
    let mut opts = SolverSettings::default().with_threads(threads.max(1) as usize);
    if let Ok(nb) = std::env::var("RLA_BENCH_NB") {
        if let Ok(nb) = nb.parse::<usize>() {
            opts.kernels.panel_nb = nb;
        }
    }
    if let Ok(g) = std::env::var("RLA_BENCH_GATES") {
        let v: Vec<usize> = g.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() == 3 {
            opts.kernels.scalar_gate = v[0];
            opts.kernels.par_gemm = v[1];
            opts.kernels.par_cdiv = v[2];
        }
    }

    let estimate_only = std::env::var("RLA_BENCH_ESTIMATE")
        .map(|v| v == "1")
        .unwrap_or(false);

    for (name, mat) in build_family(&family, &sizes) {
        if estimate_only {
            let e = match &mat {
                Mat::Sym(a) => LdltSymbolic::analyze(a, &SolverSettings::default())
                    .ok()
                    .map(|s| s.estimate_memory::<C>()),
                Mat::Unsym(a) => LuSymbolic::analyze(a, &SolverSettings::default())
                    .ok()
                    .map(|s| s.estimate_memory::<C>()),
            };
            if let Some(e) = e {
                let mb = |b: u64| b as f64 / 1e6;
                let scratch =
                    mb(e.transient_peak_bytes) - mb(e.panels_all_bytes) - mb(e.factor_bytes);
                let _ = writeln!(
                    out,
                    "{{\"name\":\"{name}\",\"n\":{},\"panels_mb\":{:.1},\"factor_mb\":{:.1},\"scratch_mb\":{:.1},\"transient_mb\":{:.1},\"freed_floor_mb\":{:.1}}}",
                    mat.n(), mb(e.panels_all_bytes), mb(e.factor_bytes), scratch.max(0.0),
                    mb(e.transient_peak_bytes), mb(e.panel_live_peak_bytes),
                );
            }
            continue;
        }
        eprintln!(
            "[bench] family={family} name={name} n={} threads={threads} metric={}",
            mat.n(),
            if mem { "mem" } else { "time" }
        );
        run_matrix(
            &mut out, &family, &name, &mat, threads, mem, &solvers, &opts,
        );
    }
}
