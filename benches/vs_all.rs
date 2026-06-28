//! Head-to-head **exact** direct solve on the complex MoM matrices - RLA's two
//! factorization paths against the field:
//!   * **RLA-LL**  - supernodal left-looking LU (the default path)
//!   * **RLA-MF**  - multifrontal LU (assembly-tree of dense fronts)
//!   * **faer**    - faer sparse LU (pure-Rust competitor)
//!   * **PARDISO** - Intel MKL PARDISO, mtype=13 (complex unsymmetric), loaded at
//!                   runtime via `mkl_rt` (bench-only FFI; the solver lib stays
//!                   100% pure Rust). Skipped with a note if MKL is absent.
//!
//! All four do a **complete** (exact) factorization, so factor time / fill / true
//! residual are apples-to-apples. SuperLU (scipy) runs from the companion
//! `benches/superlu_mom.py` on the same matrices and `b` and is reported beside
//! this table.
//!
//! Run: `cargo bench --bench vs_all` (optionally `RLA_DIAG_FILTER=spiral`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// Counting allocator: tracks **live** bytes (current allocation, not OS working
// set), so the panel-freeing transient is visible even when the system allocator
// retains freed pages. Only sees Rust allocations - MKL/PARDISO bypass it.
struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
// Counting is opt-in (`RLA_LIVE_MEM=1`): off, the allocator is a thin System
// passthrough so factor timings are not perturbed by the per-alloc atomics.
static COUNTING_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
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
    unsafe fn realloc(&self, p: *mut u8, l: Layout, ns: usize) -> *mut u8 {
        let np = System.realloc(p, l, ns);
        if !np.is_null() && COUNTING_ON.load(Ordering::Relaxed) {
            if ns >= l.size() {
                let now = LIVE.fetch_add(ns - l.size(), Ordering::Relaxed) + (ns - l.size());
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(l.size() - ns, Ordering::Relaxed);
            }
        }
        np
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;
fn live_enabled() -> bool {
    COUNTING_ON.load(Ordering::Relaxed)
}

/// Run `f` tracking the **live-bytes** peak above the entry level. Returns the
/// result and the live transient in MB (the structural memory the factor needs).
fn live_peak<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let r = f();
    let peak = PEAK.load(Ordering::Relaxed);
    (r, (peak.saturating_sub(before)) as f64 / 1e6)
}

/// Factor-memory of a Rust solver: live-bytes peak in `RLA_LIVE_MEM=1` mode (the
/// accurate structural metric), else the OS working-set transient. The tag labels
/// which was used.
fn factor_mem<R>(f: impl FnOnce() -> R) -> (R, f64, &'static str) {
    if live_enabled() {
        let (r, m) = live_peak(f);
        (r, m, "live")
    } else {
        let (r, m) = sample(f);
        (r, m, "ws")
    }
}

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use faer::{c64, Mat};
use num_complex::Complex;
use rslab::prelude::*;
use rslab::{AnalyzeOptions, FactorMethod, FactorOptions, LuSymbolic, ReorderMode};

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
type C = Complex<f64>;

// ---- peak working-set sampler (Windows), for the memory transient ----
#[cfg(windows)]
fn cur_ws_mb() -> f64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        pfc: u32,
        peak_ws: usize,
        ws: usize,
        q1: usize,
        q2: usize,
        q3: usize,
        q4: usize,
        pf: usize,
        peak_pf: usize,
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(p: isize, c: *mut Pmc, cb: u32) -> i32;
    }
    // SAFETY: POD output buffer of the documented PROCESS_MEMORY_COUNTERS size.
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

/// Run `f`, sampling the process working set every 5 ms, and return its result
/// alongside the peak transient (MB above the working set just before `f`).
fn sample<R>(f: impl FnOnce() -> R) -> (R, f64) {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    let before = cur_ws_mb();
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(before as u64));
    let (s, p) = (stop.clone(), peak.clone());
    let h = std::thread::spawn(move || {
        while !s.load(Ordering::Relaxed) {
            p.fetch_max(cur_ws_mb() as u64, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
    let r = f();
    stop.store(true, Ordering::Relaxed);
    let _ = h.join();
    (r, (peak.load(Ordering::Relaxed) as f64 - before).max(0.0))
}

fn rel_resid(a: &rslab::GeneralCsc<C>, x: &[C], b: &[C]) -> f64 {
    let mut ax = vec![Complex::new(0.0, 0.0); a.n];
    a.matvec(x, &mut ax);
    let num: f64 = (0..a.n).map(|i| (ax[i] - b[i]).norm_sqr()).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den
}

/// CSC → 0-based CSR (transpose of storage; same matrix). PARDISO wants CSR with
/// ascending column indices per row - produced here by scanning columns in order.
fn build_full_csr(a: &rslab::GeneralCsc<C>) -> (Vec<i32>, Vec<i32>, Vec<C>) {
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
        iparm[7] = 2; // iterative refinement (≤2 steps) - PARDISO's default accuracy
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
        let (fres, mem, tag) = factor_mem(|| sym.factor(&a, &opts));
        match fres {
            Ok(f) => {
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b).unwrap();
                let slv = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "  {label}  ana {ana:7.0}  fac {fac:8.1}  slv {slv:6.1}  res {:.1e}  fill {:9}  {tag} +{mem:.0}MB",
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
                let (lures, mem, tag) = factor_mem(|| fa.sp_lu());
                match lures {
                    Ok(lu) => {
                        let fac = t.elapsed().as_secs_f64() * 1e3;
                        let mut xb = Mat::<c64>::from_fn(n, 1, |i, _| c64::new(b[i].re, b[i].im));
                        let t = Instant::now();
                        lu.solve_in_place(&mut xb);
                        let slv = t.elapsed().as_secs_f64() * 1e3;
                        let xf: Vec<C> = (0..n).map(|i| Complex::new(xb[(i, 0)].re, xb[(i, 0)].im)).collect();
                        println!(
                            "  faer     ana       -  fac {fac:8.1}  slv {slv:6.1}  res {:.1e}  fill         -  {tag} +{mem:.0}MB",
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
            let (e2, mem) = sample(|| ps.call(22, ni, &ia, &ja, &va, &mut dum_b, &mut dum_x));
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
                        "  PARDISO  ana {ana_p:7.0}  fac {fac:8.1}  slv {slv:6.1}  res {:.1e}  fill {fill:9}  ws +{mem:.0}MB",
                        rel_resid(&a, &x, &b)
                    );
                }
            }
        }
    }
}

fn main() {
    faer::set_global_parallelism(faer::Par::rayon(0));
    if std::env::var("RLA_LIVE_MEM").map(|v| v == "1").unwrap_or(false) {
        COUNTING_ON.store(true, Ordering::Relaxed);
    }
    let pardiso_ok = Pardiso::try_new().is_some();
    if !pardiso_ok {
        println!("(PARDISO: mkl_rt not found - skipping; install MKL or add it to PATH)");
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
