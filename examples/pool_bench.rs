//! Factorization time of the supernodal paths on a Matrix Market file, on
//! rayon and on a lent pool whose workers wait awake between sections (as
//! a solver's evaluation workers do; a section opened inside one of its
//! pieces runs serially there): a `symmetric` file on the LDL^T path, a
//! `general` one on the LU path, real when every value is.
//!
//! `cargo run --release --example pool_bench -- <file.mtx> [threads] [--reps n]`
use num_complex::Complex;
use rslab::{
    CscMatrix, GeneralCsc, LdltSymbolic, LuSymbolic, MtxLoaded, Scalar, SolverSettings, Threads,
};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::Instant;

/// Workers that spin between sections, the caller taking pieces too; a
/// section opened inside a piece runs on the calling thread.
struct Spin {
    threads: usize,
    shared: Arc<Shared>,
}

#[derive(Default)]
struct Shared {
    epoch: AtomicU64,
    job: AtomicPtr<&'static (dyn Fn(usize) + Sync)>,
    n: AtomicUsize,
    next: AtomicUsize,
    done: AtomicUsize,
    active: AtomicUsize,
    stop: AtomicBool,
    /// A section in progress: one opened meanwhile runs on its thread.
    busy: AtomicBool,
}

impl Shared {
    fn take(&self, f: &(dyn Fn(usize) + Sync)) {
        let n = self.n.load(SeqCst);
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= n {
                return;
            }
            f(i);
            self.done.fetch_add(1, SeqCst);
        }
    }
}

impl Spin {
    fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared::default());
        for _ in 1..threads {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let mut seen = 0;
                while !sh.stop.load(SeqCst) {
                    let e = sh.epoch.load(SeqCst);
                    if e == seen {
                        std::hint::spin_loop();
                        continue;
                    }
                    seen = e;
                    sh.active.fetch_add(1, SeqCst);
                    let job = sh.job.load(SeqCst);
                    if !job.is_null() {
                        // SAFETY: the job lives until `active` drops to zero.
                        sh.take(unsafe { *job });
                    }
                    sh.active.fetch_sub(1, SeqCst);
                }
            });
        }
        Spin { threads, shared }
    }
}

impl rslab::parallel::Pool for Spin {
    fn threads(&self) -> usize {
        self.threads
    }
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let sh = &*self.shared;
        if n < 2 || sh.busy.swap(true, SeqCst) {
            (0..n).for_each(f);
            return;
        }
        // SAFETY: `f` outlives the section; the workers drop it before return.
        let f_static: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let cell = f_static;
        sh.n.store(n, SeqCst);
        sh.next.store(0, SeqCst);
        sh.done.store(0, SeqCst);
        sh.job.store(&cell as *const _ as *mut _, SeqCst);
        sh.epoch.fetch_add(1, SeqCst);
        sh.take(f);
        while sh.done.load(SeqCst) < n {
            std::hint::spin_loop();
        }
        sh.job.store(std::ptr::null_mut(), SeqCst);
        while sh.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
        sh.busy.store(false, SeqCst);
    }
}

fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn real(v: &[Complex<f64>]) -> Option<Vec<f64>> {
    v.iter()
        .all(|c| c.im == 0.0)
        .then(|| v.iter().map(|c| c.re).collect())
}

fn ldlt<T: Scalar>(a: &CscMatrix<T>, s: &SolverSettings, reps: usize) -> f64 {
    let sym = LdltSymbolic::analyze(a, s).unwrap();
    best(reps, || {
        std::hint::black_box(sym.factor(a, s).unwrap());
    })
}

fn lu<T: Scalar>(a: &GeneralCsc<T>, s: &SolverSettings, reps: usize) -> f64 {
    let sym = LuSymbolic::analyze(a, s).unwrap();
    best(reps, || {
        std::hint::black_box(sym.factor(a, s).unwrap());
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(12);
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let s = SolverSettings::default().with_threads(Threads::Fixed(threads));
    let loaded = rslab::read_mtx_any(std::path::Path::new(&args[1])).expect("read mtx");
    let pool: Arc<dyn rslab::parallel::Pool> = Arc::new(Spin::new(threads));
    let (what, rayon, lent) = match loaded {
        MtxLoaded::Symmetric(a) => match real(&a.values) {
            Some(v) => {
                let a = CscMatrix {
                    n: a.n,
                    col_ptr: a.col_ptr.clone(),
                    row_idx: a.row_idx.clone(),
                    values: v,
                };
                let r = ldlt(&a, &s, reps);
                (
                    "LDLT real",
                    r,
                    rslab::parallel::install(pool, || ldlt(&a, &s, reps)),
                )
            }
            None => {
                let r = ldlt(&a, &s, reps);
                (
                    "LDLT complex",
                    r,
                    rslab::parallel::install(pool, || ldlt(&a, &s, reps)),
                )
            }
        },
        MtxLoaded::General(a) => match real(&a.values) {
            Some(v) => {
                let a = GeneralCsc {
                    n: a.n,
                    col_ptr: a.col_ptr.clone(),
                    row_idx: a.row_idx.clone(),
                    values: v,
                };
                let r = lu(&a, &s, reps);
                (
                    "LU real",
                    r,
                    rslab::parallel::install(pool, || lu(&a, &s, reps)),
                )
            }
            None => {
                let r = lu(&a, &s, reps);
                (
                    "LU complex",
                    r,
                    rslab::parallel::install(pool, || lu(&a, &s, reps)),
                )
            }
        },
    };
    println!("{what} threads={threads} factor: rayon {rayon:.1} ms, lent {lent:.1} ms");
}
