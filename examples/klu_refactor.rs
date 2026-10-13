//! The KLU refactor and solve of a real Matrix Market matrix, sequential and
//! pipelined: per variant the best time of `--reps` runs, a buffer of
//! `--interleave <MB>` written before each run, as a solver's evaluations
//! evict the caches between its factorizations. With `--spin` the parallel
//! sections run on a lent pool whose workers wait awake between sections
//! (as a solver's evaluation workers do), not on rayon.
//!
//! `cargo run --release --example klu_refactor -- <file.mtx> [threads] [--reps n] [--interleave MB] [--spin]`
use rslab::{GeneralCsc, KluParallel, KluSettings, MtxLoaded};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::time::Instant;

/// Workers that spin between sections, the caller taking pieces too.
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
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<usize>().ok())
    };
    let reps = opt("--reps").unwrap_or(20);
    let mut buf = vec![0u8; opt("--interleave").unwrap_or(0) << 20];
    let mut pollute = || {
        for (i, b) in buf.iter_mut().enumerate().step_by(64) {
            *b = b.wrapping_add(i as u8);
        }
    };
    let threads = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
    let spin = args.iter().any(|a| a == "--spin");
    let MtxLoaded::General(a) =
        rslab::read_mtx_any(std::path::Path::new(&args[1])).expect("read mtx")
    else {
        panic!("a general matrix");
    };
    let a = GeneralCsc {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a.values.iter().map(|v| v.re).collect::<Vec<f64>>(),
    };
    let b: Vec<f64> = (0..a.n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
    println!(
        "n={} nnz={} threads={} pool={}",
        a.n,
        a.values.len(),
        rayon::current_num_threads(),
        if spin { "spinning" } else { "rayon" }
    );
    let variants = [
        (
            "sequential",
            KluSettings::default().with_parallel(KluParallel::Off),
        ),
        ("auto", KluSettings::default()),
        (
            "pipelined",
            KluSettings::default().with_parallel(KluParallel::On),
        ),
    ];
    let mut run = || {
        for (label, settings) in &variants {
            let sym = rslab::KluSymbolic::analyze(&a, settings).unwrap();
            let mut k = sym.factor(&a, settings).unwrap();
            let (mut tr, mut ts) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..reps {
                pollute();
                let t = Instant::now();
                k.refactor(&a).unwrap();
                tr = tr.min(t.elapsed().as_secs_f64());
                pollute();
                let t = Instant::now();
                let x = k.solve(&b).unwrap();
                ts = ts.min(t.elapsed().as_secs_f64());
                std::hint::black_box(x);
            }
            println!(
                "{label:>10} | refactor {:>7.3} ms  solve {:>7.3} ms",
                tr * 1e3,
                ts * 1e3,
            );
        }
    };
    if spin {
        let pool: Arc<dyn rslab::parallel::Pool> = Arc::new(Spin::new(threads.max(1)));
        rslab::parallel::install(pool, run);
    } else {
        run();
    }
}
