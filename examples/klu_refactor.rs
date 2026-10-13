//! The KLU refactor and solve of a real Matrix Market matrix, sequential and
//! pipelined: per variant the best time of `--reps` runs, a buffer of
//! `--interleave <MB>` written before each run, as a solver's evaluations
//! evict the caches between its factorizations.
//!
//! `cargo run --release --example klu_refactor -- <file.mtx> [threads] [--reps n] [--interleave MB]`
use rslab::{GeneralCsc, KluParallel, KluSettings, MtxLoaded};
use std::time::Instant;

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
        "n={} nnz={} threads={}",
        a.n,
        a.values.len(),
        rayon::current_num_threads()
    );
    let variants = [
        (
            "sequential",
            KluSettings::default().with_parallel(KluParallel::Off),
        ),
        ("auto", KluSettings::default()),
        ("pipelined", {
            let mut s = KluSettings::default().with_parallel(KluParallel::On);
            s.par_min_work = 0;
            s.par_min_ratio = 0.0;
            s
        }),
    ];
    for (label, settings) in variants {
        let sym = rslab::KluSymbolic::analyze(&a, &settings).unwrap();
        let mut k = sym.factor(&a, &settings).unwrap();
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
            "{label:>10} | refactor {:>7.3} ms  solve {:>7.3} ms | {}",
            tr * 1e3,
            ts * 1e3,
            k.diagnostics()
        );
    }
}
