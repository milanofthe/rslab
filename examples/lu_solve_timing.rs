//! Time the LU solves of a complex Matrix Market matrix over right-hand-side counts, and
//! how far a column of a block solve is from the single solve of that column (relative to
//! its largest entry; the block solve's products sum in their own order).
//! `cargo run --release --example lu_solve_timing -- <matrix.mtx> [reps] [nrhs,..]`
//!
//! The solves are applies of the factor as a block preconditioner
//! (`Preconditioner::apply_block` into a held block), on the pool the process runs
//! (`RAYON_NUM_THREADS`, `Threads::Ambient`), after one warm-up apply; the median and the
//! fastest of `reps` are printed. Environment:
//! - `RSLAB_DELAYED=1`: delayed pivots with the row matching (the near-field LU of a MoM
//!   preconditioner), else pivots stay in their front without matching;
//! - `RSLAB_TRACE=1`: the phases of the sweeps from the solver's own trace (the `debug` log),
//!   medians over a second round of `reps`, the way each sweep ran (pool or calling
//!   thread) and the solve layout of both factors;
//! - `RSLAB_COPIES=k`: `k` threads at once, each applying its own copy of the factor on
//!   its thread (run with `RAYON_NUM_THREADS=1`), against the factor's memory bandwidth;
//! - `RSLAB_RELAX=width,rows`: a relaxed amalgamation, `RSLAB_NEMIN` the amalgamation's
//!   width.
use num_complex::Complex;
use rslab::{
    LogLevel, LogSink, LuSolver, OrderingMethod, Preconditioner, SolverSettings, Threads,
    ZeroPivotAction,
};
use std::sync::Mutex;
use std::time::Instant;

type C = Complex<f64>;

/// The `debug` records of a traced round.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Collect;

impl LogSink for Collect {
    fn emit(&self, _: LogLevel, msg: &str) {
        LINES.lock().unwrap().push(msg.to_string());
    }
}

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    match v.len() {
        0 => f64::NAN,
        k if k % 2 == 1 => v[k / 2],
        k => 0.5 * (v[k / 2 - 1] + v[k / 2]),
    }
}

/// `apply_block` of `nr` columns `reps` times after one warm-up, in ms.
fn times(s: &LuSolver<C>, b: &[C], x: &mut [C], nr: usize, reps: usize) -> Vec<f64> {
    let n = b.len() / nr;
    s.apply_block(b, x, nr, n).unwrap();
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            s.apply_block(b, x, nr, n).unwrap();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect()
}

/// The `name=..ms` laps of a record, added to `sums`.
fn laps(line: &str, sums: &mut Vec<(String, f64)>) {
    for tok in line.split_whitespace() {
        let Some((k, v)) = tok.split_once('=') else {
            continue;
        };
        let Some(ms) = v.strip_suffix("ms").and_then(|m| m.parse::<f64>().ok()) else {
            continue;
        };
        match sums.iter_mut().find(|(n, _)| n == k) {
            Some(e) => e.1 += ms,
            None => sums.push((k.to_string(), ms)),
        }
    }
}

/// One traced round: per apply the sweeps' laps, the ways the sweeps ran and the whole
/// solve as the solver timed it; medians over the round.
fn trace(s: &LuSolver<C>, b: &[C], x: &mut [C], nr: usize, reps: usize) {
    let n = b.len() / nr;
    rslab::logging::set_level(LogLevel::Debug);
    let mut per: Vec<Vec<(String, f64)>> = Vec::new();
    let mut ways: Vec<String> = Vec::new();
    for _ in 0..reps {
        LINES.lock().unwrap().clear();
        s.apply_block(b, x, nr, n).unwrap();
        let lines = std::mem::take(&mut *LINES.lock().unwrap());
        let mut sums = Vec::new();
        let mut way = Vec::new();
        for l in &lines {
            if l.starts_with("forward-block:") || l.starts_with("backward-block:") {
                laps(l.split_once(':').unwrap().1, &mut sums);
            } else if let Some(rest) = l.strip_prefix("solve: sweep nr=") {
                way.push(if rest.contains("-> pool") {
                    "pool"
                } else {
                    "here"
                });
            } else if l.starts_with("lu solve:") {
                if let Some(ms) = l
                    .split_whitespace()
                    .rev()
                    .nth(1)
                    .and_then(|m| m.parse().ok())
                {
                    sums.push(("solve".to_string(), ms));
                }
            }
        }
        per.push(sums);
        ways.push(way.join("/"));
    }
    rslab::logging::set_level(LogLevel::Warning);
    let names: Vec<String> = per[0].iter().map(|(n, _)| n.clone()).collect();
    let mut out = Vec::new();
    for name in &names {
        let v: Vec<f64> = (per.iter())
            .filter_map(|p| p.iter().find(|(k, _)| k == name).map(|e| e.1))
            .collect();
        out.push(format!("{name}={:.2}", median(v)));
    }
    ways.dedup();
    println!(
        "  phases nr={nr:2} (ms, median of {reps}): {}  ways fwd/bwd: {}",
        out.join(" "),
        ways.join(", ")
    );
}

fn main() {
    let path = std::env::args().nth(1).expect("matrix path");
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let counts: Vec<usize> = std::env::args()
        .nth(3)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16]);
    let rslab::MtxLoaded::General(a) = rslab::read_mtx_any(std::path::Path::new(&path)).unwrap()
    else {
        panic!("a general matrix")
    };
    let n = a.n;
    let delayed = flag("RSLAB_DELAYED");
    let mut opts = SolverSettings::exact()
        .with_zero_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-6 })
        .with_delayed_pivots(delayed)
        .with_matching(delayed)
        .with_ordering(OrderingMethod::MetisND)
        // the pool it runs on (RAYON_NUM_THREADS), as a host that lends its workers does
        .with_threads(Threads::Ambient);
    if let Ok(r) = std::env::var("RSLAB_RELAX") {
        let v: Vec<usize> = r.split(',').map(|x| x.parse().unwrap()).collect();
        opts = opts.with_relax(Some(rslab::RelaxAmalgamation {
            max_width: v[0],
            max_extra_rows: v[1],
        }));
    }
    if let Some(n) = std::env::var("RSLAB_NEMIN")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        opts = opts.with_nemin(n);
    }
    let traced = flag("RSLAB_TRACE");
    if traced {
        rslab::logging::set_sink(Box::new(Collect));
        rslab::logging::set_level(LogLevel::Debug);
    }
    let t = Instant::now();
    let s = LuSolver::factor(&a, &opts).unwrap();
    let factor_s = t.elapsed().as_secs_f64();
    if traced {
        rslab::logging::set_level(LogLevel::Warning);
        for l in std::mem::take(&mut *LINES.lock().unwrap()) {
            if l.starts_with("solve layout") {
                println!("  {l}");
            }
        }
    }
    println!(
        "n={n} fill={} factor {factor_s:.2} s threads={} delayed={delayed}",
        s.factor_nnz(),
        rayon::current_num_threads()
    );
    let rhs = |k: usize| C::new(((k * 7) % 13) as f64 - 6.0, ((k * 3) % 5) as f64 - 2.0);
    let single: Vec<C> = (0..n).map(rhs).collect();
    let x1 = s.solve(&single).unwrap();
    let copies: usize = std::env::var("RSLAB_COPIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for nrhs in counts {
        let bb: Vec<C> = (0..n * nrhs).map(rhs).collect();
        if copies > 0 {
            // every thread its own factor, its applies on its own thread
            let start = std::sync::Barrier::new(copies);
            let per: Vec<f64> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..copies)
                    .map(|_| {
                        let (s, bb, start) = (s.clone(), &bb, &start);
                        sc.spawn(move || {
                            let mut x = vec![C::new(0.0, 0.0); n * nrhs];
                            start.wait();
                            median(times(&s, bb, &mut x, nrhs, reps))
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let lo = per.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = per.iter().copied().fold(0.0, f64::max);
            println!(
                "copies={copies} nrhs={nrhs:2}: {:7.2} ms median per apply (threads {lo:.2} .. {hi:.2})",
                median(per)
            );
            continue;
        }
        let mut x = vec![C::new(0.0, 0.0); n * nrhs];
        let t = times(&s, &bb, &mut x, nrhs, reps);
        let best = t.iter().copied().fold(f64::INFINITY, f64::min);
        let med = median(t);
        let scale = x1.iter().map(|v| v.norm()).fold(0.0, f64::max);
        let dev = x[..n]
            .iter()
            .zip(&x1)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max)
            / scale;
        println!(
            "nrhs={nrhs:2}: {med:7.2} ms median, {best:7.2} ms best, {:6.2} ms per column, {:5.1} GB/s of factor, column 0 against the single solve {dev:.1e}",
            med / nrhs as f64,
            s.factor_nnz() as f64 * 16.0 / med / 1e6,
        );
        if traced {
            trace(&s, &bb, &mut x, nrhs, reps);
        }
    }
}
