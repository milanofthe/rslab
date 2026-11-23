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
//! - `RSLAB_TRACE=dir`: a second round of `reps` applies per right-hand-side count in an
//!   rsprobe context at TRACE (the sweeps' phases, the ancestors' nodes, their triangles
//!   and products), its record written to `dir/nrhs<k>.json` for `python -m rsprobe
//!   report`; needs `--features rsprobe/on`;
//! - `RSLAB_COPIES=k`: `k` threads at once, each applying its own copy of the factor on
//!   its thread (run with `RAYON_NUM_THREADS=1`), against the factor's memory bandwidth;
//! - `RSLAB_RELAX=width,rows`: a relaxed amalgamation, `RSLAB_NEMIN` the amalgamation's
//!   width;
//! - `RSLAB_LEAF=k`: `SolveSettings::leaf_subtrees`, `RSLAB_GEMM_TASK=k`:
//!   `SolveSettings::gemm_task`.
use num_complex::Complex;
use rslab::{LuSolver, OrderingMethod, Preconditioner, SolverSettings, Threads, ZeroPivotAction};
use std::time::Instant;

type C = Complex<f64>;

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

/// One traced round of `reps` applies in a context at TRACE, its record written to
/// `dir/nrhs<nr>.json`.
fn trace(s: &LuSolver<C>, b: &[C], x: &mut [C], nr: usize, reps: usize, dir: &str) {
    let n = b.len() / nr;
    let ctx = rsprobe::Context::builder()
        .threshold(rsprobe::Level::TRACE)
        .build();
    ctx.param("n", n);
    ctx.param("nrhs", nr);
    ctx.param("threads", rayon::current_num_threads());
    {
        let _on = ctx.enter();
        for _ in 0..reps {
            s.apply_block(b, x, nr, n).unwrap();
        }
    }
    let path = std::path::Path::new(dir).join(format!("nrhs{nr}.json"));
    std::fs::write(path, ctx.record().to_string()).expect("writing the record");
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
    let env = |name: &str| std::env::var(name).ok().and_then(|s| s.parse().ok());
    if let Some(n) = env("RSLAB_NEMIN") {
        opts = opts.with_nemin(n);
    }
    if let Some(k) = env("RSLAB_LEAF") {
        opts.solve.leaf_subtrees = k;
    }
    if let Some(k) = env("RSLAB_GEMM_TASK") {
        opts.solve.gemm_task = k;
    }
    let traced = std::env::var("RSLAB_TRACE").ok();
    let t = Instant::now();
    let s = LuSolver::factor(&a, &opts).unwrap();
    let factor_s = t.elapsed().as_secs_f64();
    println!(
        "n={n} fill={} factor {factor_s:.2} s threads={} delayed={delayed} leaf_subtrees={} gemm_task={}",
        s.factor_nnz(),
        rayon::current_num_threads(),
        opts.solve.leaf_subtrees,
        opts.solve.gemm_task
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
        // FNV-1a of the solution's bits, to compare builds and thread counts
        let bits = (x.iter().flat_map(|v| [v.re.to_bits(), v.im.to_bits()]))
            .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                (h ^ b).wrapping_mul(0x100_0000_01b3)
            });
        println!(
            "nrhs={nrhs:2}: {med:7.2} ms median, {best:7.2} ms best, {:6.2} ms per column, {:5.1} GB/s of factor, column 0 against the single solve {dev:.1e}, bits {bits:016x}",
            med / nrhs as f64,
            s.factor_nnz() as f64 * 16.0 / med / 1e6,
        );
        if let Some(dir) = &traced {
            trace(&s, &bb, &mut x, nrhs, reps, dir);
        }
    }
}
