//! Time the LU solves of a complex Matrix Market matrix over right-hand-side counts, and
//! check that a column of a block solve equals the single solve of that column bit for bit.
//! `cargo run --release --example lu_solve_timing -- <matrix.mtx> [reps] [nrhs,..]`.
use num_complex::Complex;
use rslab::{LuSolver, OrderingMethod, SolverSettings, ZeroPivotAction};
use std::time::Instant;

type C = Complex<f64>;

fn main() {
    let path = std::env::args().nth(1).expect("matrix path");
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let rslab::MtxLoaded::General(a) = rslab::read_mtx_any(std::path::Path::new(&path)).unwrap()
    else {
        panic!("a general matrix")
    };
    let n = a.n;
    let opts = SolverSettings::exact()
        .with_zero_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-6 })
        .with_matching(false)
        .with_ordering(OrderingMethod::MetisND);
    let t = Instant::now();
    let s = LuSolver::factor(&a, &opts).unwrap();
    println!(
        "n={n} fill={} factor {:.2} s threads={}",
        s.factor_nnz(),
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );
    let rhs = |k: usize| C::new(((k * 7) % 13) as f64 - 6.0, ((k * 3) % 5) as f64 - 2.0);
    let single: Vec<C> = (0..n).map(rhs).collect();
    let x1 = s.solve(&single).unwrap();
    let counts: Vec<usize> = std::env::args()
        .nth(3)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16]);
    for nrhs in counts {
        let bb: Vec<C> = (0..n * nrhs).map(rhs).collect();
        let mut best = f64::INFINITY;
        let mut x = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            x = s.solve_many(&bb, nrhs).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let same = x[..n] == x1[..];
        println!(
            "nrhs={nrhs:2}: {:7.2} ms, {:6.2} ms per column, {:5.1} GB/s of factor, column 0 {}",
            best * 1e3,
            best * 1e3 / nrhs as f64,
            s.factor_nnz() as f64 * 16.0 / best / 1e9,
            if same { "bit-identical" } else { "DIFFERS" }
        );
    }
}
