//! Time the port matrix `Z = B^T A^-1 B` of sparse ports from forward sweeps
//! alone (`LdltSolver::port_matrix`) against `B^T X` from one dense block
//! solve of all ports (`solve_many`), and the sparse solve
//! (`LdltSolver::solve_sparse`) against the same dense solve.
//!
//! `cargo run --release --features matgen --example port_matrix_timing -- [case] [size] [ports] [threads] [reps] [scattered|clustered] [z-dump-file]`
//!
//! - `helmholtz`: 2D 5-point `Lap - k^2 I` on a `size x size` grid,
//!   `k^2 = 0.3 (1 + 0.02 i)` per cell (complex symmetric, lossy, indefinite
//!   off resonance), each port six neighbouring grid nodes;
//! - `curl-curl`: the shipped `matgen::fem::curl_curl` operator on a
//!   `size^3` grid (`omega = 1`, `sigma = 0.1`), each port six consecutive
//!   unknowns.
//!
//! Ports start at pseudo-random places; times are the median, minimum and
//! maximum over `reps` runs after one warm-up.
use num_complex::Complex;
use rslab::matgen::fem::curl_curl;
use rslab::{CscMatrix, LdltSolver, SolverSettings, SparseRhs, Threads};
use std::time::Instant;

type C = Complex<f64>;

fn helmholtz(m: usize, k2: C) -> CscMatrix<C> {
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(C::new(4.0, 0.0) - k2);
        if j % m + 1 < m {
            r.push(j + 1);
            c.push(j);
            v.push(C::new(-1.0, 0.0));
        }
        if j + m < n {
            r.push(j + m);
            c.push(j);
            v.push(C::new(-1.0, 0.0));
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

/// `p` ports of six unknowns each, weight 1/6, starting at pseudo-random
/// unknowns in `first..first + span`; `next(i, k)` is the `k`-th unknown of
/// the port at `i`.
fn ports(
    n: usize,
    p: usize,
    (first, span): (usize, usize),
    next: impl Fn(usize, usize) -> usize,
) -> SparseRhs<C> {
    let mut state = 0x1234_5678_9abc_def0u64;
    let cols: Vec<Vec<(usize, C)>> = (0..p)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let i = first + (state >> 33) as usize % span.max(1);
            (0..6)
                .map(|k| (next(i, k) % n, C::new(1.0 / 6.0, 0.0)))
                .collect()
        })
        .collect();
    SparseRhs::from_columns(n, &cols).unwrap()
}

/// Median, minimum and maximum seconds over `reps` runs after a warm-up.
fn time<R>(reps: usize, mut f: impl FnMut() -> R) -> ([f64; 3], R) {
    let mut out = f();
    let mut runs = Vec::with_capacity(reps);
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        out = f();
        runs.push(t.elapsed().as_secs_f64());
    }
    runs.sort_by(f64::total_cmp);
    let mid = runs.len() / 2;
    let median = if runs.len() % 2 == 1 {
        runs[mid]
    } else {
        (runs[mid - 1] + runs[mid]) / 2.0
    };
    ([median, runs[0], runs[runs.len() - 1]], out)
}

fn show([m, lo, hi]: [f64; 3]) -> String {
    format!("{m:.4} ({lo:.4}-{hi:.4}) s")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map_or("helmholtz", String::as_str);
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let p = arg(3, 200);
    let threads = arg(4, 1);
    let reps = arg(5, 5);
    // `scattered` (default): ports anywhere; `clustered`: all ports start in
    // a window of n/100 unknowns in the middle of the numbering.
    let clustered = args.get(6).is_some_and(|s| s == "clustered");
    let window = |n: usize| if clustered { (n / 2, n / 100) } else { (0, n) };
    let (a, b) = match case {
        "curl-curl" => {
            let m = arg(2, 20);
            let a = curl_curl(&[m, m, m], 1.0, 0.1);
            let n = a.n;
            (a, ports(n, p, window(n), |i, k| i + k))
        }
        _ => {
            let m = arg(2, 400);
            let a = helmholtz(m, C::new(0.3, 0.006));
            let n = a.n;
            (
                a,
                ports(n, p, window(n), move |i, k| i + (k % 3) + (k / 3) * m),
            )
        }
    };
    let n = a.n;
    let s = SolverSettings::default().with_threads(Threads::Fixed(threads));
    let t = Instant::now();
    let f = LdltSolver::factor(&a, &s).unwrap();
    println!(
        "{case}: n {n}, nnz(L) {}, {p} ports, {threads} threads, factor {:.3} s",
        f.factor_nnz(),
        t.elapsed().as_secs_f64()
    );
    let (tz, z) = time(reps, || f.port_matrix(&b).unwrap());
    // `PORT_ONLY=1`: the port matrix alone (for the process's peak memory).
    if std::env::var_os("PORT_ONLY").is_some() {
        println!("port_matrix {}", show(tz));
        return;
    }
    let dense = b.to_dense();
    let (td, zd) = time(reps, || {
        let x = f.solve_many(&dense, p).unwrap();
        b.transpose_times(&x, p).unwrap()
    });
    let (ts, _) = time(reps, || f.solve_sparse(&b).unwrap());
    // Optional: the bits of `Z` to a file, to compare builds.
    if let Some(path) = args.get(7) {
        let bytes: Vec<u8> = z
            .iter()
            .flat_map(|v| [v.re.to_le_bytes(), v.im.to_le_bytes()])
            .flatten()
            .collect();
        std::fs::write(path, bytes).unwrap();
    }
    let scale = zd.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let diff = (z.iter().zip(&zd))
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max);
    let reach: Vec<_> = (0..p)
        .map(|c| f.sparse_reach(b.column(c).0).unwrap())
        .collect();
    let mean = reach.iter().map(|r| r.panel_entries as f64).sum::<f64>() / p.max(1) as f64;
    println!(
        "port_matrix {} | {p} dense solves + B^T X {} ({:.1}x by medians) | \
         solve_sparse {} | max|dZ|/max|Z| {:.1e} | mean reach {:.1}% of the factor",
        show(tz),
        show(td),
        td[0] / tz[0],
        show(ts),
        diff / scale,
        100.0 * mean / reach.first().map_or(1, |r| r.total_panel_entries) as f64
    );
}
