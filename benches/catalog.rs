//! Sweep the generated test-matrix [`catalog`] - factor + solve each tagged
//! matrix and report dimension, fill, factor/solve time and the true residual.
//! The default consumer of the `matgen` generators: a structured benchmark across
//! size / structure / symmetry / conditioning / density without depending on any
//! external matrix files.
//!
//! Run: `cargo bench --bench catalog --features matgen`
//!   * `RLA_CAT_FILTER=helmholtz`  - only matrices whose name contains the string
//!   * `RLA_CAT_MAXN=40000`        - skip matrices larger than this (default 60k)

use std::time::Instant;

use num_complex::Complex;
use rslab::matgen::{catalog, Generated};
use rslab::{LdltSymbolic, LuSymbolic, SolverSettings};

type C = Complex<f64>;

fn rhs(n: usize) -> Vec<C> {
    (0..n)
        .map(|i| Complex::new((i % 5) as f64 - 2.0, (i % 3) as f64 - 1.0))
        .collect()
}

fn main() {
    let filter = std::env::var("RLA_CAT_FILTER").unwrap_or_default();
    let max_n: usize = std::env::var("RLA_CAT_MAXN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000);
    let opts = SolverSettings::default();

    println!(
        "matgen catalog sweep  (max_n={max_n})\n{:<22} {:>8} {:>10} {:>7} {:>8} {:>8} {:>8}",
        "name", "n", "nnz", "ana", "fac", "slv", "res"
    );
    for spec in catalog() {
        if !filter.is_empty() && !spec.name.contains(&filter) {
            continue;
        }
        if spec.size > max_n {
            println!(
                "{:<22} {:>8}  (skipped: size > max_n)",
                spec.name, spec.size
            );
            continue;
        }
        let m = spec.build();
        let (n, nnz) = (m.n(), m.nnz());
        let b = rhs(n);

        let t = Instant::now();
        let (ana, fac, slv, res) = match &m {
            Generated::Symmetric(a) => {
                let sym = LdltSymbolic::analyze(a).unwrap();
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let f = a_factor_sym(&sym, a, &opts);
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b).unwrap();
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.symv(&x, &mut ax);
                (ana, fac, slv, rel_resid(&ax, &b))
            }
            Generated::Unsymmetric(a) => {
                let sym = LuSymbolic::analyze(a).unwrap();
                let ana = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let f = sym.factor(a, &opts).unwrap();
                let fac = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let x = f.solve(&b).unwrap();
                let slv = t.elapsed().as_secs_f64() * 1e3;
                let mut ax = vec![Complex::new(0.0, 0.0); n];
                a.matvec(&x, &mut ax);
                (ana, fac, slv, rel_resid(&ax, &b))
            }
        };
        println!(
            "{:<22} {:>8} {:>10} {ana:>7.0} {fac:>8.1} {slv:>8.2} {res:>8.0e}  [{:?}/{:?}/{:?}/{:?}]",
            spec.name, n, nnz, spec.structure, spec.symmetry, spec.cond, spec.density
        );
    }
}

fn a_factor_sym(
    sym: &LdltSymbolic,
    a: &rslab::CscMatrix<C>,
    opts: &SolverSettings,
) -> rslab::LdltSolver<C> {
    sym.factor(a, opts).unwrap()
}

fn rel_resid(ax: &[C], b: &[C]) -> f64 {
    let num: f64 = (0..b.len())
        .map(|i| (ax[i] - b[i]).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    num / den.max(1e-300)
}
