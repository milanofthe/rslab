//! One standard case through every phase of its solver: analysis,
//! factorization, refactorization, a solve of one and of eight right-hand
//! sides and a refined solve. Per phase the median wall time over `reps`
//! runs after one warm-up, on one JSON line with the hash of the solution's
//! bits. The timed runs record into an rsprobe context, written to the path in
//! `RSPROBE_RECORD` (parameters: case, n, nnz, threads) for
//! `python -m rsprobe report`; with `--features rsprobe/on` only. The
//! context's threshold is INFO, or `debug` or `trace` from `RSPROBE_LEVEL`.
//!
//! `cargo run --release --features matgen --example phases -- <case> [size] [threads] [reps]`
//!
//! Cases: `laplace3d` (real SPD, LDL^T), `helmholtz3d` (complex symmetric,
//! LDL^T), `convdiff3d` (real unsymmetric, LU), `mom` (complex unsymmetric
//! BEM near field, LU), `convdiff2d-klu` (real unsymmetric, KLU). `size` is
//! the grid edge, or the dimension for `mom`.
use num_complex::Complex;
use rslab::matgen::{bem, fem, stencil};
use rslab::{
    GeneralCsc, KluSettings, KluSymbolic, LdltSymbolic, LuSymbolic, RefinePolicy, Scalar,
    SolverSettings, Threads,
};
use std::time::Instant;

type C = Complex<f64>;

const REFINE_STEPS: usize = 2;
const BLOCK_RHS: usize = 8;
const NEAR_ENTRIES_PER_COL: f64 = 60.0;
const NEAR_MAX_CUTOFF: f64 = 1.2;

#[derive(Default)]
struct Times(Vec<(&'static str, Vec<f64>)>);

impl Times {
    fn time<R>(&mut self, keep: bool, name: &'static str, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let r = f();
        if keep {
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match self.0.iter_mut().find(|p| p.0 == name) {
                Some(p) => p.1.push(ms),
                None => self.0.push((name, vec![ms])),
            }
        }
        r
    }

    fn medians(&self) -> String {
        let med = |v: &[f64]| {
            let mut v = v.to_vec();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let parts: Vec<String> = (self.0.iter())
            .map(|(n, v)| format!("\"{n}_ms\": {:.4}", med(v)))
            .collect();
        parts.join(", ")
    }
}

fn fnv1a_bits<T: Scalar>(x: &[T]) -> u64 {
    let words = x
        .iter()
        .flat_map(|v| [v.real().to_bits(), v.imag().to_bits()]);
    words.fold(0xcbf2_9ce4_8422_2325_u64, |h, w| {
        (h ^ w).wrapping_mul(0x100_0000_01b3)
    })
}

fn rhs<T: Scalar>(n: usize, nrhs: usize) -> Vec<T> {
    (0..n * nrhs)
        .map(|k| T::from_real(((k * 7) % 13) as f64 - 6.0))
        .collect()
}

macro_rules! supernodal {
    ($sym:ty, $a:expr, $opts:expr, $reps:expr, $ctx:expr) => {{
        let (a, opts) = (&$a, &$opts);
        let (mut times, mut hash) = (Times::default(), 0);
        for run in 0..=$reps {
            let keep = run > 0;
            let _on = keep.then(|| $ctx.enter());
            let sym = times.time(keep, "analyze", || <$sym>::analyze(a, opts).unwrap());
            let mut f = times.time(keep, "factor", || sym.factor(a, opts).unwrap());
            times.time(keep, "refactor", || sym.refactor(a, opts, &mut f).unwrap());
            let b = rhs(a.n, BLOCK_RHS);
            let x = times.time(keep, "solve", || f.solve(&b[..a.n]).unwrap());
            times.time(keep, "solve8", || f.solve_many(&b, BLOCK_RHS).unwrap());
            let policy = RefinePolicy::steps(REFINE_STEPS);
            times.time(keep, "refine", || {
                f.solve_refined(a, &b[..a.n], &policy).unwrap()
            });
            hash = fnv1a_bits(&x);
        }
        (times, hash)
    }};
}

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let case = arg(1).expect("case");
    let size: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let threads: usize = arg(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let reps: usize = arg(4).and_then(|s| s.parse().ok()).unwrap_or(5);
    let opts = SolverSettings::default().with_threads(Threads::Fixed(threads));
    let level = match std::env::var("RSPROBE_LEVEL").as_deref() {
        Ok("trace") => rsprobe::Level::TRACE,
        Ok("debug") => rsprobe::Level::DEBUG,
        _ => rsprobe::Level::INFO,
    };
    let ctx = rsprobe::Context::builder().threshold(level).build();
    let grid = [size; 3];
    let ((times, hash), n, nnz) = match case.as_str() {
        "laplace3d" => {
            let a = stencil::laplacian::<f64>(&grid, &Default::default());
            let (n, nnz) = (a.n, a.row_idx.len());
            (supernodal!(LdltSymbolic, a, opts, reps, ctx), n, nnz)
        }
        "helmholtz3d" => {
            let k2 = C::new(0.05, 0.02);
            let a = stencil::helmholtz(&grid, k2, &Default::default());
            let (n, nnz) = (a.n, a.row_idx.len());
            (supernodal!(LdltSymbolic, a, opts, reps, ctx), n, nnz)
        }
        "convdiff3d" => {
            let a = fem::convection_diffusion::<f64>(&grid, 0.01, fem::Flow::Rotating, false);
            let (n, nnz) = (a.n, a.row_idx.len());
            (supernodal!(LuSymbolic, a, opts, reps, ctx), n, nnz)
        }
        "mom" => {
            let cutoff = (2.0 * (NEAR_ENTRIES_PER_COL / size as f64).sqrt()).min(NEAR_MAX_CUTOFF);
            let bem = bem::BemOpts {
                cutoff,
                ..Default::default()
            };
            let a: GeneralCsc<C> = bem::kernel(size, &bem);
            let (n, nnz) = (a.n, a.row_idx.len());
            (supernodal!(LuSymbolic, a, opts, reps, ctx), n, nnz)
        }
        "convdiff2d-klu" => {
            let a =
                fem::convection_diffusion::<f64>(&[size, size], 0.01, fem::Flow::Rotating, true);
            let (n, nnz) = (a.n, a.row_idx.len());
            let s = KluSettings::default().with_threads(threads);
            let (mut times, mut hash) = (Times::default(), 0);
            for run in 0..=reps {
                let keep = run > 0;
                let _on = keep.then(|| ctx.enter());
                let sym = times.time(keep, "analyze", || KluSymbolic::analyze(&a, &s).unwrap());
                let mut k = times.time(keep, "factor", || sym.factor(&a, &s).unwrap());
                times.time(keep, "refactor", || k.refactor(&a).unwrap());
                let b = rhs(n, BLOCK_RHS);
                let x = times.time(keep, "solve", || k.solve(&b[..n]).unwrap());
                times.time(keep, "solve8", || k.solve_many(&b, BLOCK_RHS).unwrap());
                let policy = RefinePolicy::steps(REFINE_STEPS);
                times.time(keep, "refine", || {
                    k.solve_refined(&a, &b[..n], &policy).unwrap()
                });
                hash = fnv1a_bits(&x);
            }
            ((times, hash), n, nnz)
        }
        other => panic!("unknown case {other}"),
    };
    println!(
        "{{\"case\": \"{case}\", \"n\": {n}, \"nnz\": {nnz}, \"threads\": {threads}, \"reps\": {reps}, {}, \"bits\": \"{hash:016x}\"}}",
        times.medians()
    );
    ctx.param("case", case.as_str());
    ctx.param("n", n);
    ctx.param("nnz", nnz);
    ctx.param("threads", threads);
    if let Some(path) = std::env::var_os("RSPROBE_RECORD") {
        std::fs::write(path, ctx.record().to_string()).expect("writing the record");
    }
}
