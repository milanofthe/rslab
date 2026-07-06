//! Factor-quality probe for the rapidmom MoM near-field precond corpus (#14):
//! factor each matrix under a method x pivot configuration sweep, solve a
//! deterministic pseudo-random b through the factor, and report the TRUE
//! relative residual `||A x - b|| / ||b||` computed straight from the COO
//! entries — LU quality isolated from every GMRES effect. One step of
//! iterative refinement is reported as well: a factor that refines is a
//! usable preconditioner even when its direct residual is poor.
//!
//! Key question this answers: rapidmom ran BOTH `preconditioner(..)` and
//! `exact().with_pivot(PerturbToEps)` — which are the SAME LeftLooking path
//! (exact() defaults to LeftLooking) — so the multifrontal method has never
//! been measured on these matrices at all.
//!
//! Run: `cargo bench --bench mom_factor_probe [-- <substr> ...]`
//! Env:  MOM_DIR (default `C:\Repositories\rapidmom\precond_matrices`),
//!       MOM_THREADS (default 6).

use std::time::Instant;

use num_complex::Complex;
use rslab::{parse_mtx_complex_general, FactorMethod, LuSolver, SolverSettings, ZeroPivotAction};

type C = Complex<f64>;

/// Deterministic LCG in [-1, 1] — reproducible RHS without a rand dependency.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn norm2(v: &[C]) -> f64 {
    v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// r = b - A·x from the raw COO entries (duplicates sum — the file convention).
fn residual(entries: &[(usize, usize, C)], x: &[C], b: &[C]) -> Vec<C> {
    let mut r = b.to_vec();
    for &(i, j, v) in entries {
        r[i] -= v * x[j];
    }
    r
}

fn main() {
    let dir = std::env::var("MOM_DIR")
        .unwrap_or_else(|_| r"C:\Repositories\rapidmom\precond_matrices".into());
    let threads: usize = std::env::var("MOM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let filters: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with('-')).collect();

    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("MOM_DIR unreadable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
        .filter(|p| {
            filters.is_empty()
                || filters.iter().any(|f| p.file_name().unwrap().to_string_lossy().contains(f))
        })
        .collect();
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));

    let floor = |f: f64| ZeroPivotAction::PerturbToEps { abs_floor: f };
    let configs: Vec<(String, SolverSettings)> = vec![
        // What rapidmom runs today (both its historical modes collapse to this):
        ("LL u=0.1 fl=1e-12".into(),
            SolverSettings::exact().with_pivot(floor(1e-12)).with_threads(threads)),
        ("LL u=1.0 fl=1e-12".into(),
            SolverSettings::exact().with_pivot(floor(1e-12)).with_pivot_u(1.0).with_threads(threads)),
        // The never-measured multifrontal path (partial pivoting in dense fronts):
        ("MF fl=1e-12".into(),
            SolverSettings::exact().with_pivot(floor(1e-12))
                .with_method(FactorMethod::Multifrontal).with_threads(threads)),
        // Floor sweep: a LARGER floor bounds ||(LU)^-1|| in the perturbed directions
        // (the direct-solve error pattern above is floor-perturbation shaped).
        ("LL u=0.1 fl=1e-8".into(),
            SolverSettings::exact().with_pivot(floor(1e-8)).with_threads(threads)),
        ("LL u=0.1 fl=1e-6".into(),
            SolverSettings::exact().with_pivot(floor(1e-6)).with_threads(threads)),
        ("LL u=0.1 fl=1e-4".into(),
            SolverSettings::exact().with_pivot(floor(1e-4)).with_threads(threads)),
    ];

    println!(
        "{:<26}{:>9}{:>12} | {:<18}{:>10}{:>12}{:>11}{:>11}",
        "matrix", "n", "nnz", "config", "factor s", "fill nnz", "rel res", "refined"
    );
    println!("{}", "-".repeat(112));

    for path in &files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => { println!("{name}: read failed: {e}"); continue; }
        };
        let mtx = match parse_mtx_complex_general(&contents, &name) {
            Ok(m) => m,
            Err(e) => { println!("{name}: parse failed: {e:?}"); continue; }
        };
        drop(contents);
        let n = mtx.n;
        let a = match mtx.to_general_csc() {
            Ok(a) => a,
            Err(e) => { println!("{name}: csc failed: {e:?}"); continue; }
        };
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let b: Vec<C> = (0..n).map(|_| C::new(rng.next_f64(), rng.next_f64())).collect();
        let nb = norm2(&b);

        for (label, opts) in &configs {
            let t0 = Instant::now();
            let solver = match LuSolver::factor(&a, opts) {
                Ok(s) => s,
                Err(e) => {
                    println!(
                        "{:<26}{:>9}{:>12} | {:<18} FACTOR FAILED: {e:?}",
                        name, n, mtx.entries.len(), label
                    );
                    continue;
                }
            };
            let tf = t0.elapsed().as_secs_f64();
            let x = match solver.solve(&b) {
                Ok(x) => x,
                Err(e) => {
                    println!(
                        "{:<26}{:>9}{:>12} | {:<18} SOLVE FAILED: {e:?}",
                        name, n, mtx.entries.len(), label
                    );
                    continue;
                }
            };
            let r1 = residual(&mtx.entries, &x, &b);
            let res1 = norm2(&r1) / nb;
            // MULTI-STEP refinement trajectory (#14): the per-step contraction rate is
            // the quantity a Krylov method actually lives on — a factor whose one-shot
            // residual is fine but whose trajectory STAGNATES (or bounces) explains a
            // divergent preconditioned iteration that single-step probes miss.
            let mut traj = vec![res1];
            let mut xk = x.clone();
            let mut rk = r1;
            for _ in 0..8 {
                match solver.solve(&rk) {
                    Ok(dx) => {
                        for (a, d) in xk.iter_mut().zip(&dx) {
                            *a += d;
                        }
                        rk = residual(&mtx.entries, &xk, &b);
                        traj.push(norm2(&rk) / nb);
                    }
                    Err(_) => {
                        traj.push(f64::NAN);
                        break;
                    }
                }
            }
            let tstr: Vec<String> = traj.iter().map(|v| format!("{v:.1e}")).collect();
            println!(
                "{:<26}{:>9}{:>12} | {:<18}{:>10.2}{:>12} | traj {}",
                name,
                n,
                mtx.entries.len(),
                label,
                tf,
                solver.factor_nnz(),
                tstr.join(" ")
            );
            // #14 front-growth report (RSLAB_FRONT_STATS=1): the top supernodes by
            // factor magnitude — the growth localization instrument.
            let stats = rslab::take_front_stats();
            if !stats.is_empty() {
                let mut top: Vec<_> = stats.iter().collect();
                top.sort_by(|a, b| b.max_l.max(b.max_u).total_cmp(&a.max_l.max(a.max_u)));
                println!("  top fronts by max|L|/|U| (permuted cols):");
                for f in top.iter().take(10) {
                    println!(
                        "    s={:<6} cols {}..{} ({}x{})  min|piv|={:.1e}  max|L|={:.1e}  max|U|={:.1e}  perturbed={}",
                        f.s, f.first_col, f.first_col + f.ncol, f.ncol, f.nrow,
                        f.min_piv, f.max_l, f.max_u, f.perturbed
                    );
                }
            }
        }
    }
}
