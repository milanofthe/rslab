//! Full MoM solver-in-the-loop validation on the real near-field matrices.
//!
//! For each `*.mtx` in `rapidmom/precond_matrices` (complex general): build the
//! sparse near-field as a [`GeneralCsc`] `LinearOperator`, factor an **ILU**
//! preconditioner (static-pivoted + threshold-dropped, optionally `f32`), and
//! solve `A x = b` with **GMRES** using that preconditioner. Reports the
//! preconditioner fill/memory and the GMRES iteration count + residual — the
//! memory ↔ iterations tradeoff on genuine MoM data.
//!
//! Run: `cargo bench --bench mom_loop`.

use std::time::Instant;

use num_complex::Complex;
use rla::prelude::*;
use rla::LowPrecisionLu;

type C = Complex<f64>;

const DIR: &str = r"C:\Repositories\rapidmom\precond_matrices";
const GMRES_TOL: f64 = 1e-3; // a realistic MoM Krylov tolerance
const GMRES_RESTART: usize = 60;
const GMRES_MAXIT: usize = 300;

fn run_precond<M: Preconditioner<C>>(
    label: &str,
    a: &GeneralCsc<C>,
    b: &[C],
    fill: usize,
    entry_bytes: usize,
    m: &M,
) {
    let t = Instant::now();
    let res = gmres(a, b, m, GMRES_TOL, GMRES_MAXIT, GMRES_RESTART).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let mem = fill as f64 * (entry_bytes + 8) as f64 / 1e6;
    println!(
        "  {label:24} fill={fill:9}  mem={mem:7.1} MB  gmres: {:4} iters  {ms:8.1} ms  res={:.1e}{}",
        res.iters,
        res.final_res,
        if res.converged { "" } else { "  (NO CONV)" },
    );
}

fn bench_file(path: &std::path::Path) {
    let name = path.file_name().unwrap().to_string_lossy().to_string();
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            println!("{name}: read error {e}");
            return;
        }
    };
    let mtx = match parse_mtx_complex_general(&contents, &name) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}: parse error {e}");
            return;
        }
    };
    drop(contents);
    let a = match mtx.to_general_csc() {
        Ok(a) => a,
        Err(e) => {
            println!("{name}: build error {e}");
            return;
        }
    };
    let n = a.n;
    let b: Vec<C> = vec![C::new(1.0, 0.0); n];
    println!("=== {name}  n={n}  nnz={} ===", a.nnz());

    // Unpreconditioned GMRES baseline (capped).
    let t = Instant::now();
    let r0 = gmres(&a, &b, &NoPreconditioner, GMRES_TOL, 400, GMRES_RESTART).unwrap();
    println!(
        "  {:24} {:24} gmres: {:4} iters  {:8.1} ms  res={:.1e}{}",
        "unpreconditioned",
        "",
        r0.iters,
        t.elapsed().as_secs_f64() * 1e3,
        r0.final_res,
        if r0.converged { "" } else { "  (NO CONV)" },
    );

    // Exact LU near-field preconditioner (static-pivoted, equilibrated).
    let exact = LuSolver::factor(&a, &FactorOptions::preconditioner(1e-10)).unwrap();
    run_precond("f64 LU (no drop)", &a, &b, exact.factor_nnz(), 16, &exact);

    // f32 LU — half the factor memory.
    let f32lu = LowPrecisionLu::factor(&a, &FactorOptions::preconditioner(1e-10)).unwrap();
    run_precond("f32 LU (no drop)", &a, &b, f32lu.factor_nnz(), 8, &f32lu);

    // Light incomplete LU (memory ↓). Aggressive dropping is unstable on these
    // indefinite saddle matrices (post-hoc truncation, not propagated ILU).
    let ilu = LuSolver::factor(
        &a,
        &FactorOptions::preconditioner(1e-10).with_drop_tol(1e-2),
    )
    .unwrap();
    run_precond("f64 ILU τ=1e-2", &a, &b, ilu.factor_nnz(), 16, &ilu);
    println!();
}

fn main() {
    let mut files: Vec<_> = match std::fs::read_dir(DIR) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mtx"))
            .collect(),
        Err(e) => {
            println!("cannot read {DIR}: {e}");
            return;
        }
    };
    files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0));
    println!("MoM solver-in-the-loop: GeneralCsc operator + ILU preconditioner + GMRES\n");
    for f in &files {
        bench_file(f);
    }
}
