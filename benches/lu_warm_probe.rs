//! Warm LU kernel A/B probe on the rapidmom precond corpus: one matrix, the
//! production config (`exact() + PerturbToEps(1e-12)`, threshold pivot u=0.1),
//! analyze once, warm best-of-3 factor in ONE process - the LU twin of
//! `factor_probe_helmholtz`. Also reports the heuristic `tuned()` pick.
//!
//! Run: `cargo bench --bench lu_warm_probe -- <file-substr> [threads]`
//! Env:  MOM_DIR (default `C:\Repositories\rapidmom\precond_matrices`).

use std::time::Instant;

use num_complex::Complex;
use rslab::{parse_mtx_complex_general, LuSolver, LuSymbolic, SolverSettings, ZeroPivotAction};

type C = Complex<f64>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let filter = args.first().cloned().unwrap_or_else(|| "spiral".into());
    let threads: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let dir = std::env::var("MOM_DIR")
        .unwrap_or_else(|_| r"C:\Repositories\rapidmom\precond_matrices".into());

    let path = std::fs::read_dir(&dir)
        .expect("MOM_DIR unreadable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.to_string_lossy().contains(&filter))
        .expect("no matrix matches filter");
    eprintln!("matrix: {}", path.display());

    let t0 = Instant::now();
    let contents = std::fs::read_to_string(&path).expect("read mtx");
    let name = path.file_stem().unwrap().to_string_lossy().to_string();
    let mtx = parse_mtx_complex_general(&contents, &name).expect("parse mtx");
    drop(contents);
    let a = mtx.to_general_csc().expect("csc");
    eprintln!(
        "n={} nnz={}  (parse {:.1} s)",
        a.n,
        a.nnz(),
        t0.elapsed().as_secs_f64()
    );

    let opts = SolverSettings::exact()
        .with_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 })
        .with_threads(threads);
    let t0 = Instant::now();
    let sym = LuSymbolic::analyze_with(&a, &opts).expect("analyze");
    let ana = t0.elapsed().as_secs_f64();
    let est = sym.estimate_memory::<C>();
    eprintln!(
        "production cfg: ana {ana:.2} s  fill(sym) {}  flops {:.3e}",
        sym.symbolic_factor_nnz(),
        est.factor_flops as f64
    );
    let _ = sym.factor(&a, &opts).expect("warmup");
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let f = sym.factor(&a, &opts).expect("factor");
        best = best.min(t.elapsed().as_secs_f64());
        std::hint::black_box(f.factor_nnz());
    }
    eprintln!(
        "production cfg @{threads}: fac best {:.0} ms   {:.2} geom-Gflop/s",
        best * 1e3,
        est.factor_flops as f64 / best / 1e9
    );

    // Heuristic pick (what LuSolver::tuned / rapidmom-after-upgrade would use,
    // with the production pivot policy applied on top).
    let t0 = Instant::now();
    let (sym_h, s_h) = LuSolver::<C>::tuned(&a).expect("tuned");
    let ana_h = t0.elapsed().as_secs_f64();
    let mut o = s_h.clone();
    o.on_zero_pivot = ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 };
    let est_h = sym_h.estimate_memory::<C>();
    let _ = sym_h.factor(&a, &o).expect("warmup");
    let mut best_h = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let f = sym_h.factor(&a, &o).expect("factor");
        best_h = best_h.min(t.elapsed().as_secs_f64());
        std::hint::black_box(f.factor_nnz());
    }
    eprintln!(
        "heuristic pick {:?} threads {:?}: ana {ana_h:.2} s  fill(sym) {}  fac best {:.0} ms   {:.2} geom-Gflop/s",
        s_h.ordering,
        s_h.threads,
        sym_h.symbolic_factor_nnz(),
        best_h * 1e3,
        est_h.factor_flops as f64 / best_h / 1e9
    );

    // Mixed precision (issue #18): c32 factor + certified IR, production
    // pivot policy. Numeric-factor timing excludes analysis (cast + analyze
    // once, warm best-of-3), same protocol as above.
    let a32 = rslab::GeneralCsc::<num_complex::Complex<f32>> {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a
            .values
            .iter()
            .map(|v| num_complex::Complex::new(v.re as f32, v.im as f32))
            .collect(),
    };
    let opts32 = SolverSettings::exact()
        .with_pivot(ZeroPivotAction::PerturbToEps { abs_floor: 1e-12 })
        .with_threads(threads);
    let sym32 = rslab::LuSymbolic::analyze_with(&a32, &opts32).expect("analyze32");
    let _ = sym32.factor(&a32, &opts32).expect("warmup32");
    let mut best32 = f64::INFINITY;
    for _ in 0..3 {
        let t = Instant::now();
        let f = sym32.factor(&a32, &opts32).expect("factor32");
        best32 = best32.min(t.elapsed().as_secs_f64());
        std::hint::black_box(f.factor_nnz());
    }
    let m = rslab::MixedLuSolver::<C>::factor_with(&a, &opts32).expect("mixed factor");
    let b: Vec<C> = (0..a.n)
        .map(|i| C::new(((i % 13) as f64) - 6.0, ((i % 7) as f64) - 3.0))
        .collect();
    let t = Instant::now();
    let (x, info) = m.solve(&a, &b).expect("mixed solve");
    let slv = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&x);
    eprintln!(
        "mixed c32+IR: fac best {:.0} ms ({:.2}x vs c64)  solve {slv:.1} ms  ir {} gmres {} be {:.1e} certified {}",
        best32 * 1e3,
        best / best32,
        info.ir_iters,
        info.gmres_iters,
        info.backward_error,
        info.certified
    );
}
