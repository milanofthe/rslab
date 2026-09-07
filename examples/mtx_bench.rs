#![allow(clippy::needless_range_loop)]
//! Throughput of the direct paths on a Matrix Market file: per ordering the
//! analysis, factorization and solve rates, and for general (circuit)
//! matrices the KLU factor / refactor / solve. Bit-for-bit identical
//! numerics aside, this is the loop-closing view on real matrices.
//!
//! `cargo run --release --example mtx_bench -- <file.mtx> [threads]`
use num_complex::Complex;
use rslab::{KluSettings, LdltSolver, LuSolver, MtxLoaded, OrderingMethod, SolverSettings};
use std::time::Instant;

type C = Complex<f64>;

fn best<R>(reps: usize, mut f: impl FnMut() -> R) -> (f64, R) {
    let mut t_best = f64::INFINITY;
    let mut out = None;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        t_best = t_best.min(t.elapsed().as_secs_f64());
        out = Some(r);
    }
    (t_best, out.unwrap())
}

fn main() {
    let path = std::env::args().nth(1).expect("matrix path");
    let reps: usize = std::env::var("MTX_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }
    let loaded = rslab::read_mtx_any(std::path::Path::new(&path)).expect("read mtx");
    let orderings = [
        ("auto", None),
        ("amd", Some(OrderingMethod::Amd)),
        ("amf", Some(OrderingMethod::Amf)),
        ("metis", Some(OrderingMethod::MetisND)),
    ];
    // `MTX_ORDERINGS=metis,auto` restricts the orderings (large matrices).
    let only = std::env::var("MTX_ORDERINGS").ok();
    let orderings: Vec<_> = orderings
        .into_iter()
        .filter(|(name, _)| {
            only.as_ref()
                .is_none_or(|f| f.split(',').any(|x| x == *name))
        })
        .collect();
    match loaded {
        MtxLoaded::Symmetric(a) => {
            let n = a.n;
            let real = a.values.iter().all(|v| v.im == 0.0);
            if real {
                // Real symmetric: the f64 path (half the arithmetic of the complex one).
                let ar = rslab::CscMatrix {
                    n,
                    col_ptr: a.col_ptr.clone(),
                    row_idx: a.row_idx.clone(),
                    values: a.values.iter().map(|v| v.re).collect::<Vec<f64>>(),
                };
                let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
                println!(
                    "symmetric n={n} nnz(lower)={} real threads={}",
                    ar.values.len(),
                    rayon::current_num_threads()
                );
                println!(
                    "{:>7} {:>10} {:>5} | {:>8} {:>8} {:>7} {:>6} | {:>8} {:>7} {:>9} {:>7}",
                    "order",
                    "nnzL",
                    "fill",
                    "ana ms",
                    "fac ms",
                    "MDOF/s",
                    "GF/s",
                    "solve ms",
                    "MDOF/s",
                    "solve8 ms",
                    "MDOF/s"
                );
                for &(name, om) in &orderings {
                    let mut opts = SolverSettings::default();
                    if let Some(o) = om {
                        opts = opts.with_ordering(o);
                    }
                    if threads > 0 {
                        opts = opts.with_threads(threads);
                    }
                    if let Some(n) = std::env::var("MTX_NEMIN").ok().and_then(|v| v.parse().ok()) {
                        opts = opts.with_nemin(n);
                    }
                    if let Ok(m) = std::env::var("MTX_METHOD") {
                        opts = opts.with_method(if m == "multifrontal" {
                            rslab::FactorMethod::Multifrontal
                        } else {
                            rslab::FactorMethod::LeftLooking
                        });
                    }
                    let (t_all, s) = best(reps, || {
                        if om.is_none() {
                            let (sym, mut pick) = LdltSolver::<f64>::tuned(&ar).unwrap();
                            if threads > 0 {
                                pick = pick.with_threads(threads);
                            }
                            sym.factor(&ar, &pick).unwrap()
                        } else {
                            LdltSolver::factor_with(&ar, &opts).unwrap()
                        }
                    });
                    let d = s.diagnostics();
                    let r = d.rates();
                    let ana = d.stage_ms("analyze").unwrap_or(0.0);
                    let (ts, x) = best(5, || s.solve(&b).unwrap());
                    let bb: Vec<f64> = (0..n * 8).map(|k| ((k * 11) % 17) as f64 - 8.0).collect();
                    let (t8, _) = best(3, || s.solve_many(&bb, 8).unwrap());
                    let mut res = b.clone();
                    for j in 0..n {
                        for k in ar.col_ptr[j]..ar.col_ptr[j + 1] {
                            let i = ar.row_idx[k];
                            res[i] -= ar.values[k] * x[j];
                            if i != j {
                                res[j] -= ar.values[k] * x[i];
                            }
                        }
                    }
                    let rn = res.iter().map(|v| v * v).sum::<f64>().sqrt()
                        / b.iter().map(|v| v * v).sum::<f64>().sqrt();
                    println!("{name:>7} {:>10} {:>5.1} | {ana:>8.0} {:>8.1} {:>7.2} {:>6.1} | {:>8.2} {:>7.1} {:>9.2} {:>7.1}   total={t_all:.2}s res={rn:.1e}",
                        s.factor_nnz(), s.factor_nnz() as f64 / ar.values.len() as f64,
                        d.stage_ms("factor").unwrap_or(0.0), r.factor_mdof_s, r.factor_gflops,
                        ts * 1e3, n as f64 / ts / 1e6, t8 * 1e3, 8.0 * n as f64 / t8 / 1e6);
                }
                return;
            }
            println!(
                "symmetric n={n} nnz(lower)={} {} threads={}",
                a.values.len(),
                if real { "real" } else { "complex" },
                rayon::current_num_threads()
            );
            let b: Vec<C> = (0..n)
                .map(|i| C::new(((i * 7) % 13) as f64 - 6.0, 0.0))
                .collect();
            println!(
                "{:>7} {:>10} {:>5} | {:>8} {:>8} {:>7} {:>6} | {:>8} {:>7} {:>9} {:>7}",
                "order",
                "nnzL",
                "fill",
                "ana ms",
                "fac ms",
                "MDOF/s",
                "GF/s",
                "solve ms",
                "MDOF/s",
                "solve8 ms",
                "MDOF/s"
            );
            for &(name, om) in &orderings {
                let mut opts = SolverSettings::default();
                if let Some(o) = om {
                    opts = opts.with_ordering(o);
                }
                if threads > 0 {
                    opts = opts.with_threads(threads);
                }
                let (t_all, s) = best(reps, || {
                    if om.is_none() {
                        let (sym, mut pick) = LdltSolver::<C>::tuned(&a).unwrap();
                        if threads > 0 {
                            pick = pick.with_threads(threads);
                        }
                        sym.factor(&a, &pick).unwrap()
                    } else {
                        LdltSolver::factor_with(&a, &opts).unwrap()
                    }
                });
                let d = s.diagnostics();
                let r = d.rates();
                let ana = d.stage_ms("analyze").unwrap_or(0.0);
                let (ts, _) = best(5, || s.solve(&b).unwrap());
                let bb: Vec<C> = (0..n * 8)
                    .map(|k| C::new(((k * 11) % 17) as f64 - 8.0, 0.0))
                    .collect();
                let (t8, _) = best(3, || s.solve_many(&bb, 8).unwrap());
                println!("{name:>7} {:>10} {:>5.1} | {ana:>8.0} {:>8.1} {:>7.2} {:>6.1} | {:>8.2} {:>7.1} {:>9.2} {:>7.1}   total={t_all:.2}s",
                    s.factor_nnz(), s.factor_nnz() as f64 / a.values.len() as f64,
                    d.stage_ms("factor").unwrap_or(0.0), r.factor_mdof_s, r.factor_gflops,
                    ts * 1e3, n as f64 / ts / 1e6, t8 * 1e3, 8.0 * n as f64 / t8 / 1e6);
            }
        }
        MtxLoaded::General(a) => {
            let n = a.n;
            println!(
                "general n={n} nnz={} threads={}",
                a.values.len(),
                rayon::current_num_threads()
            );
            let b: Vec<C> = (0..n)
                .map(|i| C::new(((i * 7) % 13) as f64 - 6.0, 0.0))
                .collect();
            // KLU variants: pivot tolerance and BTF.
            for (label, settings) in [
                ("klu", KluSettings::default()),
                ("klu-p1", KluSettings::default().with_pivot_tol(1.0)),
                ("klu-p.1", KluSettings::default().with_pivot_tol(0.1)),
                ("klu-nobtf", KluSettings::default().with_btf(false)),
                ("klu-noscl", KluSettings::default().with_row_scaling(false)),
            ] {
                let Ok((ta, sym)) = std::panic::catch_unwind(|| {
                    best(2, || {
                        rslab::KluSymbolic::analyze_with(&a, &settings).unwrap()
                    })
                }) else {
                    println!("{label:>9} analysis failed");
                    continue;
                };
                let Ok((tf, mut k)) =
                    std::panic::catch_unwind(|| best(2, || sym.factor(&a, &settings).unwrap()))
                else {
                    println!("{label:>9} failed");
                    continue;
                };
                let (tr, _) = best(2, || k.refactor(&a).unwrap());
                let (ts, x) = best(5, || k.solve(&b).unwrap());
                let res = resid(&a, &x, &b);
                println!("{label:>9} nnzLU={:>9} blocks={:>6} | analyze {:>7.2} ms factor {:>8.2} ms ({:.2} MDOF/s) refactor {:>8.2} ms ({:.2} MDOF/s) solve {:>7.3} ms ({:.1} MDOF/s) res={res:.1e}",
                    k.factor_nnz(), k.n_blocks(), ta * 1e3, tf * 1e3, n as f64 / tf / 1e6, tr * 1e3, n as f64 / tr / 1e6, ts * 1e3, n as f64 / ts / 1e6);
            }
            // Real-valued runs when the file is real: the general Matrix
            // Market loader always yields complex values, but a circuit's
            // matrix is real and that is what the KLU comparisons measure.
            if a.values.iter().all(|v| v.im == 0.0) {
                let ar = rslab::GeneralCsc {
                    n: a.n,
                    col_ptr: a.col_ptr.clone(),
                    row_idx: a.row_idx.clone(),
                    values: a.values.iter().map(|v| v.re).collect::<Vec<f64>>(),
                };
                let br: Vec<f64> = b.iter().map(|v| v.re).collect();
                let settings = KluSettings::default();
                let (ta, sym) = best(2, || {
                    rslab::KluSymbolic::analyze_with(&ar, &settings).unwrap()
                });
                let (tf, mut k) = best(2, || sym.factor(&ar, &settings).unwrap());
                let (tr, _) = best(2, || k.refactor(&ar).unwrap());
                let (ts, x) = best(5, || k.solve(&br).unwrap());
                let mut ax = vec![0.0f64; n];
                for j in 0..n {
                    for e in ar.col_ptr[j]..ar.col_ptr[j + 1] {
                        ax[ar.row_idx[e]] += ar.values[e] * x[j];
                    }
                }
                let res = ax
                    .iter()
                    .zip(&br)
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt()
                    / br.iter().map(|v| v * v).sum::<f64>().sqrt();
                println!("  klu-f64 nnzLU={:>9} blocks={:>6} | analyze {:>7.2} ms factor {:>8.2} ms ({:.2} MDOF/s) refactor {:>8.2} ms ({:.2} MDOF/s) solve {:>7.3} ms ({:.1} MDOF/s) res={res:.1e}",
                    k.factor_nnz(), k.n_blocks(), ta * 1e3, tf * 1e3, n as f64 / tf / 1e6, tr * 1e3, n as f64 / tr / 1e6, ts * 1e3, n as f64 / ts / 1e6);
            }
            // Symbolic view of the KLU analysis: predicted vs actual fill.
            {
                let sym = rslab::KluSymbolic::analyze(&a).unwrap();
                let k = sym.factor(&a, &KluSettings::default()).unwrap();
                let d = k.diagnostics();
                println!(
                    "{:>9} symbolic nnzLU={} actual={} max_block={} | {}",
                    "klu-sym",
                    sym.symbolic_factor_nnz(),
                    k.factor_nnz(),
                    sym.max_block_size(),
                    d.summary()
                );
                for w in &d.warnings {
                    println!("          warning: {w}");
                }
            }
            // LU with full partial pivoting and refinement, for the accuracy picture.
            {
                let s = LuSolver::factor(&a, &SolverSettings::default().with_pivot_u(1.0)).unwrap();
                let x = s.solve(&b).unwrap();
                let (xr, out) = s
                    .solve_refined_with(&a, &b, &rslab::RefinePolicy::steps(3))
                    .unwrap();
                println!(
                    "{:>9} nnzLU={} perturbed={} res={:.1e} refined({} steps) res={:.1e}",
                    "lu-u1",
                    s.factor_nnz(),
                    s.n_perturbed(),
                    resid(&a, &x, &b),
                    out.steps,
                    resid(&a, &xr, &b)
                );
                let s = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
                let (xr, out) = s
                    .solve_refined_with(&a, &b, &rslab::RefinePolicy::steps(3))
                    .unwrap();
                println!(
                    "{:>9} perturbed={} refined({} steps) res={:.1e}",
                    "lu-u.1",
                    s.n_perturbed(),
                    out.steps,
                    resid(&a, &xr, &b)
                );
                let s = LuSolver::factor(
                    &a,
                    &SolverSettings::default().with_method(rslab::FactorMethod::Multifrontal),
                )
                .unwrap();
                let x = s.solve(&b).unwrap();
                println!(
                    "{:>9} nnzLU={} perturbed={} res={:.1e}",
                    "lu-mf",
                    s.factor_nnz(),
                    s.n_perturbed(),
                    resid(&a, &x, &b)
                );
            }
            // LU accuracy variants.
            for (label, opts) in [
                (
                    "lu-nogemm",
                    SolverSettings::default().with_use_gemm_schur(false),
                ),
                ("lu-norelax", SolverSettings::default().with_relax(None)),
                ("lu-nemin1", SolverSettings::default().with_nemin(1)),
                ("lu-nb8", SolverSettings::default().with_panel_nb(8)),
                ("lu-t1", SolverSettings::default().with_threads(1)),
                (
                    "lu-nb8-norelax-t1",
                    SolverSettings::default()
                        .with_panel_nb(8)
                        .with_relax(None)
                        .with_threads(1)
                        .with_nemin(1),
                ),
            ] {
                let s = LuSolver::factor(&a, &opts).unwrap();
                let x = s.solve(&b).unwrap();
                println!(
                    "{label:>18} nnzLU={} res={:.1e}",
                    s.factor_nnz(),
                    resid(&a, &x, &b)
                );
            }
            // Matching diagnostics: growth, residual and permutation sanity per variant.
            for (label, opts) in [
                (
                    "dbg-amd-u.1",
                    SolverSettings::default().with_ordering(OrderingMethod::Amd),
                ),
                (
                    "dbg-amd-u.1-nomatch",
                    SolverSettings::default()
                        .with_ordering(OrderingMethod::Amd)
                        .with_lu_matching(false),
                ),
                (
                    "dbg-amd-u0",
                    SolverSettings::default()
                        .with_ordering(OrderingMethod::Amd)
                        .with_pivot_u(0.0),
                ),
                (
                    "dbg-amf-u.1",
                    SolverSettings::default().with_ordering(OrderingMethod::Amf),
                ),
                (
                    "dbg-metis-u.1",
                    SolverSettings::default().with_ordering(OrderingMethod::MetisND),
                ),
                (
                    "dbg-amd-u.1-mf",
                    SolverSettings::default()
                        .with_ordering(OrderingMethod::Amd)
                        .with_method(rslab::FactorMethod::Multifrontal),
                ),
            ] {
                let f = rslab::factor_general_lu(&a, &opts).unwrap();
                let umax = f.u_values.iter().map(|v| v.norm()).fold(0.0, f64::max);
                let lmax = f.l_values.iter().map(|v| v.norm()).fold(0.0, f64::max);
                let mut seen = vec![false; n];
                let mut dup = 0;
                for &r in &f.perm_row {
                    if seen[r] {
                        dup += 1;
                    }
                    seen[r] = true;
                }
                let mut seenc = vec![false; n];
                let mut dupc = 0;
                for &c in &f.perm {
                    if seenc[c] {
                        dupc += 1;
                    }
                    seenc[c] = true;
                }
                let x = rslab::solve_lu(&f, &b).unwrap();
                let interchanges = f
                    .perm
                    .iter()
                    .zip(&f.perm_row)
                    .filter(|(c, r)| c != r)
                    .count();
                let u_bad = (0..n).filter(|&e| f.u_col_idx[f.u_row_ptr[e]] != e).count();
                let l_bad = (0..n).filter(|&j| f.l_row_idx[f.l_col_ptr[j]] != j).count();
                let unsorted_l = (0..n)
                    .filter(|&j| {
                        f.l_row_idx[f.l_col_ptr[j]..f.l_col_ptr[j + 1]]
                            .windows(2)
                            .any(|w| w[0] >= w[1])
                    })
                    .count();
                let unsorted_u = (0..n)
                    .filter(|&j| {
                        f.u_col_idx[f.u_row_ptr[j] + 1..f.u_row_ptr[j + 1]]
                            .windows(2)
                            .any(|w| w[0] >= w[1])
                    })
                    .count();
                // Elimination-tree nesting of the emitted L (parent = min row below the diagonal).
                let nest = |col_ptr: &[usize], row_idx: &[usize]| -> usize {
                    let mut parent = vec![usize::MAX; n];
                    for j in 0..n {
                        parent[j] = row_idx[col_ptr[j]..col_ptr[j + 1]]
                            .iter()
                            .copied()
                            .filter(|&r| r > j)
                            .min()
                            .unwrap_or(usize::MAX);
                    }
                    let mut bad = 0;
                    for j in 0..n {
                        for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                            if r <= j {
                                continue;
                            }
                            let mut p = parent[j];
                            let mut ok = false;
                            let mut steps = 0;
                            while p != usize::MAX && steps < 100000 {
                                if p == r {
                                    ok = true;
                                    break;
                                }
                                if p > r {
                                    break;
                                }
                                p = parent[p];
                                steps += 1;
                            }
                            if !ok && r != parent[j] {
                                bad += 1;
                            }
                        }
                    }
                    bad
                };
                let l_nest = nest(&f.l_col_ptr, &f.l_row_idx);
                let u_nest = nest(&f.u_row_ptr, &f.u_col_idx);
                println!("{label:>22} etree violations: L={l_nest} U^T={u_nest}");
                let sv = LuSolver::factor(&a, &opts).unwrap();
                let xs = sv.solve(&b).unwrap();
                println!("{label:>22} nnzLU={} max|U|={:.1e} max|L|={:.1e} dups={dup}/{dupc} row!=col={interchanges} diag-not-first U={u_bad} L={l_bad} unsorted L={unsorted_l} U={unsorted_u} res(csc)={:.1e} res(plan)={:.1e}", f.factor_nnz(), umax, lmax, resid(&a, &x, &b), resid(&a, &xs, &b));
            }
            // Growth and pivot-free variants.
            for (label, opts) in [
                ("lu-u0", SolverSettings::default().with_pivot_u(0.0)),
                (
                    "lu-u1-mf",
                    SolverSettings::default()
                        .with_pivot_u(1.0)
                        .with_method(rslab::FactorMethod::Multifrontal),
                ),
            ] {
                let f = rslab::factor_general_lu(&a, &opts).unwrap();
                let amax = a.values.iter().map(|v| v.norm()).fold(0.0, f64::max);
                let umax = f.u_values.iter().map(|v| v.norm()).fold(0.0, f64::max);
                let lmax = f.l_values.iter().map(|v| v.norm()).fold(0.0, f64::max);
                let x = rslab::solve_lu(&f, &b).unwrap();
                println!(
                    "{label:>18} nnzLU={} growth max|U|/max|A|={:.1e} max|L|={:.1e} res={:.1e}",
                    f.factor_nnz(),
                    umax / amax,
                    lmax,
                    resid(&a, &x, &b)
                );
            }
            {
                // Transposed system: the same structure with rows and columns swapped.
                let mut t_col: Vec<Vec<(usize, C)>> = vec![Vec::new(); n];
                for j in 0..n {
                    for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                        t_col[a.row_idx[k]].push((j, a.values[k]));
                    }
                }
                let (mut cp, mut ri, mut vv) = (vec![0usize], Vec::new(), Vec::new());
                for col in &mut t_col {
                    col.sort_by_key(|e| e.0);
                    for &(r, v) in col.iter() {
                        ri.push(r);
                        vv.push(v);
                    }
                    cp.push(ri.len());
                }
                let at = rslab::GeneralCsc {
                    n,
                    col_ptr: cp,
                    row_idx: ri,
                    values: vv,
                };
                let s = LuSolver::factor(&at, &SolverSettings::default()).unwrap();
                let x = s.solve(&b).unwrap();
                println!(
                    "{:>18} nnzLU={} res={:.1e}",
                    "lu-transposed",
                    s.factor_nnz(),
                    resid(&at, &x, &b)
                );
            }
            // Reference: the scalar CSC solve on the plain LU factors.
            {
                let f = rslab::factor_general_lu(&a, &SolverSettings::default()).unwrap();
                let x = rslab::solve_lu(&f, &b).unwrap();
                println!(
                    "{:>9} nnzLU={:>9} scalar-solve res={:.1e}",
                    "lu-csc",
                    f.factor_nnz(),
                    resid(&a, &x, &b)
                );
            }
            for &(name, om) in &orderings {
                let mut opts = SolverSettings::default();
                if let Some(o) = om {
                    opts = opts.with_ordering(o);
                }
                let r = std::panic::catch_unwind(|| {
                    if om.is_none() {
                        let (sym, pick) = LuSolver::<C>::tuned(&a).unwrap();
                        sym.factor(&a, &pick).unwrap()
                    } else {
                        LuSolver::factor(&a, &opts).unwrap()
                    }
                });
                let Ok(s) = r else {
                    println!("{name:>7} lu failed");
                    continue;
                };
                let d = s.diagnostics();
                let rt = d.rates();
                let (ts, x) = best(5, || s.solve(&b).unwrap());
                let plan_mb = d
                    .stages
                    .iter()
                    .find(|st| st.name == "solve-layout")
                    .map_or(0.0, |st| st.bytes as f64 / 1e6);
                println!("{:>7} nnzLU={:>9} fill={:>5.1} plan={:>6.1} MB | analyze {:>8.0} ms factor {:>8.1} ms ({:.2} MDOF/s, {:.1} GF/s) solve {:>7.3} ms ({:.1} MDOF/s) res={:.1e}",
                    format!("lu-{name}"), s.factor_nnz(), s.factor_nnz() as f64 / a.values.len() as f64, plan_mb,
                    d.stage_ms("analyze").unwrap_or(0.0), d.stage_ms("factor").unwrap_or(0.0), rt.factor_mdof_s, rt.factor_gflops,
                    ts * 1e3, n as f64 / ts / 1e6, resid(&a, &x, &b));
            }
        }
    }
}

fn resid(a: &rslab::GeneralCsc<C>, x: &[C], b: &[C]) -> f64 {
    let mut r = b.to_vec();
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            r[a.row_idx[k]] -= a.values[k] * x[j];
        }
    }
    let rn: f64 = r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let bn: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    rn / bn
}
