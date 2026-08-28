//! RSLAB KLU vs SuiteSparse KLU on real circuit matrices (SuiteSparse Matrix
//! Collection, the KLU paper's application domain).
//!
//! Downloads each matrix once into the matgen cache (`RLA_MATGEN_CACHE` or the
//! system temp dir), then runs the same phases on both solvers with identical
//! settings (threshold partial pivoting `tol=1e-3`, row-max scaling, BTF):
//! analyze / factor / numeric-only refactor / solve, plus a frequency-sweep
//! proxy of 20 refactor+solve cycles on unchanged values. RSLAB is measured
//! strictly sequentially and (separately) with the opt-in parallel per-block
//! factor/refactor; SuiteSparse KLU is sequential by design.
//!
//! Run: `cargo bench --bench klu_realworld --features matgen-download`
//! Env:  `RLA_KLU_SS_PREFIX` - SuiteSparse install for the reference shim
//!       (see `benches/ss_klu_ref.rs`); without it only RSLAB rows appear.
//!       `RLA_KLU_RW_BIG=1` - include the multi-million-row Freescale set.
//!       `RLA_BENCH_OUT=path.jsonl` - append one JSONL record per
//!       (matrix, solver) for `benches/klu_realworld_plot.py`.

use std::time::Instant;

use rslab::matgen::download::fetch;
use rslab::{read_mtx_any, GeneralCsc, KluSettings, KluSymbolic, MtxLoaded};

#[path = "ss_klu_ref.rs"]
mod ss_klu_ref;
use ss_klu_ref::SsKlu;

/// Real circuit / post-layout simulation matrices, ascending size. The core
/// set matches the families benchmarked in the KLU paper (Davis & Palamadai
/// Natarajan 2010): Bomhof+Hamm SPICE matrices, AT&T harmonic balance,
/// Sandia ASIC, Rajat post-layout.
const CORPUS: &[(&str, &str)] = &[
    ("Bomhof", "circuit_1"),
    ("Bomhof", "circuit_2"),
    ("Hamm", "add32"),
    ("Bomhof", "circuit_3"),
    ("Hamm", "memplus"),
    ("ATandT", "onetone2"),
    ("ATandT", "onetone1"),
    ("Rajat", "rajat15"),
    ("Bomhof", "circuit_4"),
    ("Sandia", "ASIC_100ks"),
    ("ATandT", "twotone"),
    ("Hamm", "scircuit"),
    ("Sandia", "ASIC_320ks"),
];
const CORPUS_BIG: &[(&str, &str)] = &[("Freescale", "memchip")];

const SWEEP: usize = 20;

fn resid(a: &GeneralCsc<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; a.n];
    a.matvec(x, &mut ax);
    let num = b
        .iter()
        .zip(&ax)
        .map(|(bi, axi)| (bi - axi).abs())
        .fold(0.0, f64::max);
    num / b.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-300)
}

/// Load `<group>/<name>` as a real general CSC, or explain why not.
fn load(group: &str, name: &str) -> Result<GeneralCsc<f64>, String> {
    let path = fetch(group, name)?;
    match read_mtx_any(&path).map_err(|e| format!("parse: {e}"))? {
        MtxLoaded::General(c) => {
            if c.values.iter().any(|v| v.im != 0.0) {
                return Err("complex values (real corpus expected)".into());
            }
            let a = GeneralCsc {
                n: c.n,
                col_ptr: c.col_ptr,
                row_idx: c.row_idx,
                values: c.values.iter().map(|v| v.re).collect(),
            };
            a.validate().map_err(|e| format!("validate: {e}"))?;
            Ok(a)
        }
        MtxLoaded::Symmetric(_) => Err("symmetric file (unsymmetric corpus expected)".into()),
    }
}

struct Row {
    fac_ratio: f64,   // ss / rslab-seq
    refac_ratio: f64, // ss / rslab-seq
    sweep_ratio: f64, // ss / rslab-seq
    par_fac_ratio: f64,
    par_sweep_ratio: f64,
}

/// Append one JSONL record to `RLA_BENCH_OUT` (no-op when unset).
#[allow(clippy::too_many_arguments)]
fn emit(
    group: &str,
    name: &str,
    n: usize,
    nnz: usize,
    solver: &str,
    times: [f64; 5], // ana, fac, refac, solve, sweep (seconds; NaN = n/a)
    fill: i64,
    blocks: i64,
    res: f64,
) {
    let Ok(path) = std::env::var("RLA_BENCH_OUT") else {
        return;
    };
    let rec = serde_json::json!({
        "group": group, "name": name, "n": n, "nnz": nnz, "solver": solver,
        "ana_s": times[0], "fac_s": times[1], "refac_s": times[2],
        "slv_s": times[3], "sweep_s": times[4],
        "fill": fill, "blocks": blocks, "res": res,
    });
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{rec}");
    }
}

fn main() {
    let big = std::env::var("RLA_KLU_RW_BIG")
        .map(|v| v == "1")
        .unwrap_or(false);
    let ss = SsKlu::try_new();
    if ss.is_none() {
        println!("[note] SuiteSparse KLU reference not available; RSLAB rows only.");
    }
    let mut ratios: Vec<Row> = Vec::new();

    let list: Vec<_> = CORPUS
        .iter()
        .chain(if big { CORPUS_BIG } else { &[] })
        .collect();
    for &&(group, name) in &list {
        let a = match load(group, name) {
            Ok(a) => a,
            Err(e) => {
                println!("=== {group}/{name}: SKIP ({e})");
                continue;
            }
        };
        let n = a.n;
        println!("=== {group}/{name}  n={}  nnz={} ===", n, a.nnz());
        let xt: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7) % 13) as f64 / 13.0).collect();
        let mut b = vec![0.0; n];
        a.matvec(&xt, &mut b);

        // --- RSLAB KLU, strictly sequential (default settings otherwise) ---
        let seq = KluSettings::default().with_parallel_factor(false);
        let t = Instant::now();
        let sym = match KluSymbolic::analyze_with(&a, &seq) {
            Ok(s) => s,
            Err(e) => {
                println!("  rslab analyze FAILED: {e}");
                continue;
            }
        };
        let t_ana = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut klu = match sym.factor(&a, &seq) {
            Ok(f) => f,
            Err(e) => {
                println!("  rslab factor FAILED: {e}");
                continue;
            }
        };
        let t_fac = t.elapsed().as_secs_f64();
        let t = Instant::now();
        klu.refactor(&a).unwrap();
        let t_refac = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let x = klu.solve(&b).unwrap();
        let t_slv = t.elapsed().as_secs_f64();
        let res = resid(&a, &x, &b);
        let t = Instant::now();
        for _ in 0..SWEEP {
            klu.refactor(&a).unwrap();
            let _ = klu.solve(&b).unwrap();
        }
        let t_sweep = t.elapsed().as_secs_f64();
        println!(
            "  rslab-klu:     ana {:8.1}ms  fac {:8.1}ms  refac {:8.1}ms  solve {:6.2}ms  \
             nnz {:>9}  blocks {:>5}  sweep {:8.1}ms  res {:.1e}",
            t_ana * 1e3,
            t_fac * 1e3,
            t_refac * 1e3,
            t_slv * 1e3,
            klu.factor_nnz(),
            klu.n_blocks(),
            t_sweep * 1e3,
            res
        );
        emit(
            group,
            name,
            n,
            a.nnz(),
            "rslab",
            [t_ana, t_fac, t_refac, t_slv, t_sweep],
            klu.factor_nnz() as i64,
            klu.n_blocks() as i64,
            res,
        );

        // --- RSLAB KLU, opt-in parallel per-block factor/refactor ---
        let par = KluSettings::default().with_parallel_factor(true);
        let t = Instant::now();
        let mut klu_p = sym.factor(&a, &par).unwrap();
        let t_fac_p = t.elapsed().as_secs_f64();
        assert_eq!(
            klu_p.factor_nnz(),
            klu.factor_nnz(),
            "parallel factor differs"
        );
        let t = Instant::now();
        klu_p.refactor(&a).unwrap();
        let t_refac_p = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..SWEEP {
            klu_p.refactor(&a).unwrap();
            let _ = klu_p.solve(&b).unwrap();
        }
        let t_sweep_p = t.elapsed().as_secs_f64();
        println!(
            "  rslab-klu-par: fac {:8.1}ms ({:.1}x)  refac {:8.1}ms ({:.1}x)  sweep {:8.1}ms ({:.1}x vs seq)",
            t_fac_p * 1e3,
            t_fac / t_fac_p,
            t_refac_p * 1e3,
            t_refac / t_refac_p,
            t_sweep_p * 1e3,
            t_sweep / t_sweep_p
        );
        emit(
            group,
            name,
            n,
            a.nnz(),
            "rslab-par",
            [f64::NAN, t_fac_p, t_refac_p, f64::NAN, t_sweep_p],
            klu_p.factor_nnz() as i64,
            klu_p.n_blocks() as i64,
            res,
        );

        // --- SuiteSparse KLU, same matrix, same phases ---
        if let Some(ss) = &ss {
            if let Some((r, xs)) = ss.run(&a, &b, SWEEP) {
                let res_ss = resid(&a, &xs, &b);
                println!(
                    "  ss-klu:        ana {:8.1}ms  fac {:8.1}ms  refac {:8.1}ms  solve {:6.2}ms  \
                     nnz {:>9}  blocks {:>5}  sweep {:8.1}ms  res {:.1e}",
                    r.ana_s * 1e3,
                    r.fac_s * 1e3,
                    r.refac_s * 1e3,
                    r.slv_s * 1e3,
                    r.lnz + r.unz,
                    r.nblocks,
                    r.sweep_s * 1e3,
                    res_ss
                );
                println!(
                    "  ratio ss/rslab (>1 = rslab faster): fac {:.2}x  refac {:.2}x  sweep {:.2}x  \
                     | parallel: fac {:.2}x  sweep {:.2}x",
                    r.fac_s / t_fac,
                    r.refac_s / t_refac,
                    r.sweep_s / t_sweep,
                    r.fac_s / t_fac_p,
                    r.sweep_s / t_sweep_p
                );
                ratios.push(Row {
                    fac_ratio: r.fac_s / t_fac,
                    refac_ratio: r.refac_s / t_refac,
                    sweep_ratio: r.sweep_s / t_sweep,
                    par_fac_ratio: r.fac_s / t_fac_p,
                    par_sweep_ratio: r.sweep_s / t_sweep_p,
                });
                emit(
                    group,
                    name,
                    n,
                    a.nnz(),
                    "ss-klu",
                    [r.ana_s, r.fac_s, r.refac_s, r.slv_s, r.sweep_s],
                    r.lnz + r.unz,
                    r.nblocks as i64,
                    res_ss,
                );
            }
        }
        println!();
    }

    if !ratios.is_empty() {
        let geo = |f: fn(&Row) -> f64| {
            (ratios.iter().map(|r| f(r).ln()).sum::<f64>() / ratios.len() as f64).exp()
        };
        println!(
            "geomean over {} matrices, ss/rslab (>1 = rslab faster):",
            ratios.len()
        );
        println!(
            "  sequential: fac {:.2}x  refac {:.2}x  sweep {:.2}x",
            geo(|r| r.fac_ratio),
            geo(|r| r.refac_ratio),
            geo(|r| r.sweep_ratio)
        );
        println!(
            "  parallel:   fac {:.2}x  sweep {:.2}x",
            geo(|r| r.par_fac_ratio),
            geo(|r| r.par_sweep_ratio)
        );
    }
}
