//! KLU path vs multifrontal LU on circuit-shaped matrices (#15).
//!
//! Generates MNA-like matrices: very sparse (~4-5 nnz/col), unsymmetric,
//! diagonally weighted, with reducible (block upper triangular) structure,
//! and reports for each size:
//!   * KLU: analyze / factor / refactor / solve wall time, factor nnz, blocks
//!   * multifrontal LU (defaults): factor / solve wall time, factor nnz
//!
//! plus the frequency-sweep proxy: 20 refactor+solve cycles KLU vs 20
//! factor+solve cycles multifrontal.
//!
//! Run: `cargo bench --bench klu_circuit`.

use std::time::Instant;

use rslab::{factor_general_lu, solve_lu, GeneralCsc, KluSettings, KluSymbolic, SolverSettings};

/// Deterministic xorshift.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// MNA-like circuit matrix: n nodes in `n_stages` cascaded stages (signal
/// flows one way between stages -> reducible), each stage internally coupled
/// (local 2D neighborhood + a few long-range couplings -> irreducible blocks
/// of stage size). Diagonal ~ sum of couplings (conductance-like), values
/// unsymmetric.
fn circuit(n: usize, n_stages: usize, seed: u64) -> GeneralCsc<f64> {
    let mut rng = Rng(seed | 1);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let mut colsum = vec![0.0f64; n];
    let stage = n / n_stages;
    for j in 0..n {
        let s = (j / stage).min(n_stages - 1);
        let (lo, hi) = (
            s * stage,
            if s == n_stages - 1 {
                n
            } else {
                (s + 1) * stage
            },
        );
        let span = hi - lo;
        // local neighborhood within the stage (wraps -> strongly connected);
        // slightly unsymmetric conductances
        for &t in &[1usize, 2, 17] {
            let i = lo + (j - lo + t) % span;
            if i != j {
                let g = 0.5 + rng.next_f64().abs();
                r.push(i);
                c.push(j);
                v.push(-g);
                colsum[j] += g;
                let g2 = 0.5 + rng.next_f64().abs();
                r.push(j);
                c.push(i);
                v.push(-g2);
                colsum[i] += g2;
            }
        }
        // one-directional inter-stage feed (previous stage listens)
        if s > 0 && (j - lo) % 3 == 0 {
            let i = lo - stage + (j - lo) % stage;
            let g = 0.2 + 0.3 * rng.next_f64().abs();
            r.push(i);
            c.push(j);
            v.push(-g);
            colsum[j] += g;
        }
    }
    // grounded-capacitor margin keeps every column strictly diagonally
    // dominant: the condition number stays O(1) at every size
    for (j, &cs) in colsum.iter().enumerate() {
        r.push(j);
        c.push(j);
        v.push(cs + 0.1 + 0.1 * rng.next_f64().abs());
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

// ---- SuiteSparse KLU reference (runtime FFI, LGPL library loaded only for
// measurement; nothing distributed) --------------------------------------------
// `benches/klu_shim.c` is compiled on demand against the locally installed
// SuiteSparse (Homebrew prefix by default, RLA_KLU_SS_PREFIX to override) and
// loaded via libloading - the same pattern as the MKL / Accelerate shims.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SsKluResult {
    ana_s: f64,
    fac_s: f64,
    refac_s: f64,
    slv_s: f64,
    sweep_s: f64,
    lnz: i64,
    unz: i64,
    nblocks: i32,
    ok: i32,
}

type SsKluFn = unsafe extern "C" fn(
    i32,        // n
    *const i32, // Ap
    *const i32, // Ai
    *const f64, // Ax
    *const f64, // b
    *mut f64,   // x
    i32,        // sweep_len
    *mut SsKluResult,
) -> i32;

struct SsKlu {
    _lib: libloading::Library,
    f: SsKluFn,
}

impl SsKlu {
    fn try_new() -> Option<Self> {
        let prefix = std::env::var("RLA_KLU_SS_PREFIX").unwrap_or_else(|_| "/opt/homebrew".into());
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let src = root.join("benches/klu_shim.c");
        let dylib = root.join("target/klu_shim.dylib");
        let mtime = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let stale = match (mtime(&src), mtime(&dylib)) {
            (Some(s), Some(d)) => s > d,
            _ => true,
        };
        if stale {
            let st = std::process::Command::new("cc")
                .args(["-O2", "-std=c11", "-dynamiclib"])
                .arg(format!("-I{prefix}/include/suitesparse"))
                .arg(format!("-L{prefix}/lib"))
                .arg("-lklu")
                .arg(&src)
                .arg("-o")
                .arg(&dylib)
                .status();
            if !st.map(|s| s.success()).unwrap_or(false) {
                eprintln!("[ss-klu] shim compile failed (SuiteSparse installed at {prefix}?)");
                return None;
            }
        }
        let lib = unsafe { libloading::Library::new(&dylib).ok()? };
        let f: SsKluFn = unsafe {
            let s: libloading::Symbol<SsKluFn> = lib.get(b"klu_shim_run").ok()?;
            *s
        };
        Some(SsKlu { _lib: lib, f })
    }

    fn run(&self, a: &GeneralCsc<f64>, b: &[f64], sweep: usize) -> Option<(SsKluResult, Vec<f64>)> {
        let ap: Vec<i32> = a.col_ptr.iter().map(|&v| v as i32).collect();
        let ai: Vec<i32> = a.row_idx.iter().map(|&v| v as i32).collect();
        let mut x = vec![0.0f64; a.n];
        let mut r = SsKluResult::default();
        let st = unsafe {
            (self.f)(
                a.n as i32,
                ap.as_ptr(),
                ai.as_ptr(),
                a.values.as_ptr(),
                b.as_ptr(),
                x.as_mut_ptr(),
                sweep as i32,
                &mut r,
            )
        };
        if st != 0 || r.ok != 1 {
            eprintln!("[ss-klu] failed with status {st}");
            return None;
        }
        Some((r, x))
    }
}

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

/// Settings sweep (`RLA_KLU_SWEEP=1`): grid over the three KLU knobs on the
/// circuit family plus a badly row-scaled variant, warm best-of-3 per config.
/// Evidence base for the KLU default-settings choice (heuristic defaults work,
/// 2026-07): confirms/updates that `pivot_tol=1e-3, row_scaling=on, btf=on`
/// is on the speed/fill/robustness Pareto front.
fn settings_sweep() {
    println!(
        "{:>10} {:>8} {:>5} {:>5} | {:>9} {:>9} {:>9} {:>10} {:>7} {:>9}",
        "matrix", "ptol", "scal", "btf", "ana-ms", "fac-ms", "refac-ms", "fill", "blocks", "resid"
    );
    for &(n, stages, bad_scaling) in &[
        (10_000usize, 16usize, false),
        (50_000, 32, false),
        (200_000, 64, false),
        (50_000, 32, true),
    ] {
        let mut a = circuit(n, stages, 0xC0FFEE ^ n as u64);
        if bad_scaling {
            // Row scaling over ~12 orders of magnitude: the robustness case
            // row_scaling is for (badly equilibrated device models).
            // CSC: row i of the matrix appears via row_idx - scale per row.
            let scale = |i: usize| 10f64.powi(((i * 7919) % 13) as i32 - 6);
            for (k, &i) in a.row_idx.iter().enumerate() {
                a.values[k] *= scale(i);
            }
        }
        let name = format!("{}{}", n, if bad_scaling { "-badscale" } else { "" });
        let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        for &btf in &[true, false] {
            for &scal in &[true, false] {
                for &ptol in &[1e-3f64, 1e-2, 0.1, 1.0] {
                    let s = KluSettings::default()
                        .with_pivot_tol(ptol)
                        .with_row_scaling(scal)
                        .with_btf(btf);
                    let t = Instant::now();
                    let sym = match KluSymbolic::analyze_with(&a, &s) {
                        Ok(x) => x,
                        Err(e) => {
                            println!(
                                "{name:>10} {ptol:>8.0e} {scal:>5} {btf:>5} | analyze FAILED {e}"
                            );
                            continue;
                        }
                    };
                    let ana = t.elapsed().as_secs_f64() * 1e3;
                    let mut fac_best = f64::INFINITY;
                    let mut refac_best = f64::INFINITY;
                    let mut fill = 0usize;
                    let mut blocks = 0usize;
                    let mut res = f64::NAN;
                    let mut failed = false;
                    for _ in 0..3 {
                        let t = Instant::now();
                        match sym.factor(&a, &s) {
                            Ok(mut f) => {
                                fac_best = fac_best.min(t.elapsed().as_secs_f64() * 1e3);
                                let t = Instant::now();
                                if f.refactor(&a).is_ok() {
                                    refac_best = refac_best.min(t.elapsed().as_secs_f64() * 1e3);
                                }
                                fill = f.factor_nnz();
                                blocks = f.n_blocks();
                                if let Ok(x) = f.solve(&b) {
                                    res = resid(&a, &x, &b);
                                }
                            }
                            Err(e) => {
                                println!(
                                    "{name:>10} {ptol:>8.0e} {scal:>5} {btf:>5} | factor FAILED {e}"
                                );
                                failed = true;
                                break;
                            }
                        }
                    }
                    if failed {
                        continue;
                    }
                    println!(
                        "{name:>10} {ptol:>8.0e} {scal:>5} {btf:>5} | {ana:>9.1} {fac_best:>9.1} {refac_best:>9.1} {fill:>10} {blocks:>7} {res:>9.1e}"
                    );
                }
            }
        }
    }
}

fn main() {
    const SWEEP: usize = 20;
    if std::env::var("RLA_KLU_SWEEP")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        settings_sweep();
        return;
    }
    println!(
        "{:>8} {:>9} | {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} | {:>9} {:>9} {:>9} | {:>8} {:>8}",
        "n",
        "nnz",
        "klu-ana",
        "klu-fac",
        "klu-refac",
        "klu-solve",
        "klu-nnz",
        "blocks",
        "mf-fac",
        "mf-solve",
        "mf-nnz",
        "sweep-klu",
        "sweep-mf"
    );
    for &(n, stages) in &[
        (2_000usize, 8usize),
        (10_000, 16),
        (50_000, 32),
        (200_000, 64),
    ] {
        let a = circuit(n, stages, 0xC0FFEE ^ n as u64);
        let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();

        // --- KLU ---
        let t = Instant::now();
        let sym = KluSymbolic::analyze(&a).unwrap();
        let t_ana = t.elapsed();
        let t = Instant::now();
        let mut klu = sym.factor(&a, &KluSettings::default()).unwrap();
        let t_fac = t.elapsed();
        let t = Instant::now();
        klu.refactor(&a).unwrap();
        let t_refac = t.elapsed();
        let t = Instant::now();
        let x = klu.solve(&b).unwrap();
        let t_solve = t.elapsed();
        let klu_res = resid(&a, &x, &b);
        let klu_nnz = klu.factor_nnz();
        let blocks = klu.n_blocks();

        // --- multifrontal LU (defaults) ---
        let t = Instant::now();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let t_mf_fac = t.elapsed();
        let t = Instant::now();
        let xm = solve_lu(&f, &b).unwrap();
        let t_mf_solve = t.elapsed();
        let mf_res = resid(&a, &xm, &b);
        let mf_nnz = f.factor_nnz();
        assert!(
            klu_res < 1e-8,
            "klu residual {klu_res} (mf residual {mf_res})"
        );
        assert!(mf_res < 1e-8, "multifrontal residual {mf_res}");

        // --- sweep proxy: 20 value sets, same pattern ---
        let t = Instant::now();
        for k in 0..SWEEP {
            let scale = 1.0 + 0.03 * k as f64;
            let a2 = GeneralCsc {
                n: a.n,
                col_ptr: a.col_ptr.clone(),
                row_idx: a.row_idx.clone(),
                values: a.values.iter().map(|&v| v * scale).collect(),
            };
            klu.refactor(&a2).unwrap();
            let _ = klu.solve(&b).unwrap();
        }
        let t_sweep_klu = t.elapsed();
        let t = Instant::now();
        for k in 0..SWEEP {
            let scale = 1.0 + 0.03 * k as f64;
            let a2 = GeneralCsc {
                n: a.n,
                col_ptr: a.col_ptr.clone(),
                row_idx: a.row_idx.clone(),
                values: a.values.iter().map(|&v| v * scale).collect(),
            };
            let f2 = factor_general_lu(&a2, &SolverSettings::default()).unwrap();
            let _ = solve_lu(&f2, &b).unwrap();
        }
        let t_sweep_mf = t.elapsed();

        println!(
            "{:>8} {:>9} | {:>9.2?} {:>9.2?} {:>9.2?} {:>9.2?} {:>9} {:>7} | {:>9.2?} {:>9.2?} {:>9} | {:>8.2?} {:>8.2?}",
            n,
            a.nnz(),
            t_ana,
            t_fac,
            t_refac,
            t_solve,
            klu_nnz,
            blocks,
            t_mf_fac,
            t_mf_solve,
            mf_nnz,
            t_sweep_klu,
            t_sweep_mf
        );
        println!(
            "{:>8} residuals: klu {:.1e}  mf {:.1e}",
            "", klu_res, mf_res
        );

        // --- parallel per-block factor + refactor (opt-in, ambient pool) ---
        let t = Instant::now();
        let mut klu_par = sym
            .factor(&a, &KluSettings::default().with_parallel_factor(true))
            .unwrap();
        let t_fac_par = t.elapsed();
        assert_eq!(klu_par.factor_nnz(), klu_nnz, "parallel factor differs");
        let t = Instant::now();
        klu_par.refactor(&a).unwrap();
        let t_refac_par = t.elapsed();
        let t = Instant::now();
        for k in 0..SWEEP {
            let scale = 1.0 + 0.03 * k as f64;
            let a2 = GeneralCsc {
                n: a.n,
                col_ptr: a.col_ptr.clone(),
                row_idx: a.row_idx.clone(),
                values: a.values.iter().map(|&v| v * scale).collect(),
            };
            klu_par.refactor(&a2).unwrap();
            let _ = klu_par.solve(&b).unwrap();
        }
        let t_sweep_par = t.elapsed();
        println!(
            "{:>8} parallel-blocks: fac {:>8.2?} ({:.1}x) refac {:>8.2?} ({:.1}x) sweep {:>8.2?} ({:.1}x vs sequential)",
            "",
            t_fac_par,
            t_fac.as_secs_f64() / t_fac_par.as_secs_f64(),
            t_refac_par,
            t_refac.as_secs_f64() / t_refac_par.as_secs_f64(),
            t_sweep_par,
            t_sweep_klu.as_secs_f64() / t_sweep_par.as_secs_f64()
        );

        // --- SuiteSparse KLU reference (same matrix, same phases) ---
        if let Some(ss) = SsKlu::try_new() {
            if let Some((r, xs)) = ss.run(&a, &b, SWEEP) {
                let ss_res = resid(&a, &xs, &b);
                assert!(ss_res < 1e-8, "suitesparse klu residual {ss_res}");
                println!(
                    "{:>8} suitesparse-klu: ana {:>8.2?} fac {:>8.2?} refac {:>8.2?} solve {:>8.2?} lnz+unz {:>9} blocks {:>5} sweep {:>8.2?} res {:.1e}",
                    "",
                    std::time::Duration::from_secs_f64(r.ana_s),
                    std::time::Duration::from_secs_f64(r.fac_s),
                    std::time::Duration::from_secs_f64(r.refac_s),
                    std::time::Duration::from_secs_f64(r.slv_s),
                    r.lnz + r.unz,
                    r.nblocks,
                    std::time::Duration::from_secs_f64(r.sweep_s),
                    ss_res
                );
            }
        }

        // --- wide-RHS solve: batched solve_many vs per-column loop ---
        const NRHS: usize = 8;
        let bm: Vec<f64> = (0..n * NRHS).map(|k| ((k * 3) % 17) as f64 - 8.0).collect();
        let t = Instant::now();
        let xm_batched = klu.solve_many(&bm, NRHS).unwrap();
        let t_batched = t.elapsed();
        let t = Instant::now();
        let mut xm_loop = vec![0.0; n * NRHS];
        for c in 0..NRHS {
            let bc: Vec<f64> = (0..n).map(|i| bm[i * NRHS + c]).collect();
            let xc = klu.solve(&bc).unwrap();
            for i in 0..n {
                xm_loop[i * NRHS + c] = xc[i];
            }
        }
        let t_loop = t.elapsed();
        assert_eq!(
            xm_batched, xm_loop,
            "batched solve_many must be bit-identical"
        );
        println!(
            "{:>8} solve x{}: batched {:.2?} vs looped {:.2?} ({:.1}x)",
            "",
            NRHS,
            t_batched,
            t_loop,
            t_loop.as_secs_f64() / t_batched.as_secs_f64()
        );
    }
}
