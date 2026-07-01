//! `cargo xtask` — the RSLAB meta-tuner (issue #1).
//!
//! Turns the auto-tuner from a compile-time artifact into a reproducible,
//! hardware-calibrated pipeline that emits a runtime `tuner_profile.json`:
//!
//!   sweep  -> train -> calibrate -> profile -> validate -> ship-gate
//!
//! * **sweep**    `cargo bench --bench sweep` over the corpus (shelled out).
//! * **train**    `python benches/train_tuner.py` -> two per-path model JSONs.
//! * **calibrate**  in-process hardware microbench (proxy-GFLOP/s, speedup, CV).
//! * **profile**  bundle the two models + calibration-derived guards.
//! * **validate**  factor a held-out generator corpus with the default tuner vs
//!   the candidate profile; geomean the per-matrix speedup.
//! * **ship-gate**  write the profile only if it does not regress the default.
//!
//! Subcommands: `calibrate`, `validate <profile>`, `profile <models_dir> <out>
//! [class]`, `tune <workdir>`.

use num_complex::Complex;
use rslab::tuning::{Calibration, HardwareInfo};
use rslab::{
    default_profile, recommend_settings_pathed, recommend_with_profile, CscMatrix, LdltSolver,
    LdltSymbolic, SolverPath, StructuralFeatures, TunerProfile, DEFAULT_TUNE_WEIGHT,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let rest = &args[args.len().min(1)..];
    let code = match cmd {
        "calibrate" => cmd_calibrate(),
        "validate" => cmd_validate(rest),
        "profile" => cmd_profile(rest),
        "tune" => cmd_tune(rest),
        _ => {
            eprintln!(
                "cargo xtask <command>\n  calibrate                     hardware microbench summary\n  validate <profile.json>       held-out geomean speedup vs default\n  profile <models_dir> <out.json> [class]\n                                assemble + ship-gate a profile\n  tune <workdir>                full sweep->train->profile pipeline"
            );
            2
        }
    };
    std::process::exit(code);
}

/// Probe the machine and return its (cached) calibration.
fn calibration() -> (HardwareInfo, Calibration) {
    let hw = HardwareInfo::probe();
    let calib = Calibration::load_or_measure(&hw);
    (hw, calib)
}

fn cmd_calibrate() -> i32 {
    let (hw, c) = calibration();
    println!("hardware fingerprint : {:016x}", c.fingerprint);
    println!("physical cores       : {}", hw.physical_cores);
    println!("proxy GFLOP/s (f64)  : {:.2}", c.geom_gflops);
    println!("proxy GFLOP/s (cplx) : {:.2}", c.geom_gflops_cplx);
    println!("parallel speedup     : {:.2}x @ {} threads", c.speedup, c.speedup_threads);
    println!("timing CV            : {:.3}", c.time_cv);
    println!("=> calibrated min_gain guard : {:.3}", c.min_gain());
    0
}

/// A held-out validation corpus built from the structured-grid generators (no
/// downloads): complex-symmetric curl-curl (EM) + saddle-point (Stokes/KKT),
/// spanning the LDLᵀ path's indefinite classes. Sizes stay modest so a full
/// validation runs in a few seconds.
fn holdout_corpus() -> Vec<(String, CscMatrix<Complex<f64>>)> {
    let mut v = Vec::new();
    for n in [10usize, 14, 18] {
        v.push((format!("curlcurl_{n}"), rslab::matgen::fem::curl_curl(&[n, n, n], 1.0, 0.1)));
    }
    for n in [24usize, 40, 60] {
        v.push((format!("saddle_{n}"), rslab::matgen::fem::saddle_point::<Complex<f64>>(&[n, n], 1e-3)));
    }
    v
}

/// The (features, mf/ll ratio) a tuner needs for one matrix — mirrors
/// `LdltSymbolic::tuned`'s a-priori inputs.
fn tuner_inputs(
    a: &CscMatrix<Complex<f64>>,
) -> Option<(StructuralFeatures, f64)> {
    let sym = LdltSymbolic::analyze(a).ok()?;
    let est = sym.estimate_memory::<Complex<f64>>();
    let feat = StructuralFeatures::from_symmetric(a, &sym);
    let mf_ll = if est.panel_live_peak_bytes > 0 {
        est.mf_transient_peak_bytes as f64 / est.panel_live_peak_bytes as f64
    } else {
        1.0
    };
    Some((feat, mf_ll))
}

/// Median wall-time (seconds) of an end-to-end factor with `opts`, over 3 runs.
fn time_factor(a: &CscMatrix<Complex<f64>>, opts: &rslab::SolverSettings) -> Option<f64> {
    let mut ts = Vec::new();
    for _ in 0..3 {
        let start = std::time::Instant::now();
        LdltSolver::factor_with(a, opts).ok()?;
        ts.push(start.elapsed().as_secs_f64());
    }
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    Some(ts[ts.len() / 2])
}

/// Geomean per-matrix speedup of `profile`'s recommendation over the embedded
/// default tuner on the held-out corpus (>1 = candidate faster). Prints a table.
fn validate_profile(profile: &TunerProfile) -> f64 {
    let w = DEFAULT_TUNE_WEIGHT;
    let mut log_sum = 0.0;
    let mut count = 0u32;
    println!("{:<14} {:>10} {:>10} {:>8}", "matrix", "default_ms", "cand_ms", "speedup");
    for (name, a) in holdout_corpus() {
        let Some((feat, mf_ll)) = tuner_inputs(&a) else { continue };
        let s_def = recommend_settings_pathed(&feat, w, mf_ll, SolverPath::Ldlt);
        let Some(s_cand) = recommend_with_profile(profile, &feat, w, mf_ll, SolverPath::Ldlt) else {
            continue;
        };
        let (Some(t_def), Some(t_cand)) = (time_factor(&a, &s_def), time_factor(&a, &s_cand)) else {
            continue;
        };
        let speedup = t_def / t_cand.max(1e-9);
        println!("{name:<14} {:>10.2} {:>10.2} {:>8.3}", t_def * 1e3, t_cand * 1e3, speedup);
        log_sum += speedup.max(1e-6).ln();
        count += 1;
    }
    if count == 0 {
        return 1.0;
    }
    (log_sum / count as f64).exp()
}

fn cmd_validate(rest: &[String]) -> i32 {
    let profile = match rest.first() {
        Some(p) => match TunerProfile::load(Path::new(p)) {
            Ok(pr) => pr,
            Err(e) => {
                eprintln!("cannot load profile {p}: {e}");
                return 1;
            }
        },
        None => {
            eprintln!("usage: cargo xtask validate <profile.json>");
            return 2;
        }
    };
    let geomean = validate_profile(&profile);
    println!("=> geomean speedup vs default: {geomean:.3}x");
    0
}

/// Assemble a candidate profile from trained per-path models + fresh calibration,
/// run the ship-gate, and write it out only if it does not regress the default.
fn cmd_profile(rest: &[String]) -> i32 {
    let (models_dir, out) = match (rest.first(), rest.get(1)) {
        (Some(d), Some(o)) => (PathBuf::from(d), PathBuf::from(o)),
        _ => {
            eprintln!("usage: cargo xtask profile <models_dir> <out.json> [class]");
            return 2;
        }
    };
    let class = rest.get(2).cloned().unwrap_or_else(|| "corpus".to_string());
    assemble_and_ship(&models_dir, &out, &class)
}

/// The shared profile-assembly + ship-gate used by `profile` and `tune`.
fn assemble_and_ship(models_dir: &Path, out: &Path, class: &str) -> i32 {
    let ldlt_path = models_dir.join("auto_tune_model_ldlt.json");
    let lu_path = models_dir.join("auto_tune_model_lu.json");
    let read_json = |p: &Path| -> Result<serde_json::Value, String> {
        let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
    };
    let (ldlt_model, lu_model) = match (read_json(&ldlt_path), read_json(&lu_path)) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            eprintln!("cannot read trained models:");
            for r in [a, b].into_iter() {
                if let Err(e) = r {
                    eprintln!("  {e}");
                }
            }
            return 1;
        }
    };
    // Guards: min_gain is data-calibrated (z·CV) to this machine's timing noise;
    // the method-flip and memory guards keep the proven defaults (memory is the
    // hard resource — never relaxed by calibration).
    let (_, calib) = calibration();
    let def = default_profile();
    let candidate = TunerProfile {
        class: class.to_string(),
        ldlt_model,
        lu_model,
        min_gain: calib.min_gain(),
        method_flip_gain: def.method_flip_gain,
        mem_tol_ln: def.mem_tol_ln,
    };

    println!("== ship-gate: candidate vs default ==");
    let geomean = validate_profile(&candidate);
    println!("=> candidate geomean speedup vs default: {geomean:.3}x");
    // Ship only if the candidate does not regress the default beyond the timing
    // noise floor (1% band). A candidate that is merely neutral still ships (it is
    // the freshly-trained, hardware-calibrated one); a regression is rejected.
    const SHIP_MIN: f64 = 0.99;
    if geomean < SHIP_MIN {
        eprintln!(
            "SHIP-GATE FAILED: {geomean:.3}x < {SHIP_MIN:.2}x — keeping default, not writing {}",
            out.display()
        );
        return 1;
    }
    match candidate.save(out) {
        Ok(()) => {
            println!("SHIP-GATE PASSED: wrote {} (class '{class}')", out.display());
            0
        }
        Err(e) => {
            eprintln!("failed to write {}: {e}", out.display());
            1
        }
    }
}

/// Full pipeline: sweep (unless `sweep.jsonl` exists) -> train -> profile+ship.
/// `workdir` holds the intermediate artifacts (`sweep.jsonl`, model JSONs) and
/// the final `tuner_profile.json`.
fn cmd_tune(rest: &[String]) -> i32 {
    let workdir = match rest.first() {
        Some(d) => PathBuf::from(d),
        None => {
            eprintln!("usage: cargo xtask tune <workdir>");
            return 2;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&workdir) {
        eprintln!("cannot create {}: {e}", workdir.display());
        return 1;
    }
    let sweep = workdir.join("sweep.jsonl");

    // 1) Sweep — expensive; skipped if the corpus sweep is already present so the
    // pipeline is resumable. Honors RLA_SWEEP_OUT so the bench writes into workdir.
    if !sweep.exists() {
        println!("== sweep: cargo bench --bench sweep (this is the expensive step) ==");
        let status = Command::new("cargo")
            .args(["bench", "--bench", "sweep", "--features", "matgen matgen-download tuning"])
            .env("RLA_SWEEP_OUT", &sweep)
            .env("RLA_SWEEP_MODE", "sweep")
            .status();
        match status {
            Ok(s) if s.success() => {}
            _ => {
                eprintln!("sweep failed; provide a pre-computed {} to skip it", sweep.display());
                return 1;
            }
        }
    } else {
        println!("== sweep: reusing existing {} ==", sweep.display());
    }

    // 2) Train — the Python trainer writes both per-path models into workdir.
    println!("== train: benches/train_tuner.py ==");
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python".to_string());
    let status = Command::new(&py)
        .arg("benches/train_tuner.py")
        .arg(&sweep)
        .arg(&workdir)
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("training failed (need Python + numpy). Models not produced.");
            return 1;
        }
    }

    // 3) Calibrate + 4) assemble + 5/6) validate + ship-gate.
    let out = workdir.join("tuner_profile.json");
    assemble_and_ship(&workdir, &out, "corpus")
}
