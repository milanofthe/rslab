//! `cargo xtask`: hardware calibration for the `tuning` feature.
//!
//! `calibrate` runs the in-process hardware microbench (proxy-GFLOP/s,
//! parallel speedup, timing CV) and writes the calibration cache consumed
//! by `tuned()`'s cost-model worker-count pick.

use rslab::tuning::{Calibration, HardwareInfo};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let code = match cmd {
        "calibrate" => cmd_calibrate(),
        _ => {
            eprintln!("cargo xtask <command>\n  calibrate    hardware microbench summary (writes the calibration cache)");
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
    println!(
        "parallel speedup     : {:.2}x @ {} threads",
        c.speedup, c.speedup_threads
    );
    println!("timing CV            : {:.3}", c.time_cv);
    println!("=> calibrated min_gain guard : {:.3}", c.min_gain());
    0
}
