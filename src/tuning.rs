//! Hardware calibration (feature `tuning`).
//!
//! [`HardwareInfo::probe`](crate::tuning::HardwareInfo::probe) detects cores and RAM;
//! [`Calibration::load_or_measure`](crate::tuning::Calibration::load_or_measure)
//! measures this machine's factorization throughput and thread scaling once
//! (by factoring representative grids) and caches it, keyed by a hardware
//! fingerprint. The worker-count cost model reads it; the pick is
//! deterministic given the calibration.

use crate::constants::*;
use std::path::PathBuf;

use num_complex::Complex;

use crate::diagnostics::MemoryEstimate;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::{LdltSymbolic, SolverSettings};

/// Detected machine capabilities - for budgeting and the calibration key.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
}

impl HardwareInfo {
    /// Probe the current machine (cores via std, RAM via sysinfo). In a container (a cgroup
    /// with a memory limit, Linux) the RAM is the container's: the smaller of the host's and the
    /// cgroup's total and free memory, so a memory plan checked against it fails fast instead of
    /// being OOM-killed. The cores already honour a CPU quota (`available_parallelism`).
    pub fn probe() -> Self {
        let logical = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let physical = sys.physical_core_count().unwrap_or(logical);
        let (mut total, mut available) = (sys.total_memory(), sys.available_memory());
        if let Some(cg) = sys.cgroup_limits() {
            total = total.min(cg.total_memory);
            available = available.min(cg.free_memory);
        }
        HardwareInfo {
            logical_cores: logical,
            physical_cores: physical.max(1),
            total_ram_bytes: total,
            available_ram_bytes: available,
        }
    }
    /// Stable key for the calibration cache (cores + RAM size bucket).
    pub fn fingerprint(&self) -> u64 {
        let gb = self.total_ram_bytes / FINGERPRINT_RAM_UNIT;
        ((self.physical_cores as u64) << FINGERPRINT_PHYSICAL_SHIFT)
            ^ ((self.logical_cores as u64) << FINGERPRINT_LOGICAL_SHIFT)
            ^ gb.min(FINGERPRINT_RAM_MASK)
    }
}

/// Measured cost-model constants for this machine - the cached "wisdom".
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// One-thread throughput in the `factor_flops` proxy unit, x1e9 (giga/s), for a
    /// **real** (`f64`) factorization.
    pub geom_gflops: f64,
    /// Same proxy-flops/s rate for a **complex** (`Complex<f64>`) factorization:
    /// each proxy-flop is ~4 real flops plus wider memory traffic, so the rate in
    /// the type-independent proxy unit is lower --- calibrated separately so the
    /// runtime estimate is correct for the complex-symmetric target class.
    pub geom_gflops_cplx: f64,
    /// Parallel speedup measured at `speedup_threads`.
    pub speedup: f64,
    pub speedup_threads: usize,
    /// Parallel speedup measured at 4 threads - a mid-curve support point so
    /// [`speedup_for`](Self::speedup_for) does not overestimate small thread
    /// counts by linear interpolation toward the peak.
    pub speedup4: f64,
    /// Coefficient of variation (std/mean) of the repeated single-thread factor
    /// time on this machine --- the measurement noise floor. The tuner's deviate
    /// guard is set from this (`min_gain = z*cv`) so it never chases a speedup
    /// smaller than the noise it measured on this hardware.
    pub time_cv: f64,
    pub fingerprint: u64,
}

impl Calibration {
    fn cache_path(fp: u64) -> PathBuf {
        let dir = std::env::var_os("RLA_CALIB_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("rla-calib"));
        dir.join(format!("calib-{fp:016x}.txt"))
    }

    /// Cached calibration for `hw`, or measure it (and cache) on first use.
    pub fn load_or_measure(hw: &HardwareInfo) -> Self {
        let fp = hw.fingerprint();
        if let Some(c) = Self::load(fp) {
            return c;
        }
        let c = Self::measure(hw);
        let _ = c.save();
        c
    }

    fn load(fp: u64) -> Option<Self> {
        let s = std::fs::read_to_string(Self::cache_path(fp)).ok()?;
        let mut it = s.split_whitespace();
        let geom_gflops: f64 = it.next()?.parse().ok()?;
        let speedup: f64 = it.next()?.parse().ok()?;
        let speedup_threads: usize = it.next()?.parse().ok()?;
        let geom_gflops_cplx = it
            .next()
            .and_then(|t| t.parse().ok())
            .unwrap_or(geom_gflops / CPLX_RATE_DIV);
        let time_cv = it
            .next()
            .and_then(|t| t.parse().ok())
            .unwrap_or(FALLBACK_TIME_CV);
        let speedup4 = it.next().and_then(|t| t.parse().ok()).unwrap_or_else(|| {
            if speedup_threads > 1 {
                1.0 + (speedup - 1.0) * (SPEEDUP_ANCHOR_THREADS - 1) as f64
                    / (speedup_threads - 1) as f64
            } else {
                1.0
            }
        });
        Some(Calibration {
            geom_gflops,
            geom_gflops_cplx,
            speedup,
            speedup_threads,
            speedup4,
            time_cv,
            fingerprint: fp,
        })
    }

    fn save(&self) -> std::io::Result<()> {
        let p = Self::cache_path(self.fingerprint);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(
            p,
            format!(
                "{} {} {} {} {} {}",
                self.geom_gflops,
                self.speedup,
                self.speedup_threads,
                self.geom_gflops_cplx,
                self.time_cv,
                self.speedup4
            ),
        )
    }

    fn fallback(hw: &HardwareInfo) -> Self {
        let speedup = (hw.physical_cores as f64).sqrt().max(1.0);
        Calibration {
            geom_gflops: FALLBACK_GEOM_GFLOPS,
            geom_gflops_cplx: FALLBACK_GEOM_GFLOPS / CPLX_RATE_DIV,
            speedup,
            speedup_threads: hw.physical_cores,
            speedup4: speedup.min(FALLBACK_SPEEDUP_ANCHOR_MAX),
            time_cv: FALLBACK_TIME_CV,
            fingerprint: hw.fingerprint(),
        }
    }

    /// The one-thread proxy-flops/s rate for a scalar of `value_bytes` (real `f64`
    /// = 8, `Complex<f64>` = 16): picks the complex rate for the wider type.
    pub fn rate_for(&self, value_bytes: usize) -> f64 {
        if value_bytes >= CPLX_RATE_MIN_BYTES {
            self.geom_gflops_cplx
        } else {
            self.geom_gflops
        }
    }

    /// Measure throughput by factoring representative 3D grids: the serial
    /// proxy-flops/s rate + timing noise floor on a small grid (fast), and the
    /// parallel speedup curve (at 4 threads and at `physical_cores`) on a larger
    /// grid - the regime where parallelism actually matters. A small grid
    /// understates the achievable speedup badly (fronts too small to keep
    /// workers busy), which made the cost model pick far too few workers for
    /// production-size systems.
    pub fn measure(hw: &HardwareInfo) -> Self {
        let a = grid3d_spd::<f64>(CALIB_GRID_SMALL);
        let Ok(sym) = LdltSymbolic::analyze(&a, &SolverSettings::default()) else {
            return Self::fallback(hw);
        };
        let flops = sym.estimate_memory::<f64>().factor_flops as f64;
        let time_at = |t: usize| -> Option<f64> {
            let opts = SolverSettings::default().with_threads(t);
            let start = crate::clock::Instant::now();
            sym.factor(&a, &opts).ok()?;
            Some(start.elapsed().as_secs_f64())
        };
        let mut samples = Vec::new();
        for _ in 0..CALIB_REPEATS {
            match time_at(1) {
                Some(t) => samples.push(t),
                None => return Self::fallback(hw),
            }
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let time_cv = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
        let t1 = mean;
        let geom_gflops = (flops / t1.max(TIME_FLOOR)) / GIGA;

        let ab = grid3d_spd::<Complex<f64>>(CALIB_GRID_LARGE);
        let opts_nd =
            SolverSettings::default().with_ordering(crate::symbolic::OrderingMethod::MetisND);
        let Ok(symb) = LdltSymbolic::analyze(&ab, &opts_nd) else {
            return Self::fallback(hw);
        };
        let time_b = |t: usize| -> Option<f64> {
            let opts = opts_nd.clone().with_threads(t);
            let start = crate::clock::Instant::now();
            symb.factor(&ab, &opts).ok()?;
            Some(start.elapsed().as_secs_f64())
        };
        if time_b(hw.physical_cores.max(1)).is_none() {
            return Self::fallback(hw);
        }
        let (Some(t1b), Some(t4b), Some(tn)) = (
            time_b(1),
            time_b(SPEEDUP_ANCHOR_THREADS.min(hw.physical_cores.max(1))),
            time_b(hw.physical_cores.max(1)),
        ) else {
            return Self::fallback(hw);
        };
        let ac = grid3d_spd::<Complex<f64>>(CALIB_GRID_SMALL);
        let geom_gflops_cplx = match LdltSymbolic::analyze(&ac, &SolverSettings::default()) {
            Ok(symc) => {
                let fc = symc.estimate_memory::<Complex<f64>>().factor_flops as f64;
                let start = crate::clock::Instant::now();
                match symc.factor(&ac, &SolverSettings::default().with_threads(1)) {
                    Ok(_) => (fc / start.elapsed().as_secs_f64().max(TIME_FLOOR)) / GIGA,
                    Err(_) => geom_gflops / CPLX_RATE_DIV,
                }
            }
            Err(_) => geom_gflops / CPLX_RATE_DIV,
        };
        let speedup = (t1b / tn.max(TIME_FLOOR)).max(1.0);
        Calibration {
            geom_gflops,
            geom_gflops_cplx,
            speedup,
            speedup_threads: hw.physical_cores.max(1),
            speedup4: (t1b / t4b.max(TIME_FLOOR)).max(1.0),
            time_cv,
            fingerprint: hw.fingerprint(),
        }
    }

    /// Interpolated speedup at `threads`: piecewise linear through the measured
    /// points `(1, 1) -> (4, speedup4) -> (speedup_threads, speedup)`, flat beyond
    /// the calibrated peak (sparse-direct scaling saturates). The mid point keeps
    /// the curve honest at small counts, where a straight line toward the peak
    /// systematically misjudges the knee.
    pub fn speedup_for(&self, threads: usize) -> f64 {
        if threads <= 1 {
            1.0
        } else if threads >= self.speedup_threads {
            self.speedup
        } else if threads <= SPEEDUP_ANCHOR_THREADS {
            1.0 + (self.speedup4 - 1.0) * (threads - 1) as f64 / (SPEEDUP_ANCHOR_THREADS - 1) as f64
        } else {
            let span = (self.speedup_threads - SPEEDUP_ANCHOR_THREADS).max(1) as f64;
            self.speedup4
                + (self.speedup - self.speedup4) * (threads - SPEEDUP_ANCHOR_THREADS) as f64 / span
        }
    }
}

/// One-time **install diagnosis**: probe this machine and measure (or load) its
/// factorization calibration, caching it keyed by the hardware fingerprint.
///
/// Run once per machine (e.g. from an install script or `cargo xtask calibrate`);
/// afterwards [`Threads::Auto`](crate::Threads::Auto) picks its worker count
/// from the cached calibration via the cost model instead of the structural
/// predictor. The solvers themselves **never** measure implicitly - no
/// calibration cache means the hardware-agnostic default behaviour.
pub fn install_diagnose() -> Calibration {
    let hw = HardwareInfo::probe();
    Calibration::load_or_measure(&hw)
}

pub(crate) fn cached_calibration() -> Option<(usize, Calibration)> {
    static CACHE: std::sync::OnceLock<Option<(usize, Calibration)>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let hw = HardwareInfo::probe();
        Calibration::load(hw.fingerprint()).map(|c| (hw.physical_cores, c))
    })
}

/// Cost-model worker-count selection: the fewest workers that reach (within a
/// small margin) the minimum predicted time. Because
/// [`est_runtime_ms_threaded`](MemoryEstimate::est_runtime_ms_threaded) floors at
/// the critical path of the assembly tree, adding workers past the point where the
/// parallel term hits that floor buys nothing --- so a critical-path-bound matrix
/// (a banded chain, whose critical path is nearly its whole work) gets **fewer**
/// workers, and a wide 3D tree gets more. Deterministic given the calibration.
/// `max_threads == 0` means all physical cores.
pub fn recommend_threads_cost_model(
    estimate: &MemoryEstimate,
    calib: &Calibration,
    max_threads: usize,
    all_cores: usize,
) -> usize {
    let cap = if max_threads == 0 {
        all_cores.max(1)
    } else {
        max_threads
    };
    let rate = calib.rate_for(estimate.value_bytes);
    let residual_ln = |t: usize| -> f64 {
        if t <= 1 {
            return 0.0;
        }
        let flops = estimate.factor_flops.max(1) as f64;
        let crit = estimate.critical_path_flops.max(1) as f64;
        let amdahl = (crit / flops).min(1.0);
        let width = estimate.max_tree_width.max(1) as f64;
        let r = TIME_RESIDUAL[0]
            + TIME_RESIDUAL[1] * amdahl
            + TIME_RESIDUAL[2] * (t as f64).ln()
            + TIME_RESIDUAL[3] * width.ln();
        r.clamp(-TIME_RESIDUAL_CLAMP, TIME_RESIDUAL_CLAMP)
    };
    let floor_ms = estimate.est_runtime_ms_threaded(rate, f64::MAX);
    let time = |t: usize| {
        let corrected =
            estimate.est_runtime_ms_threaded(rate, calib.speedup_for(t)) * (-residual_ln(t)).exp();
        corrected.max(floor_ms)
    };
    let mut best_t = 1;
    let mut best = time(1);
    for t in THREAD_LADDER.into_iter().filter(|&t| t <= cap) {
        let tt = time(t);
        if tt < best * THREAD_GAIN_MARGIN {
            best = tt;
            best_t = t;
        }
    }
    best_t
}

fn grid3d_spd<T: Scalar>(k: usize) -> CscMatrix<T> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals: Vec<T> = Vec::new();
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                rows.push(p);
                cols.push(p);
                vals.push(T::from_real(CALIB_STENCIL_DIAGONAL));
                let mut nb = |q: usize| {
                    let (hi, lo) = if p >= q { (p, q) } else { (q, p) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(T::from_real(-1.0));
                };
                if x + 1 < k {
                    nb(idx(x + 1, y, z));
                }
                if y + 1 < k {
                    nb(idx(x, y + 1, z));
                }
                if z + 1 < k {
                    nb(idx(x, y, z + 1));
                }
            }
        }
    }
    match CscMatrix::from_triplets(n, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(_) => CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_takes_the_containers_memory() {
        let hw = HardwareInfo::probe();
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        eprintln!(
            "probe: total {} MB, available {} MB; host total {} MB; cgroup {:?}",
            hw.total_ram_bytes >> 20,
            hw.available_ram_bytes >> 20,
            sys.total_memory() >> 20,
            sys.cgroup_limits().map(|c| c.total_memory >> 20)
        );
        assert!(hw.total_ram_bytes <= sys.total_memory());
        assert!(hw.available_ram_bytes <= hw.total_ram_bytes);
        if let Some(cg) = sys.cgroup_limits() {
            assert!(hw.total_ram_bytes <= cg.total_memory);
            assert!(hw.available_ram_bytes <= cg.free_memory);
        }
    }

    #[test]
    fn cost_model_thread_count_respects_critical_path() {
        let calib = Calibration {
            geom_gflops: 2.0,
            geom_gflops_cplx: 0.7,
            speedup: 6.0,
            speedup_threads: 12,
            speedup4: 2.8,
            time_cv: 0.1,
            fingerprint: 0,
        };
        let mut est = crate::diagnostics::MemoryEstimate {
            value_bytes: 8,
            factor_nnz: 0,
            factor_bytes: 0,
            panels_all_bytes: 0,
            panel_live_peak_bytes: 0,
            transient_peak_bytes: 0,
            factor_flops: 100_000_000_000,
            critical_path_flops: 0,
            max_tree_width: 256,
        };
        est.critical_path_flops = 1_000_000_000;
        let wide = recommend_threads_cost_model(&est, &calib, 24, 24);
        est.critical_path_flops = est.factor_flops;
        let chain = recommend_threads_cost_model(&est, &calib, 24, 24);
        assert!(
            wide > chain,
            "wide tree uses more workers ({wide}) than a chain ({chain})"
        );
        assert_eq!(
            chain, 1,
            "a critical-path-bound matrix uses a single worker"
        );
        assert!(wide >= 8, "a wide tree uses many workers ({wide})");
    }
}
