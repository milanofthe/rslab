//! Hardware-aware auto-tuning + resource governor (feature `tuning`).
//!
//! [`HardwareInfo::probe`] detects cores + RAM; [`Calibration::load_or_measure`]
//! measures this machine's factorization throughput once (by factoring a
//! representative grid) and caches it (FFTW-"wisdom" style, keyed by a hardware
//! fingerprint); [`plan`] then turns a [`MemoryEstimate`] + a [`Budget`] into a
//! concrete [`FactorPlan`] — picking the thread count, predicting peak memory and
//! wall-clock, and (when over budget) selecting approximations (mixed precision /
//! incomplete factor / BLR) or recommending fail-fast. Deterministic given a fixed
//! calibration, so solver-in-the-loop scheduling is reproducible.

use std::path::PathBuf;

use crate::diagnostics::MemoryEstimate;
use crate::sparse::csc::CscMatrix;
use crate::{BlrMode, FactorOptions, LdltSymbolic};

/// Detected machine capabilities — for budgeting and the calibration key.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
}

impl HardwareInfo {
    /// Probe the current machine (cores via std, RAM via sysinfo).
    pub fn probe() -> Self {
        let logical = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let physical = sys.physical_core_count().unwrap_or(logical);
        HardwareInfo {
            logical_cores: logical,
            physical_cores: physical.max(1),
            total_ram_bytes: sys.total_memory(),
            available_ram_bytes: sys.available_memory(),
        }
    }
    /// Stable key for the calibration cache (cores + RAM size bucket).
    pub fn fingerprint(&self) -> u64 {
        let gb = self.total_ram_bytes / (1 << 30);
        ((self.physical_cores as u64) << 40)
            ^ ((self.logical_cores as u64) << 20)
            ^ gb.min(0xFFFFF)
    }
}

/// Measured cost-model constants for this machine — the cached "wisdom".
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// One-thread throughput in the `factor_flops` proxy unit, ×1e9 (giga/s).
    pub geom_gflops: f64,
    /// Parallel speedup measured at `speedup_threads`.
    pub speedup: f64,
    pub speedup_threads: usize,
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
        Some(Calibration {
            geom_gflops: it.next()?.parse().ok()?,
            speedup: it.next()?.parse().ok()?,
            speedup_threads: it.next()?.parse().ok()?,
            fingerprint: fp,
        })
    }

    fn save(&self) -> std::io::Result<()> {
        let p = Self::cache_path(self.fingerprint);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(p, format!("{} {} {}", self.geom_gflops, self.speedup, self.speedup_threads))
    }

    /// A reasonable default if calibration cannot run (e.g. analyze fails).
    fn fallback(hw: &HardwareInfo) -> Self {
        Calibration {
            geom_gflops: 2.0,
            speedup: (hw.physical_cores as f64).sqrt().max(1.0),
            speedup_threads: hw.physical_cores,
            fingerprint: hw.fingerprint(),
        }
    }

    /// Measure throughput by factoring a representative 3D grid at 1 thread and at
    /// `physical_cores`, recording the proxy-flops/s rate and the parallel speedup.
    pub fn measure(hw: &HardwareInfo) -> Self {
        let a = grid3d_spd(24); // ≈ 13 800 DOFs, a few hundred ms
        let Ok(sym) = LdltSymbolic::analyze(&a) else {
            return Self::fallback(hw);
        };
        let flops = sym.estimate_memory::<f64>().factor_flops as f64;
        let time_at = |t: usize| -> Option<f64> {
            let opts = FactorOptions::default().with_threads(t);
            let start = std::time::Instant::now();
            sym.factor(&a, &opts).ok()?;
            Some(start.elapsed().as_secs_f64())
        };
        let (Some(t1), Some(tn)) = (time_at(1), time_at(hw.physical_cores.max(1))) else {
            return Self::fallback(hw);
        };
        Calibration {
            geom_gflops: (flops / t1.max(1e-9)) / 1e9,
            speedup: (t1 / tn.max(1e-9)).max(1.0),
            speedup_threads: hw.physical_cores.max(1),
            fingerprint: hw.fingerprint(),
        }
    }

    /// Interpolated speedup at `threads`: linear toward the calibrated peak, flat
    /// beyond it (sparse-direct scaling saturates).
    pub fn speedup_for(&self, threads: usize) -> f64 {
        if threads <= 1 {
            1.0
        } else if threads >= self.speedup_threads {
            self.speedup
        } else {
            1.0 + (self.speedup - 1.0) * (threads - 1) as f64
                / (self.speedup_threads - 1).max(1) as f64
        }
    }
}

/// Resource budget the factorization must live within.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Hard memory ceiling (bytes). `None` = no limit.
    pub max_mem_bytes: Option<u64>,
    /// Thread budget. `0` = the machine's physical cores.
    pub max_threads: usize,
    /// When over the memory budget, may the planner factor in single precision
    /// (`Complex<f32>`/`f32`) and recover accuracy by refinement? Halves the factor.
    pub allow_mixed_precision: bool,
    /// When over budget, may it drop small fill (incomplete factor)? `Some(tau)`.
    pub allow_drop_tol: Option<f64>,
    /// When over budget, may it BLR-compress the big fronts?
    pub allow_blr: bool,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            max_mem_bytes: None,
            max_threads: 0,
            allow_mixed_precision: false,
            allow_drop_tol: None,
            allow_blr: false,
        }
    }
}

/// A concrete factorization plan: tuned options + predictions + decisions.
#[derive(Debug, Clone)]
pub struct FactorPlan {
    /// Tuned options (thread count, plus drop_tol / BLR if chosen to fit budget).
    pub opts: FactorOptions,
    /// Recommendation to factor in single precision (the caller casts the matrix);
    /// not expressible in `opts` since it is a matrix-type choice.
    pub use_mixed_precision: bool,
    /// Predicted transient peak after the chosen approximations (bytes).
    pub est_peak_bytes: u64,
    /// Predicted factor wall-clock at the chosen thread count (ms).
    pub est_runtime_ms: f64,
    /// Whether the prediction fits the memory budget.
    pub fits: bool,
    /// Human-readable summary of the decisions taken.
    pub note: String,
}

/// Turn an a-priori [`MemoryEstimate`] + a [`Budget`] into a concrete plan, using
/// the machine's [`HardwareInfo`] and [`Calibration`]. Path-agnostic: pass the
/// estimate from `LuSymbolic`/`LdltSymbolic::estimate_memory`. Pure given the
/// inputs (deterministic scheduling).
pub fn plan(
    estimate: &MemoryEstimate,
    budget: &Budget,
    hw: &HardwareInfo,
    calib: &Calibration,
) -> FactorPlan {
    let threads = if budget.max_threads == 0 { hw.physical_cores.max(1) } else { budget.max_threads };
    let speedup = calib.speedup_for(threads);
    let runtime = estimate.est_runtime_ms(calib.geom_gflops, speedup);
    let mut opts = FactorOptions::default().with_threads(threads);
    let mut peak = estimate.transient_peak_bytes;
    let mut use_f32 = false;
    let mut notes: Vec<String> = Vec::new();

    if let Some(maxm) = budget.max_mem_bytes {
        // Apply approximations in increasing aggressiveness until it fits. The
        // reduction factors are conservative rules of thumb (documented as such).
        if peak > maxm && budget.allow_mixed_precision && estimate.value_bytes > 4 {
            use_f32 = true;
            peak /= 2; // single precision ≈ halves the factor + panels
            notes.push("mixed-precision (factor in single)".into());
        }
        if peak > maxm {
            if let Some(tau) = budget.allow_drop_tol {
                opts = opts.with_drop_tol(tau);
                peak = (peak as f64 * 0.7) as u64; // incomplete factor ≈ −30%
                notes.push(format!("incomplete factor (drop_tol={tau:.0e})"));
            }
        }
        if peak > maxm && budget.allow_blr {
            opts = opts.with_blr(BlrMode::contribution_blocks(1e-4));
            peak = (peak as f64 * 0.7) as u64; // BLR ≈ −30% on big fronts
            notes.push("BLR compression".into());
        }
        if peak > maxm {
            notes.push(format!(
                "STILL over budget ({:.0}>{:.0} MB) — recommend fail-fast",
                peak as f64 / 1e6,
                maxm as f64 / 1e6
            ));
        }
    }

    let fits = budget.max_mem_bytes.map_or(true, |m| peak <= m);
    if notes.is_empty() {
        notes.push(format!("exact, {threads} threads"));
    }
    FactorPlan {
        opts,
        use_mixed_precision: use_f32,
        est_peak_bytes: peak,
        est_runtime_ms: runtime,
        fits,
        note: notes.join("; "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_calibrate_plan_governor() {
        let hw = HardwareInfo::probe();
        assert!(hw.logical_cores >= 1 && hw.physical_cores >= 1);
        assert!(hw.total_ram_bytes > 0, "RAM probed");

        let calib = Calibration::measure(&hw);
        assert!(calib.geom_gflops > 0.0 && calib.speedup >= 1.0);
        assert_eq!(calib.speedup_for(1), 1.0);
        assert!(calib.speedup_for(hw.physical_cores) >= 1.0);
        // cache round-trip
        calib.save().unwrap();
        assert!(Calibration::load(hw.fingerprint()).is_some());

        let a = grid3d_spd(16);
        let est = LdltSymbolic::analyze(&a).unwrap().estimate_memory::<f64>();
        assert!(est.factor_flops > 0);

        // Generous budget → exact, fits, predicted runtime positive.
        let plan_ok = plan(&est, &Budget::default(), &hw, &calib);
        assert!(plan_ok.fits && !plan_ok.use_mixed_precision);
        assert!(plan_ok.est_runtime_ms > 0.0);
        assert_eq!(plan_ok.opts.threads, hw.physical_cores.max(1));

        // Tight budget with all approximations allowed → planner applies them.
        let tight = Budget {
            max_mem_bytes: Some(est.transient_peak_bytes / 8),
            max_threads: 0,
            allow_mixed_precision: true,
            allow_drop_tol: Some(1e-3),
            allow_blr: true,
        };
        let plan_tight = plan(&est, &tight, &hw, &calib);
        assert!(
            plan_tight.use_mixed_precision || plan_tight.opts.drop_tol.is_some(),
            "an approximation was selected"
        );
        assert!(!plan_tight.note.is_empty());
        assert!(plan_tight.est_peak_bytes < est.transient_peak_bytes, "approximations shrink the peak");
    }
}

/// 3D 7-point Laplacian (k³ grid, Dirichlet, SPD `f64`, lower triangle) — the
/// calibration's representative matrix.
fn grid3d_spd(k: usize) -> CscMatrix<f64> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                rows.push(p);
                cols.push(p);
                vals.push(6.0);
                let mut nb = |q: usize| {
                    let (hi, lo) = if p >= q { (p, q) } else { (q, p) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(-1.0);
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
        Err(_) => CscMatrix::from_triplets(1, &[0], &[0], &[1.0]).unwrap_or(CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        }),
    }
}
