//! The hardware calibration and its cost model (feature `tuning`): what
//! `install_diagnose` measures, and how `Threads::Auto` turns it into a
//! worker count. They change the worker count only, never the bits.

/// Bytes per unit of the RAM size in the calibration cache key (bytes).
/// Used by: `tuning::HardwareInfo::fingerprint`.
/// Why: a GiB bucket keeps the key stable across small reported changes.
/// Contract: format (the cache key; changing it orphans existing caches).
pub const FINGERPRINT_RAM_UNIT: u64 = 1 << 30;

/// Bit offsets of the physical and logical core counts in the cache key,
/// and the mask of its RAM field (bits).
/// Used by: `tuning::HardwareInfo::fingerprint`.
/// Why: three disjoint fields of one `u64`.
/// Contract: format (the cache key).
pub const FINGERPRINT_PHYSICAL_SHIFT: u32 = 40;

/// See [`FINGERPRINT_PHYSICAL_SHIFT`] (bits).
pub const FINGERPRINT_LOGICAL_SHIFT: u32 = 20;

/// See [`FINGERPRINT_PHYSICAL_SHIFT`] (GiB, 20 bits).
pub const FINGERPRINT_RAM_MASK: u64 = 0xFFFFF;

/// The complex rate as this fraction of the real one where it was not
/// measured: an older cache, or a failed complex run (1, a divisor).
/// Used by: `tuning::Calibration::load`, `Calibration::measure`,
/// `Calibration::fallback`.
/// Why: a conservative fallback; the calibration measures the real ratio.
/// Contract: timing.
pub const CPLX_RATE_DIV: f64 = 3.0;

/// Timing noise floor (coefficient of variation) assumed where it was not
/// measured (1).
/// Used by: `tuning::Calibration::load`, `Calibration::fallback`.
/// Why: a conservative fallback; the calibration measures it.
/// Contract: timing.
pub const FALLBACK_TIME_CV: f64 = 0.1;

/// The speedup curve's middle anchor (threads): the calibration measures
/// the speedup at this count besides the peak, and the curve interpolates
/// through it.
/// Used by: `tuning::Calibration::measure`, `Calibration::load`,
/// `Calibration::speedup_for`.
/// Why: the default worker cap, where the knee of the curve sits.
/// Contract: timing.
pub const SPEEDUP_ANCHOR_THREADS: usize = 4;

/// Scalars of at least this many bytes time at the complex rate (bytes).
/// Used by: `tuning::Calibration::rate_for`.
/// Why: `Complex<f64>` is 16 bytes; `Complex<f32>` (8) runs nearer the real
/// rate.
/// Contract: timing.
pub const CPLX_RATE_MIN_BYTES: usize = 16;

/// Edge of the small 3D calibration grid, about 13 800 unknowns (points).
/// Used by: `tuning::Calibration::measure` (one-thread rates).
/// Why: a few hundred ms per factorization.
/// Contract: timing.
pub const CALIB_GRID_SMALL: usize = 24;

/// Edge of the large complex calibration grid, about 33 000 unknowns
/// (points).
/// Used by: `tuning::Calibration::measure` (speedup curve).
/// Why: a small grid understates the achievable speedup badly (fronts too
/// small to keep workers busy).
/// Contract: timing.
pub const CALIB_GRID_LARGE: usize = 32;

/// One-thread factorizations timed for the noise floor (runs).
/// Used by: `tuning::Calibration::measure`.
/// Why: enough for a coefficient of variation, short enough to install.
/// Contract: timing.
pub const CALIB_REPEATS: usize = 4;

/// Learned additive-residual weights `[intercept, amdahl_frac,
/// ln(threads), ln(tree_width)]` correcting `log(measured / analytical
/// speedup)` (1).
/// Used by: `tuning::recommend_threads_cost_model`.
/// Why: fit offline by `benches/fit_residual.py` on the generated
/// thread-scaling corpus (54 matrices), a 26 % held-out RMSE reduction
/// under leave-one-matrix-out CV; dimensionless and hardware-agnostic.
/// Contract: timing.
pub const TIME_RESIDUAL: [f64; 4] = [-0.4621, 0.7550, -0.0791, 0.0358];

/// Clamp of that residual, so it refines the calibrated base and never
/// swings the choice wildly: `exp` of it stays in `[0.50, 2.01]` (1).
/// Used by: `tuning::recommend_threads_cost_model`.
/// Why: the fit's range of trust.
/// Contract: timing.
pub const TIME_RESIDUAL_CLAMP: f64 = 0.7;

/// The worker counts the cost model evaluates besides one (workers).
/// Used by: `tuning::recommend_threads_cost_model`.
/// Why: dense at small counts, where the speedup curve bends, coarse above.
/// Contract: timing.
pub const THREAD_LADDER: [usize; 9] = [2, 4, 6, 8, 12, 16, 20, 24, 32];

/// More workers are taken only when they predict at most this share of
/// the best time so far (1).
/// Used by: `tuning::recommend_threads_cost_model`.
/// Why: a 3 % gain is worth the extra workers, less is noise.
/// Contract: timing.
pub const THREAD_GAIN_MARGIN: f64 = 0.97;
