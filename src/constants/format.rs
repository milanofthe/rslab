//! Units and report formatting. None of these touch the numerics.

/// Milliseconds per second (1): durations are reported in ms as
/// `secs * MS_PER_S`.
/// Used by: `MemoryEstimate::est_runtime_ms`.
/// Why: the unit.
/// Contract: format.
pub const MS_PER_S: f64 = 1e3;

/// The SI prefix mega (1): megabytes, millions of unknowns or nonzeros per
/// second.
/// Used by: `diagnostics`, `memory::MemoryPlan`'s display.
/// Why: the unit (decimal megabytes, as the reports have always used).
/// Contract: format.
pub const MEGA: f64 = 1e6;

/// The SI prefix giga (1): Gflop/s.
/// Used by: `diagnostics`, `tuning`.
/// Why: the unit.
/// Contract: format.
pub const GIGA: f64 = 1e9;

/// Floor of a rate or a speedup in the runtime estimates, so an unmeasured
/// zero never divides (Gflop/s, 1).
/// Used by: `MemoryEstimate::est_runtime_ms`,
/// `MemoryEstimate::est_runtime_ms_threaded`.
/// Why: far below any real rate.
/// Contract: estimate.
pub const RATE_FLOOR: f64 = 1e-6;

/// Floor of a measured duration before it divides (s).
/// Used by: `tuning::Calibration::measure` (s).
/// Why: far below any real duration, far above zero.
/// Contract: estimate.
pub const TIME_FLOOR: f64 = 1e-9;

/// Words of the Matrix Market banner
/// `%%MatrixMarket matrix coordinate <field> <symmetry>` (words).
/// Used by: `io::mtx::read_mtx_any`.
/// Why: the format.
/// Contract: format.
pub const MTX_BANNER_WORDS: usize = 5;

/// Fields of the Matrix Market size line `rows cols entries` (fields).
/// Used by: `io::mtx::parse_mtx_with`.
/// Why: the format.
/// Contract: format.
pub const MTX_SIZE_FIELDS: usize = 3;
