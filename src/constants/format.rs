//! Units, report formatting and the log levels. None of these touch the
//! numerics.

/// Milliseconds per second (1): durations are reported in ms as
/// `secs * MS_PER_S`.
/// Used by: `clock::ms` (every stage time), `diagnostics`.
/// Why: the unit.
/// Contract: format.
pub const MS_PER_S: f64 = 1e3;

/// Seconds per millisecond (1): rates divide by `ms * S_PER_MS`.
/// Used by: `Diagnostics::rates`.
/// Why: the unit; kept as a product, not a division by [`MS_PER_S`], so the
/// rates keep their bits.
/// Contract: format.
pub const S_PER_MS: f64 = 1e-3;

/// The SI prefix mega (1): megabytes, millions of unknowns or nonzeros per
/// second.
/// Used by: `diagnostics`, `memory::MemoryPlan`'s display.
/// Why: the unit (decimal megabytes, as the reports have always used).
/// Contract: format.
pub const MEGA: f64 = 1e6;

/// The SI prefix mega as an integer, for integer flop counts (1).
/// Used by: `Diagnostics`'s display (Mflop).
/// Why: the unit.
/// Contract: format.
pub const MEGA_U64: u64 = 1_000_000;

/// The SI prefix giga (1): Gflop/s.
/// Used by: `diagnostics`, `tuning`.
/// Why: the unit.
/// Contract: format.
pub const GIGA: f64 = 1e9;

/// Percent per unit share (1).
/// Used by: `Diagnostics`'s display.
/// Why: the unit.
/// Contract: format.
pub const PERCENT: f64 = 100.0;

/// Floor of a rate or a speedup in the runtime estimates, so an unmeasured
/// zero never divides (Gflop/s, 1).
/// Used by: `MemoryEstimate::est_runtime_ms`,
/// `MemoryEstimate::est_runtime_ms_threaded`.
/// Why: far below any real rate.
/// Contract: estimate.
pub const RATE_FLOOR: f64 = 1e-6;

/// Floor of a measured duration before it divides (s or ms).
/// Used by: `Diagnostics`'s display (ms), `tuning::Calibration::measure`
/// (s).
/// Why: far below any real duration, far above zero.
/// Contract: estimate.
pub const TIME_FLOOR: f64 = 1e-9;

/// Seconds per day, hour and minute of the default log sink's time of day
/// (s).
/// Used by: `logging::timestamp`.
/// Why: the units.
/// Contract: format.
pub const SECS_PER_DAY: u64 = 86400;

/// See [`SECS_PER_DAY`] (s).
pub const SECS_PER_HOUR: u64 = 3600;

/// See [`SECS_PER_DAY`] (s).
pub const SECS_PER_MIN: u64 = 60;

/// Numeric log levels, those of the Python `logging` module, so the
/// bindings map them one to one (level).
/// Used by: `LogLevel`.
/// Why: Python's `DEBUG`, `INFO`, `WARNING`, `ERROR`; `OFF` above all.
/// Contract: format (the level values are API).
pub const LOG_DEBUG: u8 = 10;

/// See [`LOG_DEBUG`].
pub const LOG_INFO: u8 = 20;

/// See [`LOG_DEBUG`].
pub const LOG_WARNING: u8 = 30;

/// See [`LOG_DEBUG`].
pub const LOG_ERROR: u8 = 40;

/// See [`LOG_DEBUG`].
pub const LOG_OFF: u8 = 100;

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
