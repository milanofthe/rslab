//! Monotonic clock for the profiling instrumentation, portable to targets
//! without an OS clock (wasm32-unknown-unknown: `std::time::Instant::now()`
//! panics with "time not implemented on this platform"). Native builds
//! re-export [`std::time::Instant`] unchanged. On wasm32 the instant is
//! inert: every duration reads zero, so stage timings read zero instead of
//! trapping.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

/// A duration in milliseconds, as every stage time is reported.
pub(crate) fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * crate::constants::MS_PER_S
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant
    }
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
