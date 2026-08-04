//! Monotonic clock for the profiling instrumentation, portable to targets
//! without an OS clock (wasm32-unknown-unknown: `std::time::Instant::now()`
//! panics with "time not implemented on this platform", which trapped every
//! KLU factorization in the browser). Native builds re-export
//! [`std::time::Instant`] unchanged, so instrumentation cost and behavior
//! stay exactly as before. On wasm32 the instant is inert: every duration
//! reads zero, so the `RLA_*_PROF` diagnostics (whose env-var gates are
//! absent in the browser anyway) become no-ops instead of trapping.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
impl Instant {
    pub fn now() -> Self {
        Instant
    }
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
    pub fn duration_since(&self, _earlier: Instant) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
