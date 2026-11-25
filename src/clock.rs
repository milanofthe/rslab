//! Monotonic clock of the measured dispatch decisions (the calibration of
//! `tuning`), portable to targets without an
//! OS clock (wasm32-unknown-unknown: `std::time::Instant::now()` panics with
//! "time not implemented on this platform"). Native builds re-export
//! [`std::time::Instant`] unchanged. On wasm32 the instant is inert: every
//! duration reads zero instead of trapping.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;

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
