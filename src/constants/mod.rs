//! Every number the solver is tuned, bounded or defined by: thresholds,
//! budgets and size limits, tolerances, and the defaults of the settings
//! structs. One place per crate (`rslab_metis::constants` and
//! `rslab_ordering_core::constants` hold the ordering crates' own), so a
//! value is read, compared and changed here and nowhere else;
//! `tests/constants_guard.rs` fails on a numeric literal elsewhere.
//!
//! Each constant states its unit, what reads it, why it has its value, and
//! its contract:
//!
//! - **results**: changing it changes the computed bits (a different factor,
//!   ordering or iterate); `tests/contract_snapshot.rs` pins the value.
//! - **rounding**: changes the rounding of the result, not its accuracy.
//! - **timing**: scheduling or memory only, the result is bit-identical.
//! - **estimate**: feeds a prediction or a report, never the numerics.
//! - **format**: a unit, a file format or an API value.
//!
//! The submodules are re-exported flat, so a value is
//! `crate::constants::NAME` wherever in this tree it is defined.
//!
//! Published algorithm coefficients (the Bunch-Kaufman `alpha`, the FNV-1a
//! hash) stay next to their algorithm; the guard lists them as exemptions.

mod format;
mod kernels;
mod klu;
mod krylov;
mod memory;
mod numeric;
mod ordering;
mod parallel;
mod pivoting;
mod scaling;
mod solve;
mod tuning;

pub use format::*;
pub use kernels::*;
pub use klu::*;
pub use krylov::*;
pub use memory::*;
pub use numeric::*;
pub use ordering::*;
pub use parallel::*;
pub use pivoting::*;
pub use scaling::*;
pub use solve::*;
pub use tuning::*;
