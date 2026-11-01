//! The precision values are stored in ([`Stored`](crate::Stored),
//! [`single_for`](crate::single_for)).

/// How far below the tolerance data is accurate to the rounding of storing it in
/// the single-precision twin must stay (1): single where the twin's unit
/// roundoff (`f32::EPSILON / 2`, 6e-8) times this is at most the tolerance, in
/// the field itself below. The rounding then adds at most this share of the
/// tolerance to the error the data holds; measured on a dense MoM far-field
/// block compressed by ACA, the stored block's relative error stops at 3.4e-8 in
/// single precision. From `9.5e-7` on, so an ACA tolerance of `1e-6` stores single.
/// Used by: `single_for`.
/// Why: derived: the rounding one sixteenth of the tolerance at most.
/// Contract: results (the precision data is stored in).
pub const SINGLE_MARGIN: f64 = 16.0;
