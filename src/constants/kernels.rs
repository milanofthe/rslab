//! The dense kernels of the supernodal LDL^T and LU factorizations.

/// Scale-invariant singularity floor for a 2x2 Bunch-Kaufman pivot (1): a
/// block whose `|det|` falls below `GROWTH_EPS * scale^2` (scale = the
/// largest block entry magnitude) is numerically singular, rejected in exact
/// mode and lifted in static-pivot mode. Bounds the element growth `1/|det|`
/// can otherwise inject into the trailing update.
/// Used by: `ldlt::bunch_kaufman` (2x2 pivot test and static-pivot lift).
/// Why: about 45 units of double roundoff.
/// Contract: results.
pub const GROWTH_EPS: f64 = 1e-14;
