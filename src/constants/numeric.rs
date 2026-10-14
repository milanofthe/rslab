//! Tolerances of the small dense routines behind GMRES recycling (GCRO-DR):
//! the dense LU, the eigenvalue iteration and the harmonic-Ritz extraction.
//! They only shape the recycle subspace, which accelerates convergence; the
//! outer GMRES residual test stays authoritative.

/// A positive floor for magnitudes that are divided by or compared
/// relatively, so a zero never divides (1).
/// Used by: `dense_eig::eigenvalues` (deflation test),
/// `krylov::recycle::combine_ritz_real` (Ritz value magnitude).
/// Why: far below any meaningful magnitude, far above the subnormals.
/// Contract: results (only where a magnitude would be zero).
pub const TINY: f64 = 1e-300;

/// A pivot of the dense LU below this times `||A||inf` (at least 1) counts
/// as singular (1).
/// Used by: `dense_eig::lu_solve`.
/// Why: about 45 units of double roundoff; the caller then skips the
/// recycle update instead of dividing by about 0.
/// Contract: results (whether a recycle update happens).
pub const DENSE_SINGULAR_TOL: f64 = 1e-14;

/// QR iterations the eigenvalue routine may spend per dimension, and once
/// more on top: the budget is `QR_ITER_PER_DIM * d + QR_ITER_PER_DIM`
/// (iterations).
/// Used by: `dense_eig::eigenvalues`.
/// Why: LAPACK's 30 iterations per eigenvalue.
/// Contract: results (only where the QR iteration does not converge).
pub const QR_ITER_PER_DIM: usize = 30;

/// Relative perturbation of the shift of the inverse iteration, so the
/// shifted matrix is not exactly singular at a converged eigenvalue (1, of
/// the largest entry magnitude, at least 1).
/// Used by: `dense_eig::inverse_iteration`.
/// Why: well above roundoff, well below the eigenvalue separation that
/// matters for the recycle space.
/// Contract: results.
pub const INVERSE_ITERATION_SHIFT: f64 = 1e-10;

/// A harmonic-Ritz value with `|Im| <= RITZ_REAL_TOL * |theta|` counts as
/// real and contributes one real vector (1).
/// Used by: `krylov::recycle::combine_ritz_real`.
/// Why: tuned default, well above the eigenvalue routine's roundoff.
/// Contract: results.
pub const RITZ_REAL_TOL: f64 = 1e-8;

/// A recycle direction whose norm falls below this fraction of its norm
/// before orthogonalization is rank-deficient and dropped (1).
/// Used by: `krylov::recycle::orthonormalize_columns`,
/// `krylov::recycle::orthonormalize_recycle`.
/// Why: about 4500 units of double roundoff: what is left is cancellation.
/// Contract: results.
pub const RECYCLE_RANK_TOL: f64 = 1e-12;
