//! Tolerances and budgets of the small dense routines (`dense_eig`): the
//! dense LU, the Schur form and eigenvalue iteration behind [`crate::dense`]
//! and GMRES recycling (GCRO-DR), and the harmonic-Ritz extraction. For
//! GCRO-DR they only shape the recycle subspace, which accelerates
//! convergence, and the outer GMRES residual test stays authoritative; for
//! [`crate::dense`] a QR that does not converge within the budget is an error.

/// A positive floor for magnitudes that are divided by or compared
/// relatively, so a zero never divides (1).
/// Used by: `dense_eig::qr_iterate` (deflation test), `dense_eig::Schur`'s
/// eigenvectors (the floor of a back substitution's denominator),
/// `krylov::recycle::combine_ritz_real` (Ritz value magnitude).
/// Why: far below any meaningful magnitude, far above the subnormals.
/// Contract: results (only where a magnitude would be zero).
pub const TINY: f64 = 1e-300;

/// A pivot of the dense LU below this times `||A||inf` (at least 1) counts
/// as singular (1).
/// Used by: `dense_eig::lu`, `dense::lu_solve`, `dense::Schur::solve_shifted`.
/// Why: about 45 units of double roundoff; GCRO-DR then skips the recycle
/// update, `crate::dense` reports the matrix singular, instead of dividing by
/// about 0.
/// Contract: results (whether a recycle update happens, where a dense solve
/// fails).
pub const DENSE_SINGULAR_TOL: f64 = 1e-14;

/// QR iterations the eigenvalue routine may spend per dimension, and once
/// more on top: the budget is `QR_ITER_PER_DIM * d + QR_ITER_PER_DIM`
/// (iterations).
/// Used by: `dense_eig::qr_iterate`: `eigenvalues` (best effort beyond it),
/// `dense::Schur::new` (`NotConverged` beyond it).
/// Why: LAPACK's 30 iterations per eigenvalue.
/// Contract: results (only where the QR iteration does not converge).
pub const QR_ITER_PER_DIM: usize = 30;

/// QR iterations without a deflation after which an exceptional shift
/// replaces the Wilkinson shift, and every as many again (iterations): it
/// breaks the cycles a fixed shift strategy can fall into (a cyclic
/// permutation's QR does not move).
/// Used by: `dense_eig::qr_iterate`.
/// Why: LAPACK's `zlahqr` (`KEXSH = 10`).
/// Contract: results (only where 10 steps pass without a deflation).
pub const QR_EXCEPTIONAL_PERIOD: usize = 10;

/// The exceptional shift: the corner entry plus this times the magnitude of
/// the real part of the subdiagonal next to it, the top corner at the first
/// exceptional step and the bottom one at the next, alternating (1).
/// Used by: `dense_eig::qr_iterate`.
/// Why: LAPACK's `zlahqr` (`DAT1 = 3/4`).
/// Contract: results (only where exceptional shifts are taken).
pub const QR_EXCEPTIONAL_SHIFT: f64 = 0.75;

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
