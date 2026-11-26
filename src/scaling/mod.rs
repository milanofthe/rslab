//! Matrix scaling: symmetric equilibration for the LDL^T path and MC64
//! weighted matching for the LU path.
//!
//! ## Symmetric equilibration (LDL^T)
//!
//! [`compute_scaling`] turns a [`ScalingStrategy`] into a real vector `s`
//! such that the factored matrix is the congruence `A_hat = D A D`,
//! `D = diag(s)`:
//!
//! - `OnePassInfNorm` (the `SolverSettings` default): a single
//!   Knight-Ruiz step `s_i = 1/sqrt(max_j |A_ij|)`; tolerates a zero
//!   diagonal.
//! - `InfNorm`: iterative Knight-Ruiz inf-norm equilibration.
//! - `Mc64Symmetric`: matching-based scaling after Duff & Koster (2001)
//!   and Duff & Pralet (2005), computed with a pure-Rust Hungarian
//!   algorithm, so that the largest entries of `D A D` lie on the
//!   diagonal.
//! - `Identity`, and `External` (a caller-supplied vector).
//!
//! The vector is in user-order indexing (the numbering of the input CSC).
//! `LdltSolver` computes it at factor time (through this module on the
//! `|A|` magnitude pattern, except for the one-pass default and
//! `Identity`, which it handles natively), multiplies each entry
//! `a[i,j]` by `s[i] * s[j]` while permuting the values for the
//! factorization, and fuses the scaling into the permutation
//! gather/scatter of the solve:
//! `x = D * (A_hat^-1 * (D b))`. The same vector is applied on both ends,
//! not its inverse. The sparsity pattern is unaffected.
//!
//! ## Two-sided equilibration (LU)
//!
//! `twosided::equilibrate_general` turns the same strategy into row and
//! column scalings `A_hat = D_r A D_c` of a general matrix: `OnePassInfNorm`
//! (the default) one Knight-Ruiz step, `InfNorm` the step iterated (Ruiz
//! 2001) until every row and column maximum is one, `Identity` none.
//!
//! ## Unsymmetric matching (LU)
//!
//! `mc64::compute_matching_general` computes a maximum-product
//! transversal of a general square matrix (Hungarian kernel in
//! `hungarian.rs`), and `mc64::unsymmetric_scaling` turns its dual
//! variables into row and column scalings. The LU path uses the matching
//! as a row permutation that puts large entries on the diagonal, together
//! with those scalings; the BTF analysis uses it as the transversal.

use crate::error::RslabError;
use crate::sparse::csc::CscMatrix;

mod hungarian;
mod infnorm;
pub(crate) mod mc64;
mod twosided;

pub(crate) use twosided::equilibrate_general;

// https://github.com/milanofthe/rslab/issues/291
#[inline]
pub(crate) fn kr_guarded_update(d: f64, m: f64) -> f64 {
    let cand = d / m.sqrt();
    if cand.is_finite() && cand > 0.0 {
        cand
    } else {
        d
    }
}

#[inline]
pub(crate) fn inv_sqrt_scale_guarded(m: f64) -> f64 {
    if m > 0.0 {
        let cand = 1.0 / m.sqrt();
        if cand.is_finite() && cand > 0.0 {
            cand
        } else {
            1.0
        }
    } else {
        1.0
    }
}

/// The symmetric equilibration of the LDL^T path. `OnePassInfNorm` (the
/// default) is the cheapest; `InfNorm` iterates it to convergence;
/// `Mc64Symmetric` scales by a maximum-product matching, at its extra
/// cost, which helps matrices whose large entries sit off the diagonal
/// (saddle-point and KKT systems).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalingStrategy {
    /// Knight-Ruiz inf-norm iterative equilibration (the "iterative Ruiz"
    /// arm of the equilibration knob). See `infnorm::compute_infnorm`; on the
    /// LU path the two-sided form, every row and column maximum brought to one
    /// (`twosided::equilibrate_general`).
    InfNorm,
    /// One-pass symmetric inf-norm equilibration `s_i = 1/sqrt(max_j |A_ij|)` (a
    /// single Knight-Ruiz step). The [`crate::SolverSettings`] default:
    /// cheapest, tolerates a zero diagonal, no iteration. See
    /// `infnorm::compute_onepass`; on the LU path the two-sided step
    /// `1/sqrt(row max)`, `1/sqrt(column max)`.
    #[default]
    OnePassInfNorm,
    /// MC64-style symmetric matching-based scaling. Matches the
    /// default behavior of MUMPS (SYM=2) and SSIDS
    /// (options%scaling=1). Useful on matrices where matching
    /// provides better conditioning than inf-norm balancing.
    Mc64Symmetric,
    /// Identity scaling (no-op). Use for regression testing and for
    /// inputs where any scaling is inappropriate.
    Identity,
    /// User-supplied pre-computed scaling vector in user-order
    /// indexing. Length must equal the matrix dimension.
    External(Vec<f64>),
}

/// Diagnostic information about how the scaling was computed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingInfo {
    /// A non-trivial scaling vector was applied to the matrix and the
    /// solve path must undo it. Produced when MC64 matching ran to
    /// completion on a non-singular matrix, and when the caller
    /// supplied an `External` scaling vector (the factor applies
    /// `D = diag(s)` regardless of how `s` was obtained).
    Applied,
    /// MC64 matching found a partial solution; unmatched rows and
    /// columns fall back to identity scaling. `n_unmatched` is the
    /// number of variables that could not be matched. The returned
    /// scaling vector has `1.0` at the unmatched positions.
    PartialSingular { n_unmatched: usize },
    /// The scaling vector is all-ones. Produced only by
    /// `ScalingStrategy::Identity`. (`External` reports `Applied` even
    /// when its vector happens to be all-ones.)
    NotApplied,
}

/// Compute the symmetric scaling vector for a sparse symmetric
/// matrix stored in CSC with only the lower triangle, following
/// `strategy`.
///
/// Returns a vector of length `n` in **user-order** indexing that is
/// applied as the congruence transform `D * A * D` with
/// `D = diag(scaling)`. When `Mc64Symmetric` succeeds on a non-singular
/// matrix, the largest-magnitude entries of `D * A * D` lie on the
/// diagonal and the off-diagonals are bounded by 1 in absolute value.
pub fn compute_scaling(
    matrix: &CscMatrix,
    strategy: &ScalingStrategy,
) -> Result<(Vec<f64>, ScalingInfo), RslabError> {
    match strategy {
        ScalingStrategy::Identity => Ok((vec![1.0; matrix.n], ScalingInfo::NotApplied)),
        ScalingStrategy::External(s) => {
            if s.len() != matrix.n {
                return Err(RslabError::InvalidInput(format!(
                    "external scaling has length {} but matrix has n={}",
                    s.len(),
                    matrix.n,
                )));
            }
            Ok((s.clone(), ScalingInfo::Applied))
        }
        ScalingStrategy::InfNorm => Ok(infnorm::compute_infnorm(matrix)),
        ScalingStrategy::OnePassInfNorm => Ok(infnorm::compute_onepass(matrix)),
        ScalingStrategy::Mc64Symmetric => mc64::compute_symmetric(matrix),
    }
}
