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
//! - `Auto`: routes between `InfNorm` and `Mc64Symmetric` by matrix
//!   shape, falling back to `InfNorm` when the MC64 vector is unusable.
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

/// One Knight-Ruiz equilibration step `d <- d / sqrt(m)`, guarded against
/// overflow/underflow. Applies the update only when
/// the result stays finite and strictly positive; otherwise `d` is held at
/// its last good value. `m` is the row/column infinity-norm and is assumed
/// `> 0` (the caller's existing `m > 0` guard).
///
/// On well-scaled matrices `d / sqrt(m)` is always finite and positive, so
/// this is **bit-identical** to the bare division - the guard bites only on
/// extreme or subnormal couplings, where the unguarded `d` would reach
/// `+-Inf` (then `NaN` on the next sweep) or `0`. Such a value silently
/// poisons every coupled row: the factorization sees a zeroed/NaN row, the
/// static pivot perturbation "repairs" it, and the solve returns garbage
/// with no error. Keeping `d` finite each sweep - rather than only
/// sanitizing at the end - also stops one overflowing row from dragging its
/// neighbours to zero.
#[inline]
pub(crate) fn kr_guarded_update(d: f64, m: f64) -> f64 {
    let cand = d / m.sqrt();
    if cand.is_finite() && cand > 0.0 {
        cand
    } else {
        d
    }
}

/// Guarded one-pass scale factor `1 / sqrt(m)` (single Knight-Ruiz
/// step): a zero, non-finite, or overflow-prone row max
/// yields the neutral `1.0` instead of a `0`/`Inf`/`NaN` factor that would
/// silently poison the equilibrated matrix. Bit-identical to the bare
/// expression for every healthy `m` (finite, `> 0`, not extreme).
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

/// User-facing scaling strategy selector.
///
/// The [`crate::SolverSettings`] default is `OnePassInfNorm`, the cheapest
/// choice. The enum's own `Default` is `Auto`: adaptive shape-based
/// routing that picks `Mc64Symmetric` for matrices with the arrow-KKT
/// signature and `InfNorm` everywhere else (see `pick_scaling_strategy`
/// and `compute_scaling_auto`). On arrow KKTs the matching scaling can
/// markedly reduce factor time and residuals compared with inf-norm
/// balancing.
///
/// `InfNorm` (iterative Knight-Ruiz) is the safer pick where MC64 is
/// known to hurt accuracy on a well-equilibrated matrix. `Mc64Symmetric`
/// forces the matching, at its extra cost, on every matrix.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalingStrategy {
    /// Knight-Ruiz inf-norm iterative equilibration (the "iterative Ruiz"
    /// arm of the equilibration knob). See `infnorm::compute_infnorm`.
    InfNorm,
    /// One-pass symmetric inf-norm equilibration `s_i = 1/sqrt(max_j |A_ij|)` (a
    /// single Knight-Ruiz step). The [`crate::SolverSettings`] default:
    /// cheapest, tolerates a zero diagonal, no iteration. See
    /// `infnorm::compute_onepass`.
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
    /// Adaptive shape-based routing: `Mc64Symmetric` when the matrix
    /// has the arrow-KKT signature (many degree-1 "constraint slack"
    /// columns), else `InfNorm`. The routing rule is documented at
    /// `pick_scaling_strategy`; `compute_scaling_auto` adds the
    /// fallback to `InfNorm`.
    #[default]
    Auto,
}

/// Reason that `ScalingStrategy::Auto` chose InfNorm scaling instead
/// of the MC64 matching it had nominally routed to. Surfaced through
/// `ScalingInfo::Mc64FallbackToInfnorm` so callers can tell it apart from
/// a user-requested InfNorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mc64FallbackReason {
    /// `Auto` picked MC64 by shape, but the pre-MC64 InfNorm trial
    /// (`scaling_spread(in_vec) < IN_SPREAD_GUARD`) produced a tight
    /// scaling - the matrix was already well-equilibrated and the
    /// Hungarian matching never ran.
    InfNormSpreadAcceptable,
    /// MC64 ran but produced a catastrophically worse scaling than
    /// InfNorm on a matrix whose raw `|diag|` range was tame enough
    /// that MC64 had no inherent ill-conditioning to recover from.
    /// Ratio guard: `mc_off > 1e6 and mc_off / in_off > 1e5 and
    /// raw_drng < 1e6`.
    Mc64WorseThanInfnorm,
    /// MC64 ran but the scaling vector it produced is itself
    /// numerically degenerate: its own spread `max|s| / min|s|`
    /// exceeds `1 / EPS ~ 4.5e15`. `D = diag(s)` is then singular to
    /// working precision, `D*A*D` underflows during the factorization,
    /// and Bunch-Kaufman force-accepts exact-zero pivots - a silently
    /// wrong solve. Seen on saddle-point KKTs with a structurally-zero
    /// `(2,2)` block, where the symmetric matching forces extreme
    /// path-accumulated dual potentials (spreads around `1e82` have been
    /// observed on parameter-estimation KKTs).
    Mc64ScalingDegenerate,
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
    /// `ScalingStrategy::Auto` resolved to `Mc64Symmetric` by shape
    /// routing but then fell back to InfNorm. The scaling vector
    /// returned alongside this info is the InfNorm vector (so the
    /// solve path applies it).
    Mc64FallbackToInfnorm { reason: Mc64FallbackReason },
    /// The scaling vector is all-ones - applying it is a no-op, so the
    /// solve path skips pre/post scaling entirely. Produced only by
    /// `ScalingStrategy::Identity`. (`External` reports `Applied` even
    /// when its vector happens to be all-ones, since the factor still
    /// runs the scaling loop.)
    NotApplied,
}

/// Compute the symmetric scaling vector for a sparse symmetric
/// matrix stored in CSC with only the lower triangle, following
/// `strategy`.
///
/// Returns a vector of length `n` in **user-order** indexing such
/// that applying `D = diag(scaling)` as the congruence transform
/// `D * A * D` produces a matrix whose largest-magnitude entries lie
/// on the diagonal. The off-diagonals are bounded by 1 in absolute
/// value when MC64 succeeds on a non-singular matrix.
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
            // `Applied`, not `NotApplied`: the factor scales the matrix
            // by `D = diag(s)` unconditionally, so the solve MUST undo
            // it. `NotApplied` is a load-bearing invariant meaning "the
            // scaling vector is all-ones" - the solve keys off it to
            // skip pre/post scaling (`solve_sparse`). Pairing a real
            // `s` with `NotApplied` factors `D*A*D` but solves it as
            // `A`, returning `D^-1A^-1D^-1b`. `s` may itself be all-ones,
            // in which case `Applied` just does bit-exact `x1.0` no-ops.
            Ok((s.clone(), ScalingInfo::Applied))
        }
        ScalingStrategy::InfNorm => Ok(infnorm::compute_infnorm(matrix)),
        ScalingStrategy::OnePassInfNorm => Ok(infnorm::compute_onepass(matrix)),
        ScalingStrategy::Mc64Symmetric => mc64::compute_symmetric(matrix),
        ScalingStrategy::Auto => compute_scaling_auto(matrix),
    }
}

/// Resolve `ScalingStrategy::Auto` with a fallback rule: when
/// `pick_scaling_strategy` would pick `Mc64Symmetric`, fall back to
/// InfNorm if InfNorm alone already equilibrates the matrix (tight
/// InfNorm spread), if the MC64 vector is degenerate (its own spread
/// exceeds `1 / EPS`), or if MC64 has produced a scaling that is
/// catastrophically worse than InfNorm on a matrix where InfNorm would
/// have done fine.
///
/// The last rule (all three must fire):
/// 1. `raw_diag_range < RAW_GUARD` - the raw matrix's diagonal
///    spans only a few orders of magnitude. MC64 has nothing
///    to recover from raw ill-conditioning here, so any huge
///    scaled off/diag ratio it produces is pure artifact, not
///    reflection of inherent matrix difficulty.
/// 2. `mc_off > MC_OFF_GUARD` - MC64's scaled `max(|off|/|diag|)`
///    is large in absolute terms.
/// 3. `mc_off / in_off > RATIO_GUARD` - and is much larger
///    than what InfNorm produces.
///
/// The first guard is the critical one: it lets matrices like
/// MEYER3NE_0220 (raw_drng=4.77e19, but MC64 actually works) keep
/// MC64, while still catching MSS1_0009 (raw_drng=51, where MC64
/// produces noise: 1e-6 residual against 6e-12 with InfNorm).
fn compute_scaling_auto(matrix: &CscMatrix) -> Result<(Vec<f64>, ScalingInfo), RslabError> {
    const RAW_GUARD: f64 = 1e6;
    const MC_OFF_GUARD: f64 = 1e6;
    const RATIO_GUARD: f64 = 1e5;
    // When InfNorm's scaling vector spread (max|s|/min|s|) is below
    // this threshold, the matrix is already nearly equilibrated by a
    // single Knight-Ruiz pass; MC64's heavier matching is gratuitous
    // and on some KKT families (ACOPP30 cond~3e16) produces a strictly
    // worse factor. The threshold catches ACOPP30 (1.63), MSS1 (1.09)
    // and HS75 (20.8) without flipping VESUVIA/VESUVIO/VESUVIOU/
    // MEYER3NE/CRESC132 (all >> 1e3 or where MC64 strictly wins).
    const IN_SPREAD_GUARD: f64 = 1e3;
    // An MC64 scaling vector whose own spread `max|s| / min|s|`
    // exceeds `1 / EPS` is degenerate to working precision -
    // `D = diag(s)` is singular, `D*A*D` underflows, and Bunch-Kaufman
    // force-accepts exact-zero pivots, returning a silently wrong
    // solve. Legitimate MC64 spreads seen on KKT test matrices stay
    // below ~3.3e15 (ssine); saddle-point parameter-estimation KKTs can
    // blow up to ~3e82. `1 / EPS` (~4.503e15) is a hard numerical
    // limit.
    const MC64_SPREAD_GUARD: f64 = 1.0 / f64::EPSILON;

    let picked = pick_scaling_strategy(matrix);
    if !matches!(picked, ScalingStrategy::Mc64Symmetric) {
        // Auto picked InfNorm-class - no fallback needed.
        return compute_scaling(matrix, &picked);
    }

    // Pre-MC64 InfNorm trial: if Knight-Ruiz produces a tight
    // scaling vector, the matrix is already well-equilibrated and
    // MC64's matching can only hurt. This catches the ACOPP30
    // family (raw_drng=1.06e10 but in_spread=1.63), which the
    // `raw_drng >= RAW_GUARD` pre-filter below would send to MC64.
    //
    // Tag the result as `Mc64FallbackToInfnorm` so
    // downstream telemetry can distinguish a "user picked InfNorm"
    // from a "Auto routed to MC64 but fell back" outcome. The
    // underlying scaling vector is unchanged.
    let (in_vec, _in_info) = infnorm::compute_infnorm(matrix);
    if scaling_spread(&in_vec) < IN_SPREAD_GUARD {
        return Ok((
            in_vec,
            ScalingInfo::Mc64FallbackToInfnorm {
                reason: Mc64FallbackReason::InfNormSpreadAcceptable,
            },
        ));
    }

    // Compute the MC64 scaling once. Every branch below either
    // returns this vector or inspects it.
    let (mc_vec, mc_info) = mc64::compute_symmetric(matrix)?;

    // Catastrophic-spread guard. An MC64 scaling whose own spread
    // exceeds `MC64_SPREAD_GUARD` is degenerate to working precision
    // and silently corrupts the factorization (see the constant's
    // comment). Discard it and fall back to the already-computed
    // InfNorm vector. This check is placed BEFORE the `raw_diag_range`
    // fast-path so it fires regardless of raw conditioning: the
    // affected saddle-point KKTs are genuinely ill-conditioned
    // (`raw_diag_range >= RAW_GUARD`) and would otherwise take the
    // fast-path straight to the unchecked MC64 vector.
    if scaling_spread(&mc_vec) > MC64_SPREAD_GUARD {
        return Ok((
            in_vec,
            ScalingInfo::Mc64FallbackToInfnorm {
                reason: Mc64FallbackReason::Mc64ScalingDegenerate,
            },
        ));
    }

    // Cheap pre-filter: a wide raw |diag| range means MC64 has
    // genuine work to do (and the InfNorm trial above did not
    // produce a tight scaling). Skip the off-diag diagnostic and
    // commit to MC64.
    if raw_diag_range(matrix) >= RAW_GUARD {
        return Ok((mc_vec, mc_info));
    }

    let mc_off = max_off_diag_ratio(matrix, &mc_vec);
    if mc_off <= MC_OFF_GUARD {
        // MC64 produced a well-conditioned scaled matrix.
        return Ok((mc_vec, mc_info));
    }
    let in_off = max_off_diag_ratio(matrix, &in_vec);
    let ratio = if in_off > 0.0 {
        mc_off / in_off
    } else {
        f64::INFINITY
    };
    if ratio > RATIO_GUARD {
        // MC64 is catastrophically worse than InfNorm AND the raw
        // matrix is already well-behaved - fall back to InfNorm.
        // The solve path applies the InfNorm scaling vector; tag
        // the info as `Mc64FallbackToInfnorm` so callers can
        // distinguish this from a user-requested InfNorm.
        Ok((
            in_vec,
            ScalingInfo::Mc64FallbackToInfnorm {
                reason: Mc64FallbackReason::Mc64WorseThanInfnorm,
            },
        ))
    } else {
        Ok((mc_vec, mc_info))
    }
}

/// Return `max|s|/min|s|` over the nonzero entries of `s`. Returns
/// `+inf` if `s` has no nonzero entry. Used by the `Auto` fallback as a fast
/// "is the matrix already equilibrated?" probe on the InfNorm
/// scaling vector.
fn scaling_spread(s: &[f64]) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for v in s {
        let a = v.abs();
        if a > 0.0 {
            if a < lo {
                lo = a;
            }
            if a > hi {
                hi = a;
            }
        }
    }
    if lo.is_finite() && lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

/// Compute `max |A_{j,j}| / min(|A_{j,j}|)` over diagonal entries
/// that are present and nonzero. Returns `+inf` if no nonzero
/// diagonal is present. O(nnz), no allocations.
fn raw_diag_range(matrix: &CscMatrix) -> f64 {
    let n = matrix.n;
    if n == 0 {
        return 0.0;
    }
    let mut lo = f64::INFINITY;
    let mut hi = 0.0_f64;
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            if matrix.row_idx[k] == j {
                let a = matrix.values[k].abs();
                if a > 0.0 {
                    if a < lo {
                        lo = a;
                    }
                    if a > hi {
                        hi = a;
                    }
                }
            }
        }
    }
    if lo.is_finite() && lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

/// Compute `max_j (max_{i != j} |s_i * A_{i,j} * s_j|) / |s_j * A_{j,j} * s_j|`
/// over all columns of the symmetrically-scaled matrix `D * A * D`.
/// Diagonal columns with zero diagonal contribute `+inf` to the max.
/// O(nnz), no allocations.
fn max_off_diag_ratio(matrix: &CscMatrix, scaling: &[f64]) -> f64 {
    let n = matrix.n;
    if n == 0 {
        return 0.0;
    }
    let mut diag_abs = vec![0.0_f64; n];
    let mut max_off = vec![0.0_f64; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let v = (matrix.values[k] * scaling[i] * scaling[j]).abs();
            if i == j {
                diag_abs[j] = v;
            } else {
                if v > max_off[i] {
                    max_off[i] = v;
                }
                if v > max_off[j] {
                    max_off[j] = v;
                }
            }
        }
    }
    let mut worst = 0.0_f64;
    for j in 0..n {
        let r = if diag_abs[j] > 0.0 {
            max_off[j] / diag_abs[j]
        } else if max_off[j] > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        if r > worst {
            worst = r;
        }
    }
    worst
}

/// Resolve `ScalingStrategy::Auto` to a concrete strategy based on
/// matrix shape.
///
/// Routes to `Mc64Symmetric` when the matrix has the arrow-KKT
/// signature: BOTH
///   (a) many degree-1 "constraint slack" columns whose only
///       structurally nonzero entry is the diagonal
///       (`diag_only / n >= 0.30`), AND
///   (b) at least one structurally dense column whose nonzero count
///       exceeds `MAX_COL_NNZ_FOR_INFNORM = 32` - the "arrow head"
///       that creates wildly mismatched off-diagonal magnitudes
///       InfNorm cannot equalize.
///
/// Else routes to `InfNorm`.
///
/// Both counts ignore explicit stored `0.0` entries: an
/// explicit zero is not coupling and not mass, so a value-only zero
/// must not change which scaling strategy a matrix routes to. Callers
/// that refill a fixed sparsity pattern each solve (IPM backends) leave
/// such zeros in the zero-`(2,2)` block; a value-blind router would
/// split the kept and stripped forms of the same KKT.
///
/// **Why both gates are needed.** The diag_only ratio alone CANNOT
/// distinguish a 1-D banded KKT like clnlbeam (n=99999, diag_only=40%,
/// max_col_nnz=5) from a true arrow KKT like VESUVIO (n=3083,
/// diag_only=33%, max_col_nnz=1026). clnlbeam scores HIGHER on
/// diag_only/n than VESUVIO yet MC64 hurts its IPM trajectory (about
/// 4.4x the iterations), while VESUVIO benefits 6x-243x from MC64. The
/// dense column count (gate b) is what separates them: banded PDE-like
/// KKTs have small max column degree by construction; arrow KKTs
/// concentrate the slack/dual coupling in 1-8 dense columns of size
/// ~ n/3.
///
/// Threshold calibration:
///
/// | matrix          | n     | diag_only/n | max_col_nnz | MC64 helps? |
/// |-----------------|-------|-------------|-------------|-------------|
/// | clnlbeam_0000   | 99999 | 40.0%       | 5           | NO (4.4x iters) |
/// | VESUVIOU_0000   | 3083  | 33.2%       | 1026        | YES (243x)  |
/// | VESUVIO_0000    | 3083  | 33.2%       | 1026        | YES         |
/// | VESUVIA_0000    | 3083  | 33.2%       | 1026        | YES         |
/// | MUONSINE_0000   | 1537  | 33.3%       | 512         | YES         |
/// | CRESC132_0000   | 5314  | 50.0%       | 2657        | YES         |
/// | ACOPP30_0064    | 209   | 65.6%       | 29          | NO          |
///
/// `32` sits an order of magnitude above ACOPP30's max (29) and an
/// order of magnitude below MUONSINE's (512), giving the widest
/// possible margin on either side of the validation panel.
///
/// **Routing on the matrix, not on the numbering.** The head gate measures the
/// *symmetric* degree of an index, not the length of its stored column. A
/// `CscMatrix` holds one triangle, so a stored column counts couplings to one
/// side of `j` only - a property of the index order. Under the pure relabeling
/// `P(i) = n-1-i` an arrow head reports its full degree one way and its
/// diagonal-plus-one the other, and the route flips. Symmetric degree is never
/// smaller than the stored degree, so the gate can only become easier to pass:
/// no matrix loses `Mc64Symmetric` to this.
///
/// The slack-mass gate is still order-dependent; the order-invariant
/// reformulations tried so far stripped `Mc64Symmetric` from matrices that
/// need it, so it stays as it is.
///
/// Cost: one allocation-free `O(n+nnz)` pass decides every matrix that fails the
/// slack-mass gate or whose densest stored column already clears the threshold
/// (sound, by the monotonicity above). Only the ambiguous remainder allocates
/// the `n`-length degree accumulator.
pub fn pick_scaling_strategy(matrix: &CscMatrix) -> ScalingStrategy {
    /// Maximum stored column nnz for the Auto policy to consider the
    /// matrix "banded enough" that InfNorm is the safe choice even
    /// when the diag_only ratio is high. See function docs.
    const MAX_COL_NNZ_FOR_INFNORM: usize = 32;

    let n = matrix.n;
    if n == 0 {
        return ScalingStrategy::InfNorm;
    }
    let mut diag_only = 0usize;
    let mut max_col_nnz = 0usize;
    for j in 0..n {
        let start = matrix.col_ptr[j];
        let end = matrix.col_ptr[j + 1];
        // Count only structurally meaningful entries. An explicit
        // stored `0.0` is not coupling and not mass - IPM callers
        // refill a fixed pattern each iterate, leaving value-only `0.0`
        // slots in the zero-`(2,2)` block. Counting them would let a
        // value-only zero flip this router, sending the kept form of a
        // KKT to MC64 and the structurally identical stripped form to
        // InfNorm.
        let mut nnz_col = 0usize;
        let mut diag_nonzero = false;
        for k in start..end {
            if matrix.values[k] == 0.0 {
                continue;
            }
            nnz_col += 1;
            if matrix.row_idx[k] == j {
                diag_nonzero = true;
            }
        }
        if nnz_col > max_col_nnz {
            max_col_nnz = nnz_col;
        }
        if nnz_col == 1 && diag_nonzero {
            diag_only += 1;
        }
    }
    let has_slack_mass = diag_only as f64 / n as f64 >= 0.3;
    if !has_slack_mass {
        return ScalingStrategy::InfNorm;
    }
    // Stored degree is a lower bound on symmetric degree: clearing the gate on
    // the cheap pass is already decisive.
    if max_col_nnz > MAX_COL_NNZ_FOR_INFNORM {
        return ScalingStrategy::Mc64Symmetric;
    }
    // Ambiguous: the stored column is short, but the index may still be the
    // arrow head seen from the other side. Count both directions.
    let mut deg = vec![0usize; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            if matrix.values[k] == 0.0 {
                continue;
            }
            let i = matrix.row_idx[k];
            deg[i] += 1;
            if i != j {
                deg[j] += 1;
            }
        }
    }
    if deg.iter().copied().max().unwrap_or(0) > MAX_COL_NNZ_FOR_INFNORM {
        ScalingStrategy::Mc64Symmetric
    } else {
        ScalingStrategy::InfNorm
    }
}

// Hungarian kernel types, used by the `mc64` module. Not part of the
// public API.
#[allow(unused_imports)]
pub(crate) use hungarian::{hungarian_match, CostGraph, Matching};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    /// Lower-triangle CSC from triplets `(row, col, value)` with `row >= col`.
    fn from_lower(n: usize, trip: &[(usize, usize, f64)]) -> CscMatrix {
        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = trip
            .iter()
            .map(|&(i, j, v)| {
                assert!(i >= j, "lower triangle only");
                (i, j, v)
            })
            .fold((vec![], vec![], vec![]), |mut acc, (i, j, v)| {
                acc.0.push(i);
                acc.1.push(j);
                acc.2.push(v);
                acc
            });
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    /// Relabel by `P(i) = n-1-i` and re-fold into the lower triangle. This is a
    /// pure renumbering: the matrix is the same operator with its indices
    /// reversed, so every routing decision must be unchanged.
    fn reversed(n: usize, trip: &[(usize, usize, f64)]) -> Vec<(usize, usize, f64)> {
        trip.iter()
            .map(|&(i, j, v)| {
                let (pi, pj) = (n - 1 - i, n - 1 - j);
                (pi.max(pj), pi.min(pj), v)
            })
            .collect()
    }

    /// `n_iso` isolated diagonal-only indices plus a star of `leaves` indices
    /// around one head. The head's couplings are stored in its own column when
    /// the head has the smallest index of the star, and in the leaves' columns
    /// when it has the largest - which is exactly the asymmetry a stored-column
    /// gate reads and a symmetric-degree gate does not. The slack-mass gate
    /// passes in both orientations by construction (the leaves are degree-1 one
    /// way, the head and the isolated indices are degree-1 the other).
    fn star_with_isolated(n_iso: usize, leaves: usize) -> (usize, Vec<(usize, usize, f64)>) {
        let n = n_iso + 1 + leaves;
        let head = n_iso; // smallest index of the star
        let mut trip: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 4.0)).collect();
        for l in 0..leaves {
            let leaf = n_iso + 1 + l;
            trip.push((leaf.max(head), leaf.min(head), 0.5));
        }
        (n, trip)
    }

    /// `n_iso` isolated diagonal-only indices plus a genuine band of `n_band`
    /// indices with half-bandwidth `bw`: every index couples only to its
    /// neighbours, so its symmetric degree is at most `2*bw+1` no matter how the
    /// indices are numbered. This is the clnlbeam shape - high diagonal-only
    /// mass, narrow coupling - that the head gate must keep on InfNorm.
    fn banded_with_isolated(
        n_iso: usize,
        n_band: usize,
        bw: usize,
    ) -> (usize, Vec<(usize, usize, f64)>) {
        let n = n_iso + n_band;
        let mut trip: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 4.0)).collect();
        for j in n_iso..n {
            for d in 1..=bw {
                if j + d < n {
                    trip.push((j + d, j, -1.0));
                }
            }
        }
        (n, trip)
    }

    /// The route is a property of the matrix, not of the numbering.
    #[test]
    fn scaling_route_is_permutation_invariant() {
        let (n, star) = star_with_isolated(60, 35);
        let (n_small, small_star) = star_with_isolated(60, 20);
        let tri: Vec<(usize, usize, f64)> = (0..200)
            .flat_map(|i| {
                let mut e = vec![(i, i, 4.0)];
                if i + 1 < 200 {
                    e.push((i + 1, i, -1.0));
                }
                e
            })
            .collect();

        for (label, n, trip) in [
            ("star head degree 36", n, star),
            ("star head degree 21", n_small, small_star),
            ("tridiagonal", 200, tri),
        ] {
            let forward = pick_scaling_strategy(&from_lower(n, &trip));
            let backward = pick_scaling_strategy(&from_lower(n, &reversed(n, &trip)));
            assert_eq!(forward, backward, "{label}: route flipped under relabeling");
        }
    }

    /// The head gate itself still separates: a star wide enough to be an arrow
    /// head routes to MC64 from either end, a narrow one to InfNorm from either.
    #[test]
    fn scaling_head_gate_separates_by_symmetric_degree() {
        let (n, wide) = star_with_isolated(60, 35);
        let (n2, narrow) = star_with_isolated(60, 20);
        for trip in [wide.clone(), reversed(n, &wide)] {
            assert_eq!(
                pick_scaling_strategy(&from_lower(n, &trip)),
                ScalingStrategy::Mc64Symmetric
            );
        }
        for trip in [narrow.clone(), reversed(n2, &narrow)] {
            assert_eq!(
                pick_scaling_strategy(&from_lower(n2, &trip)),
                ScalingStrategy::InfNorm
            );
        }
    }

    /// Build an arrow-KKT-shaped CSC.
    ///
    /// Layout: `diag_only` degree-1 slack columns followed by
    /// `n - diag_only` columns each of which stores the diagonal plus
    /// `dense_off` off-diagonal entries. When `dense_off` is large
    /// enough that `1 + dense_off > MAX_COL_NNZ_FOR_INFNORM` (= 32),
    /// the non-slack columns form an "arrow head" that triggers the
    /// dense-column gate in `pick_scaling_strategy`.
    fn shape_csc(n: usize, diag_only: usize, dense_off: usize) -> CscMatrix {
        assert!(diag_only <= n);
        let n_dense = n - diag_only;
        assert!(dense_off < n, "dense_off must be < n");
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        col_ptr.push(0);
        for j in 0..n {
            row_idx.push(j);
            values.push(1.0);
            if j >= diag_only {
                // Walk earlier rows to fill `dense_off` off-diagonals.
                // We may have fewer earlier rows than requested on the
                // first non-slack column; cap to what's available.
                let take = dense_off.min(j);
                for k in 0..take {
                    row_idx.push(k);
                    values.push(0.1);
                }
                let _ = n_dense; // sanity hint for the reader
            }
            col_ptr.push(row_idx.len());
        }
        CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Build the parameter-estimation saddle-point KKT used as the
    /// MC64 spread-guard test oracle.
    ///
    /// `[H B^T; B 0]` stored as the lower triangle: `ntheta` dense
    /// parameter columns (graded H diagonal `1 .. theta_top`, each
    /// coupling to every constraint with coefficient `pcoef`), `nx`
    /// zero-diagonal state columns chained to `nc = nx` zero-`(2,2)`
    /// constraint columns by the constant ratio `base` (state `s`
    /// couples to constraint `s` with coefficient 1 and to constraint
    /// `s-1` with coefficient `base`). The constant ratio makes the
    /// chain translation-invariant - InfNorm equilibrates it
    /// uniformly - while MC64's symmetric matching telescopes `base`
    /// into a path-accumulated potential.
    ///
    /// `nslack` degree-1 columns with a *nonzero* unit diagonal are
    /// appended last. They model the bound slacks of a
    /// bound-constrained parameter-estimation KKT (real slack mass +
    /// zero equality duals). They are required because with the
    /// value-aware router the explicit-zero constraint/state diagonals
    /// no longer count as `diag_only`, so genuine slack mass is what
    /// routes this matrix to `Mc64Symmetric` (`nslack/n >= 0.30`,
    /// `max_col_nnz = 1 + nc > 32`). Being disconnected from the chain,
    /// MC64's matching decomposes over them (each matches itself,
    /// `log|1| = 0`, scale 1) - the chain potentials, and hence
    /// `scaling_spread`, are unchanged from the no-slack form.
    fn build_synth_kkt(
        ntheta: usize,
        nx: usize,
        theta_top: f64,
        base: f64,
        pcoef: f64,
        nslack: usize,
    ) -> CscMatrix {
        let nc = nx;
        let n = ntheta + nx + nc + nslack;
        let con0 = ntheta + nx; // first constraint global index
        let slack0 = ntheta + nx + nc; // first slack global index
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        // Parameter columns: graded H diagonal + coupling to every
        // constraint.
        for p in 0..ntheta {
            let hp = if ntheta > 1 {
                theta_top.powf(p as f64 / (ntheta - 1) as f64)
            } else {
                1.0
            };
            rows.push(p);
            cols.push(p);
            vals.push(hp);
            for c in 0..nc {
                rows.push(con0 + c);
                cols.push(p);
                vals.push(pcoef);
            }
        }
        // State columns: zero H diagonal + chain coupling.
        for s in 0..nx {
            let js = ntheta + s;
            rows.push(js);
            cols.push(js);
            vals.push(0.0);
            if s >= 1 {
                rows.push(con0 + s - 1);
                cols.push(js);
                vals.push(base);
            }
            rows.push(con0 + s);
            cols.push(js);
            vals.push(1.0);
        }
        // Constraint columns: zero (2,2) diagonal only.
        for c in 0..nc {
            let jc = con0 + c;
            rows.push(jc);
            cols.push(jc);
            vals.push(0.0);
        }
        // Slack columns: genuine degree-1 mass, nonzero unit diagonal,
        // disconnected from the chain.
        for s in 0..nslack {
            let js = slack0 + s;
            rows.push(js);
            cols.push(js);
            vals.push(1.0);
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals)
            .expect("synthetic KKT triplets are valid lower-triangle")
    }

    #[test]
    fn pick_scaling_strategy_picks_mc64_for_arrow_kkt() {
        // n=100, 80 slacks, 20 arrow-head cols each storing diag +
        // 50 earlier rows. diag_only/n=0.80 >= 0.30 AND max_col_nnz=51
        // > 32 -> MC64.
        let csc = shape_csc(100, 80, 50);
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
    }

    #[test]
    fn pick_scaling_strategy_picks_infnorm_for_banded_high_diag_only() {
        // The clnlbeam shape: high diagonal-only mass (0.60) but a narrow
        // band, so no index couples widely. Must route to InfNorm - this is
        // the entire motivation for the head gate. The band is built as
        // a band rather than as couplings to a few shared leading rows: the
        // latter is an arrow head that only the stored-column view hides.
        let (n, trip) = banded_with_isolated(60, 40, 2);
        assert_eq!(
            pick_scaling_strategy(&from_lower(n, &trip)),
            ScalingStrategy::InfNorm
        );
    }

    #[test]
    fn pick_scaling_strategy_picks_infnorm_for_dense_low_diag_only() {
        // 0 diag-only cols, but each col is dense -> fails the
        // diag_only gate even though the arrow-head gate passes.
        let csc = shape_csc(100, 0, 50);
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::InfNorm);
    }

    #[test]
    fn pick_scaling_strategy_diag_only_threshold_boundary() {
        // Dense-column gate satisfied for both rows of the table
        // (arrow-head cols have 51 nnz > 32). Only the diag_only
        // ratio varies across the boundary at 0.30.
        let below = shape_csc(100, 29, 50);
        assert_eq!(pick_scaling_strategy(&below), ScalingStrategy::InfNorm);
        let at = shape_csc(100, 30, 50);
        assert_eq!(pick_scaling_strategy(&at), ScalingStrategy::Mc64Symmetric);
    }

    #[test]
    fn pick_scaling_strategy_max_col_nnz_threshold_boundary() {
        // Diagonal-only mass is satisfied for both; only the head's symmetric
        // degree crosses the boundary at 32.
        let (n32, at32) = star_with_isolated(60, 31); // head degree 32 -> fails
        assert_eq!(
            pick_scaling_strategy(&from_lower(n32, &at32)),
            ScalingStrategy::InfNorm
        );
        let (n33, at33) = star_with_isolated(60, 32); // head degree 33 -> passes
        assert_eq!(
            pick_scaling_strategy(&from_lower(n33, &at33)),
            ScalingStrategy::Mc64Symmetric
        );
    }

    #[test]
    fn pick_scaling_strategy_empty_matrix_picks_infnorm() {
        let csc = CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        };
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::InfNorm);
    }

    /// `pick_scaling_strategy` must treat an explicit stored `0.0` as
    /// structurally absent. IPM callers refill a fixed KKT pattern each
    /// iterate, leaving value-only `0.0` slots in the zero-`(2,2)` block;
    /// a value-blind structural router counts them and flips the scaling
    /// strategy between the kept and the stripped form of the same KKT.
    ///
    /// Layout (n=100): 50 arrow-head columns each storing the diagonal
    /// plus 40 nonzero rows below it (41 nnz > 32 -> arrow head); then 50
    /// "constraint" columns whose diagonal is the variable:
    ///   - `Zero`:   one explicit `0.0` on the diagonal.
    ///   - `Absent`: structurally empty.
    ///   - `Real`:   one nonzero `1.0` on the diagonal.
    ///
    /// Oracle (hand calculation): an explicit `0.0` is not mass. `Zero`
    /// and `Absent` must route identically (-> InfNorm: no real slack
    /// mass); `Real` has 50 genuine degree-1 columns (0.50 >= 0.30) -> MC64.
    #[test]
    fn pick_scaling_strategy_explicit_zero_diag_not_slack_mass() {
        #[derive(Clone, Copy)]
        enum Cdiag {
            Zero,
            Absent,
            Real,
        }
        fn build(cdiag: Cdiag) -> CscMatrix {
            let n = 100;
            let n_dense = 50;
            let mut col_ptr = vec![0usize];
            let mut row_idx: Vec<usize> = Vec::new();
            let mut values: Vec<f64> = Vec::new();
            // Arrow-head columns: diagonal + 40 nonzero rows below.
            for j in 0..n_dense {
                row_idx.push(j);
                values.push(2.0);
                for r in (j + 1)..(j + 41) {
                    row_idx.push(r);
                    values.push(0.3);
                }
                col_ptr.push(row_idx.len());
            }
            // Constraint columns.
            for j in n_dense..n {
                match cdiag {
                    Cdiag::Zero => {
                        row_idx.push(j);
                        values.push(0.0);
                    }
                    Cdiag::Absent => {}
                    Cdiag::Real => {
                        row_idx.push(j);
                        values.push(1.0);
                    }
                }
                col_ptr.push(row_idx.len());
            }
            CscMatrix {
                n,
                col_ptr,
                row_idx,
                values,
            }
        }
        // Explicit-zero diagonals and structural absence must agree.
        assert_eq!(
            pick_scaling_strategy(&build(Cdiag::Zero)),
            ScalingStrategy::InfNorm,
            "explicit-zero constraint diagonals are not slack mass"
        );
        assert_eq!(
            pick_scaling_strategy(&build(Cdiag::Absent)),
            ScalingStrategy::InfNorm
        );
        // Genuine nonzero degree-1 columns are slack mass -> MC64.
        assert_eq!(
            pick_scaling_strategy(&build(Cdiag::Real)),
            ScalingStrategy::Mc64Symmetric
        );
    }

    /// An explicit-zero *off-diagonal* entry must neither
    /// inflate `max_col_nnz` nor disqualify an otherwise-`diag_only`
    /// column. Here the 50 constraint columns each store a nonzero
    /// diagonal AND a single explicit-zero off-diagonal; value-aware
    /// counting still sees them as 50 degree-1 columns (0.50 >= 0.30) ->
    /// MC64. (Value-blind counting would call them degree-2 and route
    /// to InfNorm.)
    #[test]
    fn pick_scaling_strategy_explicit_zero_offdiag_ignored() {
        let n = 100;
        let mut col_ptr = vec![0usize];
        let mut row_idx: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for j in 0..50 {
            row_idx.push(j);
            values.push(2.0);
            for r in (j + 1)..(j + 41) {
                row_idx.push(r);
                values.push(0.3);
            }
            col_ptr.push(row_idx.len());
        }
        for j in 50..n {
            row_idx.push(0);
            values.push(0.0); // explicit-zero off-diagonal
            row_idx.push(j);
            values.push(1.0); // nonzero diagonal
            col_ptr.push(row_idx.len());
        }
        let csc = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
    }

    /// The clnlbeam KKT scores 40% diag_only and would route to MC64 on
    /// the slack-mass gate alone, which inflates the IPM iteration count
    /// about 4x. The dense-column gate (max_col_nnz=5 fails) routes it to
    /// InfNorm. Skipped when the fixture is not present.
    #[test]
    fn pick_scaling_strategy_routes_clnlbeam_to_infnorm() {
        let path = std::path::Path::new("data/matrices/kkt-mittelmann/clnlbeam/clnlbeam_0000.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return, // fixture not present - skip
        };
        let csc = mtx.to_csc().expect("clnlbeam_0000 CSC build");
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::InfNorm);
    }

    #[test]
    fn compute_scaling_auto_routes_to_mc64_on_arrow_kkt() {
        // Build a symmetric arrow KKT large enough that the dense
        // "linking" columns clear the `max_col_nnz > 32` gate.
        // n=80: 40 diag-only slack columns + 40 dense columns where
        // column j (j >= 40) stores rows j..n. Column 40 has 40
        // entries (well above the 32 threshold).
        // Ratio diag_only/n = 40/80 = 0.50 >= 0.30 -> Auto resolves to MC64.
        let n = 80;
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        // 40 diag-only columns.
        for j in 0..40 {
            row_idx.push(j);
            values.push(2.0);
            col_ptr.push(row_idx.len());
        }
        // 40 dense columns (diagonal + all earlier dense rows).
        for j in 40..n {
            row_idx.push(j);
            values.push(2.0);
            for i in (j + 1)..n {
                row_idx.push(i);
                values.push(0.1);
            }
            col_ptr.push(row_idx.len());
        }
        let csc = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
        // Auto and explicit Mc64Symmetric must produce the same vector
        // here - this is a well-conditioned shape, so the ratio
        // fallback rule (mc_off > 1e6 and mc_off/in_off > 1e5) never fires.
        let (auto_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto routing should succeed");
        let (mc64_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).expect("MC64 should succeed");
        assert_eq!(auto_s, mc64_s);
    }

    #[test]
    fn max_off_diag_ratio_basic_well_conditioned() {
        // 3x3 well-conditioned matrix:
        //   [ 4  1  0 ]
        //   [ 1  3  1 ]
        //   [ 0  1  2 ]
        // With identity scaling, max ratio = 1/2 = 0.5.
        let csc = CscMatrix {
            n: 3,
            col_ptr: vec![0, 2, 4, 5],
            row_idx: vec![0, 1, 1, 2, 2],
            values: vec![4.0, 1.0, 3.0, 1.0, 2.0],
        };
        let s = vec![1.0; 3];
        let r = max_off_diag_ratio(&csc, &s);
        assert!((r - 0.5).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn max_off_diag_ratio_zero_diag_gives_infinity() {
        // 2x2 with zero diagonal on column 0:
        //   [ 0  1 ]
        //   [ 1  1 ]
        // Column 0 has off=1, diag=0 -> +inf. Column 1 has off=1,
        // diag=1 -> 1.0. max = +inf.
        let csc = CscMatrix {
            n: 2,
            col_ptr: vec![0, 2, 3],
            row_idx: vec![0, 1, 1],
            values: vec![0.0, 1.0, 1.0],
        };
        let s = vec![1.0; 2];
        let r = max_off_diag_ratio(&csc, &s);
        assert!(r.is_infinite(), "got {r}");
    }

    /// An arrow KKT with uniform absolute values triggers
    /// the `Auto` shape rule (high diag_only ratio + a dense arrow head
    /// of size > 32) but the pre-MC64 InfNorm trial gives a constant
    /// scaling vector (spread = 1), so `IN_SPREAD_GUARD` fires and the
    /// fallback is taken. Assert the returned `ScalingInfo` is
    /// `Mc64FallbackToInfnorm{InfNormSpreadAcceptable}`, so the
    /// fallback is visible to the caller.
    ///
    /// Construction: n=40. Column 0 stores diag + all 39 earlier-row
    /// entries with value 2.0 (40 stored entries -> exceeds the dense
    /// gate). Columns 1..39 are degree-1 with the diagonal value 2.0
    /// (39 of 40 -> diag_only/n = 0.975 >= 0.30). All stored absolute
    /// values are 2.0, so Knight-Ruiz converges to a uniform `d`.
    #[test]
    fn auto_surfaces_infnorm_spread_fallback_on_uniform_diag() {
        let n = 40;
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        // Column 0: dense (all 40 rows), uniform |a| = 2.0.
        for i in 0..n {
            row_idx.push(i);
            values.push(2.0);
        }
        col_ptr.push(row_idx.len());
        // Columns 1..n: degree-1 diagonal, value 2.0.
        for j in 1..n {
            row_idx.push(j);
            values.push(2.0);
            col_ptr.push(row_idx.len());
        }
        let csc = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        // Precondition: routing rule says MC64.
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
        // The fallback must surface the new variant with the
        // InfNormSpreadAcceptable reason.
        let (auto_s, info) = compute_scaling(&csc, &ScalingStrategy::Auto)
            .expect("Auto on uniform diag should succeed");
        match info {
            ScalingInfo::Mc64FallbackToInfnorm {
                reason: Mc64FallbackReason::InfNormSpreadAcceptable,
            } => {}
            other => panic!(
                "expected Mc64FallbackToInfnorm{{InfNormSpreadAcceptable}}, got {:?}",
                other
            ),
        }
        // And the returned vector must be the InfNorm vector
        // (so the solve path applies it, not identity).
        let (in_s, _) = compute_scaling(&csc, &ScalingStrategy::InfNorm)
            .expect("InfNorm on uniform diag should succeed");
        assert_eq!(auto_s, in_s, "fallback vector must be the InfNorm vector");
    }

    /// Fallback regression test - MSS1_0009 should resolve to InfNorm
    /// under Auto despite the diag_only/n=0.45 ratio triggering the
    /// MC64 routing rule. MC64 produces a scaled
    /// `max(|off|/|diag|) ~ 7.8e14` while InfNorm gets ~ 2.0e8 - ratio
    /// 3.9e6 is well above the 1e5 RATIO_GUARD. Skipped when the
    /// fixture is not present.
    #[test]
    fn auto_falls_back_to_infnorm_on_mss1_0009() {
        let path = std::path::Path::new("data/matrices/kkt/MSS1/MSS1_0009.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return, // fixture not present - skip
        };
        let csc = mtx.to_csc().expect("MSS1_0009 CSC build");

        // pick_scaling_strategy still picks MC64 - the routing rule
        // hasn't changed.
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);

        // But Auto should resolve to the InfNorm scaling because of
        // the fallback.
        let (auto_s, auto_info) = compute_scaling(&csc, &ScalingStrategy::Auto)
            .expect("Auto on MSS1_0009 should succeed");
        let (in_s, _) = compute_scaling(&csc, &ScalingStrategy::InfNorm)
            .expect("InfNorm on MSS1_0009 should succeed");
        let (mc_s, _) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric)
            .expect("MC64 on MSS1_0009 should succeed");
        assert_eq!(auto_s, in_s, "Auto must fall back to InfNorm on MSS1_0009");
        assert_ne!(
            auto_s, mc_s,
            "Auto must NOT use MC64 on MSS1_0009 (would regress residual to 1e-6)"
        );
        // Either fallback reason is acceptable here. The high-level invariant - "Auto falls back to
        // InfNorm on MSS1_0009" - is already proven by the
        // `assert_eq!(auto_s, in_s)` above. Empirically the earlier
        // `InfNormSpreadAcceptable` guard fires on this matrix
        // under the current IN_SPREAD_GUARD threshold, but the test
        // tolerates either variant so threshold tuning does not
        // wedge this fixture-gated test. The
        // `Mc64WorseThanInfnorm` branch is exercised explicitly by
        // the synthetic unit tests above.
        match auto_info {
            ScalingInfo::Mc64FallbackToInfnorm { .. } => {}
            other => panic!("MSS1_0009: expected Mc64FallbackToInfnorm, got {:?}", other),
        }
    }

    /// The fallback must NOT fire on the VESUVIO/CRESC class - these
    /// are the matrices MC64 scaling is routed for. MC64
    /// produces a scaled `mc_off ~ 4.84e12` for VESUVIA_0000 with
    /// `mc/in ~ 40` - well below the 1e5 RATIO_GUARD.
    #[test]
    fn auto_keeps_mc64_on_vesuvia_0000() {
        let path = std::path::Path::new("data/matrices/kkt/VESUVIA/VESUVIA_0000.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return, // fixture not present - skip
        };
        let csc = mtx.to_csc().expect("VESUVIA_0000 CSC build");
        let (auto_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto on VESUVIA_0000");
        let (mc_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).expect("MC64 on VESUVIA_0000");
        assert_eq!(auto_s, mc_s, "Auto must keep MC64 on VESUVIA_0000");
    }

    /// Same shape as `auto_keeps_mc64_on_vesuvia_0000` for the
    /// VESUVIOU subfamily - the highest mc/in ratio in the
    /// validation panel (1.05e4) is on this matrix; the threshold
    /// has 10x margin.
    #[test]
    fn auto_keeps_mc64_on_vesuviou_0000() {
        let path = std::path::Path::new("data/matrices/kkt/VESUVIOU/VESUVIOU_0000.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return, // fixture not present - skip
        };
        let csc = mtx.to_csc().expect("VESUVIOU_0000 CSC build");
        let (auto_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto on VESUVIOU_0000");
        let (mc_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).expect("MC64 on VESUVIOU_0000");
        assert_eq!(auto_s, mc_s, "Auto must keep MC64 on VESUVIOU_0000");
    }

    /// ACOPP30_0064 has raw_drng=1.06e10; MC64 scaling on it produces
    /// a catastrophic factor (zero pivot, rel_ref = 1.74e-1). The
    /// dense-column gate of `pick_scaling_strategy` (max_col_nnz=29
    /// <= 32) routes it to InfNorm directly, without needing the
    /// fallback. Skipped when the fixture is not present.
    #[test]
    fn auto_picks_infnorm_on_acopp30_0064() {
        let path = std::path::Path::new("data/matrices/kkt/ACOPP30/ACOPP30_0064.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return, // fixture not present - skip
        };
        let csc = mtx.to_csc().expect("ACOPP30_0064 CSC build");
        // Routing rule picks InfNorm directly because the
        // dense-column gate is not satisfied (max_col_nnz=29 <= 32).
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::InfNorm);
        // And Auto still resolves to the InfNorm scaling vector.
        let (auto_s, _auto_info) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto on ACOPP30_0064");
        let (in_s, _) =
            compute_scaling(&csc, &ScalingStrategy::InfNorm).expect("InfNorm on ACOPP30_0064");
        let (mc_s, _) =
            compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric).expect("MC64 on ACOPP30_0064");
        assert_eq!(
            auto_s, in_s,
            "Auto must pick InfNorm on ACOPP30_0064 (MC64 produces rel_ref=1.7e-1)"
        );
        assert_ne!(
            auto_s, mc_s,
            "Auto must NOT use MC64 on ACOPP30_0064 (regresses rel_ref to 1.7e-1)"
        );
        // No fallback variant assertion: the routing, not the
        // fallback, handles this matrix. The fallback path is
        // exercised by the
        // synthetic `auto_surfaces_infnorm_spread_fallback_on_uniform_diag`
        // test and the fixture-gated MSS1_0009 test, both of which
        // build/load matrices that still satisfy the (>=0.30 and >32)
        // routing gate.
    }

    /// HS75_0000 has in_spread ~ 20.8, so the IN_SPREAD_GUARD
    /// pre-MC64 InfNorm trial accepts InfNorm before ever calling
    /// MC64. Measured relative residuals on HS75 are 4.20e-17 with InfNorm
    /// and 1.31e-16 with MC64, so InfNorm is the right pick. Skipped
    /// when the fixture is not present.
    #[test]
    fn auto_picks_infnorm_on_hs75_0000() {
        let path = std::path::Path::new("data/matrices/kkt/HS75/HS75_0000.mtx");
        let mtx = match crate::io::mtx::read_mtx(path) {
            Ok(m) => m,
            Err(_) => return, // fixture not present - skip
        };
        let csc = mtx.to_csc().expect("HS75_0000 CSC build");
        let (auto_s, _) = compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto on HS75_0000");
        let (in_s, _) =
            compute_scaling(&csc, &ScalingStrategy::InfNorm).expect("InfNorm on HS75_0000");
        assert_eq!(
            auto_s, in_s,
            "Auto must pick InfNorm on HS75_0000 (in_spread<1e3, InfNorm strictly wins)"
        );
    }

    // ---- MC64 catastrophic-spread guard ----

    /// T1 - `scaling_spread` returns `max|s| / min|s|` over the
    /// nonzero entries. Hand-calculated oracle.
    #[test]
    fn scaling_spread_hand_oracle() {
        // 4.0 / 1e-3 = 4000.
        assert!((scaling_spread(&[1e-3, 1.0, 4.0]) - 4000.0).abs() < 1e-9);
        // Zeros and signs are ignored: 8 / 2 = 4.
        assert!((scaling_spread(&[0.0, -2.0, 8.0, 0.0]) - 4.0).abs() < 1e-12);
        // A vector with no nonzero entry has undefined spread -> +inf.
        assert!(scaling_spread(&[0.0, 0.0]).is_infinite());
    }

    /// T2 - On a saddle-point KKT where MC64 symmetric
    /// scaling produces a vector whose own spread exceeds `1/EPS`,
    /// `Auto` must discard the degenerate MC64 vector and fall back
    /// to the InfNorm vector, tagging the result
    /// `Mc64FallbackToInfnorm{Mc64ScalingDegenerate}`.
    ///
    /// Oracle: the chain block of this matrix (`base = 4.0`) has MC64
    /// spread 3.34e94 (far above `1/EPS ~ 4.50e15`) and InfNorm spread
    /// 2.00e4 (above `IN_SPREAD_GUARD = 1e3`, so the MC64 branch is
    /// genuinely reached). The 120 appended unit slack columns (they
    /// carry the genuine `diag_only` mass the value-aware router
    /// requires) are disconnected from the chain, so neither spread
    /// moves. All three preconditions below re-assert these facts so
    /// the test fails loudly if the oracle ever drifts.
    #[test]
    fn auto_falls_back_on_catastrophic_mc64_spread() {
        let csc = build_synth_kkt(8, 80, 1e8, 4.0, 0.5, 120);
        // Precondition 1: the shape router sends this to MC64.
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
        // Precondition 2: MC64's own scaling spread exceeds the guard.
        let (mc_vec, _) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric)
            .expect("MC64 scaling should succeed");
        let mc_spread = scaling_spread(&mc_vec);
        assert!(
            mc_spread > 1.0 / f64::EPSILON,
            "test oracle invalid: MC64 spread {mc_spread:.3e} must exceed the guard"
        );
        let (in_vec, _) = compute_scaling(&csc, &ScalingStrategy::InfNorm)
            .expect("InfNorm scaling should succeed");
        // Precondition 3: InfNorm spread clears IN_SPREAD_GUARD (1e3),
        // so Auto genuinely reaches the MC64 branch and the new guard
        // rather than short-circuiting on the pre-MC64 InfNorm trial.
        let in_spread = scaling_spread(&in_vec);
        assert!(
            in_spread > 1e3,
            "test oracle invalid: InfNorm spread {in_spread:.3e} must exceed IN_SPREAD_GUARD"
        );
        // The guard must fire: Auto returns the InfNorm vector with
        // the new degenerate-scaling reason.
        let (auto_vec, info) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto scaling should succeed");
        match info {
            ScalingInfo::Mc64FallbackToInfnorm {
                reason: Mc64FallbackReason::Mc64ScalingDegenerate,
            } => {}
            other => {
                panic!("expected Mc64FallbackToInfnorm{{Mc64ScalingDegenerate}}, got {other:?}")
            }
        }
        assert_eq!(auto_vec, in_vec, "fallback must return the InfNorm vector");
        assert_ne!(
            auto_vec, mc_vec,
            "fallback must NOT return the degenerate MC64 vector"
        );
    }

    /// T3 - Non-regression. When MC64's scaling spread is BELOW the
    /// guard, `Auto` must keep the MC64 vector - the guard must not be
    /// over-eager. Same builder as T2 with `base = 1.1`: the chain block
    /// has MC64 spread 9.31e6 (well under `1/EPS`) and InfNorm spread
    /// 1.05e4 (above `IN_SPREAD_GUARD`, so the MC64 branch - and thus
    /// the guard - is genuinely reached rather than short-circuited).
    /// The 120 appended unit slack columns are disconnected and move
    /// neither spread.
    #[test]
    fn auto_keeps_mc64_when_spread_below_guard() {
        let csc = build_synth_kkt(8, 80, 1e8, 1.1, 0.5, 120);
        assert_eq!(pick_scaling_strategy(&csc), ScalingStrategy::Mc64Symmetric);
        let (mc_vec, _) = compute_scaling(&csc, &ScalingStrategy::Mc64Symmetric)
            .expect("MC64 scaling should succeed");
        let mc_spread = scaling_spread(&mc_vec);
        assert!(
            mc_spread < 1.0 / f64::EPSILON,
            "test oracle invalid: MC64 spread {mc_spread:.3e} must be below the guard"
        );
        let in_spread = scaling_spread(
            &compute_scaling(&csc, &ScalingStrategy::InfNorm)
                .expect("InfNorm scaling should succeed")
                .0,
        );
        assert!(
            in_spread > 1e3,
            "test oracle invalid: InfNorm spread {in_spread:.3e} must exceed IN_SPREAD_GUARD"
        );
        let (auto_vec, info) =
            compute_scaling(&csc, &ScalingStrategy::Auto).expect("Auto scaling should succeed");
        assert_eq!(
            auto_vec, mc_vec,
            "Auto must keep the MC64 vector when spread is below the guard"
        );
        assert!(
            !matches!(
                info,
                ScalingInfo::Mc64FallbackToInfnorm {
                    reason: Mc64FallbackReason::Mc64ScalingDegenerate,
                }
            ),
            "the spread guard must not fire below the threshold, got {info:?}"
        );
    }
}
