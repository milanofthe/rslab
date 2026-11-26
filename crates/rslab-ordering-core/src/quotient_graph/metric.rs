//! Selection-metric trait abstracting AMD vs AMF differences.
//!
//! The elimination loop ([`super::run_elimination`]), the workspace and
//! the final postorder are shared by AMD and AMF. The metrics differ in:
//!
//! 1. **Bucket array length.** AMD: `n` (degrees `0..n` indexable;
//!    `select_pivot` scans `while deg < n`). AMF: `2 * n + 2` because
//!    the quantized RMF can exceed `n`.
//! 2. **Pivot selection.** AMD takes the head of the lowest non-empty
//!    bucket. AMF linear-scans a bucket in the coarse region (scores
//!    above `n` share buckets with stride `PAS = max(n / 8, 1)`).
//! 3. **Element construction.** AMD unlinks absorbed neighbours from
//!    the bucket `degree[i]`, AMF from the quantized bucket of `wf[i]`.
//! 4. **Score finalisation** at the end of Pass-2: AMD's loose-degree
//!    formula `min(deg_prev, scan2_deg) + degme - nvi` clamped at
//!    `nleft - nvi`; AMF's quantized RMF (Amestoy 1999 thesis) with
//!    the supervariable max-merge of `wf`.
//!
//! Sites (2)-(4) are the step functions of `algo.rs` (`select_pivot`,
//! `create_element`, `finalize_step` and their `_amf` counterparts).
//! They stay parallel concrete functions so that the AMD path stays
//! bit-identical to SuiteSparse; the trait selects them statically.
//!
//! Reference: Amestoy (1999) habilitation thesis; MUMPS HAMF4.

use super::algo::{
    create_element, create_element_amf, finalize_step, finalize_step_amf, select_pivot,
    select_pivot_amf, StepFlops,
};
use super::workspace::Workspace;
use crate::OrderingError;

/// Selection metric for an AMD-family bottom-up ordering.
///
/// The methods are the metric-specific steps of
/// [`super::run_elimination`], which is monomorphized per metric.
pub trait Metric {
    /// Length of the bucket head array `Workspace::head`. AMD: `n`
    /// (indexed up to `n - 1` by `select_pivot`'s `while deg < n`).
    /// AMF: `2 * n + 2`.
    fn n_buckets(n: usize) -> usize;

    /// Pick and unlink the next pivot, or `None` if no live bucket is
    /// left.
    fn select_pivot(ws: &mut Workspace) -> Option<usize>;

    /// Build the new element of pivot `me`. Returns
    /// `(pme1, pme2, nvpiv, degme)`.
    fn create_element(
        ws: &mut Workspace,
        me: usize,
    ) -> Result<(usize, usize, i32, usize), OrderingError>;

    /// Update the scores of the element's variables, detect
    /// supervariables and re-insert them into the buckets. Returns the
    /// flops of the step.
    #[allow(clippy::too_many_arguments)]
    fn finalize_step(
        ws: &mut Workspace,
        me: usize,
        pme1: usize,
        pme2_excl: usize,
        nvpiv: i32,
        degme: usize,
        elenme: i32,
        aggressive: bool,
    ) -> StepFlops;
}

/// Minimum-degree metric - the AMD selection rule of Amestoy, Davis,
/// Duff (1996).
///
/// Score is the running degree. Bucket index is the score itself.
/// All buckets are "fine" (head-only pivot selection). Supervariable
/// merge does not update the score (AMD tracks degree only via
/// `nv[i]` and the per-iteration Pass-2 monotone cap).
#[derive(Debug, Clone, Copy, Default)]
pub struct MinDegree;

impl Metric for MinDegree {
    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        n
    }

    #[inline(always)]
    fn select_pivot(ws: &mut Workspace) -> Option<usize> {
        select_pivot(ws)
    }

    #[inline(always)]
    fn create_element(
        ws: &mut Workspace,
        me: usize,
    ) -> Result<(usize, usize, i32, usize), OrderingError> {
        create_element(ws, me)
    }

    #[inline(always)]
    fn finalize_step(
        ws: &mut Workspace,
        me: usize,
        pme1: usize,
        pme2_excl: usize,
        nvpiv: i32,
        degme: usize,
        elenme: i32,
        aggressive: bool,
    ) -> StepFlops {
        finalize_step(ws, me, pme1, pme2_excl, nvpiv, degme, elenme, aggressive)
    }
}

/// Approximate Minimum Fill metric (HAMF4) - Amestoy 1999 thesis.
///
/// AMF selects the next pivot to minimise the *fill* introduced by
/// the elimination, rather than the candidate's degree. On bipartite-
/// KKT graphs with a few "hub" rows AMF can give well over an order of
/// magnitude less `nnz_L` than AMD.
///
/// Score is a quantized `RMF = DEG*(DEG-1+2*DEGME) - WF(i)` value.
/// Buckets up to and including `NORIG = n` are one bucket per integer
/// score; above `NORIG` the buckets quantize with stride
/// `PAS = max(n / 8, 1)` and the head must be linear-scanned to pick
/// the minimum-RMF entry. Supervariable absorption merges the
/// per-supervariable WF with `max`.
///
/// The lazy WF(e) cache, three-accumulator Pass-2, supervariable
/// max-merge of `wf`, saturated/regular RMF branch, and coarse-bucket
/// linear scan all live in the `_amf` step functions of `algo.rs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinFill;

impl Metric for MinFill {
    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        2 * n + 2
    }

    #[inline(always)]
    fn select_pivot(ws: &mut Workspace) -> Option<usize> {
        select_pivot_amf(ws)
    }

    #[inline(always)]
    fn create_element(
        ws: &mut Workspace,
        me: usize,
    ) -> Result<(usize, usize, i32, usize), OrderingError> {
        create_element_amf(ws, me)
    }

    #[inline(always)]
    fn finalize_step(
        ws: &mut Workspace,
        me: usize,
        pme1: usize,
        pme2_excl: usize,
        nvpiv: i32,
        degme: usize,
        elenme: i32,
        aggressive: bool,
    ) -> StepFlops {
        finalize_step_amf(ws, me, pme1, pme2_excl, nvpiv, degme, elenme, aggressive)
    }
}
