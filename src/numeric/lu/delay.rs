//! Delayed pivots of the LU factorization: what each front leaves behind
//! for its parent, its ancestors and the emit.
//!
//! The threshold test of a pivot column runs over the whole column of the
//! front, its off-block rows included (`node`). When no fully-summed row
//! passes it, an off-block row holds the only stable pivot: a row of an
//! ancestor, beyond this front's reach. The column and one of the
//! fully-summed rows still unpivoted are then left out of this front's
//! factor, carried up as they stand after its pivots (the Schur complement
//! of the pair) and become fully-summed in the parent, which has those rows
//! and can pivot on them. This is the delayed pivoting of the multifrontal
//! codes (MA41, MUMPS); any front with a parent may delay any column, so a
//! decision depends on the values alone, never on the thread count or the
//! timing.
//!
//! A delayed pair enlarges the parent's front by a row and a column, so the
//! parent outgrows its slot of the arena and factors into buffers of its
//! own (`PanelArena::grow`); its slot stays untouched. The growth over the
//! slots, summed over the forest, is bounded by
//! [`PivotSettings::delay_growth`](crate::PivotSettings::delay_growth) of
//! the factor, which the memory plan reserves. A factorization whose delays
//! would pass it stops (whether the sum passes it does not depend on the
//! order the fronts run in) and the factor is computed again without
//! delays: every pivot in its own front, as before, reported as a warning.

use super::structure::LuStructure;
use crate::numeric::supernodal::Li;

/// What a factored supernode leaves for its parent, its ancestors' updates
/// and the emit.
#[derive(Default)]
pub(super) struct NodePivots {
    /// Pivots taken: the columns of the finished panel.
    pub npiv: usize,
    /// Fully-summed rows and columns (global indices), the pivots first in
    /// pivot order, then those passed to the parent.
    pub rows: Vec<Li>,
    pub cols: Vec<Li>,
    /// With pivots passed on: the finished panels' rows and columns past the
    /// pivots, the passed ones first, then the structure's off-block ones
    /// (what the ancestors' updates read). Empty otherwise: the structure's.
    pub off_l: Vec<Li>,
    pub off_u: Vec<Li>,
    /// Pivots that failed the threshold over their whole column and could
    /// not be delayed.
    pub forced: usize,
}

impl NodePivots {
    /// Pivots passed to the parent.
    #[inline]
    pub(super) fn passed(&self) -> usize {
        self.rows.len() - self.npiv
    }

    /// The pivot count and the rows and columns past the pivots of the
    /// finished panels of supernode `k` (`ncol` columns in the analysis).
    #[inline]
    pub(super) fn off<'a>(
        &'a self,
        st: &'a LuStructure,
        k: usize,
        ncol: usize,
    ) -> (usize, &'a [Li], &'a [Li]) {
        if self.passed() > 0 {
            (self.npiv, &self.off_l, &self.off_u)
        } else {
            (self.npiv, &st.rows_l(k)[ncol..], &st.cols_u(k)[ncol..])
        }
    }
}
