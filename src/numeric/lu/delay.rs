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
//! A delayed pair enlarges the parent's front by a row and a column, and by
//! the rows and columns past the child it carries that the parent's
//! structure lacks (on a structurally unsymmetric pattern, as under the row
//! matching); an ancestor that takes updates from a front with such rows
//! takes the rows too (the rule of the structure, `structure`). So where
//! pivots may be delayed every front claims its slots of the arena when it
//! starts, at the size it has then (`PanelArena::lazy`), and only what the
//! fronts hold is ever resident. The arenas' room is the analysis's panels
//! and [`LU_DELAY_ALLOWANCE`](crate::constants::LU_DELAY_ALLOWANCE) of
//! them on top, which the memory plan reserves. A factorization whose
//! fronts would need more stops (whether they do does not depend on the
//! order the fronts run in) and the factor is computed again without
//! delays: every pivot in its own front, as before, reported as a warning.

use super::structure::LuStructure;
use crate::numeric::supernodal::Li;

pub(super) struct NodePivots<T> {
    /// Pivots taken: the columns of the finished panel.
    pub npiv: usize,
    /// Fully-summed rows and columns (global indices), the pivots first in
    /// pivot order, then those passed to the parent.
    pub rows: Vec<Li>,
    pub cols: Vec<Li>,
    /// With pivots passed on or rows and columns beyond the structure's:
    /// the finished panels' rows and columns past the pivots, the passed
    /// ones first, then the off-block ones (what the ancestors' updates
    /// read). Empty otherwise: the structure's.
    pub off_l: Vec<Li>,
    pub off_u: Vec<Li>,
    /// Its off-block rows or columns go beyond the structure's: the
    /// delayed pivots it took brought them, or an updater passed them on.
    pub extras: bool,
    /// Pivots that failed the threshold over their whole column and could
    /// not be delayed.
    pub forced: usize,
    /// The passed pivots as they stand after this front's pivots, for the
    /// parent to take: per passed column its entries over the rows past
    /// the pivots (`off_l`'s order), then per passed row its entries over
    /// the columns past the fully-summed ones (`off_u` past the passed
    /// columns).
    pub passed_cols: Vec<T>,
    pub passed_rows: Vec<T>,
}

impl<T> Default for NodePivots<T> {
    fn default() -> Self {
        NodePivots {
            npiv: 0,
            rows: Vec::new(),
            cols: Vec::new(),
            off_l: Vec::new(),
            off_u: Vec::new(),
            extras: false,
            forced: 0,
            passed_cols: Vec::new(),
            passed_rows: Vec::new(),
        }
    }
}

impl<T> crate::memory::HeapBytes for crate::numeric::supernodal::Cells<NodePivots<T>> {
    fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes as b;
        let records = (self.len() * std::mem::size_of::<NodePivots<T>>()) as u64;
        // SAFETY: called on a pool's own object, no factorization fills it.
        records
            + (0..self.len())
                .map(|s| {
                    let p = unsafe { self.get(s) };
                    b(&p.rows)
                        + b(&p.cols)
                        + b(&p.off_l)
                        + b(&p.off_u)
                        + b(&p.passed_cols)
                        + b(&p.passed_rows)
                })
                .sum::<u64>()
    }
}

impl<T> NodePivots<T> {
    #[inline]
    pub(super) fn passed(&self) -> usize {
        self.rows.len() - self.npiv
    }

    #[inline]
    pub(super) fn off<'a>(
        &'a self,
        st: &'a LuStructure,
        k: usize,
        ncol: usize,
    ) -> (usize, &'a [Li], &'a [Li]) {
        if self.passed() > 0 || self.extras {
            (self.npiv, &self.off_l, &self.off_u)
        } else {
            (self.npiv, &st.rows_l(k)[ncol..], &st.cols_u(k)[ncol..])
        }
    }
}
