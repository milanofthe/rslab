//! The LU factor types: the supernodal panel form of a numeric factorization
//! and the permutations, scalings and counters the solver keeps next to it.

use crate::numeric::supernodal::panel::PanelFactor;
use crate::refine::PivotFloor;
use crate::scalar::Scalar;

/// Everything of `P_r^T A P = L U` except the panels: the permutations, the
/// equilibration and the outcome, `perm[e]` mapping factorization position
/// `e` to the original index.
#[derive(Clone)]
pub(crate) struct LuPivots<T> {
    pub n: usize,
    /// Column permutation: factorization position -> original column index
    /// (`P^T A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position -> original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `A_hat = diag(d_row)*A*diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
    /// Parent of every supernode in the assembly tree (`usize::MAX` for a
    /// root); empty when unknown.
    pub supernode_parent: Vec<usize>,
    /// The pivot floors: the factor is of `A + E`, these the entries of `E`.
    pub floors: Vec<PivotFloor<T>>,
}

impl<T> LuPivots<T> {
    /// Heap bytes held: the permutations, the scalings, the supernode tree
    /// and the pivot floors.
    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.perm)
            + vec_bytes(&self.perm_row)
            + vec_bytes(&self.d_row)
            + vec_bytes(&self.d_col)
            + vec_bytes(&self.supernode_parent)
            + vec_bytes(&self.floors)
    }
}

/// The numeric result of a sparse LU factorization: the unit lower `L` and
/// the transposed upper factor `U^T` (its diagonal in the panel) in
/// supernodal panel form (the storage the solves run on, written by the
/// drivers without a copy), the permutations, the scalings and the outcome.
pub(crate) struct LuNumeric<T> {
    /// `L` in panel form, in elimination order (unit diagonal implicit).
    pub l: PanelFactor<T>,
    /// `U^T` in panel form: column `c` of the panel is row `c` of `U`, the
    /// diagonal of `U` at the panel's diagonal.
    pub ut: PanelFactor<T>,
    /// `perm[e]` is the original column eliminated at position `e`.
    pub perm: Vec<usize>,
    /// `perm_row[e]` is the original row that became pivot row `e`.
    pub perm_row: Vec<usize>,
    /// Row scaling applied before the factorization.
    pub d_row: Vec<f64>,
    /// Column scaling applied before the factorization.
    pub d_col: Vec<f64>,
    /// Supernode tree over the factor's supernodes (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// The pivot floors, in `A`'s indices (`B`'s rows until the row
    /// matching is folded back).
    pub floors: Vec<PivotFloor<T>>,
    /// Structural panel slots holding an exact zero (the symmetrized
    /// pattern or cancellation); the stored nonzeros are
    /// `l.nnz() + ut.nnz() - n_zeros`.
    pub n_zeros: usize,
    /// Pivots a front passed to its parent (one delayed twice counts twice).
    pub delayed: usize,
    /// Pivots that failed the threshold over their whole column and stayed
    /// in their front.
    pub forced: usize,
    /// The delayed pivots exceeded their allowance: the factor is the one
    /// without them.
    pub delays_dropped: bool,
}

impl<T: Scalar> LuNumeric<T> {
    /// Stored fill `nnz(L) + nnz(U)`: the structural panel entries minus the
    /// slots holding an exact zero.
    pub(crate) fn factor_nnz(&self) -> usize {
        self.l.nnz() + self.ut.nnz() - self.n_zeros
    }

    /// Bytes of the two panel factors.
    pub(crate) fn bytes(&self) -> usize {
        self.l.bytes() + self.ut.bytes()
    }

    /// Split into the two panel factors and the [`LuPivots`] the solver
    /// keeps next to them.
    pub(crate) fn into_parts(self) -> (PanelFactor<T>, PanelFactor<T>, LuPivots<T>) {
        let pivots = LuPivots {
            n: self.l.n,
            perm: self.perm,
            perm_row: self.perm_row,
            d_row: self.d_row,
            d_col: self.d_col,
            supernode_parent: self.supernode_parent,
            floors: self.floors,
        };
        (self.l, self.ut, pivots)
    }
}
