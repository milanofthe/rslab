//! The block diagonal and the pivot permutation of a sparse LDL^T factor.

use crate::refine::PivotFloor;

/// Everything of `P^T A P = L D L^T` except `L`, which the solve plan holds
/// in panel form: the block diagonal `D`, the permutation and the numeric
/// outcome. The permutation is symmetric, so factoring preserves symmetry.
#[derive(Debug, Clone)]
pub(crate) struct LdltPivots<T> {
    pub n: usize,
    /// Diagonal of the block-diagonal `D`, length `n`.
    pub d_diag: Vec<T>,
    /// Sub-diagonal of `D`, length `n`: `d_subdiag[k]` is the `(k+1, k)`
    /// entry of a 2x2 block starting at column `k`, zero elsewhere.
    pub d_subdiag: Vec<T>,
    /// `true` at the first column of each 2x2 pivot block.
    pub two_by_two: Vec<bool>,
    /// `perm[e]` is the original index eliminated at position `e`.
    pub perm: Vec<usize>,
    /// Parent of every supernode (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// The pivot floors: the factor is of `A + E`, these the entries of `E`
    /// (on the diagonal; a lifted 2x2 block is two).
    pub floors: Vec<PivotFloor<T>>,
    /// Inertia of the factored matrix (advisory for complex symmetric).
    pub inertia: crate::inertia::Inertia,
}

impl<T> LdltPivots<T> {
    /// Heap bytes held: `D`, the permutation, the supernode tree and the
    /// pivot floors.
    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.d_diag)
            + vec_bytes(&self.d_subdiag)
            + vec_bytes(&self.two_by_two)
            + vec_bytes(&self.perm)
            + vec_bytes(&self.supernode_parent)
            + vec_bytes(&self.floors)
    }
}
