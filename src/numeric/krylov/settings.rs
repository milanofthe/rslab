//! The settings of the Krylov solvers.

use crate::constants::{
    KRYLOV_BASIS_BUDGET_BYTES, KRYLOV_MAX_ITER, KRYLOV_ORTHO_CHUNK, KRYLOV_REORTH_ETA,
    KRYLOV_RESTART_MAX, KRYLOV_RESTART_MIN, KRYLOV_TOL,
};
use crate::scalar::Scalar;
use crate::stored::Stored;

/// Settings of [`gmres`](super::gmres), [`gmres_block`](super::gmres_block),
/// [`gmres_recycled`](super::gmres_recycled), [`cocg`](super::cocg) and
/// [`cocr`](super::cocr). [`BlockArnoldi`](super::BlockArnoldi) reads
/// [`reorth_eta`](Self::reorth_eta) and [`ortho_chunk`](Self::ortho_chunk) only.
///
/// ```
/// use rslab::KrylovSettings;
/// let s = KrylovSettings::default().with_tol(1e-10).with_restart(60);
/// assert_eq!(s.restart, Some(60));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KrylovSettings {
    /// Target of the relative residual `||b - A x|| / ||b||`, per column.
    /// Default `1e-8`.
    pub tol: f64,
    /// Iteration budget per solve; a block GMRES gives it to every column of
    /// every [`BlockGmres::solve`](super::BlockGmres::solve) call. Default `400`.
    pub max_iter: usize,
    /// GMRES restart length (ignored by COCG and COCR). `None` (default)
    /// takes the largest length in `restart_min..=restart_max` whose Krylov
    /// bases fit [`basis_budget_bytes`](Self::basis_budget_bytes).
    pub restart: Option<usize>,
    /// Memory cap of the Krylov bases for the default restart length.
    /// Default 1 GiB.
    pub basis_budget_bytes: usize,
    /// Shortest default restart length. Default `20`.
    pub restart_min: usize,
    /// Longest default restart length. Default `80`.
    pub restart_max: usize,
    /// A new basis vector is orthogonalized a second time when the first
    /// pass keeps less than this fraction of its norm (the DGKS criterion).
    /// Default `1/sqrt(2)`.
    pub reorth_eta: f64,
    /// Rows per chunk of the parallel orthogonalization sums of block GMRES and
    /// block Arnoldi.
    /// The chunks are summed in a fixed order, so the result depends on this
    /// value but not on the thread count. Default `2048`.
    pub ortho_chunk: usize,
    /// The preconditioner may differ from step to step (flexible GMRES): every
    /// column of a block GMRES keeps its preconditioned basis `Z = M^-1 V`
    /// beside `V` and forms its solution `x0 + Z y` from it. `false`, for a
    /// preconditioner that is one fixed operator (a factorization): only `V` is
    /// kept and the solution formed as `x0 + M^-1 (V y)`, one batched
    /// preconditioner apply per residual check; the bases take half the memory.
    /// Default `true`.
    pub flexible: bool,
    /// A block GMRES stores its bases `V` and `Z` in the single-precision twin of
    /// its scalar ([`Stored`](crate::Stored): `Complex<f32>` for `Complex<f64>`,
    /// `f32` for `f64`), half their memory, and reads them through the full
    /// precision: every product, the Hessenberg matrices, the rotations, the
    /// residuals and the solutions stay in it. A cycle's true residual then stalls
    /// at the rounding of its stored vectors, relative to the residual it started
    /// from (`1.3e-8` on a grid where full bases reach `3e-14`). Above
    /// [`single_for`](crate::single_for)'s tolerance (`9.5e-7`) that never shows;
    /// below it a column found above the tolerance restarts from its true residual,
    /// at the cost of more iterations (34 against 28 to `1e-10` on a grid restarting
    /// every 30 steps). Fixed at [`BlockGmres::new`](super::BlockGmres::new) for
    /// every call of the solve. The default restart
    /// ([`basis_budget_bytes`](Self::basis_budget_bytes)) and the memory
    /// ([`room_bytes`](Self::room_bytes), [`block_bytes`](Self::block_bytes)) count
    /// the bytes stored. Without effect for the single fields (their own twin).
    /// Default `false`.
    pub basis_single: bool,
    /// The memory a block GMRES may hold (bytes, [`block_bytes`](Self::block_bytes)):
    /// where the bases would outgrow it, the columns with the worst residual
    /// estimates close their cycles and wait while the others run on, and start
    /// again once those finished ([`BlockGmres::solve`](super::BlockGmres::solve)).
    /// Every column keeps the restart; the room bounds how many run at once.
    /// A live bound, the memory the process has left, is
    /// [`BlockGmres::within`](super::BlockGmres::within). `None` (default): no bound.
    pub room_bytes: Option<u64>,
}

impl Default for KrylovSettings {
    fn default() -> Self {
        Self {
            tol: KRYLOV_TOL,
            max_iter: KRYLOV_MAX_ITER,
            restart: None,
            basis_budget_bytes: KRYLOV_BASIS_BUDGET_BYTES,
            restart_min: KRYLOV_RESTART_MIN,
            restart_max: KRYLOV_RESTART_MAX,
            reorth_eta: KRYLOV_REORTH_ETA,
            ortho_chunk: KRYLOV_ORTHO_CHUNK,
            flexible: true,
            basis_single: false,
            room_bytes: None,
        }
    }
}

impl KrylovSettings {
    /// Set the residual target.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the iteration budget.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fix the GMRES restart length.
    pub fn with_restart(mut self, restart: usize) -> Self {
        self.restart = Some(restart);
        self
    }
    /// Whether the preconditioner may change between steps ([`flexible`](Self::flexible)).
    pub fn with_flexible(mut self, flexible: bool) -> Self {
        self.flexible = flexible;
        self
    }
    /// Store the bases of a block GMRES in single precision ([`basis_single`](Self::basis_single)).
    pub fn with_basis_single(mut self, single: bool) -> Self {
        self.basis_single = single;
        self
    }
    /// Bound the memory of a block GMRES ([`room_bytes`](Self::room_bytes)).
    pub fn with_room(mut self, bytes: u64) -> Self {
        self.room_bytes = Some(bytes);
        self
    }
    /// The bases a block GMRES column stores: `V` and `Z` when
    /// [`flexible`](Self::flexible), `V` alone otherwise.
    fn bases(&self) -> usize {
        if self.flexible {
            2
        } else {
            1
        }
    }

    /// The restart of a block GMRES of `s` right-hand sides of length `n` in `T`,
    /// its bases as stored.
    pub(crate) fn block_restart<T: Scalar>(&self, n: usize, s: usize) -> usize {
        let stored = Stored::<T>::scalar_bytes(self.basis_single);
        self.restart_for(n, s, stored, self.bases())
    }

    /// Bytes a step adds to the bases of a block GMRES column of length `n` in `T`.
    pub(crate) fn layer_bytes<T: Scalar>(&self, n: usize) -> u64 {
        (n * self.bases() * Stored::<T>::scalar_bytes(self.basis_single)) as u64
    }

    /// Heap bytes a block GMRES ([`gmres_block`](super::gmres_block),
    /// [`BlockGmres`](super::BlockGmres)) of `s` right-hand sides of length `n`
    /// in scalars `T` holds once its bases are `depth` vectors deep (a solve of
    /// `k` iterations without a restart: `k + 1`), at the restart these settings
    /// give: the bases `V` and (when [`flexible`](Self::flexible)) `Z` as
    /// written and stored ([`basis_single`](Self::basis_single)), the Hessenberg
    /// matrices, the panels and the reduction scratch;
    /// within [`room_bytes`](Self::room_bytes) where set (at least one column a step
    /// deep beside the rest).
    pub fn block_bytes<T: Scalar>(&self, n: usize, s: usize, depth: usize) -> u64 {
        let stored = Stored::<T>::scalar_bytes(self.basis_single);
        let m = self.block_restart::<T>(n, s);
        let plan = |d: usize| {
            super::block::block_bytes::<T>(n, s, m, self.ortho_chunk, d, self.flexible, stored)
        };
        let full = plan(depth.min(m + 1));
        match self.room_bytes {
            Some(room) => {
                // one column two vectors deep: its start and a step
                let column = (plan(2) - plan(0)) / s as u64;
                full.min(room.max(plan(0) + column))
            }
            None => full,
        }
    }

    /// What deflating a subspace of dimension `k`
    /// ([`BlockGmres::deflating`](super::BlockGmres::deflating)) adds to
    /// [`block_bytes`](Self::block_bytes): the subspace, every column's
    /// components on it per step, and a panel's projections.
    pub fn deflation_bytes<T: Scalar>(&self, n: usize, s: usize, k: usize) -> u64 {
        let m = self.block_restart::<T>(n, s);
        super::block::deflation_bytes::<T>(n, s, m, self.ortho_chunk, k)
    }

    /// The restart length for `n` unknowns, `columns` right-hand sides of
    /// `scalar_bytes` each and `bases` stored bases (two for flexible GMRES).
    pub(crate) fn restart_for(
        &self,
        n: usize,
        columns: usize,
        scalar_bytes: usize,
        bases: usize,
    ) -> usize {
        if let Some(r) = self.restart {
            return r.max(1);
        }
        let per_vector = n
            .saturating_mul(columns)
            .saturating_mul(scalar_bytes)
            .saturating_mul(bases);
        let max = self.restart_max.max(1);
        if per_vector == 0 {
            return max;
        }
        (self.basis_budget_bytes / per_vector)
            .saturating_sub(1)
            .clamp(self.restart_min.clamp(1, max), max)
    }
}

#[cfg(test)]
mod tests {
    use super::KrylovSettings;

    /// The default restart is the longest whose bases fit the budget,
    /// within the bounds; an explicit one is kept.
    #[test]
    fn the_default_restart_fits_the_basis_budget() {
        let s = KrylovSettings::default();
        assert_eq!(s.restart_for(1000, 1, 16, 2), 80);
        // 32 MB per basis vector: 33 fit into 1 GiB, one is the spare.
        assert_eq!(s.restart_for(500_000, 2, 16, 2), 32);
        assert_eq!(s.restart_for(50_000_000, 1, 16, 2), 20);
        assert_eq!(
            s.clone().with_restart(7).restart_for(50_000_000, 1, 16, 2),
            7
        );
    }
}
