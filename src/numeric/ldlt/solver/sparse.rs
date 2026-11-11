//! Sparse right-hand sides on the `LDL^T` factor: the reach of a sparse
//! column, the solve whose forward sweep visits only it, and the port
//! matrix `Z = B^T A^-1 B` from forward sweeps alone (see
//! `numeric::supernodal::solve::sparse` for the method).

use super::LdltSolver;
use crate::error::RslabError;
#[cfg(test)]
use crate::numeric::ldlt::LdltPivots;
#[cfg(test)]
use crate::numeric::supernodal::solve::SolvePlan;
use crate::numeric::supernodal::solve::SparseReach;
use crate::scalar::Scalar;
use crate::sparse::rhs::SparseRhs;

impl<T: Scalar> LdltSolver<T> {
    fn check_sparse(&self, n: usize) -> Result<(), RslabError> {
        if !self.factored {
            return Err(RslabError::InvalidInput(
                "the last refactorization failed; refactor before solving".to_string(),
            ));
        }
        if n != self.factors.n {
            return Err(RslabError::DimensionMismatch {
                expected: self.factors.n,
                got: n,
            });
        }
        Ok(())
    }

    fn record_sparse(&self, rhs: usize, t: crate::clock::Instant, what: &str) {
        let ms = crate::clock::ms(t.elapsed());
        self.solves.record(rhs, ms, 0);
        if crate::logging::enabled(crate::logging::LogLevel::Debug) {
            crate::logging::debug(&format!(
                "ldlt {what}: n={} rhs={rhs} {ms:.3} ms",
                self.factors.n
            ));
        }
    }

    /// How much of the factor the forward sweep of a right-hand side with
    /// nonzeros in the rows `rows` reaches: the supernodes, unknowns and
    /// panel entries [`solve_sparse`](Self::solve_sparse) and
    /// [`port_matrix`](Self::port_matrix) sweep for it.
    pub fn sparse_reach(&self, rows: &[usize]) -> Result<SparseReach, RslabError> {
        self.check_sparse(self.factors.n)?;
        if let Some(&r) = rows.iter().find(|&&r| r >= self.factors.n) {
            return Err(RslabError::InvalidInput(format!(
                "row {r} out of range for n = {}",
                self.factors.n
            )));
        }
        Ok(self.plan.sparse_reach(&self.factors.perm, rows))
    }

    /// Solve `A X = B` for a sparse block `B`: the dense column-major
    /// `n x ncols` solution (`x[c * n + i]`). The forward sweep of each block
    /// of columns ([`SolveSettings::port_block`](crate::SolveSettings::port_block))
    /// visits only their reach; the backward sweep is the full one of
    /// [`solve_many`](Self::solve_many). Agrees with `solve_many` on the
    /// dense `B` to rounding; the same bits for every thread count.
    pub fn solve_sparse(&self, b: &SparseRhs<T>) -> Result<Vec<T>, RslabError> {
        self.check_sparse(b.n())?;
        if b.ncols() == 0 {
            return Ok(Vec::new());
        }
        let t = crate::clock::Instant::now();
        let x = self
            .plan
            .solve_sparse(&self.factors, &self.scale, b, self.solve_threads)?;
        self.record_sparse(b.ncols(), t, "sparse solve");
        Ok(x)
    }

    /// The port matrix `Z = B^T A^-1 B` (the plain transpose: for a complex
    /// symmetric `A` not the adjoint) of a sparse block `B`, column-major
    /// `ncols x ncols` and exactly symmetric. Computed as `Y^T D^-1 Y` with
    /// `Y = L^-1 P^T S B` from forward sweeps limited to the columns' reach,
    /// with no backward sweep and no dense `n x ncols` block; the 2x2 blocks of
    /// `D` and the equilibration `S` are taken in. Agrees with `B^T X` from
    /// [`solve_many`](Self::solve_many) to rounding. The same bits for every
    /// thread count and every [`SolveSettings::port_block`](crate::SolveSettings::port_block).
    pub fn port_matrix(&self, b: &SparseRhs<T>) -> Result<Vec<T>, RslabError> {
        self.check_sparse(b.n())?;
        let t = crate::clock::Instant::now();
        let z = self
            .plan
            .port_matrix(&self.factors, &self.scale, b, self.solve_threads)?;
        self.record_sparse(b.ncols(), t, "port matrix");
        Ok(z)
    }

    /// The solve plan, the block diagonal and the equilibration (tests).
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (&SolvePlan<T>, &LdltPivots<T>, &[f64]) {
        (&self.plan, &self.factors, &self.scale)
    }
}
