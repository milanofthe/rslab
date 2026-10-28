//! What the three direct solvers share: the solve entry points with their
//! accounting and iterative refinement, and the [`Preconditioner`] and
//! [`Factorization`] impls. A solver provides [`SolveCore`], the raw
//! triangular solves; [`direct_solver!`] builds the rest from it.
//!
//! Right-hand-side blocks are column-major `n x nrhs` (`b[c * n + i]` is
//! row `i` of column `c`), like the Krylov panels.

use crate::diagnostics::SolveCounter;
use crate::error::RslabError;
use crate::refine::{refine_in_place, RefineOperator, RefineOutcome, RefinePolicy};
use crate::scalar::Scalar;

/// The scratch of a solve, lent by the caller: sized by the first solve and
/// reused by every later one, so `solve_into` and its siblings on
/// [`KluSolver`](crate::KluSolver), [`LuSolver`](crate::LuSolver) and
/// [`LdltSolver`](crate::LdltSolver) allocate nothing once warm. One work
/// serves any direct solver and any number of right-hand sides; a thread
/// that solves concurrently keeps its own.
pub struct SolveWork<T> {
    /// The right-hand side block in the factor's order.
    pub(crate) y: Vec<T>,
    /// One row of a block, staged apart from the rows it updates.
    pub(crate) row: Vec<T>,
    /// The supernodal sweeps' scratch.
    pub(crate) plan: crate::numeric::supernodal::solve::PlanWork<T>,
}

impl<T> Default for SolveWork<T> {
    fn default() -> Self {
        Self {
            y: Vec::new(),
            row: Vec::new(),
            plan: Default::default(),
        }
    }
}

impl<T> SolveWork<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The solve works a solver keeps for its own `solve*` calls: a call takes
/// one and gives it back, so every solve after the first allocates nothing
/// and touches no fresh memory (a near-field LU's multi right-hand side
/// solve spent about a fifth of its time setting up a fresh work). As many
/// as calls run at once; held as long as the solver, empty in a clone.
pub(crate) struct WorkPool<T>(std::sync::Mutex<Vec<SolveWork<T>>>);

impl<T> Default for WorkPool<T> {
    fn default() -> Self {
        Self(std::sync::Mutex::new(Vec::new()))
    }
}

impl<T> Clone for WorkPool<T> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<T> std::fmt::Debug for WorkPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WorkPool")
    }
}

impl<T> WorkPool<T> {
    /// `f` on a work of the pool, given back after.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut SolveWork<T>) -> R) -> R {
        let mut w = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default();
        let r = f(&mut w);
        self.0.lock().unwrap_or_else(|e| e.into_inner()).push(w);
        r
    }
}

/// The part of a direct solver the shared entry points build on.
pub(crate) trait SolveCore<T: Scalar> {
    /// The path's name in log records.
    const NAME: &'static str;
    fn dim(&self) -> usize;
    /// `A X = B` (or `A^T X = B`) for a column-major block `b` whose shape
    /// the caller has checked, into `x` of the same shape, without
    /// accounting.
    fn solve_raw_into(
        &self,
        b: &[T],
        nrhs: usize,
        transpose: bool,
        x: &mut [T],
        work: &mut SolveWork<T>,
    ) -> Result<(), RslabError>;
    /// [`solve_raw_into`](Self::solve_raw_into) into a fresh vector.
    fn solve_raw(&self, b: &[T], nrhs: usize, transpose: bool) -> Result<Vec<T>, RslabError> {
        let mut x = vec![T::zero(); b.len()];
        self.works()
            .with(|w| self.solve_raw_into(b, nrhs, transpose, &mut x, w))?;
        Ok(x)
    }
    fn counter(&self) -> &SolveCounter;
    /// The works of its own `solve*` calls ([`WorkPool`]).
    fn works(&self) -> &WorkPool<T>;
    /// The determinant of the matrix that was factored, from the pivots,
    /// the permutations and the scalings.
    fn log_det_raw(&self) -> Result<LogDet<T>, RslabError>;
}

/// A determinant as `phase * exp(log_abs)`: a product of many pivots
/// neither under- nor overflows in this form. `phase` has unit modulus
/// (`+-1` for a real field), or is zero for a zero factor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LogDet<T> {
    phase: T,
    log_abs: f64,
}

impl<T: Scalar> LogDet<T> {
    pub(crate) fn one() -> Self {
        Self {
            phase: T::one(),
            log_abs: 0.0,
        }
    }

    /// Multiply by `v`.
    pub(crate) fn mul(&mut self, v: T) {
        let m = v.magnitude();
        if m == 0.0 {
            self.phase = T::zero();
            self.log_abs = f64::NEG_INFINITY;
            return;
        }
        // Renormalized every step, so the phase stays on the unit circle.
        let p = self.phase * v * T::from_real(1.0 / m);
        self.phase = p * T::from_real(1.0 / p.magnitude().max(f64::MIN_POSITIVE));
        self.log_abs += m.ln();
    }

    /// Divide by `v`.
    pub(crate) fn div(&mut self, v: T) {
        let m = v.magnitude();
        self.mul(v.conj() * T::from_real(1.0 / (m * m)));
    }

    /// Divide by the positive real `s` (a scaling).
    pub(crate) fn div_scale(&mut self, s: f64) {
        self.log_abs -= s.ln();
    }

    /// Multiply by the sign of the permutation `p`.
    pub(crate) fn permutation(&mut self, p: &[usize]) {
        let mut seen = vec![false; p.len()];
        let mut odd = false;
        for start in 0..p.len() {
            let mut i = start;
            let mut len = 0usize;
            while !seen[i] {
                seen[i] = true;
                i = p[i];
                len += 1;
            }
            if len > 0 && len.is_multiple_of(2) {
                odd = !odd;
            }
        }
        if odd {
            self.phase = T::zero() - self.phase;
        }
    }

    pub(crate) fn parts(self) -> (T, f64) {
        (self.phase, self.log_abs)
    }
}

/// [`SolveCore::log_det_raw`], refused for a statically perturbed factor:
/// its determinant is that of the perturbed matrix.
pub(crate) fn log_det<T: Scalar, S: SolveCore<T>>(
    s: &S,
    n_perturbed: usize,
) -> Result<(T, f64), RslabError> {
    if n_perturbed > 0 {
        return Err(RslabError::InvalidInput(format!(
            "{}: {n_perturbed} pivots were lifted by the static floor, the factor's              determinant is not that of the matrix",
            S::NAME
        )));
    }
    Ok(s.log_det_raw()?.parts())
}

/// The dense inverse, column-major, as one block solve against the identity.
pub(crate) fn inverse<T: Scalar, S: SolveCore<T>>(s: &S) -> Result<Vec<T>, RslabError> {
    let n = s.dim();
    let mut e = vec![T::zero(); n * n];
    for i in 0..n {
        e[i * n + i] = T::one();
    }
    solve(s, &e, n, false)
}

fn record<T: Scalar, S: SolveCore<T>>(s: &S, rhs: usize, t: crate::clock::Instant, steps: usize) {
    let ms = crate::clock::ms(t.elapsed());
    s.counter().record(rhs, ms, steps);
    if crate::logging::enabled(crate::logging::LogLevel::Debug) {
        crate::logging::debug(&format!(
            "{} solve: n={} rhs={rhs} refine_steps={steps} {ms:.3} ms",
            S::NAME,
            s.dim()
        ));
    }
}

fn check(n: usize, len: usize, nrhs: usize) -> Result<(), RslabError> {
    if nrhs == 0 || len != n * nrhs {
        return Err(RslabError::DimensionMismatch {
            expected: n * nrhs.max(1),
            got: len,
        });
    }
    Ok(())
}

pub(crate) fn solve_into<T: Scalar, S: SolveCore<T>>(
    s: &S,
    b: &[T],
    nrhs: usize,
    transpose: bool,
    x: &mut [T],
    work: &mut SolveWork<T>,
) -> Result<(), RslabError> {
    check(s.dim(), b.len(), nrhs)?;
    check(s.dim(), x.len(), nrhs)?;
    let t = crate::clock::Instant::now();
    s.solve_raw_into(b, nrhs, transpose, x, work)?;
    record(s, nrhs, t, 0);
    Ok(())
}

pub(crate) fn solve<T: Scalar, S: SolveCore<T>>(
    s: &S,
    b: &[T],
    nrhs: usize,
    transpose: bool,
) -> Result<Vec<T>, RslabError> {
    check(s.dim(), b.len(), nrhs)?;
    let mut x = vec![T::zero(); b.len()];
    s.works()
        .with(|w| solve_into(s, b, nrhs, transpose, &mut x, w))?;
    Ok(x)
}

pub(crate) fn refine_into<T, S, A>(
    s: &S,
    a: &A,
    b: &[T],
    x: &mut [T],
    policy: &RefinePolicy,
) -> Result<RefineOutcome, RslabError>
where
    T: Scalar,
    S: SolveCore<T>,
    A: RefineOperator<T> + ?Sized,
{
    let n = s.dim();
    if a.dim() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: a.dim(),
        });
    }
    check(n, b.len(), 1)?;
    check(n, x.len(), 1)?;
    refine_in_place(a, b, x, policy, |r| s.solve_raw(r, 1, false))
}

pub(crate) fn solve_refined<T, S, A>(
    s: &S,
    a: &A,
    b: &[T],
    policy: &RefinePolicy,
) -> Result<(Vec<T>, RefineOutcome), RslabError>
where
    T: Scalar,
    S: SolveCore<T>,
    A: RefineOperator<T> + ?Sized,
{
    check(s.dim(), b.len(), 1)?;
    let t = crate::clock::Instant::now();
    let mut x = s.solve_raw(b, 1, false)?;
    let outcome = refine_into(s, a, b, &mut x, policy)?;
    record(s, 1, t, outcome.steps);
    Ok((x, outcome))
}

pub(crate) fn refine_into_recorded<T, S, A>(
    s: &S,
    a: &A,
    b: &[T],
    x: &mut [T],
    policy: &RefinePolicy,
) -> Result<RefineOutcome, RslabError>
where
    T: Scalar,
    S: SolveCore<T>,
    A: RefineOperator<T> + ?Sized,
{
    let t = crate::clock::Instant::now();
    let outcome = refine_into(s, a, b, x, policy)?;
    record(s, 0, t, outcome.steps);
    Ok(outcome)
}

/// The shared entry points of a direct solver `$solver<T>` (or
/// `$solver<'a, T>` for one that borrows its analysis): the solves, the
/// refinement, and the `Preconditioner` / `Factorization` impls.
macro_rules! direct_solver {
    ($solver:ident $(<$lt:lifetime>)?) => {
        impl<$($lt,)? T: $crate::scalar::Scalar> $solver<$($lt,)? T> {
            /// Solve `A x = b`.
            pub fn solve(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::solve(self, b, 1, false)
            }

            /// Solve `A X = B` for a column-major `n x nrhs` block (`b[c * n + i]`
            /// is row `i` of column `c`) in one pass over the factor, which
            /// applies each stored value to every column.
            pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::solve(self, b, nrhs, false)
            }

            /// Solve `A^T x = b` on the same factors (the plain transpose; for
            /// the adjoint `A^H x = b` conjugate `b` before and `x` after).
            pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::solve(self, b, 1, true)
            }

            /// [`solve`](Self::solve) into `x`, working in `work`: no
            /// allocation once `work` has served a solve of this size. The
            /// same bits as [`solve`](Self::solve).
            pub fn solve_into(
                &self,
                b: &[T],
                x: &mut [T],
                work: &mut $crate::SolveWork<T>,
            ) -> Result<(), $crate::RslabError> {
                $crate::numeric::direct::solve_into(self, b, 1, false, x, work)
            }

            /// [`solve_many`](Self::solve_many) into the column-major `x`,
            /// working in `work`.
            pub fn solve_many_into(
                &self,
                b: &[T],
                nrhs: usize,
                x: &mut [T],
                work: &mut $crate::SolveWork<T>,
            ) -> Result<(), $crate::RslabError> {
                $crate::numeric::direct::solve_into(self, b, nrhs, false, x, work)
            }

            /// [`solve_transpose`](Self::solve_transpose) into `x`, working in
            /// `work`.
            pub fn solve_transpose_into(
                &self,
                b: &[T],
                x: &mut [T],
                work: &mut $crate::SolveWork<T>,
            ) -> Result<(), $crate::RslabError> {
                $crate::numeric::direct::solve_into(self, b, 1, true, x, work)
            }

            /// Solve `A x = b` with iterative refinement against `a`, the matrix
            /// this factor was computed from (or the exact one it
            /// approximates), under `policy`, reporting the backward error
            /// reached.
            pub fn solve_refined<A: $crate::refine::RefineOperator<T> + ?Sized>(
                &self,
                a: &A,
                b: &[T],
                policy: &$crate::RefinePolicy,
            ) -> Result<(Vec<T>, $crate::RefineOutcome), $crate::RslabError> {
                $crate::numeric::direct::solve_refined(self, a, b, policy)
            }

            /// The determinant of `A` as `(phase, log|det A|)`: `phase` has unit
            /// modulus (`+-1` for a real field) and the log form neither under-
            /// nor overflows. The pivots, the permutations and the scalings are
            /// all accounted for. A factor whose pivots the static floor lifted
            /// ([`n_perturbed`](Self::n_perturbed) `> 0`) belongs to a perturbed
            /// matrix and is refused with [`RslabError::InvalidInput`].
            ///
            /// [`RslabError::InvalidInput`]: $crate::RslabError::InvalidInput
            pub fn log_det(&self) -> Result<(T, f64), $crate::RslabError> {
                $crate::numeric::direct::log_det(self, $solver::n_perturbed(self))
            }

            /// The determinant of `A`, `phase * exp(log|det A|)` of
            /// [`log_det`](Self::log_det): zero or infinite where it under- or
            /// overflows.
            pub fn det(&self) -> Result<T, $crate::RslabError> {
                let (phase, log_abs) = $solver::log_det(self)?;
                Ok(phase * T::from_real(log_abs.exp()))
            }

            /// The dense inverse `A^-1`, column-major `n x n` as
            /// [`solve_many`](Self::solve_many) returns it, from one block solve
            /// against the identity.
            pub fn inverse(&self) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::inverse(self)
            }

            /// Refine an existing iterate `x` of `A x = b` in place.
            pub fn refine_into<A: $crate::refine::RefineOperator<T> + ?Sized>(
                &self,
                a: &A,
                b: &[T],
                x: &mut [T],
                policy: &$crate::RefinePolicy,
            ) -> Result<$crate::RefineOutcome, $crate::RslabError> {
                $crate::numeric::direct::refine_into_recorded(self, a, b, x, policy)
            }
        }

        impl<$($lt,)? T: $crate::scalar::Scalar> $crate::Preconditioner<T>
            for $solver<$($lt,)? T>
        {
            fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), $crate::RslabError> {
                z.copy_from_slice(&$solver::solve(self, r)?);
                Ok(())
            }
            fn apply_block(
                &self,
                r: &[T],
                z: &mut [T],
                s: usize,
                n: usize,
            ) -> Result<(), $crate::RslabError> {
                z[..n * s].copy_from_slice(&$solver::solve_many(self, &r[..n * s], s)?);
                Ok(())
            }
            fn solve_threads(&self) -> $crate::Threads {
                self.solve_threads
            }
        }

        impl<$($lt,)? T: $crate::scalar::Scalar> $crate::Factorization<T>
            for $solver<$($lt,)? T>
        {
            fn n(&self) -> usize {
                $solver::n(self)
            }
            fn solve(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $solver::solve(self, b)
            }
            fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, $crate::RslabError> {
                $solver::solve_many(self, b, nrhs)
            }
            fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $solver::solve_transpose(self, b)
            }
            fn solve_refined(
                &self,
                a: &dyn $crate::refine::RefineOperator<T>,
                b: &[T],
                policy: &$crate::RefinePolicy,
            ) -> Result<(Vec<T>, $crate::RefineOutcome), $crate::RslabError> {
                $solver::solve_refined(self, a, b, policy)
            }
            fn factor_nnz(&self) -> usize {
                $solver::factor_nnz(self)
            }
            fn n_perturbed(&self) -> usize {
                $solver::n_perturbed(self)
            }
            fn log_det(&self) -> Result<(T, f64), $crate::RslabError> {
                $solver::log_det(self)
            }
            fn det(&self) -> Result<T, $crate::RslabError> {
                $solver::det(self)
            }
            fn inverse(&self) -> Result<Vec<T>, $crate::RslabError> {
                $solver::inverse(self)
            }
            fn diagnostics(&self) -> $crate::Diagnostics {
                $solver::diagnostics(self)
            }
        }
    };
}
pub(crate) use direct_solver;
