//! The solvers over closures: matrix-free operators and preconditioners
//! given as functions.

use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// Adapter: a closure block-matvec `op(x, y, s)` (`Y <- A*X`, column-major `nxs`) as a
/// [`LinearOperator`] for the matrix-free call path. `FnMut` (the Arnoldi issues applies
/// sequentially) so the operator's own scratch lives in the closure capture - no struct, no
/// interior-mutability dance at the call site. The `RefCell` is borrowed for one apply at a time.
struct FnOp<F> {
    f: std::cell::RefCell<F>,
    n: usize,
}
impl<T: Scalar, F: FnMut(&[T], &mut [T], usize)> LinearOperator<T> for FnOp<F> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        (self.f.borrow_mut())(x, y, 1)
    }
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        (self.f.borrow_mut())(x, y, s)
    }
}

/// Adapter: a closure block-preconditioner `pc(r, z, s)` (`Z <- M^-1*R`) as a [`Preconditioner`].
struct FnPc<G> {
    f: std::cell::RefCell<G>,
}
impl<T: Scalar, G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>> Preconditioner<T>
    for FnPc<G>
{
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, 1)
    }
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, _n: usize) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, s)
    }
}

/// Closure entry point for [`gmres_block`]: pass the block matvec and block preconditioner as
/// `FnMut` closures plus the dimension `n` - the natural form for a **matrix-free** MoM/FEM
/// operator that captures its own assembly data + scratch, with no `LinearOperator`/`Preconditioner`
/// boilerplate. For an unpreconditioned solve pass `|r, z, _| { z.copy_from_slice(r); Ok(()) }`.
#[allow(clippy::too_many_arguments)]
pub fn gmres_block_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    s: usize,
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres_block(&op, b, s, &pc, tol, max_iter, restart, None)
}

/// Closure entry point for [`gmres_block_mon`], [`gmres_block_fn`] plus the per-cycle
/// progress monitor and an optional WARM START `x0` (column-major `nxs`, like `b`;
/// `None` => `x_0 = 0`). Seeding with a nearby solution (e.g. the previous frequency
/// of a sweep) starts from its residual; convergence stays relative to `||b||`.
#[allow(clippy::too_many_arguments)]
pub fn gmres_block_fn_mon<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    s: usize,
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
    mon: Option<&mut dyn FnMut(usize, f64, usize) -> bool>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres_block_mon(&op, b, s, &pc, tol, max_iter, restart, x0, mon)
}

/// Closure entry point for [`gmres_recycled`] (single RHS, GCRO-DR): the operator and
/// preconditioner as `FnMut` closures, the recycle handle updated in place - see [`gmres_fn`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_recycled_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
    recycle: &mut Recycle<T>,
) -> Result<KrylovResult<T>, RslabError>
where
    T: RecycleScalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres_recycled(&op, b, &pc, tol, max_iter, restart, x0, recycle)
}

/// Closure entry point for [`gmres`] (single RHS) - see [`gmres_block_fn`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres(&op, b, &pc, tol, max_iter, restart, None)
}
