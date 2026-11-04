//! Block GMRES for many right-hand sides with CGS2 over the whole panel, as a
//! solve that can be continued to a tighter tolerance in the same Krylov space.

use super::util::*;
use super::*;

use super::recycle::RecycleScalar;
use super::settings::{BasisSwitch, BlockBasis};
use crate::error::RslabError;
use crate::scalar::Scalar;
use crate::stored::Stored;

/// Outcome of a block (multi-RHS) Krylov solve.
#[derive(Debug, Clone)]
pub struct BlockKrylovResult<T> {
    /// Solutions, column-major `nxs` (RHS `c` is the slice `[c*n, (c+1)*n)`).
    pub x: Vec<T>,
    /// Block iterations performed: steps of the batched applies, each advancing
    /// every column stepping.
    pub iters: usize,
    /// `true` iff **every** RHS reached `tol`.
    pub converged: bool,
    /// Per-RHS final relative residual `||b_c - A x_c|| / ||b_c||`.
    pub final_res: Vec<f64>,
    /// Why the block iteration stopped: `Converged` when every RHS met `tol`,
    /// `Stalled` when the monitor cancelled it, else `MaxIter`.
    pub stop: StopReason,
    /// Cycles closed before the restart because the bases filled the memory
    /// ([`KrylovSettings::room_bytes`], [`BlockGmres::within`]), over all calls so far.
    pub closed: usize,
    /// The representation the bases ended in.
    pub basis: BlockBasis,
    /// The representations they were lowered to for memory and when
    /// ([`KrylovSettings::downgrade`]), over all calls so far.
    pub lowered: Vec<BasisSwitch>,
}

/// What the monitor of a block GMRES ([`BlockGmres::solve`]) receives after every
/// check of true residuals.
#[derive(Debug, Clone, Copy)]
pub struct BlockProgress<'p> {
    /// Block iterations of this call.
    pub iterations: usize,
    /// The worst over the columns not waiting: their last true residuals and
    /// their open cycles' estimates, a non-finite one counting as infinite.
    pub worst: f64,
    /// The columns stepping next.
    pub active: usize,
    /// The columns waiting for memory ([`KrylovSettings::room_bytes`],
    /// [`BlockGmres::within`]).
    pub waiting: usize,
    /// The columns whose true residual this check formed.
    pub checked: &'p [usize],
    /// The columns of `checked` whose cycles close for memory short of the
    /// restart: the check ends no cycle of theirs.
    pub closing: &'p [usize],
    /// Every column's last true relative residual, `None` once it stepped since.
    pub residuals: &'p [Option<f64>],
    /// Heap bytes the solve holds ([`BlockGmres::resident_bytes`]).
    pub resident: u64,
    /// The representation of the bases.
    pub basis: BlockBasis,
    /// The representations they were lowered to for memory and when
    /// ([`KrylovSettings::downgrade`]), over all calls so far.
    pub lowered: &'p [BasisSwitch],
}

/// What the step monitor of a block GMRES ([`BlockGmres::solve_watched`])
/// receives after every block step: the residuals the open cycles would hand
/// out now (their least-squares estimates, in exact arithmetic the true
/// residuals), between the checks of true residuals.
#[derive(Debug, Clone, Copy)]
pub struct StepProgress<'p> {
    /// Block iterations of this call.
    pub iterations: usize,
    /// The worst over the columns not waiting: an open cycle's estimate, else
    /// the last true residual, a non-finite one counting as infinite.
    pub worst: f64,
    /// The columns stepping next.
    pub active: usize,
    /// Per column its estimate where its cycle is open, else its last true
    /// residual (`None` where it has none yet).
    pub estimates: &'p [Option<f64>],
    /// The representation of the bases.
    pub basis: BlockBasis,
}

/// One right-hand side's Arnoldi state: its basis, the Hessenberg matrix under
/// its Givens rotations and the rotated least-squares right-hand side of the
/// open restart cycle.
struct Column<T: Scalar> {
    /// The basis, layer `i` the scalars `[i*n, (i+1)*n)`: the restart's layers
    /// reserved, the ones the cycle has written resident ([`Stored`]).
    v: Stored<T>,
    /// The preconditioned basis `z_i = M^-1 v_i` of the steps taken, laid out as
    /// `v`: the solution is formed from the vectors the Arnoldi relation holds.
    /// Empty when the preconditioner is fixed ([`KrylovSettings::flexible`]
    /// `false`): `Z y = M^-1 (V y)` then.
    zb: Stored<T>,
    /// Row-major `(m+1) x m`, entry `(i, j)` at `h[i*m + j]`.
    h: Vec<T>,
    /// The same before the Givens rotations: `A Z = V H_bar` of the open cycle
    /// (`+ C B` when deflating), the matrix [`BlockGmres::harvest`] reads.
    hraw: Vec<T>,
    /// Deflating ([`BlockRecycle`] of dimension `k`): step `j`'s components
    /// `C^H A z_j` at `bm[j*k..(j+1)*k]`.
    bm: Vec<T>,
    cs: Vec<T>,
    sn: Vec<T>,
    g: Vec<T>,
    /// Steps of the open cycle; `None` before the first and after a cycle closed.
    steps: Option<usize>,
    /// Steps taken in the current [`BlockGmres::solve`] call.
    stepped: usize,
    /// Its cycle closed for room: it waits, its basis released, for the next
    /// wave ([`BlockGmres::evict`]).
    waits: bool,
    /// Its cycle closes at the next check, its basis released.
    evict: bool,
}

impl<T: Scalar> Column<T> {
    /// No basis yet: [`open`](Self::open) reserves it.
    fn new(m: usize, k: usize) -> Self {
        Column {
            v: Stored::default(),
            zb: Stored::default(),
            h: vec![T::zero(); (m + 1) * m],
            hraw: vec![T::zero(); (m + 1) * m],
            bm: vec![T::zero(); m * k],
            cs: vec![T::zero(); m],
            sn: vec![T::zero(); m],
            g: vec![T::zero(); m + 1],
            steps: None,
            stepped: 0,
            waits: false,
            evict: false,
        }
    }

    /// Whether it holds a basis.
    fn holds(&self) -> bool {
        self.v.capacity() > 0
    }

    /// Heap bytes of its bases as written.
    fn basis_bytes(&self) -> u64 {
        self.v.bytes() + self.zb.bytes()
    }

    /// The bytes its bases grow by from `j` steps of the open cycle to `to`, of
    /// `per` bytes a step: the layers an earlier cycle wrote are reused.
    fn growth(&self, n: usize, j: usize, to: usize, per: u64) -> u64 {
        let written = (self.v.len() / n).saturating_sub(1).max(j);
        to.saturating_sub(written) as u64 * per
    }

    /// Release its bases: no cycle open.
    fn release(&mut self) {
        self.v = Stored::default();
        self.zb = Stored::default();
        self.steps = None;
    }

    /// The least-squares residual of the open cycle relative to `bnorm`: the
    /// residual of the solution it would hand out now, in exact arithmetic.
    fn estimate(&self, bnorm: f64) -> f64 {
        self.steps
            .map_or(f64::INFINITY, |j| self.g[j].magnitude() / bnorm)
    }

    /// Open a cycle on the residual `r` of norm `beta > 0` (scaled to the first basis
    /// vector in place), the restart's layers of its bases reserved where it holds
    /// none (as `basis` stores them).
    fn open(&mut self, r: &mut [T], beta: f64, basis: BlockBasis) {
        let n = r.len();
        if !self.holds() {
            let m = self.cs.len();
            self.v = Stored::with_capacity(n * (m + 1), basis.single);
            if basis.flexible {
                self.zb = Stored::with_capacity(n * m, basis.single);
            }
        }
        self.h.iter_mut().for_each(|e| *e = T::zero());
        self.g.iter_mut().for_each(|e| *e = T::zero());
        self.g[0] = T::from_real(beta);
        let inv = T::from_real(1.0 / beta);
        r.iter_mut().for_each(|ri| *ri = *ri * inv);
        self.v.write(0, r);
        self.steps = Some(0);
    }

    /// The open cycle's coefficients `y` (back substitution on the leading
    /// well-conditioned block, so a rank-deficient Hessenberg breaks down
    /// deterministically instead of dividing by about zero); returns how many.
    #[allow(clippy::needless_range_loop)]
    fn coefficients(&self, m: usize, y: &mut [T]) -> usize {
        let jd = well_conditioned_dim(&self.h, m, self.steps.unwrap_or(0));
        for i in (0..jd).rev() {
            let mut acc = self.g[i];
            for k in (i + 1)..jd {
                acc = acc - self.h[i * m + k] * y[k];
            }
            y[i] = acc * self.h[i * m + i].recip();
        }
        jd
    }

    /// `out += B y`: the first `jd` vectors of the basis `b` combined by `y`.
    fn combine(b: &Stored<T>, jd: usize, y: &[T], out: &mut [T]) {
        let n = out.len();
        for (i, &yi) in y.iter().enumerate().take(jd) {
            b.axpy(i * n, yi, out);
        }
    }

    /// `out -= U B y` for the subspace `rec` deflated: the part of the open
    /// cycle's correction in its directions.
    fn deflated(&self, rec: &BlockRecycle<T>, jd: usize, y: &[T], out: &mut [T]) {
        let n = out.len();
        let k = rec.k;
        for i in 0..k {
            let t = (0..jd).fold(T::zero(), |a, j| a + self.bm[j * k + i] * y[j]);
            if t != T::zero() {
                for (o, &u) in out.iter_mut().zip(&rec.u[i * n..(i + 1) * n]) {
                    *o = *o - u * t;
                }
            }
        }
    }

    /// The open cycle's solution `x0 + Z y` into `out` from the stored
    /// preconditioned basis, deflating `rec`: `x0 + Z y - U B y`.
    fn correction(
        &self,
        m: usize,
        rec: Option<&BlockRecycle<T>>,
        y: &mut [T],
        x0: &[T],
        out: &mut [T],
    ) {
        out.copy_from_slice(x0);
        let jd = self.coefficients(m, y);
        Self::combine(&self.zb, jd, y, out);
        if let Some(rec) = rec {
            self.deflated(rec, jd, y, out);
        }
    }
}

/// A subspace a block GMRES deflates (GCRO): `k` directions `u` in the solution
/// space and their images `c = A u` under the operator they were taken on,
/// orthonormal (each column-major `n x k`). A solve of the same operator with it
/// ([`BlockGmres::deflating`]) starts every cycle from the residual's part
/// outside `range(C)` and keeps its Krylov vectors orthogonal to `C`, so the
/// slow directions the subspace holds are not found again. Taken from a solve by
/// [`BlockGmres::harvest`]; valid only for the operator of that solve (`c = A u`
/// is not recomputed), so it serves solves of one operator with different
/// right-hand sides, not a sequence of operators.
#[derive(Clone, Debug)]
pub struct BlockRecycle<T> {
    n: usize,
    k: usize,
    u: Vec<T>,
    c: Vec<T>,
}

impl<T> BlockRecycle<T> {
    /// The subspace's dimension.
    pub fn dim(&self) -> usize {
        self.k
    }

    /// The directions and their images, column-major `n x k` each.
    pub fn parts(&self) -> (&[T], &[T]) {
        (&self.u, &self.c)
    }

    /// Heap bytes of the directions and their images.
    pub fn heap_bytes(&self) -> u64 {
        crate::memory::vec_bytes(&self.u) + crate::memory::vec_bytes(&self.c)
    }
}

/// A right-preconditioned block GMRES solve of `A X = B` that can be continued:
/// [`solve`](Self::solve) runs every right-hand side to a tolerance and hands out
/// the solutions, a later call with a tighter tolerance carries on in the same
/// Krylov space instead of restarting from the solution handed out. A ladder of
/// tolerances (solve, inspect, tighten) then costs about what one solve to the
/// last tolerance costs; a fresh GMRES per rung rebuilds most of the space (a
/// 79 k unknown MoM system: 26 iterations for the rung from `1e-6` to `1e-8`
/// restarted, 8 continued).
///
/// Each right-hand side keeps its own basis, Hessenberg matrix and Givens
/// rotations, so a column converges as the single-RHS [`gmres`] does; the
/// columns share the batched operator and preconditioner applies of every step.
///
/// **Flexible:** like [`gmres`], every column keeps its preconditioned basis
/// `Z = M^-1 V` next to `V` and forms its solution as `x0 + Z y` from the vectors
/// the Arnoldi relation `A Z = V H` holds, so the least-squares residual is the
/// true one up to rounding of `A` alone, and the preconditioner may change from
/// step to step. Applying `M^-1` once more to `V y` instead breaks that
/// relation by the preconditioner's own rounding: a static-pivoted near-field LU
/// of a 79 k unknown MoM system (backward error 2e-3) left the true residuals at
/// 1e-5 to 2e-5 where the least squares said 1e-6, and every column restarted.
/// The price is the second basis in memory; a preconditioner that is one fixed
/// operator can do without it ([`KrylovSettings::flexible`] `false`).
///
/// **Room:** within [`KrylovSettings::room_bytes`] and the memory a probe reports
/// ([`within`](Self::within)) the columns run in waves: where the bases would
/// outgrow it, those with the worst estimates close their cycles and wait until
/// the others finished, every column at the restart. With a fixed preconditioner
/// ([`KrylovSettings::downgrade`]) the bases are lowered first: `Z` dropped, then
/// `V` rounded to single precision where the tolerance allows it, the open cycles
/// carried on.
///
/// A column whose least-squares residual reaches the tolerance pauses: it leaves
/// the batched applies and keeps its cycle open. Once no column can step, the
/// paused columns' solutions are formed and their true residuals `||b - A x||`
/// checked in one batched operator apply: a column above the tolerance closes
/// its cycle and restarts from its true residual, as does a column whose cycle
/// reached the restart.
pub struct BlockGmres<'a, T: Scalar, A: ?Sized, M: ?Sized> {
    op: &'a A,
    precond: &'a M,
    b: &'a [T],
    n: usize,
    s: usize,
    /// The restart: the most steps of one cycle.
    m: usize,
    /// Iterations one [`solve`](Self::solve) may take.
    max_iter: usize,
    reorth_eta: f64,
    chunk: usize,
    ortho_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    bnorm: Vec<f64>,
    /// Per column the solution its open cycle started from (its current one
    /// without an open cycle), column-major `n x s`.
    x: Vec<T>,
    /// Per column the solution as last formed: `x` plus the open cycle's
    /// correction at the last residual check.
    out: Vec<T>,
    /// The true relative residual of `out`, `None` once the column stepped since.
    res: Vec<Option<f64>>,
    cols: Vec<Column<T>>,
    iters: usize,
    deepest: usize,
    // Scratch, reused by every step and check (the hot loop allocates nothing).
    /// Compact `n x s` panel: the active columns' newest basis vectors, the
    /// solutions whose residual is checked.
    panel: Vec<T>,
    /// `M^-1` of the panel.
    z: Vec<T>,
    /// `A M^-1` of the panel, orthogonalized in place; `b - A x` in a check.
    w: Vec<T>,
    /// The two Gram-Schmidt passes' projections, `off[a]..off[a+1]` column `a`'s.
    proj1: Vec<T>,
    proj2: Vec<T>,
    /// The projections' row-chunk partial sums.
    proj_scratch: Vec<T>,
    off: Vec<usize>,
    wnorm0: Vec<f64>,
    reorth: Vec<bool>,
    y: Vec<T>,
    /// The columns of a step with their open cycles' steps, in column order.
    act: Vec<(usize, usize)>,
    /// The columns of the last residual check, in column order (their residual
    /// vectors in `w` at these positions).
    chk: Vec<usize>,
    /// The columns of `chk` closing for memory ([`BlockProgress::closing`]).
    closing: Vec<usize>,
    /// Per column its estimate for the step monitor ([`StepProgress::estimates`]).
    est: Vec<Option<f64>>,
    /// How the bases are stored now ([`KrylovSettings::flexible`],
    /// [`KrylovSettings::basis_single`]).
    basis: BlockBasis,
    /// They may be lowered for memory ([`KrylovSettings::downgrade`]).
    downgrade: bool,
    /// The representations they were lowered to ([`BlockKrylovResult::lowered`]).
    lowered: Vec<BasisSwitch>,
    /// The bytes a step adds to a column's bases.
    layer: u64,
    /// The bytes the columns' bases may hold together: the room set less what the
    /// solve holds besides ([`KrylovSettings::room_bytes`]); `None`: no bound.
    room: Option<u64>,
    /// Cycles closed for room ([`BlockKrylovResult::closed`]).
    closed: usize,
    /// The memory the process may still take ([`within`](Self::within)).
    probe: Option<&'a (dyn Fn() -> Option<u64> + Sync)>,
    /// The most the process grew by over a step beside the bases' growth, as the
    /// probe saw it.
    side: u64,
    /// The subspace deflated ([`deflating`](Self::deflating)), with its
    /// projections of a panel and their partial sums.
    rec: Option<&'a BlockRecycle<T>>,
    rproj: Vec<T>,
    rscratch: Vec<T>,
}

impl<'a, T, A, M> BlockGmres<'a, T, A, M>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    /// A solve of the `s` right-hand sides `b` (column-major `n x s`) from `x0`
    /// (column-major `n x s`, default zero), nothing computed yet. The restart,
    /// the iteration budget of every [`solve`](Self::solve) call, the
    /// reorthogonalization threshold and the reduction chunk come from
    /// `settings`; its `tol` is not read (each call names its own).
    pub fn new(
        op: &'a A,
        b: &'a [T],
        s: usize,
        precond: &'a M,
        settings: &KrylovSettings,
        x0: Option<&[T]>,
    ) -> Result<Self, RslabError> {
        let n = op.n();
        if s == 0 || b.len() != n * s {
            return Err(RslabError::DimensionMismatch {
                expected: n * s,
                got: b.len(),
            });
        }
        let mut x = match x0 {
            Some(g) if g.len() != n * s => {
                return Err(RslabError::DimensionMismatch {
                    expected: n * s,
                    got: g.len(),
                })
            }
            Some(g) => g.to_vec(),
            None => vec![T::zero(); n * s],
        };
        let m = settings.block_restart::<T>(n, s);
        let chunk = settings.ortho_chunk.max(1);
        let bnorm: Vec<f64> = (0..s).map(|c| norm2(&b[c * n..c * n + n])).collect();
        let mut res = vec![None; s];
        for c in 0..s {
            if bnorm[c] == 0.0 {
                // `A x = 0` is solved by zero exactly; a warm start's seed is dropped.
                x[c * n..c * n + n].iter_mut().for_each(|v| *v = T::zero());
                res[c] = Some(0.0);
            }
        }
        let mut gm = BlockGmres {
            op,
            precond,
            b,
            n,
            s,
            m,
            max_iter: settings.max_iter,
            reorth_eta: settings.reorth_eta,
            chunk,
            // Orthogonalize in a pool of the width the preconditioner was factored
            // with, so factor and solve share one concurrency budget (`None`:
            // the caller's pool).
            ortho_pool: solve_thread_pool(precond.solve_threads()),
            bnorm,
            out: x.clone(),
            x,
            res,
            cols: (0..s).map(|_| Column::new(m, 0)).collect(),
            basis: settings.basis(),
            downgrade: settings.downgrade,
            lowered: Vec::new(),
            est: Vec::new(),
            layer: settings.basis().layer_bytes::<T>(n),
            room: None,
            closed: 0,
            probe: None,
            side: 0,
            iters: 0,
            deepest: 0,
            panel: vec![T::zero(); n * s],
            z: vec![T::zero(); n * s],
            w: vec![T::zero(); n * s],
            proj1: vec![T::zero(); s * (m + 1)],
            proj2: vec![T::zero(); s * (m + 1)],
            proj_scratch: vec![T::zero(); n.div_ceil(chunk) * s * (m + 1)],
            off: vec![0; s + 1],
            wnorm0: vec![0.0; s],
            reorth: vec![false; s],
            y: vec![T::zero(); m],
            act: Vec::with_capacity(s),
            chk: Vec::with_capacity(s),
            closing: Vec::with_capacity(s),
            rec: None,
            rproj: Vec::new(),
            rscratch: Vec::new(),
        };
        // (no basis is written yet: what it holds now is what it holds besides them)
        gm.room = settings
            .room_bytes
            .map(|r| r.saturating_sub(gm.resident_bytes()));
        Ok(gm)
    }

    /// This solve deflating `rec` (taken by [`harvest`](Self::harvest) from a
    /// solve of the same operator): every cycle starts from the residual's part
    /// outside `range(C)` with its correction inside, and every Krylov vector is
    /// kept orthogonal to `C`. Before the first [`solve`](Self::solve).
    pub fn deflating(mut self, rec: &'a BlockRecycle<T>) -> Result<Self, RslabError> {
        if rec.n != self.n {
            return Err(RslabError::DimensionMismatch {
                expected: self.n,
                got: rec.n,
            });
        }
        let (n, s, m, k) = (self.n, self.s, self.m, rec.k);
        for col in &mut self.cols {
            col.bm = vec![T::zero(); m * k];
        }
        self.rproj = vec![T::zero(); k * s];
        self.rscratch = vec![T::zero(); n.div_ceil(self.chunk) * k * s];
        self.rec = Some(rec);
        Ok(self)
    }

    /// This solve within the memory `probe` says the process may still take
    /// (bytes, `None`: not known), asked before every step and after it: the
    /// bases grow only into it, less the most a step took beside them (what the
    /// operator and the preconditioner kept, as measured), as within
    /// [`KrylovSettings::room_bytes`]. Before the first [`solve`](Self::solve).
    pub fn within(mut self, probe: &'a (dyn Fn() -> Option<u64> + Sync)) -> Self {
        self.probe = Some(probe);
        self
    }

    /// Heap bytes the solve holds now: every column's basis as far as it is
    /// written in the precision it is stored in (the restart's reservation is
    /// address space only), the
    /// Hessenberg matrices, the panels and the reduction scratch.
    /// [`KrylovSettings::block_bytes`] predicts it for a given depth.
    pub fn resident_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        let cols: u64 = self
            .cols
            .iter()
            .map(|c| {
                c.basis_bytes()
                    + [&c.h, &c.hraw, &c.bm, &c.cs, &c.sn, &c.g]
                        .into_iter()
                        .map(vec_bytes)
                        .sum::<u64>()
            })
            .sum();
        cols + [
            &self.x,
            &self.out,
            &self.panel,
            &self.z,
            &self.w,
            &self.proj1,
            &self.proj2,
            &self.proj_scratch,
            &self.y,
            &self.rproj,
            &self.rscratch,
        ]
        .into_iter()
        .map(vec_bytes)
        .sum::<u64>()
            + self.rec.map_or(0, BlockRecycle::heap_bytes)
            + vec_bytes(&self.bnorm)
            + vec_bytes(&self.res)
            + vec_bytes(&self.wnorm0)
            + vec_bytes(&self.reorth)
            + vec_bytes(&self.off)
            + vec_bytes(&self.act)
            + vec_bytes(&self.chk)
            + vec_bytes(&self.closing)
    }

    /// The operator the solve runs on, for the caller's own products between
    /// [`solve`](Self::solve) calls (a diagnosis of an unconverged rung).
    pub fn operator(&self) -> &'a A {
        self.op
    }

    /// The dimension of the subspace deflated (`0` without one).
    pub fn rec_dim(&self) -> usize {
        self.rec.map_or(0, |r| r.k)
    }

    /// Block iterations over all [`solve`](Self::solve) calls so far.
    pub fn iterations(&self) -> usize {
        self.iters
    }

    /// The deepest basis a column has held so far, in vectors of length `n`
    /// (the open cycle's steps plus its starting vector): with
    /// [`KrylovSettings::block_bytes`] what the solve's memory came to.
    pub fn deepest(&self) -> usize {
        self.deepest
    }

    /// Run every right-hand side to the relative residual `tol` (or the
    /// iteration budget, or a cancel from `mon`) and hand out the solutions. A
    /// further call continues: columns whose cycle is open carry on in their
    /// Krylov space, so a tighter `tol` costs only the steps past the last one.
    ///
    /// **Budget:** every column takes at most `max_iter` steps a call.
    ///
    /// **Monitor:** after every check of true residuals (the start, every
    /// restart, a cycle closed for room, the hand-out) `mon` receives the
    /// [`BlockProgress`] and returns whether to go on; `false` ends the call
    /// with `StopReason::Stalled` and the solutions so far.
    pub fn solve(
        &mut self,
        tol: f64,
        mon: Option<&mut dyn FnMut(&BlockProgress<'_>) -> bool>,
    ) -> Result<BlockKrylovResult<T>, RslabError> {
        self.solve_watched(tol, mon, None)
    }

    /// [`solve`](Self::solve) with a step monitor as well: after every block
    /// step `step` receives the [`StepProgress`] (the open cycles' estimates,
    /// no true residual formed) and returns whether to go on; `false` ends the
    /// call as `mon` does (`StopReason::Stalled`, the solutions formed). A
    /// further call continues the open cycles; a caller that changes the
    /// preconditioner in between [`restart`](Self::restart)s first.
    pub fn solve_watched(
        &mut self,
        tol: f64,
        mut mon: Option<&mut dyn FnMut(&BlockProgress<'_>) -> bool>,
        mut step: Option<&mut dyn FnMut(&StepProgress<'_>) -> bool>,
    ) -> Result<BlockKrylovResult<T>, RslabError> {
        let start = self.iters;
        let mut stalled = false;
        for (c, col) in self.cols.iter_mut().enumerate() {
            col.stepped = 0;
            // a column a call before closed above this tolerance starts again
            if col.steps.is_none() && !col.waits && self.res[c].is_some_and(|r| r > tol) {
                self.res[c] = None;
            }
        }
        loop {
            self.select_active(tol);
            if self.act.is_empty() {
                self.admit();
            }
            self.evict(tol);
            let stepping = !self.act.is_empty();
            // The residual check: with columns stepping, only the full cycles
            // (they restart from their true residual); else every column that
            // stepped since its last check.
            let checked = self.check(!stepping)?;
            if checked {
                if let Some(mo) = mon.as_mut() {
                    // NaN-honest: a non-finite residual is the worst one
                    let worst = (0..self.s)
                        .filter(|&c| !self.cols[c].waits)
                        .map(|c| match self.res[c] {
                            Some(r) => r,
                            None => self.cols[c].estimate(self.bnorm[c]),
                        })
                        .fold(
                            0.0_f64,
                            |a, r| if r.is_nan() { f64::INFINITY } else { a.max(r) },
                        );
                    self.closing.clear();
                    let cols = &self.cols;
                    self.closing
                        .extend(self.chk.iter().filter(|&&c| cols[c].evict));
                    let progress = BlockProgress {
                        iterations: self.iters - start,
                        worst,
                        active: self.act.len(),
                        waiting: self.cols.iter().filter(|c| c.waits).count(),
                        checked: &self.chk,
                        closing: &self.closing,
                        residuals: &self.res,
                        resident: self.resident_bytes(),
                        basis: self.basis,
                        lowered: &self.lowered,
                    };
                    if !mo(&progress) {
                        stalled = true;
                        break;
                    }
                }
                self.restart_above(tol);
                continue;
            }
            if !stepping {
                break;
            }
            let before = self.probe.and_then(|p| p());
            let held = self.basis_bytes();
            self.step()?;
            if let (Some(b), Some(a)) = (before, self.probe.and_then(|p| p())) {
                let grown = self.basis_bytes() - held;
                self.side = self.side.max(b.saturating_sub(a).saturating_sub(grown));
            }
            if let Some(st) = step.as_mut() {
                self.est.clear();
                for c in 0..self.s {
                    let col = &self.cols[c];
                    self.est.push(match (col.steps, self.res[c]) {
                        (Some(_), None) => Some(col.estimate(self.bnorm[c])),
                        (_, r) => r,
                    });
                }
                let worst = (0..self.s)
                    .filter(|&c| !self.cols[c].waits)
                    .map(|c| self.est[c].unwrap_or(f64::INFINITY))
                    .fold(
                        0.0_f64,
                        |a, r| if r.is_nan() { f64::INFINITY } else { a.max(r) },
                    );
                let progress = StepProgress {
                    iterations: self.iters - start,
                    worst,
                    active: self.act.len(),
                    estimates: &self.est,
                    basis: self.basis,
                };
                if !st(&progress) {
                    stalled = true;
                    break;
                }
            }
        }
        // A cancel leaves the columns stepped since their last check unformed.
        self.check(true)?;
        let final_res: Vec<f64> = self
            .res
            .iter()
            .map(|r| r.unwrap_or(f64::INFINITY))
            .collect();
        let converged = final_res.iter().all(|&r| r <= tol);
        Ok(BlockKrylovResult {
            x: self.out.clone(),
            iters: self.iters - start,
            converged,
            final_res,
            stop: if converged {
                StopReason::Converged
            } else if stalled {
                StopReason::Stalled
            } else {
                StopReason::MaxIter
            },
            closed: self.closed,
            basis: self.basis,
            lowered: self.lowered.clone(),
        })
    }

    /// The representation of the bases now.
    pub fn basis(&self) -> BlockBasis {
        self.basis
    }

    /// Close every open cycle on the solution handed out by the last call: the next call
    /// starts a new one from its true residual, the Krylov space so far released. A caller
    /// that changes the preconditioner between calls restarts so: a cycle is formed under one
    /// preconditioner (`x0 + M^-1 (V y)` needs it, and the flexible bases built on the one it
    /// dropped carried no gain on the measured systems).
    pub fn restart(&mut self) {
        for c in 0..self.s {
            if self.cols[c].steps.is_some() && self.res[c].is_some() {
                self.close(c);
                // checked again (its true residual) before a new cycle opens
                self.res[c] = None;
            }
        }
    }

    /// `act`: the columns that can step toward `tol`, an open cycle short of the
    /// restart with its estimate above `tol` and steps left in the budget.
    fn select_active(&mut self, tol: f64) {
        self.act.clear();
        for c in 0..self.s {
            let col = &self.cols[c];
            if let Some(j) = col.steps {
                if j < self.m && col.stepped < self.max_iter && col.estimate(self.bnorm[c]) > tol {
                    self.act.push((c, j));
                }
            }
        }
    }

    /// The next wave: once no column steps and none waits for its restart, the
    /// columns waiting for room start again from their solutions (checked first).
    fn admit(&mut self) {
        let m = self.m;
        if (0..self.s).any(|c| self.cols[c].steps == Some(m) && self.res[c].is_none()) {
            return;
        }
        for c in 0..self.s {
            if self.cols[c].waits {
                self.cols[c].waits = false;
                self.res[c] = None;
            }
        }
    }

    /// The memory the bases may still grow by: what the room left
    /// ([`KrylovSettings::room_bytes`]) and what the probe says the process may
    /// still take, less the most a step took beside the bases ([`within`](Self::within));
    /// `None` without either.
    fn free(&self) -> Option<u64> {
        let room = self.room.map(|r| r.saturating_sub(self.basis_bytes()));
        let live = self
            .probe
            .and_then(|p| p())
            .map(|r| r.saturating_sub(self.side));
        match (room, live) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// Heap bytes of the columns' bases as written.
    fn basis_bytes(&self) -> u64 {
        self.cols.iter().map(Column::basis_bytes).sum()
    }

    /// Keep the bases within the memory ([`free`](Self::free)): where the next step
    /// of the columns in `act` would not fit, the bases are lowered first where they
    /// may be ([`lower`](Self::lower), toward `tol`); where it still does not fit, the
    /// columns not stepping give their
    /// bases up (a finished one at once, a paused one closes its cycle at the next
    /// check), then the stepping ones with the worst estimates close theirs until
    /// the rest reach the restart in what is free (at once where no step was taken
    /// since their check, else at the next check). The columns closed wait for the
    /// next wave ([`admit`](Self::admit)): the easy ones finish together, the hard
    /// ones then have the memory. At least one column steps on; one alone closes
    /// only once its cycle took a step and its next does not fit (a restart at the
    /// depth the memory holds).
    fn evict(&mut self, tol: f64) {
        let Some(mut free) = self.free() else {
            return;
        };
        let (n, m) = (self.n, self.m);
        // what the columns in `act` grow by to step `to(j)` of their cycles
        let grow = |g: &Self, i: usize, to: &dyn Fn(usize) -> usize| {
            let (c, j) = g.act[i];
            g.cols[c].growth(n, j, to(j), g.layer)
        };
        let next = |g: &Self, i: usize| grow(g, i, &|j| j + 1);
        let ahead = |g: &Self, i: usize| grow(g, i, &|_| m);
        let step = |g: &Self| (0..g.act.len()).map(|i| next(g, i)).sum::<u64>();
        while free < step(self) {
            let Some(to) = self.basis.lower::<T>(tol).filter(|_| self.downgrade) else {
                break;
            };
            // what the bases give up is free (a live probe may not see it yet)
            let held = self.basis_bytes();
            self.lower(to);
            free += held - self.basis_bytes();
        }
        if free >= step(self) {
            return;
        }
        let mut freed = 0;
        let stepping =
            |act: &[(usize, usize)], c: usize| act.binary_search_by_key(&c, |a| a.0).is_ok();
        for c in 0..self.s {
            if !self.cols[c].holds() || stepping(&self.act, c) {
                continue;
            }
            freed += self.cols[c].basis_bytes();
            if self.res[c].is_some() {
                self.close(c);
            } else {
                self.cols[c].evict = true;
            }
        }
        // the stepping columns, the worst estimate first (a non-finite one first)
        let est = |g: &Self, i: usize| {
            let c = g.act[i].0;
            g.cols[c].estimate(g.bnorm[c])
        };
        let mut order: Vec<usize> = (0..self.act.len()).collect();
        order.sort_by(|&a, &b| est(self, b).total_cmp(&est(self, a)).then(a.cmp(&b)));
        let mut need: u64 = (0..self.act.len()).map(|i| ahead(self, i)).sum();
        let mut out = vec![false; self.act.len()];
        let mut kept = self.act.len();
        for i in order {
            if free + freed >= need {
                break;
            }
            let (c, j) = self.act[i];
            if kept == 1 && (j == 0 || free + freed >= next(self, i)) {
                break;
            }
            out[i] = true;
            kept -= 1;
            freed += self.cols[c].basis_bytes();
            need -= ahead(self, i);
            self.closed += 1;
            if self.res[c].is_some() {
                // checked and not stepped since (a cycle just opened): it waits at once
                self.close(c);
                self.cols[c].waits = true;
            } else {
                self.cols[c].evict = true;
            }
        }
        let mut i = 0;
        self.act.retain(|_| {
            i += 1;
            !out[i - 1]
        });
    }

    /// Store the bases as `to` from now on, every column's open cycle carried on: `Z`
    /// dropped (the solutions then `x + M^-1 (V y)`, the preconditioner a fixed
    /// operator, [`KrylovSettings::downgrade`]), the bases rounded to single
    /// precision (a column at a time).
    fn lower(&mut self, to: BlockBasis) {
        for col in &mut self.cols {
            if !to.flexible {
                col.zb = Stored::default();
            }
            if to.single {
                col.v.round_to_single();
                col.zb.round_to_single();
            }
        }
        self.basis = to;
        self.layer = to.layer_bytes::<T>(self.n);
        self.lowered.push(BasisSwitch {
            iteration: self.iters,
            basis: to,
        });
    }

    /// Close column `c`'s cycle on the solution formed (`x <- out`), its bases
    /// released.
    fn close(&mut self, c: usize) {
        let (cb, n) = (c * self.n, self.n);
        self.x[cb..cb + n].copy_from_slice(&self.out[cb..cb + n]);
        self.cols[c].release();
    }

    /// Form the solutions of the unchecked columns (all of them with `all`, else
    /// those whose cycle reached the restart) and their true residuals; the
    /// residual vectors stay in `w` (compact, in the order of `chk`).
    /// Whether any column was checked.
    fn check(&mut self, all: bool) -> Result<bool, RslabError> {
        let (n, m) = (self.n, self.m);
        self.chk.clear();
        for c in 0..self.s {
            let col = &self.cols[c];
            // with every one: a cycle at the restart, one closing for room, one
            // starting (none open)
            if self.res[c].is_none()
                && (all || col.evict || col.steps == Some(m) || col.steps.is_none())
            {
                self.chk.push(c);
            }
        }
        let k = self.chk.len();
        if k == 0 {
            return Ok(false);
        }
        if self.basis.flexible {
            // The open cycles' solutions `x + Z y`, from the stored preconditioned basis.
            for (a, &c) in self.chk.iter().enumerate() {
                let (cb, pb) = (c * n, a * n);
                self.cols[c].correction(
                    m,
                    self.rec,
                    &mut self.y,
                    &self.x[cb..cb + n],
                    &mut self.out[cb..cb + n],
                );
                self.panel[pb..pb + n].copy_from_slice(&self.out[cb..cb + n]);
            }
        } else {
            // The open cycles' solutions `x + M^-1 (V y)`: every column's `V y`, one
            // batched preconditioner apply, then `x` and the deflated part.
            for (a, &c) in self.chk.iter().enumerate() {
                let pb = a * n;
                let col = &self.cols[c];
                let jd = col.coefficients(m, &mut self.y);
                let p = &mut self.panel[pb..pb + n];
                p.iter_mut().for_each(|e| *e = T::zero());
                Column::combine(&col.v, jd, &self.y, p);
            }
            self.precond
                .apply_block(&self.panel[..k * n], &mut self.z[..k * n], k, n)?;
            for (a, &c) in self.chk.iter().enumerate() {
                let (cb, pb) = (c * n, a * n);
                let col = &self.cols[c];
                let out = &mut self.out[cb..cb + n];
                for ((o, &x), &z) in out
                    .iter_mut()
                    .zip(&self.x[cb..cb + n])
                    .zip(&self.z[pb..pb + n])
                {
                    *o = x + z;
                }
                if let Some(rec) = self.rec {
                    let jd = col.coefficients(m, &mut self.y);
                    col.deflated(rec, jd, &self.y, out);
                }
                self.panel[pb..pb + n].copy_from_slice(out);
            }
        }
        // Their true residuals, one batched operator apply.
        self.op
            .apply_block(&self.panel[..k * n], &mut self.w[..k * n], k);
        for (a, &c) in self.chk.iter().enumerate() {
            let (cb, pb) = (c * n, a * n);
            let mut rn = 0.0;
            for i in 0..n {
                let r = self.b[cb + i] - self.w[pb + i];
                self.w[pb + i] = r;
                rn += r.magnitude_sq();
            }
            self.res[c] = Some(rn.sqrt() / self.bnorm[c]);
        }
        Ok(true)
    }

    /// Restart every column the last check left above `tol` with steps left: its
    /// cycle closes on the solution formed (`x <- out`) and a new one opens on its
    /// true residual (in `w` at its position in `chk`). A column closing for room
    /// closes alone, its bases released, and waits where it is above `tol`.
    fn restart_above(&mut self, tol: f64) {
        let n = self.n;
        for a in 0..self.chk.len() {
            let c = self.chk[a];
            let r = self.res[c].unwrap_or(f64::INFINITY);
            let done = r <= tol || self.bnorm[c] == 0.0 || self.cols[c].stepped >= self.max_iter;
            if self.cols[c].evict {
                self.cols[c].evict = false;
                self.close(c);
                self.cols[c].waits = !done;
                continue;
            }
            if done {
                continue;
            }
            let cb = c * n;
            self.x[cb..cb + n].copy_from_slice(&self.out[cb..cb + n]);
            let rv = &mut self.w[a * n..(a + 1) * n];
            let beta = match &self.rec {
                // deflating: the residual's part in `range(C)` first (`x += U C^H r`,
                // `r -= C C^H r`), the solution handed out moving with it
                Some(rec) => {
                    for i in 0..rec.k {
                        let ci = &rec.c[i * n..(i + 1) * n];
                        let t = dotc(ci, rv);
                        for (x, &ck) in rv.iter_mut().zip(ci) {
                            *x = *x - ck * t;
                        }
                        let ui = &rec.u[i * n..(i + 1) * n];
                        for (x, &u) in self.x[cb..cb + n].iter_mut().zip(ui) {
                            *x = *x + u * t;
                        }
                    }
                    self.out[cb..cb + n].copy_from_slice(&self.x[cb..cb + n]);
                    let beta = norm2(rv);
                    // its residual is the projected one: checked again before it counts
                    self.res[c] = if beta == 0.0 { Some(0.0) } else { None };
                    beta
                }
                None => r * self.bnorm[c],
            };
            if beta > 0.0 {
                self.cols[c].open(&mut self.w[a * n..(a + 1) * n], beta, self.basis);
                self.deepest = self.deepest.max(1);
            }
        }
    }

    /// One block step of the columns in `act`: their newest basis vectors through
    /// the preconditioner and the operator in one batched apply each, each
    /// result orthogonalized against its column's own basis by CGS2 (a
    /// projection pass over the whole panel, a second one for the columns whose
    /// norm collapsed), the Hessenberg columns rotated and the bases extended.
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self) -> Result<(), RslabError> {
        let (n, m, sa) = (self.n, self.m, self.act.len());
        for (a, &(c, j)) in self.act.iter().enumerate() {
            self.cols[c]
                .v
                .read(j * n, &mut self.panel[a * n..(a + 1) * n]);
            self.off[a + 1] = self.off[a] + j + 1;
        }
        self.precond
            .apply_block(&self.panel[..sa * n], &mut self.z[..sa * n], sa, n)?;
        self.op
            .apply_block(&self.z[..sa * n], &mut self.w[..sa * n], sa);
        for a in 0..sa {
            self.wnorm0[a] = norm2(&self.w[a * n..(a + 1) * n]);
        }
        let Self {
            cols,
            act,
            off,
            w,
            proj1,
            proj2,
            proj_scratch,
            ortho_pool,
            chunk,
            reorth,
            wnorm0,
            reorth_eta,
            rec,
            rproj,
            rscratch,
            ..
        } = self;
        let (cols, act, off, chunk) = (&*cols, &*act, &off[..=sa], *chunk);
        let layers = |a: usize| &cols[act[a].0].v;
        let w = &mut w[..sa * n];
        // deflating: the panel's components in `range(C)` first, kept per step
        if let Some(rec) = &*rec {
            ortho_in_pool(ortho_pool, || {
                panel_project(&rec.c, rec.k, w, sa, n, chunk, rproj, rscratch)
            });
            ortho_in_pool(ortho_pool, || {
                panel_subtract(&rec.c, rec.k, w, sa, n, chunk, rproj)
            });
        }
        ortho_in_pool(ortho_pool, || {
            column_project(&layers, off, w, n, chunk, proj1, proj_scratch)
        });
        ortho_in_pool(ortho_pool, || {
            column_subtract(&layers, off, w, n, chunk, proj1)
        });
        // The second pass, per column: only a column whose norm collapsed in the
        // first takes it (decided on serial norms, so thread-count independent).
        let mut any = false;
        for a in 0..sa {
            reorth[a] = norm2(&w[a * n..(a + 1) * n]) < *reorth_eta * wnorm0[a];
            any |= reorth[a];
        }
        if any {
            ortho_in_pool(ortho_pool, || {
                column_project(&layers, off, w, n, chunk, proj2, proj_scratch)
            });
            for a in 0..sa {
                if !reorth[a] {
                    proj2[off[a]..off[a + 1]]
                        .iter_mut()
                        .for_each(|e| *e = T::zero());
                }
            }
            ortho_in_pool(ortho_pool, || {
                column_subtract(&layers, off, w, n, chunk, proj2)
            });
        }
        for a in 0..sa {
            let (c, j) = self.act[a];
            let col = &mut self.cols[c];
            if let Some(rec) = &self.rec {
                for i in 0..rec.k {
                    col.bm[j * rec.k + i] = self.rproj[i * sa + a];
                }
            }
            for i in 0..=j {
                let p = self.off[a] + i;
                col.h[i * m + j] = if any {
                    self.proj1[p] + self.proj2[p]
                } else {
                    self.proj1[p]
                };
                col.hraw[i * m + j] = col.h[i * m + j];
            }
            if self.basis.flexible {
                col.zb.write(j * n, &self.z[a * n..(a + 1) * n]);
            }
            let wa = &mut self.w[a * n..(a + 1) * n];
            let hn = norm2(wa);
            col.h[(j + 1) * m + j] = T::from_real(hn);
            col.hraw[(j + 1) * m + j] = col.h[(j + 1) * m + j];
            // the next basis vector, scaled in the panel
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                wa.iter_mut().for_each(|wi| *wi = *wi * inv);
            } else {
                wa.iter_mut().for_each(|wi| *wi = T::zero());
            }
            col.v.write((j + 1) * n, wa);
            // The earlier rotations, then a new one zeroing `h[j+1][j]`.
            for i in 0..j {
                let (hi, hi1) = (col.h[i * m + j], col.h[(i + 1) * m + j]);
                col.h[i * m + j] = col.cs[i].conj() * hi + col.sn[i].conj() * hi1;
                col.h[(i + 1) * m + j] = -col.sn[i] * hi + col.cs[i] * hi1;
            }
            let (cj, sj) = givens(col.h[j * m + j], col.h[(j + 1) * m + j]);
            col.cs[j] = cj;
            col.sn[j] = sj;
            col.h[j * m + j] = cj.conj() * col.h[j * m + j] + sj.conj() * col.h[(j + 1) * m + j];
            col.h[(j + 1) * m + j] = T::zero();
            col.g[j + 1] = -sj * col.g[j];
            col.g[j] = cj.conj() * col.g[j];
            col.steps = Some(j + 1);
            col.stepped += 1;
            self.res[c] = None;
            self.deepest = self.deepest.max(j + 2);
        }
        self.iters += 1;
        Ok(())
    }
}

impl<'a, T, A, M> BlockGmres<'a, T, A, M>
where
    T: RecycleScalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    /// Up to `k` directions this solve's open cycles found slowest, as a
    /// [`BlockRecycle`] for later solves of the same operator
    /// ([`deflating`](Self::deflating)): per open cycle of `d` steps up to `d / 2`
    /// harmonic-Ritz pairs of smallest magnitude of its `(d+1) x d` relation
    /// `A Z = V H_bar` (`H_bar^H H_bar g = theta H^H g`) and their directions
    /// `Z g`, over all columns by ascending `|theta|`, orthonormalized with the
    /// dependent ones dropped; then their images `C = A U` by one operator
    /// product of the `k` columns, orthonormalized with `U` along. The images
    /// are computed rather than taken from the relation (`V H_bar g`): the
    /// columns' candidates overlap, and orthonormalizing nearly dependent ones
    /// amplified the relation's rounding to `|A u - c|` of 4e-2. `None` where no
    /// cycle is open or the solve deflates itself (its relation then has the
    /// deflated part too).
    pub fn harvest(&self, k: usize) -> Option<BlockRecycle<T>> {
        use crate::numeric::dense_eig::harmonic_ritz_smallest;
        type Cx = num_complex::Complex<f64>;
        if self.rec.is_some() || k == 0 {
            return None;
        }
        let (n, m) = (self.n, self.m);
        // every column's candidates: (|theta|, column, theta, g)
        let mut cands: Vec<(f64, usize, Cx, Vec<Cx>)> = Vec::new();
        for (c, col) in self.cols.iter().enumerate() {
            let Some(d) = col.steps.filter(|&d| d > 0) else {
                continue;
            };
            let hb = |i: usize, j: usize| col.hraw[i * m + j].to_c();
            // `M1 = H_bar^H H_bar`, `M2 = H^H` (row-major `d x d`)
            let mut m1 = vec![Cx::new(0.0, 0.0); d * d];
            let mut m2 = vec![Cx::new(0.0, 0.0); d * d];
            for i in 0..d {
                for j in 0..d {
                    m1[i * d + j] = (0..=d).map(|r| hb(r, i).conj() * hb(r, j)).sum();
                    m2[i * d + j] = hb(j, i).conj();
                }
            }
            // at most half the cycle's depth from one column: the harmonic Ritz pairs of a
            // short cycle are crude, and deflating crude directions slows the solves it serves
            // (GCRO-DR's rule, as `gmres_recycled` caps its subspace at half the restart)
            for (theta, g) in harmonic_ritz_smallest(&m1, &m2, d, k.min(d / 2)) {
                cands.push((theta.norm(), c, theta, g));
            }
        }
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        // the directions, orthonormal
        let mut u: Vec<T> = Vec::new();
        for (_, c, theta, g) in cands {
            let kept = u.len() / n;
            if kept >= k {
                break;
            }
            let d = g.len();
            // the Ritz vectors in the solution space: `Z g`, as `M^-1 (V g)` where `Z`
            // is not kept
            let pairs = [(theta, g)];
            let ritz = |b: &Stored<T>| {
                T::combine_ritz(&|j, a, out| b.axpy(j * n, a, out), n, d, &pairs, k - kept)
            };
            let cand = if self.basis.flexible {
                ritz(&self.cols[c].zb)
            } else {
                let vg = ritz(&self.cols[c].v);
                let mut zg = vec![T::zero(); vg.len()];
                // (no wider than the solve applies it)
                let width = self.s * n;
                let applied = vg
                    .chunks(width)
                    .zip(zg.chunks_mut(width))
                    .all(|(v, z)| self.precond.apply_block(v, z, v.len() / n, n).is_ok());
                if !applied {
                    continue;
                }
                zg
            };
            // a column's Ritz vector the others found too is not new: kept only
            // where most of it lies outside the kept ones (the DGKS ratio)
            for v in cand.chunks_exact(n) {
                orthonormal_into(&mut u, None, v, n, self.reorth_eta);
            }
        }
        let kept = u.len() / n;
        if kept == 0 {
            return None;
        }
        // their images, orthonormalized with the directions along (`C = A U` stays)
        // in blocks of at most the solve's width: the operator is applied as the solve applies
        // it (a matrix-free operator sizes its scratch for that)
        let mut au = vec![T::zero(); kept * n];
        for (ui, ai) in u.chunks(self.s * n).zip(au.chunks_mut(self.s * n)) {
            self.op.apply_block(ui, ai, ui.len() / n);
        }
        let (mut uu, mut cc) = (Vec::with_capacity(kept * n), Vec::with_capacity(kept * n));
        for (ui, ci) in u.chunks_exact(n).zip(au.chunks_exact(n)) {
            orthonormal_into(
                &mut cc,
                Some((&mut uu, ui)),
                ci,
                n,
                crate::constants::RECYCLE_RANK_TOL,
            );
        }
        let k = cc.len() / n;
        (k > 0).then_some(BlockRecycle { n, k, u: uu, c: cc })
    }
}

/// Append `v` to the orthonormal columns `q` (column-major, length `n` each) by
/// modified Gram-Schmidt, unless it keeps no more than `keep` of its norm (it
/// lies in their span to that degree). With `along = (p, w)` the same
/// combination of `w` and `p`'s columns goes to `p`, so a relation
/// `q_i = A p_i` holds for the appended column too.
fn orthonormal_into<T: Scalar>(
    q: &mut Vec<T>,
    along: Option<(&mut Vec<T>, &[T])>,
    v: &[T],
    n: usize,
    keep: f64,
) {
    let mut x = v.to_vec();
    let mut y = along.as_ref().map(|(_, w)| w.to_vec());
    let orig = norm2(&x);
    if orig == 0.0 {
        return;
    }
    for j in 0..q.len() / n {
        let qj = &q[j * n..(j + 1) * n];
        let t = dotc(qj, &x);
        for (a, &b) in x.iter_mut().zip(qj) {
            *a = *a - b * t;
        }
        if let (Some(y), Some((p, _))) = (y.as_mut(), along.as_ref()) {
            for (a, &b) in y.iter_mut().zip(&p[j * n..(j + 1) * n]) {
                *a = *a - b * t;
            }
        }
    }
    let nrm = norm2(&x);
    if nrm <= keep * orig {
        return;
    }
    let inv = T::from_real(1.0 / nrm);
    q.extend(x.iter().map(|&a| a * inv));
    if let (Some(y), Some((p, _))) = (y, along) {
        p.extend(y.iter().map(|&a| a * inv));
    }
}

/// Heap bytes of a [`BlockGmres`] of `s` right-hand sides of length `n` at
/// restart `m` and reduction chunk `chunk` once its columns' bases are `depth`
/// vectors deep, of `stored` bytes a basis scalar: the allocations of
/// [`BlockGmres::new`] and the written bases ([`BlockGmres::resident_bytes`] of a
/// solve at that depth in every column).
pub(super) fn block_bytes<T>(
    n: usize,
    s: usize,
    m: usize,
    chunk: usize,
    depth: usize,
    flexible: bool,
    stored: usize,
) -> u64 {
    let sz = std::mem::size_of::<T>();
    // per column: the basis and its preconditioned steps (flexible only), then the
    // Hessenberg matrix rotated and as built, the two rotations, `g`
    let z = if flexible {
        n * depth.saturating_sub(1)
    } else {
        0
    };
    let bases = n * depth + z;
    let column = 2 * (m + 1) * m + 2 * m + (m + 1);
    // the panels `x`, `out`, `panel`, `z`, `w`; the two projections and their partial
    // sums; `y`
    let panels = 5 * n * s; // lit: the five n x s panels of `BlockGmres::new`
    let shared = panels + 2 * s * (m + 1) + n.div_ceil(chunk.max(1)) * s * (m + 1) + m;
    // `bnorm`, `wnorm0`, `res`, `reorth`, `off` (one more), `act`, `chk`, `closing`
    let per_rhs = 2 * std::mem::size_of::<f64>()
        + std::mem::size_of::<Option<f64>>()
        + std::mem::size_of::<bool>()
        + 3 * std::mem::size_of::<usize>() // lit: `act`, `chk` and `closing`
        + std::mem::size_of::<(usize, usize)>();
    ((s * column + shared) * sz + s * bases * stored + s * per_rhs + std::mem::size_of::<usize>())
        as u64
}

/// Heap bytes [`BlockGmres::deflating`] adds to a solve of `s` right-hand sides
/// of length `n` at restart `m` and reduction chunk `chunk` for a subspace of
/// dimension `k`: the subspace, every column's components on it per step, and
/// a panel's projections with their partial sums.
pub(super) fn deflation_bytes<T>(n: usize, s: usize, m: usize, chunk: usize, k: usize) -> u64 {
    let sz = std::mem::size_of::<T>();
    ((2 * n * k + s * m * k + k * s + n.div_ceil(chunk.max(1)) * k * s) * sz) as u64
}

/// Right-preconditioned restarted **block GMRES** for `s` right-hand sides `b`
/// (column-major `nxs`) to `settings.tol`: one [`BlockGmres::solve`]. The `s`
/// systems advance in lockstep so the two expensive operations - the operator
/// matvec and the preconditioner solve - are issued once per step as **block**
/// applies ([`LinearOperator::apply_block`] / [`Preconditioner::apply_block`]),
/// each factor / matrix value touched once for all `s` columns. Each RHS keeps
/// its own Arnoldi basis / Hessenberg / Givens, so a column converges as the
/// single-RHS [`gmres`] does; the systems share only the batched operator and
/// preconditioner calls. Solves `A X = B` from the optional initial guess `x0`
/// (column-major `nxs`, default `X_0 = 0`).
///
/// **Warm start:** `x0 = Some(prev)` seeds every column from a related
/// previous solution. Each column's convergence is relative to its own
/// `||B[:,c]||`.
///
/// **Deflation:** a RHS whose least-squares residual reaches `tol` leaves the
/// batched applies, so they shrink to the active width as columns converge.
///
/// **Memory:** every column's bases `V` and `Z` (`V` alone when not
/// [`flexible`](KrylovSettings::flexible)) reserve the restart's layers of address
/// space while a cycle is open and hold what its cycles have written, in single
/// precision with [`basis_single`](KrylovSettings::basis_single); the solve
/// holds about [`KrylovSettings::block_bytes`] for the depth it reaches, within
/// [`KrylovSettings::room_bytes`] where set, lowering its bases there first with
/// [`downgrade`](KrylovSettings::downgrade). An unset
/// [`KrylovSettings::restart`] is capped to
/// [`KrylovSettings::basis_budget_bytes`]; an explicit value is honoured exactly.
///
/// **Threads:** the orthogonalization reductions run in a scoped pool derived
/// from the preconditioner's [`Threads`](crate::Threads) policy
/// ([`Preconditioner::solve_threads`]), so factor and solve share one
/// concurrency budget; [`Threads::Ambient`](crate::Threads::Ambient) (or
/// [`NoPreconditioner`]) leaves them on the caller's pool. The row-chunk fold is
/// thread-count independent, so the result is too.
///
/// **Orthogonalization:** block CGS2 (classical Gram-Schmidt with a conditional,
/// per-column second pass), not the MGS+DGKS of the single-RHS [`gmres`]: a block
/// solve with `s = 1` is not bit-identical to [`gmres`] and may differ by +/-1
/// iteration (see the module-level note).
///
/// **Budget and monitor:** see [`BlockGmres::solve`].
pub fn gmres_block<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    settings: &KrylovSettings,
    x0: Option<&[T]>,
    mon: Option<&mut dyn FnMut(&BlockProgress<'_>) -> bool>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    BlockGmres::new(op, b, s, precond, settings, x0)?.solve(settings.tol, mon)
}
