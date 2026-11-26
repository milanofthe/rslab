//! Block GMRES for many right-hand sides with delayed CGS2 over the whole panel,
//! as a solve that can be continued to a tighter tolerance in the same Krylov
//! space.

use super::util::*;
use super::*;

use super::recycle::RecycleScalar;
use super::settings::{BasisSwitch, BlockBasis};
use crate::error::RslabError;
use crate::parallel::SendPtr;
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
///
/// The basis is orthogonalized by delayed CGS2: after `j` steps of a cycle the
/// layers `0..j` hold the orthonormal `v_0..v_{j-1}` and layer `j` the pending
/// `u_j`, the residual of `A z_{j-1}` after one Gram-Schmidt pass. A step applies
/// the operator to `u_j / ||u_j||` as it is; its second pass, the projection
/// `s_j = V^H u_j / ||u_j||` that turns it into `v_j`, is taken in the same pass
/// over the basis as the first pass of the new vector. The Hessenberg matrix
/// relates the vectors the operator was applied to: `A Z = V H_bar` with
/// `z_i = M^-1 u_i / ||u_i||`, the last column (step `j - 1`) with `u_j` taken as
/// orthogonal to `V` until the next step finishes it.
struct Column<T: Scalar> {
    /// The basis, layer `i` the scalars `[i*n, (i+1)*n)`: the restart's layers
    /// reserved, the ones the cycle has written resident ([`Stored`]).
    v: Stored<T>,
    /// The preconditioned vectors `z_i = M^-1 u_i / ||u_i||` of the steps taken,
    /// laid out as `v`: the solution is formed from the vectors the Arnoldi
    /// relation holds. Empty when the preconditioner is fixed
    /// ([`KrylovSettings::flexible`] `false`): `Z y = M^-1 (V S y)` then.
    zb: Stored<T>,
    /// Row-major `(m+1) x m`, entry `(i, j)` at `h[i*m + j]`.
    h: Vec<T>,
    /// The same before the Givens rotations: `A Z = V H_bar` of the open cycle
    /// (`+ C B` when deflating), the matrix [`BlockGmres::harvest`] reads.
    hraw: Vec<T>,
    /// Row-major `m x m` upper triangular, `u_i / ||u_i|| = V S[:, i]`: the
    /// vectors the operator was applied to in the basis.
    su: Vec<T>,
    /// The least-squares right-hand side's entry of the last step before its
    /// rotation, kept to rotate again once the step is finished.
    gp: T,
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
            su: vec![T::zero(); m * m],
            gp: T::zero(),
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

    /// Open a cycle on the residual `r` of norm `beta > 0` (the pending first basis
    /// vector, normalized by the first step), the restart's layers of its bases
    /// reserved where it holds none (as `basis` stores them).
    fn open(&mut self, r: &[T], beta: f64, basis: BlockBasis) {
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

    /// `y <- S y` for the leading `jd` coefficients: those of `Z y` taken to the basis
    /// `V` (`Z = M^-1 V S` for a fixed preconditioner).
    fn lift(&self, m: usize, jd: usize, y: &mut [T]) {
        for i in 0..jd {
            y[i] = (i..jd).fold(T::zero(), |a, k| a + self.su[i * m + k] * y[k]);
        }
    }

    /// `t = B y` for the subspace `rec` deflated (`k` entries): the open cycle's
    /// correction in its directions is `- U t`.
    fn deflation_terms(&self, k: usize, jd: usize, y: &[T], t: &mut [T]) {
        for (i, ti) in t.iter_mut().enumerate().take(k) {
            *ti = (0..jd).fold(T::zero(), |a, j| a + self.bm[j * k + i] * y[j]);
        }
    }

    /// Rotate Hessenberg column `j` (taken from `hraw`) by the rotations before it
    /// and a new one zeroing its subdiagonal, the least-squares right-hand side's
    /// entries `j` and `j + 1` from `gp`, its entry `j` before.
    fn rotate(&mut self, m: usize, j: usize) {
        for i in 0..=j + 1 {
            self.h[i * m + j] = self.hraw[i * m + j];
        }
        for i in 0..j {
            let (hi, hi1) = (self.h[i * m + j], self.h[(i + 1) * m + j]);
            self.h[i * m + j] = self.cs[i].conj() * hi + self.sn[i].conj() * hi1;
            self.h[(i + 1) * m + j] = -self.sn[i] * hi + self.cs[i] * hi1;
        }
        let (cj, sj) = givens(self.h[j * m + j], self.h[(j + 1) * m + j]);
        self.cs[j] = cj;
        self.sn[j] = sj;
        self.h[j * m + j] = cj.conj() * self.h[j * m + j] + sj.conj() * self.h[(j + 1) * m + j];
        self.h[(j + 1) * m + j] = T::zero();
        self.g[j + 1] = -sj * self.gp;
        self.g[j] = cj.conj() * self.gp;
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
    /// The worker policy the orthogonalization runs under: the one the
    /// preconditioner was factored with, so factor and solve share one
    /// budget ([`Threads::run`](crate::Threads)).
    ortho_threads: crate::Threads,
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
    /// A step's products of the pass over the bases ([`delayed_project`]),
    /// `off[a]..off[a+1]` column `a`'s.
    proj: Vec<T>,
    /// The projections' row-chunk partial sums.
    proj_scratch: Vec<T>,
    off: Vec<usize>,
    /// A step's update per column ([`delayed_update`]).
    delayed: Vec<Delayed<T>>,
    /// Per column a norm (the new vectors of a step, the residuals of a check) and
    /// its row-chunk partial sums.
    norms: Vec<f64>,
    norm_scratch: Vec<f64>,
    /// The least-squares coefficients of the columns of a check, `m` each.
    y: Vec<T>,
    /// Their leading well-conditioned dimensions.
    jd: Vec<usize>,
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
    /// the iteration budget of every [`solve`](Self::solve) call and the
    /// reduction chunk come from `settings`; its `tol` is not read (each call
    /// names its own).
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
            ortho_threads: precond.solve_threads(),
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
            proj: vec![T::zero(); 2 * s * (m + 1)],
            proj_scratch: vec![T::zero(); n.div_ceil(chunk) * 2 * s * (m + 1)],
            off: vec![0; s + 1],
            delayed: Vec::with_capacity(s),
            norms: vec![0.0; s],
            norm_scratch: vec![0.0; n.div_ceil(chunk) * s],
            y: vec![T::zero(); s * m],
            jd: vec![0; s],
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
                    + [&c.h, &c.hraw, &c.su, &c.bm, &c.cs, &c.sn, &c.g]
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
            &self.proj,
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
            + vec_bytes(&self.norms)
            + vec_bytes(&self.norm_scratch)
            + vec_bytes(&self.jd)
            + vec_bytes(&self.delayed)
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
        let chunk = self.chunk;
        let kd = self.rec.map_or(0, |r| r.k);
        // every column's coefficients, and `B y` of the subspace deflated
        for (a, &c) in self.chk.iter().enumerate() {
            let col = &self.cols[c];
            let y = &mut self.y[a * m..(a + 1) * m];
            let jd = col.coefficients(m, y);
            self.jd[a] = jd;
            col.deflation_terms(kd, jd, y, &mut self.rproj[a * kd..(a + 1) * kd]);
            if !self.basis.flexible {
                col.lift(m, jd, y);
            }
        }
        let (cols, chk, jds, y, t) = (&self.cols, &self.chk, &self.jd, &self.y, &self.rproj);
        let (x, rec) = (&self.x, self.rec);
        let (pp, op) = (SendPtr::of(&mut self.panel), SendPtr::of(&mut self.out));
        let depth: usize = (0..k).map(|a| jds[a] + kd + 1).sum();
        // `out = x + Z y - U t` from the stored preconditioned basis, or `x + z - U t` from
        // `z = M^-1 (V S y)`; the panel a copy
        let form = |a: usize, r0: usize, r1: usize, z: Option<&[T]>| {
            let (c, len) = (chk[a], r1 - r0);
            // SAFETY: piece (a, chunk) alone holds these rows of column `c` of `out` and of
            // column `a` of the panel.
            let (o, p) = unsafe { (op.part(c * n + r0, len), pp.part(a * n + r0, len)) };
            o.copy_from_slice(&x[c * n + r0..c * n + r1]);
            match z {
                Some(z) => o
                    .iter_mut()
                    .zip(&z[a * n + r0..a * n + r1])
                    .for_each(|(o, &z)| *o = *o + z),
                None => {
                    for i in 0..jds[a] {
                        cols[c].zb.axpy(i * n + r0, y[a * m + i], o);
                    }
                }
            }
            if let Some(rec) = rec {
                for i in 0..kd {
                    let u = &rec.u[i * n + r0..i * n + r1];
                    let ti = t[a * kd + i];
                    o.iter_mut().zip(u).for_each(|(o, &u)| *o = *o - u * ti);
                }
            }
            p.copy_from_slice(o);
        };
        let threads = self.ortho_threads;
        if self.basis.flexible {
            threads.run(
                0,
                |cap| cap,
                || column_chunks::<T>(k, n, chunk, n * depth, |a, r0, r1| form(a, r0, r1, None)),
            );
        } else {
            threads.run(
                0,
                |cap| cap,
                || {
                    column_chunks::<T>(k, n, chunk, n * depth, |a, r0, r1| {
                        // SAFETY: piece (a, chunk) alone holds these rows of column `a`.
                        let p = unsafe { pp.part(a * n + r0, r1 - r0) };
                        p.iter_mut().for_each(|e| *e = T::zero());
                        for i in 0..jds[a] {
                            cols[chk[a]].v.axpy(i * n + r0, y[a * m + i], p);
                        }
                    })
                },
            );
            self.precond
                .apply_block(&self.panel[..k * n], &mut self.z[..k * n], k, n)?;
            let z = &self.z;
            threads.run(
                0,
                |cap| cap,
                || {
                    column_chunks::<T>(k, n, chunk, n * (kd + 2), |a, r0, r1| {
                        form(a, r0, r1, Some(z))
                    })
                },
            );
        }
        // Their true residuals, one batched operator apply.
        self.op
            .apply_block(&self.panel[..k * n], &mut self.w[..k * n], k);
        let nchunks = n.div_ceil(chunk);
        let (b, wp) = (self.b, SendPtr::of(&mut self.w));
        let sp = SendPtr::of(&mut self.norm_scratch[..k * nchunks]);
        threads.run(
            0,
            |cap| cap,
            || {
                column_chunks::<T>(k, n, chunk, 2 * k * n, |a, r0, r1| {
                    let cb = chk[a] * n;
                    // SAFETY: piece (a, chunk) alone holds these rows of column `a` and its sum.
                    let wa = unsafe { wp.part(a * n + r0, r1 - r0) };
                    let mut rn = 0.0;
                    for (wi, &bi) in wa.iter_mut().zip(&b[cb + r0..cb + r1]) {
                        *wi = bi - *wi;
                        rn += wi.magnitude_sq();
                    }
                    unsafe { sp.part(a * nchunks + r0 / chunk, 1)[0] = rn };
                })
            },
        );
        for (a, &c) in self.chk.iter().enumerate() {
            let rn: f64 = self.norm_scratch[a * nchunks..(a + 1) * nchunks]
                .iter()
                .sum();
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
                self.cols[c].open(&self.w[a * n..(a + 1) * n], beta, self.basis);
                self.deepest = self.deepest.max(1);
            }
        }
    }

    /// One block step of the columns in `act`: their pending basis vectors through
    /// the preconditioner and the operator in one batched apply each, then delayed
    /// CGS2 over the whole panel in one pass over the bases for the products
    /// ([`delayed_project`]) and one for the update ([`delayed_update`]): every
    /// pending vector orthogonalized a second time and normalized to its column's
    /// next basis vector, the operator's image orthogonalized once against the
    /// basis so extended and left pending. The step before is finished in the
    /// Hessenberg matrix, the new one entered with its pending vector taken as
    /// orthogonal, and the least-squares residual estimated from both.
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self) -> Result<(), RslabError> {
        let (n, m, sa, chunk) = (self.n, self.m, self.act.len(), self.chunk);
        {
            let (cols, act) = (&self.cols, &self.act);
            let pp = SendPtr::of(&mut self.panel[..sa * n]);
            self.ortho_threads.run(
                0,
                |cap| cap,
                || {
                    column_chunks::<T>(sa, n, chunk, sa * n, |a, r0, r1| {
                        let (c, j) = act[a];
                        // SAFETY: piece (a, chunk) alone writes these rows of column `a`.
                        cols[c]
                            .v
                            .read(j * n + r0, unsafe { pp.part(a * n + r0, r1 - r0) });
                    })
                },
            );
        }
        for a in 0..sa {
            self.off[a + 1] = self.off[a] + 2 * self.act[a].1 + 2;
        }
        self.precond
            .apply_block(&self.panel[..sa * n], &mut self.z[..sa * n], sa, n)?;
        self.op
            .apply_block(&self.z[..sa * n], &mut self.w[..sa * n], sa);
        let Self {
            cols,
            act,
            off,
            panel,
            z,
            w,
            proj,
            proj_scratch,
            ortho_threads,
            rec,
            rproj,
            rscratch,
            delayed,
            norms,
            norm_scratch,
            basis,
            ..
        } = self;
        let off = &off[..=sa];
        let (panel, z, w) = (&mut panel[..sa * n], &mut z[..sa * n], &mut w[..sa * n]);
        // deflating: the panel's components in `range(C)` first, kept per step
        if let Some(rec) = &*rec {
            ortho_threads.run(
                0,
                |cap| cap,
                || panel_project(&rec.c, rec.k, w, sa, n, chunk, rproj, rscratch),
            );
            ortho_threads.run(
                0,
                |cap| cap,
                || panel_subtract(&rec.c, rec.k, w, sa, n, chunk, rproj),
            );
        }
        {
            let (cols, act) = (&*cols, &*act);
            let layers = |a: usize| &cols[act[a].0].v;
            ortho_threads.run(
                0,
                |cap| cap,
                || delayed_project(&layers, off, panel, w, n, chunk, proj, proj_scratch),
            );
        }
        // Per column from the products: the pending vector `u` scaled to unit norm, its
        // components `s` in the basis and the norm `nrm` of the rest (Pythagoras), the
        // first pass `q` of the image `w` against the basis and `h` against the next basis
        // vector `(u - V s) / nrm`, all from one pass.
        let kd = rec.map_or(0, |r| r.k);
        delayed.clear();
        for a in 0..sa {
            let (c, k) = act[a];
            let o = &proj[off[a]..off[a + 1]];
            let unorm = o[2 * k].real().max(0.0).sqrt();
            let inv_u = if unorm > 0.0 { 1.0 / unorm } else { 0.0 };
            let iu = T::from_real(inv_u);
            let ss: f64 = o[..k].iter().map(|e| e.magnitude_sq()).sum::<f64>() * inv_u * inv_u;
            let nrm = (1.0 - ss).max(0.0).sqrt();
            let inv_nrm = if nrm > 0.0 { 1.0 / nrm } else { 0.0 };
            let sq = (0..k).fold(T::zero(), |acc, i| acc + o[i].conj() * o[k + i]);
            let h = (o[2 * k + 1] - sq) * T::from_real(inv_u * inv_u * inv_nrm);
            let col = &mut cols[c];
            for i in 0..k {
                col.su[i * m + k] = o[i] * iu;
            }
            col.su[k * m + k] = T::from_real(nrm);
            if k == 0 {
                col.g[0] = T::from_real(unorm);
            } else {
                // The step before finished: its image's remainder `u = unorm (V s' + nrm v)`.
                for i in 0..k {
                    col.hraw[i * m + k - 1] = col.hraw[i * m + k - 1] + o[i];
                }
                col.hraw[k * m + k - 1] = T::from_real(unorm * nrm);
                col.rotate(m, k - 1);
            }
            for i in 0..k {
                col.hraw[i * m + k] = o[k + i] * iu;
            }
            col.hraw[k * m + k] = h;
            for i in 0..kd {
                col.bm[k * kd + i] = rproj[i * sa + a] * iu;
            }
            delayed.push(Delayed {
                k,
                inv_u,
                inv_nrm,
                h,
                v: col.v.rows((k + 2) * n),
                z: basis.flexible.then(|| col.zb.rows((k + 1) * n)),
            });
        }
        {
            let delayed = &*delayed;
            ortho_threads.run(
                0,
                |cap| cap,
                // SAFETY: every column's rows reach its layers `k + 1` of `v` and `k` of
                // `z` (just taken), and the columns of a step are distinct.
                || unsafe {
                    delayed_update(
                        delayed,
                        off,
                        proj,
                        (panel, z, w),
                        n,
                        chunk,
                        norms,
                        norm_scratch,
                    )
                },
            );
        }
        for a in 0..sa {
            let (c, k) = self.act[a];
            let col = &mut self.cols[c];
            // SAFETY: the update wrote layers `k` and `k + 1` of `v` and layer `k` of `z`.
            unsafe {
                col.v.written((k + 2) * n);
                if self.basis.flexible {
                    col.zb.written((k + 1) * n);
                }
            }
            // The new step, its pending vector taken as orthogonal to the basis.
            col.hraw[(k + 1) * m + k] = T::from_real(self.norms[a]);
            col.gp = col.g[k];
            col.rotate(m, k);
            col.steps = Some(k + 1);
            col.stepped += 1;
            self.res[c] = None;
            self.deepest = self.deepest.max(k + 2);
        }
        self.iters += 1;
        Ok(())
    }
}

#[cfg(test)]
impl<T, A, M> BlockGmres<'_, T, A, M>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    /// The worst `||V^H V - I||_F` over the open cycles of the first `k` vectors of
    /// their bases (as many as written and finished).
    pub(super) fn orthogonality_loss(&self, k: usize) -> f64 {
        let n = self.n;
        let (mut p, mut q) = (vec![T::zero(); n], vec![T::zero(); n]);
        let mut worst = 0.0_f64;
        for col in &self.cols {
            let Some(j) = col.steps else { continue };
            let mut f = 0.0;
            for i in 0..j.min(k) {
                col.v.read(i * n, &mut p);
                for l in 0..j.min(k) {
                    col.v.read(l * n, &mut q);
                    let e = if i == l { T::one() } else { T::zero() };
                    f += (dotc(&p, &q) - e).magnitude_sq();
                }
            }
            worst = worst.max(f.sqrt());
        }
        worst
    }

    /// Column `c`'s least-squares residual estimate, relative.
    pub(super) fn estimate(&self, c: usize) -> f64 {
        self.cols[c].estimate(self.bnorm[c])
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
            let sb = |i: usize, j: usize| col.su[i * m + j].to_c();
            // `M1 = H_bar^H H_bar`, `M2 = H^H S` (row-major `d x d`): the vectors the operator
            // was applied to are `V S`
            let mut m1 = vec![Cx::new(0.0, 0.0); d * d];
            let mut m2 = vec![Cx::new(0.0, 0.0); d * d];
            for i in 0..d {
                for j in 0..d {
                    m1[i * d + j] = (0..=d).map(|r| hb(r, i).conj() * hb(r, j)).sum();
                    m2[i * d + j] = (0..=j).map(|r| hb(r, i).conj() * sb(r, j)).sum();
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
        let col_s = |c: usize, i: usize, j: usize| self.cols[c].su[i * m + j].to_c();
        // the directions, orthonormal
        let mut u: Vec<T> = Vec::new();
        for (_, c, theta, g) in cands {
            let kept = u.len() / n;
            if kept >= k {
                break;
            }
            let d = g.len();
            // the Ritz vectors in the solution space: `Z g`, as `M^-1 (V S g)` where `Z`
            // is not kept
            let ritz = |b: &Stored<T>, g: Vec<Cx>| {
                let pairs = [(theta, g)];
                T::combine_ritz(&|j, a, out| b.axpy(j * n, a, out), n, d, &pairs, k - kept)
            };
            let cand = if self.basis.flexible {
                ritz(&self.cols[c].zb, g)
            } else {
                let sg = (0..d)
                    .map(|i| (i..d).map(|j| col_s(c, i, j) * g[j]).sum())
                    .collect();
                let vg = ritz(&self.cols[c].v, sg);
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
pub(super) fn block_bytes<T: Scalar>(
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
    // the Hessenberg matrix rotated and as built, `S`, the two rotations, `g`
    let column = 2 * (m + 1) * m + m * m + 2 * m + (m + 1);
    // the panels `x`, `out`, `panel`, `z`, `w`; a step's products and their partial
    // sums; the coefficients of a check
    let panels = 5 * n * s; // lit: the five n x s panels of `BlockGmres::new`
    let nchunks = n.div_ceil(chunk.max(1));
    let shared = panels + 2 * s * (m + 1) * (1 + nchunks) + s * m;
    // `bnorm`, `norms`, their partial sums, `res`, `jd`, `off` (one more), `chk`,
    // `closing`, `act`, `delayed`
    let per_rhs = (2 + nchunks) * std::mem::size_of::<f64>()
        + std::mem::size_of::<Option<f64>>()
        + 4 * std::mem::size_of::<usize>() // lit: `jd`, `off`, `chk` and `closing`
        + std::mem::size_of::<(usize, usize)>()
        + std::mem::size_of::<Delayed<T>>();
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
/// **Orthogonalization:** delayed CGS2 (classical Gram-Schmidt with an
/// unconditional second pass taken in the next step's pass over the basis, see
/// the module-level note), not the MGS+DGKS of the single-RHS [`gmres`]: a block
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
