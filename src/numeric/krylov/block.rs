//! Block GMRES for many right-hand sides with CGS2 over the whole panel, as a
//! solve that can be continued to a tighter tolerance in the same Krylov space.

use super::util::*;
use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// Outcome of a block (multi-RHS) Krylov solve.
#[derive(Debug, Clone)]
pub struct BlockKrylovResult<T> {
    /// Solutions, column-major `nxs` (RHS `c` is the slice `[c*n, (c+1)*n)`).
    pub x: Vec<T>,
    /// Block iterations performed (the RHS advance in lockstep).
    pub iters: usize,
    /// `true` iff **every** RHS reached `tol`.
    pub converged: bool,
    /// Per-RHS final relative residual `||b_c - A x_c|| / ||b_c||`.
    pub final_res: Vec<f64>,
    /// Why the block iteration stopped: `Converged` when every RHS met `tol`,
    /// `Stalled` when the monitor cancelled it, else `MaxIter`.
    pub stop: StopReason,
}

/// One right-hand side's Arnoldi state: its basis, the Hessenberg matrix under
/// its Givens rotations and the rotated least-squares right-hand side of the
/// open restart cycle.
struct Column<T> {
    /// The basis, layer `i` the slice `v[i*n..(i+1)*n]`: the restart's layers
    /// reserved, the ones the cycle has written resident ([`basis`], [`grow`]).
    v: Vec<T>,
    /// The preconditioned basis `z_i = M^-1 v_i` of the steps taken, laid out as
    /// `v`: the solution is formed from the vectors the Arnoldi relation holds.
    zb: Vec<T>,
    /// Row-major `(m+1) x m`, entry `(i, j)` at `h[i*m + j]`.
    h: Vec<T>,
    cs: Vec<T>,
    sn: Vec<T>,
    g: Vec<T>,
    /// Steps of the open cycle; `None` before the first and after a cycle closed.
    steps: Option<usize>,
}

impl<T: Scalar> Column<T> {
    fn new(n: usize, m: usize) -> Self {
        Column {
            v: basis(n * (m + 1)),
            zb: basis(n * m),
            h: vec![T::zero(); (m + 1) * m],
            cs: vec![T::zero(); m],
            sn: vec![T::zero(); m],
            g: vec![T::zero(); m + 1],
            steps: None,
        }
    }

    /// The least-squares residual of the open cycle relative to `bnorm`: the
    /// residual of the solution it would hand out now, in exact arithmetic.
    fn estimate(&self, bnorm: f64) -> f64 {
        self.steps
            .map_or(f64::INFINITY, |j| self.g[j].magnitude() / bnorm)
    }

    /// Open a cycle on the residual `r` of norm `beta > 0`.
    fn open(&mut self, r: &[T], beta: f64) {
        let n = r.len();
        self.h.iter_mut().for_each(|e| *e = T::zero());
        self.g.iter_mut().for_each(|e| *e = T::zero());
        self.g[0] = T::from_real(beta);
        grow(&mut self.v, n);
        let inv = T::from_real(1.0 / beta);
        for (v, &ri) in self.v[..n].iter_mut().zip(r) {
            *v = ri * inv;
        }
        self.steps = Some(0);
    }

    /// The open cycle's coefficients `y` (back substitution on the leading
    /// well-conditioned block, so a rank-deficient Hessenberg breaks down
    /// deterministically instead of dividing by about zero) and the solution
    /// `x0 + Z y` into `out`.
    #[allow(clippy::needless_range_loop)]
    fn correction(&self, m: usize, y: &mut [T], x0: &[T], out: &mut [T]) {
        let n = out.len();
        out.copy_from_slice(x0);
        let jd = well_conditioned_dim(&self.h, m, self.steps.unwrap_or(0));
        for i in (0..jd).rev() {
            let mut acc = self.g[i];
            for k in (i + 1)..jd {
                acc = acc - self.h[i * m + k] * y[k];
            }
            y[i] = acc * self.h[i * m + i].recip();
        }
        for i in 0..jd {
            let yi = y[i];
            for (o, &z) in out.iter_mut().zip(&self.zb[i * n..(i + 1) * n]) {
                *o = *o + z * yi;
            }
        }
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
/// The price is the second basis in memory.
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
        // two bases a column: `V` and `Z`
        let m = settings.restart_for(n, s, std::mem::size_of::<T>(), 2);
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
        Ok(BlockGmres {
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
            cols: (0..s).map(|_| Column::new(n, m)).collect(),
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
        })
    }

    /// Heap bytes the solve holds now: every column's basis as far as it is
    /// written (the restart's reservation is address space only), the
    /// Hessenberg matrices, the panels and the reduction scratch.
    /// [`KrylovSettings::block_bytes`] predicts it for a given depth.
    pub fn resident_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        let sz = std::mem::size_of::<T>() as u64;
        let cols: u64 = self
            .cols
            .iter()
            .map(|c| {
                (c.v.len() + c.zb.len()) as u64 * sz
                    + vec_bytes(&c.h)
                    + vec_bytes(&c.cs)
                    + vec_bytes(&c.sn)
                    + vec_bytes(&c.g)
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
        ]
        .into_iter()
        .map(vec_bytes)
        .sum::<u64>()
            + vec_bytes(&self.bnorm)
            + vec_bytes(&self.res)
            + vec_bytes(&self.wnorm0)
            + vec_bytes(&self.reorth)
            + vec_bytes(&self.off)
            + vec_bytes(&self.act)
            + vec_bytes(&self.chk)
    }

    /// The operator the solve runs on, for the caller's own products between
    /// [`solve`](Self::solve) calls (a diagnosis of an unconverged rung).
    pub fn operator(&self) -> &'a A {
        self.op
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
    /// **Monitor:** after every check of true residuals (the start, every
    /// restart, the hand-out) `mon` receives `(iterations of this call, worst
    /// residual, columns stepping next)`, the worst over the columns' last true
    /// residuals and their open cycles' estimates, and returns whether to go
    /// on; `false` ends the call with `StopReason::Stalled` and the solutions
    /// so far.
    pub fn solve(
        &mut self,
        tol: f64,
        mut mon: Option<&mut dyn FnMut(usize, f64, usize) -> bool>,
    ) -> Result<BlockKrylovResult<T>, RslabError> {
        let start = self.iters;
        let mut stalled = false;
        loop {
            let budget = self.iters - start < self.max_iter;
            self.select_active(tol);
            let stepping = budget && !self.act.is_empty();
            // The residual check: with columns stepping, only the full cycles
            // (they restart from their true residual); else every column that
            // stepped since its last check.
            let checked = self.check(!stepping);
            if checked {
                if let Some(mo) = mon.as_mut() {
                    // NaN-honest: a non-finite residual is the worst one
                    let worst = (0..self.s)
                        .map(|c| match self.res[c] {
                            Some(r) => r,
                            None => self.cols[c].estimate(self.bnorm[c]),
                        })
                        .fold(
                            0.0_f64,
                            |a, r| if r.is_nan() { f64::INFINITY } else { a.max(r) },
                        );
                    if !mo(self.iters - start, worst, self.act.len()) {
                        stalled = true;
                        break;
                    }
                }
                if budget {
                    self.restart_above(tol);
                }
                continue;
            }
            if !stepping {
                break;
            }
            self.step()?;
        }
        // A cancel leaves the columns stepped since their last check unformed.
        self.check(true);
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
        })
    }

    /// `act`: the columns that can step toward `tol`, an open cycle short of the
    /// restart with its estimate above `tol`.
    fn select_active(&mut self, tol: f64) {
        self.act.clear();
        for c in 0..self.s {
            let col = &self.cols[c];
            if let Some(j) = col.steps {
                if j < self.m && col.estimate(self.bnorm[c]) > tol {
                    self.act.push((c, j));
                }
            }
        }
    }

    /// Form the solutions of the unchecked columns (all of them with `all`, else
    /// those whose cycle reached the restart) and their true residuals; the
    /// residual vectors stay in `w` (compact, in the order of `chk`).
    /// Whether any column was checked.
    fn check(&mut self, all: bool) -> bool {
        let (n, m) = (self.n, self.m);
        self.chk.clear();
        for c in 0..self.s {
            if self.res[c].is_none() && (all || self.cols[c].steps == Some(m)) {
                self.chk.push(c);
            }
        }
        let k = self.chk.len();
        if k == 0 {
            return false;
        }
        // The open cycles' solutions `x + Z y`, from the stored preconditioned basis.
        for (a, &c) in self.chk.iter().enumerate() {
            let (cb, pb) = (c * n, a * n);
            self.cols[c].correction(
                m,
                &mut self.y,
                &self.x[cb..cb + n],
                &mut self.out[cb..cb + n],
            );
            self.panel[pb..pb + n].copy_from_slice(&self.out[cb..cb + n]);
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
        true
    }

    /// Restart every column the last check left above `tol`: its cycle closes
    /// on the solution formed (`x <- out`) and a new one opens on its true
    /// residual (in `w` at its position in `chk`).
    fn restart_above(&mut self, tol: f64) {
        let n = self.n;
        for (a, &c) in self.chk.iter().enumerate() {
            let r = self.res[c].unwrap_or(f64::INFINITY);
            if r <= tol || self.bnorm[c] == 0.0 {
                continue;
            }
            let cb = c * n;
            self.x[cb..cb + n].copy_from_slice(&self.out[cb..cb + n]);
            self.cols[c].open(&self.w[a * n..(a + 1) * n], r * self.bnorm[c]);
            self.deepest = self.deepest.max(1);
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
            self.panel[a * n..(a + 1) * n].copy_from_slice(&self.cols[c].v[j * n..(j + 1) * n]);
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
            ..
        } = self;
        let (cols, act, off, chunk) = (&*cols, &*act, &off[..=sa], *chunk);
        let layers = |a: usize| &cols[act[a].0].v[..];
        let w = &mut w[..sa * n];
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
            for i in 0..=j {
                let p = self.off[a] + i;
                col.h[i * m + j] = if any {
                    self.proj1[p] + self.proj2[p]
                } else {
                    self.proj1[p]
                };
            }
            grow(&mut col.zb, (j + 1) * n);
            col.zb[j * n..(j + 1) * n].copy_from_slice(&self.z[a * n..(a + 1) * n]);
            let wa = &self.w[a * n..(a + 1) * n];
            let hn = norm2(wa);
            col.h[(j + 1) * m + j] = T::from_real(hn);
            grow(&mut col.v, (j + 2) * n);
            let next = &mut col.v[(j + 1) * n..(j + 2) * n];
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                for (v, &wi) in next.iter_mut().zip(wa) {
                    *v = wi * inv;
                }
            } else {
                next.iter_mut().for_each(|v| *v = T::zero());
            }
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
            self.res[c] = None;
            self.deepest = self.deepest.max(j + 2);
        }
        self.iters += 1;
        Ok(())
    }
}

/// Heap bytes of a [`BlockGmres`] of `s` right-hand sides of length `n` at
/// restart `m` and reduction chunk `chunk` once its columns' bases are `depth`
/// vectors deep: the allocations of [`BlockGmres::new`] and the written bases
/// ([`BlockGmres::resident_bytes`] of a solve at that depth in every column).
pub(super) fn block_bytes<T>(n: usize, s: usize, m: usize, chunk: usize, depth: usize) -> u64 {
    let sz = std::mem::size_of::<T>();
    // per column: the basis, its preconditioned steps, the Hessenberg matrix, the two
    // rotations, `g`
    let column = n * depth + n * depth.saturating_sub(1) + (m + 1) * m + 2 * m + (m + 1);
    // the panels `x`, `out`, `panel`, `z`, `w`; the two projections and their partial
    // sums; `y`
    let panels = 5 * n * s; // lit: the five n x s panels of `BlockGmres::new`
    let shared = panels + 2 * s * (m + 1) + n.div_ceil(chunk.max(1)) * s * (m + 1) + m;
    // `bnorm`, `wnorm0`, `res`, `reorth`, `off` (one more), `act`, `chk`
    let per_rhs = 2 * std::mem::size_of::<f64>()
        + std::mem::size_of::<Option<f64>>()
        + std::mem::size_of::<bool>()
        + 2 * std::mem::size_of::<usize>()
        + std::mem::size_of::<(usize, usize)>();
    ((s * column + shared) * sz + s * per_rhs + std::mem::size_of::<usize>()) as u64
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
/// **Memory:** every column's bases `V` and `Z` reserve `n*(2*restart+1)`
/// scalars of address space and hold what its cycle has written; the solve holds about
/// [`KrylovSettings::block_bytes`] for the depth it reaches. An unset
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
/// **Monitor:** see [`BlockGmres::solve`].
pub fn gmres_block<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    settings: &KrylovSettings,
    x0: Option<&[T]>,
    mon: Option<&mut dyn FnMut(usize, f64, usize) -> bool>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    BlockGmres::new(op, b, s, precond, settings, x0)?.solve(settings.tol, mon)
}
