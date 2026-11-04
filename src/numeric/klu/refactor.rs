//! The numeric-only refactorization on a frozen pattern and pivot order.

use super::*;
use crate::constants::{KLU_REPLAY_NS_PER_WORK, PIECES_PER_WORKER, PIPE_RUN};
use crate::parallel::{Choice, Dispatch, Go, PerScalar, Pick, Prior, Rayon, ScalarRates, Work};
use rspool::constants::TUNE_ARMS;

/// What a unit of replay work costs a thread, learned from the sequential
/// replays (a replay inside which a section went wide teaches it nothing),
/// and how wide the blocks replay side by side: the widths their refactors
/// were fastest on ([`Work`]).
static REPLAY_RATE: ScalarRates = ScalarRates::new(KLU_REPLAY_NS_PER_WORK);

/// How a block replays: the way its replays of its size were fastest
/// (sequentially, level by level, pipelined on 2, 4, ... threads), a way
/// not measured yet counted at what the pool's section cost predicts for
/// it ([`Choice`]). What a parallel replay pays beyond its section cost (the
/// hand-offs of a pipeline, the levels that find the workers asleep) is in
/// the measurement.
static MODES: PerScalar<Choice> = PerScalar::choices();

impl ReplayPlan {
    /// Whether the user's structural floors decide in place of the time.
    fn floors(&self) -> bool {
        self.min_work.is_some() || self.min_ratio.is_some()
    }

    /// Whether block `bp` of `bn` columns clears the user's floors.
    fn block_pays(&self, bp: &BlockPlan, bn: usize) -> bool {
        let nlev = bp.level_work.len().max(1);
        self.min_work.is_none_or(|w| bp.work >= w)
            && self.min_ratio.is_none_or(|r| bn as f64 / nlev as f64 >= r)
    }

    /// Whether the blocks together clear the user's floors.
    fn blocks_pay(&self) -> bool {
        self.min_work.is_none_or(|w| self.total >= w)
            && self
                .min_ratio
                .is_none_or(|r| self.total as f64 >= r * self.largest as f64)
    }
}

/// What a refactor decides its replay on: the section cost of the pool it
/// runs on and the replay rate, for the gates and for the ways not measured
/// yet.
pub(super) struct Decide {
    dispatch: Dispatch,
    threads: usize,
    /// `On`, or floors the user set: every section is free, and the time
    /// picks only how a block replays side by side, not whether.
    forced: bool,
    /// ns per unit of replay work.
    seq: f64,
}

impl Decide {
    fn new<T: Scalar>(plan: &ReplayPlan) -> Self {
        let threads = crate::parallel::threads();
        let forced = plan.policy == KluParallel::On || plan.floors();
        Decide {
            dispatch: if forced {
                crate::parallel::kept(true, threads)
            } else {
                crate::parallel::dispatch()
            },
            threads,
            forced,
            seq: REPLAY_RATE.of::<T>().ns_per_op(),
        }
    }

    /// On `dispatch` and `threads`, a unit of replay work a ns (tests).
    #[cfg(test)]
    pub(super) fn on(dispatch: Dispatch, threads: usize) -> Self {
        Decide {
            dispatch,
            threads,
            forced: false,
            seq: 1.0,
        }
    }

    fn width(&self, work: f64, pieces: usize) -> usize {
        self.dispatch.width(work, pieces).min(self.threads).max(1)
    }

    /// The threads level `l` of `bp` replays on, and its time on them.
    fn level(&self, bp: &BlockPlan, l: usize) -> (usize, f64) {
        let work = bp.level_work[l] as f64 * self.seq;
        let p = (bp.ptr[l + 1] - bp.ptr[l]) as usize;
        match self.width(work, p) {
            k if k >= 2 => (
                k,
                work * p.div_ceil(k) as f64 / p as f64 + self.dispatch.cost(k),
            ),
            _ => (1, work),
        }
    }

    /// The ways block `bp` of `bn` columns, `serial` ns sequentially, can
    /// replay, each at a number of its own, with what is predicted for it
    /// where it is available: 0 sequentially (unless every section is
    /// free), 1 level by level (where a level is worth a section), `a + 1`
    /// pipelined on `2^a` threads, the last one on all the pool's. A side by
    /// side replay takes no less than the block's longest chain, and a
    /// pipelined one on `k` threads is expected at Brent's bound for a
    /// greedy schedule: its share of the work, the chain on top.
    fn ways(
        &self,
        bp: &BlockPlan,
        bn: usize,
        serial: f64,
    ) -> ([Option<(Mode, Prior)>; TUNE_ARMS], usize) {
        let mut ways = [None; TUNE_ARMS];
        if !self.forced {
            ways[0] = Some((Mode::Serial, Prior::exactly(serial)));
        }
        let span = bp.span as f64 * self.seq;
        let cap = self.threads.min(bn);
        if (bp.widest as f64 * self.seq) >= self.dispatch.break_even() {
            let levels = (0..bp.level_work.len()).map(|l| self.level(bp, l).1).sum();
            let least = (serial / cap.max(1) as f64).max(span);
            ways[1] = Some((
                Mode::Levels,
                Prior {
                    expected: levels,
                    least,
                },
            ));
        }
        let mut n = 2;
        while n < TUNE_ARMS && cap >= 2 {
            let k = (1usize << (n - 1)).min(cap);
            let share = serial * bn.div_ceil(k) as f64 / bn as f64;
            let cost = self.dispatch.cost(k);
            let prior = Prior {
                expected: share + span + cost,
                least: share.max(span) + cost,
            };
            ways[n] = Some((Mode::Pipelined(k), prior));
            n += 1;
            if k == cap {
                break;
            }
        }
        (ways, n)
    }

    /// How block `bp` of `bn` columns replays: sequentially below the
    /// user's floors and below what any section is worth on the pool; else
    /// the way its replays of its size were fastest (`choices`), with the
    /// pick its replay is measured for.
    pub(super) fn mode(
        &self,
        plan: &ReplayPlan,
        bp: &BlockPlan,
        bn: usize,
        choices: &Choice,
    ) -> (Mode, Option<Pick>) {
        let serial = bp.work as f64 * self.seq;
        if plan.policy != KluParallel::On && plan.floors() && !plan.block_pays(bp, bn) {
            return (Mode::Serial, None);
        }
        if serial.is_nan() || serial < self.dispatch.break_even() {
            return (Mode::Serial, None);
        }
        let (ways, n) = self.ways(bp, bn, serial);
        let open = ways.iter().flatten().count();
        if open < 2 {
            return (
                ways.iter().flatten().next().map_or(Mode::Serial, |w| w.0),
                None,
            );
        }
        let pick = choices.pick(serial, bp.work as f64, n, |o| ways[o].map(|w| w.1));
        (ways[pick.option].map_or(Mode::Serial, |w| w.0), Some(pick))
    }

    /// Whether the blocks may replay side by side at all: two of them or
    /// more, under a policy that allows it and the user's floors.
    fn blocks_may(&self, plan: &ReplayPlan, nblocks: usize) -> bool {
        nblocks >= 2
            && plan.policy != KluParallel::Off
            && (plan.policy == KluParallel::On || !plan.floors() || plan.blocks_pay())
    }
}

impl<T: Scalar> KluSolver<T> {
    /// Numeric-only refactorization: replay the stored pattern and pivot
    /// sequence on new values with the **same** sparsity pattern. No symbolic
    /// work, no pivot search, the fast path for frequency sweeps and Newton
    /// steps. Fails with a pattern-mismatch error if `a`'s pattern deviates
    /// from the factored one, and with [`RslabError::SingularBasis`] if a
    /// frozen pivot becomes zero (re-`factor` with pivoting in that case).
    /// After an error the factorization is invalid; a subsequent successful
    /// `refactor` or a fresh `factor` makes it valid again.
    ///
    /// Every refactor decides how its blocks replay with the pool it runs on
    /// ([`KluSettings::parallel`]): sequentially, level by level, pipelined,
    /// or blocks side by side, as the replay work of the pivot-final pattern
    /// is worth there.
    // The replay loops index several parallel arrays at offset positions;
    // iterator forms would obscure the offset arithmetic.
    #[allow(clippy::needless_range_loop)]
    pub fn refactor(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
        a.validate()?;
        let t = crate::clock::Instant::now();
        // Cloned out of the factors so the borrow below stays exclusive; the
        // flag is read-only for us either way.
        let int_flag = self.factors.interrupt.clone();
        let int_flag = int_flag.as_deref();
        let f = &mut self.factors;
        if a.n != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: a.n,
            });
        }
        if a.nnz() != f.nnz_a {
            return Err(pattern_mismatch());
        }
        // Into the kept buffer; it becomes `rs_inv` once the replay succeeds.
        let mut rs_inv = std::mem::take(&mut f.rs_next);
        row_scale_inv_into(a, f.scaled, &mut rs_inv);

        // Branch-free pattern verification against the recorded program: every
        // entry must map to the exact final position it had at factor time
        // (this subsumes the old per-column F-walk and leftover checks).
        {
            let mut acc: Ki = 0;
            for (k, &r) in a.row_idx.iter().enumerate() {
                acc |= (f.pinv[r] as Ki) ^ f.scatter_expect[k];
            }
            if acc != 0 {
                return Err(pattern_mismatch());
            }
        }

        // Per-block replay jobs over disjoint value ranges: L/U/F entries and
        // `udiag` of a block are contiguous (`colptr[bs]..colptr[be]`), so the
        // mutable arrays split cleanly and the blocks replay independently,
        // side by side where it pays (bit-identical: per-block work is
        // sequential and shares nothing).
        struct RJob<'s, T> {
            b: usize,
            l_v: &'s mut [T],
            u_v: &'s mut [T],
            ud: &'s mut [T],
            f_v: &'s mut [T],
        }
        let nblocks = f.block_ptr.len() - 1;
        let mut x = std::mem::take(&mut f.replay_x);
        // How each planned block replays this time, on the pool this
        // refactor runs on and at the rates measured so far.
        let plan = &f.plan;
        let decide = Decide::new::<T>(plan);
        let rate = REPLAY_RATE.of::<T>();
        let choices = MODES.of::<T>();
        let serial = plan.total as f64 * decide.seq;
        let cheap = serial.is_nan() || serial < decide.dispatch.break_even();
        let mut modes = std::mem::take(&mut f.modes);
        let mut picks = std::mem::take(&mut f.picks);
        modes.clear();
        picks.clear();
        for bp in &plan.blocks {
            let (m, p) = if cheap {
                (Mode::Serial, None)
            } else {
                let bn = f.block_ptr[bp.b + 1] - f.block_ptr[bp.b];
                decide.mode(plan, bp, bn, choices)
            };
            modes.push(m);
            picks.push(p);
        }
        #[cfg(test)]
        if let Some(m) = plan.forced {
            modes.iter_mut().for_each(|x| *x = m);
            picks.iter_mut().for_each(|p| *p = None);
        }
        // The blocks side by side, on the width the refactors of this work
        // were fastest on (as wide as they go where every section is free).
        let work = Work::of(plan.total as f64, rate);
        let pieces = ((plan.total / plan.largest.max(1)) as usize).min(nblocks);
        let width_blocks = if cheap || !decide.blocks_may(plan, nblocks) {
            1
        } else if decide.forced {
            decide.width(serial, pieces)
        } else {
            match work.go::<Rayon>(pieces) {
                Go::Wide(k) => k,
                Go::Here => 1,
            }
        };
        let wide = width_blocks > 1 || modes.iter().any(|m| *m != Mode::Serial);
        // The pipelined replays: this refactor's number on the columns; and
        // for every parallel replay a zero work vector per worker, kept
        // across refactors.
        let mut ready = std::mem::take(&mut f.ready);
        let mut kept_xs = std::mem::take(&mut f.replay_xs);
        if modes.iter().any(|m| matches!(m, Mode::Pipelined(_))) {
            ready.next(f.n);
        }
        if wide && kept_xs.len() < decide.threads {
            kept_xs.resize_with(decide.threads, Vec::new);
        }
        let xs: Vec<std::sync::Mutex<Vec<T>>> =
            kept_xs.into_iter().map(std::sync::Mutex::new).collect();
        // The jobs in block order, each split off the front of the rest.
        let mut next_job = {
            let (mut lv, mut uv) = (f.l_val.as_mut_slice(), f.u_val.as_mut_slice());
            let (mut ud, mut fv) = (f.udiag.as_mut_slice(), f.f_val.as_mut_slice());
            let (block_ptr, l_colptr) = (&f.block_ptr, &f.l_colptr);
            let (u_colptr, f_colptr) = (&f.u_colptr, &f.f_colptr);
            move |b: usize| {
                let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
                let (a1, r1) = std::mem::take(&mut lv).split_at_mut(l_colptr[be] - l_colptr[bs]);
                lv = r1;
                let (a2, r2) = std::mem::take(&mut uv).split_at_mut(u_colptr[be] - u_colptr[bs]);
                uv = r2;
                let (a3, r3) = std::mem::take(&mut ud).split_at_mut(be - bs);
                ud = r3;
                let (a4, r4) = std::mem::take(&mut fv).split_at_mut(f_colptr[be] - f_colptr[bs]);
                fv = r4;
                RJob {
                    b,
                    l_v: a1,
                    u_v: a2,
                    ud: a3,
                    f_v: a4,
                }
            }
        };
        let (block_ptr, col_perm) = (&f.block_ptr, &f.col_perm);
        let (l_colptr, l_rowidx) = (&f.l_colptr, &f.l_rowidx);
        let (u_colptr, u_rowidx) = (&f.u_colptr, &f.u_rowidx);
        let f_colptr = &f.f_colptr;
        let scatter_target = &f.scatter_target;
        let rs_inv_ref = &rs_inv;
        // Raw column-disjoint views of one block's value arrays for the
        // parallel replays: every column writes only its own L/U/F/diag
        // slots and reads L columns complete before it (fenced by the
        // per-level join, or awaited by the pipeline), so the shared-mutable
        // access is race-free.
        struct BlockPtrs<T> {
            l_v: PanelPtr<T>,
            u_v: PanelPtr<T>,
            ud: PanelPtr<T>,
            f_v: PanelPtr<T>,
        }
        impl<T> Clone for BlockPtrs<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<T> Copy for BlockPtrs<T> {}

        // Replay one column through the recorded program: scatter, eliminate
        // in the stored topological order (bit-identical to `factor_impl`'s
        // pass 3: same `fmadd`, same per-column order), pivot, emit L.
        // SAFETY: caller guarantees exclusive ownership of column `j`'s output
        // slots and completed dependency columns (see `BlockPtrs`); `x` is this
        // caller's scratch, all-zero on entry and left all-zero.
        // With `wait`, the column runs in a pipelined replay: it consumes `L`
        // column `pr` only once `wait.0[pr]` carries the refactor's number
        // `wait.1`, so the elimination against the columns done long ago
        // overlaps the ones still being replayed.
        let replay_col = |j: usize,
                          bs: usize,
                          bases: (usize, usize, usize),
                          p: BlockPtrs<T>,
                          x: &mut [T],
                          wait: Option<(&[std::sync::atomic::AtomicU32], u32)>|
         -> Result<(), RslabError> {
            let (l_base, u_base, f_base) = bases;
            let c = col_perm[j];
            unsafe {
                for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                    let r = a.row_idx[k];
                    let sv = a.values[k] * T::from_real(rs_inv_ref[r]);
                    let tv = scatter_target[k];
                    if tv & KI_FBIT != 0 {
                        *p.f_v.get().add((tv & !KI_FBIT) as usize - f_base) = sv;
                    } else {
                        x[tv as usize - bs] = sv;
                    }
                }
                for k in u_colptr[j]..u_colptr[j + 1] {
                    let pr = u_rowidx[k] as usize;
                    if let Some((done, epoch)) = wait {
                        // Acquire pairs with the Release that published it.
                        let mut spins = 0u32;
                        while done[pr].load(std::sync::atomic::Ordering::Acquire) != epoch {
                            spins += 1;
                            if spins & crate::constants::SPIN_YIELD_MASK == 0 {
                                std::thread::yield_now();
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                    }
                    let xu = x[pr - bs];
                    x[pr - bs] = T::zero();
                    *p.u_v.get().add(k - u_base) = xu;
                    let nxu = T::zero() - xu;
                    for kl in l_colptr[pr]..l_colptr[pr + 1] {
                        let lr = l_rowidx[kl] as usize - bs;
                        debug_assert!(lr < x.len());
                        *x.get_unchecked_mut(lr) =
                            fmadd(nxu, *p.l_v.get().add(kl - l_base), *x.get_unchecked(lr));
                    }
                }
                let d = x[j - bs];
                x[j - bs] = T::zero();
                if d.magnitude() == 0.0 || !d.is_finite() {
                    return Err(RslabError::SingularBasis { column: c });
                }
                *p.ud.get().add(j - bs) = d;
                for k in l_colptr[j]..l_colptr[j + 1] {
                    let lr = l_rowidx[k] as usize - bs;
                    *p.l_v.get().add(k - l_base) = x[lr] / d;
                    x[lr] = T::zero();
                }
            }
            Ok(())
        };

        // The lower of two errors of side-by-side columns: the replay reports
        // what the sequential one does, the lowest failing column (an
        // interrupt before any).
        let keep_lowest = |slot: &std::sync::Mutex<Option<RslabError>>, e: RslabError| {
            let mut slot = slot.lock().unwrap_or_else(|e| e.into_inner());
            let lower = match (&e, &*slot) {
                (_, None) => true,
                (RslabError::Interrupted, _) => true,
                (
                    RslabError::SingularBasis { column: c1 },
                    Some(RslabError::SingularBasis { column: c0 }),
                ) => c1 < c0,
                _ => false,
            };
            if lower {
                *slot = Some(e);
            }
        };
        // A work vector of this worker's: a free kept one, else a fresh one.
        let xs_ref = &xs;
        let with_work = |k: usize, run: &mut dyn FnMut(&mut Vec<T>)| {
            let free =
                (xs_ref.iter().cycle().skip(k).take(xs_ref.len())).find_map(|m| m.try_lock().ok());
            match free {
                Some(mut w) => run(&mut w),
                None => run(&mut Vec::new()),
            }
        };
        // Level by level: each level on as many threads as it is worth, a
        // level not worth two on the calling thread; on an error the level's
        // lowest failing column is reported.
        let decide_ref = &decide;
        let replay_levels = |bp: &BlockPlan,
                             bs: usize,
                             bases: (usize, usize, usize),
                             p: BlockPtrs<T>,
                             x: &mut [T]|
         -> Result<(), RslabError> {
            let bn = x.len();
            for l in 0..bp.level_work.len() {
                interrupt_check(int_flag)?;
                let cols = &bp.cols[bp.ptr[l] as usize..bp.ptr[l + 1] as usize];
                let (k, _) = decide_ref.level(bp, l);
                if k < 2 {
                    for &j in cols {
                        if let Err(e) = replay_col(j as usize, bs, bases, p, x, None) {
                            x.fill(T::zero());
                            return Err(e);
                        }
                    }
                    continue;
                }
                let per = cols.len().div_ceil(cols.len().min(k * PIECES_PER_WORKER));
                let failed: std::sync::Mutex<Option<RslabError>> = std::sync::Mutex::new(None);
                crate::parallel::run_width(cols.len().div_ceil(per), k, &|q| {
                    with_work(q, &mut |w: &mut Vec<T>| {
                        if w.len() < bn {
                            w.resize(bn, T::zero());
                        }
                        let n = cols.len();
                        for &j in &cols[(q * per).min(n)..((q + 1) * per).min(n)] {
                            // SAFETY: the columns of a level write disjoint
                            // slots and read `L` columns of earlier levels,
                            // complete before this level started.
                            if let Err(e) = replay_col(j as usize, bs, bases, p, &mut w[..bn], None)
                            {
                                w.fill(T::zero());
                                keep_lowest(&failed, e);
                                return;
                            }
                        }
                    });
                });
                if let Some(e) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                    return Err(e);
                }
            }
            Ok(())
        };
        // A pipelined block's columns side by side on `k` threads (NICSLU's
        // pipeline mode, Chen, Wang and Yang, TCAD 2013), taken in order, one
        // at a time, from a shared counter: a column waits until every `L`
        // column it reads is replayed, all of them earlier columns, taken
        // before it by pieces that run. So the replay completes on any pool,
        // one that runs its pieces one after the other included. A failing
        // column still releases its dependants, and the lowest failing column
        // is reported, as by the sequential replay; an interrupt stops the
        // pieces between runs.
        let ready_ref = &ready;
        let replay_pipelined = |(bs, be): (usize, usize),
                                bases: (usize, usize, usize),
                                p: BlockPtrs<T>,
                                k: usize|
         -> Result<(), RslabError> {
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AOrd};
            let (bn, epoch, done) = (be - bs, ready_ref.epoch, &ready_ref.done);
            let next = AtomicUsize::new(bs);
            let stop = AtomicBool::new(false);
            let failed: std::sync::Mutex<Option<RslabError>> = std::sync::Mutex::new(None);
            let pieces = k.min(bn.div_ceil(PIPE_RUN)).max(1);
            crate::parallel::run_width(pieces, pieces, &|q| {
                with_work(q, &mut |w: &mut Vec<T>| {
                    if w.len() < bn {
                        w.resize(bn, T::zero());
                    }
                    let w = &mut w[..bn];
                    while !stop.load(AOrd::Relaxed) {
                        if let Err(e) = interrupt_check(int_flag) {
                            stop.store(true, AOrd::Relaxed);
                            keep_lowest(&failed, e);
                            return;
                        }
                        let j0 = next.fetch_add(PIPE_RUN, AOrd::Relaxed);
                        if j0 >= be {
                            return;
                        }
                        for j in j0..(j0 + PIPE_RUN).min(be) {
                            // SAFETY: column `j` writes only its own slots
                            // and reads `L` columns once published.
                            if let Err(e) = replay_col(j, bs, bases, p, w, Some((done, epoch))) {
                                w.fill(T::zero());
                                keep_lowest(&failed, e);
                            }
                            done[j].store(epoch, AOrd::Release);
                        }
                    }
                });
            });
            match failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                Some(e) => Err(e),
                None => Ok(()),
            }
        };
        // The plan of block `b`, if it has one, the plans before `*cursor`
        // passed: blocks replay in order, so a cursor finds every plan in
        // one walk (most blocks of a circuit are singletons without one).
        let plan_at = |cursor: &mut usize, b: usize| -> Option<usize> {
            while plan.blocks.get(*cursor).is_some_and(|p| p.b < b) {
                *cursor += 1;
            }
            plan.blocks
                .get(*cursor)
                .is_some_and(|p| p.b == b)
                .then_some(*cursor)
        };
        // One block, as decided (`at` its plan); a sequential replay of a
        // planned block is measured for the rate where `timed`, and a
        // planned block replayed `alone` (no other block beside it) for the
        // way it replayed where its pick asks for it.
        let (modes_ref, picks_ref) = (&modes, &picks);
        let replay_one = |job: RJob<'_, T>,
                          x: &mut Vec<T>,
                          timed: bool,
                          at: Option<usize>|
         -> Result<(), RslabError> {
            let b = job.b;
            let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
            let bases = (l_colptr[bs], u_colptr[bs], f_colptr[bs]);
            let ptrs = BlockPtrs {
                l_v: PanelPtr(job.l_v.as_mut_ptr()),
                u_v: PanelPtr(job.u_v.as_mut_ptr()),
                ud: PanelPtr(job.ud.as_mut_ptr()),
                f_v: PanelPtr(job.f_v.as_mut_ptr()),
            };
            let mode = at.map(|i| (&plan.blocks[i], modes_ref[i]));
            if let Some((_, Mode::Pipelined(k))) = mode {
                return replay_pipelined((bs, be), bases, ptrs, k);
            }
            x.clear();
            x.resize(be - bs, T::zero());
            if let Some((bp, Mode::Levels)) = mode {
                return replay_levels(bp, bs, bases, ptrs, x);
            }
            let mut run = || -> Result<(), RslabError> {
                for j in bs..be {
                    // SAFETY: this closure exclusively owns the whole block.
                    replay_col(j, bs, bases, ptrs, x, None)?;
                }
                Ok(())
            };
            match at {
                Some(i) if timed => rate.time(plan.blocks[i].work as f64, run),
                _ => run(),
            }
        };
        let replay_block = |job: RJob<'_, T>,
                            x: &mut Vec<T>,
                            timed: bool,
                            alone: bool,
                            at: Option<usize>|
         -> Result<(), RslabError> {
            let pick = at
                .and_then(|i| picks_ref[i].map(|p| (i, p)))
                .filter(|(_, p)| alone && p.measured());
            let Some((i, pick)) = pick else {
                return replay_one(job, x, timed, at);
            };
            let start = crate::clock::Instant::now();
            replay_one(job, x, timed, at)?;
            let ops = plan.blocks[i].work as f64;
            choices.record(pick, start.elapsed().as_nanos() as f64, ops);
            Ok(())
        };
        if width_blocks > 1 {
            // Runs of consecutive blocks of about equal columns, a few per
            // thread, each replayed in order over one work vector.
            let target = f.n.div_ceil(width_blocks * PIECES_PER_WORKER).max(1);
            let mut pieces: Vec<std::sync::Mutex<Vec<RJob<'_, T>>>> = Vec::new();
            let mut run: Vec<RJob<'_, T>> = Vec::new();
            let mut run_start = 0;
            for b in 0..nblocks {
                interrupt_check(int_flag)?;
                if !run.is_empty() && block_ptr[b + 1] - run_start > target {
                    pieces.push(std::sync::Mutex::new(std::mem::take(&mut run)));
                }
                if run.is_empty() {
                    run_start = block_ptr[b];
                }
                run.push(next_job(b));
            }
            pieces.push(std::sync::Mutex::new(run));
            let failed: std::sync::Mutex<Option<(usize, RslabError)>> = std::sync::Mutex::new(None);
            let section = || {
                crate::parallel::run_width(pieces.len(), width_blocks, &|q| {
                    let jobs =
                        std::mem::take(&mut *pieces[q].lock().unwrap_or_else(|e| e.into_inner()));
                    let mut x = Vec::new();
                    let mut cursor = 0;
                    for job in jobs {
                        let b = job.b;
                        let at = plan_at(&mut cursor, b);
                        let r = interrupt_check(int_flag)
                            .and_then(|()| replay_block(job, &mut x, false, false, at));
                        if let Err(e) = r {
                            let mut slot = failed.lock().unwrap_or_else(|e| e.into_inner());
                            if slot.as_ref().is_none_or(|(b0, _)| b < *b0) {
                                *slot = Some((b, e));
                            }
                            return;
                        }
                    }
                })
            };
            work.watch::<Rayon, _>(pieces.len(), width_blocks, section);
            if let Some((_, e)) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                return Err(e);
            }
        } else {
            // One block after the other: measured as a whole for the rate and
            // the blocks' width (where no block went wide), each block for
            // the way it replayed.
            let all = || -> Result<(), RslabError> {
                let mut cursor = 0;
                for b in 0..nblocks {
                    interrupt_check(int_flag)?;
                    let at = plan_at(&mut cursor, b);
                    replay_block(next_job(b), &mut x, wide, true, at)?;
                }
                Ok(())
            };
            work.serial(all)?;
        }
        f.rs_next = std::mem::replace(&mut f.rs_inv, rs_inv);
        f.replay_x = x;
        f.replay_xs = (xs.into_iter())
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        f.ready = ready;
        f.modes = modes;
        f.picks = picks;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<Ki>()) as u64;
        let nnz = self.diagnostics.factor_nnz;
        self.diagnostics.set_latest(
            "klu-refactor",
            crate::clock::ms(t.elapsed()),
            diagnostics_flops(&self.diagnostics),
            nnz * entry,
        );
        Ok(())
    }
}
