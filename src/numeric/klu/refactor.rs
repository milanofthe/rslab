//! The numeric-only refactorization on a frozen pattern and pivot order.

use super::*;
use crate::constants::{KLU_HANDOFF_NS, KLU_REPLAY_NS_PER_WORK, PIECES_PER_WORKER, PIPE_RUN};
use crate::parallel::{Dispatch, Rate, ScalarRates};
use rspool::constants::{GAIN, TUNE_EXPLORE};

/// What a unit of replay work costs a thread, learned from the sequential
/// replays only (see [`Rate`]): its first refactors replay sequentially to
/// measure it, one that would replay side by side does so sequentially now
/// and then, and a parallel replay far faster than predicted has it measured
/// anew. A rate learned from parallel replays would carry their hand-offs
/// and keep its refactors parallel or sequential whatever the work.
static REPLAY_RATE: ScalarRates = ScalarRates::new(KLU_REPLAY_NS_PER_WORK);

/// What a pipelined replay spends per hand-off beyond the replay itself
/// ([`BlockPlan::handoffs`]), learned from the pipelined replays.
static HANDOFF_RATE: Rate = Rate::new(KLU_HANDOFF_NS);

/// What a level replayed side by side costs beyond the pool's section cost,
/// learned from the level-by-level replays: sections far apart find a
/// pool's workers asleep, which the measured cost of back-to-back sections
/// does not hold.
static LEVEL_RATE: Rate = Rate::new(0.0);

/// Counts the refactors that weigh the surcharges ([`HANDOFF_RATE`],
/// [`LEVEL_RATE`]): one in [`TUNE_EXPLORE`] decides without them, so a
/// surcharge that once kept every block sequential is measured again (both
/// are what the parallel replays cost beyond the replay, learned from them
/// alone).
static SURCHARGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

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
/// runs on, the rates the replays have measured, and how much of the serial
/// time a parallel replay must save.
struct Decide {
    dispatch: Dispatch,
    threads: usize,
    /// The share of the serial time a parallel replay must stay under:
    /// `1 - GAIN`, or `1` where the policy or the user's floors want it
    /// parallel and the time only picks the way.
    bar: f64,
    /// ns per unit of replay work, per hand-off of a pipelined replay, and
    /// per level replayed side by side beyond its section cost.
    seq: f64,
    hand: f64,
    level: f64,
}

impl Decide {
    fn new<T: Scalar>(plan: &ReplayPlan) -> Self {
        let threads = crate::parallel::threads();
        // `On`, or floors the user set: every section is free, so the time
        // picks only how the replay runs side by side, not whether.
        let forced = plan.policy == KluParallel::On || plan.floors();
        let rate = REPLAY_RATE.of::<T>();
        let total = plan.total as f64;
        // Now and then the surcharges are left out, to be measured again.
        let retry = !forced
            && total > 0.0
            && SURCHARGED
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .is_multiple_of(TUNE_EXPLORE);
        Decide {
            dispatch: if forced {
                crate::parallel::kept(true, threads)
            } else {
                crate::parallel::dispatch()
            },
            threads,
            bar: if forced { 1.0 } else { 1.0 - GAIN },
            seq: rate.ns_per_op(),
            hand: if retry { 0.0 } else { HANDOFF_RATE.ns_per_op() },
            level: if forced || retry {
                0.0
            } else {
                LEVEL_RATE.ns_per_op()
            },
        }
    }

    fn width(&self, work: f64, pieces: usize) -> usize {
        self.dispatch.width(work, pieces).min(self.threads).max(1)
    }

    /// The threads level `l` of `bp` replays on, and its time on them
    /// (without the learned surcharge of a wide level, with it).
    fn level(&self, bp: &BlockPlan, l: usize) -> (usize, f64, f64) {
        let work = bp.level_work[l] as f64 * self.seq;
        let p = (bp.ptr[l + 1] - bp.ptr[l]) as usize;
        let k = (self.dispatch)
            .width_after(work, p, self.level)
            .min(self.threads)
            .max(1);
        if k < 2 {
            return (1, work, work);
        }
        let t = work * p.div_ceil(k) as f64 / p as f64 + self.dispatch.cost(k);
        (k, t, t + self.level)
    }

    /// How block `bp` of `bn` columns replays: the shortest of the
    /// sequential replay, level by level (each level on as many threads as
    /// it is worth) and pipelined (its replay plus its hand-offs, the waits
    /// on the columns it reads included), a parallel one only where it saves
    /// its share. With it the time predicted for it, its surcharges left
    /// out.
    fn mode(&self, plan: &ReplayPlan, bp: &BlockPlan, bn: usize) -> (Mode, f64) {
        let serial = bp.work as f64 * self.seq;
        if plan.policy != KluParallel::On && plan.floors() && !plan.block_pays(bp, bn) {
            return (Mode::Serial, serial);
        }
        let breakeven = self.dispatch.break_even();
        if serial.is_nan() || serial < breakeven {
            return (Mode::Serial, serial);
        }
        let (levels, levels_bare) = if bp.widest == 0 || (bp.widest as f64 * self.seq) < breakeven {
            (f64::INFINITY, f64::INFINITY)
        } else {
            (0..bp.level_work.len())
                .map(|l| self.level(bp, l))
                .fold((0.0, 0.0), |(a, b), (_, t, ts)| (a + ts, b + t))
        };
        let piped = {
            let work = serial + bp.handoffs as f64 * self.hand;
            // Its columns as the pieces: how much the dependencies stall
            // the pipeline is what the hand-off rate learns.
            let pieces = bn;
            match self.width(work, pieces) {
                k if k >= 2 => {
                    let share = pieces.div_ceil(k) as f64 / pieces as f64;
                    let cost = self.dispatch.cost(k);
                    (work * share + cost, k, serial * share + cost)
                }
                _ => (f64::INFINITY, 1, f64::INFINITY),
            }
        };
        let bar = self.bar * serial;
        if piped.0 < levels && piped.0 <= bar {
            (Mode::Pipelined(piped.1), piped.2)
        } else if levels <= bar {
            (Mode::Levels, levels_bare)
        } else {
            (Mode::Serial, serial)
        }
    }

    /// The threads the blocks replay side by side on: as many as their work
    /// is worth, in as many equal pieces as the largest block leaves room
    /// for; with it the time predicted for them.
    fn blocks(&self, plan: &ReplayPlan, nblocks: usize) -> (usize, f64) {
        let serial = plan.total as f64 * self.seq;
        if nblocks < 2 || plan.policy == KluParallel::Off {
            return (1, serial);
        }
        if plan.policy != KluParallel::On && plan.floors() && !plan.blocks_pay() {
            return (1, serial);
        }
        let pieces = ((plan.total / plan.largest.max(1)) as usize).min(nblocks);
        match self.width(serial, pieces) {
            k if k >= 2 => (
                k,
                serial * pieces.div_ceil(k) as f64 / pieces as f64 + self.dispatch.cost(k),
            ),
            _ => (1, serial),
        }
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
        let mut modes = std::mem::take(&mut f.modes);
        modes.clear();
        let serial = plan.total as f64 * decide.seq;
        let cheap = serial.is_nan() || serial < decide.dispatch.break_even();
        modes.extend(plan.blocks.iter().map(|bp| {
            if cheap {
                Mode::Serial
            } else {
                let bn = f.block_ptr[bp.b + 1] - f.block_ptr[bp.b];
                let (m, _) = decide.mode(plan, bp, bn);
                m
            }
        }));
        #[cfg(test)]
        if let Some(m) = plan.forced {
            modes.iter_mut().for_each(|x| *x = m);
        }
        let (mut width_blocks, _) = if cheap {
            (1, serial)
        } else {
            decide.blocks(plan, nblocks)
        };
        let mut wide = width_blocks > 1 || modes.iter().any(|m| *m != Mode::Serial);
        // A refactor decided side by side replays sequentially instead,
        // measured, now and then: the rate stays what a sequential replay
        // costs now (see `REPLAY_RATE`). `modes` keeps what was decided.
        let widest = modes.iter().fold(width_blocks, |k, m| match m {
            Mode::Pipelined(p) => k.max(*p),
            Mode::Levels => k.max(decide.threads),
            Mode::Serial => k,
        });
        let rate = REPLAY_RATE.of::<T>();
        #[cfg(test)]
        let pinned = plan.forced.is_some();
        #[cfg(not(test))]
        let pinned = false;
        let free = pinned || plan.policy == KluParallel::On || plan.floors();
        // now and then a refactor decided side by side replays
        // sequentially, measured: the replay rate stays the code's
        let explore = wide
            && !free
            && rspool::draw(TUNE_EXPLORE.saturating_mul(widest.saturating_sub(1).max(1) as u32));
        let sequential = explore;
        if sequential {
            width_blocks = 1;
            wide = false;
        }
        // The pipelined replays: this refactor's number on the columns; and
        // for every parallel replay a zero work vector per worker, kept
        // across refactors.
        let mut ready = std::mem::take(&mut f.ready);
        let mut kept_xs = std::mem::take(&mut f.replay_xs);
        if !sequential && modes.iter().any(|m| matches!(m, Mode::Pipelined(_))) {
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
            // The wide levels' time against what they were costed at.
            let (mut wide, mut costed, mut took) = (0usize, 0.0f64, 0.0f64);
            for l in 0..bp.level_work.len() {
                interrupt_check(int_flag)?;
                let cols = &bp.cols[bp.ptr[l] as usize..bp.ptr[l + 1] as usize];
                let (k, t, _) = decide_ref.level(bp, l);
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
                // Not learned from for the replay rate (see `REPLAY_RATE`):
                // its time beyond the costed is the level surcharge.
                let start = crate::clock::Instant::now();
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
                (wide, costed) = (wide + 1, costed + t);
                took += start.elapsed().as_nanos() as f64;
                if let Some(e) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                    return Err(e);
                }
            }
            if wide > 0 {
                LEVEL_RATE.learn((took - costed).max(1.0), wide as f64);
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
        let replay_pipelined = |bp: &BlockPlan,
                                (bs, be): (usize, usize),
                                bases: (usize, usize, usize),
                                p: BlockPtrs<T>,
                                k: usize|
         -> Result<(), RslabError> {
            use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AOrd};
            let (bn, epoch, done) = (be - bs, ready_ref.epoch, &ready_ref.done);
            let next = AtomicUsize::new(bs);
            let stop = AtomicBool::new(false);
            let failed: std::sync::Mutex<Option<RslabError>> = std::sync::Mutex::new(None);
            let pieces = k.min(bn.div_ceil(PIPE_RUN)).max(1);
            // Measured every time (a few clock reads against a parallel
            // replay): the threads' time beyond the replay is what the
            // hand-offs cost.
            let busy = AtomicU64::new(0);
            crate::parallel::run_width(pieces, pieces, &|q| {
                let start = crate::clock::Instant::now();
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
                busy.fetch_add(start.elapsed().as_nanos() as u64, AOrd::Relaxed);
            });
            let beyond = busy.into_inner() as f64 - bp.work as f64 * decide_ref.seq;
            HANDOFF_RATE.learn(beyond.max(1.0), bp.handoffs as f64);
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
        // planned block is measured for the rate where `timed`.
        let modes_ref = &modes;
        let replay_block = |job: RJob<'_, T>,
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
            let mode = at
                .filter(|_| !sequential)
                .map(|i| (&plan.blocks[i], modes_ref[i]));
            if let Some((bp, Mode::Pipelined(k))) = mode {
                return replay_pipelined(bp, (bs, be), bases, ptrs, k);
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
            crate::parallel::run_width(pieces.len(), width_blocks, &|q| {
                let jobs =
                    std::mem::take(&mut *pieces[q].lock().unwrap_or_else(|e| e.into_inner()));
                let mut x = Vec::new();
                let mut cursor = 0;
                for job in jobs {
                    let b = job.b;
                    let at = plan_at(&mut cursor, b);
                    let r = interrupt_check(int_flag)
                        .and_then(|()| replay_block(job, &mut x, false, at));
                    if let Err(e) = r {
                        let mut slot = failed.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.as_ref().is_none_or(|(b0, _)| b < *b0) {
                            *slot = Some((b, e));
                        }
                        return;
                    }
                }
            });
            if let Some((_, e)) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                return Err(e);
            }
        } else {
            // All sequential: measured as a whole, now and then.
            let mut all = || -> Result<(), RslabError> {
                let mut cursor = 0;
                for b in 0..nblocks {
                    interrupt_check(int_flag)?;
                    let at = plan_at(&mut cursor, b);
                    replay_block(next_job(b), &mut x, wide, at)?;
                }
                Ok(())
            };
            let total = plan.total as f64;
            if wide || plan.total == 0 {
                all()?;
            } else if explore {
                rate.measure(total, all)?;
            } else {
                rate.time(total, all)?;
            }
        }
        f.rs_next = std::mem::replace(&mut f.rs_inv, rs_inv);
        f.replay_x = x;
        f.replay_xs = (xs.into_iter())
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        f.ready = ready;
        f.modes = modes;
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
