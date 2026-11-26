//! The numeric-only refactorization on a frozen pattern and pivot order.

use super::*;
use crate::constants::{
    KLU_PAR_MIN_FMADDS, KLU_PAR_MIN_RATIO, KLU_REPLAY_NS_PER_WORK, PIPE_COL_WORK, PIPE_RUN,
};
use crate::parallel::{Dispatch, Go, Rayon, ScalarRates, Work};

/// What a unit of replay work costs a thread: what the blocks' replays and
/// how wide they go side by side are decided on ([`Work`]).
static REPLAY_RATE: ScalarRates = ScalarRates::new(KLU_REPLAY_NS_PER_WORK);

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

    /// How block `bp` of `bn` columns starts (`None` sequentially): side by side
    /// where its work clears the floor and its mean level width the ratio (the
    /// user's, else [`KLU_PAR_MIN_FMADDS`] and [`KLU_PAR_MIN_RATIO`]) or the
    /// policy is `On`, then pipelined (`Some(true)`) where its columns carry
    /// [`PIPE_COL_WORK`] on average and level by level where they are lighter,
    /// under `Auto` that only on a lent pool, whose workers wait awake between
    /// the levels (a rayon pool's sleep; sequentially there).
    fn start(&self, bp: &BlockPlan, bn: usize) -> Option<bool> {
        let nlev = bp.level_work.len().max(1);
        let pays = bp.work >= self.min_work.unwrap_or(KLU_PAR_MIN_FMADDS)
            && bn as f64 / nlev as f64 >= self.min_ratio.unwrap_or(KLU_PAR_MIN_RATIO);
        let on = self.policy == KluParallel::On;
        (on || pays)
            .then(|| bp.work / bn.max(1) as u64 >= PIPE_COL_WORK)
            .filter(|&pipelined| pipelined || on || crate::parallel::is_lent())
    }

    /// Whether anything of a refactor starts side by side: a block
    /// ([`start`](Self::start)), or the blocks together, where their work
    /// clears the floor and no block holds more than its share of the ratio
    /// (the user's, else the defaults), or the policy is `On`. Where nothing
    /// does, a refactor replays sequentially without looking at the pool.
    fn starts_wide(&self) -> bool {
        let blocks = self.total >= self.min_work.unwrap_or(KLU_PAR_MIN_FMADDS)
            && self.total as f64
                >= self.min_ratio.unwrap_or(KLU_PAR_MIN_RATIO) * self.largest as f64;
        self.policy == KluParallel::On
            || blocks
            || (self.blocks.iter()).any(|bp| self.start(bp, bp.cols.len()).is_some())
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
        let forced = plan.policy == KluParallel::On || plan.floors();
        let threads = if forced || plan.starts_wide() {
            crate::parallel::threads()
        } else {
            1
        };
        Decide {
            dispatch: if threads < 2 {
                Dispatch::never()
            } else if forced {
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

    /// Columns per piece of level `l` of `bp` on `k` threads: its work cut
    /// into ranges as the section cost makes them worth
    /// ([`Dispatch::ranges`]).
    fn level_per(&self, bp: &BlockPlan, l: usize, k: usize) -> usize {
        let p = (bp.ptr[l + 1] - bp.ptr[l]) as usize;
        let work = bp.level_work[l] as f64 * self.seq;
        p.div_ceil(self.dispatch.ranges(work, p, k).max(1))
    }

    /// How block `bp` of `bn` columns, `serial` ns sequentially, replays,
    /// where its structure starts it (`start`, see [`ReplayPlan::start`]):
    /// the way predicted fastest among those it starts with, or among all
    /// offered where it starts with none of them. The ways: sequentially
    /// (unless every section is free), level by level (where a level is
    /// worth a section), pipelined on 2, 4, ... threads, the last one on all
    /// the pool's, at its share of the work and the section. No chain of the
    /// elimination DAG bounds a pipeline from below: a column eliminates
    /// against the columns it reads as they finish, so the chain of whole
    /// columns overstates its critical path (ASIC_100ks: a chain ratio below
    /// two, pipelined over three times faster).
    fn fastest(&self, bp: &BlockPlan, bn: usize, serial: f64, start: Option<bool>) -> Mode {
        let started = |m: Mode| {
            matches!(
                (start, m),
                (_, Mode::Serial) | (Some(false), Mode::Levels) | (Some(true), Mode::Pipelined(_))
            )
        };
        // the fastest way it starts with, and the fastest of all
        let mut first: Option<(Mode, f64)> = None;
        let mut any = first;
        let mut offer = |m: Mode, t: f64| {
            let faster = |b: Option<(Mode, f64)>| b.is_none_or(|(_, bt)| t < bt);
            if started(m) && faster(first) {
                first = Some((m, t));
            }
            if faster(any) {
                any = Some((m, t));
            }
        };
        if !self.forced {
            offer(Mode::Serial, serial);
        }
        let cap = self.threads.min(bn);
        if (bp.widest as f64 * self.seq) >= self.dispatch.break_even() {
            offer(
                Mode::Levels,
                (0..bp.level_work.len()).map(|l| self.level(bp, l).1).sum(),
            );
        }
        let pipelined =
            |k: usize| serial * bn.div_ceil(k) as f64 / bn as f64 + self.dispatch.cost(k);
        let mut k = 2;
        while k < cap {
            offer(Mode::Pipelined(k), pipelined(k));
            k *= 2;
        }
        if cap >= 2 {
            offer(Mode::Pipelined(cap), pipelined(cap));
        }
        first.or(any).map_or(Mode::Serial, |b| b.0)
    }

    /// How block `bp` of `bn` columns replays: sequentially below the
    /// user's floors and below what any section is worth on the pool; else
    /// the way predicted fastest ([`fastest`](Self::fastest)).
    pub(super) fn mode(&self, plan: &ReplayPlan, bp: &BlockPlan, bn: usize) -> Mode {
        let serial = bp.work as f64 * self.seq;
        if plan.policy != KluParallel::On && plan.floors() && !plan.block_pays(bp, bn) {
            return Mode::Serial;
        }
        if serial.is_nan() || serial < self.dispatch.break_even() {
            return Mode::Serial;
        }
        self.fastest(bp, bn, serial, plan.start(bp, bn))
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
    /// Every refactor decides how its blocks replay on the workers of the
    /// factorization's [`KluSettings::threads`] ([`KluSettings::parallel`]):
    /// sequentially, level by level, pipelined, or blocks side by side, as
    /// the replay work of the pivot-final pattern is worth there. A refactor
    /// none of whose replays can go side by side enters no pool.
    pub fn refactor(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
        a.validate()?;
        let _s = rsprobe::scope!("refactor");
        let threads = self.factors.threads;
        if self.factors.plan.starts_wide() {
            threads.run(0, |cap| cap, || self.replay(a))
        } else {
            threads.here(|| self.replay(a))
        }
    }

    /// [`refactor`](Self::refactor) on the workers it runs on.
    // The replay loops index several parallel arrays at offset positions;
    // iterator forms would obscure the offset arithmetic.
    #[allow(clippy::needless_range_loop)]
    fn replay(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
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
        {
            let _s = rsprobe::scope!(DEBUG, "row scaling");
            row_scale_inv_into(a, f.scaled, &mut rs_inv);
        }

        // Branch-free pattern verification against the recorded program: every
        // entry must map to the exact final position it had at factor time.
        {
            let _s = rsprobe::scope!(DEBUG, "pattern check");
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
        // refactor runs on.
        let plan = &f.plan;
        let decide = Decide::new::<T>(plan);
        let rate = REPLAY_RATE.of::<T>();
        let serial = plan.total as f64 * decide.seq;
        let cheap = serial.is_nan() || serial < decide.dispatch.break_even();
        let mut modes = std::mem::take(&mut f.modes);
        modes.clear();
        for bp in &plan.blocks {
            modes.push(if cheap {
                Mode::Serial
            } else {
                let bn = f.block_ptr[bp.b + 1] - f.block_ptr[bp.b];
                decide.mode(plan, bp, bn)
            });
        }
        #[cfg(test)]
        if let Some(m) = plan.forced {
            modes.iter_mut().for_each(|x| *x = m);
        }
        // The blocks side by side, on the width their work is worth (as
        // wide as they go where every section is free).
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
        let _replay = rsprobe::scope!(DEBUG, "replay");
        rsprobe::add!(DEBUG, "work", plan.total);
        rsprobe::add!(
            DEBUG,
            "factor_nnz",
            f.l_val.len() + f.u_val.len() + f.udiag.len()
        );
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
        let (sn_last, u_dense, l_widest) = (&f.sn_last, &f.u_dense, f.l_widest);
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

        // In a pipelined replay: wait until the `L` columns `lo ..= hi` carry
        // the refactor's number.
        type Wait<'w> = (&'w [std::sync::atomic::AtomicU32], u32);
        let await_cols = |(done, epoch): Wait<'_>, lo: usize, hi: usize| {
            for c in lo..=hi {
                // Acquire pairs with the Release that published it.
                let mut spins = 0u32;
                while done[c].load(std::sync::atomic::Ordering::Acquire) != epoch {
                    spins += 1;
                    if spins & crate::constants::SPIN_YIELD_MASK == 0 {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }
        };
        // Replay one column through the recorded program: scatter, eliminate
        // against the `U` rows in their stored order, the columns of a supernode densely
        // ([`super::supernode`]), pivot, emit L.
        // SAFETY: caller guarantees exclusive ownership of column `j`'s output
        // slots and completed dependency columns (see `BlockPtrs`); `x` is this
        // caller's scratch, the block's work vector and the supernode
        // kernel's gather (`l_widest` behind it), all-zero on entry and left
        // all-zero.
        // With `wait`, the column runs in a pipelined replay: it consumes `L`
        // column `pr` only once `wait.0[pr]` carries the refactor's number
        // `wait.1`, so the elimination against the columns done long ago
        // overlaps the ones still being replayed.
        let replay_col = |j: usize,
                          bs: usize,
                          bases: (usize, usize, usize),
                          p: BlockPtrs<T>,
                          x: &mut [T],
                          wait: Option<Wait<'_>>|
         -> Result<(), RslabError> {
            let (l_base, u_base, f_base) = bases;
            let (x, tmp) = x.split_at_mut(x.len() - l_widest);
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
                if u_dense[j] {
                    // out of line, so this loop stays as small as it was
                    super::supernode::eliminate(
                        (&mut *x, &mut *tmp),
                        (j, bs),
                        (&u_rowidx[..], u_colptr[j]..u_colptr[j + 1], sn_last),
                        (l_colptr, l_rowidx),
                        (p.l_v.get(), l_base),
                        (p.u_v.get(), u_base),
                        |lo: usize, hi: usize| {
                            if let Some(wait) = wait {
                                await_cols(wait, lo, hi);
                            }
                        },
                    );
                } else {
                    for k in u_colptr[j]..u_colptr[j + 1] {
                        let pr = u_rowidx[k] as usize;
                        if let Some(wait) = wait {
                            await_cols(wait, pr, pr);
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
                let per = decide_ref.level_per(bp, l, k);
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
            let (bn, epoch, done) = (be - bs + l_widest, ready_ref.epoch, &ready_ref.done);
            let next = AtomicUsize::new(bs);
            let stop = AtomicBool::new(false);
            let failed: std::sync::Mutex<Option<RslabError>> = std::sync::Mutex::new(None);
            let pieces = k.min((be - bs).div_ceil(PIPE_RUN)).max(1);
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
        // One block, as decided (`at` its plan).
        let modes_ref = &modes;
        let replay_block =
            |job: RJob<'_, T>, x: &mut Vec<T>, at: Option<usize>| -> Result<(), RslabError> {
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
                if let Some((bp, Mode::Pipelined(k))) = mode {
                    let _s = rsprobe::scope!(DEBUG, "pipelined");
                    rsprobe::add!(DEBUG, "work", bp.work);
                    return replay_pipelined((bs, be), bases, ptrs, k);
                }
                // all-zero, kept across blocks and refactors (cleared on an error)
                let need = be - bs + l_widest;
                if x.len() < need {
                    x.resize(need, T::zero());
                }
                let x = &mut x[..need];
                if let Some((bp, Mode::Levels)) = mode {
                    let _s = rsprobe::scope!(DEBUG, "levels");
                    rsprobe::add!(DEBUG, "work", bp.work);
                    return replay_levels(bp, bs, bases, ptrs, x);
                }
                let _s = at.map(|_| rsprobe::scope!(DEBUG, "serial block"));
                if let Some(i) = at {
                    rsprobe::add!(DEBUG, "work", plan.blocks[i].work);
                }
                for j in bs..be {
                    // SAFETY: this closure exclusively owns the whole block.
                    if let Err(e) = replay_col(j, bs, bases, ptrs, x, None) {
                        x.fill(T::zero());
                        return Err(e);
                    }
                }
                Ok(())
            };
        if width_blocks > 1 {
            // Runs of consecutive blocks of about equal columns, as many as
            // the replay's work is worth ranges ([`Dispatch::ranges`]), each
            // replayed in order over one work vector.
            let runs = decide.dispatch.ranges(serial, f.n, width_blocks).max(1);
            let target = f.n.div_ceil(runs).max(1);
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
                let mut jobs = Some(std::mem::take(
                    &mut *pieces[q].lock().unwrap_or_else(|e| e.into_inner()),
                ));
                // a kept zero work vector of this worker's, not one per run
                with_work(q, &mut |x: &mut Vec<T>| {
                    let mut cursor = 0;
                    for job in jobs.take().into_iter().flatten() {
                        let b = job.b;
                        let at = plan_at(&mut cursor, b);
                        let r = interrupt_check(int_flag).and_then(|()| replay_block(job, x, at));
                        if let Err(e) = r {
                            let mut slot = failed.lock().unwrap_or_else(|e| e.into_inner());
                            if slot.as_ref().is_none_or(|(b0, _)| b < *b0) {
                                *slot = Some((b, e));
                            }
                            return;
                        }
                    }
                });
            });
            if let Some((_, e)) = failed.into_inner().unwrap_or_else(|e| e.into_inner()) {
                return Err(e);
            }
        } else {
            // One block after the other.
            let mut cursor = 0;
            for b in 0..nblocks {
                interrupt_check(int_flag)?;
                let at = plan_at(&mut cursor, b);
                replay_block(next_job(b), &mut x, at)?;
            }
        }
        f.rs_next = std::mem::replace(&mut f.rs_inv, rs_inv);
        f.replay_x = x;
        f.replay_xs = (xs.into_iter())
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        f.ready = ready;
        f.modes = modes;
        Ok(())
    }
}
