//! The numeric-only refactorization on a frozen pattern and pivot order.

use super::*;
use crate::constants::{
    KLU_PAR_MIN_FMADDS, KLU_PAR_MIN_RATIO, KLU_REPLAY_NS_PER_WORK, PIPE_COL_WORK, PIPE_RUN,
};
use crate::parallel::{Dispatch, Go, Rayon, ScalarRates, Work};

static REPLAY_RATE: ScalarRates = ScalarRates::new(KLU_REPLAY_NS_PER_WORK);

impl ReplayPlan {
    fn floors(&self) -> bool {
        self.min_work.is_some() || self.min_ratio.is_some()
    }

    fn block_pays(&self, bp: &BlockPlan, bn: usize) -> bool {
        let nlev = bp.level_work.len().max(1);
        self.min_work.is_none_or(|w| bp.work >= w)
            && self.min_ratio.is_none_or(|r| bn as f64 / nlev as f64 >= r)
    }

    // https://github.com/milanofthe/rslab/issues/290
    fn start(&self, bp: &BlockPlan, bn: usize) -> Option<bool> {
        let nlev = bp.level_work.len().max(1);
        let pays = bp.work >= self.min_work.unwrap_or(KLU_PAR_MIN_FMADDS)
            && bn as f64 / nlev as f64 >= self.min_ratio.unwrap_or(KLU_PAR_MIN_RATIO);
        let on = self.policy == KluParallel::On;
        (on || pays)
            .then(|| bp.work / bn.max(1) as u64 >= PIPE_COL_WORK)
            .filter(|&pipelined| pipelined || on || crate::parallel::is_lent())
    }

    fn starts_wide(&self) -> bool {
        let blocks = self.total >= self.min_work.unwrap_or(KLU_PAR_MIN_FMADDS)
            && self.total as f64
                >= self.min_ratio.unwrap_or(KLU_PAR_MIN_RATIO) * self.largest as f64;
        self.policy == KluParallel::On
            || blocks
            || (self.blocks.iter()).any(|bp| self.start(bp, bp.level_cols.len()).is_some())
    }

    fn blocks_pay(&self) -> bool {
        self.min_work.is_none_or(|w| self.total >= w)
            && self
                .min_ratio
                .is_none_or(|r| self.total as f64 >= r * self.largest as f64)
    }
}

pub(super) struct Decide {
    dispatch: Dispatch,
    cut: Dispatch,
    threads: usize,
    forced: bool,
    ns_per_work: f64,
}

impl Decide {
    fn new<T: Scalar>(plan: &ReplayPlan) -> Self {
        let forced = plan.policy == KluParallel::On || plan.floors();
        let threads = if forced || plan.starts_wide() {
            crate::parallel::threads()
        } else {
            1
        };
        let cut = if threads < 2 {
            Dispatch::never()
        } else {
            crate::parallel::dispatch()
        };
        Decide {
            dispatch: if forced && threads >= 2 {
                crate::parallel::kept(true, threads)
            } else {
                cut.clone()
            },
            cut,
            threads,
            forced,
            ns_per_work: REPLAY_RATE.of::<T>().ns_per_op(),
        }
    }

    #[cfg(test)]
    pub(super) fn on(dispatch: Dispatch, threads: usize) -> Self {
        Decide {
            cut: dispatch.clone(),
            dispatch,
            threads,
            forced: false,
            ns_per_work: 1.0,
        }
    }

    fn width(&self, work: f64, pieces: usize) -> usize {
        self.dispatch.width(work, pieces).min(self.threads).max(1)
    }

    fn level(&self, bp: &BlockPlan, l: usize) -> (usize, f64) {
        let work = bp.level_work[l] as f64 * self.ns_per_work;
        let p = (bp.level_ptr[l + 1] - bp.level_ptr[l]) as usize;
        match self.width(work, p) {
            k if k >= 2 => (
                k,
                work * p.div_ceil(k) as f64 / p as f64 + self.dispatch.cost(k),
            ),
            _ => (1, work),
        }
    }

    fn level_per(&self, bp: &BlockPlan, l: usize, k: usize) -> usize {
        let p = (bp.level_ptr[l + 1] - bp.level_ptr[l]) as usize;
        let work = bp.level_work[l] as f64 * self.ns_per_work;
        p.div_ceil(self.cut.ranges(work, p, k).max(1))
    }

    // https://github.com/milanofthe/rslab/issues/290
    fn fastest(&self, bp: &BlockPlan, bn: usize, serial: f64, start: Option<bool>) -> Mode {
        let started = |m: Mode| {
            matches!(
                (start, m),
                (_, Mode::Serial) | (Some(false), Mode::Levels) | (Some(true), Mode::Pipelined(_))
            )
        };
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
        if (bp.widest as f64 * self.ns_per_work) >= self.dispatch.break_even() {
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

    pub(super) fn mode(&self, plan: &ReplayPlan, bp: &BlockPlan, bn: usize) -> Mode {
        let serial = bp.work as f64 * self.ns_per_work;
        if plan.policy != KluParallel::On && plan.floors() && !plan.block_pays(bp, bn) {
            return Mode::Serial;
        }
        if serial.is_nan() || serial < self.dispatch.break_even() {
            return Mode::Serial;
        }
        self.fastest(bp, bn, serial, plan.start(bp, bn))
    }

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

    #[allow(clippy::needless_range_loop)]
    fn replay(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
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
        let mut rs_inv = std::mem::take(&mut f.rs_next);
        {
            let _s = rsprobe::scope!(DEBUG, "row scaling");
            row_scale_inv_into(a, f.scaled, &mut rs_inv);
        }

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

        struct RJob<'s, T> {
            b: usize,
            l_v: &'s mut [T],
            u_v: &'s mut [T],
            ud: &'s mut [T],
            f_v: &'s mut [T],
        }
        let nblocks = f.block_ptr.len() - 1;
        let mut x = std::mem::take(&mut f.replay_x);
        let plan = &f.plan;
        let decide = Decide::new::<T>(plan);
        let rate = REPLAY_RATE.of::<T>();
        let serial = plan.total as f64 * decide.ns_per_work;
        let cheap = serial.is_nan() || serial < decide.dispatch.break_even();
        let mut modes = std::mem::take(&mut f.modes);
        modes.clear();
        for bp in &plan.blocks {
            modes.push(if cheap {
                Mode::Serial
            } else {
                let bn = f.block_ptr[bp.block + 1] - f.block_ptr[bp.block];
                decide.mode(plan, bp, bn)
            });
        }
        #[cfg(test)]
        if let Some(m) = plan.forced {
            modes.iter_mut().for_each(|x| *x = m);
        }
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

        type Wait<'w> = (&'w [std::sync::atomic::AtomicU32], u32);
        let await_cols = |(done, epoch): Wait<'_>, lo: usize, hi: usize| {
            for c in lo..=hi {
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
            // SAFETY: the caller owns the output slots of column `j`
            // exclusively and every `L` column that `j` reads is complete:
            // a block replayed column by column is owned whole by its
            // closure; the columns of a level write disjoint slots and read
            // `L` columns of earlier levels, complete before the level
            // started; a pipelined column writes only its own slots and,
            // with `wait`, reads `L` column `pr` only once `wait.0[pr]`
            // carries the refactor's number `wait.1`. `x` is this caller's
            // scratch, the block's work vector and the supernode kernel's
            // gather (`l_widest` behind it), all-zero on entry and left
            // all-zero.
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
        let xs_ref = &xs;
        let with_work = |k: usize, run: &mut dyn FnMut(&mut Vec<T>)| {
            let free =
                (xs_ref.iter().cycle().skip(k).take(xs_ref.len())).find_map(|m| m.try_lock().ok());
            match free {
                Some(mut w) => run(&mut w),
                None => run(&mut Vec::new()),
            }
        };
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
                let cols = &bp.level_cols[bp.level_ptr[l] as usize..bp.level_ptr[l + 1] as usize];
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
        // https://doi.org/10.1109/TCAD.2012.2217964
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
        let plan_at = |cursor: &mut usize, b: usize| -> Option<usize> {
            while plan.blocks.get(*cursor).is_some_and(|p| p.block < b) {
                *cursor += 1;
            }
            plan.blocks
                .get(*cursor)
                .is_some_and(|p| p.block == b)
                .then_some(*cursor)
        };
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
                // https://github.com/milanofthe/rslab/issues/290
                if let Some((bp, Mode::Pipelined(k))) = mode {
                    let _s = rsprobe::scope!(DEBUG, "pipelined");
                    rsprobe::add!(DEBUG, "work", bp.work);
                    return replay_pipelined((bs, be), bases, ptrs, k);
                }
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
                    if let Err(e) = replay_col(j, bs, bases, ptrs, x, None) {
                        x.fill(T::zero());
                        return Err(e);
                    }
                }
                Ok(())
            };
        if width_blocks > 1 {
            let runs = decide.cut.ranges(serial, f.n, width_blocks).max(1);
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
