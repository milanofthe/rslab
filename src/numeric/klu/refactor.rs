//! The numeric-only refactorization on a frozen pattern and pivot order.

use super::*;
use crate::constants::{LEVEL_PIECE, PIPE_RUN};

impl<T: Scalar> KluSolver<T> {
    /// Numeric-only refactorization: replay the stored pattern and pivot
    /// sequence on new values with the **same** sparsity pattern. No symbolic
    /// work, no pivot search, the fast path for frequency sweeps and Newton
    /// steps. Fails with a pattern-mismatch error if `a`'s pattern deviates
    /// from the factored one, and with [`RslabError::SingularBasis`] if a
    /// frozen pivot becomes zero (re-`factor` with pivoting in that case).
    /// After an error the factorization is invalid; a subsequent successful
    /// `refactor` or a fresh `factor` makes it valid again.
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
        // in parallel when the factor-time opt-in chose it (bit-identical:
        // per-block work is sequential and shares nothing).
        struct RJob<'s, T> {
            b: usize,
            l_v: &'s mut [T],
            u_v: &'s mut [T],
            ud: &'s mut [T],
            f_v: &'s mut [T],
        }
        let nblocks = f.block_ptr.len() - 1;
        let mut x = std::mem::take(&mut f.replay_x);
        // The pipelined replays: this refactor's number on the columns, and
        // a zero work vector per worker, kept across refactors.
        let mut ready = std::mem::take(&mut f.ready);
        let mut kept_xs = std::mem::take(&mut f.replay_xs);
        if !f.pipelined.is_empty() {
            ready.next(f.n);
            kept_xs.resize_with(crate::parallel::threads(), Vec::new);
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
        // level-parallel replay: every column writes only its own L/U/F/diag
        // slots and reads L columns completed in earlier levels (fenced by the
        // per-level join), so the shared-mutable access is race-free.
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

        // A pipelined block's columns side by side (NICSLU's pipeline mode,
        // Chen, Wang and Yang, TCAD 2013), taken in order, one at a time,
        // from a shared counter: a column waits until every `L` column
        // it reads is replayed, all of them earlier columns, taken before it
        // by pieces that run. So the replay completes on any pool, one that
        // runs its pieces one after the other included. A failing column
        // still releases its dependants, and the lowest failing column is
        // reported, as by the sequential replay; an interrupt stops the
        // pieces between runs.
        let (pipelined, xs_ref, ready_ref) = (&f.pipelined, &xs, &ready);
        // A work vector of this worker's: a free kept one, else a fresh one.
        let with_work = |k: usize, run: &mut dyn FnMut(&mut Vec<T>)| {
            let free =
                (xs_ref.iter().cycle().skip(k).take(xs_ref.len())).find_map(|m| m.try_lock().ok());
            match free {
                Some(mut w) => run(&mut w),
                None => run(&mut Vec::new()),
            }
        };
        // Level by level: a level's columns side by side, at least
        // `LEVEL_PIECE` to a piece, a level of fewer on the calling thread;
        // on an error the level's lowest failing column is reported.
        let replay_levels = |ptr: &[u32],
                             cols: &[Ki],
                             bs: usize,
                             bases: (usize, usize, usize),
                             p: BlockPtrs<T>,
                             x: &mut [T]|
         -> Result<(), RslabError> {
            let (bn, threads) = (x.len(), crate::parallel::threads());
            for l in 0..ptr.len() - 1 {
                interrupt_check(int_flag)?;
                let cols = &cols[ptr[l] as usize..ptr[l + 1] as usize];
                if threads < 2 || cols.len() < 2 * LEVEL_PIECE {
                    for &j in cols {
                        if let Err(e) = replay_col(j as usize, bs, bases, p, x, None) {
                            x.fill(T::zero());
                            return Err(e);
                        }
                    }
                    continue;
                }
                let pieces = threads.min(cols.len() / LEVEL_PIECE);
                let per = cols.len().div_ceil(pieces);
                let failed: std::sync::Mutex<Option<RslabError>> = std::sync::Mutex::new(None);
                crate::parallel::for_each(pieces, 1, |k| {
                    with_work(k, &mut |w: &mut Vec<T>| {
                        if w.len() < bn {
                            w.resize(bn, T::zero());
                        }
                        let n = cols.len();
                        for &j in &cols[(k * per).min(n)..((k + 1) * per).min(n)] {
                            // SAFETY: the columns of a level write disjoint
                            // slots and read `L` columns of earlier levels,
                            // complete before this level started.
                            if let Err(e) = replay_col(j as usize, bs, bases, p, &mut w[..bn], None)
                            {
                                w.fill(T::zero());
                                let mut slot = failed.lock().unwrap_or_else(|e| e.into_inner());
                                let lower = match (&e, &*slot) {
                                    (_, None) => true,
                                    (
                                        RslabError::SingularBasis { column: c1 },
                                        Some(RslabError::SingularBasis { column: c0 }),
                                    ) => c1 < c0,
                                    _ => false,
                                };
                                if lower {
                                    *slot = Some(e);
                                }
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
        let replay_pipelined = |bs: usize,
                                be: usize,
                                bases: (usize, usize, usize),
                                p: BlockPtrs<T>|
         -> Result<(), RslabError> {
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AOrd};
            let (bn, epoch, done) = (be - bs, ready_ref.epoch, &ready_ref.done);
            let next = AtomicUsize::new(bs);
            let stop = AtomicBool::new(false);
            let failed: std::sync::Mutex<Option<RslabError>> = std::sync::Mutex::new(None);
            let fail = |e: RslabError| {
                let mut slot = failed.lock().unwrap_or_else(|e| e.into_inner());
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
            let pieces = crate::parallel::threads().min(bn.div_ceil(PIPE_RUN)).max(1);
            crate::parallel::for_each(pieces, 1, |k| {
                with_work(k, &mut |w: &mut Vec<T>| {
                    if w.len() < bn {
                        w.resize(bn, T::zero());
                    }
                    let w = &mut w[..bn];
                    while !stop.load(AOrd::Relaxed) {
                        if let Err(e) = interrupt_check(int_flag) {
                            stop.store(true, AOrd::Relaxed);
                            fail(e);
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
                                fail(e);
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
        let replay_block = |job: RJob<'_, T>, x: &mut Vec<T>| -> Result<(), RslabError> {
            let b = job.b;
            let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
            let bases = (l_colptr[bs], u_colptr[bs], f_colptr[bs]);
            let ptrs = BlockPtrs {
                l_v: PanelPtr(job.l_v.as_mut_ptr()),
                u_v: PanelPtr(job.u_v.as_mut_ptr()),
                ud: PanelPtr(job.ud.as_mut_ptr()),
                f_v: PanelPtr(job.f_v.as_mut_ptr()),
            };
            if crate::parallel::threads() >= 2 {
                match pipelined.iter().find(|(pb, _)| *pb == b).map(|(_, r)| r) {
                    Some(Replay::Pipelined) => return replay_pipelined(bs, be, bases, ptrs),
                    // A section per level costs a wake on rayon, more than
                    // a light column's replay: level by level only on a
                    // lent pool, whose workers wait awake.
                    Some(Replay::Levels { ptr, cols }) if crate::parallel::lent().is_some() => {
                        x.clear();
                        x.resize(be - bs, T::zero());
                        return replay_levels(ptr, cols, bs, bases, ptrs, x);
                    }
                    _ => {}
                }
            }
            x.clear();
            x.resize(be - bs, T::zero());
            for j in bs..be {
                // SAFETY: this closure exclusively owns the whole block.
                replay_col(j, bs, bases, ptrs, x, None)?;
            }
            Ok(())
        };
        if f.par_refactor && nblocks > 1 {
            let mut jobs = Vec::with_capacity(nblocks);
            for b in 0..nblocks {
                interrupt_check(int_flag)?;
                jobs.push(next_job(b));
            }
            let results: Vec<Result<(), RslabError>> =
                crate::parallel::map_owned(jobs, 1, |job| replay_block(job, &mut Vec::new()));
            for r in results {
                r?;
            }
        } else {
            for b in 0..nblocks {
                interrupt_check(int_flag)?;
                replay_block(next_job(b), &mut x)?;
            }
        }
        f.rs_next = std::mem::replace(&mut f.rs_inv, rs_inv);
        f.replay_x = x;
        f.replay_xs = (xs.into_iter())
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        f.ready = ready;
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
