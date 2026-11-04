//! One supernode of the left-looking LU factorization: assembly, the updates
//! of its factored descendants (`cmod`, into both `L` and `U12`), the
//! delayed pivots its children pass, then the blocked panel LU with
//! threshold partial pivoting, which may delay pivots in turn (`delay`).

use super::delay::NodePivots;
use super::factor::LlEmit;
use super::structure::LuStructure;
use crate::numeric::supernodal::Input;

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::perturb_pivot;
use crate::numeric::supernodal::ScratchPool;
use crate::numeric::supernodal::{Gloc, Li, LlSchedule, PanelPtr};
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::Ordering;

/// One running [`lu_ll_factor_node`]'s buffers: the front's row and column
/// lists, the pivot column being decided and the sequential `cmod`'s update
/// blocks.
pub(super) struct LuScratch<T> {
    rows: Vec<Li>,
    cols: Vec<Li>,
    xl: Vec<Li>,
    xu: Vec<Li>,
    col: Vec<T>,
    lupd: Vec<T>,
    uupd: Vec<T>,
}

impl<T> crate::memory::HeapBytes for LuScratch<T> {
    fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes as b;
        b(&self.rows)
            + b(&self.cols)
            + b(&self.xl)
            + b(&self.xu)
            + b(&self.col)
            + b(&self.lupd)
            + b(&self.uupd)
    }
}

impl<T> Default for LuScratch<T> {
    fn default() -> Self {
        LuScratch {
            rows: Vec::new(),
            cols: Vec::new(),
            xl: Vec::new(),
            xu: Vec::new(),
            col: Vec::new(),
            lupd: Vec::new(),
            uupd: Vec::new(),
        }
    }
}

/// The scratch pools of one LU factorization: the node kernel's buffers per
/// running node and the tiled `cmod`'s update block per slab.
pub(super) struct LuPools<T> {
    pub(super) node: ScratchPool<LuScratch<T>>,
    pub(super) slab: ScratchPool<Vec<T>>,
}

/// A copy of a factor starts with scratch of its own, grown by its first
/// refactorization.
impl<T> Clone for LuPools<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> LuPools<T> {
    pub(super) fn new() -> Self {
        LuPools {
            node: ScratchPool::new(),
            slab: ScratchPool::new(),
        }
    }
}

/// `x -= L[rows, kb..k] u` for the off-block rows of a pivot column: `x` holds
/// rows `nfs..` of the column, `u` its entries `kb..k` (the `U` of this
/// panel's pivots so far), `lbuf` the panel's columns before it (column
/// stride `ld`); with `want` the largest squared magnitude of the result.
/// Runs of [`LU_DEEP_ROWS`](crate::constants::LU_DEEP_ROWS) rows take every
/// column in turn, so a run stays in cache; row by row the same operations
/// in the same order however the rows are split, so the parallel form is
/// bit-identical to the serial one.
#[allow(clippy::too_many_arguments)]
fn deep_update<T: Scalar>(
    lbuf: &[T],
    ld: usize,
    nfs: usize,
    kb: usize,
    u: &[T],
    x: &mut [T],
    want: bool,
    par: bool,
) -> f64 {
    use std::sync::atomic::AtomicU64;
    let run = crate::constants::LU_DEEP_ROWS;
    let apply = |r0: usize, xs: &mut [T]| -> f64 {
        let mut top = 0.0f64;
        for (q, chunk) in xs.chunks_mut(run).enumerate() {
            let at = nfs + r0 + q * run;
            for (jj, &uj) in u.iter().enumerate() {
                if uj != T::zero() {
                    let lj = &lbuf[(kb + jj) * ld + at..(kb + jj) * ld + at + chunk.len()];
                    for (v, &l) in chunk.iter_mut().zip(lj) {
                        *v = *v - l * uj;
                    }
                }
            }
            if want {
                top = chunk.iter().map(|v| v.magnitude_sq()).fold(top, f64::max);
            }
        }
        top
    };
    if par {
        // Non-negative doubles order as their bits.
        let top = AtomicU64::new(0);
        let chunk = x.len().div_ceil(crate::parallel::threads()).max(1);
        crate::parallel::chunks_mut(
            x,
            chunk,
            crate::parallel::Work::ns(f64::INFINITY),
            |c, xs| {
                top.fetch_max(apply(c * chunk, xs).to_bits(), Ordering::Relaxed);
            },
        );
        f64::from_bits(top.into_inner())
    } else {
        apply(0, x)
    }
}

/// Scale the off-block rows of a pivot column by the pivot's reciprocal.
fn deep_scale<T: Scalar>(x: &mut [T], pinv: T, par: bool) {
    if par {
        let chunk = x.len().div_ceil(crate::parallel::threads()).max(1);
        crate::parallel::chunks_mut(
            x,
            chunk,
            crate::parallel::Work::ns(f64::INFINITY),
            |_, xs| xs.iter_mut().for_each(|v| *v = *v * pinv),
        );
    } else {
        x.iter_mut().for_each(|v| *v = *v * pinv);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lu_ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    inp: Input<T>,
    sched: &LlSchedule,
    st: &LuStructure,
    pools: &LuPools<T>,
    emit: &LlEmit<T>,
    perturb_floor: Option<f64>,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    let ll_gemm_gate = kt.k.scalar_gate;
    let ll_gemm_par = kt.k.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let last = first + ncol;
    let n = sym.n;
    // SAFETY: a factored supernode's record is written once, before its
    // consumers (the ancestors, this node among them) start.
    let done = |k: usize| unsafe { emit.pivots.get(k) };
    let view = |k: usize| done(k).off(st, k, sym.supernodes[k].ncol);
    // The fully-summed block: this supernode's columns and the delayed
    // pivots its children pass (`delay`), then the off-block rows.
    let d_in: usize = snode.children.iter().map(|&c| done(c).passed()).sum();
    let nfs = ncol + d_in;
    let mut lent = pools.node.take();
    let LuScratch {
        rows,
        cols,
        xl,
        xu,
        col,
        lupd,
        uupd,
    } = &mut *lent;
    // The off-block rows and columns: the structure's, and those beyond it
    // that the delayed pivots of the children carry or that an updater with
    // such rows passes on by the rule of the structure.
    xl.clear();
    xu.clear();
    fn past(v: &[Li], last: usize) -> impl Iterator<Item = Li> + '_ {
        v.iter().copied().filter(move |&g| g as usize >= last)
    }
    for &c in &snode.children {
        if done(c).passed() > 0 {
            let (_, olc, ouc) = view(c);
            xl.extend(past(olc, last));
            xu.extend(past(ouc, last));
        }
    }
    for &k in sched.updaters(s) {
        let k = k as usize;
        if done(k).extras {
            let (_, olk, ouk) = view(k);
            if LuStructure::hits(ouk, first, last) {
                xl.extend(past(olk, last));
            }
            if LuStructure::hits(olk, first, last) {
                xu.extend(past(ouk, last));
            }
        }
    }
    let (st_l, st_u) = (&st.rows_l(s)[ncol..], &st.cols_u(s)[ncol..]);
    let mut extras = false;
    for (x, own) in [(&mut *xl, st_l), (&mut *xu, st_u)] {
        if !x.is_empty() {
            x.extend_from_slice(own);
            x.sort_unstable();
            x.dedup();
            extras |= x.len() > own.len();
        }
    }
    let off_l: &[Li] = if xl.len() > st_l.len() { xl } else { st_l };
    let off_u: &[Li] = if xu.len() > st_u.len() { xu } else { st_u };
    let (cn_l, cn_u) = (off_l.len(), off_u.len());
    let (nrow_l, nrow_u) = (nfs + cn_l, nfs + cn_u);
    // The front's rows and columns (global indices) in panel order.
    rows.clear();
    cols.clear();
    rows.extend((first..last).map(|g| g as Li));
    cols.extend((first..last).map(|g| g as Li));
    for &c in &snode.children {
        let pc = done(c);
        rows.extend_from_slice(&pc.rows[pc.npiv..]);
        cols.extend_from_slice(&pc.cols[pc.npiv..]);
    }
    rows.extend_from_slice(off_l);
    cols.extend_from_slice(off_u);
    // `lbuf`: the `L` panel, `nrow_l x nfs` (the diagonal block holds `L11`
    // below and `U11` on and above the diagonal). `ut`: the `U^T` panel,
    // `nrow_u x nfs`; the factorization writes its off-block rows, `U12[p, t]`
    // at `ut[p * nrow_u + nfs + t]` (row `p` of `U` is column `p` of the
    // panel), and the emit fills the diagonal block. The node's slots: the
    // analysis's, or, where pivots may be delayed, claimed now at the size
    // the front has, within the arenas' room (the factorization stops when
    // it is spent).
    // SAFETY: this task owns supernode `s`; nobody reads its slots before the
    // node is done (its ancestors start after it).
    let (lbuf, ut): (&mut [T], &mut [T]) = unsafe {
        let (vl, vu) = (nrow_l * nfs, nrow_u * nfs);
        // Room for the rows past the pivots: past all of the block without
        // delays, else past at least one pivot.
        let past = |nrow: usize| if emit.delays { nrow } else { nrow - nfs };
        match (
            emit.l_arena.claim(s, vl, past(nrow_l)),
            emit.u_arena.claim(s, vu, past(nrow_u)),
        ) {
            (Some(l), Some(u)) => (l, u),
            _ => {
                emit.overflowed.store(true, Ordering::Release);
                return Err(RslabError::InvalidInput(
                    "delayed pivots exceed their allowance".to_string(),
                ));
            }
        }
    };
    debug_assert!(lbuf.len() == nrow_l * nfs && ut.len() == nrow_u * nfs);

    // The finished panels of a factored descendant `k` (its pivot count, the
    // rows and columns past its pivots, its panels and their heights).
    let updater = |k: usize| {
        let (nck, ol, ou) = done(k).off(st, k, sym.supernodes[k].ncol);
        // SAFETY: `k` is a factored descendant of `s`; its slots are published
        // and never written again.
        let (lk, uk): (&[T], &[T]) = unsafe { (emit.l_arena.slot(k), emit.u_arena.slot(k)) };
        (nck, ol, ou, lk, uk, nck + ol.len(), nck + ou.len())
    };
    {
        // Global to local: `L` rows into `lbuf`, `U` columns into `ut`.
        let gloc = Gloc::new(n, rows);
        let gloc_u = Gloc::second(n, cols);
        // Assemble columns of s (full) into lbuf, and the U12 rows into ut. The
        // exact structure holds every entry of `B`.
        for p in 0..ncol {
            let c = first + p;
            for (g, v) in inp.col(c) {
                debug_assert!(gloc[g] != Li::MAX, "entry outside the L structure");
                let li = gloc[g] as usize;
                lbuf[p * nrow_l + li] = lbuf[p * nrow_l + li] + v;
            }
            for (g, v) in inp.row(c) {
                debug_assert!(gloc_u[g] != Li::MAX, "entry outside the U structure");
                let lc = gloc_u[g] as usize;
                ut[p * nrow_u + lc] = ut[p * nrow_u + lc] + v;
            }
        }
        // cmod from every factored descendant, into this supernode's own
        // columns and rows (the delayed ones arrive updated, below). NOTE:
        // cmod-aggregation (K-stacking many descendant updates into one fat
        // GEMM) was measured and rejected - across MoM topologies 91-95 % of
        // cmod flop already runs as large parallel GEMMs, and the only
        // aggregation reaching those dominant updates carries an 11-15x
        // zero-pad blowup (each top-of-tree descendant touches a small,
        // distinct row/col subset of the large target).
        let plan = crate::numeric::supernodal::CmodPlan::new(
            sym,
            s,
            sched.updaters(s),
            |k| done(k).off(st, k, sym.supernodes[k].ncol),
            true,
            &kt.k,
        );
        let (spans, forks, tile_w, tiled) = (&plan.spans, plan.forks, plan.tile_w, plan.tiled);
        let tile_u = kt.k.cmod_tile(cn_u);

        // Column-tiled parallel cmod: disjoint `&mut` slabs of the target
        // buffers; per slab every updater's contribution in updater order with a
        // serial GEMM (see `CmodPlan::tiled`). The LU node has two target buffers,
        // so the tiling runs as two phases: `lbuf` slabs (the `L`/`U11` updates),
        // then runs of `U12` columns in `ut`.
        if tiled {
            let (gloc_ref, gloc_u_ref) = (&gloc, &gloc_u);
            let work = crate::parallel::Work::ns(f64::INFINITY);
            let lslab = &mut lbuf[..nrow_l * ncol];
            crate::parallel::chunks_mut(lslab, nrow_l * tile_w, work, |ti, slab| {
                let c0 = ti * tile_w;
                let c1 = (c0 + tile_w).min(ncol);
                let mut lupd = pools.slab.take();
                for sp in spans {
                    let (nck, ol, ou, lk, uk, nrk_l, nrk_u) = updater(sp.k);
                    let (p0l, (p0u, p1u)) = (sp.l.0, sp.u);
                    let q0 = p0u + ou[p0u..p1u].partition_point(|&g| (g as usize) < first + c0);
                    let q1 = p0u + ou[p0u..p1u].partition_point(|&g| (g as usize) < first + c1);
                    let npk = q1 - q0;
                    let mrows = ol.len() - p0l;
                    if npk == 0 || mrows == 0 {
                        continue;
                    }
                    lupd.clear();
                    lupd.resize(mrows * npk, T::zero());
                    // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                    unsafe {
                        crate::dense::gemm_backend::gemm(
                            mrows,
                            npk,
                            nck,
                            lupd.as_mut_ptr(),
                            mrows as isize,
                            1,
                            false,
                            lk.as_ptr().add(nck + p0l),
                            nrk_l as isize,
                            1,
                            uk.as_ptr().add(nck + q0),
                            1,
                            nrk_u as isize,
                            T::zero(),
                            T::one(),
                            false,
                            false,
                            false,
                            crate::dense::gemm_backend::GemmMode::new(
                                gemm::Parallelism::None,
                                &kt.k,
                            ),
                        );
                    }
                    for jj in 0..npk {
                        let cbase = (ou[q0 + jj] as usize - first - c0) * nrow_l;
                        let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                        for i in 0..mrows {
                            let dst = cbase + gloc_ref[ol[p0l + i] as usize] as usize;
                            slab[dst] = slab[dst] - ucol[i];
                        }
                    }
                }
            });
            if cn_u > 0 {
                // A slab is the run `[u0, u1)` of U12's columns: rows `nfs + u0..nfs + u1`
                // of every `ut` column, disjoint between slabs.
                let up = PanelPtr(ut.as_mut_ptr());
                crate::parallel::for_each(cn_u.div_ceil(tile_u), work, |ti| {
                    let u0 = ti * tile_u;
                    let u1 = (u0 + tile_u).min(cn_u);
                    let g0 = off_u[u0];
                    let g1 = if u1 < cn_u { off_u[u1] } else { Li::MAX };
                    let mut uupd = pools.slab.take();
                    for sp in spans {
                        let (nck, ol, ou, lk, uk, nrk_l, nrk_u) = updater(sp.k);
                        let ((p0l, p1l), p1u) = (sp.l, sp.u.1);
                        let t0 = p1u + ou[p1u..].partition_point(|&g| g < g0);
                        let t1 = p1u + ou[p1u..].partition_point(|&g| g < g1);
                        let (ntr, npk) = (t1 - t0, p1l - p0l);
                        if ntr == 0 || npk == 0 {
                            continue;
                        }
                        uupd.clear();
                        uupd.resize(npk * ntr, T::zero());
                        // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                        unsafe {
                            crate::dense::gemm_backend::gemm(
                                npk,
                                ntr,
                                nck,
                                uupd.as_mut_ptr(),
                                npk as isize,
                                1,
                                false,
                                lk.as_ptr().add(nck + p0l),
                                nrk_l as isize,
                                1,
                                uk.as_ptr().add(nck + t0),
                                1,
                                nrk_u as isize,
                                T::zero(),
                                T::one(),
                                false,
                                false,
                                false,
                                crate::dense::gemm_backend::GemmMode::new(
                                    gemm::Parallelism::None,
                                    &kt.k,
                                ),
                            );
                        }
                        for jj in 0..ntr {
                            let lt = gloc_u_ref[ou[t0 + jj] as usize] as usize;
                            let ucol = &uupd[jj * npk..jj * npk + npk];
                            for i in 0..npk {
                                // SAFETY: row `lt` lies in this slab's run.
                                unsafe {
                                    let d =
                                        up.get().add((ol[p0l + i] as usize - first) * nrow_u + lt);
                                    *d = *d - ucol[i];
                                }
                            }
                        }
                    }
                });
            }
        }

        // Sequential per-update cmod (small nodes / narrow panels).
        for sp in spans.iter().filter(|_| !tiled) {
            let (nck, ol, ou, lk, uk, nrk_l, nrk_u) = updater(sp.k);
            let ((p0l, p1l), (p0u, p1u)) = (sp.l, sp.u);
            // L update: rows `ol[p0l..]` times the `U` columns landing here;
            // U12 update: the `L` rows landing here times the `U` columns past here.
            let (mrows, npk_u) = (ol.len() - p0l, p1u - p0u);
            let (npk_l, ntrail) = (p1l - p0l, ou.len() - p1u);
            if (mrows * npk_u + npk_l * ntrail) * nck < ll_gemm_gate {
                // Scalar path.
                for jj in 0..npk_u {
                    let tcol = ou[p0u + jj] as usize - first;
                    for i in 0..mrows {
                        let mut acc = T::zero();
                        for ck in 0..nck {
                            acc = acc
                                + lk[(nck + p0l + i) + ck * nrk_l]
                                    * uk[ck * nrk_u + nck + p0u + jj];
                        }
                        let trow = gloc[ol[p0l + i] as usize] as usize;
                        lbuf[tcol * nrow_l + trow] = lbuf[tcol * nrow_l + trow] - acc;
                    }
                }
                for jj in 0..ntrail {
                    let tu = gloc_u[ou[p1u + jj] as usize] as usize;
                    for i in 0..npk_l {
                        let mut acc = T::zero();
                        for ck in 0..nck {
                            acc = acc
                                + lk[(nck + p0l + i) + ck * nrk_l]
                                    * uk[ck * nrk_u + nck + p1u + jj];
                        }
                        let urow = ol[p0l + i] as usize - first;
                        ut[urow * nrow_u + tu] = ut[urow * nrow_u + tu] - acc;
                    }
                }
                continue;
            }
            // `forks` folds in the join-steal guard: a small node never forks here.
            let par = |work: usize| {
                if forks && work >= ll_gemm_par {
                    gemm::Parallelism::Rayon(0)
                } else {
                    gemm::Parallelism::None
                }
            };
            if mrows > 0 && npk_u > 0 {
                // Lupd(mrows x npk_u) = L_k[rows from p0l, :] * U_k[:, landing columns].
                lupd.clear();
                lupd.resize(mrows * npk_u, T::zero());
                // SAFETY: lhs (lk off-diag rows), rhs (uk landing columns), dst
                // (lupd) are disjoint; strides in bounds.
                unsafe {
                    crate::dense::gemm_backend::gemm(
                        mrows,
                        npk_u,
                        nck,
                        lupd.as_mut_ptr(),
                        mrows as isize,
                        1,
                        false,
                        lk.as_ptr().add(nck + p0l),
                        nrk_l as isize,
                        1,
                        uk.as_ptr().add(nck + p0u),
                        1,
                        nrk_u as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        crate::dense::gemm_backend::GemmMode::new(par(mrows * npk_u * nck), &kt.k),
                    );
                }
                for jj in 0..npk_u {
                    let cbase = (ou[p0u + jj] as usize - first) * nrow_l;
                    let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                    for i in 0..mrows {
                        let dst = cbase + gloc[ol[p0l + i] as usize] as usize;
                        lbuf[dst] = lbuf[dst] - ucol[i];
                    }
                }
            }
            if npk_l > 0 && ntrail > 0 {
                // Uupd(npk_l x ntrail) = L_k[landing rows, :] * U_k[:, columns past s].
                uupd.clear();
                uupd.resize(npk_l * ntrail, T::zero());
                // SAFETY: as above; rhs is the trailing U columns of `uk`.
                unsafe {
                    crate::dense::gemm_backend::gemm(
                        npk_l,
                        ntrail,
                        nck,
                        uupd.as_mut_ptr(),
                        npk_l as isize,
                        1,
                        false,
                        lk.as_ptr().add(nck + p0l),
                        nrk_l as isize,
                        1,
                        uk.as_ptr().add(nck + p1u),
                        1,
                        nrk_u as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        crate::dense::gemm_backend::GemmMode::new(par(npk_l * ntrail * nck), &kt.k),
                    );
                }
                for jj in 0..ntrail {
                    let lt = gloc_u[ou[p1u + jj] as usize] as usize;
                    let ucol = &uupd[jj * npk_l..jj * npk_l + npk_l];
                    for i in 0..npk_l {
                        let dst = (ol[p0l + i] as usize - first) * nrow_u + lt;
                        ut[dst] = ut[dst] - ucol[i];
                    }
                }
            }
        }

        // The delayed pivots of the children, as they left them: the Schur
        // complement of each delayed column over the child's rows past its
        // pivots, and of each delayed row over its columns past them. Every
        // update from below is in (descendants reach a delayed index only
        // through the child that delayed it), and every row and column lies
        // in this front (the children's ones past it were added above).
        for &c in &snode.children {
            let pc = done(c);
            if pc.passed() == 0 {
                continue;
            }
            let (npc, olc, ouc) = pc.off(st, c, sym.supernodes[c].ncol);
            let (ldc, lduc, nfs_c) = (npc + olc.len(), npc + ouc.len(), pc.rows.len());
            // SAFETY: `c` is factored; its slots are published and never
            // written again before the emit, which runs after this node.
            let (lc, uc): (&[T], &[T]) = unsafe { (emit.l_arena.slot(c), emit.u_arena.slot(c)) };
            for (j, &gc) in pc.cols[npc..].iter().enumerate() {
                let dst = gloc_u[gc as usize] as usize * nrow_l;
                let src = &lc[(npc + j) * ldc + npc..(npc + j + 1) * ldc];
                for (&g, &v) in olc.iter().zip(src) {
                    debug_assert!(gloc[g as usize] != Li::MAX, "delayed row outside the front");
                    let li = dst + gloc[g as usize] as usize;
                    lbuf[li] = lbuf[li] + v;
                }
            }
            for (i, &gr) in pc.rows[npc..].iter().enumerate() {
                let r = gloc[gr as usize] as usize;
                let src = &uc[(npc + i) * lduc + nfs_c..(npc + i + 1) * lduc];
                for (&g, &v) in ouc[nfs_c - npc..].iter().zip(src) {
                    let g = g as usize;
                    if g < first + ncol {
                        let li = (g - first) * nrow_l + r;
                        lbuf[li] = lbuf[li] + v;
                    } else {
                        debug_assert!(gloc_u[g] != Li::MAX, "delayed column outside the front");
                        let li = r * nrow_u + gloc_u[g] as usize;
                        ut[li] = ut[li] + v;
                    }
                }
            }
        }
    }

    // cdiv: in-place **blocked** panel LU, no trailing update outside the
    // panel: a getrf - an NB-wide panel factored column by column, then the
    // dominant trailing update as a single SIMD GEMM (rank-NB) - restricted to
    // the panel: the trailing is the remaining panel columns (`lbuf`) plus the
    // `U12` rows (in `ut`), with no `A22`/CB. This routes the
    // `O(nfs^2*nrow_l)` cdiv work (the measured 77 % of the left-looking
    // factor) through BLAS-3. Within an NB panel the columns are brought up to
    // date one at a time from its earlier pivots (left-looking, a GEMV for
    // the off-block rows), so each pivot is decided on the column as it
    // stands, its off-block rows included, while every column still to come
    // keeps the state of the panel's start.
    // Panel width. Swept 32/48/64/96 on the MoM fronts: 32 optimal for typical
    // panels - but root-class WIDE panels want a fatter deferred-GEMM inner
    // dimension (k = nb), the same lever as the LDLT twin's adaptive nb. Pure
    // function of the front, never of the thread count.
    let nb_cdiv = kt.k.panel_width(kt.k.lu_panel_nb, nfs);
    // Join-steal guard (see the cmod fork gate above): a small node must not
    // fork inside its cdiv either.
    let ll_cdiv_par = kt.k.cdiv_par(nrow_l, nfs);
    // The off-block rows of a pivot column go parallel where a panel's worth
    // of them is cdiv-parallel work.
    let deep_par = cn_l * nb_cdiv * nb_cdiv >= ll_cdiv_par;
    // **Threshold** partial pivoting (UMFPACK-style): keep the diagonal pivot
    // unless it is below `THRESH` of the largest entry of its column - so a
    // well-scaled/equilibrated matrix never interchanges (no fill or accuracy
    // cost) while small/zero diagonals still get a stable pivot. The column
    // is the front's whole column, its off-block rows included, but only its
    // fully-summed rows can pivot here: where none of them passes, the column
    // is delayed to the parent (`delay`), which has the rows that can, while
    // this front's allowance lasts; past it, or at a root, the pivot is the
    // diagonal unless it is below `THRESH` of the fully-summed rows (the
    // front-confined pivot), and counts as forced. `THRESH^2` compared on
    // squared magnitudes. `THRESH = kt.pivot_threshold` (tunable, default
    // 0.1); `u = 1` recovers full partial pivoting, `u = 0` keeps the diagonal
    // unless it is exactly zero and skips the search: the "skip pivot search"
    // speed lever for fixed-pattern value sequences (solver-in-the-loop:
    // reuse a good order across a frequency sweep / time-stepping). A
    // sub-floor / zero diagonal is still caught by the pivot policy.
    let thresh_sq = kt.pivot_threshold * kt.pivot_threshold;
    // Any front with a parent may delay.
    let may_delay = emit.delays && sched.parent[s] != usize::MAX;
    let (mut forced, mut floors) = (0usize, Vec::<(usize, T)>::new());
    // Columns `[npiv, live)` are still to pivot, `[live, nfs)` delayed.
    let (mut npiv, mut live) = (0usize, nfs);
    while npiv < live {
        kt.interrupted()?;
        let kb = npiv;
        while npiv < live && npiv - kb < nb_cdiv {
            let k = npiv;
            // Column `k` from row `kb` down, brought up to date with this
            // panel's pivots `kb..k`. Where it may be delayed, in a copy, so a
            // delayed column keeps the state of the columns still to come.
            let (stable, p, best, deep) = {
                let (before, rest) = lbuf.split_at_mut(k * nrow_l);
                let own = &mut rest[kb..nrow_l];
                let x: &mut [T] = if may_delay {
                    col.clear();
                    col.extend_from_slice(own);
                    col
                } else {
                    own
                };
                let (fs, offb) = x.split_at_mut(nfs - kb);
                for j in kb..k {
                    let u = fs[j - kb];
                    if u != T::zero() {
                        let lj = &before[j * nrow_l + j + 1..j * nrow_l + nfs];
                        for (v, &l) in fs[j + 1 - kb..].iter_mut().zip(lj) {
                            *v = *v - l * u;
                        }
                    }
                }
                let want = thresh_sq > 0.0;
                let deep =
                    deep_update(before, nrow_l, nfs, kb, &fs[..k - kb], offb, want, deep_par);
                // The largest fully-summed candidate.
                let cand = &fs[k - kb..];
                let (mut p, mut best) = (0, cand[0].magnitude_sq());
                if want {
                    for (i, v) in cand.iter().enumerate().skip(1) {
                        let m = v.magnitude_sq();
                        if m > best {
                            best = m;
                            p = i;
                        }
                    }
                }
                // Some fully-summed row passes the test over the whole column.
                (best >= thresh_sq * deep, p, best, deep)
            };
            let mut piv_row = k;
            if thresh_sq > 0.0 {
                if !stable {
                    if may_delay {
                        // Delayed: to the end of the live columns, which this
                        // panel's pivots have not touched either.
                        live -= 1;
                        for i in 0..nrow_l {
                            lbuf.swap(k * nrow_l + i, live * nrow_l + i);
                        }
                        cols.swap(k, live);
                        continue;
                    }
                    forced += 1;
                }
                let reference = if stable { best.max(deep) } else { best };
                let diag = if may_delay {
                    col[k - kb]
                } else {
                    lbuf[k * nrow_l + k]
                };
                if p != 0 && diag.magnitude_sq() < thresh_sq * reference {
                    piv_row = k + p;
                }
            }
            if may_delay {
                lbuf[k * nrow_l + kb..(k + 1) * nrow_l].copy_from_slice(col);
            }
            if piv_row != k {
                for c in 0..nfs {
                    lbuf.swap(c * nrow_l + k, c * nrow_l + piv_row);
                }
                for t in nfs..nrow_u {
                    ut.swap(k * nrow_u + t, piv_row * nrow_u + t);
                }
                rows.swap(k, piv_row);
            }
            let mut piv = lbuf[k * nrow_l + k];
            match perturb_floor {
                Some(floor) if piv.magnitude() < floor => {
                    let lifted = perturb_pivot(piv, floor);
                    floors.push((k, lifted - piv));
                    piv = lifted;
                }
                None if piv == T::zero() => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            lbuf[k * nrow_l + k] = piv;
            let pinv = piv.recip();
            let (fs, deep) = lbuf[k * nrow_l + k + 1..(k + 1) * nrow_l].split_at_mut(nfs - k - 1);
            fs.iter_mut().for_each(|v| *v = *v * pinv);
            deep_scale(deep, pinv, deep_par);
            npiv += 1;
        }
        let ke = npiv;
        let pw = ke - kb;
        if pw == 0 {
            continue;
        }
        // TRSM: U = L11^-1 * (trailing panel columns of lbuf) and the U12 rows.
        // Each trailing column is an independent forward substitution reading
        // only the finished panel columns [kb, ke), so the block parallelizes
        // over disjoint column chunks - bit-identical per-column op order.
        // Profiled at 22% of cdiv CPU when serial (MoM fronts).
        if (nfs - ke) * pw * pw >= ll_cdiv_par {
            let (head, tail) = lbuf.split_at_mut(ke * nrow_l);
            crate::parallel::chunks_mut(
                tail,
                nrow_l,
                crate::parallel::Work::ns(f64::INFINITY),
                |_, col| {
                    for r in (kb + 1)..ke {
                        let mut acc = col[r];
                        for i in kb..r {
                            acc = acc - head[i * nrow_l + r] * col[i];
                        }
                        col[r] = acc;
                    }
                },
            );
        } else {
            for j in ke..nfs {
                for r in (kb + 1)..ke {
                    let mut acc = lbuf[j * nrow_l + r];
                    for i in kb..r {
                        acc = acc - lbuf[i * nrow_l + r] * lbuf[j * nrow_l + i];
                    }
                    lbuf[j * nrow_l + r] = acc;
                }
            }
        }
        // U12 rows (the `cn_u` contribution columns of U): the forward substitution
        // over the panel rows, `x_r -= L[r, i] x_i` for `i` ascending, on whole rows
        // of U12 (the contiguous runs `ut[r * nrow_u + nfs..(r + 1) * nrow_u]`), so every
        // entry sees the operations of its own column's substitution in order;
        // parallel over disjoint runs of the columns.
        let trsm_u = |t0: usize, t1: usize, u: PanelPtr<T>, lref: &[T]| {
            for r in (kb + 1)..ke {
                for i in kb..r {
                    let l = lref[i * nrow_l + r];
                    // SAFETY: rows `r != i` of `ut`, the caller's columns `[t0, t1)`.
                    unsafe {
                        let (xr, xi) = (u.get().add(r * nrow_u), u.get().add(i * nrow_u));
                        for t in t0..t1 {
                            *xr.add(t) = *xr.add(t) - l * *xi.add(t);
                        }
                    }
                }
            }
        };
        let up = PanelPtr(ut.as_mut_ptr());
        if cn_u * pw * pw >= ll_cdiv_par {
            let lref: &[T] = lbuf;
            let run = kt.k.trsm_rows;
            crate::parallel::for_each(
                cn_u.div_ceil(run),
                crate::parallel::Work::ns(f64::INFINITY),
                |b| {
                    let t0 = nfs + b * run;
                    trsm_u(t0, (t0 + run).min(nrow_u), up, lref)
                },
            );
        } else {
            trsm_u(nfs, nrow_u, up, lbuf);
        }
        // GEMM: lbuf[ke.., ke..nfs] -= L21[ke.., kb..ke] * U[kb..ke, ke..nfs].
        let mt = nrow_l - ke;
        let nt = nfs - ke;
        if mt > 0 && nt > 0 {
            let par = if (mt * nt * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let base = lbuf.as_mut_ptr();
            // SAFETY: the three sub-blocks of `lbuf` are disjoint; strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    mt,
                    nt,
                    pw,
                    base.add(ke * nrow_l + ke),
                    nrow_l as isize,
                    1,
                    true,
                    base.add(kb * nrow_l + ke),
                    nrow_l as isize,
                    1,
                    base.add(ke * nrow_l + kb),
                    nrow_l as isize,
                    1,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    crate::dense::gemm_backend::GemmMode::new(par, &kt.k),
                );
            }
        }
        // GEMM: U12[ke..nfs, :] -= L[ke..nfs, kb..ke] * U12[kb..ke, :].
        if cn_u > 0 && nt > 0 {
            let par = if (nt * cn_u * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let lptr = lbuf.as_ptr();
            let uptr = ut.as_mut_ptr();
            // SAFETY: dst (U12 rows `ke..nfs`, `ut` columns) is disjoint from the
            // read sub-blocks of `lbuf` and U12 (rows `kb..ke`); strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    nt,
                    cn_u,
                    pw,
                    uptr.add(ke * nrow_u + nfs),
                    1,
                    nrow_u as isize,
                    true,
                    lptr.add(kb * nrow_l + ke),
                    nrow_l as isize,
                    1,
                    uptr.add(kb * nrow_u + nfs),
                    1,
                    nrow_u as isize,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    crate::dense::gemm_backend::GemmMode::new(par, &kt.k),
                );
            }
        }
    }
    if !floors.is_empty() {
        let mut all = emit.floors.lock().unwrap_or_else(|e| e.into_inner());
        all.extend(floors.into_iter().map(|(k, delta)| (s, k, delta)));
    }
    // What the node leaves: its pivots, the pivots it passes on (the rows and
    // columns past its pivots, as its ancestors read them, start with them).
    let own_lists = npiv < nfs || extras;
    let pivots = NodePivots {
        npiv,
        rows: rows[..nfs].to_vec(),
        cols: cols[..nfs].to_vec(),
        off_l: if own_lists {
            rows[npiv..].to_vec()
        } else {
            Vec::new()
        },
        off_u: if own_lists {
            cols[npiv..].to_vec()
        } else {
            Vec::new()
        },
        extras,
        forced,
    };
    // SAFETY: each supernode's record is written by its own task, once.
    unsafe { emit.pivots.set(s, pivots) };
    if !emit.delays {
        // Without delayed pivots every position is the analysis's.
        // SAFETY: this node's own indices.
        unsafe { emit.place(s, emit.e_offset[s]) };
    }
    Ok(())
}

/// Heap bytes [`lu_ll_factor_node`] allocates for supernode `s` while it
/// runs, for the memory plan: the tiled cmod's per-slab buffers (the `L`
/// slabs, then the `U12` runs) on up to `workers` threads at once, or the
/// sequential cmod's buffers, which stay allocated through the cdiv, and
/// the front's row and column lists and its pivot column; the largest split
/// planes its GEMMs take on a worker (complex fields) and how many workers
/// may take them at once. Replays the kernel's plan and growth of its
/// buffers (of a front without delayed pivots).
pub(super) fn lu_node_scratch<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    st: &LuStructure,
    k: &crate::KernelSettings,
    workers: usize,
) -> (u64, usize, usize) {
    use crate::dense::gemm_backend::split_plane_entries;
    use crate::memory::grown;
    let vb = std::mem::size_of::<T>();
    let (first, ncol) = (sym.supernodes[s].first_col, sym.supernodes[s].ncol);
    let cols_u = st.cols_u(s);
    let (nrow_l, cn_u) = (st.rows_l(s).len(), cols_u.len() - ncol);
    let nb = k.panel_width(k.lu_panel_nb, ncol);
    let lists = (nrow_l + cols_u.len()) * crate::constants::IDX32_BYTES;
    let cdiv = nrow_l * vb + lists;
    // The cdiv's two GEMMs per panel: the trailing `L` update and the `U12` rows.
    let mut planes = 0;
    if T::COMPLEX {
        let par_cdiv = k.cdiv_par(nrow_l, ncol);
        let mut kb = 0;
        while kb < ncol {
            let ke = (kb + nb).min(ncol);
            let (pw, mt, nt) = (ke - kb, nrow_l - ke, ncol - ke);
            if mt > 0 && nt > 0 {
                planes = planes.max(split_plane_entries(mt, nt, pw, mt * nt * pw >= par_cdiv, k));
            }
            if cn_u > 0 && nt > 0 {
                planes = planes.max(split_plane_entries(
                    nt,
                    cn_u,
                    pw,
                    nt * cn_u * pw >= par_cdiv,
                    k,
                ));
            }
            kb = ke;
        }
    }
    let plan = crate::numeric::supernodal::CmodPlan::new(
        sym,
        s,
        sched.updaters(s),
        |k| {
            let nck = sym.supernodes[k].ncol;
            (nck, &st.rows_l(k)[nck..], &st.cols_u(k)[nck..])
        },
        true,
        k,
    );
    let updater = |kk: usize| {
        let nck = sym.supernodes[kk].ncol;
        (nck, &st.rows_l(kk)[nck..], &st.cols_u(kk)[nck..])
    };
    let workers = workers.max(1);
    if plan.tiled {
        let tw = plan.tile_w;
        let slabs = ncol.div_ceil(tw);
        let (mut worst_l, mut slab_planes) = (0, 0);
        for ti in 0..slabs {
            let (c0, c1) = (ti * tw, (ti * tw + tw).min(ncol));
            let mut lupd = 0;
            for sp in &plan.spans {
                let (nck, ol, ou) = updater(sp.k);
                let (p0l, (p0u, p1u)) = (sp.l.0, sp.u);
                let q0 = p0u + ou[p0u..p1u].partition_point(|&g| (g as usize) < first + c0);
                let q1 = p0u + ou[p0u..p1u].partition_point(|&g| (g as usize) < first + c1);
                lupd = grown(lupd, (ol.len() - p0l) * (q1 - q0));
                if T::COMPLEX && q1 > q0 && ol.len() > p0l {
                    let p = split_plane_entries(ol.len() - p0l, q1 - q0, nck, false, k);
                    slab_planes = slab_planes.max(p);
                }
            }
            worst_l = worst_l.max(lupd);
        }
        let tile_u = k.cmod_tile(cn_u);
        let runs = cn_u.div_ceil(tile_u);
        let mut worst_u = 0;
        for ti in 0..runs {
            let (u0, u1) = (ti * tile_u, (ti * tile_u + tile_u).min(cn_u));
            let g0 = cols_u[ncol + u0];
            let g1 = cols_u.get(ncol + u1).copied().unwrap_or(Li::MAX);
            let mut uupd = 0;
            for sp in &plan.spans {
                let (nck, _, ou) = updater(sp.k);
                let ((p0l, p1l), p1u) = (sp.l, sp.u.1);
                let t0 = p1u + ou[p1u..].partition_point(|&g| g < g0);
                let t1 = p1u + ou[p1u..].partition_point(|&g| g < g1);
                uupd = grown(uupd, (p1l - p0l) * (t1 - t0));
                if T::COMPLEX && t1 > t0 && p1l > p0l {
                    let p = split_plane_entries(p1l - p0l, t1 - t0, nck, false, k);
                    slab_planes = slab_planes.max(p);
                }
            }
            worst_u = worst_u.max(uupd);
        }
        let slabs_l = slabs;
        let slabs = (worst_l * slabs.min(workers)).max(worst_u * runs.min(workers));
        let copies = if slab_planes > planes {
            slabs_l.max(runs).min(workers)
        } else {
            1
        };
        (
            (slabs * vb) as u64 + cdiv as u64,
            planes.max(slab_planes),
            copies,
        )
    } else {
        let (mut lupd, mut uupd) = (0, 0);
        for sp in &plan.spans {
            let (nck, ol, ou) = updater(sp.k);
            let ((p0l, p1l), (p0u, p1u)) = (sp.l, sp.u);
            let (mrows, npk_u) = (ol.len() - p0l, p1u - p0u);
            let (npk_l, ntrail) = (p1l - p0l, ou.len() - p1u);
            if (mrows * npk_u + npk_l * ntrail) * nck < k.scalar_gate {
                continue;
            }
            let par = |work: usize| plan.forks && work >= k.par_gemm;
            if mrows > 0 && npk_u > 0 {
                lupd = grown(lupd, mrows * npk_u);
                if T::COMPLEX {
                    let p = split_plane_entries(mrows, npk_u, nck, par(mrows * npk_u * nck), k);
                    planes = planes.max(p);
                }
            }
            if npk_l > 0 && ntrail > 0 {
                uupd = grown(uupd, npk_l * ntrail);
                if T::COMPLEX {
                    let p = split_plane_entries(npk_l, ntrail, nck, par(npk_l * ntrail * nck), k);
                    planes = planes.max(p);
                }
            }
        }
        (((lupd + uupd) * vb + cdiv) as u64, planes, 1)
    }
}
