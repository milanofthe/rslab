//! One supernode of the left-looking LU factorization: assembly, the updates
//! of its factored descendants (`cmod`, into both `L` and `U12`), the
//! delayed pivots its children pass, then the blocked panel LU with
//! threshold partial pivoting, which may delay pivots in turn (`delay`).

use super::delay::NodePivots;
use super::factor::LlEmit;
use super::structure::LuStructure;
use crate::numeric::supernodal::Input;

use crate::constants::PANEL_BLOCK;
use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::panel::Layout;
use crate::numeric::supernodal::panel::PanelScratch;
use crate::numeric::supernodal::perturb_pivot;
use crate::numeric::supernodal::schur_tiles;
use crate::numeric::supernodal::{Cells, ScratchPool};
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
    lb: Vec<usize>,
    ub: Vec<usize>,
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
            + b(&self.lb)
            + b(&self.ub)
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
            lb: Vec::new(),
            ub: Vec::new(),
        }
    }
}

/// The scratch pools of one LU factorization: the node kernel's buffers per
/// running node and the tiled `cmod`'s update block per slab.
pub(super) struct LuPools<T> {
    pub(super) node: ScratchPool<LuScratch<T>>,
    pub(super) slab: ScratchPool<Vec<T>>,
    /// The buffers of the emit's [`finish_panel`](crate::numeric::supernodal::panel::finish_panel).
    pub(super) finish: ScratchPool<PanelScratch<T>>,
    /// The supernodes' pivot records, filled again by each factorization.
    pub(super) pivots: ScratchPool<Cells<NodePivots<T>>>,
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
            finish: ScratchPool::new(),
            pivots: ScratchPool::new(),
        }
    }
}

/// What a multiply-add of the descendants' updates costs, learned from
/// their serial runs: the update of a large node goes as wide as its work is
/// worth.
static UPDATE_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::GEMM_NS_PER_FLOP);

/// What a multiply-subtract of the panel kernels' row passes (the off-block
/// rows' updates and scaling, the triangular solves of `U`) costs, learned
/// from their serial runs: their sections go as wide as their work is worth.
/// Each row takes the same operations however the rows are split, so the
/// width changes no bits.
static ROW_PASS_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::SCAN_NS_PER_ENTRY);

/// The work of `ops` multiply-subtracts of a row pass on `T`.
fn row_pass<T: Scalar>(ops: usize) -> crate::parallel::Work<'static> {
    crate::parallel::Work::of(ops as f64, ROW_PASS_RATE.of::<T>())
}

/// `x -= L[rows, kb..k] u` for the off-block rows of a pivot column: `x` holds
/// the column's off-block rows, `u` its entries `kb..k` (the `U` of this
/// panel's pivots so far), `lrect` the panel's off-block rows (column `j` at
/// `j * ld`); with `want` the largest squared magnitude of the result.
/// Runs of [`LU_DEEP_ROWS`](crate::constants::LU_DEEP_ROWS) rows take every
/// column in turn, so a run stays in cache; row by row the same operations
/// in the same order however the rows are split, so the parallel form is
/// bit-identical to the serial one.
fn deep_update<T: Scalar>(
    lrect: &[T],
    ld: usize,
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
            let at = r0 + q * run;
            for (jj, &uj) in u.iter().enumerate() {
                if uj != T::zero() {
                    let lj = &lrect[(kb + jj) * ld + at..(kb + jj) * ld + at + chunk.len()];
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
        let work = row_pass::<T>(x.len() * u.len().max(1));
        crate::parallel::chunks_mut(x, chunk, work, |c, xs| {
            top.fetch_max(apply(c * chunk, xs).to_bits(), Ordering::Relaxed);
        });
        f64::from_bits(top.into_inner())
    } else {
        apply(0, x)
    }
}

/// Scale the off-block rows of a pivot column by the pivot's reciprocal.
fn deep_scale<T: Scalar>(x: &mut [T], pinv: T, par: bool) {
    if par {
        let chunk = x.len().div_ceil(crate::parallel::threads()).max(1);
        let work = row_pass::<T>(x.len());
        crate::parallel::chunks_mut(x, chunk, work, |_, xs| {
            xs.iter_mut().for_each(|v| *v = *v * pinv)
        });
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
        lb,
        ub,
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
    // The front's two panels: `lbuf` the `L` panel and `ut` the `U^T` panel,
    // each of the `nfs` fully-summed columns and the rows (columns) past
    // them ([`Layout`]). The front's entry `(i, j)` lies in `lbuf` on and
    // below the diagonal and in `ut` above it (its entry `(j, i)`, [`Front`]):
    // `U` above the diagonal, `U11` and `U12`, is written in `ut` as the
    // factorization goes and the emit adds `U`'s diagonal. The node's slots:
    // the analysis's, or, where pivots may be delayed, claimed now at the
    // size the front has, within the arenas' room (the factorization stops
    // when it is spent).
    let front = Front::new(Layout::new(nfs, cn_l), Layout::new(nfs, cn_u), lb, ub);
    // SAFETY: this task owns supernode `s`; nobody reads its slots before the
    // node is done (its ancestors start after it).
    let (lbuf, ut): (&mut [T], &mut [T]) = unsafe {
        // Room for the rows past the pivots: past all of the block without
        // delays, else past at least one pivot.
        let past = |nrow: usize| if emit.delays { nrow } else { nrow - nfs };
        match (
            emit.l_arena.claim(s, front.l.len(), past(nrow_l)),
            emit.u_arena.claim(s, front.u.len(), past(nrow_u)),
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
    debug_assert!(lbuf.len() == front.l.len() && ut.len() == front.u.len());
    let (lp, up) = (PanelPtr(lbuf.as_mut_ptr()), PanelPtr(ut.as_mut_ptr()));
    // The front's entry `(i, j)`.
    // SAFETY (its callers): an entry of this node's slots, which this task
    // alone writes, a section's pieces at disjoint entries.
    let at = |i: usize, j: usize| -> *mut T { unsafe { front.ptr(lp.get(), up.get(), i, j) } };
    // `U12`: the front's entry `(r, t)`, `r` a fully-summed row and `t` past
    // them, at `urect + r * cn_u + t - nfs` of `ut`.
    let urect = front.u.rect();

    // The finished panels of a factored descendant `k`: its pivot count, the
    // rows and columns past its pivots, and its off-block rows of `L` and of
    // `U^T` (column-major, `ol.len()` and `ou.len()` apart).
    let updater = |k: usize| {
        let (nck, ol, ou) = done(k).off(st, k, sym.supernodes[k].ncol);
        // SAFETY: `k` is a factored descendant of `s`; its slots are published
        // and never written again.
        let (lk, uk): (&[T], &[T]) = unsafe { (emit.l_arena.slot(k), emit.u_arena.slot(k)) };
        let (rl, ru) = (
            Layout::new(nck, ol.len()).rect(),
            Layout::new(nck, ou.len()).rect(),
        );
        (nck, ol, ou, &lk[rl..], &uk[ru..])
    };
    {
        // Global to local: `L` rows into `lbuf`, `U` columns into `ut`.
        let gloc = Gloc::new(n, rows);
        let gloc_u = Gloc::second(n, cols);
        // Assemble the columns of s and the U12 rows. The exact structure
        // holds every entry of `B`.
        for p in 0..ncol {
            let c = first + p;
            for (g, v) in inp.col(c) {
                debug_assert!(gloc[g] != Li::MAX, "entry outside the L structure");
                // SAFETY: an entry of the front.
                unsafe {
                    let e = at(gloc[g] as usize, p);
                    *e = *e + v;
                }
            }
            for (g, v) in inp.row(c) {
                debug_assert!(gloc_u[g] != Li::MAX, "entry outside the U structure");
                // SAFETY: an entry of the front.
                unsafe {
                    let e = at(p, gloc_u[g] as usize);
                    *e = *e + v;
                }
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
        let _cmod = rsprobe::scope!(TRACE, "cmod");
        let (spans, forks, tiled) = (&plan.spans, plan.forks, plan.forks);

        // The update in units of the front's layout: ranges of its column
        // blocks of about rspool's pieces per worker of them per worker of
        // the pool ([`unit_share`]), at most the Schur tile wide. Each unit
        // takes every updater's contribution in updater order, one product
        // per updater, over all its rows: an updater's rows are not in the
        // front's order where children pass pivots, so a range of the front's
        // rows is no range of an updater's. Two phases: the units of the
        // front's own columns (the `L` and `U11` updates), then those of
        // `U12`'s columns. The units go to the pool as pieces of their own.
        if tiled {
            let (gloc_ref, gloc_u_ref) = (&gloc, &gloc_u);
            let nb = PANEL_BLOCK;
            let (ncb, nub) = (ncol.div_ceil(nb), cn_u.div_ceil(nb));
            // Each block's multiply-adds.
            let (mut wl, mut wu) = (vec![0.0f64; ncb], vec![0.0f64; nub]);
            for sp in spans {
                let (nck, ol, ou, _, _) = updater(sp.k);
                let ((p0l, p1l), (p0u, p1u)) = (sp.l, sp.u);
                for &g in &ou[p0u..p1u] {
                    wl[(g as usize - first) / nb] += ((ol.len() - p0l) * nck) as f64;
                }
                for &g in &ou[p1u..] {
                    wu[(gloc_u[g as usize] as usize - nfs) / nb] += ((p1l - p0l) * nck) as f64;
                }
            }
            let rate = UPDATE_RATE.of::<T>();
            let work = |w: &[f64]| crate::parallel::Work::of(w.iter().sum(), rate);
            let widest = (kt.k.schur_tile / nb).max(1);
            let units = |w: &[f64]| {
                let share = crate::numeric::supernodal::unit_share(w.iter().sum(), kt.workers);
                crate::numeric::supernodal::cut(w, share, widest, 0)
            };
            let (lunits, uunits) = (units(&wl), units(&wu));
            let lblock = |u: usize| {
                let c0 = lunits[u].0 * nb;
                let c1 = (lunits[u].1 * nb).min(ncol);
                let mut lupd = pools.slab.take();
                for sp in spans {
                    let (nck, ol, ou, lk, uk) = updater(sp.k);
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
                            lk.as_ptr().add(p0l),
                            ol.len() as isize,
                            1,
                            uk.as_ptr().add(q0),
                            1,
                            ou.len() as isize,
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
                    // SAFETY: the columns of the front in this block.
                    unsafe {
                        scatter(
                            &front,
                            (lp.get(), up.get()),
                            &lupd[..mrows * npk],
                            (mrows, npk),
                            |i| gloc_ref[ol[p0l + i] as usize] as usize,
                            |jj| ou[q0 + jj] as usize - first,
                        )
                    };
                }
            };
            crate::parallel::each(lunits.len(), work(&wl), lblock);
            if cn_u > 0 {
                // A block of U12's columns `[u0, u1)`: the front's columns
                // `nfs + u0..nfs + u1`, disjoint between blocks.
                let ublock = |u: usize| {
                    let u0 = uunits[u].0 * nb;
                    let u1 = (uunits[u].1 * nb).min(cn_u);
                    let g0 = off_u[u0];
                    let g1 = if u1 < cn_u { off_u[u1] } else { Li::MAX };
                    let mut uupd = pools.slab.take();
                    for sp in spans {
                        let (nck, ol, ou, lk, uk) = updater(sp.k);
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
                                lk.as_ptr().add(p0l),
                                ol.len() as isize,
                                1,
                                uk.as_ptr().add(t0),
                                1,
                                ou.len() as isize,
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
                                // SAFETY: column `lt` lies in this run.
                                unsafe {
                                    let d = up.get().add(
                                        urect + (ol[p0l + i] as usize - first) * cn_u + lt - nfs,
                                    );
                                    *d = *d - ucol[i];
                                }
                            }
                        }
                    }
                };
                crate::parallel::each(uunits.len(), work(&wu), ublock);
            }
        }

        // Sequential per-update cmod (small nodes / narrow panels).
        for sp in spans.iter().filter(|_| !tiled) {
            let (nck, ol, ou, lk, uk) = updater(sp.k);
            let (ml, mu) = (ol.len(), ou.len());
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
                            acc = acc + lk[p0l + i + ck * ml] * uk[ck * mu + p0u + jj];
                        }
                        // SAFETY: an entry of the front.
                        unsafe {
                            let d = at(gloc[ol[p0l + i] as usize] as usize, tcol);
                            *d = *d - acc;
                        }
                    }
                }
                for jj in 0..ntrail {
                    let tu = gloc_u[ou[p1u + jj] as usize] as usize;
                    for i in 0..npk_l {
                        let mut acc = T::zero();
                        for ck in 0..nck {
                            acc = acc + lk[p0l + i + ck * ml] * uk[ck * mu + p1u + jj];
                        }
                        // SAFETY: an entry of the front.
                        unsafe {
                            let d = at(ol[p0l + i] as usize - first, tu);
                            *d = *d - acc;
                        }
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
                        lk.as_ptr().add(p0l),
                        ml as isize,
                        1,
                        uk.as_ptr().add(p0u),
                        1,
                        mu as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        crate::dense::gemm_backend::GemmMode::new(par(mrows * npk_u * nck), &kt.k),
                    );
                }
                // SAFETY: entries of the front.
                unsafe {
                    scatter(
                        &front,
                        (lp.get(), up.get()),
                        &lupd[..mrows * npk_u],
                        (mrows, npk_u),
                        |i| gloc[ol[p0l + i] as usize] as usize,
                        |jj| ou[p0u + jj] as usize - first,
                    )
                };
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
                        lk.as_ptr().add(p0l),
                        ml as isize,
                        1,
                        uk.as_ptr().add(p1u),
                        1,
                        mu as isize,
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
                        // SAFETY: an entry of the front.
                        unsafe {
                            let d = up
                                .get()
                                .add(urect + (ol[p0l + i] as usize - first) * cn_u + lt - nfs);
                            *d = *d - ucol[i];
                        }
                    }
                }
            }
        }

        // The delayed pivots of the children, as they left them in their
        // records: each delayed column over the child's rows past its
        // pivots, and each delayed row over its columns past the delayed
        // ones. Every update from below is in (descendants reach a delayed
        // index only through the child that delayed it), and every row and
        // column lies in this front (the children's ones past it were added
        // above).
        for &c in &snode.children {
            let pc = done(c);
            let passed = pc.passed();
            if passed == 0 {
                continue;
            }
            let (npc, olc, ouc) = pc.off(st, c, sym.supernodes[c].ncol);
            let (ml, mu) = (olc.len(), ouc.len() - passed);
            for (j, &gc) in pc.cols[npc..].iter().enumerate() {
                let col = gloc_u[gc as usize] as usize;
                let src = &pc.passed_cols[j * ml..(j + 1) * ml];
                for (&g, &v) in olc.iter().zip(src) {
                    debug_assert!(gloc[g as usize] != Li::MAX, "delayed row outside the front");
                    // SAFETY: an entry of the front.
                    unsafe {
                        let d = at(gloc[g as usize] as usize, col);
                        *d = *d + v;
                    }
                }
            }
            for (i, &gr) in pc.rows[npc..].iter().enumerate() {
                let r = gloc[gr as usize] as usize;
                let src = &pc.passed_rows[i * mu..(i + 1) * mu];
                for (&g, &v) in ouc[passed..].iter().zip(src) {
                    debug_assert!(
                        gloc_u[g as usize] != Li::MAX,
                        "delayed column outside the front"
                    );
                    // SAFETY: an entry of the front.
                    unsafe {
                        let d = at(r, gloc_u[g as usize] as usize);
                        *d = *d + v;
                    }
                }
            }
        }
    }

    // The pivots and the emit of the front, timed apart from its updates.
    let _pivot = rsprobe::scope!(TRACE, "pivot");
    // cdiv: in-place **blocked** panel LU, no trailing update outside the
    // panel: a getrf - an NB-wide panel factored column by column, then the
    // dominant trailing update as SIMD GEMMs (rank-NB) - restricted to the
    // front's fully-summed columns and the `U12` rows, with no `A22`/CB.
    // Within an NB panel the columns are brought up to date one at a time
    // from its earlier pivots (left-looking, a GEMV for the off-block rows),
    // so each pivot is decided on the column as it stands, its off-block rows
    // included, while every column still to come keeps the state of the
    // panel's start.
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
    // The off-block rows of `L`: column `j` at `lrect + j * cn_l`.
    let lrect = front.l.rect();
    // Columns `[npiv, live)` are still to pivot, `[live, nfs)` delayed.
    let (mut npiv, mut live) = (0usize, nfs);
    while npiv < live {
        kt.interrupted()?;
        let kb = npiv;
        {
            let _panel = rsprobe::scope!(TRACE, "panel");
            while npiv < live && npiv - kb < nb_cdiv {
                let k = npiv;
                // Column `k` from row `kb` down, brought up to date with this
                // panel's pivots `kb..k`: its fully-summed rows in a copy
                // (`U`'s in `ut`, `L`'s in `lbuf`), its off-block rows in
                // place, or in the copy too where the column may be delayed,
                // so a delayed column keeps the state of the columns still
                // to come.
                col.clear();
                // SAFETY: entries of the front.
                col.extend((kb..k).map(|i| unsafe { *at(i, k) }));
                let dk = front.l.at(k, k);
                col.extend_from_slice(&lbuf[dk..dk + nfs - k]);
                if may_delay {
                    col.extend_from_slice(&lbuf[lrect + k * cn_l..lrect + (k + 1) * cn_l]);
                }
                let (stable, p, best, deep) = {
                    let (ldiag, lrect_s) = lbuf.split_at_mut(lrect);
                    let (before, rest) = lrect_s.split_at_mut(k * cn_l);
                    let (fs, offb) = col.split_at_mut(nfs - kb);
                    let offb = if may_delay { offb } else { &mut rest[..cn_l] };
                    for j in kb..k {
                        let u = fs[j - kb];
                        if u != T::zero() {
                            let l0 = front.l.at(j + 1, j);
                            let lj = &ldiag[l0..l0 + nfs - j - 1];
                            for (v, &l) in fs[j + 1 - kb..].iter_mut().zip(lj) {
                                *v = *v - l * u;
                            }
                        }
                    }
                    let want = thresh_sq > 0.0;
                    let deep = deep_update(before, cn_l, kb, &fs[..k - kb], offb, want, deep_par);
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
                                // SAFETY: entries of the front.
                                unsafe { std::ptr::swap(at(i, k), at(i, live)) };
                            }
                            cols.swap(k, live);
                            continue;
                        }
                        forced += 1;
                    }
                    let reference = if stable { best.max(deep) } else { best };
                    if p != 0 && col[k - kb].magnitude_sq() < thresh_sq * reference {
                        piv_row = k + p;
                    }
                }
                for (i, &v) in (kb..k).zip(col.iter()) {
                    // SAFETY: an entry of the front.
                    unsafe { *at(i, k) = v };
                }
                let dk = front.l.at(k, k);
                lbuf[dk..dk + nfs - k].copy_from_slice(&col[k - kb..nfs - kb]);
                if may_delay {
                    lbuf[lrect + k * cn_l..lrect + (k + 1) * cn_l]
                        .copy_from_slice(&col[nfs - kb..]);
                }
                if piv_row != k {
                    for c in 0..nrow_u {
                        // SAFETY: entries of the front.
                        unsafe { std::ptr::swap(at(k, c), at(piv_row, c)) };
                    }
                    rows.swap(k, piv_row);
                }
                let mut piv = lbuf[dk];
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
                lbuf[dk] = piv;
                let pinv = piv.recip();
                lbuf[dk + 1..dk + nfs - k]
                    .iter_mut()
                    .for_each(|v| *v = *v * pinv);
                deep_scale(
                    &mut lbuf[lrect + k * cn_l..lrect + (k + 1) * cn_l],
                    pinv,
                    deep_par,
                );
                npiv += 1;
            }
        }
        let ke = npiv;
        let pw = ke - kb;
        if pw == 0 {
            continue;
        }
        let _update = rsprobe::scope!(TRACE, "schur");
        // TRSM: the rows `kb..ke` of `U` past the panel, `U[r, t]` for the
        // front's columns `t` in `ke..nrow_u` (the trailing fully-summed ones
        // and U12), `x_r -= L[r, i] x_i` for `i` ascending, on whole rows of
        // `U` (contiguous in `ut`, once in its diagonal part and once in its
        // rectangle), so every entry sees the operations of its own column's
        // substitution in order; parallel over disjoint runs of the columns.
        let trsm = |t0: usize, t1: usize| {
            for r in (kb + 1)..ke {
                for i in kb..r {
                    let l = lbuf[front.l.at(r, i)];
                    // SAFETY: rows `r != i` of `U`, the caller's columns `[t0, t1)`,
                    // all in one part of `ut`.
                    unsafe {
                        let (xr, xi) = (at(r, t0), at(i, t0));
                        for t in 0..t1 - t0 {
                            *xr.add(t) = *xr.add(t) - l * *xi.add(t);
                        }
                    }
                }
            }
        };
        let parts = [(ke, nfs), (nfs.max(ke), nrow_u)];
        if (nrow_u - ke) * pw * pw >= ll_cdiv_par {
            let run = kt.k.trsm_rows;
            let runs: Vec<(usize, usize)> = (parts.iter())
                .flat_map(|&(p0, p1)| (p0..p1).step_by(run).map(move |t| (t, (t + run).min(p1))))
                .collect();
            crate::parallel::for_each(
                runs.len(),
                row_pass::<T>((nrow_u - ke) * pw * pw / 2),
                |q| trsm(runs[q].0, runs[q].1),
            );
        } else {
            for &(p0, p1) in &parts {
                if p1 > p0 {
                    trsm(p0, p1);
                }
            }
        }
        // The trailing update of the front's columns `ke..nfs`: on and below
        // the diagonal in `lbuf`, `L[i, c] -= L[i, kb..ke] U[kb..ke, c]`, and
        // above it in `ut`, `U^T[t, c] -= U^T[t, kb..ke] L[c, kb..ke]^T`
        // (U11 and U12): two lower updates of the step's blocks, `G_l` the
        // `L` rows past the panel and `G_u` the `U^T` rows past it.
        let (mt_l, mt_u, cw) = (nrow_l - ke, nrow_u - ke, nfs - ke);
        if cw > 0 {
            let gather = |g: &mut Vec<T>, buf: &[T], lay: Layout, mt: usize| {
                g.clear();
                g.resize(mt * pw, T::zero());
                for q in 0..pw {
                    for r in lay.runs(ke..ke + mt, kb + q..kb + q + 1) {
                        let (r0, len) = (r.rows.start - ke, r.rows.len());
                        g[r0 + q * mt..r0 + len + q * mt].copy_from_slice(&buf[r.off..r.off + len]);
                    }
                }
            };
            gather(lupd, lbuf, front.l, mt_l);
            gather(uupd, ut, front.u, mt_u);
            let tiles = |mt: usize| {
                if mt * cw * pw >= ll_cdiv_par {
                    (PANEL_BLOCK, PANEL_BLOCK)
                } else {
                    (kt.k.schur_tile, mt)
                }
            };
            // SAFETY: the trailing columns of this node's own slots; the
            // operands are its scratch.
            unsafe {
                schur_tiles(
                    lp.get(),
                    front.l,
                    ke,
                    mt_l,
                    pw,
                    &lupd[..],
                    (&uupd[..], mt_u),
                    (0, cw),
                    tiles(mt_l),
                    mt_l * cw * pw >= ll_cdiv_par,
                    &pools.slab,
                    &kt.k,
                );
                schur_tiles(
                    up.get(),
                    front.u,
                    ke,
                    mt_u,
                    pw,
                    &uupd[..],
                    (&lupd[..], mt_l),
                    (0, cw),
                    tiles(mt_u),
                    mt_u * cw * pw >= ll_cdiv_par,
                    &pools.slab,
                    &kt.k,
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
    // Into the record an earlier factorization left, its buffers kept.
    let own_lists = npiv < nfs || extras;
    // SAFETY: each supernode's record is written by its own task, once,
    // before any reader.
    let pv: &mut NodePivots<T> = unsafe { emit.pivots.get_mut(s) };
    let fill = |v: &mut Vec<Li>, from: &[Li]| {
        v.clear();
        v.extend_from_slice(from);
    };
    pv.npiv = npiv;
    fill(&mut pv.rows, &rows[..nfs]);
    fill(&mut pv.cols, &cols[..nfs]);
    fill(&mut pv.off_l, if own_lists { &rows[npiv..] } else { &[] });
    fill(&mut pv.off_u, if own_lists { &cols[npiv..] } else { &[] });
    pv.extras = extras;
    pv.forced = forced;
    pv.passed_cols.clear();
    pv.passed_rows.clear();
    if npiv < nfs {
        // The passed pivots into the record, then the panels of the pivots
        // taken in the layouts of `npiv` columns, the passed rows and columns
        // the first ones past them, as the ancestors and the emit read them.
        for j in npiv..nfs {
            // SAFETY: entries of the front.
            pv.passed_cols
                .extend((npiv..nrow_l).map(|i| unsafe { *at(i, j) }));
        }
        for i in npiv..nfs {
            // SAFETY: entries of the front.
            pv.passed_rows
                .extend((nfs..nrow_u).map(|t| unsafe { *at(i, t) }));
        }
        let mut tmp = pools.slab.take();
        for (buf, from, m) in [(&mut *lbuf, front.l, nrow_l), (&mut *ut, front.u, nrow_u)] {
            tmp.clear();
            tmp.extend_from_slice(buf);
            relayout(&tmp, from, buf, Layout::new(npiv, m - npiv));
        }
    }
    if !emit.delays {
        // Without delayed pivots every position is the analysis's.
        // SAFETY: this node's own indices.
        unsafe { emit.place(s, emit.e_offset[s]) };
    }
    Ok(())
}

/// The front of an LU node: its `L` panel (laid out as `l`) holds the
/// entries on and below the diagonal, its `U^T` panel (laid out as `u`) the
/// ones above it, transposed. `lb[j]` and `ub[j]` place column `j`'s rows of
/// the fully-summed block: row `i` of `L`'s column `j` at `lb[j] + i`, of
/// `U^T`'s at `ub[j] + i`.
#[derive(Clone, Copy)]
struct Front<'a> {
    l: Layout,
    u: Layout,
    lb: &'a [usize],
    ub: &'a [usize],
}

impl<'a> Front<'a> {
    /// The front laid out as `l` and `u`, its tables in `lb` and `ub`.
    fn new(l: Layout, u: Layout, lb: &'a mut Vec<usize>, ub: &'a mut Vec<usize>) -> Self {
        lb.clear();
        lb.extend((0..l.w).map(|j| l.at(j, j) - j));
        ub.clear();
        ub.extend((0..u.w).map(|j| u.at(j, j) - j));
        Front { l, u, lb, ub }
    }

    /// Entry `(i, j)` of the front in the panels `lp` and `up`.
    ///
    /// # Safety
    /// `lp` and `up` are the panels laid out as `self`.
    #[inline]
    unsafe fn ptr<T>(&self, lp: *mut T, up: *mut T, i: usize, j: usize) -> *mut T {
        let nfs = self.l.w;
        if i >= j {
            if i < nfs {
                lp.add(self.lb[j] + i)
            } else {
                lp.add(self.l.rect() + j * self.l.m + i - nfs)
            }
        } else if j < nfs {
            up.add(self.ub[i] + j)
        } else {
            up.add(self.u.rect() + i * self.u.m + j - nfs)
        }
    }
}

/// `front(row(i), col(j)) -= upd[i + j * m]` for the `m x n` update
/// `upd`, `col(j)` a fully-summed column: column by column, each column's
/// places in the panels taken once.
///
/// # Safety
/// `lp` and `up` are the front's panels; its entries addressed are the
/// caller's.
unsafe fn scatter<T: Scalar>(
    front: &Front<'_>,
    (lp, up): (*mut T, *mut T),
    upd: &[T],
    (m, n): (usize, usize),
    row: impl Fn(usize) -> usize,
    col: impl Fn(usize) -> usize,
) {
    let nfs = front.l.w;
    for j in 0..n {
        let c = col(j);
        debug_assert!(c < nfs);
        // rows `r` of column `c`: `L`'s fully-summed at `ld + r`, its
        // off-block ones at `lr + r`, `U`'s above the diagonal in `ut`
        let (ld, lr) = (front.lb[c], front.l.rect() + c * front.l.m - nfs);
        let u = &upd[j * m..(j + 1) * m];
        for (i, &v) in u.iter().enumerate() {
            let r = row(i);
            let d = if r >= nfs {
                lp.add(lr + r)
            } else if r >= c {
                lp.add(ld + r)
            } else {
                up.add(front.ub[r] + c)
            };
            *d = *d - v;
        }
    }
}

/// The first `to.w` columns of the panel `src` (laid out as `from`, the same
/// rows) laid out as `to` in `dst`.
fn relayout<T: Copy>(src: &[T], from: Layout, dst: &mut [T], to: Layout) {
    debug_assert!(to.w <= from.w && to.w + to.m == from.w + from.m);
    for j in 0..to.w {
        let jb = j / crate::constants::PANEL_BLOCK * crate::constants::PANEL_BLOCK;
        for i in jb..to.w + to.m {
            dst[to.at(i, j)] = src[from.at(i, j)];
        }
    }
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
    // The pivot column, the step's `L` and `U^T` rows past each panel, and
    // the trailing updates' diagonal tiles on up to `workers` threads.
    let tile = PANEL_BLOCK.max(k.schur_tile).min(ncol);
    let cdiv = (nrow_l + (nrow_l + cols_u.len()) * nb + tile * tile * workers.max(1)) * vb + lists;
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
    if plan.forks {
        // a unit at most the Schur tile wide
        let tw = (k.schur_tile / PANEL_BLOCK).max(1) * PANEL_BLOCK;
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
        let tile_u = tw;
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
