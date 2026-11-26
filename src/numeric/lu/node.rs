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

pub(super) struct LuPools<T> {
    pub(super) node: ScratchPool<LuScratch<T>>,
    pub(super) slab: ScratchPool<Vec<T>>,
    pub(super) finish: ScratchPool<PanelScratch<T>>,
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

static UPDATE_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::GEMM_NS_PER_FLOP);

static ROW_PASS_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::SCAN_NS_PER_ENTRY);

fn row_pass<T: Scalar>(ops: usize) -> crate::parallel::Work<'static> {
    crate::parallel::Work::of(ops as f64, ROW_PASS_RATE.of::<T>())
}

pub(super) fn deep_update<T: Scalar>(
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
        let top = AtomicU64::new(0);
        let work = row_pass::<T>(x.len() * u.len().max(1));
        crate::parallel::chunks_mut::<crate::parallel::Rayon, _>(x, run, work, |i, xs| {
            let top_of_run = apply(i * run, xs);
            debug_assert!(top_of_run.is_sign_positive());
            top.fetch_max(top_of_run.to_bits(), Ordering::Relaxed);
        });
        f64::from_bits(top.into_inner())
    } else {
        apply(0, x)
    }
}

pub(super) fn deep_scale<T: Scalar>(x: &mut [T], pinv: T, par: bool) {
    if par {
        let work = row_pass::<T>(x.len());
        crate::parallel::chunks_mut::<crate::parallel::Rayon, _>(x, 1, work, |_, xs| {
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
    let front = Front::new(Layout::new(nfs, cn_l), Layout::new(nfs, cn_u), lb, ub);
    // SAFETY: this task owns supernode `s`; nobody reads its slots before the
    // node is done (its ancestors start after it).
    let (lbuf, ut): (&mut [T], &mut [T]) = unsafe {
        let rows_past_pivots = |nrow: usize| if emit.delays { nrow } else { nrow - nfs };
        match (
            emit.l_arena
                .claim(s, front.l.len(), rows_past_pivots(nrow_l)),
            emit.u_arena
                .claim(s, front.u.len(), rows_past_pivots(nrow_u)),
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
    // SAFETY: `(i, j)` is an entry of this node's slots, which this task
    // alone writes, the pieces of a section at disjoint entries.
    let at = |i: usize, j: usize| -> *mut T { unsafe { front.ptr(lp.get(), up.get(), i, j) } };
    let urect = front.u.rect();

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
        let gloc = Gloc::new(n, rows);
        let gloc_u = Gloc::second(n, cols);
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
        // https://github.com/milanofthe/rslab/issues/288
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

        if tiled {
            let (gloc_ref, gloc_u_ref) = (&gloc, &gloc_u);
            let nb = PANEL_BLOCK;
            let (ncb, nub) = (ncol.div_ceil(nb), cn_u.div_ceil(nb));
            let (mut wl, mut wu) = (vec![0.0f64; ncb], vec![0.0f64; nub]);
            for sp in spans {
                let (nck, ol, ou, _, _) = updater(sp.k);
                let ((p0l, p1l), (p0u, p1u)) = (sp.l, sp.u);
                for &g in &ou[p0u..p1u] {
                    wl[(g as usize - first) / nb] += ((ol.len() - p0l) * nck) as f64;
                }
                if p1l > p0l {
                    for &g in &ou[p1u..] {
                        debug_assert!(
                            gloc_u[g as usize] != Li::MAX,
                            "U12 column outside the front"
                        );
                        wu[(gloc_u[g as usize] as usize - nfs) / nb] += ((p1l - p0l) * nck) as f64;
                    }
                }
            }
            let rate = UPDATE_RATE.of::<T>();
            let work = |w: &[f64]| crate::parallel::Work::of(w.iter().sum(), rate);
            let widest = (kt.k.schur_tile / nb).max(1);
            // https://github.com/milanofthe/rslab/issues/288
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
                            debug_assert!((nfs + u0..nfs + u1).contains(&lt));
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

        for sp in spans.iter().filter(|_| !tiled) {
            let (nck, ol, ou, lk, uk) = updater(sp.k);
            let (ml, mu) = (ol.len(), ou.len());
            let ((p0l, p1l), (p0u, p1u)) = (sp.l, sp.u);
            let (mrows, npk_u) = (ol.len() - p0l, p1u - p0u);
            let (npk_l, ntrail) = (p1l - p0l, ou.len() - p1u);
            if (mrows * npk_u + npk_l * ntrail) * nck < ll_gemm_gate {
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
            let par = |work: usize| {
                if forks && work >= ll_gemm_par {
                    gemm::Parallelism::Rayon(0)
                } else {
                    gemm::Parallelism::None
                }
            };
            if mrows > 0 && npk_u > 0 {
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

    let _pivot = rsprobe::scope!(TRACE, "pivot");
    let nb_cdiv = kt.k.panel_width(kt.k.lu_panel_nb, nfs);
    let ll_cdiv_par = kt.k.cdiv_par(nrow_l, nfs);
    let deep_par = cn_l * nb_cdiv * nb_cdiv >= ll_cdiv_par;
    let thresh_sq = kt.pivot_threshold * kt.pivot_threshold;
    let may_delay = emit.delays && sched.parent[s] != usize::MAX;
    let (mut forced, mut floors) = (0usize, Vec::<(usize, T)>::new());
    let lrect = front.l.rect();
    let (mut npiv, mut live) = (0usize, nfs);
    while npiv < live {
        debug_assert!(live <= nfs);
        kt.interrupted()?;
        let kb = npiv;
        {
            let _panel = rsprobe::scope!(TRACE, "panel");
            // https://github.com/milanofthe/rslab/issues/288
            while npiv < live && npiv - kb < nb_cdiv {
                let k = npiv;
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
                    // https://github.com/milanofthe/rslab/issues/288
                    let want = thresh_sq > 0.0;
                    let deep = deep_update(before, cn_l, kb, &fs[..k - kb], offb, want, deep_par);
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
                    (best >= thresh_sq * deep, p, best, deep)
                };
                let mut piv_row = k;
                if thresh_sq > 0.0 {
                    if !stable {
                        if may_delay {
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
            crate::parallel::ranges::<crate::parallel::Rayon>(
                runs.len(),
                row_pass::<T>((nrow_u - ke) * pw * pw / 2),
                |r| r.for_each(|q| trsm(runs[q].0, runs[q].1)),
            );
        } else {
            for &(p0, p1) in &parts {
                if p1 > p0 {
                    trsm(p0, p1);
                }
            }
        }
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
        // SAFETY: this node's own indices.
        unsafe { emit.place(s, emit.e_offset[s]) };
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Front<'a> {
    l: Layout,
    u: Layout,
    lb: &'a [usize],
    ub: &'a [usize],
}

impl<'a> Front<'a> {
    fn new(l: Layout, u: Layout, lb: &'a mut Vec<usize>, ub: &'a mut Vec<usize>) -> Self {
        lb.clear();
        lb.extend((0..l.w).map(|j| l.at(j, j) - j));
        ub.clear();
        ub.extend((0..u.w).map(|j| u.at(j, j) - j));
        Front { l, u, lb, ub }
    }

    #[inline]
    unsafe fn ptr<T>(&self, lp: *mut T, up: *mut T, i: usize, j: usize) -> *mut T {
        let nfs = self.l.w;
        if i >= j {
            if i < nfs {
                debug_assert_eq!(self.lb[j] + i, self.l.at(i, j));
                lp.add(self.lb[j] + i)
            } else {
                lp.add(self.l.rect() + j * self.l.m + i - nfs)
            }
        } else if j < nfs {
            debug_assert_eq!(self.ub[i] + j, self.u.at(j, i));
            up.add(self.ub[i] + j)
        } else {
            up.add(self.u.rect() + i * self.u.m + j - nfs)
        }
    }
}

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

fn relayout<T: Copy>(src: &[T], from: Layout, dst: &mut [T], to: Layout) {
    debug_assert!(to.w <= from.w && to.w + to.m == from.w + from.m);
    for j in 0..to.w {
        let jb = j / crate::constants::PANEL_BLOCK * crate::constants::PANEL_BLOCK;
        for i in jb..to.w + to.m {
            dst[to.at(i, j)] = src[from.at(i, j)];
        }
    }
}

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
    let tile = PANEL_BLOCK.max(k.schur_tile).min(ncol);
    let cdiv = (nrow_l + (nrow_l + cols_u.len()) * nb + tile * tile * workers.max(1)) * vb + lists;
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
