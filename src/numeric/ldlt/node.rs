//! One supernode of the left-looking LDL^T factorization: assembly of `A`,
//! the updates of its factored descendants (`cmod`), then the panel
//! factorization.

use super::bunch_kaufman::ll_cdiv_emit;
use super::bunch_kaufman::BkPools;
use super::factor::LlEmitLdlt;
use super::gemm::lower_tile_gemm;
use crate::constants::PANEL_BLOCK;
use crate::numeric::supernodal::panel::Layout;
use crate::numeric::supernodal::{Input, Span};

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::LlSchedule;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;

static UPDATE_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::GEMM_NS_PER_FLOP);

#[inline]
fn local_of(
    gloc: &crate::numeric::supernodal::Gloc<'_>,
    g: crate::numeric::supernodal::Li,
) -> usize {
    gloc[g as usize] as usize
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    inp: Input<T>,
    sched: &LlSchedule,
    pools: &BkPools<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    let ll_gemm_gate = kt.k.scalar_gate;
    let ll_gemm_par = kt.k.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(s).len();
    let n = sym.n;
    // SAFETY: this task owns supernode `s`; nobody reads its slot before the
    // node is done (its ancestors start after it).
    let panel: &mut [T] = unsafe { emit.arena.slot_mut(s) };
    let lay = Layout::new(ncol, nrow - ncol);
    debug_assert_eq!(panel.len(), lay.len());

    // https://github.com/milanofthe/rslab/issues/288
    let gloc = crate::numeric::supernodal::Gloc::new(n, sched.rows(s));
    for p in 0..ncol {
        let c = first + p;
        for (g, v) in inp.col(c) {
            let at = lay.at(gloc[g] as usize, p);
            panel[at] = panel[at] + v;
        }
    }
    let plan = crate::numeric::supernodal::CmodPlan::new(
        sym,
        s,
        sched.updaters(s),
        |k| {
            let nck = sym.supernodes[k].ncol;
            (nck, &sched.rows(k)[nck..], &sched.rows(k)[nck..])
        },
        false,
        &kt.k,
    );
    {
        let _cmod = rsprobe::scope!(TRACE, "cmod");
        let (spans, tiled) = (&plan.spans, plan.forks);
        let seq_gemm_par = if plan.forks { ll_gemm_par } else { usize::MAX };
        if tiled {
            let nb = PANEL_BLOCK;
            let ncb = ncol.div_ceil(nb);
            let nrb = ncb + (nrow - ncol).div_ceil(nb);
            let rows_s = sched.rows(s);
            let row_block = |loc: usize| {
                if loc < ncol {
                    loc / nb
                } else {
                    ncb + (loc - ncol) / nb
                }
            };
            let block_rows = |rb: usize| {
                if rb < ncb {
                    (rb * nb, ((rb + 1) * nb).min(ncol))
                } else {
                    let r = ncol + (rb - ncb) * nb;
                    (r, (r + nb).min(nrow))
                }
            };
            let mut weight = vec![0.0f64; ncb * nrb];
            let mut landing: Vec<(usize, u32, u32, u32)> = Vec::new();
            for (
                si,
                &Span {
                    k: kk, l: (p0, p1), ..
                },
            ) in spans.iter().enumerate()
            {
                let nck = sym.supernodes[kk].ncol;
                let ok = &sched.rows(kk)[nck..];
                let local = |r: usize| local_of(&gloc, ok[r]);
                let mut c = p0;
                while c < p1 {
                    let cb = local(c) / nb;
                    let ce = c + ok[c..p1].partition_point(|&g| local_of(&gloc, g) / nb == cb);
                    landing.push((cb, si as u32, c as u32, ce as u32));
                    let mut r = c;
                    while r < ok.len() {
                        let rb = row_block(local(r));
                        let re =
                            r + ok[r..].partition_point(|&g| row_block(local_of(&gloc, g)) == rb);
                        weight[cb * nrb + rb] += ((ce - c) * (re - r) * nck) as f64;
                        r = re;
                    }
                    c = ce;
                }
            }
            landing.sort_by_key(|e| e.0);
            debug_assert!(landing
                .windows(2)
                .all(|w| w[0].0 < w[1].0 || w[0].1 <= w[1].1));
            let mut by_cb = vec![0usize; ncb + 1];
            for e in &landing {
                by_cb[e.0 + 1] += 1;
            }
            for cb in 0..ncb {
                by_cb[cb + 1] += by_cb[cb];
            }
            let flops: f64 = weight.iter().sum();
            let share = crate::numeric::supernodal::unit_share(flops, kt.workers);
            let col_w: Vec<f64> = (0..ncb)
                .map(|cb| weight[cb * nrb..(cb + 1) * nrb].iter().sum())
                .collect();
            let widest = (kt.k.schur_tile / nb).max(1);
            let mut units: Vec<(usize, usize, usize, usize)> = Vec::new();
            for (ca, ce) in crate::numeric::supernodal::cut(&col_w, share, widest, 0) {
                let row_w: Vec<f64> = (ca..nrb)
                    .map(|rb| (ca..ce).map(|cb| weight[cb * nrb + rb]).sum())
                    .collect();
                for (r0, r1) in crate::numeric::supernodal::cut(&row_w, share, usize::MAX, ce - ca)
                {
                    units.push((ca, ce, ca + r0, ca + r1));
                }
            }
            let work = crate::parallel::Work::of(flops, UPDATE_RATE.of::<T>());
            let gloc_ref = &gloc;
            let base = crate::numeric::supernodal::PanelPtr(panel.as_mut_ptr());
            let unit = |u: usize| {
                let (ca, ce, rb0, rb1) = units[u];
                let mut bufs = pools.cmod.take();
                let (_, vd_buf, u_buf) = &mut *bufs;
                let mut ups: Vec<(u32, u32, u32)> = landing[by_cb[ca]..by_cb[ce]]
                    .iter()
                    .map(|e| (e.1, e.2, e.3))
                    .collect();
                ups.sort_by_key(|e| e.0);
                ups.dedup_by(|b, a| {
                    let same = a.0 == b.0;
                    if same {
                        (a.1, a.2) = (a.1.min(b.1), a.2.max(b.2));
                    }
                    same
                });
                let (r0, r1) = (block_rows(rb0).0, block_rows(rb1 - 1).1);
                let diag = rb0 == ca;
                for &(si, q0, q1) in &ups {
                    let kk = spans[si as usize].k;
                    let (q0, q1) = (q0 as usize, q1 as usize);
                    let nck = sym.supernodes[kk].ncol;
                    let ok = &sched.rows(kk)[nck..];
                    let nok = ok.len();
                    let rk = Layout::new(nck, nok).rect();
                    let npk = q1 - q0;
                    let below = |r: usize| q0 + ok[q0..].partition_point(|&g| g < rows_s[r]);
                    let a = if diag { q0 } else { below(r0) };
                    let b = if r1 == nrow { nok } else { below(r1) };
                    if b <= a {
                        continue;
                    }
                    // SAFETY: `kk` is a factored descendant of `s`, its cells
                    // are written and never mutated again.
                    let pk: &[T] = unsafe { emit.arena.slot(kk) };
                    let (dk, dsub_k, two_k) = unsafe { emit.d_of(kk, nck) };
                    crate::memory::grow(vd_buf, npk * nck);
                    let mut ck = 0;
                    while ck < nck {
                        if two_k[ck] {
                            let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                            for i in 0..npk {
                                let x = pk[rk + q0 + i + ck * nok];
                                let y = pk[rk + q0 + i + (ck + 1) * nok];
                                vd_buf[i + ck * npk] = d11 * x + d21 * y;
                                vd_buf[i + (ck + 1) * npk] = d21 * x + d22 * y;
                            }
                            ck += 2;
                        } else {
                            let dkc = dk[ck];
                            for i in 0..npk {
                                vd_buf[i + ck * npk] = pk[rk + q0 + i + ck * nok] * dkc;
                            }
                            ck += 1;
                        }
                    }
                    let mrows = b - a;
                    crate::memory::grow(u_buf, mrows * npk);
                    // SAFETY: lhs (read), rhs (read), dst (write) pairwise
                    // disjoint; strides in bounds.
                    unsafe {
                        if diag {
                            lower_tile_gemm(
                                &mut *u_buf,
                                mrows,
                                npk,
                                nck,
                                pk.as_ptr().add(rk + a),
                                nok as isize,
                                vd_buf.as_ptr(),
                                npk as isize,
                                usize::MAX,
                                &kt.k,
                            )
                        } else {
                            crate::dense::gemm_backend::gemm(
                                mrows,
                                npk,
                                nck,
                                u_buf.as_mut_ptr(),
                                mrows as isize,
                                1,
                                false,
                                pk.as_ptr().add(rk + a),
                                nok as isize,
                                1,
                                vd_buf.as_ptr(),
                                1,
                                npk as isize,
                                T::zero(),
                                T::one(),
                                false,
                                false,
                                false,
                                crate::dense::gemm_backend::GemmMode::new(
                                    gemm::Parallelism::None,
                                    &kt.k,
                                ),
                            )
                        }
                    };
                    for c in 0..npk {
                        let tcol = ok[q0 + c] as usize - first;
                        let (ucol, col) = (&u_buf[c * mrows..c * mrows + mrows], lay.col(tcol));
                        for r in a.max(q0 + c)..b {
                            // SAFETY: an entry of this unit's columns and rows,
                            // on or below the diagonal.
                            unsafe {
                                let dst = base.get().add(col.at(gloc_ref[ok[r] as usize] as usize));
                                *dst = *dst - ucol[r - a];
                            }
                        }
                    }
                }
            };
            crate::parallel::each(units.len(), work, unit);
        }

        let mut bufs = pools.cmod.take();
        let (vc, vd_buf, u_buf) = &mut *bufs;
        for &Span {
            k: kk, l: (p0, p1), ..
        } in spans.iter().filter(|_| !tiled)
        {
            let nck = sym.supernodes[kk].ncol;
            let ok = &sched.rows(kk)[nck..];
            let nok = ok.len();
            let rk = Layout::new(nck, nok).rect();
            // SAFETY: `kk` is a factored descendant of `s` (its update reaches `s`),
            // so its panel/dval cells are written and never mutated again.
            let pk: &[T] = unsafe { emit.arena.slot(kk) };
            let (dk, dsub_k, two_k) = unsafe { emit.d_of(kk, nck) };
            let npk = p1 - p0;
            if (nok - p0) * npk * nck < ll_gemm_gate {
                crate::memory::grow(vc, nck);
                for c_idx in p0..p1 {
                    let tcol = ok[c_idx] as usize - first;
                    let mut ck = 0;
                    while ck < nck {
                        let a = pk[rk + c_idx + ck * nok];
                        if two_k[ck] {
                            let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                            let b = pk[rk + c_idx + (ck + 1) * nok];
                            vc[ck] = d11 * a + d21 * b;
                            vc[ck + 1] = d21 * a + d22 * b;
                            ck += 2;
                        } else {
                            vc[ck] = dk[ck] * a;
                            ck += 1;
                        }
                    }
                    for r_idx in c_idx..nok {
                        let trow = gloc[ok[r_idx] as usize] as usize;
                        let mut acc = T::zero();
                        for ck in 0..nck {
                            acc = acc + pk[rk + r_idx + ck * nok] * vc[ck];
                        }
                        let at = lay.at(trow, tcol);
                        panel[at] = panel[at] - acc;
                    }
                }
            } else {
                crate::memory::grow(vd_buf, npk * nck);
                // https://github.com/milanofthe/rslab/issues/288
                let mut ck = 0;
                while ck < nck {
                    if two_k[ck] {
                        let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                        for i in 0..npk {
                            let a = pk[rk + p0 + i + ck * nok];
                            let b = pk[rk + p0 + i + (ck + 1) * nok];
                            vd_buf[i + ck * npk] = d11 * a + d21 * b;
                            vd_buf[i + (ck + 1) * npk] = d21 * a + d22 * b;
                        }
                        ck += 2;
                    } else {
                        let dkc = dk[ck];
                        for i in 0..npk {
                            vd_buf[i + ck * npk] = pk[rk + p0 + i + ck * nok] * dkc;
                        }
                        ck += 1;
                    }
                }
                // https://github.com/milanofthe/rslab/issues/288
                let mrows = nok - p0;
                crate::memory::grow(u_buf, mrows * npk);
                // SAFETY: lhs (`pk` off-diag block from row p0, read), rhs
                // (`vd_buf`, read), dst (`u_buf`, write) are pairwise-disjoint;
                // strides in bounds.
                unsafe {
                    lower_tile_gemm(
                        &mut *u_buf,
                        mrows,
                        npk,
                        nck,
                        pk.as_ptr().add(rk + p0),
                        nok as isize,
                        vd_buf.as_ptr(),
                        npk as isize,
                        seq_gemm_par,
                        &kt.k,
                    )
                };
                for c in 0..npk {
                    let tcol = ok[p0 + c] as usize - first;
                    let (ucol, col) = (&u_buf[c * mrows..c * mrows + mrows], lay.col(tcol));
                    for r in (p0 + c)..nok {
                        let dst = col.at(gloc[ok[r] as usize] as usize);
                        panel[dst] = panel[dst] - ucol[r - p0];
                    }
                }
            }
        }
    }
    let _pivot = rsprobe::scope!(TRACE, "pivot");
    ll_cdiv_emit(s, sym, sched, pools, emit, perturb_floor, kt, panel)
}

pub(super) fn ll_node_scratch<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    k: &crate::KernelSettings,
    workers: usize,
) -> (u64, usize, usize) {
    use super::gemm::lower_tile_planes;
    use crate::memory::grown;
    let vb = std::mem::size_of::<T>();
    let ncol = sym.supernodes[s].ncol;
    let (cdiv, mut planes, cdiv_copies) =
        super::bunch_kaufman::ll_cdiv_scratch::<T>(ncol, sched.rows(s).len(), k, workers);
    let plan = crate::numeric::supernodal::CmodPlan::new(
        sym,
        s,
        sched.updaters(s),
        |k| {
            let nck = sym.supernodes[k].ncol;
            (nck, &sched.rows(k)[nck..], &sched.rows(k)[nck..])
        },
        false,
        k,
    );
    let updater = |kk: usize| {
        let nck = sym.supernodes[kk].ncol;
        (nck, &sched.rows(kk)[nck..])
    };
    if plan.forks {
        let nb = ((k.schur_tile / PANEL_BLOCK).max(1) * PANEL_BLOCK).min(ncol.max(1));
        let (mut vd, mut u, mut slab_planes) = (0, 0, 0);
        for sp in &plan.spans {
            let (nck, ok) = updater(sp.k);
            let rows = ok.len() - sp.l.0;
            vd = grown(vd, nb * nck);
            u = grown(u, rows * nb);
            if T::COMPLEX {
                slab_planes = slab_planes.max(lower_tile_planes(rows, nb, nck, usize::MAX, k));
            }
        }
        let at_once = workers.max(1);
        let copies = if slab_planes > 0 {
            at_once.max(cdiv_copies)
        } else {
            cdiv_copies
        };
        let bytes = (((vd + u) * at_once * vb) as u64).max(cdiv);
        (bytes, planes.max(slab_planes), copies)
    } else {
        let (mut vc, mut vd, mut u) = (0, 0, 0);
        for sp in &plan.spans {
            let (nck, ok) = updater(sp.k);
            let (p0, p1) = sp.l;
            let npk = p1 - p0;
            if (ok.len() - p0) * npk * nck < k.scalar_gate {
                vc = grown(vc, nck);
            } else {
                vd = grown(vd, npk * nck);
                u = grown(u, (ok.len() - p0) * npk);
                if T::COMPLEX {
                    let par = if plan.forks { k.par_gemm } else { usize::MAX };
                    planes = planes.max(lower_tile_planes(ok.len() - p0, npk, nck, par, k));
                }
            }
        }
        (((vc + vd + u) * vb) as u64 + cdiv, planes, cdiv_copies)
    }
}
