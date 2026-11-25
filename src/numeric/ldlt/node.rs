//! One supernode of the left-looking LDL^T factorization: assembly of `A`,
//! the updates of its factored descendants (`cmod`), then the panel
//! factorization.

use super::bunch_kaufman::ll_cdiv_emit;
use super::bunch_kaufman::BkPools;
use super::factor::LlEmitLdlt;
use super::gemm::{grow_scratch, lower_tile_gemm};
use crate::constants::PANEL_BLOCK;
use crate::numeric::supernodal::panel::Layout;
use crate::numeric::supernodal::{Input, Span};

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::LlSchedule;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;

/// What a multiply-add of the descendants' updates costs: the update of a large
/// node goes as wide as its work is worth.
static UPDATE_RATE: crate::parallel::ScalarRates =
    crate::parallel::ScalarRates::new(crate::constants::GEMM_NS_PER_FLOP);

/// The front's row of the global index `g`.
#[inline]
fn local_of(
    gloc: &crate::numeric::supernodal::Gloc<'_>,
    g: crate::numeric::supernodal::Li,
) -> usize {
    gloc[g as usize] as usize
}

/// Factor one supernode's panel: assemble `A`, apply every descendant's `cmod`
/// update (BLAS-3 with scalar fallback), then `cdiv` (partial 1x1 LDL^T). Reads
/// only already-factored descendant panels and their `D` from the emit state,
/// so sibling subtrees run concurrently. Writes the factored panel and `D`
/// there.
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

    // Global-to-local rows (narrow entries halve the table's random-access
    // footprint); restored when the node returns.
    let gloc = crate::numeric::supernodal::Gloc::new(n, sched.rows(s));
    // Assemble A's lower-triangle columns of this supernode.
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
            // The update in units of the panel's layout: ranges of its column
            // blocks, each with its rows from its diagonal down or, where a
            // range alone holds more than a unit's share, cut into ranges of
            // row blocks too, about rspool's pieces per worker of them per
            // worker of the pool ([`unit_share`]), a range at most the Schur
            // tile wide. Each unit takes every updater's contribution in
            // updater order, one product per updater, so its sums depend on
            // the node and the thread count alone; the units go to the pool
            // as pieces of their own.
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
            // Per column block the updaters landing in it, in updater order,
            // and their landing rows `[q0, q1)` there; the multiply-adds of
            // every column block and row block.
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
            // stable: each column block's updaters stay in updater order
            landing.sort_by_key(|e| e.0);
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
            // The units `(ca, ce, rb0, rb1)`: column blocks `ca..ce`, row blocks `rb0..rb1`.
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
                // The updaters landing in the unit's columns, each once, in
                // updater order, with their landing rows there.
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
                    // The updater's off-block rows, column `ck` at `rk + ck * nok`.
                    let rk = Layout::new(nck, nok).rect();
                    let npk = q1 - q0;
                    // Its rows in the unit's row blocks (from its landing rows on).
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
                    // G = (kk's rows q0..q1) * D, column-major npk x nck.
                    grow_scratch(vd_buf, npk * nck);
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
                    grow_scratch(u_buf, mrows * npk);
                    // SAFETY: lhs (read), rhs (read), dst (write) pairwise
                    // disjoint; strides in bounds.
                    unsafe {
                        if diag {
                            // From the landing rows down: on and below the
                            // diagonal only.
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

        // Sequential per-update cmod (small nodes / small total update work).
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
            // Bunch-Kaufman block structure of `kk`'s D (pivoted column order). The
            // cmod `L*D*L^T` is invariant under `kk`'s internal column permutation, so
            // only the block-diagonal `D`-apply has to honor the 2x2 blocks.
            let npk = p1 - p0;
            // Gate on the REAL work (rows >= p0); the scalar path already
            // iterates from the target block, so small tails route there.
            if (nok - p0) * npk * nck < ll_gemm_gate {
                grow_scratch(vc, nck);
                for c_idx in p0..p1 {
                    let tcol = ok[c_idx] as usize - first;
                    // vc = D * (column `c_idx` of kk's off-diagonal block), with D
                    // block-diagonal (1x1 and complex-symmetric 2x2 blocks).
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
                grow_scratch(vd_buf, npk * nck);
                // G = (kk's in-panel off-diagonal block) * D, stored column-major as
                // `vd_buf[c + ck*npk]`. D is block-diagonal (1x1 and 2x2 blocks); a
                // 2x2 block mixes its two columns. GEMM below is unchanged.
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
                // Only rows >= p0 land in (or below) the target block: computing
                // the full `nok`-tall product and discarding rows `< p0` in the
                // write-back wasted `p0*npk*nck` flops per update - large for
                // updates into high supernodes, where most of the updater's
                // off-diagonal rows lie above the target. Mirror the LU twin:
                // offset the lhs by `p0` and compute `mrows = nok - p0` rows.
                // The write-back below also reads only rows `>= c` per column
                // (the symmetric lower part), so the product is computed
                // tile-wise from each tile's diagonal downward - the same
                // `lower_tile_gemm` that serves the panel Schur updates. For
                // updates into the topmost supernodes (`mrows ~ npk`) the full
                // rectangle wasted another ~half of the flops.
                let mrows = nok - p0;
                grow_scratch(u_buf, mrows * npk);
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

/// Heap bytes [`ll_factor_node`] allocates for supernode `s` while it runs,
/// for the memory plan: the tiled cmod's per-slab buffers on up to
/// `workers` threads at once, or the sequential cmod's buffers, which stay
/// allocated through the cdiv ([`ll_cdiv_scratch`](super::bunch_kaufman::ll_cdiv_scratch));
/// the largest split planes its GEMMs take on a worker (complex fields)
/// and how many workers may take them at once. Replays the kernel's plan
/// and growth of its buffers.
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
        // A unit's buffers per worker: `L D` of an updater over the unit's
        // columns (at most the Schur tile), its product over the rows below.
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
