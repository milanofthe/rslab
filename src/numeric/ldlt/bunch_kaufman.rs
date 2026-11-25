//! The dense panel factorization of the LDL^T kernel (`cdiv`): blocked
//! Bunch-Kaufman with 1x1 and 2x2 pivots bounded to a supernode's
//! fully-summed block, the row-parallel replay of each panel's transform on
//! the deep rows, and the deferred trailing update with panel lookahead.

use super::factor::LlEmitLdlt;
use crate::constants::{GROWTH_EPS, PANEL_BLOCK};
use crate::numeric::supernodal::perturb_pivot;
use crate::numeric::supernodal::schur_tiles;

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::panel::Layout;
use crate::numeric::supernodal::LlSchedule;
use crate::numeric::supernodal::PanelPtr as LdltPanelPtr;
use crate::numeric::supernodal::ScratchPool;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::Ordering;

/// One running [`ll_cdiv_emit`]'s buffers (see there).
pub(super) struct BkScratch<T> {
    d: Vec<T>,
    d_subdiag: Vec<T>,
    two_by_two: Vec<bool>,
    lperm: Vec<usize>,
    l1: Vec<T>,
    l2: Vec<T>,
    l21buf: Vec<T>,
    gbuf: Vec<T>,
    deep_swaps: Vec<usize>,
    mult_snap: Vec<T>,
    l1b: Vec<T>,
    l2b: Vec<T>,
    deep_swaps_b: Vec<usize>,
    mult_snap_b: Vec<T>,
}

impl<T> crate::memory::HeapBytes for BkScratch<T> {
    fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes as b;
        b(&self.d)
            + b(&self.d_subdiag)
            + b(&self.two_by_two)
            + b(&self.lperm)
            + b(&self.l1)
            + b(&self.l2)
            + b(&self.l21buf)
            + b(&self.gbuf)
            + b(&self.deep_swaps)
            + b(&self.mult_snap)
            + b(&self.l1b)
            + b(&self.l2b)
            + b(&self.deep_swaps_b)
            + b(&self.mult_snap_b)
    }
}

impl<T> Default for BkScratch<T> {
    fn default() -> Self {
        BkScratch {
            d: Vec::new(),
            d_subdiag: Vec::new(),
            two_by_two: Vec::new(),
            lperm: Vec::new(),
            l1: Vec::new(),
            l2: Vec::new(),
            l21buf: Vec::new(),
            gbuf: Vec::new(),
            deep_swaps: Vec::new(),
            mult_snap: Vec::new(),
            l1b: Vec::new(),
            l2b: Vec::new(),
            deep_swaps_b: Vec::new(),
            mult_snap_b: Vec::new(),
        }
    }
}

/// The scratch pools of one LDL^T factorization: the panel kernel's buffers
/// per running node, the deep-row replay's per row chunk, and the `cmod`
/// update's (`D` times a column, `D` times a block, the product) per node
/// or column slab.
pub(super) struct BkPools<T> {
    pub(super) node: ScratchPool<BkScratch<T>>,
    pub(super) w: ScratchPool<Vec<T>>,
    pub(super) cmod: ScratchPool<(Vec<T>, Vec<T>, Vec<T>)>,
    /// The buffers of the emit's [`finish_panel`](crate::numeric::supernodal::panel::finish_panel).
    pub(super) finish: ScratchPool<crate::numeric::supernodal::panel::PanelScratch<T>>,
}

/// A copy of a factor starts with scratch of its own, grown by its first
/// refactorization.
impl<T> Clone for BkPools<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> BkPools<T> {
    pub(super) fn new() -> Self {
        BkPools {
            node: ScratchPool::new(),
            w: ScratchPool::new(),
            cmod: ScratchPool::new(),
            finish: ScratchPool::new(),
        }
    }
}

/// `v` as `len` copies of `x`, in its storage.
fn reset<V: Clone>(v: &mut Vec<V>, len: usize, x: V) {
    v.clear();
    v.resize(len, x);
}

/// Apply a factored Bunch-Kaufman panel's transform sequence to rows
/// `[r0, r1)` of the panel `base` (laid out as `lay`; the rows all in its
/// diagonal part or all in its off-block rectangle), for pivot steps
/// `[kb, ke)`. Bit-identical to the corresponding rows of the full-height
/// panel factorization: per 1x1 step the in-panel updates use the **final**
/// column-`k` multipliers (`w_j*d^-1` is exactly the stored `L(j,k)`), then
/// the column is scaled by `d^-1`; per 2x2 step the multiplier pair is
/// rebuilt from the (already perturbed) stored `D` block with the same
/// expressions and order. Deep rows are never pivot candidates, so each
/// caller's row range is independent - the lever that lifts the dominant
/// `O((nrow-ke)*pw^2)` panel work off the serial getf2 path onto all idle
/// workers (ports the LU twin's `apply_panel_trailing` to Bunch-Kaufman).
///
/// `deep_swaps[k - kb]` records the pivot interchange partner of step `k`
/// (`usize::MAX` when the step did not interchange): getf2 bounds its swaps
/// to the panel rows, so the deep-row segments of each interchange are
/// replayed here, immediately before the step's transform - the original
/// full-height order, row by row.
///
/// `mult_snap` holds the in-panel multipliers **as of each step's time**
/// (`mult_snap[(k - kb)*nb + (j - kb)]` is step `k`'s coefficient for
/// in-panel row `j`). Reading them from the final panel would be wrong:
/// later symmetric interchanges permute the rows of earlier multiplier
/// columns (unlike LU, where produced pivot rows never move again).
///
/// SAFETY: `[r0, r1)` must be this caller's exclusive rows of the panel
/// `base`, at or below row `ke`.
#[allow(clippy::too_many_arguments)]
unsafe fn apply_bk_panel_trailing<T: Scalar>(
    base: *mut T,
    lay: Layout,
    kb: usize,
    ke: usize,
    d_diag: &[T],
    d_subdiag: &[T],
    two_by_two: &[bool],
    deep_swaps: &[usize],
    mult_snap: &[T],
    nb: usize,
    r0: usize,
    r1: usize,
    sb: usize,
    w: &mut Vec<T>,
) {
    // Blocked form of the per-pivot sweep. Pivots are taken in sub-blocks of
    // `sb` columns ([`KernelSettings::trailing_block`](crate::KernelSettings::trailing_block);
    // 16 measured best: 32 doubles the scalar within-block work, 8 halves
    // the GEMM efficiency): inside a sub-block a pivot's rank-1 (rank-2)
    // update reaches only the sub-block's remaining columns (scalar loops),
    // while its contribution to the columns beyond the sub-block is deferred
    // and applied as one GEMM `B[:, ke2..ke] -= W * M` per sub-block, where
    // `W` holds the pivot columns before their `D^{-1}` scaling and `M` the
    // multipliers of `mult_snap`. A pivot swap that reaches beyond the
    // sub-block first flushes the pending pivots (the swapped-in column must
    // carry every earlier update, as it does in the sequential sweep). Rows
    // are independent, so any row range gives the same values.
    let deep = r1.saturating_sub(r0);
    if deep == 0 || ke <= kb {
        return;
    }
    debug_assert!(r1 <= lay.w || r0 >= lay.w, "rows in one part of the panel");
    // Column `j`'s entry of row `r0`; its rows follow it.
    let col = |j: usize| base.add(lay.at(r0, j));
    // One spare column: a 2x2 pivot that starts on a sub-block's last
    // column extends the sub-block by one, the pair is never split.
    let sb = sb.max(1);
    reset(w, deep * (sb + 1), T::zero());
    let mut kb2 = kb;
    while kb2 < ke {
        let mut ke2 = (kb2 + sb).min(ke);
        if ke2 < ke && two_by_two[ke2 - 1] {
            ke2 += 1;
        }
        // Pending pivots [pend0, k) whose deferred update has not been applied.
        let mut pend0 = kb2;
        let mut k = kb2;
        while k < ke2 {
            let kp = deep_swaps[k - kb];
            if kp != usize::MAX {
                if kp >= ke2 {
                    flush_trailing(
                        base, lay, kb, ke, kb2, ke2, pend0, k, w, deep, mult_snap, nb, r0,
                    );
                    pend0 = k;
                }
                let src = if two_by_two[k] { k + 1 } else { k };
                let (ca, cb) = (col(src), col(kp));
                for i in 0..deep {
                    core::ptr::swap(ca.add(i), cb.add(i));
                }
            }
            if two_by_two[k] {
                let (d11, d21, d22) = (d_diag[k], d_subdiag[k], d_diag[k + 1]);
                let det = d11 * d22 - d21 * d21;
                let detinv = det.recip();
                let colk = col(k);
                let colk1 = col(k + 1);
                for j in (k + 2)..ke2 {
                    let l1j = mult_snap[(k - kb) * nb + (j - kb)];
                    let l2j = mult_snap[(k + 1 - kb) * nb + (j - kb)];
                    let colj = col(j);
                    for i in 0..deep {
                        *colj.add(i) = *colj.add(i) - *colk.add(i) * l1j - *colk1.add(i) * l2j;
                    }
                }
                let (head, tail) = w.split_at_mut((k + 1 - kb2) * deep);
                let wk = &mut head[(k - kb2) * deep..];
                let wk1 = &mut tail[..deep];
                for i in 0..deep {
                    let wik = *colk.add(i);
                    let wik1 = *colk1.add(i);
                    wk[i] = wik;
                    wk1[i] = wik1;
                    *colk.add(i) = (d22 * wik - d21 * wik1) * detinv;
                    *colk1.add(i) = (d11 * wik1 - d21 * wik) * detinv;
                }
                k += 2;
            } else {
                let dinv = d_diag[k].recip();
                let colk = col(k);
                for j in (k + 1)..ke2 {
                    let wj_dinv = mult_snap[(k - kb) * nb + (j - kb)];
                    if wj_dinv != T::zero() {
                        let colj = col(j);
                        for i in 0..deep {
                            *colj.add(i) = *colj.add(i) - *colk.add(i) * wj_dinv;
                        }
                    }
                }
                let wk = &mut w[(k - kb2) * deep..(k - kb2 + 1) * deep];
                for (i, wi) in wk.iter_mut().enumerate() {
                    let v = *colk.add(i);
                    *wi = v;
                    *colk.add(i) = v * dinv;
                }
                k += 1;
            }
        }
        flush_trailing(
            base, lay, kb, ke, kb2, ke2, pend0, ke2, w, deep, mult_snap, nb, r0,
        );
        kb2 = ke2;
    }
}

/// Apply the deferred updates of pivots `[p0, p1)` of the sub-block
/// `[kb2, ke2)` (their unscaled columns in `w`, indexed from `kb2`) to the
/// columns `[ke2, ke)` of rows `r0..r0 + deep`:
/// `B[:, ke2..ke] -= W[:, p0..p1] * M[p0..p1, ke2..ke]`.
///
/// # Safety
/// `base` is the panel laid out as `lay`; the row range is this task's own,
/// in one part of the panel.
#[allow(clippy::too_many_arguments)]
unsafe fn flush_trailing<T: Scalar>(
    base: *mut T,
    lay: Layout,
    kb: usize,
    ke: usize,
    kb2: usize,
    ke2: usize,
    p0: usize,
    p1: usize,
    w: &[T],
    deep: usize,
    mult_snap: &[T],
    nb: usize,
    r0: usize,
) {
    let npend = p1 - p0;
    let ncols = ke - ke2;
    if npend == 0 || ncols == 0 {
        return;
    }
    let lhs = w.as_ptr().add((p0 - kb2) * deep);
    // The direct kernel, not the backend entry: these products are skinny
    // (k = trailing_block), where the complex split's plane copies cost more
    // than they save (measured: 1.84 s against 1.98 s single-core). One per
    // run of the columns `[ke2, ke)` of uniform stride.
    for run in lay.runs(r0..r0 + deep, ke2..ke) {
        // M rows p0..p1, the run's columns: row-major with stride `nb`.
        let rhs = mult_snap
            .as_ptr()
            .add((p0 - kb) * nb + (run.cols.start - kb));
        gemm::gemm(
            deep,
            run.cols.len(),
            npend,
            base.add(run.off),
            run.ld as isize,
            1,
            true,
            lhs,
            deep as isize,
            1,
            rhs,
            1,
            nb as isize,
            T::one(),
            T::zero() - T::one(),
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

/// One blocked Bunch-Kaufman panel step of the left-looking cdiv over the
/// fully-summed columns `[kb, ke)`: the in-panel getf2 (rows `< ke`) plus the
/// row-parallel deep replay. Touches ONLY panel columns `[kb, ke)` (their full
/// height), which is what makes the cdiv panel lookahead sound: the
/// step for panel `p+1` may run concurrently with the wide part of panel
/// `p`'s deferred Schur update (columns `>= ke2`), the two column ranges are
/// disjoint. Pushes each pivot floor, `(column of the node, what the floor
/// added to the diagonal)`, to `floors`.
///
/// # Safety
/// `base` is the panel laid out as `lay`, of `nrow` rows; nobody else
/// touches its columns `[kb, ke)` meanwhile.
#[allow(clippy::too_many_arguments)]
unsafe fn ll_bk_panel_step<T: Scalar>(
    base: *mut T,
    lay: Layout,
    nrow: usize,
    kb: usize,
    ke: usize,
    nb: usize,
    alpha: f64,
    perturb_floor: Option<f64>,
    ll_cdiv_par: usize,
    trailing_block: usize,
    d: &mut [T],
    d_subdiag: &mut [T],
    two_by_two: &mut [bool],
    lperm: &mut [usize],
    l1: &mut [T],
    l2: &mut [T],
    deep_swaps: &mut [usize],
    mult_snap: &mut [T],
    wpool: &ScratchPool<Vec<T>>,
    floors: &mut Vec<(usize, T)>,
) -> Result<(), RslabError> {
    // getf2: unblocked Bunch-Kaufman over the panel columns [kb, ke), with
    // EVERYTHING bounded to the panel rows `< ke`: pivot candidates,
    // rank-1/rank-2 updates, interchanges. The deep rows `[ke, nrow)` -
    // the dominant `O((nrow-ke)*pw^2)` share on tall panels - are lifted
    // off this serial path into the parallel `apply_bk_panel_trailing`
    // below (bit-identical replay; ports the LU twin's lever).
    for ds in deep_swaps.iter_mut() {
        *ds = usize::MAX;
    }
    // Entry `(i, j)`, on or below the diagonal.
    let a = |i: usize, j: usize| base.add(lay.at(i, j));
    let mut k = kb;
    while k < ke {
        let absakk = (*a(k, k)).magnitude();
        // colmax over the in-panel candidate rows (k+1)..ke.
        let mut colmax_sq = 0.0;
        let mut imax = k;
        for i in (k + 1)..ke {
            let m = (*a(i, k)).magnitude_sq();
            if m > colmax_sq {
                colmax_sq = m;
                imax = i;
            }
        }
        let colmax = colmax_sq.sqrt();

        let kstep;
        let kp;
        if absakk.max(colmax) == 0.0 {
            if perturb_floor.is_none() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            kstep = 1;
            kp = k;
        } else if absakk >= alpha * colmax {
            kstep = 1;
            kp = k;
        } else {
            // rowmax in row `imax`, restricted to the panel.
            let mut rowmax_sq = 0.0;
            for j in k..imax {
                let m = (*a(imax, j)).magnitude_sq();
                if m > rowmax_sq {
                    rowmax_sq = m;
                }
            }
            for i in (imax + 1)..ke {
                let m = (*a(i, imax)).magnitude_sq();
                if m > rowmax_sq {
                    rowmax_sq = m;
                }
            }
            let rowmax = rowmax_sq.sqrt();
            if absakk >= alpha * colmax * (colmax / rowmax) {
                kstep = 1;
                kp = k;
            } else if (*a(imax, imax)).magnitude() >= alpha * rowmax {
                kstep = 1;
                kp = imax;
            } else {
                kstep = 2;
                kp = imax;
            }
        }

        if kstep == 1 {
            if kp != k {
                swap_sym_lower_bounded(base, lay, k, kp, ke);
                lperm.swap(k, kp);
                deep_swaps[k - kb] = kp;
            }
            let mut dk = *a(k, k);
            match perturb_floor {
                Some(floor) if dk.magnitude() < floor => {
                    let lifted = perturb_pivot(dk, floor);
                    floors.push((k, lifted - dk));
                    dk = lifted;
                    *a(k, k) = dk;
                }
                None if dk == T::zero() => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            d[k] = dk;
            let dinv = dk.recip();
            // Update the in-panel trailing columns (k+1)..ke over the
            // panel rows, then scale column k's panel rows (deep rows
            // replayed in the parallel apply).
            let ck = a(k, k);
            for j in (k + 1)..ke {
                let wj_dinv = *ck.add(j - k) * dinv;
                mult_snap[(k - kb) * nb + (j - kb)] = wj_dinv;
                if wj_dinv != T::zero() {
                    let cj = a(j, j);
                    for i in 0..ke - j {
                        *cj.add(i) = *cj.add(i) - *ck.add(j - k + i) * wj_dinv;
                    }
                }
            }
            for i in 1..ke - k {
                *ck.add(i) = *ck.add(i) * dinv;
            }
            k += 1;
        } else {
            if kp != k + 1 {
                swap_sym_lower_bounded(base, lay, k + 1, kp, ke);
                lperm.swap(k + 1, kp);
                deep_swaps[k - kb] = kp;
            }
            let mut d11 = *a(k, k);
            let d21 = *a(k + 1, k);
            let mut d22 = *a(k + 1, k + 1);
            let mut det = d11 * d22 - d21 * d21;
            let scale = d11.magnitude().max(d22.magnitude()).max(d21.magnitude());
            let growth_floor = GROWTH_EPS * scale * scale;
            match perturb_floor {
                Some(floor) => {
                    let fl = (floor * floor).max(growth_floor);
                    if det.magnitude() < fl {
                        // Lift both diagonal entries in steps of `step` until
                        // the block clears the floor, so the stored block is
                        // the exact block of the matrix plus the lift: the
                        // determinant is a monic quadratic in the lift whose
                        // roots, when it is still below the floor at one
                        // step, lie within a few steps of zero.
                        let step = floor.max(scale * GROWTH_EPS.sqrt());
                        let mut lift = step;
                        while ((d11 + T::from_real(lift)) * (d22 + T::from_real(lift)) - d21 * d21)
                            .magnitude()
                            < fl
                        {
                            lift += step;
                        }
                        d11 = d11 + T::from_real(lift);
                        d22 = d22 + T::from_real(lift);
                        det = d11 * d22 - d21 * d21;
                        floors.push((k, T::from_real(lift)));
                        floors.push((k + 1, T::from_real(lift)));
                    }
                }
                None if det.magnitude() <= growth_floor => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            let detinv = det.recip();
            d[k] = d11;
            d_subdiag[k] = d21;
            d[k + 1] = d22;
            two_by_two[k] = true;
            for i in (k + 2)..ke {
                let wik = *a(i, k);
                let wik1 = *a(i, k + 1);
                l1[i] = (d22 * wik - d21 * wik1) * detinv;
                l2[i] = (d11 * wik1 - d21 * wik) * detinv;
                mult_snap[(k - kb) * nb + (i - kb)] = l1[i];
                mult_snap[(k + 1 - kb) * nb + (i - kb)] = l2[i];
            }
            let (ck, ck1) = (a(k, k), a(k + 1, k + 1));
            for j in (k + 2)..ke {
                let l1j = l1[j];
                let l2j = l2[j];
                let cj = a(j, j);
                for i in 0..ke - j {
                    *cj.add(i) =
                        *cj.add(i) - *ck.add(j - k + i) * l1j - *ck1.add(j - k - 1 + i) * l2j;
                }
            }
            for i in (k + 2)..ke {
                *a(i, k) = l1[i];
                *a(i, k + 1) = l2[i];
            }
            k += 2;
        }
    }
    // Deep rows [ke, nrow): replay this panel's interchanges + pivot
    // transforms row-parallel (bit-identical to the old full-height
    // getf2 - same per-row op sequence). This is the dominant panel
    // work on tall supernodes; it now runs on all idle workers instead
    // of the serial getf2 path.
    if nrow > ke {
        let deep = nrow - ke;
        let pw = ke - kb;
        // The deep rows in chunks within one part of the panel each: its
        // diagonal part's rows past the step, then the off-block rows.
        let parts = [(ke, lay.w.max(ke)), (lay.w.max(ke), nrow)];
        let replay = |r0: usize, r1: usize, w: &mut Vec<T>| {
            // SAFETY: disjoint row chunk; see `apply_bk_panel_trailing`.
            unsafe {
                apply_bk_panel_trailing(
                    base,
                    lay,
                    kb,
                    ke,
                    d,
                    d_subdiag,
                    two_by_two,
                    deep_swaps,
                    mult_snap,
                    nb,
                    r0,
                    r1,
                    trailing_block,
                    w,
                )
            }
        };
        if deep * pw * pw >= ll_cdiv_par {
            let pp = LdltPanelPtr(base);
            let nthreads = crate::parallel::threads();
            let cs = deep.div_ceil(nthreads).max(1);
            let chunks: Vec<(usize, usize)> = (parts.iter())
                .flat_map(|&(p0, p1)| (p0..p1).step_by(cs).map(move |r| (r, (r + cs).min(p1))))
                .collect();
            let (d, d_subdiag, two_by_two) = (&*d, &*d_subdiag, &*two_by_two);
            let (deep_swaps, mult_snap) = (&*deep_swaps, &*mult_snap);
            crate::parallel::for_each(
                chunks.len(),
                crate::parallel::Work::ns(f64::INFINITY),
                |c| {
                    let (r0, r1) = chunks[c];
                    let mut w = wpool.take();
                    // SAFETY: disjoint row chunk; see `apply_bk_panel_trailing`.
                    unsafe {
                        apply_bk_panel_trailing(
                            pp.get(),
                            lay,
                            kb,
                            ke,
                            d,
                            d_subdiag,
                            two_by_two,
                            deep_swaps,
                            mult_snap,
                            nb,
                            r0,
                            r1,
                            trailing_block,
                            &mut w,
                        )
                    };
                },
            );
        } else {
            // Row chunks that keep a chunk of every panel column in L1 across
            // the whole pivot sweep (the sweep streams each column once per
            // pivot; unchunked, a deep panel is re-read from L2 `pw` times).
            // Rows are independent, so every row's arithmetic is unchanged:
            // bit-identical to one call over all rows.
            let mut w = wpool.take();
            for &(p0, p1) in &parts {
                if p1 > p0 {
                    replay(p0, p1, &mut w);
                }
            }
        }
    }

    Ok(())
}

/// Heap bytes [`ll_cdiv_emit`] allocates for an `nrow x ncol` panel besides
/// what it keeps in the node's slot (`D`, the 2x2 flags and the row
/// permutation): the pivot and multiplier scratch, the deferred update's
/// blocks as they grow panel by panel, the deep-row replay's sub-block
/// buffers and the Schur tiles' diagonal blocks on up to `workers` threads
/// at once; and the largest split planes its Schur GEMMs take on a worker
/// (complex fields) and how many workers may take them at once. Replays the
/// kernel's panel loop, for the memory plan.
pub(super) fn ll_cdiv_scratch<T: Scalar>(
    ncol: usize,
    nrow: usize,
    k: &crate::KernelSettings,
    workers: usize,
) -> (u64, usize, usize) {
    use super::gemm::schur_tile_planes;
    use crate::memory::grown;
    let par_cdiv = k.cdiv_par(nrow, ncol);
    let nb = k.panel_width(k.panel_nb, ncol);
    // A node narrower than a panel is one panel: its scratch (the `nb x nb`
    // multiplier snapshots) sized to its width, the same blocking.
    let nb = nb.min(ncol.max(1));
    let (mut l21, mut diag, mut planes, mut copies) = (0, 0, 0, 1);
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + nb).min(ncol);
        let (pw, cw, mt) = (ke - kb, ncol - ke, nrow - ke);
        if pw > 0 && cw > 0 && mt > 0 && k.use_gemm_schur {
            l21 = grown(l21, mt * pw);
            let par = mt * cw * pw >= par_cdiv;
            let tiles = if par {
                (PANEL_BLOCK, PANEL_BLOCK)
            } else {
                (k.schur_tile, mt)
            };
            let cw_n = (ke + nb).min(ncol) - ke;
            for range in [(0, cw_n), (cw_n, cw)] {
                if range.1 > range.0 {
                    let (p, pieces) = schur_tile_planes(mt, range, pw, tiles, k);
                    let at_once = if par { pieces.min(workers.max(1)) } else { 1 };
                    diag = diag.max(tiles.0.min(cw).pow(2) * at_once);
                    if T::COMPLEX && p > 0 {
                        planes = planes.max(p);
                        copies = copies.max(at_once);
                    }
                }
            }
        }
        kb = ke;
    }
    // l1, l2 and their lookahead twins, both multiplier snapshots, `l21buf`
    // and `gbuf`, the replay's sub-blocks over all deep rows and the tiles'
    // diagonal blocks.
    let entries = crate::constants::BK_ROW_BUFFERS * nrow
        + 2 * nb * nb
        + 2 * l21
        + nrow * (k.trailing_block.max(1) + 1)
        + diag;
    let bytes = entries * std::mem::size_of::<T>() + 2 * nb * std::mem::size_of::<usize>();
    (bytes as u64, planes, copies)
}

/// cdiv + store + emit for supernode `s` on an already fully cmod-updated
/// `panel` - the tail of [`ll_factor_node`], extracted so the spine
/// pipeline executor can drive assembly/cmod itself and reuse
/// the identical factor kernel. Takes `panel` and the global->local scratch
/// `gloc` by value (`gloc` is returned to the thread-local scratch slot on
/// every exit path).
#[allow(clippy::too_many_arguments)]
pub(super) fn ll_cdiv_emit<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    pools: &BkPools<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    kt: KernelTuning,
    panel: &mut [T],
) -> Result<(), RslabError> {
    let snode = &sym.supernodes[s];
    let ncol = snode.ncol;
    let nrow = sched.rows(s).len();
    // cdiv: partial **blocked** Bunch-Kaufman LDL^T (1x1 and 2x2 pivots), the
    // rectangular `nrow x ncol` analogue of `factor_front`'s panel kernel. The
    // fully-summed columns are factored in panels of width `NB` with pivoting
    // **bounded to the panel** (candidate rows `(k+1)..ke`), then each panel's
    // trailing update - the remaining panel columns `[ke, ncol)` over all rows
    // `[ke, nrow)` - is deferred to one SIMD GEMM (the BLAS-3 bulk, replacing the
    // scalar rank-1/rank-2 sweeps that dominated wide separators). Unlike
    // `factor_front` there is **no `A22` block** (the panel has no columns beyond
    // `ncol`; that Schur update is the ancestors' `cmod`), so the trailing region
    // is the rectangular `(nrow-ke) x (ncol-ke)` lower part. Pivoting stays inside
    // `0..ncol`, so the off-diagonal rows `[ncol, nrow)` keep their identity and
    // `s`'s contribution to ancestors is unaffected by this internal permutation.
    //
    // Adaptive panel width: wide separators get double-width panels - the
    // deferred Schur GEMM's inner dimension is `nb`, and k = 64 is too thin
    // to reach peak on root-class panels (measured ~79 Gflop/s-eq). The
    // extra serial getf2 work is O(nb^3) per panel - negligible against the
    // GEMM gain at this size. The global nb sweep said 128 loses overall
    // because SMALL panels pay; widening only from `wide_panel_min_ncol` (a
    // pure function of the node, thread-count independent) keeps them at
    // default.
    let nb = kt.k.panel_width(kt.k.panel_nb, ncol);
    // A node narrower than a panel is one panel: its scratch (the `nb x nb`
    // multiplier snapshots) sized to its width, the same blocking.
    let nb = nb.min(ncol.max(1));
    // Same join-steal guard as cmod: a small node must not fork inside its
    // cdiv (deep-row apply / deferred Schur GEMM) - the blocked join steals
    // foreign subtree work and stalls this node's dependents. Total cdiv
    // work ~ nrow*ncol^2 (panel + trailing updates).
    let ll_cdiv_par = kt.k.cdiv_par(nrow, ncol);
    let alpha = bk_alpha();
    let lay = Layout::new(ncol, nrow - ncol);
    debug_assert!(panel.len() >= lay.len());
    let base = LdltPanelPtr(panel.as_mut_ptr());
    // The node's buffers, on loan from the factorization's pool and reset
    // to the values they start from: the pivot outputs (`D`, the 2x2 flags,
    // the row permutation; read back through the emit cells), the 2x2
    // multiplier scratch (only `[k+2, nrow)` is read each step), the
    // deferred-GEMM buffers, the per-step interchange partners
    // (`usize::MAX` = none) and the time-of-step multipliers (`nb x nb`,
    // column = step) the deep-row replay consumes, and the same set again
    // for the lookahead's next-panel step.
    let mut lent = pools.node.take();
    let BkScratch {
        d,
        d_subdiag,
        two_by_two,
        lperm,
        l1,
        l2,
        l21buf,
        gbuf,
        deep_swaps,
        mult_snap,
        l1b,
        l2b,
        deep_swaps_b,
        mult_snap_b,
    } = &mut *lent;
    reset(d, ncol, T::zero());
    reset(d_subdiag, ncol, T::zero());
    reset(two_by_two, ncol, false);
    lperm.clear();
    lperm.extend(0..nrow);
    for (v, len) in [
        (&mut *l1, nrow),
        (&mut *l2, nrow),
        (&mut *l1b, nrow),
        (&mut *l2b, nrow),
    ] {
        reset(v, len, T::zero());
    }
    reset(mult_snap, nb * nb, T::zero());
    reset(mult_snap_b, nb * nb, T::zero());
    reset(deep_swaps, nb, usize::MAX);
    reset(deep_swaps_b, nb, usize::MAX);
    for v in [&mut *l21buf, &mut *gbuf] {
        v.clear();
    }
    // The node's pivot floors, `(column of the node, what the floor added)`.
    let mut floors: Vec<(usize, T)> = Vec::new();
    // The high-water mark of columns already factored ahead by the lookahead
    // join.
    let mut done_through = 0usize;
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + nb).min(ncol);
        if kb >= done_through {
            let _p = rsprobe::scope!(TRACE, "panel");
            // SAFETY: the node's own slot, no other task touches it.
            let r = unsafe {
                ll_bk_panel_step(
                    base.get(),
                    lay,
                    nrow,
                    kb,
                    ke,
                    nb,
                    alpha,
                    perturb_floor,
                    ll_cdiv_par,
                    kt.k.trailing_block,
                    &mut *d,
                    &mut *d_subdiag,
                    &mut *two_by_two,
                    &mut *lperm,
                    &mut *l1,
                    &mut *l2,
                    &mut *deep_swaps,
                    &mut *mult_snap,
                    &pools.w,
                    &mut floors,
                )
            };
            r?;
        }
        // Deferred panel trailing update: panel[ke.., ke..ncol] -= L21*D*R^T, where
        // L21 = panel rows [ke,nrow) x panel cols [kb,ke) (mtxpw), G = L21*D (block-
        // diagonal D), and R = the first `cw` rows of L21 (the rows that are
        // themselves remaining panel columns [ke,ncol)); only the lower part
        // (rows `>= ` the column) is updated.
        let pw = ke - kb;
        let cw = ncol - ke; // remaining fully-summed columns to update
        let mt = nrow - ke; // trailing rows (left-factor height)
        if pw > 0 && cw > 0 && mt > 0 {
            l21buf.clear();
            l21buf.resize(mt * pw, T::zero());
            for cc in 0..pw {
                let c = kb + cc;
                for run in lay.runs(ke..nrow, c..c + 1) {
                    let (r0, len) = (run.rows.start - ke, run.rows.len());
                    l21buf[r0 + cc * mt..r0 + len + cc * mt]
                        .copy_from_slice(&panel[run.off..run.off + len]);
                }
            }
            gbuf.clear();
            gbuf.resize(mt * pw, T::zero());
            let mut cc = 0;
            while cc < pw {
                let c = kb + cc;
                if two_by_two[c] {
                    let (d11, d21, d22) = (d[c], d_subdiag[c], d[c + 1]);
                    for rr in 0..mt {
                        let a = l21buf[rr + cc * mt];
                        let b = l21buf[rr + (cc + 1) * mt];
                        gbuf[rr + cc * mt] = a * d11 + b * d21;
                        gbuf[rr + (cc + 1) * mt] = a * d21 + b * d22;
                    }
                    cc += 2;
                } else {
                    let dc = d[c];
                    for rr in 0..mt {
                        gbuf[rr + cc * mt] = l21buf[rr + cc * mt] * dc;
                    }
                    cc += 1;
                }
            }
            // The trailing columns in tiles straight into the panel
            // (`schur_tiles`), on the pool where the update is parallel work:
            // narrow tiles that spread over the workers there, else the
            // Schur tile. Both gates are pure functions of the node shape, so
            // the tiles, and therefore the bits, are deterministic per
            // matrix; `ll_thread_determinism` holds.
            let par = mt * cw * pw >= ll_cdiv_par;
            let tiles = if par {
                (PANEL_BLOCK, PANEL_BLOCK)
            } else {
                (kt.k.schur_tile, mt)
            };
            let schur = |range: (usize, usize)| {
                let _s = rsprobe::scope!(TRACE, "schur");
                // SAFETY: the trailing rows of the columns in `range`, the
                // caller's own; `gbuf` and `l21buf` hold `mt * pw`.
                unsafe {
                    schur_tiles(
                        base.get(),
                        lay,
                        ke,
                        mt,
                        pw,
                        &gbuf[..],
                        (&l21buf[..], mt),
                        range,
                        tiles,
                        par,
                        &pools.w,
                        &kt.k,
                    )
                }
            };
            // Panel lookahead: the next panel's columns [ke, ke2) first, then
            // factor the next panel concurrently with the update of the rest
            // ([ke2, ncol)) - the two touch disjoint column ranges.
            let ke2 = (ke + nb).min(ncol);
            let cw_n = ke2 - ke;
            let wide = cw - cw_n;
            let look = kt.k.use_gemm_schur && wide > 0 && mt * wide * pw >= kt.k.par_cdiv;
            if look {
                schur((0, cw_n));
                let floors_ref = &mut floors;
                let (step_res, ()) = crate::parallel::join(
                    crate::parallel::Work::ns(f64::INFINITY),
                    || {
                        let _p = rsprobe::scope!(TRACE, "panel");
                        // SAFETY: the columns `[ke, ke2)`, which the update
                        // beside it does not touch.
                        unsafe {
                            ll_bk_panel_step(
                                base.get(),
                                lay,
                                nrow,
                                ke,
                                ke2,
                                nb,
                                alpha,
                                perturb_floor,
                                ll_cdiv_par,
                                kt.k.trailing_block,
                                &mut *d,
                                &mut *d_subdiag,
                                &mut *two_by_two,
                                &mut *lperm,
                                &mut *l1b,
                                &mut *l2b,
                                &mut *deep_swaps_b,
                                &mut *mult_snap_b,
                                &pools.w,
                                floors_ref,
                            )
                        }
                    },
                    || schur((cw_n, cw)),
                );
                step_res?;
                done_through = ke2;
            } else if kt.k.use_gemm_schur {
                schur((0, cw));
            } else {
                for cc2 in 0..cw {
                    let c = ke + cc2;
                    for rr in cc2..mt {
                        let mut acc = T::zero();
                        for kk2 in 0..pw {
                            acc = acc + gbuf[rr + kk2 * mt] * l21buf[cc2 + kk2 * mt];
                        }
                        let dst = lay.at(ke + rr, c);
                        panel[dst] = panel[dst] - acc;
                    }
                }
            }
        }
        kb = ke;
    }
    let eoff = emit.e_offset[s];
    if !floors.is_empty() {
        let mut all = emit.floors.lock().unwrap_or_else(|e| e.into_inner());
        all.extend(floors.into_iter().map(|(k, delta)| (eoff + k, delta)));
    }
    // Populate the O(n) emit maps + inertia for `s` (block-aware over its 1x1/2x2
    // Bunch-Kaufman D), mirroring the legacy pass-1 emit. The `e`-numbering is one
    // position per column, so `e_offset[s] + p` is column `p`'s elimination index.
    let (mut ipos, mut ineg, mut izero) = (0usize, 0usize, 0usize);
    let mut pp = 0;
    while pp < ncol {
        let g = sched.rows(s)[lperm[pp]] as usize;
        let e = eoff + pp;
        // SAFETY: each global index / position is written by exactly one supernode.
        unsafe {
            emit.e_of_g.set(g, e);
            emit.perm.set(e, sym.perm[g]);
            emit.d_diag.set(e, d[pp]);
        }
        if two_by_two[pp] {
            let g2 = sched.rows(s)[lperm[pp + 1]] as usize;
            unsafe {
                emit.e_of_g.set(g2, e + 1);
                emit.perm.set(e + 1, sym.perm[g2]);
                emit.d_diag.set(e + 1, d[pp + 1]);
                emit.d_subdiag.set(e, d_subdiag[pp]);
                emit.two_by_two.set(e, true);
            }
            let det_r = (d[pp] * d[pp + 1] - d_subdiag[pp] * d_subdiag[pp]).real();
            let tr_r = (d[pp] + d[pp + 1]).real();
            if det_r < 0.0 {
                ipos += 1;
                ineg += 1;
            } else if det_r > 0.0 {
                if tr_r >= 0.0 {
                    ipos += 2;
                } else {
                    ineg += 2;
                }
            } else {
                izero += 1;
                if tr_r >= 0.0 {
                    ipos += 1;
                } else {
                    ineg += 1;
                }
            }
            pp += 2;
        } else {
            let r = d[pp].real();
            if r > 0.0 {
                ipos += 1;
            } else if r < 0.0 {
                ineg += 1;
            } else {
                izero += 1;
            }
            pp += 1;
        }
    }
    emit.inertia_pos.fetch_add(ipos, Ordering::Relaxed);
    emit.inertia_neg.fetch_add(ineg, Ordering::Relaxed);
    emit.inertia_zero.fetch_add(izero, Ordering::Relaxed);
    debug_assert!(
        lperm[ncol..]
            .iter()
            .enumerate()
            .all(|(i, &p)| p == ncol + i),
        "the off-diagonal rows keep their order"
    );
    Ok(())
}

/// The Bunch-Kaufman pivot threshold `alpha = (1 + sqrt17)/8 ~ 0.6404`.
#[inline]
pub(crate) fn bk_alpha() -> f64 {
    (1.0 + 17.0_f64.sqrt()) / 8.0
}

/// Symmetric interchange of rows and columns `p < q` in the lower triangle
/// of the panel `base` (laid out as `lay`), with the below-`q`
/// column-segment swap bounded to rows `< row_limit`. The blocked Bunch-Kaufman panel kernels keep their pivot
/// interchanges inside the panel rows and replay the deep-row segments later
/// in the parallel trailing apply (`apply_bk_panel_trailing`), so the
/// interchange sequence reaches every row exactly once, in step order.
///
/// # Safety
/// `base` is the panel laid out as `lay`, `row_limit <= lay.w`.
pub(crate) unsafe fn swap_sym_lower_bounded<T: Scalar>(
    base: *mut T,
    lay: Layout,
    p: usize,
    q: usize,
    row_limit: usize,
) {
    debug_assert!(p < q && q < lay.w && q < row_limit && row_limit <= lay.w);
    let a = |i: usize, j: usize| base.add(lay.at(i, j));
    // Column segment strictly below q: (i, p) <-> (i, q) for i > q.
    for i in (q + 1)..row_limit {
        std::ptr::swap(a(i, p), a(i, q));
    }
    // Middle cross strip: (i, p) <-> (q, i) for p < i < q.
    for i in (p + 1)..q {
        std::ptr::swap(a(i, p), a(q, i));
    }
    // Diagonal: (p, p) <-> (q, q).
    std::ptr::swap(a(p, p), a(q, q));
    // Left row segments: (p, j) <-> (q, j) for j < p.
    for j in 0..p {
        std::ptr::swap(a(p, j), a(q, j));
    }
}
