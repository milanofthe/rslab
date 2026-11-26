//! The numeric factorization: per-block Gilbert-Peierls LU with threshold
//! pivoting, and the replay plan of the refactorization.

use super::*;
use crate::constants::{
    COL_WORK, KLU_FACTOR_NS_PER_ENTRY, KLU_F_RESERVE_DIV, KLU_L_RESERVE, KLU_U_RESERVE,
    SCAN_NS_PER_ENTRY,
};
use crate::parallel::{Rate, ScalarRates, Work};

pub(super) static FACTOR_RATE: ScalarRates = ScalarRates::new(KLU_FACTOR_NS_PER_ENTRY);

static SPLICE_RATE: Rate = Rate::new(SCAN_NS_PER_ENTRY);

pub(super) fn planned(bn: usize, nlev: usize, parallel: KluParallel) -> bool {
    parallel != KluParallel::Off && bn >= 2 && nlev >= 2
}

impl BlockPlan {
    fn new(b: usize, bs: usize, level: &[Ki], colw: &[u64], nlev: usize) -> Self {
        let mut level_ptr = vec![0u32; nlev + 1];
        let mut level_work = vec![0u64; nlev];
        for (&l, &w) in level.iter().zip(colw) {
            level_ptr[l as usize + 1] += 1;
            level_work[l as usize] += w;
        }
        for l in 0..nlev {
            level_ptr[l + 1] += level_ptr[l];
        }
        let mut fill = level_ptr.clone();
        let mut level_cols = vec![0 as Ki; level.len()];
        for (k, &l) in level.iter().enumerate() {
            level_cols[fill[l as usize] as usize] = (bs + k) as Ki;
            fill[l as usize] += 1;
        }
        let widest = (0..nlev)
            .filter(|&l| level_ptr[l + 1] - level_ptr[l] >= 2)
            .map(|l| level_work[l])
            .max()
            .unwrap_or(0);
        BlockPlan {
            block: b,
            work: colw.iter().sum(),
            widest,
            level_ptr,
            level_cols,
            level_work,
        }
    }
}

fn compute_replay_plan(
    block_ptr: &[usize],
    l_colptr: &[usize],
    u_colptr: &[usize],
    u_rowidx: &[Ki],
    settings: &KluSettings,
) -> ReplayPlan {
    let mut plan = ReplayPlan {
        policy: settings.parallel,
        min_work: settings.par_min_work,
        min_ratio: settings.par_min_ratio,
        ..ReplayPlan::default()
    };
    if settings.parallel == KluParallel::Off {
        return plan;
    }
    let (mut level, mut colw): (Vec<Ki>, Vec<u64>) = Default::default();
    for b in 0..block_ptr.len() - 1 {
        let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
        let bn = be - bs;
        colw.clear();
        colw.resize(bn, 0);
        level.clear();
        level.resize(bn, 0);
        let mut nlev = 1usize;
        for j in bs..be {
            let (mut l, mut w): (Ki, u64) = (0, COL_WORK);
            for &pk in &u_rowidx[u_colptr[j]..u_colptr[j + 1]] {
                let p = pk as usize;
                w += (l_colptr[p + 1] - l_colptr[p]) as u64;
                l = l.max(level[p - bs] + 1);
            }
            level[j - bs] = l;
            colw[j - bs] = w;
            nlev = nlev.max(l as usize + 1);
        }
        let w_b: u64 = colw.iter().sum();
        plan.total += w_b;
        plan.largest = plan.largest.max(w_b);
        if planned(bn, nlev, settings.parallel) {
            plan.blocks.push(BlockPlan::new(b, bs, &level, &colw, nlev));
        }
    }
    plan
}

pub(super) fn factor_csc<T: Scalar>(
    n: usize,
    colptr: &[usize],
    rowidx: &[Ki],
    val: &[T],
    diag: impl Fn(usize) -> Option<T>,
) -> GeneralCsc<T> {
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::with_capacity(val.len() + n);
    let mut values = Vec::with_capacity(val.len() + n);
    let mut col: Vec<(usize, T)> = Vec::new();
    for j in 0..n {
        col.clear();
        col.extend((colptr[j]..colptr[j + 1]).map(|k| (rowidx[k] as usize, val[k])));
        col.extend(diag(j).map(|d| (j, d)));
        col.sort_unstable_by_key(|e| e.0);
        for &(r, v) in &col {
            row_idx.push(r);
            values.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    GeneralCsc {
        n,
        col_ptr,
        row_idx,
        values,
    }
}

pub(super) fn row_scale_inv<T: Scalar>(a: &GeneralCsc<T>, enabled: bool) -> Vec<f64> {
    let mut rs = Vec::new();
    row_scale_inv_into(a, enabled, &mut rs);
    rs
}

pub(super) fn row_scale_inv_into<T: Scalar>(a: &GeneralCsc<T>, enabled: bool, rs: &mut Vec<f64>) {
    rs.clear();
    rs.resize(a.n, 0.0);
    if enabled {
        for (k, &i) in a.row_idx.iter().enumerate() {
            let m = a.values[k].magnitude();
            if m > rs[i] {
                rs[i] = m;
            }
        }
    }
    for m in rs.iter_mut() {
        *m = if *m > 0.0 && m.is_finite() {
            1.0 / *m
        } else {
            1.0
        };
    }
}

pub(super) struct BlockOut<T> {
    fin_abs: Vec<Ki>,
    l_colptr: Vec<usize>,
    l_rowidx: Vec<Ki>,
    l_val: Vec<T>,
    u_colptr: Vec<usize>,
    u_rowidx: Vec<Ki>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    f_colptr: Vec<usize>,
    f_pre: Vec<Ki>,
    f_val: Vec<T>,
    f_k: Vec<Ki>,
    prog_in: Vec<(Ki, Ki)>,
}

impl<T> Default for BlockOut<T> {
    fn default() -> Self {
        Self {
            fin_abs: Vec::new(),
            l_colptr: Vec::new(),
            l_rowidx: Vec::new(),
            l_val: Vec::new(),
            u_colptr: Vec::new(),
            u_rowidx: Vec::new(),
            u_val: Vec::new(),
            udiag: Vec::new(),
            f_colptr: Vec::new(),
            f_pre: Vec::new(),
            f_val: Vec::new(),
            f_k: Vec::new(),
            prog_in: Vec::new(),
        }
    }
}

impl<T: Scalar> BlockOut<T> {
    // https://github.com/milanofthe/rslab/issues/290
    fn reset(&mut self, bn: usize, annz: usize) {
        self.fin_abs.clear();
        self.fin_abs.resize(bn, KI_UNSET);
        self.l_colptr.clear();
        self.l_colptr.push(0);
        self.l_rowidx.clear();
        self.l_rowidx.reserve(annz * KLU_L_RESERVE);
        self.l_val.clear();
        self.l_val.reserve(annz * KLU_L_RESERVE);
        self.u_colptr.clear();
        self.u_colptr.push(0);
        self.u_rowidx.clear();
        self.u_rowidx.reserve(annz * KLU_U_RESERVE);
        self.u_val.clear();
        self.u_val.reserve(annz * KLU_U_RESERVE);
        self.udiag.clear();
        self.udiag.reserve(bn);
        self.f_colptr.clear();
        self.f_colptr.push(0);
        self.f_pre.clear();
        self.f_val.clear();
        self.f_k.clear();
        self.prog_in.clear();
        self.prog_in.reserve(annz);
    }
}

// https://github.com/milanofthe/rslab/issues/290
pub(super) struct KluScratch<T> {
    mark: Vec<[Ki; 2]>,
    prune_end: Vec<Ki>,
    x: Vec<T>,
    node_stack: Vec<Ki>,
    cur_stack: Vec<usize>,
    topo: Vec<Ki>,
    nonpiv: Vec<Ki>,
}

impl<T: Scalar> KluScratch<T> {
    fn new(max_bn: usize) -> Self {
        Self {
            mark: vec![[0, KI_UNSET]; max_bn],
            prune_end: vec![KI_UNSET; max_bn],
            x: vec![T::zero(); max_bn],
            node_stack: vec![0 as Ki; max_bn],
            cur_stack: vec![0usize; max_bn],
            topo: Vec::with_capacity(max_bn),
            nonpiv: Vec::with_capacity(max_bn),
        }
    }

    fn reset(&mut self, bn: usize) {
        self.mark[..bn].fill([0, KI_UNSET]);
        self.prune_end[..bn].fill(KI_UNSET);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn factor_block<T: Scalar>(
    sym: &KluSymbolic,
    a: &GeneralCsc<T>,
    settings: &KluSettings,
    rs_inv: &[f64],
    pinv_pre: &[Ki],
    bs: usize,
    be: usize,
    scratch: &mut KluScratch<T>,
    out: &mut BlockOut<T>,
) -> Result<(), RslabError> {
    let bn = be - bs;

    if bn == 1 {
        let c = sym.col_perm[bs];
        out.reset(1, 0);
        out.fin_abs[0] = bs as Ki;
        out.l_colptr.push(0);
        out.u_colptr.push(0);
        let mut diag: Option<T> = None;
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let pre = pinv_pre[a.row_idx[k]] as usize;
            let sv = a.values[k] * T::from_real(rs_inv[a.row_idx[k]]);
            if pre == bs {
                diag = Some(sv);
                out.prog_in.push((k as Ki, bs as Ki));
            } else if pre < bs {
                out.f_pre.push(pre as Ki);
                out.f_val.push(sv);
                out.f_k.push(k as Ki);
            } else {
                return Err(pattern_mismatch());
            }
        }
        let d = diag.ok_or(RslabError::SingularBasis { column: c })?;
        if d.magnitude() == 0.0 || !d.is_finite() {
            return Err(RslabError::SingularBasis { column: c });
        }
        out.udiag.push(d);
        out.f_colptr.push(out.f_pre.len());
        return Ok(());
    }

    scratch.reset(bn);
    let mark = scratch.mark.as_mut_slice();
    // https://doi.org/10.1137/0613017
    let prune_end = scratch.prune_end.as_mut_slice();
    let x = scratch.x.as_mut_slice();
    let node_stack = scratch.node_stack.as_mut_slice();
    let cur_stack = scratch.cur_stack.as_mut_slice();
    let topo = &mut scratch.topo;
    let nonpiv = &mut scratch.nonpiv;
    // https://doi.org/10.1145/1824801.1824814
    // https://github.com/milanofthe/rslab/issues/290
    let mut diag_row: Vec<Ki> = Vec::new();
    let mut diag_col: Vec<Ki> = Vec::new();

    let annz: usize = (bs..be)
        .map(|j| {
            let c = sym.col_perm[j];
            a.col_ptr[c + 1] - a.col_ptr[c]
        })
        .sum();
    out.reset(bn, annz);

    for lj in 0..bn {
        let j = bs + lj;
        let c = sym.col_perm[j];
        let stamp = (lj + 1) as Ki;
        topo.clear();
        nonpiv.clear();

        let col_end = |p: usize, prune_end: &[Ki], l_colptr: &[usize]| -> usize {
            let lp = prune_end[p];
            if lp == KI_UNSET {
                l_colptr[p + 1]
            } else {
                lp as usize
            }
        };
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let pre = pinv_pre[a.row_idx[k]] as usize;
            if pre < bs {
                continue;
            }
            if pre >= be {
                return Err(pattern_mismatch());
            }
            let lb = pre - bs;
            let m = mark[lb];
            if m[0] == stamp {
                continue;
            }
            mark[lb][0] = stamp;
            if m[1] == KI_UNSET {
                nonpiv.push(lb as Ki);
                continue;
            }
            let mut d = 0usize;
            node_stack[0] = lb as Ki;
            cur_stack[0] = out.l_colptr[m[1] as usize];
            let mut end = col_end(m[1] as usize, prune_end, &out.l_colptr);
            loop {
                let mut descended = false;
                while cur_stack[d] < end {
                    debug_assert!(cur_stack[d] < out.l_rowidx.len());
                    let ch = unsafe { *out.l_rowidx.get_unchecked(cur_stack[d]) } as usize;
                    cur_stack[d] += 1;
                    debug_assert!(ch < bn);
                    let mch = unsafe { *mark.get_unchecked(ch) };
                    if mch[0] == stamp {
                        continue;
                    }
                    unsafe { mark.get_unchecked_mut(ch)[0] = stamp };
                    if mch[1] == KI_UNSET {
                        nonpiv.push(ch as Ki);
                        continue;
                    }
                    d += 1;
                    node_stack[d] = ch as Ki;
                    let p = mch[1] as usize;
                    cur_stack[d] = out.l_colptr[p];
                    end = col_end(p, prune_end, &out.l_colptr);
                    descended = true;
                    break;
                }
                if descended {
                    continue;
                }
                topo.push(node_stack[d]);
                if d == 0 {
                    break;
                }
                d -= 1;
                let up = mark[node_stack[d] as usize][1] as usize;
                end = col_end(up, prune_end, &out.l_colptr);
            }
        }

        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[k];
            let pre = pinv_pre[r] as usize;
            let sv = a.values[k] * T::from_real(rs_inv[r]);
            if pre < bs {
                out.f_pre.push(pre as Ki);
                out.f_val.push(sv);
                out.f_k.push(k as Ki);
            } else {
                x[pre - bs] = sv;
                out.prog_in.push((k as Ki, (pre - bs) as Ki));
            }
        }

        let u_start = out.u_rowidx.len();

        for &u in topo.iter().rev() {
            let u = u as usize;
            let p = mark[u][1];
            let xu = x[u];
            x[u] = T::zero();
            out.u_rowidx.push(p);
            out.u_val.push(xu);
            let nxu = T::zero() - xu;
            let p = p as usize;
            for k in out.l_colptr[p]..out.l_colptr[p + 1] {
                debug_assert!(k < out.l_rowidx.len());
                let lr = unsafe { *out.l_rowidx.get_unchecked(k) } as usize;
                debug_assert!(lr < bn);
                // https://github.com/milanofthe/rslab/issues/290
                unsafe {
                    *x.get_unchecked_mut(lr) =
                        fmadd(nxu, *out.l_val.get_unchecked(k), *x.get_unchecked(lr));
                }
            }
        }

        let mut piv = UNSET;
        let mut maxmag = 0.0f64;
        for &np in nonpiv.iter() {
            let m = x[np as usize].magnitude();
            if m > maxmag {
                maxmag = m;
                piv = np as usize;
            }
        }
        if piv == UNSET || maxmag == 0.0 || !maxmag.is_finite() {
            return Err(RslabError::SingularBasis { column: c });
        }
        let d = if diag_row.is_empty() {
            lj
        } else {
            diag_row[lj] as usize
        };
        if mark[d][0] == stamp && mark[d][1] == KI_UNSET {
            let dm = x[d].magnitude();
            if dm > 0.0 && dm >= settings.pivot_threshold * maxmag {
                piv = d;
            }
        }
        if piv != d {
            if diag_row.is_empty() {
                diag_row = (0..bn as Ki).collect();
                diag_col = (0..bn as Ki).collect();
            }
            debug_assert!(mark[d][1] == KI_UNSET);
            let k2 = diag_col[piv];
            debug_assert!(k2 == KI_UNSET || k2 as usize > lj);
            if k2 != KI_UNSET {
                diag_row[k2 as usize] = d as Ki;
                diag_col[d] = k2;
            }
            diag_col[piv] = KI_UNSET;
        }

        let dval = x[piv];
        x[piv] = T::zero();
        mark[piv][1] = lj as Ki;
        out.udiag.push(dval);
        // https://github.com/milanofthe/rslab/issues/290
        for &np in nonpiv.iter() {
            let np = np as usize;
            if np == piv {
                continue;
            }
            out.l_rowidx.push(np as Ki);
            out.l_val.push(x[np] / dval);
            x[np] = T::zero();
        }
        out.l_colptr.push(out.l_rowidx.len());
        out.u_colptr.push(out.u_rowidx.len());
        out.f_colptr.push(out.f_pre.len());

        let pivk = piv as Ki;
        for ui in u_start..out.u_rowidx.len() {
            let s = out.u_rowidx[ui] as usize;
            if prune_end[s] != KI_UNSET {
                continue;
            }
            let (cs, ce) = (out.l_colptr[s], out.l_colptr[s + 1]);
            if !out.l_rowidx[cs..ce].contains(&pivk) {
                continue;
            }
            let mut head = cs;
            for k in cs..ce {
                let r = out.l_rowidx[k] as usize;
                if mark[r][1] != KI_UNSET {
                    out.l_rowidx.swap(head, k);
                    out.l_val.swap(head, k);
                    head += 1;
                }
            }
            prune_end[s] = head as Ki;
        }
    }

    for m in mark[..bn].iter() {
        debug_assert_ne!(m[1], KI_UNSET);
    }
    for (lb, m) in mark[..bn].iter().enumerate() {
        out.fin_abs[lb] = (bs as Ki) + m[1];
    }
    for ri in out.l_rowidx.iter_mut() {
        *ri = out.fin_abs[*ri as usize];
    }
    for ri in out.u_rowidx.iter_mut() {
        *ri += bs as Ki;
    }
    // https://github.com/milanofthe/rslab/issues/290
    for e in out.prog_in.iter_mut() {
        e.1 = out.fin_abs[e.1 as usize];
    }
    Ok(())
}

pub(super) fn factor_impl<T: Scalar>(
    sym: &KluSymbolic,
    a: &GeneralCsc<T>,
    settings: &KluSettings,
) -> Result<KluFactors<T>, RslabError> {
    a.validate()?;
    let n = sym.n;
    if a.n != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: a.n,
        });
    }
    if a.nnz() != sym.nnz {
        return Err(pattern_mismatch());
    }
    if n as u64 >= KI_FBIT as u64 || sym.nnz as u64 >= KI_FBIT as u64 {
        return Err(RslabError::InvalidInput(
            "klu: dimension/nnz exceeds the 31-bit index range of this path".to_string(),
        ));
    }

    let rs_inv = row_scale_inv(a, settings.row_scaling);
    let mut pinv_pre = vec![0 as Ki; n];
    for (k, &r) in sym.pre_row_perm.iter().enumerate() {
        pinv_pre[r] = k as Ki;
    }

    let nblocks = sym.block_ptr.len() - 1;
    let parallel = sym.parallel_blocks::<T>(settings);
    let max_bn = sym.max_block_size();
    let rate = FACTOR_RATE.of::<T>();
    let entries = sym.nnz as f64;
    let forced = settings.parallel == KluParallel::On;
    let side = parallel;

    let mut l_colptr: Vec<usize> = Vec::with_capacity(n + 1);
    l_colptr.push(0);
    let mut u_colptr: Vec<usize> = Vec::with_capacity(n + 1);
    u_colptr.push(0);
    let mut f_colptr: Vec<usize> = Vec::with_capacity(n + 1);
    f_colptr.push(0);
    let mut udiag: Vec<T> = Vec::with_capacity(n);
    let mut l_rowidx: Vec<Ki> = Vec::new();
    let mut l_val: Vec<T> = Vec::new();
    let mut u_rowidx: Vec<Ki> = Vec::new();
    let mut u_val: Vec<T> = Vec::new();
    let mut f_rowidx: Vec<Ki> = Vec::new();
    let mut f_val: Vec<T> = Vec::new();
    let mut scatter_expect = vec![0 as Ki; sym.nnz];
    let mut scatter_target = vec![0 as Ki; sym.nnz];
    let mut fin_of_pre = vec![0 as Ki; n];

    // https://github.com/milanofthe/rslab/issues/290
    if !side {
        if nblocks > 1 {
            l_rowidx.reserve(sym.nnz * KLU_L_RESERVE);
            l_val.reserve(sym.nnz * KLU_L_RESERVE);
            u_rowidx.reserve(sym.nnz * KLU_U_RESERVE);
            u_val.reserve(sym.nnz * KLU_U_RESERVE);
            f_rowidx.reserve(sym.nnz / KLU_F_RESERVE_DIV);
            f_val.reserve(sym.nnz / KLU_F_RESERVE_DIV);
        }
        let mut scratch = KluScratch::<T>::new(max_bn);
        let mut out = BlockOut::<T>::default();
        for b in 0..nblocks {
            settings.interrupted()?;
            let bs = sym.block_ptr[b];
            let be = sym.block_ptr[b + 1];
            factor_block(
                sym,
                a,
                settings,
                &rs_inv,
                &pinv_pre,
                bs,
                be,
                &mut scratch,
                &mut out,
            )?;
            fin_of_pre[bs..be].copy_from_slice(&out.fin_abs);
            let (lo, uo, fo) = (l_rowidx.len(), u_rowidx.len(), f_rowidx.len());
            if nblocks == 1 {
                l_rowidx = std::mem::take(&mut out.l_rowidx);
                l_val = std::mem::take(&mut out.l_val);
                u_rowidx = std::mem::take(&mut out.u_rowidx);
                u_val = std::mem::take(&mut out.u_val);
                l_colptr = std::mem::take(&mut out.l_colptr);
                u_colptr = std::mem::take(&mut out.u_colptr);
                udiag = std::mem::take(&mut out.udiag);
            } else {
                l_rowidx.extend_from_slice(&out.l_rowidx);
                l_val.extend_from_slice(&out.l_val);
                u_rowidx.extend_from_slice(&out.u_rowidx);
                u_val.extend_from_slice(&out.u_val);
                l_colptr.extend(out.l_colptr[1..].iter().map(|&p| lo + p));
                u_colptr.extend(out.u_colptr[1..].iter().map(|&p| uo + p));
                udiag.extend_from_slice(&out.udiag);
            }
            f_rowidx.extend(out.f_pre.iter().map(|&pre| fin_of_pre[pre as usize]));
            f_val.extend_from_slice(&out.f_val);
            f_colptr.extend(out.f_colptr[1..].iter().map(|&p| fo + p));
            for (i, &k) in out.f_k.iter().enumerate() {
                scatter_expect[k as usize] = f_rowidx[fo + i];
                scatter_target[k as usize] = KI_FBIT | (fo + i) as Ki;
            }
            for &(k, fin) in &out.prog_in {
                scatter_expect[k as usize] = fin;
                scatter_target[k as usize] = fin;
            }
        }
    } else {
        // https://github.com/milanofthe/rslab/issues/290
        let blocks: Vec<Result<BlockOut<T>, RslabError>> =
            crate::parallel::map_init::<crate::parallel::Rayon, _, _>(
                nblocks,
                if forced {
                    Work::ns(f64::INFINITY)
                } else {
                    Work::of(entries, rate)
                },
                || KluScratch::<T>::new(max_bn),
                |scratch, b| {
                    settings.interrupted()?;
                    let mut out = BlockOut::<T>::default();
                    factor_block(
                        sym,
                        a,
                        settings,
                        &rs_inv,
                        &pinv_pre,
                        sym.block_ptr[b],
                        sym.block_ptr[b + 1],
                        scratch,
                        &mut out,
                    )?;
                    Ok(out)
                },
            );

        let outs: Vec<BlockOut<T>> = {
            let mut v = Vec::with_capacity(nblocks);
            for o in blocks {
                v.push(o?);
            }
            v
        };
        let mut l_off = vec![0usize; nblocks + 1];
        let mut u_off = vec![0usize; nblocks + 1];
        let mut f_off = vec![0usize; nblocks + 1];
        for (b, out) in outs.iter().enumerate() {
            l_off[b + 1] = l_off[b] + out.l_rowidx.len();
            u_off[b + 1] = u_off[b] + out.u_rowidx.len();
            f_off[b + 1] = f_off[b] + out.f_pre.len();
        }

        for (b, out) in outs.iter().enumerate() {
            let bs = sym.block_ptr[b];
            fin_of_pre[bs..bs + out.fin_abs.len()].copy_from_slice(&out.fin_abs);
        }

        l_rowidx = vec![0; l_off[nblocks]];
        l_val = vec![T::zero(); l_off[nblocks]];
        u_rowidx = vec![0; u_off[nblocks]];
        u_val = vec![T::zero(); u_off[nblocks]];
        f_rowidx = vec![0; f_off[nblocks]];
        f_val = vec![T::zero(); f_off[nblocks]];
        {
            struct Chunks<'s, T> {
                l_ri: &'s mut [Ki],
                l_v: &'s mut [T],
                u_ri: &'s mut [Ki],
                u_v: &'s mut [T],
                f_ri: &'s mut [Ki],
                f_v: &'s mut [T],
            }
            let mut jobs: Vec<(Chunks<'_, T>, &BlockOut<T>)> = Vec::with_capacity(nblocks);
            let (mut lri, mut lv) = (l_rowidx.as_mut_slice(), l_val.as_mut_slice());
            let (mut uri, mut uv) = (u_rowidx.as_mut_slice(), u_val.as_mut_slice());
            let (mut fri, mut fv) = (f_rowidx.as_mut_slice(), f_val.as_mut_slice());
            for out in &outs {
                let (a1, r1) = std::mem::take(&mut lri).split_at_mut(out.l_rowidx.len());
                lri = r1;
                let (a2, r2) = std::mem::take(&mut lv).split_at_mut(out.l_val.len());
                lv = r2;
                let (a3, r3) = std::mem::take(&mut uri).split_at_mut(out.u_rowidx.len());
                uri = r3;
                let (a4, r4) = std::mem::take(&mut uv).split_at_mut(out.u_val.len());
                uv = r4;
                let (a5, r5) = std::mem::take(&mut fri).split_at_mut(out.f_pre.len());
                fri = r5;
                let (a6, r6) = std::mem::take(&mut fv).split_at_mut(out.f_val.len());
                fv = r6;
                jobs.push((
                    Chunks {
                        l_ri: a1,
                        l_v: a2,
                        u_ri: a3,
                        u_v: a4,
                        f_ri: a5,
                        f_v: a6,
                    },
                    out,
                ));
            }
            let fin = &fin_of_pre;
            let copy_one = |(c, out): (Chunks<'_, T>, &BlockOut<T>)| {
                c.l_ri.copy_from_slice(&out.l_rowidx);
                c.l_v.copy_from_slice(&out.l_val);
                c.u_ri.copy_from_slice(&out.u_rowidx);
                c.u_v.copy_from_slice(&out.u_val);
                for (dst, &pre) in c.f_ri.iter_mut().zip(&out.f_pre) {
                    *dst = fin[pre as usize];
                }
                c.f_v.copy_from_slice(&out.f_val);
            };
            let copied = (l_off[nblocks] + u_off[nblocks] + f_off[nblocks]) as f64;
            crate::parallel::for_each_owned(jobs, Work::of(copied, &SPLICE_RATE), copy_one);
        }

        for (b, out) in outs.iter().enumerate() {
            l_colptr.extend(out.l_colptr[1..].iter().map(|&p| l_off[b] + p));
            u_colptr.extend(out.u_colptr[1..].iter().map(|&p| u_off[b] + p));
            f_colptr.extend(out.f_colptr[1..].iter().map(|&p| f_off[b] + p));
            udiag.extend_from_slice(&out.udiag);
            for (i, &k) in out.f_k.iter().enumerate() {
                let k = k as usize;
                scatter_expect[k] = f_rowidx[f_off[b] + i];
                scatter_target[k] = KI_FBIT | (f_off[b] + i) as Ki;
            }
            for &(k, fin) in &out.prog_in {
                scatter_expect[k as usize] = fin;
                scatter_target[k as usize] = fin;
            }
        }
    }

    let mut row_perm = vec![0usize; n];
    let mut pinv = vec![0usize; n];
    for (&fin, &orig) in fin_of_pre.iter().zip(&sym.pre_row_perm) {
        row_perm[fin as usize] = orig;
        pinv[orig] = fin as usize;
    }

    let (sn_last, u_dense) = super::supernode::layout(
        &sym.block_ptr,
        &l_colptr,
        &mut l_rowidx,
        &mut l_val,
        &u_colptr,
        &mut u_rowidx,
        &mut u_val,
    );
    let l_widest = (0..n)
        .map(|j| l_colptr[j + 1] - l_colptr[j])
        .max()
        .unwrap_or(0);

    let plan = compute_replay_plan(&sym.block_ptr, &l_colptr, &u_colptr, &u_rowidx, settings);

    Ok(KluFactors {
        interrupt: settings.interrupt.clone(),
        threads: settings.threads,
        n,
        nnz_a: sym.nnz,
        block_ptr: sym.block_ptr.clone(),
        row_perm,
        pinv,
        col_perm: sym.col_perm.clone(),
        rs_inv,
        scaled: settings.row_scaling,
        l_colptr,
        l_rowidx,
        l_val,
        u_colptr,
        u_rowidx,
        u_val,
        udiag,
        sn_last,
        u_dense,
        l_widest,
        f_colptr,
        f_rowidx,
        f_val,
        scatter_expect,
        scatter_target,
        plan,
        rs_next: Vec::new(),
        replay_x: Vec::new(),
        replay_xs: Vec::new(),
        ready: Ready::default(),
        modes: Vec::new(),
    })
}
