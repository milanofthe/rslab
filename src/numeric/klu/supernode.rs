//! The supernodes of `L` and the dense kernel the refactorization and the
//! solve apply them with.
//!
//! Columns `c` and `c + 1` of one block are in one supernode when the rows
//! of `L(:, c)` are `c + 1` and the rows of `L(:, c + 1)`; a supernode
//! `[p, q]` shares the rows `S` of `L(:, q)` below it. Each of its columns
//! is stored as its rows inside the supernode, ascending (`c + 1 ..= q`),
//! then `S` in the order of `L(:, q)`, so the columns `r ..= e` of a
//! supernode read the same rows `e + 1 ..= q`, `S` from their tails: one
//! gather of the work vector, a dense update over the tails, one scatter.
//!
//! A column `j` that reads column `r` of a supernode reads every column of
//! it from `r` up to `j - 1` or `q` (`L(c + 1, c)` is nonzero, so the reach
//! of `j` through `c` holds `c + 1`): with the `U` rows of the column
//! ascending, a valid elimination order since `L` is lower triangular in
//! the final positions, those columns are consecutive entries of `U(:, j)`.

use super::Ki;
use crate::constants::{
    KLU_SN_COLUMN_ACCESSES, KLU_SN_GATHER_SCATTER_ACCESSES, KLU_SN_PASS, KLU_SN_PASS_ACCESSES,
    KLU_SN_TRIANGLE_ACCESSES,
};
use crate::scalar::{fmadd, Scalar};

fn dense_pays(w: usize, s: usize) -> bool {
    let (t, p) = (w * (w - 1) / 2, KLU_SN_PASS);
    let column = KLU_SN_COLUMN_ACCESSES * (t + w * s);
    let dense = KLU_SN_TRIANGLE_ACCESSES * t
        + w * s
        + KLU_SN_PASS_ACCESSES * s * (w / p + w % p)
        + KLU_SN_GATHER_SCATTER_ACCESSES * s;
    dense < column
}

pub(super) fn layout<T: Scalar>(
    block_ptr: &[usize],
    l_colptr: &[usize],
    l_rowidx: &mut [Ki],
    l_val: &mut [T],
    u_colptr: &[usize],
    u_rowidx: &mut [Ki],
    u_val: &mut [T],
) -> (Vec<Ki>, Vec<bool>) {
    let n = l_colptr.len() - 1;
    let mut sn_last: Vec<Ki> = (0..n as Ki).collect();
    let mut mark = vec![false; n];
    let mut pairs: Vec<(Ki, T)> = Vec::new();
    let mut shared_rank = vec![0 as Ki; n];
    for b in 0..block_ptr.len() - 1 {
        let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
        let mut c = be;
        while c > bs {
            c -= 1;
            if c + 1 < be {
                let (lo, hi) = (l_colptr[c], l_colptr[c + 1]);
                let (lo1, hi1) = (l_colptr[c + 1], l_colptr[c + 2]);
                let joins = hi - lo == hi1 - lo1 + 1 && {
                    for &r in &l_rowidx[lo1..hi1] {
                        mark[r as usize] = true;
                    }
                    let inside =
                        (l_rowidx[lo..hi].iter()).all(|&r| r as usize == c + 1 || mark[r as usize]);
                    for &r in &l_rowidx[lo1..hi1] {
                        mark[r as usize] = false;
                    }
                    inside
                };
                if joins {
                    sn_last[c] = sn_last[c + 1];
                }
            }
        }
        let mut c = bs;
        while c < be {
            let q = sn_last[c] as usize;
            if !dense_pays(q + 1 - c, l_colptr[q + 1] - l_colptr[q]) {
                for (l, k) in sn_last[c..=q].iter_mut().zip(c..) {
                    *l = k as Ki;
                }
            } else {
                for (i, &r) in l_rowidx[l_colptr[q]..l_colptr[q + 1]].iter().enumerate() {
                    shared_rank[r as usize] = (q - c + 1 + i) as Ki;
                }
                for p in c..q {
                    let (lo, hi) = (l_colptr[p], l_colptr[p + 1]);
                    pairs.clear();
                    pairs.extend((lo..hi).map(|k| (l_rowidx[k], l_val[k])));
                    pairs.sort_unstable_by_key(|&(r, _)| {
                        if (r as usize) <= q {
                            r as usize - c
                        } else {
                            shared_rank[r as usize] as usize
                        }
                    });
                    for (k, &(r, v)) in (lo..hi).zip(&pairs) {
                        l_rowidx[k] = r;
                        l_val[k] = v;
                    }
                }
            }
            c = q + 1;
        }
    }
    let u_dense: Vec<bool> = (0..n)
        .map(|j| {
            (u_rowidx[u_colptr[j]..u_colptr[j + 1]].iter())
                .any(|&p| (sn_last[p as usize] as usize).min(j - 1) > p as usize)
        })
        .collect();
    for j in (0..n).filter(|&j| u_dense[j]) {
        let (lo, hi) = (u_colptr[j], u_colptr[j + 1]);
        pairs.clear();
        pairs.extend((lo..hi).map(|k| (u_rowidx[k], u_val[k])));
        pairs.sort_unstable_by_key(|&(r, _)| r);
        for (k, &(r, v)) in (lo..hi).zip(&pairs) {
            u_rowidx[k] = r;
            u_val[k] = v;
        }
    }
    (sn_last, u_dense)
}

#[inline(never)]
pub(super) unsafe fn apply<T: Scalar>(
    x: &mut [T],
    off: usize,
    tmp: &mut [T],
    l_colptr: &[usize],
    l_rowidx: &[Ki],
    (lv, lbase): (*const T, usize),
    (r, e): (usize, usize),
) {
    debug_assert!(off <= r && r <= e && e - off < x.len());
    debug_assert!(tmp.len() >= l_colptr[e + 1] - l_colptr[e]);
    debug_assert!((l_rowidx[l_colptr[e]..l_colptr[e + 1]].iter())
        .all(|&i| i as usize >= off && i as usize - off < x.len()));
    for (c, &start) in (r..e).zip(&l_colptr[r..e]) {
        let xc = T::zero() - *x.get_unchecked(c - off);
        if xc == T::zero() {
            continue;
        }
        let base = start - lbase;
        for i in 0..e - c {
            let t = x.get_unchecked_mut(c + 1 + i - off);
            *t = fmadd(xc, *lv.add(base + i), *t);
        }
    }
    let rows = &l_rowidx[l_colptr[e]..l_colptr[e + 1]];
    let m = rows.len();
    let tmp = &mut tmp[..m];
    for (t, &i) in tmp.iter_mut().zip(rows) {
        *t = *x.get_unchecked(i as usize - off);
    }
    let tail = |c: usize| lv.add(l_colptr[c] - lbase + (e - c));
    let mut c = r;
    while c + KLU_SN_PASS <= e + 1 {
        let a: [T; KLU_SN_PASS] =
            std::array::from_fn(|k| T::zero() - *x.get_unchecked(c + k - off));
        let l: [*const T; KLU_SN_PASS] = std::array::from_fn(|k| tail(c + k));
        for (i, t) in tmp.iter_mut().enumerate() {
            let mut v = *t;
            for (&ak, lk) in a.iter().zip(&l) {
                v = fmadd(ak, *lk.add(i), v);
            }
            *t = v;
        }
        c += KLU_SN_PASS;
    }
    while c <= e {
        let a = T::zero() - *x.get_unchecked(c - off);
        let l = tail(c);
        for (i, t) in tmp.iter_mut().enumerate() {
            *t = fmadd(a, *l.add(i), *t);
        }
        c += 1;
    }
    for (t, &i) in tmp.iter_mut().zip(rows) {
        *x.get_unchecked_mut(i as usize - off) = *t;
        *t = T::zero();
    }
}

#[inline(never)]
pub(super) unsafe fn eliminate<T: Scalar>(
    (x, tmp): (&mut [T], &mut [T]),
    (j, bs): (usize, usize),
    (u_rowidx, ks, sn_last): (&[Ki], std::ops::Range<usize>, &[Ki]),
    (l_colptr, l_rowidx): (&[usize], &[Ki]),
    (lv, lbase): (*const T, usize),
    (uv, ubase): (*mut T, usize),
    wait: impl Fn(usize, usize),
) {
    debug_assert!(u_rowidx[ks.clone()].windows(2).all(|w| w[0] < w[1]));
    let mut k = ks.start;
    while k < ks.end {
        let pr = u_rowidx[k] as usize;
        let e = (sn_last[pr] as usize).min(j - 1);
        wait(pr, e);
        if e > pr {
            debug_assert_eq!(u_rowidx[k + e - pr] as usize, e);
            apply(x, bs, tmp, l_colptr, l_rowidx, (lv, lbase), (pr, e));
        } else {
            let nxu = T::zero() - x[pr - bs];
            let (lo, hi) = (l_colptr[pr], l_colptr[pr + 1]);
            for (kl, &row) in (lo..hi).zip(&l_rowidx[lo..hi]) {
                let t = x.get_unchecked_mut(row as usize - bs);
                *t = fmadd(nxu, *lv.add(kl - lbase), *t);
            }
        }
        for c in pr..=e {
            *uv.add(k + c - pr - ubase) = x[c - bs];
            x[c - bs] = T::zero();
        }
        k += e - pr + 1;
    }
}
