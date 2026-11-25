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
use crate::constants::KLU_SN_PASS;
use crate::scalar::{fmadd, Scalar};

/// Whether a supernode of `w` columns over `s` shared rows is applied by the
/// dense kernel: where [`apply`] reads and writes memory fewer times than
/// the column-by-column replay for a full application. With
/// `t = w (w - 1) / 2` entries in its triangle and `w s` below it:
///
/// ```text
/// column by column  4 (t + w s)          index, L value, x read, x write
/// dense kernel      3 t                  L value, x read, x write
///                 + w s                  L value
///                 + 2 s (w / P + w % P)  `tmp` read and write a pass
///                 + 7 s                  gather: index, x read, tmp write;
///                                        scatter: index, tmp read, x write,
///                                        tmp clear
/// ```
///
/// with `P` = [`KLU_SN_PASS`] columns a full pass, the rest one a pass: a
/// supernode of three columns never pays, one of four or more always does,
/// one of two only without shared rows.
fn dense_pays(w: usize, s: usize) -> bool {
    let (t, p) = (w * (w - 1) / 2, KLU_SN_PASS);
    let column = 4 * (t + w * s); // lit: index, L value, x read, x write
    let dense = 3 * t + w * s + 2 * s * (w / p + w % p) + 7 * s; // lit: the accesses counted above
    dense < column
}

/// The supernodes of the factor's `L` the dense kernel applies
/// ([`dense_pays`]) and the storage order that applies them densely: the
/// rows of every `L` column put in the order above and the rows of every
/// `U` column that applies one ascending (values moved with them). Returns
/// per column the last column of its supernode, and per column whether it
/// reads two columns of one supernode or more (applies the dense kernel at
/// all).
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
    let mut last: Vec<Ki> = (0..n as Ki).collect();
    let mut mark = vec![false; n];
    let mut pairs: Vec<(Ki, T)> = Vec::new();
    // `rank[row]`: the position of `row` in the shared rows of the
    // current supernode.
    let mut rank = vec![0 as Ki; n];
    for b in 0..block_ptr.len() - 1 {
        let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
        // the supernodes, from the last column of the block backwards
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
                    last[c] = last[c + 1];
                }
            }
        }
        // the rows of every column of a supernode worth the kernel in the
        // shared order; another one is taken apart into single columns
        let mut c = bs;
        while c < be {
            let q = last[c] as usize;
            if !dense_pays(q + 1 - c, l_colptr[q + 1] - l_colptr[q]) {
                for (l, k) in last[c..=q].iter_mut().zip(c..) {
                    *l = k as Ki;
                }
            } else {
                for (i, &r) in l_rowidx[l_colptr[q]..l_colptr[q + 1]].iter().enumerate() {
                    rank[r as usize] = (q - c + 1 + i) as Ki;
                }
                for p in c..q {
                    let (lo, hi) = (l_colptr[p], l_colptr[p + 1]);
                    pairs.clear();
                    pairs.extend((lo..hi).map(|k| (l_rowidx[k], l_val[k])));
                    // rows inside the supernode first (by row), then `S`
                    pairs.sort_unstable_by_key(|&(r, _)| {
                        if (r as usize) <= q {
                            r as usize - c
                        } else {
                            rank[r as usize] as usize
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
    // A column that reads two columns of one supernode or more takes its
    // `U` rows ascending, so the columns of a supernode it reads are
    // consecutive; another keeps the factor's topological order, which
    // spaces dependent updates apart.
    let dense: Vec<bool> = (0..n)
        .map(|j| {
            (u_rowidx[u_colptr[j]..u_colptr[j + 1]].iter())
                .any(|&p| (last[p as usize] as usize).min(j - 1) > p as usize)
        })
        .collect();
    for j in (0..n).filter(|&j| dense[j]) {
        let (lo, hi) = (u_colptr[j], u_colptr[j + 1]);
        pairs.clear();
        pairs.extend((lo..hi).map(|k| (u_rowidx[k], u_val[k])));
        pairs.sort_unstable_by_key(|&(r, _)| r);
        for (k, &(r, v)) in (lo..hi).zip(&pairs) {
            u_rowidx[k] = r;
            u_val[k] = v;
        }
    }
    (last, dense)
}

/// Applies the columns `r ..= e` of one supernode to the work vector `x`
/// (row `i` at `x[i - off]`): the triangle `x[c + 1 ..= e] -= L(c + 1 ..= e,
/// c) x[c]` column after column, then `x[R] -= L(R, r ..= e) x[r ..= e]`
/// over the shared rows `R` (the rows of `L(:, e)`), gathered into `tmp`.
/// A triangle column whose `x[c]` is zero adds nothing and is passed over.
/// The `L` value at storage position `k` is at `lv + k - lbase`. `tmp` is
/// left all-zero, as it is found.
///
/// # Safety
/// The rows of `L(:, e)` and `r ..= e` are in `x` (minus `off`), `tmp`
/// holds `|L(:, e)|` entries, and `lv` points at `L` values complete for
/// the columns `r ..= e`.
// Out of line: the column loop that calls it stays small.
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
    // `KLU_SN_PASS` columns a pass: `tmp` loaded and stored once for them
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

/// The elimination of column `j` (block from `bs`) in its work vector `x`
/// against the `L` columns `U(:, j)` names (slots `ks`, their rows
/// ascending): column by column, the columns `r ..= e` of one supernode
/// together by [`apply`]. Writes `U(:, j)` at `uv + k - ubase` and leaves the
/// rows read zero; `wait(lo, hi)` is called before the `L` columns
/// `lo ..= hi` are read.
///
/// # Safety
/// As [`apply`] for every supernode read; `uv` points at the `U` slots
/// `ks` of `j`, writable.
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
    let mut k = ks.start;
    while k < ks.end {
        let pr = u_rowidx[k] as usize;
        // the columns `pr ..= e` of `pr`'s supernode `j` reads, consecutive
        // in `U(:, j)`
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
