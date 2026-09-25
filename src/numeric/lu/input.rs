//! The LU input stage: the permutation program that splits the matched,
//! permuted matrix into each supernode's column part and `U12` row part.

use crate::numeric::supernodal::Li;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;

/// The permuted input of the numeric phase, split the way the supernodes read
/// it: entry `(i, j)` of `P^T B P` (`B` the matrix factored, `A` or its
/// row-matched form) goes to the columns of `j`'s supernode if row `i` lies in
/// its diagonal block or below, else to the `U12` rows of `i`'s supernode (`j`
/// then lies past it). Every entry is read once, so the values are one array
/// of `nnz`: the column part `[..split]` (by column: `col_ptr`, `row_idx`),
/// then the row part (by row: `row_ptr`, `col_idx`, offsets from `split`).
/// The structure is fixed per analysis; a factorization scatters its values
/// through `pos` (entry `k` of `A` to slot `pos[k]`), row matching included.
pub(super) struct LuScatter {
    pub(super) col_ptr: Vec<usize>,
    pub(super) row_idx: Vec<Li>,
    pub(super) row_ptr: Vec<usize>,
    pub(super) col_idx: Vec<Li>,
    pub(super) pos: Vec<usize>,
}

impl LuScatter {
    /// `b_row[r]`: the row of `B` that row `r` of `A` becomes (`None`: `B = A`).
    pub(super) fn build(
        a: &GeneralCsc<impl Scalar>,
        b_row: Option<&[usize]>,
        sym: &SymbolicFactorization,
    ) -> Self {
        let n = a.n;
        let mut first = vec![0usize; n];
        for sn in &sym.supernodes {
            first[sn.first_col..sn.first_col + sn.ncol].fill(sn.first_col);
        }
        let at = |j: usize, k: usize| {
            let r = a.row_idx[k];
            (sym.perm_inv[b_row.map_or(r, |b| b[r])], sym.perm_inv[j])
        };
        // Count per target column / row, place `(index, entry)` pairs, sort each
        // column / row by index (the assembly walks them in that order).
        let (mut col_ptr, mut row_ptr) = (vec![0usize; n + 1], vec![0usize; n + 1]);
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let (gi, gj) = at(j, k);
                if gi >= first[gj] {
                    col_ptr[gj + 1] += 1;
                } else {
                    row_ptr[gi + 1] += 1;
                }
            }
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
            row_ptr[c + 1] += row_ptr[c];
        }
        let split = col_ptr[n];
        let mut pairs: Vec<(Li, usize)> = vec![(0, 0); a.row_idx.len()];
        let (mut cc, mut rc) = (col_ptr[..n].to_vec(), row_ptr[..n].to_vec());
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let (gi, gj) = at(j, k);
                if gi >= first[gj] {
                    pairs[cc[gj]] = (gi as Li, k);
                    cc[gj] += 1;
                } else {
                    pairs[split + rc[gi]] = (gj as Li, k);
                    rc[gi] += 1;
                }
            }
        }
        drop((first, cc, rc));
        let mut pos = vec![0usize; pairs.len()];
        let mut idx = |ptr: &[usize], base: usize| -> Vec<Li> {
            let mut out = vec![0 as Li; ptr[n]];
            for c in 0..n {
                let run = &mut pairs[base + ptr[c]..base + ptr[c + 1]];
                run.sort_unstable_by_key(|&(g, _)| g);
                for (p, &(g, k)) in (ptr[c]..ptr[c + 1]).zip(run.iter()) {
                    out[p] = g;
                    pos[k] = base + p;
                }
            }
            out
        };
        let row_idx = idx(&col_ptr, 0);
        let col_idx = idx(&row_ptr, split);
        LuScatter {
            col_ptr,
            row_idx,
            row_ptr,
            col_idx,
            pos,
        }
    }
}

/// One factorization's permuted input: the [`LuScatter`] structure and its values.
#[derive(Clone, Copy)]
pub(super) struct LuInput<'a, T> {
    pub(super) sc: &'a LuScatter,
    pub(super) vals: &'a [T],
}

impl<'a, T: Scalar> LuInput<'a, T> {
    /// Rows and values of column `c` in its supernode's columns (row `>=` the
    /// supernode's first column).
    #[inline]
    pub(super) fn col(self, c: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let r = self.sc.col_ptr[c]..self.sc.col_ptr[c + 1];
        let (idx, vals) = (&self.sc.row_idx[r.clone()], &self.vals[r]);
        idx.iter().zip(vals).map(|(&g, &v)| (g as usize, v))
    }

    /// Columns and values of row `r` in its supernode's `U12` (column past the
    /// supernode).
    #[inline]
    pub(super) fn row(self, r: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let split = self.sc.col_ptr[self.sc.col_ptr.len() - 1];
        let k = split + self.sc.row_ptr[r]..split + self.sc.row_ptr[r + 1];
        let idx = &self.sc.col_idx[self.sc.row_ptr[r]..self.sc.row_ptr[r + 1]];
        idx.iter()
            .zip(&self.vals[k])
            .map(|(&g, &v)| (g as usize, v))
    }
}
