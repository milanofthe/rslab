//! The input stage shared by the LDL^T and LU factorizations: the frozen
//! program that reads a value set of `A` in the factorization order, and the
//! view the supernodes assemble from.
//!
//! The permuted structure is a pure function of the analyzed pattern and the
//! permutation, so it is built once per analysis; every (re)factorization
//! then reads `A`'s values where they are, through `src` (the scaling
//! applied on the way), with no counting sort, no per-column sorting, no copy
//! of the indices and no permuted copy of the values.

use super::Li;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;

/// Where the entries of the permuted matrix live: a column part (by column:
/// `col_ptr`, its indices `idx[..split]`) and, on the LU path, a row part (by
/// row: `row_ptr`, its indices from `split`). Indices are sorted within each
/// column and row, the order the assembly walks them in. A slot is a place
/// in `idx`.
pub(crate) struct InputProgram {
    n: usize,
    col_ptr: Vec<usize>,
    row_ptr: Vec<usize>,
    /// The row of a slot in the column part, the column of one in the row
    /// part.
    idx: Vec<Li>,
    /// `src[p]`: the entry of `A` (its place in `A`'s values) at slot `p`.
    src: Vec<Li>,
}

impl InputProgram {
    /// Heap bytes held by the program.
    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.col_ptr)
            + vec_bytes(&self.row_ptr)
            + vec_bytes(&self.idx)
            + vec_bytes(&self.src)
    }

    /// The symmetric fold of a lower triangle (LDL^T): entry `(i, j)` lands
    /// at `(max(gi, gj), min(gi, gj))` of `P^T A P`, `g = perm_inv[*]`.
    pub fn symmetric(col_ptr: &[usize], row_idx: &[usize], perm_inv: &[usize]) -> Self {
        Self::build(col_ptr, row_idx, |i, j| {
            let (gi, gj) = (perm_inv[i], perm_inv[j]);
            Part::Col {
                col: gi.min(gj),
                row: gi.max(gj),
            }
        })
    }

    /// The split of a general matrix (LU): entry `(i, j)` of `P^T B P` (`B`
    /// the matrix factored; row `r` of `A` is row `row_map[r]` of `B`) goes to
    /// the columns of `j`'s supernode when its row lies in that supernode's
    /// diagonal block or below, else to the `U12` rows of `i`'s supernode.
    pub fn general(
        col_ptr: &[usize],
        row_idx: &[usize],
        row_map: Option<&[usize]>,
        sym: &SymbolicFactorization,
    ) -> Self {
        let mut first = vec![0usize; sym.n];
        for sn in &sym.supernodes {
            first[sn.first_col..sn.first_col + sn.ncol].fill(sn.first_col);
        }
        Self::build(col_ptr, row_idx, |i, j| {
            let (gi, gj) = (sym.perm_inv[row_map.map_or(i, |m| m[i])], sym.perm_inv[j]);
            if gi >= first[gj] {
                Part::Col { col: gj, row: gi }
            } else {
                Part::Row { row: gi, col: gj }
            }
        })
    }

    fn build(col_ptr: &[usize], row_idx: &[usize], place: impl Fn(usize, usize) -> Part) -> Self {
        let n = col_ptr.len() - 1;
        let nnz = row_idx.len();
        assert!(
            nnz <= Li::MAX as usize,
            "the input program requires fewer than 2^32 entries"
        );
        let entries = || (0..n).flat_map(|j| (col_ptr[j]..col_ptr[j + 1]).map(move |k| (j, k)));
        // Count per target column / row, place every entry's index and its
        // place in `A` at its slot, then sort each column / row by index.
        let (mut cptr, mut rptr) = (vec![0usize; n + 1], vec![0usize; n + 1]);
        for (j, k) in entries() {
            match place(row_idx[k], j) {
                Part::Col { col, .. } => cptr[col + 1] += 1,
                Part::Row { row, .. } => rptr[row + 1] += 1,
            }
        }
        for c in 0..n {
            cptr[c + 1] += cptr[c];
            rptr[c + 1] += rptr[c];
        }
        let split = cptr[n];
        let mut idx = vec![0 as Li; nnz];
        let mut src = vec![0 as Li; nnz];
        let (mut cc, mut rc) = (cptr[..n].to_vec(), rptr[..n].to_vec());
        for (j, k) in entries() {
            let (p, g) = match place(row_idx[k], j) {
                Part::Col { col, row } => {
                    cc[col] += 1;
                    (cc[col] - 1, row)
                }
                Part::Row { row, col } => {
                    rc[row] += 1;
                    (split + rc[row] - 1, col)
                }
            };
            idx[p] = g as Li;
            src[p] = k as Li;
        }
        drop((cc, rc));
        // (the indices of a run are distinct: the unstable sort is
        // deterministic)
        let mut run: Vec<(Li, Li)> = Vec::new();
        let mut sort = |a: usize, b: usize| {
            run.clear();
            run.extend(idx[a..b].iter().copied().zip(src[a..b].iter().copied()));
            run.sort_unstable_by_key(|&(g, _)| g);
            for (p, &(g, k)) in (a..b).zip(&run) {
                idx[p] = g;
                src[p] = k;
            }
        };
        for c in 0..n {
            sort(cptr[c], cptr[c + 1]);
            sort(split + rptr[c], split + rptr[c + 1]);
        }
        InputProgram {
            n,
            col_ptr: cptr,
            row_ptr: rptr,
            idx,
            src,
        }
    }
}

/// Target of one entry of `A` in the permuted matrix.
enum Part {
    Col { col: usize, row: usize },
    Row { row: usize, col: usize },
}

/// One factorization's permuted input: the program over `A`'s values and
/// the scaling the factorization applies to them.
#[derive(Clone, Copy)]
pub(crate) struct Input<'a, T> {
    prog: &'a InputProgram,
    /// `A`'s values, in `A`'s order.
    vals: &'a [T],
    /// The row and the column factor of the scaling by elimination index:
    /// entry `(g, c)` of the permuted matrix is `A`'s times `row[g] * col[c]`.
    scale: Option<(&'a [f64], &'a [f64])>,
}

impl<'a, T: Scalar> Input<'a, T> {
    pub fn new(
        prog: &'a InputProgram,
        vals: &'a [T],
        scale: Option<(&'a [f64], &'a [f64])>,
    ) -> Self {
        debug_assert_eq!(vals.len(), prog.src.len());
        Input { prog, vals, scale }
    }

    /// The value at slot `p`, entry `(row, col)` of the permuted matrix.
    #[inline]
    fn value(self, p: usize, row: usize, col: usize) -> T {
        let v = self.vals[self.prog.src[p] as usize];
        match self.scale {
            None => v,
            Some((r, c)) => v * T::from_real(r[row] * c[col]),
        }
    }

    /// Every value, in no particular order.
    pub fn values(self) -> impl Iterator<Item = T> + 'a {
        (0..self.prog.n).flat_map(move |c| self.col(c).chain(self.row(c)).map(|(_, v)| v))
    }

    /// Rows and values of column `c` (in its supernode's columns, rows
    /// ascending).
    #[inline]
    pub fn col(self, c: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        (self.prog.col_ptr[c]..self.prog.col_ptr[c + 1]).map(move |p| {
            let g = self.prog.idx[p] as usize;
            (g, self.value(p, g, c))
        })
    }

    /// Columns and values of row `r` in its supernode's `U12` (columns past
    /// the supernode, ascending); empty for the symmetric fold.
    #[inline]
    pub fn row(self, r: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let split = self.prog.col_ptr[self.prog.n];
        (split + self.prog.row_ptr[r]..split + self.prog.row_ptr[r + 1]).map(move |p| {
            let g = self.prog.idx[p] as usize;
            (g, self.value(p, r, g))
        })
    }
}
