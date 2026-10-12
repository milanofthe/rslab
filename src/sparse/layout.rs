//! Fixed entry layouts: where the entries of an assembled matrix land.
//!
//! A matrix assembled again and again over the same entry positions (a
//! Newton Jacobian, a frequency sweep, a stamped circuit) has one layout:
//! the distinct positions in compressed-column order and, for each entry in
//! the caller's order, the slot it sums into. [`CscLayout`] computes it once,
//! and every later value set is a scatter into it; [`Triplets`] is the entry
//! list it is built from.

use crate::error::RslabError;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

/// Sparse entries as parallel `(rows, cols, vals)` lists (coordinate form),
/// duplicate positions summing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Triplets<T = f64> {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<T>,
}

impl<T> Triplets<T> {
    /// No entries, room for `cap`.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            rows: Vec::with_capacity(cap),
            cols: Vec::with_capacity(cap),
            vals: Vec::with_capacity(cap),
        }
    }

    /// The number of entries (duplicates counted).
    pub fn len(&self) -> usize {
        self.vals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }

    /// Append the entry `(row, col, val)`.
    pub fn push(&mut self, row: usize, col: usize, val: T) {
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }

    /// The entries `(row, col, value)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (self.rows.iter().zip(&self.cols).zip(&self.vals)).map(|((&r, &c), v)| (r, c, v))
    }

    /// The layout of these entries in an `n x n` matrix.
    pub fn layout(&self, n: usize) -> Result<CscLayout, RslabError> {
        CscLayout::new(n, &self.rows, &self.cols)
    }
}

impl<T: Scalar> Triplets<T> {
    /// The `n x n` matrix of these entries, duplicates summed.
    pub fn to_csc(&self, n: usize) -> Result<GeneralCsc<T>, RslabError> {
        GeneralCsc::from_triplets(n, &self.rows, &self.cols, &self.vals)
    }

    /// The `n x n` dense matrix, by rows, duplicates summed in entry order.
    pub fn to_dense(&self, n: usize) -> Vec<Vec<T>> {
        let mut m = vec![vec![T::zero(); n]; n];
        for (r, c, &v) in self.iter() {
            m[r][c] = m[r][c] + v;
        }
        m
    }
}

/// The compressed-column layout of a fixed list of entry positions: the
/// distinct positions sorted by column, then row, and the slot each entry
/// (in the caller's order) sums into.
///
/// Values are given per entry and [`scatter`](Self::scatter)ed into one value
/// per slot, duplicates summed in entry order; the result is the
/// [`GeneralCsc`] of [`matrix`](Self::matrix), so the pattern work (sorting,
/// deduplication, the structural checks) is paid once per layout, not per
/// value set.
#[derive(Clone, Debug)]
pub struct CscLayout {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// Entry `k` -> its slot; duplicate positions share one.
    slot: Vec<usize>,
}

impl CscLayout {
    /// The layout of the entries at `(rows[k], cols[k])` in an `n x n`
    /// matrix. Fails if the lists differ in length or an entry lies outside
    /// the matrix.
    pub fn new(n: usize, rows: &[usize], cols: &[usize]) -> Result<Self, RslabError> {
        if rows.len() != cols.len() {
            return Err(RslabError::InvalidInput(format!(
                "CscLayout: {} rows but {} cols",
                rows.len(),
                cols.len()
            )));
        }
        if let Some((&r, &c)) = (rows.iter().zip(cols)).find(|&(&r, &c)| r >= n || c >= n) {
            return Err(RslabError::InvalidInput(format!(
                "CscLayout: index ({r}, {c}) out of bounds for {n}x{n}"
            )));
        }
        // Bucket the entries by column, then sort each column by row.
        let mut col_start = vec![0usize; n + 1];
        for &c in cols {
            col_start[c + 1] += 1;
        }
        for j in 0..n {
            col_start[j + 1] += col_start[j];
        }
        let mut order = vec![0usize; rows.len()];
        let mut next = col_start[..n].to_vec();
        for (k, &c) in cols.iter().enumerate() {
            order[next[c]] = k;
            next[c] += 1;
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        col_ptr.push(0);
        let mut row_idx = Vec::with_capacity(rows.len());
        let mut slot = vec![0usize; rows.len()];
        for j in 0..n {
            let col = &mut order[col_start[j]..col_start[j + 1]];
            col.sort_unstable_by_key(|&k| rows[k]);
            let mut prev = usize::MAX;
            for &k in col.iter() {
                if rows[k] != prev {
                    prev = rows[k];
                    row_idx.push(prev);
                }
                slot[k] = row_idx.len() - 1;
            }
            col_ptr.push(row_idx.len());
        }
        Ok(CscLayout {
            n,
            col_ptr,
            row_idx,
            slot,
        })
    }

    /// The layout of `a`'s pattern, each stored entry its own slot (the
    /// entry order is `a`'s storage order). Fails if `a` is not canonical
    /// (see [`GeneralCsc::validate`]).
    pub fn of<T: Scalar>(a: &GeneralCsc<T>) -> Result<Self, RslabError> {
        a.validate()?;
        Ok(CscLayout {
            n: a.n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            slot: (0..a.row_idx.len()).collect(),
        })
    }

    /// Heap bytes the layout holds.
    pub fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.col_ptr) + vec_bytes(&self.row_idx) + vec_bytes(&self.slot)
    }

    /// The dimension `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of distinct positions (slots).
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    /// The number of entries the layout was built from.
    pub fn entries(&self) -> usize {
        self.slot.len()
    }

    /// Column `j` occupies `col_ptr[j]..col_ptr[j + 1]` of [`row_idx`](Self::row_idx).
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// The row of each slot, ascending within a column.
    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// Entry `k` -> the slot its value sums into.
    pub fn slot(&self) -> &[usize] {
        &self.slot
    }

    /// The entry values `vals` (one per entry, the caller's order) summed
    /// into `out`, one per slot.
    pub fn scatter<T: Scalar>(&self, vals: &[T], out: &mut [T]) {
        debug_assert_eq!(vals.len(), self.slot.len());
        debug_assert_eq!(out.len(), self.row_idx.len());
        out.fill(T::zero());
        for (&s, &v) in self.slot.iter().zip(vals) {
            out[s] = out[s] + v;
        }
    }

    /// [`scatter`](Self::scatter) of values given as consecutive runs of the
    /// entry order, each times its scale, and every entry after the runs at
    /// `rest` (an augmentation such as a regularizing diagonal): the same
    /// sums in the same order as scattering the concatenated values.
    pub fn scatter_scaled<T: Scalar>(&self, parts: &[(&[T], T)], rest: T, out: &mut [T]) {
        debug_assert_eq!(out.len(), self.row_idx.len());
        out.fill(T::zero());
        let mut k = 0;
        for &(vals, scale) in parts {
            for &v in vals {
                let s = self.slot[k];
                out[s] = out[s] + v * scale;
                k += 1;
            }
        }
        for &s in &self.slot[k..] {
            out[s] = out[s] + rest;
        }
    }

    /// The layout's matrix with `values`, one per slot.
    pub fn matrix<T>(&self, values: Vec<T>) -> GeneralCsc<T> {
        debug_assert_eq!(values.len(), self.row_idx.len());
        GeneralCsc {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values,
        }
    }

    /// The layout's matrix of the entry values `vals`, duplicates summed.
    pub fn assemble<T: Scalar>(&self, vals: &[T]) -> GeneralCsc<T> {
        let mut values = vec![T::zero(); self.nnz()];
        self.scatter(vals, &mut values);
        self.matrix(values)
    }

    /// The structural rank of the pattern: the size of a maximum matching of
    /// columns onto rows over the positions. Below `n` the matrix is singular
    /// for every value set.
    pub fn structural_rank(&self) -> usize {
        crate::ordering::btf::hk_transversal(self.n, &self.col_ptr, &self.row_idx, true).1
    }

    /// For each slot `(r, c)`, the slot of `(c, r)`; `None` when a partner is
    /// missing (a structurally unsymmetric pattern).
    pub fn transpose_slots(&self) -> Option<Vec<usize>> {
        let mut tpair = vec![0usize; self.row_idx.len()];
        let (cp, ri) = (&self.col_ptr, &self.row_idx);
        for c in 0..self.n {
            let col = cp[c]..cp[c + 1];
            for (t, &r) in tpair[col.clone()].iter_mut().zip(&ri[col]) {
                *t = cp[r] + ri[cp[r]..cp[r + 1]].binary_search(&c).ok()?;
            }
        }
        Some(tpair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sorts_dedups_and_maps_entries() {
        // Entries out of order with a duplicate (0, 0).
        let rows = [1, 0, 0, 0];
        let cols = [1, 1, 0, 0];
        let l = CscLayout::new(2, &rows, &cols).unwrap();
        assert_eq!(l.col_ptr(), &[0, 1, 3]);
        assert_eq!(l.row_idx(), &[0, 0, 1]);
        assert_eq!(l.slot(), &[2, 1, 0, 0]);
        assert_eq!((l.nnz(), l.entries()), (3, 4));
        let a = l.assemble(&[3.0, 1.0, 1.5, 0.5]);
        assert_eq!(a.values, vec![2.0, 1.0, 3.0]);
        a.validate().unwrap();
    }

    #[test]
    fn scatter_scaled_matches_concatenated_scatter() {
        let rows = [0, 1, 0, 0, 1];
        let cols = [0, 1, 1, 0, 1];
        let l = CscLayout::new(2, &rows, &cols).unwrap();
        let (a, b) = ([1.0, 2.0], [3.0]);
        let mut scaled = vec![0.0; l.nnz()];
        l.scatter_scaled(&[(&a, 2.0), (&b, -1.0)], 0.25, &mut scaled);
        let mut plain = vec![0.0; l.nnz()];
        l.scatter(&[2.0, 4.0, -3.0, 0.25, 0.25], &mut plain);
        assert_eq!(scaled, plain);
    }

    #[test]
    fn layout_of_a_matrix_is_its_pattern() {
        let a = GeneralCsc::from_triplets(2, &[1, 0, 0], &[0, 0, 1], &[1.0, 2.0, 3.0]).unwrap();
        let l = CscLayout::of(&a).unwrap();
        assert_eq!(
            (l.col_ptr(), l.row_idx(), l.slot()),
            (&a.col_ptr[..], &a.row_idx[..], &[0, 1, 2][..])
        );
        assert_eq!(l.assemble(&a.values).values, a.values);
    }

    #[test]
    fn layout_rejects_bad_input() {
        assert!(CscLayout::new(2, &[0, 2], &[0, 0]).is_err());
        assert!(CscLayout::new(2, &[0], &[0, 1]).is_err());
    }

    #[test]
    fn structural_rank_and_transpose_slots() {
        // Column 1 empty: rank 1.
        let l = CscLayout::new(2, &[0, 1], &[0, 0]).unwrap();
        assert_eq!(l.structural_rank(), 1);
        assert!(l.transpose_slots().is_none());
        // Full 2x2: rank 2, slots (0,0) (1,0) (0,1) (1,1) pair 0 2 1 3.
        let l = CscLayout::new(2, &[0, 1, 0, 1], &[0, 0, 1, 1]).unwrap();
        assert_eq!(l.structural_rank(), 2);
        assert_eq!(l.transpose_slots().unwrap(), vec![0, 2, 1, 3]);
    }

    #[test]
    fn triplets_round_trip() {
        let mut t = Triplets::with_capacity(3);
        t.push(0, 0, 2.0);
        t.push(1, 0, 1.0);
        t.push(0, 0, 1.0);
        assert_eq!(t.to_dense(2), vec![vec![3.0, 0.0], vec![1.0, 0.0]]);
        let a = t.to_csc(2).unwrap();
        assert_eq!((a.row_idx, a.values), (vec![0, 1], vec![3.0, 1.0]));
    }
}
