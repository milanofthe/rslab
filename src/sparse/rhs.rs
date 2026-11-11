//! Sparse right-hand sides: an `n x ncols` block stored by columns, for
//! [`LdltSolver::port_matrix`](crate::LdltSolver::port_matrix), which visits
//! only the part of the factor a right-hand side reaches.

use crate::error::RslabError;
use crate::scalar::Scalar;

/// An `n x ncols` block of sparse columns in compressed-column form: column
/// `c` holds the rows `row_idx[col_ptr[c]..col_ptr[c + 1]]` with their
/// values. Rows need not be sorted; a row listed twice in a column counts
/// with the sum of its values; a column may be empty.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseRhs<T = f64> {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T: Scalar> SparseRhs<T> {
    /// The block from its compressed-column arrays, checked: `col_ptr`
    /// starts at `0`, never decreases and ends at the length of `row_idx`
    /// and `values`, and every row is below `n`.
    pub fn new(
        n: usize,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, RslabError> {
        let ends_right = col_ptr.first() == Some(&0) && col_ptr.last() == Some(&row_idx.len());
        if !ends_right || col_ptr.windows(2).any(|p| p[0] > p[1]) {
            return Err(RslabError::InvalidInput(
                "sparse right-hand side: col_ptr must start at 0, not decrease and end at the entry count"
                    .to_string(),
            ));
        }
        if values.len() != row_idx.len() {
            return Err(RslabError::DimensionMismatch {
                expected: row_idx.len(),
                got: values.len(),
            });
        }
        if let Some(&r) = row_idx.iter().find(|&&r| r >= n) {
            return Err(RslabError::InvalidInput(format!(
                "sparse right-hand side: row {r} out of range for n = {n}"
            )));
        }
        Ok(SparseRhs {
            n,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// The block from one `(row, value)` list per column.
    pub fn from_columns(n: usize, columns: &[Vec<(usize, T)>]) -> Result<Self, RslabError> {
        let mut col_ptr = Vec::with_capacity(columns.len() + 1);
        col_ptr.push(0);
        let (mut row_idx, mut values) = (Vec::new(), Vec::new());
        for c in columns {
            for &(r, v) in c {
                row_idx.push(r);
                values.push(v);
            }
            col_ptr.push(row_idx.len());
        }
        Self::new(n, col_ptr, row_idx, values)
    }

    /// Rows of the block.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Columns of the block.
    pub fn ncols(&self) -> usize {
        self.col_ptr.len() - 1
    }

    /// Stored entries.
    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    /// Rows and values of column `c`.
    pub fn column(&self, c: usize) -> (&[usize], &[T]) {
        let (a, b) = (self.col_ptr[c], self.col_ptr[c + 1]);
        (&self.row_idx[a..b], &self.values[a..b])
    }

    /// The block as a dense column-major `n x ncols` array (`x[c * n + i]`),
    /// duplicates summed.
    pub fn to_dense(&self) -> Vec<T> {
        let mut x = vec![T::zero(); self.n * self.ncols()];
        for c in 0..self.ncols() {
            let (rows, vals) = self.column(c);
            for (&r, &v) in rows.iter().zip(vals) {
                x[c * self.n + r] = x[c * self.n + r] + v;
            }
        }
        x
    }

    /// `B^T X` (the plain transpose) for a dense column-major `n x k` block
    /// `x`: the `ncols x k` result column-major.
    pub fn transpose_times(&self, x: &[T], k: usize) -> Result<Vec<T>, RslabError> {
        if x.len() != self.n * k {
            return Err(RslabError::DimensionMismatch {
                expected: self.n * k,
                got: x.len(),
            });
        }
        let p = self.ncols();
        let mut out = vec![T::zero(); p * k];
        for j in 0..k {
            for c in 0..p {
                let (rows, vals) = self.column(c);
                let mut s = T::zero();
                for (&r, &v) in rows.iter().zip(vals) {
                    s = s + v * x[j * self.n + r];
                }
                out[j * p + c] = s;
            }
        }
        Ok(out)
    }
}
