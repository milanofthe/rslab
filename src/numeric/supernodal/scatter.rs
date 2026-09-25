//! The frozen permutation program that moves a value set into the
//! factorization order in one pass.

/// Cached scatter program for the numeric phase's permuted matrix (the KLU
/// pattern applied to the supernodal twins): the permuted CSC *structure* is a
/// pure function of the analyzed pattern and the fill-reducing permutation, so
/// it is built once per symbolic analysis; every (re)factorization then only
/// scatters the new values through `pos` - one linear pass, no counting sort,
/// no per-column sorting.
pub(crate) struct PermScatter {
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    /// `pos[k]` = slot of original entry `k` in the permuted values array.
    pub pos: Vec<usize>,
}

impl PermScatter {
    /// Build for the symmetric lower-triangle fold `P^T A P` (LDL^T path):
    /// original entry `(i, j)` lands at permuted `(max(gi, gj), min(gi, gj))`
    /// with `g = perm_inv[*]`. Columns come out row-sorted.
    pub fn build_lower(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(
            n,
            a_col_ptr,
            a_row_idx,
            |gi, gj| {
                if gi >= gj {
                    (gi, gj)
                } else {
                    (gj, gi)
                }
            },
            perm_inv,
        )
    }

    fn build_with(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        target: impl Fn(usize, usize) -> (usize, usize),
        perm_inv: &[usize],
    ) -> Self {
        let nnz = a_row_idx.len();
        // Pass 1: count entries per target column.
        let mut col_ptr = vec![0usize; n + 1];
        for (j, &gj) in perm_inv.iter().enumerate() {
            for k in a_col_ptr[j]..a_col_ptr[j + 1] {
                let (_, c) = target(perm_inv[a_row_idx[k]], gj);
                col_ptr[c + 1] += 1;
            }
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
        }
        // Pass 2: place (row, original-entry) pairs per column.
        let mut cursor = col_ptr[..n].to_vec();
        let mut pairs: Vec<(usize, usize)> = vec![(0, 0); nnz];
        for (j, &gj) in perm_inv.iter().enumerate() {
            for k in a_col_ptr[j]..a_col_ptr[j + 1] {
                let (r, c) = target(perm_inv[a_row_idx[k]], gj);
                let p = cursor[c];
                cursor[c] += 1;
                pairs[p] = (r, k);
            }
        }
        // Pass 3: sort each column by row (once, at build time) and freeze the
        // structure + position map.
        let mut row_idx = vec![0usize; nnz];
        let mut pos = vec![0usize; nnz];
        for c in 0..n {
            let (s, e) = (col_ptr[c], col_ptr[c + 1]);
            pairs[s..e].sort_unstable_by_key(|&(r, _)| r);
            for (p, &(r, k)) in (s..e).zip(pairs[s..e].iter()) {
                row_idx[p] = r;
                pos[k] = p;
            }
        }
        PermScatter {
            col_ptr,
            row_idx,
            pos,
        }
    }

    /// Scatter a fresh value set of `a` (the pattern this was built from)
    /// through the frozen structure in one linear pass, applying the symmetric
    /// equilibration `a_ij * (s_i * s_j)` on the way when `scale` is given, so
    /// no scaled copy of `a` is ever held.
    pub fn scatter<T: crate::scalar::Scalar>(
        &self,
        a: &crate::sparse::csc::CscMatrix<T>,
        scale: Option<&[f64]>,
    ) -> Vec<T> {
        debug_assert_eq!(a.values.len(), self.pos.len());
        let mut out = vec![T::zero(); a.values.len()];
        match scale {
            None => {
                for (k, &p) in self.pos.iter().enumerate() {
                    out[p] = a.values[k];
                }
            }
            Some(s) => {
                for j in 0..a.n {
                    for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                        out[self.pos[k]] = a.values[k] * T::from_real(s[a.row_idx[k]] * s[j]);
                    }
                }
            }
        }
        out
    }
}
