use crate::ordering::elimination_tree::EliminationTree;
use crate::sparse::csc::CscPattern;

/// Compute the number of nonzeros in each column of the Cholesky factor L.
///
/// Uses elimination graph simulation: process columns left to right,
/// maintaining the fill pattern. For column j, `L[:,j]` contains:
/// - The diagonal entry (j,j)
/// - All original entries (i,j) with i > j
/// - All fill entries propagated from earlier columns
///
/// For indefinite factorization (LDL^T), the fill pattern is the same as
/// Cholesky: pivoting is confined to each supernode's fully-summed block
/// (no delayed pivots), so it changes values but not structure.
///
/// Input `pattern` should be the full symmetric pattern (both triangles).
///
/// Returns a vector of length n where `counts[j]` is the number of nonzeros
/// in column j of L (including the diagonal).
///
/// Test-only reference oracle: the production pipeline uses
/// [`column_counts_gnp`] exclusively; this O(n^2) simulation is kept solely
/// so the tests here and in `supernode.rs` can cross-check GNP against a
/// first-principles implementation.
#[cfg(test)]
pub fn column_counts(pattern: &CscPattern, _etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }

    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, col_j) in col_rows.iter_mut().enumerate() {
        for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
            let i = pattern.row_idx[k];
            if i > j {
                col_j.push(i);
            }
        }
        col_j.sort_unstable();
        col_j.dedup();
    }

    let mut counts = vec![1usize; n];

    for j in 0..n {
        let rows = std::mem::take(&mut col_rows[j]);
        counts[j] += rows.len();

        if rows.len() > 1 {
            let min_row = rows[0];
            for &row in &rows[1..] {
                if !col_rows[min_row].contains(&row) {
                    col_rows[min_row].push(row);
                }
            }
            col_rows[min_row].sort_unstable();
            col_rows[min_row].dedup();
        }
    }

    counts
}

/// Compute the total number of nonzeros in L from column counts.
pub fn total_factor_nnz(counts: &[usize]) -> usize {
    counts.iter().sum()
}

/// Gilbert-Ng-Peyton column counts (O(nnz(A) + n*alpha(n))).
///
/// Equivalent to `column_counts` but asymptotically much faster on
/// dense-ish patterns. Uses Liu's row-subtree characterization:
///
/// - L(i, j) != 0 iff j in T^r_i (the row subtree for row i)
/// - `c[j] = |{i : j in T^r_i}|` (column count)
/// - `c[j]` = sum over T^r_i of (leaves at j) - (LCAs at j), accumulated
///   up the etree
///
/// References:
/// - Gilbert, Ng, Peyton, *SIAM J. Matrix Anal. Appl.* 15(4):1075-1091, 1994
/// - Davis, *Direct Methods for Sparse Linear Systems* section 4.4
/// - CSparse `cs_counts.c` (BSD, structural reference)
pub fn column_counts_gnp(pattern: &CscPattern, etree: &EliminationTree) -> Vec<usize> {
    gnp(
        pattern.n,
        etree,
        |i| {
            pattern.row_idx[pattern.col_ptr[i]..pattern.col_ptr[i + 1]]
                .iter()
                .copied()
        },
        |_| 1,
    )
}

pub(crate) fn gnp<I: Iterator<Item = usize>>(
    n: usize,
    etree: &EliminationTree,
    cols: impl Fn(usize) -> I,
    weight: impl Fn(usize) -> i64,
) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    let post = etree.postorder();
    let first = etree.first_descendants(&post);
    let mut has_child = vec![false; n];
    for p in etree.parent.iter().flatten() {
        has_child[*p] = true;
    }

    let mut delta: Vec<i64> = (0..n)
        .map(|i| if has_child[i] { 0 } else { weight(i) })
        .collect();

    let mut maxfirst = vec![-1i64; n];
    let mut prevleaf = vec![-1i64; n];
    let mut ancestor: Vec<usize> = (0..n).collect();

    for &i in &post {
        if let Some(p) = etree.parent[i] {
            delta[p] -= weight(i);
        }

        let fi = first[i] as i64;
        for partner in cols(i) {
            if partner <= i {
                continue;
            }
            if fi > maxfirst[partner] {
                delta[i] += weight(partner);
                let pl = prevleaf[partner];
                if pl != -1 {
                    let mut q = pl as usize;
                    while ancestor[q] != q {
                        q = ancestor[q];
                    }
                    let root = q;
                    let mut cur = pl as usize;
                    while cur != root {
                        let next = ancestor[cur];
                        ancestor[cur] = root;
                        cur = next;
                    }
                    delta[root] -= weight(partner);
                }
                prevleaf[partner] = i as i64;
                maxfirst[partner] = fi;
            }
        }
        if let Some(p) = etree.parent[i] {
            ancestor[i] = p;
        }
    }

    for &i in &post {
        if let Some(p) = etree.parent[i] {
            delta[p] += delta[i];
        }
    }

    delta.iter().map(|&d| d as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn gnp_matches_reference_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_tridiagonal() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_arrow() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_block_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_banded() {
        let rows = vec![0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 6];
        let cols = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6];
        let m = CscMatrix::from_triplets(7, &rows, &cols, &vec![1.0; rows.len()]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }
}
