use crate::sparse::csc::CscPattern;

/// Elimination tree of a symmetric matrix.
///
/// For a symmetric matrix A, the elimination tree has the property:
/// `parent[j] = min { i > j : L(i,j) != 0 }` where L is the Cholesky factor.
/// For indefinite matrices, the same structure applies to the fill pattern.
///
/// Constructed from the symmetric sparsity pattern using union-find with
/// path compression (George & Liu 1981, Chapter 4).
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// `parent[j] = Some(i)` where `i > j`, or `None` if `j` is a root.
    pub parent: Vec<Option<usize>>,
    pub n: usize,
}

impl EliminationTree {
    /// Build the elimination tree from a symmetric sparsity pattern.
    ///
    /// Uses the column-by-column algorithm with path compression
    /// (Liu 1990, based on George & Liu 1981):
    ///
    /// For each column j (in order 0..n), examine all rows i < j in column j
    /// (the upper triangle entries). Walk from i up the partially built tree
    /// using path compression until finding a root or reaching j. Make j the
    /// parent of that root. This produces parent[j] = min { i > j : L(i,j) != 0 }.
    pub fn from_pattern(pattern: &CscPattern) -> Self {
        Self::from_cols(pattern.n, |j| {
            pattern.row_idx[pattern.col_ptr[j]..pattern.col_ptr[j + 1]]
                .iter()
                .copied()
        })
    }

    pub(crate) fn from_cols<I: Iterator<Item = usize>>(
        n: usize,
        cols: impl Fn(usize) -> I,
    ) -> Self {
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut ancestor = vec![0usize; n];

        for j in 0..n {
            ancestor[j] = j;
            for i in cols(j) {
                if i >= j {
                    continue;
                }

                let mut r = i;
                while ancestor[r] != r {
                    r = ancestor[r];
                }
                let mut node = i;
                while node != r {
                    let next = ancestor[node];
                    ancestor[node] = r;
                    node = next;
                }

                if r != j {
                    parent[r] = Some(j);
                    ancestor[r] = j;
                }
            }
        }

        EliminationTree { parent, n }
    }

    pub(crate) fn children_flat(&self) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut ptr = vec![0usize; n + 1];
        for p in self.parent.iter().flatten() {
            ptr[p + 1] += 1;
        }
        for v in 0..n {
            ptr[v + 1] += ptr[v];
        }
        let mut next = ptr[..n].to_vec();
        let mut idx = vec![0usize; ptr[n]];
        for (j, &p) in self.parent.iter().enumerate() {
            if let Some(p) = p {
                idx[next[p]] = j;
                next[p] += 1;
            }
        }
        (ptr, idx)
    }

    pub fn children(&self) -> Vec<Vec<usize>> {
        let mut ch = vec![Vec::new(); self.n];
        for (j, &p) in self.parent.iter().enumerate() {
            if let Some(p) = p {
                ch[p].push(j);
            }
        }
        ch
    }

    /// Return root nodes (nodes with no parent).
    pub fn roots(&self) -> Vec<usize> {
        (0..self.n).filter(|&j| self.parent[j].is_none()).collect()
    }

    /// Compute subtree sizes (number of nodes in each subtree, including self).
    pub fn subtree_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![1usize; self.n];
        for j in 0..self.n {
            if let Some(p) = self.parent[j] {
                debug_assert!(p > j);
                sizes[p] += sizes[j];
            }
        }
        sizes
    }

    /// Postorder traversal of the etree forest. Returns a Vec of
    /// node indices in postorder (each subtree's children listed
    /// before the subtree's root; roots of the forest come last).
    ///
    /// Iterative DFS using an explicit stack so deep trees don't
    /// blow the call stack.
    pub fn postorder(&self) -> Vec<usize> {
        let n = self.n;
        let mut out = Vec::with_capacity(n);
        let (ptr, idx) = self.children_flat();
        let mut next_child = ptr[..n].to_vec();
        let mut stack: Vec<usize> = Vec::with_capacity(n);

        for root in self.roots() {
            stack.push(root);
            while let Some(&node) = stack.last() {
                let k = next_child[node];
                if k < ptr[node + 1] {
                    next_child[node] = k + 1;
                    stack.push(idx[k]);
                } else {
                    out.push(node);
                    stack.pop();
                }
            }
        }
        out
    }

    /// First-descendant numbering used by the Gilbert-Ng-Peyton
    /// column-count algorithm.
    ///
    /// Given a postorder `post` (as returned by [`postorder`]), the
    /// result `first[i]` is the smallest postorder number taken by
    /// any descendant of node i (including i itself). Leaves have
    /// `first[i]` equal to their own postorder index.
    pub fn first_descendants(&self, post: &[usize]) -> Vec<usize> {
        let n = self.n;
        debug_assert_eq!(post.len(), n);
        let mut post_of = vec![0usize; n];
        for (pnum, &node) in post.iter().enumerate() {
            post_of[node] = pnum;
        }
        let mut first = post_of.clone();
        for &node in post {
            if let Some(p) = self.parent[node] {
                if first[node] < first[p] {
                    first[p] = first[node];
                }
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_etree_tridiagonal() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);

        assert_eq!(etree.parent[0], Some(1));
        assert_eq!(etree.parent[1], Some(2));
        assert_eq!(etree.parent[2], Some(3));
        assert_eq!(etree.parent[3], Some(4));
        assert_eq!(etree.parent[4], None);
    }

    #[test]
    fn test_etree_arrow() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);

        assert_eq!(etree.parent[4], None);
        assert_eq!(etree.roots(), vec![4]);
    }

    fn chain_etree(n: usize) -> EliminationTree {
        let mut parent = vec![None; n];
        for j in 0..n.saturating_sub(1) {
            parent[j] = Some(j + 1);
        }
        EliminationTree { parent, n }
    }

    #[test]
    fn test_postorder_chain() {
        let et = chain_etree(5);
        let post = et.postorder();
        assert_eq!(post, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_postorder_star() {
        let parent = vec![Some(4), Some(4), Some(4), Some(4), None];
        let et = EliminationTree { parent, n: 5 };
        let post = et.postorder();
        assert_eq!(*post.last().unwrap(), 4);
        assert_eq!(post.len(), 5);
        let mut leaves: Vec<_> = post[..4].to_vec();
        leaves.sort_unstable();
        assert_eq!(leaves, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_postorder_two_roots() {
        let parent = vec![Some(1), None, Some(3), None];
        let et = EliminationTree { parent, n: 4 };
        let post = et.postorder();
        assert_eq!(post.len(), 4);
        let p0 = post.iter().position(|&x| x == 0).unwrap();
        let p1 = post.iter().position(|&x| x == 1).unwrap();
        let p2 = post.iter().position(|&x| x == 2).unwrap();
        let p3 = post.iter().position(|&x| x == 3).unwrap();
        assert!(p0 < p1);
        assert!(p2 < p3);
    }

    #[test]
    fn test_first_descendants_chain() {
        let et = chain_etree(5);
        let post = et.postorder();
        let first = et.first_descendants(&post);
        assert_eq!(first, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_first_descendants_leaves() {
        let parent = vec![Some(4), Some(4), Some(4), Some(4), None];
        let et = EliminationTree { parent, n: 5 };
        let post = et.postorder();
        let first = et.first_descendants(&post);
        for leaf in 0..4 {
            let ppos = post.iter().position(|&x| x == leaf).unwrap();
            assert_eq!(first[leaf], ppos);
        }
        assert_eq!(first[4], 0);
    }

    #[test]
    fn test_etree_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);

        for j in 0..4 {
            assert_eq!(etree.parent[j], None);
        }
        assert_eq!(etree.roots().len(), 4);
    }

    #[test]
    fn test_etree_children() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let ch = etree.children();

        assert_eq!(ch[0], Vec::<usize>::new());
        assert_eq!(ch[1], vec![0]);
        assert_eq!(ch[2], vec![1]);
        assert_eq!(ch[3], vec![2]);
    }

    #[test]
    fn test_subtree_sizes() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let sizes = etree.subtree_sizes();

        assert_eq!(sizes[0], 1);
        assert_eq!(sizes[1], 2);
        assert_eq!(sizes[2], 3);
        assert_eq!(sizes[3], 4);
    }
}
