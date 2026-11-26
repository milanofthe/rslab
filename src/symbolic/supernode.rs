use crate::numeric::settings::AmalgamationSettings;
use crate::ordering::elimination_tree::EliminationTree;

/// Relaxed (fill-tolerant) amalgamation thresholds. See
/// [`AmalgamationSettings::relax`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelaxAmalgamation {
    /// Cap on the merged supernode width (eliminated columns).
    pub max_width: usize,
    /// Maximum explicit-zero rows a single relaxed merge may introduce.
    pub max_extra_rows: usize,
}

/// Fronts up to 256 columns wide, at most 64 explicit-zero rows per merge.
impl Default for RelaxAmalgamation {
    fn default() -> Self {
        Self {
            max_width: crate::constants::RELAX_MAX_WIDTH,
            max_extra_rows: crate::constants::RELAX_MAX_EXTRA_ROWS,
        }
    }
}

/// How the amalgamation reaches the children it wants to merge.
///
/// A merged supernode must hold consecutive columns. In a postordered tree
/// only one child of a parent sits next to it, so a parent with several
/// children can absorb at most one of them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AmalgamationStrategy {
    /// Merge only children that are already adjacent.
    Adjacency,
    /// Predict the merges, then postorder the tree again with the children
    /// to be merged next to their parents, so every predicted merge can
    /// happen (SSIDS's renumbering). Wins on bushy trees; on path-like trees
    /// it over-merges.
    Renumber,
    /// `Adjacency` for path-like trees, `Renumber` for bushy ones (see
    /// [`pick_amalgamation_strategy`]).
    #[default]
    Auto,
}

/// A supernode: consecutive columns with nested row structures, factored as
/// one dense panel.
#[derive(Debug, Clone)]
pub struct Supernode {
    /// First of the supernode's columns (in the analysis numbering).
    pub first_col: usize,
    /// Number of columns.
    pub ncol: usize,
    /// Rows of the panel: the columns themselves and the rows below.
    pub nrow: usize,
    /// Child supernodes.
    pub children: Vec<usize>,
}

impl Supernode {
    /// Rows below the supernode's own columns.
    #[inline]
    pub fn contrib_nrow(&self) -> usize {
        self.nrow - self.ncol
    }
}

/// Resolve [`AmalgamationStrategy::Auto`] from the tree's shape: `Adjacency`
/// for a path-like tree (fewer than `path_like_fraction` of the internal
/// nodes with several children; measured trees sat at 0.002 where
/// `Renumber` loses and from 0.20 up where it wins), else `Renumber`.
pub fn pick_amalgamation_strategy(
    etree: &EliminationTree,
    path_like_fraction: f64,
) -> AmalgamationStrategy {
    let n = etree.n;
    if n == 0 {
        return AmalgamationStrategy::Adjacency;
    }
    let mut child_count = vec![0usize; n];
    for &p in &etree.parent {
        if let Some(par) = p {
            child_count[par] += 1;
        }
    }
    let n_internal = child_count.iter().filter(|&&c| c > 0).count();
    if n_internal == 0 {
        return AmalgamationStrategy::Adjacency;
    }
    let n_multi_child = child_count.iter().filter(|&&c| c >= 2).count();
    if (n_multi_child as f64) < path_like_fraction * n_internal as f64 {
        AmalgamationStrategy::Adjacency
    } else {
        AmalgamationStrategy::Renumber
    }
}

/// Detect fundamental supernodes and apply amalgamation.
///
/// A fundamental supernode is a maximal set of consecutive columns j, j+1, ..., j+k
/// where each column's row structure is identical (the same set of row indices,
/// minus the column being eliminated). See `find_fundamental_supernodes` for
/// the detection conditions.
///
/// After detecting fundamental supernodes, amalgamation merges an adjacent
/// child into its parent using the SSIDS merge rule:
/// 1. Trivial chain: parent has exactly 1 column AND the parent's column count
///    is the child's last column count minus one (same row structure minus
///    the eliminated column).
/// 2. Size-based: both parent AND child have < nemin columns.
///
/// On larger problems `params.relax` adds relaxed merges (from
/// `relax_min_n` unknowns) and a width cap keeps the root supernode from
/// growing too wide (from 1024 unknowns).
///
/// Returns supernodes in postorder (children before parents).
pub fn find_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &AmalgamationSettings,
) -> Vec<Supernode> {
    let n = etree.n;
    if n == 0 {
        return Vec::new();
    }

    let relax = if n >= params.relax_min_n {
        params.relax
    } else {
        None
    };
    let relax_width = relax.map(|r| r.max_width);
    let relax_rows = relax.map_or(0, |r| r.max_extra_rows);
    let force_renumber = relax.is_some();

    let fund = find_fundamental_supernodes(etree, col_counts);
    let snode_starts = fund.snode_starts;
    let mut snode_ncols = fund.snode_ncols;
    let snode_parent = fund.snode_parent;
    let n_snodes = snode_starts.len();

    let mut merged_into = vec![None::<usize>; n_snodes];
    let mut snode_first_col: Vec<usize> = snode_starts;

    let mut snode_nrow: Vec<usize> = (0..n_snodes)
        .map(|s| col_counts[snode_first_col[s]].max(snode_ncols[s]))
        .collect();

    let reverse = force_renumber || matches!(params.strategy, AmalgamationStrategy::Renumber);
    let order: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..n_snodes).rev())
    } else {
        Box::new(0..n_snodes)
    };

    for s in order {
        let sp = snode_parent[s];
        if let Some(p) = sp {
            if find_root(s, &merged_into) != s {
                continue;
            }

            let root_s = find_root(s, &merged_into);
            let root_p = find_root(p, &merged_into);
            if root_s == root_p {
                continue;
            }

            let s_first = snode_first_col[root_s];
            let s_ncol = snode_ncols[root_s];
            let p_first = snode_first_col[root_p];
            if s_first + s_ncol != p_first {
                continue;
            }

            let child_ncol = snode_ncols[root_s];
            let parent_ncol = snode_ncols[root_p];

            let trivial_chain = parent_ncol == 1 && {
                let child_last = s_first + s_ncol - 1;
                col_counts[p_first] + 1 == col_counts[child_last]
            };

            let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;

            let parent_is_root = snode_parent[root_p].is_none();
            let merged_ncol = child_ncol + parent_ncol;
            let root_cap = if n >= params.root_cap_min_n {
                ((n as f64 * params.root_cap_fraction) as usize).min(params.root_cap_max)
            } else {
                usize::MAX
            };
            let root_cap_exceeded = parent_is_root && merged_ncol > root_cap;

            let relaxed = relax_width.is_some_and(|w| {
                let child_last = s_first + s_ncol - 1;
                let extra = col_counts[child_last].saturating_sub(1 + col_counts[p_first]);
                merged_ncol <= w && extra <= relax_rows
            });

            if (trivial_chain || size_based || relaxed) && !root_cap_exceeded {
                merged_into[root_s] = Some(root_p);
                debug_assert!(s_first < p_first);
                snode_ncols[root_p] = merged_ncol;
                snode_first_col[root_p] = s_first;
                snode_nrow[root_p] += child_ncol;
            }
        }
    }

    let mut final_snodes: Vec<Supernode> = Vec::new();
    let mut new_snode_id = vec![0usize; n_snodes];

    for s in 0..n_snodes {
        if merged_into[s].is_some() {
            continue;
        }

        let first_col = snode_first_col[s];
        let ncol = snode_ncols[s];
        let nrow = snode_nrow[s].max(ncol);

        new_snode_id[s] = final_snodes.len();

        final_snodes.push(Supernode {
            first_col,
            ncol,
            nrow,
            children: Vec::new(),
        });
    }

    for s in 0..n_snodes {
        if merged_into[s].is_some() {
            continue;
        }
        if let Some(p) = snode_parent[s] {
            let root_p = find_root(p, &merged_into);
            if root_p != s {
                let new_child = new_snode_id[s];
                let new_parent = new_snode_id[root_p];
                final_snodes[new_parent].children.push(new_child);
            }
        }
    }

    final_snodes
}

fn find_root(s: usize, merged_into: &[Option<usize>]) -> usize {
    let mut node = s;
    while let Some(parent) = merged_into[node] {
        node = parent;
    }
    node
}

pub(crate) struct FundamentalSupernodes {
    pub(crate) snode_starts: Vec<usize>,
    pub(crate) snode_ncols: Vec<usize>,
    pub(crate) snode_parent: Vec<Option<usize>>,
}

pub(crate) fn find_fundamental_supernodes(
    etree: &EliminationTree,
    col_counts: &[usize],
) -> FundamentalSupernodes {
    let n = etree.n;
    if n == 0 {
        return FundamentalSupernodes {
            snode_starts: Vec::new(),
            snode_ncols: Vec::new(),
            snode_parent: Vec::new(),
        };
    }

    let mut snode_id = vec![0usize; n];
    let mut snode_starts: Vec<usize> = Vec::new();

    let mut n_children = vec![0usize; n];
    for &p in etree.parent.iter().flatten() {
        n_children[p] += 1;
    }

    snode_starts.push(0);
    snode_id[0] = 0;
    for j in 1..n {
        let same_snode = etree.parent[j - 1] == Some(j)
            && col_counts[j] + 1 == col_counts[j - 1]
            && n_children[j] == 1;
        if same_snode {
            snode_id[j] = snode_id[j - 1];
        } else {
            snode_id[j] = snode_starts.len();
            snode_starts.push(j);
        }
    }

    let n_snodes = snode_starts.len();
    let mut snode_ncols = vec![0usize; n_snodes];
    let mut snode_parent: Vec<Option<usize>> = vec![None; n_snodes];
    for j in 0..n {
        snode_ncols[snode_id[j]] += 1;
    }
    for s in 0..n_snodes {
        let last_col = snode_starts[s] + snode_ncols[s] - 1;
        if let Some(p) = etree.parent[last_col] {
            snode_parent[s] = Some(snode_id[p]);
        }
    }

    FundamentalSupernodes {
        snode_starts,
        snode_ncols,
        snode_parent,
    }
}

pub(crate) fn predict_merges(
    etree: &EliminationTree,
    col_counts: &[usize],
    params: &AmalgamationSettings,
) -> Vec<bool> {
    let n = etree.n;
    let mut bias = vec![false; n];
    if n == 0 {
        return bias;
    }
    let fund = find_fundamental_supernodes(etree, col_counts);
    let n_snodes = fund.snode_starts.len();

    for s in 0..n_snodes {
        let p = match fund.snode_parent[s] {
            Some(p) => p,
            None => continue,
        };
        let child_ncol = fund.snode_ncols[s];
        let parent_ncol = fund.snode_ncols[p];

        let s_first = fund.snode_starts[s];
        let p_first = fund.snode_starts[p];
        let child_last = s_first + child_ncol - 1;
        let trivial_chain = parent_ncol == 1 && col_counts[p_first] + 1 == col_counts[child_last];
        let size_based = child_ncol < params.nemin && parent_ncol < params.nemin;

        if trivial_chain || size_based {
            for b in bias.iter_mut().skip(s_first).take(child_ncol) {
                *b = true;
            }
        }
    }

    bias
}

/// Parent of every supernode in the assembly tree (`usize::MAX` for a root),
/// renumbered over the supernodes for which `kept` holds (an empty supernode
/// contributes no factor columns; its children attach to the nearest kept
/// ancestor). The result is indexed by the kept supernodes in order.
pub fn supernode_parents(supernodes: &[Supernode], kept: &[bool]) -> Vec<usize> {
    let ns = supernodes.len();
    let mut parent = vec![usize::MAX; ns];
    for (s, sn) in supernodes.iter().enumerate() {
        for &c in &sn.children {
            parent[c] = s;
        }
    }
    let mut new_index = vec![usize::MAX; ns];
    let mut next = 0;
    for s in 0..ns {
        if kept[s] {
            new_index[s] = next;
            next += 1;
        }
    }
    let mut out = Vec::with_capacity(next);
    for s in 0..ns {
        if !kept[s] {
            continue;
        }
        let mut p = parent[s];
        while p != usize::MAX && !kept[p] {
            p = parent[p];
        }
        out.push(if p == usize::MAX {
            usize::MAX
        } else {
            new_index[p]
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;
    use crate::symbolic::column_counts::column_counts;

    #[test]
    fn test_supernodes_tridiagonal() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = AmalgamationSettings {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);
        assert_eq!(snodes.len(), 3);

        let total_cols: usize = snodes.iter().map(|s| s.ncol).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn test_supernodes_tridiagonal_amalgamated() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = AmalgamationSettings {
            nemin: 32,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        let total_cols: usize = snodes.iter().map(|s| s.ncol).sum();
        assert_eq!(total_cols, 4);
        assert_eq!(snodes.len(), 1);
    }

    #[test]
    fn test_supernodes_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = AmalgamationSettings {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        assert_eq!(snodes.len(), 1);
        assert_eq!(snodes[0].ncol, 3);
        assert_eq!(snodes[0].nrow, 3);
        assert_eq!(snodes[0].contrib_nrow(), 0);
    }

    #[test]
    fn test_supernodes_block_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = AmalgamationSettings {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        assert_eq!(snodes.len(), 2);
        assert_eq!(snodes[0].ncol, 2);
        assert_eq!(snodes[1].ncol, 2);
    }

    #[test]
    fn test_supernodes_diagonal_no_amalg() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = AmalgamationSettings {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        assert_eq!(snodes.len(), 4);
    }

    #[test]
    fn test_supernodes_total_columns() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        for nemin in [1, 5, 32] {
            let params = AmalgamationSettings {
                nemin,
                ..Default::default()
            };
            let snodes = find_supernodes(&etree, &counts, &params);
            let total: usize = snodes.iter().map(|s| s.ncol).sum();
            assert_eq!(total, 5, "nemin={}: total columns {} != 5", nemin, total);
        }
    }

    #[test]
    fn test_supernode_children_valid() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        let params = AmalgamationSettings {
            nemin: 1,
            ..Default::default()
        };
        let snodes = find_supernodes(&etree, &counts, &params);

        for (i, s) in snodes.iter().enumerate() {
            for &child in &s.children {
                assert!(child < snodes.len(), "invalid child index");
                assert!(
                    child < i,
                    "child {} should come before parent {} in postorder",
                    child,
                    i
                );
            }
        }
    }
}
