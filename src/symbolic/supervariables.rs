//! Indistinguishable-vertex compression of the ordering graph.
//!
//! Vertices with the same closed adjacency (neighbours plus themselves) are
//! eliminated identically by any fill-reducing ordering, so they can be
//! ordered as one vertex and expanded afterwards. Vector-valued finite
//! elements produce them in bulk: second-order Nedelec elements carry two
//! unknowns per edge with the same neighbours, which halves the graph a
//! nested dissection has to partition. METIS applies the same compression
//! (`METIS_OPTION_COMPRESS`); weighting each compressed vertex by its group
//! size keeps the bisections balanced as on the full graph.
use rayon::prelude::*;

use crate::sparse::csc::CscPattern;

/// Groups of indistinguishable vertices of a symmetric pattern.
///
/// Group `g` holds the original vertices `members[ptr[g]..ptr[g + 1]]` in
/// ascending order; groups are numbered by their smallest member, so the
/// grouping is a pure function of the pattern.
pub(crate) struct Supervariables {
    /// `group_of[v]` is the group of original vertex `v`.
    pub group_of: Vec<usize>,
    /// Group boundaries into `members`, `n_groups + 1` entries.
    pub ptr: Vec<usize>,
    /// Original vertices, grouped.
    pub members: Vec<usize>,
}

impl Supervariables {
    /// Find the groups of `pattern` (full symmetric, sorted rows per column).
    pub fn of(pattern: &CscPattern) -> Self {
        let n = pattern.n;
        let closed = |v: usize| {
            let rows = &pattern.row_idx[pattern.col_ptr[v]..pattern.col_ptr[v + 1]];
            let at = rows.partition_point(|&r| r < v);
            let has_self = rows.get(at) == Some(&v);
            (rows, at, has_self)
        };
        // Order-independent key of the closed adjacency: its length and a
        // hash over the sorted rows with `v` merged in.
        let key = |v: usize| -> (usize, u64) {
            let (rows, at, has_self) = closed(v);
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut mix = |x: usize| {
                h ^= x as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            };
            rows[..at].iter().for_each(|&r| mix(r));
            mix(v);
            rows[at + usize::from(has_self)..]
                .iter()
                .for_each(|&r| mix(r));
            (rows.len() + usize::from(!has_self), h)
        };
        let same = |a: usize, b: usize| -> bool {
            let merged = |v: usize| {
                let (rows, at, has_self) = closed(v);
                rows[..at]
                    .iter()
                    .copied()
                    .chain(std::iter::once(v))
                    .chain(rows[at + usize::from(has_self)..].iter().copied())
            };
            merged(a).eq(merged(b))
        };

        let keys: Vec<(usize, u64)> = (0..n).into_par_iter().map(key).collect();
        let mut by_key: Vec<usize> = (0..n).collect();
        by_key.par_sort_unstable_by_key(|&v| (keys[v], v));

        // Within a run of equal keys, split into exact-equality classes
        // (hash collisions are rare; the leader is the smallest member).
        const NONE: usize = usize::MAX;
        let mut leader = vec![NONE; n];
        let mut run = 0;
        while run < n {
            let mut end = run + 1;
            while end < n && keys[by_key[end]] == keys[by_key[run]] {
                end += 1;
            }
            for i in run..end {
                let v = by_key[i];
                if leader[v] != NONE {
                    continue;
                }
                leader[v] = v;
                for &u in &by_key[i + 1..end] {
                    if leader[u] == NONE && same(v, u) {
                        leader[u] = v;
                    }
                }
            }
            run = end;
        }

        // Number the groups by leader; `v` ascending visits every leader
        // before its members, so members come out ascending too.
        let mut group_of = vec![NONE; n];
        let mut sizes: Vec<usize> = Vec::new();
        for v in 0..n {
            let g = if leader[v] == v {
                sizes.push(0);
                sizes.len() - 1
            } else {
                group_of[leader[v]]
            };
            group_of[v] = g;
            sizes[g] += 1;
        }
        let ptr: Vec<usize> = std::iter::once(0)
            .chain(sizes.iter().scan(0, |total, &s| {
                *total += s;
                Some(*total)
            }))
            .collect();
        let mut fill = ptr[..sizes.len()].to_vec();
        let mut members = vec![0; n];
        for (v, &g) in group_of.iter().enumerate() {
            members[fill[g]] = v;
            fill[g] += 1;
        }
        Supervariables {
            group_of,
            ptr,
            members,
        }
    }

    /// Number of groups.
    pub fn len(&self) -> usize {
        self.ptr.len() - 1
    }

    /// Group sizes, the vertex weights of the compressed graph.
    pub fn weights(&self) -> Vec<i32> {
        self.ptr
            .windows(2)
            .map(|w| i32::try_from(w[1] - w[0]).unwrap_or(i32::MAX))
            .collect()
    }

    /// The compressed graph: one vertex per group, an edge between two
    /// groups whenever their members are adjacent, no self-loops, rows
    /// sorted. The members of a group share their neighbours, so the
    /// group's first member stands for all of them.
    pub fn compress(&self, pattern: &CscPattern) -> CscPattern {
        let n_groups = self.len();
        let mut col_ptr = Vec::with_capacity(n_groups + 1);
        col_ptr.push(0);
        let mut row_idx = Vec::new();
        let mut col: Vec<usize> = Vec::new();
        for g in 0..n_groups {
            let v = self.members[self.ptr[g]];
            col.clear();
            col.extend(
                pattern.row_idx[pattern.col_ptr[v]..pattern.col_ptr[v + 1]]
                    .iter()
                    .map(|&r| self.group_of[r])
                    .filter(|&h| h != g),
            );
            col.sort_unstable();
            col.dedup();
            row_idx.extend_from_slice(&col);
            col_ptr.push(row_idx.len());
        }
        CscPattern {
            n: n_groups,
            col_ptr,
            row_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Full symmetric pattern with the diagonal, from an edge list.
    fn pattern(n: usize, edges: &[(usize, usize)]) -> CscPattern {
        let mut cols: Vec<Vec<usize>> = (0..n).map(|v| vec![v]).collect();
        for &(a, b) in edges {
            cols[a].push(b);
            cols[b].push(a);
        }
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for mut c in cols {
            c.sort_unstable();
            c.dedup();
            row_idx.extend(c);
            col_ptr.push(row_idx.len());
        }
        CscPattern {
            n,
            col_ptr,
            row_idx,
        }
    }

    #[test]
    fn twins_group_and_compress() {
        // 0 and 1 are adjacent twins (same closed neighbourhood {0,1,2});
        // 3 and 4 are adjacent twins ({2,3,4}); 2 is alone.
        let p = pattern(5, &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]);
        let s = Supervariables::of(&p);
        assert_eq!(s.len(), 3);
        assert_eq!(s.group_of, vec![0, 0, 1, 2, 2]);
        assert_eq!(s.weights(), vec![2, 1, 2]);
        let c = s.compress(&p);
        assert_eq!(c.n, 3);
        // Groups {0, 1}, {2}, {3, 4}: the pairs touch only the middle.
        assert_eq!(c.col_ptr, vec![0, 1, 3, 4]);
        assert_eq!(c.row_idx, vec![1, 0, 2, 1]);
    }

    #[test]
    fn a_path_has_no_twins() {
        let p = pattern(4, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(Supervariables::of(&p).len(), 4);
    }

    #[test]
    fn diagonal_presence_does_not_matter() {
        // Same graph with and without stored diagonal entries.
        let with = pattern(3, &[(0, 1), (0, 2), (1, 2)]);
        let without = CscPattern {
            n: 3,
            col_ptr: vec![0, 2, 4, 6],
            row_idx: vec![1, 2, 0, 2, 0, 1],
        };
        assert_eq!(Supervariables::of(&with).len(), 1);
        assert_eq!(Supervariables::of(&without).len(), 1);
    }
}
