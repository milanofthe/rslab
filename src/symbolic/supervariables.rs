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
use crate::sparse::csc::CscPattern;

/// FNV-1a 64 offset basis (published constant of the hash).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64 prime (published constant of the hash).
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

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
    pub fn of(pattern: &CscPattern<i32>) -> Self {
        let n = pattern.n;
        let closed = |v: usize| {
            let rows = pattern.col(v);
            let at = rows.partition_point(|&r| (r as usize) < v);
            let has_self = rows.get(at).is_some_and(|&r| r as usize == v);
            (rows, at, has_self)
        };
        // Order-independent key of the closed adjacency: its length and a
        // hash over the sorted rows with `v` merged in.
        let key = |v: usize| -> (usize, u64) {
            let (rows, at, has_self) = closed(v);
            let mut h: u64 = FNV_OFFSET;
            let mut mix = |x: usize| {
                h ^= x as u64;
                h = h.wrapping_mul(FNV_PRIME);
            };
            rows[..at].iter().for_each(|&r| mix(r as usize));
            mix(v);
            rows[at + usize::from(has_self)..]
                .iter()
                .for_each(|&r| mix(r as usize));
            (rows.len() + usize::from(!has_self), h)
        };
        let same = |a: usize, b: usize| -> bool {
            let merged = |v: usize| {
                let (rows, at, has_self) = closed(v);
                rows[..at]
                    .iter()
                    .map(|&r| r as usize)
                    .chain(std::iter::once(v))
                    .chain(
                        rows[at + usize::from(has_self)..]
                            .iter()
                            .map(|&r| r as usize),
                    )
            };
            merged(a).eq(merged(b))
        };

        let keys: Vec<(usize, u64)> = crate::parallel::map(n, 1, key);
        let mut by_key: Vec<usize> = (0..n).collect();
        crate::parallel::sort_unstable_by_key(&mut by_key, |&v| (keys[v], v));

        // Within a run of equal keys, split into exact-equality classes
        // (hash collisions are rare; the leader is the smallest member).
        // Runs are contiguous in `by_key` and independent, so they split in
        // parallel: `lead[i]` is the leader of `by_key[i]`.
        const NONE: usize = usize::MAX;
        let mut runs: Vec<&mut [usize]> = Vec::new();
        let mut lead = vec![NONE; n];
        let mut rest: &mut [usize] = &mut lead;
        let mut run = 0;
        while run < n {
            let mut end = run + 1;
            while end < n && keys[by_key[end]] == keys[by_key[run]] {
                end += 1;
            }
            let (head, tail) = rest.split_at_mut(end - run);
            runs.push(head);
            rest = tail;
            run = end;
        }
        let mut offsets = Vec::with_capacity(runs.len());
        let mut at = 0;
        for r in &runs {
            offsets.push(at);
            at += r.len();
        }
        let runs: Vec<(&mut [usize], usize)> = runs.into_iter().zip(offsets).collect();
        let min_runs = crate::constants::SUPERVARIABLE_PIECE_MIN_RUNS;
        crate::parallel::for_each_owned(runs, min_runs, |(lead, start)| {
            let members = &by_key[start..start + lead.len()];
            for i in 0..members.len() {
                if lead[i] != NONE {
                    continue;
                }
                lead[i] = members[i];
                for k in i + 1..members.len() {
                    if lead[k] == NONE && same(members[i], members[k]) {
                        lead[k] = members[i];
                    }
                }
            }
        });
        let mut leader = vec![NONE; n];
        for (i, &v) in by_key.iter().enumerate() {
            leader[v] = lead[i];
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
    /// sorted. The members of a group share their neighbours, so the group's
    /// first member stands for all of them. Blocks of groups compress in
    /// parallel; the graph is smaller than `pattern`, so its indices fit.
    pub fn compress(&self, pattern: &CscPattern<i32>) -> CscPattern<i32> {
        let n_groups = self.len();
        let blocks = (crate::constants::FINE_PIECES_PER_WORKER * crate::parallel::threads())
            .clamp(1, n_groups.max(1));
        let parts: Vec<(Vec<i32>, Vec<i32>)> = crate::parallel::map(blocks, 1, |b| {
            let (g0, g1) = (b * n_groups / blocks, (b + 1) * n_groups / blocks);
            let (mut lens, mut rows) = (Vec::with_capacity(g1 - g0), Vec::new());
            let mut col: Vec<i32> = Vec::new();
            for g in g0..g1 {
                let v = self.members[self.ptr[g]];
                col.clear();
                col.extend(
                    pattern
                        .col(v)
                        .iter()
                        .map(|&r| self.group_of[r as usize])
                        .filter(|&h| h != g)
                        .map(|h| h as i32),
                );
                col.sort_unstable();
                col.dedup();
                lens.push(col.len() as i32);
                rows.extend_from_slice(&col);
            }
            (lens, rows)
        });
        let mut col_ptr = Vec::with_capacity(n_groups + 1);
        col_ptr.push(0i32);
        let mut row_idx = Vec::with_capacity(parts.iter().map(|p| p.1.len()).sum());
        for (lens, rows) in parts {
            for l in lens {
                col_ptr.push(col_ptr[col_ptr.len() - 1] + l);
            }
            row_idx.extend_from_slice(&rows);
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
    fn pattern(n: usize, edges: &[(usize, usize)]) -> CscPattern<i32> {
        let mut cols: Vec<Vec<i32>> = (0..n).map(|v| vec![v as i32]).collect();
        for &(a, b) in edges {
            cols[a].push(b as i32);
            cols[b].push(a as i32);
        }
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for mut c in cols {
            c.sort_unstable();
            c.dedup();
            row_idx.extend(c);
            col_ptr.push(row_idx.len() as i32);
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
        let g = s.compress(&p);
        // Groups {0, 1}, {2}, {3, 4}: the pairs touch only the middle.
        assert_eq!(g.col_ptr, vec![0, 1, 3, 4]);
        assert_eq!(g.row_idx, vec![1, 0, 2, 1]);
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
