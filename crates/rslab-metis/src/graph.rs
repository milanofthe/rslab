//! CSR graph representation for the multilevel partitioning pipeline.
//!
//! Mirrors METIS 5.2.0's `graph_t` (see `libmetis/struct.h`) with
//! Rust-idiomatic `Vec` storage. Diagonal entries of the input
//! matrix (self-loops in the graph) are dropped - they carry no
//! information for fill-reducing ordering and would otherwise need
//! to be stripped in every downstream kernel.
//!
//! Indices are held as `i32` end-to-end: it matches the crate
//! contract, the METIS reference, and keeps cache density high for
//! the partitioning kernels without touching `usize`-shaped
//! `Vec<T>` storage.

use rslab_ordering_core::{CscPattern, OrderingError};

/// CSR graph for partitioning.
///
/// All arrays are `i32`-indexed. Adjacency lists are **not**
/// guaranteed sorted: METIS's kernels don't require sortedness, and
/// enforcing it would be a constant-factor cost in every coarsening
/// level. `from_csc_pattern` happens to emit them sorted because it
/// reads a CSC with sorted row indices, but callers must not rely
/// on this after coarsening.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices. Must equal `xadj.len() - 1` and fit in `i32`.
    pub nvtxs: i32,
    /// Adjacency offsets, length `nvtxs + 1`. `xadj[0] == 0`,
    /// non-decreasing. `xadj[nvtxs]` equals `adjncy.len()` and
    /// therefore `2 * |E|` for undirected graphs without self-loops.
    pub xadj: Vec<i32>,
    /// Neighbor lists, length `2 * |E|`. Each undirected edge `{u,v}`
    /// appears twice (once in `u`'s list, once in `v`'s).
    pub adjncy: Vec<i32>,
    /// Vertex weights, length `nvtxs`. Default: all 1.
    pub vwgt: Vec<i32>,
    /// Edge weights, aligned with `adjncy`, or empty where every edge
    /// weighs one: the input graph and the subgraphs extracted from it, so
    /// only the coarse graphs, whose contraction sums the weights, carry
    /// them (half the bytes of an input graph). Read with
    /// [`wgt`](Self::wgt).
    pub adjwgt: Vec<i32>,
}

impl Graph {
    /// Build a `Graph` from a full-symmetric `CscPattern`.
    ///
    /// Diagonal entries are dropped. The input is assumed to be
    /// already structurally symmetric per the contract (debug-
    /// asserted elsewhere). Row indices within each CSC column are
    /// sorted ascending - `CscPattern::new` enforces this - so any
    /// duplicates are adjacent and a running "last seen" check drops
    /// all of them without needing a hash set.
    ///
    /// Complexity: `O(nnz)` time, `O(nnz + n)` space.
    pub fn from_csc_pattern(pattern: &CscPattern<'_>) -> Result<Self, OrderingError> {
        let n = pattern.n;
        if n > i32::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }
        let nvtxs = n as i32;
        let mut xadj: Vec<i32> = Vec::with_capacity(n + 1);
        let mut adjncy: Vec<i32> = Vec::with_capacity(pattern.row_idx.len());
        xadj.push(0);
        for j in 0..n {
            let lo = pattern.col_ptr[j] as usize;
            let hi = pattern.col_ptr[j + 1] as usize;
            if hi > pattern.row_idx.len() || lo > hi {
                return Err(OrderingError::MalformedInput);
            }
            let jj = j as i32;
            let mut last: i32 = -1;
            for &r in &pattern.row_idx[lo..hi] {
                if r == jj {
                    continue;
                }
                if r < 0 || r >= nvtxs {
                    return Err(OrderingError::MalformedInput);
                }
                debug_assert!(r >= last);
                if r == last {
                    continue;
                }
                adjncy.push(r);
                last = r;
            }
            if adjncy.len() > i32::MAX as usize {
                return Err(OrderingError::IndexOverflow);
            }
            xadj.push(adjncy.len() as i32);
        }
        let vwgt = vec![1i32; n];
        Ok(Graph {
            nvtxs,
            xadj,
            adjncy,
            vwgt,
            adjwgt: Vec::new(),
        })
    }

    /// Number of undirected edges (each stored twice in `adjncy`).
    pub fn nedges(&self) -> usize {
        self.adjncy.len() / 2
    }

    /// Degree of vertex `v`.
    pub fn degree(&self, v: i32) -> i32 {
        self.xadj[(v + 1) as usize] - self.xadj[v as usize]
    }

    /// Borrow the adjacency slice of vertex `v`.
    pub fn neighbors(&self, v: i32) -> &[i32] {
        let lo = self.xadj[v as usize] as usize;
        let hi = self.xadj[(v + 1) as usize] as usize;
        &self.adjncy[lo..hi]
    }

    /// Weight of the edge at `adjncy[k]`.
    #[inline]
    pub fn wgt(&self, k: usize) -> i32 {
        if self.adjwgt.is_empty() {
            1
        } else {
            self.adjwgt[k]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::{arrow, csc_from_triples, grid_2d};

    fn diag(n: usize) -> (Vec<i32>, Vec<i32>) {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
        }
        csc_from_triples(n, &t)
    }

    fn tridiag(n: usize) -> (Vec<i32>, Vec<i32>) {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            if i + 1 < n {
                t.push((i, i + 1));
            }
        }
        csc_from_triples(n, &t)
    }

    fn assert_graph_invariants(g: &Graph) {
        assert_eq!(g.xadj.len(), g.nvtxs as usize + 1, "xadj length");
        assert_eq!(g.xadj[0], 0, "xadj[0]");
        assert_eq!(
            g.xadj[g.nvtxs as usize] as usize,
            g.adjncy.len(),
            "xadj[nvtxs] == adjncy.len()"
        );
        for i in 0..g.nvtxs as usize {
            assert!(g.xadj[i] <= g.xadj[i + 1], "xadj non-decreasing at {}", i);
        }
        assert_eq!(g.vwgt.len(), g.nvtxs as usize, "vwgt length");
        assert!(
            g.adjwgt.is_empty() || g.adjwgt.len() == g.adjncy.len(),
            "adjwgt length"
        );
        for (i, &r) in g.adjncy.iter().enumerate() {
            assert!(r >= 0 && r < g.nvtxs, "adjncy[{}]={} OOB", i, r);
        }
        for v in 0..g.nvtxs {
            for &u in g.neighbors(v) {
                assert_ne!(u, v, "self-loop at {}", v);
            }
        }
        for v in 0..g.nvtxs {
            for &u in g.neighbors(v) {
                assert!(
                    g.neighbors(u).contains(&v),
                    "asymmetric edge: {} -> {} but not back",
                    v,
                    u
                );
            }
        }
        assert_eq!(g.nedges() * 2, g.adjncy.len(), "edge-count parity");
    }

    #[test]
    fn diag_has_no_edges() {
        let (cp, ri) = diag(4);
        let pat = CscPattern::new(4, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 4);
        assert_eq!(g.nedges(), 0);
        for v in 0..4 {
            assert_eq!(g.degree(v), 0);
        }
    }

    #[test]
    fn arrow_5_degree_sequence() {
        let (cp, ri) = arrow(5);
        let pat = CscPattern::new(5, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 5);
        assert_eq!(g.nedges(), 4);
        assert_eq!(g.degree(0), 4);
        for v in 1..5 {
            assert_eq!(g.degree(v), 1);
            assert_eq!(g.neighbors(v), &[0]);
        }
        let mut hub: Vec<i32> = g.neighbors(0).to_vec();
        hub.sort();
        assert_eq!(hub, vec![1, 2, 3, 4]);
    }

    #[test]
    fn tridiag_10_degree_sequence() {
        let (cp, ri) = tridiag(10);
        let pat = CscPattern::new(10, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 10);
        assert_eq!(g.nedges(), 9);
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(9), 1);
        for v in 1..9 {
            assert_eq!(g.degree(v), 2);
        }
    }

    #[test]
    fn grid_3x3_structure() {
        let (cp, ri) = grid_2d(3, 3);
        let pat = CscPattern::new(9, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_graph_invariants(&g);
        assert_eq!(g.nvtxs, 9);
        assert_eq!(g.nedges(), 12);
        let mut by_degree = [0u32; 5];
        for v in 0..9 {
            by_degree[g.degree(v) as usize] += 1;
        }
        assert_eq!(by_degree[2], 4, "four corners with degree 2");
        assert_eq!(by_degree[3], 4, "four edge-centres with degree 3");
        assert_eq!(by_degree[4], 1, "one interior with degree 4");
    }

    #[test]
    fn weights_default_to_one() {
        let (cp, ri) = tridiag(5);
        let pat = CscPattern::new(5, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert!(g.vwgt.iter().all(|&w| w == 1));
        assert!((0..g.adjncy.len()).all(|k| g.wgt(k) == 1));
    }

    #[test]
    fn empty_graph_is_legal() {
        let cp: Vec<i32> = vec![0];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_eq!(g.nvtxs, 0);
        assert_eq!(g.nedges(), 0);
        assert_eq!(g.xadj, vec![0]);
    }

    #[test]
    fn diagonal_entries_are_dropped() {
        let (cp, ri) = diag(3);
        assert_eq!(ri.len(), 3);
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        assert_eq!(g.adjncy.len(), 0);
    }
}
