//! Multilevel node-separator refinement, ported from METIS 5.2.0
//! (`libmetis/sfm.c`, `libmetis/srefine.c`).
//!
//! This is the quality-critical piece of `METIS_NodeND`: the node
//! separator is constructed once (at the coarsest level, from the best
//! edge bisection via min vertex cover) and then **refined as a node
//! separator at every uncoarsening step** with an FM that moves
//! separator vertices to a side, pulling their far-side neighbors into
//! the separator, with hill-climbing (negative-gain moves) and
//! best-prefix rollback.
//!
//! Evidence for why this matters (2026-07, 40^3 7-point grid, exact
//! scalar nnz(L)): edge-FM through the hierarchy + final König
//! conversion (the previous rslab-metis pipeline) gives 20.9 M fill;
//! *perfect geometric plane separators* give 21.3 M; METIS 5 NodeND
//! gives 14.1 M and MKL PARDISO's ND 12.5 M. Separator *size* is not
//! the lever - refining the node separator through the hierarchy is.
//! See `dev/research/metis-node-separator-2026-07.md`.
//!
//! Gain model (all functions): for a separator vertex `v` and target
//! side `to`, moving `v` out of the separator sheds `vwgt[v]` but pulls
//! every neighbor on the far side into the separator, so
//! `gain = vwgt[v] - edegrees[v][other]` where `edegrees[v][side]` is
//! the total vertex weight of `v`'s neighbors on `side`.

use crate::fm_refine::PART_SEP;
use crate::graph::Graph;
#[cfg(test)]
use crate::initial_partition::{PART_A, PART_B};
use crate::rng::SplitMix;

/// Indexed binary max-heap over vertices keyed by i64 gain, with
/// position tracking for update/delete (METIS `gk_rpq` equivalent).
/// Deterministic: heap order is a pure function of the operation
/// sequence.
struct MaxHeap {
    heap: Vec<i32>,
    pos: Vec<i32>,
    key: Vec<i64>,
}

impl MaxHeap {
    fn new(n: usize) -> Self {
        MaxHeap {
            heap: Vec::with_capacity(n),
            pos: vec![-1; n],
            key: vec![0; n],
        }
    }

    fn clear(&mut self) {
        for &v in &self.heap {
            self.pos[v as usize] = -1;
        }
        self.heap.clear();
    }

    fn contains(&self, v: i32) -> bool {
        self.pos[v as usize] >= 0
    }

    fn insert(&mut self, v: i32, k: i64) {
        debug_assert!(self.pos[v as usize] < 0);
        self.key[v as usize] = k;
        self.pos[v as usize] = self.heap.len() as i32;
        self.heap.push(v);
        self.sift_up(self.heap.len() - 1);
    }

    fn update(&mut self, v: i32, k: i64) {
        let old = self.key[v as usize];
        self.key[v as usize] = k;
        let i = self.pos[v as usize] as usize;
        if k > old {
            self.sift_up(i);
        } else if k < old {
            self.sift_down(i);
        }
    }

    fn pop(&mut self) -> Option<i32> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0];
        self.pos[top as usize] = -1;
        let last = self.heap.pop().expect("non-empty");
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.pos[last as usize] = 0;
            self.sift_down(0);
        }
        Some(top)
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.key[self.heap[i] as usize] > self.key[self.heap[p] as usize] {
                self.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut m = i;
            if l < n && self.key[self.heap[l] as usize] > self.key[self.heap[m] as usize] {
                m = l;
            }
            if r < n && self.key[self.heap[r] as usize] > self.key[self.heap[m] as usize] {
                m = r;
            }
            if m == i {
                break;
            }
            self.swap(i, m);
            i = m;
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        self.pos[self.heap[i] as usize] = i as i32;
        self.pos[self.heap[j] as usize] = j as i32;
    }
}

/// Per-vertex working state shared by balance and refinement.
struct NodeState {
    /// `pwgts[0/1]` = side weights, `pwgts[2]` = separator weight.
    pwgts: [i64; 3],
    /// `edeg[v][side]` = total weight of `v`'s neighbors on `side`.
    /// Valid only while `labels[v] == PART_SEP`.
    edeg: Vec<[i64; 2]>,
    /// Separator membership list with O(1) delete (METIS bnd list).
    bndind: Vec<i32>,
    /// Position of `v` in `bndind`, or -1.
    bndptr: Vec<i32>,
}

impl NodeState {
    /// METIS `Compute2WayNodePartitionParams`.
    fn compute(graph: &Graph, labels: &[u8]) -> Self {
        let n = graph.nvtxs as usize;
        let mut st = NodeState {
            pwgts: [0; 3],
            edeg: vec![[0, 0]; n],
            bndind: Vec::new(),
            bndptr: vec![-1; n],
        };
        for v in 0..n {
            let l = labels[v];
            st.pwgts[l as usize] += graph.vwgt[v] as i64;
            if l == PART_SEP {
                st.bndptr[v] = st.bndind.len() as i32;
                st.bndind.push(v as i32);
                let mut e = [0i64; 2];
                for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
                    let u = graph.adjncy[k] as usize;
                    let lu = labels[u];
                    if lu != PART_SEP {
                        e[lu as usize] += graph.vwgt[u] as i64;
                    }
                }
                st.edeg[v] = e;
            }
        }
        st
    }

    fn bnd_insert(&mut self, v: usize) {
        debug_assert_eq!(self.bndptr[v], -1);
        self.bndptr[v] = self.bndind.len() as i32;
        self.bndind.push(v as i32);
    }

    fn bnd_delete(&mut self, v: usize) {
        let p = self.bndptr[v];
        debug_assert!(p >= 0);
        let last = *self.bndind.last().expect("non-empty bnd");
        self.bndind[p as usize] = last;
        self.bndptr[last as usize] = p;
        self.bndind.pop();
        self.bndptr[v] = -1;
    }
}

/// One-sided node-separator FM (METIS `FM_2WayNodeRefine1Sided`).
///
/// Runs up to `2 * niter` passes, alternating the target side each
/// pass, starting with the lighter side. Each pass greedily moves the
/// max-gain separator vertex to the pass's side (pulling far-side
/// neighbors into the separator), keeps going past negative gains up
/// to a breakout limit, and rolls back to the best separator seen.
/// Returns the final separator weight.
pub(crate) fn fm_node_refine_1sided(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    niter: u32,
    rng: &mut SplitMix,
) -> i64 {
    let n = graph.nvtxs as usize;
    let mut st = NodeState::compute(graph, labels);
    let total = st.pwgts[0] + st.pwgts[1] + st.pwgts[2];
    let badmaxpwgt = (0.5 * (1.0 + max_imbalance) * total as f64) as i64;

    let mut queue = MaxHeap::new(n);
    let mut swaps: Vec<i32> = Vec::with_capacity(n);
    // Rollback bookkeeping: mind[mptr[s]..mptr[s+1]] are the vertices
    // pulled into the separator by swap s.
    let mut mind: Vec<i32> = Vec::with_capacity(2 * n);
    let mut mptr: Vec<usize> = Vec::with_capacity(n + 1);
    let mut order: Vec<i32> = Vec::new();

    let mut to = if st.pwgts[0] < st.pwgts[1] { 1 } else { 0 };
    let mut pass = 0u32;
    loop {
        if pass >= 2 * niter {
            break;
        }
        let other = to;
        to = 1 - to;

        queue.clear();
        let mut mincutorder: i64 = -1;
        let initcut = st.pwgts[2];
        let mut mincut = st.pwgts[2];

        order.clear();
        order.extend_from_slice(&st.bndind);
        rng.shuffle(&mut order);
        for &v in &order {
            debug_assert_eq!(labels[v as usize], PART_SEP);
            queue.insert(v, graph.vwgt[v as usize] as i64 - st.edeg[v as usize][other]);
        }

        let nbnd0 = st.bndind.len();
        let limit = (3 * nbnd0).min(300) as i64;

        swaps.clear();
        mind.clear();
        mptr.clear();
        mptr.push(0);
        let mut mindiff = (st.pwgts[0] - st.pwgts[1]).abs();

        while let Some(higain) = queue.pop() {
            let hv = higain as usize;
            let hw = graph.vwgt[hv] as i64;

            if mind.len() + (graph.xadj[hv + 1] - graph.xadj[hv]) as usize >= 2 * n - 1 {
                break;
            }
            if st.pwgts[to] + hw > badmaxpwgt {
                break;
            }

            let gain = hw - st.edeg[hv][other];
            st.pwgts[2] -= gain;

            let newdiff = (st.pwgts[to] + hw - (st.pwgts[other] - st.edeg[hv][other])).abs();
            if st.pwgts[2] < mincut || (st.pwgts[2] == mincut && newdiff < mindiff) {
                mincut = st.pwgts[2];
                mincutorder = swaps.len() as i64;
                mindiff = newdiff;
            } else if (swaps.len() as i64 - mincutorder) > 3 * limit
                || ((swaps.len() as i64 - mincutorder) > limit
                    && st.pwgts[2] as f64 > 1.10 * mincut as f64)
            {
                st.pwgts[2] += gain;
                break;
            }

            st.bnd_delete(hv);
            st.pwgts[to] += hw;
            labels[hv] = to as u8;
            swaps.push(higain);

            for k in graph.xadj[hv] as usize..graph.xadj[hv + 1] as usize {
                let u = graph.adjncy[k] as usize;
                if labels[u] == PART_SEP {
                    // Neighbor stays in the separator; only its
                    // edegree toward `to` grows, which does not affect
                    // this pass's queue key (vwgt - edeg[other]).
                    st.edeg[u][to] += hw;
                } else if labels[u] == other as u8 {
                    // Pulled into the separator.
                    st.bnd_insert(u);
                    mind.push(u as i32);
                    labels[u] = PART_SEP;
                    st.pwgts[other] -= graph.vwgt[u] as i64;
                    let mut e = [0i64; 2];
                    for kk in graph.xadj[u] as usize..graph.xadj[u + 1] as usize {
                        let w = graph.adjncy[kk] as usize;
                        let lw = labels[w];
                        if lw != PART_SEP {
                            e[lw as usize] += graph.vwgt[w] as i64;
                        } else {
                            st.edeg[w][other] -= graph.vwgt[u] as i64;
                            // One-sided moves: w is still in the queue.
                            if queue.contains(w as i32) {
                                queue.update(
                                    w as i32,
                                    graph.vwgt[w] as i64 - st.edeg[w][other],
                                );
                            }
                        }
                    }
                    st.edeg[u] = e;
                    queue.insert(u as i32, graph.vwgt[u] as i64 - e[other]);
                }
            }
            mptr.push(mind.len());
        }

        // Roll back to the best prefix.
        while swaps.len() as i64 > mincutorder + 1 {
            let higain = swaps.pop().expect("non-empty swaps");
            let hv = higain as usize;
            debug_assert_eq!(labels[hv], to as u8);
            let hw = graph.vwgt[hv] as i64;

            st.pwgts[2] += hw;
            st.pwgts[to] -= hw;
            labels[hv] = PART_SEP;
            st.bnd_insert(hv);

            let mut e = [0i64; 2];
            for k in graph.xadj[hv] as usize..graph.xadj[hv + 1] as usize {
                let u = graph.adjncy[k] as usize;
                let lu = labels[u];
                if lu == PART_SEP {
                    st.edeg[u][to] -= hw;
                } else {
                    e[lu as usize] += graph.vwgt[u] as i64;
                }
            }
            st.edeg[hv] = e;

            // Push the vertices this swap pulled in back out.
            let s = swaps.len();
            for &kv in &mind[mptr[s]..mptr[s + 1]] {
                let u = kv as usize;
                debug_assert_eq!(labels[u], PART_SEP);
                labels[u] = other as u8;
                st.pwgts[other] += graph.vwgt[u] as i64;
                st.pwgts[2] -= graph.vwgt[u] as i64;
                st.bnd_delete(u);
                for kk in graph.xadj[u] as usize..graph.xadj[u + 1] as usize {
                    let w = graph.adjncy[kk] as usize;
                    if labels[w] == PART_SEP {
                        st.edeg[w][other] += graph.vwgt[u] as i64;
                    }
                }
            }
            mptr.pop();
        }

        debug_assert_eq!(mincut, st.pwgts[2]);
        pass += 1;

        if pass.is_multiple_of(2) && (mincutorder == -1 || mincut >= initcut) {
            break;
        }
    }
    st.pwgts[2]
}

/// Node-separator balance pass (METIS `FM_2WayNodeBalance`). Moves
/// separator vertices greedily into the lighter side until the sides
/// are balanced. No rollback - every accepted move is kept.
pub(crate) fn fm_node_balance(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    rng: &mut SplitMix,
) {
    let n = graph.nvtxs as usize;
    let mut st = NodeState::compute(graph, labels);
    let mult = 0.5 * (1.0 + max_imbalance);

    let badmaxpwgt = (mult * (st.pwgts[0] + st.pwgts[1]) as f64) as i64;
    if st.pwgts[0].max(st.pwgts[1]) < badmaxpwgt {
        return;
    }
    let total = st.pwgts[0] + st.pwgts[1] + st.pwgts[2];
    if (st.pwgts[0] - st.pwgts[1]).abs() < 3 * total / n.max(1) as i64 {
        return;
    }

    let to = if st.pwgts[0] < st.pwgts[1] { 0usize } else { 1 };
    let other = 1 - to;

    let mut queue = MaxHeap::new(n);
    let mut moved: Vec<bool> = vec![false; n];

    let mut order: Vec<i32> = st.bndind.clone();
    rng.shuffle(&mut order);
    for &v in &order {
        queue.insert(v, graph.vwgt[v as usize] as i64 - st.edeg[v as usize][other]);
    }

    while let Some(higain) = queue.pop() {
        let hv = higain as usize;
        let hw = graph.vwgt[hv] as i64;
        moved[hv] = true;

        let gain = hw - st.edeg[hv][other];
        let badmaxpwgt = (mult * (st.pwgts[0] + st.pwgts[1]) as f64) as i64;

        if st.pwgts[to] > st.pwgts[other] {
            break;
        }
        if gain < 0 && st.pwgts[other] < badmaxpwgt {
            break;
        }
        if st.pwgts[to] + hw > badmaxpwgt {
            continue;
        }

        st.pwgts[2] -= gain;
        st.bnd_delete(hv);
        st.pwgts[to] += hw;
        labels[hv] = to as u8;

        for k in graph.xadj[hv] as usize..graph.xadj[hv + 1] as usize {
            let u = graph.adjncy[k] as usize;
            if labels[u] == PART_SEP {
                st.edeg[u][to] += hw;
            } else if labels[u] == other as u8 {
                st.bnd_insert(u);
                labels[u] = PART_SEP;
                st.pwgts[other] -= graph.vwgt[u] as i64;
                let mut e = [0i64; 2];
                for kk in graph.xadj[u] as usize..graph.xadj[u + 1] as usize {
                    let w = graph.adjncy[kk] as usize;
                    let lw = labels[w];
                    if lw != PART_SEP {
                        e[lw as usize] += graph.vwgt[w] as i64;
                    } else {
                        st.edeg[w][other] -= graph.vwgt[u] as i64;
                        if !moved[w] && queue.contains(w as i32) {
                            queue.update(w as i32, graph.vwgt[w] as i64 - st.edeg[w][other]);
                        }
                    }
                }
                st.edeg[u] = e;
                queue.insert(u as i32, graph.vwgt[u] as i64 - e[other]);
            }
        }
    }
}

/// Debug validation: labels form a valid tri-section (every A-B edge
/// is blocked by the separator).
#[cfg(test)]
pub(crate) fn is_valid_trisection(graph: &Graph, labels: &[u8]) -> bool {
    for v in 0..graph.nvtxs as usize {
        let lv = labels[v];
        if lv != PART_A && lv != PART_B {
            continue;
        }
        for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
            let u = graph.adjncy[k] as usize;
            let lu = labels[u];
            if (lv == PART_A && lu == PART_B) || (lv == PART_B && lu == PART_A) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fm_refine::separator_weight;
    use crate::initial_partition::initial_bisect_ggp;
    use crate::separator::construct_separator;
    use rslab_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
            set.insert((i, j));
            set.insert((j, i));
        }
        let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
        for &(r, c) in &set {
            cols[c].push(r as i32);
        }
        for col in &mut cols {
            col.sort();
        }
        let mut col_ptr: Vec<i32> = vec![0];
        let mut row_idx: Vec<i32> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as i32);
        }
        (col_ptr, row_idx)
    }

    fn grid(m: usize, n: usize) -> Graph {
        let idx = |r: usize, c: usize| r * n + c;
        let total = m * n;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        let (cp, ri) = csc_from_triples(total, &t);
        let pat = CscPattern::new(total, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    /// Build a valid trisection on a grid via GGP + König, then check
    /// the refiner's invariants.
    fn refined_grid_case(m: usize, n: usize, seed: u64) -> (Graph, Vec<u8>, i64) {
        let g = grid(m, n);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(seed);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        construct_separator(&g, &mut labels);
        let before = separator_weight(&g, &labels);
        let after = fm_node_refine_1sided(&g, &mut labels, 0.20, 10, &mut rng);
        (g, labels, before - after)
    }

    #[test]
    fn refine_preserves_trisection_and_bookkeeping() {
        for seed in [1u64, 7, 21, 33] {
            let (g, labels, _) = refined_grid_case(12, 12, seed);
            assert!(is_valid_trisection(&g, &labels), "seed {seed}");
            // Returned weight must match a from-scratch recount.
            let mut labels2 = labels.clone();
            let mut rng = SplitMix::new(99);
            let w = fm_node_refine_1sided(&g, &mut labels2, 0.20, 10, &mut rng);
            assert_eq!(w, separator_weight(&g, &labels2), "I1 bookkeeping");
            assert!(is_valid_trisection(&g, &labels2));
        }
    }

    #[test]
    fn refine_never_grows_separator() {
        for seed in [1u64, 5, 17] {
            let (_, _, saved) = refined_grid_case(16, 16, seed);
            assert!(saved >= 0, "separator grew by {} (seed {seed})", -saved);
        }
    }

    #[test]
    fn refine_finds_thin_separator_on_grid_band() {
        // 8x8 grid with a fat 3-column separator band: columns 3,4,5
        // SEP, cols 0-2 A, cols 6-7 B. Optimal is a single column (8).
        let g = grid(8, 8);
        let mut labels: Vec<u8> = (0..64u8)
            .map(|k| match k % 8 {
                0..=2 => PART_A,
                3..=5 => PART_SEP,
                _ => PART_B,
            })
            .collect();
        let before = separator_weight(&g, &labels);
        assert_eq!(before, 24);
        let mut rng = SplitMix::new(3);
        let after = fm_node_refine_1sided(&g, &mut labels, 0.20, 10, &mut rng);
        assert!(is_valid_trisection(&g, &labels));
        assert_eq!(after, separator_weight(&g, &labels), "I1 bookkeeping");
        assert!(
            after <= 10,
            "node FM must thin a 3-wide band toward a column, got {after}"
        );
    }

    #[test]
    fn balance_moves_toward_lighter_side() {
        // Heavily imbalanced trisection on a 12x12 grid: col 1 = SEP,
        // col 0 = A (12 vertices), cols 2.. = B (120 vertices).
        let g = grid(12, 12);
        let mut labels: Vec<u8> = (0..144u16)
            .map(|k| match k % 12 {
                0 => PART_A,
                1 => PART_SEP,
                _ => PART_B,
            })
            .collect();
        assert!(is_valid_trisection(&g, &labels));
        let mut rng = SplitMix::new(11);
        fm_node_balance(&g, &mut labels, 0.20, &mut rng);
        assert!(is_valid_trisection(&g, &labels));
        let a: i64 = labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == PART_A)
            .map(|(v, _)| g.vwgt[v] as i64)
            .sum();
        let b: i64 = labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == PART_B)
            .map(|(v, _)| g.vwgt[v] as i64)
            .sum();
        assert!(
            a.max(b) < 132,
            "balance must reduce the 12/120 imbalance, got {a}/{b}"
        );
    }

    #[test]
    fn refine_deterministic_with_seed() {
        let g = grid(14, 14);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mk = || {
            let mut rng = SplitMix::new(42);
            let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
            construct_separator(&g, &mut labels);
            let w = fm_node_refine_1sided(&g, &mut labels, 0.20, 10, &mut rng);
            (labels, w)
        };
        let (l1, w1) = mk();
        let (l2, w2) = mk();
        assert_eq!(w1, w2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn empty_separator_is_noop() {
        let g = grid(4, 4);
        let mut labels = vec![PART_A; 16];
        let mut rng = SplitMix::new(1);
        let w = fm_node_refine_1sided(&g, &mut labels, 0.20, 10, &mut rng);
        assert_eq!(w, 0);
        assert_eq!(labels, vec![PART_A; 16]);
        fm_node_balance(&g, &mut labels, 0.20, &mut rng);
        assert_eq!(labels, vec![PART_A; 16]);
    }
}
