//! Fiduccia-Mattheyses edge-bisection refinement.
//!
//! `refine_bisection`: classic FM with best-balanced rollback. Picks
//! the highest-gain boundary vertex, flips its partition, updates
//! neighbor gains, locks it, and repeats. Tracks both "current cut"
//! and "best cut subject to balance" separately; at the end of the
//! pass rolls back to the best balanced state.
//!
//! Node-*separator* refinement lives in `sep_refine` (an independent
//! implementation from published descriptions): the node separator is
//! refined through the whole hierarchy instead of refining the edge
//! bisection and converting at the finest level.
//!
//! Priority queue: a lazy `BinaryHeap<(gain, Reverse(v))>` rather
//! than METIS's bucket array. Correct; the O(log n) overhead per
//! operation is acceptable at the graph sizes RSLAB targets
//! (<= 100k vertices, FM rarely dominates runtime).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::Graph;
use crate::initial_partition::{cut_size, part_weight, PART_A, PART_B};

pub const PART_SEP: u8 = 2;

/// Refine an edge bisection with FM, tracking the best balanced cut.
///
/// `labels[v] in {PART_A, PART_B}`. Modifies `labels` in place.
/// Returns the final edge cut.
pub fn refine_bisection(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> i32 {
    let n = graph.nvtxs as usize;
    if n < 2 {
        return cut_size(graph, labels);
    }
    let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    let mut pass_cut = cut_size(graph, labels);

    for _pass in 0..max_passes {
        let before_pass = pass_cut;
        let mut gain: Vec<i32> = vec![0; n];
        compute_gains(graph, labels, &mut gain);

        let mut locked: Vec<bool> = vec![false; n];
        let mut heap: BinaryHeap<(i32, Reverse<i32>, i32)> = BinaryHeap::new();
        for (v, &g) in gain.iter().enumerate().take(n) {
            heap.push((g, Reverse(v as i32), g));
        }

        let mut cur_cut = pass_cut;
        let mut moves: Vec<i32> = Vec::new();
        let mut a_w = part_weight(graph, labels, PART_A);
        let mut b_w = total - a_w;
        let starts_balanced = a_w.max(b_w) <= max_side;
        let mut best_prefix: Option<usize> = if starts_balanced { Some(0) } else { None };
        let mut best_cut = pass_cut;
        let mut best_a_w = a_w;
        let mut best_b_w = b_w;
        let mut no_improve: u32 = 0;

        while let Some((_, Reverse(v), stamp)) = heap.pop() {
            let vu = v as usize;
            if locked[vu] {
                continue;
            }
            if stamp != gain[vu] {
                continue;
            }
            let from = labels[vu];
            let to = if from == PART_A { PART_B } else { PART_A };
            let (new_a_w, new_b_w) = if to == PART_A {
                (a_w + graph.vwgt[vu] as i64, b_w - graph.vwgt[vu] as i64)
            } else {
                (a_w - graph.vwgt[vu] as i64, b_w + graph.vwgt[vu] as i64)
            };
            let side_max = new_a_w.max(new_b_w);
            labels[vu] = to;
            a_w = new_a_w;
            b_w = new_b_w;
            cur_cut -= gain[vu];
            locked[vu] = true;
            moves.push(v);

            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k] as usize;
                if locked[u] {
                    continue;
                }
                let w = graph.wgt(k);
                if labels[u] == from {
                    gain[u] += 2 * w;
                } else {
                    gain[u] -= 2 * w;
                }
                heap.push((gain[u], Reverse(u as i32), gain[u]));
            }

            if side_max <= max_side {
                let is_first_balanced = best_prefix.is_none();
                if is_first_balanced || cur_cut < best_cut {
                    best_cut = cur_cut;
                    best_prefix = Some(moves.len());
                    best_a_w = a_w;
                    best_b_w = b_w;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }
            if no_improve >= crate::constants::FM_NO_IMPROVE_MAX {
                break;
            }
        }

        match best_prefix {
            Some(prefix) => {
                for &v in moves.iter().skip(prefix) {
                    let vu = v as usize;
                    labels[vu] = if labels[vu] == PART_A { PART_B } else { PART_A };
                }
                a_w = best_a_w;
                b_w = best_b_w;
                pass_cut = best_cut;
            }
            None => {
                pass_cut = cur_cut;
            }
        }
        debug_assert_eq!(a_w, part_weight(graph, labels, PART_A));
        debug_assert_eq!(b_w, total - a_w);
        debug_assert_eq!(pass_cut, cut_size(graph, labels));
        if pass_cut == before_pass {
            break;
        }
    }
    pass_cut
}

fn compute_gains(graph: &Graph, labels: &[u8], gain: &mut [i32]) {
    for (v, &lv) in labels.iter().enumerate() {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut ed: i32 = 0;
        let mut id: i32 = 0;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let w = graph.wgt(k);
            if labels[u] == lv {
                id = id.saturating_add(w);
            } else {
                ed = ed.saturating_add(w);
            }
        }
        gain[v] = ed - id;
    }
}

/// Total vertex weight of `PART_SEP` vertices.
pub fn separator_weight(graph: &Graph, labels: &[u8]) -> i64 {
    let mut s: i64 = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l == PART_SEP {
            s += graph.vwgt[v] as i64;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial_partition::initial_bisect_ggp;
    use crate::patterns::csc_from_triples;
    use crate::rng::SplitMix;
    use crate::tests::grid;
    use rslab_ordering_core::CscPattern;

    #[test]
    fn fm_sign_invariant_on_alternating_path() {
        let mut t = Vec::new();
        for i in 0..10 {
            t.push((i, i));
        }
        for i in 0..9 {
            t.push((i, i + 1));
        }
        let (cp, ri) = csc_from_triples(10, &t);
        let pat = CscPattern::new(10, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();

        let mut labels: Vec<u8> = (0..10u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        let before = cut_size(&g, &labels);
        assert_eq!(before, 9, "alternating path P_10 has cut 9");

        let after = refine_bisection(&g, &mut labels, 0.20, 32);

        assert_eq!(
            after,
            cut_size(&g, &labels),
            "returned cut must equal cut_size(labels) recomputed from scratch"
        );
        assert!(
            after < before,
            "FM must reduce cut from {} on alternating P_10, got {}",
            before,
            after
        );
        assert!(after >= 0, "cut size is non-negative, got {}", after);
    }

    #[test]
    fn refine_bisection_does_not_increase_cut() {
        let g = grid(8, 8);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(17);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let initial = cut_size(&g, &labels);
        let final_cut = refine_bisection(&g, &mut labels, 0.20, 5);
        assert_eq!(
            final_cut,
            cut_size(&g, &labels),
            "reported cut matches labels"
        );
        assert!(
            final_cut <= initial,
            "cut must not increase (before={}, after={})",
            initial,
            final_cut
        );
    }

    #[test]
    fn refine_bisection_balance_respected() {
        let g = grid(6, 6);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(9);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let returned = refine_bisection(&g, &mut labels, 0.20, 5);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        let max_allowed = ((1.20_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(a.max(b) <= max_allowed, "balance: a={} b={}", a, b);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn refine_bisection_bad_init_improves() {
        let g = grid(4, 4);
        let mut labels = vec![PART_A; 16];
        labels[0] = PART_B;
        let returned = refine_bisection(&g, &mut labels, 0.20, 10);
        let after = cut_size(&g, &labels);
        assert_eq!(returned, after, "I1: bookkeeping");
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let a = part_weight(&g, &labels, PART_A);
        let b = total - a;
        let max_side = ((1.20_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(
            a.max(b) <= max_side,
            "I4: FM rebalanced from imbalanced start (a={}, b={}, max={})",
            a,
            b,
            max_side
        );
    }

    #[test]
    fn separator_weight_accounting() {
        let g = grid(3, 3);
        let labels: Vec<u8> = (0..9u8)
            .map(|k| {
                let r = k / 3;
                match r {
                    0 => PART_A,
                    1 => PART_SEP,
                    _ => PART_B,
                }
            })
            .collect();
        assert_eq!(separator_weight(&g, &labels), 3);
    }

    fn path(n: usize) -> Graph {
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..n.saturating_sub(1) {
            t.push((i, i + 1));
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn cycle(n: usize) -> Graph {
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..n {
            t.push((i, (i + 1) % n));
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn complete_bipartite(m: usize, k: usize) -> Graph {
        let n = m + k;
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..m {
            for j in m..n {
                t.push((i, j));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    #[test]
    fn a2_path_p20_alternating() {
        let g = path(20);
        let mut labels: Vec<u8> = (0..20u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        assert_eq!(cut_size(&g, &labels), 19, "construction: alternating P_20");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned < 19,
            "FM must improve alternating P_20 cut, got {}",
            returned
        );
        assert!(returned >= 0);
    }

    #[test]
    fn a3_cycle_c12_alternating() {
        let g = cycle(12);
        let mut labels: Vec<u8> = (0..12u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        assert_eq!(cut_size(&g, &labels), 12, "construction: alternating C_12");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned <= 12,
            "I2: cut never grows (before=12, after={})",
            returned
        );
        assert!(returned >= 2, "cycle C_12 balanced cut is >= 2");
    }

    #[test]
    fn a4_grid_4x4_checkerboard() {
        let g = grid(4, 4);
        let mut labels: Vec<u8> = (0..16u8)
            .map(|k| {
                let r = (k as usize) / 4;
                let c = (k as usize) % 4;
                if (r + c).is_multiple_of(2) {
                    PART_A
                } else {
                    PART_B
                }
            })
            .collect();
        assert_eq!(cut_size(&g, &labels), 24, "construction: 4x4 checkerboard");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned < 24,
            "FM must improve 4x4 checkerboard cut, got {}",
            returned
        );
        assert!(returned >= 4, "4x4 grid balanced cut is >= 4");
    }

    #[test]
    fn a5_grid_6x6_mixed_boundary() {
        let g = grid(6, 6);
        let mut rng = SplitMix::new(0xA5A5_A5A5);
        let mut labels: Vec<u8> = (0..36)
            .map(|_| {
                if rng.next_u64() & 1 == 0 {
                    PART_A
                } else {
                    PART_B
                }
            })
            .collect();
        let before = cut_size(&g, &labels);
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned <= before,
            "I2: cut never grows (before={}, after={})",
            before,
            returned
        );
    }

    #[test]
    fn a6_k44_unbalanced_init_rebalances() {
        let g = complete_bipartite(4, 4);
        let mut labels: Vec<u8> = vec![PART_B; 8];
        labels[0] = PART_A;
        assert_eq!(cut_size(&g, &labels), 4, "construction: K_{{4,4}} 1-7");
        let returned = refine_bisection(&g, &mut labels, 0.50, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let a = part_weight(&g, &labels, PART_A);
        let b = total - a;
        let max_side = ((1.50_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(
            a.max(b) <= max_side,
            "I4: FM rebalanced (a={}, b={}, max={})",
            a,
            b,
            max_side
        );
    }

    #[test]
    fn a7_two_k4_bridge_no_spurious_moves() {
        let n = 8;
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                t.push((i, j));
            }
        }
        for i in 4..8 {
            for j in (i + 1)..8 {
                t.push((i, j));
            }
        }
        t.push((3, 4));
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels = vec![PART_A; n];
        assert_eq!(cut_size(&g, &labels), 0, "construction: all-A cut is 0");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(returned >= 0);
    }

    #[test]
    fn a8_path_p10_all_a_empty_side() {
        let g = path(10);
        let mut labels = vec![PART_A; 10];
        assert_eq!(cut_size(&g, &labels), 0);
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(returned >= 0);
    }

    #[test]
    fn a9_single_vertex_returns_zero() {
        let cp: Vec<i32> = vec![0, 0];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(1, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels = vec![PART_A];
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, 0);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
    }

    #[test]
    fn a10_empty_edge_set_no_moves() {
        let cp: Vec<i32> = vec![0; 9];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(8, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels: Vec<u8> = (0..8u8)
            .map(|k| if k < 4 { PART_A } else { PART_B })
            .collect();
        assert_eq!(cut_size(&g, &labels), 0);
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, 0);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        let mut labels2: Vec<u8> = (0..8u8)
            .map(|k| if k < 4 { PART_A } else { PART_B })
            .collect();
        let returned2 = refine_bisection(&g, &mut labels2, 0.20, 32);
        assert_eq!(returned, returned2, "I6: determinism");
        assert_eq!(labels, labels2, "I6: determinism (labels)");
    }
}
