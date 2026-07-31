//! Node-separator refinement for the multilevel ND pipeline.
//!
//! Independent implementation written from published descriptions of
//! separator-local refinement; no third-party partitioning source code
//! was consulted for this module. Sources:
//!
//! - Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for
//!   Partitioning Irregular Graphs" (SIAM J. Sci. Comput. 1998):
//!   multilevel scheme, refinement applied at every uncoarsening step.
//! - Hendrickson & Rothberg, "Improving the Run Time and Quality of
//!   Nested Dissection Ordering" (SIAM J. Sci. Comput. 1998):
//!   Fiduccia-Mattheyses-style refinement operating directly on a
//!   vertex separator, with hill climbing past locally bad moves.
//! - Ashcraft & Liu, "Applications of the Dulmage-Mendelsohn
//!   Decomposition and Network Flow to Graph Bisection Improvement"
//!   (SIAM J. Matrix Anal. 1998): separator-improvement framing.
//! - `dev/research/metis-node-separator-2026-07.md`: RSLAB's own
//!   evidence chain for why hierarchical node-separator refinement is
//!   the fill lever, and the exact-fill evaluation harness
//!   (`examples/grid_fill.rs`) this module's policies were tuned on.
//!
//! ## The move
//!
//! State is a tri-section `labels[v] ∈ {A, B, SEP}` where SEP blocks
//! every A-B edge. The elementary refinement step takes a separator
//! vertex `v` and assigns it to a chosen side `s`; every neighbor of
//! `v` on the far side must then enter the separator to keep the
//! tri-section valid. Writing `far(v, s)` for the total weight of
//! those neighbors, the separator weight changes by
//! `far(v, s) - w(v)`, so the move's gain is `w(v) - far(v, s)`.
//!
//! ## The pass
//!
//! One pass fixes a target side and repeatedly applies the best-gain
//! move from a lazy max-heap, hill-climbing through negative-gain
//! moves until the heap is exhausted (with a generous patience /
//! overshoot brake purely as termination hygiene). Every label
//! transition is appended to a journal; when the pass ends, the
//! journal suffix past the best observed separator is unwound,
//! restoring the best prefix. Rounds alternate the target side,
//! lighter side first, and stop as soon as a full round yields no net
//! improvement.
//!
//! ## Tie-breaking: newest first
//!
//! Equal-gain heap entries pop **newest first** (a per-pass push
//! counter is the secondary key). Freshly pulled-in vertices and
//! just-updated neighbors are therefore preferred over stale
//! same-gain candidates, so the pass drills along the moving
//! separator front instead of scattering across it; LIFO gain
//! buckets are a known-good choice in the FM literature. Measured on
//! the exact-fill harness (40^3 7-point grid, scalar nnz(L)): random
//! static tie ranks 14.75 M, newest-first 11.06 M, against 20.6 M
//! for AMD and 12.5 M for the best vendor ND ordering we measured
//! (see the research note). Seeding order is shuffled per pass, so
//! different seeds still explore genuinely different climbs.
//!
//! Determinism: given the same seed the result is bit-identical.

use std::collections::BinaryHeap;

use crate::fm_refine::PART_SEP;
use crate::graph::Graph;
use crate::initial_partition::{PART_A, PART_B};
use crate::rng::SplitMix;

/// Hill-climb patience: a pass ends after this many consecutive moves
/// without a new best separator. Effectively unbounded on the graph
/// sizes RSLAB targets; the exact-fill harness saturates two orders of
/// magnitude below this. Exists purely so a pathological climb cannot
/// run away.
const PATIENCE: usize = 1 << 20;

/// Termination hygiene: give up on a climb that has wandered this far
/// (as a multiple of the best weight) above the best separator seen.
/// Verified on the harness to leave results untouched.
const MAX_OVERSHOOT: f64 = 4.0;

/// Weight of the neighbors of `v` lying on side `s` (0 or 1).
#[inline]
fn far_weight(graph: &Graph, labels: &[u8], v: usize, s: usize) -> i64 {
    let mut w = 0i64;
    for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
        let u = graph.adjncy[k] as usize;
        if labels[u] == s as u8 {
            w += graph.vwgt[u] as i64;
        }
    }
    w
}

/// Total weight per label class `[A, B, SEP]`.
fn class_weights(graph: &Graph, labels: &[u8]) -> [i64; 3] {
    let mut w = [0i64; 3];
    for v in 0..graph.nvtxs as usize {
        w[labels[v] as usize] += graph.vwgt[v] as i64;
    }
    w
}

/// Lazy max-heap entry: `(gain, push_seq, vertex, version)`. Entries
/// are never removed eagerly; a popped entry is valid only if the
/// vertex is still in the separator and its version matches the
/// per-vertex counter (the `fm_refine` staleness pattern). The push
/// sequence number makes equal-gain entries pop newest first.
type HeapEntry = (i64, u64, u32, u32);

/// Push the current key of separator vertex `v` for target side `into`.
#[inline]
fn push_key(
    heap: &mut BinaryHeap<HeapEntry>,
    graph: &Graph,
    labels: &[u8],
    version: &[u32],
    seq: &mut u64,
    v: usize,
    into: usize,
) {
    let gain = graph.vwgt[v] as i64 - far_weight(graph, labels, v, 1 - into);
    *seq += 1;
    heap.push((gain, *seq, v as u32, version[v]));
}

/// One journal record: `vertex` previously carried `old_label`.
#[derive(Clone, Copy)]
struct Transition {
    vertex: u32,
    old_label: u8,
}

/// Assign separator vertex `v` to side `into`, pulling its far-side
/// neighbors into the separator. Appends every label transition to the
/// journal, bumps the version of every separator vertex whose key
/// changed, and re-pushes their keys. Returns the separator-weight
/// delta (negative = the separator shrank).
#[allow(clippy::too_many_arguments)]
fn apply_move(
    graph: &Graph,
    labels: &mut [u8],
    class_w: &mut [i64; 3],
    journal: &mut Vec<Transition>,
    heap: &mut BinaryHeap<HeapEntry>,
    version: &mut [u32],
    seq: &mut u64,
    v: usize,
    into: usize,
) -> i64 {
    let from_side = 1 - into;
    let wv = graph.vwgt[v] as i64;

    journal.push(Transition {
        vertex: v as u32,
        old_label: PART_SEP,
    });
    labels[v] = into as u8;
    class_w[PART_SEP as usize] -= wv;
    class_w[into] += wv;
    version[v] = version[v].wrapping_add(1);
    let mut delta = -wv;

    // Far-side neighbors enter the separator; separator vertices
    // adjacent to them get changed keys and are re-pushed lazily.
    for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
        let u = graph.adjncy[k] as usize;
        let lu = labels[u];
        if lu == from_side as u8 {
            let wu = graph.vwgt[u] as i64;
            journal.push(Transition {
                vertex: u as u32,
                old_label: lu,
            });
            labels[u] = PART_SEP;
            class_w[from_side] -= wu;
            class_w[PART_SEP as usize] += wu;
            delta += wu;
            version[u] = version[u].wrapping_add(1);
            push_key(heap, graph, labels, version, seq, u, into);
            // `u` leaving the far side changes the key of every
            // separator vertex adjacent to it.
            for kk in graph.xadj[u] as usize..graph.xadj[u + 1] as usize {
                let t = graph.adjncy[kk] as usize;
                if labels[t] == PART_SEP && t != u {
                    version[t] = version[t].wrapping_add(1);
                    push_key(heap, graph, labels, version, seq, t, into);
                }
            }
        }
    }
    delta
}

/// Undo journal entries beyond `keep`, restoring labels and class
/// weights to the state after the first `keep` transitions.
fn unwind(
    graph: &Graph,
    labels: &mut [u8],
    class_w: &mut [i64; 3],
    journal: &mut Vec<Transition>,
    keep: usize,
) {
    while journal.len() > keep {
        let t = journal.pop().expect("journal non-empty");
        let v = t.vertex as usize;
        let wv = graph.vwgt[v] as i64;
        class_w[labels[v] as usize] -= wv;
        class_w[t.old_label as usize] += wv;
        labels[v] = t.old_label;
    }
}

/// One refinement pass toward `into`. Returns the separator weight
/// after the pass (the best state visited, journal-unwound).
fn side_pass(
    graph: &Graph,
    labels: &mut [u8],
    into: usize,
    side_cap: i64,
    rng: &mut SplitMix,
) -> i64 {
    let n = graph.nvtxs as usize;

    let mut class_w = class_weights(graph, labels);
    let mut version: Vec<u32> = vec![0; n];
    let mut seq: u64 = 0;
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

    // Shuffled seeding: with newest-first ties, the seeding order is
    // the tie order among untouched vertices, so the shuffle is what
    // lets different seeds explore different climbs.
    let mut order: Vec<u32> = (0..n as u32)
        .filter(|&v| labels[v as usize] == PART_SEP)
        .collect();
    rng.shuffle(&mut order);
    for &v in &order {
        push_key(
            &mut heap, graph, labels, &version, &mut seq, v as usize, into,
        );
    }

    let mut journal: Vec<Transition> = Vec::new();
    let mut sep_w = class_w[PART_SEP as usize];
    let mut best_w = sep_w;
    let mut best_len = 0usize;
    let mut since_best = 0usize;

    while let Some((_, _, v32, ver)) = heap.pop() {
        let v = v32 as usize;
        if labels[v] != PART_SEP || version[v] != ver {
            continue; // stale entry
        }
        if class_w[into] + graph.vwgt[v] as i64 > side_cap {
            continue; // too heavy for the target side right now
        }

        sep_w += apply_move(
            graph,
            labels,
            &mut class_w,
            &mut journal,
            &mut heap,
            &mut version,
            &mut seq,
            v,
            into,
        );

        if sep_w < best_w {
            best_w = sep_w;
            best_len = journal.len();
            since_best = 0;
        } else {
            since_best += 1;
            let overshoot = sep_w - best_w;
            if since_best > PATIENCE || overshoot as f64 > MAX_OVERSHOOT * best_w.max(1) as f64 {
                break;
            }
        }
    }

    unwind(graph, labels, &mut class_w, &mut journal, best_len);
    debug_assert_eq!(class_w[PART_SEP as usize], best_w);
    best_w
}

/// Refine a node separator in place. Runs up to `max_rounds` rounds of
/// two side-passes each (lighter side first); stops early once a full
/// round brings no improvement. Returns the final separator weight.
///
/// `labels[v] ∈ {PART_A, PART_B, PART_SEP}` must form a valid
/// tri-section on entry; it does on exit. Deterministic per seed.
pub(crate) fn refine_node_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_rounds: u32,
    rng: &mut SplitMix,
) -> i64 {
    let class_w = class_weights(graph, labels);
    let mut sep_w = class_w[PART_SEP as usize];
    if sep_w == 0 {
        return 0;
    }
    // Cap on either side's weight: half the total graph weight plus
    // the imbalance allowance. The base includes the separator, since
    // its vertices eventually land on one of the sides; excluding it
    // would freeze refinement whenever the separator is fat.
    let total = class_w[0] + class_w[1] + class_w[2];
    let side_cap = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    for _round in 0..max_rounds {
        let w = class_weights(graph, labels);
        let first = if w[PART_A as usize] <= w[PART_B as usize] {
            PART_A as usize
        } else {
            PART_B as usize
        };
        let after_first = side_pass(graph, labels, first, side_cap, rng);
        let after_second = side_pass(graph, labels, 1 - first, side_cap, rng);
        let round_end = after_first.min(after_second);
        if round_end >= sep_w {
            sep_w = round_end;
            break;
        }
        sep_w = round_end;
    }
    sep_w
}

/// Rebalance a tri-section by shifting separator vertices into the
/// lighter side. Accepts negative-gain moves; keeps every accepted
/// move (no rollback); stops once the sides are within the imbalance
/// tolerance or the lighter side would overtake the heavier one.
pub(crate) fn balance_node_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    rng: &mut SplitMix,
) {
    let n = graph.nvtxs as usize;
    let mut class_w = class_weights(graph, labels);
    let a = class_w[PART_A as usize];
    let b = class_w[PART_B as usize];
    let tolerance = ((a + b) as f64 * max_imbalance / 2.0).ceil() as i64;
    if (a - b).abs() <= tolerance.max(1) {
        return;
    }
    let into = if a < b {
        PART_A as usize
    } else {
        PART_B as usize
    };
    let heavy = 1 - into;

    let mut version: Vec<u32> = vec![0; n];
    let mut seq: u64 = 0;
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut order: Vec<u32> = (0..n as u32)
        .filter(|&v| labels[v as usize] == PART_SEP)
        .collect();
    rng.shuffle(&mut order);
    for &v in &order {
        push_key(
            &mut heap, graph, labels, &version, &mut seq, v as usize, into,
        );
    }

    let mut journal: Vec<Transition> = Vec::new(); // kept, never unwound
    while let Some((_, _, v32, ver)) = heap.pop() {
        let v = v32 as usize;
        if labels[v] != PART_SEP || version[v] != ver {
            continue;
        }
        // Never overtake the heavy side; a lighter candidate may
        // still fit, so skip rather than stop.
        if class_w[into] + graph.vwgt[v] as i64 > class_w[heavy] {
            continue;
        }
        apply_move(
            graph,
            labels,
            &mut class_w,
            &mut journal,
            &mut heap,
            &mut version,
            &mut seq,
            v,
            into,
        );
        if (class_w[PART_A as usize] - class_w[PART_B as usize]).abs() <= tolerance.max(1) {
            break;
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
        let after = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
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
            let w = refine_node_separator(&g, &mut labels2, 0.20, 10, &mut rng);
            assert_eq!(w, separator_weight(&g, &labels2), "bookkeeping");
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
        let after = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
        assert!(is_valid_trisection(&g, &labels));
        assert_eq!(after, separator_weight(&g, &labels), "bookkeeping");
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
        balance_node_separator(&g, &mut labels, 0.20, &mut rng);
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
            let w = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
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
        let w = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
        assert_eq!(w, 0);
        assert_eq!(labels, vec![PART_A; 16]);
        balance_node_separator(&g, &mut labels, 0.20, &mut rng);
        assert_eq!(labels, vec![PART_A; 16]);
    }
}
