//! Multilevel coarsening.
//!
//! Sorted Heavy-Edge Matching (SHEM): iterate vertices in ascending
//! degree order (ties broken by a seeded random permutation) and, for
//! each unmatched vertex, pair it with its unmatched neighbor of
//! maximum edge weight (ties broken by lower vertex id). Visiting
//! low-degree vertices first - the "sorted" in SHEM - keeps a leaf
//! from being stranded when its sole neighbor is claimed by a
//! higher-degree vertex. If the coarse graph keeps more than the
//! configured share of the vertices (default 0.85 per METIS 5.2.0), a
//! simple 2-hop fallback pairs unmatched vertices that share a common
//! neighbor. This keeps power-law-degree KKT graphs from stalling
//! the coarsening pipeline.
//!
//! Contraction then folds the fine graph into a coarse graph:
//! parallel edges between the same coarse endpoints are **summed**,
//! self-loops (edges inside a merged pair) are **dropped**, vertex
//! weights accumulate.
//!
//! References:
//! - Karypis & Kumar, "A Fast and High Quality Multilevel Scheme",
//!   section 3.1 (matching) and section 3.2 (graph contraction).
//! - METIS 5.2.0 `libmetis/coarsen.c` (`Match_SHEM`, `Match_2Hop`,
//!   `CreateCoarseGraph`).

use crate::graph::Graph;
use crate::rng::SplitMix;
use crate::MetisOptions;
use rslab_ordering_core::OrderingError;

const UNMATCHED: i32 = -1;

/// Output of one coarsening level.
#[derive(Debug)]
pub struct CoarseGraph {
    /// Coarser graph.
    pub graph: Graph,
    /// Map from fine vertex index to coarse vertex index. Length
    /// `fine.nvtxs`.
    pub cmap: Vec<i32>,
}

/// Coarsening-pass diagnostics (accumulated into `MetisStats`).
#[derive(Debug, Default)]
pub struct CoarsenCounters {
    pub n_two_hop_fallbacks: u32,
}

/// Coarsen a single level. Returns the coarse graph plus a fine-to-
/// coarse vertex map. Always produces a valid result; when matching
/// is sparse the coarse graph can be nearly as large as the fine
/// graph (caller is responsible for detecting stalled coarsening).
pub fn coarsen_level(
    fine: &Graph,
    rng: &mut SplitMix,
    two_hop_threshold: f64,
    parallel_min_edges: usize,
    counters: &mut CoarsenCounters,
) -> CoarseGraph {
    let n = fine.nvtxs as usize;
    let mut match_: Vec<i32> = vec![UNMATCHED; n];
    let mut cmap: Vec<i32> = vec![-1; n];

    let mut order: Vec<i32> = (0..fine.nvtxs).collect();
    rng.shuffle(&mut order);
    order.sort_by_key(|&v| fine.xadj[v as usize + 1] - fine.xadj[v as usize]);

    let mut cnvtxs: i32 = 0;
    for &v in &order {
        let vu = v as usize;
        if match_[vu] != UNMATCHED {
            continue;
        }
        let lo = fine.xadj[vu] as usize;
        let hi = fine.xadj[vu + 1] as usize;
        let mut best: i32 = -1;
        let mut best_w: i32 = -1;
        for k in lo..hi {
            let u = fine.adjncy[k];
            if match_[u as usize] != UNMATCHED {
                continue;
            }
            let w = fine.wgt(k);
            if w > best_w || (w == best_w && (best < 0 || u < best)) {
                best = u;
                best_w = w;
            }
        }
        if best >= 0 {
            match_[vu] = best;
            match_[best as usize] = v;
            cmap[vu] = cnvtxs;
            cmap[best as usize] = cnvtxs;
        } else {
            match_[vu] = v;
            cmap[vu] = cnvtxs;
        }
        cnvtxs += 1;
    }

    let reduction_ratio = cnvtxs as f64 / fine.nvtxs.max(1) as f64;
    if reduction_ratio > two_hop_threshold && fine.nvtxs >= crate::constants::TWO_HOP_MIN_VERTICES {
        counters.n_two_hop_fallbacks += 1;
        cnvtxs = two_hop_pass(fine, &mut match_, &mut cmap);
    }

    let graph = contract(fine, &cmap, cnvtxs, parallel_min_edges);
    CoarseGraph { graph, cmap }
}

/// Run `coarsen_level` in a loop until the graph is small enough or
/// coarsening stalls (reduction ratio < 5%). Returns the hierarchy
/// of coarse graphs, finest first, or `Stopped` once the caller's stop
/// flag is set.
pub fn coarsen(
    fine: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    counters: &mut CoarsenCounters,
) -> Result<Vec<CoarseGraph>, OrderingError> {
    let mut levels: Vec<CoarseGraph> = Vec::new();
    let mut prev_nvtxs = fine.nvtxs;
    loop {
        let cur = levels.last().map_or(fine, |level| &level.graph);
        crate::node_nd::stopped(opts)?;
        if cur.nvtxs <= opts.coarsen_floor as i32 {
            break;
        }
        let level = coarsen_level(
            cur,
            rng,
            opts.two_hop_ratio_threshold,
            opts.parallel_min_edges
                .unwrap_or(crate::constants::PAR_MIN_EDGES),
            counters,
        );
        let new_nvtxs = level.graph.nvtxs;
        if new_nvtxs == 0
            || new_nvtxs as f64 > crate::constants::COARSEN_STALL_RATIO * prev_nvtxs as f64
        {
            if new_nvtxs > 0 && new_nvtxs < prev_nvtxs {
                levels.push(level);
            }
            break;
        }
        prev_nvtxs = new_nvtxs;
        levels.push(level);
    }
    Ok(levels)
}

fn two_hop_pass(fine: &Graph, match_: &mut [i32], cmap: &mut [i32]) -> i32 {
    for v in 0..fine.nvtxs {
        let vu = v as usize;
        if match_[vu] != v {
            continue;
        }
        let lo = fine.xadj[vu] as usize;
        let hi = fine.xadj[vu + 1] as usize;
        let mut partner: i32 = -1;
        'outer: for k in lo..hi {
            let mid = fine.adjncy[k];
            let mlo = fine.xadj[mid as usize] as usize;
            let mhi = fine.xadj[(mid as usize) + 1] as usize;
            for &w in &fine.adjncy[mlo..mhi] {
                if w == v {
                    continue;
                }
                if match_[w as usize] == w {
                    partner = w;
                    break 'outer;
                }
            }
        }
        if partner >= 0 {
            match_[vu] = partner;
            match_[partner as usize] = v;
        }
    }

    cmap.iter_mut().for_each(|c| *c = -1);
    let mut cnvtxs: i32 = 0;
    for v in 0..fine.nvtxs {
        let vu = v as usize;
        if cmap[vu] >= 0 {
            continue;
        }
        cmap[vu] = cnvtxs;
        let u = match_[vu];
        if u != v {
            cmap[u as usize] = cnvtxs;
        }
        cnvtxs += 1;
    }
    cnvtxs
}

fn contract(fine: &Graph, cmap: &[i32], cnvtxs: i32, parallel_min_edges: usize) -> Graph {
    let cn = cnvtxs as usize;
    let n = fine.nvtxs as usize;
    let mut vwgt: Vec<i32> = vec![0; cn];
    for v in 0..n {
        vwgt[cmap[v] as usize] = vwgt[cmap[v] as usize].saturating_add(fine.vwgt[v]);
    }
    let mut head: Vec<i32> = vec![-1; cn];
    let mut next: Vec<i32> = vec![-1; n];
    for v in 0..n {
        let c = cmap[v] as usize;
        next[v] = head[c];
        head[c] = v as i32;
    }
    let block = |(slot, touched): &mut (Vec<[i32; 2]>, Vec<i32>),
                 (c0, c1, adj, wgt): (usize, usize, &mut [i32], &mut [i32])| {
        let mut ends = Vec::with_capacity(c1 - c0);
        let mut len = 0;
        for (c, &first) in head.iter().enumerate().take(c1).skip(c0) {
            touched.clear();
            let mut v = first;
            while v >= 0 {
                let vu = v as usize;
                let lo = fine.xadj[vu] as usize;
                let hi = fine.xadj[vu + 1] as usize;
                for k in lo..hi {
                    let nbr = fine.adjncy[k];
                    let cn2 = cmap[nbr as usize];
                    if cn2 == c as i32 {
                        continue;
                    }
                    let sl = &mut slot[cn2 as usize];
                    if sl[0] != c as i32 {
                        *sl = [c as i32, fine.wgt(k)];
                        touched.push(cn2);
                    } else {
                        sl[1] = sl[1].saturating_add(fine.wgt(k));
                    }
                }
                v = next[vu];
            }
            for &tgt in touched.iter() {
                adj[len] = tgt;
                wgt[len] = slot[tgt as usize][1];
                len += 1;
            }
            ends.push(len);
        }
        (ends, len)
    };

    let blocks = if fine.adjncy.len() >= parallel_min_edges && rayon::current_num_threads() > 1 {
        (crate::constants::FINE_PIECES_PER_WORKER * rayon::current_num_threads()).min(cn)
    } else {
        1
    };
    let first = |b: usize| b * cn / blocks;
    let bounds: Vec<usize> = (0..blocks)
        .map(|b| {
            let mut bound = 0usize;
            for &f in &head[first(b)..first(b + 1)] {
                let mut v = f;
                while v >= 0 {
                    let vu = v as usize;
                    bound += (fine.xadj[vu + 1] - fine.xadj[vu]) as usize;
                    v = next[vu];
                }
            }
            bound
        })
        .collect();
    let mut adjncy = vec![0i32; fine.adjncy.len()];
    let mut adjwgt = vec![0i32; fine.adjncy.len()];
    let mut pieces = Vec::with_capacity(blocks);
    let (mut adj_rest, mut wgt_rest) = (&mut adjncy[..], &mut adjwgt[..]);
    for (b, &bound) in bounds.iter().enumerate() {
        let (adj, a_rest) = std::mem::take(&mut adj_rest).split_at_mut(bound);
        let (wgt, w_rest) = std::mem::take(&mut wgt_rest).split_at_mut(bound);
        (adj_rest, wgt_rest) = (a_rest, w_rest);
        pieces.push((first(b), first(b + 1), adj, wgt));
    }
    let init = || (vec![[-1i32, 0]; cn], Vec::new());
    let parts: Vec<(Vec<usize>, usize)> = if blocks <= 1 {
        let mut state = init();
        pieces.into_iter().map(|p| block(&mut state, p)).collect()
    } else {
        use rayon::prelude::*;
        pieces.into_par_iter().map_init(init, block).collect()
    };
    let mut xadj: Vec<i32> = Vec::with_capacity(cn + 1);
    xadj.push(0);
    let (mut src, mut len) = (0, 0);
    for ((ends, count), bound) in parts.into_iter().zip(bounds) {
        adjncy.copy_within(src..src + count, len);
        adjwgt.copy_within(src..src + count, len);
        xadj.extend(ends.iter().map(|&e| (len + e) as i32));
        len += count;
        src += bound;
    }
    for v in [&mut adjncy, &mut adjwgt] {
        v.truncate(len);
        v.shrink_to_fit();
    }
    Graph {
        nvtxs: cnvtxs,
        xadj,
        adjncy,
        vwgt,
        adjwgt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::csc_from_triples;
    use crate::tests::grid;
    use rslab_ordering_core::CscPattern;

    #[test]
    fn parallel_contract_matches_serial() {
        let g = grid(500, 500);
        assert!(g.adjncy.len() >= 200_000);
        let level = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut rng = SplitMix::new(3);
                coarsen_level(&g, &mut rng, 0.95, 200_000, &mut CoarsenCounters::default())
            })
        };
        let (a, b) = (level(1), level(4));
        assert_eq!(a.cmap, b.cmap);
        assert_eq!(a.graph.nvtxs, b.graph.nvtxs);
        assert_eq!(a.graph.xadj, b.graph.xadj);
        assert_eq!(a.graph.adjncy, b.graph.adjncy);
        assert_eq!(a.graph.adjwgt, b.graph.adjwgt);
        assert_eq!(a.graph.vwgt, b.graph.vwgt);
    }

    fn tridiag(n: usize) -> Graph {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            if i + 1 < n {
                t.push((i, i + 1));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn assert_valid_coarse(fine: &Graph, cg: &CoarseGraph) {
        assert_eq!(cg.cmap.len(), fine.nvtxs as usize);
        for &c in &cg.cmap {
            assert!(c >= 0 && c < cg.graph.nvtxs, "cmap entry OOB: {}", c);
        }
        let fine_total: i64 = fine.vwgt.iter().map(|&x| x as i64).sum();
        let coarse_total: i64 = cg.graph.vwgt.iter().map(|&x| x as i64).sum();
        assert_eq!(fine_total, coarse_total, "vwgt conservation");
        for v in 0..cg.graph.nvtxs {
            let lo = cg.graph.xadj[v as usize] as usize;
            let hi = cg.graph.xadj[(v + 1) as usize] as usize;
            for &u in &cg.graph.adjncy[lo..hi] {
                assert_ne!(u, v, "self-loop in coarse graph at {}", v);
            }
        }
        for v in 0..cg.graph.nvtxs {
            let lo = cg.graph.xadj[v as usize] as usize;
            let hi = cg.graph.xadj[(v + 1) as usize] as usize;
            for &u in &cg.graph.adjncy[lo..hi] {
                let ulo = cg.graph.xadj[u as usize] as usize;
                let uhi = cg.graph.xadj[(u + 1) as usize] as usize;
                assert!(
                    cg.graph.adjncy[ulo..uhi].contains(&v),
                    "asymmetric coarse edge {} -> {}",
                    v,
                    u
                );
            }
        }
    }

    #[test]
    fn coarsen_grid_8x8_halves_vertices() {
        let g = grid(8, 8);
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, 200_000, &mut ctr);
        assert_valid_coarse(&g, &cg);
        assert!(
            cg.graph.nvtxs < g.nvtxs,
            "coarse graph must shrink (was {}, now {})",
            g.nvtxs,
            cg.graph.nvtxs
        );
        assert!(
            cg.graph.nvtxs as f64 <= 0.70 * g.nvtxs as f64,
            "reduction ratio on 8x8 grid should be <= 0.70, got {}/{}",
            cg.graph.nvtxs,
            g.nvtxs
        );
    }

    #[test]
    fn shem_matches_low_degree_before_hub() {
        let t = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 2),
        ];
        let (cp, ri) = csc_from_triples(4, &t);
        let pat = CscPattern::new(4, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, 200_000, &mut ctr);
        assert_valid_coarse(&g, &cg);
        assert_eq!(
            cg.graph.nvtxs, 2,
            "SHEM must match the degree-1 leaf with the hub and pair \
             the remaining two vertices for 2 coarse vertices; got {}",
            cg.graph.nvtxs
        );
    }

    #[test]
    fn coarsen_tridiag_10() {
        let g = tridiag(10);
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let cg = coarsen_level(&g, &mut rng, 0.85, 200_000, &mut ctr);
        assert_valid_coarse(&g, &cg);
        assert!(cg.graph.nvtxs <= 6);
    }

    #[test]
    fn coarsen_is_deterministic_with_seed() {
        let g = grid(6, 6);
        let mut r1 = SplitMix::new(42);
        let mut r2 = SplitMix::new(42);
        let mut c1 = CoarsenCounters::default();
        let mut c2 = CoarsenCounters::default();
        let a = coarsen_level(&g, &mut r1, 0.85, 200_000, &mut c1);
        let b = coarsen_level(&g, &mut r2, 0.85, 200_000, &mut c2);
        assert_eq!(a.cmap, b.cmap);
        assert_eq!(a.graph.xadj, b.graph.xadj);
        assert_eq!(a.graph.adjncy, b.graph.adjncy);
    }

    #[test]
    fn coarsen_hierarchy_shrinks_monotonically() {
        let g = grid(12, 12);
        let opts = MetisOptions {
            coarsen_floor: 20,
            ..Default::default()
        };
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let levels = coarsen(&g, &opts, &mut rng, &mut ctr).unwrap();
        assert!(!levels.is_empty(), "must produce at least one level");
        let mut prev = g.nvtxs;
        for (i, lvl) in levels.iter().enumerate() {
            assert!(
                lvl.graph.nvtxs < prev,
                "level {} did not shrink: {} >= {}",
                i,
                lvl.graph.nvtxs,
                prev
            );
            prev = lvl.graph.nvtxs;
        }
        assert!(
            prev <= opts.coarsen_floor as i32
                || levels.last().expect("non-empty").graph.nvtxs < g.nvtxs
        );
    }

    #[test]
    fn stall_keeps_first_level_that_shrank() {
        let mut t: Vec<(usize, usize)> = vec![(0, 0)];
        for l in 1..=24usize {
            t.push((l, l));
            t.push((0, l));
        }
        let (cp, ri) = csc_from_triples(25, &t);
        let pat = CscPattern::new(25, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let opts = MetisOptions {
            coarsen_floor: 4,
            two_hop_ratio_threshold: 10.0,
            ..Default::default()
        };
        let mut rng = SplitMix::new(1);
        let mut ctr = CoarsenCounters::default();
        let levels = coarsen(&g, &opts, &mut rng, &mut ctr).unwrap();
        assert!(
            !levels.is_empty(),
            "a first level that shrank 25->24 must be kept, not discarded"
        );
        assert_eq!(levels.last().expect("non-empty").graph.nvtxs, 24);
    }

    #[test]
    fn contract_sums_parallel_edges() {
        let n = 4;
        let t = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 1),
            (3, 2),
        ];
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let fine = Graph::from_csc_pattern(&pat).unwrap();
        let cmap: Vec<i32> = vec![0, 1, 2, 0];
        let coarse = contract(&fine, &cmap, 3, 200_000);
        let mut got: Vec<(i32, i32)> = Vec::new();
        let lo = coarse.xadj[0] as usize;
        let hi = coarse.xadj[1] as usize;
        for k in lo..hi {
            got.push((coarse.adjncy[k], coarse.adjwgt[k]));
        }
        got.sort();
        assert_eq!(got, vec![(1, 2), (2, 2)]);
        for v in 0..coarse.nvtxs {
            let lo = coarse.xadj[v as usize] as usize;
            let hi = coarse.xadj[(v + 1) as usize] as usize;
            for k in lo..hi {
                let u = coarse.adjncy[k];
                let w = coarse.adjwgt[k];
                let ulo = coarse.xadj[u as usize] as usize;
                let uhi = coarse.xadj[(u + 1) as usize] as usize;
                let mut ok = false;
                for kk in ulo..uhi {
                    if coarse.adjncy[kk] == v {
                        assert_eq!(coarse.adjwgt[kk], w, "edge-weight symmetry");
                        ok = true;
                        break;
                    }
                }
                assert!(ok, "edge {}->{} has no reverse", v, u);
            }
        }
    }
}
