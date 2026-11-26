//! Recursive nested-dissection driver.
//!
//! Top-level algorithm (Karypis & Kumar 1998, section 4, George 1973):
//!
//! 1. Split graph into connected components. Each component is
//!    ordered independently and its numbering is concatenated.
//! 2. On a single component:
//!    - If its size is below `nd_to_amd_switch`, hand off to AMD.
//!      (AMD dominates at small scales where ND overhead would
//!      exceed its quality benefit; METIS 5.2.0 switches at 200.)
//!    - Otherwise, run a multilevel node bisection, number the
//!      separator last, and recurse on the two sides.
//!
//! The numbering convention is "permutation" = new-position -> old-id.
//! Internally we populate `iperm[original_vertex] = new_position` and
//! invert at the end.

use crate::coarsen::{coarsen, CoarsenCounters};
use crate::fm_refine::refine_bisection;
use crate::graph::Graph;
use crate::initial_partition::{initial_bisect_bfs, initial_bisect_ggp, PART_A, PART_B};
use crate::rng::SplitMix;
use crate::sep_refine::{balance_node_separator, refine_node_separator};
use crate::separator::construct_separator;
use crate::{MetisOptions, MetisStats};
use rslab_ordering_core::{Counter, CscPattern, OrderingError, Progress};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};

struct IpermWriter<'a> {
    pos: Vec<AtomicI32>,
    placed: Counter<'a>,
}

impl<'a> IpermWriter<'a> {
    fn new(n: usize, progress: Option<&'a Progress>) -> Self {
        Self {
            pos: (0..n).map(|_| AtomicI32::new(-1)).collect(),
            placed: Counter::new(progress, n),
        }
    }

    // https://github.com/milanofthe/rslab/issues/291
    #[inline]
    fn set(&self, orig: usize, pos: i32) {
        debug_assert_eq!(self.pos[orig].load(Ordering::Relaxed), -1);
        self.pos[orig].store(pos, Ordering::Relaxed);
    }

    fn into_vec(self) -> Vec<i32> {
        self.placed.finish();
        self.pos.into_iter().map(|a| a.into_inner()).collect()
    }
}

#[derive(Default)]
struct AtomicStats {
    n_levels: AtomicU32,
    n_separator_vertices: AtomicU32,
    n_fm_passes: AtomicU32,
    n_two_hop_fallbacks: AtomicU32,
    n_amd_leaf_calls: AtomicU32,
}

impl AtomicStats {
    fn add(&self, s: &MetisStats) {
        self.n_levels.fetch_add(s.n_levels, Ordering::Relaxed);
        self.n_separator_vertices
            .fetch_add(s.n_separator_vertices, Ordering::Relaxed);
        self.n_fm_passes.fetch_add(s.n_fm_passes, Ordering::Relaxed);
        self.n_two_hop_fallbacks
            .fetch_add(s.n_two_hop_fallbacks, Ordering::Relaxed);
        self.n_amd_leaf_calls
            .fetch_add(s.n_amd_leaf_calls, Ordering::Relaxed);
    }
}

// https://github.com/milanofthe/rslab/issues/291
fn child_seed(seed: u64, which: u64) -> u64 {
    SplitMix::new(seed ^ which.wrapping_mul(crate::rng::GOLDEN_GAMMA)).next_u64()
}

pub(crate) fn nd_order(
    pattern: &CscPattern<'_>,
    vwgt: Option<&[i32]>,
    opts: &MetisOptions,
    stats: &mut MetisStats,
) -> Result<Vec<i32>, OrderingError> {
    let mut graph = Graph::from_csc_pattern(pattern)?;
    if let Some(w) = vwgt {
        if w.len() != graph.nvtxs as usize || w.iter().any(|&x| x < 1) {
            return Err(OrderingError::MalformedInput);
        }
        graph.vwgt = w.to_vec();
    }
    let n = graph.nvtxs as usize;
    let writer = IpermWriter::new(n, opts.progress.as_ref());
    let acc = AtomicStats::default();

    let (cc_label, ncc) = connected_components(&graph);
    stats.n_components = ncc as u32;
    let mut work: Vec<(Graph, Vec<i32>, usize, u64)> = Vec::new();
    if ncc == 1 {
        let vtx_map = (0..graph.nvtxs).collect();
        work.push((graph, vtx_map, 0, child_seed(opts.seed, 0)));
    } else {
        let mut offset: usize = 0;
        for c in 0..ncc {
            let (sub, vtx_map) = extract_by_label(&graph, &cc_label, c as i32);
            let count = sub.nvtxs as usize;
            if count > 0 {
                work.push((sub, vtx_map, offset, child_seed(opts.seed, c as u64)));
            }
            offset += count;
        }
        drop(graph);
    }
    let results: Vec<Result<(), OrderingError>> = {
        use rayon::prelude::*;
        work.into_par_iter()
            .map(|(sub, vtx_map, offset, seed)| {
                nd_subproblem(sub, vtx_map, offset, seed, opts, &writer, &acc)
            })
            .collect()
    };
    for r in results {
        r?;
    }
    stats.n_levels += acc.n_levels.load(Ordering::Relaxed);
    stats.n_separator_vertices += acc.n_separator_vertices.load(Ordering::Relaxed);
    stats.n_fm_passes += acc.n_fm_passes.load(Ordering::Relaxed);
    stats.n_two_hop_fallbacks += acc.n_two_hop_fallbacks.load(Ordering::Relaxed);
    stats.n_amd_leaf_calls += acc.n_amd_leaf_calls.load(Ordering::Relaxed);
    invert_iperm(&writer.into_vec(), n)
}

fn nd_subproblem(
    subgraph: Graph,
    vtx_map: Vec<i32>,
    offset: usize,
    seed: u64,
    opts: &MetisOptions,
    writer: &IpermWriter,
    acc: &AtomicStats,
) -> Result<(), OrderingError> {
    stopped(opts)?;
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        writer.set(vtx_map[0] as usize, offset as i32);
        writer.placed.add(1);
        return Ok(());
    }
    let mut local = MetisStats::default();

    let (cc_label, ncc) = connected_components(&subgraph);
    if ncc > 1 {
        let mut parts = Vec::with_capacity(ncc);
        let mut off = offset;
        for c in 0..ncc {
            let (sub, map) = extract_by_label(&subgraph, &cc_label, c as i32);
            let map_to_orig: Vec<i32> = map.iter().map(|&local| vtx_map[local as usize]).collect();
            let count = sub.nvtxs as usize;
            if count > 0 {
                let seed = child_seed(seed, crate::constants::CHILD_SEED_OFFSET + c as u64);
                parts.push((sub, map_to_orig, off, seed));
            }
            off += count;
        }
        drop(subgraph);
        drop(vtx_map);
        for (sub, map, off, seed) in parts {
            nd_subproblem(sub, map, off, seed, opts, writer, acc)?;
        }
        return Ok(());
    }

    if n <= opts.nd_to_amd_switch as usize {
        amd_leaf(&subgraph, &vtx_map, offset, writer, &mut local)?;
        acc.add(&local);
        return Ok(());
    }

    let mut rng = SplitMix::new(seed);
    let labels = multilevel_node_bisection(&subgraph, opts, &mut rng, &mut local)?;
    let mut a_verts: Vec<i32> = Vec::new();
    let mut b_verts: Vec<i32> = Vec::new();
    let mut s_verts: Vec<i32> = Vec::new();
    for (v, &l) in labels.iter().enumerate() {
        match l {
            PART_A => a_verts.push(v as i32),
            PART_B => b_verts.push(v as i32),
            _ => s_verts.push(v as i32),
        }
    }

    let big = a_verts.len().max(b_verts.len());
    if a_verts.is_empty()
        || b_verts.is_empty()
        || big as f64 >= crate::constants::UNBALANCED_SPLIT_FRACTION * n as f64
    {
        amd_leaf(&subgraph, &vtx_map, offset, writer, &mut local)?;
        acc.add(&local);
        return Ok(());
    }

    local.n_separator_vertices += s_verts.len() as u32;
    let na = a_verts.len();
    let nb = b_verts.len();
    for (i, &v) in s_verts.iter().enumerate() {
        writer.set(vtx_map[v as usize] as usize, (offset + na + nb + i) as i32);
    }
    writer.placed.add(s_verts.len());

    let (sub_a, map_a_local) = extract_by_list(&subgraph, &a_verts);
    let map_a: Vec<i32> = map_a_local
        .iter()
        .map(|&local| vtx_map[local as usize])
        .collect();
    let (sub_b, map_b_local) = extract_by_list(&subgraph, &b_verts);
    let map_b: Vec<i32> = map_b_local
        .iter()
        .map(|&local| vtx_map[local as usize])
        .collect();
    drop(subgraph);
    drop(vtx_map);
    acc.add(&local);

    let (seed_a, seed_b) = (child_seed(seed, 1), child_seed(seed, 2));
    let min_vertices = opts
        .parallel_min_vertices
        .unwrap_or(crate::constants::PAR_MIN_VERTICES);
    if n >= min_vertices && rayon::current_num_threads() > 1 {
        let (ra, rb) = rayon::join(
            || nd_subproblem(sub_a, map_a, offset, seed_a, opts, writer, acc),
            || nd_subproblem(sub_b, map_b, offset + na, seed_b, opts, writer, acc),
        );
        ra?;
        rb
    } else {
        nd_subproblem(sub_a, map_a, offset, seed_a, opts, writer, acc)?;
        nd_subproblem(sub_b, map_b, offset + na, seed_b, opts, writer, acc)
    }
}

// https://github.com/milanofthe/rslab/issues/291
fn multilevel_node_bisection(
    graph: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Result<Vec<u8>, OrderingError> {
    let mut counters = CoarsenCounters::default();
    let mut levels = coarsen(graph, opts, rng, &mut counters)?;
    stats.n_two_hop_fallbacks += counters.n_two_hop_fallbacks;
    stats.n_levels += levels.len() as u32;

    let coarsest: &Graph = match levels.last() {
        Some(cg) => &cg.graph,
        None => graph,
    };
    let total: i64 = coarsest.vwgt.iter().map(|&w| w as i64).sum();
    let target = total / 2;

    let mut best_labels: Vec<u8> = vec![PART_A; coarsest.nvtxs as usize];
    let mut best_cut: i32 = i32::MAX;
    for trial in 0..opts.niparts {
        let mut trial_labels = if trial % 2 == 0 {
            initial_bisect_ggp(coarsest, rng, target)
        } else {
            initial_bisect_bfs(coarsest, rng, target)
        };
        let cut = refine_bisection(
            coarsest,
            &mut trial_labels,
            opts.max_imbalance,
            opts.fm_passes,
        );
        stats.n_fm_passes += opts.fm_passes;
        if cut < best_cut {
            best_cut = cut;
            best_labels = trial_labels;
        }
    }
    let mut labels = best_labels;

    construct_separator(coarsest, &mut labels);
    refine_node_separator(coarsest, &mut labels, opts, rng);
    stats.n_fm_passes += opts.fm_passes;

    while let Some(cg) = levels.pop() {
        stopped(opts)?;
        let prev_graph: &Graph = levels.last().map_or(graph, |l| &l.graph);
        let prev_n = prev_graph.nvtxs as usize;
        let mut proj: Vec<u8> = vec![PART_A; prev_n];
        for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
            let c = cg.cmap[v] as usize;
            *p = labels[c];
        }
        labels = proj;
        balance_node_separator(prev_graph, &mut labels, opts.max_imbalance, rng);
        refine_node_separator(prev_graph, &mut labels, opts, rng);
        stats.n_fm_passes += opts.fm_passes;
    }

    Ok(labels)
}

pub(crate) fn stopped(opts: &MetisOptions) -> Result<(), OrderingError> {
    match &opts.stop {
        Some(f) if f.load(Ordering::Relaxed) => Err(OrderingError::Stopped),
        _ => Ok(()),
    }
}

fn amd_leaf(
    subgraph: &Graph,
    vtx_map: &[i32],
    offset: usize,
    iperm: &IpermWriter,
    stats: &mut MetisStats,
) -> Result<(), OrderingError> {
    stats.n_amd_leaf_calls += 1;
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    let (col_ptr, row_idx) = graph_to_csc_pattern(subgraph);
    let pattern = CscPattern::new(n, &col_ptr, &row_idx).ok_or(OrderingError::MalformedInput)?;
    let perm_local = rslab_amd::amd_order(&pattern)?;
    debug_assert_eq!(perm_local.len(), n);
    for (new_pos, &local_id) in perm_local.iter().enumerate() {
        let orig = vtx_map[local_id as usize];
        iperm.set(orig as usize, (offset + new_pos) as i32);
    }
    iperm.placed.add(n);
    Ok(())
}

fn graph_to_csc_pattern(graph: &Graph) -> (Vec<i32>, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<i32> = Vec::with_capacity(graph.adjncy.len() + n);
    col_ptr.push(0);
    for v in 0..n {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut diag_inserted = false;
        for &u in &graph.adjncy[lo..hi] {
            if !diag_inserted && u as usize > v {
                row_idx.push(v as i32);
                diag_inserted = true;
            }
            row_idx.push(u);
        }
        if !diag_inserted {
            row_idx.push(v as i32);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}

fn invert_iperm(iperm: &[i32], n: usize) -> Result<Vec<i32>, OrderingError> {
    let mut perm: Vec<i32> = vec![-1; n];
    for (old, &new_pos) in iperm.iter().enumerate() {
        if new_pos < 0 || (new_pos as usize) >= n {
            return Err(OrderingError::Internal(
                "metis nd produced invalid permutation",
            ));
        }
        let np = new_pos as usize;
        if perm[np] >= 0 {
            return Err(OrderingError::Internal(
                "metis nd produced duplicate position",
            ));
        }
        perm[np] = old as i32;
    }
    Ok(perm)
}

fn connected_components(graph: &Graph) -> (Vec<i32>, usize) {
    let n = graph.nvtxs as usize;
    let mut cc: Vec<i32> = vec![-1; n];
    let mut ncc: i32 = 0;
    let mut queue: Vec<i32> = Vec::new();
    for start in 0..n {
        if cc[start] >= 0 {
            continue;
        }
        cc[start] = ncc;
        queue.clear();
        queue.push(start as i32);
        while let Some(v) = queue.pop() {
            let vu = v as usize;
            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for &u in &graph.adjncy[lo..hi] {
                if cc[u as usize] < 0 {
                    cc[u as usize] = ncc;
                    queue.push(u);
                }
            }
        }
        ncc += 1;
    }
    (cc, ncc as usize)
}

fn extract_by_label(graph: &Graph, label: &[i32], c: i32) -> (Graph, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut vtx_map: Vec<i32> = Vec::new();
    let mut local_id: Vec<i32> = vec![-1; n];
    for v in 0..n {
        if label[v] == c {
            local_id[v] = vtx_map.len() as i32;
            vtx_map.push(v as i32);
        }
    }
    (build_induced(graph, &vtx_map, &local_id), vtx_map)
}

fn extract_by_list(graph: &Graph, verts: &[i32]) -> (Graph, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut local_id: Vec<i32> = vec![-1; n];
    for (i, &v) in verts.iter().enumerate() {
        local_id[v as usize] = i as i32;
    }
    let vtx_map = verts.to_vec();
    (build_induced(graph, &vtx_map, &local_id), vtx_map)
}

fn build_induced(graph: &Graph, vtx_map: &[i32], local_id: &[i32]) -> Graph {
    let sub_n = vtx_map.len();
    let bound: usize = vtx_map.iter().map(|&v| graph.degree(v) as usize).sum();
    let unit = graph.adjwgt.is_empty();
    let mut xadj: Vec<i32> = Vec::with_capacity(sub_n + 1);
    let mut adjncy: Vec<i32> = Vec::with_capacity(bound);
    let mut adjwgt: Vec<i32> = Vec::with_capacity(if unit { 0 } else { bound });
    let mut vwgt: Vec<i32> = Vec::with_capacity(sub_n);
    xadj.push(0);
    for &orig in vtx_map {
        let v = orig as usize;
        vwgt.push(graph.vwgt[v]);
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let lu = local_id[u];
            if lu >= 0 {
                adjncy.push(lu);
                if !unit {
                    adjwgt.push(graph.adjwgt[k]);
                }
            }
        }
        xadj.push(adjncy.len() as i32);
    }
    adjncy.shrink_to_fit();
    adjwgt.shrink_to_fit();
    Graph {
        nvtxs: sub_n as i32,
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
    use rslab_ordering_core::CscPattern;

    fn grid_triples(m: usize, n: usize) -> Vec<(usize, usize)> {
        let idx = |r: usize, c: usize| r * n + c;
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
        t
    }

    fn assert_permutation(perm: &[i32]) {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in perm {
            let p = p as usize;
            assert!(p < n, "index {} out of bounds", p);
            assert!(!seen[p], "duplicate {}", p);
            seen[p] = true;
        }
    }

    #[test]
    fn invert_iperm_rejects_duplicate_positions() {
        assert_eq!(invert_iperm(&[2, 0, 1], 3).unwrap(), vec![1, 2, 0]);
        assert!(matches!(
            invert_iperm(&[0, 0, 2], 3),
            Err(OrderingError::Internal(_))
        ));
        assert!(matches!(
            invert_iperm(&[3, 0, 1], 3),
            Err(OrderingError::Internal(_))
        ));
    }

    #[test]
    fn cc_disconnected_blocks() {
        let mut t = grid_triples(3, 3);
        for &(i, j) in grid_triples(3, 3).iter() {
            t.push((i + 9, j + 9));
        }
        let (cp, ri) = csc_from_triples(18, &t);
        let pat = CscPattern::new(18, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let (_, ncc) = connected_components(&g);
        assert_eq!(ncc, 2);
    }

    #[test]
    fn nd_order_small_grid_is_permutation() {
        let t = grid_triples(10, 10);
        let (cp, ri) = csc_from_triples(100, &t);
        let pat = CscPattern::new(100, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, None, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 100);
        assert_permutation(&perm);
        assert!(stats.n_amd_leaf_calls >= 1);
    }

    #[test]
    fn nd_order_large_grid_uses_multilevel() {
        let t = grid_triples(20, 20);
        let (cp, ri) = csc_from_triples(400, &t);
        let pat = CscPattern::new(400, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, None, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 400);
        assert_permutation(&perm);
        assert!(
            stats.n_separator_vertices > 0,
            "expected a top-level separator"
        );
    }

    #[test]
    fn nd_order_deterministic() {
        let t = grid_triples(12, 12);
        let n = 144;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut s1 = MetisStats::default();
        let mut s2 = MetisStats::default();
        let p1 = nd_order(&pat, None, &opts, &mut s1).unwrap();
        let p2 = nd_order(&pat, None, &opts, &mut s2).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn nd_order_stops_on_the_flag_and_only_then() {
        let t = grid_triples(40, 40);
        let n = 1600;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let opts = MetisOptions {
            stop: Some(flag.clone()),
            ..MetisOptions::default()
        };
        let free = nd_order(
            &pat,
            None,
            &MetisOptions::default(),
            &mut MetisStats::default(),
        );
        let armed = nd_order(&pat, None, &opts, &mut MetisStats::default());
        assert_eq!(free.unwrap(), armed.unwrap());
        flag.store(true, Ordering::Relaxed);
        let stopped = nd_order(&pat, None, &opts, &mut MetisStats::default());
        assert_eq!(stopped, Err(OrderingError::Stopped));
    }

    #[test]
    fn nd_order_handles_disconnected_graph() {
        let mut t = grid_triples(6, 6);
        for &(i, j) in grid_triples(6, 6).iter() {
            t.push((i + 36, j + 36));
        }
        let (cp, ri) = csc_from_triples(72, &t);
        let pat = CscPattern::new(72, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, None, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 72);
        assert_permutation(&perm);
        assert_eq!(stats.n_components, 2);
    }

    #[test]
    #[ignore]
    fn grid3d_separator_probe() {
        let m = 40usize;
        let idx = |x: usize, y: usize, z: usize| (z * m + y) * m + x;
        let mut t = Vec::new();
        for z in 0..m {
            for y in 0..m {
                for x in 0..m {
                    let k = idx(x, y, z);
                    t.push((k, k));
                    if x + 1 < m {
                        t.push((k, idx(x + 1, y, z)));
                    }
                    if y + 1 < m {
                        t.push((k, idx(x, y + 1, z)));
                    }
                    if z + 1 < m {
                        t.push((k, idx(x, y, z + 1)));
                    }
                }
            }
        }
        let n = m * m * m;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let graph = Graph::from_csc_pattern(&pat).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let mut rng = crate::rng::SplitMix::new(opts.seed);
        let labels = multilevel_node_bisection(&graph, &opts, &mut rng, &mut stats).unwrap();
        let mut na = 0usize;
        let mut nb = 0usize;
        let mut ns = 0usize;
        for &l in &labels {
            match l {
                PART_A => na += 1,
                PART_B => nb += 1,
                _ => ns += 1,
            }
        }
        println!(
            "40^3 top-level bisection: |A|={} |B|={} |S|={} (ideal S=1600), levels={}",
            na, nb, ns, stats.n_levels
        );

        let mut work: Vec<(Graph, usize)> = vec![(graph, 0)];
        let mut per_depth: Vec<(usize, f64, f64)> = Vec::new();
        let mut fill_proxy: f64 = 0.0;
        while let Some((g, depth)) = work.pop() {
            let gn = g.nvtxs as usize;
            if gn <= opts.nd_to_amd_switch as usize {
                continue;
            }
            let mut st = MetisStats::default();
            let labels = multilevel_node_bisection(&g, &opts, &mut rng, &mut st).unwrap();
            let mut a_verts: Vec<i32> = Vec::new();
            let mut b_verts: Vec<i32> = Vec::new();
            let mut ns = 0usize;
            for (v, &l) in labels.iter().enumerate() {
                match l {
                    PART_A => a_verts.push(v as i32),
                    PART_B => b_verts.push(v as i32),
                    _ => ns += 1,
                }
            }
            let big = a_verts.len().max(b_verts.len());
            if a_verts.is_empty()
                || b_verts.is_empty()
                || big as f64 >= crate::constants::UNBALANCED_SPLIT_FRACTION * gn as f64
            {
                continue;
            }
            let ratio = ns as f64 / (gn as f64).powf(2.0 / 3.0);
            fill_proxy += 0.5 * (ns * (ns + 1)) as f64;
            if per_depth.len() <= depth {
                per_depth.resize(depth + 1, (0, 0.0, 0.0));
            }
            per_depth[depth].0 += 1;
            per_depth[depth].1 += ratio;
            per_depth[depth].2 += ns as f64;
            if depth <= 4 {
                println!(
                    "  d={} n={} sep={} bal={:.2} sep/n^(2/3)={:.2}",
                    depth,
                    gn,
                    ns,
                    a_verts.len().min(b_verts.len()) as f64 / big as f64,
                    ratio
                );
            }
            let (sub_a, _) = extract_by_list(&g, &a_verts);
            let (sub_b, _) = extract_by_list(&g, &b_verts);
            work.push((sub_a, depth + 1));
            work.push((sub_b, depth + 1));
        }
        for (d, &(cnt, sr, ssep)) in per_depth.iter().enumerate() {
            println!(
                "depth {}: bisections={} mean sep/n^(2/3)={:.2} total sep={}",
                d,
                cnt,
                sr / cnt.max(1) as f64,
                ssep as usize
            );
        }
        println!("sep-clique fill proxy: {:.3e}", fill_proxy);

        let mut stats2 = MetisStats::default();
        let perm = nd_order(&pat, None, &opts, &mut stats2).unwrap();
        assert_eq!(perm.len(), n);
        println!(
            "full nd: sep_total={} amd_leaves={} levels={} two_hop={}",
            stats2.n_separator_vertices,
            stats2.n_amd_leaf_calls,
            stats2.n_levels,
            stats2.n_two_hop_fallbacks
        );
    }

    #[test]
    fn extract_induced_preserves_edges() {
        let t = grid_triples(3, 3);
        let (cp, ri) = csc_from_triples(9, &t);
        let pat = CscPattern::new(9, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let (sub, map) = extract_by_list(&g, &[0, 1, 2]);
        assert_eq!(sub.nvtxs, 3);
        assert_eq!(map, vec![0, 1, 2]);
        assert_eq!(sub.adjncy.len(), 4);
    }
}
