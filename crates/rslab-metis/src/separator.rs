//! Convert an edge bisection to a node separator.
//!
//! Given `labels[v] in {PART_A, PART_B}`, we compute a vertex set
//! `S subset boundary(A) union boundary(B)` such that removing `S` disconnects
//! the remaining A-vertices from the remaining B-vertices. The
//! minimum-weight such set is the minimum vertex cover of the
//! bipartite graph between A-boundary and B-boundary induced by
//! crossing edges (Konig's theorem).
//!
//! Algorithm:
//!
//! 1. Identify the A-boundary and B-boundary (vertices on each side
//!    incident to a crossing edge).
//! 2. Build the bipartite crossing-edge graph.
//! 3. Compute a maximum matching with Kuhn's augmenting-path
//!    algorithm (O(V*E); sufficient at our graph sizes).
//! 4. Use Konig's construction to extract the minimum vertex cover
//!    from the matching, and relabel those vertices as `PART_SEP`.
//!
//! References:
//! - Karypis & Kumar, *A Fast and High Quality Multilevel Scheme for
//!   Partitioning Irregular Graphs*, 1998 (section 5).
//! - METIS 5.2.0 `libmetis/separator.c::ConstructMinCoverSeparator`.

use crate::fm_refine::PART_SEP;
use crate::graph::Graph;
use crate::initial_partition::{PART_A, PART_B};

/// Convert an edge bisection (labels in {PART_A, PART_B}) to a node
/// separator (labels in {PART_A, PART_B, PART_SEP}) in place.
///
/// Returns the number of separator vertices.
pub fn construct_separator(graph: &Graph, labels: &mut [u8]) -> usize {
    let n = graph.nvtxs as usize;
    debug_assert_eq!(labels.len(), n);

    let mut is_bnd_a: Vec<bool> = vec![false; n];
    let mut is_bnd_b: Vec<bool> = vec![false; n];
    for v in 0..n {
        let lv = labels[v];
        if lv != PART_A && lv != PART_B {
            continue;
        }
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for &u in &graph.adjncy[lo..hi] {
            let u = u as usize;
            if labels[u] != lv && (labels[u] == PART_A || labels[u] == PART_B) {
                if lv == PART_A {
                    is_bnd_a[v] = true;
                } else {
                    is_bnd_b[v] = true;
                }
                break;
            }
        }
    }

    let mut a_list: Vec<i32> = Vec::new();
    let mut b_list: Vec<i32> = Vec::new();
    let mut a_idx: Vec<i32> = vec![-1; n];
    let mut b_idx: Vec<i32> = vec![-1; n];
    for v in 0..n {
        if is_bnd_a[v] {
            a_idx[v] = a_list.len() as i32;
            a_list.push(v as i32);
        }
        if is_bnd_b[v] {
            b_idx[v] = b_list.len() as i32;
            b_list.push(v as i32);
        }
    }
    if a_list.is_empty() || b_list.is_empty() {
        return 0;
    }

    let na = a_list.len();
    let nb = b_list.len();
    let mut adj: Vec<Vec<i32>> = vec![Vec::new(); na];
    for (ai, &v) in a_list.iter().enumerate() {
        let vu = v as usize;
        let lo = graph.xadj[vu] as usize;
        let hi = graph.xadj[vu + 1] as usize;
        for &u in &graph.adjncy[lo..hi] {
            let u = u as usize;
            let bi = b_idx[u];
            if bi >= 0 {
                adj[ai].push(bi);
            }
        }
    }

    let mut match_a: Vec<i32> = vec![-1; na];
    let mut match_b: Vec<i32> = vec![-1; nb];
    for ai in 0..na {
        let mut visited: Vec<bool> = vec![false; nb];
        try_augment(ai as i32, &adj, &mut match_a, &mut match_b, &mut visited);
    }

    let mut in_z_a: Vec<bool> = vec![false; na];
    let mut in_z_b: Vec<bool> = vec![false; nb];
    let mut stack: Vec<i32> = Vec::new();
    for ai in 0..na {
        if match_a[ai] < 0 {
            in_z_a[ai] = true;
            stack.push(ai as i32);
        }
    }
    while let Some(ai) = stack.pop() {
        for &bi in &adj[ai as usize] {
            if !in_z_b[bi as usize] {
                in_z_b[bi as usize] = true;
                let m = match_b[bi as usize];
                if m >= 0 && !in_z_a[m as usize] {
                    in_z_a[m as usize] = true;
                    stack.push(m);
                }
            }
        }
    }

    let mut sep_count: usize = 0;
    for (ai, &v) in a_list.iter().enumerate() {
        if !in_z_a[ai] {
            labels[v as usize] = PART_SEP;
            sep_count += 1;
        }
    }
    for (bi, &v) in b_list.iter().enumerate() {
        if in_z_b[bi] {
            labels[v as usize] = PART_SEP;
            sep_count += 1;
        }
    }
    sep_count
}

// https://github.com/milanofthe/rslab/issues/291
fn try_augment(
    start: i32,
    adj: &[Vec<i32>],
    match_a: &mut [i32],
    match_b: &mut [i32],
    visited: &mut [bool],
) -> bool {
    fn dfs(
        u: i32,
        adj: &[Vec<i32>],
        match_a: &mut [i32],
        match_b: &mut [i32],
        visited: &mut [bool],
    ) -> bool {
        for &v in &adj[u as usize] {
            if visited[v as usize] {
                continue;
            }
            visited[v as usize] = true;
            if match_b[v as usize] < 0 || dfs(match_b[v as usize], adj, match_a, match_b, visited) {
                match_a[u as usize] = v;
                match_b[v as usize] = u;
                return true;
            }
        }
        false
    }
    dfs(start, adj, match_a, match_b, visited)
}

/// Verify that labels form a valid node separator: no edge connects a
/// PART_A vertex directly to a PART_B vertex.
#[cfg(test)]
pub fn is_valid_separator(graph: &Graph, labels: &[u8]) -> bool {
    for v in 0..graph.nvtxs as usize {
        let lv = labels[v];
        if lv != PART_A && lv != PART_B {
            continue;
        }
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
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
    use crate::initial_partition::{cut_size, initial_bisect_ggp, part_weight};
    use crate::rng::SplitMix;
    use crate::tests::grid;

    #[test]
    fn separator_disconnects_parts_on_grid() {
        let g = grid(6, 6);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(21);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let sep_count = construct_separator(&g, &mut labels);
        assert!(sep_count > 0, "expected non-trivial separator");
        assert!(is_valid_separator(&g, &labels));
    }

    #[test]
    fn separator_empty_if_no_cut() {
        let g = grid(3, 3);
        let mut labels: Vec<u8> = vec![PART_A; 9];
        let sep_count = construct_separator(&g, &mut labels);
        assert_eq!(sep_count, 0);
        assert!(is_valid_separator(&g, &labels));
    }

    #[test]
    fn separator_on_4x4_grid_balanced() {
        let g = grid(4, 4);
        let mut labels: Vec<u8> = (0..16u8)
            .map(|k| if (k % 4) < 2 { PART_A } else { PART_B })
            .collect();
        let cut = cut_size(&g, &labels);
        assert!(cut > 0);
        let sep_count = construct_separator(&g, &mut labels);
        assert!(sep_count > 0);
        assert!(is_valid_separator(&g, &labels));
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn min_cover_no_larger_than_lighter_boundary() {
        let g = grid(8, 8);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(33);
        let labels_init = initial_bisect_ggp(&g, &mut rng, total / 2);
        let n = g.nvtxs as usize;
        let mut bnd_a = 0usize;
        let mut bnd_b = 0usize;
        for v in 0..n {
            let lv = labels_init[v];
            let lo = g.xadj[v] as usize;
            let hi = g.xadj[v + 1] as usize;
            let mut is_bnd = false;
            for k in lo..hi {
                if labels_init[g.adjncy[k] as usize] != lv {
                    is_bnd = true;
                    break;
                }
            }
            if is_bnd {
                if lv == PART_A {
                    bnd_a += 1;
                } else {
                    bnd_b += 1;
                }
            }
        }
        let mut labels = labels_init.clone();
        let sep_count = construct_separator(&g, &mut labels);
        assert!(sep_count <= bnd_a.min(bnd_b).max(1));
    }
}
