//! Block triangular form (BTF) for unsymmetric matrices.
//!
//! Two purely structural, sequential, deterministic passes used by the KLU
//! path ([`crate::numeric::klu`]):
//!
//! 1. **Maximum transversal** ([`max_transversal`]): an augmenting-path
//!    (Hopcroft-Karp) matching that pairs every column with a distinct row
//!    holding a structural nonzero, giving the permuted matrix a zero-free
//!    diagonal. An incomplete matching proves the matrix structurally
//!    singular (its structural rank is the matching size).
//! 2. **Tarjan SCC** ([`block_triangular_form`]): strongly connected
//!    components of the matched matrix's directed graph, emitted in reverse
//!    topological order, yield the symmetric permutation to **block upper
//!    triangular** form: all entries outside the diagonal blocks lie above
//!    them, and each diagonal block is irreducible (strongly connected).
//!
//! Both passes are iterative (explicit stacks, no recursion, safe on
//! arbitrarily deep graphs) and allocation-linear in `n + nnz`.

#[derive(Debug, Clone)]
pub(crate) struct BtfForm {
    /// Row permutation, new-to-old: permuted row `k` is original row
    /// `row_perm[k]`.
    pub row_perm: Vec<usize>,
    /// Column permutation, new-to-old.
    pub col_perm: Vec<usize>,
    /// Diagonal-block boundaries in the permuted matrix: block `b` spans
    /// rows/columns `block_ptr[b]..block_ptr[b + 1]`. `block_ptr.len()` is
    /// `n_blocks + 1`, first entry `0`, last entry `n`.
    pub block_ptr: Vec<usize>,
}

#[cfg(test)]
pub(crate) fn max_transversal(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> (Vec<usize>, usize) {
    hk_transversal(n, col_ptr, row_idx, true)
}

// https://github.com/milanofthe/rslab/issues/291
pub(crate) fn hk_transversal(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    diag_seed: bool,
) -> (Vec<usize>, usize) {
    const UNMATCHED: usize = usize::MAX;
    let mut row_match = vec![UNMATCHED; n];
    let mut col_of_row = vec![UNMATCHED; n];
    let mut n_matched = 0usize;

    // https://github.com/milanofthe/rslab/issues/291
    if diag_seed {
        for j in 0..n {
            for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if r == j {
                    col_of_row[j] = j;
                    row_match[j] = j;
                    n_matched += 1;
                    break;
                }
            }
        }
    }
    for j in 0..n {
        if row_match[j] != UNMATCHED {
            continue;
        }
        for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            if col_of_row[r] == UNMATCHED {
                col_of_row[r] = j;
                row_match[j] = r;
                n_matched += 1;
                break;
            }
        }
    }
    if n_matched == n {
        return (row_match, n_matched);
    }

    const UNSET: usize = usize::MAX;
    let mut level = vec![UNSET; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    let mut phase_cursor: Vec<usize> = vec![0; n];
    let mut col_stack = vec![0usize; n];
    let mut path_row = vec![0usize; n];

    loop {
        level.iter_mut().for_each(|v| *v = UNSET);
        queue.clear();
        for c in 0..n {
            if row_match[c] == UNMATCHED {
                level[c] = 0;
                queue.push(c);
            }
        }
        let mut shortest = UNSET;
        let mut head = 0usize;
        while head < queue.len() {
            let c = queue[head];
            head += 1;
            if level[c] >= shortest {
                break;
            }
            for &r in &row_idx[col_ptr[c]..col_ptr[c + 1]] {
                let c2 = col_of_row[r];
                if c2 == UNMATCHED {
                    if shortest == UNSET {
                        shortest = level[c];
                    }
                } else if level[c2] == UNSET {
                    level[c2] = level[c] + 1;
                    queue.push(c2);
                }
            }
        }
        if shortest == UNSET {
            break;
        }

        phase_cursor[..n].copy_from_slice(&col_ptr[..n]);
        for jstart in 0..n {
            if row_match[jstart] != UNMATCHED || level[jstart] != 0 {
                continue;
            }
            let mut depth = 0usize;
            col_stack[0] = jstart;
            'dfs: loop {
                let c = col_stack[depth];
                let mut descended = false;
                while phase_cursor[c] < col_ptr[c + 1] {
                    let r = row_idx[phase_cursor[c]];
                    phase_cursor[c] += 1;
                    let c2 = col_of_row[r];
                    if c2 == UNMATCHED {
                        path_row[depth] = r;
                        for k in (0..=depth).rev() {
                            let ck = col_stack[k];
                            let rk = path_row[k];
                            col_of_row[rk] = ck;
                            row_match[ck] = rk;
                            level[ck] = UNSET;
                        }
                        n_matched += 1;
                        break 'dfs;
                    }
                    if level[c2] == level[c] + 1 {
                        path_row[depth] = r;
                        depth += 1;
                        col_stack[depth] = c2;
                        descended = true;
                        break;
                    }
                }
                if descended {
                    continue;
                }
                level[c] = UNSET;
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
        }
    }
    (row_match, n_matched)
}

pub(crate) fn mc21_transversal(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    budget: usize,
) -> Option<(Vec<usize>, usize)> {
    const UNMATCHED: usize = usize::MAX;
    let mut row_match = vec![UNMATCHED; n];
    let mut col_of_row = vec![UNMATCHED; n];
    let mut n_matched = 0usize;
    for j in 0..n {
        for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            if col_of_row[r] == UNMATCHED {
                col_of_row[r] = j;
                row_match[j] = r;
                n_matched += 1;
                break;
            }
        }
    }
    if n_matched < n {
        n_matched = mc21_augment(
            n,
            col_ptr,
            row_idx,
            &mut row_match,
            &mut col_of_row,
            n_matched,
            budget,
        )?;
    }
    Some((row_match, n_matched))
}

#[allow(clippy::too_many_arguments)]
fn mc21_augment(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    row_match: &mut [usize],
    col_of_row: &mut [usize],
    mut n_matched: usize,
    budget: usize,
) -> Option<usize> {
    const UNMATCHED: usize = usize::MAX;
    let mut work = 0usize;
    let mut visited = vec![UNMATCHED; n];
    let mut cheap: Vec<usize> = col_ptr[..n].to_vec();
    let mut col_stack = vec![0usize; n];
    let mut cursor = vec![0usize; n];
    let mut path_row = vec![0usize; n];

    for jstart in 0..n {
        if row_match[jstart] != UNMATCHED {
            continue;
        }
        let stamp = jstart;
        let mut depth = 0usize;
        col_stack[0] = jstart;
        cursor[0] = col_ptr[jstart];
        visited[jstart] = stamp;
        let mut augment_depth = None;

        'dfs: while augment_depth.is_none() {
            let c = col_stack[depth];
            while cheap[c] < col_ptr[c + 1] {
                let r = row_idx[cheap[c]];
                cheap[c] += 1;
                work += 1;
                if col_of_row[r] == UNMATCHED {
                    path_row[depth] = r;
                    augment_depth = Some(depth);
                    continue 'dfs;
                }
            }
            let mut advanced = false;
            while cursor[depth] < col_ptr[c + 1] {
                let r = row_idx[cursor[depth]];
                cursor[depth] += 1;
                work += 1;
                let c2 = col_of_row[r];
                if visited[c2] == stamp {
                    continue;
                }
                visited[c2] = stamp;
                path_row[depth] = r;
                depth += 1;
                col_stack[depth] = c2;
                cursor[depth] = col_ptr[c2];
                advanced = true;
                break;
            }
            if advanced {
                continue;
            }
            if depth == 0 {
                break;
            }
            depth -= 1;
        }
        if work > budget {
            return None;
        }

        if let Some(d) = augment_depth {
            for k in (0..=d).rev() {
                let c = col_stack[k];
                let r = path_row[k];
                col_of_row[r] = c;
                row_match[c] = r;
            }
            n_matched += 1;
        }
    }
    Some(n_matched)
}

pub(crate) fn matching_candidates(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Option<Vec<Vec<usize>>> {
    let (m1, k1) = hk_transversal(n, col_ptr, row_idx, true);
    if k1 != n {
        return None;
    }
    // https://github.com/milanofthe/rslab/issues/291
    if m1.iter().enumerate().all(|(j, &r)| r == j) {
        return Some(vec![m1]);
    }
    let mut out = vec![m1];
    let (m2, k2) = hk_transversal(n, col_ptr, row_idx, false);
    if k2 == n && !out.contains(&m2) {
        out.push(m2);
    }
    let nnz = col_ptr[n];
    let step_budget =
        (crate::constants::MC21_STEPS_PER_ENTRY * (nnz + n)).min(crate::constants::MC21_MAX_STEPS);
    if let Some((m3, k3)) = mc21_transversal(n, col_ptr, row_idx, step_budget) {
        if k3 == n && !out.contains(&m3) {
            out.push(m3);
        }
    }
    Some(out)
}

#[cfg(test)]
pub(crate) fn block_triangular_form(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Option<BtfForm> {
    if n == 0 {
        return Some(BtfForm {
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            block_ptr: vec![0],
        });
    }
    let (row_match, n_matched) = max_transversal(n, col_ptr, row_idx);
    if n_matched != n {
        return None;
    }
    Some(btf_from_matching(n, col_ptr, row_idx, row_match))
}

// https://github.com/milanofthe/rslab/issues/291
pub(crate) fn btf_from_matching(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    row_match: Vec<usize>,
) -> BtfForm {
    if n == 0 {
        return BtfForm {
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            block_ptr: vec![0],
        };
    }
    debug_assert!(n < u32::MAX as usize);
    debug_assert!(row_match.len() == n && row_match.iter().all(|&r| r < n));
    let mut col_of_row = vec![0u32; n];
    for (j, &r) in row_match.iter().enumerate() {
        col_of_row[r] = j as u32;
    }

    const UNSET32: u32 = u32::MAX;
    let mut discovery = vec![UNSET32; n];
    let mut lowlink = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut scc_stack: Vec<u32> = Vec::with_capacity(n);
    let mut frame_node = vec![0u32; n];
    let mut frame_cursor = vec![0usize; n];
    let mut next_index = 0u32;
    let mut col_perm: Vec<usize> = Vec::with_capacity(n);
    let mut block_ptr: Vec<usize> = vec![0];

    for root in 0..n {
        if discovery[root] != UNSET32 {
            continue;
        }
        let mut top = 0usize;
        frame_node[0] = root as u32;
        frame_cursor[0] = col_ptr[root];
        discovery[root] = next_index;
        lowlink[root] = next_index;
        next_index += 1;
        scc_stack.push(root as u32);
        on_stack[root] = true;

        loop {
            let v = frame_node[top] as usize;
            let mut descended = false;
            while frame_cursor[top] < col_ptr[v + 1] {
                let w = col_of_row[row_idx[frame_cursor[top]]] as usize;
                frame_cursor[top] += 1;
                if discovery[w] == UNSET32 {
                    discovery[w] = next_index;
                    lowlink[w] = next_index;
                    next_index += 1;
                    scc_stack.push(w as u32);
                    on_stack[w] = true;
                    top += 1;
                    frame_node[top] = w as u32;
                    frame_cursor[top] = col_ptr[w];
                    descended = true;
                    break;
                } else if on_stack[w] && discovery[w] < lowlink[v] {
                    lowlink[v] = discovery[w];
                }
            }
            if descended {
                continue;
            }
            if lowlink[v] == discovery[v] {
                let block_start = col_perm.len();
                while let Some(w) = scc_stack.pop_if(|w| discovery[*w as usize] >= discovery[v]) {
                    let w = w as usize;
                    on_stack[w] = false;
                    col_perm.push(w);
                }
                col_perm[block_start..].sort_unstable_by_key(|&w| discovery[w]);
                block_ptr.push(col_perm.len());
            }
            if top == 0 {
                break;
            }
            top -= 1;
            let parent = frame_node[top] as usize;
            if lowlink[v] < lowlink[parent] {
                lowlink[parent] = lowlink[v];
            }
        }
    }
    debug_assert_eq!(col_perm.len(), n);

    let row_perm: Vec<usize> = col_perm.iter().map(|&j| row_match[j]).collect();
    BtfForm {
        row_perm,
        col_perm,
        block_ptr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(n: usize, entries: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c) in entries {
            cols[c].push(r);
        }
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        for c in cols {
            let mut c = c;
            c.sort_unstable();
            row_idx.extend_from_slice(&c);
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx)
    }

    #[test]
    fn transversal_identity_diagonal() {
        let (cp, ri) = pattern(3, &[(0, 0), (1, 1), (2, 2), (1, 0)]);
        let (m, k) = max_transversal(3, &cp, &ri);
        assert_eq!(k, 3);
        assert_eq!(m, vec![0, 1, 2]);
    }

    #[test]
    fn transversal_needs_augmenting_path() {
        let (cp, ri) = pattern(2, &[(0, 0), (1, 0), (0, 1)]);
        let (m, k) = max_transversal(2, &cp, &ri);
        assert_eq!(k, 2);
        assert_eq!(m, vec![1, 0]);
    }

    #[test]
    fn transversal_detects_structural_singularity() {
        let (cp, ri) = pattern(3, &[(0, 0), (1, 1), (2, 0)]);
        let (_, k) = max_transversal(3, &cp, &ri);
        assert_eq!(k, 2);
    }

    fn assert_block_upper(n: usize, col_ptr: &[usize], row_idx: &[usize], f: &BtfForm) {
        let mut inv_row = vec![0usize; n];
        for (k, &r) in f.row_perm.iter().enumerate() {
            inv_row[r] = k;
        }
        let mut block_of = vec![0usize; n];
        for b in 0..f.block_ptr.len() - 1 {
            for k in f.block_ptr[b]..f.block_ptr[b + 1] {
                block_of[k] = b;
            }
        }
        for (kc, &jc) in f.col_perm.iter().enumerate() {
            for &r in &row_idx[col_ptr[jc]..col_ptr[jc + 1]] {
                let kr = inv_row[r];
                assert!(
                    block_of[kr] <= block_of[kc],
                    "entry ({kr}, {kc}) below its diagonal block"
                );
            }
        }
    }

    #[test]
    fn btf_lower_triangular_becomes_upper() {
        let n = 5;
        let mut e = Vec::new();
        for i in 0..n {
            e.push((i, i));
            if i + 1 < n {
                e.push((i + 1, i));
            }
        }
        let (cp, ri) = pattern(n, &e);
        let f = block_triangular_form(n, &cp, &ri).unwrap();
        assert_eq!(f.block_ptr.len() - 1, n, "all blocks must be 1x1");
        assert_block_upper(n, &cp, &ri, &f);
    }

    #[test]
    fn btf_finds_irreducible_blocks() {
        let entries = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (1, 0),
            (2, 3),
            (3, 2),
            (0, 2),
        ];
        let (cp, ri) = pattern(4, &entries);
        let f = block_triangular_form(4, &cp, &ri).unwrap();
        assert_eq!(f.block_ptr, vec![0, 2, 4]);
        assert_block_upper(4, &cp, &ri, &f);
        let first: Vec<usize> = f.col_perm[0..2].to_vec();
        assert!(first.contains(&0) && first.contains(&1));
    }

    #[test]
    fn btf_fully_coupled_is_one_block() {
        let n = 4;
        let mut e = Vec::new();
        for i in 0..n {
            e.push((i, i));
            e.push(((i + 1) % n, i));
        }
        let (cp, ri) = pattern(n, &e);
        let f = block_triangular_form(n, &cp, &ri).unwrap();
        assert_eq!(f.block_ptr, vec![0, n]);
    }

    #[test]
    fn btf_structurally_singular_is_none() {
        let (cp, ri) = pattern(3, &[(0, 0), (1, 1), (2, 0)]);
        assert!(block_triangular_form(3, &cp, &ri).is_none());
    }

    #[test]
    fn btf_empty_matrix() {
        let f = block_triangular_form(0, &[0], &[]).unwrap();
        assert_eq!(f.block_ptr, vec![0]);
        assert!(f.row_perm.is_empty());
    }
}
