//! Hungarian algorithm for the minimum-cost perfect bipartite
//! matching problem.
//!
//! The core kernel for MC64-style scaling. Given a sparse
//! non-negative cost matrix in CSC form, finds a perfect matching
//! (row-to-column) minimizing total cost, and returns both the
//! matching permutation and the optimal dual variables from the
//! LP dual. Those duals are what get exponentiated into the
//! row/column scalings in `mc64.rs`.
//!
//! Reference: Duff & Koster (2001), section 4. Source model: SPRAL
//! `src/scaling.f90::hungarian_match`, itself a clean-room rewrite of
//! HSL_MC80. The algorithm is the standard shortest-augmenting-path
//! variant - each augmenting path is a Dijkstra-like search on the
//! reduced-cost graph, and the dual variables are updated to preserve
//! complementary slackness.
//!
//! Follows SPRAL's `hungarian_match` / `hungarian_init_heurisitic`
//! line-by-line, with Rust-idiomatic naming and a custom index-based
//! binary min-heap that supports decrease-key (mirroring SPRAL's
//! `q`/`d`/`l` arrays).
//! The custom heap is used instead of `std::collections::BinaryHeap`
//! because BinaryHeap lacks decrease-key, and lazy deletion would
//! require wrapping `f64` distances in a tie-breakable ordered type -
//! mirroring SPRAL's index-based heap produces cleaner code.

#[derive(Debug, Clone)]
pub(crate) struct CostGraph<'a> {
    pub n: usize,
    pub col_ptr: std::borrow::Cow<'a, [usize]>,
    pub row_idx: std::borrow::Cow<'a, [usize]>,
    pub cost: Vec<f64>,
}

#[derive(Debug, Clone)]
pub(crate) struct Matching {
    /// `perm[j]` is the row matched to column `j`. `usize::MAX`
    /// sentinel for unmatched columns (only populated in the
    /// partial-matching case).
    pub perm: Vec<usize>,
    /// Dual variable `u[i]` for row `i` (length `n`).
    pub u: Vec<f64>,
    /// Dual variable `v[j]` for column `j` (length `n`).
    pub v: Vec<f64>,
    /// Number of columns successfully matched. `n_matched == n`
    /// means a full perfect matching was found; a smaller value
    /// indicates structural singularity on the cost graph.
    pub n_matched: usize,
}

const NONE: usize = usize::MAX;

const RINF: f64 = f64::MAX / 2.0;

struct IndexHeap {
    heap: Vec<usize>,
    pos: Vec<usize>,
    len: usize,
}

impl IndexHeap {
    fn new(m: usize) -> Self {
        rsprobe::add!(DEBUG, "heap slots", m);
        IndexHeap {
            heap: vec![0; m + 1],
            pos: vec![0; m],
            len: 0,
        }
    }

    fn reset(&mut self, rows: &[usize]) {
        rsprobe::add!(DEBUG, "heap slots", rows.len());
        for &i in rows {
            self.pos[i] = 0;
        }
        self.len = 0;
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn peek(&self) -> usize {
        self.heap[1]
    }

    fn contains(&self, i: usize) -> bool {
        self.pos[i] != 0
    }

    fn update(&mut self, i: usize, d: &[f64]) {
        let mut p = self.pos[i];
        if p <= 1 {
            self.heap[p] = i;
            return;
        }
        let v = d[i];
        while p > 1 {
            let parent_pos = p / 2;
            let parent_idx = self.heap[parent_pos];
            if v >= d[parent_idx] {
                break;
            }
            self.heap[p] = parent_idx;
            self.pos[parent_idx] = p;
            p = parent_pos;
        }
        self.heap[p] = i;
        self.pos[i] = p;
    }

    fn insert(&mut self, i: usize, d: &[f64]) {
        debug_assert!(!self.contains(i));
        self.len += 1;
        self.pos[i] = self.len;
        self.update(i, d);
    }

    fn delete(&mut self, pos0: usize, d: &[f64]) {
        let removed = self.heap[pos0];
        self.pos[removed] = 0;
        if self.len == pos0 {
            self.len -= 1;
            return;
        }
        let idx = self.heap[self.len];
        let v = d[idx];
        self.len -= 1;
        let mut p = pos0;

        if p > 1 {
            loop {
                let parent = p / 2;
                let pk = self.heap[parent];
                if v >= d[pk] {
                    break;
                }
                self.heap[p] = pk;
                self.pos[pk] = p;
                p = parent;
                if p <= 1 {
                    break;
                }
            }
        }
        self.heap[p] = idx;
        self.pos[idx] = p;
        if p != pos0 {
            return;
        }

        loop {
            let mut child = 2 * p;
            if child > self.len {
                break;
            }
            let mut dk = d[self.heap[child]];
            if child < self.len {
                let dr = d[self.heap[child + 1]];
                if dk > dr {
                    child += 1;
                    dk = dr;
                }
            }
            if v <= dk {
                break;
            }
            let qk = self.heap[child];
            self.heap[p] = qk;
            self.pos[qk] = p;
            p = child;
        }
        self.heap[p] = idx;
        self.pos[idx] = p;
    }

    fn pop(&mut self, d: &[f64]) -> usize {
        let top = self.heap[1];
        self.delete(1, d);
        top
    }
}

fn hungarian_init_heuristic(
    cost: &CostGraph<'_>,
    iperm: &mut [usize],
    jperm: &mut [usize],
    u: &mut [f64],
) -> usize {
    let n = cost.n;
    let m = n;
    let mut num = 0usize;

    let mut row_min_edge: Vec<usize> = vec![NONE; m];
    let mut col_reduced_min: Vec<f64> = vec![0.0; n];
    let mut search_from: Vec<usize> = (0..n).map(|j| cost.col_ptr[j]).collect();

    for ui in u.iter_mut().take(m) {
        *ui = RINF;
    }
    for j in 0..n {
        for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
            let i = cost.row_idx[k];
            if cost.cost[k] <= u[i] {
                u[i] = cost.cost[k];
                iperm[i] = j;
                row_min_edge[i] = k;
            }
        }
    }

    for i in 0..m {
        let j = iperm[i];
        if j == NONE {
            continue;
        }
        iperm[i] = NONE;
        if jperm[j] != NONE {
            continue;
        }
        let col_len = cost.col_ptr[j + 1] - cost.col_ptr[j];
        if col_len > m / crate::constants::HUNGARIAN_DENSE_COL_DIV
            && m > crate::constants::HUNGARIAN_DENSE_MIN_ROWS
        {
            continue;
        }
        num += 1;
        iperm[i] = j;
        jperm[j] = row_min_edge[i];
    }
    if num == n {
        return num;
    }

    let mut length2_scans = 0u64;
    'improve_assign: for j in 0..n {
        if jperm[j] != NONE {
            continue;
        }
        if cost.col_ptr[j] >= cost.col_ptr[j + 1] {
            continue;
        }
        let start = cost.col_ptr[j];
        let end = cost.col_ptr[j + 1];
        let mut i0 = cost.row_idx[start];
        let mut vj = cost.cost[start] - u[i0];
        let mut k0 = start;
        for k in (start + 1)..end {
            let i = cost.row_idx[k];
            let di = cost.cost[k] - u[i];
            if di > vj {
                continue;
            }
            if di == vj && di != RINF {
                if iperm[i] != NONE || iperm[i0] == NONE {
                    continue;
                }
            }
            vj = di;
            i0 = i;
            k0 = k;
        }
        col_reduced_min[j] = vj;
        if iperm[i0] == NONE {
            num += 1;
            jperm[j] = k0;
            iperm[i0] = j;
            search_from[j] = k0 + 1;
            continue;
        }
        for k in k0..end {
            let i = cost.row_idx[k];
            if (cost.cost[k] - u[i]) > vj {
                continue;
            }
            let jj = iperm[i];
            if jj == NONE {
                continue;
            }
            let jj_end = cost.col_ptr[jj + 1];
            for kk in search_from[jj]..jj_end {
                length2_scans += 1;
                let ii = cost.row_idx[kk];
                if iperm[ii] != NONE {
                    continue;
                }
                if (cost.cost[kk] - u[ii]) <= col_reduced_min[jj] {
                    jperm[jj] = kk;
                    iperm[ii] = jj;
                    search_from[jj] = kk + 1;
                    num += 1;
                    jperm[j] = k;
                    iperm[i] = j;
                    search_from[j] = k + 1;
                    continue 'improve_assign;
                }
            }
            search_from[jj] = jj_end;
        }
    }
    rsprobe::add!(DEBUG, "length-2 scans", length2_scans);

    num
}

pub(crate) fn hungarian_match(cost: &CostGraph<'_>) -> Matching {
    let n = cost.n;
    let m = n;

    let mut iperm: Vec<usize> = vec![NONE; m];
    let mut jperm: Vec<usize> = vec![NONE; n];
    let mut u: Vec<f64> = vec![0.0; m];
    let mut v: Vec<f64> = vec![0.0; n];

    if n == 0 {
        return Matching {
            perm: Vec::new(),
            u,
            v,
            n_matched: 0,
        };
    }

    let mut num = hungarian_init_heuristic(cost, &mut iperm, &mut jperm, &mut u);

    for ui in u.iter_mut() {
        if *ui >= RINF {
            *ui = 0.0;
        }
    }

    if num == n {
        finalize_duals(cost, &jperm, &u, &mut v);
        return build_matching(cost, iperm, jperm, u, v, num);
    }

    let mut d: Vec<f64> = vec![RINF; m];
    let mut pr: Vec<usize> = vec![NONE; n];
    let mut out_idx: Vec<usize> = vec![NONE; n];
    let mut visited: Vec<bool> = vec![false; m];
    let mut touched: Vec<usize> = Vec::with_capacity(m);
    let mut visited_rows: Vec<usize> = Vec::with_capacity(m);
    let mut heap = IndexHeap::new(m);
    let (mut searches, mut touched_rows, mut edge_scans) = (0u64, 0u64, 0u64);

    let steps = crate::progress::Steps::new(rsprobe::active!(), n as u64);
    let report = |step: Option<u64>| {
        if let Some(step) = step {
            rsprobe::progress!(step, crate::constants::PROGRESS_STEPS; "matching");
        }
    };
    for jord in 0..n {
        report(steps.as_ref().and_then(|st| st.reach(jord as u64)));
        if jperm[jord] != NONE {
            continue;
        }
        searches += 1;

        let mut shortest_cost = RINF;
        let mut terminal_edge: usize = NONE;
        let mut terminal_col: usize = NONE;
        visited_rows.clear();
        touched.clear();

        let j = jord;
        pr[j] = NONE;

        edge_scans += (cost.col_ptr[j + 1] - cost.col_ptr[j]) as u64;
        for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
            let i = cost.row_idx[k];
            let dnew = cost.cost[k] - u[i];
            if dnew >= shortest_cost {
                continue;
            }
            if iperm[i] == NONE {
                shortest_cost = dnew;
                terminal_edge = k;
                terminal_col = j;
            } else if dnew < d[i] {
                if d[i] == RINF {
                    touched.push(i);
                }
                d[i] = dnew;
                let jj = iperm[i];
                out_idx[jj] = k;
                pr[jj] = j;
                if heap.contains(i) {
                    heap.update(i, &d);
                } else {
                    heap.insert(i, &d);
                }
            }
        }

        loop {
            if heap.is_empty() {
                break;
            }
            let top = heap.peek();
            if d[top] >= shortest_cost {
                break;
            }
            let q0 = heap.pop(&d);
            visited[q0] = true;
            visited_rows.push(q0);
            let dq0 = d[q0];

            let j2 = iperm[q0];
            let vj = dq0 - cost.cost[jperm[j2]] + u[q0];
            edge_scans += (cost.col_ptr[j2 + 1] - cost.col_ptr[j2]) as u64;
            for k in cost.col_ptr[j2]..cost.col_ptr[j2 + 1] {
                let i = cost.row_idx[k];
                if visited[i] {
                    continue;
                }
                let dnew = vj + cost.cost[k] - u[i];
                if dnew >= shortest_cost {
                    continue;
                }
                if iperm[i] == NONE {
                    shortest_cost = dnew;
                    terminal_edge = k;
                    terminal_col = j2;
                } else {
                    let di = d[i];
                    if di <= dnew {
                        continue;
                    }
                    if d[i] == RINF {
                        touched.push(i);
                    }
                    d[i] = dnew;
                    if heap.contains(i) {
                        heap.update(i, &d);
                    } else {
                        heap.insert(i, &d);
                    }
                    let jj = iperm[i];
                    out_idx[jj] = k;
                    pr[jj] = j2;
                }
            }
        }

        if shortest_cost < RINF {
            num += 1;
            let i_term = cost.row_idx[terminal_edge];
            iperm[i_term] = terminal_col;
            jperm[terminal_col] = terminal_edge;
            let mut j_cur = terminal_col;
            for _ in 0..num {
                let jj = pr[j_cur];
                if jj == NONE {
                    break;
                }
                let k = out_idx[j_cur];
                let i_tree = cost.row_idx[k];
                iperm[i_tree] = jj;
                jperm[jj] = k;
                j_cur = jj;
            }
            for &i in &visited_rows {
                u[i] += d[i] - shortest_cost;
            }
        }

        for &i in &touched {
            d[i] = RINF;
        }
        for &i in &visited_rows {
            visited[i] = false;
        }
        touched_rows += touched.len() as u64;
        heap.reset(&touched);
    }
    report(steps.as_ref().and_then(|st| st.finish()));
    rsprobe::add!(DEBUG, "searches", searches);
    rsprobe::add!(DEBUG, "touched", touched_rows);
    rsprobe::add!(DEBUG, "edge scans", edge_scans);

    finalize_duals(cost, &jperm, &u, &mut v);
    build_matching(cost, iperm, jperm, u, v, num)
}

fn finalize_duals(cost: &CostGraph, jperm: &[usize], u: &[f64], v: &mut [f64]) {
    for (j, vj) in v.iter_mut().enumerate() {
        if jperm[j] != NONE {
            let k = jperm[j];
            let i = cost.row_idx[k];
            *vj = cost.cost[k] - u[i];
        } else {
            *vj = 0.0;
        }
    }
}

fn build_matching(
    cost: &CostGraph<'_>,
    iperm: Vec<usize>,
    jperm: Vec<usize>,
    mut u: Vec<f64>,
    v: Vec<f64>,
    num: usize,
) -> Matching {
    let n = cost.n;
    let mut perm = vec![NONE; n];
    for (j, &k) in jperm.iter().enumerate() {
        if k != NONE {
            perm[j] = cost.row_idx[k];
        }
    }
    for (i, &col) in iperm.iter().enumerate() {
        if col == NONE {
            u[i] = 0.0;
        }
    }
    Matching {
        perm,
        u,
        v,
        n_matched: num,
    }
}

#[cfg(test)]
mod tests {
    //! Hungarian kernel unit tests.
    //!
    //! These tests exercise the `hungarian_match` function directly
    //! on small cost graphs where the answer can be hand-derived.
    //!
    //! Hand-derivation method: any minimum-cost perfect matching on
    //! a bipartite graph satisfies the LP optimality conditions
    //!   `u[i] + v[j] <= cost[i][j]`   for all edges,
    //!   `u[i] + v[j] == cost[i][j]`  on matched edges,
    //! so the matching plus any feasible dual that makes the total
    //! `sum(u) + sum(v)` equal to `sum(cost[matched])` is optimal.

    use super::*;
    use crate::test_constants::linearithmic_growth;

    fn build_cost_graph(n: usize, entries: &[(usize, usize, f64)]) -> CostGraph<'static> {
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in entries {
            by_col[c].push((r, v));
        }
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::new();
        let mut cost = Vec::new();
        for j in 0..n {
            by_col[j].sort_by_key(|&(r, _)| r);
            for &(r, v) in &by_col[j] {
                row_idx.push(r);
                cost.push(v);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        CostGraph {
            n,
            col_ptr: col_ptr.into(),
            row_idx: row_idx.into(),
            cost,
        }
    }

    fn assert_matching_optimal(cost: &CostGraph, m: &Matching) {
        let n = cost.n;
        assert_eq!(m.u.len(), n);
        assert_eq!(m.v.len(), n);
        assert_eq!(m.perm.len(), n);

        let mut matched_row = vec![false; n];
        for j in 0..n {
            if m.perm[j] != usize::MAX {
                matched_row[m.perm[j]] = true;
            }
        }

        for j in 0..n {
            for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
                let i = cost.row_idx[k];
                let c = cost.cost[k];
                let reduced = m.u[i] + m.v[j];
                assert!(
                    reduced <= c,
                    "edge ({},{}) has cost {} but u+v={} (reduced > cost)",
                    i,
                    j,
                    c,
                    reduced
                );
                if m.perm[j] == i {
                    assert_eq!(
                        reduced, c,
                        "matched edge ({},{}) has cost {} but u+v={} (not tight)",
                        i, j, c, reduced
                    );
                }
            }
        }
    }

    #[test]
    fn match_diagonal_3x3_identity() {
        let cost = build_cost_graph(3, &[(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)]);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm, vec![0, 1, 2]);
        assert_matching_optimal(&cost, &m);
    }

    #[test]
    fn match_permutation_3x3() {
        let cost = build_cost_graph(3, &[(1, 0, 0.0), (2, 1, 0.0), (0, 2, 0.0)]);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm[0], 1, "col 0 should match row 1");
        assert_eq!(m.perm[1], 2, "col 1 should match row 2");
        assert_eq!(m.perm[2], 0, "col 2 should match row 0");
        assert_matching_optimal(&cost, &m);
    }

    #[test]
    fn match_hand_computed_3x3() {
        let cost = build_cost_graph(
            3,
            &[
                (0, 0, 3.0),
                (1, 0, 1.0),
                (0, 1, 2.0),
                (2, 1, 4.0),
                (1, 2, 5.0),
                (2, 2, 0.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm[0], 1, "col 0 matches row 1 (cost 1)");
        assert_eq!(m.perm[1], 0, "col 1 matches row 0 (cost 2)");
        assert_eq!(m.perm[2], 2, "col 2 matches row 2 (cost 0)");
        assert_matching_optimal(&cost, &m);
        let total: f64 = (0..3).map(|j| m.u[m.perm[j]] + m.v[j]).sum();
        assert_eq!(
            total, 3.0,
            "total matching cost should be 3 (1+2+0), got {}",
            total
        );
    }

    #[test]
    fn match_dense_4x4() {
        let n = 4;
        let mat = [
            [1.0_f64, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 1.0, 2.0],
            [4.0, 8.0, 2.0, 1.0],
        ];
        let mut entries = Vec::new();
        for (i, row) in mat.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                entries.push((i, j, v));
            }
        }
        let cost = build_cost_graph(n, &entries);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, n);
        assert_matching_optimal(&cost, &m);
        let total: f64 = (0..n).map(|j| mat[m.perm[j]][j]).sum();
        assert_eq!(
            total, 6.0,
            "total matching cost should be 6, got {} with perm {:?}",
            total, m.perm
        );
    }

    #[test]
    fn match_sparse_5x5() {
        let cost = build_cost_graph(
            5,
            &[
                (0, 0, 10.0),
                (1, 0, 1.0),
                (0, 1, 1.0),
                (2, 1, 10.0),
                (1, 2, 10.0),
                (3, 2, 1.0),
                (2, 3, 1.0),
                (4, 3, 10.0),
                (3, 4, 10.0),
                (4, 4, 1.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 5);
        assert_matching_optimal(&cost, &m);
        let mut total = 0.0;
        for j in 0..5 {
            for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
                if cost.row_idx[k] == m.perm[j] {
                    total += cost.cost[k];
                }
            }
        }
        assert_eq!(
            total, 5.0,
            "total matching cost should be 5, got {} with perm {:?}",
            total, m.perm
        );
    }

    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() >> 33) as usize % bound
        }
    }

    fn gen_random_sparse(n: usize, deg: usize, seed: u64) -> CostGraph<'static> {
        let mut rng = Lcg(seed);
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for col in by_col.iter_mut() {
            let mut rows: Vec<usize> = Vec::with_capacity(deg);
            while rows.len() < deg {
                let r = rng.below(n);
                if !rows.contains(&r) {
                    rows.push(r);
                }
            }
            for &r in &rows {
                let c = 1.0 + (rng.below(100) as f64);
                col.push((r, c));
            }
        }
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::new();
        let mut cost = Vec::new();
        for j in 0..n {
            by_col[j].sort_by_key(|&(r, _)| r);
            for &(r, v) in &by_col[j] {
                row_idx.push(r);
                cost.push(v);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        CostGraph {
            n,
            col_ptr: col_ptr.into(),
            row_idx: row_idx.into(),
            cost,
        }
    }

    fn matching_work(cost: &CostGraph<'_>) -> Option<rsprobe::Record> {
        let ctx = rsprobe::Context::builder()
            .threshold(rsprobe::Level::DEBUG)
            .build();
        let _on = ctx.enter();
        if !rsprobe::active!(DEBUG) {
            return None;
        }
        {
            let _s = rsprobe::scope!(ERROR, "matching");
            hungarian_match(cost);
        }
        Some(ctx.record())
    }

    #[test]
    fn matching_work_is_linearithmic() {
        let mut length2 = Vec::new();
        let mut nnz = Vec::new();
        for n in [1000usize, 2000, 4000, 8000] {
            let cost = gen_random_sparse(n, 3, 0x1234_5678);
            let Some(r) = matching_work(&cost) else {
                return;
            };
            let node = &r.nodes[r.find(&["matching"]).expect("the matching's scope") as usize];
            assert!(node.get("searches") > 0, "n={n}: no augmenting search ran");
            assert_eq!(
                node.get("heap slots"),
                n as u64 + node.get("touched"),
                "n={n}: the heap zeroes more than n plus the touched rows"
            );
            length2.push(node.get("length-2 scans") as f64);
            nnz.push(cost.row_idx.len());
        }
        assert!(length2[0] > 0.0, "the length-2 augmentation never ran");
        let growth = length2[3] / length2[0];
        let bound = linearithmic_growth(nnz[0], nnz[3]);
        assert!(
            growth <= bound,
            "length-2 scans grew {growth:.2}x over nnz {} -> {}, bound {bound:.2}x",
            nnz[0],
            nnz[3]
        );
    }

    #[test]
    fn match_random_sparse_is_optimal() {
        for n in [1000usize, 2000, 4000, 8000] {
            let cost = gen_random_sparse(n, 3, 0x1234_5678);
            assert_matching_optimal(&cost, &hungarian_match(&cost));
        }
    }

    #[test]
    fn match_structurally_singular_3x3() {
        let cost = build_cost_graph(
            3,
            &[
                (0, 0, 1.0),
                (1, 0, 2.0),
                (0, 1, 3.0),
                (1, 1, 4.0),
                (0, 2, 5.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 2, "only 2 of 3 columns should match");
        let n_unmatched = m.perm.iter().filter(|&&p| p == usize::MAX).count();
        assert_eq!(n_unmatched, 1);
        assert_matching_optimal(&cost, &m);
    }
}
