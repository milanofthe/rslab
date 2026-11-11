//! Sparse right-hand sides on the plan: the reach of a sparse column in the
//! supernodal factor, forward sweeps limited to it, and the port matrix
//! `Z = B^T A^-1 B` from forward sweeps alone.
//!
//! ## Reach
//!
//! Column `j` of `L` has its nonzeros in rows of the supernodes its
//! supernode's off-block rows lie in, so `L^-1 b` is zero outside the
//! closure of `b`'s nonzeros under "the supernodes of a supernode's
//! off-block rows" (Gilbert and Peierls 1988; Davis, *Direct Methods for
//! Sparse Linear Systems*, SIAM 2006, sec. 3.2). On an elimination tree that
//! is the path from each nonzero's supernode to the root. The closure is
//! taken over the row lists themselves, so it holds on any structure the
//! plan accepts. A supernode's off-block rows lie in later supernodes, so
//! the reach in ascending order is an order a forward sweep may take.
//!
//! ## Port matrix
//!
//! With `P^T S A S P = L D L^T` (`S` the real equilibration, `P` the
//! elimination order, `D` with 1x1 and 2x2 blocks):
//!
//! ```text
//! Z = B^T A^-1 B = (L^-1 P^T S B)^T D^-1 (L^-1 P^T S B) = Y^T D^-1 Y
//! ```
//!
//! with the plain transpose (complex symmetric, not Hermitian). `Y` is the
//! forward sweep of the scaled, permuted columns, restricted to their reach;
//! no backward sweep is needed. A 2x2 block of `D` couples two columns of
//! one supernode, which a reach holds together.
//!
//! The columns go in blocks of [`SolveSettings::port_block`](crate::SolveSettings::port_block),
//! ordered by the first supernode each one reaches (columns starting in one
//! subtree share most of their reach). A block's `Y` is swept on the union
//! of its columns' reaches in a compact buffer of just those rows, each
//! column in its own lane by the same operations whatever the block. An
//! entry of `Z` sums `y_a v_c` (`v = D^-1 y`) over the rows two blocks
//! share, in ascending order; the rows of a block's union outside a
//! column's own reach hold exact zeros there. So `Z` does not depend on the
//! block width, the column grouping or the thread count, and it is exactly
//! symmetric (each pair is computed once). (A column step whose lanes are
//! all zero is skipped, so exact zeros may differ in sign between
//! groupings; values do not.)

use super::{packed_blocks, solve_diagonal, PlanWork, SolvePlan, NONE};
use crate::constants::SOLVE_PAR_MIN_ENTRY_RHS;
use crate::constants::SOLVE_RHS_TILE;
use crate::error::RslabError;
use crate::numeric::ldlt::LdltPivots;
use crate::numeric::settings::Threads;
use crate::parallel::Work;
use crate::scalar::Scalar;
use crate::sparse::rhs::SparseRhs;

/// How much of the factor a sparse right-hand side reaches
/// ([`LdltSolver::sparse_reach`](crate::LdltSolver::sparse_reach)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseReach {
    /// Supernodes in the reach.
    pub supernodes: usize,
    /// Factor columns (unknowns) in the reach.
    pub columns: usize,
    /// Panel entries of the reach's supernodes: the factor values a forward
    /// sweep limited to it reads.
    pub panel_entries: usize,
    /// Supernodes of the factor.
    pub total_supernodes: usize,
    /// Panel entries of the factor.
    pub total_panel_entries: usize,
}

/// The column maps of one call: supernode of every factor column, and the
/// factor column of every original row.
pub(crate) struct ColumnMaps {
    sn_of: Vec<u32>,
    elim_of: Vec<usize>,
}

/// One worker's marks and buffers.
struct ReachScratch<T> {
    /// `mark[s] == stamp` once supernode `s` is in the current reach.
    mark: Vec<u32>,
    stamp: u32,
    stack: Vec<u32>,
    /// Factor column -> compact row of the current block (`NONE` outside).
    map: Vec<u32>,
    /// A node's products of its off-block rows.
    t: Vec<T>,
}

impl<T> Default for ReachScratch<T> {
    fn default() -> Self {
        Self {
            mark: Vec::new(),
            stamp: 0,
            stack: Vec::new(),
            map: Vec::new(),
            t: Vec::new(),
        }
    }
}

impl<T> ReachScratch<T> {
    fn fit(&mut self, ns: usize, n: usize) {
        if self.mark.len() != ns {
            self.mark = vec![0; ns];
            self.stamp = 0;
        }
        if self.map.len() != n {
            self.map = vec![NONE; n];
        }
    }
}

/// `Y` of one block of sparse columns on the union of their reaches.
struct ForwardBlock<T> {
    /// Columns of `B` in lane order.
    cols: Vec<usize>,
    /// Supernodes of the reach, ascending.
    nodes: Vec<u32>,
    /// Compact row of each node's first column; a node's columns are
    /// consecutive rows.
    offsets: Vec<usize>,
    /// Row-major `rows x lanes` (`y[q * lanes + c]`).
    y: Vec<T>,
    /// `D^-1 y`, the same layout (port matrix only).
    v: Vec<T>,
}

impl<T: Scalar> SolvePlan<T> {
    /// The maps of one call over the elimination order `perm`.
    pub(crate) fn column_maps(&self, perm: &[usize]) -> ColumnMaps {
        let mut sn_of = vec![0u32; self.n];
        for s in 0..self.sn_col.len() - 1 {
            sn_of[self.sn_col[s] as usize..self.sn_col[s + 1] as usize].fill(s as u32);
        }
        let mut elim_of = vec![0usize; self.n];
        for (e, &i) in perm.iter().enumerate() {
            elim_of[i] = e;
        }
        ColumnMaps { sn_of, elim_of }
    }

    /// The reach of the factor columns `cols` into `out`, ascending.
    fn reach_into(
        &self,
        maps: &ColumnMaps,
        cols: impl Iterator<Item = usize>,
        sc: &mut ReachScratch<T>,
        out: &mut Vec<u32>,
    ) {
        sc.fit(self.sn_col.len() - 1, self.n);
        if sc.stamp == u32::MAX {
            sc.mark.fill(0);
            sc.stamp = 0;
        }
        sc.stamp += 1;
        let stamp = sc.stamp;
        out.clear();
        for c in cols {
            let s0 = maps.sn_of[c];
            if sc.mark[s0 as usize] == stamp {
                continue;
            }
            sc.mark[s0 as usize] = stamp;
            sc.stack.push(s0);
            while let Some(s) = sc.stack.pop() {
                out.push(s);
                let s = s as usize;
                let mut last = NONE;
                for &r in &self.rows[self.row_ptr[s]..self.row_ptr[s + 1]] {
                    let a = maps.sn_of[r as usize];
                    if a != last && sc.mark[a as usize] != stamp {
                        sc.mark[a as usize] = stamp;
                        sc.stack.push(a);
                    }
                    last = a;
                }
            }
        }
        out.sort_unstable();
    }

    /// Reach statistics of the original rows `rows`.
    pub(crate) fn sparse_reach(&self, perm: &[usize], rows: &[usize]) -> SparseReach {
        let maps = self.column_maps(perm);
        let mut sc = ReachScratch::<T>::default();
        let mut nodes = Vec::new();
        self.reach_into(
            &maps,
            rows.iter().map(|&i| maps.elim_of[i]),
            &mut sc,
            &mut nodes,
        );
        let entries = |s: usize| self.val_rng[s].1 - self.val_rng[s].0;
        SparseReach {
            supernodes: nodes.len(),
            columns: nodes
                .iter()
                .map(|&s| (self.sn_col[s as usize + 1] - self.sn_col[s as usize]) as usize)
                .sum(),
            panel_entries: nodes.iter().map(|&s| entries(s as usize)).sum(),
            total_supernodes: self.sn_col.len() - 1,
            total_panel_entries: self.vals.len(),
        }
    }

    /// The forward sweep `L y = y` of node `s` on the compact buffer `y`
    /// (`nr` lanes, the node's columns at rows `base..`), its off-block rows
    /// found through `map`. Column by column in the diagonal block; the
    /// off-block rows take their terms summed over the node's columns in
    /// ascending order, then once.
    fn fwd_node_compact(
        &self,
        s: u32,
        base: usize,
        nr: usize,
        y: &mut [T],
        map: &[u32],
        t: &mut Vec<T>,
    ) {
        let (_, w, r0, m, _, panel) = self.node(s);
        t.clear();
        t.resize(m * nr, T::zero());
        for (jb, je, off, ld) in packed_blocks(w, m) {
            for k in jb..je {
                let col = &panel[off + (k - jb) * ld..off + (k - jb + 1) * ld];
                let (head, tail) = y.split_at_mut((base + k + 1) * nr);
                let xk = &head[(base + k) * nr..];
                // A column whose lanes are all exactly zero changes nothing
                // (up to the sign of exact zeros): common near a port's
                // start, where most of a node's columns are still zero.
                if xk.iter().all(|&x| x == T::zero()) {
                    continue;
                }
                for i in k + 1..w {
                    let l = col[i - jb];
                    let yi = &mut tail[(i - k - 1) * nr..(i - k) * nr];
                    for (a, &x) in yi.iter_mut().zip(xk) {
                        *a = *a - l * x;
                    }
                }
                for r in 0..m {
                    let l = col[w + r - jb];
                    for (a, &x) in t[r * nr..(r + 1) * nr].iter_mut().zip(xk) {
                        *a = *a + l * x;
                    }
                }
            }
        }
        for (r, &row) in self.rows[r0..r0 + m].iter().enumerate() {
            let q = map[row as usize] as usize;
            for (a, &d) in y[q * nr..(q + 1) * nr].iter_mut().zip(&t[r * nr..]) {
                *a = *a - d;
            }
        }
    }

    /// `Y` of the columns `cols` of `b`: their reach, the scaled, permuted
    /// entries placed in a compact buffer, and the forward sweep over it;
    /// with `f`, also `D^-1 Y`.
    fn sparse_forward(
        &self,
        maps: &ColumnMaps,
        scale: &[f64],
        b: &SparseRhs<T>,
        cols: &[usize],
        sc: &mut ReachScratch<T>,
        f: Option<&LdltPivots<T>>,
    ) -> Result<ForwardBlock<T>, RslabError> {
        let nr = cols.len();
        let mut nodes = Vec::new();
        let starts = cols
            .iter()
            .flat_map(|&c| b.column(c).0.iter().map(|&i| maps.elim_of[i]));
        self.reach_into(maps, starts, sc, &mut nodes);
        let mut offsets = Vec::with_capacity(nodes.len());
        let mut rows = 0usize;
        for &s in &nodes {
            offsets.push(rows);
            for c in self.sn_col[s as usize]..self.sn_col[s as usize + 1] {
                sc.map[c as usize] = rows as u32;
                rows += 1;
            }
        }
        let mut y = vec![T::zero(); rows * nr];
        for (lane, &c) in cols.iter().enumerate() {
            let (rs, vs) = b.column(c);
            for (&i, &v) in rs.iter().zip(vs) {
                let q = sc.map[maps.elim_of[i]] as usize;
                y[q * nr + lane] = y[q * nr + lane] + v * T::from_real(scale[i]);
            }
        }
        let mut t = std::mem::take(&mut sc.t);
        for (&s, &base) in nodes.iter().zip(&offsets) {
            self.fwd_node_compact(s, base, nr, &mut y, &sc.map, &mut t);
        }
        sc.t = t;
        for &s in &nodes {
            sc.map[self.sn_col[s as usize] as usize..self.sn_col[s as usize + 1] as usize]
                .fill(NONE);
        }
        let v = match f {
            Some(f) => self.diagonal_compact(f, &nodes, &offsets, nr, &y)?,
            None => Vec::new(),
        };
        Ok(ForwardBlock {
            cols: cols.to_vec(),
            nodes,
            offsets,
            y,
            v,
        })
    }

    /// `D^-1 y` on a compact buffer, 1x1 and 2x2 blocks as
    /// [`solve_diagonal`] takes them.
    fn diagonal_compact(
        &self,
        f: &LdltPivots<T>,
        nodes: &[u32],
        offsets: &[usize],
        nr: usize,
        y: &[T],
    ) -> Result<Vec<T>, RslabError> {
        let mut v = y.to_vec();
        for (&s, &base) in nodes.iter().zip(offsets) {
            let (c0, c1) = (
                self.sn_col[s as usize] as usize,
                self.sn_col[s as usize + 1] as usize,
            );
            let rows = &mut v[base * nr..(base + c1 - c0) * nr];
            // A 2x2 block never straddles a supernode: its pivots are chosen
            // inside one fully-summed block. `solve_diagonal` on the node's
            // rows is the global one restricted to them.
            let local = LocalPivots { f, c0, c1 };
            local.solve(rows, nr)?;
        }
        Ok(v)
    }

    /// The task budget and whether a call over sparse columns runs on the
    /// pool: the plan's gate for a sweep of `nr` right-hand sides.
    fn sparse_parallel(&self, nr: usize, threads: Threads) -> bool {
        let budget = threads.resolve(|cap| cap);
        let ops = self.vals.len().saturating_mul(nr.min(SOLVE_RHS_TILE));
        budget > 1
            && match self.cfg.par_min_work {
                Some(min) => ops >= min,
                None => ops >= SOLVE_PAR_MIN_ENTRY_RHS,
            }
    }

    /// `f(i)` for every block index, on the pool when `par`, each piece with
    /// its own scratch; results in order.
    fn map_blocks<R: Send>(
        par: bool,
        threads: Threads,
        n: usize,
        f: impl Fn(&mut ReachScratch<T>, usize) -> R + Sync + Send,
    ) -> Vec<R> {
        if !par || n < 2 {
            let mut sc = ReachScratch::default();
            return (0..n).map(|i| f(&mut sc, i)).collect();
        }
        Self::in_pool(true, threads, || {
            crate::parallel::map_init(
                n,
                Work::ns(f64::INFINITY),
                ReachScratch::default,
                |sc, i| f(sc, i),
            )
        })
    }

    /// The columns of `b` in the order the blocks take them: by the first
    /// supernode each one reaches, then by index; empty columns last.
    fn sparse_order(&self, maps: &ColumnMaps, b: &SparseRhs<T>) -> Vec<usize> {
        let key = |c: usize| {
            (b.column(c).0.iter())
                .map(|&i| maps.sn_of[maps.elim_of[i]])
                .min()
                .unwrap_or(NONE)
        };
        let mut order: Vec<(u32, usize)> = (0..b.ncols()).map(|c| (key(c), c)).collect();
        order.sort_unstable();
        order.into_iter().map(|(_, c)| c).collect()
    }

    /// `Z = B^T A^-1 B` (column-major `ncols x ncols`, exactly symmetric) by
    /// forward sweeps only; see the module docs.
    pub(crate) fn port_matrix(
        &self,
        f: &LdltPivots<T>,
        scale: &[f64],
        b: &SparseRhs<T>,
        threads: Threads,
    ) -> Result<Vec<T>, RslabError> {
        let p = b.ncols();
        let maps = self.column_maps(&f.perm);
        let order = self.sparse_order(&maps, b);
        let groups: Vec<&[usize]> = order.chunks(self.cfg.port_block.max(1)).collect();
        let par = self.sparse_parallel(self.cfg.port_block.min(p), threads);
        let blocks = Self::map_blocks(par, threads, groups.len(), |sc, i| {
            self.sparse_forward(&maps, scale, b, groups[i], sc, Some(f))
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let pairs: Vec<(usize, usize)> = (0..blocks.len())
            .flat_map(|i| (i..blocks.len()).map(move |j| (i, j)))
            .collect();
        let products = Self::map_blocks(par, threads, pairs.len(), |_, k| {
            let (i, j) = pairs[k];
            self.block_product(&blocks[i], &blocks[j])
        });
        let mut z = vec![T::zero(); p * p];
        for (&(i, j), prod) in pairs.iter().zip(&products) {
            let (bi, bj) = (&blocks[i], &blocks[j]);
            let lj = bj.cols.len();
            for (a, &pa) in bi.cols.iter().enumerate() {
                for (c, &pc) in bj.cols.iter().enumerate() {
                    if i == j && c < a {
                        continue;
                    }
                    let v = prod[a * lj + c];
                    z[pc * p + pa] = v;
                    z[pa * p + pc] = v;
                }
            }
        }
        Ok(z)
    }

    /// `Y_I^T V_J` over the rows both blocks reach, ascending; row-major
    /// `lanes_I x lanes_J`.
    fn block_product(&self, bi: &ForwardBlock<T>, bj: &ForwardBlock<T>) -> Vec<T> {
        let (li, lj) = (bi.cols.len(), bj.cols.len());
        let mut out = vec![T::zero(); li * lj];
        let (mut p, mut q) = (0, 0);
        while p < bi.nodes.len() && q < bj.nodes.len() {
            let (s, t) = (bi.nodes[p], bj.nodes[q]);
            if s != t {
                if s < t {
                    p += 1;
                } else {
                    q += 1;
                }
                continue;
            }
            let w = (self.sn_col[s as usize + 1] - self.sn_col[s as usize]) as usize;
            let (oi, oj) = (bi.offsets[p], bj.offsets[q]);
            for k in 0..w {
                let yi = &bi.y[(oi + k) * li..(oi + k + 1) * li];
                let vj = &bj.v[(oj + k) * lj..(oj + k + 1) * lj];
                for (a, &ya) in yi.iter().enumerate() {
                    for (o, &vc) in out[a * lj..(a + 1) * lj].iter_mut().zip(vj) {
                        *o = *o + ya * vc;
                    }
                }
            }
            p += 1;
            q += 1;
        }
        out
    }

    /// `A X = B` for sparse `B`: the forward sweep limited to each block's
    /// reach, then the diagonal and the full backward sweep on all columns.
    /// The dense column-major `n x ncols` solution.
    pub(crate) fn solve_sparse(
        &self,
        f: &LdltPivots<T>,
        scale: &[f64],
        b: &SparseRhs<T>,
        threads: Threads,
    ) -> Result<Vec<T>, RslabError> {
        let (n, p) = (self.n, b.ncols());
        let maps = self.column_maps(&f.perm);
        let order = self.sparse_order(&maps, b);
        let groups: Vec<&[usize]> = order.chunks(self.cfg.port_block.max(1)).collect();
        let par = self.sparse_parallel(self.cfg.port_block.min(p), threads);
        let blocks = Self::map_blocks(par, threads, groups.len(), |sc, i| {
            self.sparse_forward(&maps, scale, b, groups[i], sc, None)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        let mut y = vec![T::zero(); n * p];
        for blk in &blocks {
            let nr = blk.cols.len();
            for (&s, &base) in blk.nodes.iter().zip(&blk.offsets) {
                let c0 = self.sn_col[s as usize] as usize;
                let w = self.sn_col[s as usize + 1] as usize - c0;
                for k in 0..w {
                    let row = &blk.y[(base + k) * nr..(base + k + 1) * nr];
                    for (lane, &c) in blk.cols.iter().enumerate() {
                        y[(c0 + k) * p + c] = row[lane];
                    }
                }
            }
        }
        drop(blocks);
        solve_diagonal(f, &mut y, p)?;
        self.backward(p, &mut y, &mut PlanWork::default(), threads);
        let mut x = vec![T::zero(); n * p];
        for (e, &i) in f.perm.iter().enumerate() {
            let si = T::from_real(scale[i]);
            for c in 0..p {
                x[c * n + i] = y[e * p + c] * si;
            }
        }
        Ok(x)
    }
}

/// The block diagonal of the columns `c0..c1` of one supernode.
struct LocalPivots<'a, T> {
    f: &'a LdltPivots<T>,
    c0: usize,
    c1: usize,
}

impl<T: Scalar> LocalPivots<'_, T> {
    /// `D z = y` on the node's rows (`nr` lanes), as [`solve_diagonal`] does
    /// on the whole factor, with its error on a singular block.
    fn solve(&self, y: &mut [T], nr: usize) -> Result<(), RslabError> {
        let f = self.f;
        let mut k = self.c0;
        while k < self.c1 {
            let row = (k - self.c0) * nr;
            if f.two_by_two[k] && k + 1 >= self.c1 {
                // A 2x2 block is chosen inside one fully-summed block, so it
                // cannot start at a supernode's last column; if it does, the
                // factor and the plan disagree and `Z` would be wrong.
                debug_assert!(false, "2x2 pivot at the last column {k} of a supernode");
                return Err(RslabError::InvalidInput(format!(
                    "2x2 pivot at column {k} straddles a supernode boundary: \
                     the factor does not match its solve plan"
                )));
            }
            if f.two_by_two[k] {
                let (d11, d21, d22) = (f.d_diag[k], f.d_subdiag[k], f.d_diag[k + 1]);
                let det = d11 * d22 - d21 * d21;
                if det == T::zero() {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                let detinv = det.recip();
                let (r0, r1) = y[row..row + 2 * nr].split_at_mut(nr);
                for c in 0..nr {
                    let (z0, z1) = (r0[c], r1[c]);
                    r0[c] = (d22 * z0 - d21 * z1) * detinv;
                    r1[c] = (d11 * z1 - d21 * z0) * detinv;
                }
                k += 2;
            } else {
                if f.d_diag[k] == T::zero() {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                let dinv = f.d_diag[k].recip();
                for v in &mut y[row..row + nr] {
                    *v = *v * dinv;
                }
                k += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::settings::SolverSettings;
    use crate::sparse::csc::CscMatrix;
    use crate::LdltSolver;
    use num_complex::Complex;

    type C = Complex<f64>;

    /// A complex-symmetric 2D 5-point Helmholtz-like matrix (lower triangle).
    fn grid(m: usize) -> CscMatrix<C> {
        let n = m * m;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            r.push(j);
            c.push(j);
            v.push(C::new(3.7, 0.05 + 0.001 * (j % 7) as f64));
            if j % m + 1 < m {
                r.push(j + 1);
                c.push(j);
                v.push(C::new(-1.0, 0.0));
            }
            if j + m < n {
                r.push(j + m);
                c.push(j);
                v.push(C::new(-1.0, 0.0));
            }
        }
        CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
    }

    /// A 2x2 pivot flagged at a supernode's last column is refused (and
    /// asserted in debug builds), never taken as a 1x1 pivot.
    #[test]
    fn straddling_two_by_two_is_refused() {
        let a = grid(6);
        let f = LdltSolver::factor(&a, &SolverSettings::default().with_threads(1)).unwrap();
        let (plan, pv, _) = f.parts();
        let mut bad = pv.clone();
        let (c0, c1) = (plan.sn_col[0] as usize, plan.sn_col[1] as usize);
        bad.two_by_two[c1 - 1] = true;
        let local = LocalPivots { f: &bad, c0, c1 };
        let mut y = vec![C::new(1.0, 0.0); c1 - c0];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| local.solve(&mut y, 1)));
        if cfg!(debug_assertions) {
            assert!(r.is_err(), "debug builds assert");
        } else {
            assert!(matches!(r, Ok(Err(RslabError::InvalidInput(_)))));
        }
    }

    /// The reach holds every nonzero of the full forward sweep, and the
    /// compact sweep equals the plan's own forward sweep on it to rounding.
    #[test]
    fn reach_holds_the_forward_sweep_and_compact_equals_full() {
        let a = grid(18);
        let f = LdltSolver::factor(&a, &SolverSettings::default().with_threads(1)).unwrap();
        let (plan, pv, scale) = f.parts();
        let maps = plan.column_maps(&pv.perm);
        let mut sc = ReachScratch::default();
        for &row in &[0usize, 17, 150, 323] {
            let b = SparseRhs::from_columns(a.n, &[vec![(row, C::new(1.0, 0.5))]]).unwrap();
            let blk = (plan.sparse_forward(&maps, scale, &b, &[0], &mut sc, None)).unwrap();
            // the full forward sweep of the same scaled, permuted column
            let mut y = vec![C::new(0.0, 0.0); a.n];
            y[maps.elim_of[row]] = C::new(1.0, 0.5) * scale[row];
            plan.forward(1, &mut y, &mut PlanWork::default(), Threads::Fixed(1));
            let mut covered = vec![false; a.n];
            for (&s, &base) in blk.nodes.iter().zip(&blk.offsets) {
                let c0 = plan.sn_col[s as usize] as usize;
                for k in 0..(plan.sn_col[s as usize + 1] as usize - c0) {
                    covered[c0 + k] = true;
                    let d = (blk.y[base + k] - y[c0 + k]).norm();
                    assert!(
                        d <= 1e-13 * (1.0 + y[c0 + k].norm()),
                        "row {row} col {}",
                        c0 + k
                    );
                }
            }
            for e in 0..a.n {
                assert!(
                    covered[e] || y[e] == C::new(0.0, 0.0),
                    "nonzero outside the reach"
                );
            }
        }
    }
}
