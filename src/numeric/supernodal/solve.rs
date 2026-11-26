//! Supernodal, tree-parallel triangular solves for the sparse LDL^T factor.
//!
//! A scalar CSC factor is the wrong layout for a solve: every entry costs a
//! value, a `usize` row index and a random access into the right-hand side,
//! on one core. The plan works on the factor in *supernodal panel form*
//! ([`PanelFactor`], laid out as [`Layout`]): per supernode its diagonal
//! part in column blocks and its off-block rows as one rectangle, with a
//! single shared list of `u32` row indices. The factorizations write that
//! layout, so the plan takes the panels as its storage as they are. A
//! forward or backward sweep then streams
//! contiguous panels, touches each row index once per panel instead of once
//! per entry, and runs the inner loops over contiguous memory. Every node
//! takes all right-hand sides at once, its dense parts as GEMMs: its
//! triangle in column blocks, the rows below it in one product. A column of
//! a block solve therefore agrees with the solve of that column alone to
//! rounding, the products summing in their own order (on a 95 k near-field
//! LU about 4e-10 relative), not to the bit.
//!
//! Parallelism comes from the supernodal elimination tree. The tree is cut
//! into a fixed number of independent leaf subtrees (`SolveSettings::leaf_subtrees`, not a
//! function of the thread count) plus the ancestors above the cut. In the
//! forward sweep the subtrees run in parallel; updates that leave a subtree
//! (into ancestor rows) go to a per-subtree accumulator. The ancestors then
//! run as a task DAG over the 64-column tiles of their dense nodes: in the
//! forward sweep a tile of an ancestor starts once the row chunks of its
//! children that reach it are done, takes in the accumulations for its
//! columns in the order a level-by-level sweep reduces them, and the products
//! of a solved tile run while the next tile solves its triangle; in the
//! backward sweep an ancestor starts when its parent is done, and the
//! subtrees follow in parallel reading only final values. Every sum is formed
//! in the same order for every thread count, so the solution is
//! bit-identical from one thread to many.

#![allow(clippy::needless_range_loop)]

use crate::constants::{
    IDX32_BYTES, PANEL_BLOCK, SOLVE_BUILD_NODE_BYTES, SOLVE_GEMV_ROWS, SOLVE_PAR_MIN_ENTRY_RHS,
    SOLVE_RHS_TILE, TILE_ROWS, VEC_HEADER_BYTES, WORD_BYTES,
};
use crate::error::RslabError;
use crate::memory::grow;
use crate::numeric::ldlt::LdltPivots;
use crate::numeric::settings::Threads;
use crate::numeric::supernodal::panel::{Layout, PanelFactor, PanelStorage};
use crate::scalar::Scalar;

const NONE: u32 = u32::MAX;

const FIRST_WRITE: u32 = !(u32::MAX >> 1);

mod dag;
mod sparse;
pub use sparse::SparseReach;

#[derive(Clone)]
pub(crate) struct Subtree {
    nodes: Vec<u32>,
    // https://github.com/milanofthe/rslab/issues/289
    ext: Vec<u32>,
}

#[derive(Clone)]
pub(crate) struct SolvePlan<T> {
    n: usize,
    sn_col: Vec<u32>,
    row_ptr: Vec<usize>,
    rows: Vec<u32>,
    ext_slot: Vec<u32>,
    val_rng: Vec<(usize, usize)>,
    vals: Vec<T>,
    spare_rows: Vec<u32>,
    diag_inv: Vec<T>,
    subtrees: Vec<Subtree>,
    pub(crate) top_levels: Vec<Vec<u32>>,
    top_index: Vec<u32>,
    top_acc_ptr: Vec<Vec<usize>>,
    top: Vec<u32>,
    top_roots: Vec<u32>,
    top_parent: Vec<u32>,
    top_kids_ptr: Vec<usize>,
    top_kids: Vec<u32>,
    top_acc_off: Vec<usize>,
    pull_ptr: Vec<usize>,
    pull: Vec<(u32, u32, u32)>,
    chain: bool,
    subtree_work: usize,
    cfg: crate::SolveSettings,
}

// https://github.com/milanofthe/rslab/issues/289
pub(crate) struct Scratch<T> {
    t: Vec<T>,
    g: Vec<T>,
    acc: Vec<T>,
}

impl<T> Default for Scratch<T> {
    fn default() -> Self {
        Self {
            t: Vec::new(),
            g: Vec::new(),
            acc: Vec::new(),
        }
    }
}

pub(crate) struct PlanWork<T> {
    subtrees: Vec<Scratch<T>>,
    level: Vec<Scratch<T>>,
    apex: Scratch<T>,
    top_acc: Vec<T>,
    dag: dag::DagWork,
    parallel: bool,
}

impl<T> Default for PlanWork<T> {
    fn default() -> Self {
        Self {
            subtrees: Vec::new(),
            level: Vec::new(),
            apex: Scratch::default(),
            top_acc: Vec::new(),
            dag: dag::DagWork::default(),
            parallel: false,
        }
    }
}

impl<T> PlanWork<T> {
    pub(crate) fn on_pool(&self) -> bool {
        self.parallel
    }

    fn fit(&mut self, subtrees: usize, widest: usize) {
        if self.subtrees.len() < subtrees {
            self.subtrees.resize_with(subtrees, Scratch::default);
        }
        if self.level.len() < widest {
            self.level.resize_with(widest, Scratch::default);
        }
    }
}

#[derive(Clone, Copy)]
struct Shared<T>(*mut T, usize);
// SAFETY: concurrent users write disjoint index sets (a subtree writes only
// its own columns) and read only indices no other task writes during the
// phase (its own columns, or ancestor columns that are final).
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Sync> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline]
    fn ptr(&self) -> *mut T {
        self.0
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0, self.1)
    }
}

fn cut(
    parent: &[u32],
    children: &[Vec<u32>],
    leaf_subtrees: usize,
    own: impl Fn(usize) -> u64,
) -> (Vec<bool>, Vec<u32>) {
    let ns = parent.len();
    let mut work = vec![0u64; ns];
    for s in 0..ns {
        work[s] += own(s);
        if parent[s] != NONE {
            let w = work[s];
            work[parent[s] as usize] += w;
        }
    }
    let roots = || (0..ns).filter(|&s| parent[s] == NONE);
    let leaf_cap = roots().map(|s| work[s]).sum::<u64>() / leaf_subtrees.max(1) as u64;
    let mut heap: std::collections::BinaryHeap<(u64, std::cmp::Reverse<u32>)> = roots()
        .map(|s| (work[s], std::cmp::Reverse(s as u32)))
        .collect();
    let mut is_top = vec![false; ns];
    let mut leaf_roots: Vec<u32> = Vec::new();
    while let Some((wk, std::cmp::Reverse(s))) = heap.pop() {
        if wk <= leaf_cap || children[s as usize].is_empty() {
            leaf_roots.push(s);
        } else {
            is_top[s as usize] = true;
            for &c in &children[s as usize] {
                heap.push((work[c as usize], std::cmp::Reverse(c)));
            }
        }
    }
    leaf_roots.sort_unstable();
    (is_top, leaf_roots)
}

pub(crate) struct LayoutSize {
    /// Bytes of the schedule (all but the values and the reciprocal diagonal).
    pub held: u64,
    /// Peak bytes of the build's temporaries.
    pub build: u64,
    /// Entries of `T` per right-hand side a solve allocates besides the
    /// right-hand side itself: the accumulators, the apex buffers and each
    /// thread's node scratch.
    pub solve_per_rhs: u64,
}

pub(crate) fn layout_size<'r>(
    parent: &[usize],
    w: &[usize],
    off_rows: &dyn Fn(usize) -> &'r [u32],
    cfg: &crate::SolveSettings,
    threads: usize,
) -> LayoutSize {
    use crate::memory::pushed;
    let ns = parent.len();
    let n: usize = w.iter().sum();
    let m: Vec<usize> = (0..ns).map(|s| off_rows(s).len()).collect();
    let rows: usize = m.iter().sum();
    let parent: Vec<u32> = (0..ns)
        .map(|s| match parent[s] {
            p if p != usize::MAX && p < ns && p > s => p as u32,
            _ => NONE,
        })
        .collect();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); ns];
    for s in 0..ns {
        if parent[s] != NONE {
            children[parent[s] as usize].push(s as u32);
        }
    }
    let (is_top, leaf_roots) = cut(&parent, &children, cfg.leaf_subtrees, |s| {
        (w[s] * (w[s] + m[s])) as u64
    });
    let mut sn_of = vec![0u32; n];
    let mut c = 0;
    for s in 0..ns {
        sn_of[c..c + w[s]].fill(s as u32);
        c += w[s];
    }
    let mut subtree_of = vec![NONE; ns];
    let (mut ext, mut ext_runs, mut sizes) = (Vec::new(), Vec::new(), Vec::new());
    let mut list: Vec<u32> = Vec::new();
    for (t, &r) in leaf_roots.iter().enumerate() {
        let mut nodes = vec![r];
        let mut stack = vec![r];
        while let Some(s) = stack.pop() {
            for &k in &children[s as usize] {
                nodes.push(k);
                stack.push(k);
            }
        }
        for &s in &nodes {
            subtree_of[s as usize] = t as u32;
        }
        list.clear();
        for &s in &nodes {
            list.extend(
                off_rows(s as usize)
                    .iter()
                    .filter(|&&r| subtree_of[sn_of[r as usize] as usize] != t as u32),
            );
        }
        list.sort_unstable();
        list.dedup();
        ext.push(list.len());
        ext_runs.push(runs(&list, &sn_of));
        sizes.push(nodes.len());
    }
    let mut depth = vec![0usize; ns];
    let mut level_len: Vec<usize> = Vec::new();
    for s in (0..ns).rev().filter(|&s| is_top[s]) {
        let p = parent[s];
        depth[s] = if p == NONE { 0 } else { depth[p as usize] + 1 };
        if level_len.len() <= depth[s] {
            level_len.resize(depth[s] + 1, 0);
        }
        level_len[depth[s]] += 1;
    }
    let top: Vec<usize> = (0..ns).filter(|&s| is_top[s]).collect();
    let (idx, word, vec) = (IDX32_BYTES, WORD_BYTES, VEC_HEADER_BYTES);
    let sn_col_bytes = idx * (ns + 1);
    let row_ptr_bytes = word * (ns + 1);
    let val_rng_bytes = 2 * word * ns;
    let rows_and_ext_slot_bytes = word * rows;
    let top_index_bytes = idx * ns;
    let fixed =
        sn_col_bytes + row_ptr_bytes + val_rng_bytes + rows_and_ext_slot_bytes + top_index_bytes;
    let subtrees = leaf_roots.len() * std::mem::size_of::<Subtree>()
        + (0..leaf_roots.len())
            .map(|t| idx * pushed(sizes[t]) + idx * ext[t])
            .sum::<usize>();
    let levels = 2 * level_len.len() * vec
        + (level_len.iter())
            .map(|&l| idx * pushed(l) + word * pushed(l + 1))
            .sum::<usize>();
    let roots = top.iter().filter(|&&t| parent[t] == NONE).count();
    let pulls = ext_runs.iter().sum::<usize>()
        + top
            .iter()
            .map(|&t| runs(off_rows(t), &sn_of))
            .sum::<usize>();
    let top_ptr_bytes = word * (top.len() + 1);
    let (top_kids_ptr, top_acc_off, pull_ptr) = (top_ptr_bytes, top_ptr_bytes, top_ptr_bytes);
    let schedule = idx * (top.len() + roots + top.len() + top.len() - roots)
        + top_kids_ptr
        + top_acc_off
        + pull_ptr
        + std::mem::size_of::<(u32, u32, u32)>() * pulls;
    let build = idx * n + ns * SOLVE_BUILD_NODE_BYTES;
    let max_ld = (0..ns).map(|s| w[s] + m[s]).max().unwrap_or(0);
    let solve_per_rhs = ext.iter().sum::<usize>()
        + top.iter().map(|&t| m[t]).sum::<usize>()
        + threads.max(1) * 2 * max_ld;
    LayoutSize {
        held: (fixed + subtrees + levels + schedule) as u64,
        build: build as u64,
        solve_per_rhs: solve_per_rhs as u64,
    }
}

fn runs(list: &[u32], sn_of: &[u32]) -> usize {
    let mut k = 0;
    let mut last = NONE;
    for &r in list {
        let a = sn_of[r as usize];
        if a != last {
            k += 1;
            last = a;
        }
    }
    k
}

impl<T: Scalar> SolvePlan<T> {
    pub(crate) fn diag_inv(&self) -> &[T] {
        &self.diag_inv
    }

    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::{nested_bytes, vec_bytes};
        vec_bytes(&self.sn_col)
            + vec_bytes(&self.row_ptr)
            + vec_bytes(&self.rows)
            + vec_bytes(&self.ext_slot)
            + vec_bytes(&self.val_rng)
            + vec_bytes(&self.vals)
            + vec_bytes(&self.spare_rows)
            + vec_bytes(&self.diag_inv)
            + vec_bytes(&self.subtrees)
            + self
                .subtrees
                .iter()
                .map(|t| vec_bytes(&t.nodes) + vec_bytes(&t.ext))
                .sum::<u64>()
            + nested_bytes(&self.top_levels)
            + vec_bytes(&self.top_index)
            + nested_bytes(&self.top_acc_ptr)
            + vec_bytes(&self.top)
            + vec_bytes(&self.top_roots)
            + vec_bytes(&self.top_parent)
            + vec_bytes(&self.top_kids_ptr)
            + vec_bytes(&self.top_kids)
            + vec_bytes(&self.top_acc_off)
            + vec_bytes(&self.pull_ptr)
            + vec_bytes(&self.pull)
    }

    /// Build the schedule over a factor in panel form, taking the panels as
    /// the plan's storage (no copy). `supernode_parent` is the supernode
    /// tree of the analysis (`usize::MAX` for a root); an empty or
    /// mismatched one is replaced by the parent implied by the first
    /// off-block row of every supernode.
    pub fn from_panels(
        factor: PanelFactor<T>,
        supernode_parent: &[usize],
        unit: bool,
        cfg: crate::SolveSettings,
    ) -> Self {
        let n = factor.n;
        let ns = factor.n_supernodes();
        let sn_col: Vec<u32> = factor.sn_col.clone();
        let mut sn_of = vec![0u32; n];
        for s in 0..ns {
            for c in sn_col[s] as usize..sn_col[s + 1] as usize {
                sn_of[c] = s as u32;
            }
        }
        let diag_inv: Vec<T> = if unit {
            Vec::new()
        } else {
            let mut d = Vec::with_capacity(n);
            for s in 0..ns {
                let lay = factor.layout(s);
                for k in 0..lay.w {
                    d.push(factor.panel(s)[lay.at(k, k)].recip());
                }
            }
            d
        };
        let val_rng: Vec<(usize, usize)> = (0..ns)
            .map(|s| {
                (
                    factor.val_off[s],
                    factor.val_off[s] + factor.layout(s).len(),
                )
            })
            .collect();
        let PanelFactor {
            rows,
            row_ptr,
            mut vals,
            ..
        } = factor;
        debug_assert!(
            (0..ns).all(|s| rows[row_ptr[s]..row_ptr[s + 1]]
                .windows(2)
                .all(|p| p[0] < p[1])),
            "off-block rows ascending"
        );
        super::panel::release_room(&mut vals);
        let known = supernode_parent.len() == ns;

        let mut parent = vec![NONE; ns];
        let mut children: Vec<Vec<u32>> = vec![Vec::new(); ns];
        if known {
            for s in 0..ns {
                let p = supernode_parent[s];
                if p != usize::MAX && p < ns && p > s {
                    parent[s] = p as u32;
                    children[p].push(s as u32);
                }
            }
        } else {
            for s in 0..ns {
                if row_ptr[s + 1] > row_ptr[s] {
                    let p = sn_of[rows[row_ptr[s]] as usize];
                    parent[s] = p;
                    children[p as usize].push(s as u32);
                }
            }
        }
        let width = |s: usize| (sn_col[s + 1] - sn_col[s]) as usize;
        let (is_top, leaf_roots) = cut(&parent, &children, cfg.leaf_subtrees, |s| {
            let w = width(s) as u64;
            w * (w + (row_ptr[s + 1] - row_ptr[s]) as u64)
        });

        let mut subtree_of = vec![NONE; ns];
        let mut subtrees: Vec<Subtree> = Vec::with_capacity(leaf_roots.len());
        for (t, &r) in leaf_roots.iter().enumerate() {
            let mut nodes = vec![r];
            let mut stack = vec![r];
            while let Some(s) = stack.pop() {
                for &c in &children[s as usize] {
                    nodes.push(c);
                    stack.push(c);
                }
            }
            nodes.sort_unstable();
            for &s in &nodes {
                subtree_of[s as usize] = t as u32;
            }
            subtrees.push(Subtree {
                nodes,
                ext: Vec::new(),
            });
        }
        for (t, st) in subtrees.iter_mut().enumerate() {
            let mut ext: Vec<u32> = st
                .nodes
                .iter()
                .flat_map(|&s| {
                    rows[row_ptr[s as usize]..row_ptr[s as usize + 1]]
                        .iter()
                        .copied()
                })
                .filter(|&r| subtree_of[sn_of[r as usize] as usize] != t as u32)
                .collect();
            ext.sort_unstable();
            ext.dedup();
            ext.shrink_to_fit();
            st.ext = ext;
        }
        let top: Vec<u32> = (0..ns as u32).filter(|&s| is_top[s as usize]).collect();
        let mut depth = vec![0u32; ns];
        for &t in top.iter().rev() {
            let p = parent[t as usize];
            debug_assert!(p == NONE || p > t);
            depth[t as usize] = if p == NONE { 0 } else { depth[p as usize] + 1 };
        }
        let max_depth = top.iter().map(|&t| depth[t as usize]).max().unwrap_or(0) as usize;
        let mut top_levels: Vec<Vec<u32>> =
            vec![Vec::new(); if top.is_empty() { 0 } else { max_depth + 1 }];
        for &t in &top {
            top_levels[depth[t as usize] as usize].push(t);
        }

        let mut ext_slot = vec![NONE; rows.len()];
        let mut on_path = vec![false; ns];
        let mut degenerate = false;
        let mark = |from: u32, on: bool, on_path: &mut [bool]| {
            let mut a = from;
            while a != NONE {
                on_path[a as usize] = on;
                a = parent[a as usize];
            }
        };
        for st in &subtrees {
            let Some(&root) = st.nodes.last() else {
                continue;
            };
            mark(parent[root as usize], true, &mut on_path);
            let mut written = vec![false; st.ext.len()];
            for &s in &st.nodes {
                for e in row_ptr[s as usize]..row_ptr[s as usize + 1] {
                    let r = rows[e];
                    let a = sn_of[r as usize];
                    if subtree_of[a as usize] != subtree_of[s as usize] {
                        if on_path[a as usize] {
                            let slot = st.ext.partition_point(|&x| x < r);
                            debug_assert_eq!(st.ext.get(slot), Some(&r));
                            let first = !std::mem::replace(&mut written[slot], true);
                            ext_slot[e] = slot as u32 | if first { FIRST_WRITE } else { 0 };
                        } else {
                            degenerate = true;
                        }
                    }
                }
            }
            mark(parent[root as usize], false, &mut on_path);
        }
        for &t in &top {
            mark(parent[t as usize], true, &mut on_path);
            let r0 = row_ptr[t as usize];
            for e in r0..row_ptr[t as usize + 1] {
                if on_path[sn_of[rows[e] as usize] as usize] {
                    ext_slot[e] = (e - r0) as u32 | FIRST_WRITE;
                } else {
                    degenerate = true;
                }
            }
            mark(parent[t as usize], false, &mut on_path);
        }
        let chain = degenerate;
        let (subtrees, top, top_levels, ext_slot) = if degenerate {
            rsprobe::debug!("solve layout: structure not tree-nested, sequential sweeps");
            let all: Vec<u32> = (0..ns as u32).collect();
            let levels: Vec<Vec<u32>> = all.iter().rev().map(|&s| vec![s]).collect();
            (Vec::new(), all, levels, vec![NONE; rows.len()])
        } else {
            (subtrees, top, top_levels, ext_slot)
        };
        let top_index = {
            let mut ix = vec![NONE; ns];
            for (i, &t) in top.iter().enumerate() {
                ix[t as usize] = i as u32;
            }
            ix
        };
        let top_acc_ptr: Vec<Vec<usize>> = top_levels
            .iter()
            .map(|level| {
                std::iter::once(0)
                    .chain(level.iter().scan(0usize, |o, &s| {
                        *o += row_ptr[s as usize + 1] - row_ptr[s as usize];
                        Some(*o)
                    }))
                    .collect()
            })
            .collect();
        let top_parent: Vec<u32> = top
            .iter()
            .map(|&t| match parent[t as usize] {
                NONE => NONE,
                p => top_index[p as usize],
            })
            .collect();
        let mut top_kids_ptr = vec![0usize; top.len() + 1];
        for &p in &top_parent {
            if p != NONE {
                top_kids_ptr[p as usize + 1] += 1;
            }
        }
        for i in 0..top.len() {
            top_kids_ptr[i + 1] += top_kids_ptr[i];
        }
        let mut top_kids = vec![0u32; top_kids_ptr[top.len()]];
        let mut fill = top_kids_ptr.clone();
        for (i, &p) in top_parent.iter().enumerate() {
            if p != NONE {
                top_kids[fill[p as usize]] = i as u32;
                fill[p as usize] += 1;
            }
        }
        let mut top_roots: Vec<u32> = (0..top.len() as u32)
            .filter(|&i| top_parent[i as usize] == NONE)
            .collect();
        top_roots.shrink_to_fit();
        let mut top_acc_off = vec![0usize; top.len() + 1];
        for (i, &t) in top.iter().enumerate() {
            top_acc_off[i + 1] = top_acc_off[i] + row_ptr[t as usize + 1] - row_ptr[t as usize];
        }
        let mut pulls: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); top.len()];
        if !chain {
            let runs = |src: u32, list: &[u32], pulls: &mut Vec<Vec<(u32, u32, u32)>>| {
                debug_assert!(list.windows(2).all(|r| r[0] < r[1]));
                let mut lo = 0;
                while lo < list.len() {
                    let a = sn_of[list[lo] as usize];
                    let hi = lo + list[lo..].partition_point(|&r| sn_of[r as usize] == a);
                    pulls[top_index[a as usize] as usize].push((src, lo as u32, hi as u32));
                    lo = hi;
                }
            };
            for (t, st) in subtrees.iter().enumerate() {
                runs(t as u32, &st.ext, &mut pulls);
            }
            for level in top_levels.iter().rev() {
                for &d in level {
                    let src = (subtrees.len() + top_index[d as usize] as usize) as u32;
                    runs(
                        src,
                        &rows[row_ptr[d as usize]..row_ptr[d as usize + 1]],
                        &mut pulls,
                    );
                }
            }
        }
        let mut pull_ptr = vec![0usize; top.len() + 1];
        for i in 0..top.len() {
            pull_ptr[i + 1] = pull_ptr[i] + pulls[i].len();
        }
        let mut pull: Vec<(u32, u32, u32)> = Vec::with_capacity(pull_ptr[top.len()]);
        pull.extend(pulls.into_iter().flatten());
        let entries = |s: u32| val_rng[s as usize].1 - val_rng[s as usize].0;
        let subtree_work = (subtrees.iter())
            .map(|st| st.nodes.iter().map(|&s| entries(s)).sum::<usize>())
            .sum();
        let plan = Self {
            n,
            sn_col,
            row_ptr,
            rows,
            ext_slot,
            val_rng,
            vals,
            diag_inv,
            subtrees,
            top_levels,
            top_index,
            top_acc_ptr,
            top: {
                let mut top = top;
                top.shrink_to_fit();
                top
            },
            top_roots,
            top_parent: {
                let mut p = top_parent;
                p.shrink_to_fit();
                p
            },
            top_kids_ptr,
            top_kids,
            top_acc_off,
            pull_ptr,
            pull,
            chain,
            subtree_work,
            spare_rows: Vec::new(),
            cfg,
        };
        rsprobe::debug!("{}", plan.describe());
        plan
    }

    pub(crate) fn take_storage(&mut self) -> PanelStorage<T> {
        PanelStorage {
            vals: std::mem::take(&mut self.vals),
            rows: std::mem::take(&mut self.spare_rows),
        }
    }

    /// Take `factor`, the next factorization of this plan's analysis, as
    /// [`from_panels`](Self::from_panels) would. When its supernodes and rows
    /// are the plan's and `cfg` is unchanged, only the panels and the
    /// reciprocal diagonal change and the schedule stays; a factorization
    /// whose pivoting or cancellation changed the rows gets a new plan.
    pub fn refill(
        &mut self,
        factor: PanelFactor<T>,
        supernode_parent: &[usize],
        unit: bool,
        cfg: crate::SolveSettings,
    ) {
        let cfg_now = cfg;
        let ns = factor.n_supernodes();
        let same = cfg_now == self.cfg
            && factor.n == self.n
            && unit == self.diag_inv.is_empty()
            && factor.sn_col == self.sn_col
            && factor.row_ptr == self.row_ptr
            && factor.rows == self.rows;
        if !same {
            rsprobe::add!("layouts rebuilt", 1);
            *self = Self::from_panels(factor, supernode_parent, unit, cfg);
            return;
        }
        rsprobe::add!("layouts kept", 1);
        if !unit {
            for s in 0..ns {
                let (c0, lay) = (self.sn_col[s] as usize, factor.layout(s));
                for k in 0..lay.w {
                    self.diag_inv[c0 + k] = factor.panel(s)[lay.at(k, k)].recip();
                }
            }
        }
        for s in 0..ns {
            self.val_rng[s] = (
                factor.val_off[s],
                factor.val_off[s] + factor.layout(s).len(),
            );
        }
        let PanelFactor { vals, rows, .. } = factor;
        self.vals = vals;
        self.spare_rows = rows;
    }

    fn describe(&self) -> String {
        let entries = |s: u32| self.val_rng[s as usize].1 - self.val_rng[s as usize].0;
        let sub: Vec<usize> = (self.subtrees.iter())
            .map(|st| st.nodes.iter().map(|&s| entries(s)).sum())
            .collect();
        let sub_max = sub.iter().copied().max().unwrap_or(0);
        let top_work = self.nodes_work(&self.top);
        let path = self.critical_path();
        let critical = self.nodes_work(&path.iter().map(|&t| self.top[t]).collect::<Vec<_>>());
        let (mut tri, mut off, mut wide) = (0usize, 0usize, Vec::new());
        for &t in &path {
            let (_, w, _, m, _, _) = self.node(self.top[t]);
            tri += w * (w + 1) / 2;
            off += w * m;
            if w >= PANEL_BLOCK {
                wide.push((w, m));
            }
        }
        let widths: Vec<usize> = self.top_levels.iter().map(Vec::len).collect();
        let slots: usize = self.subtrees.iter().map(|st| st.ext.len()).sum();
        let big = self.top.iter().copied().max_by_key(|&s| entries(s));
        let big = big.map_or(String::new(), |s| {
            let (_, w, _, m, _, _) = self.node(s);
            format!(" largest ancestor {} entries (w={w} m={m})", entries(s))
        });
        format!(
            "solve layout: n={} chain={} entries={} subtrees={} entries={} (max {} mean {:.0}) \
             ancestors={} entries={} levels={} widths={widths:?} critical path={} \
             (triangles {tri}, rows below {off}, wide nodes (w, m) {wide:?}) slots={slots}{big}",
            self.n,
            self.chain,
            self.subtree_work + top_work,
            self.subtrees.len(),
            self.subtree_work,
            sub_max,
            self.subtree_work as f64 / sub.len().max(1) as f64,
            self.top.len(),
            top_work,
            self.top_levels.len(),
            critical,
        )
    }

    fn critical_path(&self) -> Vec<usize> {
        let entries = |s: u32| self.val_rng[s as usize].1 - self.val_rng[s as usize].0;
        let mut path = vec![0usize; self.top.len()];
        let mut next = vec![NONE; self.top.len()];
        for ti in 0..self.top.len() {
            let kids = &self.top_kids[self.top_kids_ptr[ti]..self.top_kids_ptr[ti + 1]];
            debug_assert!(kids.iter().all(|&k| (k as usize) < ti));
            if let Some(&k) = kids.iter().max_by_key(|&&k| path[k as usize]) {
                next[ti] = k;
                path[ti] = path[k as usize];
            }
            path[ti] += entries(self.top[ti]);
        }
        let mut chain = Vec::new();
        let mut at = (0..self.top.len() as u32).max_by_key(|&t| path[t as usize]);
        while let Some(t) = at.filter(|&t| t != NONE) {
            chain.push(t as usize);
            at = Some(next[t as usize]);
        }
        chain
    }

    // https://github.com/milanofthe/rslab/issues/289
    fn node_chunks(&self, par: bool, rows: usize, depth: usize, nr: usize) -> usize {
        let whole = usize::from(rows > 0);
        if !par {
            return whole;
        }
        let by_work = (rows * depth * nr).div_ceil(self.cfg.gemm_task.max(1));
        by_work.min(rows / chunk_align(nr)).max(whole)
    }

    #[inline]
    fn node(&self, s: u32) -> (usize, usize, usize, usize, usize, &[T]) {
        let s = s as usize;
        let c0 = self.sn_col[s] as usize;
        let w = self.sn_col[s + 1] as usize - c0;
        let r0 = self.row_ptr[s];
        let m = self.row_ptr[s + 1] - r0;
        let (v0, v1) = self.val_rng[s];
        let panel = &self.vals[v0..v1];
        (c0, w, r0, m, w + m, panel)
    }

    // https://github.com/milanofthe/rslab/issues/289
    fn in_pool<R: Send>(par: bool, threads: Threads, f: impl FnOnce() -> R + Send) -> R {
        if !par {
            return threads.here(f);
        }
        match threads {
            _ if crate::parallel::is_lent() => crate::parallel::enter(f64::INFINITY, f),
            Threads::Ambient => crate::parallel::enter(f64::INFINITY, f),
            budget => budget.run(0, |cap| cap, f),
        }
    }

    fn each<A: Sync, S: Send>(
        par: bool,
        items: &[A],
        scratch: &mut [S],
        f: impl Fn(usize, &A, &mut S) + Sync + Send,
    ) {
        if par {
            crate::parallel::zip_mut(items, scratch, crate::parallel::Work::ns(f64::INFINITY), f);
        } else {
            for (i, (a, s)) in items.iter().zip(scratch.iter_mut()).enumerate() {
                f(i, a, s);
            }
        }
    }

    fn nodes_work(&self, nodes: &[u32]) -> usize {
        (nodes.iter())
            .map(|&s| self.val_rng[s as usize].1 - self.val_rng[s as usize].0)
            .sum()
    }

    fn fit<'w>(
        &self,
        work: &'w mut PlanWork<T>,
        nr: usize,
        threads: Threads,
    ) -> &'w mut PlanWork<T> {
        let widest = self.top_levels.iter().map(Vec::len).max().unwrap_or(0);
        work.fit(self.subtrees.len(), widest);
        let budget = threads.resolve(|cap| cap);
        let ops = self.vals.len().saturating_mul(nr.min(SOLVE_RHS_TILE));
        work.parallel =
            budget > 1 && ops >= self.cfg.par_min_work.unwrap_or(SOLVE_PAR_MIN_ENTRY_RHS);
        rsprobe::debug!(
            "sweep nr={nr} budget={budget} -> {}",
            if work.parallel { "pool" } else { "here" }
        );
        work
    }

    pub(crate) fn sweep<R: Send>(
        &self,
        work: &mut PlanWork<T>,
        nr: usize,
        threads: Threads,
        sweep: impl FnOnce(&mut PlanWork<T>) -> R + Send,
    ) -> R {
        let work = self.fit(work, nr, threads);
        Self::in_pool(work.parallel, threads, || sweep(work))
    }

    fn flows(&self, work: &PlanWork<T>) -> bool {
        work.parallel && !self.chain && !self.top.is_empty()
    }

    fn top_forward(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let par = work.parallel;
        let nt = crate::parallel::threads();
        let shared = Shared(y.as_mut_ptr(), y.len());
        for (li, level) in self.top_levels.iter().enumerate().rev() {
            let node_par = !par || level.len() >= nt || nt == 1;
            let ptr = &self.top_acc_ptr[li];
            grow(&mut work.top_acc, ptr[level.len()] * nr);
            let accs = Shared(work.top_acc.as_mut_ptr(), work.top_acc.len());
            let sweep = |i: usize, s: u32, y: &mut [T], par: bool, sc: &mut Scratch<T>| {
                // SAFETY: node `i` owns its range of the level's block.
                let acc = unsafe { &mut accs.slice()[ptr[i] * nr..ptr[i + 1] * nr] };
                let _s = rsprobe::scope!(TRACE, "node");
                self.fwd_node_gemm(s, nr, y, acc, par, &mut sc.t);
            };
            if node_par {
                Self::each(par, level, &mut work.level[..level.len()], |i, &s, sc| {
                    // SAFETY: see `Shared`; the node writes only its own
                    // columns, its off-block rows go to its accumulator. A
                    // large node keeps its row tasks inside the level's
                    // (the same arithmetic either way): one large node per
                    // level otherwise ran on one thread while the others
                    // finished their small ones.
                    let yv = unsafe { shared.slice() };
                    sweep(i, s, yv, par, sc);
                });
            } else {
                for (i, &s) in level.iter().enumerate() {
                    sweep(i, s, y, true, &mut work.apex);
                }
            }
            for (i, &s) in level.iter().enumerate() {
                let own = &self.rows[self.row_ptr[s as usize]..self.row_ptr[s as usize + 1]];
                let acc = &work.top_acc[ptr[i] * nr..ptr[i + 1] * nr];
                for (k, &c) in own.iter().enumerate() {
                    sub_assign(
                        &mut y[c as usize * nr..(c as usize + 1) * nr],
                        &acc[k * nr..(k + 1) * nr],
                    );
                }
            }
        }
    }

    fn top_backward(&self, nr: usize, x: &mut [T], work: &mut PlanWork<T>) {
        let par = work.parallel;
        let nt = crate::parallel::threads();
        let shared = Shared(x.as_mut_ptr(), x.len());
        for level in &self.top_levels {
            let node_par = !par || level.len() >= nt || nt == 1;
            let sweep = |s: u32, x: &mut [T], par: bool, sc: &mut Scratch<T>| {
                let _s = rsprobe::scope!(TRACE, "node");
                self.bwd_node_gemm(s, nr, x, par, &mut sc.g);
            };
            if node_par {
                Self::each(par, level, &mut work.level[..level.len()], |_, &s, sc| {
                    // SAFETY: see `Shared`; the node writes only its own
                    // columns and reads final ancestor columns. Its row
                    // tasks nest as in the forward sweep.
                    let xv = unsafe { shared.slice() };
                    sweep(s, xv, par, sc);
                });
            } else {
                for &s in level {
                    sweep(s, x, true, &mut work.apex);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fwd_node_gemm(
        &self,
        s: u32,
        nr: usize,
        y: &mut [T],
        acc: &mut [T],
        par: bool,
        t: &mut Vec<T>,
    ) {
        let (c0, w, r0, m, _, _) = self.node(s);
        self.fwd_node_core(s, nr, &mut y[c0 * nr..(c0 + w) * nr], par, t);
        if m == 0 {
            return;
        }
        let rows = &self.rows[r0..r0 + m];
        let slots = &self.ext_slot[r0..r0 + m];
        let _s = rsprobe::scope!(TRACE, "scatter");
        for (i, ti) in t[..m * nr].chunks_exact(nr).enumerate() {
            let sl = slots[i];
            if sl == NONE {
                let r = rows[i] as usize;
                sub_assign(&mut y[r * nr..(r + 1) * nr], ti);
            } else {
                let a = (sl & !FIRST_WRITE) as usize;
                let to = &mut acc[a * nr..(a + 1) * nr];
                if sl & FIRST_WRITE != 0 {
                    to.copy_from_slice(ti);
                } else {
                    add_assign(to, ti);
                }
            }
        }
    }

    pub(super) fn fwd_node_core(&self, s: u32, nr: usize, yb: &mut [T], par: bool, t: &mut Vec<T>) {
        let (c0, w, _, m, _, panel) = self.node(s);
        let lay = Layout::new(w, m);
        let dinv = self.diag_inv.get(c0..c0 + w).unwrap_or(&[]);
        let pp = Shared(panel.as_ptr() as *mut T, panel.len());
        for (jb, je, off, ld) in lay.blocks() {
            let d = dinv.get(jb..je).unwrap_or(&[]);
            {
                let _s = rsprobe::scope!(TRACE, "triangle");
                tri_rows(
                    &mut yb[jb * nr..je * nr],
                    &panel[off..],
                    d,
                    ld,
                    nr,
                    0,
                    je - jb,
                );
            }
            let k = je - jb;
            let own = self.node_chunks(par, w - je, k, nr);
            let yp = Shared(yb.as_mut_ptr(), yb.len());
            run_chunks(own, |i| {
                let r = chunk_rows(i, own, w - je, nr);
                // SAFETY: a chunk writes its own rows of `yb` (below `je`); the
                // block's solved rows `jb..je` and the panel are only read.
                unsafe {
                    fwd_own(
                        yp.ptr(),
                        pp.ptr().add(off),
                        ld,
                        jb,
                        je,
                        je + r.start..je + r.end,
                        nr,
                    )
                }
            });
        }
        if m == 0 {
            return;
        }
        grow(t, m * nr);
        let below = self.node_chunks(par, m, w, nr);
        let (yp, tp) = (
            Shared(yb.as_mut_ptr(), yb.len()),
            Shared(t.as_mut_ptr(), t.len()),
        );
        run_chunks(below, |i| {
            let r = chunk_rows(i, below, m, nr);
            for (b, (jb, je, _, _)) in lay.blocks().enumerate() {
                // SAFETY: a chunk writes its own rows of `t`; the solved rows of `yb`
                // and the panel are only read.
                unsafe {
                    fwd_below(
                        tp.ptr(),
                        pp.ptr().add(lay.rect()),
                        m,
                        jb,
                        je,
                        yp.ptr(),
                        r.clone(),
                        nr,
                        b > 0,
                    )
                }
            }
        });
    }

    fn bwd_node_gemm(&self, s: u32, nr: usize, x: &mut [T], par: bool, g: &mut Vec<T>) {
        let (c0, w, r0, m, _, panel) = self.node(s);
        let lay = Layout::new(w, m);
        let dinv = self.diag_inv.get(c0..c0 + w).unwrap_or(&[]);
        if m > 0 {
            let _s = rsprobe::scope!(TRACE, "gather");
            g.clear();
            for &r in &self.rows[r0..r0 + m] {
                g.extend_from_slice(&x[r as usize * nr..(r as usize + 1) * nr]);
            }
        }
        let xb = &mut x[c0 * nr..(c0 + w) * nr];
        let (pp, gp, xp) = (
            Shared(panel.as_ptr() as *mut T, panel.len()),
            Shared(g.as_mut_ptr(), g.len()),
            Shared(xb.as_mut_ptr(), xb.len()),
        );
        if m > 0 {
            let nb = w.div_ceil(PANEL_BLOCK);
            let last_rows = w - (nb - 1) * PANEL_BLOCK;
            let (full, last) = (
                self.node_chunks(par, PANEL_BLOCK, m, nr),
                self.node_chunks(par, last_rows, m, nr),
            );
            run_chunks((nb - 1) * full + last, |i| {
                let (b, q, count) = match i.checked_sub((nb - 1) * full) {
                    None => (i / full, i % full, full),
                    Some(q) => (nb - 1, q, last),
                };
                let Some((jb, je, _, _)) = lay.blocks().nth(b) else {
                    return;
                };
                let r = chunk_rows(q, count, je - jb, nr);
                // SAFETY: the chunk writes its own rows of the node; the panel and
                // the gathered rows are only read.
                unsafe {
                    bwd_off(
                        xp.ptr(),
                        pp.ptr().add(lay.rect()),
                        m,
                        gp.ptr(),
                        jb + r.start..jb + r.end,
                        nr,
                    )
                }
            });
        }
        for (jb, je, off, ld) in lay.blocks().rev() {
            if je < w {
                let count = self.node_chunks(par, je - jb, w - je, nr);
                run_chunks(count, |q| {
                    let r = chunk_rows(q, count, je - jb, nr);
                    // SAFETY: the chunk writes its own rows of `jb..je`, rows `je..w`
                    // are read: disjoint.
                    unsafe {
                        bwd_in(
                            xp.ptr(),
                            pp.ptr().add(off),
                            ld,
                            w,
                            jb,
                            je,
                            jb + r.start..jb + r.end,
                            nr,
                        )
                    }
                });
            }
            let _s = rsprobe::scope!(TRACE, "triangle");
            tri_up(xb, &panel[off..], dinv, ld, nr, jb, je);
        }
    }

    pub(crate) fn solve_block_inner(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        nr: usize,
        work: &mut PlanWork<T>,
    ) -> Result<(), RslabError> {
        debug_assert_eq!(y.len(), self.n * nr);
        self.forward_block(nr, y, work);
        {
            let _s = rsprobe::scope!(DEBUG, "diagonal");
            solve_diagonal(f, y, nr, 0..f.n)?;
        }
        self.backward_block(nr, y, work);
        Ok(())
    }

    pub(crate) fn forward_block(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let _s = rsprobe::scope!(DEBUG, "forward");
        let shared = Shared(y.as_mut_ptr(), y.len());
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        {
            let _s = rsprobe::scope!(DEBUG, "subtrees");
            Self::each(par, &self.subtrees, scratch, |_, st, sc| {
                grow(&mut sc.acc, st.ext.len() * nr);
                // SAFETY: see `Shared`.
                let yv = unsafe { shared.slice() };
                for &s in &st.nodes {
                    self.fwd_node_gemm(s, nr, yv, &mut sc.acc, false, &mut sc.t);
                }
            });
        }
        if self.flows(work) {
            let _s = rsprobe::scope!(DEBUG, "ancestors");
            self.forward_dag(nr, y, work);
        } else {
            {
                let _s = rsprobe::scope!(DEBUG, "reduce");
                for (st, sc) in self.subtrees.iter().zip(&work.subtrees) {
                    for (i, &c) in st.ext.iter().enumerate() {
                        let yr = &mut y[c as usize * nr..(c as usize + 1) * nr];
                        sub_assign(yr, &sc.acc[i * nr..(i + 1) * nr]);
                    }
                }
            }
            let _s = rsprobe::scope!(DEBUG, "ancestors");
            self.top_forward(nr, y, work);
        }
    }

    pub(crate) fn backward_block(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let _s = rsprobe::scope!(DEBUG, "backward");
        {
            let _s = rsprobe::scope!(DEBUG, "ancestors");
            if self.flows(work) {
                self.backward_dag(nr, y, work);
            } else {
                self.top_backward(nr, y, work);
            }
        }
        let _s = rsprobe::scope!(DEBUG, "subtrees");
        let shared = Shared(y.as_mut_ptr(), y.len());
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        Self::each(par, &self.subtrees, scratch, |_, st, sc| {
            // SAFETY: see `Shared`.
            let xv = unsafe { shared.slice() };
            for &s in st.nodes.iter().rev() {
                self.bwd_node_gemm(s, nr, xv, false, &mut sc.g);
            }
        });
    }
}

#[inline(always)]
fn sub_assign<T: Scalar>(y: &mut [T], x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi - xi;
    }
}

#[inline(always)]
fn add_assign<T: Scalar>(y: &mut [T], x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi + xi;
    }
}

// https://github.com/milanofthe/rslab/issues/289
fn tri_rows<T: Scalar>(
    v: &mut [T],
    panel: &[T],
    diag_inv: &[T],
    ld: usize,
    nr: usize,
    jb: usize,
    je: usize,
) {
    let mut c0 = 0;
    while c0 < nr {
        let w = (nr - c0).min(SOLVE_RHS_TILE);
        match w {
            4 => tri_tile::<T, 4>(v, panel, diag_inv, ld, nr, c0, jb, je),
            3 => tri_tile::<T, 3>(v, panel, diag_inv, ld, nr, c0, jb, je),
            2 => tri_tile::<T, 2>(v, panel, diag_inv, ld, nr, c0, jb, je),
            _ => tri_tile::<T, 1>(v, panel, diag_inv, ld, nr, c0, jb, je),
        }
        c0 += w;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tri_tile<T: Scalar, const W: usize>(
    v: &mut [T],
    panel: &[T],
    diag_inv: &[T],
    ld: usize,
    nr: usize,
    c0: usize,
    jb: usize,
    je: usize,
) {
    let mut i = jb;
    while i < je {
        let rn = TILE_ROWS.min(je - i);
        let mut acc = [[T::zero(); W]; TILE_ROWS];
        for r in 0..rn {
            acc[r].copy_from_slice(&v[(i + r) * nr + c0..(i + r) * nr + c0 + W]);
        }
        for k in jb..i {
            let vk = &v[k * nr + c0..k * nr + c0 + W];
            let col = &panel[k * ld + i..k * ld + i + rn];
            for (ar, &l) in acc.iter_mut().zip(col) {
                for (a, &x) in ar.iter_mut().zip(vk) {
                    *a = *a - l * x;
                }
            }
        }
        for r in 0..rn {
            for k in i..i + r {
                let l = panel[k * ld + i + r];
                let xk = acc[k - i];
                for (a, x) in acc[r].iter_mut().zip(xk) {
                    *a = *a - l * x;
                }
            }
            if let Some(&d) = diag_inv.get(i + r) {
                for a in acc[r].iter_mut() {
                    *a = *a * d;
                }
            }
        }
        for r in 0..rn {
            v[(i + r) * nr + c0..(i + r) * nr + c0 + W].copy_from_slice(&acc[r]);
        }
        i += rn;
    }
}

fn tri_up<T: Scalar>(
    v: &mut [T],
    block: &[T],
    diag_inv: &[T],
    ld: usize,
    nr: usize,
    jb: usize,
    je: usize,
) {
    let mut c0 = 0;
    while c0 < nr {
        let w = (nr - c0).min(SOLVE_RHS_TILE);
        match w {
            4 => tri_up_tile::<T, 4>(v, block, diag_inv, ld, nr, c0, jb, je),
            3 => tri_up_tile::<T, 3>(v, block, diag_inv, ld, nr, c0, jb, je),
            2 => tri_up_tile::<T, 2>(v, block, diag_inv, ld, nr, c0, jb, je),
            _ => tri_up_tile::<T, 1>(v, block, diag_inv, ld, nr, c0, jb, je),
        }
        c0 += w;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tri_up_tile<T: Scalar, const W: usize>(
    v: &mut [T],
    block: &[T],
    diag_inv: &[T],
    ld: usize,
    nr: usize,
    c0: usize,
    jb: usize,
    je: usize,
) {
    for k in (jb..je).rev() {
        let col = &block[(k - jb) * ld..];
        let mut acc = [T::zero(); W];
        acc.copy_from_slice(&v[k * nr + c0..k * nr + c0 + W]);
        for i in k + 1..je {
            let l = col[i - jb];
            let vi = &v[i * nr + c0..i * nr + c0 + W];
            for (a, &x) in acc.iter_mut().zip(vi) {
                *a = *a - l * x;
            }
        }
        if let Some(&d) = diag_inv.get(k) {
            for a in acc.iter_mut() {
                *a = *a * d;
            }
        }
        v[k * nr + c0..k * nr + c0 + W].copy_from_slice(&acc);
    }
}

fn chunk_align(nr: usize) -> usize {
    if nr == 1 {
        SOLVE_GEMV_ROWS
    } else {
        2
    }
}

fn chunk_rows(i: usize, count: usize, rows: usize, nr: usize) -> std::ops::Range<usize> {
    let align = chunk_align(nr);
    let units = rows / align;
    let at = |c: usize| {
        if c >= count {
            rows
        } else {
            c * units / count * align
        }
    };
    at(i)..at(i + 1)
}

fn run_chunks(count: usize, f: impl Fn(usize) + Sync + Send) {
    let _s = rsprobe::scope!(TRACE, "product");
    rsprobe::add!(TRACE, "chunks", count);
    if count > 1 {
        crate::parallel::each(count, crate::parallel::Work::ns(f64::INFINITY), f);
    } else {
        (0..count).for_each(f);
    }
}

fn node_mode() -> crate::dense::gemm_backend::GemmMode {
    crate::dense::gemm_backend::GemmMode::new(
        gemm::Parallelism::None,
        &crate::KernelSettings::default(),
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn fwd_own<T: Scalar>(
    y: *mut T,
    block: *const T,
    ld: usize,
    jb: usize,
    je: usize,
    rows: std::ops::Range<usize>,
    nr: usize,
) {
    debug_assert!(rows.start >= je);
    crate::dense::gemm_backend::gemm(
        rows.len(),
        nr,
        je - jb,
        y.add(rows.start * nr),
        1,
        nr as isize,
        true,
        block.add(rows.start - jb),
        ld as isize,
        1,
        y.add(jb * nr) as *const T,
        1,
        nr as isize,
        T::one(),
        T::from_real(-1.0),
        false,
        false,
        false,
        node_mode(),
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn fwd_below<T: Scalar>(
    t: *mut T,
    rect: *const T,
    m: usize,
    jb: usize,
    je: usize,
    y: *const T,
    rows: std::ops::Range<usize>,
    nr: usize,
    add: bool,
) {
    crate::dense::gemm_backend::gemm(
        rows.len(),
        nr,
        je - jb,
        t.add(rows.start * nr),
        1,
        nr as isize,
        add,
        rect.add(jb * m + rows.start),
        m as isize,
        1,
        y.add(jb * nr),
        1,
        nr as isize,
        T::one(),
        T::one(),
        false,
        false,
        false,
        node_mode(),
    );
}

unsafe fn bwd_off<T: Scalar>(
    x: *mut T,
    rect: *const T,
    m: usize,
    g: *const T,
    rows: std::ops::Range<usize>,
    nr: usize,
) {
    crate::dense::gemm_backend::gemm(
        rows.len(),
        nr,
        m,
        x.add(rows.start * nr),
        1,
        nr as isize,
        true,
        rect.add(rows.start * m),
        1,
        m as isize,
        g,
        1,
        nr as isize,
        T::one(),
        T::from_real(-1.0),
        false,
        false,
        false,
        node_mode(),
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn bwd_in<T: Scalar>(
    x: *mut T,
    block: *const T,
    ld: usize,
    w: usize,
    jb: usize,
    je: usize,
    rows: std::ops::Range<usize>,
    nr: usize,
) {
    debug_assert!(jb <= rows.start && rows.end <= je);
    crate::dense::gemm_backend::gemm(
        rows.len(),
        nr,
        w - je,
        x.add(rows.start * nr),
        1,
        nr as isize,
        true,
        block.add(je - jb + (rows.start - jb) * ld),
        1,
        ld as isize,
        x.add(je * nr) as *const T,
        1,
        nr as isize,
        T::one(),
        T::from_real(-1.0),
        false,
        false,
        false,
        node_mode(),
    );
}

fn solve_diagonal<T: Scalar>(
    f: &LdltPivots<T>,
    y: &mut [T],
    nr: usize,
    cols: std::ops::Range<usize>,
) -> Result<(), RslabError> {
    let mut k = cols.start;
    while k < cols.end {
        let row = (k - cols.start) * nr;
        if f.two_by_two[k] {
            debug_assert!(
                k + 1 < cols.end,
                "2x2 pivot at column {k} crosses the range end"
            );
            let d11 = f.d_diag[k];
            let d21 = f.d_subdiag[k];
            let d22 = f.d_diag[k + 1];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            let (r0, r1) = y[row..row + 2 * nr].split_at_mut(nr);
            for c in 0..nr {
                let z0 = r0[c];
                let z1 = r1[c];
                r0[c] = (d22 * z0 - d21 * z1) * detinv;
                r1[c] = (d11 * z1 - d21 * z0) * detinv;
            }
            k += 2;
        } else {
            let d = f.d_diag[k];
            if d == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let dinv = d.recip();
            for v in &mut y[row..row + nr] {
                *v = *v * dinv;
            }
            k += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::test_constants::{solve_backward_error, U64};
    use crate::{CscMatrix, LdltSolver, SolverSettings};

    fn grid(m: usize, shift: f64) -> CscMatrix<f64> {
        let n = m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            row_idx.push(j);
            values.push(4.0 - shift + 0.1 * ((j * 7919) % 13) as f64);
            if x + 1 < m {
                row_idx.push(j + 1);
                values.push(-1.0);
            }
            if y + 1 < m {
                row_idx.push(j + m);
                values.push(-1.0);
            }
            col_ptr.push(row_idx.len());
        }
        CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    fn grid3d(m: usize) -> CscMatrix<f64> {
        let n = m * m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y, z) = (j % m, (j / m) % m, j / (m * m));
            row_idx.push(j);
            values.push(6.0 + 0.01 * ((j * 7919) % 13) as f64);
            for (cond, nb) in [
                (x + 1 < m, j + 1),
                (y + 1 < m, j + m),
                (z + 1 < m, j + m * m),
            ] {
                if cond {
                    row_idx.push(nb);
                    values.push(-1.0);
                }
            }
            col_ptr.push(row_idx.len());
        }
        CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    fn backward_error(a: &CscMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.n;
        let (mut r, mut rows) = (b.iter().map(|v| -v).collect::<Vec<f64>>(), vec![0.0; n]);
        for j in 0..n {
            for e in a.col_ptr[j]..a.col_ptr[j + 1] {
                let (i, v) = (a.row_idx[e], a.values[e]);
                r[i] += v * x[j];
                rows[i] += v.abs();
                if i != j {
                    r[j] += v * x[i];
                    rows[j] += v.abs();
                }
            }
        }
        let inf = |v: &[f64]| v.iter().map(|z| z.abs()).fold(0.0, f64::max);
        inf(&r) / (inf(&rows) * inf(x) + inf(b))
    }

    fn dominant_solve_bound(n: usize) -> f64 {
        let nu = n as f64 * f64::EPSILON / 2.0;
        2.0 * nu / (1.0 - nu)
    }

    #[test]
    fn plan_solve_matches_scalar_kernel_and_solves_alike_on_every_pool() {
        let cases: Vec<(&str, CscMatrix<f64>, bool)> = vec![
            ("grid 7", grid(7, 0.0), true),
            ("grid 40", grid(40, 0.0), true),
            ("grid 40 indefinite", grid(40, 3.7), false),
            ("grid3d 26", grid3d(26), true),
        ];
        for (name, a, dominant) in &cases {
            let n = a.n;
            let within = |x: &[f64], b: &[f64], what: &str| {
                let eta = backward_error(a, x, b);
                assert!(
                    !dominant || eta <= dominant_solve_bound(n),
                    "{name} {what}: backward error {eta:e} over {:e}",
                    dominant_solve_bound(n)
                );
            };
            let opts = SolverSettings::default()
                .with_threads(1)
                .with_ordering(crate::OrderingMethod::MetisND);
            let s = LdltSolver::factor(a, &opts).unwrap();
            let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
            let x1 = s.solve(&b).unwrap();
            within(&x1, &b, "one thread");
            let nrhs = 3;
            let bb: Vec<f64> = (0..n * nrhs)
                .map(|k| ((k * 13) % 11) as f64 - 5.0)
                .collect();
            let xb = s.solve_many(&bb, nrhs).unwrap();
            for c in 0..nrhs {
                let bc: Vec<f64> = (0..n).map(|i| bb[c * n + i]).collect();
                let xc = s.solve(&bc).unwrap();
                let omega = backward_error(a, &xb[c * n..(c + 1) * n], &bc)
                    .max(backward_error(a, &xc, &bc));
                assert!(
                    omega <= solve_backward_error(n, U64),
                    "{name} column {c}: backward error {omega:e}"
                );
            }
            let ambient = opts.clone().with_threads(crate::Threads::Ambient);
            let one = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            let sa = one.install(|| LdltSolver::factor(a, &ambient).unwrap());
            for threads in [1usize, 2, 5] {
                let st = LdltSolver::factor(a, &opts.clone().with_threads(threads)).unwrap();
                let xs = st.solve(&b).unwrap();
                within(&xs, &b, &format!("{threads} threads"));
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let xt = pool.install(|| sa.solve(&b).unwrap());
                assert_eq!(xt, x1, "ambient threads={threads}");
                let xbt = pool.install(|| sa.solve_many(&bb, nrhs).unwrap());
                assert_eq!(xbt, xb, "ambient block threads={threads}");
            }
        }
    }

    #[test]
    fn plan_cuts_the_tree() {
        let a = grid(60, 0.0);
        let s = LdltSolver::factor(&a, &SolverSettings::default().with_threads(1)).unwrap();
        assert!(s.plan.subtrees.len() > 1);
        assert!(!s.plan.top_levels.is_empty());
    }

    #[test]
    fn plan_falls_back_on_a_structure_off_the_tree() {
        use super::super::panel::PanelFactor;
        use super::{PlanWork, SolvePlan};
        let factor = PanelFactor {
            n: 3,
            sn_col: vec![0, 1, 2, 3],
            rows: vec![1, 2, 2],
            row_ptr: vec![0, 2, 3, 3],
            val_off: vec![0, 3, 5],
            vals: vec![1.0, 0.5, 0.25, 1.0, -2.0, 1.0],
        };
        let plan = SolvePlan::from_panels(
            factor,
            &[2, 2, usize::MAX],
            true,
            crate::SolveSettings::default(),
        );
        assert!(plan.subtrees.is_empty(), "solved as one sequential chain");
        let mut work = PlanWork::default();
        let mut y = vec![1.0, 2.5, -0.75];
        let one = crate::Threads::Fixed(1);
        plan.sweep(&mut work, 1, one, |w| plan.forward_block(1, &mut y, w));
        assert_eq!(y, [1.0, 2.0, 3.0]);
        let mut y = vec![2.75, -4.0, 3.0];
        plan.sweep(&mut work, 1, one, |w| plan.backward_block(1, &mut y, w));
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }

    fn convdiff(m: usize) -> crate::GeneralCsc<f64> {
        let n = m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            if y > 0 {
                row_idx.push(j - m);
                values.push(-1.0);
            }
            if x > 0 {
                row_idx.push(j - 1);
                values.push(-1.0 - 0.4);
            }
            row_idx.push(j);
            values.push(4.0 + 0.05 * ((j * 7919) % 13) as f64);
            if x + 1 < m {
                row_idx.push(j + 1);
                values.push(-1.0 + 0.4);
            }
            if y + 1 < m {
                row_idx.push(j + m);
                values.push(-1.0);
            }
            col_ptr.push(row_idx.len());
        }
        crate::GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    fn general_backward_error(a: &crate::GeneralCsc<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.n;
        let (mut r, mut rows) = (b.to_vec(), vec![0.0; n]);
        for j in 0..n {
            for e in a.col_ptr[j]..a.col_ptr[j + 1] {
                r[a.row_idx[e]] -= a.values[e] * x[j];
                rows[a.row_idx[e]] += a.values[e].abs();
            }
        }
        let inf = |v: &[f64]| v.iter().map(|z| z.abs()).fold(0.0, f64::max);
        inf(&r) / (inf(&rows) * inf(x) + inf(b))
    }

    #[test]
    fn ancestor_tasks_solve_as_the_levels_do() {
        use crate::{LuSolver, OrderingMethod};
        let opts = SolverSettings {
            solve: crate::SolveSettings {
                leaf_subtrees: 4,
                par_min_work: Some(0),
                gemm_task: 1,
            },
            ..SolverSettings::default()
        }
        .with_ordering(OrderingMethod::MetisND)
        .with_threads(crate::Threads::Ambient);
        let on = |t: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let nrhs = 64;
        let a = convdiff(80);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
        let bb: Vec<f64> = (0..n * nrhs)
            .map(|k| ((k * 13) % 11) as f64 - 5.0)
            .collect();
        use super::SolvePlan;
        let chunked = |p: &SolvePlan<f64>| {
            p.top.iter().any(|&s| {
                let (_, w, _, m, _, _) = p.node(s);
                p.node_chunks(true, m, w, 1) > 1
            })
        };
        let lu = on(1).install(|| LuSolver::factor(&a, &opts).unwrap());
        assert!(
            !lu.plan_l.top.is_empty() && !lu.plan_l.chain,
            "a tree above the cut"
        );
        assert!(chunked(&lu.plan_l), "LU products in chunks");
        let (x1, xb1) =
            on(1).install(|| (lu.solve(&b).unwrap(), lu.solve_many(&bb, nrhs).unwrap()));
        let g = grid3d(14);
        let gb: Vec<f64> = (0..g.n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let gbb: Vec<f64> = (0..g.n * nrhs)
            .map(|k| ((k * 5) % 9) as f64 - 4.0)
            .collect();
        let ld = on(1).install(|| LdltSolver::factor(&g, &opts).unwrap());
        assert!(
            !ld.plan.top.is_empty() && !ld.plan.chain,
            "a tree above the cut"
        );
        assert!(chunked(&ld.plan), "LDL^T products in chunks");
        let tiled =
            (ld.plan.top.iter()).any(|&s| ld.plan.node(s).1 > 2 * crate::constants::PANEL_BLOCK);
        assert!(tiled, "an ancestor of three tiles or more");
        let (g1, gb1) =
            on(1).install(|| (ld.solve(&gb).unwrap(), ld.solve_many(&gbb, nrhs).unwrap()));
        for t in [2usize, 5, 8] {
            let pool = on(t);
            assert_eq!(pool.install(|| lu.solve(&b).unwrap()), x1, "LU threads={t}");
            assert_eq!(
                pool.install(|| lu.solve_many(&bb, nrhs).unwrap()),
                xb1,
                "LU block threads={t}"
            );
            assert_eq!(
                pool.install(|| ld.solve(&gb).unwrap()),
                g1,
                "LDLT threads={t}"
            );
            assert_eq!(
                pool.install(|| ld.solve_many(&gbb, nrhs).unwrap()),
                gb1,
                "LDLT block threads={t}"
            );
        }
    }

    #[test]
    fn lu_plans_solve_and_are_thread_invariant() {
        use crate::{LuSolver, OrderingMethod};
        for m in [30usize, 60] {
            let a = convdiff(m);
            let n = a.n;
            let opts = SolverSettings::default()
                .with_threads(1)
                .with_ordering(OrderingMethod::MetisND);
            let s = LuSolver::factor(&a, &opts).unwrap();
            let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
            let x1 = s.solve(&b).unwrap();
            assert!(
                general_backward_error(&a, &x1, &b) <= solve_backward_error(n, U64),
                "backward error m={m}"
            );
            let nrhs = 3;
            let bb: Vec<f64> = (0..n * nrhs)
                .map(|k| ((k * 13) % 11) as f64 - 5.0)
                .collect();
            let xb = s.solve_many(&bb, nrhs).unwrap();
            for c in 0..nrhs {
                let bc: Vec<f64> = (0..n).map(|i| bb[c * n + i]).collect();
                let xc = s.solve(&bc).unwrap();
                let omega = general_backward_error(&a, &xb[c * n..(c + 1) * n], &bc)
                    .max(general_backward_error(&a, &xc, &bc));
                assert!(
                    omega <= solve_backward_error(n, U64),
                    "m={m} column {c}: backward error {omega:e}"
                );
            }
            let (xr, out) = s
                .solve_refined(&a, &b, &crate::RefinePolicy::steps(2))
                .unwrap();
            assert!(out.steps <= 2);
            assert!(general_backward_error(&a, &xr, &b) <= solve_backward_error(n, U64));
            for threads in [1usize, 2, 5] {
                let st = LuSolver::factor(&a, &opts.clone().with_threads(threads)).unwrap();
                assert_eq!(st.solve(&b).unwrap(), x1, "threads={threads}");
                assert_eq!(
                    st.solve_many(&bb, nrhs).unwrap(),
                    xb,
                    "block threads={threads}"
                );
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let ambient = opts.clone().with_threads(crate::Threads::Ambient);
                let sa = pool.install(|| LuSolver::factor(&a, &ambient).unwrap());
                let xt = pool.install(|| sa.solve(&b).unwrap());
                assert_eq!(xt, x1, "ambient threads={threads}");
                let xbt = pool.install(|| sa.solve_many(&bb, nrhs).unwrap());
                assert_eq!(xbt, xb, "ambient block threads={threads}");
            }
        }
    }
}
