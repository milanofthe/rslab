//! Supernodal, tree-parallel triangular solves for the sparse LDL^T factor.
//!
//! A scalar CSC factor is the wrong layout for a solve: every entry costs a
//! value, a `usize` row index and a random access into the right-hand side,
//! on one core. The plan works on the factor in *supernodal panel form*
//! ([`PanelFactor`]): one dense column-major panel per supernode (unit-lower
//! diagonal block on top, the off-block rows below) with a single shared
//! list of `u32` row indices. The LDL^T drivers produce that form directly,
//! so the plan takes the panels as its storage without a copy; a CSC factor
//! (the LU path) is converted once. A forward or backward sweep then streams
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
//! run as a task tree: in the forward sweep an ancestor starts when its
//! children are done and first takes in the accumulations for its columns,
//! in the order a level-by-level sweep reduces them; in the backward sweep it
//! starts when its parent is done, and the subtrees follow in parallel
//! reading only final values. A large ancestor runs its products in row
//! chunks on the pool, and a wide one takes the subtrees' terms in column
//! chunks. Every sum is formed in the same order for every thread count, so
//! the solution is bit-identical from one thread to many.

// The kernels index several parallel arrays (panels, row lists, slots,
// right-hand sides) at offset positions; index loops read closer to the
// linear algebra than zipped iterators would.
#![allow(clippy::needless_range_loop)]

use crate::constants::{
    GEMM_NS_PER_FLOP, IDX32_BYTES, SOLVE_BUILD_NODE_BYTES, SOLVE_DIAG_BLOCK, SOLVE_NS_PER_ENTRY,
    SOLVE_PAR_MIN_ENTRY_RHS, SOLVE_PULL_COLS, SOLVE_PULL_PAR_MIN, SOLVE_RHS_TILE, TILE_ROWS,
    VEC_HEADER_BYTES, WORD_BYTES,
};
use crate::error::RslabError;
use crate::numeric::ldlt::LdltPivots;
use crate::numeric::settings::Threads;
use crate::numeric::supernodal::panel::{PanelFactor, PanelStorage};
use crate::parallel::{Choice, PerScalar, Pick, Prior, Rate, ScalarRates, Work};
use crate::scalar::Scalar;
use std::sync::atomic::{AtomicU64, Ordering};

const NONE: u32 = u32::MAX;

mod sparse;
pub use sparse::SparseReach;

/// Whether a sweep runs on the calling thread (0) or on the pool (1): the
/// way the sweeps of its kind and size were fastest ([`Choice`]), a sweep
/// starting on the pool from [`SOLVE_PAR_MIN_ENTRY_RHS`] on before either
/// is measured. Per kind ([`Sweep`]): a forward sweep takes its subtrees'
/// terms where a backward one does not, and on a near-field LU took two
/// and a half times as long on as many panel entries; and per one or a
/// block of right-hand sides, whose kernels differ as their rates do
/// ([`sweep_rate`](SolvePlan::sweep_rate)).
static SWEEP_WAYS: [[PerScalar<Choice>; 2]; 3] = [
    [PerScalar::choices(), PerScalar::choices()],
    [PerScalar::choices(), PerScalar::choices()],
    [PerScalar::choices(), PerScalar::choices()],
];

/// What a sweep does: its own measured ways ([`SWEEP_WAYS`]).
#[derive(Clone, Copy)]
enum Sweep {
    Forward,
    Backward,
    /// The forward and backward sweeps of an `L D L^T` solve in one.
    Whole,
}

impl Sweep {
    /// The ways of this kind of sweep of `nr` right-hand sides.
    fn ways<T: Scalar>(self, nr: usize) -> &'static Choice {
        SWEEP_WAYS[self as usize][usize::from(nr > 1)].of::<T>()
    }
}

/// What a panel entry costs a sweep of one right-hand side (the register
/// tiles), learned from the sweeps.
static SOLVE_RATE: ScalarRates = ScalarRates::new(SOLVE_NS_PER_ENTRY);

/// What a panel entry costs a sweep per right-hand side of a block (the
/// GEMM node kernels), learned from the sweeps.
static SOLVE_BLOCK_RATE: ScalarRates = ScalarRates::new(GEMM_NS_PER_FLOP);

/// One leaf subtree of the cut: its supernodes in elimination order and the
/// rows outside it its updates accumulate into.
#[derive(Clone)]
pub(crate) struct Subtree {
    nodes: Vec<u32>,
    /// The accumulator's slots: the rows outside the subtree its nodes' off-block
    /// rows reach, ascending (they lie in the ancestors above the cut). An
    /// accumulator of the ancestor columns alone would hold every column of
    /// the subtree's ancestor path, most of them never written (a 79 k
    /// near-field LU: 128 MB a right-hand side over all accumulators).
    ext: Vec<u32>,
}

/// The factor in solve layout plus the tree schedule.
#[derive(Clone)]
pub(crate) struct SolvePlan<T> {
    n: usize,
    /// Supernode column ranges, `ns + 1`.
    sn_col: Vec<u32>,
    /// Off-block row ranges into `rows` / `ext_slot`, `ns + 1`.
    row_ptr: Vec<usize>,
    /// Off-block row indices per supernode, ascending.
    rows: Vec<u32>,
    /// Per off-block row: accumulator slot when the row lies outside the
    /// supernode's leaf subtree, `NONE` when it is written directly.
    ext_slot: Vec<u32>,
    /// The panels in the factor's own storage ([`PanelFactor`]), packed in
    /// place by column blocks ([`packed_blocks`]): supernode `s` is
    /// `vals[val_rng[s].0..val_rng[s].1]`.
    val_rng: Vec<(usize, usize)>,
    vals: Vec<T>,
    /// A row list of the factor's size, the buffer the next refactorization
    /// emits its rows into (see [`take_storage`](Self::take_storage)).
    spare_rows: Vec<u32>,
    /// Reciprocal diagonal per column for a non-unit triangular factor (the
    /// `U^T` of an LU); empty for a unit-diagonal factor.
    diag_inv: Vec<T>,
    subtrees: Vec<Subtree>,
    /// The ancestors (supernodes above the cut) by depth from the roots;
    /// siblings of a level are independent.
    pub(crate) top_levels: Vec<Vec<u32>>,
    /// Position of a supernode in `top` (`NONE` below the cut).
    top_index: Vec<u32>,
    /// Per ancestor level, the start of each node's accumulator (its own
    /// off-block rows) in the level's block, in slots (`level.len() + 1`).
    top_acc_ptr: Vec<Vec<usize>>,
    /// The ancestors by `top_index` (ascending), the schedule of the
    /// parallel sweeps: every ancestor a task that starts when the nodes it
    /// depends on are done (see `top_forward_flow`).
    top: Vec<u32>,
    /// The roots among the ancestors, by `top_index`.
    top_roots: Vec<u32>,
    /// Each ancestor's parent, by `top_index` (`NONE` for a root).
    top_parent: Vec<u32>,
    /// Each ancestor's children above the cut, by `top_index` (CSR over
    /// `top_kids`, `top.len() + 1`).
    top_kids_ptr: Vec<usize>,
    top_kids: Vec<u32>,
    /// Where each ancestor's accumulator starts in the one block of all of
    /// them, in slots (`top.len() + 1`).
    top_acc_off: Vec<usize>,
    /// What each ancestor takes into its columns before its sweep (CSR over
    /// `pull`, `top.len() + 1`): `(source, lo, hi)`, the slots `lo..hi` of a
    /// leaf subtree's accumulator (`source < subtrees.len()`, the rows its
    /// `ext` lists there) or of an ancestor's (`source - subtrees.len()` its
    /// `top_index`, the rows of its own off-block list there), the ones that
    /// lie in this ancestor's columns. In the order the level sweeps reduce
    /// them: the subtrees in order, then the ancestors from the deepest level
    /// up, a level in node order.
    pull_ptr: Vec<usize>,
    pull: Vec<(u32, u32, u32)>,
    /// The structure is no elimination tree (see `from_panels`): the level
    /// sweeps alone, one node after another.
    chain: bool,
    /// The panel entries of the leaf subtrees, the work their sweeps share.
    subtree_work: usize,
    /// The blocking of the sweeps.
    cfg: crate::SolveSettings,
}

/// Per-task scratch vectors, reused across nodes: the big nodes need
/// buffers of tens to hundreds of kilobytes, and allocating those per node
/// (page faults under the allocator lock) serialises the parallel phases.
pub(crate) struct Scratch<T> {
    /// A node's product of its rows below the diagonal block (forward).
    t: Vec<T>,
    /// The final values of a node's rows below it, gathered (backward).
    g: Vec<T>,
    /// A leaf subtree's accumulator of its off-tree updates.
    acc: Vec<T>,
}

/// Scratch for the ancestor tasks, taken by a node for its sweep and given
/// back after it: as many as nodes run at once rather than one per ancestor
/// (a near-field LU's 440 ancestors held about 250 MB for 23 right-hand
/// sides). Not per worker: a node waiting inside its sections may run
/// another node on its thread meanwhile.
pub(crate) struct ScratchPool<T>(std::sync::Mutex<Vec<Scratch<T>>>);

impl<T> Default for ScratchPool<T> {
    fn default() -> Self {
        Self(std::sync::Mutex::new(Vec::new()))
    }
}

impl<T> ScratchPool<T> {
    fn take(&self) -> Scratch<T> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_default()
    }

    fn give(&self, sc: Scratch<T>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).push(sc);
    }
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

/// The sweeps' scratch across solves (part of [`crate::SolveWork`]): a
/// [`Scratch`] per leaf subtree and per node of the widest node-parallel
/// ancestor level, the apex nodes' scratch, and an ancestor level's
/// accumulator block. Sized by the first solve through a plan, reused
/// after, so a warm sweep allocates nothing.
pub(crate) struct PlanWork<T> {
    subtrees: Vec<Scratch<T>>,
    level: Vec<Scratch<T>>,
    apex: Scratch<T>,
    /// The ancestor tasks' scratch ([`ScratchPool`]).
    pool: ScratchPool<T>,
    top_acc: Vec<T>,
    /// Per ancestor, the children whose forward sweep is still to finish
    /// (`top_forward_flow`).
    pending: Vec<std::sync::atomic::AtomicU32>,
    /// This sweep's decision: on the pool, or on the calling thread.
    parallel: bool,
    /// The pick it was made as and its work in ops, where its run is to
    /// be measured.
    pick: Option<(Pick, f64)>,
}

impl<T> Default for PlanWork<T> {
    fn default() -> Self {
        Self {
            subtrees: Vec::new(),
            level: Vec::new(),
            apex: Scratch::default(),
            pool: ScratchPool::default(),
            top_acc: Vec::new(),
            pending: Vec::new(),
            parallel: false,
            pick: None,
        }
    }
}

impl<T> PlanWork<T> {
    /// At least a scratch per subtree and per node of the widest level of
    /// `plan`, and a count per ancestor (`top` of them); never shrinks, so
    /// one work serves several plans.
    fn fit(&mut self, subtrees: usize, widest: usize, top: usize) {
        if self.pending.len() < top {
            self.pending
                .resize_with(top, || std::sync::atomic::AtomicU32::new(0));
        }
        if self.subtrees.len() < subtrees {
            self.subtrees.resize_with(subtrees, Scratch::default);
        }
        if self.level.len() < widest {
            self.level.resize_with(widest, Scratch::default);
        }
    }
}

/// Shared mutable view of a right-hand side for the subtree phases.
///
/// SAFETY contract: concurrent users write disjoint index sets (a subtree
/// writes only its own columns) and read only indices no other task writes
/// during the phase (its own columns, or ancestor columns that are final).
#[derive(Clone, Copy)]
struct Shared<T>(*mut T, usize);
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Sync> Sync for Shared<T> {}
impl<T> Shared<T> {
    /// The pointer, for a task that takes a part of it (the whole `Shared`
    /// captured, not its field).
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

/// The cut of the tree ([`SolvePlan::from_panels`]): split the heaviest
/// subtree until every leaf subtree holds at most `1 / leaf_subtrees` of the
/// work (or is a single supernode). `own(s)` is the work of supernode `s`,
/// its panel entries. Returns which supernodes lie above the cut and the
/// leaf subtree roots, ascending.
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

/// The heap of a [`SolvePlan`] not built yet, for the memory plan: the
/// schedule around the panels, its build and a solve's work vectors.
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

/// The [`LayoutSize`] of the plan [`SolvePlan::from_panels`] builds on the
/// supernode tree (`parent`, `usize::MAX` for a root) with each supernode's
/// width `w` and off-block rows `off_rows(s)` (ascending), solved on `threads`
/// threads. Mirrors `from_panels` on a tree-nested structure; where a
/// factorization keeps fewer rows than its structure has (cancellation), an
/// upper bound.
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
    // per leaf subtree its nodes, and its rows outside it (`ext`) and their
    // runs by ancestor (its pulls)
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
    // sn_col, row_ptr, val_rng (a pair per supernode), rows, ext_slot, top_index.
    let (idx, word, vec) = (IDX32_BYTES, WORD_BYTES, VEC_HEADER_BYTES);
    let fixed = idx * (ns + 1) + word * (ns + 1) + 2 * word * ns + word * rows + idx * ns;
    let subtrees = leaf_roots.len() * std::mem::size_of::<Subtree>()
        + (0..leaf_roots.len())
            .map(|t| idx * pushed(sizes[t]) + idx * ext[t])
            .sum::<usize>();
    let levels = level_len.len() * vec + level_len.iter().map(|&l| idx * pushed(l)).sum::<usize>();
    // The task schedule: the ancestors, their roots, parents and children,
    // the accumulator offsets and the pulls (a run of a source's rows per
    // ancestor they lie in: every leaf subtree's and every ancestor's).
    let roots = top.iter().filter(|&&t| parent[t] == NONE).count();
    let pulls = ext_runs.iter().sum::<usize>()
        + top
            .iter()
            .map(|&t| runs(off_rows(t), &sn_of))
            .sum::<usize>();
    let offsets = word * (top.len() + 1); // the children's, the accumulators', the pulls'
    let schedule = idx * (top.len() + roots + top.len() + top.len() - roots)
        + offsets
        + offsets
        + offsets
        + std::mem::size_of::<(u32, u32, u32)>() * pulls;
    // sn_of, the tree (parent, children), work, heap, flags and slots.
    let build = idx * n + ns * SOLVE_BUILD_NODE_BYTES;
    // Accumulators of the subtrees (their `ext`) and of every ancestor (its
    // own off-block rows; the task sweep keeps them all to its end), and per
    // thread a node's product and gathered rows twice: a subtree sweep's, and
    // the pooled scratch of the ancestor tasks running at once.
    let max_ld = (0..ns).map(|s| w[s] + m[s]).max().unwrap_or(0);
    let solve_per_rhs = ext.iter().sum::<usize>()
        + top.iter().map(|&t| m[t]).sum::<usize>()
        + threads.max(1) * 2 * (2 * max_ld);
    LayoutSize {
        held: (fixed + subtrees + levels + schedule) as u64,
        build: build as u64,
        solve_per_rhs: solve_per_rhs as u64,
    }
}

/// The runs of an ascending row list by the supernode its rows lie in.
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
    /// Heap bytes held: the panels (the factor's values) and the schedule.
    /// The reciprocals of the diagonal of a non-unit factor (empty for a
    /// unit one), in elimination order.
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
                let (_, w, m) = factor.shape(s);
                let ld = w + m;
                for k in 0..w {
                    d.push(factor.panel(s)[k * ld + k].recip());
                }
            }
            d
        };
        let PanelFactor {
            rows,
            row_ptr,
            val_off,
            mut vals,
            ..
        } = factor;
        debug_assert!(
            (0..ns).all(|s| rows[row_ptr[s]..row_ptr[s + 1]]
                .windows(2)
                .all(|p| p[0] < p[1])),
            "off-block rows ascending"
        );
        let val_rng = pack_panels(&mut vals, &sn_col, &row_ptr, &val_off);
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

        // Subtree membership and node lists (ascending = elimination order).
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
        // Ancestors by depth from the roots (level 0 = roots); an ancestor's
        // accumulator is its own off-block rows.
        let mut depth = vec![0u32; ns];
        for &t in top.iter().rev() {
            // Parents carry larger indices: descending order sets them first.
            let p = parent[t as usize];
            depth[t as usize] = if p == NONE { 0 } else { depth[p as usize] + 1 };
        }
        let max_depth = top.iter().map(|&t| depth[t as usize]).max().unwrap_or(0) as usize;
        let mut top_levels: Vec<Vec<u32>> =
            vec![Vec::new(); if top.is_empty() { 0 } else { max_depth + 1 }];
        for &t in &top {
            top_levels[depth[t as usize] as usize].push(t);
        }

        // Accumulator slots of the off-tree rows: a subtree's row its position
        // in the subtree's `ext`, an ancestor's off-block row its position in
        // the ancestor's own rows. A row whose supernode is not an ancestor of
        // the writer (the structure no longer follows an elimination tree)
        // makes the structure degenerate; such a factor is solved without the
        // subtree phase (everything above the cut, sequential in the tree
        // phases).
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
            for &s in &st.nodes {
                for e in row_ptr[s as usize]..row_ptr[s as usize + 1] {
                    let r = rows[e];
                    let a = sn_of[r as usize];
                    if subtree_of[a as usize] != subtree_of[s as usize] {
                        if on_path[a as usize] {
                            // `ext` holds every such row
                            ext_slot[e] = st.ext.partition_point(|&x| x < r) as u32;
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
                    ext_slot[e] = (e - r0) as u32;
                } else {
                    degenerate = true;
                }
            }
            mark(parent[t as usize], false, &mut on_path);
        }
        let chain = degenerate;
        let (subtrees, top, top_levels, ext_slot) = if degenerate {
            // The structure does not follow an elimination tree (an LU whose
            // row pivoting moved rows across subtrees): one sequential chain, every node an ancestor
            // without slots. Levels run from the root, so the deepest level
            // (processed first by the forward sweep) is the first column.
            if crate::logging::enabled(crate::logging::LogLevel::Debug) {
                crate::logging::debug("solve layout: structure not tree-nested, sequential sweeps");
            }
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
        // Per ancestor level, where each node's accumulator starts in the
        // level's block, in slots.
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
        // The task schedule of the ancestors.
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
        // The pulls, in the level sweeps' order of reduction: per source the
        // runs of its accumulator rows by the ancestor they lie in (`ext` and
        // an ancestor's rows are ascending, an ancestor's columns contiguous).
        let mut pulls: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); top.len()];
        if !chain {
            let runs = |src: u32, list: &[u32], pulls: &mut Vec<Vec<(u32, u32, u32)>>| {
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
        // The work of the subtrees' sweeps.
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
        plan.log_layout();
        plan
    }

    /// The panels' buffer and a spare row list, taken out for the next
    /// factorization of the same analysis to fill; [`refill`](Self::refill)
    /// puts the result back. The plan cannot solve until then.
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
        // (the supernodes and their rows fix the panels' shapes, packed or not)
        let same = cfg_now == self.cfg
            && factor.n == self.n
            && unit == self.diag_inv.is_empty()
            && factor.sn_col == self.sn_col
            && factor.row_ptr == self.row_ptr
            && factor.rows == self.rows;
        if !same {
            *self = Self::from_panels(factor, supernode_parent, unit, cfg);
            return;
        }
        // The new row list equals the plan's: kept as the next spare, and the
        // buffer keeps its room for the next factorization to fill again.
        let PanelFactor {
            mut vals,
            rows,
            val_off,
            ..
        } = factor;
        if !unit {
            for s in 0..ns {
                let (c0, c1) = (self.sn_col[s] as usize, self.sn_col[s + 1] as usize);
                let ld = c1 - c0 + self.row_ptr[s + 1] - self.row_ptr[s];
                let panel = &vals[val_off[s]..];
                for k in 0..c1 - c0 {
                    self.diag_inv[c0 + k] = panel[k * ld + k].recip();
                }
            }
        }
        self.val_rng = pack_panels(&mut vals, &self.sn_col, &self.row_ptr, &val_off);
        self.vals = vals;
        self.spare_rows = rows;
    }

    /// Bytes of the panel storage (values plus row indices).
    pub fn bytes(&self) -> usize {
        self.vals.len() * std::mem::size_of::<T>() + self.rows.len() * IDX32_BYTES
    }

    /// The schedule's bounds at the `debug` level, in panel entries (a
    /// sweep's work per right-hand side): the leaf subtrees' share and their
    /// balance, the ancestors' share, the longest chain of ancestors to a
    /// root (the critical path of their task tree), the widths of the top
    /// levels and the accumulator slots the ancestors take.
    fn log_layout(&self) {
        if !crate::logging::enabled(crate::logging::LogLevel::Debug) {
            return;
        }
        let entries = |s: u32| self.val_rng[s as usize].1 - self.val_rng[s as usize].0;
        let sub: Vec<usize> = (self.subtrees.iter())
            .map(|st| st.nodes.iter().map(|&s| entries(s)).sum())
            .collect();
        let sub_max = sub.iter().copied().max().unwrap_or(0);
        let top_work = self.nodes_work(&self.top);
        let path = self.critical_path();
        let critical = self.nodes_work(&path.iter().map(|&t| self.top[t]).collect::<Vec<_>>());
        // the nodes of that chain: their triangles' and their rows below's entries, and
        // the widths of the large ones (root first)
        let (mut tri, mut off, mut wide) = (0usize, 0usize, Vec::new());
        for &t in &path {
            let (_, w, _, m, _, _) = self.node(self.top[t]);
            tri += w * (w + 1) / 2;
            off += w * m;
            if w >= SOLVE_DIAG_BLOCK {
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
        crate::logging::debug(&format!(
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
        ));
    }

    /// The longest chain of ancestors to a root in panel entries, the
    /// critical path of their task tree: its `top` indices, root first.
    fn critical_path(&self) -> Vec<usize> {
        let entries = |s: u32| self.val_rng[s as usize].1 - self.val_rng[s as usize].0;
        // ancestors ascend, a parent after its children
        let mut path = vec![0usize; self.top.len()];
        let mut next = vec![NONE; self.top.len()];
        for ti in 0..self.top.len() {
            let kids = &self.top_kids[self.top_kids_ptr[ti]..self.top_kids_ptr[ti + 1]];
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

    /// The row chunks a node's product of `rows` rows and depth `depth` for `nr`
    /// right-hand sides runs in on the pool (`par`, the node an ancestor): about
    /// [`SolveSettings::gemm_task`](crate::SolveSettings::gemm_task) entries times
    /// right-hand sides each, at least two rows each (a single row is a
    /// matrix-vector product inside the GEMM, with its own order of sums). One,
    /// the whole product, on the calling thread; none for no rows. Within a call
    /// the GEMM blocks a product's depth alike for any row count (depth up to 64
    /// here in the forward sweep, row counts up to 64 in the backward one), so a
    /// row's sums are the same in a chunk as in the whole product, and a node's
    /// are the same on one thread as on many.
    fn node_chunks(&self, par: bool, rows: usize, depth: usize, nr: usize) -> usize {
        let whole = usize::from(rows > 0);
        if !par {
            return whole;
        }
        let by_work = (rows * depth * nr).div_ceil(self.cfg.gemm_task.max(1));
        by_work.min(rows / 2).max(whole)
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

    // -----------------------------------------------------------------------
    // Single right-hand side
    // -----------------------------------------------------------------------

    /// Run `f` inside a pool when `par`: the factor's scoped pool of its
    /// worker budget, or under [`Threads::Ambient`] the current one. Every
    /// parallel section is cheap to start from a worker and expensive to
    /// inject from outside, so a solve enters the pool once.
    fn in_pool<R: Send>(par: bool, threads: Threads, f: impl FnOnce() -> R + Send) -> R {
        if !par {
            return f();
        }
        match threads {
            _ if crate::parallel::is_lent() => crate::parallel::enter(f64::INFINITY, f),
            Threads::Ambient => crate::parallel::enter(f64::INFINITY, f),
            budget => budget.run(0, |cap| cap, f),
        }
    }

    /// What a panel entry costs a sweep per right-hand side, of one or of a
    /// block of `nr`.
    fn sweep_rate(nr: usize) -> &'static Rate {
        if nr == 1 {
            SOLVE_RATE.of::<T>()
        } else {
            SOLVE_BLOCK_RATE.of::<T>()
        }
    }

    /// The work of a sweep over `entries` panel entries and `nr` right-hand
    /// sides, at the sweeps' rate (its runs on the calling thread measured).
    fn sweep_work(entries: usize, nr: usize) -> Work<'static> {
        Work::of((entries * nr) as f64, Self::sweep_rate(nr))
    }

    /// `f` over the items with their scratch: as wide as the pool when
    /// `par`, in order on the calling thread otherwise, `work` all of them
    /// measured there; the same sums either way.
    fn each<A: Sync, S: Send>(
        par: bool,
        work: Work<'_>,
        items: &[A],
        scratch: &mut [S],
        f: impl Fn(usize, &A, &mut S) + Sync + Send,
    ) {
        if par {
            crate::parallel::zip_mut(items, scratch, crate::parallel::Work::ns(f64::INFINITY), f);
        } else {
            work.serial(|| {
                for (i, (a, s)) in items.iter().zip(scratch.iter_mut()).enumerate() {
                    f(i, a, s);
                }
            });
        }
    }

    /// The panel entries of the supernodes `nodes`.
    fn nodes_work(&self, nodes: &[u32]) -> usize {
        (nodes.iter())
            .map(|&s| self.val_rng[s as usize].1 - self.val_rng[s as usize].0)
            .sum()
    }

    /// The panel entries of the leaf subtrees.
    fn subtrees_work(&self) -> usize {
        self.subtree_work
    }

    /// `work` fitted to a sweep of `nr` right-hand sides under the thread
    /// budget `threads`: a scratch per subtree and per node of the widest
    /// ancestor level, and the decision to run on the pool: the way the
    /// sweeps of its kind and work were fastest ([`SWEEP_WAYS`]), on the pool
    /// expected at its work over the budget's threads where its panel
    /// entries times the right-hand sides up to four reach
    /// [`SOLVE_PAR_MIN_ENTRY_RHS`] (whatever the sweeps' rate, which a
    /// sweep on the pool does not measure, says), else only explored, and
    /// the calling thread the other way round; or
    /// where they clear
    /// [`SolveSettings::par_min_work`](crate::SolveSettings::par_min_work),
    /// where that is set.
    fn fit<'w>(
        &self,
        work: &'w mut PlanWork<T>,
        nr: usize,
        threads: Threads,
        kind: Sweep,
    ) -> &'w mut PlanWork<T> {
        let widest = self.top_levels.iter().map(Vec::len).max().unwrap_or(0);
        work.fit(self.subtrees.len(), widest, self.top.len());
        let budget = threads.resolve(|cap| cap);
        let ops = self.vals.len().saturating_mul(nr.min(SOLVE_RHS_TILE));
        work.pick = None;
        work.parallel = budget > 1
            && match self.cfg.par_min_work {
                Some(min) => ops >= min,
                None => {
                    let entries = self.vals.len().saturating_mul(nr) as f64;
                    let serial = entries * Self::sweep_rate(nr).ns_per_op();
                    let starts = ops >= SOLVE_PAR_MIN_ENTRY_RHS;
                    let ways = [
                        // The way the start does not take is only explored
                        // until measured: the sweeps' rate a sequential one
                        // is estimated at may be a prior far off (a block
                        // of right-hand sides: the GEMM rate, a sweep seven
                        // times slower).
                        Prior {
                            expected: if starts { f64::INFINITY } else { serial },
                            least: 0.0,
                        },
                        Prior {
                            expected: if starts {
                                serial / budget as f64
                            } else {
                                f64::INFINITY
                            },
                            least: serial / budget as f64,
                        },
                    ];
                    let pick = kind
                        .ways::<T>(nr)
                        .pick(serial, entries, 2, |o| Some(ways[o]));
                    if crate::logging::enabled(crate::logging::LogLevel::Debug) {
                        crate::logging::debug(&format!(
                            "solve: sweep nr={nr} budget={budget} serial~{:.2} ms pool~{:.2} ms -> {} (measured {})",
                            crate::clock::ms(std::time::Duration::from_nanos(serial as u64)),
                            crate::clock::ms(std::time::Duration::from_nanos(ways[1].expected as u64)),
                            if pick.option == 1 { "pool" } else { "here" },
                            pick.measured()
                        ));
                    }
                    work.pick = pick.measured().then_some((pick, entries));
                    pick.option == 1
                }
            };
        work
    }

    /// `sweep` of `nr` right-hand sides on `work` fitted for it, on the pool
    /// or on the calling thread as decided, timed for the way it ran where
    /// that is to be measured (once there).
    fn sweep<R: Send>(
        &self,
        work: &mut PlanWork<T>,
        nr: usize,
        threads: Threads,
        kind: Sweep,
        sweep: impl FnOnce(&mut PlanWork<T>) -> R + Send,
    ) -> R {
        let work = self.fit(work, nr, threads, kind);
        let timed = work.pick.take();
        // timed in the pool: building it on its first use and measuring
        // its section cost is no time of the way
        Self::in_pool(work.parallel, threads, || {
            let t = crate::clock::Instant::now();
            let r = sweep(work);
            if let Some((pick, ops)) = timed {
                let ns = t.elapsed().as_nanos() as f64;
                kind.ways::<T>(nr).record(pick, ns, ops);
                if crate::logging::enabled(crate::logging::LogLevel::Debug) {
                    crate::logging::debug(&format!(
                        "solve: sweep way {} took {:.2} ms",
                        pick.option,
                        crate::clock::ms(std::time::Duration::from_nanos(ns as u64))
                    ));
                }
            }
            r
        })
    }

    /// Forward sweep `L y = y` in place on `nr` row-major right-hand sides,
    /// on at most `threads` workers.
    pub fn forward(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>, threads: Threads) {
        self.sweep(work, nr, threads, Sweep::Forward, |work| {
            self.forward_block(nr, y, work)
        })
    }

    /// Backward sweep `L^T x = x` (or `U x = x` for a non-unit factor) in
    /// place on `nr` row-major right-hand sides, on at most `threads`
    /// workers.
    pub fn backward(&self, nr: usize, x: &mut [T], work: &mut PlanWork<T>, threads: Threads) {
        self.sweep(work, nr, threads, Sweep::Backward, |work| {
            self.backward_block(nr, x, work)
        })
    }

    /// Solve `L D L^T y = y` in place on the permuted, scaled right-hand
    /// side, on at most `threads` workers.
    pub fn solve_in_place(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        work: &mut PlanWork<T>,
        threads: Threads,
    ) -> Result<(), RslabError> {
        self.sweep(work, 1, threads, Sweep::Whole, |work| {
            self.solve_block_inner(f, y, 1, work)
        })
    }

    // -----------------------------------------------------------------------
    // Ancestor supernodes: a task tree, sections inside the large nodes
    // -----------------------------------------------------------------------
    //
    // The supernodes above the cut form the top of the elimination tree.
    // On the pool they run as a task tree (`top_forward_flow`,
    // `top_backward_flow`), with no level waiting for its slowest node and
    // no recursion along the tree's height: on a
    // 32 k near-field LU the level sweeps ran most levels (fewer nodes than
    // threads) one node after another, 16 threads slower than 4. A node's
    // products run as GEMMs ([`fwd_node_gemm`](Self::fwd_node_gemm),
    // [`bwd_node_gemm`](Self::bwd_node_gemm)), a large one's in row chunks on
    // the pool, and a wide node takes the leaf subtrees' terms in column
    // chunks. On one thread, or on a factor whose structure is no elimination
    // tree, the levels run in order (`top_forward`, `top_backward`), with the
    // same arithmetic per node and per column.

    /// Whether the ancestors run as tasks (`top_forward_flow`,
    /// `top_backward_flow`) rather than level by level.
    fn flows(&self, work: &PlanWork<T>) -> bool {
        work.parallel && !self.chain && !self.top.is_empty()
    }

    /// The forward sweep of the ancestors as a task tree: an ancestor runs
    /// once its children are done, with no level waiting for its slowest
    /// node. Every leaf of the tree starts a walk toward the root; a walk
    /// goes on to the parent when it finished the parent's last child to
    /// finish (the pending counts' acquire-release orders the children's
    /// writes before the parent), and ends otherwise. No task is allocated
    /// and nothing recurses: a walk is a loop. Before its sweep an ancestor
    /// takes into its columns what the
    /// leaf subtrees and its descendants above the cut accumulated for them
    /// (`pull`), in the order the level sweeps reduce them, so every column
    /// takes its terms in the same order and the result is the level
    /// sweeps' to the bit. Every ancestor keeps its accumulator to the end
    /// of the sweep, in one block. The subtrees' reduction is part of it.
    fn top_forward_flow(
        &self,
        nr: usize,
        y: &mut [T],
        work: &mut PlanWork<T>,
        trace: Option<&SweepTrace>,
    ) {
        let nt = self.top.len();
        let nsub = self.subtrees.len();
        work.top_acc.clear();
        work.top_acc.resize(self.top_acc_off[nt] * nr, T::zero());
        let ys = Shared(y.as_mut_ptr(), y.len());
        let accs = Shared(work.top_acc.as_mut_ptr(), work.top_acc.len());
        let pool = &work.pool;
        let subs = &work.subtrees;
        let node = |ti: usize, tr: Option<&NodeTimes>| {
            let s = self.top[ti];
            let (c0, w, _, _, _, _) = self.node(s);
            // SAFETY: see `Shared`. This task alone writes the columns of
            // `s` and its own accumulator; the accumulators it reads belong
            // to nodes that are done (its descendants, ordered before it by
            // the pending counts' acquire-release).
            let y = unsafe { ys.slice() };
            let pulls = &self.pull[self.pull_ptr[ti]..self.pull_ptr[ti + 1]];
            // the columns `j0..` of this node take every source's terms for them:
            // of a source's run, the slots whose rows lie in those columns
            let take = |j0: usize, yc: &mut [T]| {
                let (lo_r, hi_r) = ((c0 + j0) as u32, (c0 + j0 + yc.len() / nr) as u32);
                for &(src, lo, hi) in pulls {
                    let (src, lo, hi) = (src as usize, lo as usize, hi as usize);
                    let list = if src < nsub {
                        &self.subtrees[src].ext[lo..hi]
                    } else {
                        let d = self.top[src - nsub] as usize;
                        &self.rows[self.row_ptr[d] + lo..self.row_ptr[d] + hi]
                    };
                    let a = list.partition_point(|&r| r < lo_r);
                    let b = list.partition_point(|&r| r < hi_r);
                    if a == b {
                        continue;
                    }
                    let from = if src < nsub {
                        &subs[src].acc[(lo + a) * nr..(lo + b) * nr]
                    } else {
                        let o = (self.top_acc_off[src - nsub] + lo + a) * nr;
                        unsafe { std::slice::from_raw_parts(accs.ptr().add(o), (b - a) * nr) }
                    };
                    for (k, &r) in list[a..b].iter().enumerate() {
                        let i = (r - lo_r) as usize;
                        sub_assign(&mut yc[i * nr..(i + 1) * nr], &from[k * nr..(k + 1) * nr]);
                    }
                }
            };
            let yc = &mut y[c0 * nr..(c0 + w) * nr];
            timed(
                tr,
                |t| &t.take,
                || {
                    if pulls.len() * w * nr >= SOLVE_PULL_PAR_MIN {
                        // a wide node near the root takes the terms of every leaf
                        // subtree: its columns in chunks on the pool, each column
                        // summing its sources in order
                        crate::parallel::chunks_mut(
                            yc,
                            SOLVE_PULL_COLS * nr,
                            crate::parallel::Work::ns(f64::INFINITY),
                            |ci, c| take(ci * SOLVE_PULL_COLS, c),
                        );
                    } else {
                        take(0, yc);
                    }
                },
            );
            let (a0, a1) = (self.top_acc_off[ti] * nr, self.top_acc_off[ti + 1] * nr);
            let acc = unsafe { std::slice::from_raw_parts_mut(accs.ptr().add(a0), a1 - a0) };
            let mut sc = pool.take();
            self.fwd_node_gemm(s, nr, y, acc, true, &mut sc.t, tr);
            pool.give(sc);
        };
        let kids_ptr = &self.top_kids_ptr;
        let pending = &work.pending[..nt];
        for (t, p) in pending.iter().enumerate() {
            p.store((kids_ptr[t + 1] - kids_ptr[t]) as u32, Ordering::Relaxed);
        }
        let walk = |leaf: usize| {
            if kids_ptr[leaf + 1] > kids_ptr[leaf] {
                return;
            }
            let mut t = leaf;
            loop {
                SweepTrace::node(trace, t, |tr| node(t, tr));
                match self.top_parent[t] {
                    NONE => return,
                    p if pending[p as usize].fetch_sub(1, Ordering::AcqRel) == 1 => t = p as usize,
                    _ => return,
                }
            }
        };
        crate::parallel::for_each(nt, crate::parallel::Work::ns(f64::INFINITY), walk);
    }

    /// The backward sweep of the ancestors as a task tree: an ancestor runs
    /// when its parent is done (every column it reads is then final), then
    /// its children's trees side by side. Each node's
    /// arithmetic is the level sweep's. A chain of single children runs in
    /// a loop; the fork-join on the stack (no task is allocated) nests only
    /// where the tree branches, fewer than
    /// [`leaf_subtrees`](crate::SolveSettings::leaf_subtrees) times on any
    /// path whatever the tree's height: every ancestor holds more than that
    /// share of the work (`cut`), so each branch passed leaves more than it
    /// behind.
    fn top_backward_flow(
        &self,
        nr: usize,
        x: &mut [T],
        work: &mut PlanWork<T>,
        trace: Option<&SweepTrace>,
    ) {
        let xs = Shared(x.as_mut_ptr(), x.len());
        let pool = &work.pool;
        let node = |ti: usize| {
            let s = self.top[ti];
            // SAFETY: see `Shared`; this task alone writes the columns of
            // `s`, and the ancestors' columns it reads are final.
            let x = unsafe { xs.slice() };
            let mut sc = pool.take();
            SweepTrace::node(trace, ti, |tr| {
                self.bwd_node_gemm(s, nr, x, true, &mut sc.g, tr)
            });
            pool.give(sc);
        };
        /// The tree under ancestor `ti`: its branches side by side.
        fn tree(ti: usize, node: &(dyn Fn(usize) + Sync), kids_ptr: &[usize], kids: &[u32]) {
            let mut t = ti;
            loop {
                node(t);
                match &kids[kids_ptr[t]..kids_ptr[t + 1]] {
                    [] => return,
                    [only] => t = *only as usize,
                    k => {
                        return crate::parallel::each(
                            k.len(),
                            crate::parallel::Work::ns(f64::INFINITY),
                            |c| tree(k[c] as usize, node, kids_ptr, kids),
                        )
                    }
                }
            }
        }
        let roots = &self.top_roots;
        crate::parallel::each(roots.len(), crate::parallel::Work::ns(f64::INFINITY), |c| {
            tree(roots[c] as usize, &node, &self.top_kids_ptr, &self.top_kids)
        });
    }

    fn top_forward(
        &self,
        nr: usize,
        y: &mut [T],
        work: &mut PlanWork<T>,
        trace: Option<&SweepTrace>,
    ) {
        let par = work.parallel;
        let nt = crate::parallel::threads();
        let shared = Shared(y.as_mut_ptr(), y.len());
        for (li, level) in self.top_levels.iter().enumerate().rev() {
            let node_par = !par || level.len() >= nt || nt == 1;
            // One accumulator block for the level, a disjoint slice per node.
            let ptr = &self.top_acc_ptr[li];
            work.top_acc.clear();
            work.top_acc.resize(ptr[level.len()] * nr, T::zero());
            let accs = Shared(work.top_acc.as_mut_ptr(), work.top_acc.len());
            let sweep = |i: usize, s: u32, y: &mut [T], par: bool, sc: &mut Scratch<T>| {
                // SAFETY: node `i` owns its range of the level's block.
                let acc = unsafe { &mut accs.slice()[ptr[i] * nr..ptr[i + 1] * nr] };
                let ti = self.top_index[s as usize] as usize;
                SweepTrace::node(trace, ti, |tr| {
                    self.fwd_node_gemm(s, nr, y, acc, par, &mut sc.t, tr)
                });
            };
            if node_par {
                // a node's own sections nest inside: estimated, not measured
                let sweep_work = Self::sweep_work(self.nodes_work(level), nr);
                Self::each(
                    par,
                    sweep_work,
                    level,
                    &mut work.level[..level.len()],
                    |i, &s, sc| {
                        // SAFETY: see `Shared`; the node writes only its own
                        // columns, its off-block rows go to its accumulator. A
                        // large node keeps its row tasks inside the level's
                        // (the same arithmetic either way): one large node per
                        // level otherwise ran on one thread while the others
                        // finished their small ones.
                        let yv = unsafe { shared.slice() };
                        sweep(i, s, yv, par, sc);
                    },
                );
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

    fn top_backward(
        &self,
        nr: usize,
        x: &mut [T],
        work: &mut PlanWork<T>,
        trace: Option<&SweepTrace>,
    ) {
        let par = work.parallel;
        let nt = crate::parallel::threads();
        let shared = Shared(x.as_mut_ptr(), x.len());
        for level in &self.top_levels {
            let node_par = !par || level.len() >= nt || nt == 1;
            let sweep = |s: u32, x: &mut [T], par: bool, sc: &mut Scratch<T>| {
                let ti = self.top_index[s as usize] as usize;
                SweepTrace::node(trace, ti, |tr| {
                    self.bwd_node_gemm(s, nr, x, par, &mut sc.g, tr)
                });
            };
            if node_par {
                // a node's own sections nest inside: estimated, not measured
                let sweep_work = Self::sweep_work(self.nodes_work(level), nr);
                Self::each(
                    par,
                    sweep_work,
                    level,
                    &mut work.level[..level.len()],
                    |_, &s, sc| {
                        // SAFETY: see `Shared`; the node writes only its own
                        // columns and reads final ancestor columns. Its row
                        // tasks nest as in the forward sweep.
                        let xv = unsafe { shared.slice() };
                        sweep(s, xv, par, sc);
                    },
                );
            } else {
                for &s in level {
                    sweep(s, x, true, &mut work.apex);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Blocked right-hand sides (row-major `n x nrhs`)
    // -----------------------------------------------------------------------

    /// The forward sweep of node `s` for `nr` right-hand sides with its dense
    /// products as GEMMs, block by block of the packed panel ([`packed_blocks`]):
    /// each block's triangle on the register tiles, then the rows below it,
    /// the node's own and its rows below the node (`L_off X`, the blocks'
    /// products summed into `t`), in one product (in row chunks on the pool
    /// when `par`, [`node_chunks`](Self::node_chunks)), each row of `t` taken off `y` where the node writes it
    /// directly and into the accumulator `acc` at its slot otherwise. A row's
    /// sums run in the products' order, so a column of a block agrees with its
    /// single solve to rounding, not to the bit.
    #[allow(clippy::too_many_arguments)]
    fn fwd_node_gemm(
        &self,
        s: u32,
        nr: usize,
        y: &mut [T],
        acc: &mut [T],
        par: bool,
        t: &mut Vec<T>,
        tr: Option<&NodeTimes>,
    ) {
        let (c0, w, r0, m, _, _) = self.node(s);
        self.fwd_node_core(s, nr, &mut y[c0 * nr..(c0 + w) * nr], par, t, tr);
        if m == 0 {
            return;
        }
        let rows = &self.rows[r0..r0 + m];
        let slots = &self.ext_slot[r0..r0 + m];
        timed(
            tr,
            |t| &t.scatter,
            || {
                for (i, ti) in t[..m * nr].chunks_exact(nr).enumerate() {
                    if slots[i] == NONE {
                        let r = rows[i] as usize;
                        sub_assign(&mut y[r * nr..(r + 1) * nr], ti);
                    } else {
                        let sl = slots[i] as usize;
                        add_assign(&mut acc[sl * nr..(sl + 1) * nr], ti);
                    }
                }
            },
        );
    }

    /// The node part of [`fwd_node_gemm`](Self::fwd_node_gemm), shared with
    /// the reach-limited sweep of sparse right-hand sides: the triangle of
    /// node `s` on its own rows `yb` (`nr` wide), then `L_off X` into
    /// `t[..m * nr]` (left untouched when the node has no off-block rows). The
    /// caller takes `t` off the rows the node's off-block rows map to. Timed
    /// into `tr` where traced.
    pub(super) fn fwd_node_core(
        &self,
        s: u32,
        nr: usize,
        yb: &mut [T],
        par: bool,
        t: &mut Vec<T>,
        tr: Option<&NodeTimes>,
    ) {
        let (c0, w, _, m, _, panel) = self.node(s);
        let dinv = self.diag_inv.get(c0..c0 + w).unwrap_or(&[]);
        let mode = crate::dense::gemm_backend::GemmMode::new(
            gemm::Parallelism::None,
            &crate::KernelSettings::default(),
        );
        let (one, minus) = (T::one(), T::from_real(-1.0));
        // the first block's product overwrites every entry of `t`: no zeroing, only growth
        if m > 0 && t.len() < m * nr {
            t.resize(m * nr, T::zero());
        }
        for (b, (jb, je, off, ld)) in packed_blocks(w, m).enumerate() {
            let d = dinv.get(jb..je).unwrap_or(&[]);
            timed(
                tr,
                |t| &t.tri,
                || {
                    tri_rows(
                        &mut yb[jb * nr..je * nr],
                        &panel[off..],
                        d,
                        ld,
                        nr,
                        0,
                        je - jb,
                    )
                },
            );
            // the rows below the block's triangle take its columns: the node's own
            // `je..w` off `yb`, its rows below the node into `t`
            let k = je - jb;
            let own = self.node_chunks(par, w - je, k, nr);
            let below = self.node_chunks(par, m, k, nr);
            let (pp, yp, tp) = (
                Shared(panel.as_ptr() as *mut T, panel.len()),
                Shared(yb.as_mut_ptr(), yb.len()),
                Shared(t.as_mut_ptr(), t.len()),
            );
            run_chunks(own + below, tr, |i| {
                let (dst, at, read, lhs, beta, r) = if i < own {
                    let r = chunk_rows(i, own, w - je);
                    (
                        yp.ptr(),
                        (je + r.start) * nr,
                        true,
                        off + k + r.start,
                        minus,
                        r,
                    )
                } else {
                    let r = chunk_rows(i - own, below, m);
                    (
                        tp.ptr(),
                        r.start * nr,
                        b > 0,
                        off + w - jb + r.start,
                        one,
                        r,
                    )
                };
                // SAFETY: a chunk writes its own rows of `yb` (below `je`) or of `t`;
                // the block's solved rows `jb..je` of `yb` and the panel are only read.
                unsafe {
                    crate::dense::gemm_backend::gemm(
                        r.len(),
                        nr,
                        k,
                        dst.add(at),
                        1,
                        nr as isize,
                        read,
                        pp.ptr().add(lhs) as *const T,
                        ld as isize,
                        1,
                        yp.ptr().add(jb * nr) as *const T,
                        1,
                        nr as isize,
                        one,
                        beta,
                        false,
                        false,
                        false,
                        mode,
                    );
                }
            });
        }
    }

    /// The backward sweep of node `s` for `nr` right-hand sides with its dense
    /// products as GEMMs, block by block of the packed panel ([`packed_blocks`]):
    /// the final values of the rows below the node gathered and taken off the
    /// node's rows as one product `L_off^T G`, then the triangle from the last
    /// block up: each block takes off the rows below it in the node by one
    /// product and solves its own triangle column by column, last to first, each
    /// column scaled after its terms. The products run in row chunks on the
    /// pool when `par` ([`node_chunks`](Self::node_chunks)), within a block's rows.
    fn bwd_node_gemm(
        &self,
        s: u32,
        nr: usize,
        x: &mut [T],
        par: bool,
        g: &mut Vec<T>,
        tr: Option<&NodeTimes>,
    ) {
        let (c0, w, r0, m, _, panel) = self.node(s);
        let dinv = self.diag_inv.get(c0..c0 + w).unwrap_or(&[]);
        let mode = crate::dense::gemm_backend::GemmMode::new(
            gemm::Parallelism::None,
            &crate::KernelSettings::default(),
        );
        let (one, minus) = (T::one(), T::from_real(-1.0));
        if m > 0 {
            timed(
                tr,
                |t| &t.take,
                || {
                    g.clear();
                    for &r in &self.rows[r0..r0 + m] {
                        g.extend_from_slice(&x[r as usize * nr..(r as usize + 1) * nr]);
                    }
                },
            );
        }
        let xb = &mut x[c0 * nr..(c0 + w) * nr];
        if m > 0 {
            // the node's rows `X[i] -= sum_r L[r, i] G[r]`, each block's rows in its
            // chunks: every full block cut alike, the last one by its own rows
            let nb = w.div_ceil(SOLVE_DIAG_BLOCK);
            let last_rows = w - (nb - 1) * SOLVE_DIAG_BLOCK;
            let (full, last) = (
                self.node_chunks(par, SOLVE_DIAG_BLOCK, m, nr),
                self.node_chunks(par, last_rows, m, nr),
            );
            let (pp, gp, xp) = (
                Shared(panel.as_ptr() as *mut T, panel.len()),
                Shared(g.as_mut_ptr(), g.len()),
                Shared(xb.as_mut_ptr(), xb.len()),
            );
            run_chunks((nb - 1) * full + last, tr, |i| {
                let (b, q, count) = match i.checked_sub((nb - 1) * full) {
                    None => (i / full, i % full, full),
                    Some(q) => (nb - 1, q, last),
                };
                let Some((jb, je, off, ld)) = packed_blocks(w, m).nth(b) else {
                    return;
                };
                let r = chunk_rows(q, count, je - jb);
                // SAFETY: the chunk writes its own rows of the node; the panel and
                // the gathered rows are only read.
                unsafe {
                    crate::dense::gemm_backend::gemm(
                        r.len(),
                        nr,
                        m,
                        xp.ptr().add((jb + r.start) * nr),
                        1,
                        nr as isize,
                        true,
                        pp.ptr().add(off + w - jb + r.start * ld) as *const T,
                        1,
                        ld as isize,
                        gp.ptr() as *const T,
                        1,
                        nr as isize,
                        one,
                        minus,
                        false,
                        false,
                        false,
                        mode,
                    );
                }
            });
        }
        for (jb, je, off, ld) in packed_blocks(w, m).rev() {
            if je < w {
                let k = w - je;
                let count = self.node_chunks(par, je - jb, k, nr);
                let (pp, xp) = (
                    Shared(panel.as_ptr() as *mut T, panel.len()),
                    Shared(xb.as_mut_ptr(), xb.len()),
                );
                run_chunks(count, tr, |q| {
                    let r = chunk_rows(q, count, je - jb);
                    // SAFETY: the chunk writes its own rows of `jb..je`, rows `je..w`
                    // are read: disjoint; the block's rows `je..w` lie in it.
                    unsafe {
                        crate::dense::gemm_backend::gemm(
                            r.len(),
                            nr,
                            k,
                            xp.ptr().add((jb + r.start) * nr),
                            1,
                            nr as isize,
                            true,
                            pp.ptr().add(off + je - jb + r.start * ld) as *const T,
                            1,
                            ld as isize,
                            xp.ptr().add(je * nr) as *const T,
                            1,
                            nr as isize,
                            one,
                            minus,
                            false,
                            false,
                            false,
                            mode,
                        );
                    }
                });
            }
            let block = &panel[off..];
            timed(
                tr,
                |t| &t.tri,
                || {
                    for k in (jb..je).rev() {
                        let col = &block[(k - jb) * ld..(k - jb + 1) * ld];
                        let (head, tail) = xb.split_at_mut((k + 1) * nr);
                        let xk = &mut head[k * nr..];
                        for i in k + 1..je {
                            let l = col[i - jb];
                            let xi = &tail[(i - k - 1) * nr..(i - k) * nr];
                            for (a, &v) in xk.iter_mut().zip(xi) {
                                *a = *a - l * v;
                            }
                        }
                        if let Some(&d) = dinv.get(k) {
                            for a in xk.iter_mut() {
                                *a = *a * d;
                            }
                        }
                    }
                },
            );
        }
    }

    /// Solve `L D L^T Y = Y` in place on a row-major `n x nrhs` block, on at
    /// most `threads` workers.
    pub fn solve_block_in_place(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        nr: usize,
        work: &mut PlanWork<T>,
        threads: Threads,
    ) -> Result<(), RslabError> {
        self.sweep(work, nr, threads, Sweep::Whole, |work| {
            self.solve_block_inner(f, y, nr, work)
        })
    }

    fn solve_block_inner(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        nr: usize,
        work: &mut PlanWork<T>,
    ) -> Result<(), RslabError> {
        debug_assert_eq!(y.len(), self.n * nr);
        let mut phases = PhaseTrace::start();
        self.forward_block(nr, y, work);
        phases.lap("forward");
        solve_diagonal(f, y, nr, 0..f.n)?;
        phases.lap("diag");
        self.backward_block(nr, y, work);
        phases.lap("backward");
        phases.finish("solve-block", work.parallel);
        Ok(())
    }

    fn forward_block(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let shared = Shared(y.as_mut_ptr(), y.len());
        let mut phases = PhaseTrace::start();
        let trace = SweepTrace::start();
        let sub_tr = trace.as_ref().map(|t| &t.subtrees);
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        let sweep_work = Self::sweep_work(self.subtrees_work(), nr);
        Self::each(par, sweep_work, &self.subtrees, scratch, |_, st, sc| {
            timed(
                sub_tr,
                |t| &t.zero,
                || {
                    sc.acc.clear();
                    sc.acc.resize(st.ext.len() * nr, T::zero());
                },
            );
            // SAFETY: see `Shared`.
            let yv = unsafe { shared.slice() };
            for &s in &st.nodes {
                self.fwd_node_gemm(s, nr, yv, &mut sc.acc, false, &mut sc.t, sub_tr);
            }
        });
        phases.lap("fwd-subtrees");
        if self.flows(work) {
            self.top_forward_flow(nr, y, work, trace.as_ref());
            phases.lap("fwd-top");
        } else {
            for (st, sc) in self.subtrees.iter().zip(&work.subtrees) {
                for (i, &c) in st.ext.iter().enumerate() {
                    let yr = &mut y[c as usize * nr..(c as usize + 1) * nr];
                    sub_assign(yr, &sc.acc[i * nr..(i + 1) * nr]);
                }
            }
            phases.lap("fwd-reduce");
            self.top_forward(nr, y, work, trace.as_ref());
            phases.lap("fwd-top");
        }
        phases.finish("forward-block", work.parallel);
        if let Some(trace) = trace {
            trace.finish(self, "fwd");
        }
    }

    fn backward_block(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let mut phases = PhaseTrace::start();
        let trace = SweepTrace::start();
        let sub_tr = trace.as_ref().map(|t| &t.subtrees);
        if self.flows(work) {
            self.top_backward_flow(nr, y, work, trace.as_ref());
        } else {
            self.top_backward(nr, y, work, trace.as_ref());
        }
        phases.lap("bwd-top");
        let shared = Shared(y.as_mut_ptr(), y.len());
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        let sweep_work = Self::sweep_work(self.subtrees_work(), nr);
        Self::each(par, sweep_work, &self.subtrees, scratch, |_, st, sc| {
            // SAFETY: see `Shared`.
            let xv = unsafe { shared.slice() };
            for &s in st.nodes.iter().rev() {
                self.bwd_node_gemm(s, nr, xv, false, &mut sc.g, sub_tr);
            }
        });
        phases.lap("bwd-subtrees");
        phases.finish("backward-block", work.parallel);
        if let Some(trace) = trace {
            trace.finish(self, "bwd");
        }
    }
}

/// `y -= x`.
#[inline(always)]
fn sub_assign<T: Scalar>(y: &mut [T], x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi - xi;
    }
}

/// `y += x`.
#[inline(always)]
fn add_assign<T: Scalar>(y: &mut [T], x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi + xi;
    }
}

/// Per-phase wall times of one solve, emitted at the `debug` log level
/// (nothing is measured otherwise).
struct PhaseTrace {
    t0: Option<crate::clock::Instant>,
    laps: Vec<(&'static str, f64)>,
}

impl PhaseTrace {
    fn start() -> Self {
        let on = crate::logging::enabled(crate::logging::LogLevel::Debug);
        Self {
            t0: on.then(crate::clock::Instant::now),
            laps: Vec::new(),
        }
    }

    fn lap(&mut self, name: &'static str) {
        if let Some(t0) = self.t0 {
            self.laps.push((name, crate::clock::ms(t0.elapsed())));
            self.t0 = Some(crate::clock::Instant::now());
        }
    }

    /// Log the laps of a sweep, run on the pool when `parallel`.
    fn finish(&self, what: &str, parallel: bool) {
        if self.t0.is_some() {
            let total: f64 = self.laps.iter().map(|l| l.1).sum();
            let parts: Vec<String> = self
                .laps
                .iter()
                .map(|(n, ms)| format!("{n}={ms:.2}ms"))
                .collect();
            crate::logging::debug(&format!(
                "{what}: {total:.2}ms threads={} {}",
                if parallel {
                    crate::parallel::threads()
                } else {
                    1
                },
                parts.join(" ")
            ));
        }
    }
}

/// Wall times inside a sweep, in ns, of one ancestor or summed over the
/// leaf subtrees' tasks ([`SweepTrace`]); only at the `debug` log level.
#[derive(Default)]
pub(super) struct NodeTimes {
    /// The node's whole sweep.
    total: AtomicU64,
    /// The take of the terms for its columns (forward) or the gather of its
    /// rows below (backward).
    take: AtomicU64,
    /// The zeroing of a leaf subtree's accumulator.
    zero: AtomicU64,
    /// The triangles on the node's thread.
    tri: AtomicU64,
    /// The products on the node's thread, whole.
    gemm: AtomicU64,
    /// The products' chunks in the pool sections, summed.
    pooled: AtomicU64,
    /// The scatter of `L_off X` into the right-hand sides and accumulators.
    scatter: AtomicU64,
    /// The pool sections of the products: their count, wall and chunks.
    sections: AtomicU64,
    section: AtomicU64,
    chunks: AtomicU64,
}

/// `f()`, its wall time added to the counter `at` of `tr` where traced.
#[inline]
fn timed<R>(
    tr: Option<&NodeTimes>,
    at: impl Fn(&NodeTimes) -> &AtomicU64,
    f: impl FnOnce() -> R,
) -> R {
    match tr {
        None => f(),
        Some(tr) => {
            let t0 = crate::clock::Instant::now();
            let r = f();
            at(tr).fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            r
        }
    }
}

/// The node times of one sweep, emitted at the `debug` log level
/// (`examples/lu_solve_timing.rs` aggregates them): a line per ancestor
/// with its shape, entries and whether it lies on the critical path, and a
/// line for the leaf subtrees.
#[derive(Default)]
struct SweepTrace {
    nodes: std::sync::Mutex<Vec<(usize, NodeTimes)>>,
    subtrees: NodeTimes,
}

impl SweepTrace {
    /// A trace when the `debug` log level is on.
    fn start() -> Option<Self> {
        crate::logging::enabled(crate::logging::LogLevel::Debug).then(Self::default)
    }

    /// `f(times)` for ancestor `ti`, its times kept where traced.
    fn node(trace: Option<&Self>, ti: usize, f: impl FnOnce(Option<&NodeTimes>)) {
        let Some(trace) = trace else {
            return f(None);
        };
        let times = NodeTimes::default();
        timed(Some(&times), |t| &t.total, || f(Some(&times)));
        let mut nodes = trace.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes.push((ti, times));
    }

    fn finish<T: Scalar>(&self, plan: &SolvePlan<T>, what: &str) {
        let line = |t: &NodeTimes| {
            let v = |a: &AtomicU64| a.load(Ordering::Relaxed);
            format!(
                "total={} take={} zero={} tri={} gemm={} pooled={} scatter={} sections={} section={} chunks={}",
                v(&t.total),
                v(&t.take),
                v(&t.zero),
                v(&t.tri),
                v(&t.gemm),
                v(&t.pooled),
                v(&t.scatter),
                v(&t.sections),
                v(&t.section),
                v(&t.chunks)
            )
        };
        let path = plan.critical_path();
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        for (ti, t) in nodes.iter() {
            let s = plan.top[*ti];
            let (_, w, _, m, _, _) = plan.node(s);
            let (v0, v1) = plan.val_rng[s as usize];
            crate::logging::debug(&format!(
                "solve node {what} w={w} m={m} entries={} path={} {}",
                v1 - v0,
                u8::from(path.contains(ti)),
                line(t)
            ));
        }
        crate::logging::debug(&format!(
            "solve subtrees {what} entries={} {}",
            plan.subtree_work,
            line(&self.subtrees)
        ));
    }
}

/// The unit-lower triangle of the columns `jb..je` (scaled by `diag_inv` where it is given,
/// the `U^T` of an LU) on the row-major right-hand sides `v` (`nr` wide), ROW by row: row
/// `i` takes `- l_ik v_k` for `k` ascending from `jb`, then its scale. That is the very
/// sequence of operations the column-oriented sweep applies to it (column `k` subtracts its
/// multiple from every later row, in ascending `k`), so the result is the same to the bit;
/// the row form keeps a tile of [`TILE_ROWS`] rows by up to four right-hand sides in
/// registers across all columns above it, where the column form read and wrote a row of
/// `nr` values per entry.
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

/// [`tri_rows`] for the right-hand sides `c0..c0 + W`.
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
        // the rows above the tile are final
        for k in jb..i {
            let vk = &v[k * nr + c0..k * nr + c0 + W];
            let col = &panel[k * ld + i..k * ld + i + rn];
            for (ar, &l) in acc.iter_mut().zip(col) {
                for (a, &x) in ar.iter_mut().zip(vk) {
                    *a = *a - l * x;
                }
            }
        }
        // inside the tile, each row once the rows before it are final
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

/// The column blocks of a node of width `w` with `m` off-block rows in the plan's packed
/// panel: `(jb, je, offset, ld)`, block `jb..je` the dense column-major `ld x (je - jb)`
/// block at `offset` from the node's start, holding rows `jb..w + m` of its columns (`ld = w +
/// m - jb`). A sweep reads no entry above a block's rows (the triangle in column blocks, the
/// rows below in products), so those are not stored: of the diagonal block's unused upper
/// triangle only the part inside each block's own square remains.
fn packed_blocks(
    w: usize,
    m: usize,
) -> impl DoubleEndedIterator<Item = (usize, usize, usize, usize)> + ExactSizeIterator {
    let nb = SOLVE_DIAG_BLOCK;
    // `k` full blocks before block `k`: `sum_{i < k} nb (w + m - i nb)`
    let before = move |k: usize| k * nb * (w + m) - nb * nb * (k * k.saturating_sub(1) / 2);
    (0..w.div_ceil(nb)).map(move |k| {
        let (jb, je) = (k * nb, ((k + 1) * nb).min(w));
        (jb, je, before(k), w + m - jb)
    })
}

/// The rows of chunk `i` of `count` over `rows` rows, the chunks as even as
/// they can be.
fn chunk_rows(i: usize, count: usize, rows: usize) -> std::ops::Range<usize> {
    i * rows / count..(i + 1) * rows / count
}

/// `f(i)` for the `count` chunks of a node's product
/// ([`node_chunks`](SolvePlan::node_chunks)): each its own task on the pool
/// where there are several; timed into `tr` where traced.
fn run_chunks(count: usize, tr: Option<&NodeTimes>, f: impl Fn(usize) + Sync + Send) {
    if count > 1 {
        let f = |i| timed(tr, |t| &t.pooled, || f(i));
        timed(
            tr,
            |t| &t.section,
            || crate::parallel::each(count, crate::parallel::Work::ns(f64::INFINITY), f),
        );
        if let Some(tr) = tr {
            tr.sections.fetch_add(1, Ordering::Relaxed);
            tr.chunks.fetch_add(count as u64, Ordering::Relaxed);
        }
    } else {
        (0..count).for_each(|i| timed(tr, |t| &t.gemm, || f(i)));
    }
}

/// Entries of a node's packed panel ([`packed_blocks`]).
pub(crate) fn packed_len(w: usize, m: usize) -> usize {
    packed_blocks(w, m)
        .map(|(jb, je, _, ld)| ld * (je - jb))
        .sum()
}

/// Pack the panels of `vals` (supernode `s` the column-major `(w + m) x w` block at
/// `full_off[s]`, `sn_col` and `row_ptr` giving `w` and `m`) in place into the plan's layout
/// ([`packed_blocks`]) and return each one's packed range. The panels go in the order they
/// lie in, every column moves to an offset no larger than its own and the columns go in
/// order, so no source is overwritten before it is read. The buffer keeps its room.
fn pack_panels<T: Copy>(
    vals: &mut Vec<T>,
    sn_col: &[u32],
    row_ptr: &[usize],
    full_off: &[usize],
) -> Vec<(usize, usize)> {
    let ns = sn_col.len() - 1;
    let mut order: Vec<usize> = (0..ns).collect();
    if full_off.windows(2).any(|p| p[0] > p[1]) {
        order.sort_unstable_by_key(|&s| full_off[s]);
    }
    let mut rng = vec![(0usize, 0usize); ns];
    let mut at = 0usize;
    for s in order {
        let w = (sn_col[s + 1] - sn_col[s]) as usize;
        let m = row_ptr[s + 1] - row_ptr[s];
        let ld = w + m;
        for (jb, je, off, ldb) in packed_blocks(w, m) {
            for k in jb..je {
                let src = full_off[s] + k * ld + jb;
                vals.copy_within(src..src + ldb, at + off + (k - jb) * ldb);
            }
        }
        rng[s] = (at, at + packed_len(w, m));
        at = rng[s].1;
    }
    vals.truncate(at);
    rng
}

/// `D z = y` for the block diagonal (1x1 and 2x2 pivots) of the columns
/// `cols`, on `nr` right-hand sides stored row-major (`y` holds the rows of
/// `cols`). A 2x2 block lies inside one supernode's fully-summed block, so a
/// range of whole supernodes never cuts one.
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
    use crate::{CscMatrix, LdltSolver, SolverSettings};

    /// A 2D grid Laplacian shifted to be indefinite (2x2 pivots appear).
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

    /// A 3D grid Laplacian: wide separator supernodes exercise the blocked,
    /// parallel ancestor kernels.
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

    fn residual(a: &CscMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.n;
        let mut r = b.to_vec();
        for j in 0..n {
            for e in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[e];
                r[i] -= a.values[e] * x[j];
                if i != j {
                    r[j] -= a.values[e] * x[i];
                }
            }
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        rn / bn
    }

    #[test]
    fn plan_solve_matches_scalar_kernel_and_is_thread_invariant() {
        let cases: Vec<(String, CscMatrix<f64>)> = vec![
            ("grid 7".into(), grid(7, 0.0)),
            ("grid 40".into(), grid(40, 0.0)),
            ("grid 40 indefinite".into(), grid(40, 3.7)),
            ("grid3d 26".into(), grid3d(26)),
        ];
        for (name, a) in &cases {
            let (m, shift) = (name.as_str(), 0.0);
            let n = a.n;
            let opts = SolverSettings::default()
                .with_threads(1)
                .with_ordering(crate::OrderingMethod::MetisND);
            let s = LdltSolver::factor(a, &opts).unwrap();
            let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
            let x1 = s.solve(&b).unwrap();
            assert!(residual(a, &x1, &b) < 1e-10, "residual m={m} shift={shift}");
            // Block solve, columns must agree with single solves.
            let nrhs = 3;
            let bb: Vec<f64> = (0..n * nrhs)
                .map(|k| ((k * 13) % 11) as f64 - 5.0)
                .collect();
            let xb = s.solve_many(&bb, nrhs).unwrap();
            for c in 0..nrhs {
                let bc: Vec<f64> = (0..n).map(|i| bb[c * n + i]).collect();
                let xc = s.solve(&bc).unwrap();
                for i in 0..n {
                    assert!((xb[c * n + i] - xc[i]).abs() <= 1e-9 * (1.0 + xc[i].abs()));
                }
            }
            // Bit-identical for every thread budget: the factor's own, and
            // the ambient pool's.
            for threads in [1usize, 2, 5] {
                let st = LdltSolver::factor(a, &opts.clone().with_threads(threads)).unwrap();
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
                let sa = pool.install(|| LdltSolver::factor(a, &ambient).unwrap());
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
        // Unit `L` of three 1-column supernodes on the tree 0 -> 2, 1 -> 2,
        // supernode 0 holding row 1 (a sibling, not an ancestor):
        // L = [[1, 0, 0], [0.5, 1, 0], [0.25, -2, 1]].
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
        // L x = b and L^T x = c for x = [1, 2, 3].
        let mut y = vec![1.0, 2.5, -0.75];
        plan.forward(1, &mut y, &mut work, crate::Threads::Fixed(1));
        assert_eq!(y, [1.0, 2.0, 3.0]);
        let mut y = vec![2.75, -4.0, 3.0];
        plan.backward(1, &mut y, &mut work, crate::Threads::Fixed(1));
        assert_eq!(y, [1.0, 2.0, 3.0]);
    }

    /// Unsymmetric convection-diffusion grid for the LU plans.
    fn convdiff(m: usize) -> crate::GeneralCsc<f64> {
        let n = m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            // Column j holds the entries A[i, j]; A[i, j] for neighbors i.
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

    fn residual_general(a: &crate::GeneralCsc<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.n;
        let mut r = b.to_vec();
        for j in 0..n {
            for e in a.col_ptr[j]..a.col_ptr[j + 1] {
                r[a.row_idx[e]] -= a.values[e] * x[j];
            }
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        rn / bn
    }

    /// The ancestors as tasks (`top_forward_flow`, `top_backward_flow`) solve
    /// as the level sweeps do, to the bit: a tree cut into few subtrees, so
    /// most of it lies above the cut, every sweep on the pool, against one
    /// thread (the level sweeps); LU and LDL^T, one and several right-hand
    /// sides, enough of them that the large nodes run their products in row
    /// chunks ([`node_chunks`](super::SolvePlan::node_chunks)).
    #[test]
    fn ancestor_tasks_solve_as_the_levels_do() {
        use crate::{LuSolver, OrderingMethod};
        let opts = SolverSettings {
            solve: crate::SolveSettings {
                leaf_subtrees: 4,
                par_min_work: Some(0),
                ..crate::SolveSettings::default()
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
        // an ancestor whose first block's products below it run in several chunks
        use super::{SolvePlan, SOLVE_DIAG_BLOCK};
        let chunked = |p: &SolvePlan<f64>| {
            p.top.iter().any(|&s| {
                let (_, w, _, m, _, _) = p.node(s);
                let k = w.min(SOLVE_DIAG_BLOCK);
                p.node_chunks(true, w - k + m, k, nrhs) > 1 || p.node_chunks(true, k, m, nrhs) > 1
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
            assert!(residual_general(&a, &x1, &b) < 1e-10, "residual m={m}");
            let nrhs = 3;
            let bb: Vec<f64> = (0..n * nrhs)
                .map(|k| ((k * 13) % 11) as f64 - 5.0)
                .collect();
            let xb = s.solve_many(&bb, nrhs).unwrap();
            for c in 0..nrhs {
                let bc: Vec<f64> = (0..n).map(|i| bb[c * n + i]).collect();
                let xc = s.solve(&bc).unwrap();
                for i in 0..n {
                    assert!((xb[c * n + i] - xc[i]).abs() <= 1e-9 * (1.0 + xc[i].abs()));
                }
            }
            // Refinement runs through the plans too.
            let (xr, out) = s
                .solve_refined(&a, &b, &crate::RefinePolicy::steps(2))
                .unwrap();
            assert!(out.steps <= 2);
            assert!(residual_general(&a, &xr, &b) < 1e-12);
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
