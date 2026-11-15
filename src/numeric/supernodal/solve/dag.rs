//! The ancestors' sweeps on the pool: a task DAG over the tiles of their dense
//! nodes, each tile [`SOLVE_DIAG_BLOCK`] columns of a node's packed panel
//! ([`packed_blocks`]).
//!
//! Forward, per ancestor with `nb` tiles and `nc` row chunks below it
//! ([`SolvePlan::node_chunks`] at the depth of one tile):
//! - `Take(b)`: tile `b`'s columns take the terms the leaf subtrees and the
//!   descendants above the cut accumulated for them, in the order of `pull`;
//!   open while a child's row chunk that reaches the tile, or a child's own
//!   descendants, are not done.
//! - `Tri(b)`: the triangle of tile `b` ([`tri_rows`]), once every earlier
//!   tile's product reached its rows (tile 0: once it took its terms).
//! - `Own(g, s)`: tile `g`'s rows take the solved tile `s < g` ([`fwd_own`]),
//!   after `Tri(s)` and after `Own(g, s - 1)` (`Take(g)` for `s = 0`).
//! - `Below(c, s)`: chunk `c` of the rows below takes tile `s` into the
//!   node's accumulator ([`fwd_below`]), after `Tri(s)` and `Below(c, s - 1)`.
//!   The last one done completes the chunk; a node is complete once its
//!   chunks and its children are.
//!
//! The critical path through a node is its triangles, one tile product each
//! between them and the last tile's row chunk; the parent's first tile starts
//! once the chunks that reach it are done. Backward, per ancestor, once its
//! parent is complete (its rows below are final), in the arithmetic of
//! [`bwd_node_gemm`](SolvePlan::bwd_node_gemm):
//! - `Start`: the rows below gathered into the node's accumulator block.
//! - `Off(b, q)`: chunk `q` of tile `b`'s rows takes the rows below
//!   ([`bwd_off`]).
//! - `In(b, q)`: chunk `q` of tile `b`'s rows takes the solved tiles after it
//!   in one product ([`bwd_in`]), after `Off(b, ..)` and `Tri(b + 1)`.
//! - `Tri(b)`: the triangle of tile `b` ([`tri_up`]) after `In(b, ..)`; the last
//!   (`b = 0`) starts the children.
//!
//! Every row takes its terms in the order of the level sweeps (a tile's
//! products in tile order, the terms of the sources in `pull` order), each
//! product cut by rows only, so the solution is the one of a single thread to
//! the bit whatever runs when. A task counts its open inputs; the task that
//! closes the last one runs it, the first such on its own thread and the rest
//! spawned in the scope, so the critical successor (the next tile's product
//! after a triangle, the triangle after its last product) runs where its
//! inputs were made.

use super::{
    bwd_in, bwd_off, chunk_align, chunk_rows, fwd_below, fwd_own, packed_blocks, sub_assign,
    tri_rows, tri_up, NodeTimes, PlanWork, Shared, SolvePlan, SweepTrace, NONE,
};
use crate::constants::SOLVE_DIAG_BLOCK;
use crate::parallel::Scope;
use crate::scalar::Scalar;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// The open inputs of one sweep's tasks ([`SolvePlan::forward_dag`],
/// [`SolvePlan::backward_dag`]), laid out anew by each sweep and kept across
/// them, so a warm sweep allocates nothing.
#[derive(Default)]
pub(crate) struct DagWork {
    open: Vec<AtomicU32>,
    /// Per ancestor, the start of its counters in `open` (`top.len() + 1`).
    base: Vec<usize>,
    /// Forward: per ancestor, the start of its row chunks in `tiles`
    /// (`top.len() + 1`).
    chunk_ptr: Vec<usize>,
    /// Forward: per row chunk, the parent's tiles its rows lie in, `lo..hi`.
    tiles: Vec<(u32, u32)>,
}

impl DagWork {
    /// `open[i]` closed once: whether that was its last open input.
    #[inline]
    fn close(&self, i: usize) -> bool {
        // acquire-release: the writes of every input happen before the task
        // that the last one starts
        self.open[i].fetch_sub(1, Ordering::AcqRel) == 1
    }

    fn reset(&mut self, len: usize) {
        if self.open.len() < len {
            self.open.resize_with(len, || AtomicU32::new(0));
        }
        for o in &self.open[..len] {
            o.store(0, Ordering::Relaxed);
        }
    }
}

/// A DAG of tasks of kind `Task`, run in a scope.
trait Dag: Sync {
    type Task: Copy + Send + Sync;
    /// Run `task`, handing every task it makes ready to `ready`.
    fn step<'s>(&'s self, task: Self::Task, ready: &mut Ready<'_, '_, 's, Self>);
}

/// The tasks a task made ready: the first one runs next on this thread, the
/// rest are spawned.
struct Ready<'r, 'b, 's, D: Dag + ?Sized> {
    sc: &'r Scope<'b, 's>,
    dag: &'s D,
    next: Option<D::Task>,
}

impl<'s, D: Dag> Ready<'_, '_, 's, D> {
    fn push(&mut self, t: D::Task) {
        if self.next.is_none() {
            self.next = Some(t);
        } else {
            let dag = self.dag;
            self.sc.spawn(move |sc| run(dag, sc, t));
        }
    }
}

/// `task` and every task it makes ready that runs on this thread after it.
fn run<'s, D: Dag>(dag: &'s D, sc: &Scope<'_, 's>, task: D::Task) {
    let mut next = Some(task);
    while let Some(t) = next {
        let mut ready = Ready {
            sc,
            dag,
            next: None,
        };
        dag.step(t, &mut ready);
        next = ready.next;
    }
}

/// The node times of a sweep's DAG where it is traced: per ancestor, its
/// tasks' times and the span from its first task's start to its last one's
/// end (`total`).
struct DagTrace {
    t0: crate::clock::Instant,
    nodes: Vec<(NodeTimes, AtomicU64, AtomicU64)>,
}

impl DagTrace {
    fn new(trace: Option<&SweepTrace>, nt: usize) -> Option<Self> {
        trace.map(|_| Self {
            t0: crate::clock::Instant::now(),
            nodes: (0..nt)
                .map(|_| {
                    let n = NodeTimes::default();
                    (n, AtomicU64::new(u64::MAX), AtomicU64::new(0))
                })
                .collect(),
        })
    }

    fn finish(self, trace: Option<&SweepTrace>) {
        let Some(trace) = trace else { return };
        for (ti, (times, from, to)) in self.nodes.into_iter().enumerate() {
            let (from, to) = (from.into_inner(), to.into_inner());
            if from <= to {
                times.total.store(to - from, Ordering::Relaxed);
                trace.add(ti, times);
            }
        }
    }
}

/// `f()` of ancestor `ti`, timed into its `at` where traced.
#[inline]
fn timed<R>(
    tr: Option<&DagTrace>,
    ti: usize,
    at: impl Fn(&NodeTimes) -> &AtomicU64,
    f: impl FnOnce() -> R,
) -> R {
    let Some(tr) = tr else { return f() };
    let a = tr.t0.elapsed().as_nanos() as u64;
    let r = f();
    let b = tr.t0.elapsed().as_nanos() as u64;
    let (times, from, to) = &tr.nodes[ti];
    at(times).fetch_add(b - a, Ordering::Relaxed);
    from.fetch_min(a, Ordering::Relaxed);
    to.fetch_max(b, Ordering::Relaxed);
    r
}

/// The tiles of an ancestor of width `w`.
fn tiles_of(w: usize) -> usize {
    w.div_ceil(SOLVE_DIAG_BLOCK)
}

/// A forward task ([`Fwd`]): the ancestor's `top` index first.
#[derive(Clone, Copy)]
enum FwdTask {
    Take(u32, u32),
    Tri(u32, u32),
    Own(u32, u32, u32),
    Below(u32, u32, u32),
}

/// The shape of an ancestor in the forward DAG.
#[derive(Clone, Copy)]
struct FwdNode {
    c0: usize,
    w: usize,
    m: usize,
    nb: usize,
    nc: usize,
    /// The last tile has fewer rows than a product's least ([`chunk_align`]):
    /// its rows join the tile before it in the products of the tiles before
    /// that (in the level sweeps they lie inside one product of more rows).
    merged: bool,
    base: usize,
}

impl FwdNode {
    fn take(&self, b: usize) -> usize {
        self.base + b
    }

    fn own(&self, g: usize, s: usize) -> usize {
        self.base + self.nb + g * (g - 1) / 2 + s
    }

    fn below(&self, c: usize, s: usize) -> usize {
        self.base + self.nb * (self.nb + 1) / 2 + c * self.nb + s
    }

    fn done(&self) -> usize {
        self.base + self.nb * (self.nb + 1) / 2 + self.nc * self.nb
    }

    fn kids(&self) -> usize {
        self.done() + 1
    }

    fn len(&self) -> usize {
        self.nb * (self.nb + 1) / 2 + self.nc * self.nb + 2
    }

    /// The task tile `g`'s product of tile `s` is part of: `g` itself, or
    /// none where it joined the tile before it.
    fn own_task(&self, g: usize, s: usize) -> Option<usize> {
        (!(self.merged && g == self.nb - 1 && s + 2 < self.nb)).then_some(g)
    }

    /// The node rows of the product of tile `s` into tile `g`.
    fn own_rows(&self, g: usize, s: usize) -> std::ops::Range<usize> {
        let b = SOLVE_DIAG_BLOCK;
        if self.merged && g + 2 == self.nb && s < g {
            g * b..self.w
        } else {
            g * b..((g + 1) * b).min(self.w)
        }
    }
}

struct Fwd<'a, T> {
    plan: &'a SolvePlan<T>,
    nr: usize,
    y: Shared<T>,
    acc: Shared<T>,
    subs: &'a [super::Scratch<T>],
    dag: &'a DagWork,
    trace: Option<&'a DagTrace>,
}

/// The shape of ancestor `ti` in the forward DAG laid out in `dag`.
fn fwd_node<T: Scalar>(plan: &SolvePlan<T>, dag: &DagWork, nr: usize, ti: usize) -> FwdNode {
    let (c0, w, _, m, _, _) = plan.node(plan.top[ti]);
    let nb = tiles_of(w);
    FwdNode {
        c0,
        w,
        m,
        nb,
        nc: dag.chunk_ptr[ti + 1] - dag.chunk_ptr[ti],
        merged: nb >= 3 && w - (nb - 1) * SOLVE_DIAG_BLOCK < chunk_align(nr),
        base: dag.base[ti],
    }
}

impl<T: Scalar> Fwd<'_, T> {
    fn node(&self, ti: usize) -> FwdNode {
        fwd_node(self.plan, self.dag, self.nr, ti)
    }

    /// The columns `j0..` of ancestor `ti` (`yc`, from column `j0` of the
    /// factor) take every source's terms for them: of a source's run, the
    /// slots whose rows lie in those columns.
    fn take(&self, ti: usize, j0: usize, yc: &mut [T]) {
        let (p, nr) = (self.plan, self.nr);
        let nsub = p.subtrees.len();
        let (lo_r, hi_r) = (j0 as u32, (j0 + yc.len() / nr) as u32);
        for &(src, lo, hi) in &p.pull[p.pull_ptr[ti]..p.pull_ptr[ti + 1]] {
            let (src, lo, hi) = (src as usize, lo as usize, hi as usize);
            let list = if src < nsub {
                &p.subtrees[src].ext[lo..hi]
            } else {
                let d = p.top[src - nsub] as usize;
                &p.rows[p.row_ptr[d] + lo..p.row_ptr[d] + hi]
            };
            let a = list.partition_point(|&r| r < lo_r);
            let b = list.partition_point(|&r| r < hi_r);
            if a == b {
                continue;
            }
            let from = if src < nsub {
                &self.subs[src].acc[(lo + a) * nr..(lo + b) * nr]
            } else {
                let o = (p.top_acc_off[src - nsub] + lo + a) * nr;
                // SAFETY: the source is done (the take's open inputs), and no
                // task writes its accumulator any more.
                unsafe { std::slice::from_raw_parts(self.acc.ptr().add(o), (b - a) * nr) }
            };
            for (k, &r) in list[a..b].iter().enumerate() {
                let i = (r - lo_r) as usize;
                sub_assign(&mut yc[i * nr..(i + 1) * nr], &from[k * nr..(k + 1) * nr]);
            }
        }
    }

    /// Chunk `c` of ancestor `ti`'s rows below is done.
    fn chunk_done<'s>(
        &'s self,
        ti: usize,
        nd: &FwdNode,
        c: usize,
        ready: &mut Ready<'_, '_, 's, Self>,
    ) {
        let p = self.plan.top_parent[ti];
        if p != NONE {
            let pn = self.node(p as usize);
            let (lo, hi) = self.dag.tiles[self.dag.chunk_ptr[ti] + c];
            for b in lo as usize..hi as usize {
                if self.dag.close(pn.take(b)) {
                    ready.push(FwdTask::Take(p, b as u32));
                }
            }
        }
        if self.dag.close(nd.done()) {
            self.complete(ti, ready);
        }
    }

    /// Ancestor `ti` is complete: its chunks and its descendants are done.
    fn complete<'s>(&'s self, mut ti: usize, ready: &mut Ready<'_, '_, 's, Self>) {
        loop {
            let p = self.plan.top_parent[ti];
            if p == NONE {
                return;
            }
            let pn = self.node(p as usize);
            if !self.dag.close(pn.kids()) {
                return;
            }
            // every descendant of `p` is complete: its parent's tiles take
            // from them
            let pp = self.plan.top_parent[p as usize];
            if pp != NONE {
                let ppn = self.node(pp as usize);
                for b in 0..ppn.nb {
                    if self.dag.close(ppn.take(b)) {
                        ready.push(FwdTask::Take(pp, b as u32));
                    }
                }
            }
            if !self.dag.close(pn.done()) {
                return;
            }
            ti = p as usize;
        }
    }
}

impl<T: Scalar> Dag for Fwd<'_, T> {
    type Task = FwdTask;

    fn step<'s>(&'s self, task: FwdTask, ready: &mut Ready<'_, '_, 's, Self>) {
        let (plan, nr, dag) = (self.plan, self.nr, self.dag);
        let tr = self.trace;
        let ti = match task {
            FwdTask::Take(t, ..)
            | FwdTask::Tri(t, ..)
            | FwdTask::Own(t, ..)
            | FwdTask::Below(t, ..) => t as usize,
        };
        let nd = self.node(ti);
        let (_, w, _, m, _, panel) = plan.node(plan.top[ti]);
        let block = |b: usize| packed_blocks(w, m).nth(b).unwrap_or((0, 0, 0, 0));
        // SAFETY: the node's rows; every task of the node writes rows of it no
        // other task writes meanwhile, and reads only rows that are final
        // (the open inputs order them).
        let yn = unsafe { self.y.ptr().add(nd.c0 * nr) };
        match task {
            FwdTask::Take(_, b) => {
                let b = b as usize;
                let (j0, j1) = (b * SOLVE_DIAG_BLOCK, ((b + 1) * SOLVE_DIAG_BLOCK).min(w));
                // SAFETY: the tile's rows, this task's alone.
                let yc = unsafe { std::slice::from_raw_parts_mut(yn.add(j0 * nr), (j1 - j0) * nr) };
                timed(tr, ti, |t| &t.take, || self.take(ti, nd.c0 + j0, yc));
                if b == 0 {
                    ready.push(FwdTask::Tri(ti as u32, 0));
                } else {
                    let g = if nd.merged && b + 1 == nd.nb {
                        b - 1
                    } else {
                        b
                    };
                    if dag.close(nd.own(g, 0)) {
                        ready.push(FwdTask::Own(ti as u32, g as u32, 0));
                    }
                }
            }
            FwdTask::Tri(_, b) => {
                let b = b as usize;
                let (jb, je, off, ld) = block(b);
                let d = plan.diag_inv.get(nd.c0 + jb..nd.c0 + je).unwrap_or(&[]);
                // SAFETY: the tile's rows, this task's alone.
                let v = unsafe { std::slice::from_raw_parts_mut(yn.add(jb * nr), (je - jb) * nr) };
                timed(
                    tr,
                    ti,
                    |t| &t.tri,
                    || tri_rows(v, &panel[off..], d, ld, nr, 0, je - jb),
                );
                // the next tile's product first: it runs here
                for g in b + 1..nd.nb {
                    if let Some(g) = nd.own_task(g, b) {
                        if dag.close(nd.own(g, b)) {
                            ready.push(FwdTask::Own(ti as u32, g as u32, b as u32));
                        }
                    }
                }
                for c in 0..nd.nc {
                    if dag.close(nd.below(c, b)) {
                        ready.push(FwdTask::Below(ti as u32, c as u32, b as u32));
                    }
                }
                if nd.nc == 0 && b + 1 == nd.nb && dag.close(nd.done()) {
                    self.complete(ti, ready);
                }
            }
            FwdTask::Own(_, g, s) => {
                let (g, s) = (g as usize, s as usize);
                let (jb, je, off, ld) = block(s);
                let rows = nd.own_rows(g, s);
                let joined = rows.end > ((g + 1) * SOLVE_DIAG_BLOCK).min(w);
                timed(
                    tr,
                    ti,
                    |t| &t.pooled,
                    || unsafe { fwd_own(yn, panel.as_ptr().add(off), ld, jb, je, rows, nr) },
                );
                if s + 1 == g {
                    ready.push(FwdTask::Tri(ti as u32, g as u32));
                    if joined && dag.close(nd.own(g + 1, g)) {
                        ready.push(FwdTask::Own(ti as u32, g as u32 + 1, g as u32));
                    }
                } else if dag.close(nd.own(g, s + 1)) {
                    ready.push(FwdTask::Own(ti as u32, g as u32, s as u32 + 1));
                }
            }
            FwdTask::Below(_, c, s) => {
                let (c, s) = (c as usize, s as usize);
                let (jb, je, off, ld) = block(s);
                let rows = chunk_rows(c, nd.nc, m, nr);
                let acc = unsafe { self.acc.ptr().add(plan.top_acc_off[ti] * nr) };
                timed(
                    tr,
                    ti,
                    |t| &t.pooled,
                    || unsafe {
                        fwd_below(
                            acc,
                            panel.as_ptr().add(off),
                            ld,
                            w,
                            jb,
                            je,
                            yn,
                            rows,
                            nr,
                            s > 0,
                        )
                    },
                );
                if s + 1 < nd.nb {
                    if dag.close(nd.below(c, s + 1)) {
                        ready.push(FwdTask::Below(ti as u32, c as u32, s as u32 + 1));
                    }
                } else {
                    self.chunk_done(ti, &nd, c, ready);
                }
            }
        }
    }
}

/// A backward task ([`Bwd`]): the ancestor's `top` index first.
#[derive(Clone, Copy)]
enum BwdTask {
    Start(u32),
    Off(u32, u32, u32),
    In(u32, u32, u32),
    Tri(u32, u32),
}

struct Bwd<'a, T> {
    plan: &'a SolvePlan<T>,
    nr: usize,
    x: Shared<T>,
    g: Shared<T>,
    dag: &'a DagWork,
    trace: Option<&'a DagTrace>,
}

impl<T: Scalar> Bwd<'_, T> {
    /// The row chunks of tile `jb..je` of a node with `m` rows below: of its
    /// product of the rows below, and of its product of the tiles after it
    /// (as [`bwd_node_gemm`](SolvePlan::bwd_node_gemm) cuts them).
    fn chunks(&self, w: usize, m: usize, jb: usize, je: usize) -> (usize, usize) {
        let p = self.plan;
        let off = if m > 0 {
            p.node_chunks(true, je - jb, m, self.nr)
        } else {
            0
        };
        let inner = if je < w {
            p.node_chunks(true, je - jb, w - je, self.nr)
        } else {
            0
        };
        (off, inner)
    }

    /// Tile `b` took its rows below and the tiles after it are solved: its
    /// product of those.
    fn stepped<'s>(&'s self, ti: usize, b: usize, ready: &mut Ready<'_, '_, 's, Self>) {
        let (_, w, _, m, _, _) = self.plan.node(self.plan.top[ti]);
        let (jb, je, _, _) = packed_blocks(w, m).nth(b).unwrap_or((0, 0, 0, 0));
        let (_, inner) = self.chunks(w, m, jb, je);
        if inner == 0 {
            return ready.push(BwdTask::Tri(ti as u32, b as u32));
        }
        for q in 0..inner {
            ready.push(BwdTask::In(ti as u32, b as u32, q as u32));
        }
    }
}

impl<T: Scalar> Dag for Bwd<'_, T> {
    type Task = BwdTask;

    fn step<'s>(&'s self, task: BwdTask, ready: &mut Ready<'_, '_, 's, Self>) {
        let (plan, nr, dag) = (self.plan, self.nr, self.dag);
        let tr = self.trace;
        let ti = match task {
            BwdTask::Start(t) | BwdTask::Off(t, ..) | BwdTask::In(t, ..) | BwdTask::Tri(t, ..) => {
                t as usize
            }
        };
        let (c0, w, r0, m, _, panel) = plan.node(plan.top[ti]);
        let nb = tiles_of(w);
        let base = dag.base[ti];
        let (step, tri_open) = (|b: usize| base + b, |b: usize| base + nb + b);
        let block = |b: usize| packed_blocks(w, m).nth(b).unwrap_or((0, 0, 0, 0));
        // SAFETY: as in the forward DAG; the rows below (`g`) are written by
        // `Start` alone, before any task reads them.
        let xn = unsafe { self.x.ptr().add(c0 * nr) };
        let g = unsafe { self.g.ptr().add(plan.top_acc_off[ti] * nr) };
        match task {
            BwdTask::Start(_) => {
                timed(
                    tr,
                    ti,
                    |t| &t.take,
                    || {
                        for (k, &r) in plan.rows[r0..r0 + m].iter().enumerate() {
                            // SAFETY: the rows below lie in complete ancestors.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    self.x.ptr().add(r as usize * nr),
                                    g.add(k * nr),
                                    nr,
                                )
                            }
                        }
                    },
                );
                if m == 0 {
                    return ready.push(BwdTask::Tri(ti as u32, nb as u32 - 1));
                }
                // the last tile's chunks first: they lead to its triangle
                for b in (0..nb).rev() {
                    let (jb, je, _, _) = block(b);
                    for q in 0..self.chunks(w, m, jb, je).0 {
                        ready.push(BwdTask::Off(ti as u32, b as u32, q as u32));
                    }
                }
            }
            BwdTask::Off(_, b, q) => {
                let b = b as usize;
                let (jb, je, off, ld) = block(b);
                let r = chunk_rows(q as usize, self.chunks(w, m, jb, je).0, je - jb, nr);
                timed(
                    tr,
                    ti,
                    |t| &t.pooled,
                    || unsafe {
                        bwd_off(
                            xn,
                            panel.as_ptr().add(off),
                            ld,
                            w,
                            m,
                            jb,
                            g,
                            jb + r.start..jb + r.end,
                            nr,
                        )
                    },
                );
                if dag.close(step(b)) {
                    self.stepped(ti, b, ready);
                }
            }
            BwdTask::In(_, b, q) => {
                let b = b as usize;
                let (jb, je, off, ld) = block(b);
                let r = chunk_rows(q as usize, self.chunks(w, m, jb, je).1, je - jb, nr);
                timed(
                    tr,
                    ti,
                    |t| &t.pooled,
                    || unsafe {
                        bwd_in(
                            xn,
                            panel.as_ptr().add(off),
                            ld,
                            w,
                            jb,
                            je,
                            jb + r.start..jb + r.end,
                            nr,
                        )
                    },
                );
                if dag.close(tri_open(b)) {
                    ready.push(BwdTask::Tri(ti as u32, b as u32));
                }
            }
            BwdTask::Tri(_, b) => {
                let b = b as usize;
                let (jb, je, off, ld) = block(b);
                let d = plan.diag_inv.get(c0 + jb..c0 + je).unwrap_or(&[]);
                // SAFETY: the tile's rows, this task's alone.
                let v = unsafe { std::slice::from_raw_parts_mut(xn.add(jb * nr), (je - jb) * nr) };
                timed(
                    tr,
                    ti,
                    |t| &t.tri,
                    || tri_up(v, &panel[off..], d, ld, nr, 0, je - jb),
                );
                if b > 0 {
                    if dag.close(step(b - 1)) {
                        self.stepped(ti, b - 1, ready);
                    }
                } else {
                    let kids = &plan.top_kids[plan.top_kids_ptr[ti]..plan.top_kids_ptr[ti + 1]];
                    for &k in kids {
                        ready.push(BwdTask::Start(k));
                    }
                }
            }
        }
    }
}

impl<T: Scalar> SolvePlan<T> {
    /// The forward sweep of the ancestors on the pool as the tile DAG (see
    /// the module), after the leaf subtrees: every ancestor's accumulator is
    /// its rows below, kept to the end of the sweep in one block.
    pub(super) fn forward_dag(
        &self,
        nr: usize,
        y: &mut [T],
        work: &mut PlanWork<T>,
        trace: Option<&SweepTrace>,
    ) {
        let nt = self.top.len();
        // every slot's first term is stored: no zeroing, only growth
        super::grow(&mut work.top_acc, self.top_acc_off[nt] * nr);
        let dag = &mut work.dag;
        dag.base.clear();
        dag.chunk_ptr.clear();
        let (mut at, mut chunks) = (0, 0);
        for ti in 0..nt {
            let (_, w, _, m, _, _) = self.node(self.top[ti]);
            let nb = tiles_of(w);
            let nc = self.node_chunks(true, m, w.min(SOLVE_DIAG_BLOCK), nr);
            dag.base.push(at);
            dag.chunk_ptr.push(chunks);
            at += nb * (nb + 1) / 2 + nc * nb + 2;
            chunks += nc;
        }
        dag.base.push(at);
        dag.chunk_ptr.push(chunks);
        dag.reset(at);
        // the open inputs, and per row chunk the parent's tiles it reaches
        let mut tiles = std::mem::take(&mut dag.tiles);
        tiles.clear();
        let dag = &work.dag;
        for ti in 0..nt {
            let nd = fwd_node(self, dag, nr, ti);
            debug_assert_eq!(nd.len(), dag.base[ti + 1] - dag.base[ti]);
            let set = |i: usize, v: usize| dag.open[i].store(v as u32, Ordering::Relaxed);
            let add = |i: usize| dag.open[i].fetch_add(1, Ordering::Relaxed);
            for g in 1..nd.nb {
                for s in 0..g {
                    set(nd.own(g, s), 2);
                }
            }
            if nd.merged {
                set(nd.own(nd.nb - 2, 0), 3);
            }
            for c in 0..nd.nc {
                set(nd.below(c, 0), 1);
                for s in 1..nd.nb {
                    set(nd.below(c, s), 2);
                }
            }
            let kids = self.top_kids_ptr[ti + 1] - self.top_kids_ptr[ti];
            set(nd.kids(), kids);
            // its chunks, its descendants, and without chunks its last triangle
            set(nd.done(), nd.nc.max(1) + usize::from(kids > 0));
            let p = self.top_parent[ti];
            let pn = (p != NONE).then(|| fwd_node(self, dag, nr, p as usize));
            for c in 0..nd.nc {
                let mut reach = (0, 0);
                if let Some(pn) = &pn {
                    let r = chunk_rows(c, nd.nc, nd.m, nr);
                    let r0 = self.row_ptr[self.top[ti] as usize];
                    let list = &self.rows[r0 + r.start..r0 + r.end];
                    let a = list.partition_point(|&x| (x as usize) < pn.c0);
                    let e = list.partition_point(|&x| (x as usize) < pn.c0 + pn.w);
                    if a < e {
                        let tile = |x: u32| ((x as usize - pn.c0) / SOLVE_DIAG_BLOCK) as u32;
                        reach = (tile(list[a]), tile(list[e - 1]) + 1);
                    }
                    for b in reach.0..reach.1 {
                        add(pn.take(b as usize));
                    }
                }
                tiles.push(reach);
            }
            if let Some(pn) = pn.filter(|_| kids > 0) {
                for b in 0..pn.nb {
                    add(pn.take(b));
                }
            }
        }
        work.dag.tiles = tiles;
        let tr = DagTrace::new(trace, nt);
        let fwd = Fwd {
            plan: self,
            nr,
            y: Shared(y.as_mut_ptr(), y.len()),
            acc: Shared(work.top_acc.as_mut_ptr(), work.top_acc.len()),
            subs: &work.subtrees,
            dag: &work.dag,
            trace: tr.as_ref(),
        };
        let fwd = &fwd;
        crate::parallel::scope(crate::parallel::Work::ns(f64::INFINITY), |sc| {
            for ti in 0..nt {
                let nd = fwd.node(ti);
                for b in 0..nd.nb {
                    if fwd.dag.open[nd.take(b)].load(Ordering::Relaxed) == 0 {
                        let t = FwdTask::Take(ti as u32, b as u32);
                        sc.spawn(move |sc| run(fwd, sc, t));
                    }
                }
            }
        });
        if let Some(tr) = tr {
            tr.finish(trace);
        }
    }

    /// The backward sweep of the ancestors on the pool as the tile DAG (see
    /// the module), before the leaf subtrees: an ancestor starts once its
    /// parent is complete, its rows below gathered into the accumulator block
    /// the forward sweep used.
    pub(super) fn backward_dag(
        &self,
        nr: usize,
        x: &mut [T],
        work: &mut PlanWork<T>,
        trace: Option<&SweepTrace>,
    ) {
        let nt = self.top.len();
        super::grow(&mut work.top_acc, self.top_acc_off[nt] * nr);
        let dag = &mut work.dag;
        dag.base.clear();
        let mut at = 0;
        for ti in 0..nt {
            let (_, w, _, _, _, _) = self.node(self.top[ti]);
            dag.base.push(at);
            at += 2 * tiles_of(w);
        }
        dag.base.push(at);
        dag.reset(at);
        let tr = DagTrace::new(trace, nt);
        let bwd = Bwd {
            plan: self,
            nr,
            x: Shared(x.as_mut_ptr(), x.len()),
            g: Shared(work.top_acc.as_mut_ptr(), work.top_acc.len()),
            dag: &work.dag,
            trace: tr.as_ref(),
        };
        for ti in 0..nt {
            let (_, w, _, m, _, _) = self.node(self.top[ti]);
            let (nb, base) = (tiles_of(w), bwd.dag.base[ti]);
            for (b, (jb, je, _, _)) in packed_blocks(w, m).enumerate() {
                let (off, inner) = bwd.chunks(w, m, jb, je);
                let set = |i: usize, v: usize| bwd.dag.open[i].store(v as u32, Ordering::Relaxed);
                set(base + b, off + usize::from(b + 1 < nb));
                set(base + nb + b, inner);
            }
        }
        let bwd = &bwd;
        crate::parallel::scope(crate::parallel::Work::ns(f64::INFINITY), |sc| {
            for &r in &self.top_roots {
                sc.spawn(move |sc| run(bwd, sc, BwdTask::Start(r)));
            }
        });
        if let Some(tr) = tr {
            tr.finish(trace);
        }
    }
}
