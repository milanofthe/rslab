//! The ancestors' sweeps on the pool: a task DAG over the tiles of their dense
//! nodes, each tile [`PANEL_BLOCK`] columns of a node's panel
//! ([`Layout`]).
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
    bwd_in, bwd_off, chunk_align, chunk_rows, fwd_below, fwd_own, sub_assign, tri_rows, tri_up,
    PlanWork, Shared, SolvePlan, NONE,
};
use crate::constants::PANEL_BLOCK;
use crate::numeric::supernodal::panel::Layout;
use crate::parallel::Scope;
use crate::scalar::Scalar;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Default)]
pub(crate) struct DagWork {
    open: Vec<AtomicU32>,
    base: Vec<usize>,
    chunk_ptr: Vec<usize>,
    tiles: Vec<(u32, u32)>,
    first: Vec<(u32, u32)>,
    len: usize,
}

impl DagWork {
    #[inline]
    fn close(&self, i: usize) -> bool {
        self.open[i].fetch_sub(1, Ordering::AcqRel) == 1
    }

    fn reset(&mut self, len: usize) {
        self.len = len;
        if self.open.len() < len {
            self.open.resize_with(len, || AtomicU32::new(0));
        }
        for o in &self.open[..len] {
            o.store(0, Ordering::Relaxed);
        }
    }

    fn drained(&self) -> bool {
        (self.open[..self.len].iter()).all(|o| o.load(Ordering::Relaxed) == 0)
    }
}

trait Dag: Sync {
    type Task: Copy + Send + Sync;
    /// Run `task`, handing every task it makes ready to `ready`.
    fn step<'s>(&'s self, task: Self::Task, ready: &mut Ready<'_, '_, 's, Self>);
}

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

fn tiles_of(w: usize) -> usize {
    w.div_ceil(PANEL_BLOCK)
}

#[derive(Clone, Copy)]
enum FwdTask {
    Take(u32, u32),
    Tri(u32, u32),
    Own(u32, u32, u32),
    Below(u32, u32, u32),
}

#[derive(Clone, Copy)]
struct FwdNode {
    c0: usize,
    w: usize,
    m: usize,
    nb: usize,
    nc: usize,
    last_tile_merged: bool,
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

    fn own_task(&self, g: usize, s: usize) -> Option<usize> {
        (!(self.last_tile_merged && g == self.nb - 1 && s + 2 < self.nb)).then_some(g)
    }

    fn own_rows(&self, g: usize, s: usize) -> std::ops::Range<usize> {
        let b = PANEL_BLOCK;
        if self.last_tile_merged && g + 2 == self.nb && s < g {
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
}

fn fwd_node<T: Scalar>(plan: &SolvePlan<T>, dag: &DagWork, nr: usize, ti: usize) -> FwdNode {
    let (c0, w, _, m, _, _) = plan.node(plan.top[ti]);
    let nb = tiles_of(w);
    FwdNode {
        c0,
        w,
        m,
        nb,
        nc: dag.chunk_ptr[ti + 1] - dag.chunk_ptr[ti],
        last_tile_merged: nb > 2 && w - (nb - 1) * PANEL_BLOCK < chunk_align(nr),
        base: dag.base[ti],
    }
}

impl<T: Scalar> Fwd<'_, T> {
    fn node(&self, ti: usize) -> FwdNode {
        fwd_node(self.plan, self.dag, self.nr, ti)
    }

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
        let ti = match task {
            FwdTask::Take(t, ..)
            | FwdTask::Tri(t, ..)
            | FwdTask::Own(t, ..)
            | FwdTask::Below(t, ..) => t as usize,
        };
        let nd = self.node(ti);
        let (_, w, _, m, _, panel) = plan.node(plan.top[ti]);
        let lay = Layout::new(w, m);
        let block = |b: usize| lay.blocks().nth(b).unwrap_or((0, 0, 0, 0));
        // SAFETY: the node's rows; every task of the node writes rows of it no
        // other task writes meanwhile, and reads only rows that are final
        // (the open inputs order them).
        let yn = unsafe { self.y.ptr().add(nd.c0 * nr) };
        match task {
            FwdTask::Take(_, b) => {
                let b = b as usize;
                let (j0, j1) = (b * PANEL_BLOCK, ((b + 1) * PANEL_BLOCK).min(w));
                // SAFETY: the tile's rows, this task's alone.
                let yc = unsafe { std::slice::from_raw_parts_mut(yn.add(j0 * nr), (j1 - j0) * nr) };
                {
                    let _s = rsprobe::scope!(TRACE, "take");
                    self.take(ti, nd.c0 + j0, yc);
                }
                if b == 0 {
                    ready.push(FwdTask::Tri(ti as u32, 0));
                } else {
                    let g = if nd.last_tile_merged && b + 1 == nd.nb {
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
                {
                    let _s = rsprobe::scope!(TRACE, "triangle");
                    tri_rows(v, &panel[off..], d, ld, nr, 0, je - jb);
                }
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
                let joined = rows.end > ((g + 1) * PANEL_BLOCK).min(w);
                {
                    let _s = rsprobe::scope!(TRACE, "product");
                    unsafe { fwd_own(yn, panel.as_ptr().add(off), ld, jb, je, rows, nr) };
                }
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
                let (jb, je, _, _) = block(s);
                let rows = chunk_rows(c, nd.nc, m, nr);
                let acc = unsafe { self.acc.ptr().add(plan.top_acc_off[ti] * nr) };
                {
                    let _s = rsprobe::scope!(TRACE, "product");
                    unsafe {
                        fwd_below(
                            acc,
                            panel.as_ptr().add(lay.rect()),
                            m,
                            jb,
                            je,
                            yn,
                            rows,
                            nr,
                            s > 0,
                        )
                    };
                }
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
}

impl<T: Scalar> Bwd<'_, T> {
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

    fn stepped<'s>(&'s self, ti: usize, b: usize, ready: &mut Ready<'_, '_, 's, Self>) {
        let (_, w, _, m, _, _) = self.plan.node(self.plan.top[ti]);
        let (jb, je, _, _) = Layout::new(w, m).blocks().nth(b).unwrap_or((0, 0, 0, 0));
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
        let ti = match task {
            BwdTask::Start(t) | BwdTask::Off(t, ..) | BwdTask::In(t, ..) | BwdTask::Tri(t, ..) => {
                t as usize
            }
        };
        let (c0, w, r0, m, _, panel) = plan.node(plan.top[ti]);
        let nb = tiles_of(w);
        let base = dag.base[ti];
        let (step, tri_open) = (|b: usize| base + b, |b: usize| base + nb + b);
        let lay = Layout::new(w, m);
        let block = |b: usize| lay.blocks().nth(b).unwrap_or((0, 0, 0, 0));
        // SAFETY: as in the forward DAG; the rows below (`g`) are written by
        // `Start` alone, before any task reads them.
        let xn = unsafe { self.x.ptr().add(c0 * nr) };
        let g = unsafe { self.g.ptr().add(plan.top_acc_off[ti] * nr) };
        match task {
            BwdTask::Start(_) => {
                {
                    let _s = rsprobe::scope!(TRACE, "take");
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
                }
                if m == 0 {
                    return ready.push(BwdTask::Tri(ti as u32, nb as u32 - 1));
                }
                for b in (0..nb).rev() {
                    let (jb, je, _, _) = block(b);
                    for q in 0..self.chunks(w, m, jb, je).0 {
                        ready.push(BwdTask::Off(ti as u32, b as u32, q as u32));
                    }
                }
            }
            BwdTask::Off(_, b, q) => {
                let b = b as usize;
                let (jb, je, _, _) = block(b);
                let r = chunk_rows(q as usize, self.chunks(w, m, jb, je).0, je - jb, nr);
                {
                    let _s = rsprobe::scope!(TRACE, "product");
                    unsafe {
                        bwd_off(
                            xn,
                            panel.as_ptr().add(lay.rect()),
                            m,
                            g,
                            jb + r.start..jb + r.end,
                            nr,
                        )
                    };
                }
                if dag.close(step(b)) {
                    self.stepped(ti, b, ready);
                }
            }
            BwdTask::In(_, b, q) => {
                let b = b as usize;
                let (jb, je, off, ld) = block(b);
                let r = chunk_rows(q as usize, self.chunks(w, m, jb, je).1, je - jb, nr);
                {
                    let _s = rsprobe::scope!(TRACE, "product");
                    unsafe {
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
                    };
                }
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
                {
                    let _s = rsprobe::scope!(TRACE, "triangle");
                    tri_up(v, &panel[off..], d, ld, nr, 0, je - jb);
                }
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
    pub(super) fn forward_dag(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let nt = self.top.len();
        crate::memory::grow(&mut work.top_acc, self.top_acc_off[nt] * nr);
        let dag = &mut work.dag;
        dag.base.clear();
        dag.chunk_ptr.clear();
        let (mut at, mut chunks) = (0, 0);
        for ti in 0..nt {
            let (_, w, _, m, _, _) = self.node(self.top[ti]);
            let nb = tiles_of(w);
            let nc = self.node_chunks(true, m, w.min(PANEL_BLOCK), nr);
            dag.base.push(at);
            dag.chunk_ptr.push(chunks);
            at += nb * (nb + 1) / 2 + nc * nb + 2;
            chunks += nc;
        }
        dag.base.push(at);
        dag.chunk_ptr.push(chunks);
        dag.reset(at);
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
                    if nd.own_task(g, s).is_some() {
                        set(nd.own(g, s), 2);
                    }
                }
            }
            if nd.last_tile_merged {
                add(nd.own(nd.nb - 2, 0));
            }
            for c in 0..nd.nc {
                set(nd.below(c, 0), 1);
                for s in 1..nd.nb {
                    set(nd.below(c, s), 2);
                }
            }
            let kids = self.top_kids_ptr[ti + 1] - self.top_kids_ptr[ti];
            set(nd.kids(), kids);
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
                        let tile = |x: u32| ((x as usize - pn.c0) / PANEL_BLOCK) as u32;
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
        let mut first = std::mem::take(&mut work.dag.first);
        first.clear();
        for ti in 0..nt {
            let nd = fwd_node(self, &work.dag, nr, ti);
            for b in 0..nd.nb {
                if work.dag.open[nd.take(b)].load(Ordering::Relaxed) == 0 {
                    first.push((ti as u32, b as u32));
                }
            }
        }
        work.dag.first = first;
        let fwd = Fwd {
            plan: self,
            nr,
            y: Shared(y.as_mut_ptr(), y.len()),
            acc: Shared(work.top_acc.as_mut_ptr(), work.top_acc.len()),
            subs: &work.subtrees,
            dag: &work.dag,
        };
        let fwd = &fwd;
        crate::parallel::scope(crate::parallel::Work::ns(f64::INFINITY), |sc| {
            for &(ti, b) in &fwd.dag.first {
                let t = FwdTask::Take(ti, b);
                sc.spawn(move |sc| run(fwd, sc, t));
            }
        });
        assert!(
            work.dag.drained(),
            "solve DAG: a task of the forward sweep did not run once"
        );
    }

    pub(super) fn backward_dag(&self, nr: usize, x: &mut [T], work: &mut PlanWork<T>) {
        let nt = self.top.len();
        crate::memory::grow(&mut work.top_acc, self.top_acc_off[nt] * nr);
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
        let bwd = Bwd {
            plan: self,
            nr,
            x: Shared(x.as_mut_ptr(), x.len()),
            g: Shared(work.top_acc.as_mut_ptr(), work.top_acc.len()),
            dag: &work.dag,
        };
        for ti in 0..nt {
            let (_, w, _, m, _, _) = self.node(self.top[ti]);
            let (nb, base) = (tiles_of(w), bwd.dag.base[ti]);
            for (b, (jb, je, _, _)) in Layout::new(w, m).blocks().enumerate() {
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
        assert!(
            work.dag.drained(),
            "solve DAG: a task of the backward sweep did not run once"
        );
    }
}
