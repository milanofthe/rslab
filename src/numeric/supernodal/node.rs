//! Per-node machinery shared by the LDL^T and LU kernels: the cmod plan and
//! the global-to-local row map.

use super::{Li, LlSchedule};
use crate::scalar::Scalar;

/// Work above which a node forks inside its cmod. A small node that forks
/// pays rayon's join-steal latency: while its join waits for a stolen slab,
/// the waiting thread steals other work, often a whole sibling subtree, and
/// this node (and every dependent on its chain) stalls for tens of ms doing
/// almost no flops (measured: 74 ms of cmod at 0.03 Gflop on a 1046x170
/// node). Below the gate the node's cmod runs strictly serially; the
/// tree-level parallelism covers it.
const CMOD_FORK_MIN_FLOPS: usize = 100_000_000;

/// How a node applies its descendants' updates (`cmod`), shared by the LDL^T
/// and LU kernels. Every field is a pure function of the node and the
/// kernel thresholds, never of the thread count or of timing: a
/// timing-dependent choice broke bit-identity twice (a chain-phase-dependent
/// `tiled`, and a fork below the gate while few nodes were in flight, which
/// switched GEMMs between serial and parallel mode, not bit-identical for
/// complex scalars); see `tests/ll_thread_determinism.rs`.
pub(crate) struct CmodPlan {
    /// `(updater, p0, p1)`: the updater's off-diagonal rows `[p0, p1)` land in
    /// this node's columns.
    pub spans: Vec<(usize, usize, usize)>,
    /// The node's update work is large enough to fork (see
    /// [`CMOD_FORK_MIN_FLOPS`], raised to the parallel-GEMM threshold).
    pub forks: bool,
    /// Width of the column slabs of the tiled mode.
    pub tile_w: usize,
    /// Tiled mode: the panel is cut into column slabs (disjoint `&mut`
    /// chunks) and each slab receives every updater's contribution in updater
    /// order with a serial GEMM. One fan-out per node instead of one per
    /// update, the slab stays cache-hot across the updaters, and a root
    /// separator running alone with hundreds of updaters still parallelizes.
    /// Not bit-identical to the sequential mode (that one routes small updates
    /// through the scalar kernel, and cutting an update at slab boundaries
    /// changes the GEMM shapes, whose per-element bits are shape-dependent),
    /// which is why the pick is a pure function of the node.
    pub tiled: bool,
}

impl CmodPlan {
    /// The plan of supernode `s`. With `count_u` the U-side rows beyond each
    /// landing range count as work too (the LU kernel also updates `U12`).
    pub fn new(
        sym: &crate::symbolic::SymbolicFactorization,
        sched: &LlSchedule,
        s: usize,
        count_u: bool,
        par_gemm: usize,
    ) -> Self {
        let (first, ncol) = (sym.supernodes[s].first_col, sym.supernodes[s].ncol);
        let mut spans = Vec::with_capacity(sched.updaters(s).len());
        let mut flops: usize = 0;
        for &kk in sched.updaters(s) {
            let kk = kk as usize;
            let nck = sym.supernodes[kk].ncol;
            let ok = &sched.rows(kk)[nck..];
            let nok = ok.len();
            let p0 = ok.partition_point(|&g| (g as usize) < first);
            let p1 = ok.partition_point(|&g| (g as usize) < first + ncol);
            let npk = p1 - p0;
            if npk == 0 {
                continue;
            }
            flops += (nok - p0) * npk * nck;
            if count_u {
                flops += npk * (nok - p1) * nck;
            }
            spans.push((kk, p0, p1));
        }
        let forks = flops >= CMOD_FORK_MIN_FLOPS.max(par_gemm);
        let tile_w = (ncol / 16).clamp(32, 256);
        CmodPlan {
            spans,
            forks,
            tile_w,
            tiled: forks && ncol >= 2 * tile_w,
        }
    }
}

thread_local! {
    /// Per-worker global-to-local row map, held at all-`Li::MAX` between nodes.
    static GLOC_SCRATCH: std::cell::RefCell<Vec<Li>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Global-to-local row map of one supernode: `map[rows[li]] == li`, every other
/// global index maps to `Li::MAX`. Borrows the worker's scratch and restores
/// it on drop, so every way out of a node (errors included) leaves the
/// invariant intact for the next node on this thread.
pub(crate) struct Gloc<'a> {
    map: Vec<Li>,
    rows: &'a [Li],
}

impl<'a> Gloc<'a> {
    pub fn new(n: usize, rows: &'a [Li]) -> Self {
        let mut map = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
        if map.len() < n {
            map.resize(n, Li::MAX);
        }
        for (li, &g) in rows.iter().enumerate() {
            map[g as usize] = li as Li;
        }
        Gloc { map, rows }
    }
}

impl std::ops::Deref for Gloc<'_> {
    type Target = [Li];
    fn deref(&self) -> &[Li] {
        &self.map
    }
}

impl Drop for Gloc<'_> {
    fn drop(&mut self) {
        for &g in self.rows {
            self.map[g as usize] = Li::MAX;
        }
        let map = std::mem::take(&mut self.map);
        GLOC_SCRATCH.with(|c| *c.borrow_mut() = map);
    }
}

/// Static-pivot perturbation, the complex-symmetric analogue of rslab's f64
/// `perturb_to_floor` (`dense::factor`): lift a pivot whose magnitude is below
/// `abs_floor` up to that floor, preserving phase. For `T = f64` this reduces
/// to `sign(d)*max(|d|, abs_floor)`, matching the real kernel.
#[inline]
pub(crate) fn perturb_pivot<T: Scalar>(d: T, abs_floor: f64) -> T {
    let mag = d.magnitude();
    if mag >= abs_floor {
        d
    } else if mag == 0.0 {
        T::from_real(abs_floor)
    } else {
        d * T::from_real(abs_floor / mag)
    }
}
