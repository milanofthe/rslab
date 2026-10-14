//! The dependency-count scheduler over the assembly forest.

use super::LlSchedule;
use crate::error::RslabError;

/// What releasing one updater costs, learned from the releases.
static FREE_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::FREE_NS_PER_UPDATER);

/// The left-looking factorization of the whole assembly forest, shared
/// by the LDLT/LU twins and scheduled by dependency counts instead of fork-join:
/// every node is one task, started by whichever thread finishes its last child.
/// `factor_node` and `emit_free` carry the path-specific kernels.
///
/// Under fork-join a parent runs on the thread that forked its children, after
/// that thread's wait returns - and a rayon wait executes other ready work
/// meanwhile, often a whole unrelated subtree, so the parent started long after
/// its last child had finished. On a 338k FEM system that gap was more than
/// half of the critical chain (1.0 to 1.1 s of 1.8 s at 12 threads). Here
/// nobody waits: a finished node counts down its parent and, as the last child,
/// hands the parent on as the next task. A stolen task is one node, never a
/// subtree. Per node the work is unchanged (factor, free every updater whose
/// last consumer it was, emit itself if nothing above reads it), and what a
/// node computes does not depend on the thread running it. There is no
/// recursion either, so deep chain trees need no large worker stacks here.
///
/// The nodes run as tasks only where the forest's work (its fronts'
/// multiply-adds at `ns_per_flop`) is worth more than one thread across its
/// leaves; else in index order on the calling thread, each node still as
/// wide inside as its own work is worth.
pub(crate) fn ll_forest(
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
    refcount: &[std::sync::atomic::AtomicUsize],
    factor_node: &(dyn Fn(usize) -> Result<(), RslabError> + Sync),
    emit_free: &(dyn Fn(usize) + Sync),
    ns_per_flop: f64,
) -> Result<(), RslabError> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    const NONE: usize = usize::MAX;
    let failed = AtomicBool::new(false);
    let error = std::sync::Mutex::new(None);

    // Factor `s` and release what it was the last consumer of; the parent's
    // task is the caller's to start when this was its last child.
    let node = |s: usize| -> bool {
        if failed.load(Ordering::Relaxed) {
            return false;
        }
        if let Err(e) = factor_node(s) {
            failed.store(true, Ordering::Relaxed);
            if let Ok(mut slot) = error.lock() {
                slot.get_or_insert(e);
            }
            return false;
        }
        // Disjoint `k`, so the wide top-of-tree free runs in parallel.
        let free = |k: usize| {
            if refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                emit_free(k);
            }
        };
        let up = sched.updaters(s);
        let work = crate::parallel::Work::of(up.len() as f64, &FREE_RATE);
        crate::parallel::for_each(up.len(), work, |i| free(up[i] as usize));
        if refcount[s].load(Ordering::Relaxed) == 0 {
            emit_free(s);
        }
        true
    };
    let flops: f64 = (0..sym.supernodes.len())
        .map(|s| {
            let nrow = sched.rows(s).len() as f64;
            nrow * nrow * sym.supernodes[s].ncol as f64
        })
        .sum();
    if crate::parallel::width(flops * ns_per_flop, sched.leaves.len()) < 2 {
        // Not worth tasks: the supernodes in index order, children before their
        // parents, with no task per node.
        for s in 0..sym.supernodes.len() {
            if !node(s) {
                break;
            }
        }
    } else {
        fn run<'s>(
            scope: &crate::parallel::Scope<'_, 's>,
            s: usize,
            node: &'s (dyn Fn(usize) -> bool + Sync),
            parent: &'s [usize],
            pending: &'s [AtomicUsize],
        ) {
            if !node(s) {
                return;
            }
            let p = parent[s];
            if p != NONE && pending[p].fetch_sub(1, Ordering::AcqRel) == 1 {
                scope.spawn(move |sc| run(sc, p, node, parent, pending));
            }
        }
        let pending: Vec<AtomicUsize> = sym
            .supernodes
            .iter()
            .map(|sn| AtomicUsize::new(sn.children.len()))
            .collect();
        let node: &(dyn Fn(usize) -> bool + Sync) = &node;
        let (parent, pending) = (&sched.parent[..], &pending[..]);
        crate::parallel::scope(crate::parallel::Work::ns(f64::INFINITY), |sc| {
            for &leaf in &sched.leaves {
                sc.spawn(move |sc| run(sc, leaf, node, parent, pending));
            }
        });
    }
    match error.into_inner() {
        Ok(Some(e)) => Err(e),
        _ => Ok(()),
    }
}
