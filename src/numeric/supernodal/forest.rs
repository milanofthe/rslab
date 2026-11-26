//! The dependency-count scheduler over the assembly forest.

use super::LlSchedule;
use crate::error::RslabError;

pub(crate) fn ll_forest(
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
    refcount: &[std::sync::atomic::AtomicUsize],
    factor_node: &(dyn Fn(usize) -> Result<(), RslabError> + Sync),
    emit_free: &(dyn Fn(usize) + Sync),
) -> Result<(), RslabError> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    const NONE: usize = usize::MAX;
    let _s = rsprobe::scope!(DEBUG, "tree");
    // https://github.com/milanofthe/rslab/issues/289
    let panel_entries = |s: usize| (sym.supernodes[s].nrow * sym.supernodes[s].ncol) as u64;
    let active = rsprobe::active!();
    let total: u64 = match active {
        true => (0..sym.supernodes.len()).map(panel_entries).sum(),
        false => 0,
    };
    let steps = crate::progress::Steps::new(active, total);
    let failed = AtomicBool::new(false);
    let error = std::sync::Mutex::new(None);

    let node = |s: usize| -> bool {
        if failed.load(Ordering::Relaxed) {
            return false;
        }
        let _front = rsprobe::scope!(TRACE, "front");
        rsprobe::add!(DEBUG, "fronts", 1);
        if let Err(e) = factor_node(s) {
            failed.store(true, Ordering::Relaxed);
            if let Ok(mut slot) = error.lock() {
                slot.get_or_insert(e);
            }
            return false;
        }
        let free = |k: usize| {
            if refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                emit_free(k);
            }
        };
        let up = sched.updaters(s);
        debug_assert!(up.windows(2).all(|w| w[0] < w[1]));
        if up.len() >= crate::constants::FREE_PAR_MIN_UPDATERS {
            crate::parallel::ranges::<crate::parallel::Rayon>(
                up.len(),
                crate::parallel::Work::ns(f64::INFINITY),
                |r| r.for_each(|i| free(up[i] as usize)),
            );
        } else {
            up.iter().for_each(|&k| free(k as usize));
        }
        if refcount[s].load(Ordering::Relaxed) == 0 {
            emit_free(s);
        }
        if let Some(step) = steps.as_ref().and_then(|st| st.add(panel_entries(s))) {
            rsprobe::progress!(step, crate::constants::PROGRESS_STEPS);
        }
        true
    };
    if crate::parallel::threads() == 1 {
        for s in 0..sym.supernodes.len() {
            debug_assert!(sym.supernodes[s].children.iter().all(|&c| c < s));
            if !node(s) {
                break;
            }
        }
    } else {
        // https://github.com/milanofthe/rslab/issues/289
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
