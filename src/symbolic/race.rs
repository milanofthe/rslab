//! The ordering candidates of an analysis and the race among them.
//!
//! A candidate runs only the *prefix* of the analysis: its ordering, the
//! elimination tree and the column counts, read through the permutation,
//! which give its exact factor size. The race keeps the smallest; the
//! winner alone gets the *finish*: the postorder, the merge-biased
//! renumbering and the supernodes. Both the tree and the counts are
//! invariant under the postorder (a relabelling of the tree), so the
//! candidates skip it.

use super::column_counts::total_factor_nnz;
use super::ordering_graph::{structure, OrderingGraph};
use super::progress::{Parts, Reports};
use super::supernode::{
    find_supernodes, pick_amalgamation_strategy, predict_merges, AmalgamationStrategy,
};
use super::{OrderingMethod, SymbolicFactorization};
use crate::error::RslabError;
use crate::numeric::settings::{AmalgamationSettings, OrderingSettings};
use crate::ordering::elimination_tree::EliminationTree;
use crate::ordering::postorder::{biased_postorder, postorder};
use crate::sparse::csc::CscPattern;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub(super) struct Prefix {
    perm: Vec<usize>,
    etree: EliminationTree,
    col_counts: Vec<usize>,
    factor_nnz: usize,
    method: OrderingMethod,
}

fn prefix_flops(px: &Prefix) -> u64 {
    px.col_counts.iter().map(|&c| (c * c) as u64).sum()
}

fn prefix_work(px: &Prefix, workers: usize) -> u64 {
    let mut chain = vec![0u64; px.col_counts.len()];
    let mut longest = 0;
    for (j, &c) in px.col_counts.iter().enumerate() {
        let here = chain[j] + (c * c) as u64;
        longest = longest.max(here);
        if let Some(p) = px.etree.parent[j] {
            debug_assert!(p > j, "not an elimination tree");
            chain[p] = chain[p].max(here);
        }
    }
    (prefix_flops(px) / workers.max(1) as u64).max(longest)
}

pub(super) fn alone(
    graph: &OrderingGraph,
    ordering: &OrderingSettings,
    method: OrderingMethod,
) -> Result<Prefix, RslabError> {
    let ordered = ordering.permutation.is_none();
    let parts = Parts::new(usize::from(ordered), 1, false, 0);
    let reports = match ordered {
        true => parts.stage(0),
        false => parts.known(),
    };
    let px = prefix(graph, ordering, method, ordering.nd.seed, None, &reports);
    parts.finish();
    px
}

fn prefix(
    graph: &OrderingGraph,
    ordering: &OrderingSettings,
    method: OrderingMethod,
    nd_seed: u64,
    stop: Option<&Arc<AtomicBool>>,
    reports: &Reports,
) -> Result<Prefix, RslabError> {
    match &ordering.permutation {
        Some(p) => given(graph, p, method, reports),
        None => Ok(from(graph.order(method, nd_seed, stop, reports)?, method)),
    }
}

fn given(
    graph: &OrderingGraph,
    p: &[usize],
    method: OrderingMethod,
    reports: &Reports,
) -> Result<Prefix, RslabError> {
    let perm = checked_permutation(p, graph.pattern.n)?;
    let s = structure(graph.pattern, perm, reports.structure.as_ref());
    Ok(from(s, method))
}

fn from(s: super::ordering_graph::Structure, method: OrderingMethod) -> Prefix {
    Prefix {
        factor_nnz: total_factor_nnz(&s.col_counts),
        perm: s.perm,
        etree: s.etree,
        col_counts: s.col_counts,
        method,
    }
}

fn checked_permutation(p: &[usize], n: usize) -> Result<Vec<usize>, RslabError> {
    let mut seen = vec![false; n];
    if p.len() != n
        || !p
            .iter()
            .all(|&k| k < n && !std::mem::replace(&mut seen[k], true))
    {
        return Err(RslabError::InvalidInput(format!(
            "given ordering is not a permutation of 0..{n}"
        )));
    }
    Ok(p.to_vec())
}

pub(super) fn race(
    graph: &OrderingGraph,
    ordering: &OrderingSettings,
    amd: Option<&[usize]>,
) -> Result<Prefix, RslabError> {
    use std::sync::atomic::Ordering;
    use std::sync::OnceLock;
    let r = &ordering.race;
    if r.candidates.is_empty() {
        return Err(RslabError::InvalidInput(
            "ordering race without candidates".to_string(),
        ));
    }
    if let Some(bad) = r
        .candidates
        .iter()
        .find(|m| matches!(m, OrderingMethod::Auto | OrderingMethod::MetisND))
    {
        return Err(RslabError::InvalidInput(format!(
            "{bad:?} is not a race candidate (nested dissection joins by its own gates)"
        )));
    }
    let seed = ordering.nd.seed;
    let n = graph.pattern.n;
    let makes_nd_count =
        |px: &Prefix| n > r.nd_min_n && prefix_work(px, r.assumed_workers) >= r.nd_min_work;
    let is_amd = |m: &OrderingMethod| *m == OrderingMethod::Amd;
    let ordering_ns =
        graph.pattern.row_idx.len() as f64 * super::ordering_graph::ORDER_RATE.ns_per_op();
    let amd = amd.filter(|_| r.candidates.iter().any(is_amd));
    let seeded = amd.is_some();
    let k = r.candidates.len() - usize::from(seeded);
    let ensemble = r.ensemble_size.max(1);
    let parts = Parts::new(k, k + usize::from(seeded), n > r.nd_min_n, ensemble);
    let known = match amd {
        Some(p) => {
            let work = crate::parallel::Work::ns(ordering_ns);
            let (known, ()) = crate::parallel::join(
                work,
                || given(graph, p, OrderingMethod::Amd, &parts.known()),
                || graph.prepare(),
            );
            Some(known?)
        }
        _ => {
            graph.prepare();
            None
        }
    };
    let todo: Vec<OrderingMethod> = (r.candidates.iter())
        .filter(|m| !(seeded && is_amd(m)))
        .copied()
        .collect();
    let all = crate::parallel::Work::ns(ordering_ns * todo.len() as f64);
    let side = crate::parallel::width(all.time(), todo.len()) > 1;
    let speculate = n > r.nd_min_n && side && known.as_ref().is_none_or(makes_nd_count);
    let stop = Arc::new(AtomicBool::new(false));
    let nd: Vec<OnceLock<Option<Prefix>>> = (0..ensemble).map(|_| OnceLock::new()).collect();
    let started: Vec<AtomicBool> = (0..ensemble).map(|_| AtomicBool::new(false)).collect();
    let scoped = if n > r.nd_min_n {
        crate::parallel::Work::ns(f64::INFINITY)
    } else {
        all
    };
    let (mut best, last_err, seeds) = crate::parallel::scope(scoped, |sc| {
        let start = |i: usize| {
            if !started[i].swap(true, Ordering::Relaxed) {
                let (nd, stop, parts) = (&nd, &stop, &parts);
                sc.spawn(move |_| {
                    let px = prefix(
                        graph,
                        ordering,
                        OrderingMethod::MetisND,
                        seed + i as u64,
                        Some(stop),
                        &parts.seed(i),
                    );
                    let _ = nd[i].set(px.ok());
                });
            }
        };
        if speculate {
            start(0);
        }
        let ran = match todo.split_first() {
            Some((&first, rest)) => {
                let (head, tail) = crate::parallel::join(
                    all,
                    || prefix(graph, ordering, first, seed, None, &parts.stage(0)),
                    || {
                        let work = crate::parallel::Work::ns(ordering_ns * rest.len() as f64);
                        crate::parallel::map::<crate::parallel::Rayon, _>(rest.len(), work, |i| {
                            prefix(graph, ordering, rest[i], seed, None, &parts.stage(1 + i))
                        })
                    },
                );
                std::iter::once(head).chain(tail).collect()
            }
            None => Vec::new(),
        };
        let (mut known, mut ran) = (known.map(Ok), ran.into_iter());
        let results = (r.candidates.iter()).filter_map(|m| {
            if seeded && is_amd(m) {
                known.take()
            } else {
                ran.next()
            }
        });
        let mut best: Option<Prefix> = None;
        let mut last_err: Option<RslabError> = None;
        for r in results {
            match r {
                Ok(prefix) => {
                    if best
                        .as_ref()
                        .is_none_or(|b| prefix.factor_nnz < b.factor_nnz)
                    {
                        best = Some(prefix);
                    }
                }
                Err(e) => last_err = Some(e),
            }
        }
        let seeds = match &best {
            Some(champ) if makes_nd_count(champ) => {
                if r.ensemble && prefix_flops(champ) >= r.ensemble_min_flops {
                    ensemble
                } else {
                    1
                }
            }
            _ => 0,
        };
        if seeds == 0 {
            stop.store(true, Ordering::Relaxed);
        }
        parts.decide(seeds);
        for i in 0..seeds {
            start(i);
        }
        (best, last_err, seeds)
    });
    parts.finish();
    let nd = nd
        .into_iter()
        .take(seeds)
        .filter_map(|slot| slot.into_inner().flatten())
        .reduce(|a, b| if b.factor_nnz < a.factor_nnz { b } else { a });
    if let (Some(nd), Some(champ)) = (nd, &best) {
        if nd.factor_nnz < champ.factor_nnz {
            best = Some(nd);
        }
    }
    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            RslabError::InvalidInput("ordering race: no candidate succeeded".to_string())
        })
    })
}

pub(super) fn finish(
    px: Prefix,
    full: CscPattern<i32>,
    params: &AmalgamationSettings,
) -> Result<SymbolicFactorization, RslabError> {
    let n = full.n;
    let (post, post_inv) = postorder(&px.etree);
    let mut perm: Vec<usize> = post.iter().map(|&p| px.perm[p]).collect();
    let mut etree = relabel(&px.etree, &post, &post_inv);
    let mut col_counts: Vec<usize> = post.iter().map(|&old| px.col_counts[old]).collect();

    let strategy = match params.strategy {
        AmalgamationStrategy::Auto => pick_amalgamation_strategy(&etree, params.path_like_fraction),
        concrete => concrete,
    };
    let params = AmalgamationSettings {
        strategy,
        ..params.clone()
    };
    if strategy == AmalgamationStrategy::Renumber {
        let bias = predict_merges(&etree, &col_counts, &params);
        if bias.iter().any(|&b| b) {
            let (post2, post2_inv) = biased_postorder(&etree, &bias);
            perm = post2.iter().map(|&p| perm[p]).collect();
            etree = relabel(&etree, &post2, &post2_inv);
            col_counts = post2.iter().map(|&old| col_counts[old]).collect();
        }
    }
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    debug_assert!({
        let s = structure(&full, perm.clone(), None);
        s.etree.parent == etree.parent && s.col_counts == col_counts
    });
    let supernodes = find_supernodes(&etree, &col_counts, &params);
    Ok(SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz: px.factor_nnz,
        pattern: full,
        resolved_method: px.method,
        resolved_amalgamation: strategy,
    })
}

fn relabel(etree: &EliminationTree, post: &[usize], post_inv: &[usize]) -> EliminationTree {
    EliminationTree {
        parent: post
            .iter()
            .map(|&old| etree.parent[old].map(|p| post_inv[p]))
            .collect(),
        n: etree.n,
    }
}
