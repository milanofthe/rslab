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

/// A candidate: its ordering (new-to-old, before the postorder) with the
/// elimination tree, column counts and exact scalar factor size under it.
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

/// Predicted factor time of a candidate in flops: the work shared among
/// `workers`, or the longest elimination chain where that is longer,
/// since no worker count shortens it. Minimum-degree orderings of some 3D
/// meshes leave most of the work on one chain (a 23k waveguide: 69 percent);
/// judged by total flops alone they never met the dissection floor, although
/// dissection halved their factor time.
fn prefix_work(px: &Prefix, workers: usize) -> u64 {
    // Every child precedes its parent in an elimination tree.
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

/// The candidate for the concrete `method` (or the given permutation where
/// the settings carry one); nested dissection fails once `stop` is set.
pub(super) fn prefix(
    graph: &OrderingGraph,
    ordering: &OrderingSettings,
    method: OrderingMethod,
    nd_seed: u64,
    stop: Option<&Arc<AtomicBool>>,
) -> Result<Prefix, RslabError> {
    match &ordering.permutation {
        Some(p) => given(graph, p, method),
        None => Ok(from(graph.order(method, nd_seed, stop)?, method)),
    }
}

/// The candidate of the given ordering `p` of the pattern, labelled `method`.
fn given(graph: &OrderingGraph, p: &[usize], method: OrderingMethod) -> Result<Prefix, RslabError> {
    let s = structure(graph.pattern, checked_permutation(p, graph.pattern.n)?);
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

/// A given ordering, checked to be a permutation of `0..n`.
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

/// Race the candidates on `graph`; the one with the smallest exact factor.
/// `amd`, where given, is the minimum-degree candidate's ordering, computed
/// before the race (the automatic path's KLU analysis orders the pattern so
/// to route it): the race takes it in place of its own and knows its
/// predicted work before stage 1.
pub(super) fn race(
    graph: &OrderingGraph,
    ordering: &OrderingSettings,
    amd: Option<&[usize]>,
) -> Result<Prefix, RslabError> {
    use std::sync::atomic::Ordering;
    use std::sync::OnceLock;
    // Stage 1: the cheap candidates run concurrently (each prefix is itself
    // mostly sequential, so the race wall is roughly the slowest candidate);
    // the pick is deterministic - smallest exact factor nnz, candidate order
    // breaking ties - regardless of completion order.
    //
    // With more than one worker, the first nested-dissection seed starts
    // speculatively with stage 1 instead of after it: the ND prefixes are
    // the longest part of the race, and stage 1 leaves most workers idle.
    // Where stage 1 decides that no seed counts, the speculative one is
    // stopped, so a discarded dissection ends with stage 1 instead of
    // outlasting it; where a candidate known before stage 1 (a given
    // minimum-degree ordering) predicts too little work for any seed to
    // count, none is speculated. The other ensemble seeds start as soon as
    // stage 1 has decided that they count. Stage 2 decides from stage 1
    // alone which seeds count, exactly as it would have run them, so the
    // result does not depend on the thread count or on whether anything
    // was speculated.
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
    // Whether a champion makes nested dissection count.
    let counts =
        |px: &Prefix| n > r.nd_min_n && prefix_work(px, r.assumed_workers) >= r.nd_min_work;
    let is_amd = |m: &OrderingMethod| *m == OrderingMethod::Amd;
    // The candidates' work, estimated at what an ordering has cost per entry
    // (learned where an ordering runs, `ordering_graph`): a candidate's
    // prefix is more than its ordering, so the race is not measured for
    // that rate.
    let ordering_ns =
        graph.pattern.row_idx.len() as f64 * super::ordering_graph::ORDER_RATE.ns_per_op();
    // One ordering graph for every candidate: built per candidate it cost as
    // much as a minimum-degree ordering. The given candidate's structure
    // (at most an ordering's work) is counted beside it.
    let known = match amd {
        Some(p) if r.candidates.iter().any(is_amd) => {
            let work = crate::parallel::Work::ns(ordering_ns);
            let (known, ()) = crate::parallel::join(
                work,
                || given(graph, p, OrderingMethod::Amd),
                || graph.prepare(),
            );
            Some(known?)
        }
        _ => {
            graph.prepare();
            None
        }
    };
    let seeded = known.is_some();
    // The candidates stage 1 orders.
    let todo: Vec<OrderingMethod> = (r.candidates.iter())
        .filter(|m| !(seeded && is_amd(m)))
        .copied()
        .collect();
    // The candidates side by side where their orderings are worth more
    // than one thread; nested dissection is speculated on only then.
    let all = crate::parallel::Work::ns(ordering_ns * todo.len() as f64);
    let side = crate::parallel::width(all.time(), todo.len()) > 1;
    let speculate = n > r.nd_min_n && side && known.as_ref().is_none_or(counts);
    // Set where stage 1 discards the speculative dissection.
    let stop = Arc::new(AtomicBool::new(false));
    // One slot per ensemble seed, filled by its (at most one) ND prefix.
    let ensemble = r.ensemble_size.max(1);
    let nd: Vec<OnceLock<Option<Prefix>>> = (0..ensemble).map(|_| OnceLock::new()).collect();
    let started: Vec<AtomicBool> = (0..ensemble).map(|_| AtomicBool::new(false)).collect();
    // Nested dissection, where it may run, is worth a scope on its own.
    let scoped = if n > r.nd_min_n {
        crate::parallel::Work::ns(f64::INFINITY)
    } else {
        all
    };
    let (mut best, last_err, seeds) = crate::parallel::scope(scoped, |sc| {
        let start = |i: usize| {
            if !started[i].swap(true, Ordering::Relaxed) {
                let (nd, stop) = (&nd, &stop);
                sc.spawn(move |_| {
                    let px = prefix(
                        graph,
                        ordering,
                        OrderingMethod::MetisND,
                        seed + i as u64,
                        Some(stop),
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
                    || prefix(graph, ordering, first, seed, None),
                    || {
                        let work = crate::parallel::Work::ns(ordering_ns * rest.len() as f64);
                        crate::parallel::map(rest.len(), work, |i| {
                            prefix(graph, ordering, rest[i], seed, None)
                        })
                    },
                );
                std::iter::once(head).chain(tail).collect()
            }
            None => Vec::new(),
        };
        // In candidate order, the known one in its place.
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
        // Stage 2: nested dissection, only where its cost can amortize, with
        // the seed ensemble above `ensemble_min_flops`.
        let seeds = match &best {
            Some(champ) if counts(champ) => {
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
        for i in 0..seeds {
            start(i);
        }
        (best, last_err, seeds)
    });
    // The ensemble's pick: the smallest exact factor nnz, the lowest seed
    // breaking ties; it replaces the cheap champion only when smaller.
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

/// The analysis of the adopted candidate: its ordering composed with a
/// postorder of the tree (so every supernode's columns are consecutive), the
/// merge-biased renumbering where the amalgamation asks for it, and the
/// supernodes. `full` is the pattern it was ordered on, kept for the
/// schedule.
pub(super) fn finish(
    px: Prefix,
    full: CscPattern<i32>,
    params: &AmalgamationSettings,
) -> Result<SymbolicFactorization, RslabError> {
    let n = full.n;
    // The postorder relabels the tree; the counts carry over.
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
    // Renumber: a second postorder placing the children the amalgamation
    // wants to merge next to their parents, so their columns become
    // consecutive and the merges can happen.
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
        let s = structure(&full, perm.clone());
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

/// The tree relabelled by a postorder `post` (new-to-old) with inverse
/// `post_inv`.
fn relabel(etree: &EliminationTree, post: &[usize], post_inv: &[usize]) -> EliminationTree {
    EliminationTree {
        parent: post
            .iter()
            .map(|&old| etree.parent[old].map(|p| post_inv[p]))
            .collect(),
        n: etree.n,
    }
}
