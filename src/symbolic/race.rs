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
    SupernodeParams,
};
use super::{OrderingMethod, SymbolicFactorization};
use crate::error::RslabError;
use crate::ordering::amd::permute_pattern;
use crate::ordering::elimination_tree::EliminationTree;
use crate::ordering::postorder::{biased_postorder, postorder};
use crate::sparse::csc::CscPattern;

/// Seeds of the nested-dissection ensemble. Multilevel dissection is
/// seed-sensitive, mostly in one direction (on a 40^3 grid seeds 1, 3 and 4
/// give 11.2 M entries in `L`, seed 2 gives 13.2 M), so the race keeps the
/// best of several seeds where the factorization is heavy enough to pay for
/// them; the lowest seed breaks ties.
pub(super) const ND_SEED_CANDIDATES: &[u64] = &[1, 2, 3];

/// The seed of a single nested dissection: what an explicit `MetisND`
/// request and the cheap race stage use.
pub(super) const ND_SEED: u64 = 1;

/// Predicted flops (sum of squared column counts of the cheap champion) from
/// which the race runs the seed ensemble. The ensemble buys a few percent of
/// fill (up to 15 percent on 3D meshes) for two more dissections.
const ND_SEED_RACE_MIN_FLOPS: u64 = 50_000_000_000;

/// The cheap candidates, always raced: minimum degree, minimum fill, and the
/// band reducer (which wins on banded and structured patterns).
const RACE_CHEAP: &[OrderingMethod] = &[
    OrderingMethod::Amd,
    OrderingMethod::Amf,
    OrderingMethod::Rcm,
];

/// Predicted factor time (see [`prefix_work`]) from which nested dissection
/// joins the race: below it the factorization is sub-second and a dissection
/// cannot pay for itself.
const ND_RACE_MIN_WORK: u64 = 1_250_000_000;

/// Workers the time prediction assumes: the default thread cap.
const RACE_WORKERS: u64 = 4;

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
/// [`RACE_WORKERS`], or the longest elimination chain where that is longer,
/// since no worker count shortens it. Minimum-degree orderings of some 3D
/// meshes leave most of the work on one chain (a 23k waveguide: 69 percent);
/// judged by total flops alone they never met the dissection floor, although
/// dissection halved their factor time.
fn prefix_work(px: &Prefix) -> u64 {
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
    (prefix_flops(px) / RACE_WORKERS).max(longest)
}

/// The candidate for the concrete `method` (or the given permutation where
/// the parameters carry one).
pub(super) fn prefix(
    graph: &OrderingGraph,
    params: &SupernodeParams,
    method: OrderingMethod,
    nd_seed: u64,
) -> Result<Prefix, RslabError> {
    let s = match &params.given_perm {
        Some(p) => structure(graph.pattern, checked_permutation(p, graph.pattern.n)?),
        None => graph.order(method, nd_seed)?,
    };
    Ok(Prefix {
        factor_nnz: total_factor_nnz(&s.col_counts),
        perm: s.perm,
        etree: s.etree,
        col_counts: s.col_counts,
        method,
    })
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

/// Race the candidates and finish the one with the smallest exact factor.
pub(super) fn race(
    full: &CscPattern,
    params: &SupernodeParams,
) -> Result<SymbolicFactorization, RslabError> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::OnceLock;
    // Stage 1: the cheap candidates run concurrently (each prefix is itself
    // mostly sequential, so the race wall is roughly the slowest candidate);
    // the pick is deterministic - smallest exact factor nnz, candidate order
    // breaking ties - regardless of completion order.
    //
    // With more than one worker, the first nested-dissection seed starts
    // speculatively instead of after stage 1: the ND prefixes are the longest
    // part of the race, and stage 1 leaves most workers idle. On patterns
    // with at least `EAGER_ND_MIN_NNZ` entries it starts at once; on smaller
    // ones only once the AMD prefix predicts enough work to pass the ND gate,
    // because a discarded ND run on a small pattern outlasts the whole stage 1
    // (it doubled the race on a 128k power grid). The other ensemble seeds
    // start as soon as stage 1 has decided that they count. Stage 2 decides
    // from stage 1 alone which seeds count, exactly as it would have run them,
    // so the result does not depend on the thread count or on whether
    // anything was speculated.
    const EAGER_ND_MIN_NNZ: usize = 2_000_000;
    debug_assert_eq!(RACE_CHEAP[0], OrderingMethod::Amd);
    let n = full.n;
    let parallel = n > 10_000 && rayon::current_num_threads() > 1;
    // The lower triangle's entry count, as the gate was calibrated on it.
    let eager = parallel && (full.row_idx.len() + n) / 2 >= EAGER_ND_MIN_NNZ;
    // One ordering graph for every candidate: built per candidate it cost as
    // much as a minimum-degree ordering.
    let graph = &OrderingGraph::new(full);
    graph.prepare()?;
    // One slot per ensemble seed, filled by its (at most one) ND prefix.
    let nd: [OnceLock<Option<Prefix>>; ND_SEED_CANDIDATES.len()] = Default::default();
    let started: [AtomicBool; ND_SEED_CANDIDATES.len()] = Default::default();
    let (mut best, last_err, seeds) = rayon::scope(|sc| {
        let start = |i: usize| {
            if !started[i].swap(true, Ordering::Relaxed) {
                let nd = &nd;
                sc.spawn(move |_| {
                    let px = prefix(
                        graph,
                        params,
                        OrderingMethod::MetisND,
                        ND_SEED_CANDIDATES[i],
                    );
                    let _ = nd[i].set(px.ok());
                });
            }
        };
        if eager {
            start(0);
        }
        let (amd, rest) = rayon::join(
            || {
                let amd = prefix(graph, params, RACE_CHEAP[0], ND_SEED);
                if parallel && matches!(&amd, Ok(p) if prefix_work(p) >= ND_RACE_MIN_WORK) {
                    start(0);
                }
                amd
            },
            || {
                RACE_CHEAP[1..]
                    .par_iter()
                    .map(|&cand| prefix(graph, params, cand, ND_SEED))
                    .collect::<Vec<_>>()
            },
        );
        let mut best: Option<Prefix> = None;
        let mut last_err: Option<RslabError> = None;
        for r in std::iter::once(amd).chain(rest) {
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
        // Stage 2: the expensive ND candidate, only where its cost can
        // amortize (see [`ND_RACE_MIN_WORK`]), with the seed ensemble above
        // [`ND_SEED_RACE_MIN_FLOPS`].
        let seeds = match &best {
            Some(champ) if n > 10_000 && prefix_work(champ) >= ND_RACE_MIN_WORK => {
                if params.nd_ensemble && prefix_flops(champ) >= ND_SEED_RACE_MIN_FLOPS {
                    ND_SEED_CANDIDATES.len()
                } else {
                    1
                }
            }
            _ => 0,
        };
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
    let Some(winner) = best else {
        return Err(last_err.unwrap_or_else(|| {
            RslabError::InvalidInput("ordering race: no candidate succeeded".to_string())
        }));
    };
    crate::logging::timed(
        || "analysis: finish".into(),
        || finish(winner, full, params),
    )
}

/// The analysis of the adopted candidate: its ordering composed with a
/// postorder of the tree (so every supernode's columns are consecutive), the
/// merge-biased renumbering where the amalgamation asks for it, and the
/// supernodes.
pub(super) fn finish(
    px: Prefix,
    full: &CscPattern,
    params: &SupernodeParams,
) -> Result<SymbolicFactorization, RslabError> {
    let n = full.n;
    // The postorder relabels the tree; the counts carry over.
    let (post, post_inv) = postorder(&px.etree);
    let mut perm: Vec<usize> = post.iter().map(|&p| px.perm[p]).collect();
    let mut etree = relabel(&px.etree, &post, &post_inv);
    let mut col_counts: Vec<usize> = post.iter().map(|&old| px.col_counts[old]).collect();

    let strategy = match params.amalgamation_strategy {
        AmalgamationStrategy::Auto => pick_amalgamation_strategy(&etree),
        concrete => concrete,
    };
    let params = SupernodeParams {
        amalgamation_strategy: strategy,
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
    let permuted_pattern = permute_pattern(full, &perm);
    debug_assert_eq!(
        etree.parent,
        EliminationTree::from_pattern(&permuted_pattern).parent
    );
    debug_assert_eq!(
        col_counts,
        super::column_counts::column_counts_gnp(&permuted_pattern, &etree)
    );
    let supernodes = find_supernodes(&etree, &col_counts, &params);
    Ok(SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz: px.factor_nnz,
        permuted_pattern,
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
