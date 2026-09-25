//! The graph the fill-reducing orderings run on.
//!
//! Every candidate of the ordering race orders the same graph: the full
//! symmetric pattern, compressed to its groups of indistinguishable vertices
//! when they shrink it (see [`supervariables`](super::supervariables)), in
//! the `i32` form the ordering crates take. It is built once per pattern and
//! shared; per candidate it cost as much as the minimum-degree ordering.

use super::supervariables::Supervariables;
use super::OrderingMethod;
use crate::error::RslabError;
use crate::ordering::elimination_tree::EliminationTree;
use crate::sparse::csc::CscPattern;
use std::sync::OnceLock;

/// Compress the ordering graph only when the groups of indistinguishable
/// vertices shrink it to at most this share of its vertices.
const COMPRESS_MAX_RATIO: f64 = 0.95;

pub(super) struct OrderingGraph<'a> {
    /// The full symmetric pattern (both triangles, sorted rows).
    pub pattern: &'a CscPattern,
    /// The graph handed to the ordering crates, built on first use (a given
    /// permutation needs none). Build it with [`prepare`](Self::prepare)
    /// before orderings run concurrently: a worker waiting inside the build
    /// could otherwise steal a task that waits on the same build.
    ordered: OnceLock<Result<Ordered, String>>,
}

struct Ordered {
    /// The groups, when the graph is ordered compressed.
    groups: Option<Supervariables>,
    /// Group sizes, the vertex weights of the compressed graph.
    weights: Option<Vec<i32>>,
    /// The ordered graph (compressed or not) as `i32` arrays.
    col_ptr: Vec<i32>,
    row_idx: Vec<i32>,
}

impl<'a> OrderingGraph<'a> {
    pub fn new(pattern: &'a CscPattern) -> Self {
        OrderingGraph {
            pattern,
            ordered: OnceLock::new(),
        }
    }

    /// Build the ordered graph now.
    pub fn prepare(&self) -> Result<(), RslabError> {
        self.ordered().map(|_| ())
    }

    fn ordered(&self) -> Result<&Ordered, RslabError> {
        let pattern = self.pattern;
        self.ordered
            .get_or_init(|| {
                crate::logging::timed(
                    || "analysis: ordering graph".into(),
                    || {
                        let groups = Supervariables::of(pattern);
                        let groups = ((groups.len() as f64)
                            <= COMPRESS_MAX_RATIO * pattern.n as f64)
                            .then_some(groups);
                        let compressed = groups.as_ref().map(|g| g.compress(pattern));
                        let (col_ptr, row_idx) = to_i32(compressed.as_ref().unwrap_or(pattern))?;
                        Ok(Ordered {
                            weights: groups.as_ref().map(Supervariables::weights),
                            groups,
                            col_ptr,
                            row_idx,
                        })
                    },
                )
            })
            .as_ref()
            .map_err(|e| RslabError::InvalidInput(e.clone()))
    }

    /// Order the graph with the concrete `method` (the seeds apply to
    /// nested dissection) and return the permutation of the pattern's
    /// vertices, new-to-old: `perm[k]` is the original column that becomes
    /// column `k`.
    pub fn order(
        &self,
        method: OrderingMethod,
        nd_seeds: &[u64],
    ) -> Result<Vec<usize>, RslabError> {
        let n = self.pattern.n;
        let g = self.ordered()?;
        let pat = rslab_ordering_core::CscPattern::new(g.col_ptr.len() - 1, &g.col_ptr, &g.row_idx)
            .ok_or_else(|| RslabError::InvalidInput("malformed CSC pattern".to_string()))?;
        let expand = |p: Vec<i32>| match &g.groups {
            Some(g) => g.expand(&p),
            None => p,
        };
        let perm = match method {
            OrderingMethod::Amd => rslab_amd::amd_order(&pat).map(expand),
            OrderingMethod::Amf => rslab_amf::amf_order(&pat).map(expand),
            OrderingMethod::MetisND => self.metis_seed_race(g, &pat, nd_seeds),
            OrderingMethod::Rcm => rslab_ordering_core::rcm_order(&pat).map(expand),
            OrderingMethod::Auto | OrderingMethod::AutoRace => {
                unreachable!("resolved by symbolic_factorize_with_method")
            }
        }
        .map_err(|e| RslabError::InvalidInput(format!("external ordering failed: {e}")))?;
        if perm.len() != n {
            return Err(RslabError::InvalidInput(format!(
                "external ordering returned {} entries for n={n}",
                perm.len()
            )));
        }
        perm.into_iter()
            .map(|x| {
                usize::try_from(x).ok().filter(|&u| u < n).ok_or_else(|| {
                    RslabError::InvalidInput(
                        "external ordering returned an out-of-range index".to_string(),
                    )
                })
            })
            .collect()
    }

    /// Best-of-seeds nested dissection (see `ND_SEED_CANDIDATES`): each seed
    /// scored by its exact scalar nnz(L), the lowest seed breaking ties.
    fn metis_seed_race(
        &self,
        g: &Ordered,
        pat: &rslab_ordering_core::CscPattern<'_>,
        seeds: &[u64],
    ) -> Result<Vec<i32>, rslab_ordering_core::OrderingError> {
        use rayon::prelude::*;
        // One nested dissection of `pat` (weighted when it is the compressed
        // graph), returned as an ordering of the pattern's vertices.
        let order = |seed: u64| -> Result<Vec<i32>, rslab_ordering_core::OrderingError> {
            let opts = rslab_metis::MetisOptions {
                seed,
                ..Default::default()
            };
            let (perm, _, _) = match &g.weights {
                Some(w) => rslab_metis::metis_order_weighted(pat, w, &opts)?,
                None => rslab_metis::metis_order_full(pat, &opts)?,
            };
            Ok(match &g.groups {
                Some(groups) => groups.expand(&perm),
                None => perm,
            })
        };
        if let [seed] = seeds {
            return order(*seed);
        }
        let scored: Vec<(usize, u64, Vec<i32>)> = seeds
            .par_iter()
            .filter_map(|&seed| {
                let perm_i32 = order(seed).ok()?;
                Some((fill(self.pattern, &perm_i32), seed, perm_i32))
            })
            .collect();
        scored
            .into_iter()
            .min_by_key(|&(fill, seed, _)| (fill, seed))
            .map(|(_, _, perm)| perm)
            .ok_or(rslab_ordering_core::OrderingError::MalformedInput)
    }
}

/// Exact scalar nnz(L) of `pattern` under the ordering `perm` (new-to-old),
/// read through the permutation.
fn fill(pattern: &CscPattern, perm: &[i32]) -> usize {
    let perm: Vec<usize> = perm.iter().map(|&x| x as usize).collect();
    let mut perm_inv = vec![0usize; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    let etree = EliminationTree::from_permuted_pattern(pattern, &perm, &perm_inv);
    super::total_factor_nnz(&super::column_counts_permuted(
        pattern, &perm, &perm_inv, &etree,
    ))
}

/// The pattern as the `i32` arrays of the ordering crates.
fn to_i32(pattern: &CscPattern) -> Result<(Vec<i32>, Vec<i32>), String> {
    let conv = |v: &[usize]| -> Result<Vec<i32>, String> {
        v.iter()
            .map(|&x| i32::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| "matrix too large for i32-indexed ordering crates".to_string())
    };
    Ok((conv(&pattern.col_ptr)?, conv(&pattern.row_idx)?))
}
