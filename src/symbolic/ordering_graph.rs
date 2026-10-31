//! The graph the fill-reducing orderings run on.
//!
//! Every candidate of the ordering race orders the same graph: the full
//! symmetric pattern, held in the `i32` form the ordering crates take and
//! handed to them as it is, or the graph of its groups of indistinguishable
//! vertices when they shrink it (see [`supervariables`](super::supervariables)).
//! The compressed graph is built once per pattern and shared; per candidate
//! it cost as much as the minimum-degree ordering.

use super::supervariables::Supervariables;
use super::OrderingMethod;
use crate::error::RslabError;
use crate::numeric::settings::OrderingSettings;
use crate::ordering::elimination_tree::EliminationTree;
use crate::sparse::csc::CscPattern;
use std::sync::OnceLock;

pub(super) struct OrderingGraph<'a> {
    /// The full symmetric pattern (both triangles, sorted rows), the graph
    /// the ordering crates order where it is not compressed.
    pub pattern: &'a CscPattern<i32>,
    /// The options of the orderings and the compression.
    settings: &'a OrderingSettings,
    /// The compressed graph where the groups shrink the pattern enough,
    /// built on first use (a given permutation needs none). Build it with
    /// [`prepare`](Self::prepare) before orderings run concurrently: a
    /// worker waiting inside the build could otherwise steal a task that
    /// waits on the same build.
    compressed: OnceLock<Option<Compressed>>,
}

struct Compressed {
    groups: Supervariables,
    /// Group sizes, the vertex weights of the compressed graph.
    weights: Vec<i32>,
    /// One vertex per group.
    graph: CscPattern<i32>,
}

impl<'a> OrderingGraph<'a> {
    pub fn new(pattern: &'a CscPattern<i32>, settings: &'a OrderingSettings) -> Self {
        OrderingGraph {
            pattern,
            settings,
            compressed: OnceLock::new(),
        }
    }

    /// Build the compressed graph now.
    pub fn prepare(&self) {
        self.compressed();
    }

    fn compressed(&self) -> Option<&Compressed> {
        let (pattern, ratio) = (self.pattern, self.settings.compress_max_ratio);
        self.compressed
            .get_or_init(|| {
                crate::logging::timed(
                    || "analysis: ordering graph".into(),
                    || {
                        let groups = Supervariables::of(pattern);
                        ((groups.len() as f64) <= ratio * pattern.n as f64).then(|| Compressed {
                            weights: groups.weights(),
                            graph: groups.compress(pattern),
                            groups,
                        })
                    },
                )
            })
            .as_ref()
    }

    /// Order the graph with the concrete `method` (nested dissection with
    /// `nd_seed`), and return the ordering of the pattern's vertices with its
    /// elimination tree and column counts. On the compressed graph these come
    /// from a count weighted by the group sizes, a fraction of the work on
    /// the full pattern.
    pub fn order(&self, method: OrderingMethod, nd_seed: u64) -> Result<Structure, RslabError> {
        let c = self.compressed();
        let g = c.map_or(self.pattern, |c| &c.graph);
        let pat = rslab_ordering_core::CscPattern::new(g.n, &g.col_ptr, &g.row_idx)
            .ok_or_else(|| RslabError::InvalidInput("malformed CSC pattern".to_string()))?;
        let order = crate::logging::timed(
            || format!("analysis: ordering {method:?} seed {nd_seed}"),
            || match method {
                OrderingMethod::Amd => {
                    rslab_amd::amd_order_opts(&pat, &self.settings.amd).map(|(perm, _)| perm)
                }
                OrderingMethod::Amf => {
                    rslab_amf::amf_order_opts(&pat, &self.settings.amf).map(|(perm, _)| perm)
                }
                OrderingMethod::MetisND => {
                    let opts = rslab_metis::MetisOptions {
                        seed: nd_seed,
                        ..self.settings.nd.clone()
                    };
                    match c {
                        Some(c) => rslab_metis::metis_order_weighted(&pat, &c.weights, &opts),
                        None => rslab_metis::metis_order_full(&pat, &opts),
                    }
                    .map(|(perm, _, _)| perm)
                }
                OrderingMethod::Rcm => rslab_ordering_core::rcm_order(&pat),
                OrderingMethod::Auto => unreachable!("the race resolves Auto"),
            },
        )
        .map_err(|e| RslabError::InvalidInput(format!("ordering failed: {e}")))?;
        let order = checked(&order, g.n)?;
        let t = crate::clock::Instant::now();
        let s = match c {
            Some(c) => grouped_structure(c, &order),
            None => structure(self.pattern, order),
        };
        if crate::logging::enabled(crate::logging::LogLevel::Debug) {
            crate::logging::debug(&format!(
                "analysis: prefix tail {method:?} (etree, column counts): {:.1} ms",
                crate::clock::ms(t.elapsed())
            ));
        }
        Ok(s)
    }
}

/// An ordering of the pattern's vertices (new-to-old) with the elimination
/// tree and the column counts of the ordered pattern.
pub(super) struct Structure {
    pub perm: Vec<usize>,
    pub etree: EliminationTree,
    pub col_counts: Vec<usize>,
}

/// The structure under the ordering `perm` of the full `pattern`, read
/// through the permutation: column `new` is the original column
/// `perm[new]` with its rows mapped by the inverse, so the permuted pattern
/// is never built or sorted.
pub(super) fn structure(pattern: &CscPattern<i32>, perm: Vec<usize>) -> Structure {
    let mut perm_inv = vec![0usize; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    let cols = |new: usize| pattern.col(perm[new]).iter().map(|&r| perm_inv[r as usize]);
    let etree = EliminationTree::from_cols(pattern.n, cols);
    let col_counts = super::column_counts::gnp(pattern.n, &etree, cols, |_| 1);
    Structure {
        perm,
        etree,
        col_counts,
    }
}

/// The structure under the group ordering `order` of the compressed graph,
/// expanded to the vertices. The members of a group are eliminated
/// consecutively and share their structure, so they form a chain in the
/// tree whose last member hangs below the first member of the parent group,
/// and member `t` of a group with weighted count `c` has `c - t` entries.
fn grouped_structure(c: &Compressed, order: &[usize]) -> Structure {
    let (groups, ng) = (&c.groups, order.len());
    let mut inv = vec![0usize; ng];
    for (k, &v) in order.iter().enumerate() {
        inv[v] = k;
    }
    let cols = |k: usize| c.graph.col(order[k]).iter().map(|&r| inv[r as usize]);
    let etree = EliminationTree::from_cols(ng, cols);
    let size = |k: usize| groups.ptr[order[k] + 1] - groups.ptr[order[k]];
    let counts = super::column_counts::gnp(ng, &etree, cols, |k| size(k) as i64);
    let n = groups.members.len();
    let mut start = Vec::with_capacity(ng + 1);
    let mut perm = Vec::with_capacity(n);
    for &v in order {
        start.push(perm.len());
        perm.extend_from_slice(&groups.members[groups.ptr[v]..groups.ptr[v + 1]]);
    }
    start.push(n);
    let mut parent = vec![None; n];
    let mut col_counts = vec![0usize; n];
    for k in 0..ng {
        let (a, b) = (start[k], start[k + 1]);
        for t in a..b {
            col_counts[t] = counts[k] - (t - a);
            parent[t] = if t + 1 < b {
                Some(t + 1)
            } else {
                etree.parent[k].map(|kp| start[kp])
            };
        }
    }
    Structure {
        perm,
        etree: EliminationTree { parent, n },
        col_counts,
    }
}

/// An ordering from an ordering crate, checked to be a permutation of
/// `0..n`.
fn checked(order: &[i32], n: usize) -> Result<Vec<usize>, RslabError> {
    let mut seen = vec![false; n];
    let perm: Option<Vec<usize>> = order
        .iter()
        .map(|&x| {
            usize::try_from(x)
                .ok()
                .filter(|&u| u < n && !std::mem::replace(&mut seen[u], true))
        })
        .collect();
    match perm {
        Some(p) if p.len() == n => Ok(p),
        _ => Err(RslabError::InvalidInput(
            "ordering is not a permutation".to_string(),
        )),
    }
}
