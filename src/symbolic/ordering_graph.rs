//! The graph the fill-reducing orderings run on.
//!
//! Every candidate of the ordering race orders the same graph: the full
//! symmetric pattern, held in the `i32` form the ordering crates take and
//! handed to them as it is, or the graph of its groups of indistinguishable
//! vertices when they shrink it (see [`supervariables`](super::supervariables)).
//! The compressed graph is built once per pattern and shared; per candidate
//! it cost as much as the minimum-degree ordering.

use super::progress::Reports;
use super::supervariables::Supervariables;
use super::OrderingMethod;
use crate::error::RslabError;
use crate::numeric::settings::OrderingSettings;
use crate::ordering::elimination_tree::EliminationTree;
use crate::sparse::csc::CscPattern;
use rslab_ordering_core::{Counter, Progress};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

pub(super) static ORDER_RATE: crate::parallel::Rate =
    crate::parallel::Rate::new(crate::constants::ORDER_NS_PER_ENTRY);

fn units(least: f64, ns_per_unit: f32) -> usize {
    let n = (least / ns_per_unit as f64).ceil();
    if n.is_finite() && n < usize::MAX as f64 {
        n as usize
    } else {
        usize::MAX
    }
}

pub(super) struct OrderingGraph<'a> {
    /// The full symmetric pattern (both triangles, sorted rows), the graph
    /// the ordering crates order where it is not compressed.
    pub pattern: &'a CscPattern<i32>,
    settings: &'a OrderingSettings,
    compressed: OnceLock<Option<Compressed>>,
}

struct Compressed {
    groups: Supervariables,
    weights: Vec<i32>,
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

    // https://github.com/milanofthe/rslab/issues/291
    /// Build the compressed graph now.
    pub fn prepare(&self) {
        self.compressed();
    }

    fn compressed(&self) -> Option<&Compressed> {
        let (pattern, ratio) = (self.pattern, self.settings.compress_max_ratio);
        self.compressed
            .get_or_init(|| {
                let _s = rsprobe::scope!(DEBUG, "graph");
                let groups = Supervariables::of(pattern);
                ((groups.len() as f64) <= ratio * pattern.n as f64).then(|| Compressed {
                    weights: groups.weights(),
                    graph: groups.compress(pattern),
                    groups,
                })
            })
            .as_ref()
    }

    /// Order the graph with the concrete `method` (nested dissection with
    /// `nd_seed`), and return the ordering of the pattern's vertices with its
    /// elimination tree and column counts. On the compressed graph these come
    /// from a count weighted by the group sizes, a fraction of the work on
    /// the full pattern. Nested dissection returns an error once `stop` is
    /// set. `reports` hears of the vertices ordered and of the columns of
    /// the structure.
    pub fn order(
        &self,
        method: OrderingMethod,
        nd_seed: u64,
        stop: Option<&Arc<AtomicBool>>,
        reports: &Reports,
    ) -> Result<Structure, RslabError> {
        let c = self.compressed();
        let g = c.map_or(self.pattern, |c| &c.graph);
        let pat = rslab_ordering_core::CscPattern::new(g.n, &g.col_ptr, &g.row_idx)
            .ok_or_else(|| RslabError::InvalidInput("malformed CSC pattern".to_string()))?;
        let entries = g.row_idx.len() as f64;
        let progress = &reports.ordering;
        let order = {
            match method {
                OrderingMethod::Amd => {
                    let _s = rsprobe::scope!(DEBUG, "amd");
                    rsprobe::add!(DEBUG, "entries", entries);
                    let opts = rslab_amd::AmdOptions {
                        progress: progress
                            .clone()
                            .or_else(|| self.settings.amd.progress.clone()),
                        ..self.settings.amd.clone()
                    };
                    rslab_amd::amd_order_opts(&pat, &opts).map(|(perm, _)| perm)
                }
                OrderingMethod::Amf => {
                    let _s = rsprobe::scope!(DEBUG, "amf");
                    rsprobe::add!(DEBUG, "entries", entries);
                    let opts = rslab_amf::AmfOptions {
                        progress: progress
                            .clone()
                            .or_else(|| self.settings.amf.progress.clone()),
                        ..self.settings.amf.clone()
                    };
                    rslab_amf::amf_order_opts(&pat, &opts).map(|(perm, _)| perm)
                }
                OrderingMethod::MetisND => {
                    let _s = rsprobe::scope!(DEBUG, "nested dissection");
                    rsprobe::add!(DEBUG, "entries", entries);
                    let mut opts = rslab_metis::MetisOptions {
                        seed: nd_seed,
                        stop: stop.cloned(),
                        progress: progress
                            .clone()
                            .or_else(|| self.settings.nd.progress.clone()),
                        ..self.settings.nd.clone()
                    };
                    let d = crate::parallel::rayon_dispatch();
                    let blocks = crate::constants::FINE_PIECES_PER_WORKER * d.threads();
                    opts.parallel_min_vertices.get_or_insert_with(|| {
                        units(d.least(2), crate::constants::ND_NS_PER_VERTEX)
                    });
                    opts.parallel_min_edges.get_or_insert_with(|| {
                        units(d.least(blocks), crate::constants::CONTRACT_NS_PER_ENTRY)
                    });
                    match c {
                        Some(c) => rslab_metis::metis_order_weighted(&pat, &c.weights, &opts),
                        None => rslab_metis::metis_order_full(&pat, &opts),
                    }
                    .map(|(perm, _, _)| perm)
                }
                OrderingMethod::Rcm => {
                    let _s = rsprobe::scope!(DEBUG, "rcm");
                    rslab_ordering_core::rcm_order(&pat)
                }
                OrderingMethod::Auto => unreachable!("the race resolves Auto"),
            }
        }
        .map_err(|e| RslabError::InvalidInput(format!("ordering failed: {e}")))?;
        let order = checked(&order, g.n)?;
        if let Some(p) = progress {
            p.report(g.n, g.n);
        }
        let _s = rsprobe::scope!(DEBUG, "etree");
        let progress = reports.structure.as_ref();
        Ok(match c {
            Some(c) => grouped_structure(c, &order, progress),
            None => structure(self.pattern, order, progress),
        })
    }
}

pub(super) struct Structure {
    pub perm: Vec<usize>,
    pub etree: EliminationTree,
    pub col_counts: Vec<usize>,
}

pub(super) fn structure(
    pattern: &CscPattern<i32>,
    perm: Vec<usize>,
    progress: Option<&Progress>,
) -> Structure {
    let mut perm_inv = vec![0usize; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    let read = Counter::new(progress, 2 * pattern.n);
    let cols = |new: usize| {
        read.add(1);
        pattern.col(perm[new]).iter().map(|&r| perm_inv[r as usize])
    };
    let etree = EliminationTree::from_cols(pattern.n, cols);
    let col_counts = super::column_counts::gnp(pattern.n, &etree, cols, |_| 1);
    read.finish();
    Structure {
        perm,
        etree,
        col_counts,
    }
}

fn grouped_structure(c: &Compressed, order: &[usize], progress: Option<&Progress>) -> Structure {
    let (groups, ng) = (&c.groups, order.len());
    let mut inv = vec![0usize; ng];
    for (k, &v) in order.iter().enumerate() {
        inv[v] = k;
    }
    let read = Counter::new(progress, 2 * ng);
    let cols = |k: usize| {
        read.add(1);
        c.graph.col(order[k]).iter().map(|&r| inv[r as usize])
    };
    let etree = EliminationTree::from_cols(ng, cols);
    let size = |k: usize| groups.ptr[order[k] + 1] - groups.ptr[order[k]];
    let counts = super::column_counts::gnp(ng, &etree, cols, |k| size(k) as i64);
    read.finish();
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
