//! Symbolic analysis of a symmetric pattern: the fill-reducing ordering, the
//! elimination tree, the column counts and the supernodes.
//!
//! [`analyze`] takes the lower triangle of the pattern (the symmetrized
//! pattern on the LU path). With [`OrderingMethod::Auto`] it races the
//! orderings on the exact size of their factors (see `race`); an explicit
//! method or a given permutation runs that one alone.

pub mod column_counts;
mod ordering_graph;
mod progress;
mod race;
pub mod supernode;
pub(crate) mod supervariables;

use crate::error::RslabError;
use crate::numeric::settings::{AmalgamationSettings, OrderingSettings};
use crate::sparse::csc::CscPattern;
use ordering_graph::OrderingGraph;

pub use column_counts::{column_counts_gnp, total_factor_nnz};
pub use supernode::{
    find_supernodes, pick_amalgamation_strategy, supernode_parents, AmalgamationStrategy,
    RelaxAmalgamation, Supernode,
};

/// The fill-reducing ordering of an analysis.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingMethod {
    /// Race the orderings below on the exact size of their factors and keep
    /// the smallest: minimum degree, minimum fill and the band reducer
    /// always; nested dissection where the predicted factor time can pay for
    /// it, with a seed ensemble on heavy factorizations where
    /// [`RaceSettings::ensemble`](crate::RaceSettings::ensemble) asks for it.
    #[default]
    Auto,
    /// Approximate minimum degree (Amestoy, Davis and Duff), on the
    /// quotient graph with aggressive absorption.
    Amd,
    /// Approximate minimum fill (Amestoy's HAMF): the quotient-graph
    /// elimination scored by approximate fill instead of degree.
    Amf,
    /// Multilevel nested dissection, one run; the two sides of every
    /// bisection are ordered in parallel, and the result depends on the seed
    /// only, not on the thread count.
    MetisND,
    /// Reverse Cuthill-McKee: a band and profile reducer for banded and
    /// structured patterns, where dissection over-separates.
    Rcm,
}

/// The result of the symbolic analysis, everything the numeric phase needs.
#[derive(Debug)]
pub struct SymbolicFactorization {
    /// Matrix dimension.
    pub n: usize,
    /// The ordering, new-to-old: column `perm[k]` becomes column `k`.
    pub perm: Vec<usize>,
    /// Its inverse, old-to-new.
    pub perm_inv: Vec<usize>,
    /// Supernodes in postorder (children before parents).
    pub supernodes: Vec<Supernode>,
    /// Entries of the scalar factor `L`, diagonal included (before the
    /// amalgamation adds explicit zeros).
    pub factor_nnz: usize,
    /// The full symmetric pattern the ordering was computed on, in the
    /// original numbering (`perm_inv` maps it to the factor's). The
    /// supernodal analysis empties it once its schedule is built (the
    /// pattern's last reader).
    pub pattern: CscPattern<i32>,
    /// The ordering used (the race winner under `Auto`).
    pub resolved_method: OrderingMethod,
    /// The amalgamation strategy used (`Auto` resolved).
    pub resolved_amalgamation: AmalgamationStrategy,
}

impl SymbolicFactorization {
    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        vec_bytes(&self.perm)
            + vec_bytes(&self.perm_inv)
            + vec_bytes(&self.supernodes)
            + self
                .supernodes
                .iter()
                .map(|s| vec_bytes(&s.children))
                .sum::<u64>()
            + vec_bytes(&self.pattern.col_ptr)
            + vec_bytes(&self.pattern.row_idx)
    }
}

/// The full symmetric pattern of the lower triangle (`col_ptr`, `row_idx`)
/// of an `n x n` pattern, in the `i32` form the analysis orders.
pub fn full_pattern(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Result<CscPattern<i32>, RslabError> {
    let _s = rsprobe::scope!(DEBUG, "pattern");
    crate::sparse::csc::symmetric_pattern(n, col_ptr, row_idx).ok_or_else(|| {
        RslabError::InvalidInput("matrix too large for i32-indexed ordering crates".to_string())
    })
}

/// Analyze the lower triangle (`col_ptr`, `row_idx`) of an `n x n` pattern.
pub fn analyze(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    ordering: &OrderingSettings,
    amalgamation: &AmalgamationSettings,
) -> Result<SymbolicFactorization, RslabError> {
    analyze_pattern(full_pattern(n, col_ptr, row_idx)?, ordering, amalgamation)
}

/// Analyze a full symmetric pattern ([`full_pattern`]), which the result
/// keeps for the schedule.
pub fn analyze_pattern(
    full: CscPattern<i32>,
    ordering: &OrderingSettings,
    amalgamation: &AmalgamationSettings,
) -> Result<SymbolicFactorization, RslabError> {
    analyze_pattern_from(full, ordering, amalgamation, None)
}

pub(crate) fn analyze_pattern_from(
    full: CscPattern<i32>,
    ordering: &OrderingSettings,
    amalgamation: &AmalgamationSettings,
    amd: Option<&[usize]>,
) -> Result<SymbolicFactorization, RslabError> {
    let method = ordering.method;
    let px = {
        let _s = rsprobe::scope!(DEBUG, "ordering");
        let graph = OrderingGraph::new(&full, ordering);
        if method == OrderingMethod::Auto && ordering.permutation.is_none() {
            race::race(&graph, ordering, amd)?
        } else {
            race::alone(&graph, ordering, method)?
        }
    };
    let _s = rsprobe::scope!(DEBUG, "supernodes");
    race::finish(px, full, amalgamation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;
    use crate::SolverSettings;

    fn try_run(
        m: &CscMatrix<f64>,
        s: &SolverSettings,
        method: OrderingMethod,
    ) -> Result<SymbolicFactorization, RslabError> {
        let ordering = OrderingSettings {
            method,
            ..s.ordering.clone()
        };
        analyze(m.n, &m.col_ptr, &m.row_idx, &ordering, &s.amalgamation)
    }

    fn run(
        m: &CscMatrix<f64>,
        s: &SolverSettings,
        method: OrderingMethod,
    ) -> SymbolicFactorization {
        try_run(m, s, method).unwrap()
    }

    fn grid(k: usize) -> CscMatrix<f64> {
        let (mut r, mut c) = (Vec::new(), Vec::new());
        for y in 0..k {
            for x in 0..k {
                let i = y * k + x;
                r.push(i);
                c.push(i);
                if x + 1 < k {
                    r.push(i + 1);
                    c.push(i);
                }
                if y + 1 < k {
                    r.push(i + k);
                    c.push(i);
                }
            }
        }
        CscMatrix::from_triplets(k * k, &r, &c, &vec![1.0; r.len()]).unwrap()
    }

    fn assert_permutation(p: &[usize], n: usize) {
        let mut seen = vec![false; n];
        for &k in p {
            assert!(k < n && !seen[k], "not a permutation");
            seen[k] = true;
        }
    }

    #[test]
    fn every_method_gives_a_consistent_analysis() {
        let a = grid(30);
        for method in [
            OrderingMethod::Auto,
            OrderingMethod::Amd,
            OrderingMethod::Amf,
            OrderingMethod::MetisND,
            OrderingMethod::Rcm,
        ] {
            let s = run(&a, &SolverSettings::default(), method);
            assert_permutation(&s.perm, a.n);
            assert!((0..a.n).all(|k| s.perm_inv[s.perm[k]] == k));
            assert_eq!(s.supernodes.iter().map(|sn| sn.ncol).sum::<usize>(), a.n);
            assert!(s.supernodes.iter().all(|sn| sn.nrow >= sn.ncol));
            assert!(s.factor_nnz >= a.n);
            if method != OrderingMethod::Auto {
                assert_eq!(s.resolved_method, method);
            }
        }
    }

    #[test]
    fn the_race_is_never_worse_than_its_candidates() {
        let a = grid(40);
        let raced = run(&a, &SolverSettings::default(), OrderingMethod::Auto).factor_nnz;
        for method in [
            OrderingMethod::Amd,
            OrderingMethod::Amf,
            OrderingMethod::Rcm,
        ] {
            assert!(raced <= run(&a, &SolverSettings::default(), method).factor_nnz);
        }
    }

    #[test]
    fn a_given_minimum_degree_ordering_runs_in_its_place() {
        let a = grid(40);
        let s = SolverSettings::default();
        let full = full_pattern(a.n, &a.col_ptr, &a.row_idx).unwrap();
        let own = OrderingGraph::new(&full, &s.ordering)
            .order(
                OrderingMethod::Amd,
                s.ordering.nd.seed,
                None,
                &Default::default(),
            )
            .unwrap();
        let analyze = |amd: Option<&[usize]>| {
            analyze_pattern_from(full.clone(), &s.ordering, &s.amalgamation, amd).unwrap()
        };
        let (raced, given) = (analyze(None), analyze(Some(&own.perm)));
        assert_eq!(raced.perm, given.perm);
        assert_eq!(raced.factor_nnz, given.factor_nnz);
        let natural: Vec<usize> = (0..a.n).collect();
        let other = analyze(Some(&natural));
        assert_permutation(&other.perm, a.n);
        assert!(other.factor_nnz <= run(&a, &s, OrderingMethod::Amf).factor_nnz);
    }

    #[test]
    fn the_race_settings_steer_nested_dissection() {
        let a = grid(40);
        let mut s = SolverSettings::default();
        s.ordering.race.nd_min_n = 0;
        s.ordering.race.nd_min_work = 0;
        let one = run(&a, &s, OrderingMethod::Auto);
        let nd = run(&a, &s, OrderingMethod::MetisND);
        assert!(one.factor_nnz <= nd.factor_nnz);
        s.ordering.race.ensemble = true;
        s.ordering.race.ensemble_min_flops = 0;
        let ens = run(&a, &s, OrderingMethod::Auto);
        assert!(ens.factor_nnz <= one.factor_nnz);
        let mut rcm_only = SolverSettings::default();
        rcm_only.ordering.race.candidates = vec![OrderingMethod::Rcm];
        let r = run(&a, &rcm_only, OrderingMethod::Auto);
        assert_eq!(r.resolved_method, OrderingMethod::Rcm);
        for bad in [
            vec![],
            vec![OrderingMethod::MetisND],
            vec![OrderingMethod::Auto],
        ] {
            let mut s = SolverSettings::default();
            s.ordering.race.candidates = bad;
            assert!(try_run(&a, &s, OrderingMethod::Auto).is_err());
        }
    }

    #[test]
    fn a_given_permutation_is_honoured() {
        let a = grid(10);
        let rev: Vec<usize> = (0..a.n).rev().collect();
        let params = SolverSettings::default().with_permutation(rev.clone().into());
        let s = run(&a, &params, OrderingMethod::Auto);
        let direct = run(&a, &params, OrderingMethod::Amd);
        assert_eq!(s.factor_nnz, direct.factor_nnz);
        let bad = SolverSettings::default().with_permutation(vec![0; a.n].into());
        assert!(try_run(&a, &bad, OrderingMethod::Amd).is_err());
    }

    #[test]
    fn supernode_heights_match_the_built_row_sets() {
        use crate::numeric::supernodal::LlSchedule;
        let a = grid(20);
        for nemin in [1usize, 16, 32] {
            let params = SolverSettings::default().with_nemin(nemin);
            let sym = run(&a, &params, OrderingMethod::Amd);
            let sched = LlSchedule::build(&sym);
            assert!(
                sym.supernodes.iter().any(|sn| sn.ncol > 1),
                "nothing merged"
            );
            for (s, sn) in sym.supernodes.iter().enumerate() {
                assert_eq!(sn.nrow, sched.rows(s).len(), "nemin={nemin}, supernode {s}");
            }
        }
    }

    #[test]
    fn the_grouped_structure_equals_the_full_one() {
        let (k, d) = (16usize, 2usize);
        let (mut r, mut c) = (Vec::new(), Vec::new());
        let node = |x: usize, y: usize| y * k + x;
        for y in 0..k {
            for x in 0..k {
                let mut nbrs = vec![node(x, y)];
                if x + 1 < k {
                    nbrs.push(node(x + 1, y));
                }
                if y + 1 < k {
                    nbrs.push(node(x, y + 1));
                }
                for &m in &nbrs {
                    for a in 0..d {
                        for b in 0..d {
                            let (i, j) = (m * d + a, node(x, y) * d + b);
                            if i >= j {
                                r.push(i);
                                c.push(j);
                            }
                        }
                    }
                }
            }
        }
        let n = k * k * d;
        let a = CscMatrix::from_triplets(n, &r, &c, &vec![1.0; r.len()]).unwrap();
        let full = full_pattern(n, &a.col_ptr, &a.row_idx).unwrap();
        let settings = OrderingSettings::default();
        let graph = OrderingGraph::new(&full, &settings);
        for method in [
            OrderingMethod::Amd,
            OrderingMethod::Amf,
            OrderingMethod::MetisND,
            OrderingMethod::Rcm,
        ] {
            let grouped = graph.order(method, 1, None, &Default::default()).unwrap();
            let direct = ordering_graph::structure(&full, grouped.perm.clone(), None);
            assert_eq!(grouped.etree.parent, direct.etree.parent, "{method:?}");
            assert_eq!(grouped.col_counts, direct.col_counts, "{method:?}");
        }
    }

    #[test]
    fn a_dense_pattern_is_one_supernode() {
        let n = 12;
        let (mut r, mut c) = (Vec::new(), Vec::new());
        for j in 0..n {
            for i in j..n {
                r.push(i);
                c.push(j);
            }
        }
        let a = CscMatrix::from_triplets(n, &r, &c, &vec![1.0; r.len()]).unwrap();
        let s = run(&a, &SolverSettings::default(), OrderingMethod::Amd);
        assert_eq!(s.supernodes.len(), 1);
        assert_eq!(s.factor_nnz, n * (n + 1) / 2);
    }
}
