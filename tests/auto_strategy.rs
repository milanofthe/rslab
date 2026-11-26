//! `AmalgamationStrategy::Auto` dispatch tests.
//!
//! Verifies the cheap O(n) etree shape predicate routes:
//!   * pure path etree -> `Adjacency` (Renumber over-merging case)
//!   * bushy etree -> `Renumber` (IPM-KKT amalgamation case)
//!   * empty / leaf-only forest -> `Adjacency`

#![allow(clippy::assertions_on_constants, clippy::needless_range_loop)]
use rslab::ordering::elimination_tree::EliminationTree;
use rslab::symbolic::{pick_amalgamation_strategy, AmalgamationStrategy};

fn default_fraction() -> f64 {
    rslab::AmalgamationSettings::default().path_like_fraction
}

fn pick(etree: &EliminationTree) -> AmalgamationStrategy {
    pick_amalgamation_strategy(etree, default_fraction())
}

fn path_etree(n: usize) -> EliminationTree {
    let mut parent: Vec<Option<usize>> = vec![None; n];
    for i in 0..n.saturating_sub(1) {
        parent[i] = Some(i + 1);
    }
    EliminationTree { parent, n }
}

fn binary_tree_etree(depth: usize) -> EliminationTree {
    let n = (1usize << (depth + 1)) - 1;
    let mut new_label = vec![0usize; n];
    let mut counter = 0usize;
    fn visit(i: usize, n: usize, new_label: &mut [usize], counter: &mut usize) {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n {
            visit(l, n, new_label, counter);
        }
        if r < n {
            visit(r, n, new_label, counter);
        }
        new_label[i] = *counter;
        *counter += 1;
    }
    visit(0, n, &mut new_label, &mut counter);

    let mut parent: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n {
            parent[new_label[l]] = Some(new_label[i]);
        }
        if r < n {
            parent[new_label[r]] = Some(new_label[i]);
        }
    }
    assert!(
        parent
            .iter()
            .enumerate()
            .all(|(j, p)| p.is_none_or(|p| p > j)),
        "the postorder labels put every parent after its children"
    );
    EliminationTree { parent, n }
}

#[test]
fn path_dispatches_to_adjacency() {
    let etree = path_etree(100);
    assert_eq!(
        pick(&etree),
        AmalgamationStrategy::Adjacency,
        "pure path must dispatch Adjacency"
    );
}

#[test]
fn complete_binary_tree_dispatches_to_renumber() {
    let etree = binary_tree_etree(5);
    assert_eq!(etree.n, 63, "nodes of a complete binary tree of depth 5");
    assert_eq!(
        pick(&etree),
        AmalgamationStrategy::Renumber,
        "complete binary tree must dispatch Renumber"
    );
}

#[test]
fn empty_etree_dispatches_to_adjacency() {
    let etree = EliminationTree {
        parent: Vec::new(),
        n: 0,
    };
    assert_eq!(pick(&etree), AmalgamationStrategy::Adjacency,);
}

#[test]
fn leaf_only_forest_dispatches_to_adjacency() {
    let etree = EliminationTree {
        parent: vec![None; 10],
        n: 10,
    };
    assert_eq!(
        pick(&etree),
        AmalgamationStrategy::Adjacency,
        "no internal nodes means no merging opportunities; fall back"
    );
}

#[test]
fn auto_default_resolves_under_threshold() {
    let etree = path_etree(5);
    let strat = pick(&etree);
    assert_eq!(strat, AmalgamationStrategy::Adjacency);
}

#[test]
fn near_path_with_one_branch_still_adjacency() {
    let n = 100;
    let n2 = n + 1;
    let mut parent2 = vec![None::<usize>; n2];
    for i in 0..n - 1 {
        parent2[i] = Some(i + 1);
    }
    let junction = 50;
    parent2[n] = Some(junction);
    let etree = EliminationTree {
        parent: parent2,
        n: n2,
    };
    let (multi_child, internal) = (1.0, (n - 1) as f64);
    assert!(
        multi_child / internal < default_fraction(),
        "one junction among {internal} internal nodes is below the path-like fraction"
    );
    assert_eq!(pick(&etree), AmalgamationStrategy::Adjacency,);
}

#[test]
fn fan_at_root_dispatches_to_renumber() {
    assert!(
        default_fraction() <= 1.0,
        "a root as the only internal node reaches the path-like fraction"
    );
    let n = 20;
    let mut parent = vec![None::<usize>; n];
    for i in 0..n - 1 {
        parent[i] = Some(n - 1);
    }
    let etree = EliminationTree { parent, n };
    assert_eq!(pick(&etree), AmalgamationStrategy::Renumber,);
}
