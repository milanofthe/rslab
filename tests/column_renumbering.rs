//! Column-renumbering amalgamation strategy tests.
//!
//! Each test constructs a structurally simple matrix where the
//! current `Adjacency` strategy must under-merge (because the
//! postorder ordering blocks sibling merges), and asserts that the
//! `Renumber` strategy produces the SSIDS-correct number of
//! supernodes.

use rslab::symbolic::SymbolicFactorization;
use rslab::{
    AmalgamationSettings, AmalgamationStrategy, CscMatrix, OrderingMethod, OrderingSettings,
};

fn analyze(m: &CscMatrix<f64>, params: &AmalgamationSettings) -> SymbolicFactorization {
    let ordering = OrderingSettings {
        method: OrderingMethod::Amd,
        ..Default::default()
    };
    rslab::symbolic::analyze(m.n, &m.col_ptr, &m.row_idx, &ordering, params).unwrap()
}

fn arrow_matrix(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n - 1 {
        rows.push(j);
        cols.push(j);
        vals.push(2.0);
        rows.push(n - 1);
        cols.push(j);
        vals.push(-1.0);
    }
    rows.push(n - 1);
    cols.push(n - 1);
    vals.push((n - 1) as f64 + 0.5);
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

fn bushy_fan(n: usize) -> CscMatrix {
    arrow_matrix(n)
}

fn tridiagonal(n: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        rows.push(j);
        cols.push(j);
        vals.push(2.0);
        if j + 1 < n {
            rows.push(j + 1);
            cols.push(j);
            vals.push(-1.0);
        }
    }
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

#[test]
fn arrow_matrix_collapses_under_renumber() {
    let n = 8;
    let nemin = 32;
    assert!(nemin >= n, "every merge stays within the size rule");
    let m = arrow_matrix(n);
    let adj_params = AmalgamationSettings {
        nemin,
        strategy: AmalgamationStrategy::Adjacency,
        ..Default::default()
    };
    let adj_sym = analyze(&m, &adj_params);
    assert!(
        adj_sym.supernodes.len() >= 2,
        "adjacency strategy on arrow matrix should under-merge; got {} supernodes",
        adj_sym.supernodes.len()
    );
    let renum_params = AmalgamationSettings {
        nemin,
        strategy: AmalgamationStrategy::Renumber,
        ..Default::default()
    };
    let renum_sym = analyze(&m, &renum_params);
    assert_eq!(
        renum_sym.supernodes.len(),
        1,
        "renumber strategy on arrow matrix should collapse to 1 supernode"
    );
    assert_eq!(renum_sym.supernodes[0].ncol, n);
}

#[test]
fn bushy_fan_collapses_under_renumber() {
    let m = bushy_fan(33);
    let params = AmalgamationSettings {
        nemin: 64,
        strategy: AmalgamationStrategy::Renumber,
        ..Default::default()
    };
    let sym = analyze(&m, &params);
    assert_eq!(
        sym.supernodes.len(),
        1,
        "bushy fan should collapse to 1 supernode under renumber"
    );
    assert_eq!(sym.supernodes[0].ncol, 33);
}

#[test]
fn tridiagonal_renumber_is_at_least_as_aggressive() {
    let m = tridiagonal(8);
    let adj_params = AmalgamationSettings {
        nemin: 32,
        strategy: AmalgamationStrategy::Adjacency,
        ..Default::default()
    };
    let renum_params = AmalgamationSettings {
        nemin: 32,
        strategy: AmalgamationStrategy::Renumber,
        ..Default::default()
    };
    let adj = analyze(&m, &adj_params);
    let renum = analyze(&m, &renum_params);
    assert!(
        renum.supernodes.len() <= adj.supernodes.len(),
        "Renumber must not produce more supernodes than Adjacency; got {} vs {}",
        renum.supernodes.len(),
        adj.supernodes.len()
    );
    let adj_total: usize = adj.supernodes.iter().map(|s| s.ncol).sum();
    let renum_total: usize = renum.supernodes.iter().map(|s| s.ncol).sum();
    assert_eq!(adj_total, 8);
    assert_eq!(renum_total, 8);
    let ncols =
        |sym: &SymbolicFactorization| sym.supernodes.iter().map(|s| s.ncol).collect::<Vec<_>>();
    assert_eq!(
        ncols(&renum),
        ncols(&adj),
        "renumbering a chain whose postorder is already adjacent changes the supernodes"
    );
}

#[test]
fn perm_is_valid_bijection_under_renumber() {
    for n in [4, 8, 33, 64] {
        let m = arrow_matrix(n);
        let params = AmalgamationSettings {
            nemin: 32,
            strategy: AmalgamationStrategy::Renumber,
            ..Default::default()
        };
        let sym = analyze(&m, &params);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            (0..n).collect::<Vec<_>>(),
            "n={}: perm is not a bijection on 0..n",
            n
        );
        for i in 0..n {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
            assert_eq!(sym.perm_inv[sym.perm[i]], i);
        }
    }
}
