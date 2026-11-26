//! Oracle match: compare `rslab-amd` output against the pinned
//! SuiteSparse AMD fixtures under `tests/data/amd_oracle/`.
//!
//! rslab-amd's output matches the SuiteSparse AMD reference
//! byte-for-byte. We assert:
//!
//! - permutation exactly matches the oracle;
//! - `ncmpa`, `ndiv`, `nms_ldl`, `nms_lu`, `n_dense_deferred`
//!   exactly match the oracle.
//!
//! A few focused tests additionally exercise the `n_mass_elim` and
//! `n_supervar_merge` counters so a regression that silently
//! disables mass elimination or supervariable detection would surface
//! even if the perm happened to still line up.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rslab_amd::{amd_order_with_stats, CscPattern};

#[path = "common/patterns.rs"]
mod patterns;
use patterns::{arrow, csc_from_triples, grid_2d};

fn band(n: usize, b: usize) -> (Vec<i32>, Vec<i32>) {
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i));
        for k in 1..=b {
            if i + k < n {
                t.push((i, i + k));
            }
        }
    }
    csc_from_triples(n, &t)
}

struct Oracle {
    n: usize,
    ncmpa: u32,
    n_dense: u32,
    ndiv: u64,
    nms_ldl: u64,
    nms_lu: u64,
    perm: Vec<i32>,
}

fn parse_oracle(path: &Path) -> Oracle {
    let text = fs::read_to_string(path).expect("read oracle fixture");
    let mut map: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        map.insert(k.trim().to_string(), v.trim().to_string());
    }
    let n: usize = map["n"].parse().expect("parse n");
    let ncmpa: u32 = map["ncmpa"].parse().expect("parse ncmpa");
    let n_dense: u32 = map["n_dense"].parse().expect("parse n_dense");
    let ndiv: u64 = map["ndiv"].parse().expect("parse ndiv");
    let nms_ldl: u64 = map["nms_ldl"].parse().expect("parse nms_ldl");
    let nms_lu: u64 = map["nms_lu"].parse().expect("parse nms_lu");
    let perm: Vec<i32> = map["perm"]
        .split_whitespace()
        .map(|s| s.parse().expect("parse perm entry"))
        .collect();
    assert_eq!(perm.len(), n, "oracle perm length mismatch in {:?}", path);
    Oracle {
        n,
        ncmpa,
        n_dense,
        ndiv,
        nms_ldl,
        nms_lu,
        perm,
    }
}

fn oracle_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/amd_oracle")
        .join(format!("{}.txt", name))
}

fn run_fixture(name: &str, cp: &[i32], ri: &[i32]) {
    let oracle = parse_oracle(&oracle_path(name));
    let pattern = CscPattern::new(oracle.n, cp, ri).expect("valid CSC");
    let (perm, stats) = amd_order_with_stats(&pattern).expect("amd_order");
    assert_eq!(perm, oracle.perm, "{}: permutation mismatch", name);
    assert_eq!(stats.ncmpa, oracle.ncmpa, "{}: ncmpa mismatch", name);
    assert_eq!(
        stats.n_dense_deferred, oracle.n_dense,
        "{}: n_dense_deferred mismatch",
        name
    );
    assert_eq!(stats.ndiv, oracle.ndiv, "{}: ndiv mismatch", name);
    assert_eq!(stats.nms_ldl, oracle.nms_ldl, "{}: nms_ldl mismatch", name);
    assert_eq!(stats.nms_lu, oracle.nms_lu, "{}: nms_lu mismatch", name);
}

#[test]
fn oracle_diag_4() {
    let (cp, ri) = band(4, 0);
    run_fixture("diag_4", &cp, &ri);
}

#[test]
fn oracle_tridiag_10() {
    let (cp, ri) = band(10, 1);
    run_fixture("tridiag_10", &cp, &ri);
}

#[test]
fn oracle_arrow_5() {
    let (cp, ri) = arrow(5);
    run_fixture("arrow_5", &cp, &ri);
}

#[test]
fn oracle_arrow_200() {
    let (cp, ri) = arrow(200);
    run_fixture("arrow_200", &cp, &ri);
}

#[test]
fn oracle_band_20_3() {
    let (cp, ri) = band(20, 3);
    run_fixture("band_20_3", &cp, &ri);
}

#[test]
fn oracle_grid_7x7() {
    let (cp, ri) = grid_2d(7, 7);
    run_fixture("grid_7x7", &cp, &ri);
}

#[test]
fn oracle_amd_demo_24() {
    let (cp, ri) = grid_2d(6, 4);
    run_fixture("amd_demo_24", &cp, &ri);
}

#[test]
fn mass_elim_fires_on_tridiag_10() {
    let (cp, ri) = band(10, 1);
    let pattern = CscPattern::new(10, &cp, &ri).unwrap();
    let (_perm, stats) = amd_order_with_stats(&pattern).unwrap();
    assert!(
        stats.n_mass_elim > 0,
        "tridiag_10 must trigger mass elimination (got {})",
        stats.n_mass_elim
    );
}

#[test]
fn mass_elim_fires_on_band_20_3() {
    let (cp, ri) = band(20, 3);
    let pattern = CscPattern::new(20, &cp, &ri).unwrap();
    let (_perm, stats) = amd_order_with_stats(&pattern).unwrap();
    assert!(
        stats.n_mass_elim > 0,
        "band_20_3 must trigger mass elimination (got {})",
        stats.n_mass_elim
    );
}

#[test]
fn mass_elim_fires_on_grid_7x7() {
    let (cp, ri) = grid_2d(7, 7);
    let pattern = CscPattern::new(49, &cp, &ri).unwrap();
    let (_perm, stats) = amd_order_with_stats(&pattern).unwrap();
    assert!(
        stats.n_mass_elim > 0,
        "grid_7x7 must trigger mass elimination (got {})",
        stats.n_mass_elim
    );
}

#[test]
fn supervar_merge_fires_on_grid_7x7() {
    let (cp, ri) = grid_2d(7, 7);
    let pattern = CscPattern::new(49, &cp, &ri).unwrap();
    let (_perm, stats) = amd_order_with_stats(&pattern).unwrap();
    assert!(
        stats.n_supervar_merge > 0,
        "grid_7x7 must trigger at least one supervariable merge (got {})",
        stats.n_supervar_merge
    );
}

#[test]
fn supervar_merge_fires_on_band_20_3() {
    let (cp, ri) = band(20, 3);
    let pattern = CscPattern::new(20, &cp, &ri).unwrap();
    let (_perm, stats) = amd_order_with_stats(&pattern).unwrap();
    assert!(
        stats.n_supervar_merge > 0,
        "band_20_3 must trigger at least one supervariable merge (got {})",
        stats.n_supervar_merge
    );
}
