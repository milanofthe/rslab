//! Time one nested-dissection ordering on regular grids, per option variant.
use rslab_metis::{metis_order_full, MetisOptions};
use rslab_ordering_core::CscPattern;
use std::time::Instant;

fn grid2d(m: usize) -> (Vec<i32>, Vec<i32>) {
    let n = m * m;
    let (mut cp, mut ri) = (Vec::with_capacity(n + 1), Vec::with_capacity(5 * n));
    cp.push(0);
    for j in 0..n {
        let (x, y) = (j % m, j / m);
        let mut nb = Vec::new();
        if y > 0 {
            nb.push(j - m);
        }
        if x > 0 {
            nb.push(j - 1);
        }
        nb.push(j);
        if x + 1 < m {
            nb.push(j + 1);
        }
        if y + 1 < m {
            nb.push(j + m);
        }
        ri.extend(nb.iter().map(|&r| r as i32));
        cp.push(ri.len() as i32);
    }
    (cp, ri)
}

fn grid3d(m: usize) -> (Vec<i32>, Vec<i32>) {
    let n = m * m * m;
    let (mut cp, mut ri) = (Vec::with_capacity(n + 1), Vec::with_capacity(7 * n));
    cp.push(0);
    for j in 0..n {
        let (x, y, z) = (j % m, (j / m) % m, j / (m * m));
        let mut nb = Vec::new();
        if z > 0 {
            nb.push(j - m * m);
        }
        if y > 0 {
            nb.push(j - m);
        }
        if x > 0 {
            nb.push(j - 1);
        }
        nb.push(j);
        if x + 1 < m {
            nb.push(j + 1);
        }
        if y + 1 < m {
            nb.push(j + m);
        }
        if z + 1 < m {
            nb.push(j + m * m);
        }
        ri.extend(nb.iter().map(|&r| r as i32));
        cp.push(ri.len() as i32);
    }
    (cp, ri)
}

/// Exact scalar nnz(L) of the permuted full symmetric pattern (etree + GNP
/// column counts, as the ordering race scores its candidates).
fn exact_fill(cp: &[i32], ri: &[i32], perm: &[i32]) -> usize {
    let n = cp.len() - 1;
    let pattern = rslab::CscPattern {
        n,
        col_ptr: cp.iter().map(|&x| x as usize).collect(),
        row_idx: ri.iter().map(|&x| x as usize).collect(),
    };
    let perm: Vec<usize> = perm.iter().map(|&x| x as usize).collect();
    let permuted = rslab::ordering::amd::permute_pattern(&pattern, &perm);
    let etree = rslab::ordering::elimination_tree::EliminationTree::from_pattern(&permuted);
    rslab::symbolic::total_factor_nnz(&rslab::symbolic::column_counts_gnp(&permuted, &etree))
}

fn run(name: &str, cp: &[i32], ri: &[i32], variants: &[(&str, MetisOptions)]) {
    let pat = CscPattern {
        n: cp.len() - 1,
        col_ptr: cp,
        row_idx: ri,
    };
    for (label, opts) in variants {
        let t = Instant::now();
        let (perm, _, ms) = metis_order_full(&pat, opts).unwrap();
        let el = t.elapsed().as_secs_f64();
        let fill = exact_fill(cp, ri, &perm);
        println!(
            "{name:>12} {label:>22} {:>8.0} ms  {:>6.2} us/node  nnzL={fill:>10}  seps={} fm_passes={} amd_leafs={}",
            el * 1e3, el * 1e6 / pat.n as f64, ms.n_separator_vertices, ms.n_fm_passes, ms.n_amd_leaf_calls
        );
    }
}

fn main() {
    let d = MetisOptions::default();
    let variants = vec![
        ("seed=1", d.clone()),
        (
            "seed=2",
            MetisOptions {
                seed: 2,
                ..d.clone()
            },
        ),
        (
            "seed=3",
            MetisOptions {
                seed: 3,
                ..d.clone()
            },
        ),
        (
            "fm_passes=2",
            MetisOptions {
                fm_passes: 2,
                ..d.clone()
            },
        ),
        (
            "niparts=1",
            MetisOptions {
                niparts: 1,
                ..d.clone()
            },
        ),
    ];
    let which = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if which.ends_with(".mtx") {
        // A symmetric Matrix Market file: symmetrize the lower pattern.
        let rslab::MtxLoaded::Symmetric(a) =
            rslab::read_mtx_any(std::path::Path::new(&which)).unwrap()
        else {
            panic!("symmetric mtx expected");
        };
        let n = a.n;
        let mut adj: Vec<Vec<i32>> = vec![Vec::new(); n];
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                adj[j].push(i as i32);
                if i != j {
                    adj[i].push(j as i32);
                }
            }
        }
        let (mut cp, mut ri) = (vec![0i32], Vec::new());
        for l in &mut adj {
            l.sort_unstable();
            l.dedup();
            ri.extend_from_slice(l);
            cp.push(ri.len() as i32);
        }
        let name = std::path::Path::new(&which)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let sel: Vec<(&str, MetisOptions)> = if std::env::var("ALL_VARIANTS").is_ok() {
            variants.clone()
        } else {
            vec![variants[0].clone()]
        };
        run(&name, &cp, &ri, &sel);
        return;
    }
    if which == "all" || which == "2d" {
        let (cp, ri) = grid2d(500);
        run("grid2d 500", &cp, &ri, &variants);
    }
    if which == "all" || which == "3d" {
        let (cp, ri) = grid3d(40);
        run("grid3d 40", &cp, &ri, &variants);
    }
    if which == "scaling" {
        for m in [100usize, 200, 400, 800] {
            let (cp, ri) = grid2d(m);
            run(&format!("grid2d {m}"), &cp, &ri, &variants[..1]);
        }
    }
}
