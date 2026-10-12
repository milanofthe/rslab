//! Factorizations from inside another rayon pool: a parallel sweep whose
//! every task factors on the supernodal paths must not nest the tasks into
//! each other while one waits for its factorization's pool.

use rayon::prelude::*;
use rslab::{AutoBackend, AutoSettings, AutoSymbolic, CscLayout, SolverSettings, Threads};

/// The `(rows, cols)` of an `m x m` grid Laplacian's pattern.
fn grid(m: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut rows, mut cols) = (Vec::new(), Vec::new());
    for i in 0..m * m {
        rows.push(i);
        cols.push(i);
        for j in [i + 1, i + m] {
            if j < m * m && (j != i + 1 || j % m != 0) {
                rows.extend([i, j]);
                cols.extend([j, i]);
            }
        }
    }
    (rows, cols)
}

#[test]
fn sweep_on_a_foreign_pool_does_not_nest() {
    let m = 24;
    let (rows, cols) = grid(m);
    let n = m * m;
    let layout = CscLayout::new(n, &rows, &cols).unwrap();
    let entries: Vec<f64> = (rows.iter().zip(&cols))
        .map(|(&r, &c)| if r == c { 4.0 } else { -1.0 })
        .collect();
    let mut vals = vec![0.0; layout.nnz()];
    layout.scatter(&entries, &mut vals);
    // Every pattern supernodal, one worker per factorization (a sweep's).
    let settings = AutoSettings::default()
        .with_supernodal_flops_per_entry(0.0)
        .with_supernodal(SolverSettings::default().with_threads(Threads::Fixed(1)));
    let sym = AutoSymbolic::analyze(layout, &vals, &settings).unwrap();
    // Small worker stacks: a task nested per waiting task overflows them.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .stack_size(256 * 1024)
        .build()
        .unwrap();
    let x0: Vec<f64> = pool.install(|| {
        (0..2000)
            .into_par_iter()
            .map_init(
                || sym.factorizer::<f64>(),
                |f, k| {
                    f.values_mut().copy_from_slice(&vals);
                    f.values_mut()[0] = 4.0 + k as f64 * 1e-6;
                    assert_eq!(f.factor(&settings).unwrap(), AutoBackend::Ldlt);
                    f.solve(&vec![1.0; n]).unwrap()[0]
                },
            )
            .collect()
    });
    assert!(x0.iter().all(|v| v.is_finite()));
}
