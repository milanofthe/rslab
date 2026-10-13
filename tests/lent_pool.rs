//! Every path on a lent pool (`rslab::parallel::install`) computes what it
//! computes on rayon, bit for bit: on a pool running its pieces in order,
//! in reverse order (nothing waits for a piece that has not started), and on
//! threads of its own.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use num_complex::Complex;
use rslab::parallel::{install, Pool};
use rslab::{CscMatrix, GeneralCsc, KluSettings, KluSolver, LdltSolver, LuSolver, SolverSettings};

/// Pieces one after the other on the calling thread, forwards or backwards.
struct Serial {
    reverse: bool,
}

impl Pool for Serial {
    fn threads(&self) -> usize {
        4
    }
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.reverse {
            (0..n).rev().for_each(f)
        } else {
            (0..n).for_each(f)
        }
    }
}

/// Pieces on `threads` threads started per run, each taking the next one.
struct Threaded(usize);

impl Pool for Threaded {
    fn threads(&self) -> usize {
        self.0
    }
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..self.0.min(n) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    f(i);
                });
            }
        });
    }
}

fn pools() -> Vec<(&'static str, Arc<dyn Pool>)> {
    vec![
        ("serial", Arc::new(Serial { reverse: false })),
        ("reversed", Arc::new(Serial { reverse: true })),
        ("threaded", Arc::new(Threaded(4))),
    ]
}

/// 3D 7-point Laplacian (k^3 grid, SPD, lower triangle).
fn grid3d(k: usize) -> CscMatrix<f64> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                r.push(p);
                c.push(p);
                v.push(6.0);
                for (q, ok) in [
                    (idx((x + 1) % k, y, z), x + 1 < k),
                    (idx(x, (y + 1) % k, z), y + 1 < k),
                    (idx(x, y, (z + 1) % k), z + 1 < k),
                ] {
                    if ok {
                        r.push(p.max(q));
                        c.push(p.min(q));
                        v.push(-1.0);
                    }
                }
            }
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

/// 3D convection-diffusion grid (unsymmetric).
fn grid3d_conv(k: usize) -> GeneralCsc<f64> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                r.push(p);
                c.push(p);
                v.push(6.0);
                for (q, ok, down, up) in [
                    (idx((x + 1) % k, y, z), x + 1 < k, -1.3, -0.7),
                    (idx(x, (y + 1) % k, z), y + 1 < k, -1.2, -0.8),
                    (idx(x, y, (z + 1) % k), z + 1 < k, -1.1, -0.9),
                ] {
                    if ok {
                        r.extend([q, p]);
                        c.extend([p, q]);
                        v.extend([down, up]);
                    }
                }
            }
        }
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

fn bits<T: Copy>(x: &[T]) -> Vec<u8> {
    // SAFETY: plain scalars, read as their bytes.
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) }
        .to_vec()
}

/// `solve` on rayon at 8 threads, then on every lent pool, the same bits.
fn same_on_every_pool<T: Copy>(what: &str, solve: impl Fn() -> Vec<T>) {
    let reference = bits(&solve());
    for (name, pool) in pools() {
        let x = install(pool, &solve);
        assert!(bits(&x) == reference, "{what} differs on the {name} pool");
    }
}

const K: usize = if cfg!(debug_assertions) { 12 } else { 20 };

#[test]
fn ldlt_on_a_lent_pool() {
    let a = grid3d(K);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let s = SolverSettings::default().with_threads(8);
    same_on_every_pool("LDLT", || {
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
    let ac = CscMatrix::<Complex<f64>> {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a.values.iter().map(|&v| Complex::new(v, 0.1 * v)).collect(),
    };
    let bc: Vec<Complex<f64>> = b.iter().map(|&v| Complex::new(v, 1.0)).collect();
    same_on_every_pool("complex LDLT", || {
        LdltSolver::factor(&ac, &s).unwrap().solve(&bc).unwrap()
    });
}

#[test]
fn lu_on_a_lent_pool() {
    let a = grid3d_conv(K);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let s = SolverSettings::default().with_threads(8);
    same_on_every_pool("LU", || {
        LuSolver::<f64>::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
    let ac = GeneralCsc::<Complex<f64>> {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: (a.values.iter().enumerate())
            .map(|(k, &v)| Complex::new(v, 0.1 * v + 0.01 * (k % 7) as f64))
            .collect(),
    };
    let bc: Vec<Complex<f64>> = b.iter().map(|&v| Complex::new(v, 1.0)).collect();
    same_on_every_pool("complex LU", || {
        LuSolver::<Complex<f64>>::factor(&ac, &s)
            .unwrap()
            .solve(&bc)
            .unwrap()
    });
}

#[test]
fn klu_on_a_lent_pool() {
    let a = grid3d_conv(K);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let s = KluSettings::default();
    same_on_every_pool("KLU", || {
        let mut k = KluSolver::factor(&a, &s).unwrap();
        k.refactor(&a).unwrap();
        k.solve(&b).unwrap()
    });
}

/// A dense matrix in sparse form, diagonally dominant: one front, whose
/// products are large enough to split.
fn dense(n: usize, lower: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            if lower && i < j {
                continue;
            }
            r.push(i);
            c.push(j);
            v.push(if i == j {
                2.0 * n as f64
            } else {
                ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5
            });
        }
    }
    (r, c, v)
}

#[test]
fn dense_fronts_on_a_lent_pool() {
    let n = 600;
    let b: Vec<f64> = (0..n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let s = SolverSettings::default().with_threads(8);
    let (r, c, v) = dense(n, true);
    let a = CscMatrix::from_triplets(n, &r, &c, &v).unwrap();
    same_on_every_pool("dense LDLT", || {
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
    let (r, c, v) = dense(n, false);
    let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
    same_on_every_pool("dense LU", || {
        LuSolver::<f64>::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
    let ac = GeneralCsc::<Complex<f64>> {
        n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: a.values.iter().map(|&x| Complex::new(x, 0.3 * x)).collect(),
    };
    let bc: Vec<Complex<f64>> = b.iter().map(|&x| Complex::new(x, -1.0)).collect();
    same_on_every_pool("dense complex LU", || {
        LuSolver::<Complex<f64>>::factor(&ac, &s)
            .unwrap()
            .solve(&bc)
            .unwrap()
    });
}
