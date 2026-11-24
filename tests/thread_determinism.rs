//! What every thread count gives: a factorization whose solves meet the
//! error bound of the problem, and at a fixed thread count the same result
//! from run to run. The tiles of a node's updates follow the workers
//! (`KernelSettings::update_tile`), so different thread counts round
//! differently; each is held to the normwise backward error of a stable
//! solve, `|A x - b| / (|A| |x| + |b|) <= GROWTH * gamma_m` with
//! `gamma_m = m u / (1 - m u)`, `u` the unit roundoff and `m` the dimension
//! (twice it for complex arithmetic). The test matrices are diagonally
//! dominant, where elimination grows the entries at most by `GROWTH`.

use num_complex::Complex;
use rslab::{
    CscMatrix, GeneralCsc, KluParallel, KluSettings, KluSolver, LdltSolver, LdltSymbolic, LuSolver,
    OrderingMethod, Scalar, SolverSettings, Threads,
};

/// The most a diagonally dominant matrix's entries grow under elimination.
const GROWTH: f64 = 2.0;

/// The thread counts every solve is checked at.
const THREADS: [usize; 3] = [1, 2, 8];

/// The backward error bound of a solve of dimension `n` in `T`.
fn bound<T: Scalar>(n: usize) -> f64 {
    let u = f64::EPSILON / 2.0;
    let m = (if T::COMPLEX { 2 * n } else { n }) as f64;
    GROWTH * m * u / (1.0 - m * u)
}

/// `|A x - b|_inf / (|A|_inf |x|_inf + |b|_inf)` for `A` given by its
/// entries `(row, column, value)`.
fn backward_error<T: Scalar>(n: usize, entries: &[(usize, usize, T)], x: &[T], b: &[T]) -> f64 {
    let (mut r, mut rows) = (
        b.iter().map(|&v| T::zero() - v).collect::<Vec<T>>(),
        vec![0.0; n],
    );
    for &(i, j, v) in entries {
        r[i] = r[i] + v * x[j];
        rows[i] += v.magnitude();
    }
    let inf = |v: &[T]| v.iter().map(|z| z.magnitude()).fold(0.0, f64::max);
    let a = rows.iter().copied().fold(0.0, f64::max);
    inf(&r) / (a * inf(x) + inf(b))
}

/// The entries of the symmetric matrix whose lower triangle is `a`.
fn symmetric<T: Scalar>(a: &CscMatrix<T>) -> Vec<(usize, usize, T)> {
    let mut e = Vec::new();
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            e.push((i, j, a.values[k]));
            if i != j {
                e.push((j, i, a.values[k]));
            }
        }
    }
    e
}

fn general<T: Scalar>(a: &GeneralCsc<T>) -> Vec<(usize, usize, T)> {
    (0..a.n)
        .flat_map(|j| (a.col_ptr[j]..a.col_ptr[j + 1]).map(move |k| (a.row_idx[k], j, a.values[k])))
        .collect()
}

/// Every thread count of [`THREADS`] solves within the bound, and the last
/// one gives the same bits run after run.
fn check<T: Scalar>(
    what: &str,
    entries: &[(usize, usize, T)],
    b: &[T],
    solve: impl Fn(usize) -> Vec<T>,
) {
    let n = b.len();
    for t in THREADS {
        let x = solve(t);
        let eta = backward_error(n, entries, &x, b);
        assert!(
            eta <= bound::<T>(n),
            "{what}: backward error {eta:e} at {t} threads over {:e}",
            bound::<T>(n)
        );
    }
    let last = THREADS[THREADS.len() - 1];
    let x = solve(last);
    for _ in 0..2 {
        assert!(
            solve(last) == x,
            "{what}: other bits run to run at {last} threads"
        );
    }
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
                let mut nb = |q: usize| {
                    let (hi, lo) = if p >= q { (p, q) } else { (q, p) };
                    r.push(hi);
                    c.push(lo);
                    v.push(-1.0);
                };
                if x + 1 < k {
                    nb(idx(x + 1, y, z));
                }
                if y + 1 < k {
                    nb(idx(x, y + 1, z));
                }
                if z + 1 < k {
                    nb(idx(x, y, z + 1));
                }
            }
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

/// 3D convection-diffusion grid (unsymmetric, general CSC).
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
                let mut nb = |q: usize, down: f64, up: f64| {
                    r.push(q);
                    c.push(p);
                    v.push(down);
                    r.push(p);
                    c.push(q);
                    v.push(up);
                };
                if x + 1 < k {
                    nb(idx(x + 1, y, z), -1.3, -0.7);
                }
                if y + 1 < k {
                    nb(idx(x, y + 1, z), -1.2, -0.8);
                }
                if z + 1 < k {
                    nb(idx(x, y, z + 1), -1.1, -0.9);
                }
            }
        }
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

/// Grid edge sizes: larger in release; debug (the CI profile) shrinks them
/// so the factors fit the runner budget. Every size has nodes whose updates
/// run in tiles at eight threads.
const K_LDLT: usize = if cfg!(debug_assertions) { 16 } else { 24 };
const K_LDLT_C: usize = 16;
const K_LU: usize = if cfg!(debug_assertions) { 14 } else { 22 };

#[test]
fn ldlt_meets_the_bound_at_every_thread_count() {
    let a = grid3d(K_LDLT);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    check("LDL^T", &symmetric(&a), &b, |t| {
        let s = SolverSettings::default().with_threads(t);
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
}

#[test]
fn complex_ldlt_meets_the_bound_at_every_thread_count() {
    let ar = grid3d(K_LDLT_C);
    let a = CscMatrix::<Complex<f64>> {
        n: ar.n,
        col_ptr: ar.col_ptr.clone(),
        row_idx: ar.row_idx.clone(),
        values: (ar.values.iter())
            .map(|&v| Complex::new(v, 0.1 * v))
            .collect(),
    };
    let b: Vec<Complex<f64>> = (0..a.n)
        .map(|i| Complex::new(((i % 11) as f64) - 5.0, 1.0))
        .collect();
    check("complex LDL^T", &symmetric(&a), &b, |t| {
        let s = SolverSettings::default().with_threads(t);
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
}

/// A complex factorization under a one-thread budget gives the bits it
/// gives on a rayon pool of one (`Threads::Ambient` inside it, the kernels
/// on that pool's worker): the budget runs the kernels on the calling
/// thread under a lent pool of one, where the parallel products once ran
/// in column blocks and rounded otherwise. A 12^3 curl-curl grid with the
/// parallel gates lowered to 10^6, so that its fronts take those products.
#[cfg(feature = "matgen")]
#[test]
fn complex_factor_on_a_one_thread_budget_matches_a_pool_of_one() {
    let a = rslab::matgen::fem::curl_curl(&[12, 12, 12], 1.0, 0.1);
    let b: Vec<Complex<f64>> = (0..a.n)
        .map(|i| Complex::new(((i % 11) as f64) - 5.0, 1.0))
        .collect();
    let solve = |threads: Threads| -> Vec<Complex<f64>> {
        let mut s = SolverSettings::default().with_threads(threads);
        let gate = 1_000_000;
        (
            s.kernels.par_gemm,
            s.kernels.par_cdiv,
            s.kernels.fork_min_flops,
        ) = (gate, gate, gate);
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    };
    assert!(
        solve(Threads::Fixed(1)) == in_pool(1, || solve(Threads::Ambient)),
        "complex factor differs between a one-thread budget and a pool of one"
    );
}

#[test]
fn lu_meets_the_bound_at_every_thread_count() {
    let a = grid3d_conv(K_LU);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    check("LU", &general(&a), &b, |t| {
        let s = SolverSettings::default().with_threads(t);
        LuSolver::<f64>::factor(&a, &s).unwrap().solve(&b).unwrap()
    });
}

#[test]
fn complex_lu_meets_the_bound_at_every_thread_count() {
    let ar = grid3d_conv(K_LU);
    let a = GeneralCsc::<Complex<f64>> {
        n: ar.n,
        col_ptr: ar.col_ptr.clone(),
        row_idx: ar.row_idx.clone(),
        values: (ar.values.iter().enumerate())
            .map(|(k, &v)| Complex::new(v, 0.1 * v + 0.01 * (k % 7) as f64))
            .collect(),
    };
    let b: Vec<Complex<f64>> = (0..a.n)
        .map(|i| Complex::new(((i % 11) as f64) - 5.0, (i % 3) as f64))
        .collect();
    check("complex LU", &general(&a), &b, |t| {
        let s = SolverSettings::default().with_threads(t);
        LuSolver::<Complex<f64>>::factor(&a, &s)
            .unwrap()
            .solve(&b)
            .unwrap()
    });
}

/// A random unsymmetric matrix with a dominant diagonal; with `holes` every
/// seventh diagonal entry is left out, its column carrying the pivot of the
/// next row instead (a 2x2 swap the row matching has to find).
fn unsymmetric(n: usize, holes: bool) -> GeneralCsc<f64> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        if holes && j % 7 == 0 && j + 1 < n {
            r.extend([j + 1, j]);
            c.extend([j, j + 1]);
            v.extend([4.0, 4.0]);
        } else {
            r.push(j);
            c.push(j);
            v.push(4.0 + (j % 3) as f64);
        }
        for _ in 0..3 {
            let i = (next() % n as u64) as usize;
            if i != j {
                r.push(i);
                c.push(j);
                v.push(((next() % 200) as f64) / 100.0 - 1.0);
            }
        }
    }
    GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
}

fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

/// The ordering race, with nested dissection and its seed ensemble forced in
/// and the dissection started speculatively, picks the same ordering whatever
/// the analysis pool: the factors under one and eight analysis workers solve
/// bit-identically.
#[test]
fn the_race_is_thread_count_invariant() {
    let a = grid3d(16);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 13) as f64) - 6.0).collect();
    let mut open = SolverSettings::default().with_nd_ensemble(true);
    let race = &mut open.ordering.race;
    (race.nd_min_n, race.nd_min_work, race.ensemble_min_flops) = (0, 0, 0);
    let solve = |t: usize| -> Vec<f64> {
        let sym = LdltSymbolic::analyze(&a, &open.clone().with_threads(t)).unwrap();
        sym.factor(&a, &SolverSettings::default().with_threads(4))
            .unwrap()
            .solve(&b)
            .unwrap()
    };
    assert!(solve(1) == solve(8));
}

/// The LU path with the row matching meets the bound at every thread count;
/// the KLU path with parallel blocks factors every block on one thread, so
/// its bits are the same at one and eight workers.
#[test]
fn lu_with_matching_and_klu_across_threads() {
    let a = unsymmetric(if cfg!(debug_assertions) { 2000 } else { 20_000 }, true);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    check("LU with matching", &general(&a), &b, |t| {
        let s = SolverSettings::default()
            .with_threads(t)
            .with_ordering(OrderingMethod::Amd);
        let f = LuSolver::factor(&a, &s).unwrap();
        assert!(f.diagnostics().decisions.scaling.contains("Mc64"));
        f.solve(&b).unwrap()
    });
    let klu = |t: usize| {
        let s = (KluSettings::default())
            .with_parallel(KluParallel::On)
            .with_threads(Threads::Ambient);
        in_pool(t, || KluSolver::factor(&a, &s).unwrap().solve(&b).unwrap())
    };
    assert!(klu(1) == klu(8));
}
