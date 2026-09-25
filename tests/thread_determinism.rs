//! Bit-identity of the analysis, the factorizations and the solves across
//! thread counts and across repeated runs at the same thread count.
//!
//! Regression test for the racy tiled-cmod mode pick: the `chain_phase`
//! signal (nodes currently in flight, a timing artifact) must never select
//! between the tiled and the sequential cmod path, because the sequential
//! path routes sub-`scalar_gate` updates through the scalar kernel (plain
//! mul+add) while the tiled path runs everything through FMA GEMM
//! micro-kernels - different rounding, different bits. The 3D grid below
//! places its separator nodes squarely in the once-racy dispatch zone
//! (`cmod_flops` between `par_gemm` and the deterministic fork gate), where
//! the drift was measured at up to ~1800 last-ulp entries between 1 and 8
//! threads and a few hundred entries run-to-run at 8 threads.

use num_complex::Complex;
use rslab::{
    CscMatrix, GeneralCsc, KluParallel, KluSettings, KluSolver, LdltSolver, LdltSymbolic, LuSolver,
    OrderingMethod, SolverSettings,
};

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

fn bits_f64(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn bits_c64(x: &[Complex<f64>]) -> Vec<(u64, u64)> {
    x.iter().map(|v| (v.re.to_bits(), v.im.to_bits())).collect()
}

/// Grid edge sizes: release runs the sizes the original drift was measured
/// at; debug (the CI profile) shrinks the f64 grids so the numeric-heavy
/// factors fit the runner budget. Every size below was verified to
/// reproduce the original racy-dispatch drift in its profile (the shrunk
/// debug sizes included), so the test discriminates on both.
const K_LDLT: usize = if cfg!(debug_assertions) { 16 } else { 24 };
const K_LDLT_C: usize = 16;
const K_LU: usize = if cfg!(debug_assertions) { 14 } else { 22 };

#[test]
fn ll_ldlt_bit_identical_across_threads_and_runs() {
    let a = grid3d(K_LDLT);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let solve = |t: usize| -> Vec<f64> {
        let s = SolverSettings::default().with_threads(t);
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    };
    let x1 = solve(1);
    let x8 = solve(8);
    assert_eq!(
        bits_f64(&x1),
        bits_f64(&x8),
        "LL LDLT solution differs between 1 and 8 threads"
    );
    for _ in 0..3 {
        assert_eq!(
            bits_f64(&x8),
            bits_f64(&solve(8)),
            "LL LDLT solution differs run-to-run at 8 threads"
        );
    }
}

#[test]
fn ll_ldlt_complex_bit_identical_across_threads() {
    // Complex-typed variant of the same grid: the complex kernels dispatch
    // through the same racy-prone gates.
    let ar = grid3d(K_LDLT_C);
    let a = CscMatrix::<Complex<f64>> {
        n: ar.n,
        col_ptr: ar.col_ptr.clone(),
        row_idx: ar.row_idx.clone(),
        values: ar
            .values
            .iter()
            .map(|&v| Complex::new(v, 0.1 * v))
            .collect(),
    };
    let b: Vec<Complex<f64>> = (0..a.n)
        .map(|i| Complex::new(((i % 11) as f64) - 5.0, 1.0))
        .collect();
    let solve = |t: usize| -> Vec<Complex<f64>> {
        let s = SolverSettings::default().with_threads(t);
        LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap()
    };
    assert_eq!(
        bits_c64(&solve(1)),
        bits_c64(&solve(8)),
        "LL complex LDLT solution differs between 1 and 8 threads"
    );
}

#[test]
fn ll_lu_bit_identical_across_threads_and_runs() {
    let a = grid3d_conv(K_LU);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let solve = |t: usize| -> Vec<f64> {
        let s = SolverSettings::default().with_threads(t);
        LuSolver::<f64>::factor(&a, &s).unwrap().solve(&b).unwrap()
    };
    let x1 = solve(1);
    let x8 = solve(8);
    assert_eq!(
        bits_f64(&x1),
        bits_f64(&x8),
        "LL LU solution differs between 1 and 8 threads"
    );
    for _ in 0..3 {
        let xr = solve(8);
        assert_eq!(
            bits_f64(&x8),
            bits_f64(&xr),
            "LL LU solution differs run-to-run at 8 threads"
        );
    }
}

#[test]
fn ll_lu_complex_bit_identical_across_threads_and_runs() {
    // The complex twin: its GEMMs are not bit-identical between the serial and the
    // parallel mode, so a timing-dependent choice between them (the former chain-phase
    // gates) made the factor differ run to run.
    let ar = grid3d_conv(K_LU);
    let a = GeneralCsc::<Complex<f64>> {
        n: ar.n,
        col_ptr: ar.col_ptr.clone(),
        row_idx: ar.row_idx.clone(),
        values: ar
            .values
            .iter()
            .enumerate()
            .map(|(k, &v)| Complex::new(v, 0.1 * v + 0.01 * (k % 7) as f64))
            .collect(),
    };
    let b: Vec<Complex<f64>> = (0..a.n)
        .map(|i| Complex::new(((i % 11) as f64) - 5.0, (i % 3) as f64))
        .collect();
    let solve = |t: usize| -> Vec<Complex<f64>> {
        let s = SolverSettings::default().with_threads(t);
        LuSolver::<Complex<f64>>::factor(&a, &s)
            .unwrap()
            .solve(&b)
            .unwrap()
    };
    let x1 = solve(1);
    for _ in 0..4 {
        assert_eq!(
            bits_c64(&x1),
            bits_c64(&solve(8)),
            "LL complex LU solution differs between 1 and 8 threads or run to run"
        );
    }
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
    race.eager_nd_min_nnz = 0;
    let solve = |t: usize| -> Vec<f64> {
        let sym = LdltSymbolic::analyze(&a, &open.clone().with_threads(t)).unwrap();
        sym.factor(&a, &SolverSettings::default().with_threads(4))
            .unwrap()
            .solve(&b)
            .unwrap()
    };
    assert_eq!(bits_f64(&solve(1)), bits_f64(&solve(8)));
}

/// The LU path with the row matching and the KLU path with parallel blocks
/// are bit-identical at one and eight workers.
#[test]
fn lu_with_matching_and_klu_are_bit_identical_across_threads() {
    let a = unsymmetric(if cfg!(debug_assertions) { 2000 } else { 20_000 }, true);
    let b: Vec<f64> = (0..a.n).map(|i| ((i % 11) as f64) - 5.0).collect();
    let lu = |t: usize| {
        let s = SolverSettings::default()
            .with_threads(t)
            .with_ordering(OrderingMethod::Amd);
        let f = LuSolver::factor(&a, &s).unwrap();
        assert!(f.diagnostics().decisions.scaling.contains("Mc64"));
        f.solve(&b).unwrap()
    };
    assert_eq!(bits_f64(&lu(1)), bits_f64(&lu(8)));
    let klu = |t: usize| {
        let s = KluSettings::default().with_parallel(KluParallel::On);
        in_pool(t, || KluSolver::factor(&a, &s).unwrap().solve(&b).unwrap())
    };
    assert_eq!(bits_f64(&klu(1)), bits_f64(&klu(8)));
}
