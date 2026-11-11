//! Sparse right-hand sides on the LDL^T factor: `port_matrix` and
//! `solve_sparse` against `B^T X` and `X` from the dense block solve
//! (`solve_many`), on real symmetric and complex symmetric matrices,
//! matrices that take 2x2 pivots, a near-singular one, empty and duplicate
//! columns; and their bits across thread counts and port blocks.

use num_complex::Complex;
use rslab::{CscMatrix, LdltSolver, Scalar, SolveSettings, SolverSettings, SparseRhs, Threads};

type C = Complex<f64>;

/// SplitMix64 in `[-1, 1)`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 52) as f64 * 2.0 - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        (((self.next() + 1.0) / 2.0) * n as f64) as usize % n
    }
}

trait Draw: Scalar {
    fn draw(rng: &mut Rng) -> Self;
}

impl Draw for f64 {
    fn draw(rng: &mut Rng) -> Self {
        rng.next()
    }
}

impl Draw for C {
    fn draw(rng: &mut Rng) -> Self {
        C::new(rng.next(), rng.next())
    }
}

impl Draw for f32 {
    fn draw(rng: &mut Rng) -> Self {
        rng.next() as f32
    }
}

/// A random sparse symmetric matrix (lower triangle): `deg` random
/// off-diagonals per column and the diagonal `diag + noise`; an indefinite
/// one for a small `diag`.
fn random_sym<T: Draw>(n: usize, deg: usize, diag: f64, seed: u64) -> CscMatrix<T> {
    let mut rng = Rng(seed);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(T::from_real(diag) + T::draw(&mut rng));
        for _ in 0..deg {
            let i = rng.below(n);
            if i > j {
                r.push(i);
                c.push(j);
                v.push(T::draw(&mut rng));
            }
        }
    }
    // a chain so the matrix is irreducible
    for j in 0..n - 1 {
        r.push(j + 1);
        c.push(j);
        v.push(T::draw(&mut rng));
    }
    sum_duplicates(n, &r, &c, &v)
}

fn sum_duplicates<T: Scalar>(n: usize, r: &[usize], c: &[usize], v: &[T]) -> CscMatrix<T> {
    let mut e: Vec<(usize, usize, T)> = (0..r.len()).map(|k| (c[k], r[k], v[k])).collect();
    e.sort_by_key(|x| (x.0, x.1));
    let mut out: Vec<(usize, usize, T)> = Vec::new();
    for (cc, rr, vv) in e {
        match out.last_mut() {
            Some(l) if l.0 == cc && l.1 == rr => l.2 = l.2 + vv,
            _ => out.push((cc, rr, vv)),
        }
    }
    let rows: Vec<usize> = out.iter().map(|x| x.1).collect();
    let cols: Vec<usize> = out.iter().map(|x| x.0).collect();
    let vals: Vec<T> = out.iter().map(|x| x.2).collect();
    CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
}

/// 2D 5-point `Lap - k2 I` on an `m x m` grid, Dirichlet (lower triangle).
fn helmholtz(m: usize, k2: C) -> CscMatrix<C> {
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        r.push(j);
        c.push(j);
        v.push(C::new(4.0, 0.0) - k2);
        if j % m + 1 < m {
            r.push(j + 1);
            c.push(j);
            v.push(C::new(-1.0, 0.0));
        }
        if j + m < n {
            r.push(j + m);
            c.push(j);
            v.push(C::new(-1.0, 0.0));
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

/// Pairs `(2i, 2i + 1)` with a zero diagonal coupled by `1`, weakly coupled
/// to the next pair: every 1x1 pivot fails the Bunch-Kaufman test, so the
/// factor takes 2x2 pivots.
fn zero_diagonal_pairs<T: Draw>(pairs: usize, seed: u64) -> CscMatrix<T> {
    let n = 2 * pairs;
    let mut rng = Rng(seed);
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for p in 0..pairs {
        let (a, b) = (2 * p, 2 * p + 1);
        r.extend([a, b, b]);
        c.extend([a, a, b]);
        v.extend([
            T::zero(),
            T::from_real(1.0) + T::draw(&mut rng) * T::from_real(0.1),
            T::zero(),
        ]);
        if p + 1 < pairs {
            // both rows of a pair see the same neighbours, so a pair is
            // one supervariable and its two columns share a supernode
            for (i, j) in [(a + 2, a), (b + 2, a), (a + 2, b), (b + 2, b)] {
                r.push(i);
                c.push(j);
                v.push(T::draw(&mut rng) * T::from_real(0.05));
            }
        }
    }
    CscMatrix::from_triplets(n, &r, &c, &v).unwrap()
}

/// `p` ports of `k` random rows each with random weights.
fn ports<T: Draw>(n: usize, p: usize, k: usize, seed: u64) -> SparseRhs<T> {
    let mut rng = Rng(seed);
    let cols: Vec<Vec<(usize, T)>> = (0..p)
        .map(|_| (0..k).map(|_| (rng.below(n), T::draw(&mut rng))).collect())
        .collect();
    SparseRhs::from_columns(n, &cols).unwrap()
}

fn settings(threads: usize) -> SolverSettings {
    SolverSettings::default().with_threads(Threads::Fixed(threads))
}

/// `B^T X` from the dense block solve.
fn reference<T: Scalar>(f: &LdltSolver<T>, b: &SparseRhs<T>) -> Vec<T> {
    let x = f.solve_many(&b.to_dense(), b.ncols()).unwrap();
    b.transpose_times(&x, b.ncols()).unwrap()
}

fn max_rel_diff<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    let scale = b.iter().map(|x| x.magnitude()).fold(0.0, f64::max);
    let diff = (a.iter().zip(b))
        .map(|(&x, &y)| (x - y).magnitude())
        .fold(0.0, f64::max);
    diff / scale.max(f64::MIN_POSITIVE)
}

fn assert_symmetric<T: Scalar>(z: &[T], p: usize) {
    for i in 0..p {
        for j in 0..p {
            assert!(
                z[i * p + j] == z[j * p + i],
                "Z not exactly symmetric at ({i}, {j})"
            );
        }
    }
}

fn check<T: Scalar>(a: &CscMatrix<T>, b: &SparseRhs<T>, tol: f64) -> LdltSolver<T> {
    let f = LdltSolver::factor(a, &settings(1)).unwrap();
    let z = f.port_matrix(b).unwrap();
    let p = b.ncols();
    assert_eq!(z.len(), p * p);
    assert_symmetric(&z, p);
    let r = reference(&f, b);
    let d = max_rel_diff(&z, &r);
    assert!(d <= tol, "port matrix differs from B^T X by {d:e}");
    let x = f.solve_sparse(b).unwrap();
    let xr = f.solve_many(&b.to_dense(), p).unwrap();
    let d = max_rel_diff(&x, &xr);
    assert!(d <= tol, "sparse solve differs from solve_many by {d:e}");
    f
}

#[test]
fn real_symmetric_random() {
    for seed in 1..4 {
        let a = random_sym::<f64>(600, 3, 6.0, seed);
        check(&a, &ports(600, 23, 4, seed + 10), 1e-11);
    }
}

#[test]
fn real_symmetric_indefinite_random() {
    let a = random_sym::<f64>(500, 3, 0.3, 7);
    check(&a, &ports(500, 17, 3, 8), 1e-8);
}

#[test]
fn complex_symmetric_random() {
    for seed in 1..4 {
        let a = random_sym::<C>(700, 4, 0.5, seed);
        check(&a, &ports(700, 30, 6, seed + 20), 1e-9);
    }
}

#[test]
fn complex_symmetric_helmholtz_grid() {
    let a = helmholtz(40, C::new(0.7, 0.02));
    check(&a, &ports(1600, 40, 6, 3), 1e-11);
}

#[test]
fn two_by_two_pivots_real_and_complex() {
    let a = zero_diagonal_pairs::<f64>(300, 5);
    let f = check(&a, &ports(600, 25, 2, 6), 1e-11);
    assert!(
        f.diagnostics().numeric.two_by_two.unwrap_or(0) > 0,
        "2x2 pivots taken"
    );
    let a = zero_diagonal_pairs::<C>(300, 9);
    let f = check(&a, &ports(600, 25, 2, 10), 1e-11);
    assert!(
        f.diagnostics().numeric.two_by_two.unwrap_or(0) > 0,
        "2x2 pivots taken"
    );
    // a port on exactly one row of a pair and one on both
    let b = SparseRhs::from_columns(
        600,
        &[
            vec![(10, C::new(1.0, 0.0))],
            vec![(10, C::new(1.0, 0.0)), (11, C::new(0.0, 1.0))],
        ],
    )
    .unwrap();
    check(&a, &b, 1e-11);
}

#[test]
fn near_singular_helmholtz() {
    // the first Dirichlet eigenvalue of the 5-point Laplacian on 30 x 30
    let m = 30;
    let s = (std::f64::consts::PI / (2.0 * (m as f64 + 1.0))).sin();
    let lambda = 8.0 * s * s;
    let a = helmholtz(m, C::new(lambda * (1.0 + 1e-9), 0.0));
    let f = LdltSolver::factor(&a, &settings(1)).unwrap();
    let b = ports::<C>(m * m, 20, 4, 4);
    let z = f.port_matrix(&b).unwrap();
    assert_symmetric(&z, 20);
    let d = max_rel_diff(&z, &reference(&f, &b));
    // conditioning ~1e10: both are accurate to about kappa * u
    assert!(d <= 1e-4, "near-singular port matrix differs by {d:e}");
    let norm = z.iter().map(|x| x.norm()).fold(0.0, f64::max);
    assert!(norm > 1e6, "the near-singular mode dominates Z ({norm:e})");
}

#[test]
fn empty_and_duplicate_columns() {
    let a = random_sym::<C>(400, 3, 0.7, 2);
    let one = C::new(1.0, 0.0);
    let b = SparseRhs::from_columns(
        400,
        &[
            vec![(5, one), (77, C::new(0.5, -0.5))],
            vec![],
            // the same row twice: counts with the sum of its values
            vec![
                (5, C::new(0.25, 0.0)),
                (5, C::new(0.75, 0.0)),
                (77, C::new(0.5, -0.5)),
            ],
            vec![(399, one)],
            vec![],
        ],
    )
    .unwrap();
    let f = check(&a, &b, 1e-11);
    let z = f.port_matrix(&b).unwrap();
    let p = 5;
    for j in 0..p {
        assert_eq!(z[p + j], C::new(0.0, 0.0), "empty column");
        assert_eq!(z[4 * p + j], C::new(0.0, 0.0), "empty column");
    }
    // columns 0 and 2 are the same right-hand side
    for j in 0..p {
        assert!((z[j] - z[2 * p + j]).norm() <= 1e-12 * z[j].norm().max(1.0));
    }
    // no columns at all
    let none = SparseRhs::<C>::from_columns(400, &[]).unwrap();
    assert!(f.port_matrix(&none).unwrap().is_empty());
    assert!(f.solve_sparse(&none).unwrap().is_empty());
}

#[test]
fn same_bits_for_every_thread_count_and_block() {
    let a = helmholtz(60, C::new(1.3, 0.05));
    let b = ports::<C>(3600, 70, 6, 12);
    let sym = rslab::LdltSymbolic::analyze(&a, &settings(1)).unwrap();
    let base = sym.factor(&a, &settings(1)).unwrap();
    let z1 = base.port_matrix(&b).unwrap();
    let x1 = base.solve_sparse(&b).unwrap();
    for threads in [2, 4, 8] {
        // `par_min_work: Some(0)` puts every section on the pool
        for par in [None, Some(0)] {
            let mut s = settings(threads);
            s.solve.par_min_work = par;
            let f = sym.factor(&a, &s).unwrap();
            assert!(f.port_matrix(&b).unwrap() == z1, "Z at {threads} threads");
            assert!(f.solve_sparse(&b).unwrap() == x1, "X at {threads} threads");
        }
    }
    for block in [1, 3, 8, 64] {
        let mut s = settings(4);
        s.solve = SolveSettings {
            port_block: block,
            ..SolveSettings::default()
        };
        let f = sym.factor(&a, &s).unwrap();
        // values equal (a signed zero aside) for any block width
        assert!(f.port_matrix(&b).unwrap() == z1, "Z with block {block}");
    }
}

#[test]
fn single_precision_fields() {
    let a = random_sym::<f32>(300, 3, 6.0, 4);
    check(&a, &ports(300, 12, 3, 5), 1e-4);
}

#[test]
fn reach_is_bounded_by_the_factor() {
    let a = helmholtz(50, C::new(0.4, 0.01));
    let f = LdltSolver::factor(&a, &settings(1)).unwrap();
    let r = f.sparse_reach(&[0]).unwrap();
    assert!(r.supernodes >= 1 && r.supernodes <= r.total_supernodes);
    assert!(r.panel_entries <= r.total_panel_entries);
    let all: Vec<usize> = (0..2500).collect();
    let full = f.sparse_reach(&all).unwrap();
    assert_eq!(full.supernodes, full.total_supernodes);
    assert_eq!(full.columns, 2500);
    assert_eq!(full.panel_entries, full.total_panel_entries);
    // a corner node reaches a small part of a nested-dissection-like factor
    assert!(r.panel_entries < full.total_panel_entries);
    assert!(f.sparse_reach(&[2500]).is_err());
}

#[test]
fn inputs_are_checked() {
    assert!(SparseRhs::<f64>::new(3, vec![0, 2], vec![0, 3], vec![1.0, 1.0]).is_err());
    assert!(SparseRhs::<f64>::new(3, vec![1, 2], vec![0, 1], vec![1.0, 1.0]).is_err());
    assert!(SparseRhs::<f64>::new(3, vec![0, 2, 1], vec![0, 1], vec![1.0, 1.0]).is_err());
    assert!(SparseRhs::<f64>::new(3, vec![0, 2], vec![0, 1], vec![1.0]).is_err());
    let a = random_sym::<f64>(50, 2, 6.0, 1);
    let f = LdltSolver::factor(&a, &settings(1)).unwrap();
    let wrong = SparseRhs::<f64>::from_columns(49, &[vec![(0, 1.0)]]).unwrap();
    assert!(f.port_matrix(&wrong).is_err());
    assert!(f.solve_sparse(&wrong).is_err());
}
