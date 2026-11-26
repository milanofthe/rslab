//! Parametrized test-matrix generators for benchmarking and testing the solver
//! across the axes that matter for a sparse direct solver: **size**, **sparsity
//! structure**, **symmetry**, **conditioning**, and **density**.
//!
//! Design follows the established packages (MatrixDepot.jl, MATLAB `gallery`,
//! LAPACK `xLATMS`): parametrized *generators* for structured coverage plus an
//! optional *downloader* (module `download`, feature `matgen-download`) for
//! real SuiteSparse / Matrix Market matrices. Conditioning is steered
//! **structurally**: diagonal dominance, PDE refinement, near-resonance shift,
//! coefficient jumps.
//!
//! The real-valued structural families are generic over [`Scalar`]; the inherently
//! complex families (Helmholtz, BEM/MoM kernel) produce `Complex<f64>`.

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

pub(crate) fn build_sym<T: Scalar>(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> CscMatrix<T> {
    built(CscMatrix::from_triplets(n, rows, cols, vals))
}

pub(crate) fn build_gen<T: Scalar>(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> GeneralCsc<T> {
    built(GeneralCsc::from_triplets(n, rows, cols, vals))
}

/// The matrix a generator assembled from its triplets; an error is a bug of
/// the generator.
fn built<M>(m: Result<M, crate::error::RslabError>) -> M {
    match m {
        Ok(m) => m,
        Err(e) => panic!("matgen: internal invariant - invalid triplets: {e}"),
    }
}

pub mod bem;
#[cfg(feature = "matgen-download")]
pub mod download;
pub mod fem;
pub mod random;
pub mod stencil;
pub mod structured;

/// Small, fast, deterministic PRNG (xorshift64*). Pure Rust, no `rand` dep - so
/// generated matrices are exactly reproducible from a seed across runs/platforms.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    /// Standard normal via Box-Muller (one of the pair).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = (self.unit()).max(crate::constants::TINY);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod integration {
    //! Every family must produce matrices the solver actually factors and solves -
    //! the whole point. Small instances, exact factorization, true residual.
    use super::*;
    use crate::test_constants::{solve_backward_error, U32, U64, UC32, UC64};
    use crate::refine::RefineOperator;
    use crate::{Factorization, LdltSymbolic, LuSymbolic, SolverSettings};
    use num_complex::Complex;

    type C = Complex<f64>;

    fn rhs(n: usize) -> Vec<C> {
        (0..n)
            .map(|i| Complex::new((i % 5) as f64 - 2.0, (i % 3) as f64 - 1.0))
            .collect()
    }

    /// The normwise backward error `max|A x - b| / (||A||_inf max|x| + max|b|)`.
    pub(super) fn backward_error<T: crate::Scalar>(
        a: &impl RefineOperator<T>,
        x: &[T],
        b: &[T],
    ) -> f64 {
        let n = b.len();
        let mut ax = vec![T::zero(); n];
        a.apply(x, &mut ax);
        let mut sums = vec![0.0; n];
        a.apply_abs(&vec![T::one(); n], &mut sums);
        let max = |v: &[T]| v.iter().map(|z| z.magnitude()).fold(0.0, f64::max);
        let r: Vec<T> = ax.iter().zip(b).map(|(&p, &q)| p - q).collect();
        max(&r) / (sums.iter().copied().fold(0.0, f64::max) * max(x) + max(b))
    }

    /// The backward error of the solve of `rhs` with the factor `f` of `a`.
    fn resid(a: &impl RefineOperator<C>, f: impl Factorization<C>) -> f64 {
        let b = rhs(a.n());
        let x = f.solve(&b).unwrap();
        backward_error(a, &x, &b)
    }

    fn ldlt_resid(a: &CscMatrix<C>) -> f64 {
        let s = SolverSettings::default();
        resid(a, LdltSymbolic::analyze(a, &s).unwrap().factor(a, &s).unwrap())
    }

    fn lu_resid(a: &GeneralCsc<C>) -> f64 {
        let s = SolverSettings::default();
        resid(a, LuSymbolic::analyze(a, &s).unwrap().factor(a, &s).unwrap())
    }

    #[test]
    fn symmetric_families_factor_and_solve() {
        let cases: Vec<(&str, CscMatrix<C>)> = vec![
            (
                "poisson3d",
                stencil::laplacian(&[8, 8, 8], &stencil::StencilOpts::default()),
            ),
            (
                "helmholtz",
                stencil::helmholtz(
                    &[8, 8, 8],
                    Complex::new(3.0, 0.1),
                    &stencil::StencilOpts::default(),
                ),
            ),
            ("banded", structured::banded(400, 6, 1.0, 1)),
            ("kkt_arrow", structured::arrow(400, 16, 1e-2, 1)),
            ("spectral_ill", random::spectral(200, 1e8, false, 1)),
            ("rand_spd", random::random_spd(500, 12, 1.0, 1)),
        ];
        for (name, a) in cases {
            let r = ldlt_resid(&a);
            assert!(
                r <= solve_backward_error(a.n, UC64),
                "{name}: LDL^T backward error {r:.1e} too large"
            );
        }
    }

    #[test]
    fn unsymmetric_families_factor_and_solve() {
        let bem = bem::kernel(600, &bem::BemOpts::default());
        assert!(
            lu_resid(&bem) <= solve_backward_error(bem.n, UC64),
            "BEM LU backward error too large"
        );
        let r = random::random_unsym::<C>(500, 12, 2.0, 1);
        assert!(
            lu_resid(&r) <= solve_backward_error(r.n, UC64),
            "random unsymmetric LU backward error too large"
        );
    }

    #[test]
    fn solver_is_type_agnostic_all_four_scalars() {
        use crate::Scalar;
        fn sym_ok<T: Scalar>(u: f64) {
            let a = structured::banded::<T>(200, 6, 1.0, 1);
            let b: Vec<T> = (0..a.n)
                .map(|i| T::from_real((i % 7) as f64 - 3.0))
                .collect();
            let f = LdltSymbolic::analyze(&a, &SolverSettings::default())
                .unwrap()
                .factor(&a, &SolverSettings::default())
                .unwrap();
            let x = f.solve(&b).unwrap();
            let omega = backward_error(&a, &x, &b);
            assert!(
                omega <= solve_backward_error(a.n, u),
                "sym {} backward error {:.2e}",
                std::any::type_name::<T>(),
                omega
            );
        }
        fn unsym_ok<T: Scalar>(u: f64) {
            let a = random::random_unsym::<T>(200, 12, 3.0, 1);
            let b: Vec<T> = (0..a.n)
                .map(|i| T::from_real((i % 5) as f64 - 2.0))
                .collect();
            let f = LuSymbolic::analyze(&a, &SolverSettings::default())
                .unwrap()
                .factor(&a, &SolverSettings::default())
                .unwrap();
            let x = f.solve(&b).unwrap();
            let omega = backward_error(&a, &x, &b);
            assert!(
                omega <= solve_backward_error(a.n, u),
                "unsym {} backward error {:.2e}",
                std::any::type_name::<T>(),
                omega
            );
        }
        sym_ok::<f64>(U64);
        sym_ok::<f32>(U32);
        sym_ok::<C>(UC64);
        sym_ok::<Complex<f32>>(UC32);
        unsym_ok::<f64>(U64);
        unsym_ok::<f32>(U32);
        unsym_ok::<C>(UC64);
        unsym_ok::<Complex<f32>>(UC32);

        let a32 = random::random_unsym::<f32>(150, 8, 2.0, 1);
        let e32 = LuSymbolic::analyze(&a32, &SolverSettings::default())
            .unwrap()
            .estimate_memory::<f32>();
        let ac64 = random::random_unsym::<Complex<f64>>(150, 8, 2.0, 1);
        let ec64 = LuSymbolic::analyze(&ac64, &SolverSettings::default())
            .unwrap()
            .estimate_memory::<Complex<f64>>();
        assert_eq!(e32.value_bytes, 4);
        assert_eq!(ec64.value_bytes, 16);
        assert!(
            ec64.transient_peak_bytes > e32.transient_peak_bytes,
            "16B estimate > 4B"
        );
    }

    #[test]
    fn diagnostics_and_estimate_wired_on_both_paths() {
        let a = structured::banded::<C>(500, 8, 1.0, 1);
        let opts = SolverSettings::default().with_threads(3);
        let f = LdltSymbolic::analyze(&a, &SolverSettings::default())
            .unwrap()
            .factor(&a, &opts)
            .unwrap();
        let d = f.diagnostics();
        assert_eq!(d.threads, 3, "thread budget recorded");
        assert!(d.factor_bytes > 0, "factor bytes");
        assert!(d.factor_nnz > 0);
        let est = d.estimate.expect("a-priori estimate attached");
        assert_eq!(est.value_bytes, 16);
        assert!(
            est.transient_peak_bytes > est.factor_bytes,
            "transient > factor alone"
        );

        let g = bem::kernel(600, &bem::BemOpts::default());
        let o2 = SolverSettings::default().with_threads(0);
        let lf = LuSymbolic::analyze(&g, &SolverSettings::default())
            .unwrap()
            .factor(&g, &o2)
            .unwrap();
        let ld = lf.diagnostics();
        assert!(ld.threads >= 1);
        assert!(ld.estimate.is_some());
        assert_eq!(ld.estimate.unwrap().value_bytes, 16);
    }
}
