//! Parametrized test-matrix generators for benchmarking and testing the solver
//! across the axes that matter for a sparse direct solver: **size**, **sparsity
//! structure**, **symmetry**, **conditioning**, and **density**.
//!
//! Design follows the established packages (MatrixDepot.jl, MATLAB `gallery`,
//! LAPACK `xLATMS`): parametrized *generators* for structured coverage plus an
//! optional *downloader* ([`download`], feature `matgen-download`) for real
//! SuiteSparse / Matrix Market matrices. Conditioning is steered **structurally**
//! (diagonal dominance, PDE refinement, near-resonance shift, coefficient jumps)
//! for the sparse families, and **spectrally** (prescribed eigenvalues, `xLATMS`
//! style) for the small dense [`spectral`] family where exact κ is needed.
//!
//! The real-valued structural families are generic over [`Scalar`]; the inherently
//! complex families (Helmholtz, BEM/MoM kernel) produce `Complex<f64>`. The
//! [`catalog`] is `Complex<f64>` (the solver's primary EM/MoM type) and tags each
//! entry so benchmarks can sweep e.g. "all SPD small" or "all ill-conditioned".

use num_complex::Complex;

use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

pub mod bem;
#[cfg(feature = "matgen-download")]
pub mod download;
pub mod random;
pub mod stencil;
pub mod structured;

/// Small, fast, deterministic PRNG (xorshift64*). Pure Rust, no `rand` dep — so
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
    /// Standard normal via Box–Muller (one of the pair).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = (self.unit()).max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// --------------------------------------------------------------------------
// Catalog: tagged, named matrices for benchmark sweeps.
// --------------------------------------------------------------------------

/// Sparsity / origin structure of a catalog matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structure {
    Stencil2D,
    Stencil3D,
    Bem,
    Banded,
    Arrow,
    Random,
    Spectral,
}

/// Symmetry class — selects the solver path (LDLᵀ vs LU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// Real or complex symmetric positive definite.
    Spd,
    /// Complex-symmetric (A = Aᵀ, not Hermitian) — the EM-FEM case.
    ComplexSymmetric,
    /// Symmetric indefinite (saddle/KKT).
    SymIndefinite,
    /// Unsymmetric (MoM/BEM).
    Unsymmetric,
}

/// Rough conditioning class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    Well,
    Moderate,
    Ill,
}

/// Rough nonzeros-per-row class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Density {
    Sparse,
    Medium,
    Dense,
}

/// A generated matrix in the form the solver consumes: symmetric matrices as a
/// lower-triangle [`CscMatrix`] (→ LDLᵀ), unsymmetric as a full [`GeneralCsc`]
/// (→ LU).
pub enum Generated {
    Symmetric(CscMatrix<Complex<f64>>),
    Unsymmetric(GeneralCsc<Complex<f64>>),
}

impl Generated {
    pub fn n(&self) -> usize {
        match self {
            Generated::Symmetric(a) => a.n,
            Generated::Unsymmetric(a) => a.n,
        }
    }
    pub fn nnz(&self) -> usize {
        match self {
            Generated::Symmetric(a) => a.values.len(),
            Generated::Unsymmetric(a) => a.values.len(),
        }
    }
}

/// A catalog entry: a tagged, named, on-demand matrix builder.
pub struct MatrixSpec {
    pub name: &'static str,
    pub structure: Structure,
    pub symmetry: Symmetry,
    pub cond: Cond,
    pub density: Density,
    /// Approximate dimension (the actual `n` may round to the grid size).
    pub size: usize,
    build: fn() -> Generated,
}

impl MatrixSpec {
    pub fn build(&self) -> Generated {
        (self.build)()
    }
}

/// The full catalog of named, tagged test matrices. Filter with the iterator
/// adapters, e.g. `catalog().into_iter().filter(|m| m.cond == Cond::Ill)`.
pub fn catalog() -> Vec<MatrixSpec> {
    let mut c = Vec::new();
    stencil::add_to_catalog(&mut c);
    bem::add_to_catalog(&mut c);
    structured::add_to_catalog(&mut c);
    random::add_to_catalog(&mut c);
    c
}
