// Deny `.unwrap()` and `.expect()` in production code, but allow them in
// test modules (inside `#[cfg(test)]` blocks) where panics are acceptable.
// This is a structural enforcement of the CLAUDE.md hard rule against
// unwrap in `src/`, replacing the ad-hoc grep check in CI.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Style lints that fire only in test scaffolding — relaxed under cfg(test).
// The lib build keeps default clippy strictness.
#![cfg_attr(test, allow(clippy::needless_range_loop))]

//! # RLA — a pure-Rust sparse symmetric direct solver and preconditioner
//!
//! A self-contained replacement for PARDISO's sparse symmetric path, with no
//! MKL or other native dependency. RLA factors **real symmetric** (`f64`,
//! PARDISO `mtype 2`) and **complex symmetric** (`Complex<f64>`, `mtype 6`)
//! matrices as `Pᵀ A P = L D Lᵀ` by a rayon-parallel multifrontal
//! Bunch-Kaufman method with a SIMD (`gemm`) Schur kernel.
//!
//! Two intended uses:
//! * **FEM direct solve** — factor once, solve many right-hand sides.
//! * **MoM sparse preconditioner** — a robust, memory-light approximate factor
//!   (static pivoting, `f32` mixed precision, incomplete-factor dropping)
//!   driving a [`cocg`]/[`cocr`] iteration.
//!
//! ## PARDISO-style phased workflow (FEM)
//!
//! Analyze the sparsity pattern once, then factor many value sets that share it
//! (Newton steps, time stepping, frequency sweep):
//!
//! ```
//! # fn main() -> Result<(), rla::FeralError> {
//! use rla::prelude::*;
//! // Real symmetric matrix, lower triangle (i ≥ j).
//! let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!                                         &[2.0, 2.0, 2.0, -1.0])?;
//! let analysis = SymbolicAnalysis::analyze(&a)?;                 // phase 1
//! let factor = analysis.factor(&a, &GenericFactorOptions::default())?; // 2/3
//! let x = factor.solve(&[1.0, 2.0, 3.0])?;
//! # let _ = x; Ok(()) }
//! ```
//!
//! ## Complex-symmetric MoM preconditioner
//!
//! A robust, low-memory factor (never-fail static pivoting + incomplete
//! dropping) used to precondition COCG:
//!
//! ```
//! # fn main() -> Result<(), rla::FeralError> {
//! use rla::prelude::*;
//! use num_complex::Complex;
//! let c = |re, im| Complex::new(re, im);
//! let a = CscMatrix::<Complex<f64>>::from_triplets(
//!     3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!     &[c(4.0, 1.0), c(4.0, 1.0), c(4.0, 1.0), c(-1.0, 0.2)])?;
//! let opts = GenericFactorOptions {
//!     on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
//!     drop_tol: Some(1e-2),
//! };
//! let m = SparseSymmetricLdlt::factor_with(&a, &opts)?;          // preconditioner
//! let b = vec![c(1.0, 0.0); 3];
//! let res = cocg(&a, &b, &m, 1e-10, 100)?;
//! assert!(res.converged);
//! # Ok(()) }
//! ```

pub mod dense;
pub mod error;
pub mod inertia;
pub mod io;
pub mod numeric;
pub mod ordering;
pub mod scalar;
pub mod scaling;
pub mod sparse;
pub mod symbolic;

// Flat public API re-exported at crate root:
pub use dense::factor::{
    factor, factor_single_front, BunchKaufmanParams, Factors, ZeroPivotAction,
};
pub use dense::matrix::SymmetricMatrix;
pub use dense::solve::{solve, solve_refined};
pub use error::FeralError;
/// Ergonomic alias for the crate error type ([`FeralError`]).
pub use error::FeralError as RlaError;
pub use scalar::Scalar;
// Generic (real + complex-symmetric) sparse direct solver — the RLA entry point.
pub use dense::ldlt_generic::{factor_ldlt, solve_ldlt, LdltFactors};
pub use numeric::multifrontal_generic::{
    factor_sparse_ldlt, factor_sparse_ldlt_with, set_use_gemm_schur, GenericFactorOptions,
};
pub use numeric::multifrontal_generic::{analyze, factor_numeric, GenericSymbolic};
pub use numeric::multifrontal_lu::{
    analyze_general, factor_general_lu, factor_general_lu_numeric, solve_lu, solve_lu_refined,
    LuFactors, LuSymbolic,
};
pub use numeric::sparse_solver::{SparseSymmetricLdlt, SymbolicAnalysis};
pub use numeric::iterative::{
    cocg, cocr, gmres, Factorization, KrylovResult, LinearOperator, LowPrecisionLu,
    LowPrecisionPreconditioner, NoPreconditioner, Preconditioner,
};
pub use inertia::Inertia;
pub use io::mtx::{
    parse_mtx, parse_mtx_complex, parse_mtx_complex_general, read_mtx, read_mtx_complex, MtxMatrix,
};
pub use numeric::condition::{estimate_condition_1norm, estimate_inverse_norm_1, matrix_norm_1};
pub use numeric::factorize::{
    factorize_multifrontal_with_schur, LdltExport, NumericParams, ProfileReport, SchurBlock,
};
pub use numeric::solve::{
    solve_sparse, solve_sparse_refined, solve_sparse_refined_with_diagnostics,
    RefinementDiagnostics, RefinementStep,
};
pub use numeric::solver::{FactorStats, FactorStatus, QualityLevel, Solver};
pub use sparse::csc::{CscMatrix, CscPattern};
pub use sparse::general::GeneralCsc;
pub use symbolic::SymbolicProfileReport;

/// Ergonomic imports for embedding RLA as a PARDISO-style sparse solver /
/// preconditioner. `use rla::prelude::*;` brings in the matrix type, the
/// phased analysis/factor API, the iterative solvers and preconditioners, the
/// options enums, and the Matrix Market loaders.
pub mod prelude {
    pub use crate::{
        analyze, cocg, cocr, factor_general_lu, factor_numeric, gmres, parse_mtx,
        parse_mtx_complex, parse_mtx_complex_general, read_mtx, read_mtx_complex, solve_lu,
        solve_lu_refined, CscMatrix, Factorization, FeralError, GeneralCsc, GenericFactorOptions,
        GenericSymbolic, KrylovResult, LinearOperator, LowPrecisionPreconditioner, LuFactors,
        MtxMatrix, NoPreconditioner, Preconditioner, Scalar, SparseSymmetricLdlt, SymbolicAnalysis,
        ZeroPivotAction,
    };
}
