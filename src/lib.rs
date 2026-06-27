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
//! let analysis = LdltSymbolic::analyze(&a)?;                 // phase 1
//! let factor = analysis.factor(&a, &FactorOptions::default())?; // 2/3
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
//! let opts = FactorOptions {
//!     on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
//!     drop_tol: Some(1e-2),
//! };
//! let m = LdltSolver::factor_with(&a, &opts)?;          // preconditioner
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
// MC64 max-product matching + equilibration. Currently only the symbolic MC64
// cache is wired; the numeric consumption (and the dense f64 scaling helpers)
// will be re-attached to the generic path during the feral feature port, so the
// not-yet-wired items are allowed dead for now rather than deleted.
#[allow(dead_code)]
pub mod scaling;
pub mod sparse;
pub mod symbolic;

// Flat public API re-exported at crate root — a single data-type-generic
// (`Scalar`: f64, Complex<f64>, f32, Complex<f32>) sparse direct + iterative
// stack. (The legacy f64-dedicated multifrontal path has been removed.)
pub use dense::matrix::SymmetricMatrix;
pub use error::FeralError;
/// Ergonomic alias for the crate error type ([`FeralError`]).
pub use error::FeralError as RlaError;
pub use scalar::Scalar;
// Generic dense LDLᵀ kernel (the multifrontal fronts reduce to this).
pub use dense::ldlt_generic::{factor_ldlt, solve_ldlt, LdltFactors};
// Generic symmetric LDLᵀ multifrontal direct solver.
pub use numeric::multifrontal_ldlt::{
    analyze, factor_numeric, factor_sparse_ldlt, factor_sparse_ldlt_with, set_use_gemm_schur,
    FactorOptions, MultifrontalSymbolic, ZeroPivotAction,
};
// Generic unsymmetric LU multifrontal direct solver.
pub use numeric::multifrontal_lu::{
    analyze_general, factor_general_lu, factor_general_lu_numeric, solve_lu, solve_lu_refined,
    LuFactors, LuSymbolic,
};
pub use numeric::sparse_solver::{LdltSolver, LdltSymbolic};
pub use numeric::iterative::{
    cocg, cocr, gmres, Factorization, KrylovResult, LinearOperator, LowPrecisionLu,
    LowPrecisionPreconditioner, NoPreconditioner, Preconditioner,
};
pub use inertia::Inertia;
pub use io::mtx::{
    parse_mtx, parse_mtx_complex, parse_mtx_complex_general, read_mtx, read_mtx_complex, MtxMatrix,
};
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
        solve_lu_refined, CscMatrix, Factorization, FeralError, GeneralCsc, FactorOptions,
        MultifrontalSymbolic, KrylovResult, LinearOperator, LowPrecisionPreconditioner, LuFactors,
        MtxMatrix, NoPreconditioner, Preconditioner, Scalar, LdltSolver, LdltSymbolic,
        ZeroPivotAction,
    };
}
