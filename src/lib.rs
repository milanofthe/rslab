// Deny `.unwrap()` and `.expect()` in production code, but allow them in
// test modules (inside `#[cfg(test)]` blocks) where panics are acceptable.
// This is a structural enforcement of the CLAUDE.md hard rule against
// unwrap in `src/`, replacing the ad-hoc grep check in CI.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Style lints that fire only in test scaffolding — relaxed under cfg(test).
// The lib build keeps default clippy strictness.
#![cfg_attr(test, allow(clippy::needless_range_loop))]

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
pub use scalar::Scalar;
// Generic (real + complex-symmetric) sparse direct solver — the RLA entry point.
pub use dense::ldlt_generic::{factor_ldlt, solve_ldlt, LdltFactors};
pub use numeric::multifrontal_generic::{
    factor_sparse_ldlt, factor_sparse_ldlt_with, set_use_gemm_schur, GenericFactorOptions,
};
pub use numeric::multifrontal_generic::{analyze, factor_numeric, GenericSymbolic};
pub use numeric::sparse_solver::{SparseSymmetricLdlt, SymbolicAnalysis};
pub use numeric::iterative::{
    cocg, cocr, KrylovResult, LowPrecisionPreconditioner, NoPreconditioner, Preconditioner,
};
pub use inertia::Inertia;
pub use io::mtx::{parse_mtx, parse_mtx_complex, read_mtx, read_mtx_complex, MtxMatrix};
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
pub use symbolic::SymbolicProfileReport;
