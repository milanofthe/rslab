//! Small dense complex linear algebra for the projected matrices of Krylov
//! methods: [`lu_solve`], the [`Schur`] form with eigenvectors and shifted
//! solves, [`eigen`]. The dense kernels of the factorizations (`gemm_backend`)
//! are internal.

pub(crate) mod gemm_backend;
#[cfg(test)]
pub(crate) mod matrix;
mod small;

pub use small::{eigen, lu_solve, Eigen, Schur};
