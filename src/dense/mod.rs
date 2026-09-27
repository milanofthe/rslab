#[cfg(rslab_blas)]
pub(crate) mod blas;
pub mod gemm_backend;
#[cfg(test)]
pub(crate) mod matrix;
