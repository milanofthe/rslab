//! Links a system CBLAS for the large dense products (`src/dense/blas.rs`).
//!
//! On macOS that is Accelerate, which runs on the matrix units of Apple
//! silicon (`RSLAB_NO_ACCELERATE=1` keeps a build on the portable kernels).
//! Elsewhere `RSLAB_BLAS_LIB` names the library (`mkl_rt`, `openblas`) and
//! `RSLAB_BLAS_DIR` adds its folder to the search path; without it every
//! product runs on the portable kernels. `RSLAB_DENSE_LIBRARY` carries the
//! choice into the crate for [`dense_library`](crate::dense_library).

use std::env;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rslab_blas)");
    for var in ["RSLAB_BLAS_LIB", "RSLAB_BLAS_DIR", "RSLAB_NO_ACCELERATE"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let library = if os == "macos" && env::var_os("RSLAB_NO_ACCELERATE").is_none() {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        Some("Accelerate".to_string())
    } else if let Some(lib) = env::var("RSLAB_BLAS_LIB").ok().filter(|l| !l.is_empty()) {
        if let Ok(dir) = env::var("RSLAB_BLAS_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
        println!("cargo:rustc-link-lib={lib}");
        if os == "linux" {
            println!("cargo:rustc-link-lib=dl"); // dlsym on glibc before 2.34
        }
        Some(lib)
    } else {
        None
    };
    if library.is_some() {
        println!("cargo:rustc-cfg=rslab_blas");
    }
    println!(
        "cargo:rustc-env=RSLAB_DENSE_LIBRARY={}",
        library.as_deref().unwrap_or("")
    );
}
