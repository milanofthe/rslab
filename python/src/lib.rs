//! `rslab._rslab`: the compiled extension behind the `rslab` Python package.
//!
//! The module is organised by concern: `settings` (the configuration
//! objects), `symbolic` (analysis handles and the one-shot factor functions),
//! `factor` (the `Ldlt` / `Lu` / `Klu` handles), `krylov` (iterative solvers)
//! and `logging`. The Python package `rslab/__init__.py` adds the SciPy-facing
//! wrappers (matrix conversion, symmetry detection, `spsolve`).

#![allow(clippy::useless_conversion)]

mod common;
mod factor;
mod krylov;
mod logging;
mod settings;
mod symbolic;

use pyo3::prelude::*;

/// One-time machine calibration: measure this machine's factorization
/// throughput and thread-speedup curve and cache them, so later factor calls
/// pick their worker count from the measurement. Returns the measured values.
#[pyfunction]
fn install_diagnose(py: Python<'_>) -> PyResult<PyObject> {
    let c = py.allow_threads(rslab::tuning::install_diagnose);
    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("gflops_f64", c.geom_gflops)?;
    d.set_item("gflops_complex", c.geom_gflops_cplx)?;
    d.set_item("speedup", c.speedup)?;
    d.set_item("speedup_threads", c.speedup_threads)?;
    d.set_item("speedup_at_4", c.speedup4)?;
    d.set_item("timing_cv", c.time_cv)?;
    Ok(d.into())
}

#[pymodule]
fn _rslab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<settings::PySettings>()?;
    m.add_class::<settings::PyKluSettings>()?;
    m.add_class::<settings::PyInterrupt>()?;
    m.add_class::<factor::Ldlt>()?;
    m.add_class::<factor::Lu>()?;
    m.add_class::<factor::Klu>()?;
    m.add_class::<factor::PyRecycle>()?;
    m.add_class::<symbolic::PyLdltSymbolic>()?;
    m.add_class::<symbolic::PyLuSymbolic>()?;
    m.add_class::<symbolic::PyKluSymbolic>()?;
    m.add_class::<krylov::PyKrylovResult>()?;
    m.add_function(wrap_pyfunction!(symbolic::ldlt_factor, m)?)?;
    m.add_function(wrap_pyfunction!(symbolic::lu_factor, m)?)?;
    m.add_function(wrap_pyfunction!(symbolic::klu_factor, m)?)?;
    m.add_function(wrap_pyfunction!(symbolic::analyze_ldlt, m)?)?;
    m.add_function(wrap_pyfunction!(symbolic::analyze_lu, m)?)?;
    m.add_function(wrap_pyfunction!(symbolic::analyze_klu, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::gmres_plain, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::gmres_block_plain, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::cocg_plain, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::cocr_plain, m)?)?;
    m.add_function(wrap_pyfunction!(install_diagnose, m)?)?;
    m.add_function(wrap_pyfunction!(logging::set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(logging::log_level, m)?)?;
    m.add_function(wrap_pyfunction!(logging::set_log_sink, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
