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
    m.add_function(wrap_pyfunction!(common::lower_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(common::is_symmetric, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::gmres_plain, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::gmres_block_plain, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::cocg_plain, m)?)?;
    m.add_function(wrap_pyfunction!(krylov::cocr_plain, m)?)?;
    m.add_function(wrap_pyfunction!(install_diagnose, m)?)?;
    m.add_function(wrap_pyfunction!(logging::set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(logging::log_level, m)?)?;
    m.add_function(wrap_pyfunction!(logging::set_log_sink, m)?)?;
    #[cfg(feature = "alloc-stats")]
    {
        m.add_function(wrap_pyfunction!(_alloc_stats, m)?)?;
        m.add_function(wrap_pyfunction!(_reset_alloc_peak, m)?)?;
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// See the `mimalloc` note in Cargo.toml: the core stays allocator-agnostic,
// the extension module picks the allocator for the process.
#[cfg(not(feature = "alloc-stats"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Feature `alloc-stats`: mimalloc behind byte counters, so the benches can read
// the exact heap peak of a phase (`_alloc_stats`, `_reset_alloc_peak`) and hold
// the memory estimate against it. Measurement builds only: every allocation
// pays two atomic updates.
#[cfg(feature = "alloc-stats")]
#[global_allocator]
static GLOBAL: counting::Counting = counting::Counting;

#[cfg(feature = "alloc-stats")]
mod counting {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    /// Bytes currently allocated, and the most at any point since the last reset.
    pub static LIVE: AtomicUsize = AtomicUsize::new(0);
    pub static PEAK: AtomicUsize = AtomicUsize::new(0);

    pub struct Counting;

    fn grow(bytes: usize) {
        let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
        PEAK.fetch_max(live, Relaxed);
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = mimalloc::MiMalloc.alloc(layout);
            if !p.is_null() {
                grow(layout.size());
            }
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let p = mimalloc::MiMalloc.alloc_zeroed(layout);
            if !p.is_null() {
                grow(layout.size());
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
            mimalloc::MiMalloc.dealloc(p, layout);
            LIVE.fetch_sub(layout.size(), Relaxed);
        }
        unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let q = mimalloc::MiMalloc.realloc(p, layout, size);
            if !q.is_null() {
                if size >= layout.size() {
                    grow(size - layout.size());
                } else {
                    LIVE.fetch_sub(layout.size() - size, Relaxed);
                }
            }
            q
        }
    }
}

/// `(live, peak)` bytes of the Rust heap (feature `alloc-stats` only).
#[cfg(feature = "alloc-stats")]
#[pyfunction]
fn _alloc_stats() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (counting::LIVE.load(Relaxed), counting::PEAK.load(Relaxed))
}

/// Restarts the peak at the current live bytes (feature `alloc-stats` only).
#[cfg(feature = "alloc-stats")]
#[pyfunction]
fn _reset_alloc_peak() {
    use std::sync::atomic::Ordering::Relaxed;
    counting::PEAK.store(counting::LIVE.load(Relaxed), Relaxed);
}
