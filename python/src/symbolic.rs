//! Symbolic analysis as first-class Python objects: analyze a pattern once,
//! factor many value sets on it. Also the one-shot factor functions, which
//! are analyze + factor under one GIL release.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rslab::{
    CscMatrix, GeneralCsc, KluSolver, KluSymbolic, LdltSolver, LdltSymbolic, LuSolver, LuSymbolic,
    MemoryEstimate, RslabError, SolverSettings,
};

use crate::common::{map_err, memory_estimate_dict, scalar_bytes, vector, with_dtype, Pattern};
use crate::factor::{Field, Klu, Ldlt, Lu, Pair};
use crate::settings::{klu_settings_from, settings_from, PyKluSettings, PySettings};

/// The a-priori estimate for a value size: the core's estimator only depends
/// on `size_of::<T>()`, so dispatch on the byte width.
fn estimate_for<F>(bytes: usize, est: F) -> MemoryEstimate
where
    F: Fn(usize) -> MemoryEstimate,
{
    est(bytes)
}

// ---------------------------------------------------------------------------
// LDL^T
// ---------------------------------------------------------------------------

/// Analysis of a symmetric pattern (ordering, elimination tree, supernodes),
/// from :func:`rslab.analyze` with ``path='ldlt'``. Factor any value set on
/// the same pattern with :meth:`factor`; the analysis is paid once.
#[pyclass(name = "LdltSymbolic", module = "rslab")]
pub struct PyLdltSymbolic {
    sym: LdltSymbolic,
    pattern: Pattern,
    settings: PySettings,
}

/// Heuristic-pick or explicit analysis, as the one-shot factor does it.
fn analyze_ldlt_core<T: Field>(
    a: &CscMatrix<T>,
    st: &PySettings,
) -> Result<(LdltSymbolic, SolverSettings), RslabError> {
    let mut opts = st.resolved();
    if st.explicit_ordering {
        return Ok((LdltSymbolic::analyze_with(a, &opts)?, opts));
    }
    let (sym, pick) = LdltSolver::<T>::tuned(a)?;
    if !st.explicit_threads {
        opts.threads = pick.threads;
    }
    Ok((sym, opts))
}

fn adopt(st: &PySettings, opts: SolverSettings) -> PySettings {
    let mut s = st.clone();
    s.inner = opts;
    s
}

#[pymethods]
impl PyLdltSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.sym.n()
    }

    /// Predicted factor entries (the fill of ``L``).
    #[getter]
    fn factor_nnz(&self) -> usize {
        self.sym.symbolic_factor_nnz()
    }

    /// Levels of the supernodal elimination tree.
    #[getter]
    fn n_levels(&self) -> usize {
        self.sym.n_levels()
    }

    /// Supernodes per tree level, root level first.
    #[getter]
    fn level_widths(&self) -> Vec<usize> {
        self.sym.level_widths()
    }

    /// ``(columns, rows)`` of every front.
    #[getter]
    fn front_dims(&self) -> Vec<(usize, usize)> {
        self.sym.front_dims()
    }

    /// The settings the analysis adopted (including the heuristic thread
    /// pick); the defaults for :meth:`factor`.
    #[getter]
    fn settings(&self) -> PySettings {
        self.settings.clone()
    }

    /// A-priori memory and work estimate for a factor in the given dtype:
    /// ``factor_bytes`` / ``factor_mb``, ``transient_peak_bytes`` /
    /// ``transient_peak_mb``, ``factor_flops``, ``critical_path_flops``.
    #[pyo3(signature = (dtype = "float64"))]
    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        let e = estimate_for(scalar_bytes(dtype)?, |b| match b {
            4 => self.sym.estimate_memory::<f32>(),
            16 => self.sym.estimate_memory::<crate::common::C64>(),
            _ => self.sym.estimate_memory::<f64>(),
        });
        memory_estimate_dict(py, &e)
    }

    /// Numeric factorization of ``data`` (the CSC value array of the lower
    /// triangle, in the analyzed pattern's order, in any supported dtype).
    /// Numeric settings (``threads``, ``preconditioner``, ``drop_tol``,
    /// ``pivot_u``, ``scaling`` ...) may be overridden per call.
    #[pyo3(signature = (data, settings = None, **kwargs))]
    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Ldlt> {
        let st = settings_from(
            Some(settings.unwrap_or_else(|| self.settings.clone())),
            kwargs,
        )?;
        let opts = st.resolved();
        with_dtype!(data, |d: T| {
            let a = self.pattern.csc::<T>(d)?;
            let s = py
                .allow_threads(|| self.sym.factor(&a, &opts))
                .map_err(map_err)?;
            Ok(Ldlt {
                inner: T::ldlt(Pair::new(s, a)),
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LdltSymbolic(n={}, factor_nnz={}, n_levels={})",
            self.n(),
            self.factor_nnz(),
            self.n_levels()
        )
    }
}

/// Analyze the lower-triangle CSC pattern ``(n, indptr, indices, data)``
/// for the LDL^T path.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, **kwargs))]
pub fn analyze_ldlt(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PySettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyLdltSymbolic> {
    let st = settings_from(settings, kwargs)?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pattern.csc::<T>(d)?;
        let (sym, opts) = py
            .allow_threads(|| analyze_ldlt_core(&a, &st))
            .map_err(map_err)?;
        Ok(PyLdltSymbolic {
            sym,
            pattern,
            settings: adopt(&st, opts),
        })
    })
}

/// Analyze and factor in one step (one GIL release).
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, **kwargs))]
pub fn ldlt_factor(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PySettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Ldlt> {
    let st = settings_from(settings, kwargs)?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pattern.csc::<T>(d)?;
        let s = py
            .allow_threads(|| {
                let (sym, opts) = analyze_ldlt_core(&a, &st)?;
                sym.factor(&a, &opts)
            })
            .map_err(map_err)?;
        Ok(Ldlt {
            inner: T::ldlt(Pair::new(s, a)),
        })
    })
}

// ---------------------------------------------------------------------------
// LU
// ---------------------------------------------------------------------------

/// Analysis of a general pattern for the supernodal LU path, from
/// :func:`rslab.analyze` with ``path='lu'``. Factor any value set on the
/// same pattern with :meth:`factor`.
#[pyclass(name = "LuSymbolic", module = "rslab")]
pub struct PyLuSymbolic {
    sym: LuSymbolic,
    pattern: Pattern,
    settings: PySettings,
}

fn analyze_lu_core<T: Field>(
    a: &GeneralCsc<T>,
    st: &PySettings,
) -> Result<(LuSymbolic, SolverSettings), RslabError> {
    let mut opts = st.resolved();
    if st.explicit_ordering {
        return Ok((LuSymbolic::analyze_with(a, &opts)?, opts));
    }
    let (sym, pick) = LuSolver::<T>::tuned(a)?;
    if !st.explicit_threads {
        opts.threads = pick.threads;
    }
    Ok((sym, opts))
}

#[pymethods]
impl PyLuSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.sym.n()
    }

    /// Predicted factor entries, ``nnz(L) + nnz(U)``.
    #[getter]
    fn factor_nnz(&self) -> usize {
        self.sym.symbolic_factor_nnz()
    }

    /// Levels of the supernodal elimination tree.
    #[getter]
    fn n_levels(&self) -> usize {
        self.sym.n_levels()
    }

    /// Supernodes per tree level, root level first.
    #[getter]
    fn level_widths(&self) -> Vec<usize> {
        self.sym.level_widths()
    }

    /// ``(columns, rows)`` of every front.
    #[getter]
    fn front_dims(&self) -> Vec<(usize, usize)> {
        self.sym.front_dims()
    }

    /// The settings the analysis adopted; the defaults for :meth:`factor`.
    #[getter]
    fn settings(&self) -> PySettings {
        self.settings.clone()
    }

    /// A-priori memory and work estimate for a factor in the given dtype
    /// (same keys as :meth:`LdltSymbolic.estimate_memory`).
    #[pyo3(signature = (dtype = "float64"))]
    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        let e = estimate_for(scalar_bytes(dtype)?, |b| match b {
            4 => self.sym.estimate_memory::<f32>(),
            16 => self.sym.estimate_memory::<crate::common::C64>(),
            _ => self.sym.estimate_memory::<f64>(),
        });
        memory_estimate_dict(py, &e)
    }

    /// Numeric factorization of ``data`` (the full CSC value array in the
    /// analyzed pattern's order). Numeric settings may be overridden per call.
    #[pyo3(signature = (data, settings = None, **kwargs))]
    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Lu> {
        let st = settings_from(
            Some(settings.unwrap_or_else(|| self.settings.clone())),
            kwargs,
        )?;
        let opts = st.resolved();
        with_dtype!(data, |d: T| {
            let a = self.pattern.general::<T>(d)?;
            let s = py
                .allow_threads(|| self.sym.factor(&a, &opts))
                .map_err(map_err)?;
            Ok(Lu {
                inner: T::lu(Pair::new(s, a)),
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LuSymbolic(n={}, factor_nnz={}, n_levels={})",
            self.n(),
            self.factor_nnz(),
            self.n_levels()
        )
    }
}

/// Analyze the CSC pattern ``(n, indptr, indices, data)`` for the LU path.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, **kwargs))]
pub fn analyze_lu(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PySettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyLuSymbolic> {
    let st = settings_from(settings, kwargs)?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pattern.general::<T>(d)?;
        let (sym, opts) = py
            .allow_threads(|| analyze_lu_core(&a, &st))
            .map_err(map_err)?;
        Ok(PyLuSymbolic {
            sym,
            pattern,
            settings: adopt(&st, opts),
        })
    })
}

/// Analyze and factor in one step (one GIL release).
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, **kwargs))]
pub fn lu_factor(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PySettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Lu> {
    let st = settings_from(settings, kwargs)?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pattern.general::<T>(d)?;
        let s = py
            .allow_threads(|| {
                let (sym, opts) = analyze_lu_core(&a, &st)?;
                sym.factor(&a, &opts)
            })
            .map_err(map_err)?;
        Ok(Lu {
            inner: T::lu(Pair::new(s, a)),
        })
    })
}

// ---------------------------------------------------------------------------
// KLU
// ---------------------------------------------------------------------------

/// Analysis of a general pattern for the KLU path (block triangular form,
/// per-block AMD), from :func:`rslab.analyze` with ``path='klu'``. Factor any
/// value set on the same pattern with :meth:`factor`.
#[pyclass(name = "KluSymbolic", module = "rslab")]
pub struct PyKluSymbolic {
    sym: KluSymbolic,
    pattern: Pattern,
    settings: PyKluSettings,
}

#[pymethods]
impl PyKluSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.sym.n()
    }

    /// Number of diagonal blocks of the block triangular form.
    #[getter]
    fn n_blocks(&self) -> usize {
        self.sym.n_blocks()
    }

    /// Dimension of the largest diagonal block.
    #[getter]
    fn max_block_size(&self) -> usize {
        self.sym.max_block_size()
    }

    /// Block boundaries in the permuted order (``n_blocks + 1`` entries).
    #[getter]
    fn block_ptr(&self) -> Vec<usize> {
        self.sym.block_ptr().to_vec()
    }

    /// Predicted factor entries.
    #[getter]
    fn factor_nnz(&self) -> usize {
        self.sym.symbolic_factor_nnz()
    }

    /// The settings the analysis adopted; the defaults for :meth:`factor`.
    #[getter]
    fn settings(&self) -> PyKluSettings {
        self.settings.clone()
    }

    /// A-priori memory and work estimate for a factor in the given dtype
    /// (same keys as :meth:`LdltSymbolic.estimate_memory`).
    #[pyo3(signature = (dtype = "float64"))]
    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        let e = estimate_for(scalar_bytes(dtype)?, |b| match b {
            4 => self.sym.estimate_memory::<f32>(),
            16 => self.sym.estimate_memory::<crate::common::C64>(),
            _ => self.sym.estimate_memory::<f64>(),
        });
        memory_estimate_dict(py, &e)
    }

    /// Numeric factorization of ``data`` (the full CSC value array in the
    /// analyzed pattern's order). KLU settings may be overridden per call.
    #[pyo3(signature = (data, settings = None, **kwargs))]
    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<PyKluSettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Klu> {
        let st = klu_settings_from(
            Some(settings.unwrap_or_else(|| self.settings.clone())),
            kwargs,
        )?;
        with_dtype!(data, |d: T| {
            let a = self.pattern.general::<T>(d)?;
            let s = py
                .allow_threads(|| self.sym.factor(&a, &st.inner))
                .map_err(map_err)?;
            Ok(Klu {
                inner: T::klu(Pair::new(s, a)),
            })
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "KluSymbolic(n={}, n_blocks={}, factor_nnz={})",
            self.n(),
            self.n_blocks(),
            self.factor_nnz()
        )
    }
}

/// Analyze the CSC pattern ``(n, indptr, indices, data)`` for the KLU path.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, **kwargs))]
pub fn analyze_klu(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PyKluSettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyKluSymbolic> {
    let st = klu_settings_from(settings, kwargs)?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pattern.general::<T>(d)?;
        let sym = py
            .allow_threads(|| KluSymbolic::analyze_with(&a, &st.inner))
            .map_err(map_err)?;
        Ok(PyKluSymbolic {
            sym,
            pattern,
            settings: st,
        })
    })
}

/// Analyze and factor in one step (one GIL release).
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, **kwargs))]
pub fn klu_factor(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PyKluSettings>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Klu> {
    let st = klu_settings_from(settings, kwargs)?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pattern.general::<T>(d)?;
        let s = py
            .allow_threads(|| KluSolver::<T>::factor(&a, &st.inner))
            .map_err(map_err)?;
        Ok(Klu {
            inner: T::klu(Pair::new(s, a)),
        })
    })
}

// Silence the unused-import lint for `vector` on targets where the symbolic
// module does not extract vectors directly.
#[allow(dead_code)]
fn _uses_vector(b: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    vector::<f64>(b, "x")
}
