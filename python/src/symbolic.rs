//! Symbolic analysis as first-class Python objects: analyze a pattern once,
//! factor many value sets on it. Also the one-shot factor functions, which
//! are analyze + factor under one GIL release.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rslab::{
    CscMatrix, GeneralCsc, KluSettings, KluSymbolic, LdltSymbolic, LuSymbolic, MemoryEstimate,
    MemoryPlan, RslabError, SolverSettings,
};

use crate::common::{
    csc_bytes, heavy, map_err, memory_estimate_dict, memory_plan_dict, vector, with_dtype,
    with_dtype_name, Pattern, Values,
};
use crate::factor::{Auto, Field, Klu, Ldlt, Lu, Pair};
use crate::settings::{klu_settings_from, settings_from, PyKluSettings, PySettings};

pub trait Analysis: Sized + Send + Sync {
    type Settings: Clone;
    type Options: Sync;
    type Matrix<T: Field>: Sync;
    type Factor;
    const LOWER: bool;
    fn settings(
        settings: Option<Self::Settings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self::Settings>;
    fn options(st: &Self::Settings) -> Self::Options;
    fn adopted(st: &Self::Settings, opts: Self::Options) -> Self::Settings;
    fn matrix<T: Field>(p: &Pattern, values: Vec<T>) -> PyResult<Self::Matrix<T>>;
    fn analyze_matrix<T: Field>(
        a: &Self::Matrix<T>,
        opts: &Self::Options,
    ) -> Result<Self, RslabError>;
    fn factor_matrix<T: Field>(
        &self,
        py: Python<'_>,
        a: Self::Matrix<T>,
        opts: &Self::Options,
        st: &Self::Settings,
    ) -> PyResult<Self::Factor>;
    fn estimate<T: Field>(&self) -> MemoryEstimate;
    fn plan<T: Field>(&self, opts: &Self::Options, nrhs: usize) -> MemoryPlan;
    fn bytes(&self) -> u64;
}

impl Analysis for LdltSymbolic {
    type Settings = PySettings;
    type Options = SolverSettings;
    type Matrix<T: Field> = CscMatrix<T>;
    type Factor = Ldlt;
    const LOWER: bool = true;
    fn settings(
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySettings> {
        settings_from(settings, kwargs)
    }
    fn options(st: &PySettings) -> SolverSettings {
        st.resolved()
    }
    fn adopted(st: &PySettings, opts: SolverSettings) -> PySettings {
        adopt(st, opts)
    }
    fn matrix<T: Field>(p: &Pattern, values: Vec<T>) -> PyResult<CscMatrix<T>> {
        p.csc(values)
    }
    fn analyze_matrix<T: Field>(
        a: &CscMatrix<T>,
        opts: &SolverSettings,
    ) -> Result<Self, RslabError> {
        LdltSymbolic::analyze(a, opts)
    }
    fn factor_matrix<T: Field>(
        &self,
        py: Python<'_>,
        a: CscMatrix<T>,
        opts: &SolverSettings,
        st: &PySettings,
    ) -> PyResult<Ldlt> {
        T::ldlt_factor(py, self, a, opts, st.factor_dtype.as_deref()).map(|inner| Ldlt { inner })
    }
    fn estimate<T: Field>(&self) -> MemoryEstimate {
        self.estimate_memory::<T>()
    }
    fn plan<T: Field>(&self, opts: &SolverSettings, nrhs: usize) -> MemoryPlan {
        self.memory_plan::<T>(opts, nrhs)
    }
    fn bytes(&self) -> u64 {
        self.heap_bytes()
    }
}

impl Analysis for LuSymbolic {
    type Settings = PySettings;
    type Options = SolverSettings;
    type Matrix<T: Field> = GeneralCsc<T>;
    type Factor = Lu;
    const LOWER: bool = false;
    fn settings(
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySettings> {
        settings_from(settings, kwargs)
    }
    fn options(st: &PySettings) -> SolverSettings {
        st.resolved()
    }
    fn adopted(st: &PySettings, opts: SolverSettings) -> PySettings {
        adopt(st, opts)
    }
    fn matrix<T: Field>(p: &Pattern, values: Vec<T>) -> PyResult<GeneralCsc<T>> {
        p.general(values)
    }
    fn analyze_matrix<T: Field>(
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<Self, RslabError> {
        LuSymbolic::analyze(a, opts)
    }
    fn factor_matrix<T: Field>(
        &self,
        py: Python<'_>,
        a: GeneralCsc<T>,
        opts: &SolverSettings,
        st: &PySettings,
    ) -> PyResult<Lu> {
        T::lu_factor(py, self, a, opts, st.factor_dtype.as_deref()).map(|inner| Lu { inner })
    }
    fn estimate<T: Field>(&self) -> MemoryEstimate {
        self.estimate_memory::<T>()
    }
    fn plan<T: Field>(&self, opts: &SolverSettings, nrhs: usize) -> MemoryPlan {
        self.memory_plan::<T>(opts, nrhs)
    }
    fn bytes(&self) -> u64 {
        self.heap_bytes()
    }
}

impl Analysis for KluSymbolic {
    type Settings = PyKluSettings;
    type Options = KluSettings;
    type Matrix<T: Field> = GeneralCsc<T>;
    type Factor = Klu;
    const LOWER: bool = false;
    fn settings(
        settings: Option<PyKluSettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyKluSettings> {
        klu_settings_from(settings, kwargs)
    }
    fn options(st: &PyKluSettings) -> KluSettings {
        st.inner.clone()
    }
    fn adopted(st: &PyKluSettings, _opts: KluSettings) -> PyKluSettings {
        st.clone()
    }
    fn matrix<T: Field>(p: &Pattern, values: Vec<T>) -> PyResult<GeneralCsc<T>> {
        p.general(values)
    }
    fn analyze_matrix<T: Field>(a: &GeneralCsc<T>, opts: &KluSettings) -> Result<Self, RslabError> {
        KluSymbolic::analyze(a, opts)
    }
    fn factor_matrix<T: Field>(
        &self,
        py: Python<'_>,
        a: GeneralCsc<T>,
        opts: &KluSettings,
        _st: &PyKluSettings,
    ) -> PyResult<Klu> {
        let s = heavy(py, || self.factor(&a, opts)).map_err(map_err)?;
        Ok(Klu {
            inner: T::klu(Pair::new(s, a)),
        })
    }
    fn estimate<T: Field>(&self) -> MemoryEstimate {
        self.estimate_memory::<T>()
    }
    fn plan<T: Field>(&self, opts: &KluSettings, nrhs: usize) -> MemoryPlan {
        self.memory_plan::<T>(opts, nrhs)
    }
    fn bytes(&self) -> u64 {
        self.heap_bytes()
    }
}

fn adopt(st: &PySettings, opts: SolverSettings) -> PySettings {
    let mut s = st.clone();
    s.inner = opts;
    s
}

pub struct Analyzed<S: Analysis> {
    sym: S,
    pattern: Pattern,
    settings: S::Settings,
}

impl<S: Analysis> Analyzed<S> {
    fn analyze(
        py: Python<'_>,
        n: usize,
        indptr: PyReadonlyArray1<i64>,
        indices: PyReadonlyArray1<i64>,
        data: &Bound<'_, PyAny>,
        settings: Option<S::Settings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let st = S::settings(settings, kwargs)?;
        let pattern = Pattern::from_py(n, &indptr, &indices)?;
        with_dtype!(data, |T| {
            let a = S::matrix::<T>(&pattern, vector(data, "data")?)?;
            let opts = S::options(&st);
            let sym = heavy(py, || S::analyze_matrix(&a, &opts)).map_err(map_err)?;
            Ok(Self {
                sym,
                pattern,
                settings: S::adopted(&st, opts),
            })
        })
    }

    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<S::Settings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<S::Factor> {
        let st = S::settings(
            Some(settings.unwrap_or_else(|| self.settings.clone())),
            kwargs,
        )?;
        let opts = S::options(&st);
        let p = &self.pattern;
        let values = Values::of(py, data, &p.col_ptr, &p.row_idx, S::LOWER)?;
        with_dtype!(values.data(), |T| {
            let a = S::matrix::<T>(p, values.read()?)?;
            self.sym.factor_matrix(py, a, &opts, &st)
        })
    }

    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        let e = with_dtype_name!(dtype, |T| self.sym.estimate::<T>())?;
        memory_estimate_dict(py, &e)
    }

    fn memory_plan(
        &self,
        py: Python<'_>,
        dtype: &str,
        nrhs: usize,
        settings: Option<S::Settings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let st = S::settings(
            Some(settings.unwrap_or_else(|| self.settings.clone())),
            kwargs,
        )?;
        let (sym, opts) = (&self.sym, S::options(&st));
        let (plan, vb) = with_dtype_name!(dtype, |T| (
            heavy(py, || sym.plan::<T>(&opts, nrhs)),
            std::mem::size_of::<T>()
        ))?;
        memory_plan_dict(py, plan, &self.pattern, vb)
    }

    fn heap_bytes(&self) -> u64 {
        let p = &self.pattern;
        self.sym.bytes() + csc_bytes(&p.col_ptr, &p.row_idx, &Vec::<u8>::new())
    }
}

/// Analysis of a symmetric pattern (ordering, elimination tree, supernodes),
/// from :func:`rslab.analyze` with ``path='ldlt'``. Factor any value set on
/// the same pattern with :meth:`factor`; the analysis is paid once.
#[pyclass(name = "LdltSymbolic", module = "rslab")]
pub struct PyLdltSymbolic {
    inner: Analyzed<LdltSymbolic>,
}

#[pymethods]
impl PyLdltSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.inner.sym.n()
    }

    /// Predicted factor entries (the fill of ``L``).
    #[getter]
    fn factor_nnz(&self) -> usize {
        self.inner.sym.symbolic_factor_nnz()
    }

    /// Heap bytes this analysis holds on the Rust side, the stored pattern
    /// included.
    #[getter]
    fn heap_bytes(&self) -> u64 {
        self.inner.heap_bytes()
    }

    /// Levels of the supernodal elimination tree.
    #[getter]
    fn n_levels(&self) -> usize {
        self.inner.sym.n_levels()
    }

    /// Supernodes per tree level, root level first.
    #[getter]
    fn level_widths(&self) -> Vec<usize> {
        self.inner.sym.level_widths()
    }

    /// ``(columns, rows)`` of every front.
    #[getter]
    fn front_dims(&self) -> Vec<(usize, usize)> {
        self.inner.sym.front_dims()
    }

    /// The settings of the analysis; the defaults for :meth:`factor`.
    #[getter]
    fn settings(&self) -> PySettings {
        self.inner.settings.clone()
    }

    /// A-priori memory and work estimate for a factor in the given dtype.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use (``'float64'``, ``'complex128'``,
    ///     ``'float32'``, ``'complex64'``).
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``factor_bytes`` / ``factor_mb``, ``transient_peak_bytes`` /
    ///     ``transient_peak_mb`` (the peak during the factorization),
    ///     ``factor_flops``, ``critical_path_flops``.
    #[pyo3(signature = (dtype = "float64"))]
    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        self.inner.estimate_memory(py, dtype)
    }

    /// The heap a factorization needs, predicted from the analysis before
    /// any numeric work: for a preflight check against the memory available
    /// and for scheduling factorizations side by side.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use.
    /// nrhs : int, default 1
    ///     Right-hand sides per solve.
    /// settings : Settings, optional
    ///     The settings :meth:`factor` will run with (the threads matter: the kernels' scratch grows with them); the
    ///     analysis settings by default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``peak_bytes``, the heap peak from here through a factorization
    ///     and a solve (the number to compare against the memory available),
    ///     ``resident_bytes``, what stays held while the factor is kept, and
    ///     their parts: ``analysis_bytes``, ``analysis_growth_bytes``,
    ///     ``factor_bytes``, ``factor_peak_bytes`` (above what was live when
    ///     the factorization began), ``solve_bytes`` and ``kept_bytes``
    ///     (what the worker threads, and a refactorized factor's scratch,
    ///     keep for the next factorization), with ``threads``
    ///     and ``nrhs``. The copies this binding keeps (the pattern, the
    ///     matrix in the factor, a solve's right-hand sides) are included;
    ///     NumPy's own arrays are not.
    #[pyo3(signature = (dtype = "float64", nrhs = 1, settings = None, **kwargs))]
    fn memory_plan(
        &self,
        py: Python<'_>,
        dtype: &str,
        nrhs: usize,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        self.inner.memory_plan(py, dtype, nrhs, settings, kwargs)
    }

    /// Numeric factorization of ``data`` (the CSC value array of the lower
    /// triangle, in the analyzed pattern's order, in any supported dtype).
    /// Numeric settings (``threads``, ``preconditioner``,
    /// ``pivot_threshold``, ``scaling`` ...) may be overridden per call.
    #[pyo3(signature = (data, settings = None, **kwargs))]

    /// Numeric factorization of new values on the analyzed pattern.
    ///
    /// Parameters
    /// ----------
    /// data : scipy.sparse matrix or ndarray
    ///     The matrix itself (any SciPy sparse matrix with the analyzed
    ///     pattern: its lower triangle is taken, sorted and summed like at analysis
    ///     time, and the pattern is checked entry by entry), or the CSC value
    ///     array of that lower triangle in the order of the analysis.
    /// settings : Settings, optional
    ///     Numeric settings for this factorization; the analysis settings by
    ///     default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    ///
    /// Returns
    /// -------
    /// factor handle
    ///     The numeric factor with the solve methods and diagnostics.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the pattern or the value count differs from the analysis.
    /// RuntimeError
    ///     If a pivot is numerically zero in exact mode.
    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Ldlt> {
        self.inner.factor(py, data, settings, kwargs)
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
    Analyzed::analyze(py, n, indptr, indices, data, settings, kwargs)
        .map(|inner| PyLdltSymbolic { inner })
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
    Analyzed::<LdltSymbolic>::analyze(py, n, indptr, indices, data, settings, kwargs)?
        .factor(py, data, None, None)
}

/// Analysis of a general pattern for the supernodal LU path, from
/// :func:`rslab.analyze` with ``path='lu'``. Factor any value set on the
/// same pattern with :meth:`factor`.
#[pyclass(name = "LuSymbolic", module = "rslab")]
pub struct PyLuSymbolic {
    inner: Analyzed<LuSymbolic>,
}

#[pymethods]
impl PyLuSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.inner.sym.n()
    }

    /// Predicted factor entries, ``nnz(L) + nnz(U)``.
    #[getter]
    fn factor_nnz(&self) -> usize {
        self.inner.sym.symbolic_factor_nnz()
    }

    /// Heap bytes this analysis holds on the Rust side, the stored pattern
    /// included.
    #[getter]
    fn heap_bytes(&self) -> u64 {
        self.inner.heap_bytes()
    }

    /// Levels of the supernodal elimination tree.
    #[getter]
    fn n_levels(&self) -> usize {
        self.inner.sym.n_levels()
    }

    /// Supernodes per tree level, root level first.
    #[getter]
    fn level_widths(&self) -> Vec<usize> {
        self.inner.sym.level_widths()
    }

    /// ``(columns, rows)`` of every front.
    #[getter]
    fn front_dims(&self) -> Vec<(usize, usize)> {
        self.inner.sym.front_dims()
    }

    /// The settings the analysis adopted; the defaults for :meth:`factor`.
    #[getter]
    fn settings(&self) -> PySettings {
        self.inner.settings.clone()
    }

    /// A-priori memory and work estimate for a factor in the given dtype.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use (``'float64'``, ``'complex128'``,
    ///     ``'float32'``, ``'complex64'``).
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``factor_bytes`` / ``factor_mb``, ``transient_peak_bytes`` /
    ///     ``transient_peak_mb`` (the peak during the factorization),
    ///     ``factor_flops``, ``critical_path_flops``.
    #[pyo3(signature = (dtype = "float64"))]
    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        self.inner.estimate_memory(py, dtype)
    }

    /// The heap a factorization needs, predicted from the analysis before
    /// any numeric work: for a preflight check against the memory available
    /// and for scheduling factorizations side by side.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use.
    /// nrhs : int, default 1
    ///     Right-hand sides per solve.
    /// settings : Settings, optional
    ///     The settings :meth:`factor` will run with (the threads matter: the kernels' scratch grows with them); the
    ///     analysis settings by default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``peak_bytes``, the heap peak from here through a factorization
    ///     and a solve (the number to compare against the memory available),
    ///     ``resident_bytes``, what stays held while the factor is kept, and
    ///     their parts: ``analysis_bytes``, ``analysis_growth_bytes``,
    ///     ``factor_bytes``, ``factor_peak_bytes`` (above what was live when
    ///     the factorization began), ``solve_bytes`` and ``kept_bytes``
    ///     (what the worker threads, and a refactorized factor's scratch,
    ///     keep for the next factorization), with ``threads``
    ///     and ``nrhs``. The copies this binding keeps (the pattern, the
    ///     matrix in the factor, a solve's right-hand sides) are included;
    ///     NumPy's own arrays are not.
    #[pyo3(signature = (dtype = "float64", nrhs = 1, settings = None, **kwargs))]
    fn memory_plan(
        &self,
        py: Python<'_>,
        dtype: &str,
        nrhs: usize,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        self.inner.memory_plan(py, dtype, nrhs, settings, kwargs)
    }

    /// Numeric factorization of ``data`` (the full CSC value array in the
    /// analyzed pattern's order). Numeric settings may be overridden per call.
    #[pyo3(signature = (data, settings = None, **kwargs))]

    /// Numeric factorization of new values on the analyzed pattern.
    ///
    /// Parameters
    /// ----------
    /// data : scipy.sparse matrix or ndarray
    ///     The matrix itself (any SciPy sparse matrix with the analyzed
    ///     pattern: its matrix is taken, sorted and summed like at analysis
    ///     time, and the pattern is checked entry by entry), or the CSC value
    ///     array of that matrix in the order of the analysis.
    /// settings : Settings, optional
    ///     Numeric settings for this factorization; the analysis settings by
    ///     default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    ///
    /// Returns
    /// -------
    /// factor handle
    ///     The numeric factor with the solve methods and diagnostics.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the pattern or the value count differs from the analysis.
    /// RuntimeError
    ///     If a pivot is numerically zero in exact mode.
    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Lu> {
        self.inner.factor(py, data, settings, kwargs)
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
    Analyzed::analyze(py, n, indptr, indices, data, settings, kwargs)
        .map(|inner| PyLuSymbolic { inner })
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
    Analyzed::<LuSymbolic>::analyze(py, n, indptr, indices, data, settings, kwargs)?
        .factor(py, data, None, None)
}

/// Analysis of a general pattern for the KLU path (block triangular form,
/// per-block AMD), from :func:`rslab.analyze` with ``path='klu'``. Factor any
/// value set on the same pattern with :meth:`factor`.
#[pyclass(name = "KluSymbolic", module = "rslab")]
pub struct PyKluSymbolic {
    inner: Analyzed<KluSymbolic>,
}

#[pymethods]
impl PyKluSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.inner.sym.n()
    }

    /// Number of diagonal blocks of the block triangular form.
    #[getter]
    fn n_blocks(&self) -> usize {
        self.inner.sym.n_blocks()
    }

    /// Dimension of the largest diagonal block.
    #[getter]
    fn max_block_size(&self) -> usize {
        self.inner.sym.max_block_size()
    }

    /// Block boundaries in the permuted order (``n_blocks + 1`` entries).
    #[getter]
    fn block_ptr(&self) -> Vec<usize> {
        self.inner.sym.block_ptr().to_vec()
    }

    /// Predicted factor entries.
    #[getter]
    fn factor_nnz(&self) -> usize {
        self.inner.sym.symbolic_factor_nnz()
    }

    /// Heap bytes this analysis holds on the Rust side, the stored pattern
    /// included.
    #[getter]
    fn heap_bytes(&self) -> u64 {
        self.inner.heap_bytes()
    }

    /// The settings the analysis adopted; the defaults for :meth:`factor`.
    #[getter]
    fn settings(&self) -> PyKluSettings {
        self.inner.settings.clone()
    }

    /// A-priori memory and work estimate for a factor in the given dtype.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use (``'float64'``, ``'complex128'``,
    ///     ``'float32'``, ``'complex64'``).
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``factor_bytes`` / ``factor_mb``, ``transient_peak_bytes`` /
    ///     ``transient_peak_mb`` (the peak during the factorization),
    ///     ``factor_flops``, ``critical_path_flops``.
    #[pyo3(signature = (dtype = "float64"))]
    fn estimate_memory(&self, py: Python<'_>, dtype: &str) -> PyResult<PyObject> {
        self.inner.estimate_memory(py, dtype)
    }

    /// The heap a factorization needs, predicted from the analysis before
    /// any numeric work: for a preflight check against the memory available
    /// and for scheduling factorizations side by side.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use.
    /// nrhs : int, default 1
    ///     Right-hand sides per solve.
    /// settings : KluSettings, optional
    ///     The settings :meth:`factor` will run with (``parallel`` decides the block buffers); the
    ///     analysis settings by default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``peak_bytes``, the heap peak from here through a factorization
    ///     and a solve (the number to compare against the memory available),
    ///     ``resident_bytes``, what stays held while the factor is kept, and
    ///     their parts: ``analysis_bytes``, ``analysis_growth_bytes``,
    ///     ``factor_bytes``, ``factor_peak_bytes`` (above what was live when
    ///     the factorization began), ``solve_bytes`` and ``kept_bytes``
    ///     (what the worker threads, and a refactorized factor's scratch,
    ///     keep for the next factorization), with ``threads``
    ///     and ``nrhs``. The copies this binding keeps (the pattern, the
    ///     matrix in the factor, a solve's right-hand sides) are included;
    ///     NumPy's own arrays are not.
    #[pyo3(signature = (dtype = "float64", nrhs = 1, settings = None, **kwargs))]
    fn memory_plan(
        &self,
        py: Python<'_>,
        dtype: &str,
        nrhs: usize,
        settings: Option<PyKluSettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        self.inner.memory_plan(py, dtype, nrhs, settings, kwargs)
    }

    /// Numeric factorization of ``data`` (the full CSC value array in the
    /// analyzed pattern's order). KLU settings may be overridden per call.
    #[pyo3(signature = (data, settings = None, **kwargs))]

    /// Numeric factorization of new values on the analyzed pattern.
    ///
    /// Parameters
    /// ----------
    /// data : scipy.sparse matrix or ndarray
    ///     The matrix itself (any SciPy sparse matrix with the analyzed
    ///     pattern: its matrix is taken, sorted and summed like at analysis
    ///     time, and the pattern is checked entry by entry), or the CSC value
    ///     array of that matrix in the order of the analysis.
    /// settings : Settings, optional
    ///     Numeric settings for this factorization; the analysis settings by
    ///     default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    ///
    /// Returns
    /// -------
    /// factor handle
    ///     The numeric factor with the solve methods and diagnostics.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the pattern or the value count differs from the analysis.
    /// RuntimeError
    ///     If a pivot is numerically zero in exact mode.
    fn factor(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        settings: Option<PyKluSettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Klu> {
        self.inner.factor(py, data, settings, kwargs)
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
    Analyzed::analyze(py, n, indptr, indices, data, settings, kwargs)
        .map(|inner| PyKluSymbolic { inner })
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
    Analyzed::<KluSymbolic>::analyze(py, n, indptr, indices, data, settings, kwargs)?
        .factor(py, data, None, None)
}

/// The automatic path's settings: the supernodal paths' (``settings`` and
/// the keywords, also returned as given), KLU's, and the routing between
/// them.
pub fn auto_settings(
    settings: Option<PySettings>,
    klu_settings: Option<PyKluSettings>,
    supernodal_flops_per_entry: Option<f64>,
    symmetry_tol: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(rslab::AutoSettings, PySettings)> {
    let supernodal = settings_from(settings, kwargs)?;
    let mut s = rslab::AutoSettings::default()
        .with_supernodal(supernodal.inner.clone())
        .with_klu(klu_settings_from(klu_settings, None)?.inner);
    if let Some(f) = supernodal_flops_per_entry {
        s = s.with_supernodal_flops_per_entry(f);
    }
    if let Some(t) = symmetry_tol {
        s = s.with_symmetry_tol(t);
    }
    Ok((s, supernodal))
}

/// A pattern analyzed for the automatic path, from :func:`rslab.analyze`
/// with ``path='auto'``: KLU's analysis predicts the work, and a pattern of
/// dense work is analyzed for the supernodal paths. Factor any value set on
/// the same pattern with :meth:`factor`; every factor shares the analysis.
#[pyclass(name = "AutoSymbolic", module = "rslab")]
pub struct PyAutoSymbolic {
    sym: std::sync::Arc<rslab::AutoSymbolic>,
    pattern: Pattern,
    settings: rslab::AutoSettings,
    supernodal: PySettings,
}

#[pymethods]
impl PyAutoSymbolic {
    /// Matrix dimension ``n``.
    #[getter]
    fn n(&self) -> usize {
        self.sym.n()
    }

    /// The path family the pattern's work selected: ``'klu'`` or
    /// ``'supernodal'`` (whose factors are ``'ldlt'`` or ``'lu'`` by the
    /// values).
    #[getter]
    fn route(&self) -> &'static str {
        match self.sym.is_supernodal() {
            Some(true) => "supernodal",
            _ => "klu",
        }
    }

    /// Heap bytes this analysis holds on the Rust side, the stored pattern
    /// included.
    #[getter]
    fn heap_bytes(&self) -> u64 {
        let p = &self.pattern;
        self.sym.heap_bytes() + csc_bytes(&p.col_ptr, &p.row_idx, &Vec::<u8>::new())
    }

    /// The heap a factorization needs, predicted from the analysis (see
    /// :meth:`LuSymbolic.memory_plan`). On the supernodal paths a factor
    /// keeps the ``L D L^T`` and the ``L U`` side by side once value sets of
    /// both kinds came, and the plan counts both.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, default 'float64'
    ///     The value type the factor will use.
    /// nrhs : int, default 1
    ///     Right-hand sides per solve.
    /// settings : Settings, optional
    ///     The supernodal settings the factorization will run with; the
    ///     analysis settings by default.
    /// **kwargs
    ///     Any settings keyword, overriding ``settings``.
    #[pyo3(signature = (dtype = "float64", nrhs = 1, settings = None, **kwargs))]
    fn memory_plan(
        &self,
        py: Python<'_>,
        dtype: &str,
        nrhs: usize,
        settings: Option<PySettings>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let st = settings_from(
            Some(settings.unwrap_or_else(|| self.supernodal.clone())),
            kwargs,
        )?;
        let opts = self.settings.clone().with_supernodal(st.inner);
        let (plan, vb) = with_dtype_name!(dtype, |T| (
            heavy(py, || self.sym.memory_plan::<T>(&opts, nrhs)),
            std::mem::size_of::<T>()
        ))?;
        let plan = plan.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("the pattern is not analyzed yet")
        })?;
        memory_plan_dict(py, plan, &self.pattern, vb)
    }

    /// Numeric factorization of ``data`` on the analyzed pattern.
    ///
    /// Parameters
    /// ----------
    /// data : scipy.sparse matrix or ndarray
    ///     The matrix itself (with the analyzed pattern) or its CSC value
    ///     array in the order of the analysis.
    ///
    /// Returns
    /// -------
    /// Auto
    ///     The factor, sharing this analysis.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the pattern or the value count differs from the analysis.
    /// RuntimeError
    ///     If the matrix is numerically singular.
    fn factor(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Auto> {
        let p = &self.pattern;
        let values = Values::of(py, data, &p.col_ptr, &p.row_idx, false)?;
        with_dtype!(values.data(), |T| {
            let a = p.general::<T>(values.read()?)?;
            let mut s = rslab::AutoSolver::<T>::new(self.sym.clone());
            s.values_mut().copy_from_slice(&a.values);
            heavy(py, || s.refactor(&self.settings)).map_err(map_err)?;
            Ok(Auto {
                inner: T::auto(Pair::new(s, a)),
                settings: self.settings.clone(),
            })
        })
    }

    fn __repr__(&self) -> String {
        format!("AutoSymbolic(n={}, route='{}')", self.n(), self.route())
    }
}

/// Analyze the CSC pattern ``(n, indptr, indices, data)`` for the automatic
/// path, on the values ``data``.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, klu_settings = None,
                    supernodal_flops_per_entry = None, symmetry_tol = None, **kwargs))]
#[allow(clippy::too_many_arguments)]
pub fn analyze_auto(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PySettings>,
    klu_settings: Option<PyKluSettings>,
    supernodal_flops_per_entry: Option<f64>,
    symmetry_tol: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyAutoSymbolic> {
    let (st, supernodal) = auto_settings(
        settings,
        klu_settings,
        supernodal_flops_per_entry,
        symmetry_tol,
        kwargs,
    )?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |T| {
        let a = pattern.general::<T>(vector(data, "data")?)?;
        let sym = heavy(py, || {
            rslab::AutoSymbolic::analyze(rslab::CscLayout::of(&a)?, &a.values, &st)
        })
        .map_err(map_err)?;
        Ok(PyAutoSymbolic {
            sym: std::sync::Arc::new(sym),
            pattern: pattern.clone(),
            settings: st.clone(),
            supernodal: supernodal.clone(),
        })
    })
}

/// One-shot factorization of ``(n, indptr, indices, data)`` on the automatic
/// path.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, settings = None, klu_settings = None,
                    supernodal_flops_per_entry = None, symmetry_tol = None, **kwargs))]
#[allow(clippy::too_many_arguments)]
pub fn auto_factor(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    settings: Option<PySettings>,
    klu_settings: Option<PyKluSettings>,
    supernodal_flops_per_entry: Option<f64>,
    symmetry_tol: Option<f64>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Auto> {
    let (st, _) = auto_settings(
        settings,
        klu_settings,
        supernodal_flops_per_entry,
        symmetry_tol,
        kwargs,
    )?;
    let pattern = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |T| {
        let a = pattern.general::<T>(vector(data, "data")?)?;
        let s = heavy(py, || rslab::AutoSolver::<T>::factor(&a, &st)).map_err(map_err)?;
        Ok(Auto {
            inner: T::auto(Pair::new(s, a)),
            settings: st.clone(),
        })
    })
}
