//! The Krylov solvers: the shared implementations behind the handle methods
//! (factor as preconditioner) and the unpreconditioned module functions.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rslab::{
    cocg as cocg_core, cocr as cocr_core, gmres as gmres_core, gmres_block as gmres_block_core,
    gmres_recycled as gmres_recycled_core, BlockKrylovResult, KrylovResult, LinearOperator,
    NoPreconditioner, Preconditioner,
};

use crate::common::{
    adaptive_restart, array1, array2, block, map_err, vector, with_dtype, Pattern,
};
use crate::factor::{Field, PyRecycle};

/// An explicit CSC operator handed in from Python: `(n, indptr, indices, data)`.
pub type Operator<'py> = (
    usize,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    Bound<'py, PyAny>,
);

/// Outcome of an iterative solve.
///
/// Attributes
/// ----------
/// x : ndarray
///     The iterate (``n`` or ``n x nrhs``).
/// converged : bool
///     Whether the residual target was met.
/// iters : int
///     Iterations (matrix-vector products) run.
/// final_res : float or ndarray
///     Final relative residual (one value per column for block solves).
/// stop : str
///     ``'converged'``, ``'max_iter'``, ``'breakdown'`` or ``'stalled'``.
///
/// Unpacks as the 5-tuple ``(x, converged, iters, final_res, stop)``.
#[pyclass(name = "KrylovResult", module = "rslab")]
pub struct PyKrylovResult {
    #[pyo3(get)]
    pub x: PyObject,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iters: usize,
    #[pyo3(get)]
    pub final_res: PyObject,
    #[pyo3(get)]
    pub stop: String,
}

#[pymethods]
impl PyKrylovResult {
    fn __len__(&self) -> usize {
        5
    }

    fn __getitem__(&self, py: Python<'_>, i: isize) -> PyResult<PyObject> {
        let i = if i < 0 { i + 5 } else { i };
        Ok(match i {
            0 => self.x.clone_ref(py),
            1 => self.converged.into_py(py),
            2 => self.iters.into_py(py),
            3 => self.final_res.clone_ref(py),
            4 => self.stop.clone().into_py(py),
            _ => {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "KrylovResult index out of range",
                ))
            }
        })
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let t = pyo3::types::PyTuple::new_bound(
            py,
            [
                self.x.clone_ref(py),
                self.converged.into_py(py),
                self.iters.into_py(py),
                self.final_res.clone_ref(py),
                self.stop.clone().into_py(py),
            ],
        );
        Ok(t.as_any().iter()?.into_any().unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "KrylovResult(converged={}, iters={}, final_res={}, stop='{}')",
            if self.converged { "True" } else { "False" },
            self.iters,
            self.final_res.bind(py).repr()?,
            self.stop
        ))
    }
}

fn single<T: Field>(py: Python<'_>, r: KrylovResult<T>) -> PyResult<PyObject> {
    Ok(Py::new(
        py,
        PyKrylovResult {
            x: array1(py, r.x),
            converged: r.converged,
            iters: r.iters,
            final_res: r.final_res.into_py(py),
            stop: r.stop.as_str().to_string(),
        },
    )?
    .into_any())
}

fn blocked<T: Field>(
    py: Python<'_>,
    r: BlockKrylovResult<T>,
    n: usize,
    nrhs: usize,
) -> PyResult<PyObject> {
    Ok(Py::new(
        py,
        PyKrylovResult {
            x: array2(py, &r.x, n, nrhs)?,
            converged: r.converged,
            iters: r.iters,
            final_res: array1(py, r.final_res),
            stop: r.stop.as_str().to_string(),
        },
    )?
    .into_any())
}

#[allow(clippy::too_many_arguments)]
pub fn gmres<T: Field, M: Preconditioner<T> + Sync + ?Sized>(
    py: Python<'_>,
    op: &(dyn LinearOperator<T> + Sync),
    pc: &M,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
    restart: Option<usize>,
    x0: Option<&Bound<'_, PyAny>>,
    recycle: Option<&Bound<'_, PyRecycle>>,
) -> PyResult<PyObject> {
    let rhs = vector::<T>(b, "rhs")?;
    let restart =
        restart.unwrap_or_else(|| adaptive_restart(rhs.len(), 1, std::mem::size_of::<T>(), 2));
    let x0v = x0.map(|g| vector::<T>(g, "x0")).transpose()?;
    let r = match recycle {
        None => py
            .allow_threads(|| gmres_core(op, &rhs, pc, tol, maxit, restart, x0v.as_deref()))
            .map_err(map_err)?,
        Some(rc) => {
            let rc = rc.borrow();
            let mut guard = rc.inner.borrow_mut();
            let handle = T::recycle_mut(&mut guard).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "recycle dtype does not match the factor dtype",
                )
            })?;
            py.allow_threads(|| {
                gmres_recycled_core(op, &rhs, pc, tol, maxit, restart, x0v.as_deref(), handle)
            })
            .map_err(map_err)?
        }
    };
    single(py, r)
}

#[allow(clippy::too_many_arguments)]
pub fn gmres_block<T: Field, M: Preconditioner<T> + Sync + ?Sized>(
    py: Python<'_>,
    op: &(dyn LinearOperator<T> + Sync),
    pc: &M,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
    restart: Option<usize>,
    x0: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let (n, nrhs, cm) = block::<T>(b, "rhs")?;
    let restart = restart.unwrap_or_else(|| adaptive_restart(n, nrhs, std::mem::size_of::<T>(), 1));
    let x0v = match x0 {
        Some(g) => {
            let (gn, gs, gcm) = block::<T>(g, "x0")?;
            if gn != n || gs != nrhs {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "x0 shape must match B (n x nrhs)",
                ));
            }
            Some(gcm)
        }
        None => None,
    };
    let r = py
        .allow_threads(|| gmres_block_core(op, &cm, nrhs, pc, tol, maxit, restart, x0v.as_deref()))
        .map_err(map_err)?;
    blocked(py, r, n, nrhs)
}

pub fn cocg<T: Field, M: Preconditioner<T> + Sync + ?Sized>(
    py: Python<'_>,
    op: &(dyn LinearOperator<T> + Sync),
    pc: &M,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
) -> PyResult<PyObject> {
    let rhs = vector::<T>(b, "rhs")?;
    let r = py
        .allow_threads(|| cocg_core(op, &rhs, pc, tol, maxit))
        .map_err(map_err)?;
    single(py, r)
}

pub fn cocr<T: Field, M: Preconditioner<T> + Sync + ?Sized>(
    py: Python<'_>,
    op: &(dyn LinearOperator<T> + Sync),
    pc: &M,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
) -> PyResult<PyObject> {
    let rhs = vector::<T>(b, "rhs")?;
    let r = py
        .allow_threads(|| cocr_core(op, &rhs, pc, tol, maxit))
        .map_err(map_err)?;
    single(py, r)
}

// ---------------------------------------------------------------------------
// Unpreconditioned module functions (the Python wrappers route the
// preconditioned case through the handle methods).
// ---------------------------------------------------------------------------

/// Unpreconditioned flexible GMRES on the CSC matrix ``(n, indptr, indices, data)``.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, b, tol = 1e-8, maxit = 400, restart = None, x0 = None))]
#[allow(clippy::too_many_arguments)]
pub fn gmres_plain(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
    restart: Option<usize>,
    x0: Option<Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let pat = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pat.general::<T>(d)?;
        gmres::<T, _>(
            py,
            &a,
            &NoPreconditioner,
            b,
            tol,
            maxit,
            restart,
            x0.as_ref(),
            None,
        )
    })
}

/// Unpreconditioned block GMRES on the CSC matrix ``(n, indptr, indices, data)``.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, b, tol = 1e-8, maxit = 400, restart = None, x0 = None))]
#[allow(clippy::too_many_arguments)]
pub fn gmres_block_plain(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
    restart: Option<usize>,
    x0: Option<Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let pat = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pat.general::<T>(d)?;
        gmres_block::<T, _>(
            py,
            &a,
            &NoPreconditioner,
            b,
            tol,
            maxit,
            restart,
            x0.as_ref(),
        )
    })
}

/// Unpreconditioned COCG on the CSC matrix ``(n, indptr, indices, data)``.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, b, tol = 1e-8, maxit = 400))]
#[allow(clippy::too_many_arguments)]
pub fn cocg_plain(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
) -> PyResult<PyObject> {
    let pat = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pat.general::<T>(d)?;
        cocg::<T, _>(py, &a, &NoPreconditioner, b, tol, maxit)
    })
}

/// Unpreconditioned COCR on the CSC matrix ``(n, indptr, indices, data)``.
#[pyfunction]
#[pyo3(signature = (n, indptr, indices, data, b, tol = 1e-8, maxit = 400))]
#[allow(clippy::too_many_arguments)]
pub fn cocr_plain(
    py: Python<'_>,
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    tol: f64,
    maxit: usize,
) -> PyResult<PyObject> {
    let pat = Pattern::from_py(n, &indptr, &indices)?;
    with_dtype!(data, |d: T| {
        let a = pat.general::<T>(d)?;
        cocr::<T, _>(py, &a, &NoPreconditioner, b, tol, maxit)
    })
}
