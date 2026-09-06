//! Shared plumbing of the bindings: error mapping, the scalar-field dispatch,
//! NumPy <-> Rust buffer conversion, and the dict renderings of the core's
//! report types.

use num_complex::Complex;
use numpy::{
    Element, IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rslab::{Diagnostics, MemoryEstimate, RslabError, Scalar};

pub type C64 = Complex<f64>;
pub type C32 = Complex<f32>;

pub fn map_err(e: RslabError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

pub fn unsupported_dtype() -> PyErr {
    PyValueError::new_err("unsupported dtype: expected float64, float32, complex128, or complex64")
}

/// The four scalar fields the core supports, by NumPy dtype name.
pub fn scalar_bytes(dtype: &str) -> PyResult<usize> {
    Ok(match dtype {
        "float64" | "f8" | "d" => 8,
        "float32" | "f4" | "f" => 4,
        "complex128" | "c16" | "D" => 16,
        "complex64" | "c8" | "F" => 8,
        other => {
            return Err(PyValueError::new_err(format!(
                "dtype must be 'float64', 'float32', 'complex128' or 'complex64', got '{other}'"
            )))
        }
    })
}

/// Run `$body` with `$T` bound to the scalar type of the 1-D NumPy array
/// `$data` and `$d` to its contents (copied into a Rust-owned `Vec`, so the
/// body may release the GIL).
macro_rules! with_dtype {
    ($data:expr, |$d:ident : $T:ident| $body:block) => {{
        let data: &::pyo3::Bound<'_, ::pyo3::PyAny> = $data;
        if let Ok(arr) = data.extract::<::numpy::PyReadonlyArray1<f64>>() {
            type $T = f64;
            let $d: Vec<$T> = arr.as_slice()?.to_vec();
            $body
        } else if let Ok(arr) = data.extract::<::numpy::PyReadonlyArray1<$crate::common::C64>>() {
            type $T = $crate::common::C64;
            let $d: Vec<$T> = arr.as_slice()?.to_vec();
            $body
        } else if let Ok(arr) = data.extract::<::numpy::PyReadonlyArray1<f32>>() {
            type $T = f32;
            let $d: Vec<$T> = arr.as_slice()?.to_vec();
            $body
        } else if let Ok(arr) = data.extract::<::numpy::PyReadonlyArray1<$crate::common::C32>>() {
            type $T = $crate::common::C32;
            let $d: Vec<$T> = arr.as_slice()?.to_vec();
            $body
        } else {
            Err($crate::common::unsupported_dtype())
        }
    }};
}
pub(crate) use with_dtype;

/// A sparsity pattern in CSC form (what a symbolic analysis is a function of).
#[derive(Clone)]
pub struct Pattern {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
}

impl Pattern {
    pub fn new(n: usize, indptr: &[i64], indices: &[i64]) -> PyResult<Self> {
        if indptr.len() != n + 1 {
            return Err(PyValueError::new_err(format!(
                "indptr has length {}, expected n + 1 = {}",
                indptr.len(),
                n + 1
            )));
        }
        let conv = |v: &[i64]| -> PyResult<Vec<usize>> {
            v.iter()
                .map(|&x| {
                    usize::try_from(x)
                        .map_err(|_| PyValueError::new_err("negative index in the CSC pattern"))
                })
                .collect()
        };
        Ok(Self {
            n,
            col_ptr: conv(indptr)?,
            row_idx: conv(indices)?,
        })
    }

    pub fn from_py(
        n: usize,
        indptr: &PyReadonlyArray1<i64>,
        indices: &PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        Self::new(n, indptr.as_slice()?, indices.as_slice()?)
    }

    pub fn nnz(&self) -> usize {
        self.row_idx.len()
    }

    /// Check that a value array fits this pattern.
    pub fn check_values(&self, len: usize) -> PyResult<()> {
        if len != self.nnz() {
            return Err(PyValueError::new_err(format!(
                "data length {len} does not match the pattern nnz {}",
                self.nnz()
            )));
        }
        Ok(())
    }

    pub fn csc<T: Scalar>(&self, values: Vec<T>) -> PyResult<rslab::CscMatrix<T>> {
        self.check_values(values.len())?;
        let m = rslab::CscMatrix {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values,
        };
        m.validate().map_err(map_err)?;
        Ok(m)
    }

    pub fn general<T: Scalar>(&self, values: Vec<T>) -> PyResult<rslab::GeneralCsc<T>> {
        self.check_values(values.len())?;
        let m = rslab::GeneralCsc {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values,
        };
        m.validate().map_err(map_err)?;
        Ok(m)
    }
}

/// A 1-D right-hand side of the factor's scalar type, copied out.
pub fn vector<T: Element + Clone>(b: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<T>> {
    let arr: PyReadonlyArray1<T> = b.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} dtype does not match the factor dtype"))
    })?;
    Ok(arr.as_slice()?.to_vec())
}

/// A 2-D `n x nrhs` block of the factor's scalar type as `(n, nrhs, column-major data)`.
pub fn block<T: Element + Copy + Default>(
    b: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<(usize, usize, Vec<T>)> {
    let arr: PyReadonlyArray2<T> = b.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} dtype does not match the factor dtype"))
    })?;
    let shape = arr.shape();
    let (n, nrhs) = (shape[0], shape[1]);
    let rm = arr.as_slice()?;
    let mut cm = vec![T::default(); n * nrhs];
    for i in 0..n {
        for c in 0..nrhs {
            cm[c * n + i] = rm[i * nrhs + c];
        }
    }
    Ok((n, nrhs, cm))
}

/// A 2-D `n x nrhs` block in its native row-major layout, `(n, nrhs, data)`.
pub fn block_row_major<T: Element + Clone>(
    b: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<(usize, usize, Vec<T>)> {
    let arr: PyReadonlyArray2<T> = b.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} dtype does not match the factor dtype"))
    })?;
    let shape = arr.shape();
    Ok((shape[0], shape[1], arr.as_slice()?.to_vec()))
}

/// A row-major `n x nrhs` buffer as a NumPy array.
pub fn array2_row_major<T: Element>(
    py: Python<'_>,
    rm: Vec<T>,
    n: usize,
    nrhs: usize,
) -> PyResult<PyObject> {
    Ok(rm
        .into_pyarray_bound(py)
        .reshape([n, nrhs])?
        .into_any()
        .unbind())
}

pub fn array1<T: Element>(py: Python<'_>, v: Vec<T>) -> PyObject {
    v.into_pyarray_bound(py).into_any().unbind()
}

/// A column-major `n x nrhs` block as a row-major NumPy array.
pub fn array2<T: Element + Copy + Default>(
    py: Python<'_>,
    cm: &[T],
    n: usize,
    nrhs: usize,
) -> PyResult<PyObject> {
    let mut out = vec![T::default(); n * nrhs];
    for c in 0..nrhs {
        for i in 0..n {
            out[i * nrhs + c] = cm[c * n + i];
        }
    }
    Ok(out
        .into_pyarray_bound(py)
        .reshape([n, nrhs])?
        .into_any()
        .unbind())
}

pub fn memory_estimate_dict(py: Python<'_>, e: &MemoryEstimate) -> PyResult<PyObject> {
    let est = PyDict::new_bound(py);
    est.set_item("value_bytes", e.value_bytes)?;
    est.set_item("factor_nnz", e.factor_nnz)?;
    est.set_item("factor_bytes", e.factor_bytes)?;
    est.set_item("factor_mb", e.factor_mb())?;
    est.set_item("transient_peak_bytes", e.transient_peak_bytes)?;
    est.set_item("transient_peak_mb", e.transient_peak_mb())?;
    est.set_item("factor_flops", e.factor_flops)?;
    est.set_item("critical_path_flops", e.critical_path_flops)?;
    Ok(est.into_any().unbind())
}

pub fn diagnostics_dict(py: Python<'_>, d: &Diagnostics) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("n", d.n)?;
    out.set_item("nnz_a", d.nnz_a)?;
    out.set_item("factor_nnz", d.factor_nnz)?;
    out.set_item("fill_ratio", d.fill_ratio())?;
    out.set_item("threads", d.threads)?;
    let stages = pyo3::types::PyList::empty_bound(py);
    for s in &d.stages {
        let st = PyDict::new_bound(py);
        st.set_item("name", s.name)?;
        st.set_item("wall_ms", s.wall_ms)?;
        st.set_item("flops", s.flops)?;
        st.set_item("bytes", s.bytes)?;
        stages.append(st)?;
    }
    out.set_item("stages", stages)?;
    out.set_item("total_ms", d.total_ms())?;
    let dec = PyDict::new_bound(py);
    dec.set_item("ordering_requested", &d.decisions.ordering_requested)?;
    dec.set_item("ordering_used", &d.decisions.ordering_used)?;
    dec.set_item("preprocess", &d.decisions.preprocess)?;
    dec.set_item("amalgamation", &d.decisions.amalgamation)?;
    dec.set_item("scaling", &d.decisions.scaling)?;
    dec.set_item("method", &d.decisions.method)?;
    dec.set_item("n_supernodes", d.decisions.n_supernodes)?;
    dec.set_item("max_front", d.decisions.max_front)?;
    dec.set_item("tree_levels", d.decisions.tree_levels)?;
    dec.set_item("btf_blocks", d.decisions.btf_blocks)?;
    out.set_item("decisions", dec)?;
    let num = PyDict::new_bound(py);
    num.set_item("perturbed", d.numeric.perturbed)?;
    num.set_item("two_by_two", d.numeric.two_by_two)?;
    num.set_item("inertia", d.numeric.inertia)?;
    out.set_item("numeric", num)?;
    let sv = PyDict::new_bound(py);
    sv.set_item("calls", d.solves.calls)?;
    sv.set_item("rhs", d.solves.rhs)?;
    sv.set_item("wall_ms", d.solves.wall_ms)?;
    sv.set_item("refine_steps", d.solves.refine_steps)?;
    out.set_item("solves", sv)?;
    out.set_item("warnings", d.warnings.clone())?;
    let r = d.rates();
    let rates = PyDict::new_bound(py);
    rates.set_item("analyze_mdof_s", r.analyze_mdof_s)?;
    rates.set_item("factor_mdof_s", r.factor_mdof_s)?;
    rates.set_item("factor_gflops", r.factor_gflops)?;
    rates.set_item("factor_mnnz_s", r.factor_mnnz_s)?;
    rates.set_item("total_mdof_s", r.total_mdof_s)?;
    rates.set_item("solve_mdof_s", r.solve_mdof_s)?;
    out.set_item("rates", rates)?;
    match &d.estimate {
        Some(e) => out.set_item("estimate", memory_estimate_dict(py, e)?)?,
        None => out.set_item("estimate", py.None())?,
    }
    out.set_item("summary", d.summary())?;
    Ok(out.into_any().unbind())
}

// GMRES restart / basis-memory policy (issue #12).
//
// The Arnoldi basis is allocated up front, so its size is fixed by `restart`,
// not by how few iterations actually run: block GMRES holds one basis of
// `n * nrhs * (restart+1)` scalars, flexible single-RHS GMRES the `V` + `Z`
// pair, `2 * n * (restart+1)`. With a fixed `restart=80` a large `n*nrhs`
// allocates silently (`n=100k, nrhs=10, complex128` is about 13 GB). When the
// caller does not pin `restart`, the binding caps it so the basis stays under
// `GMRES_BASIS_BUDGET_BYTES`, clamped to a still-useful `[MIN, MAX]`. An
// explicit `restart=` always wins, even past the budget.
const GMRES_BASIS_BUDGET_BYTES: usize = 1 << 30;
const GMRES_RESTART_MIN: usize = 20;
const GMRES_RESTART_MAX: usize = 80;

pub fn adaptive_restart(n: usize, columns: usize, scalar_bytes: usize, bases: usize) -> usize {
    let per_layer = n
        .saturating_mul(columns)
        .saturating_mul(scalar_bytes)
        .saturating_mul(bases);
    if per_layer == 0 {
        return GMRES_RESTART_MAX;
    }
    let cap = (GMRES_BASIS_BUDGET_BYTES / per_layer).saturating_sub(1);
    cap.clamp(GMRES_RESTART_MIN, GMRES_RESTART_MAX)
}
