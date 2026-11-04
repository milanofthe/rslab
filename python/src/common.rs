//! Shared plumbing of the bindings: error mapping, the scalar-field dispatch,
//! NumPy <-> Rust buffer conversion, and the dict renderings of the core's
//! report types.

use crate::constants::{C32_BYTES, C64_BYTES, F32_BYTES, F64_BYTES};
use num_complex::Complex;
use numpy::{Element, IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rslab::{Diagnostics, MemoryEstimate, RslabError, Scalar};

pub type C64 = Complex<f64>;
pub type C32 = Complex<f32>;

/// Run an analysis or factorization without the GIL, then hand the memory the
/// allocator cached during it back to the system. mimalloc keeps freed pages in
/// per-thread heaps, which is what makes the per-product buffers of the dense
/// kernels cheap inside one call; the scoped pool's workers exit afterwards and
/// leave those heaps holding up to the factor size again until forced.
pub fn heavy<R: pyo3::marker::Ungil>(
    py: Python<'_>,
    f: impl pyo3::marker::Ungil + FnOnce() -> R,
) -> R {
    let out = py.allow_threads(f);
    collect();
    out
}

/// Last field of the objects that own a factor: dropped after the factor
/// itself, it hands the freed pages back as [`heavy`] does.
#[derive(Default)]
pub struct Release;

impl Drop for Release {
    fn drop(&mut self) {
        collect();
    }
}

fn collect() {
    // SAFETY: a plain allocator call without preconditions.
    unsafe { libmimalloc_sys::mi_collect(true) };
}

pub fn map_err(e: RslabError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

pub fn unsupported_dtype() -> PyErr {
    PyValueError::new_err("unsupported dtype: expected float64, float32, complex128, or complex64")
}

/// The four scalar fields the core supports, by NumPy dtype name: the size
/// in bytes and whether the field is complex.
pub fn scalar_kind(dtype: &str) -> PyResult<(usize, bool)> {
    Ok(match dtype {
        "float64" | "f8" | "d" => (F64_BYTES, false),
        "float32" | "f4" | "f" => (F32_BYTES, false),
        "complex128" | "c16" | "D" => (C64_BYTES, true),
        "complex64" | "c8" | "F" => (C32_BYTES, true),
        other => {
            return Err(PyValueError::new_err(format!(
                "dtype must be 'float64', 'float32', 'complex128' or 'complex64', got '{other}'"
            )))
        }
    })
}

/// Evaluate `$body` with `$T` bound to the scalar type named by the dtype
/// string `$dtype`; a `PyResult` of the body's value.
macro_rules! with_dtype_name {
    ($dtype:expr, |$T:ident| $body:expr) => {{
        match $crate::common::scalar_kind($dtype) {
            Ok(($crate::constants::F64_BYTES, false)) => {
                type $T = f64;
                Ok($body)
            }
            Ok(($crate::constants::F32_BYTES, _)) => {
                type $T = f32;
                Ok($body)
            }
            Ok(($crate::constants::C64_BYTES, _)) => {
                type $T = $crate::common::C64;
                Ok($body)
            }
            Ok(_) => {
                type $T = $crate::common::C32;
                Ok($body)
            }
            Err(e) => Err(e),
        }
    }};
}
pub(crate) use with_dtype_name;

/// Run `$body` with `$T` bound to the scalar type of the 1-D NumPy array
/// `$data` (the array itself is not read).
macro_rules! with_dtype {
    ($data:expr, |$T:ident| $body:block) => {{
        let data: &::pyo3::Bound<'_, ::pyo3::PyAny> = $data;
        if data.extract::<::numpy::PyReadonlyArray1<f64>>().is_ok() {
            type $T = f64;
            $body
        } else if data
            .extract::<::numpy::PyReadonlyArray1<$crate::common::C64>>()
            .is_ok()
        {
            type $T = $crate::common::C64;
            $body
        } else if data.extract::<::numpy::PyReadonlyArray1<f32>>().is_ok() {
            type $T = f32;
            $body
        } else if data
            .extract::<::numpy::PyReadonlyArray1<$crate::common::C32>>()
            .is_ok()
        {
            type $T = $crate::common::C32;
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

/// Heap bytes of a CSC matrix's three arrays (their capacities).
pub fn csc_bytes<T>(col_ptr: &Vec<usize>, row_idx: &Vec<usize>, values: &Vec<T>) -> u64 {
    ((col_ptr.capacity() + row_idx.capacity()) * std::mem::size_of::<usize>()
        + values.capacity() * std::mem::size_of::<T>()) as u64
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

    /// The matrix of `values` on this pattern, unchecked: the core validates
    /// what it analyzes or factors.
    pub fn csc<T: Scalar>(&self, values: Vec<T>) -> PyResult<rslab::CscMatrix<T>> {
        self.check_values(values.len())?;
        Ok(rslab::CscMatrix {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values,
        })
    }

    /// [`csc`](Self::csc) for a general matrix.
    pub fn general<T: Scalar>(&self, values: Vec<T>) -> PyResult<rslab::GeneralCsc<T>> {
        self.check_values(values.len())?;
        Ok(rslab::GeneralCsc {
            n: self.n,
            col_ptr: self.col_ptr.clone(),
            row_idx: self.row_idx.clone(),
            values,
        })
    }

    /// A Krylov operator of `values` on this pattern: validated here, since
    /// the core applies an operator without a check of its own.
    pub fn operator<T: Scalar>(&self, values: Vec<T>) -> PyResult<rslab::GeneralCsc<T>> {
        let m = self.general(values)?;
        m.validate().map_err(map_err)?;
        Ok(m)
    }
}

/// The values a factorization reads on the analyzed pattern `(col_ptr,
/// row_idx)`: a value array in the pattern's order, or a SciPy matrix with
/// the pattern, sorted and summed like at analysis time and its lower
/// triangle taken (`lower`) or the whole.
pub struct Values<'py, 'p> {
    data: Bound<'py, PyAny>,
    /// `(indptr, indices)` of a matrix; none for a value array.
    index: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    col_ptr: &'p [usize],
    row_idx: &'p [usize],
    lower: bool,
}

impl<'py, 'p> Values<'py, 'p> {
    pub fn of(
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        col_ptr: &'p [usize],
        row_idx: &'p [usize],
        lower: bool,
    ) -> PyResult<Self> {
        let mut values = Self {
            data: obj.clone(),
            index: None,
            col_ptr,
            row_idx,
            lower,
        };
        if !obj.hasattr("indptr")? {
            return Ok(values);
        }
        let m = py
            .import_bound("rslab")?
            .getattr("_full_csc")?
            .call1((obj,))?;
        let shape: (usize, usize) = m.getattr("shape")?.extract()?;
        let n = col_ptr.len() - 1;
        if shape != (n, n) {
            return Err(PyValueError::new_err(format!(
                "matrix is {}x{}, the analyzed pattern is {n}x{n}",
                shape.0, shape.1
            )));
        }
        values.data = m.getattr("data")?;
        values.index = Some((m.getattr("indptr")?, m.getattr("indices")?));
        Ok(values)
    }

    /// The value array, whose dtype is the factorization's.
    pub fn data(&self) -> &Bound<'py, PyAny> {
        &self.data
    }

    /// The values in the pattern's order: the value array (its length the
    /// caller's to check), or the matrix's entries column by column, each
    /// column checked against the pattern as it is copied (one pass).
    pub fn read<T: Element + Copy>(&self) -> PyResult<Vec<T>> {
        let Some((indptr, indices)) = &self.index else {
            return vector::<T>(&self.data, "data");
        };
        let data: PyReadonlyArray1<T> = self
            .data
            .extract()
            .map_err(|_| PyValueError::new_err("data dtype does not match the factor dtype"))?;
        let data = data.as_slice()?;
        let (col_ptr, row_idx, lower) = (self.col_ptr, self.row_idx, self.lower);
        macro_rules! try_index {
            ($I:ty) => {
                if let (Ok(ip), Ok(ix)) = (
                    indptr.extract::<PyReadonlyArray1<$I>>(),
                    indices.extract::<PyReadonlyArray1<$I>>(),
                ) {
                    let (ip, ix) = (ip.as_slice()?, ix.as_slice()?);
                    return gather(ip, ix, data, col_ptr, row_idx, lower).ok_or_else(|| {
                        PyValueError::new_err(
                            "the matrix has a different sparsity pattern than the one that was analyzed (same positions, any values, are required; re-analyze for a new pattern)",
                        )
                    });
                }
            };
        }
        try_index!(i32);
        try_index!(i64);
        Err(PyValueError::new_err(
            "indptr and indices must both be int32 or both int64",
        ))
    }
}

/// The values of the canonical CSC matrix `(ip, ix, data)` (each column's
/// rows from the diagonal down with `lower`) in the order of the pattern
/// `(col_ptr, row_idx)`, or none if its pattern is another.
fn gather<I: CscIndex, T: Copy>(
    ip: &[I],
    ix: &[I],
    data: &[T],
    col_ptr: &[usize],
    row_idx: &[usize],
    lower: bool,
) -> Option<Vec<T>> {
    if ip.len() != col_ptr.len() || ix.len() != data.len() {
        return None;
    }
    let mut out = Vec::with_capacity(row_idx.len());
    for (j, (p, want)) in ip.windows(2).zip(col_ptr.windows(2)).enumerate() {
        let (start, end) = (p[0].to_usize()?, p[1].to_usize()?);
        let col = ix.get(start..end)?;
        let skip = if lower {
            let diag = I::from_usize(j)?;
            col.partition_point(|&r| r < diag)
        } else {
            0
        };
        let want = &row_idx[want[0]..want[1]];
        if col.len() - skip != want.len()
            || col[skip..]
                .iter()
                .zip(want)
                .any(|(&r, &w)| r.to_usize() != Some(w))
        {
            return None;
        }
        out.extend_from_slice(&data[start + skip..end]);
    }
    Some(out)
}

/// A 1-D right-hand side of the factor's scalar type, copied out (strided
/// views included).
pub fn vector<T: Element + Clone>(b: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<T>> {
    let arr: PyReadonlyArray1<T> = b.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} dtype does not match the factor dtype"))
    })?;
    Ok(match arr.as_slice() {
        Ok(s) => s.to_vec(),
        Err(_) => arr.as_array().iter().cloned().collect(),
    })
}

/// A 2-D `n x nrhs` block of the factor's scalar type as `(n, nrhs, column-major data)`.
pub fn block<T: Element + Copy>(
    b: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<(usize, usize, Vec<T>)> {
    let arr: PyReadonlyArray2<T> = b.extract().map_err(|_| {
        PyValueError::new_err(format!("{what} dtype does not match the factor dtype"))
    })?;
    let view = arr.as_array();
    let (n, nrhs) = view.dim();
    // Iterating the transposed view walks the block column by column in
    // logical order, whatever memory order NumPy hands over.
    Ok((n, nrhs, view.t().iter().copied().collect()))
}

pub fn array1<T: Element>(py: Python<'_>, v: Vec<T>) -> PyObject {
    v.into_pyarray_bound(py).into_any().unbind()
}

/// One value as a NumPy scalar of its dtype.
pub fn scalar<T: Element>(py: Python<'_>, v: T) -> PyResult<PyObject> {
    Ok(array1(py, vec![v]).bind(py).get_item(0)?.unbind())
}

/// Indices (a permutation, block boundaries) as an int64 NumPy array.
pub fn index_array(py: Python<'_>, v: &[usize]) -> PyObject {
    array1(py, v.iter().map(|&i| i as i64).collect::<Vec<i64>>())
}

/// A square compressed-column matrix as a SciPy `csc_matrix` (int64
/// indices).
pub fn csc_matrix<T: Element + Clone>(
    py: Python<'_>,
    m: &rslab::GeneralCsc<T>,
) -> PyResult<PyObject> {
    let parts = (
        array1(py, m.values.clone()),
        index_array(py, &m.row_idx),
        index_array(py, &m.col_ptr),
    );
    let csc = py.import_bound("scipy.sparse")?.getattr("csc_matrix")?;
    Ok(csc.call1((parts, (m.n, m.n)))?.unbind())
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

/// A memory plan as a dict, with the copies this binding keeps on top of
/// the core's: the analyzed pattern, the matrix a factor handle holds for
/// residuals and refinement (copied during the factorization), and the
/// right-hand sides a solve copies in.
pub fn memory_plan_dict(
    py: Python<'_>,
    mut p: rslab::MemoryPlan,
    pattern: &Pattern,
    value_bytes: usize,
) -> PyResult<PyObject> {
    let (n, nnz, vb) = (
        pattern.n as u64,
        pattern.row_idx.len() as u64,
        value_bytes as u64,
    );
    let idx = crate::constants::INDEX_BYTES;
    let matrix = idx * (n + 1) + (idx + vb) * nnz;
    p.analysis_bytes += csc_bytes(&pattern.col_ptr, &pattern.row_idx, &Vec::<u8>::new());
    p.factor_bytes += matrix;
    p.factor_peak_bytes += matrix;
    p.solve_bytes += n * p.nrhs as u64 * vb;
    let d = PyDict::new_bound(py);
    d.set_item("threads", p.threads)?;
    d.set_item("nrhs", p.nrhs)?;
    d.set_item("peak_bytes", p.peak_bytes())?;
    d.set_item("resident_bytes", p.resident_bytes())?;
    d.set_item("analysis_bytes", p.analysis_bytes)?;
    d.set_item("analysis_growth_bytes", p.analysis_growth_bytes)?;
    d.set_item("factor_bytes", p.factor_bytes)?;
    d.set_item("factor_peak_bytes", p.factor_peak_bytes)?;
    d.set_item("solve_bytes", p.solve_bytes)?;
    d.set_item("kept_bytes", p.kept_bytes)?;
    Ok(d.into_any().unbind())
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
    num.set_item("delayed", d.numeric.delayed)?;
    num.set_item("forced", d.numeric.forced)?;
    num.set_item("min_pivot", d.numeric.min_pivot)?;
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

/// Index type of a SciPy CSC matrix (`int32` or `int64`).
trait CscIndex: Element + Copy + Ord {
    fn to_usize(self) -> Option<usize>;
    fn from_usize(v: usize) -> Option<Self>;
}

impl CscIndex for i32 {
    fn to_usize(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
    fn from_usize(v: usize) -> Option<Self> {
        i32::try_from(v).ok()
    }
}

impl CscIndex for i64 {
    fn to_usize(self) -> Option<usize> {
        usize::try_from(self).ok()
    }
    fn from_usize(v: usize) -> Option<Self> {
        i64::try_from(v).ok()
    }
}

/// Lower triangle (rows at or below the diagonal) of an `n x n` CSC matrix in
/// canonical format: sorted rows, no duplicates. Each column's lower part is a
/// suffix of the column, found by binary search and copied as one slice.
fn lower_suffixes<I: CscIndex, T: Copy>(
    n: usize,
    indptr: &[I],
    indices: &[I],
    data: &[T],
) -> Option<(Vec<I>, Vec<I>, Vec<T>)> {
    if indptr.len() != n + 1 || indices.len() != data.len() {
        return None;
    }
    let ptr: Vec<usize> = indptr
        .iter()
        .map(|&p| p.to_usize())
        .collect::<Option<_>>()?;
    if ptr[0] != 0 || ptr[n] != indices.len() || ptr.windows(2).any(|w| w[0] > w[1]) {
        return None;
    }
    let starts: Vec<usize> = (0..n)
        .map(|j| {
            let col = &indices[ptr[j]..ptr[j + 1]];
            I::from_usize(j).map(|jj| ptr[j] + col.partition_point(|&r| r < jj))
        })
        .collect::<Option<_>>()?;
    let mut out_ptr = Vec::with_capacity(n + 1);
    let mut total = 0usize;
    out_ptr.push(I::from_usize(0)?);
    for j in 0..n {
        total += ptr[j + 1] - starts[j];
        out_ptr.push(I::from_usize(total)?);
    }
    let mut out_idx = Vec::with_capacity(total);
    let mut out_val = Vec::with_capacity(total);
    for j in 0..n {
        out_idx.extend_from_slice(&indices[starts[j]..ptr[j + 1]]);
        out_val.extend_from_slice(&data[starts[j]..ptr[j + 1]]);
    }
    Some((out_ptr, out_idx, out_val))
}

/// Lower triangle of a canonical `n x n` CSC matrix as `(indptr, indices,
/// data)`, in the index and value types it came in. The fast path of the
/// Python package's lower-triangle extraction; `_lower_csc` keeps the SciPy
/// route for matrices that are not in canonical format.
#[pyfunction]
pub fn lower_triangle(
    py: Python<'_>,
    n: usize,
    indptr: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    data: &Bound<'_, PyAny>,
) -> PyResult<(PyObject, PyObject, PyObject)> {
    fn by_value<I: CscIndex>(
        py: Python<'_>,
        n: usize,
        ip: &[I],
        ix: &[I],
        data: &Bound<'_, PyAny>,
    ) -> PyResult<(PyObject, PyObject, PyObject)> {
        macro_rules! try_value {
            ($T:ty) => {
                if let Ok(d) = data.extract::<PyReadonlyArray1<$T>>() {
                    let (p, i, v) = lower_suffixes(n, ip, ix, d.as_slice()?).ok_or_else(|| {
                        PyValueError::new_err("lower_triangle: malformed CSC arrays")
                    })?;
                    return Ok((array1(py, p), array1(py, i), array1(py, v)));
                }
            };
        }
        try_value!(f64);
        try_value!(C64);
        try_value!(f32);
        try_value!(C32);
        Err(PyValueError::new_err(
            "lower_triangle: data must be float32, float64, complex64 or complex128",
        ))
    }
    macro_rules! try_index {
        ($I:ty) => {
            if let (Ok(ip), Ok(ix)) = (
                indptr.extract::<PyReadonlyArray1<$I>>(),
                indices.extract::<PyReadonlyArray1<$I>>(),
            ) {
                return by_value::<$I>(py, n, ip.as_slice()?, ix.as_slice()?, data);
            }
        };
    }
    try_index!(i32);
    try_index!(i64);
    Err(PyValueError::new_err(
        "lower_triangle: indptr and indices must both be int32 or both int64",
    ))
}
