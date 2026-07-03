//! PyO3 / NumPy bindings for RSLAB.
//!
//! A **thin** FFI wrapper: this crate does no numeric work of its own. It
//! converts NumPy/SciPy CSC buffers into the core `rslab` matrix types, calls
//! the pure-Rust factorization, and hands results back as NumPy arrays. All
//! four scalar fields the core supports are exposed transparently - the array
//! `dtype` selects the path (`float64`/`float32` -> real, `complex128`/
//! `complex64` -> complex-symmetric), so the same Python call works for real
//! and complex matrices.
//!
//! The clean keyword-argument surface lives in the Python package
//! (`python/rslab/__init__.py`); here we expose two factor builders
//! (`ldlt_factor`, `lu_factor`) plus the `Ldlt` / `Lu` factor objects.

// pyo3's `#[pyfunction]`/`#[pymethods]` macros expand to `.into()` on the return
// type, which clippy flags as a useless conversion at our signature spans. The
// generated code is not ours to change, so silence that one lint crate-wide.
#![allow(clippy::useless_conversion)]

use num_complex::Complex;
use numpy::{
    Complex32, Complex64, IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use rslab::{
    gmres as gmres_core, gmres_block as gmres_block_core, CscMatrix, FactorMethod, GeneralCsc,
    LdltSolver, LuSolver, MemoryMode, RslabError, Scalar, SolverSettings, ZeroPivotAction,
};

/// Map a core solver error onto a Python `RuntimeError` carrying its message.
fn map_err(e: RslabError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// GMRES restart / basis-memory policy (issue #12).
//
// The Arnoldi basis is allocated **up front**, so its size is fixed by `restart`,
// not by how few iterations actually run:
//   * block  `gmres_block`: one basis `vbas` = `n · nrhs · (restart+1)` scalars;
//   * single `gmres`      : the flexible `V` + `Z` pair = `2 · n · (restart+1)`.
// With the old fixed `restart=80` a large `n·nrhs` allocates silently: e.g.
// `n=100k, nrhs=10, complex128` ⇒ `100000·10·81·16 B ≈ 13 GB`. When the caller
// does **not** pin `restart`, the binding instead caps it so the basis stays under
// `GMRES_BASIS_BUDGET_BYTES`, clamped to a still-useful `[MIN, MAX]`. An explicit
// `restart=` argument always wins (honoured exactly, even past the budget).
const GMRES_BASIS_BUDGET_BYTES: usize = 1 << 30; // 1 GiB
const GMRES_RESTART_MIN: usize = 20;
const GMRES_RESTART_MAX: usize = 80;

/// Restart length keeping the up-front `bases · n · columns · (restart+1)`-scalar
/// Arnoldi basis under [`GMRES_BASIS_BUDGET_BYTES`], clamped to `[MIN, MAX]`.
/// `bases` is `1` for `gmres_block`'s single block basis, `2` for the single-RHS
/// FGMRES `V`+`Z` pair. Predictable and monotone in `n·columns`.
fn adaptive_restart(n: usize, columns: usize, scalar_bytes: usize, bases: usize) -> usize {
    let per_layer = n
        .saturating_mul(columns)
        .saturating_mul(scalar_bytes)
        .saturating_mul(bases);
    if per_layer == 0 {
        return GMRES_RESTART_MAX;
    }
    // basis = per_layer · (restart+1) ≤ budget  ⇒  restart ≤ budget/per_layer − 1.
    let cap = (GMRES_BASIS_BUDGET_BYTES / per_layer).saturating_sub(1);
    cap.clamp(GMRES_RESTART_MIN, GMRES_RESTART_MAX)
}

/// Translate the Python-side keyword arguments into a core [`SolverSettings`].
#[allow(clippy::too_many_arguments)]
fn build_opts(
    threads: Option<usize>,
    preconditioner: Option<f64>,
    drop_tol: Option<f64>,
    method: &str,
    memory: &str,
    force_accept: bool,
) -> PyResult<SolverSettings> {
    let mut o = match preconditioner {
        Some(floor) => SolverSettings::preconditioner(floor),
        None => SolverSettings::default(),
    };
    // `threads=None` keeps the core default (Threads::Auto { max: 4 } - the
    // per-matrix predictor capped at 4 workers); an explicit value pins the count.
    if let Some(n) = threads {
        o = o.with_threads(n);
    }
    if let Some(tau) = drop_tol {
        o = o.with_drop_tol(tau);
    }
    o = o.with_method(match method {
        "multifrontal" | "mf" => FactorMethod::Multifrontal,
        "left_looking" | "leftlooking" | "ll" => FactorMethod::LeftLooking,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'left_looking' or 'multifrontal', got '{other}'"
            )))
        }
    });
    o = o.with_memory(match memory {
        "eager" => MemoryMode::Eager,
        "low" | "low_memory" => MemoryMode::LowMemory,
        other => {
            return Err(PyValueError::new_err(format!(
                "memory must be 'low' or 'eager', got '{other}'"
            )))
        }
    });
    // `force_accept` only matters in non-preconditioner mode; preconditioner()
    // already sets a never-fail static-pivot policy.
    if force_accept && preconditioner.is_none() {
        o = o.with_pivot(ZeroPivotAction::ForceAccept);
    }
    Ok(o)
}

/// Build a lower-triangle [`CscMatrix`] from SciPy CSC buffers (already reduced
/// to the lower triangle, indices sorted ascending, duplicates summed - done in
/// the Python wrapper). Indices arrive as `int64` and are widened to `usize`.
fn build_csc<T: Scalar + numpy::Element>(
    n: usize,
    indptr: &[i64],
    indices: &[i64],
    data: &[T],
) -> PyResult<CscMatrix<T>> {
    let m = CscMatrix::<T> {
        n,
        col_ptr: indptr.iter().map(|&x| x as usize).collect(),
        row_idx: indices.iter().map(|&x| x as usize).collect(),
        values: data.to_vec(),
    };
    m.validate().map_err(map_err)?;
    Ok(m)
}

/// Build a full (both-triangles) [`GeneralCsc`] from SciPy CSC buffers.
fn build_general<T: Scalar + numpy::Element>(
    n: usize,
    indptr: &[i64],
    indices: &[i64],
    data: &[T],
) -> PyResult<GeneralCsc<T>> {
    let m = GeneralCsc::<T> {
        n,
        col_ptr: indptr.iter().map(|&x| x as usize).collect(),
        row_idx: indices.iter().map(|&x| x as usize).collect(),
        values: data.to_vec(),
    };
    m.validate().map_err(map_err)?;
    Ok(m)
}

// ---------------------------------------------------------------------------
// Symmetric LDLᵀ factor
// ---------------------------------------------------------------------------

/// A factored symmetric matrix over one of the four scalar fields. The original
/// matrix is kept alongside the factor to support residual-driven iterative
/// refinement (`solve(refine=...)`), the recipe for the static-pivot
/// preconditioner mode.
enum LdltAny {
    F64(LdltSolver<f64>, CscMatrix<f64>),
    F32(LdltSolver<f32>, CscMatrix<f32>),
    C64(LdltSolver<Complex<f64>>, CscMatrix<Complex<f64>>),
    C32(LdltSolver<Complex<f32>>, CscMatrix<Complex<f32>>),
}

/// A factored symmetric (real or complex-symmetric) matrix, ready to solve
/// against many right-hand sides. Created by `rslab.ldlt(...)`.
/// A reusable symmetric factor handle, ``Pᵀ A P = L D Lᵀ``.
///
/// Returned by :func:`rslab.ldlt` (or :func:`rslab.spsolve` internally). Holds the
/// Bunch-Kaufman factor and the fill-reducing permutation, so the expensive
/// factorization is paid once and amortized over many :meth:`solve` /
/// :meth:`solve_many` calls. Immutable and cheap to keep around.
///
/// Attributes
/// ----------
/// n : int
///     Matrix dimension.
/// factor_nnz : int
///     Stored nonzeros in ``L`` (the fill).
/// n_perturbed : int
///     Count of statically perturbed pivots (nonzero only in preconditioner mode).
/// inertia : tuple of int
///     ``(positive, negative, zero)`` eigenvalue counts.
/// dtype : str
///     The factor's NumPy dtype name.
#[pyclass]
struct Ldlt {
    inner: LdltAny,
}

/// Solve (optionally with iterative refinement) for one scalar type, returning a
/// fresh NumPy 1-D array of the same dtype.
macro_rules! ldlt_solve_arm {
    ($py:expr, $b:expr, $refine:expr, $s:expr, $a:expr, $T:ty) => {{
        let bb: PyReadonlyArray1<$T> = $b.extract().map_err(|_| {
            PyValueError::new_err("rhs dtype does not match the factor dtype")
        })?;
        let x = if $refine > 0 {
            $s.solve_refined($a, bb.as_slice()?, $refine)
        } else {
            $s.solve(bb.as_slice()?)
        }
        .map_err(map_err)?;
        Ok(x.into_pyarray_bound($py).into_any().unbind())
    }};
}

/// Multi-RHS solve for one scalar type; `B` is a C-contiguous `n x nrhs` array,
/// returned in the same shape.
macro_rules! ldlt_solve_many_arm {
    ($py:expr, $b:expr, $s:expr, $T:ty) => {{
        let bb: PyReadonlyArray2<$T> = $b.extract().map_err(|_| {
            PyValueError::new_err("rhs dtype does not match the factor dtype")
        })?;
        let shape = bb.shape();
        let (n, nrhs) = (shape[0], shape[1]);
        let x = $s.solve_many(bb.as_slice()?, nrhs).map_err(map_err)?;
        let arr = x.into_pyarray_bound($py);
        Ok(arr.reshape([n, nrhs])?.into_any().unbind())
    }};
}

/// Right-preconditioned **block GMRES** for one scalar type. `B` is a
/// C-contiguous (row-major) `n x nrhs` array; the core wants column-major
/// (each RHS contiguous), so transpose in, run the block solve with the stored
/// matrix as operator and this factor as the preconditioner, transpose out.
/// Returns the full diagnostics tuple `(X, converged, iters, final_res)` so a
/// solver-in-the-loop caller can branch on convergence instead of silently
/// accepting a stalled iterate (issue #6): `converged` is `True` only when every
/// column reached `tol`, `iters` is the block iteration count, and `final_res`
/// is the per-column relative residual `‖B[:,c] − A X[:,c]‖ / ‖B[:,c]‖`.
macro_rules! gmres_block_arm {
    ($py:expr, $b:expr, $x0:expr, $op:expr, $pc:expr, $tol:expr, $maxit:expr, $restart:expr, $T:ty) => {{
        let bb: PyReadonlyArray2<$T> = $b
            .extract()
            .map_err(|_| PyValueError::new_err("rhs dtype does not match the factor dtype"))?;
        let shape = bb.shape();
        let (n, nrhs) = (shape[0], shape[1]);
        // Resolve the restart length (issue #12): an explicit value is honoured
        // exactly; `None` caps it so the up-front `n·nrhs·(restart+1)` basis stays
        // under the memory budget.
        let restart: usize = match $restart {
            Some(r) => r,
            None => adaptive_restart(n, nrhs, std::mem::size_of::<$T>(), 1),
        };
        let rm = bb.as_slice()?; // row-major n x nrhs
        let mut cm = vec![<$T>::default(); n * nrhs];
        for i in 0..n {
            for c in 0..nrhs {
                cm[c * n + i] = rm[i * nrhs + c];
            }
        }
        // Optional warm start `x0` (same `n x nrhs` layout as B), transposed to the
        // column-major block the core expects.
        let x0v: Option<Vec<$T>> = match $x0 {
            Some(g) => {
                let ga: PyReadonlyArray2<$T> = g.extract().map_err(|_| {
                    PyValueError::new_err("x0 dtype does not match the factor dtype")
                })?;
                let gs = ga.shape();
                if gs[0] != n || gs[1] != nrhs {
                    return Err(PyValueError::new_err("x0 shape must match B (n x nrhs)"));
                }
                let grm = ga.as_slice()?;
                let mut gcm = vec![<$T>::default(); n * nrhs];
                for i in 0..n {
                    for c in 0..nrhs {
                        gcm[c * n + i] = grm[i * nrhs + c];
                    }
                }
                Some(gcm)
            }
            None => None,
        };
        let res =
            gmres_block_core($op, &cm, nrhs, $pc, $tol, $maxit, restart, x0v.as_deref()).map_err(map_err)?;
        let mut out = vec![<$T>::default(); n * nrhs];
        for c in 0..nrhs {
            for i in 0..n {
                out[i * nrhs + c] = res.x[c * n + i];
            }
        }
        let arr = out.into_pyarray_bound($py);
        let x_obj = arr.reshape([n, nrhs])?.into_any().unbind();
        let fr = res.final_res.into_pyarray_bound($py).into_any().unbind();
        Ok((x_obj, res.converged, res.iters, fr).into_py($py))
    }};
}

/// Right-preconditioned single-RHS **GMRES** for one scalar type. `b` is a 1-D
/// array of length `n`; the stored matrix is the operator and this factor the
/// preconditioner. Returns `(x, converged, iters, final_res)` - the scalar
/// analogue of `gmres_block_arm!`, exposing the Rust `KrylovResult` diagnostics
/// that were previously unavailable from Python (issue #6).
macro_rules! gmres_arm {
    ($py:expr, $b:expr, $x0:expr, $op:expr, $pc:expr, $tol:expr, $maxit:expr, $restart:expr, $T:ty) => {{
        let bb: PyReadonlyArray1<$T> = $b
            .extract()
            .map_err(|_| PyValueError::new_err("rhs dtype does not match the factor dtype"))?;
        let rhs = bb.as_slice()?;
        // Resolve the restart length (issue #12): explicit value honoured exactly;
        // `None` caps it so the up-front `2·n·(restart+1)` FGMRES `V`+`Z` basis
        // stays under the memory budget.
        let restart: usize = match $restart {
            Some(r) => r,
            None => adaptive_restart(rhs.len(), 1, std::mem::size_of::<$T>(), 2),
        };
        // Optional warm start `x0` (length-`n` vector), seeding the iteration.
        let x0v: Option<Vec<$T>> = match $x0 {
            Some(g) => {
                let ga: PyReadonlyArray1<$T> = g.extract().map_err(|_| {
                    PyValueError::new_err("x0 dtype does not match the factor dtype")
                })?;
                Some(ga.as_slice()?.to_vec())
            }
            None => None,
        };
        let res = gmres_core($op, rhs, $pc, $tol, $maxit, restart, x0v.as_deref()).map_err(map_err)?;
        let x_obj = res.x.into_pyarray_bound($py).into_any().unbind();
        Ok((x_obj, res.converged, res.iters, res.final_res).into_py($py))
    }};
}

#[pymethods]
impl Ldlt {
    /// Matrix dimension `n`.
    #[getter]
    fn n(&self) -> usize {
        match &self.inner {
            LdltAny::F64(s, _) => s.n(),
            LdltAny::F32(s, _) => s.n(),
            LdltAny::C64(s, _) => s.n(),
            LdltAny::C32(s, _) => s.n(),
        }
    }

    /// Stored nonzeros in the factor `L` (the fill).
    #[getter]
    fn factor_nnz(&self) -> usize {
        match &self.inner {
            LdltAny::F64(s, _) => s.factor_nnz(),
            LdltAny::F32(s, _) => s.factor_nnz(),
            LdltAny::C64(s, _) => s.factor_nnz(),
            LdltAny::C32(s, _) => s.factor_nnz(),
        }
    }

    /// Number of statically perturbed pivots (nonzero only in preconditioner
    /// mode; the stored factor is then of a perturbed `A + E`, solve with
    /// `refine`).
    #[getter]
    fn n_perturbed(&self) -> usize {
        match &self.inner {
            LdltAny::F64(s, _) => s.n_perturbed(),
            LdltAny::F32(s, _) => s.n_perturbed(),
            LdltAny::C64(s, _) => s.n_perturbed(),
            LdltAny::C32(s, _) => s.n_perturbed(),
        }
    }

    /// Inertia `(positive, negative, zero)` eigenvalue counts. Exact for real
    /// symmetric matrices; advisory (pivot real-part signs) for complex.
    #[getter]
    fn inertia(&self) -> (usize, usize, usize) {
        let i = match &self.inner {
            LdltAny::F64(s, _) => s.inertia(),
            LdltAny::F32(s, _) => s.inertia(),
            LdltAny::C64(s, _) => s.inertia(),
            LdltAny::C32(s, _) => s.inertia(),
        };
        (i.positive, i.negative, i.zero)
    }

    /// The factor's NumPy dtype name (`'float64'`, `'float32'`, `'complex128'`,
    /// `'complex64'`).
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            LdltAny::F64(..) => "float64",
            LdltAny::F32(..) => "float32",
            LdltAny::C64(..) => "complex128",
            LdltAny::C32(..) => "complex64",
        }
    }

    /// Solve ``A x = b`` for a single right-hand side.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     Right-hand side of length ``n``. Its dtype must match :attr:`dtype`
    ///     (the underlying solve is strict; :func:`rslab.spsolve` casts for you).
    /// refine : int, default 0
    ///     Steps of iterative refinement against the original matrix. Use with a
    ///     preconditioner / static-pivot factor to recover full accuracy from the
    ///     inexact factor; ``0`` is a plain triangular solve.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     The solution ``x``, a fresh array of the same dtype and shape as ``b``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``b``'s dtype does not match the factor's dtype.
    #[pyo3(signature = (b, refine = 0))]
    fn solve(&self, py: Python<'_>, b: &Bound<'_, PyAny>, refine: usize) -> PyResult<PyObject> {
        match &self.inner {
            LdltAny::F64(s, a) => ldlt_solve_arm!(py, b, refine, s, a, f64),
            LdltAny::F32(s, a) => ldlt_solve_arm!(py, b, refine, s, a, f32),
            LdltAny::C64(s, a) => ldlt_solve_arm!(py, b, refine, s, a, Complex64),
            LdltAny::C32(s, a) => ldlt_solve_arm!(py, b, refine, s, a, Complex32),
        }
    }

    /// Solve ``A X = B`` for several right-hand sides at once.
    ///
    /// Batches the triangular solves across the columns of ``B`` (one pass over the
    /// factor for all right-hand sides), which is markedly faster than looping
    /// :meth:`solve` when ``nrhs`` is large.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     A C-contiguous ``n x nrhs`` block; its dtype must match :attr:`dtype`.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     The solutions ``X``, an ``n x nrhs`` array of the same dtype as ``B``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``B``'s dtype does not match the factor's dtype.
    fn solve_many(&self, py: Python<'_>, b: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        match &self.inner {
            LdltAny::F64(s, _) => ldlt_solve_many_arm!(py, b, s, f64),
            LdltAny::F32(s, _) => ldlt_solve_many_arm!(py, b, s, f32),
            LdltAny::C64(s, _) => ldlt_solve_many_arm!(py, b, s, Complex64),
            LdltAny::C32(s, _) => ldlt_solve_many_arm!(py, b, s, Complex32),
        }
    }

    /// Solve ``A X = B`` iteratively by **block GMRES**, preconditioned by this
    /// factor.
    ///
    /// Drives all ``nrhs`` right-hand sides in lockstep, this factor acting as the
    /// preconditioner :math:`M^{-1}` and the factored matrix as the operator. The
    /// payoff over :meth:`solve_many` is when the factor is *inexact* - built with
    /// ``preconditioner=...`` (static pivoting) or ``drop_tol=...`` (incomplete) -
    /// so a few preconditioned iterations recover the true solution while the
    /// factor stays cheap / memory-light. The multi-RHS orthogonalization is
    /// block-CGS2, which parallelizes across the worker pool (cap it with
    /// :func:`rslab.with_threads` in Rust, or via the ambient pool).
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     An ``n x nrhs`` block; its dtype must match :attr:`dtype`.
    /// tol : float, default 1e-8
    ///     Target relative residual :math:`\\lVert B - A X\\rVert / \\lVert B\\rVert`,
    ///     per column.
    /// maxit : int, default 400
    ///     Maximum total inner iterations.
    /// restart : int, optional
    ///     GMRES restart length (the Krylov basis depth per cycle). The Arnoldi
    ///     basis is allocated **up front**, so memory scales with ``restart``:
    ///     ``n * nrhs * (restart+1)`` scalars for :meth:`gmres_block` (one basis)
    ///     and ``2 * n * (restart+1)`` for :meth:`gmres` (the flexible ``V``+``Z``
    ///     pair). Default ``None`` caps ``restart`` within ``[20, 80]`` so the
    ///     basis stays under ~1 GiB (``n=100k, nrhs=10, complex128`` would
    ///     otherwise take ~13 GB at ``restart=80``). Pass an integer to pin
    ///     ``restart`` exactly - honoured even if it exceeds that budget.
    /// x0 : numpy.ndarray, optional
    ///     Warm-start initial guess, same shape and dtype as the right-hand side.
    ///     On a sequence of related solves (slowly varying operator or RHS),
    ///     seeding with the previous solution typically cuts the iteration count
    ///     substantially. Convergence is still measured relative to the RHS norm.
    ///     ``None`` (default) starts from zero.
    ///
    /// Returns
    /// -------
    /// X : numpy.ndarray
    ///     The solutions ``X`` as an ``n x nrhs`` array of the factor's dtype.
    /// converged : bool
    ///     ``True`` only if **every** column reached ``tol``. When ``maxit`` is hit
    ///     the core returns the best iterate with ``converged=False`` (it does not
    ///     raise), so an in-the-loop caller must branch on this flag rather than
    ///     assume success.
    /// iters : int
    ///     Block iterations performed (all columns advance in lockstep).
    /// final_res : numpy.ndarray
    ///     Per-column relative residual
    ///     :math:`\\lVert B_{:,c} - A X_{:,c}\\rVert / \\lVert B_{:,c}\\rVert`,
    ///     a length-``nrhs`` ``float64`` array.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``B``'s dtype does not match the factor's dtype.
    #[pyo3(signature = (b, tol = 1e-8, maxit = 400, restart = None, x0 = None))]
    fn gmres_block(
        &self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        tol: f64,
        maxit: usize,
        restart: Option<usize>,
        x0: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        match &self.inner {
            LdltAny::F64(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f64),
            LdltAny::F32(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f32),
            LdltAny::C64(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex64),
            LdltAny::C32(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex32),
        }
    }

    /// Solve ``A x = b`` for a single right-hand side by preconditioned **GMRES**.
    ///
    /// The single-RHS companion to :meth:`gmres_block`: the stored matrix is the
    /// operator and this (possibly inexact) factor the preconditioner
    /// :math:`M^{-1}`. Use it when the factor is built with ``preconditioner=...``
    /// (static pivoting) or ``drop_tol=...`` (incomplete) so a few preconditioned
    /// iterations recover the true solution.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     Right-hand side of length ``n``; its dtype must match :attr:`dtype`.
    /// tol : float, default 1e-8
    ///     Target relative residual :math:`\\lVert b - A x\\rVert / \\lVert b\\rVert`.
    /// maxit : int, default 400
    ///     Maximum total inner iterations.
    /// restart : int, optional
    ///     GMRES restart length (the Krylov basis depth per cycle). The Arnoldi
    ///     basis is allocated **up front**, so memory scales with ``restart``:
    ///     ``n * nrhs * (restart+1)`` scalars for :meth:`gmres_block` (one basis)
    ///     and ``2 * n * (restart+1)`` for :meth:`gmres` (the flexible ``V``+``Z``
    ///     pair). Default ``None`` caps ``restart`` within ``[20, 80]`` so the
    ///     basis stays under ~1 GiB (``n=100k, nrhs=10, complex128`` would
    ///     otherwise take ~13 GB at ``restart=80``). Pass an integer to pin
    ///     ``restart`` exactly - honoured even if it exceeds that budget.
    /// x0 : numpy.ndarray, optional
    ///     Warm-start initial guess, same shape and dtype as the right-hand side.
    ///     On a sequence of related solves (slowly varying operator or RHS),
    ///     seeding with the previous solution typically cuts the iteration count
    ///     substantially. Convergence is still measured relative to the RHS norm.
    ///     ``None`` (default) starts from zero.
    ///
    /// Returns
    /// -------
    /// x : numpy.ndarray
    ///     The solution ``x`` of length ``n`` and the factor's dtype.
    /// converged : bool
    ///     ``True`` iff the relative residual reached ``tol``; ``False`` (with the
    ///     best iterate returned, no exception) when ``maxit`` is hit.
    /// iters : int
    ///     Inner iterations performed.
    /// final_res : float
    ///     The final relative residual :math:`\\lVert b - A x\\rVert / \\lVert b\\rVert`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``b``'s dtype does not match the factor's dtype.
    #[pyo3(signature = (b, tol = 1e-8, maxit = 400, restart = None, x0 = None))]
    fn gmres(
        &self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        tol: f64,
        maxit: usize,
        restart: Option<usize>,
        x0: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        match &self.inner {
            LdltAny::F64(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f64),
            LdltAny::F32(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f32),
            LdltAny::C64(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex64),
            LdltAny::C32(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex32),
        }
    }
}

/// Factor a symmetric matrix from its lower-triangle SciPy CSC buffers. The
/// `data` array's dtype picks the scalar field.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (n, indptr, indices, data, threads, preconditioner, drop_tol, method, memory, force_accept))]
fn ldlt_factor(
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    threads: Option<usize>,
    preconditioner: Option<f64>,
    drop_tol: Option<f64>,
    method: &str,
    memory: &str,
    force_accept: bool,
) -> PyResult<Ldlt> {
    let opts = build_opts(threads, preconditioner, drop_tol, method, memory, force_accept)?;
    let ip = indptr.as_slice()?;
    let ii = indices.as_slice()?;
    macro_rules! try_build {
        ($T:ty, $variant:ident) => {
            if let Ok(d) = data.extract::<PyReadonlyArray1<$T>>() {
                let a = build_csc::<$T>(n, ip, ii, d.as_slice()?)?;
                let s = LdltSolver::<$T>::factor_with(&a, &opts).map_err(map_err)?;
                return Ok(Ldlt {
                    inner: LdltAny::$variant(s, a),
                });
            }
        };
    }
    try_build!(f64, F64);
    try_build!(Complex64, C64);
    try_build!(f32, F32);
    try_build!(Complex32, C32);
    Err(PyValueError::new_err(
        "unsupported dtype: expected float64, float32, complex128, or complex64",
    ))
}

// ---------------------------------------------------------------------------
// Unsymmetric LU factor
// ---------------------------------------------------------------------------

/// A factored general (unsymmetric) matrix over one of the four scalar fields,
/// kept with its original matrix for iterative refinement.
enum LuAny {
    F64(LuSolver<f64>, GeneralCsc<f64>),
    F32(LuSolver<f32>, GeneralCsc<f32>),
    C64(LuSolver<Complex<f64>>, GeneralCsc<Complex<f64>>),
    C32(LuSolver<Complex<f32>>, GeneralCsc<Complex<f32>>),
}

/// A reusable general (unsymmetric) factor handle, ``Pᵀ A P = L U``.
///
/// Returned by :func:`rslab.lu` (or :func:`rslab.spsolve` internally). Holds the
/// multifrontal ``L U`` factor, the fill-reducing permutation, and a copy of the
/// original matrix (for iterative refinement), so the factorization is paid once
/// and amortized over many :meth:`solve` / :meth:`solve_many` calls.
///
/// Attributes
/// ----------
/// n : int
///     Matrix dimension.
/// factor_nnz : int
///     Stored fill, ``nnz(L) + nnz(U)``.
/// n_perturbed : int
///     Count of statically perturbed pivots (nonzero only in preconditioner mode).
/// dtype : str
///     The factor's NumPy dtype name.
#[pyclass]
struct Lu {
    inner: LuAny,
}

macro_rules! lu_solve_arm {
    ($py:expr, $b:expr, $refine:expr, $s:expr, $a:expr, $T:ty) => {{
        let bb: PyReadonlyArray1<$T> = $b.extract().map_err(|_| {
            PyValueError::new_err("rhs dtype does not match the factor dtype")
        })?;
        let x = if $refine > 0 {
            $s.solve_refined($a, bb.as_slice()?, $refine)
        } else {
            $s.solve(bb.as_slice()?)
        }
        .map_err(map_err)?;
        Ok(x.into_pyarray_bound($py).into_any().unbind())
    }};
}

macro_rules! lu_solve_many_arm {
    ($py:expr, $b:expr, $s:expr, $T:ty) => {{
        let bb: PyReadonlyArray2<$T> = $b.extract().map_err(|_| {
            PyValueError::new_err("rhs dtype does not match the factor dtype")
        })?;
        let shape = bb.shape();
        let (n, nrhs) = (shape[0], shape[1]);
        let x = $s.solve_many(bb.as_slice()?, nrhs).map_err(map_err)?;
        let arr = x.into_pyarray_bound($py);
        Ok(arr.reshape([n, nrhs])?.into_any().unbind())
    }};
}

#[pymethods]
impl Lu {
    /// Matrix dimension `n`.
    #[getter]
    fn n(&self) -> usize {
        match &self.inner {
            LuAny::F64(s, _) => s.n(),
            LuAny::F32(s, _) => s.n(),
            LuAny::C64(s, _) => s.n(),
            LuAny::C32(s, _) => s.n(),
        }
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    #[getter]
    fn factor_nnz(&self) -> usize {
        match &self.inner {
            LuAny::F64(s, _) => s.factor_nnz(),
            LuAny::F32(s, _) => s.factor_nnz(),
            LuAny::C64(s, _) => s.factor_nnz(),
            LuAny::C32(s, _) => s.factor_nnz(),
        }
    }

    /// Number of statically perturbed pivots (nonzero only in preconditioner
    /// mode; the stored factor is then of a perturbed `A + E`, solve with
    /// `refine`).
    #[getter]
    fn n_perturbed(&self) -> usize {
        match &self.inner {
            LuAny::F64(s, _) => s.n_perturbed(),
            LuAny::F32(s, _) => s.n_perturbed(),
            LuAny::C64(s, _) => s.n_perturbed(),
            LuAny::C32(s, _) => s.n_perturbed(),
        }
    }

    /// The factor's NumPy dtype name (`'float64'`, `'float32'`, `'complex128'`,
    /// `'complex64'`).
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            LuAny::F64(..) => "float64",
            LuAny::F32(..) => "float32",
            LuAny::C64(..) => "complex128",
            LuAny::C32(..) => "complex64",
        }
    }

    /// Solve ``A x = b`` for a single right-hand side.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     Right-hand side of length ``n``; its dtype must match :attr:`dtype`.
    /// refine : int, default 0
    ///     Steps of iterative refinement against the original matrix (recovers
    ///     accuracy from a preconditioner / static-pivot factor); ``0`` is a plain
    ///     triangular solve.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     The solution ``x``, a fresh array of the same dtype and shape as ``b``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``b``'s dtype does not match the factor's dtype.
    #[pyo3(signature = (b, refine = 0))]
    fn solve(&self, py: Python<'_>, b: &Bound<'_, PyAny>, refine: usize) -> PyResult<PyObject> {
        match &self.inner {
            LuAny::F64(s, a) => lu_solve_arm!(py, b, refine, s, a, f64),
            LuAny::F32(s, a) => lu_solve_arm!(py, b, refine, s, a, f32),
            LuAny::C64(s, a) => lu_solve_arm!(py, b, refine, s, a, Complex64),
            LuAny::C32(s, a) => lu_solve_arm!(py, b, refine, s, a, Complex32),
        }
    }

    /// Solve ``A X = B`` for several right-hand sides at once.
    ///
    /// Batches the triangular solves across the columns of ``B`` (one pass over the
    /// factor for all right-hand sides) - faster than looping :meth:`solve`.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     A C-contiguous ``n x nrhs`` block; its dtype must match :attr:`dtype`.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     The solutions ``X``, an ``n x nrhs`` array of the same dtype as ``B``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``B``'s dtype does not match the factor's dtype.
    fn solve_many(&self, py: Python<'_>, b: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        match &self.inner {
            LuAny::F64(s, _) => lu_solve_many_arm!(py, b, s, f64),
            LuAny::F32(s, _) => lu_solve_many_arm!(py, b, s, f32),
            LuAny::C64(s, _) => lu_solve_many_arm!(py, b, s, Complex64),
            LuAny::C32(s, _) => lu_solve_many_arm!(py, b, s, Complex32),
        }
    }

    /// Solve ``A X = B`` iteratively by **block GMRES**, preconditioned by this
    /// factor.
    ///
    /// Drives all ``nrhs`` right-hand sides in lockstep, this factor acting as the
    /// preconditioner :math:`M^{-1}` and the factored matrix as the operator. The
    /// payoff over :meth:`solve_many` is when the factor is *inexact* - built with
    /// ``preconditioner=...`` (static pivoting) or ``drop_tol=...`` (incomplete) -
    /// so a few preconditioned iterations recover the true solution while the
    /// factor stays cheap / memory-light. This is the many-excitations MoM / FEM
    /// path (factor the near-field once, drive all right-hand sides in one block
    /// iteration). The multi-RHS orthogonalization is block-CGS2, which
    /// parallelizes across the worker pool.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     An ``n x nrhs`` block; its dtype must match :attr:`dtype`.
    /// tol : float, default 1e-8
    ///     Target relative residual :math:`\\lVert B - A X\\rVert / \\lVert B\\rVert`,
    ///     per column.
    /// maxit : int, default 400
    ///     Maximum total inner iterations.
    /// restart : int, optional
    ///     GMRES restart length (the Krylov basis depth per cycle). The Arnoldi
    ///     basis is allocated **up front**, so memory scales with ``restart``:
    ///     ``n * nrhs * (restart+1)`` scalars for :meth:`gmres_block` (one basis)
    ///     and ``2 * n * (restart+1)`` for :meth:`gmres` (the flexible ``V``+``Z``
    ///     pair). Default ``None`` caps ``restart`` within ``[20, 80]`` so the
    ///     basis stays under ~1 GiB (``n=100k, nrhs=10, complex128`` would
    ///     otherwise take ~13 GB at ``restart=80``). Pass an integer to pin
    ///     ``restart`` exactly - honoured even if it exceeds that budget.
    /// x0 : numpy.ndarray, optional
    ///     Warm-start initial guess, same shape and dtype as the right-hand side.
    ///     On a sequence of related solves (slowly varying operator or RHS),
    ///     seeding with the previous solution typically cuts the iteration count
    ///     substantially. Convergence is still measured relative to the RHS norm.
    ///     ``None`` (default) starts from zero.
    ///
    /// Returns
    /// -------
    /// X : numpy.ndarray
    ///     The solutions ``X`` as an ``n x nrhs`` array of the factor's dtype.
    /// converged : bool
    ///     ``True`` only if **every** column reached ``tol``. When ``maxit`` is hit
    ///     the core returns the best iterate with ``converged=False`` (it does not
    ///     raise), so an in-the-loop caller must branch on this flag rather than
    ///     assume success.
    /// iters : int
    ///     Block iterations performed (all columns advance in lockstep).
    /// final_res : numpy.ndarray
    ///     Per-column relative residual
    ///     :math:`\\lVert B_{:,c} - A X_{:,c}\\rVert / \\lVert B_{:,c}\\rVert`,
    ///     a length-``nrhs`` ``float64`` array.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``B``'s dtype does not match the factor's dtype.
    #[pyo3(signature = (b, tol = 1e-8, maxit = 400, restart = None, x0 = None))]
    fn gmres_block(
        &self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        tol: f64,
        maxit: usize,
        restart: Option<usize>,
        x0: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        match &self.inner {
            LuAny::F64(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f64),
            LuAny::F32(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f32),
            LuAny::C64(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex64),
            LuAny::C32(s, a) => gmres_block_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex32),
        }
    }

    /// Solve ``A x = b`` for a single right-hand side by preconditioned **GMRES**.
    ///
    /// The single-RHS companion to :meth:`gmres_block`: the stored general matrix
    /// is the operator and this (possibly inexact) LU factor the preconditioner
    /// :math:`M^{-1}`. The natural Krylov method for the unsymmetric MoM / FEM
    /// systems this factor targets; use it with an inexact factor
    /// (``preconditioner=...`` or ``drop_tol=...``) to recover the true solution in
    /// a few iterations.
    ///
    /// Parameters
    /// ----------
    /// b : numpy.ndarray
    ///     Right-hand side of length ``n``; its dtype must match :attr:`dtype`.
    /// tol : float, default 1e-8
    ///     Target relative residual :math:`\\lVert b - A x\\rVert / \\lVert b\\rVert`.
    /// maxit : int, default 400
    ///     Maximum total inner iterations.
    /// restart : int, optional
    ///     GMRES restart length (the Krylov basis depth per cycle). The Arnoldi
    ///     basis is allocated **up front**, so memory scales with ``restart``:
    ///     ``n * nrhs * (restart+1)`` scalars for :meth:`gmres_block` (one basis)
    ///     and ``2 * n * (restart+1)`` for :meth:`gmres` (the flexible ``V``+``Z``
    ///     pair). Default ``None`` caps ``restart`` within ``[20, 80]`` so the
    ///     basis stays under ~1 GiB (``n=100k, nrhs=10, complex128`` would
    ///     otherwise take ~13 GB at ``restart=80``). Pass an integer to pin
    ///     ``restart`` exactly - honoured even if it exceeds that budget.
    /// x0 : numpy.ndarray, optional
    ///     Warm-start initial guess, same shape and dtype as the right-hand side.
    ///     On a sequence of related solves (slowly varying operator or RHS),
    ///     seeding with the previous solution typically cuts the iteration count
    ///     substantially. Convergence is still measured relative to the RHS norm.
    ///     ``None`` (default) starts from zero.
    ///
    /// Returns
    /// -------
    /// x : numpy.ndarray
    ///     The solution ``x`` of length ``n`` and the factor's dtype.
    /// converged : bool
    ///     ``True`` iff the relative residual reached ``tol``; ``False`` (with the
    ///     best iterate returned, no exception) when ``maxit`` is hit.
    /// iters : int
    ///     Inner iterations performed.
    /// final_res : float
    ///     The final relative residual :math:`\\lVert b - A x\\rVert / \\lVert b\\rVert`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``b``'s dtype does not match the factor's dtype.
    #[pyo3(signature = (b, tol = 1e-8, maxit = 400, restart = None, x0 = None))]
    fn gmres(
        &self,
        py: Python<'_>,
        b: &Bound<'_, PyAny>,
        tol: f64,
        maxit: usize,
        restart: Option<usize>,
        x0: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        match &self.inner {
            LuAny::F64(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f64),
            LuAny::F32(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, f32),
            LuAny::C64(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex64),
            LuAny::C32(s, a) => gmres_arm!(py, b, x0.as_ref(), a, s, tol, maxit, restart, Complex32),
        }
    }
}

/// Factor a general matrix from its full SciPy CSC buffers. The `data` dtype
/// picks the scalar field.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (n, indptr, indices, data, threads, preconditioner, drop_tol, method, memory, force_accept))]
fn lu_factor(
    n: usize,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i64>,
    data: &Bound<'_, PyAny>,
    threads: Option<usize>,
    preconditioner: Option<f64>,
    drop_tol: Option<f64>,
    method: &str,
    memory: &str,
    force_accept: bool,
) -> PyResult<Lu> {
    let opts = build_opts(threads, preconditioner, drop_tol, method, memory, force_accept)?;
    let ip = indptr.as_slice()?;
    let ii = indices.as_slice()?;
    macro_rules! try_build {
        ($T:ty, $variant:ident) => {
            if let Ok(d) = data.extract::<PyReadonlyArray1<$T>>() {
                let a = build_general::<$T>(n, ip, ii, d.as_slice()?)?;
                let s = LuSolver::<$T>::factor(&a, &opts).map_err(map_err)?;
                return Ok(Lu {
                    inner: LuAny::$variant(s, a),
                });
            }
        };
    }
    try_build!(f64, F64);
    try_build!(Complex64, C64);
    try_build!(f32, F32);
    try_build!(Complex32, C32);
    Err(PyValueError::new_err(
        "unsupported dtype: expected float64, float32, complex128, or complex64",
    ))
}

/// The compiled core imported by the Python package as `rslab._rslab`.
#[pymodule]
fn _rslab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Ldlt>()?;
    m.add_class::<Lu>()?;
    m.add_function(wrap_pyfunction!(ldlt_factor, m)?)?;
    m.add_function(wrap_pyfunction!(lu_factor, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
