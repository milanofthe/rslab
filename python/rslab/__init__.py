"""RSLAB - a pure-Rust sparse direct solver and preconditioner for NumPy/SciPy.

A thin wrapper over the Rust core. It factors **symmetric** matrices by a
complex-symmetric/real LDLᵀ (Bunch-Kaufman) method and **general** matrices by
an unsymmetric LU, then solves against one or many right-hand sides. It is
type-agnostic: the matrix dtype selects the field, so ``float64``/``float32``
run the real path and ``complex128``/``complex64`` the complex path with the
same call.

Quick start
-----------
>>> import numpy as np, scipy.sparse as sp, rslab
>>> A = sp.random(2000, 2000, density=1e-3, format="csc") + sp.eye(2000) * 10
>>> A = A + A.T                                   # symmetric
>>> b = np.random.rand(2000)
>>> x = rslab.spsolve(A, b)                       # one-shot
>>> f = rslab.ldlt(A); x = f.solve(b)             # factor once, solve many

The factor configuration is passed as keyword arguments (see :func:`ldlt` /
:func:`lu`): ``threads``, ``preconditioner``, ``drop_tol``, ``method``,
``memory``, ``force_accept``.
"""

from __future__ import annotations

import numpy as np

from . import _rslab
from ._rslab import Ldlt, Lu

__all__ = ["ldlt", "lu", "spsolve", "Ldlt", "Lu"]
__version__ = _rslab.__version__

# The four scalar fields the Rust core supports, by NumPy dtype.
_SUPPORTED = (np.float64, np.float32, np.complex128, np.complex64)


def _require_scipy():
    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "rslab needs SciPy for its sparse-matrix input; `pip install scipy`"
        ) from exc
    return sp


def _normalize_dtype(data: np.ndarray) -> np.ndarray:
    """Coerce ``data`` to one of the four supported dtypes (widening if needed)."""
    if data.dtype.type in _SUPPORTED:
        return data
    if np.iscomplexobj(data):
        return data.astype(np.complex128)
    return data.astype(np.float64)


def _lower_csc(A):
    """Lower triangle of a symmetric matrix as a sorted, summed CSC matrix."""
    sp = _require_scipy()
    L = sp.tril(sp.csc_matrix(A)).tocsc()
    L.sum_duplicates()
    L.sort_indices()
    return L


def _full_csc(A):
    """Full matrix as a sorted, summed CSC matrix (both triangles, as given)."""
    sp = _require_scipy()
    A = sp.csc_matrix(A)
    A.sum_duplicates()
    A.sort_indices()
    return A


def _opts(threads, preconditioner, drop_tol, method, memory, force_accept):
    return (
        None if threads is None else int(threads),
        None if preconditioner is None else float(preconditioner),
        None if drop_tol is None else float(drop_tol),
        str(method),
        str(memory),
        bool(force_accept),
    )


def ldlt(
    A,
    *,
    threads: int | None = None,
    preconditioner: float | None = None,
    drop_tol: float | None = None,
    method: str = "left_looking",
    memory: str = "low",
    force_accept: bool = False,
) -> Ldlt:
    """Factor a **symmetric** matrix ``A`` as ``Pᵀ A P = L D Lᵀ``.

    Only the lower triangle of ``A`` is used (it is extracted automatically), so
    ``A`` may be stored full or triangular. Works for real symmetric
    (``float64``/``float32``) and complex-symmetric (``complex128``/
    ``complex64``) matrices; the dtype selects the path.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The symmetric system matrix.
    threads : int, optional
        Worker-thread budget. ``None`` (default) uses the **auto** per-matrix
        predictor (thin/tiny systems stay low where they would only regress, big
        BLAS-3-rich systems use the cores), up to all logical cores. An explicit
        integer fixes the count (``0`` = all cores); use a small fixed value for
        many concurrent solves sharing the machine. Bit-identical either way.
    preconditioner : float, optional
        If set, never-fail **static-pivoting** mode: any pivot below this
        absolute floor (typically ``eps_rel * ‖A‖``) is lifted, so the factor
        is of a perturbed ``A + E``. Drive ``solve(..., refine=k)`` against the
        original matrix for accuracy. A good starting floor is ``1e-4``.
    drop_tol : float, optional
        Incomplete-factor threshold: fill entries below this (relative to the
        column) are dropped, trading accuracy for memory. Makes the factor a
        preconditioner.
    method : {'left_looking', 'multifrontal'}, default 'left_looking'
        Numeric algorithm. Both produce the same factor; they differ in the
        transient-memory / scheduling profile.
    memory : {'low', 'eager'}, default 'low'
        Factor emit strategy. ``'low'`` frees each front as it is emitted
        (lower peak RSS, bit-identical factors).
    force_accept : bool, default False
        In exact mode, accept tiny pivots at face value instead of failing.
        Ignored when ``preconditioner`` is set.

    Returns
    -------
    Ldlt
        A factor object exposing ``solve``, ``solve_many``, ``n``,
        ``factor_nnz``, ``n_perturbed``, ``inertia`` and ``dtype``.
    """
    L = _lower_csc(A)
    data = _normalize_dtype(L.data)
    return _rslab.ldlt_factor(
        L.shape[0],
        L.indptr.astype(np.int64),
        L.indices.astype(np.int64),
        data,
        *_opts(threads, preconditioner, drop_tol, method, memory, force_accept),
    )


def lu(
    A,
    *,
    threads: int | None = None,
    preconditioner: float | None = None,
    drop_tol: float | None = None,
    method: str = "left_looking",
    memory: str = "low",
    force_accept: bool = False,
) -> Lu:
    """Factor a **general** (unsymmetric) matrix ``A`` as ``Pᵀ A P = L U``.

    The full matrix (both triangles) is used. Works for real and complex dtypes;
    the dtype selects the path. The keyword arguments match :func:`ldlt`.

    Returns
    -------
    Lu
        A factor object exposing ``solve``, ``solve_many``, ``n``,
        ``factor_nnz``, ``n_perturbed`` and ``dtype``.
    """
    M = _full_csc(A)
    data = _normalize_dtype(M.data)
    return _rslab.lu_factor(
        M.shape[0],
        M.indptr.astype(np.int64),
        M.indices.astype(np.int64),
        data,
        *_opts(threads, preconditioner, drop_tol, method, memory, force_accept),
    )


def _is_symmetric(A, tol: float = 1e-12) -> bool:
    """Cheap structural+value symmetry test for picking the LDLᵀ vs LU path."""
    sp = _require_scipy()
    A = sp.csc_matrix(A)
    if A.shape[0] != A.shape[1]:
        return False
    d = A - A.T
    if d.nnz == 0:
        return True
    return float(abs(d).max()) <= tol * (float(abs(A).max()) or 1.0)


def _match_dtype(b: np.ndarray, dtype_name: str) -> np.ndarray:
    """Cast a right-hand side to the factor's dtype (the Rust solve is strict)."""
    return np.ascontiguousarray(b, dtype=np.dtype(dtype_name))


def spsolve(
    A,
    b,
    *,
    symmetric: bool | None = None,
    refine: int = 0,
    **opts,
):
    """One-shot solve of ``A x = b`` (factor + solve).

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The system matrix.
    b : array-like
        Right-hand side; a 1-D vector or a 2-D ``n x nrhs`` block. Cast to the
        factor dtype automatically.
    symmetric : bool, optional
        Force the symmetric LDLᵀ path (``True``) or the unsymmetric LU path
        (``False``). Auto-detected from ``A`` when omitted.
    refine : int, default 0
        Iterative-refinement steps against the original matrix (use with
        ``preconditioner=...``).
    **opts
        Forwarded to :func:`ldlt` / :func:`lu` (``threads``, ``preconditioner``,
        ``drop_tol``, ``method``, ``memory``, ``force_accept``).

    Returns
    -------
    numpy.ndarray
        The solution, matching the shape of ``b``.
    """
    if symmetric is None:
        symmetric = _is_symmetric(A)
    f = ldlt(A, **opts) if symmetric else lu(A, **opts)
    b = np.asarray(b)
    rhs = _match_dtype(b, f.dtype)
    if rhs.ndim == 1:
        return f.solve(rhs, refine)
    if rhs.ndim == 2:
        if refine:
            # Multi-RHS refinement: refine each column independently.
            cols = [f.solve(np.ascontiguousarray(rhs[:, c]), refine) for c in range(rhs.shape[1])]
            return np.stack(cols, axis=1)
        return f.solve_many(rhs)
    raise ValueError("b must be 1-D or 2-D")
