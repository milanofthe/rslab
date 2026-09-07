"""RSLAB - a pure-Rust sparse direct solver and preconditioner for NumPy/SciPy.

A thin, allocation-light wrapper over the RSLAB Rust core: a drop-in
alternative to :func:`scipy.sparse.linalg.spsolve` / Intel MKL PARDISO for the
factor-once, solve-many workloads that dominate FEM, method-of-moments, and
circuit-extraction codes. Three factorization paths cover the operator
classes:

**Symmetric** matrices (real, or complex-symmetric ``A = A^T``)
are factored by a supernodal Bunch-Kaufman method,

.. math::

    P^T A P = L D L^T,

**general unsymmetric** matrices by a supernodal / multifrontal LU with
threshold partial pivoting,

.. math::

    P_r^T A P_c = L U,

and **circuit-shaped** matrices (MNA / SPICE class: extremely sparse,
unsymmetric, near-triangularizable) by a KLU-style path - block triangular
form plus a per-block Gilbert-Peierls LU - whose numeric-only ``refactor``
makes fixed-pattern sweeps cheap.

The solver is **type-agnostic**: the matrix ``dtype`` selects the arithmetic
field, so ``float64`` / ``float32`` run the real path and ``complex128`` /
``complex64`` the complex path through the *same* call, at half the memory
for the 32-bit fields.

The API has four layers, each a thin step below the previous one:

1. :func:`spsolve` - one-shot factor, solve, discard.
2. :func:`ldlt` / :func:`lu` / :func:`klu` - factor once into a handle
   (:class:`Ldlt`, :class:`Lu`, :class:`Klu`), solve many.
3. :func:`analyze` - the symbolic analysis alone (:class:`LdltSymbolic`,
   :class:`LuSymbolic`, :class:`KluSymbolic`); factor many value sets on one
   pattern.
4. :func:`gmres` / :func:`gmres_block` / :func:`cocg` / :func:`cocr` -
   Krylov solvers, optionally preconditioned by any factor handle.

Configuration is a :class:`Settings` (LDL^T / LU) or :class:`KluSettings`
object, or the same keywords given to the factor functions directly.

Example
-------
.. code-block:: python

    import numpy as np, scipy.sparse as sp, rslab

    A = sp.random(2000, 2000, density=1e-3, format="csc") + sp.eye(2000) * 10
    A = A + A.T                       # make it symmetric
    b = np.random.rand(2000)
    x = rslab.spsolve(A, b)           # one-shot: factor + solve + discard

    f = rslab.ldlt(A)                 # factor once ...
    x1 = f.solve(b)                   # ... solve many
    X  = f.solve_many(np.random.rand(2000, 8))   # 8 right-hand sides at once

    sym = rslab.analyze(A, path="ldlt")          # analyze once ...
    for scale in (1.0, 2.0, 3.0):
        f = sym.factor(A.data * scale)           # ... factor many value sets

Note
----
The numeric factor is **bit-identical regardless of the thread count**; the
worker budget affects wall time and transient memory, not the result. By
default the factorization uses at most 4 workers (the pareto-optimal
throughput-per-core point on typical sparse factorizations); pass an explicit
``threads`` to override.

References
----------
.. [1] Bunch, J. R., & Kaufman, L. (1977). "Some stable methods for calculating
       inertia and solving symmetric linear systems." *Mathematics of
       Computation*, 31(137), 163-179. :doi:`10.1090/S0025-5718-1977-0428694-0`
.. [2] Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
       :doi:`10.1137/1.9780898718881`
.. [3] Davis, T. A., & Palamadai Natarajan, E. (2010). "Algorithm 907: KLU, a
       direct sparse solver for circuit simulation problems." *ACM Transactions
       on Mathematical Software*, 37(3). :doi:`10.1145/1824801.1824814`
"""

from __future__ import annotations

import numpy as np

from . import _rslab
from ._rslab import (
    Interrupt,
    Klu,
    KluSettings,
    KluSymbolic,
    KrylovResult,
    Ldlt,
    LdltSymbolic,
    Lu,
    LuSymbolic,
    Recycle,
    Settings,
    install_diagnose,
    log_level,
    set_log_level,
    set_log_sink,
)

__all__ = [
    # one-shot
    "spsolve",
    # factor handles
    "ldlt",
    "lu",
    "klu",
    "Ldlt",
    "Lu",
    "Klu",
    # symbolic analysis
    "analyze",
    "LdltSymbolic",
    "LuSymbolic",
    "KluSymbolic",
    # configuration
    "Settings",
    "KluSettings",
    "Interrupt",
    # Krylov
    "gmres",
    "gmres_block",
    "cocg",
    "cocr",
    "KrylovResult",
    "Recycle",
    # machine and logging
    "install_diagnose",
    "set_log_level",
    "log_level",
    "set_log_sink",
]

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


def _parts(M):
    """``(n, indptr, indices, data)`` of a prepared CSC matrix, as the core wants them."""
    n = M.shape[0]
    if M.shape[1] != n:
        raise ValueError(f"matrix must be square, got {M.shape}")
    return (
        n,
        np.ascontiguousarray(M.indptr, dtype=np.int64),
        np.ascontiguousarray(M.indices, dtype=np.int64),
        np.ascontiguousarray(_normalize_dtype(M.data)),
    )


def _csc_parts(A, path: str):
    return _parts(_lower_csc(A) if path == "ldlt" else _full_csc(A))


# ---------------------------------------------------------------------------
# Factor handles
# ---------------------------------------------------------------------------


def ldlt(A, *, settings: Settings | None = None, **kwargs) -> Ldlt:
    """Factor a **symmetric** matrix as ``P^T A P = L D L^T``.

    A supernodal Bunch-Kaufman ``L D L^T`` factorization with a
    fill-reducing ordering ``P``, for real symmetric (``float64`` /
    ``float32``) and **complex-symmetric** (``complex128`` / ``complex64``,
    i.e. ``A = A^T``, *not* Hermitian) matrices; the ``dtype``
    selects the path. Only the lower triangle is read (extracted
    automatically), so ``A`` may be stored full or triangular.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The symmetric ``n x n`` system matrix. Converted to CSC and
        its lower triangle taken; duplicate entries are summed.
    settings : Settings, optional
        A prepared :class:`Settings` object.
    **kwargs
        Any :class:`Settings` keyword (``threads``, ``preconditioner``,
        ``drop_tol``, ``method``, ``memory``, ``force_accept``, ``ordering``,
        ``scaling``, ``pivot_u``, ``nemin``, ``relax``, ``reorder``, ``blr``,
        ``panel_nb``, ``interrupt`` ...), overriding ``settings``.

    Returns
    -------
    Ldlt
        A reusable factor handle: :meth:`Ldlt.solve`, :meth:`Ldlt.solve_many`,
        the Krylov methods with the factor as preconditioner, and
        :meth:`Ldlt.diagnostics`.

    Raises
    ------
    RuntimeError
        If a pivot is numerically zero in exact mode (the matrix is rank
        deficient). Set ``preconditioner=...`` (recommended) or
        ``force_accept=True`` to proceed.
    ValueError
        On an unsupported dtype, a non-square matrix or an invalid setting.

    Example
    -------
    .. code-block:: python

        f = rslab.ldlt(A)                       # heuristic defaults
        f = rslab.ldlt(A, ordering="metis", threads=2)
        f = rslab.ldlt(A, preconditioner=1e-4)  # never-fail static pivoting
        x = f.solve(b, refine=2)
        print(f.inertia, f.diagnostics()["summary"])
    """
    return _rslab.ldlt_factor(*_csc_parts(A, "ldlt"), settings=settings, **kwargs)


def lu(A, *, settings: Settings | None = None, **kwargs) -> Lu:
    """Factor a **general** (unsymmetric) matrix as ``P_r^T A P_c = L U``.

    A supernodal left-looking (default) or multifrontal LU with threshold
    partial pivoting and two-sided equilibration, over the same four scalar
    fields as :func:`ldlt`. The full matrix is read.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The ``n x n`` system matrix. Converted to CSC; duplicates
        summed.
    settings : Settings, optional
        A prepared :class:`Settings` object.
    **kwargs
        Any :class:`Settings` keyword, overriding ``settings``. ``pivot_u``
        (default 0.1) is the threshold-pivoting tolerance of this path;
        ``scaling`` is ignored here (the LU path scales two-sided) and
        reported under ``diagnostics()['warnings']``.

    Returns
    -------
    Lu
        A reusable factor handle (:meth:`Lu.solve`, :meth:`Lu.solve_many`,
        Krylov methods, :meth:`Lu.diagnostics`).

    Raises
    ------
    RuntimeError
        On a numerically zero pivot in exact mode.
    ValueError
        On an unsupported dtype, a non-square matrix or an invalid setting.

    Example
    -------
    .. code-block:: python

        f = rslab.lu(A)
        x = f.solve(b)
        r = f.gmres(b, tol=1e-10)               # the factor as preconditioner
        x, converged, iters, res, stop = rslab.lu(A, drop_tol=1e-2).gmres(b)
    """
    return _rslab.lu_factor(*_csc_parts(A, "lu"), settings=settings, **kwargs)


def klu(A, *, settings: KluSettings | None = None, **kwargs) -> Klu:
    """Factor a general matrix through the **KLU** path (circuit-shaped systems).

    Block triangular form (BTF) plus a per-block AMD-ordered Gilbert-Peierls
    left-looking LU with diagonal-preference pivoting, following KLU
    (Davis & Palamadai Natarajan 2010). Built for MNA / SPICE matrices:
    extremely sparse, unsymmetric, nearly triangular. The
    :meth:`Klu.refactor` fast path re-factors new values on the same pattern
    without symbolic work or pivot search.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The ``n x n`` system matrix (full, CSC after conversion).
    settings : KluSettings, optional
        A prepared :class:`KluSettings` object.
    **kwargs
        Any :class:`KluSettings` keyword (``pivot_tol``, ``row_scaling``,
        ``btf``, ``parallel``, ``interrupt``), overriding ``settings``.

    Returns
    -------
    Klu
        A reusable factor handle with :meth:`Klu.solve`, :meth:`Klu.solve_many`,
        :meth:`Klu.solve_transpose`, :meth:`Klu.refactor`, the Krylov methods
        and :meth:`Klu.diagnostics`.

    Raises
    ------
    RuntimeError
        If the matrix is structurally singular (no complete matching) or a
        block hits a numerically zero pivot.

    Example
    -------
    .. code-block:: python

        f = rslab.klu(A)                        # factor once
        x = f.solve(b)
        f.refactor(A2.data)                     # same pattern, new values
        x2 = f.solve(b)
        y = f.solve_transpose(b)                # A^T y = b on the same factors
    """
    return _rslab.klu_factor(*_csc_parts(A, "klu"), settings=settings, **kwargs)


# ---------------------------------------------------------------------------
# Symbolic analysis
# ---------------------------------------------------------------------------


def analyze(A, path: str = "auto", *, settings=None, **kwargs):
    """Symbolic analysis of the pattern of ``A``: analyze once, factor many.

    The analysis (ordering, elimination tree, supernodes; for KLU the block
    triangular form) depends only on the sparsity pattern, so a sequence of
    matrices with the same pattern (frequency sweeps, Newton iterations,
    parameter studies) pays it once and factors each value set through
    ``symbolic.factor(data)``.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The ``n x n`` matrix whose pattern (and, for the heuristic
        ordering pick, values) is analyzed.
    path : {'auto', 'ldlt', 'lu', 'klu'}, default 'auto'
        The factorization path: ``'ldlt'`` for symmetric matrices (the lower
        triangle is analyzed), ``'lu'`` for general ones, ``'klu'`` for
        circuit-shaped ones. ``'auto'`` picks ``'ldlt'`` when ``A`` is
        symmetric and ``'lu'`` otherwise.
    settings : Settings or KluSettings, optional
        Analysis-time settings (``ordering``, ``nemin``, ``relax``,
        ``reorder`` for LDL^T / LU; ``btf`` for KLU). Numeric settings given
        here become the defaults of ``factor``.
    **kwargs
        The same keywords, overriding ``settings``.

    Returns
    -------
    LdltSymbolic or LuSymbolic or KluSymbolic
        The analysis handle: fill and memory estimates, and ``factor(data)``.

    Example
    -------
    .. code-block:: python

        sym = rslab.analyze(A, path="lu", ordering="amd")
        print(sym.factor_nnz, sym.estimate_memory()["factor_mb"])
        for omega in frequencies:
            f = sym.factor((K + 1j * omega * C).data)    # same pattern
            x = f.solve(b)

    Note
    ----
    ``factor(data)`` takes the CSC value array in the analyzed pattern's
    order, i.e. of a matrix prepared the same way (sorted indices, summed
    duplicates). Build the sweep matrices from one pattern to guarantee it.
    """
    if path == "auto":
        path = "ldlt" if _is_symmetric(A) else "lu"
    if path == "ldlt":
        return _rslab.analyze_ldlt(*_csc_parts(A, "ldlt"), settings=settings, **kwargs)
    if path == "lu":
        return _rslab.analyze_lu(*_csc_parts(A, "lu"), settings=settings, **kwargs)
    if path == "klu":
        return _rslab.analyze_klu(*_csc_parts(A, "klu"), settings=settings, **kwargs)
    raise ValueError(f"path must be 'auto', 'ldlt', 'lu' or 'klu', got {path!r}")


# ---------------------------------------------------------------------------
# Krylov solvers
# ---------------------------------------------------------------------------


def gmres(A, b, M=None, *, tol: float = 1e-8, maxit: int = 400, restart: int | None = None,
          x0=None, recycle: Recycle | None = None) -> KrylovResult:
    """Flexible restarted GMRES on ``A x = b``, optionally preconditioned.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The operator (any square matrix; converted to CSC).
    b : ndarray, shape (n,)
        Right-hand side; cast to the operator's (or preconditioner's) dtype.
    M : Ldlt or Lu or Klu, optional
        A factor handle used as the preconditioner, e.g. an incomplete or
        low-precision factor of ``A`` or a factor of a nearby matrix.
        ``None`` runs unpreconditioned.
    tol : float, default 1e-8
        Relative residual target ``||b - A x|| <= tol * ||b||``.
    maxit : int, default 400
        Iteration budget.
    restart : int, optional
        Arnoldi basis size; by default chosen so the basis stays under 1 GiB
        (between 20 and 80).
    x0 : ndarray, optional
        Initial guess (warm start).
    recycle : Recycle, optional
        Deflation subspace carried across calls (needs ``M``; create it with
        ``M.recycle(k)``).

    Returns
    -------
    KrylovResult
        ``x``, ``converged``, ``iters``, ``final_res``, ``stop``; unpacks as
        a 5-tuple.

    Example
    -------
    .. code-block:: python

        M = rslab.lu(A, drop_tol=1e-3)                    # ILU-style preconditioner
        x, ok, iters, res, stop = rslab.gmres(A, b, M, tol=1e-10)
        r = rslab.gmres(A2, b, M)                         # M reused on a nearby A2
    """
    parts = _csc_parts(A, "lu")
    if M is None:
        b = np.ascontiguousarray(b, dtype=parts[3].dtype)
        return _rslab.gmres_plain(*parts, b, tol=tol, maxit=maxit, restart=restart, x0=x0)
    b = _match_dtype(b, M.dtype)
    return M.gmres(b, tol=tol, maxit=maxit, restart=restart, x0=x0, recycle=recycle,
                   operator=_cast_operator(parts, M.dtype))


def gmres_block(A, B, M=None, *, tol: float = 1e-8, maxit: int = 400,
                restart: int | None = None, x0=None) -> KrylovResult:
    """Block GMRES on ``A X = B`` for an ``n x nrhs`` block, optionally preconditioned.

    Parameters
    ----------
    A : scipy.sparse matrix
        The ``n x n`` operator.
    B : ndarray, shape (n, nrhs)
        Right-hand side block.
    M : factor handle, optional
        A factor used as the preconditioner.
    tol : float, default 1e-8
        Relative residual target per column.
    maxit : int, default 400
        Iteration budget.
    restart : int, optional
        Block Arnoldi basis size (see :func:`gmres`).
    x0 : ndarray, shape (n, nrhs), optional
        Initial guess.

    Returns
    -------
    KrylovResult
        ``x`` of shape ``(n, nrhs)``, ``converged``, ``iters``, ``final_res``
        (one residual per column), ``stop``.
    """
    parts = _csc_parts(A, "lu")
    if M is None:
        B = np.ascontiguousarray(B, dtype=parts[3].dtype)
        return _rslab.gmres_block_plain(*parts, B, tol=tol, maxit=maxit, restart=restart, x0=x0)
    B = _match_dtype(B, M.dtype)
    return M.gmres_block(B, tol=tol, maxit=maxit, restart=restart, x0=x0,
                         operator=_cast_operator(parts, M.dtype))


def cocg(A, b, M=None, *, tol: float = 1e-8, maxit: int = 400) -> KrylovResult:
    """Conjugate orthogonal conjugate gradient (COCG) on ``A x = b``.

    The short-recurrence Krylov method for **complex-symmetric**
    (``A = A^T``) and real symmetric operators: constant
    memory, one matrix-vector product per iteration, no restart.

    Parameters
    ----------
    A : scipy.sparse matrix
        The ``n x n`` operator.
    b : ndarray, shape (n,)
        Right-hand side.
    M : factor handle, optional
        A factor (``ldlt``, ``lu``, ``klu``, typically incomplete or
        low-rank) used as the preconditioner.
    tol : float, default 1e-8
        Relative residual target ``||b - A x|| <= tol * ||b||``.
    maxit : int, default 400
        Iteration budget.

    Returns
    -------
    KrylovResult
        ``x``, ``converged``, ``iters``, ``final_res``, ``stop``.
    """
    parts = _csc_parts(A, "lu")
    if M is None:
        b = np.ascontiguousarray(b, dtype=parts[3].dtype)
        return _rslab.cocg_plain(*parts, b, tol=tol, maxit=maxit)
    b = _match_dtype(b, M.dtype)
    return M.cocg(b, tol=tol, maxit=maxit, operator=_cast_operator(parts, M.dtype))


def cocr(A, b, M=None, *, tol: float = 1e-8, maxit: int = 400) -> KrylovResult:
    """Conjugate orthogonal conjugate residual (COCR) on ``A x = b``.

    The minimal-residual sibling of :func:`cocg` for complex-symmetric
    operators (smoother residual history).

    Parameters
    ----------
    A : scipy.sparse matrix
        The ``n x n`` operator.
    b : ndarray, shape (n,)
        Right-hand side.
    M : factor handle, optional
        A factor used as the preconditioner.
    tol : float, default 1e-8
        Relative residual target.
    maxit : int, default 400
        Iteration budget.

    Returns
    -------
    KrylovResult
        ``x``, ``converged``, ``iters``, ``final_res``, ``stop``.
    """
    parts = _csc_parts(A, "lu")
    if M is None:
        b = np.ascontiguousarray(b, dtype=parts[3].dtype)
        return _rslab.cocr_plain(*parts, b, tol=tol, maxit=maxit)
    b = _match_dtype(b, M.dtype)
    return M.cocr(b, tol=tol, maxit=maxit, operator=_cast_operator(parts, M.dtype))


def _cast_operator(parts, dtype_name: str):
    """The operator tuple with its values cast to the preconditioner's dtype."""
    n, indptr, indices, data = parts
    return (n, indptr, indices, np.ascontiguousarray(data, dtype=np.dtype(dtype_name)))


# ---------------------------------------------------------------------------
# One-shot
# ---------------------------------------------------------------------------


def _is_symmetric(A, tol: float = 1e-12) -> bool:
    """Structural + value symmetry test (``A - A^T`` small relative to ``max|A|``)."""
    sp = _require_scipy()
    A = sp.csc_matrix(A)
    if A.shape[0] != A.shape[1]:
        return False
    d = (A - A.T).tocsc().data
    if d.size == 0:
        return True
    return float(abs(d).max()) <= tol * (float(abs(A).max()) or 1.0)


def _match_dtype(b: np.ndarray, dtype_name: str) -> np.ndarray:
    """``b`` as a contiguous array in the factor's dtype."""
    return np.ascontiguousarray(b, dtype=np.dtype(dtype_name))


def spsolve(A, b, *, symmetric: bool | None = None, refine: int = 0, **kwargs):
    """One-shot solve of ``A x = b`` (factor, solve, discard).

    Detects symmetry, factors through :func:`ldlt` or :func:`lu`, solves,
    and drops the factor. For repeated solves against one matrix keep the
    handle instead (:func:`ldlt` / :func:`lu` / :func:`klu`).

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The ``n x n`` system matrix.
    b : ndarray
        Right-hand side: a 1-D vector of length ``n`` or a 2-D ``n x nrhs``
        block. Cast to the factor's dtype automatically.
    symmetric : bool, optional
        Force the symmetric ``L D L^T`` path (``True``) or the
        unsymmetric ``L U`` path (``False``). When omitted, symmetry is
        auto-detected from ``A`` (a structural + value test).
    refine : int, default 0
        Steps of iterative refinement against the original matrix, per
        right-hand side. Meaningful with ``preconditioner=...`` /
        ``drop_tol=...``, where the factor is inexact.
    **kwargs
        Forwarded to :func:`ldlt` / :func:`lu` (any :class:`Settings` keyword).

    Returns
    -------
    ndarray
        The solution, matching the shape of ``b``.

    Raises
    ------
    ValueError
        If ``b`` is neither 1-D nor 2-D.
    RuntimeError
        If the (exact-mode) factorization hits a zero pivot; set
        ``preconditioner=...`` to use never-fail static pivoting.

    Example
    -------
    .. code-block:: python

        x = rslab.spsolve(A, b)                          # auto-detects symmetry
        X = rslab.spsolve(A, np.random.rand(n, 5))       # 5 right-hand sides
        x = rslab.spsolve(A, b, preconditioner=1e-4, refine=2)
    """
    if symmetric is None:
        symmetric = _is_symmetric(A)
    f = ldlt(A, **kwargs) if symmetric else lu(A, **kwargs)
    rhs = _match_dtype(b, f.dtype)
    if rhs.ndim == 1:
        return f.solve(rhs, refine)
    if rhs.ndim == 2:
        if refine:
            # Multi-RHS refinement: refine each column independently.
            cols = [f.solve(np.ascontiguousarray(rhs[:, c]), refine) for c in range(rhs.shape[1])]
            return np.stack(cols, axis=1)
        return f.solve_many(rhs)
    raise ValueError("b must be 1-D (vector) or 2-D (n x nrhs block)")
