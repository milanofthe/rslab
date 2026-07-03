"""RSLAB - a pure-Rust sparse direct solver and preconditioner for NumPy/SciPy.

A thin, allocation-light wrapper over the RSLAB Rust core: a drop-in
alternative to :func:`scipy.sparse.linalg.spsolve` / Intel MKL PARDISO for the
factor-once, solve-many workloads that dominate FEM and method-of-moments codes.
**Symmetric** matrices are factored by a real / complex-symmetric
:math:`L D L^{\\mathsf{T}}` (Bunch-Kaufman) method and **general** matrices by an
unsymmetric :math:`L U`; either factor then solves against one or many
right-hand sides.

The solver is **type-agnostic** - the matrix ``dtype`` selects the arithmetic
field, so ``float64`` / ``float32`` run the real path and ``complex128`` /
``complex64`` the complex path through the *same* call, at half the memory for
the 32-bit fields.

Two usage modes
---------------
The one-shot :func:`spsolve` (auto-detects symmetry, factors, solves, discards),
and the explicit factor handle from :func:`ldlt` / :func:`lu` when the same
matrix is solved against many right-hand sides.

Examples
--------
.. code-block:: python

    import numpy as np, scipy.sparse as sp, rslab

    A = sp.random(2000, 2000, density=1e-3, format="csc") + sp.eye(2000) * 10
    A = A + A.T                       # make it symmetric
    b = np.random.rand(2000)

    x = rslab.spsolve(A, b)           # one-shot: factor + solve + discard

    f = rslab.ldlt(A)                 # factor once ...
    x1 = f.solve(b)                   # ... solve many
    X  = f.solve_many(np.random.rand(2000, 8))   # 8 right-hand sides at once

The factor configuration is passed as keyword arguments (see :func:`ldlt` /
:func:`lu`): ``threads``, ``preconditioner``, ``drop_tol``, ``method``,
``memory``, ``force_accept``.

Notes
-----
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
    """Factor a **symmetric** matrix as :math:`P^{\\mathsf{T}} A P = L D L^{\\mathsf{T}}`.

    A supernodal Bunch-Kaufman :math:`L D L^{\\mathsf{T}}` factorization with a
    fill-reducing ordering :math:`P`, for real symmetric
    (``float64`` / ``float32``) and **complex-symmetric** (``complex128`` /
    ``complex64``, i.e. :math:`A = A^{\\mathsf{T}}`, *not* Hermitian) matrices; the
    ``dtype`` selects the path. Only the lower triangle is read (extracted
    automatically), so ``A`` may be stored full or triangular. Returns a reusable
    factor handle - factor once, then :meth:`Ldlt.solve` against as many
    right-hand sides as needed.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The symmetric :math:`n \\times n` system matrix. Converted to CSC and its
        lower triangle taken; duplicate entries are summed.
    threads : int, optional
        Worker-thread budget for the (scoped) factorization pool. ``None``
        (default) uses the **auto** per-matrix predictor - thin / tiny systems
        stay low where extra threads only regress, larger BLAS-3-rich systems
        scale up - **capped at 4 workers**, the pareto-optimal
        throughput-per-core point. An explicit integer pins the count
        (``0`` = all logical cores); pin a small value for many concurrent solves
        sharing the machine. The factor is bit-identical either way.
    preconditioner : float, optional
        Enable never-fail **static pivoting**: any pivot with magnitude below this
        absolute floor (typically ``eps_rel * ‖A‖``) is lifted to it, so the
        stored factor is of a perturbed :math:`A + E`. The factorization then
        never fails on a (near-)singular pivot, at the cost of an inexact factor -
        recover full accuracy by driving ``solve(b, refine=k)``, which does ``k``
        steps of iterative refinement against the original ``A``. A good starting
        floor is ``1e-4``.
    drop_tol : float, optional
        Incomplete-factorization threshold. Fill entries whose magnitude is below
        this value *relative to the column* are discarded, trading factor accuracy
        for memory and turning the factor into an ILU-style preconditioner (pair
        with ``refine`` or an outer Krylov iteration). ``None`` keeps the complete
        factor.
    method : {'left_looking', 'multifrontal'}, default 'left_looking'
        Numeric factorization schedule. Both produce the **same** factor and
        differ only in the transient-memory / parallel-scheduling profile:
        ``'left_looking'`` has the lower transient working set, ``'multifrontal'``
        exposes more front-level parallelism.
    memory : {'low', 'eager'}, default 'low'
        Factor-emit strategy. ``'low'`` frees each front as soon as it is emitted
        (lower peak RSS); ``'eager'`` keeps them resident. Bit-identical factors
        either way.
    force_accept : bool, default False
        In exact mode (no ``preconditioner``), accept tiny pivots at face value
        instead of raising on rank deficiency. Ignored when ``preconditioner`` is
        set. Use only when you know the system is well-conditioned.

    Returns
    -------
    Ldlt
        A reusable factor handle exposing :meth:`~Ldlt.solve`,
        :meth:`~Ldlt.solve_many`, :meth:`~Ldlt.gmres` (preconditioned single-RHS
        iterative solve), :meth:`~Ldlt.gmres_block` (preconditioned multi-RHS
        iterative solve), and the read-only attributes ``n``,
        ``factor_nnz`` (fill), ``n_perturbed``, ``inertia`` and ``dtype``.

    Raises
    ------
    RuntimeError
        If a pivot is numerically zero in exact mode (the matrix is rank
        deficient). Set ``preconditioner=...`` (recommended) or
        ``force_accept=True`` to proceed.

    See Also
    --------
    lu : the unsymmetric counterpart, :math:`P^{\\mathsf{T}} A P = L U`.
    spsolve : one-shot factor-and-solve with automatic symmetry detection.

    Notes
    -----
    The complex path uses the *unconjugated* bilinear form throughout, which is
    the correct geometry for complex-symmetric (:math:`A = A^{\\mathsf{T}}`)
    operators such as those from time-harmonic Maxwell / MoM discretizations -
    it is **not** a Hermitian solver.

    Examples
    --------
    .. code-block:: python

        import numpy as np, scipy.sparse as sp, rslab

        A = sp.random(5000, 5000, density=5e-4, format="csc")
        A = A + A.T + sp.eye(5000) * 20          # symmetric, diagonally dominant

        f = rslab.ldlt(A)                        # exact factor
        x = f.solve(np.random.rand(5000))
        print(f.factor_nnz, f.inertia)           # fill and (n+, n-, n0)

        # Near-singular system: static pivoting + iterative refinement.
        g = rslab.ldlt(A, preconditioner=1e-4)
        x = g.solve(np.random.rand(5000), refine=2)

    References
    ----------
    .. [1] Bunch, J. R., & Kaufman, L. (1977). "Some stable methods for
           calculating inertia and solving symmetric linear systems."
           *Mathematics of Computation*, 31(137), 163-179.
           :doi:`10.1090/S0025-5718-1977-0428694-0`
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
    """Factor a **general** (unsymmetric) matrix as :math:`P^{\\mathsf{T}} A P = L U`.

    A supernodal multifrontal :math:`L U` factorization with a fill-reducing
    ordering :math:`P`, for real and complex dtypes (the ``dtype`` selects the
    path). The full matrix - both triangles - is used, so this is the path for any
    non-symmetric operator (e.g. convection-diffusion, non-reciprocal MoM).
    Returns a reusable factor handle; factor once, then :meth:`Lu.solve` against
    many right-hand sides.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The general :math:`n \\times n` system matrix. Converted to CSC; duplicate
        entries are summed.
    threads : int, optional
        Worker-thread budget for the (scoped) factorization pool. ``None``
        (default) uses the **auto** per-matrix predictor, **capped at 4 workers**
        (the pareto-optimal throughput-per-core point); an explicit integer pins
        the count (``0`` = all logical cores). The factor is bit-identical either
        way.
    preconditioner : float, optional
        Enable never-fail **static pivoting**: a pivot below this absolute floor is
        lifted, so the stored factor is of a perturbed :math:`A + E` and the
        factorization never fails on a (near-)singular pivot. Recover accuracy with
        ``solve(b, refine=k)`` (iterative refinement against the original ``A``). A
        good starting floor is ``1e-4``.
    drop_tol : float, optional
        Incomplete-factorization threshold: fill below this value (relative to the
        column) is dropped, trading accuracy for memory and yielding an ILU-style
        preconditioner. ``None`` keeps the complete factor.
    method : {'left_looking', 'multifrontal'}, default 'left_looking'
        Numeric factorization schedule. Both produce the **same** factor and differ
        only in transient memory / parallel scheduling.
    memory : {'low', 'eager'}, default 'low'
        Factor-emit strategy; ``'low'`` frees each front as emitted for a lower
        peak RSS. Bit-identical either way.
    force_accept : bool, default False
        In exact mode, accept tiny pivots instead of raising on rank deficiency.
        Ignored when ``preconditioner`` is set.

    Returns
    -------
    Lu
        A reusable factor handle exposing :meth:`~Lu.solve`,
        :meth:`~Lu.solve_many`, :meth:`~Lu.gmres` (preconditioned single-RHS
        iterative solve), :meth:`~Lu.gmres_block` (preconditioned multi-RHS
        iterative solve), and the read-only attributes ``n``, ``factor_nnz`` (fill
        in ``L + U``), ``n_perturbed`` and ``dtype``.

    Raises
    ------
    RuntimeError
        If a pivot is numerically zero in exact mode (rank-deficient matrix). Set
        ``preconditioner=...`` (recommended) or ``force_accept=True`` to proceed.

    See Also
    --------
    ldlt : the symmetric counterpart, :math:`P^{\\mathsf{T}} A P = L D L^{\\mathsf{T}}`.
    spsolve : one-shot factor-and-solve with automatic symmetry detection.

    Examples
    --------
    .. code-block:: python

        import numpy as np, scipy.sparse as sp, rslab

        A = sp.random(4000, 4000, density=1e-3, format="csc") + sp.eye(4000) * 10
        f = rslab.lu(A)                          # unsymmetric factor
        x = f.solve(np.random.rand(4000))
        X = f.solve_many(np.random.rand(4000, 4))

    References
    ----------
    .. [1] Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM,
           chs. 5-6 (multifrontal / supernodal LU). :doi:`10.1137/1.9780898718881`
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
    """One-shot solve of :math:`A x = b` (factor, solve, discard).

    The convenience entry point mirroring :func:`scipy.sparse.linalg.spsolve`:
    detects whether ``A`` is symmetric, factors it with :func:`ldlt` or
    :func:`lu` accordingly, solves, and drops the factor. Use :func:`ldlt` /
    :func:`lu` directly when the same matrix is solved repeatedly, to reuse the
    (expensive) factorization.

    Parameters
    ----------
    A : scipy.sparse matrix or array-like
        The :math:`n \\times n` system matrix.
    b : array-like
        Right-hand side: a 1-D vector of length ``n`` or a 2-D ``n x nrhs`` block.
        Cast to the factor's dtype automatically.
    symmetric : bool, optional
        Force the symmetric :math:`L D L^{\\mathsf{T}}` path (``True``) or the
        unsymmetric :math:`L U` path (``False``). When omitted, symmetry is
        auto-detected from ``A`` (a structural + value test); pass it explicitly to
        skip the check or to override a borderline case.
    refine : int, default 0
        Steps of iterative refinement against the original matrix, applied per
        right-hand side. Meaningful together with ``preconditioner=...`` /
        ``drop_tol=...``, where the factor is inexact.
    **opts
        Forwarded to :func:`ldlt` / :func:`lu`: ``threads``, ``preconditioner``,
        ``drop_tol``, ``method``, ``memory``, ``force_accept``.

    Returns
    -------
    numpy.ndarray
        The solution, matching the shape of ``b`` (1-D for a vector, ``n x nrhs``
        for a block).

    Raises
    ------
    ValueError
        If ``b`` is neither 1-D nor 2-D.
    RuntimeError
        If the (exact-mode) factorization hits a zero pivot; set
        ``preconditioner=...`` to use never-fail static pivoting.

    See Also
    --------
    ldlt, lu : reusable factor handles for the factor-once, solve-many workflow.

    Examples
    --------
    .. code-block:: python

        import numpy as np, scipy.sparse as sp, rslab

        A = sp.random(3000, 3000, density=1e-3, format="csc") + sp.eye(3000) * 10
        b = np.random.rand(3000)
        x = rslab.spsolve(A, b)                          # auto-detects symmetry

        X = rslab.spsolve(A, np.random.rand(3000, 5))    # 5 right-hand sides

        # Never-fail static pivoting + refinement for an indefinite system:
        x = rslab.spsolve(A, b, preconditioner=1e-4, refine=2)
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
