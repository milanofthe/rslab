"""Smoke tests for the RSLAB Python bindings.

Run with: maturin develop && pytest python/tests
"""

import numpy as np
import scipy.sparse as sp
import pytest

import rslab


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng)
    A = (A + A.T) + sp.eye(n) * (n)  # diagonally dominant -> SPD
    return A.tocsc()


def _complex_sym(n, seed=1):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng)
    A = A + A.T + sp.eye(n) * n
    A = A.astype(np.complex128)
    A.data += 1j * (0.3 * A.data.real)  # keep it complex-symmetric
    return A.tocsc()


def _residual(A, x, b):
    return np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1.0)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ldlt_real(dtype):
    A = _spd(200).astype(dtype)
    b = np.ones(200, dtype=dtype)
    f = rslab.ldlt(A)
    assert f.dtype == np.dtype(dtype).name
    x = f.solve(b)
    assert _residual(A, x, b) < 1e-4


def test_ldlt_complex():
    A = _complex_sym(200)
    b = np.ones(200, dtype=np.complex128)
    f = rslab.ldlt(A)
    assert f.dtype == "complex128"
    x = f.solve(b)
    assert _residual(A, x, b) < 1e-8
    assert f.n == 200
    assert f.factor_nnz > 0


def test_lu_unsymmetric():
    rng = np.random.default_rng(2)
    n = 200
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * n
    A = A.tocsc()  # not symmetric
    b = np.arange(n, dtype=np.float64)
    f = rslab.lu(A)
    x = f.solve(b)
    assert _residual(A, x, b) < 1e-8


def test_spsolve_autodetect():
    A = _spd(150)
    b = np.random.default_rng(3).standard_normal(150)
    x = rslab.spsolve(A, b)
    assert _residual(A, x, b) < 1e-6


def test_solve_many():
    A = _spd(120)
    B = np.random.default_rng(4).standard_normal((120, 5))
    f = rslab.ldlt(A)
    X = f.solve_many(np.ascontiguousarray(B))
    assert X.shape == (120, 5)
    for c in range(5):
        assert _residual(A, X[:, c], B[:, c]) < 1e-6


def test_preconditioner_refine():
    # An indefinite-ish system where static pivoting + refinement is the recipe.
    A = _spd(200)
    b = np.ones(200)
    f = rslab.ldlt(A, preconditioner=1e-4)
    x = f.solve(b, refine=20)
    assert _residual(A, x, b) < 1e-6


def test_lu_gmres_block_preconditioned():
    # Incomplete factor as preconditioner, block GMRES drives all RHS to the true
    # solution of the full (unsymmetric) system.
    rng = np.random.default_rng(5)
    n = 400
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 10
    A = A.tocsc()
    B = rng.standard_normal((n, 6))
    X = rslab.lu(A, drop_tol=1e-2).gmres_block(B, tol=1e-10, maxit=400, restart=80)
    assert X.shape == (n, 6)
    for c in range(6):
        assert _residual(A, X[:, c], B[:, c]) < 1e-8


def test_ldlt_gmres_block_complex():
    # Complex-symmetric multi-RHS via block GMRES preconditioned by the LDLᵀ factor.
    A = _complex_sym(200)
    B = (np.random.default_rng(6).standard_normal((200, 4))
         + 1j * np.random.default_rng(7).standard_normal((200, 4)))
    X = rslab.ldlt(A).gmres_block(B.astype(np.complex128), tol=1e-10)
    assert X.shape == (200, 4)
    for c in range(4):
        assert _residual(A, X[:, c], B[:, c]) < 1e-8
