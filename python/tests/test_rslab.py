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
