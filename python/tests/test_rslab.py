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
    # solution of the full (unsymmetric) system. gmres_block now returns the full
    # diagnostics tuple (X, converged, iters, final_res, stop).
    rng = np.random.default_rng(5)
    n = 400
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 10
    A = A.tocsc()
    B = rng.standard_normal((n, 6))
    X, converged, iters, final_res, stop = rslab.lu(A, drop_tol=1e-2).gmres_block(
        B, tol=1e-10, maxit=400, restart=80
    )
    assert X.shape == (n, 6)
    assert converged is True
    assert stop == "converged"
    assert isinstance(iters, int) and iters > 0
    assert final_res.shape == (6,)
    assert np.all(final_res <= 1e-10)
    for c in range(6):
        assert _residual(A, X[:, c], B[:, c]) < 1e-8


def test_ldlt_gmres_block_complex():
    # Complex-symmetric multi-RHS via block GMRES preconditioned by the LDLᵀ factor.
    A = _complex_sym(200)
    B = (np.random.default_rng(6).standard_normal((200, 4))
         + 1j * np.random.default_rng(7).standard_normal((200, 4)))
    X, converged, iters, final_res, stop = rslab.ldlt(A).gmres_block(
        B.astype(np.complex128), tol=1e-10
    )
    assert X.shape == (200, 4)
    assert converged is True
    assert stop == "converged"
    assert final_res.shape == (4,)
    for c in range(4):
        assert _residual(A, X[:, c], B[:, c]) < 1e-8


def test_lu_gmres_single_rhs():
    # Single-RHS gmres exposed on the Lu factor, returning (x, converged, iters,
    # final_res, stop). Incomplete factor as preconditioner recovers the true solution.
    rng = np.random.default_rng(8)
    n = 300
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 10
    A = A.tocsc()
    b = rng.standard_normal(n)
    x, converged, iters, final_res, stop = rslab.lu(A, drop_tol=1e-2).gmres(
        b, tol=1e-10, maxit=400, restart=80
    )
    assert x.shape == (n,)
    assert converged is True
    assert stop == "converged"
    assert isinstance(iters, int) and iters > 0
    assert float(final_res) <= 1e-10
    assert _residual(A, x, b) < 1e-8


def test_gmres_reports_non_convergence():
    # With maxit too small to converge, the solver must return converged=False and
    # a truthful residual rather than silently passing off a bad iterate as good.
    rng = np.random.default_rng(9)
    n = 300
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 10
    A = A.tocsc()
    b = rng.standard_normal(n)
    # A single iteration cannot reach 1e-12 with the identity-ish weak setup here.
    x, converged, iters, final_res, stop = rslab.lu(A, drop_tol=5e-1).gmres(
        b, tol=1e-12, maxit=1, restart=1
    )
    assert converged is False
    assert stop == "max_iter"
    assert float(final_res) > 1e-12


def test_gmres_warm_start_cuts_iterations():
    # Warm start (issue #5): a sequence of related systems with a slowly rotating
    # RHS. Seeding each solve with the previous solution must cut the total
    # iteration count vs cold-starting from zero every time.
    rng = np.random.default_rng(11)
    n = 400
    # Weakly preconditioned so each solve takes many iterations (warm start has
    # room to help); unsymmetric operator.
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 4.0
    A = A.tocsc()
    fac = rslab.lu(A, drop_tol=8e-1)  # deliberately weak preconditioner
    b0 = rng.standard_normal(n)
    b1 = rng.standard_normal(n)
    steps = 8

    def bk(k):
        th = 3e-4 * k
        return np.cos(th) * b0 + np.sin(th) * b1

    cold_total = 0
    for k in range(steps):
        x, converged, iters, _, _ = fac.gmres(bk(k), tol=1e-9, maxit=4000, restart=60)
        assert converged
        cold_total += iters

    warm_total = 0
    prev = None
    for k in range(steps):
        x, converged, iters, _, _ = fac.gmres(bk(k), tol=1e-9, maxit=4000, restart=60, x0=prev)
        assert converged
        warm_total += iters
        prev = x

    assert warm_total < 0.7 * cold_total, f"cold={cold_total}, warm={warm_total}"


def _stagnating_operator(m=18, rank=3, seed=4):
    # A convection-diffusion grid plus a global near-low-rank coupling U Uᵀ that an
    # *incomplete* LU factor cannot capture - so the preconditioned system keeps a
    # small near-invariant cluster and restarted GMRES genuinely stagnates. This is
    # the regime GCRO-DR recycling targets.
    n = m * m
    rng = np.random.default_rng(seed)
    I = sp.eye(m)
    main = 2.0 * np.ones(m)
    up = -0.9 * np.ones(m - 1)
    lo = -0.05 * np.ones(m - 1)
    T = sp.diags([lo, main, up], [-1, 0, 1])
    S = sp.kron(I, T) + sp.kron(T, I)
    U = rng.standard_normal((n, rank)) * 0.6
    G = sp.csc_matrix(U @ U.T)
    return (S + G).tocsc(), rng


def test_gmres_recycle_cross_solve_beats_warm_and_cold():
    # GCRO-DR recycling (issue #5) across a sequence of related solves (slowly
    # rotating RHS, weak incomplete factor → stagnation). Total iterations must
    # order recycled < warm-start < cold, with recycled a clear margin below cold.
    A, rng = _stagnating_operator(m=18, rank=3, seed=4)
    n = A.shape[0]
    fac = rslab.lu(A, drop_tol=0.4)
    b0 = rng.standard_normal(n)
    b1 = rng.standard_normal(n)
    steps, tol, restart, maxit = 10, 1e-9, 10, 200000

    def bk(k):
        th = 0.02 * k
        return np.cos(th) * b0 + np.sin(th) * b1

    def run(mode, kdim=10):
        total = 0
        prev = None
        rec = fac.recycle(kdim) if mode == "rec" else None
        for k in range(steps):
            kw = {}
            if mode in ("warm", "rec") and prev is not None:
                kw["x0"] = prev
            if mode == "rec":
                kw["recycle"] = rec
            x, converged, iters, _, _ = fac.gmres(
                bk(k), tol=tol, maxit=maxit, restart=restart, **kw
            )
            assert converged, f"{mode} solve {k} did not converge"
            assert _residual(A, x, bk(k)) < 1e-7
            total += iters
            prev = x
        return total, rec

    cold, _ = run("cold")
    warm, _ = run("warm")
    recycled, rec = run("rec")
    print(f"\n[python cross-solve] cold={cold} warm={warm} recycled={recycled}")
    assert rec.active > 0, "recycle handle was never populated"
    assert rec.dtype == "float64"
    assert warm < cold, f"warm ({warm}) !< cold ({cold})"
    assert recycled < warm, f"recycled ({recycled}) !< warm ({warm})"
    assert recycled < 0.85 * cold, f"recycled ({recycled}) not clearly below cold ({cold})"


def test_gmres_recycle_complex_handle_and_compose():
    # End-to-end handle exercise on a complex system: the recycle dtype tracks the
    # factor, composes with x0, refreshes across solves, and clear() empties it.
    A, rng = _stagnating_operator(m=14, rank=2, seed=6)
    A = A.astype(np.complex128)
    A.data += 1j * 0.05 * A.data.real
    A = A.tocsc()
    n = A.shape[0]
    fac = rslab.lu(A, drop_tol=0.4)
    b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)

    rec = fac.recycle(8)
    assert rec.k == 8 and rec.active == 0 and rec.dtype == "complex128"

    x1, c1, _, _, _ = fac.gmres(b, tol=1e-9, maxit=50000, restart=10, recycle=rec)
    assert c1 and _residual(A, x1, b) < 1e-7
    assert rec.active > 0, "handle not refreshed after first solve"

    # Compose recycle with an x0 warm start on a related RHS.
    b2 = b * (0.99 + 0.01j)
    x2, c2, _, _, _ = fac.gmres(b2, tol=1e-9, maxit=50000, restart=10, x0=x1, recycle=rec)
    assert c2 and _residual(A, x2, b2) < 1e-7

    rec.clear()
    assert rec.active == 0 and rec.k == 8


def test_gmres_recycle_dtype_mismatch_raises():
    # A recycle handle built from one factor's dtype cannot be used with a solve of
    # a different dtype.
    A, rng = _stagnating_operator(m=10, rank=1, seed=1)
    fac_r = rslab.lu(A.tocsc(), drop_tol=0.5)  # float64
    fac_c = rslab.lu(A.astype(np.complex128).tocsc(), drop_tol=0.5)  # complex128
    rec_c = fac_c.recycle(6)  # complex128 handle
    b = rng.standard_normal(A.shape[0])
    with pytest.raises(ValueError):
        fac_r.gmres(b, recycle=rec_c)  # float64 solve, complex128 handle


def test_gmres_block_warm_start_matches_and_helps():
    # Warm start also works for the block (multi-RHS) path: seeding with the exact
    # solution converges immediately (0 iters), and the result still matches a
    # cold solve.
    rng = np.random.default_rng(12)
    n = 300
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 10
    A = A.tocsc()
    fac = rslab.lu(A, drop_tol=1e-2)
    B = rng.standard_normal((n, 4))
    X, converged, iters, _, _ = fac.gmres_block(B, tol=1e-10, maxit=400, restart=80)
    assert converged
    # Re-solve warm-started from the converged solution: must need no iterations.
    X2, conv2, iters2, _, _ = fac.gmres_block(B, tol=1e-10, maxit=400, restart=80, x0=X)
    assert conv2
    assert iters2 == 0, f"warm start from exact solution should take 0 iters, got {iters2}"
    for c in range(4):
        assert _residual(A, X2[:, c], B[:, c]) < 1e-8


def test_gmres_explicit_restart_is_honored():
    # Issue #12: `restart` is the per-cycle Krylov dimension (and the up-front
    # n*(cols)*(restart+1) basis size). It must be used verbatim, never silently
    # replaced by the adaptive default. Cap each solve to a single cycle (maxit =
    # restart) on a weakly-preconditioned system that needs more than a few steps:
    # a short restart then only explores a small Krylov space and leaves a larger
    # residual, while a long restart drives the residual far lower. Directly
    # observable proof the value is honored, independent of the convergence rate.
    rng = np.random.default_rng(23)
    n = 300
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 4.0
    A = A.tocsc()
    b = rng.standard_normal(n)
    fac = rslab.lu(A, drop_tol=8e-1)  # deliberately weak preconditioner

    # One cycle each (maxit == restart), tol unreachable so neither exits early.
    _, conv_short, iters_short, res_short, _ = fac.gmres(b, tol=1e-14, maxit=4, restart=4)
    _, _, _, res_long, _ = fac.gmres(b, tol=1e-14, maxit=40, restart=40)
    assert not conv_short and iters_short == 4, "restart=4 must cap the cycle at 4 steps"
    assert res_long < res_short, (
        f"explicit restart ignored? res(restart=4)={res_short}, res(restart=40)={res_long}"
    )


def test_gmres_default_restart_is_adaptive():
    # With `restart` unspecified the binding picks an adaptive default (capped so
    # the up-front basis stays under a memory budget, clamped to [20, 80]). For a
    # modest problem the cap is not binding, so the default still converges - and
    # matches an explicit restart=80 on the same system.
    rng = np.random.default_rng(24)
    n = 300
    A = sp.random(n, n, density=5.0 / n, format="csc", random_state=rng) + sp.eye(n) * 10
    A = A.tocsc()
    b = rng.standard_normal(n)
    B = rng.standard_normal((n, 4))
    fac = rslab.lu(A, drop_tol=1e-2)

    x, conv, iters, res, _ = fac.gmres(b, tol=1e-10, maxit=400)  # restart defaulted
    assert conv and _residual(A, x, b) < 1e-8
    xe, conve, iterse, _, _ = fac.gmres(b, tol=1e-10, maxit=400, restart=80)
    assert conve and iters == iterse  # cap not binding here -> same as explicit 80

    X, convb, _, _, _ = fac.gmres_block(B, tol=1e-10, maxit=400)  # block, restart defaulted
    assert convb
    for c in range(4):
        assert _residual(A, X[:, c], B[:, c]) < 1e-8


# ---------------------------------------------------------------------------
# KLU path
# ---------------------------------------------------------------------------


def _circuit(n, seed=7):
    """Sparse, unsymmetric, column-diagonally-dominant MNA-like matrix."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=4.0 / n, format="csc", random_state=rng)
    A = A - sp.diags(A.diagonal())
    colsum = np.abs(A).sum(axis=0).A1
    A = A + sp.diags(colsum + 1.0 + rng.random(n))
    return A.tocsc()


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_klu_factor_solve(dtype):
    A = _circuit(300).astype(dtype)
    b = np.arange(300).astype(dtype)
    f = rslab.klu(A)
    assert f.dtype == np.dtype(dtype).name
    assert f.n == 300
    assert f.n_blocks >= 1
    assert f.n_perturbed == 0
    assert f.factor_nnz > 0
    x = f.solve(b)
    assert _residual(A, x, b) < 1e-10
    X = f.solve_many(np.column_stack([b, 2 * b]))
    assert _residual(A, X[:, 1], 2 * b) < 1e-10


def test_klu_refactor_sweep():
    A = _circuit(300, seed=9)
    b = np.ones(300)
    f = rslab.klu(A)
    x0 = f.solve(b)
    # same pattern, new values: numeric-only refactor
    A2 = A.copy()
    A2.data *= 1.7
    f.refactor(A2.data)
    x2 = f.solve(b)
    assert _residual(A2, x2, b) < 1e-10
    assert np.allclose(x2, x0 / 1.7)
    # wrong length must raise, factor must recover via full refactor
    with pytest.raises(ValueError):
        f.refactor(A2.data[:-1])


def test_klu_structurally_singular_raises():
    # empty column -> no complete matching, detected before numeric work
    A = sp.csc_matrix((np.array([1.0, 2.0]), (np.array([0, 1]), np.array([0, 1]))), shape=(3, 3))
    with pytest.raises(RuntimeError, match="structurally singular"):
        rslab.klu(A)


def test_klu_deterministic():
    A = _circuit(250, seed=11)
    b = np.linspace(-1, 1, 250)
    x1 = rslab.klu(A).solve(b)
    x2 = rslab.klu(A).solve(b)
    assert np.array_equal(x1, x2)  # bit-identical


def test_klu_gmres_composes():
    A = _circuit(300, seed=13)
    b = np.ones(300)
    f = rslab.klu(A)
    x, conv, iters, res, stop = f.gmres(b, tol=1e-12)
    assert conv and stop == "converged"
    assert iters <= 2  # exact preconditioner
    assert _residual(A, x, b) < 1e-10
