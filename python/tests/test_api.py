"""The configuration objects, symbolic reuse, Krylov entry points, logging
and the API reference (kept in sync with the docstrings)."""

import logging
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import rslab


def _spd(n, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.02, format="csc", random_state=rng)
    A = (A + A.T) + sp.eye(n) * n
    return A.astype(dtype).tocsc()


def _general(n, seed=1, dtype=np.float64):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.02, format="csc", random_state=rng) + sp.eye(n) * n
    return A.astype(dtype).tocsc()


def _circuit(n, seed=2):
    rng = np.random.default_rng(seed)
    d = sp.diags([rng.uniform(1, 2, n)], [0])
    off = sp.random(n, n, density=3.0 / n, format="csc", random_state=rng) * 0.1
    return (d + off).tocsc()


def _res(A, x, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_roundtrip_and_repr():
    s = rslab.Settings(threads=2, ordering="metis", preconditioner=1e-4, relax=(128, 32), blr=1e-6)
    d = s.to_dict()
    assert d["threads"] == 2
    assert d["ordering"] == "metis"
    assert d["preconditioner"] == 1e-4
    assert d["relax"] == (128, 32)
    assert d["blr"] == 1e-6
    assert "Settings(" in repr(s) and "ordering='metis'" in repr(s)
    # Defaults: heuristic pick, no explicit ordering.
    assert rslab.Settings().to_dict()["ordering"] is None
    assert rslab.Settings().to_dict()["threads"] is None


def test_lu_matching_setting():
    assert rslab.Settings().to_dict()["matching"] is True
    assert rslab.Settings(matching=False).to_dict()["matching"] is False
    A = _general(200)
    assert rslab.lu(A).diagnostics()["decisions"]["scaling"] == "Mc64RowMatching"
    assert rslab.lu(A, matching=False).diagnostics()["decisions"]["scaling"] == "TwoSidedRowCol"


def test_klu_matching_setting():
    assert rslab.KluSettings().to_dict()["matching"] is True
    A = _circuit(300)
    f1 = rslab.klu(A)
    f0 = rslab.klu(A, matching=False)
    assert f1.n == f0.n
    b = np.ones(300)
    assert _res(A, f1.solve(b), b) < 1e-10 and _res(A, f0.solve(b), b) < 1e-10


def test_settings_reject_unknown_and_invalid():
    with pytest.raises(TypeError):
        rslab.Settings(threds=2)
    with pytest.raises(ValueError):
        rslab.Settings(ordering="banana")
    with pytest.raises(ValueError):
        rslab.Settings(method="fast")
    with pytest.raises(TypeError):
        rslab.ldlt(_spd(50), no_such_option=1)


def test_settings_object_and_kwargs_compose():
    A = _spd(200)
    s = rslab.Settings(ordering="amd", threads=1)
    f1 = rslab.ldlt(A, settings=s)
    f2 = rslab.ldlt(A, settings=s, ordering="rcm")
    assert f1.diagnostics()["decisions"]["ordering_used"].lower().startswith("amd")
    assert f2.diagnostics()["decisions"]["ordering_used"].lower().startswith("rcm")
    b = np.arange(200, dtype=float)
    assert _res(A, f2.solve(b), b) < 1e-10


def test_klu_settings():
    s = rslab.KluSettings(pivot_tol=0.5, parallel=False)
    d = s.to_dict()
    assert d["pivot_tol"] == 0.5 and d["parallel"] is False and d["btf"] is True
    assert rslab.KluSettings().to_dict()["parallel"] is None
    with pytest.raises(TypeError):
        rslab.KluSettings(ordering="amd")
    f = rslab.klu(_circuit(300), settings=s)
    assert f.n == 300


def test_interrupt_cancels_factorization():
    stop = rslab.Interrupt()
    assert not stop.is_set
    stop.cancel()
    assert stop.is_set
    with pytest.raises(RuntimeError):
        rslab.lu(_general(400), interrupt=stop)
    stop.reset()
    assert rslab.lu(_general(400), interrupt=stop).n == 400


# ---------------------------------------------------------------------------
# Symbolic reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["ldlt", "lu", "klu"])
def test_analyze_then_factor_sweep(path):
    n = 300
    A = {"ldlt": _spd(n), "lu": _general(n), "klu": _circuit(n)}[path]
    sym = rslab.analyze(A, path=path)
    assert sym.n == n
    assert sym.factor_nnz > 0
    est = sym.estimate_memory()
    assert est["factor_bytes"] > 0 and est["factor_mb"] >= 0
    if path != "klu":
        assert sym.n_levels >= 1
        assert len(sym.level_widths) == sym.n_levels
        assert len(sym.front_dims) >= 1
    else:
        assert sym.n_blocks >= 1 and len(sym.block_ptr) == sym.n_blocks + 1
    prepared = rslab._lower_csc(A) if path == "ldlt" else rslab._full_csc(A)
    b = np.arange(n, dtype=float)
    for scale in (1.0, 2.5):
        f = sym.factor(prepared.data * scale)
        assert _res(A * scale, f.solve(b), b) < 1e-10
    # Numeric settings can be overridden at factor time.
    f = sym.factor(prepared.data, threads=1) if path != "klu" else sym.factor(prepared.data, pivot_tol=1.0)
    assert _res(A, f.solve(b), b) < 1e-10


def test_analyze_auto_picks_path():
    assert isinstance(rslab.analyze(_spd(80)), rslab.LdltSymbolic)
    assert isinstance(rslab.analyze(_general(80)), rslab.LuSymbolic)
    assert "LdltSymbolic(" in repr(rslab.analyze(_spd(80)))


def test_symbolic_rejects_wrong_nnz():
    sym = rslab.analyze(_general(100), path="lu")
    with pytest.raises(ValueError):
        sym.factor(np.ones(3))


def test_symbolic_settings_carry_explicit_ordering():
    sym = rslab.analyze(_spd(200), path="ldlt", ordering="rcm")
    assert sym.settings.to_dict()["ordering"] == "rcm"
    f = sym.factor(rslab._lower_csc(_spd(200)).data)
    assert f.diagnostics()["decisions"]["ordering_used"].lower().startswith("rcm")


# ---------------------------------------------------------------------------
# Krylov entry points
# ---------------------------------------------------------------------------


def test_krylov_result_unpacks_and_has_attributes():
    A = _general(200)
    b = np.ones(200)
    r = rslab.gmres(A, b, tol=1e-10, maxit=2000)
    x, ok, iters, res, stop = r
    assert ok and stop == "converged" and iters == r.iters
    assert len(r) == 5 and r[-1] == "converged"
    assert _res(A, r.x, b) < 1e-8
    assert "KrylovResult(" in repr(r)


def test_gmres_preconditioned_by_other_factor():
    A = _general(300)
    A2 = A + sp.eye(300) * 0.5      # a nearby operator
    M = rslab.lu(A)
    b = np.ones(300)
    r = rslab.gmres(A2, b, M, tol=1e-10)
    assert r.converged and _res(A2, r.x, b) < 1e-8
    R = rslab.gmres_block(A2, np.column_stack([b, 2 * b]), M, tol=1e-10)
    assert R.converged and R.x.shape == (300, 2)
    assert _res(A2, R.x[:, 1], 2 * b) < 1e-8
    rec = M.recycle(4)
    r2 = rslab.gmres(A2, b, M, tol=1e-10, recycle=rec)
    assert r2.converged


@pytest.mark.parametrize("solver", [rslab.cocg, rslab.cocr])
def test_cocg_cocr_complex_symmetric(solver):
    n = 200
    K = _spd(n).astype(np.complex128)
    A = (K + 1j * 0.3 * sp.eye(n)).tocsc()
    b = np.ones(n, dtype=np.complex128)
    r = solver(A, b, tol=1e-10, maxit=5000)
    assert r.converged and _res(A, r.x, b) < 1e-8
    M = rslab.ldlt(A, drop_tol=1e-2)
    r2 = solver(A, b, M, tol=1e-10, maxit=5000)
    assert r2.converged and r2.iters <= r.iters


def test_handle_krylov_methods_accept_operator_override():
    A = _spd(150)
    f = rslab.ldlt(A)
    A2 = rslab._full_csc(A + sp.eye(150) * 0.1)
    b = np.ones(150)
    r = f.cocg(b, tol=1e-10, operator=rslab._parts(A2))
    assert r.converged and _res(A2, r.x, b) < 1e-8


# ---------------------------------------------------------------------------
# Refinement policy and logging
# ---------------------------------------------------------------------------


def test_diagnostics_report_throughput_rates():
    A = _spd(300)
    f = rslab.ldlt(A)
    f.solve(np.ones(300))
    r = f.diagnostics()["rates"]
    for key in ("analyze_mdof_s", "factor_mdof_s", "factor_gflops", "factor_mnnz_s", "total_mdof_s", "solve_mdof_s"):
        assert r[key] >= 0.0
    assert r["factor_mdof_s"] > 0 and r["solve_mdof_s"] > 0
    assert "MDOF/s" in f.diagnostics()["summary"]
    k = rslab.klu(_circuit(300))
    k.refactor(rslab._full_csc(_circuit(300)).data)
    assert k.diagnostics()["rates"]["factor_mdof_s"] > 0


def test_solve_refine_target_and_measure():
    A = _spd(200)
    f = rslab.ldlt(A, preconditioner=1e-2)
    b = np.arange(200, dtype=float)
    x = f.solve(b, refine=10, target=1e-14, measure="componentwise")
    assert _res(A, x, b) < 1e-10
    with pytest.raises(ValueError):
        f.solve(b, refine=1, measure="sideways")


def test_log_sink_and_level():
    seen = []
    rslab.set_log_sink(lambda level, msg: seen.append((level, msg)))
    old = rslab.log_level()
    try:
        rslab.set_log_level("debug")
        assert rslab.log_level() == "debug"
        rslab.lu(_general(200))
        assert seen, "no log messages reached the sink at debug level"
        assert all(lvl in ("debug", "info", "warning", "error") for lvl, _ in seen)
    finally:
        rslab.set_log_level(old)
        rslab.set_log_sink(None)
    with pytest.raises(ValueError):
        rslab.set_log_level("loud")


def test_python_logging_bridge():
    log = logging.getLogger("rslab-test")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    rslab.set_log_sink(lambda level, msg: log.log(logging.getLevelName(level.upper()), msg))
    old = rslab.log_level()
    try:
        rslab.set_log_level("info")
        rslab.klu(_circuit(300))
    finally:
        rslab.set_log_level(old)
        rslab.set_log_sink(None)
        log.removeHandler(handler)
    assert records


# ---------------------------------------------------------------------------
# API reference
# ---------------------------------------------------------------------------


def test_api_reference_is_current():
    """docs/api.md is generated from the docstrings; regenerate with
    `python tools/gen_api_reference.py` when they change."""
    root = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "tools" / "gen_api_reference.py"), "--check"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_symbolic_factor_and_refactor_accept_the_matrix():
    import numpy as np
    import scipy.sparse as sp

    n = 40
    rng = np.random.default_rng(3)
    A = sp.random(n, n, density=0.15, random_state=rng, format="csc") + 10.0 * sp.eye(n, format="csc")
    S = (A + A.T).tocsc()
    b = rng.standard_normal(n)
    # LDLT: the symbolic factor takes the full symmetric matrix (lower triangle extracted)
    sym = rslab.analyze(S, "ldlt")
    f = sym.factor(S)
    assert np.linalg.norm(S @ f.solve(b) - b) < 1e-9 * np.linalg.norm(b)
    S2 = (S * 2.0).tocsc()
    f2 = sym.factor(S2)
    assert np.linalg.norm(S2 @ f2.solve(b) - b) < 1e-9 * np.linalg.norm(b)
    # the value array in the analysis order still works
    f3 = sym.factor(sp.tril(S2).tocsc().data)
    assert np.allclose(f3.solve(b), f2.solve(b))
    # LU and KLU take the full matrix
    G = A.tocsc()
    lu = rslab.analyze(G, "lu").factor(G)
    assert np.linalg.norm(G @ lu.solve(b) - b) < 1e-9 * np.linalg.norm(b)
    k = rslab.klu(G)
    G2 = (G * 3.0).tocsc()
    k.refactor(G2)
    assert np.linalg.norm(G2 @ k.solve(b) - b) < 1e-9 * np.linalg.norm(b)
    # a different pattern is refused
    D = (G + sp.eye(n, k=1, format="csc")).tocsc()
    with pytest.raises(ValueError):
        rslab.analyze(G, "lu").factor(D)
