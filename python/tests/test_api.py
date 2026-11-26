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
from constants import U64, solve_backward_error
from residuals import backward_error, krylov_bound, recursive_drift, residual_norm, unit_roundoff


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


def test_settings_roundtrip_and_repr():
    floor = 1e-4
    s = rslab.Settings(threads=2, ordering="metis", preconditioner=floor, relax=(128, 32))
    d = s.to_dict()
    assert d["threads"] == 2
    assert d["ordering"] == "metis"
    assert d["preconditioner"] == floor
    assert d["relax"] == (128, 32)
    assert "Settings(" in repr(s) and "ordering='metis'" in repr(s)
    assert rslab.Settings().to_dict()["ordering"] == "auto"
    assert rslab.Settings().to_dict()["threads"] is None


def test_lu_matching_setting():
    assert rslab.Settings().to_dict()["matching"] is True
    assert rslab.Settings(matching=False).to_dict()["matching"] is False
    A = _general(200)
    assert rslab.lu(A).diagnostics()["decisions"]["scaling"] == "TwoSidedRowCol"
    P = sp.eye(200, format="csc")[np.r_[1, 0, 2:200]]
    B = (P @ A).tocsc()
    assert rslab.lu(B).diagnostics()["decisions"]["scaling"] == "Mc64RowMatching"
    assert rslab.lu(B, matching=False).diagnostics()["decisions"]["scaling"] == "TwoSidedRowCol"


def test_lu_equilibration_setting():
    rng = np.random.default_rng(3)
    A = _general(400)
    r, c = 10.0 ** rng.integers(-6, 7, 400), 10.0 ** rng.integers(-6, 7, 400)
    S = (sp.diags(r) @ A @ sp.diags(c)).tocsc()
    b = S @ np.ones(400)
    one = rslab.lu(S, preconditioner=1e-6, matching=False)
    it = rslab.lu(S, preconditioner=1e-6, matching=False, scaling="inf_norm")
    assert one.diagnostics()["decisions"]["scaling"] == "TwoSidedRowCol"
    assert it.diagnostics()["decisions"]["scaling"] == "TwoSidedRuiz"
    assert not it.diagnostics()["warnings"]
    assert it.n_perturbed <= one.n_perturbed
    x = it.solve(b)
    omega = np.max(np.abs(S @ x - b) / (abs(S) @ np.abs(x) + np.abs(b)))
    assert omega <= solve_backward_error(400, U64)
    assert rslab.lu(S, scaling="identity").diagnostics()["decisions"]["scaling"] == "Identity"
    assert rslab.lu(S, scaling="mc64").diagnostics()["warnings"]


def test_klu_matching_setting():
    assert rslab.KluSettings().to_dict()["matching"] is True
    A = _circuit(300)
    f1 = rslab.klu(A)
    f0 = rslab.klu(A, matching=False)
    assert f1.n == f0.n
    b = np.ones(300)
    bound = solve_backward_error(300, U64)
    assert backward_error(A, f1.solve(b), b) <= bound and backward_error(A, f0.solve(b), b) <= bound


def test_settings_reject_unknown_and_invalid():
    with pytest.raises(TypeError):
        rslab.Settings(threds=2)
    with pytest.raises(ValueError):
        rslab.Settings(ordering="banana")
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
    assert backward_error(A, f2.solve(b), b) <= solve_backward_error(200, U64)


def test_klu_settings():
    s = rslab.KluSettings(pivot_threshold=0.5, parallel=False)
    d = s.to_dict()
    assert d["pivot_threshold"] == 0.5 and d["parallel"] is False and d["btf"] is True
    assert rslab.KluSettings().to_dict()["parallel"] is None
    assert rslab.KluSettings().to_dict()["threads"] is None
    assert rslab.KluSettings(threads=2).to_dict()["threads"] == 2
    assert rslab.KluSettings(threads="ambient").to_dict()["threads"] == "ambient"
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


@pytest.mark.parametrize("path", ["ldlt", "lu", "klu"])
def test_analyze_then_factor_sweep(path):
    n = 300
    A = {"ldlt": _spd(n), "lu": _general(n), "klu": _circuit(n)}[path]
    sym = rslab.analyze(A, path=path)
    assert sym.n == n
    assert sym.factor_nnz > 0
    est = sym.estimate_memory()
    assert est["factor_bytes"] > 0 and est["factor_mb"] >= 0
    plan = sym.memory_plan("complex128", nrhs=4)
    assert plan["nrhs"] == 4
    assert plan["peak_bytes"] >= plan["resident_bytes"] >= plan["factor_bytes"] > 0
    if path != "klu":
        assert sym.memory_plan(threads=2)["threads"] == 2
        assert sym.memory_plan("complex64")["factor_bytes"] < plan["factor_bytes"]
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
        assert backward_error(A * scale, f.solve(b), b) <= solve_backward_error(n, U64)
    f = sym.factor(prepared.data, threads=1) if path != "klu" else sym.factor(prepared.data, pivot_threshold=1.0)
    assert backward_error(A, f.solve(b), b) <= solve_backward_error(n, U64)


def _grid(m, dtype=np.float64):
    t = sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(m, m))
    s = sp.diags([-1.0, -1.0], [-1, 1], shape=(m, m))
    return (sp.kron(sp.eye(m), t) + sp.kron(s, sp.eye(m))).astype(dtype).tocsc()


def test_analyze_auto_routes_by_work():
    small = rslab.analyze(_spd(80))
    assert isinstance(small, rslab.AutoSymbolic) and small.route == "klu"
    assert "AutoSymbolic(" in repr(small)
    A = _grid(100)
    big = rslab.analyze(A)
    assert big.route == "supernodal"
    assert big.memory_plan()["factor_bytes"] > 0 and big.heap_bytes > 0
    b = np.ones(A.shape[0])
    f = big.factor(A)
    assert f.backend == "ldlt"
    assert backward_error(A, f.solve(b), b) <= solve_backward_error(A.shape[0], U64)
    A2 = A.tolil()
    A2[0, 1] = -0.5
    A2 = A2.tocsc()
    f.refactor(A2)
    assert f.backend == "lu"
    assert backward_error(A2, f.solve(b), b) <= solve_backward_error(A.shape[0], U64)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128, np.complex64])
def test_auto_factor_every_field(dtype):
    assert rslab.auto(_general(200, dtype=dtype)).backend == "lu"
    A = _circuit(200).astype(dtype).tocsc()
    f = rslab.auto(A)
    assert f.backend == "klu" and f.dtype == np.dtype(dtype).name
    b = np.arange(200).astype(dtype)
    bound = solve_backward_error(200, unit_roundoff(dtype))
    assert backward_error(A, f.solve(b), b) <= bound
    assert backward_error(A.T, f.solve_transpose(b), b) <= bound
    assert backward_error(A, f.solve_many(np.stack([b, 2 * b], axis=1))[:, 1], 2 * b) <= bound
    assert "Auto(" in repr(f) and f.diagnostics()["solves"]["rhs"] == 4


def test_auto_symmetry_tolerance():
    asymmetry = 1e-14
    A = _grid(100).tolil()
    A[0, 1] = A[0, 1] * (1 + asymmetry)
    A = A.tocsc()
    assert rslab.auto(A).backend == "lu"
    relaxed = rslab.auto(A, symmetry_tol=2 * asymmetry)
    assert relaxed.backend == "ldlt"


def test_spsolve_takes_the_automatic_path():
    A = _grid(100)
    b = np.ones(A.shape[0])
    assert backward_error(A, rslab.spsolve(A, b), b) <= solve_backward_error(A.shape[0], U64)
    G = _general(150)
    assert backward_error(G, rslab.spsolve(G, np.ones(150)), np.ones(150)) <= solve_backward_error(150, U64)
    x = rslab.spsolve(G, np.ones(150), preconditioner=1e-8, refine=2)
    assert backward_error(G, x, np.ones(150)) <= solve_backward_error(150, U64)


def test_symbolic_rejects_wrong_nnz():
    sym = rslab.analyze(_general(100), path="lu")
    with pytest.raises(ValueError):
        sym.factor(np.ones(3))


def test_symbolic_settings_carry_explicit_ordering():
    sym = rslab.analyze(_spd(200), path="ldlt", ordering="rcm")
    assert sym.settings.to_dict()["ordering"] == "rcm"
    f = sym.factor(rslab._lower_csc(_spd(200)).data)
    assert f.diagnostics()["decisions"]["ordering_used"].lower().startswith("rcm")


def test_krylov_result_unpacks_and_has_attributes():
    A = _general(200)
    b = np.ones(200)
    tol = 1e-10
    r = rslab.gmres(A, b, tol=tol, maxit=2000)
    x, ok, iters, res, stop = r
    assert ok and stop == "converged" and iters == r.iters
    assert len(r) == 5 and r[-1] == "converged"
    assert residual_norm(A, r.x, b) <= krylov_bound(A, r.x, b, tol)
    assert "KrylovResult(" in repr(r)


def test_gmres_preconditioned_by_other_factor():
    A = _general(300)
    A2 = A + sp.eye(300) * 0.5
    M = rslab.lu(A)
    b = np.ones(300)
    tol = 1e-10
    r = rslab.gmres(A2, b, M, tol=tol)
    assert r.converged and residual_norm(A2, r.x, b) <= krylov_bound(A2, r.x, b, tol)
    R = rslab.gmres_block(A2, np.column_stack([b, 2 * b]), M, tol=tol)
    assert R.converged and R.x.shape == (300, 2)
    assert residual_norm(A2, R.x[:, 1], 2 * b) <= krylov_bound(A2, R.x[:, 1], 2 * b, tol)
    rec = M.recycle(4)
    r2 = rslab.gmres(A2, b, M, tol=tol, recycle=rec)
    assert r2.converged


@pytest.mark.parametrize("solver", [rslab.cocg, rslab.cocr])
def test_cocg_cocr_complex_symmetric(solver):
    n = 200
    K = _spd(n).astype(np.complex128)
    A = (K + 1j * 0.3 * sp.eye(n)).tocsc()
    b = np.ones(n, dtype=np.complex128)
    tol = 1e-10
    r = solver(A, b, tol=tol, maxit=5000)
    assert r.converged
    if solver is rslab.cocg:
        drift = recursive_drift(lambda j: solver(A, b, tol=tol, maxit=j), A, b, r.iters)
        assert residual_norm(A, r.x, b) <= krylov_bound(A, r.x, b, tol) + drift
    else:
        assert np.linalg.norm(A @ r.x - b) / np.linalg.norm(b) < 1e-8
    M = rslab.ldlt(A, preconditioner=1e-8)
    r2 = solver(A, b, M, tol=tol, maxit=5000)
    assert r2.converged and r2.iters <= r.iters


def test_handle_krylov_methods_accept_operator_override():
    A = _spd(150)
    f = rslab.ldlt(A)
    A2 = rslab._full_csc(A + sp.eye(150) * 0.1)
    b = np.ones(150)
    tol = 1e-10
    operator = rslab._parts(A2)
    r = f.cocg(b, tol=tol, operator=operator)
    drift = recursive_drift(lambda j: f.cocg(b, tol=tol, maxit=j, operator=operator), A2, b, r.iters)
    assert r.converged and residual_norm(A2, r.x, b) <= krylov_bound(A2, r.x, b, tol) + drift


def test_diagnostics_report_counts():
    A = _spd(300)
    f = rslab.ldlt(A)
    f.solve(np.ones(300))
    d = f.diagnostics()
    assert d["factor_bytes"] > 0 and d["solves"] == {"calls": 1, "rhs": 1, "refine_steps": 0}
    assert "nnz(L)=" in d["summary"]


def test_solve_refine_target_and_measure():
    A = _spd(200)
    f = rslab.ldlt(A, preconditioner=1e-2)
    b = np.arange(200, dtype=float)
    target = 1e-14
    x = f.solve(b, refine=10, target=target, measure="componentwise")
    componentwise = np.max(np.abs(A @ x - b) / (abs(A) @ np.abs(x) + np.abs(b)))
    assert componentwise <= target
    with pytest.raises(ValueError):
        f.solve(b, refine=1, measure="sideways")


def _floored_saddle(m=10, n2=24):
    n1 = m * m
    t = []
    for j in range(n1):
        x, y = j % m, j // m
        t.append((j, j, 4.0 + 0.3j))
        if x > 0:
            t.append((j - 1, j, -1.2 + 0.1j))
        if x + 1 < m:
            t.append((j + 1, j, -0.8))
        if y > 0:
            t.append((j - m, j, -1.2 + 0.1j))
        if y + 1 < m:
            t.append((j + m, j, -0.8))
    for c in range(n2):
        i, j = (4 * (c - 1) if c % 3 == 2 else 4 * c), n1 + c
        t += [(i, j, 1.0), (i + 1, j, -1.0), (j, i, 1.0), (j, i + 1, -1.0)]
    r, c, v = zip(*t)
    n = n1 + n2
    return sp.csc_matrix((np.array(v, dtype=complex), (r, c)), shape=(n, n))


def test_solve_refine_factored_system():
    A = _floored_saddle()
    f = rslab.lu(A, preconditioner=1e-2)
    rows, cols, deltas = f.perturbations()
    assert len(deltas) == f.n_perturbed > 0
    E = sp.csc_matrix((deltas, (rows, cols)), shape=A.shape)
    b = np.arange(A.shape[0]) + 1j
    x = f.solve(b, refine=3, system="factored")
    bound = solve_backward_error(A.shape[0], U64)
    assert backward_error(A + E, x, b) <= bound
    assert backward_error(A, x, b) > bound
    with pytest.raises(ValueError):
        f.solve(b, refine=1, system="sideways")


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
        assert any(msg.startswith("factor  ") for _, msg in seen), seen
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
        rslab.set_log_level("debug")
        rslab.klu(_circuit(300))
    finally:
        rslab.set_log_level(old)
        rslab.set_log_sink(None)
        log.removeHandler(handler)
    assert records


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
    sym = rslab.analyze(S, "ldlt")
    f = sym.factor(S)
    bound = solve_backward_error(n, U64)
    assert backward_error(S, f.solve(b), b) <= bound
    S2 = (S * 2.0).tocsc()
    f2 = sym.factor(S2)
    assert backward_error(S2, f2.solve(b), b) <= bound
    f3 = sym.factor(sp.tril(S2).tocsc().data)
    assert np.allclose(f3.solve(b), f2.solve(b))
    G = A.tocsc()
    lu = rslab.analyze(G, "lu").factor(G)
    assert backward_error(G, lu.solve(b), b) <= bound
    k = rslab.klu(G)
    G2 = (G * 3.0).tocsc()
    k.refactor(G2)
    assert backward_error(G2, k.solve(b), b) <= bound
    D = (G + sp.eye(n, k=1, format="csc")).tocsc()
    with pytest.raises(ValueError):
        rslab.analyze(G, "lu").factor(D)


@pytest.mark.parametrize("path", ["ldlt", "lu", "klu", "auto"])
@pytest.mark.parametrize("index", ["int32", "int64"])
def test_symbolic_factor_reads_the_matrix_as_its_values(path, index):
    import numpy as np
    import scipy.sparse as sp

    n = 60
    rng = np.random.default_rng(5)
    A = sp.random(n, n, density=0.1, random_state=rng, format="csc") + 10.0 * sp.eye(n, format="csc")
    S = (A + A.T).tocsc()
    S.indptr, S.indices = S.indptr.astype(index), S.indices.astype(index)
    b = rng.standard_normal(n)
    sym = rslab.analyze(S, path)
    prepared = rslab._lower_csc(S) if path == "ldlt" else rslab._full_csc(S)
    x = sym.factor(prepared.data).solve(b)
    assert np.array_equal(sym.factor(S).solve(b), x)
    indptr, indices, values = [0], [], []
    for j in range(n):
        rows = S.indices[S.indptr[j]:S.indptr[j + 1]][::-1]
        vals = S.data[S.indptr[j]:S.indptr[j + 1]][::-1].copy()
        vals[0] /= 2.0
        indices += [*rows, rows[0]]
        values += [*vals, vals[0]]
        indptr.append(len(indices))
    U = sp.csc_matrix((np.array(values), np.array(indices), np.array(indptr)), shape=S.shape)
    U.indptr, U.indices = U.indptr.astype(index), U.indices.astype(index)
    assert not U.has_canonical_format
    kept = (U.indptr.copy(), U.indices.copy(), U.data.copy())
    assert np.allclose(sym.factor(U).solve(b), x)
    assert all(np.array_equal(a, k) for a, k in zip((U.indptr, U.indices, U.data), kept))
    D = (S + sp.eye(n, k=-1, format="csc")).tocsc()
    with pytest.raises(ValueError):
        sym.factor(D)


def test_every_setting_round_trips():
    """Every Settings and KluSettings keyword is accepted and reported by to_dict."""
    import numpy as np

    s = rslab.Settings(
        threads=("auto", 2),
        preconditioner=1e-6,
        ordering="amf",
        scaling="mc64",
        pivot_threshold=0.5,
        matching=False,
        nemin=8,
        nd_ensemble=True,
        relax=(4, 12),
        panel_nb=32,
        scalar_gate=1000,
        par_gemm=100000,
        par_cdiv=200000,
        use_gemm_schur=False,
    )
    d = s.to_dict()
    assert d["threads"] == ("auto", 2)
    assert d["ordering"] == "amf" and d["scaling"] == "mc64"
    assert d["relax"] == (4, 12) and d["nd_ensemble"] is True
    assert (d["panel_nb"], d["scalar_gate"], d["par_gemm"], d["par_cdiv"]) == (32, 1000, 100000, 200000)
    assert d["use_gemm_schur"] is False and d["matching"] is False
    ext = rslab.Settings(scaling=np.ones(3))
    assert ext.to_dict()["scaling"] == "external"
    k = rslab.KluSettings(pivot_threshold=0.5, row_scaling=False, btf=True, matching=False, parallel=True)
    kd = k.to_dict()
    assert kd["pivot_threshold"] == 0.5 and kd["row_scaling"] is False and kd["parallel"] is True
    with pytest.raises(TypeError):
        rslab.Settings(no_such_option=1)
    with pytest.raises(TypeError):
        rslab.Settings(blr=1e-6)


def test_every_tuning_knob_round_trips():
    """Each plain Settings / KluSettings keyword is set and reported back."""
    for cls in (rslab.Settings, rslab.KluSettings):
        for key, v in cls().to_dict().items():
            if key == "relax":
                continue
            if isinstance(v, bool):
                new = not v
            elif isinstance(v, int):
                new = v + 1
            elif isinstance(v, float):
                new = v * 0.5
            else:
                continue
            assert cls(**{key: new}).to_dict()[key] == new, key


def test_tuned_knobs_still_solve():
    """Nondefault tuning constants change the schedule, not the solution."""
    import numpy as np
    import scipy.sparse as sp

    k = 30
    t = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    A = (sp.kron(sp.eye(k), t) + sp.kron(t, sp.eye(k))).tocsc()
    b = np.arange(A.shape[0], dtype=float)
    knobs = dict(
        ordering="metis",
        nd_fm_passes=2,
        nd_init_trials=3,
        nd_leaf_size=50,
        compress_max_ratio=1.0,
        amalgamation="renumber",
        nemin=4,
        panel_nb=16,
        trailing_block=4,
        schur_tile=32,
        fork_min_flops=0,
        solve_leaf_subtrees=4,
    )
    bound = solve_backward_error(A.shape[0], U64)
    x = rslab.ldlt(A, **knobs).solve(b)
    assert backward_error(A, x, b) <= bound
    y = rslab.lu(A, race_candidates=["rcm", "amd"], pivot_threshold=1.0).solve(b)
    assert backward_error(A, y, b) <= bound


@pytest.mark.parametrize("path", ["ldlt", "lu", "klu"])
def test_solve_transpose_on_every_handle(path):
    A = {"ldlt": _spd(300), "lu": _general(300), "klu": _circuit(300)}[path]
    f = getattr(rslab, path)(A)
    b = np.arange(300, dtype=float)
    y = f.solve_transpose(b)
    assert backward_error(A.T, y, b) <= solve_backward_error(300, U64)
    B = np.stack([b, b[::-1]], axis=1)
    X = f.solve_many(B)
    assert np.allclose(X[:, 0], f.solve(b)) and np.allclose(X[:, 1], f.solve(b[::-1]))


@pytest.mark.parametrize("path", ["ldlt", "lu"])
@pytest.mark.parametrize("dtype, low", [(np.float64, "float32"), (np.complex128, "complex64")])
def test_mixed_precision_factor(path, dtype, low):
    n = 400
    A = (_spd(n) if path == "ldlt" else _general(n)).astype(dtype)
    b = np.arange(n, dtype=dtype) + 1
    full = getattr(rslab, path)(A)
    f = getattr(rslab, path)(A, factor_dtype=low)
    assert f.dtype == np.dtype(dtype).name and f.factor_dtype == low
    assert f.heap_bytes < full.heap_bytes

    x = f.solve(b)
    single = unit_roundoff(np.dtype(low))
    assert x.dtype == dtype and backward_error(A, x, b) <= solve_backward_error(n, single) + single
    assert backward_error(A, f.solve(b, refine=5), b) <= solve_backward_error(n, U64)
    tol = 1e-12
    xg = f.gmres(b, tol=tol).x
    assert residual_norm(A, xg, b) <= krylov_bound(A, xg, b, tol)
    sym = rslab.analyze(A, path=path)
    assert sym.factor(A, factor_dtype=low).factor_dtype == low
    with pytest.raises(ValueError):
        getattr(rslab, path)(_spd(n).astype(np.float32), factor_dtype="complex64")


def test_krylov_defaults_are_the_core_constants():
    """The Python Krylov defaults are the core's, exported by the extension."""
    import inspect

    ext = rslab._rslab
    funcs = [rslab.gmres, rslab.gmres_block, rslab.cocg, rslab.cocr]
    funcs += [ext.gmres_plain, ext.gmres_block_plain, ext.cocg_plain, ext.cocr_plain]
    for cls in (rslab.Ldlt, rslab.Lu, rslab.Klu, rslab.Auto):
        funcs += [cls.gmres, cls.gmres_block, cls.cocg, cls.cocr]
    for f in funcs:
        p = inspect.signature(f).parameters
        assert p["tol"].default == ext.KRYLOV_TOL, f.__qualname__
        assert p["maxit"].default == ext.KRYLOV_MAX_ITER, f.__qualname__
