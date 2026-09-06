# rslab (Python bindings)

NumPy/SciPy bindings for [RSLAB](https://github.com/milanofthe/rslab), a
pure-Rust sparse direct solver and preconditioner: complex/real symmetric LDL^T
(Bunch-Kaufman), unsymmetric LU, and a KLU-style path for circuit-shaped
matrices. A thin wrapper, all numeric work happens in Rust.

## Install

```bash
pip install rslab
```

## Usage

```python
import numpy as np
import scipy.sparse as sp
import rslab

# Symmetric system (real or complex; the dtype selects the path).
A = sp.random(5000, 5000, density=1e-3, format="csc") + sp.eye(5000) * 10
A = A + A.T
b = np.random.rand(5000)

# One-shot solve.
x = rslab.spsolve(A, b)

# Factor once, solve many right-hand sides.
f = rslab.ldlt(A)
x1 = f.solve(b)
X = f.solve_many(np.random.rand(5000, 8))   # n x nrhs

print(f.n, f.factor_nnz, f.inertia, f.dtype)
```

Complex-symmetric matrices (EM/FEM, PARDISO `mtype 6`) work identically:

```python
A = A.astype(np.complex128); A.data += 1j * 0.3 * A.data.real
x = rslab.ldlt(A).solve(np.ones(A.shape[0], dtype=np.complex128))
```

Unsymmetric matrices use the LU path:

```python
f = rslab.lu(A_general)
x = f.solve(b)
```

Circuit-shaped matrices (MNA / SPICE-class: very sparse, unsymmetric,
near-triangularizable) use the KLU path, bit-deterministic, with a
numeric-only `refactor` for fixed-pattern sweeps:

```python
f = rslab.klu(A_circuit)
x = f.solve(b)
A_circuit.data *= 1.5            # frequency sweep: same pattern, new values
f.refactor(A_circuit.data)       # no symbolic work, no pivot search
x2 = f.solve(b)
y = f.solve_transpose(b)         # A.T @ y = b on the same factors (adjoint)
```

`solve_transpose` is the plain transpose; for the conjugate-transpose adjoint
use `f.solve_transpose(b.conj()).conj()`.

### Preconditioner mode

Never-fail static pivoting plus iterative refinement for hard/indefinite
systems:

```python
f = rslab.ldlt(A, preconditioner=1e-4)
x = f.solve(b, refine=20)        # refine against the original A
```

## Configuration

`ldlt`, `lu` and `spsolve` take a `Settings` object (or the same keywords
directly), `klu` a `KluSettings`. By default the factor uses RSLAB's
deterministic heuristic pick: the adaptive ordering plus an exact
nested-dissection bakeoff on large systems, and at most 4 workers (the
calibrated count after a one-time `rslab.install_diagnose()`). Keywords
override the pick:

```python
s = rslab.Settings(ordering="metis", threads=2, preconditioner=1e-4)
f = rslab.ldlt(A, settings=s)
f = rslab.lu(A, ordering="amd", pivot_u=0.5)        # the same, as keywords
print(s.to_dict())
```

| `Settings` keyword | default | meaning |
|---|---|---|
| `ordering` | heuristic pick | `"auto"`, `"auto_race"`, `"amd"`, `"amf"`, `"metis"` (one nested-dissection run), `"rcm"` |
| `nemin`, `relax`, `reorder` | 16, on, `"hybrid_liu"` | supernode amalgamation and elimination-tree reordering |
| `threads` | predictor, max 4 | int (`0` = all cores), `"auto"`, `"ambient"`; the factor is bit-identical either way |
| `preconditioner` | `None` | static-pivot floor (e.g. `1e-4`): never-fail, refine to solve |
| `force_accept` | `False` | accept tiny pivots in exact mode instead of failing |
| `drop_tol` | `None` | incomplete-factor threshold (ILU-style preconditioner) |
| `method`, `memory` | `"left_looking"`, `"low"` | numeric schedule and factor emit strategy |
| `pivot_u` | 0.1 | threshold-pivoting tolerance of the LU path |
| `scaling` | `"one_pass"` | LDL^T equilibration: `"inf_norm"`, `"mc64"`, `"auto"`, `"identity"` |
| `blr`, `panel_nb` | off, 64 | block-low-rank tolerance, dense panel width |
| `scalar_gate`, `par_gemm`, `par_cdiv`, `use_gemm_schur` | calibrated | kernel tuning knobs |
| `interrupt` | `None` | an `rslab.Interrupt` cancellation flag |

`KluSettings`: `pivot_tol` (1e-3), `row_scaling` (on), `btf` (on),
`parallel` (`None` = structural auto gate, `True` / `False` force), `interrupt`.

Settings a path ignores are reported under `diagnostics()["warnings"]`.
`solve` and `solve_many` on `Ldlt` and `Lu` handles are supernodal and
tree-parallel (leaf subtrees of the elimination tree in parallel, parallel
sections inside the wide top separators), bit-identical for every thread
count. Throughput is always reported: `diagnostics()["rates"]` holds the analysis,
factorization and solve rates in million unknowns per second (MDOF/s), the
factorization flop rate (GFlop/s) and the factor-entry rate (Mnnz/s); the
`summary` line and the `info` log carry the factor rate too.

## Symbolic reuse

The analysis depends only on the sparsity pattern. Pay it once and factor
each value set of a sweep on it:

```python
sym = rslab.analyze(A, path="lu")                # LdltSymbolic / LuSymbolic / KluSymbolic
print(sym.factor_nnz, sym.estimate_memory()["factor_mb"])
for omega in frequencies:
    f = sym.factor((K + 1j * omega * C).data)    # same pattern, new values
    x = f.solve(b)
```

## Krylov solvers

`gmres`, `gmres_block`, `cocg` and `cocr` run on any matrix, optionally
preconditioned by any factor handle (an incomplete factor, or the factor of
a nearby matrix):

```python
M = rslab.lu(A, drop_tol=1e-3)                   # ILU-style preconditioner
x, converged, iters, res, stop = rslab.gmres(A, b, M, tol=1e-10)
r = rslab.gmres(A_next, b, M, recycle=M.recycle(8))   # reuse M, deflate across solves
```

## Logging

```python
rslab.set_log_level("info")                      # or the RLA_LOG environment variable
log = logging.getLogger("rslab")
rslab.set_log_sink(lambda level, msg: log.log(logging.getLevelName(level.upper()), msg))
```

## API

The full reference, generated from the docstrings, is in
[`docs/api.md`](docs/api.md) (`help(rslab.ldlt)` etc. show the same text).

| name | meaning |
|------|---------|
| `spsolve(A, b, **kw)` | one-shot factor-and-solve; detects symmetry and picks the LDL^T or LU path |
| `ldlt(A, **kw) -> Ldlt` | factor a real/complex **symmetric** matrix (Bunch-Kaufman LDL^T) |
| `lu(A, **kw) -> Lu` | factor a general unsymmetric matrix (supernodal LU) |
| `klu(A, **kw) -> Klu` | factor a circuit-shaped matrix (BTF + per-block Gilbert-Peierls LU) |
| `analyze(A, path, **kw)` | the symbolic analysis alone; `.factor(data)` per value set |
| `Settings`, `KluSettings`, `Interrupt` | configuration objects |
| `gmres`, `gmres_block`, `cocg`, `cocr` | Krylov solvers, `M=` any factor handle |
| `install_diagnose()` | one-time machine calibration |
| `set_log_level`, `log_level`, `set_log_sink` | the core's logger |

Factor handles share `solve(b, refine=0, target=None, measure="normwise")`,
`solve_many(B)`, `gmres`, `gmres_block`, `cocg`, `cocr`, `recycle(k)`,
`diagnostics()` and the attributes `n`, `factor_nnz`, `n_perturbed`,
`dtype`; `Ldlt` adds `inertia`, `Klu` adds `n_blocks`, `solve_transpose(b)`
and the numeric-only `refactor(data)`.

Supported dtypes: `float64`, `float32`, `complex128`, `complex64`.

## License

MIT.
