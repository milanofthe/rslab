# rslab (Python bindings)

NumPy/SciPy bindings for [RSLAB](https://github.com/milanofthe/rslab), a
pure-Rust sparse direct solver: symmetric and complex-symmetric LDL^T
(Bunch-Kaufman), unsymmetric LU, and a KLU path for circuit-shaped matrices.
All numeric work happens in Rust. Supported dtypes: `float64`, `float32`,
`complex128`, `complex64`.

```bash
pip install rslab
```

## Usage

```python
import numpy as np
import scipy.sparse as sp
import rslab

x = rslab.spsolve(A, b)              # one-shot, picks LDL^T or LU by symmetry

f = rslab.ldlt(A)                    # symmetric or complex-symmetric
x = f.solve(b)
X = f.solve_many(B)                  # n x nrhs
print(f.factor_nnz, f.inertia, f.diagnostics()["summary"])

f = rslab.lu(A_general)              # unsymmetric
y = f.solve_transpose(b)             # A.T @ y = b, on every handle
```

Circuit-shaped matrices (MNA / SPICE class) take the KLU path, with a
numeric-only refactor for fixed-pattern sweeps and its factors exported:

```python
f = rslab.klu(A)
x = f.solve(b)
f.refactor(A_next.data)              # same pattern, new values, no pivot search
y = f.solve_transpose(b)             # A.T @ y = b (conjugate b and y for A^H)
L, U, F = f.L, f.U, f.F              # (R A)[perm_r][:, perm_c] = L @ U + F
```

## Symbolic reuse

The analysis depends on the pattern only; pay it once per sweep:

```python
sym = rslab.analyze(A)               # LdltSymbolic, LuSymbolic or KluSymbolic
print(sym.estimate_memory()["factor_mb"])
for omega in frequencies:
    f = sym.factor(K + 1j * omega * C)   # the matrix or its data array
    x = f.solve(b)
```

## Preconditioners and Krylov solvers

Static pivoting never fails; refinement or a Krylov method recovers the
accuracy. `gmres`, `gmres_block`, `cocg` and `cocr` take any factor handle as
preconditioner:

```python
f = rslab.ldlt(A, preconditioner=1e-4)
x = f.solve(b, refine=20)

M = rslab.lu(A, drop_tol=1e-3)       # incomplete factor
x, converged, iters, res, stop = rslab.gmres(A, b, M, tol=1e-10)
```

## Settings

Factor knobs are keyword arguments of `ldlt`, `lu`, `spsolve` and `analyze`,
or a `rslab.Settings` object (`rslab.KluSettings` for `klu`). Every tuning
constant of the solver is one (`nd_fm_passes`, `race_candidates`,
`solve_block`, ...), with the tuned value as default;
`rslab.Settings().to_dict()` lists them all. Common overrides:

```python
f = rslab.lu(A, ordering="metis", threads=8, pivot_threshold=0.5)
s = rslab.Settings(preconditioner=1e-6, drop_tol=1e-3)
```

Settings a path does not read are listed under
`diagnostics()["warnings"]`. Every keyword, handle and method is documented
in the generated reference [`docs/api.md`](docs/api.md) (`help(rslab.lu)`
shows the same text).

## Logging

```python
rslab.set_log_level("info")          # or the RLA_LOG environment variable
log = logging.getLogger("rslab")
rslab.set_log_sink(lambda level, msg: log.log(logging.getLevelName(level.upper()), msg))
```

## License

MIT.
