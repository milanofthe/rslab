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

| `Settings` keyword | default | what it does | options |
|---|---|---|---|
| `ordering` | heuristic pick | fill-reducing ordering of the symmetric pattern (LDL^T and LU) | `"auto"` heuristic pick, `"auto_race"` AMD and nested dissection raced on the fill estimate, `"amd"`, `"amf"`, `"metis"` one nested-dissection run, `"rcm"` |
| `nemin` | 16 | supernode amalgamation threshold | int; smaller means finer supernodes (less fill, more per-front overhead) |
| `relax` | on | relaxed, fill-tolerant amalgamation | `True` built-in thresholds, `False` off, `(max_width, max_extra_rows)` explicit |
| `reorder` | `"hybrid_liu"` | child order of the elimination tree | `"hybrid_liu"` smaller contribution-stack peak, `"off"` natural leaf order (more leaf parallelism) |
| `threads` | predictor, max 4 | worker budget of the factorization pool; the factor is bit-identical for every value | int (`0` = all cores), `"auto"` predictor without the cap, `"ambient"` the caller's rayon pool |
| `preconditioner` | `None` | static-pivot floor: pivots below it are lifted, the factorization never fails (factor of a nearby `A + E`; recover with `solve(b, refine=k)`) | float, e.g. `1e-4` |
| `force_accept` | `False` | accept tiny pivots in exact mode instead of raising on rank deficiency | bool |
| `drop_tol` | `None` | incomplete factorization: fill below the threshold (relative to its column) is dropped, an ILU-style preconditioner | float, `None` keeps the complete factor |
| `method` | `"left_looking"` | numeric schedule (same factor, different transient memory and parallel profile) | `"left_looking"`, `"multifrontal"` |
| `memory` | `"low"` | when fronts are released | `"low"` each front freed as it is emitted, `"eager"` fronts stay resident |
| `pivot_u` | 0.1 | threshold partial-pivoting tolerance of the LU path (`1.0` is full partial pivoting); ignored on LDL^T | float in `[0, 1]` |
| `matching` | `True` | MC64 row matching and scaling before the LU analysis (bounded pivot growth); LU path only | bool |
| `scaling` | `"one_pass"` | symmetric equilibration before LDL^T (the LU path scales its own way) | `"one_pass"`, `"inf_norm"`, `"mc64"`, `"auto"`, `"identity"` |
| `blr` | off | block-low-rank compression of the contribution blocks with a relative tolerance | float tolerance, `False` exact dense fronts |
| `panel_nb` | 64 | panel width (blocking factor) of the dense kernels | int |
| `scalar_gate` | calibrated | flop count below which an update runs as a scalar loop | int |
| `par_gemm` | calibrated | flop count at or above which the front GEMM runs in parallel | int |
| `par_cdiv` | calibrated | flop count at or above which the panel-trailing update runs in parallel | int |
| `use_gemm_schur` | `True` | SIMD GEMM (`True`) or the scalar loop for the front Schur update | bool |
| `interrupt` | `None` | cancellation flag polled by the numeric phase | an `rslab.Interrupt` |

`KluSettings` (the circuit path):

| keyword | default | what it does | options |
|---|---|---|---|
| `pivot_tol` | `1e-3` | diagonal preference: the diagonal is the pivot when `abs(a_jj) >= pivot_tol * max_i abs(a_ij)` | float, `1.0` is plain partial pivoting |
| `row_scaling` | `True` | divide each row by its largest magnitude before factoring | bool |
| `btf` | `True` | permute to block upper triangular form first | bool (keep it on) |
| `matching` | `True` | MC64 maximum-product transversal of the block triangular form (the diagonal-preference pivoting rarely leaves the diagonal); needs `btf` | bool |
| `parallel` | auto | per-block parallel factor and refactor over the BTF blocks; bit-identical in every mode | `None` structural auto gate, `True`, `False` |
| `interrupt` | `None` | cancellation flag polled by the numeric phase | an `rslab.Interrupt` |

Solve-time options of every handle (`solve(b, refine=0, target=None, measure="normwise")`):

| keyword | default | what it does | options |
|---|---|---|---|
| `refine` | 0 | iterative refinement steps on the factor | int |
| `target` | `None` | stop refining once the backward error is below it | float |
| `measure` | `"normwise"` | the backward error used by `target` and reported in the diagnostics | `"normwise"`, `"componentwise"` |

The symbolic handles factor new values on the analyzed pattern with
`sym.factor(A)`, where `A` is the matrix itself (its lower triangle is
taken on the LDL^T path, the pattern is checked) or the CSC value array in
the order of the analysis; `Klu.refactor(A)` takes the same forms.

`KluSettings`: `pivot_tol` (1e-3), `row_scaling` (on), `btf` (on),
`matching` (on: MC64 row matching as the BTF transversal),
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
