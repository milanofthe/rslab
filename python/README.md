# rslab (Python bindings)

NumPy/SciPy bindings for [RSLAB](https://github.com/milanofthe/rslab), a
pure-Rust sparse direct solver and preconditioner: complex/real symmetric LDLᵀ
(Bunch-Kaufman) plus unsymmetric LU. A thin wrapper, all numeric work happens in
Rust.

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

### Preconditioner mode

Never-fail static pivoting plus iterative refinement for hard/indefinite
systems:

```python
f = rslab.ldlt(A, preconditioner=1e-4)
x = f.solve(b, refine=20)        # refine against the original A
```

## Configuration (keyword arguments)

`ldlt`, `lu` and `spsolve` accept:

| kwarg            | default          | meaning                                                        |
|------------------|------------------|----------------------------------------------------------------|
| `threads`        | `2`              | worker-thread budget (`0` = all cores); result is bit-identical |
| `preconditioner` | `None`           | static-pivot floor (e.g. `1e-4`); never-fail, refine to solve  |
| `drop_tol`       | `None`           | incomplete-factor threshold (preconditioner)                   |
| `method`         | `"left_looking"` | `"left_looking"` or `"multifrontal"`                           |
| `memory`         | `"low"`          | `"low"` or `"eager"` factor emit strategy                      |
| `force_accept`   | `False`          | accept tiny pivots in exact mode instead of failing            |

Supported dtypes: `float64`, `float32`, `complex128`, `complex64`.

## License

MIT.
