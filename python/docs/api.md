# rslab Python API reference

Generated from the docstrings of `rslab` 0.34.0 by `tools/gen_api_reference.py`; do not edit by hand.

## Package

RSLAB - a pure-Rust sparse direct solver and preconditioner for NumPy/SciPy.

A thin, allocation-light wrapper over the RSLAB Rust core: a drop-in
alternative to `scipy.sparse.linalg.spsolve` / Intel MKL PARDISO for the
factor-once, solve-many workloads that dominate FEM, method-of-moments, and
circuit-extraction codes. Three factorization paths cover the operator
classes:

**Symmetric** matrices (real, or complex-symmetric `A = A^T`)
are factored by a supernodal Bunch-Kaufman method,

```math
P^T A P = L D L^T,
```
**general unsymmetric** matrices by a supernodal / multifrontal LU with
threshold partial pivoting,

```math
P_r^T A P_c = L U,
```
and **circuit-shaped** matrices (MNA / SPICE class: extremely sparse,
unsymmetric, near-triangularizable) by a KLU-style path - block triangular
form plus a per-block Gilbert-Peierls LU - whose numeric-only `refactor`
makes fixed-pattern sweeps cheap.

The solver is **type-agnostic**: the matrix `dtype` selects the arithmetic
field, so `float64` / `float32` run the real path and `complex128` /
`complex64` the complex path through the *same* call, at half the memory
for the 32-bit fields.

The API has four layers, each a thin step below the previous one:

1. `spsolve` - one-shot factor, solve, discard.
2. `ldlt` / `lu` / `klu` - factor once into a handle
   (`Ldlt`, `Lu`, `Klu`), solve many.
3. `analyze` - the symbolic analysis alone (`LdltSymbolic`,
   `LuSymbolic`, `KluSymbolic`); factor many value sets on one
   pattern.
4. `gmres` / `gmres_block` / `cocg` / `cocr` -
   Krylov solvers, optionally preconditioned by any factor handle.

Configuration is a `Settings` (LDL^T / LU) or `KluSettings`
object, or the same keywords given to the factor functions directly.

**Example**

```python
import numpy as np, scipy.sparse as sp, rslab

A = sp.random(2000, 2000, density=1e-3, format="csc") + sp.eye(2000) * 10
A = A + A.T                       # make it symmetric
b = np.random.rand(2000)
x = rslab.spsolve(A, b)           # one-shot: factor + solve + discard

f = rslab.ldlt(A)                 # factor once ...
x1 = f.solve(b)                   # ... solve many
X  = f.solve_many(np.random.rand(2000, 8))   # 8 right-hand sides at once

sym = rslab.analyze(A, path="ldlt")          # analyze once ...
for scale in (1.0, 2.0, 3.0):
    f = sym.factor(A.data * scale)           # ... factor many value sets
```
**Note**

The numeric factor is **bit-identical regardless of the thread count**; the
worker budget affects wall time and transient memory, not the result. By
default the factorization uses at most 4 workers (the pareto-optimal
throughput-per-core point on typical sparse factorizations); pass an explicit
`threads` to override.

**References**

.. [1] Bunch, J. R., & Kaufman, L. (1977). "Some stable methods for calculating
       inertia and solving symmetric linear systems." *Mathematics of
       Computation*, 31(137), 163-179. `10.1090/S0025-5718-1977-0428694-0`
.. [2] Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
       `10.1137/1.9780898718881`
.. [3] Davis, T. A., & Palamadai Natarajan, E. (2010). "Algorithm 907: KLU, a
       direct sparse solver for circuit simulation problems." *ACM Transactions
       on Mathematical Software*, 37(3). `10.1145/1824801.1824814`

## One-shot solve

### `spsolve(A, b, *, symmetric: 'bool | None' = None, refine: 'int' = 0, **kwargs)`

One-shot solve of `A x = b` (factor, solve, discard).

Detects symmetry, factors through `ldlt` or `lu`, solves,
and drops the factor. For repeated solves against one matrix keep the
handle instead (`ldlt` / `lu` / `klu`).

**Parameters**

- `A` (scipy.sparse matrix or array-like): The `n x n` system matrix.
- `b` (ndarray): Right-hand side: a 1-D vector of length `n` or a 2-D `n x nrhs` block. Cast to the factor's dtype automatically.
- `symmetric` (bool, optional): Force the symmetric `L D L^T` path (`True`) or the unsymmetric `L U` path (`False`). When omitted, symmetry is auto-detected from `A` (a structural + value test).
- `refine` (int, default 0): Steps of iterative refinement against the original matrix, per right-hand side. Meaningful with `preconditioner=...` / `drop_tol=...`, where the factor is inexact.
- `**kwargs`: Forwarded to `ldlt` / `lu` (any `Settings` keyword).

**Returns**

- `ndarray`: The solution, matching the shape of `b`.

**Raises**

- `ValueError`: If `b` is neither 1-D nor 2-D.
- `RuntimeError`: If the (exact-mode) factorization hits a zero pivot; set `preconditioner=...` to use never-fail static pivoting.

**Example**

```python
x = rslab.spsolve(A, b)                          # auto-detects symmetry
X = rslab.spsolve(A, np.random.rand(n, 5))       # 5 right-hand sides
x = rslab.spsolve(A, b, preconditioner=1e-4, refine=2)
```

## Factor handles

### `ldlt(A, *, settings: 'Settings | None' = None, **kwargs) -> 'Ldlt'`

Factor a **symmetric** matrix as `P^T A P = L D L^T`.

A supernodal Bunch-Kaufman `L D L^T` factorization with a
fill-reducing ordering `P`, for real symmetric (`float64` /
`float32`) and **complex-symmetric** (`complex128` / `complex64`,
i.e. `A = A^T`, *not* Hermitian) matrices; the `dtype`
selects the path. Only the lower triangle is read (extracted
automatically), so `A` may be stored full or triangular.

**Parameters**

- `A` (scipy.sparse matrix or array-like): The symmetric `n x n` system matrix. Converted to CSC and its lower triangle taken; duplicate entries are summed.
- `settings` (Settings, optional): A prepared `Settings` object.
- `**kwargs`: Any `Settings` keyword (`threads`, `preconditioner`, `drop_tol`, `method`, `memory`, `force_accept`, `ordering`, `scaling`, `pivot_u`, `nemin`, `relax`, `reorder`, `blr`, `panel_nb`, `interrupt` ...), overriding `settings`.

**Returns**

- `Ldlt`: A reusable factor handle: `Ldlt.solve`, `Ldlt.solve_many`, the Krylov methods with the factor as preconditioner, and `Ldlt.diagnostics`.

**Raises**

- `RuntimeError`: If a pivot is numerically zero in exact mode (the matrix is rank deficient). Set `preconditioner=...` (recommended) or `force_accept=True` to proceed.
- `ValueError`: On an unsupported dtype, a non-square matrix or an invalid setting.

**Example**

```python
f = rslab.ldlt(A)                       # heuristic defaults
f = rslab.ldlt(A, ordering="metis", threads=2)
f = rslab.ldlt(A, preconditioner=1e-4)  # never-fail static pivoting
x = f.solve(b, refine=2)
print(f.inertia, f.diagnostics()["summary"])
```

### `lu(A, *, settings: 'Settings | None' = None, **kwargs) -> 'Lu'`

Factor a **general** (unsymmetric) matrix as `P_r^T A P_c = L U`.

A supernodal left-looking (default) or multifrontal LU with threshold
partial pivoting and two-sided equilibration, over the same four scalar
fields as `ldlt`. The full matrix is read.

**Parameters**

- `A` (scipy.sparse matrix or array-like): The `n x n` system matrix. Converted to CSC; duplicates summed.
- `settings` (Settings, optional): A prepared `Settings` object.
- `**kwargs`: Any `Settings` keyword, overriding `settings`. `pivot_u` (default 0.1) is the threshold-pivoting tolerance of this path; `scaling` is ignored here (the LU path scales two-sided) and reported under `diagnostics()['warnings']`.

**Returns**

- `Lu`: A reusable factor handle (`Lu.solve`, `Lu.solve_many`, Krylov methods, `Lu.diagnostics`).

**Raises**

- `RuntimeError`: On a numerically zero pivot in exact mode.
- `ValueError`: On an unsupported dtype, a non-square matrix or an invalid setting.

**Example**

```python
f = rslab.lu(A)
x = f.solve(b)
r = f.gmres(b, tol=1e-10)               # the factor as preconditioner
x, converged, iters, res, stop = rslab.lu(A, drop_tol=1e-2).gmres(b)
```

### `klu(A, *, settings: 'KluSettings | None' = None, **kwargs) -> 'Klu'`

Factor a general matrix through the **KLU** path (circuit-shaped systems).

Block triangular form (BTF) plus a per-block AMD-ordered Gilbert-Peierls
left-looking LU with diagonal-preference pivoting, following KLU
(Davis & Palamadai Natarajan 2010). Built for MNA / SPICE matrices:
extremely sparse, unsymmetric, nearly triangular. The
`Klu.refactor` fast path re-factors new values on the same pattern
without symbolic work or pivot search.

**Parameters**

- `A` (scipy.sparse matrix or array-like): The `n x n` system matrix (full, CSC after conversion).
- `settings` (KluSettings, optional): A prepared `KluSettings` object.
- `**kwargs`: Any `KluSettings` keyword (`pivot_tol`, `row_scaling`, `btf`, `parallel`, `interrupt`), overriding `settings`.

**Returns**

- `Klu`: A reusable factor handle with `Klu.solve`, `Klu.solve_many`, `Klu.solve_transpose`, `Klu.refactor`, the Krylov methods and `Klu.diagnostics`.

**Raises**

- `RuntimeError`: If the matrix is structurally singular (no complete matching) or a block hits a numerically zero pivot.

**Example**

```python
f = rslab.klu(A)                        # factor once
x = f.solve(b)
f.refactor(A2.data)                     # same pattern, new values
x2 = f.solve(b)
y = f.solve_transpose(b)                # A^T y = b on the same factors
```

### class `Ldlt`

A symmetric factor `P^T A P = L D L^T` (Bunch-Kaufman), from
`rslab.ldlt` or `LdltSymbolic.factor`.

Holds the supernodal factor, the permutation, the equilibration and a
copy of the original lower triangle (for refinement and as the default
Krylov operator), so the factorization is paid once and amortized over
many solves.

**Attributes**

- `dtype`: NumPy dtype name of the factor (``'float64'``, ``'float32'``, ``'complex128'`` or ``'complex64'``).
- `factor_nnz`: Stored factor entries (the fill).
- `inertia`: Inertia ``(n_pos, n_neg, n_zero)``: the eigenvalue sign counts of ``A`` read off ``D`` (Sylvester's law).
- `n`: Matrix dimension ``n``.
- `n_perturbed`: Statically perturbed pivots (nonzero only in preconditioner mode).

#### `Ldlt.cocg(b, tol=1e-08, maxit=400, operator=None)`

Conjugate orthogonal conjugate gradient (COCG) with this factor
as the preconditioner: the short-recurrence method for
complex-symmetric (`A = A^T`) and real symmetric operators.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

#### `Ldlt.cocr(b, tol=1e-08, maxit=400, operator=None)`

Conjugate orthogonal conjugate residual (COCR) with this factor
as the preconditioner; the minimal-residual sibling of
`cocg` for complex-symmetric operators.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

#### `Ldlt.diagnostics()`

The factorization report as a dictionary: `stages` (name,
wall_ms, flops, bytes per phase), `decisions` (ordering,
scaling, method, supernode counts), `numeric` (perturbed
pivots, 2x2 pivots, inertia), `solves` (accumulated solve
calls, right-hand sides, wall time, refinement steps),
`warnings` (settings that were ignored on this path), the
throughput `rates` (`analyze_mdof_s`, `factor_mdof_s`,
`factor_gflops`, `factor_mnnz_s`, `total_mdof_s`,
`solve_mdof_s`; million unknowns per second and GFlop/s),
the a-priori `estimate` and a one-line `summary`.

#### `Ldlt.gmres(b, tol=1e-08, maxit=400, restart=None, x0=None, recycle=None, operator=None)`

Flexible restarted GMRES with this factor as the preconditioner.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target `||b - A x|| <= tol * ||b||`.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Arnoldi basis size; by default chosen so the basis stays under 1 GiB (between 20 and 80).
- `x0` (ndarray, optional): Initial guess (warm start).
- `recycle` (Recycle, optional): Deflation subspace carried across calls; see `recycle`.
- `operator` (tuple, optional): `(n, indptr, indices, data)` of a different CSC matrix to iterate on, with this factor as the preconditioner (the Python wrapper `rslab.gmres` builds it from a SciPy matrix). By default the factored matrix is the operator.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`; unpacks as a 5-tuple.

#### `Ldlt.gmres_block(b, tol=1e-08, maxit=400, restart=None, x0=None, operator=None)`

Block GMRES for an `n x nrhs` right-hand side block with this
factor as the preconditioner.

**Parameters**

- `b` (ndarray, shape (n, nrhs)): Right-hand side block.
- `tol` (float, default 1e-8): Relative residual target per column.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Block Arnoldi basis size (see `gmres`).
- `x0` (ndarray, shape (n, nrhs), optional): Initial guess.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x` of shape `(n, nrhs)`, `converged`, `iters`, `final_res` (one value per column), `stop`.

#### `Ldlt.recycle(k)`

A `Recycle` workspace holding up to `k` deflation
vectors for a sequence of `gmres` calls.

**Parameters**

- `k` (int): Maximum number of deflation vectors kept.

**Returns**

- `Recycle`: The workspace to pass as `recycle=` to `gmres`.

#### `Ldlt.solve(b, refine=0, target=None, measure='normwise')`

Solve `A x = b` for one right-hand side.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side in the factor's dtype.
- `refine` (int, default 0): Steps of iterative refinement against the original matrix.
- `target` (float, optional): Stop refining once the backward error is below this value (by default all `refine` steps run).
- `measure` ({'normwise', 'componentwise'}, default 'normwise'): The backward-error measure `target` refers to.

#### `Ldlt.solve_many(b)`

Solve `A X = B` for an `n x nrhs` block in one batched pass.

**Parameters**

- `b` (ndarray, shape (n, nrhs)): Right-hand side block in the factor's dtype.

**Returns**

- `ndarray, shape (n, nrhs)`: The solutions, one column per right-hand side.

### class `Lu`

A general (unsymmetric) factor `P_r^T A P_c = L U` (supernodal
left-looking or multifrontal LU with threshold pivoting), from
`rslab.lu` or `LuSymbolic.factor`.

**Attributes**

- `dtype`: NumPy dtype name of the factor (``'float64'``, ``'float32'``, ``'complex128'`` or ``'complex64'``).
- `factor_nnz`: Stored factor entries (the fill).
- `n`: Matrix dimension ``n``.
- `n_perturbed`: Statically perturbed pivots (nonzero only in preconditioner mode).

#### `Lu.cocg(b, tol=1e-08, maxit=400, operator=None)`

Conjugate orthogonal conjugate gradient (COCG) with this factor
as the preconditioner: the short-recurrence method for
complex-symmetric (`A = A^T`) and real symmetric operators.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

#### `Lu.cocr(b, tol=1e-08, maxit=400, operator=None)`

Conjugate orthogonal conjugate residual (COCR) with this factor
as the preconditioner; the minimal-residual sibling of
`cocg` for complex-symmetric operators.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

#### `Lu.diagnostics()`

The factorization report as a dictionary: `stages` (name,
wall_ms, flops, bytes per phase), `decisions` (ordering,
scaling, method, supernode counts), `numeric` (perturbed
pivots, 2x2 pivots, inertia), `solves` (accumulated solve
calls, right-hand sides, wall time, refinement steps),
`warnings` (settings that were ignored on this path), the
throughput `rates` (`analyze_mdof_s`, `factor_mdof_s`,
`factor_gflops`, `factor_mnnz_s`, `total_mdof_s`,
`solve_mdof_s`; million unknowns per second and GFlop/s),
the a-priori `estimate` and a one-line `summary`.

#### `Lu.gmres(b, tol=1e-08, maxit=400, restart=None, x0=None, recycle=None, operator=None)`

Flexible restarted GMRES with this factor as the preconditioner.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target `||b - A x|| <= tol * ||b||`.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Arnoldi basis size; by default chosen so the basis stays under 1 GiB (between 20 and 80).
- `x0` (ndarray, optional): Initial guess (warm start).
- `recycle` (Recycle, optional): Deflation subspace carried across calls; see `recycle`.
- `operator` (tuple, optional): `(n, indptr, indices, data)` of a different CSC matrix to iterate on, with this factor as the preconditioner (the Python wrapper `rslab.gmres` builds it from a SciPy matrix). By default the factored matrix is the operator.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`; unpacks as a 5-tuple.

#### `Lu.gmres_block(b, tol=1e-08, maxit=400, restart=None, x0=None, operator=None)`

Block GMRES for an `n x nrhs` right-hand side block with this
factor as the preconditioner.

**Parameters**

- `b` (ndarray, shape (n, nrhs)): Right-hand side block.
- `tol` (float, default 1e-8): Relative residual target per column.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Block Arnoldi basis size (see `gmres`).
- `x0` (ndarray, shape (n, nrhs), optional): Initial guess.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x` of shape `(n, nrhs)`, `converged`, `iters`, `final_res` (one value per column), `stop`.

#### `Lu.recycle(k)`

A `Recycle` workspace holding up to `k` deflation
vectors for a sequence of `gmres` calls.

**Parameters**

- `k` (int): Maximum number of deflation vectors kept.

**Returns**

- `Recycle`: The workspace to pass as `recycle=` to `gmres`.

#### `Lu.solve(b, refine=0, target=None, measure='normwise')`

Solve `A x = b` for one right-hand side.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side in the factor's dtype.
- `refine` (int, default 0): Steps of iterative refinement against the original matrix.
- `target` (float, optional): Stop refining once the backward error is below this value (by default all `refine` steps run).
- `measure` ({'normwise', 'componentwise'}, default 'normwise'): The backward-error measure `target` refers to.

#### `Lu.solve_many(b)`

Solve `A X = B` for an `n x nrhs` block in one batched pass.

**Parameters**

- `b` (ndarray, shape (n, nrhs)): Right-hand side block in the factor's dtype.

**Returns**

- `ndarray, shape (n, nrhs)`: The solutions, one column per right-hand side.

### class `Klu`

A KLU-path factor (block triangular form plus per-block Gilbert-Peierls
LU) for circuit-shaped matrices, from `rslab.klu` or
`KluSymbolic.factor`. Supports the numeric-only `refactor`
for fixed-pattern sweeps and `solve_transpose`.

**Attributes**

- `dtype`: NumPy dtype name of the factor (``'float64'``, ``'float32'``, ``'complex128'`` or ``'complex64'``).
- `factor_nnz`: Stored factor entries (the fill).
- `n`: Matrix dimension ``n``.
- `n_blocks`: Number of diagonal blocks of the block triangular form.
- `n_perturbed`: Statically perturbed pivots (nonzero only in preconditioner mode).

#### `Klu.cocg(b, tol=1e-08, maxit=400, operator=None)`

Conjugate orthogonal conjugate gradient (COCG) with this factor
as the preconditioner: the short-recurrence method for
complex-symmetric (`A = A^T`) and real symmetric operators.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

#### `Klu.cocr(b, tol=1e-08, maxit=400, operator=None)`

Conjugate orthogonal conjugate residual (COCR) with this factor
as the preconditioner; the minimal-residual sibling of
`cocg` for complex-symmetric operators.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

#### `Klu.diagnostics()`

The factorization report as a dictionary: `stages` (name,
wall_ms, flops, bytes per phase), `decisions` (ordering,
scaling, method, supernode counts), `numeric` (perturbed
pivots, 2x2 pivots, inertia), `solves` (accumulated solve
calls, right-hand sides, wall time, refinement steps),
`warnings` (settings that were ignored on this path), the
throughput `rates` (`analyze_mdof_s`, `factor_mdof_s`,
`factor_gflops`, `factor_mnnz_s`, `total_mdof_s`,
`solve_mdof_s`; million unknowns per second and GFlop/s),
the a-priori `estimate` and a one-line `summary`.

#### `Klu.gmres(b, tol=1e-08, maxit=400, restart=None, x0=None, recycle=None, operator=None)`

Flexible restarted GMRES with this factor as the preconditioner.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side.
- `tol` (float, default 1e-8): Relative residual target `||b - A x|| <= tol * ||b||`.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Arnoldi basis size; by default chosen so the basis stays under 1 GiB (between 20 and 80).
- `x0` (ndarray, optional): Initial guess (warm start).
- `recycle` (Recycle, optional): Deflation subspace carried across calls; see `recycle`.
- `operator` (tuple, optional): `(n, indptr, indices, data)` of a different CSC matrix to iterate on, with this factor as the preconditioner (the Python wrapper `rslab.gmres` builds it from a SciPy matrix). By default the factored matrix is the operator.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`; unpacks as a 5-tuple.

#### `Klu.gmres_block(b, tol=1e-08, maxit=400, restart=None, x0=None, operator=None)`

Block GMRES for an `n x nrhs` right-hand side block with this
factor as the preconditioner.

**Parameters**

- `b` (ndarray, shape (n, nrhs)): Right-hand side block.
- `tol` (float, default 1e-8): Relative residual target per column.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Block Arnoldi basis size (see `gmres`).
- `x0` (ndarray, shape (n, nrhs), optional): Initial guess.
- `operator` (tuple, optional): A different CSC matrix to iterate on, as in `gmres`.

**Returns**

- `KrylovResult`: `x` of shape `(n, nrhs)`, `converged`, `iters`, `final_res` (one value per column), `stop`.

#### `Klu.recycle(k)`

A `Recycle` workspace holding up to `k` deflation
vectors for a sequence of `gmres` calls.

**Parameters**

- `k` (int): Maximum number of deflation vectors kept.

**Returns**

- `Recycle`: The workspace to pass as `recycle=` to `gmres`.

#### `Klu.refactor(data)`

Numeric-only refactorization with new values on the **same** pattern:
no symbolic work, no pivot search. `data` is either the matrix
itself (any SciPy sparse matrix with the factored pattern; it is
sorted and summed like at analysis time, and the pattern is checked)
or its CSC value array in the factor's dtype, in the order of the
matrix that was factored.
The handle is invalid until a successful `refactor` or a fresh
factor if this raises.

**Parameters**

- `data` (scipy.sparse matrix or ndarray): The matrix with the factored pattern, or its CSC value array.

**Raises**

- `ValueError`: If the pattern differs or the value count does not match.
- `RuntimeError`: If a pivot is numerically zero.

#### `Klu.solve(b, refine=0, target=None, measure='normwise')`

Solve `A x = b` for one right-hand side.

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side in the factor's dtype.
- `refine` (int, default 0): Steps of iterative refinement against the original matrix.
- `target` (float, optional): Stop refining once the backward error is below this value (by default all `refine` steps run).
- `measure` ({'normwise', 'componentwise'}, default 'normwise'): The backward-error measure `target` refers to.

#### `Klu.solve_many(b)`

Solve `A X = B` for an `n x nrhs` block in one batched pass.

**Parameters**

- `b` (ndarray, shape (n, nrhs)): Right-hand side block in the factor's dtype.

**Returns**

- `ndarray, shape (n, nrhs)`: The solutions, one column per right-hand side.

#### `Klu.solve_transpose(b)`

Solve `A^T y = b` on the same factors (plain transpose, not the
conjugate transpose; conjugate the right-hand side and the result for
`A^H`).

**Parameters**

- `b` (ndarray, shape (n,)): Right-hand side in the factor's dtype.

**Returns**

- `ndarray, shape (n,)`: The solution `y`.

## Symbolic analysis

### `analyze(A, path: 'str' = 'auto', *, settings=None, **kwargs)`

Symbolic analysis of the pattern of `A`: analyze once, factor many.

The analysis (ordering, elimination tree, supernodes; for KLU the block
triangular form) depends only on the sparsity pattern, so a sequence of
matrices with the same pattern (frequency sweeps, Newton iterations,
parameter studies) pays it once and factors each value set through
`symbolic.factor(data)`.

**Parameters**

- `A` (scipy.sparse matrix or array-like): The `n x n` matrix whose pattern (and, for the heuristic ordering pick, values) is analyzed.
- `path` ({'auto', 'ldlt', 'lu', 'klu'}, default 'auto'): The factorization path: `'ldlt'` for symmetric matrices (the lower triangle is analyzed), `'lu'` for general ones, `'klu'` for circuit-shaped ones. `'auto'` picks `'ldlt'` when `A` is symmetric and `'lu'` otherwise.
- `settings` (Settings or KluSettings, optional): Analysis-time settings (`ordering`, `nemin`, `relax`, `reorder` for LDL^T / LU; `btf` for KLU). Numeric settings given here become the defaults of `factor`.
- `**kwargs`: The same keywords, overriding `settings`.

**Returns**

- `LdltSymbolic or LuSymbolic or KluSymbolic`: The analysis handle: fill and memory estimates, and `factor(data)`.

**Example**

```python
sym = rslab.analyze(A, path="lu", ordering="amd")
print(sym.factor_nnz, sym.estimate_memory()["factor_mb"])
for omega in frequencies:
    f = sym.factor((K + 1j * omega * C).data)    # same pattern
    x = f.solve(b)
```
**Note**

`factor(data)` takes the CSC value array in the analyzed pattern's
order, i.e. of a matrix prepared the same way (sorted indices, summed
duplicates). Build the sweep matrices from one pattern to guarantee it.

### class `LdltSymbolic`

Analysis of a symmetric pattern (ordering, elimination tree, supernodes),
from `rslab.analyze` with `path='ldlt'`. Factor any value set on
the same pattern with `factor`; the analysis is paid once.

**Attributes**

- `factor_nnz`: Predicted factor entries (the fill of ``L``).
- `front_dims`: ``(columns, rows)`` of every front.
- `level_widths`: Supernodes per tree level, root level first.
- `n`: Matrix dimension ``n``.
- `n_levels`: Levels of the supernodal elimination tree.
- `settings`: The settings the analysis adopted (including the heuristic thread pick); the defaults for :meth:`factor`.

#### `LdltSymbolic.estimate_memory(dtype='float64')`

A-priori memory and work estimate for a factor in the given dtype.

**Parameters**

- `dtype` (str, default 'float64'): The value type the factor will use (`'float64'`, `'complex128'`, `'float32'`, `'complex64'`).

**Returns**

- `dict`: `factor_bytes` / `factor_mb`, `transient_peak_bytes` / `transient_peak_mb` (the peak during the factorization), `factor_flops`, `critical_path_flops`.

#### `LdltSymbolic.factor(data, settings=None, **kwargs)`

Numeric factorization of `data` (the CSC value array of the lower
triangle, in the analyzed pattern's order, in any supported dtype).
Numeric settings (`threads`, `preconditioner`, `drop_tol`,
`pivot_u`, `scaling` ...) may be overridden per call.
Numeric factorization of new values on the analyzed pattern.

**Parameters**

- `data` (scipy.sparse matrix or ndarray): The matrix itself (any SciPy sparse matrix with the analyzed pattern: its lower triangle is taken, sorted and summed like at analysis time, and the pattern is checked entry by entry), or the CSC value array of that lower triangle in the order of the analysis.
- `settings` (Settings, optional): Numeric settings for this factorization; the analysis settings by default.
- `**kwargs`: Any settings keyword, overriding `settings`.

**Returns**

- `factor handle`: The numeric factor with the solve methods and diagnostics.

**Raises**

- `ValueError`: If the pattern or the value count differs from the analysis.
- `RuntimeError`: If a pivot is numerically zero in exact mode.

### class `LuSymbolic`

Analysis of a general pattern for the supernodal LU path, from
`rslab.analyze` with `path='lu'`. Factor any value set on the
same pattern with `factor`.

**Attributes**

- `factor_nnz`: Predicted factor entries, ``nnz(L) + nnz(U)``.
- `front_dims`: ``(columns, rows)`` of every front.
- `level_widths`: Supernodes per tree level, root level first.
- `n`: Matrix dimension ``n``.
- `n_levels`: Levels of the supernodal elimination tree.
- `settings`: The settings the analysis adopted; the defaults for :meth:`factor`.

#### `LuSymbolic.estimate_memory(dtype='float64')`

A-priori memory and work estimate for a factor in the given dtype.

**Parameters**

- `dtype` (str, default 'float64'): The value type the factor will use (`'float64'`, `'complex128'`, `'float32'`, `'complex64'`).

**Returns**

- `dict`: `factor_bytes` / `factor_mb`, `transient_peak_bytes` / `transient_peak_mb` (the peak during the factorization), `factor_flops`, `critical_path_flops`.

#### `LuSymbolic.factor(data, settings=None, **kwargs)`

Numeric factorization of `data` (the full CSC value array in the
analyzed pattern's order). Numeric settings may be overridden per call.
Numeric factorization of new values on the analyzed pattern.

**Parameters**

- `data` (scipy.sparse matrix or ndarray): The matrix itself (any SciPy sparse matrix with the analyzed pattern: its matrix is taken, sorted and summed like at analysis time, and the pattern is checked entry by entry), or the CSC value array of that matrix in the order of the analysis.
- `settings` (Settings, optional): Numeric settings for this factorization; the analysis settings by default.
- `**kwargs`: Any settings keyword, overriding `settings`.

**Returns**

- `factor handle`: The numeric factor with the solve methods and diagnostics.

**Raises**

- `ValueError`: If the pattern or the value count differs from the analysis.
- `RuntimeError`: If a pivot is numerically zero in exact mode.

### class `KluSymbolic`

Analysis of a general pattern for the KLU path (block triangular form,
per-block AMD), from `rslab.analyze` with `path='klu'`. Factor any
value set on the same pattern with `factor`.

**Attributes**

- `block_ptr`: Block boundaries in the permuted order (``n_blocks + 1`` entries).
- `factor_nnz`: Predicted factor entries.
- `max_block_size`: Dimension of the largest diagonal block.
- `n`: Matrix dimension ``n``.
- `n_blocks`: Number of diagonal blocks of the block triangular form.
- `settings`: The settings the analysis adopted; the defaults for :meth:`factor`.

#### `KluSymbolic.estimate_memory(dtype='float64')`

A-priori memory and work estimate for a factor in the given dtype.

**Parameters**

- `dtype` (str, default 'float64'): The value type the factor will use (`'float64'`, `'complex128'`, `'float32'`, `'complex64'`).

**Returns**

- `dict`: `factor_bytes` / `factor_mb`, `transient_peak_bytes` / `transient_peak_mb` (the peak during the factorization), `factor_flops`, `critical_path_flops`.

#### `KluSymbolic.factor(data, settings=None, **kwargs)`

Numeric factorization of `data` (the full CSC value array in the
analyzed pattern's order). KLU settings may be overridden per call.
Numeric factorization of new values on the analyzed pattern.

**Parameters**

- `data` (scipy.sparse matrix or ndarray): The matrix itself (any SciPy sparse matrix with the analyzed pattern: its matrix is taken, sorted and summed like at analysis time, and the pattern is checked entry by entry), or the CSC value array of that matrix in the order of the analysis.
- `settings` (Settings, optional): Numeric settings for this factorization; the analysis settings by default.
- `**kwargs`: Any settings keyword, overriding `settings`.

**Returns**

- `factor handle`: The numeric factor with the solve methods and diagnostics.

**Raises**

- `ValueError`: If the pattern or the value count differs from the analysis.
- `RuntimeError`: If a pivot is numerically zero in exact mode.

## Configuration

### class `Settings`

Settings of the symmetric LDL^T and the unsymmetric LU factorizations.

Wraps the core's `SolverSettings`. Construct it from keyword arguments
(`rslab.Settings(threads=2, ordering="metis")`) and pass it as
`settings=` to `rslab.ldlt` / `rslab.lu` /
`rslab.analyze`, or give the same keywords to those functions
directly. Unknown keywords raise `TypeError`; invalid values `ValueError`.

**Parameters**

- `ordering` ({'auto', 'auto_race', 'amd', 'amf', 'metis', 'rcm'}, optional): Fill-reducing ordering. `None` (default) uses the heuristic pick, the adaptive ordering plus an exact nested-dissection bakeoff on large systems (with a small seed ensemble once the factorization is heavy enough to pay for it); an explicit value analyzes with exactly that ordering, `'metis'` being one nested-dissection run. The ordering actually used is reported in `diagnostics()['decisions']`.
- `nemin` (int, optional): Supernode amalgamation threshold (default 16). Smaller means finer supernodes: less fill, more per-front overhead.
- `relax` (bool or (int, int), optional): Relaxed (fill-tolerant) amalgamation. `True` (default) keeps the built-in thresholds, `False` disables it, a pair `(max_width, max_extra_rows)` sets them explicitly.
- `reorder` ({'hybrid_liu', 'off'}, optional): Child reordering of the elimination tree: `'hybrid_liu'` (default) shrinks the contribution-stack peak, `'off'` keeps the natural leaf order for maximum leaf parallelism. Worker budget of the scoped factorization pool. `None` (default) is the per-matrix predictor capped at 4 workers (or the calibrated pick after `rslab.install_diagnose`); an `int` pins the count (`0` = all logical cores); `'auto'` is the predictor without the cap, `('auto', max)` the predictor capped at `max`; `'ambient'` runs on the caller's rayon pool. The factor is bit-identical for every value.
- `preconditioner` (float, optional): Static-pivot floor: a pivot with magnitude below it is lifted to it, so the factorization never fails and produces the factor of a nearby `A + E`. Recover accuracy with `solve(b, refine=k)`. `1e-4` is a good start.
- `force_accept` (bool, default False): In exact mode, accept tiny pivots instead of raising on rank deficiency. Ignored when `preconditioner` is set.
- `drop_tol` (float, optional): Incomplete-factorization threshold: fill below it (relative to the column) is discarded, turning the factor into an ILU-style preconditioner. `None` keeps the complete factor.
- `method` ({'left_looking', 'multifrontal'}, default 'left_looking'): Numeric schedule; same factor, different transient memory and parallel profile.
- `memory` ({'low', 'eager'}, default 'low'): Factor emit strategy: `'low'` frees each front as soon as it is emitted, `'eager'` keeps them resident.
- `pivot_u` (float, optional): Threshold partial-pivoting tolerance of the LU path in `[0, 1]` (default 0.1; `1.0` is full partial pivoting). Ignored, and reported in the diagnostics, on the LDL^T path.
- `scaling` ({'one_pass', 'inf_norm', 'mc64', 'auto', 'identity'} or array, optional): Symmetric equilibration before the LDL^T factorization: a named strategy, or a float array `s` of length `n` applying the external scaling `diag(s) A diag(s)`. The LU path uses its own two-sided scaling and reports a set value.
- `matching` (bool, default True): Maximum-product row matching (MC64) before the LU analysis: rows are permuted so the matched entries form the diagonal and both sides are scaled to unit magnitude there, which keeps the element growth of the front-restricted pivoting bounded. LU path only.
- `blr` (float or False or dict, optional): Block-low-rank compression of the contribution blocks. A float is the relative tolerance with the default block parameters; a dict `{'eps': tol, 'min_cnrow': 256, 'b': 256, 'adaptive': False}` sets the smallest contribution block that is compressed, the block size and adaptive per-vector precision; `False` (default) keeps exact dense fronts.
- `panel_nb` (int, optional): Panel width (blocking factor) of the dense kernels, default 64.
- `interrupt` (Interrupt, optional): A cancellation flag polled by the numeric phase.

**Other Parameters**

- `scalar_gate` (int, optional): Flop count below which an update runs as a scalar loop (benchmark knob; the default is calibrated).
- `par_gemm` (int, optional): Flop count at or above which the front GEMM runs in parallel (calibrated default).
- `par_cdiv` (int, optional): Flop count at or above which the panel-trailing update runs in parallel (calibrated default).
- `use_gemm_schur` (bool, optional): Use the SIMD GEMM (`True`, default) or the scalar loop for the front Schur update.

#### `Settings.to_dict()`

The settings as a plain dictionary, one key per keyword argument.

### class `KluSettings`

Settings of the KLU (circuit) path.

**Parameters**

- `pivot_tol` (float, default 1e-3): Diagonal-preference threshold: the diagonal entry is the pivot when `|a_jj| >= pivot_tol * max_i |a_ij|`; `1.0` is plain partial pivoting.
- `row_scaling` (bool, default True): Divide each row by its max-magnitude entry before factoring.
- `btf` (bool, default True): Permute to block upper triangular form first (keep it on).
- `matching` (bool, default True): Maximum-product row matching (MC64) as the transversal of the block triangular form, so the diagonal-preference pivoting rarely leaves the diagonal; needs `btf`.
- `parallel` (bool, optional): Per-block parallel factor / refactor over the BTF blocks. `None` (default) is the structural auto gate (at least 4 blocks, 8000 nonzeros, no dominant block); `True` / `False` force it. The result is bit-identical in every mode.
- `interrupt` (Interrupt, optional): A cancellation flag polled by the numeric phase.

#### `KluSettings.to_dict()`

The settings as a plain dictionary, one key per keyword argument.

### class `Interrupt`

A caller-owned cancellation flag for a running factorization.

Pass it as `interrupt=` to `Settings` / `KluSettings` (or
as a keyword to the factor functions). The numeric phase polls the flag at
supernode and panel boundaries and stops with `RuntimeError("interrupted")`
once `cancel` was called, typically from another thread while the
factorization runs with the GIL released. `reset` re-arms the flag.

**Example**

```python
stop = rslab.Interrupt()
threading.Timer(2.0, stop.cancel).start()      # give up after 2 s
f = rslab.lu(A, interrupt=stop)
```

**Attributes**

- `is_set`: ``True`` once :meth:`cancel` was called and not yet :meth:`reset`.

#### `Interrupt.cancel()`

Request cancellation of the factorization(s) carrying this flag.

#### `Interrupt.reset()`

Clear the flag so later factorizations run to completion again.

## Krylov solvers

### `gmres(A, b, M=None, *, tol: 'float' = 1e-08, maxit: 'int' = 400, restart: 'int | None' = None, x0=None, recycle: 'Recycle | None' = None) -> 'KrylovResult'`

Flexible restarted GMRES on `A x = b`, optionally preconditioned.

**Parameters**

- `A` (scipy.sparse matrix or array-like): The operator (any square matrix; converted to CSC).
- `b` (ndarray, shape (n,)): Right-hand side; cast to the operator's (or preconditioner's) dtype.
- `M` (Ldlt or Lu or Klu, optional): A factor handle used as the preconditioner, e.g. an incomplete or low-precision factor of `A` or a factor of a nearby matrix. `None` runs unpreconditioned.
- `tol` (float, default 1e-8): Relative residual target `||b - A x|| <= tol * ||b||`.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Arnoldi basis size; by default chosen so the basis stays under 1 GiB (between 20 and 80).
- `x0` (ndarray, optional): Initial guess (warm start).
- `recycle` (Recycle, optional): Deflation subspace carried across calls (needs `M`; create it with `M.recycle(k)`).

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`; unpacks as a 5-tuple.

**Example**

```python
M = rslab.lu(A, drop_tol=1e-3)                    # ILU-style preconditioner
x, ok, iters, res, stop = rslab.gmres(A, b, M, tol=1e-10)
r = rslab.gmres(A2, b, M)                         # M reused on a nearby A2
```

### `gmres_block(A, B, M=None, *, tol: 'float' = 1e-08, maxit: 'int' = 400, restart: 'int | None' = None, x0=None) -> 'KrylovResult'`

Block GMRES on `A X = B` for an `n x nrhs` block, optionally preconditioned.

**Parameters**

- `A` (scipy.sparse matrix): The `n x n` operator.
- `B` (ndarray, shape (n, nrhs)): Right-hand side block.
- `M` (factor handle, optional): A factor used as the preconditioner.
- `tol` (float, default 1e-8): Relative residual target per column.
- `maxit` (int, default 400): Iteration budget.
- `restart` (int, optional): Block Arnoldi basis size (see `gmres`).
- `x0` (ndarray, shape (n, nrhs), optional): Initial guess.

**Returns**

- `KrylovResult`: `x` of shape `(n, nrhs)`, `converged`, `iters`, `final_res` (one residual per column), `stop`.

### `cocg(A, b, M=None, *, tol: 'float' = 1e-08, maxit: 'int' = 400) -> 'KrylovResult'`

Conjugate orthogonal conjugate gradient (COCG) on `A x = b`.

The short-recurrence Krylov method for **complex-symmetric**
(`A = A^T`) and real symmetric operators: constant
memory, one matrix-vector product per iteration, no restart.

**Parameters**

- `A` (scipy.sparse matrix): The `n x n` operator.
- `b` (ndarray, shape (n,)): Right-hand side.
- `M` (factor handle, optional): A factor (`ldlt`, `lu`, `klu`, typically incomplete or low-rank) used as the preconditioner.
- `tol` (float, default 1e-8): Relative residual target `||b - A x|| <= tol * ||b||`.
- `maxit` (int, default 400): Iteration budget.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

### `cocr(A, b, M=None, *, tol: 'float' = 1e-08, maxit: 'int' = 400) -> 'KrylovResult'`

Conjugate orthogonal conjugate residual (COCR) on `A x = b`.

The minimal-residual sibling of `cocg` for complex-symmetric
operators (smoother residual history).

**Parameters**

- `A` (scipy.sparse matrix): The `n x n` operator.
- `b` (ndarray, shape (n,)): Right-hand side.
- `M` (factor handle, optional): A factor used as the preconditioner.
- `tol` (float, default 1e-8): Relative residual target.
- `maxit` (int, default 400): Iteration budget.

**Returns**

- `KrylovResult`: `x`, `converged`, `iters`, `final_res`, `stop`.

### class `KrylovResult`

Outcome of an iterative solve.

**Attributes**

- `x` (ndarray): The iterate (`n` or `n x nrhs`).
- `converged` (bool): Whether the residual target was met.
- `iters` (int): Iterations (matrix-vector products) run.
- `final_res` (float or ndarray): Final relative residual (one value per column for block solves).
- `stop` (str): `'converged'`, `'max_iter'`, `'breakdown'` or `'stalled'`.

- `Unpacks as the 5-tuple `(x, converged, iters, final_res, stop)`.`:

**Attributes**

- `converged`
- `final_res`
- `iters`
- `stop`
- `x`

### class `Recycle`

A deflation-subspace carrier for `Ldlt.gmres` / `Lu.gmres` /
`Klu.gmres` sweeps, created by `handle.recycle(k)`.

Holds up to `k` recycle vectors across calls (harvested Ritz vectors of
the preconditioned operator), so a sequence of related solves converges in
fewer iterations than cold or warm starts.

**Attributes**

- `k` (int): Target subspace dimension.
- `active` (int): Vectors currently held.
- `dtype` (str): Scalar field, matching the handle that created it.

**Attributes**

- `active`
- `dtype`
- `k`

#### `Recycle.clear()`

Drop the held vectors (the next solve starts cold).

## Machine calibration and logging

### `install_diagnose()`

One-time machine calibration: measure this machine's factorization
throughput and thread-speedup curve and cache them, so later factor calls
pick their worker count from the measurement. Returns the measured values.

### `set_log_level(level)`

Set the log level of the solver core.

**Parameters**

- `level` (str): `'debug'`, `'info'`, `'warning'` (default), `'error'` or `'off'`. The environment variable `RLA_LOG` sets the initial level.

**Returns**

- `None`:

### `log_level()`

The current log level of the solver core.

**Returns**

- `str`: The level name in lowercase.

### `set_log_sink(sink)`

Route the core's log messages to `sink(level, message)` instead of the
default stdout / stderr writer; `None` restores the default. The sink
receives the lowercase level name and the bare message (no timestamp), so
a `logging.Logger` fits directly::

    log = logging.getLogger("rslab")
    rslab.set_log_sink(lambda level, msg: log.log(logging.getLevelName(level.upper()), msg))

The sink may be called from solver worker threads.

**Parameters**

- `sink` (callable or None): `sink(level, message)`; `None` restores the default writer.

**Returns**

- `None`:
