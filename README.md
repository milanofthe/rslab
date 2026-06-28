# RSLAB

> **R**ust **S**parse **L**inear **A**lgebra **B**ackend — a pure‑Rust sparse **direct
> solver** (symmetric **LDLᵀ** + unsymmetric **LU**) and **preconditioner**: a
> PARDISO‑style, embeddable, data‑type‑agnostic replacement with **no MKL or other
> native dependency**.

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![pure Rust](https://img.shields.io/badge/deps-pure%20Rust-orange.svg)](#why-rslab)

RSLAB factors **real** and **complex** sparse matrices as `Pᵀ A P = L D Lᵀ`
(complex‑symmetric, PARDISO `mtype 6`) or `Pᵀ A P = L U` (unsymmetric, `mtype 13`)
with a rayon‑parallel supernodal kernel and a SIMD Schur update — then solves
against many right‑hand sides, or drives a Krylov iteration as a robust
low‑memory preconditioner.

---

## Why RSLAB

- **Pure Rust, zero native deps** in the solver core. No MKL, no OpenBLAS, no FFI.
  (Optional bench/tooling features may load external libraries; the library itself
  never does.)
- **One code path for every scalar.** Generic over `Scalar`: `f64`, `f32`,
  `Complex<f64>`, `Complex<f32>` — verified end‑to‑end, not just claimed.
- **Two factorizations, one API.** Complex‑symmetric **LDLᵀ** with Bunch‑Kaufman
  1×1/2×2 pivoting (exploits `A = Aᵀ` for 2× savings over a general LU), and a
  threshold‑pivoted **LU** for unsymmetric/MoM matrices.
- **Low transient memory.** A supernodal **left‑looking** path frees each dense
  panel the instant its last consumer has pulled from it (no contribution‑block
  stack), so the working set stays close to the factor size.
- **Deterministic.** The numeric factor is **bit‑identical regardless of thread
  count** — reproducible results for solver‑in‑the‑loop pipelines.
- **Solver‑in‑the‑loop infrastructure.** Scoped per‑solve thread pools (so many
  concurrent solves share the machine), per‑call diagnostics, an **a‑priori**
  peak‑memory + runtime estimate (before any numeric work), and a hardware‑aware
  **budget governor**.
- **Iterative side included.** Restarted GMRES (+ block / multi‑RHS), COCG/COCR,
  with the (possibly incomplete / mixed‑precision) factor as the preconditioner.

---

## Benchmarks

Measured on 12 cores / 24 threads, against **faer** (pure‑Rust sparse LU) and
**Intel MKL PARDISO**, on generated test matrices (3D complex‑symmetric Helmholtz
for the LDLᵀ path; MoM near‑field kernels for the LU path) and the real MoM
`precond_matrices`. Figures have transparent backgrounds and neutral axes — they
read on light and dark pages alike. Reproduce with `python benches/run_bench.py`.

### Scaling with problem size (DOFs)

![Factor time vs DOFs](benches/bench_out/scaling_factor.png)
![Factor memory vs DOFs](benches/bench_out/scaling_memory.png)

RSLAB‑LL beats faer on factor **time and memory** across the sweep and tracks the
same complexity as PARDISO with a larger constant. PARDISO leads on raw factor
throughput (decade‑tuned MKL kernels); RSLAB sits between PARDISO and faer.

### Thread scaling

![Thread scaling](benches/bench_out/thread_scaling.png)

Sparse‑direct factorization is fundamentally hard to scale (work concentrates in a
few large supernodes). PARDISO reaches ~5×, RSLAB ~2× before saturating; both are
far from ideal — see the [architecture notes](#architecture).

### Wall‑clock & memory breakdown

![WCT breakdown](benches/bench_out/wct_breakdown.png)
![A-priori memory breakdown](benches/bench_out/memory_breakdown.png)

The memory breakdown is the **a‑priori estimate** (no factoring required): dense
panels + compact factor + input/scratch, with the panel‑freed live floor marked.
RSLAB predicts its own peak memory before allocating, within ~1.0–1.2× of the
measured value (conservative, never under).

### Real MoM matrices (realism anchor)

![Real MoM matrices](benches/bench_out/real_matrices.png)

On the real application matrices, RSLAB‑LL is **lighter than faer in both time and
memory** and ~2× lighter than its own multifrontal path (panel‑freeing).

---

## Quick start

```toml
[dependencies]
rslab = "0.11"
```

### Symmetric direct solve (LDLᵀ)

```rust
use rslab::prelude::*;

// Real symmetric, lower triangle (i ≥ j).
let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0],
                                        &[2.0, 2.0, 2.0, -1.0])?;
let sym = LdltSymbolic::analyze(&a)?;            // phase 1: analyze pattern once
let f   = sym.factor(&a, &FactorOptions::default())?;  // phases 2–3: factor
let x   = f.solve(&[1.0, 2.0, 3.0])?;            // solve A x = b
# Ok::<(), rslab::RslabError>(())
```

### Unsymmetric direct solve (LU)

```rust
use rslab::prelude::*;
use num_complex::Complex;

let c = |re, im| Complex::new(re, im);
let a = GeneralCsc::from_triplets(2, &[0, 1, 0, 1], &[0, 1, 1, 0],
                                  &[c(2., 0.), c(2., 0.), c(1., 0.), c(-1., 0.)])?;
let f = LuSymbolic::analyze(&a)?.factor(&a, &FactorOptions::default())?;
let x = f.solve(&[c(1., 0.), c(0., 1.)])?;
# Ok::<(), rslab::RslabError>(())
```

### Preconditioned iteration (the MoM workflow)

```rust
use rslab::prelude::*;
# use num_complex::Complex;
# let c = |re, im| Complex::new(re, im);
# let a = CscMatrix::<Complex<f64>>::from_triplets(3, &[0,1,2,1], &[0,1,2,0],
#     &[c(4.,1.), c(4.,1.), c(4.,1.), c(-1.,0.2)])?;
// Never‑fail static pivoting + incomplete dropping ⇒ a robust, light preconditioner.
let opts = FactorOptions::preconditioner(1e-8).with_drop_tol(1e-2);
let m = LdltSolver::factor_with(&a, &opts)?;
let b = vec![c(1.0, 0.0); 3];
let res = cocg(&a, &b, &m, 1e-10, 100)?;
assert!(res.converged);
# Ok::<(), rslab::RslabError>(())
```

---

## API reference

### Phased workflow

RSLAB follows PARDISO's analyze‑once / factor‑many model:

| Phase | Symmetric | Unsymmetric |
|-------|-----------|-------------|
| 1 — analyze pattern | `LdltSymbolic::analyze(&a)` | `LuSymbolic::analyze(&a)` |
| 2–3 — factor values | `sym.factor(&a, &opts)` → `LdltSolver<T>` | `sym.factor(&a, &opts)` → `LuSolver<T>` |
| solve | `f.solve(&b)` / `f.solve_many(&b, nrhs)` | `f.solve(&b)` / `f.solve_many(&b, nrhs)` |

One‑shot convenience: `LdltSolver::factor(&a)` / `LuSolver::factor(&a, &opts)`.

### `FactorOptions`

Composable builder over the numeric and execution knobs:

```rust
use rslab::{FactorOptions, FactorMethod, BlrMode};
let opts = FactorOptions::default()      // exact, fail on singular pivot
    .with_threads(0)                     // 0 = all cores; default 2 (in‑the‑loop)
    .with_method(FactorMethod::LeftLooking); // or ::Multifrontal
// Preconditioner mode: never‑fail static pivoting + optional incomplete drop / BLR.
let pc = FactorOptions::preconditioner(1e-8)
    .with_drop_tol(1e-2)
    .with_blr(BlrMode::contribution_blocks(1e-4));
```

| Knob | Method | Meaning |
|------|--------|---------|
| pivot policy | `preconditioner(floor)` / `exact()` | static perturbation vs fail‑fast |
| incomplete | `with_drop_tol(τ)` | drop fill `< τ` (relative) |
| BLR | `with_blr(BlrMode::…)` | block‑low‑rank compression of big fronts |
| method | `with_method(FactorMethod::…)` | `LeftLooking` (default) or `Multifrontal` |
| **threads** | `with_threads(n)` | **scoped** pool of `n` workers (`0` = all, **default 2**) |
| memory | `with_memory(MemoryMode::…)` | emit/transient strategy |

The factor is **bit‑identical regardless of `threads`** — the thread budget only
affects time and the transient working set, never the result.

### Solver handles

```rust
# use rslab::prelude::*;
# fn demo(f: &LdltSolver<f64>, b: &[f64]) -> Result<(), rslab::RslabError> {
let x  = f.solve(b)?;                 // single RHS
let xs = f.solve_many(b, 4)?;         // 4 RHS at once (row‑major n×nrhs), amortized
let nnz = f.factor_nnz();             // fill (nnz of L, or L+U)
let diag = f.diagnostics();           // per‑call factor diagnostics (below)
# Ok(()) }
```

### Diagnostics — `solver.diagnostics()`

Per‑call, concurrency‑safe (no global state): measured factor time, fill, thread
budget, and the a‑priori memory estimate. Print with `Display`.

```rust
# use rslab::prelude::*;
# fn demo(f: &LdltSolver<f64>) {
let d = f.diagnostics();
println!("{d}");                      // stages, threads, factor_nnz
let est = d.estimate.unwrap();        // the a‑priori MemoryEstimate
# }
```

### A‑priori estimate — `sym.estimate_memory::<T>()`

A pure, deterministic function of the analyzed structure — call it **before** any
numeric work to fail‑fast or pick an approximation:

```rust
use rslab::prelude::*;
use num_complex::Complex;
# fn demo(a: &CscMatrix<Complex<f64>>) -> Result<(), rslab::RslabError> {
let sym = LdltSymbolic::analyze(a)?;
let est = sym.estimate_memory::<Complex<f64>>();
println!("{est}");                                  // transient peak, factor nnz, …
let runtime_ms = est.est_runtime_ms(2.0 /*Gflop/s*/, 4.0 /*speedup*/);
if !est.fits_in(8 << 30) { /* > 8 GiB — approximate or queue */ }
# Ok(()) }
```

### Iterative solvers

Right‑preconditioned **GMRES** (restarted, DGKS reorthogonalization), block /
multi‑RHS GMRES, and **COCG/COCR** for complex‑symmetric systems. Any
`LinearOperator` + `Preconditioner` compose — the factor is a preconditioner.
Mixed precision is free: factor in `Complex<f32>` and use it as a
`LowPrecisionPreconditioner` for an `f64` GMRES.

### Hardware‑aware tuning (`tuning` feature)

```rust
# #[cfg(feature = "tuning")]
# fn demo(sym: &rslab::LuSymbolic) {
use rslab::tuning::{HardwareInfo, Calibration, Budget, plan};
let hw    = HardwareInfo::probe();                  // cores + RAM
let calib = Calibration::load_or_measure(&hw);      // measured throughput, cached
let est   = sym.estimate_memory::<f64>();
let budget = Budget { max_mem_bytes: Some(4 << 30), allow_mixed_precision: true,
                      allow_drop_tol: Some(1e-3), ..Default::default() };
let p = plan(&est, &budget, &hw, &calib);           // tuned opts + predictions
// p.opts (threads / drop / BLR), p.use_mixed_precision, p.est_peak_bytes,
// p.est_runtime_ms, p.fits, p.note
# }
```

`plan` is a pure function of `(estimate, budget, hw, calibration)` — given a cached
calibration, scheduling decisions are **reproducible**.

### Test‑matrix generators (`matgen` feature)

Parametrized generators spanning size / structure / symmetry / conditioning /
density, plus a tagged catalog and an optional SuiteSparse downloader:

```rust
# #[cfg(feature = "matgen")]
# fn demo() {
use rslab::matgen::{self, stencil, bem};
let a = stencil::laplacian::<f64>(&[64, 64, 64], &stencil::StencilOpts::default()); // 3D Poisson
let k = bem::kernel(8000, &bem::BemOpts::default());            // complex‑unsymmetric MoM
for spec in matgen::catalog() { let _ = spec.name; }            // tagged catalog
# }
```

---

## Determinism & type‑agnosticism

- **Type‑agnostic:** the entire `analyze → factor → solve` pipeline is generic over
  `Scalar` (`f64`/`f32`/`Complex<f64>`/`Complex<f32>`); the estimator scales with
  `size_of::<T>()`. This is enforced by a test that factors all four types through
  both paths.
- **Deterministic:** the factor's `L`/`U`/`D` are bit‑identical for any thread
  count (verified via fill/value invariance), and all estimates are pure functions
  of the symbolic structure.

## Architecture

- **Ordering** — nested dissection (METIS/Scotch) with an AMD/AMF fallback chosen
  by a size/structure heuristic.
- **Supernodal left‑looking (default)** — each panel pulls BLAS‑3 updates from its
  factored descendants, then a blocked in‑place panel factorization (Bunch‑Kaufman
  for LDLᵀ, threshold partial pivoting for LU). Panels are compacted and freed the
  moment their last consumer is done → low transient memory.
- **Multifrontal (opt‑in)** — assembly tree of dense fronts; kept for cross‑checking
  and fronts where the extract layout is preferable.
- **Parallelism** — rayon over the assembly tree + a SIMD (`gemm`) complex Schur
  kernel, run in a **scoped pool**. Sparse‑direct thread scaling saturates early
  because the work concentrates in a few large top‑of‑tree supernodes; this is a
  property of the method, not the kernel (a standalone complex GEMM scales ~10×).

## Cargo features

| Feature | Adds |
|---------|------|
| *(default)* | the solver core — pure Rust, no extra deps |
| `matgen` | parametrized test‑matrix generators + tagged catalog |
| `matgen-download` | SuiteSparse / Matrix Market fetcher (pure‑Rust HTTP/gzip/tar) |
| `tuning` | hardware probe + calibration cache + budget governor (pulls `sysinfo`) |

## License

MIT © 2026 Milan Rother. See [LICENSE](LICENSE).
