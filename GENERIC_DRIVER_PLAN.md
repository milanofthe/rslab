# Generic-driver refactor — plan

**Goal:** make rslab's *optimized* f64 multifrontal path generic over the
`Scalar` field so `Complex<f64>` (PARDISO mtype 6) reuses the mature driver
(delayed pivoting, supernode amalgamation, scaling) instead of the slower
correctness-first reimplementation in `numeric/multifrontal_generic.rs` +
`dense/ldlt_generic.rs`. The f64 SIMD blocked kernel stays as the f64
specialization; complex gets a scalar leaf kernel.

**Key technical unlock:** the per-front dense factorization becomes a *trait
method*. `f64` implements it via the existing blocked/SIMD kernel; `Complex<f64>`
via a scalar Bunch-Kaufman kernel (already present in `ldlt_generic` /
`multifrontal_generic::factor_front`). The driver calls `T::factor_front(...)`;
monomorphization picks the impl at compile time — no `unstable` specialization.

## Why this is one monolithic change

The value-carrying types cascade and cannot be genericized in isolation:

```
FrontalFactors.{l,d_diag,d_subdiag,contrib}: Vec<f64>
  → ContribBlock.contrib: Vec<f64>
    → NodeFactors / SparseFactors
      → numeric/factorize.rs driver (assembly, extend-add, delayed pivots)
```

All must move to `Vec<T>` together; the f64 tests only go green again at the end.
Do it on this branch, in one focused push, with `cargo test` as the guard.

## Stages (tracked as tasks #11–#14)

- **A. Front-kernel trait seam.** Genericize `FrontalFactors<T>` / `Factors<T>`
  (value fields → `Vec<T>`; `d_eq` stays `Vec<f64>` — equilibration is real;
  `zero_tol` stays `f64`). Gate `Inertia`: it is meaningless for complex
  symmetric (no real inertia) — keep the field but compute it only on the f64
  path (zeroed otherwise), or make it `Option`. Define the `factor_front` trait
  method: f64 → existing blocked kernel, complex → scalar kernel.
- **B. Driver generic.** `numeric/factorize.rs` over `T`: `SparseFactors<T>`,
  `ContribBlock<T>`, extend-add, delayed pivoting. Inertia gated.
- **C. Complex over the optimized path.** Point `SparseSymmetricLdlt` at the
  optimized generic driver; validate complex (residuals + oracle); benchmark
  against PARDISO mtype 6 (MKL C API via ctypes, since pypardiso is real-only).
- **D. Retire duplicates.** Fold the scalar leaf kernel into the trait impl;
  remove `multifrontal_generic` + `ldlt_generic`; unify the public API.

## Guardrails

- Keep `dense/schur_kernel.rs` (pulp x86-v3) f64-only — complex arithmetic does
  not map onto f64 SIMD lanes; a complex kernel is intrinsic (PARDISO has
  separate real/complex code paths too).
- No loosening of f64 test tolerances. The ~600-test f64 suite is the contract.
- `#![deny(clippy::unwrap_used / expect_used)]` in `src/` still holds.
