# RLA — Rust Linear Algebra

A pure-Rust **sparse direct solver** for **complex symmetric** systems (A = Aᵀ with
complex entries, PARDISO `mtype 6`). The goal is a PARDISO-style supernodal /
multifrontal **LDLᵀ factorization**, parallelized with **rayon**. No BLAS, no LAPACK,
no Fortran — pure Rust on the stable toolchain.

## Status

Forked from [feral](https://github.com/jkitchin/feral) (MIT, © John Kitchin) and thinned
down to the core path: `sparse → ordering → symbolic → numeric → dense`. The real `f64`
path is feral's and is validated against MUMPS 5.8.2 + SPRAL/SSIDS. Generalizing the
scalar type to `Complex<f64>` (complex-symmetric pivoting, no conjugation) is in progress.

## Layout

| Path | Contents |
|------|----------|
| `src/sparse` | CSC sparse matrix |
| `src/ordering` | AMD elimination ordering, elimination tree, postorder |
| `crates/feral-{amd,amf,metis,scotch,kahip}` | Fill-reducing orderings (AMD + nested dissection, pure Rust) |
| `src/symbolic` | Symbolic factorization, supernode amalgamation |
| `src/numeric` | Multifrontal numeric factorization (rayon), solve, iterative refinement |
| `src/dense` | Dense per-front LDLᵀ kernel (Bunch-Kaufman pivoting) |
| `src/scaling` | MC64 / inf-norm equilibration |

## Build

```
cargo build
cargo test
```

## License

MIT. Based on feral © 2026 John Kitchin — see `LICENSE` and `NOTICE`.
