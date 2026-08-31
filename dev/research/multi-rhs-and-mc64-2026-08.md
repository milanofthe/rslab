# Multi-RHS solve and the MC64 heap: two ports measured and declined (2026-08-31)

Packages 5 and 6a of `dev/plans/feral-uptake-2026-08.md`. Both were taken from
upstream measurements that do not reproduce here, for reasons that are specific
to how RSLAB is built. Apple M3, 8 threads, minimum over repetitions.

## Multi-RHS: no alignment penalty to recover

Upstream found their BLAS-3 multi-RHS path losing 1.4-1.8x when `nrhs` is not a
multiple of 8, because a row stride of exactly `nrhs` puts every panel row across
cache lines, and recovered 1.36x geomean by padding the stride to whole 8-column
tiles.

RSLAB has no BLAS-3 multi-RHS path: `solve_ldlt_many` is a block AXPY over a
row-major `n x nrhs` buffer, parallelized by splitting the right-hand sides. The
question was therefore whether the same misalignment costs us anything. It does
not. Helmholtz $28^3$ ($n = 21952$, fill 3.31M), cost per right-hand side and the
rate the whole solve achieves:

| nrhs | solve ms | ms per RHS | vs nrhs=32 | GF/s |
|--:|--:|--:|--:|--:|
| 8 | 36.4 | 4.545 | 3.25 | 5.0 |
| 15 | 31.6 | 2.107 | 1.51 | 10.3 |
| 16 | 31.7 | 1.981 | 1.42 | 11.0 |
| 17 | 32.3 | 1.901 | 1.36 | 11.5 |
| 30 | 43.8 | 1.461 | 1.05 | 26.5 |
| 31 | 42.9 | 1.385 | 1.04 | 28.7 |
| 32 | 42.7 | 1.333 | 1.00 | 29.8 |
| 33 | 42.6 | 1.290 | 0.96 | 30.8 |
| 48 | 54.7 | 1.139 | 0.85 | 34.9 |
| 63 | 60.8 | 0.966 | 0.72 | 41.2 |
| 64 | 61.9 | 0.968 | 0.72 | 41.1 |

The per-RHS cost falls smoothly with the block width; `nrhs = 31` sits between 30
and 32, and 15 and 17 sit either side of 16 in the same monotone sequence. There
is no residue hump, because the AXPY kernel streams the factor once and applies
each entry to the whole row, which never assumes a tile boundary.

The same table answers the second half of the package, the BLAS-3 panel path
itself. The rate column counts two triangular sweeps at one complex FMA per
stored entry and right-hand side, six real flops each. At `nrhs = 64` the block
solve reaches **41 GF/s**, which is where the packed complex GEMM lands on hot
operands of comparable shape (31-40 GF/s, `lu-convdiff-2026-08.md`). A panel path
would have to beat that rate while adding the gather that assembles panels from
the compact factor. There is no headroom to build into. Declined.

## MC64 Hungarian heap: the inline key costs more than it saves

Upstream stores the distance key inline in the heap entry instead of reading
`d[heap[p]]` during the sift, and measured 4-5% on large matchings,
bit-identical. Implemented here the same way (a `key` array parallel to `heap`,
maintained by every move), it is a **regression**. The matching alone, on a
banded-plus-arrow cost graph:

| n | nnz | old heap | inline key |
|--:|--:|--:|--:|
| 20 000 | 121 497 | **37.9 ms** | 40.3 ms |
| 60 000 | 363 009 | **98.6 ms** | 103.9 ms |

The heap here holds the *touched* set of one augmenting search, not the whole
row set: it is reset per search and stays small, so `d[idx]` is already resident
and the indirection costs nothing, while the inline key doubles the bytes each
move writes. Declined; the heap keeps its index-only form.

## The other two items of package 6

* **Scaling cache fingerprint.** Upstream fixed a cache that rejected matrices
  against their own fingerprint. RSLAB has no cross-call scaling cache: the MC64
  matching is computed and consumed inside one analysis (`compute_matching` ->
  `scaling_from_cache`). Nothing to fix.
* **Thread-pool fallback.** `in_scoped_pool` already falls back to running the
  closure on the calling thread when the pool cannot be built
  (`Err(_) => f()`), which is the behaviour the upstream fix introduced.
