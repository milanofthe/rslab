# KLU numeric-kernel parity plan (vs SuiteSparse KLU)

Status: DONE (2026-08-01). Landed: packages 0, 1, 2, 3, 4, and most of 5,
plus Eisenstat-Liu symmetric pruning and a lazy (explicit-only) a-priori
estimate in `factor()`. Result (same-run, Apple M3, f64 MNA):
sequential factor within 1.2-1.6x of SuiteSparse KLU (was 2.8-4.3x behind),
refactor within 1.15-1.4x, and the opt-in bit-identical parallel per-block
factor 1.5-2.3x FASTER than SuiteSparse (2k/10k/50k/200k: 0.7/2.6/19.8/81 ms
vs 1.1/5.9/32.3/152 ms). Fill and block structure unchanged (parity kept).

Follow-up (2026-08-01, same day): worker-side exact reserves, a two-phase
parallel splice (prefix-sum offsets, disjoint per-block chunk copies), and a
parallel per-block refactor sharing the factor-time opt-in. Parallel factor
speedup 3.1-3.9x, refactor 3.9-4.5x (no splice on that path); vs SuiteSparse
KLU: factor 1.6-2.7x faster, refactor 2.1-3.4x faster, 20-point sweep ~2.4x
faster. Negative result, kept for the record: software prefetch of the known
future scatter targets (prfm pldl1keep, 8 entries ahead) made every phase
measurably slower on the M3; the out-of-order window already covers the
latency, and the per-iteration prefetch overhead dominates. Reverted.

Original plan below. Baseline measurement in `benches/klu_circuit.rs`
(`suitesparse-klu` rows, Apple M3, f64 MNA family): structure parity (identical
BTF blocks, fill within 1.5%, RSLAB slightly less), but the C numeric kernel is
2.8-4.3x faster on factor and 1.5-2.3x on refactor. The gap is kernel maturity
in the per-column Gilbert-Peierls loop, not structure.

| n | RSLAB factor | SS factor | RSLAB refactor | SS refactor |
|--:|--:|--:|--:|--:|
| 2k | 7.2 ms | 2.5 ms | 2.2 ms | 1.5 ms |
| 10k | 25.3 ms | 5.8 ms | 6.4 ms | 3.4 ms |
| 50k | 118 ms | 31.6 ms | 31.1 ms | 17.9 ms |
| 200k | 357 ms | 128 ms | 125 ms | 78.3 ms |

## Work packages, by expected leverage

0. **Profile first.** Split factor time into DFS-reach / scatter-gather /
   pivot-search on the 50k matrix (`RLA_PROFILE`-style counters). Decisions
   below get re-ranked by what actually dominates.

1. **32-bit indices in the KLU path.** The kernel is index-chasing and
   memory-bound; `usize` doubles index traffic vs KLU's `int32`. Lossless for
   `n < 2^31`. Pattern exists in-house (`CompressedLdltFactors`). Expected:
   the single largest step toward parity.

2. **Bounds-check-free hot loops.** DFS reach, column scatter, refactor
   replay: iterators or `get_unchecked` with `debug_assert!` backing.
   Reciprocal-multiply instead of per-pivot division.

3. **Refactor as a precompiled scatter program (overtake opportunity).**
   Pattern and pivots are frozen after factor; record a flat list of
   source/destination position spans once, then refactor becomes a single
   linear replay without index resolution. Fuse refactor+solve for the sweep
   loop. SuiteSparse KLU re-resolves positions every refactor; this is our
   structural advantage for the frequency-sweep use case.

4. **Parallel factor over BTF blocks (overtake opportunity).** Diagonal
   blocks factor independently; the MNA family has 8-64 blocks. A scoped pool
   over blocks stays bit-deterministic (each block is sequential; the result
   does not depend on scheduling). Opt-in via `Threads::Ambient`-style knob;
   default stays strictly sequential for embedded use. Realistic 3-5x on
   factor at 64 blocks / 8 cores.

5. **Small-column hygiene** (4-5 nnz/col makes loop overhead dominant):
   exact preallocation from the pattern-only pass, L+U of a column contiguous
   in one stream, timestamp marks instead of clearing in the DFS, accept
   unsorted column patterns (topological DFS order suffices for substitution).

## Targets

Single-thread factor: parity to slightly ahead. Refactor sweep: clearly ahead
(package 3). Factor with block parallelism (package 4): clearly ahead. Gate
every step on `klu_circuit` (times + residuals) and the bit-determinism tests.
