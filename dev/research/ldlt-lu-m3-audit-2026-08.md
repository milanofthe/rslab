# LDLT / LU kernel audit on Apple M3 (2026-08)

Status: MEASURED, no code changes warranted in the kernels. The per-call
kernel knobs shipped from the x86 tuning sprint are verified at their
optimum on Apple Silicon; the remaining gap to ideal scaling is structural
(top-of-tree concurrency), not parameter placement. Quantified below so the
next sprint starts from evidence, not intuition.

Reference: 40^3 helmholtz (`bench_suite`, sym/unsym families, size 64000,
c64, 8 threads, M3). Sweeps used the new `RLA_BENCH_NB` / `RLA_BENCH_GATES`
overrides in `bench_suite`.

## Baselines

| path | config | ana | factor | fill |
|---|---|--:|--:|--:|
| LDLT LL | fixed default (AMD) | 52 ms | 1287 ms | 20.6 M |
| LDLT auto | heuristic pick (new ND) | 626 ms | 633 ms | 11.2 M |
| LU LL | fixed default | 51 ms | 2403 ms | 41.2 M |
| LU auto | heuristic pick | 616 ms | 1145 ms | 22.5 M |

The clean-room ordering rewrite carries both paths: fill roughly halves
against the fixed AMD config, and the pick's factor time follows.

## Knob sweeps: flat on M3

- `panel_nb` in {32, 48, 64, 96}: 1287-1310 ms, within noise. Default 64 ok.
- Gates `(scalar, par_gemm, par_cdiv)` from (4k, 100k, 500k) to
  (16k, 1M, 8M): 1275-1305 ms, within noise. Shipped defaults ok.

No retuning of the fixed-config knobs is warranted for Apple Silicon.

## Structural ceiling

- Thread scaling LL (both paths): 3.7x at 8 threads (1.74x at 2, 2.9x at 4);
  Amdahl serial share about 17%.
- Concurrency histograms: mean active nodes 2.27-2.28; 57-59% of wall at
  1-2 active nodes, 0-1% at 5+. The tree narrows at the top and the few big
  chain nodes dominate; the in-node parallel kernels (tiled cmod, parallel
  BK panel trailing, parallel deferred GEMM) already exist and are gated
  correctly (fork-gate pathologies documented in the cmod comments).
- Small-node cmod runs at 3-6 Gflop/s (vs 27-38 on big nodes): inherent
  extend-add gather/scatter overhead at small update sizes; buffers are
  already hoisted/reused, no allocation waste found.

## Ranked future levers (structural, multi-day)

1. DAG pipelining / lookahead across parent-chain panels: overlap a parent's
   early panels with the tail of its children instead of level-style
   fork-join. Directly attacks the 57-59% of wall spent at 1-2 active nodes;
   plausibly 1.4-1.8x wall on the references. Must preserve the bit-identity
   guarantee (panel results are order-independent per column; scheduling must
   not change per-column operation order).
2. Batch small descendant updates that target the same column slab (one
   gather map + one GEMM for several updaters) to lift the small-node
   Gflop/s floor.
3. 32-bit index streams in the LL gather/scatter maps (`gloc`, row sets):
   the KLU sprint measured real wins from halving index traffic.
4. Parallel BK panel inside the multifrontal `factor_front` (the LL panel
   already has `apply_bk_panel_trailing`; MF does not). MF is not the
   shipped pick, so this only pays where the tuner selects MF.

## Note on `ana` in the pick

`auto`'s analyze is ~620 ms on the references (exact ND bakeoff plus the
stronger `sep_refine` hill climbs). Phased use amortizes it; for one-shot
factors it is now roughly half of end-to-end. The refinement patience knobs
saturate (grid_fill fill identical from patience 4096 up), so cutting ana
means cutting bakeoff exactness or refinement rounds, both of which buy the
fill that makes the factor fast. Leave as is.

## Follow-up (2026-08-01): cdiv panel lookahead

Issue #20's spine pipeline executor (inter-node update pipelining) was
already built and measured SLOWER in July (638.7 vs 616.2 ms; discarded,
see the issue's close note). The remaining lever it identified, the cdiv
sequence itself, was attacked here with a cheaper sub-case the executor did
not try: within-node depth-1 panel lookahead in the LL cdiv, factoring
panel p+1 (getf2 + deep replay, columns [ke, ke2)) concurrently with the
wide part of panel p's deferred Schur GEMM (columns [ke2, ncol)). The
narrow/wide GEMM split boundary and the gate (mt*wide*pw >= par_cdiv) are
pure functions of the node shape, so bits stay thread-invariant
(`ll_thread_determinism` green).

Measured (3-run means, 40^3 helmholtz, 8 threads): ll 1295 -> 1267 ms
(-2.2%), auto 625 -> 619 ms (-1%); single-thread neutral (4689 -> 4728 ms,
single runs, noise). Gate variations (par_cdiv 500k-8M) flat. Kept: small
but consistent, zero-risk, and it confirms the #20 conclusion - the chain
critical path is the pivoted cdiv sequence, and only cross-node cdiv
pipelining through pivot value dependencies could compress it further
(fundamentally harder, unclear headroom, not warranted at the current
PARDISO gap).
