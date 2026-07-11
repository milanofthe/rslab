# LL factor utilization: column-tiled cmod + join-steal guard (2026-07)

Status: **LANDED** on `feat/cmod-throughput`. Follow-up to the cmod
roadmap item in `factor-throughput-2026-07.md`; baseline is the heuristic
default (MetisND via bakeoff, calibrated 12 threads) at ~687 ms on the
helmholtz 40³ reference. Result after both changes, measured on an idle
machine (warm best-of-3, three independent processes): **553-575 ms**
(49-50 geom-Gflop/s), i.e. **−18 %**. Factor trajectory on this case:
1552 ms (pre-audit) → 1216 (kernel fixes) → 774/695 (ND + heuristic
defaults) → **~560 ms**; MKL PARDISO is 168-221 ms (gap ~2.6-3.3×).

## Diagnosis (new instrumentation, `RLA_PROFILE=1`)

Two additions to the LL profiler:

* `[RLA_LDLT_CONC]` - wall-clock histogram of the number of
  `ll_factor_node` calls in flight (mutex-timestamped enter/exit).
* `[RLA_LDLT_NODES]` - top-12 most expensive supernodes with per-phase
  wall (asm/cmod/cdiv), updater count, and cmod Gflop → achieved Gflop/s.

Findings at 12 workers:

1. **Mean node concurrency was 2.2-2.4**; 50-58 % of the wall runs with
   only 1-2 nodes active. The separator-chain head dominates; whole-tree
   work-stealing cannot help there - only intra-node parallelism can.
2. **The 1600×1600 root (top separator) alone cost 226 ms = 1/3 of the
   factor wall**: cmod 151 ms over 378 updaters + cdiv 72 ms, running
   solo.
3. **Join-steal stalls**: small nodes showed e.g. 74 ms of "cmod" at
   0.03 Gflop. A node that forks (per-update parallel GEMM, or the new
   tiled cmod) and then waits on its join lets rayon steal *other* work
   onto the waiting thread - often a whole sibling subtree - so the small
   node (and every dependent on its chain) stalls for tens of ms doing
   nothing. Pre-existing hazard (per-update parallel GEMMs), made visible
   by the per-node profile.

## Changes

1. **Column-tiled parallel cmod** (commit `5519abf`): partition the target
   panel into column slabs (`par_chunks_mut`, width a pure function of
   `ncol` - never of the thread count), and per slab apply *all* updaters
   in updater order with a serial `lower_tile_gemm`. One rayon fan-out per
   node instead of per update; the slab stays cache-hot across the
   updaters. Bit-identical: each panel entry lives in exactly one slab and
   receives its contributions in the same updater order with the same
   kernel. Root: cmod 151 → 78 ms (5.13 Gflop at ~73-75 Gflop/s), node
   wall 226 → 150 ms; helmholtz 687 → ~600 ms; mean concurrency 2.3 → 3.3,
   idle 26 % → 10 %.
2. **Join-steal guard** (commit `1689473`): a node forks inside cmod/cdiv
   only above ~1e8 flops of node-local work (`LL_CMOD_FORK_MIN_FLOPS`,
   same idea for `ll_cdiv_par` via `nrow·ncol²`); below it the node runs
   strictly serial and can never block on stolen foreign work. The
   0-Gflop stall entries disappear from the top list; the 2260×425 node
   went 71 → 38 ms.

Verification: full workspace suite green (incl. the bit-identical
across-thread-counts fixtures); the bit-identity argument is structural
(see above), not just empirical.

## Adaptive fork dispatch (feat/cmod-batching)

"Batching" small same-target updates by concatenation was **rejected at
design time**: merging updaters into one GEMM needs zero-padding to the
union row/column space - exactly the flop blow-up the cmod trimming
removed. Instead the fork *dispatch* became adaptive: an always-on relaxed
counter of in-flight `ll_factor_node` calls; a small node forks its
cmod/cdiv exactly when ≤ 2 nodes are active (chain phase: workers idle,
little foreign work for a blocked join to steal). The dispatch is
bit-neutral - both paths accumulate each panel entry in the same updater
order - so a racy counter read is benign and bit-identity across thread
counts holds. Measured (idle machine, 3 processes): 553-575 → **532-547
ms**. Remaining low-rate nodes (4-6 Gflop/s, gather-bound) run in the busy
phase where serial-under-tree-parallelism is the right call.

## Open (next candidates, evidence-ranked)

Post-tiling profile: root = cmod 71-78 ms (≈73 Gflop/s, fair for
tall-skinny slabs) + cdiv 70-73 ms; ~60 % of wall still at 1-2 active
nodes; mid-chain nodes with many small updates run serial at 4-6 Gflop/s
(gather-bound).

1. Batching small same-target updates (shared gather across updaters).
2. Root-chain cdiv: getf2 serial share within the 70 ms root cdiv.
3. Inter-node pipelining along the separator chain (complex; only if 1-2
   plateau).

Measurement hygiene note: one A/B window was discarded because a
concurrent zig build saturated the machine - always check
`Win32_Processor.LoadPercentage` before trusting a warm series on this
shared box.
