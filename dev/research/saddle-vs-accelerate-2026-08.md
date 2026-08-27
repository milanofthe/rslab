# Saddle/KKT class vs Accelerate: diagnosis (2026-08-27)

Trigger: the Accelerate head-to-head shows the saddle-point family as the
one LDLT class where rslab trails (0.65-0.83x; e.g. k=127 2D Stokes,
n=48387: rslab 156-181 ms vs Accelerate 123 ms).

## What it is NOT

* Not fill: the ordering race picks Amf at 3.12 M exact nnz(L);
  MetisND ties (3.14 M). Fill is competitive.
* Not pivoting trouble: n_perturbed = 0, clean inertia (32258 pos /
  16129 neg on k=127 - the pressure block factors as proper 2x2s).
* Not the analyze: 45 ms of the ~200 ms total.
* Not allocator churn: `sample` showed ~10% of wall in
  madvise/bzero/malloc, but recycling the dense panels through a
  FrontPool measured 4-6% SLOWER across saddle AND helmholtz - fresh
  mallocs get lazily zeroed pages from the OS while pooled buffers
  need a full explicit memset. Discarded (branch perf/ll-panel-pool,
  deleted).

## What it IS

The class is critical-chain/overhead dominated: ~0.2 GFlop of factor
work spread over many small supernodes (avg column height ~65) runs at
~1 GF/s effective. The 10 s `sample` profile of the factor loop
(k=156, 8 threads) sorts by top-of-stack as: __psynch_cvwait 14416
(idle workers - the chain does not feed 8 threads), gemm microkernel
11585, apply_bk_panel_trailing 1760, ll_factor_node 1459,
ll_bk_panel_step 421. Accelerate is not fast here either (123 ms for
0.2 GFlop) - it is ~1.5x less overhead per small node.

## Verdict

The remaining gap is the small-node kernel efficiency + chain
serialization - the same "medium-node floor" the M3 audit note tracks,
with two batching-flavored fixes and one allocator fix already
measured and rejected. Closing it means a fused small-node kernel
(assembly + D-apply + update in one pass, no per-updater dispatch) or
a fundamentally different schedule; both are multi-day projects, now
properly scoped by this evidence.
