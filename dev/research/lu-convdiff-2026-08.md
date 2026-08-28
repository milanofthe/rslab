# Why the LU path trails Accelerate on convection-diffusion (2026-08-28)

Trigger: in the v0.28 Accelerate head-to-head the LU path is the weakest of the
three, at 2.31x Accelerate's factor wall time in geomean, and the convection-
diffusion 2D class carries it at 3.9x. This note locates the cost.

Reference matrices: the head-to-head grid's own convdiff arms
(`convdiff2d_{9216,13924,21025,31684}`, complex, M3, 8 threads unless stated).

## It is not the ordering

Full bakeoff on convdiff2d_13924 (fill = nnz(L)+nnz(U), factor at the shipped
thread policy):

| ordering | fill | factor |
|---|--:|--:|
| AMD | 602 612 | 51.4 ms |
| AMF | 515 456 | 46.9 ms |
| RCM | 2 232 206 | 190.6 ms |
| MetisND | 617 402 | 47.6 ms |
| AutoRace (shipped) | 515 456 | 47.1 ms |

The race already picks the best candidate, and in 2D nested dissection is worse
than AMF, not better. Factor time tracks fill across the whole table, so there is
no ordering left on the table.

Scoring the race by predicted *parallel* time instead of fill does not help
either. MetisND does buy a wider tree (mean level width 113 vs 56 on
convdiff2d_31684, thread scaling 3.34x vs 2.82x), but the extra fill eats it: at
8 threads ND wins only on the largest matrix and only by 4% (94.5 vs 98.7 ms).

## It is the concurrency

`sample` over a 10 s factor loop (convdiff2d_21025, 8 threads), sorted by top of
stack:

| symbol | samples |
|---|--:|
| `__psynch_cvwait` (idle workers) | 36 279 |
| gemm microkernel | 15 330 |
| left-looking driver closures | 8 206 |
| madvise / bzero / munmap | ~2 300 |
| gemm pack_lhs / pack_rhs | ~850 |

59% of the thread time is workers waiting. Thread scaling confirms it: 1 to 8
threads buys 2.4-2.8x, i.e. 33% efficiency.

| matrix | 1 thread | 2 | 4 | 8 |
|---|--:|--:|--:|--:|
| convdiff2d_9216 | 56.0 | 36.5 | 27.5 | 22.7 |
| convdiff2d_13924 | 101.9 | 65.6 | 46.5 | 41.3 |
| convdiff2d_21025 | 162.7 | 102.6 | 69.3 | 61.8 |
| convdiff2d_31684 | 279.1 | 162.7 | 109.9 | 95.8 |

Sizing the gap with convdiff2d_21025: Accelerate factors it in 19.5 ms, we take
59.5 ms at 8 threads and 162.7 ms at one. Our single-thread work is within an 8x
of Accelerate's parallel time, so with ideal scaling we would land at ~20 ms,
which is parity. The whole deficit on this class is scaling, not arithmetic - and
the gemm microkernel is only 15k of the 25k busy samples, so Apple's AMX
advantage is not what limits us here.

Note on the reported ratios: the shipped default is `Threads::Auto { max: 4 }`,
while Accelerate uses all cores. On this class the cap alone costs 12-17%
(the 4 vs 8 column above). The head-to-head numbers keep the shipped default,
which is the honest configuration to publish, but the cap is a policy choice,
not a capability limit.

## Rejected: picking multifrontal for this class

The multifrontal twin is faster on convdiff2d (6-16%), so a method pick looked
like cheap money. It is not, because it loses everywhere else and no structural
statistic separates the two cases. `mf_gain` is left-looking time over
multifrontal time, so above 1.0 means multifrontal wins:

| matrix | fronts | mean rows | flop-weighted rows | top-front flop share | mf_gain |
|---|--:|--:|--:|--:|--:|
| convdiff2d_21025 | 277 | 75.9 | 238 | 0.019 | **1.163** |
| convdiff2d_13924 | 196 | 71.0 | 232 | 0.029 | **1.161** |
| convdiff3d_4096 | 194 | 21.1 | 231 | 0.123 | **1.100** |
| convdiff2d_31684 | 390 | 81.2 | 241 | 0.012 | **1.086** |
| convdiff2d_9216 | 145 | 63.6 | 230 | 0.052 | **1.071** |
| convdiff3d_5832 | 178 | 32.8 | 226 | 0.135 | 0.937 |
| helmholtz_4096 | 194 | 21.1 | 231 | 0.123 | 0.912 |
| saddle_48387 | 3263 | 14.8 | 242 | 0.010 | 0.912 |
| mom_48035 | 570 | 84.3 | 781 | 0.400 | 0.891 |
| curlcurl_20577 | 464 | 44.3 | 906 | 0.488 | 0.831 |
| helmholtz_5832 | 178 | 32.8 | 226 | 0.135 | 0.804 |
| curlcurl_31944 | 736 | 43.4 | 1222 | 0.482 | 0.798 |

The decisive rows are `helmholtz_4096` and `convdiff3d_4096`: identical front
count, identical mean and flop-weighted front height, identical top-front share,
opposite outcome. The two differ only in which twin factors them (LDLT vs LU),
so no front-shape predicate can route them. Weighted by wall time the LU-path
gain is negative anyway, because the one class where multifrontal loses badly
(MoM, -11% on a 1.5 s factor) outweighs the convdiff2d wins.

Left-looking stays the default for both twins. Discarded, no code change.

## Rejected: scheduling on the exact updater DAG

`ll_subtree` makes a node wait for its whole child subtrees, while a
left-looking node only consumes `sched.updaters(s)`. Dropping that false
dependency looked like free concurrency, so the executor was rebuilt as a
dataflow scheduler: pending counter per node, consumers as the transpose of the
updater lists, `rayon::scope` spawning a node when its last updater retires,
same refcount protocol for the panel lifetimes.

It is bit-identical (verified by hashing the solution bits of nine matrices
across both executors: all equal, which is what the fixed per-node update order
buys), and mostly slower. Factor wall time, 8 threads, forest recursion ->
dataflow:

| matrix | recursion | DAG | |
|---|--:|--:|---|
| convdiff2d_9216 | 22.7 | 31.1 | -37% |
| convdiff2d_13924 | 42.2 | 61.7 | -46% |
| convdiff2d_21025 | 62.0 | 88.8 | -43% |
| convdiff2d_31684 | 137.3 | 129.6 | +6% |
| convdiff3d_5832 | 94.6 | 67.9 | **+28%** |
| helmholtz_9261 | 91.5 | 71.5 | **+22%** |
| mom_48035 | 1477 | 1634 | -11% |
| curlcurl_20577 | 709 | 1216 | -71% |
| saddle_48387 | 175 | 277 | -59% |

The false dependencies are therefore not what costs the time. Two effects
dominate instead. The recursion keeps a subtree on one worker, so its panels stay
hot; the dataflow scheduler spreads the same nodes across workers and pays the
misses. And where the flops concentrate in a few large fronts (curl-curl carries
48-53% of them in a single front), the recursion's idle workers are exactly what
the *in-node* parallel kernels steal to help, while the dataflow scheduler keeps
them busy on unrelated nodes and starves the critical front. Saddle adds a third:
3263 tiny fronts make the per-node spawn and the atomic edge decrements visible.
Discarded, no code kept.

## Rejected: opening the in-node fork gates

If the idle workers cannot be filled with other nodes, the other route is to let
them into the node that is running. Scaling the shipped gates
(`scalar_gate` 4096, `par_gemm` 1e6, `par_cdiv` 8e6) down by 4x, 16x and 64x on
the convdiff2d matrices: 24.7 -> 22.4, 45.8 -> 41.6, 67.1 -> 60.7, 99.0 -> 97.8
ms at the best setting per matrix. Consistent in direction, 5-10%, no setting
wins everywhere, and the largest matrix is flat. Not worth a default change on
this evidence, and it does not touch the structural gap.

## What is left

The 59% idle share is not a scheduling artifact: the tree at these sizes does not
contain more independent work, and the workers that would absorb it are needed
inside the nodes, where the medium-node kernels run at 2-4 GF/s (lever 2 of
`ldlt-lu-m3-audit-2026-08.md`). The remaining lever is therefore the same one the
saddle note scopes: a fused small/medium-node kernel that does assembly, update
and elimination in one pass instead of dispatching per updater. That raises the
floor the idle time is measured against, rather than redistributing it.
