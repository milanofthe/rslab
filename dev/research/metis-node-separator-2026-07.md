# rslab-metis: multilevel node-separator refinement (2026-07)

Status: **LANDED**. Fill on the PARDISO reference class dropped ~33 %
(exact scalar nnz(L) 20.9 M → 14.0 M on the 40³ grid; helmholtz probe
symbolic 25.1 M → 16.8 M) and the LL factor time went 1116 → 774 ms.
This documents the evidence chain, because the intuitive diagnosis
("our separators are too big") was wrong twice before the real lever
was found.

Reference pattern: 40³ 7-point grid (the `helmholtz_64000` pattern),
metric = **exact scalar nnz(L)** incl. diagonal via elimination-tree
column counts (no supernode padding, rslab's `symbolic_factor_nnz`
and MKL's `iparm(18)` both include padding and read ~0.3-2 M higher).
Harness: `cargo run --release -p rslab-metis --example grid_fill [m]`.
The column-count computation was validated against brute-force
symbolic elimination on 3³-6³ grids with random permutations.

## The evidence chain

| ordering | exact nnz(L) 40³ |
|---|---|
| rslab-metis before (edge-FM hierarchy + final König + greedy) | 20.88 M |
| AMD | 20.62 M |
| geometric oracle: perfect mid-plane ND, ND leaves | 21.29 M |
| geometric oracle + AMD leaves (30/120/200/1000/4000) | 21.3 / 21.1 / 21.1 / 20.5 / 20.5 M |
| **METIS 5.2.0 `METIS_NodeND`** (vendored via metis-sys) | **14.08 M** |
| **MKL PARDISO ND** (perm returned via `iparm(5)=2`, recounted) | **12.52 M** |
| **rslab-metis after (this work)** | **14.00 M** |

Knob sweeps on the old pipeline (imbalance, fm_passes, niparts,
amd_switch, coarsen_floor, seeds) all landed within ±3 % of 20.9 M,
no parameter headroom existed.

Findings that shaped the fix:

1. **Separator size is NOT the lever.** The old pipeline's top-level
   separator on 40³ was exactly 1600 = the perfect 40×40 plane, and
   per-level `sep/n^(2/3)` ratios were ≤ 1 throughout. Even *perfect*
   plane separators with any leaf treatment stay at ~21 M, worse than
   METIS's 14.1 M, whose top separator is a *wavy* y≈20 surface of
   1683 vertices (bigger than the plane!).
2. **The fill excess is distributed, not top-heavy.** Fill by
   elimination-position decile: MKL [0.5..1.0, 5.3] M vs geometric
   [1.0..3.7, 4.4] M, the top dense block is identical (1.281 M in
   both), the loss is 2-4× in every band below it, i.e. in how the
   mid-level separators couple to their ancestors.
3. What METIS does differently (`ometis.c`/`sfm.c`/`srefine.c`): the
   node separator is constructed ONCE at the coarsest level (best edge
   bisection → min vertex cover) and then **refined as a node
   separator at every uncoarsening step**, `FM_2WayNodeBalance` +
   `FM_2WayNodeRefine1Sided` (hill-climbing FM with negative-gain
   moves, breakout limit `min(3·nbnd, 300)`, best-prefix rollback,
   gain = `vwgt[v] − edegrees[v][other]`). Our old pipeline refined
   the *edge* bisection through the hierarchy and converted at the
   finest level, where the greedy positive-gain pass cannot reshape
   anything. The node-FM-refined (wavy, locally re-optimized at every
   scale) separators reduce exactly the inter-level coupling term the
   deciles exposed.

## The change

`crates/rslab-metis/src/node_refine.rs`: faithful port of METIS
`FM_2WayNodeRefine1Sided` + `FM_2WayNodeBalance` (indexed max-heap
with position tracking = `gk_rpq`; per-pass random boundary order from
the seeded SplitMix, deterministic). `node_nd.rs`: pipeline reordered
to coarsest-level König + node-FM, then project-balance-refine per
level (METIS `Refine2WayNode`); the finest-level conversion and the
greedy `refine_separator` are gone.

## Results

40³ grid exact fill 14.00 M (seeds 2-4: 14.3/14.5/14.9 M); flops
2.29e10 → 1.53e10. Scaling: 24³ 1.88 → 1.53 M (−19 % vs AMD);
56³ 108.2 (AMD) → 63.8 M (−41 %).

helmholtz probe (warm best-of-3, @8 threads, target-cpu=native):

| config | before | after |
|---|---|---|
| symbolic MetisND fill (incl. padding) | 25.06 M | 16.80 M |
| geom-flops / crit-path | 4.41e10 / 1.32e10 | 2.79e10 / 9.89e9 |
| `cfg MetisND` factor | 1116 ms | **774 ms** |
| `cfg default` (AUTO→AMF) factor | 1202 ms | 1202 ms (unchanged) |

Cumulative factor trajectory on this case: 1552 ms (pre-audit)
→ 1216 (kernel fixes) → 774 ms (this work) = **2.0×**; MKL PARDISO is
~170-220 ms, so the head-to-head gap is now ~3.5-4.5× (was 8.8×).

## Follow-ups

1. **AUTO still routes to AMF** (issues #67/#73 predate this work and
   their corpus A/Bs measured the *old* MetisND). Rerun the ordering
   corpus bakeoff; expect MetisND to win at least the 3D/FEM classes
   now. Until then only explicit `with_ordering(MetisND)` and the
   `tuned()` ND bakeoff benefit.
2. Remaining 12 % vs MKL (14.0 vs 12.5 M): candidates from the METIS
   source we did not port, `MlevelNodeBisectionL2` (pre-coarsen 4
   levels, 5 independent runs, keep best separator), `nseps` multi-try
   at every bisection, 2-sided refinement at the coarsest level,
   CoarsenTo = clamp(n/8, 40, 100) vs our floor 120.
3. `MMDSWITCH`-analog: our AMD-leaf switch at 200 measured neutral
   (60: 13.86 M, 400: 14.45 M), leave at 200.
4. MKL-perm probe + METIS reference live in the session scratchpad
   (`mkl_nd_probe`); the durable harness is the `grid_fill` example.

## 2026-07-31: clean-room replacement (`sep_refine.rs`)

The refinement module was replaced by an independent implementation
written from the published sources cited in its module docs, as part
of the licensing cleanup (the previous `node_refine.rs` had been
ported from METIS 5.2.0 `sfm.c`/`srefine.c`, i.e. Apache-2.0-derived;
see LICENSE-THIRD-PARTY). Design differences: lazy `BinaryHeap` with
per-vertex version stamps (the house `fm_refine` staleness pattern)
instead of an indexed heap; a flat label-transition journal with
best-prefix unwind instead of surgical state restoration; overweight
candidates are skipped, not pass-terminating; termination knobs are
effectively unbounded (patience 2^20, overshoot 4x) after measuring
that small budgets cost quality.

Knob study on the exact-fill harness (`grid_fill 40`, metis default
row, scalar nnz(L)):

| patience / overshoot / tie      | fill    |
|---|---|
| 256 / 0.25 / random rank        | 16.60 M |
| 4096 / unbounded / random rank  | 14.75 M |
| 4096+ / unbounded / newest-first| 11.06 M |

The newest-first (LIFO) gain tie-break is the surprise lever: the
pass drills along the moving separator front instead of scattering,
and beats every ordering we have measured on this pattern (old port
14.00 M, METIS 5.2.0 NodeND 14.08 M, MKL PARDISO ND 12.52 M). Checked
m=30 (3.29 M) and m=50 (36.7 M vs AMD 61.6 M); seeds land at
11.1-13.9 M. Ordering wall time roughly doubles (466 -> 560 ms at
40^3), paid back many times over in factor flops (1.90e10 -> 0.95e10).


## Update (2026-08-27): seed ensemble

Current state of the same 40^3 harness: metis default reaches 11.24 M
exact nnz(L) - past MKL PARDISO's 12.52 M. The remaining structural
finding is one-sided seed sensitivity: seeds {1, 3, 4} land at
11.24-11.30 M, seed 2 at 13.16 M (+17%), and on other matrices the
default seed can be the outlier. `MetisND` therefore now runs a
deterministic best-of-{1,2,3} ensemble (parallel candidates, exact
scalar-nnz(L) scoring via permuted-etree column counts, minimum fill
with lowest seed on ties). Measured: curl_curl 20^3 fill 18.13 -> 17.52 M
(-3.4%), factor -5-8%; saddle 24^3 fill -1.0%; helmholtz unchanged
(seed 1 already optimal there); analyze wall +5-8% (candidates run in
parallel; CPU cost 3x, amortized by the phased workflow).
