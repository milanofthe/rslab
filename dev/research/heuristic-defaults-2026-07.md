# Heuristic default settings + install-time hardware diagnosis (2026-07)

Status: **LANDED**. `LdltSolver::factor()` / `LuSolver::tuned()` no longer
consult the ML performance model; the default settings pick is a
deterministic, hardware-agnostic heuristic, with an optional one-time
hardware calibration feeding the worker-count choice. The ML tuner stays
available as the explicit opt-in (`factor_auto` / `tuned_model`,
`TunerProfile`, `RSLAB_TUNER_PROFILE`, `cargo xtask tune`) for tuning to a
specific problem class on specific hardware.

## Why

The embedded-MLP default path had a measured regression on the PARDISO
reference class (bench_suite helmholtz 64k: auto **2565 ms** vs default
1488 ms, plus ~2.4 s analysis for the model + backstop machinery), and its
corpus can never see matrix provenance. Everything the model reliably
delivered (the curl-curl ND-class win) is decidable from **exact** a-priori
quantities - which is what the heuristic uses.

## The heuristic default (`tuned()`, behind `factor()`)

1. analysis with the adaptive ordering heuristic (`choose_adaptive`);
2. the proven default kernel configuration (left-looking, low-memory,
   measured panel/GEMM knobs - the helmholtz config sweep showed no
   parameter headroom off the default);
3. **exact ND bakeoff** on large systems (n ≥ 10k, ≥ 5e9 flops): re-analyze
   with `MetisND`, adopt only on ≥ 25% predicted-flops win with no
   regression in exact fill or transient peak (both paths: LDLᵀ and now LU);
4. worker count: `Threads::Auto{max:4}` (structural predictor) - or, when an
   **install diagnosis** cache exists, the calibrated cost model
   (`recommend_threads_cost_model`, critical-path-aware, uncapped to
   physical cores).

## Install diagnosis (feature `tuning`)

`tuning::install_diagnose()` (= `cargo xtask calibrate`) measures once and
caches per hardware fingerprint: serial proxy-GFLOP/s (f64 and complex
separately), the timing noise floor, and the parallel speedup curve. The
solvers only ever **read** the cache (`cached_calibration`, memoized) -
no calibration file, no behaviour change.

Calibration fixes found while validating (each measured on the 12-core
reference machine):

* **Speedup must be measured in the flop-dense regime.** The old 24³ f64
  grid measured 1.37×@12 (bandwidth-bound + too small), which made the cost
  model pick `Fixed(2)` for helmholtz → 1609 ms (2.1× worse than @8). Now:
  complex 32³ grid, **MetisND-ordered** (measure what the machine *can* do
  on a parallelizable tree; per-matrix structure is the cost model's job via
  the critical-path floor), warm-up factor before timing → 2.38×@12.
* **Mid-curve support point** `speedup4` (measured at 4 threads) so
  `speedup_for` doesn't misjudge the knee; cache format extended
  backward-compatibly.
* **Full-ladder scan** in `recommend_threads_cost_model` instead of
  break-at-first-non-improvement (a knee-shaped curve can be flat 2→4 and
  still win at 8).
* **Test isolation**: `probe_calibrate_plan_governor` measures while the
  test binary saturates all cores; it now writes to an `RLA_CALIB_CACHE`
  scratch dir instead of poisoning the machine's real cache.

## Measured result (helmholtz 40³ complex, warm best-of-3)

| path | pick | analysis | factor |
|---|---|---|---|
| old default (AMF, fixed cfg @8) | AMF | ~0.5 s | 1202 ms |
| old ML auto (bench_suite) | — | ~2.4 s | 2565 ms |
| **new `factor()` heuristic** | MetisND (bakeoff) + `Fixed(12)` (calibrated) | 0.37 s | **695 ms** |

Combined with the metis node-separator work this closes the tuner
regression (roadmap item 3 in `factor-throughput-2026-07.md`) and puts the
default entry point on the fastest known configuration for this class.

KLU has no model path: `KluSettings` defaults (BTF on, row scaling on) are
already the heuristic choice; its per-block AMD ordering is fixed by design.

## Family validation (bench_suite, cold single shot, ll = fixed default @8)

sym family (`auto` = the new heuristic default; ratio = auto/ll factor time):

| matrix | ratio | fill ll → auto |
|---|---|---|
| helmholtz_8000 | 1.03 | equal (below bakeoff threshold) |
| helmholtz_12167 | 1.15 | equal (cold-shot noise band ±30%) |
| helmholtz_19683 | 0.69 | 3.39 M → 2.60 M |
| curlcurl_27783 | **0.18** | 50.1 M → 17.7 M |
| curlcurl_41472 | **0.11** | 106.8 M → 32.2 M |
| curlcurl_65856 | **0.07** | 268.8 M → 63.5 M |
| saddle_89787 | 0.92 | equal |
| saddle_120000 | 0.50 | 15.3 M → 8.9 M |

Geomean 0.39 (~2.6× faster). unsym family (LU path, new bakeoff):
convdiff3d_27000 **0.69** (11.2 M → 8.1 M), convdiff2d_64009 0.89,
convdiff2d_44944 0.97, convdiff3d_15625 1.08 (below threshold, noise).
No fill regression anywhere in either family.

## Follow-ups

* The `bench_suite` "auto" rows now measure the heuristic default; a corpus
  rerun would confirm no family regresses vs the fixed default (the picks
  differ only via bakeoff adoptions, which are fill/mem-guarded, and thread
  counts, which are calibration-guarded).
* The calibrated thread pick is per-factorization; solver-in-the-loop users
  who run many factorizations concurrently should keep explicit
  `with_threads` (documented on `install_diagnose`).
