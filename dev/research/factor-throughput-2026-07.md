# Factor throughput vs PARDISO — measurement log & roadmap (2026-07)

Reference case: `helmholtz([40,40,40], 0.02+0.01i)` (the h2h `helmholtz_64000`),
`SolverSettings::default().with_threads(8)`, LL path. Probe:
`cargo run --release --features matgen --bin factor_probe_helmholtz`.
All A/B numbers below are warm best-of-3 in one process — `bench_suite` is a
cold single shot with ±30 % run-to-run spread (system/thermal) and is only
trusted for *within-run* ratios.

## Gap decomposition (why 8.8×)

MKL PARDISO (same threads, same matrix): fac ~168-221 ms, fill 12.9 M.
rslab default (AMF ordering): fill 20.6 M actual / 24.7 M symbolic,
geom-flops 5.05e10, **critical path 3.20e10 (63 % of all flops!)**,
tree-width 723.

* **Ordering ≈ 3-4× of the gap.** Our own MetisND on this matrix: fill
  25.1 M, flops 4.41e10 — *worse fill than AMF* and nowhere near MKL's ND
  (12.9 M). The `tuned()` ND bakeoff correctly rejects it. rslab-metis
  separator quality on 3D grids is the single largest lever; it also fixes
  the critical path (ND: 1.32e10 → Amdahl ceiling 3.4× instead of 1.6×)
  and, for free, the remaining 1.18× solve gap.
* **Rate ≈ 2-3×.** The `gemm` crate is NOT the ceiling (measured c64 peak
  on this machine: 492-791 Gflop/s parallel, ~80 serial — MKL-class). The
  loss is in how the LL driver feeds it.

## Kernel fixes landed on this branch (warm A/B, default config @8)

| step | fac ms | note |
|---|---|---|
| baseline | 1552 | (same-process run; cold single shots showed 1488-2473) |
| + parallel deep-row BK panel apply | ~1500 | getf2 CPU 470→~200 ms; ports the LU twin's `apply_panel_trailing` to Bunch-Kaufman. Two subtleties cost a debug cycle each: the time-of-step multipliers must be **snapshotted** (later symmetric interchanges permute earlier multiplier columns' rows — unlike LU, where produced pivot rows never move), and the interchanges' deep segments are replayed per step (`swap_sym_lower_bounded` + `deep_swaps`). |
| + cmod GEMM trimmed to target rows (`mrows = nok - p0`, LU parity) | 1325 | the old code computed ALL updater off-diag rows and discarded rows above the target in the write-back |
| + cmod lower-triangle tiling (`lower_tile_gemm`, generalized strides) | **1216** | the write-back reads only `row >= col`; full rectangle wasted ~half for near-root targets |

MetisND config: 1379 → **1116 ms**. Cumulative: **−22 %** on the reference.

Phase split after the fixes (`RLA_LDLT_LL`, new instrumentation, CPU-ms
across threads): asm ~3 %, **cmod ~62 %**, cdiv ~35 % (getf2 share of cdiv
down from 44 % to ~17-30 %). `RLA_LDLT_CMOD_DIST`: 97.8 % of cmod flops in
parallel GEMMs — the dispatch is right; the remaining loss is likely (a)
node-execution overlap ~1.4× (measured: Σ node-wall / wall), (b) per-update
gather/writeback traffic around the GEMMs, (c) `gemm` parallel efficiency on
tall-skinny updates when the tree recursion holds workers.

## Config sweep (all measured, all worse than default)

nb32 +19 %, nb128 +18 %, par_gemm 2e5 +16 %, par_cdiv 2e6 +24 %,
Multifrontal +24 % (also with ND). The knobs are already tuned; no
parameter-level headroom remains on this class.

## Roadmap (evidence-ranked)

1. **rslab-metis separator quality on 3D meshes** — **DONE 2026-07**
   (see `metis-node-separator-2026-07.md`): multilevel node-separator
   refinement (METIS `FM_2WayNodeRefine1Sided` port) took the helmholtz
   reference to fill 16.8 M / crit-path 9.89e9 / **774 ms** MetisND @8
   (cumulative 1552 → 774 = 2.0×). Note: AUTO still routes to AMF
   (1202 ms) pending a corpus bakeoff rerun (#67/#73 measured the old
   MetisND).
2. **cmod residual** (~62 % of kernel time): measure node-overlap and GEMM
   wall separately; candidates: pipelining assembly with updates, batching
   small same-target updates, checking `gemm` Rayon behavior under the
   work-stealing tree.
3. `auto` tuner regression on this class (slower than default with equal
   fill, ana 2.4 s) — file as issue.
