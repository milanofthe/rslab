# Release history vs Apple Accelerate (2026-08-27)

Question: how did the head-to-head against Apple's vendor solver move over the
project's releases, and which release bought which part of it?

## Method

`benches/run_accel_history.sh`. One `cargo bench --no-run` per release tag into a
worktree, binaries stashed, then ROUNDS round-robin passes over *all* versions on
the same size grid (9 log-spaced sizes 4k-110k for the generator families, 8
sizes 4k-200k for the circuit family), minimum per cell over the rounds. Rotating
the versions inside each round instead of finishing one version at a time means
thermal drift hits every version equally.

The comparison is valid because the harness is frozen: `src/matgen` and
`build_family` are byte-identical from v0.19.1 to HEAD (only import moves and
dead-code deletions), the `auto` path calls the same public `tuned()` entry
point, and the JSONL emit format is unchanged. Verified empirically too - `nnz`
per matrix name is identical across all ten measured versions, residuals stay
below 6e-13 everywhere. Accelerate is an external library, so it is measured once
per round by the newest binary (its variant pick and the AMD-vs-Metis bakeoff are
part of the harness, not of rslab).

Data: `benches/bench_out/accel_history.jsonl` (committed - it needs the old
binaries and cannot be regenerated from HEAD alone). Figures and tables:
`benches/accel_story.py`.

## Result

Geomean speedup over Accelerate (factor only / one-shot analyze+factor+solve):

| version | LDLT path | LU path | KLU path | mean |
|---|:-:|:-:|:-:|:-:|
| v0.20.0 | 0.94x / 1.44x | 0.35x / 0.92x | - | 0.57x / 1.15x |
| v0.21.0 | 0.95x / 1.45x | 0.35x / 0.92x | - | 0.57x / 1.16x |
| v0.22.0 | 0.89x / 1.38x | 0.34x / 0.90x | - | 0.55x / 1.12x |
| v0.23.0 | 0.86x / 1.34x | 0.34x / 0.91x | 0.39x / 1.48x | 0.49x / 1.21x |
| v0.24.0 | 0.93x / 1.21x | 0.36x / 0.86x | 0.76x / 2.61x | 0.63x / 1.39x |
| v0.25.0 | 0.93x / 1.20x | 0.36x / 0.86x | 0.77x / 2.66x | 0.64x / 1.40x |
| v0.26.0 | 0.94x / 1.23x | 0.36x / 0.87x | 2.56x / 6.02x | 0.96x / 1.86x |
| v0.26.4 | 0.93x / 1.21x | 0.36x / 0.86x | 2.53x / 5.96x | 0.95x / 1.84x |
| v0.27.0 | 0.92x / 1.21x | 0.36x / 0.86x | 2.60x / 5.89x | 0.95x / 1.83x |
| v0.28.0 | 0.94x / 1.19x | 0.36x / 0.86x | 2.41x / 5.64x | 0.94x / 1.79x |

* The **KLU path** is the whole visible arc: 0.39x (v0.23, sequential reference)
  to 0.76x (v0.24 kernel work: 32-bit index streams, symmetric pruning, packed
  DFS marks) to 2.56x the moment v0.26 made the parallel per-block factor the
  shipped default. v0.27/v0.28 are flat here by construction - the Hopcroft-Karp
  BTF changes fill, not this grid's factor time, and the v0.28 pipelined refactor
  is a *refactor* win that a one-shot bench cannot see.
* The **LDLT/LU factor** numbers are essentially flat across ten releases
  (0.86-0.95x and 0.34-0.36x). Every kernel-level lever tried in that window was
  measured and shelved (see `ldlt-lu-m3-audit-2026-08.md`); this history is the
  independent confirmation that nothing regressed either.
* The **one-shot** LDLT curve *falls* from 1.44x to 1.19x. That is not a factor
  regression, it is analyze cost - see below.

## Finding: the ordering stage is not budgeted against the factor it saves

Per-matrix analyze/factor, v0.23 -> v0.24 (clean-room ND ordering) -> v0.28
(seed ensemble + exact two-stage race), ms:

| matrix | ana v0.23 | ana v0.24 | ana v0.28 | fac v0.23 | fac v0.24 | fill v0.23 | fill v0.28 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| helmholtz_110592 | 651 | 1157 | 1239 | 2692 | 1666 | 30.9M | 24.0M |
| convdiff3d_110592 | 664 | 1175 | 1232 | 4755 | 2996 | 61.8M | 48.0M |
| mom_72690 | 1530 | 7138 | 7601 | 10696 | 10389 | 128.26M | 127.87M |
| mom_48035 | 580 | 1452 | 1685 | 1653 | 1703 | 34.13M | 34.18M |
| saddle_73008 | 390 | 655 | 718 | 268 | 269 | 4.84M | 4.57M |

Where the ordering finds structure it pays for itself several times over
(Helmholtz: +506 ms analyze buys -1026 ms factor; convdiff3d: +511 buys -1759).
Where it does not, the cost is charged anyway: the near-dense MoM block spends
5.6 s of extra analyze for a 0.3% fill change and no factor change at all, and
saddle_73008 spends 265 ms of analyze on a 200 ms factor.

The existing gates are floors on *work* (`ND_RACE_MIN_FLOPS = 5e9`, `n > 10_000`)
- they ask whether the factor is big enough to be worth ordering for, not whether
the ordering stage is cheap enough relative to that factor. On the dense-ish
classes both are large, so the floor passes and the stage runs at 70% of the
factor time it cannot improve.

The holistic form of the fix is a *budget*, not a class switch: the prefix
already gives an estimated factor cost (`prefix_flops`), so the ND/race stage can
be admitted only while its own measured cost stays under a fixed fraction of that
estimate, and abandoned mid-stage when it exceeds it. That keeps one rule for all
matrices, keeps the exact-fill tie-break, and removes the pathological case.
Not implemented - recorded here as the next lever on the direct paths.

## Also worth noting

Accelerate's own analyze is not free either (6.0 s over the sym grid, 6.9 s over
unsym, vs RSLAB's 4.0 s and 10.6 s), which is why the one-shot column is the
friendlier of the two metrics for RSLAB on the LDLT path and the harsher one on
the saddle/MoM classes.
