# KLU pipelined refactorization (2026-08-27)

NICSLU-style within-block parallel refactor replay (Chen/Wang/Yang, TCAD
2013, pipeline mode): worker w owns columns bs+w, bs+w+nw, ... of a block
and spin-waits just-in-time on each U-dependency's ready flag before
consuming that L column. Per-column arithmetic and writes are untouched,
so the replay is bit-identical to sequential (asserted by
klu_pipelined_refactor_is_bit_identical_to_sequential on full L/U/diag
value equality). Dedicated OS threads (std::thread::scope), NOT rayon
tasks: a spinning rayon task could starve the owner of the column it
waits for when spawned tasks exceed pool workers.

Admission is computed once at factor time from the frozen pattern
(compute_pipelined_blocks): bn >= 512, mean elimination-DAG level width
>= 2, and replay work >= 50 M fmadds. The work gate is the decisive one.

Measured (M3, 8 threads, 10-refactor mean, release):

| matrix | seq | pipelined | speedup |
|---|--:|--:|--:|
| Sandia/ASIC_100ks | 0.400 s | 0.150 s | 2.7x |
| ATandT/onetone1 | 2.84 s | 0.97 s | 2.9x |
| ATandT/twotone | 0.79 s | 0.36 s | 2.2x |
| Rajat/rajat15 | 0.054 s | 0.028 s | 2.0x |
| synthetic 200k 1-SCC | 0.107 s | 0.074 s | 1.4x |
| Hamm/scircuit (~30 M, gated off) | 0.033 s | 0.033 s | 1.0x |
| Bomhof/circuit_4 (gated off) | 0.006 s | 0.006 s | 1.0x |

Tried first and rejected: pure level scheduling (cluster mode) with
per-level rayon joins. Engaged fine (163k of 170k scircuit columns sit in
wide levels) but the refactor WORK sits in the dense tail chain, and the
per-level barriers plus a zero-filling scratch pool cost more than the
wide-but-light levels return: scircuit regressed up to +49% before the
zero-invariant pool fix and stayed +8% after. The level computation
survives as the admission gate; the executor is the pipeline.
