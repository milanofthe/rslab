# Consolidation audit (2026-08-27)

Status: MEASURED inventory of the whole tree, no code changed. Goal: cut
tracked Rust LOC by >= 50% through deletion of dead/experimental residue and
structural unification, without losing anything the three downstream
consumers (rapidmom, rapidfem, sane) or the python bindings actually use.

## Baseline

390 commits since 2026-06-26. Tracked: 544 files, 74,087 Rust LOC, of which

| area | files | LOC |
|---|--:|--:|
| src/ | 53 | 37,243 (approx 12,200 of that is inline #[cfg(test)] code) |
| crates/ (6 ordering crates) | 45 | 17,437 |
| benches/ | 27 | 7,215 (+2,362 LOC python plot scripts) |
| python/src | 1 | 1,856 (+1,129 py) |
| tests/ | 11 | 1,428 |
| xtask | 1 | 409 |

Non-code freight in git: 190 json, 61 mtx, 30 png, 26 pdf
(tests/data 4.4M, benches/bench_out 4.2M, docs/report 0.9M).

Downstream API surface actually used (verified by grep in the three repos):

- rapidmom: CscMatrix, GeneralCsc, LdltSolver, LuSolver, SolverSettings,
  ZeroPivotAction, BlrMode::contribution_blocks, gmres_block_fn(_mon), RslabError.
- rapidfem: CscMatrix, LdltSolver/LdltSymbolic (analyze_with), SolverSettings,
  FactorMethod (CLI maps "ll"/"mf"), OrderingMethod (CLI maps amd/amf/metis/scotch),
  tuning::HardwareInfo::probe.
- sane: KluSolver/KluSymbolic/KluSettings, LdltSolver, LuSolver, GeneralCsc,
  FactorMethod, CscMatrix.
- python bindings: Ldlt/Lu/Klu factor+solve+refactor, gmres, gmres_block,
  recycle (gmres_recycled), LdltSolver::tuned / LuSolver::tuned (heuristic,
  NOT the MLP path).

## Tier 1: dead or orphaned, delete outright (approx 15,000 LOC + data)

1. rslab-scotch (3,427) and rslab-kahip (4,114). Neither is ever returned by
   pick_default_method (a test even pins "never returns KahipND"); the
   session-08 bakeoff note records KaHIP ties METIS on fill at 4-6x cost.
   ScotchND is reachable only by explicit opt-in; no downstream sets it
   except a rapidfem CLI string mapping that can be dropped there in one
   line. Remove both crates, their OrderingMethod variants, their symbolic
   tests, Cargo wiring. History stays in git. -7,541
2. Schur-complement analysis path: symbolic_factorize_with_schur +
   merge_schur_tail_supernodes + split_straddling_supernode (approx 430 in
   symbolic/mod.rs), src/ordering/schur.rs (340), schur_constrained_postorder
   in postorder.rs, plus their tests (approx 250). Zero callers anywhere
   in-tree or downstream. -900
3. src/numeric/condition.rs (329): hager_higham_inverse_norm_1 /
   ConditionOperator exported at root, used by nothing in-tree, benches,
   python, or downstream. -330
4. Mixed-precision solvers: src/numeric/mixed.rs (524),
   src/bin/factor_probe_helmholtz.rs (194), benches/lu_warm_probe.rs (136).
   Only consumers are the probe bin and the bench; research thread is
   concluded in dev/research/mixed-precision-2026-07.md. -850
5. FactorMethod::RightLooking: driver block in factor_numeric plus enum,
   docs, one inline test; exercised only by benches being cut. Keep
   LeftLooking (default) and Multifrontal (tuner candidate). -80
6. One-off study benches (22 files, 3,719 LOC): deflation_study,
   recycle_study, mom_diag, mom_factor_probe, block_gmres_scaling, perf_axes,
   leftlook_mem, precond, mom_loop, fgmres_accounting, front_profile,
   phased_reuse, front_mem, block_gmres, catalog, solve_many,
   adaptive_restart, gemm_peak, precond_gmres, probe_complex, validate_axes,
   lu_warm_probe (already above). Their conclusions live in dev/research and
   the report. Keep: bench_suite, sweep (tuner corpus), klu_circuit,
   klu_realworld, ss_klu_ref shim loader. Also delete the matching
   [[bench]] entries and most plot scripts (approx 1,800 of 2,362 py). -3,700 rs
7. Tests that permanently SKIP on every machine: issue64_arrow_ordering (50)
   and issue67_thin_ordering (67) reference tests/data/large/ which does not
   exist (regen script dev/scripts/regen_r05_kkt.sh also gone);
   amf_corpus_oracle (276) needs external_benchmarks/ which is not in the
   repo. Either restore fixtures or delete; as-is they are green noise. -390
8. Orphaned data: tests/data/parity (245 tracked files, 4.4M, mumps/ssids
   sidecars) and tests/data/lu_trace are read by NOTHING. benches/bench_out
   (4.2M pngs/jsons/pdf) is build output committed to git; keep the current
   report figures in docs/report only. Delete + gitignore. -8.6M repo weight
9. Debug instrumentation compiled into production paths: LlConcProf /
   LuConcProf guards, ll_node_cost/lu_node_cost, FrontStat/take_front_stats,
   take_blr_cb_stats, and the env switches RLA_PROFILE, RLA_KLU_PROF,
   RLA_NO_FREE, RLA_GEMM_SERIAL, RLA_BLR_PROBE, RSLAB_FRONT_STATS,
   RSLAB_MC64_TRACE, RLA_TUNER_PARITY(_PATH). All were one-sprint measuring
   tools; the numbers are recorded in dev/research. Keeping them costs LOC,
   env-var surface, and atomics in hot paths. -500
10. src/bin/bench_sparse.rs (116, bench-era tool) and the stale
    "rslab-diagnostics 144 binaries" comment in Cargo.toml (crate no longer
    exists). -120
11. src/symbolic/profiler.rs (175) + tests/symbolic_profiler.rs (218) +
    SymbolicProfileReport root export: consumed by front_profile bench
    (cut above) and its own test. -390

Tier 1 total: approx 15,000 Rust LOC (20%), plus approx 1,800 py, plus 8.6M
binary/data freight, 9 env vars, 2 crates, 22 bench targets.

## Tier 2: structural consolidation (approx 12,000-14,000 LOC)

1. LDLT/LU twin unification. multifrontal_ldlt.rs (4,467) and
   multifrontal_lu.rs (3,926) are the same architecture specialized twice:
   identical profiling scaffolds, FrontPool sharing already crosses the
   files, LlStore/LuLlStore, LlEmitLdlt/LlEmit, PanelPtr/LdltPanelPtr,
   CompactL/CompactNode, byte-for-byte-parallel factor_subtree, twin
   left-looking drivers, twin solve_many/solve_refined plumbing, and
   duplicated high-level driver logic (tuned / nd_bakeoff / factor_auto /
   tuned_model exist in both LdltSolver and LuSolver). Factor the schedule,
   store, emit, pool, threading, and driver layers over a small FrontKernel
   trait (BK-LDLT vs threshold-partial-pivot LU as the two impls). Realistic:
   8,400 combined -> approx 5,000. -3,000 to -3,500
2. Inline test rationalization. 12,200 LOC of #[cfg(test)] in src, much of it
   historical issue-pinning with per-file matrix-builder scaffolding
   (postorder.rs is 608/611 LOC test, btf.rs 635/671, column_counts.rs
   317/343). Extract one shared test-matrix helper (or reuse matgen), keep
   invariant + oracle tests, drop superseded issue-regression pins whose
   guarded code no longer exists. Target: 12,200 -> approx 6,000. -6,000
3. iterative.rs (4,073). gmres single-RHS is a separate implementation next
   to gmres_block; reimplement as s=1 of the block path and keep gmres_fn/
   gmres_block_fn(_mon) as the thin adapters they already are (rapidmom needs
   them). Fold the _mon variants into the base signatures (Option<mon> is
   already the pattern). -400 to -600
4. Python bindings (1,856): the Ldlt/Lu/Klu pyclasses triplicate solve/
   solve_many/gmres/gmres_block/recycle bodies; a macro_rules over the three
   solver types collapses it. -600
5. matgen (1,885): after the bench cut, prune to the generators the kept
   benches/tests use (stencil, random, fem; likely drop bem/structured/
   download unless klu_realworld keeps download). -500 to -900
6. scaling/: InfNorm iterative Ruiz is opt-in since 2026-04 (OnePass is the
   default, Mc64 the matched arm). If no caller sets InfNorm, fold it away
   and keep one equilibration + MC64. Verify first. -200 to -400

## Tier 3: subsystem decisions (approx 4,500-6,500 LOC, needs owner call)

1. MLP auto-tuner: auto_tune.rs (817) + 2 embedded JSON models (195KB) +
   analysis.rs StructuralFeatures (595) + xtask calibration driver (409) +
   tuner tests (tuner_profile_env/apply, tuner_memory_backstop,
   auto_strategy, approx 350) + sweep.rs training corpus harness (1,246).
   The default path (factor, tuned) is the model-free heuristic; python and
   all three downstreams use only that. The MLP is opt-in (factor_auto/
   tuned_model/RSLAB_TUNER_PROFILE). Dropping the MLP but keeping tuned +
   pick_default_method keeps observable behavior for every known consumer.
   -3,000 to -3,400 (plus 195KB models)
2. Recycled/deflated GMRES: gmres_recycled + Recycle + RecycleScalar +
   combine_ritz_real/complex + dense_eig.rs (594). Exposed in python
   (.recycle) but unused by rapidmom/sane; research benches (recycle_study,
   deflation_study) concluded. If the python recycle API is not in active
   use, drop the subsystem. -1,600
3. FactorMethod::Multifrontal: LL is the shipped default and wins on the
   references (see ldlt-lu-m3-audit); MF survives as tuner candidate and
   cross-check. If the tuner (or its slimmed successor) stops proposing MF,
   the MF front kernels + CB-stack path in both files fold away. rapidfem's
   "mf" CLI mapping is a one-line downstream edit. Decide after the Tier-2
   unification, where MF becomes just a second schedule over the same
   kernel. -1,200 to -1,800 (post-unification accounting)

## Projected outcome

| step | LOC after |
|---|--:|
| baseline | 74,087 |
| Tier 1 | approx 59,000 |
| Tier 2 | approx 46,000 |
| Tier 3 | approx 34,000-36,500 |

Total reduction approx 51-54%, with the embedder-visible surface preserved
except: removed OrderingMethod::{ScotchND,KahipND}, removed condition/mixed/
profiler/schur exports (all verified unused), and the Tier-3 items behind
explicit decisions. Everything removed remains recoverable from git history;
the 12 stale feature branches on origin should be pruned in the same pass.

## ReSolve (ORNL) read-across: a GPU path for rslab

ReSolve (BSD-3, ECP-funded, C++) targets exactly rslab's KLU use case:
sequences of solves on a fixed sparsity pattern (ACOPF / power-grid sweeps,
i.e. the same shape as sane's circuit sweeps). Its architecture:

1. First factorization on CPU with SuiteSparse KLU (pivot order fixed there).
2. Numeric-only REfactorization + triangular solves on GPU
   (cuSolverRf/cuSolverGLU on CUDA, rocsolverRf on HIP): no pivoting on the
   device, just replay of the numeric factorization on the frozen pattern.
3. FGMRES iterative refinement wrapped around the direct solve to recover
   the accuracy lost to static pivoting/low precision.
4. Backend abstraction: workspace objects owning device buffers that persist
   across the sweep, matrix/vector handlers per backend (cpu/cuda/hip),
   MemoryUtils hiding alloc/copy.

Lessons for rslab:

- rslab already has the exact phase split ReSolve monetizes on GPU:
  KluSymbolic::analyze / factor / KluSolver::refactor, and LDLT/LU
  analyze / factor_numeric. A GPU path is therefore NOT a new solver: it is
  an alternative numeric backend for the refactor+solve phases on a frozen
  pattern with frozen pivot order. The first factorization stays on CPU,
  bit-deterministic, exactly as today.
- The consolidation above directly serves this: one unified supernodal
  kernel layer (Tier 2.1) gives a single seam where a device backend plugs
  in (panel factor stays scalar/CPU-like, the deferred Schur GEMMs and the
  triangular solves are the offload targets; for KLU, the per-block
  Gilbert-Peierls replay is the offload unit).
- ReSolve validates static-pivot + iterative refinement as the accuracy
  story; rslab already ships solve_refined and gmres. Wiring
  "GPU refactor + CPU-verified refinement fallback" mirrors
  LinSolverIterativeFGMRES over LinSolverDirectRf.
- Workspace ownership matters more than kernels: ReSolve's win is keeping
  factors + workspaces device-resident across the sweep. Any rslab GPU path
  should expose the same phased object (symbolic on CPU, numeric state
  resident on device, refactor(values) streaming only the new values).
- Backend choice on our hardware: Apple M3 means Metal (wgpu or
  objc2-metal) rather than CUDA/HIP; wgpu keeps the pure-Rust,
  no-native-deps story intact. ReSolve's cpu/cuda/hip layout maps to a
  backend trait with a cpu reference impl.
- ReSolve also ships HyKKT (GPU KKT solver), interesting later for the
  IPM/KKT matrices rslab's ldlt_compress path targets.

Suggested sequencing: consolidate first (Tiers 1-2), then prototype the GPU
refactor backend on the KLU path (smallest kernel, sequential semantics,
sane as the driving consumer), then consider the supernodal GEMM offload.

## Outcome (2026-08-27, branch refactor/consolidation)

Executed per owner decisions: MLP tuner removed, recycled GMRES kept,
Multifrontal kept as opt-in method. 11 commits, all tests green
(289 lib + 309 with matgen/tuning features + integration tests).

Measured: 74,087 -> 45,953 Rust LOC (-38%), 544 -> 189 tracked files,
repo freight -8.6M (parity data, bench_out, orphaned fixtures).

Landed: Tier 1 complete (benches purge, scotch+kahip, Schur path,
condition/mixed/bins, instrumentation + 17 debug env vars, RightLooking,
orphaned data/tests), Tier 3.1 complete (auto_tune + models + xtask
tune/validate/profile + sweep + trainer), Tier 2 partial (matgen catalog,
shared ll_common scaffolding: SlotStore/PanelPtr/tuned driver).

Consciously NOT done, with reasons:

1. gmres single-RHS as s=1 of gmres_block: the two implementations use
   different orthogonalization schedules; folding them changes solver
   numerics observable by every embedder. Needs convergence-parity
   validation on the MoM corpus first.
2. Inline-test mass reduction (-6k hoped): on inspection the big test
   blocks (iterative, scaling, klu) pin distinct behavioral properties
   (bit-identity across threads, corpus routing counterexamples,
   convergence classes) - deleting them trades real coverage for a
   number. Remaining legitimate lever: table-driven folding of the
   per-corpus pin tests, low priority.
3. Deep LDLT/LU kernel unification (-2.5k further): the remaining twin
   code is the numeric kernels themselves (BK panel vs threshold-pivot
   LU). A FrontKernel-trait redesign is a multi-day, benchmark-gated
   project; the shared scaffolding extracted here is the preparatory
   step and the natural seam for a future GPU backend as well.
4. python/src macro-fold across the three solver classes: needs pyo3's
   multiple-pymethods/inventory feature; most of the LOC is docstrings.
   Skipped as a poor elegance trade.

docs/report/rslab.tex still describes v0.27 (mixed precision, tuner,
scotch/kahip); needs a refresh pass alongside the existing
chore/readme-report-refresh branch.
