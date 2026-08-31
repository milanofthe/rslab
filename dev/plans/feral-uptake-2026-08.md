# Uptake from feral (plan, 2026-08-31)

RSLAB forked feral on 2026-06-26 and last ported from it on 2026-07-19 (issues
92/93/94/119/144). Since then feral has shipped 0.15.0, 0.16.0, 0.17.0 and an
unreleased line, 132 commits. This plan takes what applies to us.

## Out of scope

Most of feral's recent volume is LP/interior-point basis machinery for its
hosts: Suhl-Suhl basis triangularization, the dense-bump route, reach-limited
hyper-sparse `ftran`/`btran`, threshold-Markowitz as the basis-LU default, the
Bartels-Golub update, refactorization scheduling. RSLAB has no basis LU with
update; none of it transfers. Also skipped: their `FERAL_*` env-knob parse
policy (we have no numeric runtime knobs to parse), their C ABI, and their
IPM-specific scaling routes.

## Principles for the uptake

1. **One contract per concept, across all three paths.** Anything that becomes
   caller-visible (cancellation, refinement quality) lands once and applies to
   LDLT, LU and KLU. The failure mode to avoid is feral's own shape here: a
   flag on one solver type, a second entry point on another.
2. **Configuration goes where configuration already lives.** `SolverSettings`
   for the supernodal twins, `KluSettings` for the circuit path, errors in
   `RslabError`. No new env knobs, no globals, no per-call parameter that
   duplicates a settings field.
3. **The house guarantees hold.** Bit-identity across thread counts, results
   independent of the worker count, analyze-phase choices a pure function of
   the pattern. Anything that changes numbers is measured interleaved and
   reported as a minimum over rounds (this machine drifts 2x across a session).
4. **Each package lands with the test that would have caught the bug** and, for
   performance packages, an A/B on the head-to-head grid.

## WP1 - Scaling router: route on the matrix, not on the index order

`pick_scaling_strategy` (`src/scaling/mod.rs:561`) gates on the stored column
count. `CscMatrix` holds one triangle, so that measures couplings to one side of
`j`: a pure relabeling changes the route. feral measured the flip on VESUVIO
(stored max degree 1026 one way, 11 the other) and fixed it by counting the
*symmetric* degree; the change is monotone, so a matrix can only gain
`Mc64Symmetric`, never lose it.

- **Integration.** Same function, same two gates, symmetric degree in the head
  gate. Keep feral's gate ordering so the common path stays allocation-free: the
  stored-degree pass decides everything that fails the slack-mass gate or already
  clears the threshold (sound, since symmetric degree is never smaller), and only
  the ambiguous remainder pays for an `n`-length accumulator.
- **Test.** A permutation-invariance property test: for a set of shapes
  (arrow-KKT, saddle, banded, grid), assert the route is unchanged under
  `P(i) = n-1-i`. This is a new class of test for us and belongs next to the
  determinism suite, not inside the scaling unit tests.
- **Risk.** Low. More matrices route to MC64; MC64 is already the more expensive
  scaling, so watch analyze time on the corpus.

## WP2 - Exact `Supernode.nrow` after amalgamation

`find_supernodes` (`src/symbolic/supernode.rs:436`) sets
`nrow = col_counts[first_col].max(ncol)`. That is exact for a fundamental
supernode and an underestimate after a size-based merge, because the merged
group's first column misses the rows only the parent contributes.

- **What is *not* affected in RSLAB.** The memory estimate takes its heights from
  `sched.rows(s).len()`, the true row set, so the a-priori bound is unaffected
  (feral's estimate did read the stale value; ours does not).
- **What is affected.** The Liu child-reorder heuristic
  (`multifrontal_ldlt.rs:1634`), and `front_dims()`, which is the public
  accessor the amalgamation study of `dev/research/amalgamation-2026-08.md` read
  its front statistics from. Those statistics need re-deriving after the fix, and
  the note needs a correction line.
- **Integration.** Track the union cardinality during amalgamation instead of
  re-deriving it from the first column; the merge already walks both row sets.
- **Test.** For a grid Laplacian at `nemin` 1, 16 and 32, assert
  `snode.nrow == true row-set size` for every supernode.

## WP3 - Cooperative cancellation

feral's design is right and we should take it as-is in mechanism: a caller-owned
`Arc<AtomicBool>` that the solver only ever reads, polled at supernode and dense
panel boundaries, `Interrupted` as a distinct outcome, no clock inside the
library. Their motivating measurement: a host with a 5 s budget returned after
48.8 s because one factorization ran 44 s uninterrupted.

- **Integration.** The flag is a field on `SolverSettings` and on `KluSettings`
  (`with_interrupt(Arc<AtomicBool>)`), not a new solver type or a parallel entry
  point: settings are already threaded to every driver. `RslabError::Interrupted`
  joins the error enum. Poll sites, one per path, all at existing boundaries:
  the supernodal twins in `ll_subtree`/`ll_dag` before `factor_node` and in the
  panel loop of `ll_cdiv_emit`; the KLU path per block in the factor driver and
  per column-range in the pipelined replay, where the workers already poll ready
  flags.
- **Contract.** On `Interrupted` the factor is invalid exactly as after any
  failed factor; a later `factor` with a cleared flag re-runs cleanly; no partial
  result is promised. Only the numeric phase is interruptible - the symbolic
  analysis is not, and the docs say so.
- **Cost.** Must be zero when unarmed: the poll is behind `Option::is_some`, no
  atomic is touched. Verify with a paired A/B on the reference matrices.
- **Test.** Arm the flag from a watchdog thread mid-factor on a matrix that takes
  ~200 ms and assert: `Interrupted` returned, no panic, no leaked pool, and a
  clean re-factor afterwards. Also assert the unarmed path is byte-identical.

## WP4 - One refinement contract

Today `solve_refined` exists four times (LDLT, LU free function, LU solver, KLU)
and each runs a fixed step count against an absolute max-norm residual, returns
only the solution, and allocates. There is no achieved-quality report and no
caller-defined stopping rule. feral has moved this to: a per-call step budget, a
caller-defined convergence predicate, a componentwise-certifying default, and
in-place entry points.

- **Integration.** One `RefinePolicy` (max steps, target, which backward error:
  normwise or componentwise) and one `RefineOutcome` (steps taken, achieved
  omega, whether the target was certified), defined once and used by all four
  entry points; the existing `solve_refined` signature keeps working through a
  default policy. `*_into` variants take the output buffer, so a sweep host can
  solve without allocating. The Krylov layer's true-residual stop test and this
  contract must agree on the definition of the backward error, so the formula
  lives in one place.
- **Numerics.** Componentwise: `omega = max_i |r_i| / (|A||x| + |b|)_i`, with the
  zero-denominator convention feral documents. This is the MA57/MUMPS default
  criterion; a normwise-only stop is what lets a solve report success on a
  componentwise-bad answer.
- **Test.** A matrix where normwise and componentwise disagree by orders of
  magnitude (badly scaled rows) must certify under one and not the other; the
  refinement must stop at the target rather than burning the full budget.
- **Note.** `MixedLdltSolver`/`MixedLuSolver` were removed in the consolidation;
  the certified ladder they carried should come back as this contract plus the
  existing `LowPrecisionPreconditioner`, not as new solver types.

## WP5 - Multi-RHS solve

feral measured their BLAS-3 multi-RHS path losing 1.4-1.8x when `nrhs` is not a
multiple of 8, because a row stride of exactly `nrhs` puts every panel row across
cache lines; padding the stride to whole 8-column tiles is bit-identical and
worth 1.36x geomean at `nrhs = 33`.

- **Where we differ.** We have no BLAS-3 multi-RHS path at all: `solve_ldlt_many`
  is an AXPY block solve over a row-major `n x nrhs` buffer, parallelized by
  splitting columns. So this is two packages, in order:
  1. **Stride padding** for the existing kernel. Cheap, bit-identical (padding
     columns carry a zero RHS and are never read back), measure first - the
     effect on a streaming AXPY kernel is smaller than on a panel GEMM.
  2. **A panel path above a threshold**, sharing one multi-RHS core between the
     LDLT, LU and KLU solves rather than three copies. This is the package that
     has to be designed against our existing structure: the fused equilibration
     scale, the parallel chunking, and the bit-identity-across-thread-counts
     guarantee all live in the current kernel and must survive.
- **Test.** A `to_bits()` equality test across every `nrhs % 8` residue against
  the current kernel, and the existing thread-invariance tests extended to the
  panel path.

## WP6 - Small, self-contained

- **MC64 Hungarian heap:** store the key inline in the heap entry (feral: 4-5%
  on large matchings, bit-identical).
- **Scaling cache fingerprint:** feral found their cache rejecting matrices
  against their own fingerprint. Check whether our scaling reuse has the same
  hole; we may not have the cache at all, in which case this is a no-op.
- **Thread-pool fallback:** a failed scoped pool should fall back to the
  sequential driver instead of propagating; check `in_scoped_pool`.

## Sequencing

Correctness first, then contracts, then performance:

1. WP1 (router) and WP2 (`nrow`) - small, correctness-relevant, independent.
2. WP3 (cancellation) - touches every driver, so it wants a quiet tree.
3. WP4 (refinement contract) - API surface, one release note.
4. WP5.1 (stride), measure, then decide WP5.2 (panel path) on the number.
5. WP6 alongside whatever is convenient.

Each package: its own branch and PR, the test that would have caught the bug,
and for WP3/WP5 an interleaved A/B on the head-to-head grid before and after.
WP2 additionally re-derives the front statistics in the amalgamation note.
