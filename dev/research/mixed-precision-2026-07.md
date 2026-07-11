# Mixed precision: c32 factor + certified IR (issue #18, stage 1-2)

Status: **stage 1-2 LANDED** on `feat/mixed-precision`. `MixedLdltSolver` /
`MixedLuSolver` (`src/numeric/mixed.rs`): factor `cast_f32(A)` with the
heuristic pick, solve via the explicit refinement ladder - plain IR
(high-precision residuals against the original A, low-precision
corrections, stagnation guard) escalating to GMRES-IR (the c32 factor as
preconditioner, warm-started) - and report an honest certificate
([`MixedInfo`]): normwise backward error vs. a target (default
`64·eps_f64`), iteration counts, `certified` flag. `solve_to` takes an
explicit target for preconditioner-grade use.

## Measured

helmholtz 40³ c64 reference (probe, warm best-of-3, calibrated 12 threads,
analysis excluded on both sides):

| | c64 | c32 (mixed) |
|---|---|---|
| numeric factor | 625 ms | **380 ms (1.64x)** |
| solve | 34 ms | 67 ms (= 3 solves + 3 matvecs, 2 IR steps) |
| factor bytes | 1x | **0.5x** |
| result | eps-level | be 1.8e-16, **certified**, no GMRES needed |

End-to-end factor+solve: 659 -> 447 ms (**1.47x**).

rapidmom precond corpus, spiral_D200 (production pivot policy):
c32 LU factor **270 ms vs 366 ms (1.36x)** - but `certified: false`
(be 1.7e-6 after the full 120-iteration GMRES budget): the MoM near-field
system's conditioning is beyond eps-level mixed recovery. That is the
certificate working as designed; for rapidmom's actual use (factor as a
GMRES **preconditioner** with outer tolerances 1e-4..1e-6) the c32 factor
is fit for purpose at 1.36x speed and half the memory - `solve_to(.., 1e-6)`
or the pre-existing `LowPrecisionLu` preconditioner adapter are the right
entry points there.

## Design notes

- The certificate is **backward** stability - the same class of guarantee a
  double-precision direct factorization gives. A first negative test
  expected `!certified` on a kappa~4e16 near-singular pair system and
  learned the ladder legitimately certifies it at be ~1e-31 (huge x makes
  the normwise backward error small; forward error is a conditioning
  property no direct solver certifies). The honest negative test cripples
  the factor itself (drop_tol 0.9) and exhausts the GMRES budget.
- Existing `LowPrecisionPreconditioner`/`LowPrecisionLu` (c64-only, no
  ladder/certificate) remain as the bring-your-own-Krylov building blocks;
  the mixed module is the self-contained driver.

## Open (issue #18 stages 3-4)

- Low-precision-STORAGE solves (factor in c64, store c32 for the solve
  phase - the memory-accessor trick; value analog of the 32-bit index
  compression) for factor-once/solve-many.
- Heuristic auto-switch (structural features -> mixed by default where the
  corpus says it certifies) + budget-planner wiring of
  `allow_mixed_precision` to this real path.
