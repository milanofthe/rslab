# Supernodal triangular solve — design plan (audit follow-up P4)

Status: **planned** (not implemented). This is the scoped design for the one
audit finding deliberately deferred from the 2026-07 audit branch: the
single-RHS triangular solves are element-wise sequential CSC/CSR sweeps
(`solve_ldlt`, `solve_lu`), while PARDISO solves supernodally — dense
triangular kernels over panels plus level-scheduled parallelism. For the
factor-once/solve-many FEM workload this is the largest remaining structural
gap after the factor-side audit fixes.

## Measured negative result (2026-07-11): scalar-DAG level scheduling is a dead end

Stage 1 as originally sketched — level-schedule the **scalar** column/row
elimination DAG of the flat factor (CSR copy of strict `L`, up-looking
forward, per-level rayon) — was implemented on `feat/parallel-solve`,
verified bit-identical to the flat solve, measured, and **rejected**:

```
3D 7-point grid, LdltSolver::factor defaults, best-of-3 over 50 solves
n= 21952  levels=2720 | build  27 ms | flat  4.34 ms | plan@1  6.46 | @4 6.58 | @8  6.77 ms
n= 64000  levels=6178 | build 221 ms | flat 24.6  ms | plan@1 34.9  | @4 47.0 | @8 64.7 ms
```

Two structural killers, not tuning artifacts:

1. **The filled factor's scalar DAG is a chain.** 2720 levels at n = 22k
   (~8 columns/level average): fill makes the root region's columns depend
   pairwise, so the levels that carry most of `nnz(L)` have width 1-2.
   Amdahl caps any speedup near 1 regardless of scheduling.
2. **The wide levels are the cheap ones.** Leaf levels are wide but their
   rows hold a handful of nonzeros — a rayon fan-out per level costs more
   than the work it distributes, which is why the parallel runs get *slower*
   with more threads.

Conclusion: parallel solve granularity must be the **supernode tree**
(depth ~tens, dense panel ops per node), exactly as PARDISO does. Any
future attempt at scalar-level scheduling should be rejected on sight —
the numbers above are the evidence.

The one keeper from the experiment: the dot-form sweep with per-element
`fmadd(L, -y_j, ·)` reproduces the flat scatter sweep bit-for-bit, so a
supernodal implementation can (and should) keep "bit-identical to the flat
solve" as a testable acceptance criterion.

## What the audit branch already did (baseline for the measurement)

* All solve sweeps are branchless (diagonal-first layout) and FMA-fused
  (`scalar::fmadd`, `target-cpu=native` repo builds).
* `solve_many` amortizes across RHS; the gap is the **single-RHS latency**
  and the lack of any parallelism inside one solve.

## Design

### 1. Keep the panels — a supernodal factor view

Both emit paths already produce per-supernode compact CSC fragments
(`CompactL` in the LDLᵀ path, `CompactNode` in the LU path) before
concatenating them into the flat global CSC. The plan is NOT to change
`LdltFactors`/`LuFactors` (public, load-bearing) but to add an **optional
sidecar** built at emit time:

```rust
pub struct SupernodalView {
    /// Per-supernode: first elimination position, ncol, and the sorted
    /// below-panel row positions (union over the panel's columns).
    panels: Vec<PanelMeta>,
    /// Dense panel values, column-major `nrow_p x ncol` per panel
    /// (trapezoid: unit diagonal implicit, D separate as today).
    values: Vec<T>,
    /// Level schedule over the assembly tree (children before parents):
    /// levels[l] lists panels whose inputs are complete after level l-1.
    levels: Vec<Vec<usize>>,
}
```

Memory cost: the dense trapezoid stores explicit zeros the sparse columns
drop — bounded by the same relaxed-amalgamation fill already accepted at
factor time (the panels ARE the factor layout the numeric phase produced).
Gate the sidecar behind `SolverSettings` (`with_supernodal_solve(bool)`,
default off until the bakeoff below says otherwise).

### 2. Forward solve (L): panel-parallel scatter

Per level (rayon scope), per panel: gather the RHS entries of the panel's
rows, run a small dense `trsv` on the `ncol x ncol` unit-lower block, then a
dense `gemv` for the below-panel rows. Writes of different panels in one
level target disjoint elimination positions **only for the trsv part**; the
gemv scatter can collide → accumulate per-panel into a thread-local sparse
update buffer and combine per level in panel order (deterministic, same
trick as `ORTHO_CHUNK` reductions). Determinism bar: fixed panel order per
level, fixed chunking — bit-identical across thread counts, matching the
factor's guarantee.

### 3. Backward solve (Lᵀ / U): gather form

Backward is already a gather (`acc -= L(i,j)·y[i]`) — panel version is a
dense `gemv`(transposed) per panel walking levels root-to-leaves. No write
collisions at all (each panel writes only its own columns), so this side is
embarrassingly level-parallel.

### 4. When it pays / bakeoff gate

Level-scheduled solves only pay when levels are wide and panels are fat;
banded/1D factors degenerate to a serial chain where the dense kernels only
add overhead. Reuse the existing machinery:

* a-priori: `max_tree_width`, mean `ncol` from `front_dims()` — cheap gate
  (e.g. width ≥ 8 and mean ncol ≥ 16, calibrate on the corpus);
* measured: extend `benches/solve_many` with a single-RHS latency column,
  RSLAB vs MKL PARDISO phase 33, across the h2h corpus;
* auto: let `tuned()` flip `with_supernodal_solve` from the structural
  features once the sweep data exists (same pattern as the ND bakeoff).

### 5. Acceptance criteria

1. bit-identical across thread counts (the house guarantee);
2. exact same solution as the flat sweep within the documented fmadd ulp
   (ideally: identical op order per element ⇒ bit-identical to flat);
3. single-RHS solve ≥ 2x faster than the flat sweep on the h2h FEM corpus
   median at 8 threads, and **never** > 5 % slower where the gate opens;
4. memory: sidecar ≤ 1.15x factor bytes on the corpus median (else keep
   default off).

## Why deferred

The change touches the factor storage contract, both emit paths, the Python
surface, and needs a corpus benchmark loop to calibrate the gate — a
self-contained branch with its own measurement cycle, not a tail commit on
an audit branch. Estimated effort: the sidecar + solves are ~2-3 days, the
calibration sweep another day on the bench corpus.
