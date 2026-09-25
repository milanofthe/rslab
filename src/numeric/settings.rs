//! Solver settings shared by the LDL^T and LU paths: the numeric and
//! analysis knobs of [`SolverSettings`], the static-pivot policy, and the
//! worker-thread policy with the scoped pools the factorizations run in.

use crate::diagnostics::MemoryEstimate;
use crate::error::RslabError;
use crate::symbolic::{OrderingMethod, RelaxAmalgamation, SymbolicFactorization};

/// Action to take when a near-zero pivot is encountered during factorization.
///
/// This is the static-pivoting policy knob shared by the symmetric LDL^T and the
/// unsymmetric LU paths (via [`SolverSettings`] and the LU options).
#[derive(Debug, Clone)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot at face value (zero the column, count as a zero in
    /// the inertia signature, flag for iterative refinement). The perturbation
    /// magnitude is unbounded - use only when downstream code tolerates sign
    /// loss in the perturbed positions and re-checks inertia.
    ForceAccept,
    /// Return [`RslabError::NumericallyRankDeficient`].
    Fail,
    /// Replace the tiny pivot with `sign(d) * max(|d|, abs_floor)`, keeping the
    /// column live (LAPACK / MA57-style static pivoting). The factor satisfies
    /// `L*D*L^T = A + delta ` for the produced `L`, `D`; `delta ` is bounded in the worst
    /// case by `||A[:,k]||^2 / abs_floor`, so drive iterative refinement against
    /// the unperturbed `A` for tight tolerances. A typical recipe is
    /// `abs_floor = eps_rel * ||A||inf` with `eps_rel in [1e-12, 1e-8]`.
    PerturbToEps { abs_floor: f64 },
}

/// The factor path a [`SolverSettings`] is applied to; each reads a different
/// subset of the settings (see [`SolverSettings::ignored_on`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorPath {
    /// Symmetric `LDL^T` ([`LdltSolver`](crate::LdltSolver)).
    Ldlt,
    /// Unsymmetric LU ([`LuSolver`](crate::LuSolver)).
    Lu,
}

/// Options controlling the sparse LDL^T and LU factorizations. Defaults give an
/// **exact** complete factorization that fails on rank deficiency. Relaxing
/// them turns the factorization into a robust, memory-light **preconditioner**.
/// All knobs compose via the `with_*` builders.
#[derive(Debug, Clone)]
pub struct SolverSettings {
    /// Near-zero pivot policy. Reuses rslab's [`ZeroPivotAction`]: `Fail`
    /// (exact, default) returns [`RslabError::NumericallyRankDeficient`] on a
    /// singular pivot; `PerturbToEps { abs_floor }` is robust static pivoting -
    /// a pivot below `abs_floor` is lifted to that floor (the
    /// complex-symmetric analogue of rslab's f64 `perturb_to_floor`), so the
    /// factorization never fails and produces `L D L^T = A + E` for small `E`.
    /// That is exactly the never-fail behaviour a preconditioner needs.
    pub on_zero_pivot: ZeroPivotAction,
    /// Threshold dropping for incomplete factorization. When `Some(tau)`, fill
    /// entries of `L` with magnitude below `tau` (relative to the column) are
    /// discarded, trading factor accuracy for memory. `None` = complete
    /// factorization. (Wired in a later stage.)
    pub drop_tol: Option<f64>,
    /// Worker-thread policy for this factorization, run in a **scoped** rayon pool
    /// (not the global pool). Either a [`Fixed`](Threads::Fixed) count or
    /// [`Auto`](Threads::Auto) - the data-driven per-matrix predictor, capped at a
    /// user-defined maximum. **Default [`Auto`](Threads::Auto)** (predict, up to
    /// all cores). The numeric result is bit-identical regardless of this value.
    pub threads: Threads,

    /// Caller-owned cancellation flag for the numeric factorization. The solver
    /// only ever *reads* it, at supernode and dense-panel boundaries; on the
    /// first observation of `true` the factorization stops and returns
    /// [`RslabError::Interrupted`](crate::RslabError::Interrupted). Re-arming
    /// after an interrupt is the caller's `store(false)`. Taking a flag rather
    /// than a deadline keeps the library clock-agnostic and leaves
    /// wall-versus-CPU budget policy with the host. **Default `None`**, which
    /// costs one `Option` branch per boundary and touches no atomic.
    pub interrupt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,

    // ---- Analysis-phase knobs (read by `analyze_with`; ignored by `factor`) ----
    /// Fill-reducing ordering (the cuDSS `REORDERING_ALG` analogue). Analyze-time.
    /// Default [`OrderingMethod::Auto`] (the race on exact fill).
    pub ordering: OrderingMethod,
    /// Supernode amalgamation `nemin` (merge-candidate column threshold). Default
    /// `16`. Smaller = finer supernodes (less fill, more per-front overhead).
    /// Analyze-time.
    pub nemin: usize,
    /// Relaxed (fill-tolerant) amalgamation thresholds, the supernode throughput
    /// lever. `Some` (default `<=256` wide, `<=64` extra rows) trades a little
    /// explicit-zero fill for wider, higher-rank dense fronts. Analyze-time.
    pub relax: Option<RelaxAmalgamation>,
    /// A fill-reducing ordering to use instead of `ordering` (see
    /// [`with_permutation`](Self::with_permutation)). Analyze-time.
    pub permutation: Option<std::sync::Arc<[usize]>>,

    // ---- Kernel scheduling knobs (formerly process-wide atomics) ----
    /// Bunch-Kaufman / LU panel width (blocking factor). Default `64`. Changes the
    /// pivot search window (a different but equally valid factor), not the answer.
    /// Clamped to at least 8 on use.
    pub panel_nb: usize,
    /// Below this flop count a contribution update runs as a scalar triple loop
    /// instead of a SIMD GEMM. Default [`DEFAULT_SCALAR_GATE`](crate::DEFAULT_SCALAR_GATE).
    pub scalar_gate: usize,
    /// At/above this flop count a cmod-class GEMM runs rayon-parallel. Default
    /// [`DEFAULT_PAR_GEMM`](crate::DEFAULT_PAR_GEMM).
    pub par_gemm: usize,
    /// At/above this flop count the panel-trailing / Schur / LU-front GEMM runs
    /// rayon-parallel (the top-of-tree node-parallelism lever). Default
    /// [`DEFAULT_PAR_CDIV`](crate::DEFAULT_PAR_CDIV).
    pub par_cdiv: usize,
    /// Use the SIMD GEMM (vs the scalar triple loop) for the front Schur update.
    /// Default `true`. A kernel A/B knob for benchmarking.
    pub use_gemm_schur: bool,
    /// Threshold partial-pivoting tolerance `u in [0, 1]` for the LU path. The diagonal pivot is
    /// kept unless it falls below `u * |colmax|` in its fully-summed block. `u = 1`
    /// is full partial pivoting; `u -> 0` keeps the diagonal unless exactly zero
    /// (least fill, least stable). Default
    /// `DEFAULT_PIVOT_U = 0.1` (a `gemm_tuning` internal constant).
    /// Ignored by the LDL^T path (Bunch-Kaufman). Numeric-phase knob; a lower `u` trades a little
    /// stability (backed by the near-zero pivot policy) for less fill and speed on
    /// well-scaled / diagonally-dominant systems.
    pub pivot_u: f64,
    /// Symmetric equilibration strategy `A_hat = D A D` applied by [`LdltSolver`](crate::LdltSolver)
    /// before factoring. Default [`OnePassInfNorm`](crate::ScalingStrategy::OnePassInfNorm)
    /// (the historical one-pass inf-norm, bit-identical to before this knob).
    /// [`Identity`](crate::ScalingStrategy::Identity) disables scaling;
    /// [`InfNorm`](crate::ScalingStrategy::InfNorm) is the iterative Knight-Ruiz
    /// (Ruiz) equilibration; [`Auto`](crate::ScalingStrategy::Auto) routes to
    /// MC64 matching on the arrow-KKT signature else inf-norm. Scaling changes only
    /// values (not the pattern), so the a-priori memory estimate is unaffected.
    /// Consumed by the symmetric path; the unsymmetric LU path uses its own
    /// two-sided row/column equilibration.
    pub scaling: crate::scaling::ScalingStrategy,
    /// Maximum-product row matching (MC64) before the **LU** analysis, where
    /// the matrix needs it: rows are permuted so the matched,
    /// largest-product entries form the diagonal and both sides are scaled
    /// to make them unit magnitude, so the front-local pivot search rarely
    /// needs an off-diagonal pivot and the element growth of the
    /// block-restricted pivoting stays bounded (on the ibmpg1 power grid the
    /// residual improves from 4e-5 to roundoff). Applied only when a
    /// diagonal entry is missing, zero or negligible against its column;
    /// with a usable diagonal the permutation costs fill and pivot quality
    /// and is skipped. Default `true`; `false` never matches. Ignored by
    /// the symmetric and KLU paths.
    pub lu_matching: bool,
}

/// Worker-thread policy for a factorization. The numeric result is bit-identical
/// regardless of which is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threads {
    /// Exactly this many workers. `0` = all logical cores. Use a small fixed
    /// budget for **solver-in-the-loop** (many concurrent solves coexisting on
    /// the machine without oversubscription).
    Fixed(usize),
    /// Predict the worker count per-matrix from the structural fingerprint (the
    /// validated [`recommend_threads_from`](crate::recommend_threads_from)
    /// policy: thin / tiny systems stay low where they would only regress, big
    /// BLAS-3-rich systems use the cores), **capped at `max`** (`0` = all logical
    /// cores). The single-solve default: best throughput without oversubscribing
    /// the matrices that do not scale.
    Auto {
        /// Upper bound on the predicted worker count (`0` = all logical cores).
        max: usize,
    },
    /// Use the **current** rayon pool as-is, without building a scoped pool. The
    /// solver-in-the-loop path: build **one** bounded pool (e.g. 4 workers) with
    /// [`with_threads`](crate::with_threads) and run the factorization *and* every
    /// iterative solve inside it, so both phases share the same capped pool with no
    /// per-call thread spawn. The numeric factor is unchanged.
    Ambient,
}

impl Default for Threads {
    fn default() -> Self {
        // Cap at 4 workers by default: our strong-scaling data puts the efficiency
        // knee at ~4-6 threads, so 4 is the pareto-optimal throughput-per-core point
        // and the safe default for concurrent / embedded (solver-in-the-loop) use.
        // `Auto` still predicts a smaller count per matrix where more would regress.
        Threads::Auto { max: 4 }
    }
}

/// All logical cores (the `0` sentinel resolution).
fn all_cores() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

impl Threads {
    /// Resolve to a concrete worker count. `recommend(cap)` is the structural
    /// predictor (already clamped to `cap`); it is only invoked in [`Auto`] mode.
    ///
    /// [`Auto`]: Threads::Auto
    pub(crate) fn resolve(self, recommend: impl FnOnce(usize) -> usize) -> usize {
        match self {
            Threads::Fixed(0) => all_cores(),
            Threads::Fixed(n) => n,
            Threads::Auto { max } => recommend(if max == 0 { all_cores() } else { max }),
            Threads::Ambient => rayon::current_num_threads().max(1),
        }
    }

    /// Run `f` under this thread policy. [`Ambient`](Threads::Ambient) runs on the
    /// current rayon pool with **no new pool spawned** (solver-in-the-loop); every
    /// other policy resolves a worker count and runs `f` in a scoped pool of that
    /// width with a `stack_bytes` worker stack. Centralizes the dispatch so all
    /// factorization paths honour `Ambient` identically.
    pub(crate) fn run<R: Send>(
        self,
        stack_bytes: usize,
        recommend: impl FnOnce(usize) -> usize,
        f: impl FnOnce() -> R + Send,
    ) -> R {
        match self {
            Threads::Ambient => f(),
            policy => in_scoped_pool(policy.resolve(recommend), stack_bytes, f),
        }
    }
}

/// Run `f` inside a **scoped** rayon thread pool of `threads` workers, so this
/// factorization's parallelism is bounded and concurrent solves coexist instead of
/// each grabbing the global pool. Falls back to running on the current pool if the
/// build fails. `threads == 0` means all logical cores.
pub(crate) fn in_scoped_pool<R: Send>(
    threads: usize,
    stack_bytes: usize,
    f: impl FnOnce() -> R + Send,
) -> R {
    let n = if threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        threads
    };
    let mut builder = rayon::ThreadPoolBuilder::new().num_threads(n);
    if stack_bytes > 0 {
        builder = builder.stack_size(stack_bytes);
    }
    match builder.build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// Run `f` in a scoped rayon pool of `threads` workers (`0` = all logical cores),
/// then tear the pool down. The **solver-in-the-loop / embedded** entry point:
/// build **one** capped pool and drive many solves through it without a per-call
/// thread spawn.
///
/// Typical pattern - factor once (its own bounded, depth-stacked pool via the
/// default [`Threads::Auto`]`{max:4}`), then run the multi-RHS GMRES loop capped
/// at the same width:
/// ```ignore
/// let lu = factor_general_lu(&a, &SolverSettings::default())?;   // Auto{max:4}
/// with_threads(4, || {
///     for rhs in batches { let _ = gmres_block(&a, rhs, s, &lu, tol, it, m, None)?; }
///     Ok::<_, RslabError>(())
/// })?;
/// ```
/// The block GMRES orthogonalization picks up this pool automatically (it uses the
/// ambient rayon pool). To also run the *factorization* on this shared pool (e.g.
/// re-factoring every Newton step), pass [`Threads::Ambient`] in the settings.
///
/// The pool gets a 16 MB worker stack (the factorization stack floor), so an
/// `Ambient` factorization inside is safe for typical assembly-tree depths; the
/// iterative solvers do not deep-recurse. For pathologically deep trees (banded /
/// 1D at low `nemin`) build the pool yourself with a larger `stack_size`.
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    in_scoped_pool(threads, 16 * 1024 * 1024, f)
}

/// Maximum supernode-tree height (root-to-leaf), the recursion depth of the tree
/// factorization. Computed by an **iterative** post-order DFS (its own heap stack,
/// so it never recurses) and so is correct for any supernode numbering - not
/// assuming children precede their parent. O(#supernodes).
pub(crate) fn supernode_tree_depth(sym: &SymbolicFactorization) -> usize {
    let nsuper = sym.supernodes.len();
    let mut height = vec![0usize; nsuper];
    let mut is_child = vec![false; nsuper];
    for s in 0..nsuper {
        for &c in &sym.supernodes[s].children {
            is_child[c] = true;
        }
    }
    let mut max_h = 0;
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (node, next child index)
    for (r, &child) in is_child.iter().enumerate() {
        if child {
            continue;
        }
        stack.push((r, 0));
        while let Some(&(node, ci)) = stack.last() {
            let children = &sym.supernodes[node].children;
            if ci < children.len() {
                if let Some(top) = stack.last_mut() {
                    top.1 += 1;
                }
                stack.push((children[ci], 0));
            } else {
                let mut h = 1;
                for &c in children {
                    h = h.max(height[c] + 1);
                }
                height[node] = h;
                max_h = max_h.max(h);
                stack.pop();
            }
        }
    }
    max_h
}

/// Worker-thread stack size for a tree of the given depth. The recursive tree
/// factorization (`factor_subtree` / `ll_factor_subtree` and the LU twins) uses
/// O(depth) native stack; depth is O(log n) for nested-dissection orderings but
/// O(#supernodes) for deep chain trees - banded / 1D-like patterns, especially
/// with low `nemin`. Sizing the worker stack to the analyzed depth keeps the
/// factorization robust for every knob setting (the address space is reserved,
/// committed only as the recursion descends), instead of a fixed guess that a
/// deep enough chain overflows. `0` (shallow trees) keeps the rayon default.
pub(crate) fn stack_for_depth(depth: usize) -> usize {
    const FRAME: usize = 32 * 1024; // per-frame budget (LL ~6.7 KB measured; MF larger)
    const MIN: usize = 16 * 1024 * 1024; // floor (>= the rayon default; covers ~depth 500)
                                         // 8 GB cap (depth ~256k) on 64-bit; 1 GB on 32-bit targets (wasm32), where
                                         // the 64-bit literal would overflow usize at const evaluation.
    const MAX: usize = if usize::BITS >= 64 { 8 << 30 } else { 1 << 30 };
    // Always set an explicit, depth-proportional stack - never fall back to the
    // small rayon default, which a moderate depth (a few hundred supernodes, as a
    // banded matrix amalgamates to) already overflows.
    depth.saturating_mul(FRAME).clamp(MIN, MAX)
}

impl Default for SolverSettings {
    fn default() -> Self {
        use crate::numeric::gemm_tuning::{
            DEFAULT_PANEL_NB, DEFAULT_PAR_CDIV, DEFAULT_PAR_GEMM, DEFAULT_PIVOT_U,
            DEFAULT_SCALAR_GATE,
        };
        Self {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: None,
            threads: Threads::default(),
            interrupt: None,
            // Analysis-phase defaults (reproduce the historically-tuned analysis).
            ordering: OrderingMethod::default(),
            nemin: 16,
            // Relaxed amalgamation OFF. It was tuned in June on the MoM and FEM
            // classes, where padding narrow fundamental supernodes into wider
            // dense fronts pays; on the grid classes that entered the corpus
            // later it is a large pessimization, because the padded fronts carry
            // their explicit zeros through every update. Measured over the
            // 18-matrix head-to-head grid on the M3, relaxed vs off, interleaved,
            // minimum of three: geomean 0.654 for off, 16 of 18 matrices faster,
            // convection-diffusion 2D 2.6-4x, worst case curl-curl 14739 at
            // +12%. Fill is identical or lower without it (MoM 34.2M -> 32.1M).
            // Opt in per call with
            // `with_relax(Some(..))` where the fronts are dense enough to want it.
            relax: None,
            permutation: None,
            // Kernel defaults (reproduce the former process-wide atomic defaults).
            panel_nb: DEFAULT_PANEL_NB,
            scalar_gate: DEFAULT_SCALAR_GATE,
            par_gemm: DEFAULT_PAR_GEMM,
            par_cdiv: DEFAULT_PAR_CDIV,
            use_gemm_schur: true,
            pivot_u: DEFAULT_PIVOT_U,
            scaling: crate::scaling::ScalingStrategy::OnePassInfNorm,
            lu_matching: true,
        }
    }
}

impl SolverSettings {
    /// Exact, complete factorization (the default): fail on a singular pivot,
    /// no fill dropping. Use for a direct solve where accuracy is required.
    pub fn exact() -> Self {
        Self::default()
    }

    /// Robust never-fail **preconditioner** mode: static pivoting replaces any
    /// pivot below `abs_floor` (typically `eps_rel*||A||`) so the factorization
    /// always succeeds. Compose with [`with_drop_tol`](Self::with_drop_tol) for
    /// an incomplete preconditioner.
    pub fn preconditioner(abs_floor: f64) -> Self {
        Self {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor },
            ..Self::default()
        }
    }

    /// Builder: enable incomplete-factor threshold dropping (`|fill| < tau` is
    /// discarded, relative to the column/row).
    pub fn with_drop_tol(mut self, tau: f64) -> Self {
        self.drop_tol = Some(tau);
        self
    }

    /// Builder: set the near-zero pivot policy.
    pub fn with_pivot(mut self, policy: ZeroPivotAction) -> Self {
        self.on_zero_pivot = policy;
        self
    }

    /// Builder: set a **fixed** worker-thread budget (`0` = all logical cores).
    /// The factor runs in a scoped pool of this size so concurrent solves don't
    /// oversubscribe. Overrides the default [`Auto`](Threads::Auto) prediction.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Threads::Fixed(threads);
        self
    }

    /// Builder: use the **auto** per-matrix thread predictor, capped at `max`
    /// (`0` = all logical cores). This is the default policy; use it to bound the
    /// predictor below the full core count.
    pub fn with_auto_threads(mut self, max: usize) -> Self {
        self.threads = Threads::Auto { max };
        self
    }

    /// Builder: set the worker-thread policy directly.
    pub fn with_thread_policy(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }

    /// Builder: set the fill-reducing ordering method (analyze-time).
    pub fn with_ordering(mut self, ordering: OrderingMethod) -> Self {
        self.ordering = ordering;
        self
    }

    /// Builder: set the supernode amalgamation `nemin` (analyze-time).
    pub fn with_nemin(mut self, nemin: usize) -> Self {
        self.nemin = nemin;
        self
    }

    /// Builder: set the relaxed-amalgamation thresholds (`None` restricts to
    /// structural/size merges). Analyze-time.
    pub fn with_relax(mut self, relax: Option<RelaxAmalgamation>) -> Self {
        self.relax = relax;
        self
    }

    /// Builder: analyse with this fill-reducing ordering (`perm[k]` the column that
    /// becomes column `k`) instead of computing one. For a sequence of nearby patterns - a
    /// sweep whose drop tolerances move a few entries - the previous analysis's
    /// [`LuSymbolic::permutation`](crate::LuSymbolic::permutation) keeps its fill quality at
    /// the cost of the elimination tree and column counts alone, not of a new ordering.
    /// With the LU row matching the ordering is one of the row-matched matrix, so it carries
    /// over only where the (value-dependent) matching does. Analyze-time.
    pub fn with_permutation(mut self, perm: std::sync::Arc<[usize]>) -> Self {
        self.permutation = Some(perm);
        self
    }

    /// Builder: set the Bunch-Kaufman / LU panel width (kernel blocking factor).
    pub fn with_panel_nb(mut self, nb: usize) -> Self {
        self.panel_nb = nb;
        self
    }

    /// Builder: set the GEMM scheduling thresholds (scalar/SIMD and serial/parallel
    /// cutoffs) in one shot.
    pub fn with_gemm_thresholds(mut self, t: crate::numeric::gemm_tuning::GemmThresholds) -> Self {
        self.scalar_gate = t.scalar_gate;
        self.par_gemm = t.par_gemm;
        self.par_cdiv = t.par_cdiv;
        self
    }

    /// Builder: toggle the SIMD GEMM Schur update (vs the scalar triple loop).
    pub fn with_use_gemm_schur(mut self, on: bool) -> Self {
        self.use_gemm_schur = on;
        self
    }

    /// Builder: set the left-looking LU threshold partial-pivoting tolerance
    /// `u in [0, 1]` (clamped). Default `0.1`; `1.0` is full partial pivoting.
    /// See [`pivot_u`](Self::pivot_u).
    pub fn with_pivot_u(mut self, u: f64) -> Self {
        self.pivot_u = u.clamp(0.0, 1.0);
        self
    }

    /// Builder: set the symmetric equilibration strategy (analyze/factor-time,
    /// symmetric path). See [`scaling`](Self::scaling).
    pub fn with_scaling(mut self, scaling: crate::scaling::ScalingStrategy) -> Self {
        self.scaling = scaling;
        self
    }

    /// Enable or disable the MC64 row matching of the LU path (see
    /// [`SolverSettings::lu_matching`]).
    pub fn with_lu_matching(mut self, on: bool) -> Self {
        self.lu_matching = on;
        self
    }

    /// The kernel scheduling knobs as a cheap `Copy` bundle, threaded into the
    /// dense-front / left-looking kernels (replaces the former atomic loads).
    pub(crate) fn kernel(&self) -> crate::numeric::gemm_tuning::KernelTuning<'_> {
        crate::numeric::gemm_tuning::KernelTuning {
            scalar_gate: self.scalar_gate,
            par_gemm: self.par_gemm,
            par_cdiv: self.par_cdiv,
            panel_nb: self.panel_nb.max(8),
            use_gemm_schur: self.use_gemm_schur,
            pivot_u: self.pivot_u.clamp(0.0, 1.0),
            interrupt: self.interrupt.as_deref(),
        }
    }

    /// Builder: arm the numeric factorization with a caller-owned cancellation
    /// flag (see [`interrupt`](Self::interrupt)).
    pub fn with_interrupt(mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.interrupt = Some(flag);
        self
    }

    /// A static upper bound on the worker count for *reporting*, without the
    /// structural predictor: a fixed count resolves exactly; an
    /// [`Auto`](Threads::Auto) policy reports its cap (all cores for `0`). The
    /// concrete count actually used is resolved at factor time and recorded in
    /// the [`Diagnostics`](crate::Diagnostics).
    pub fn resolved_threads(&self) -> usize {
        match self.threads {
            Threads::Fixed(0) | Threads::Auto { max: 0 } => all_cores(),
            Threads::Fixed(n) | Threads::Auto { max: n } => n,
            Threads::Ambient => rayon::current_num_threads().max(1),
        }
    }

    /// The settings set to a non-default value that `path` does not read, each
    /// as one sentence naming the field and why. A factorization logs them as
    /// `Warning` records and carries them in its
    /// [`Diagnostics::warnings`](crate::Diagnostics::warnings), so a setting
    /// with no effect is never silent. Empty when every set field is honoured.
    pub fn ignored_on(&self, path: FactorPath) -> Vec<String> {
        let d = SolverSettings::default();
        let mut out = Vec::new();
        match path {
            FactorPath::Ldlt => {
                if self.pivot_u != d.pivot_u {
                    out.push(format!(
                        "pivot_u = {} is ignored by the LDL^T path (Bunch-Kaufman pivots the \
                         fully-summed block; the knob belongs to the left-looking LU)",
                        self.pivot_u
                    ));
                }
            }
            FactorPath::Lu => {
                if self.scaling != d.scaling {
                    out.push(format!(
                        "scaling = {:?} is ignored by the LU path (it equilibrates rows and \
                         columns with its own two-sided scaling)",
                        self.scaling
                    ));
                }
                if self.panel_nb != d.panel_nb {
                    out.push(format!(
                        "panel_nb = {} is ignored by the LU path (the panel width is an LDL^T \
                         kernel knob)",
                        self.panel_nb
                    ));
                }
                if self.use_gemm_schur != d.use_gemm_schur {
                    out.push(
                        "use_gemm_schur is ignored by the LU path (an LDL^T kernel A/B knob)"
                            .to_string(),
                    );
                }
            }
        }
        out
    }
}

/// The deterministic heuristic settings pick shared by `LdltSolver::tuned` and
/// `LuSolver::tuned`: default settings, an exact ND bakeoff on large systems,
/// and (feature `tuning`, when the one-time install diagnosis has run) the
/// calibrated cost-model worker count.
pub(crate) fn tuned<A: ?Sized, S>(
    a: &A,
    base: &SolverSettings,
    analyze_with: impl Fn(&A, &SolverSettings) -> Result<S, RslabError>,
    estimate: impl Fn(&S) -> MemoryEstimate,
) -> Result<(S, SolverSettings), RslabError> {
    #[cfg(not(feature = "tuning"))]
    let _ = &estimate;
    // The ordering race, whatever `base` asks for.
    #[allow(unused_mut)]
    let mut s = base.clone().with_ordering(OrderingMethod::Auto);
    let sym = analyze_with(a, &s)?;
    // Install-diagnosed worker count: only when a calibration cache exists
    // (written once by `tuning::install_diagnose`); never measures here.
    #[cfg(feature = "tuning")]
    if let Some((cores, calib)) = crate::tuning::cached_calibration() {
        let est = estimate(&sym);
        let t = crate::tuning::recommend_threads_cost_model(&est, &calib, 0, cores);
        s.threads = Threads::Fixed(t);
    }
    Ok((sym, s))
}
