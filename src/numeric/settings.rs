//! Solver settings shared by the LDL^T and LU paths: the numeric and
//! analysis knobs of [`SolverSettings`], grouped by phase, the static-pivot
//! policy, and the worker-thread policy with the scoped pools the
//! factorizations run in.
//!
//! Every tuning constant of the analysis, the kernels and the solve is a
//! field here, with the tuned value as its default. The groups compose
//! with struct update syntax:
//!
//! ```
//! use rslab::{KernelSettings, SolverSettings};
//! let mut s = SolverSettings::default().with_threads(8);
//! s.ordering.nd.fm_passes = 4;
//! s.kernels = KernelSettings { panel_nb: 32, ..Default::default() };
//! ```

use crate::constants::*;
use crate::scaling::ScalingStrategy;
use crate::symbolic::{
    AmalgamationStrategy, OrderingMethod, RelaxAmalgamation, SymbolicFactorization,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub use rslab_amd::AmdOptions;
pub use rslab_amf::AmfOptions;
pub use rslab_metis::MetisOptions;

/// Action to take when a near-zero pivot is encountered during factorization.
#[derive(Debug, Clone, PartialEq)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot at face value (zero the column, count as a zero in
    /// the inertia signature, flag for iterative refinement). The perturbation
    /// magnitude is unbounded - use only when downstream code tolerates sign
    /// loss in the perturbed positions and re-checks inertia.
    ForceAccept,
    /// Return [`RslabError::NumericallyRankDeficient`](crate::RslabError::NumericallyRankDeficient).
    Fail,
    /// Replace the tiny pivot with `sign(d) * max(|d|, abs_floor)`, keeping the
    /// column live (LAPACK / MA57-style static pivoting). The factor satisfies
    /// `L*D*L^T = A + delta ` for the produced `L`, `D`; `delta ` is bounded in the worst
    /// case by `||A[:,k]||^2 / abs_floor`, so drive iterative refinement against
    /// the unperturbed `A` for tight tolerances. The pivots it is compared
    /// with are those of the equilibrated matrix (entries of order one), so
    /// `abs_floor` is relative to it: about `sqrt(eps)` (1e-8) lifts only
    /// pivots that vanish to working precision, a larger floor also lifts
    /// genuine small pivots (the charge block of a saddle system has them at
    /// 1e-8 to 1e-6) and the factor becomes one of a perturbed matrix.
    /// [`NumericReport::min_pivot`](crate::diagnostics::NumericReport::min_pivot)
    /// shows how close the pivots came.
    PerturbToEps { abs_floor: f64 },
}

/// The factor path a [`SolverSettings`] is applied to; each reads a different
/// subset of the settings (see [`SolverSettings::ignored_on`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FactorPath {
    /// Symmetric `LDL^T` ([`LdltSolver`](crate::LdltSolver)).
    Ldlt,
    /// Unsymmetric LU ([`LuSolver`](crate::LuSolver)).
    Lu,
}

/// Settings of the sparse LDL^T and LU factorizations. Defaults give an
/// **exact** complete factorization that fails on rank deficiency; relaxing
/// [`pivoting`](Self::pivoting) turns it into a robust preconditioner. The
/// analysis groups
/// ([`ordering`](Self::ordering), [`amalgamation`](Self::amalgamation)) are
/// read when analyzing, the rest when factoring and solving.
#[derive(Debug, Clone)]
pub struct SolverSettings {
    /// Pivot selection and the near-zero pivot policy.
    pub pivoting: PivotSettings,
    /// Symmetric equilibration `A_hat = D A D` of the LDL^T path.
    /// [`Identity`](ScalingStrategy::Identity) disables it. The LU path uses
    /// its own two-sided scaling and ignores this.
    pub scaling: ScalingStrategy,
    /// The row matching of the LU path.
    pub matching: MatchingSettings,
    /// The fill-reducing ordering. Analyze-time.
    pub ordering: OrderingSettings,
    /// How columns merge into supernodes. Analyze-time.
    pub amalgamation: AmalgamationSettings,
    /// Blocking and scheduling of the dense kernels. They change the
    /// rounding at most, never the answer beyond it.
    pub kernels: KernelSettings,
    /// Scheduling of the triangular solves.
    pub solve: SolveSettings,
    /// Worker-thread policy. The factorization runs in a scoped rayon pool
    /// of this width; the numeric result does not depend on it.
    pub threads: Threads,
    /// Caller-owned cancellation flag for the numeric factorization. The solver
    /// only ever *reads* it, at supernode and dense-panel boundaries; on the
    /// first observation of `true` the factorization stops and returns
    /// [`RslabError::Interrupted`](crate::RslabError::Interrupted). Re-arming
    /// after an interrupt is the caller's `store(false)`. Default `None`.
    pub interrupt: Option<Arc<AtomicBool>>,
}

/// Pivot selection.
#[derive(Debug, Clone, PartialEq)]
pub struct PivotSettings {
    /// Threshold partial pivoting of the LU path, `u in [0, 1]`: the diagonal
    /// stays the pivot unless it falls below `u * |colmax|` in its
    /// fully-summed block. `1` is full partial pivoting, `0` keeps the
    /// diagonal unless it is exactly zero. Default `0.1`. The LDL^T path
    /// pivots by Bunch-Kaufman and ignores it.
    pub threshold: f64,
    /// What a pivot that is still too small does. Default
    /// [`Fail`](ZeroPivotAction::Fail).
    pub on_zero_pivot: ZeroPivotAction,
}

impl Default for PivotSettings {
    fn default() -> Self {
        Self {
            threshold: LU_PIVOT_THRESHOLD,
            on_zero_pivot: ZeroPivotAction::Fail,
        }
    }
}

/// Maximum-product row matching (MC64) before the LU analysis: rows are
/// permuted so the matched, largest-product entries form the diagonal and
/// both sides are scaled to make them unit magnitude, which keeps the
/// block-restricted pivoting stable (on the ibmpg1 power grid the residual
/// improves from 4e-5 to roundoff). With a usable diagonal the permutation
/// costs fill and pivot quality, so it is applied only where needed.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchingSettings {
    /// Allow the matching. Default `true`; `false` never matches.
    pub enabled: bool,
    /// A diagonal entry counts as missing below this fraction of its
    /// column's largest magnitude; the matching runs when any is missing.
    /// Default `1e-10`.
    pub negligible_diagonal: f64,
}

impl Default for MatchingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            negligible_diagonal: MATCHING_NEGLIGIBLE_DIAGONAL,
        }
    }
}

/// The fill-reducing ordering.
#[derive(Debug, Clone)]
pub struct OrderingSettings {
    /// The method. Default [`Auto`](OrderingMethod::Auto), the race of
    /// [`race`](Self::race) on exact fill.
    pub method: OrderingMethod,
    /// An ordering to use instead of computing one (`perm[k]` the column that
    /// becomes column `k`). For a sequence of nearby patterns the previous
    /// analysis's [`LuSymbolic::permutation`](crate::LuSymbolic::permutation)
    /// keeps its fill quality at the cost of the elimination tree and column
    /// counts alone. With the LU row matching the ordering is one of the
    /// row-matched matrix, so it carries over only where the matching does.
    pub permutation: Option<Arc<[usize]>>,
    /// Order the graph of indistinguishable-vertex groups only when the
    /// groups shrink it to at most this share of its vertices. Default
    /// `0.95`.
    pub compress_max_ratio: f64,
    /// The ordering race of [`OrderingMethod::Auto`].
    pub race: RaceSettings,
    /// Nested dissection.
    pub nd: MetisOptions,
    /// Approximate minimum degree.
    pub amd: AmdOptions,
    /// Approximate minimum fill.
    pub amf: AmfOptions,
}

impl Default for OrderingSettings {
    fn default() -> Self {
        Self {
            method: OrderingMethod::Auto,
            permutation: None,
            compress_max_ratio: COMPRESS_MAX_RATIO,
            race: RaceSettings::default(),
            nd: MetisOptions::default(),
            amd: AmdOptions::default(),
            amf: AmfOptions::default(),
        }
    }
}

/// The ordering race: the cheap candidates always run, nested dissection
/// joins where the predicted factorization can pay for it, and the smallest
/// exact factor wins, candidate order breaking ties.
#[derive(Debug, Clone, PartialEq)]
pub struct RaceSettings {
    /// The cheap candidates, run concurrently. The first one also decides,
    /// on smaller patterns, whether nested dissection starts before the
    /// others finish. Default AMD, AMF, RCM; nested dissection and `Auto`
    /// are not allowed here.
    pub candidates: Vec<OrderingMethod>,
    /// Nested dissection joins only above this many unknowns. Default
    /// `10_000`.
    pub nd_min_n: usize,
    /// ... and only when the best cheap candidate predicts at least this
    /// factor time, in flops per worker or on the longest elimination chain.
    /// Default `1.25e9`.
    pub nd_min_work: u64,
    /// Workers the time prediction assumes. Default `4`.
    pub assumed_workers: usize,
    /// Patterns with at least this many entries (lower triangle) start the
    /// dissection at once, speculatively, where more than one worker is
    /// available. Default `2_000_000`.
    pub eager_nd_min_nnz: usize,
    /// Keep the best of several dissection seeds on heavy factorizations.
    /// Buys up to a few percent of fill (15 percent on 3D meshes) for more
    /// dissections, which pays over many refactorizations of one analysis.
    /// Default `false`.
    pub ensemble: bool,
    /// Seeds of the ensemble, `nd.seed` upwards. Default `3`.
    pub ensemble_size: usize,
    /// The ensemble runs from this predicted flop count (sum of squared
    /// column counts). Default `5e10`.
    pub ensemble_min_flops: u64,
}

impl Default for RaceSettings {
    fn default() -> Self {
        Self {
            candidates: vec![
                OrderingMethod::Amd,
                OrderingMethod::Amf,
                OrderingMethod::Rcm,
            ],
            nd_min_n: RACE_ND_MIN_N,
            nd_min_work: RACE_ND_MIN_WORK,
            assumed_workers: RACE_ASSUMED_WORKERS,
            eager_nd_min_nnz: RACE_EAGER_ND_MIN_NNZ,
            ensemble: false,
            ensemble_size: RACE_ENSEMBLE_SIZE,
            ensemble_min_flops: RACE_ENSEMBLE_MIN_FLOPS,
        }
    }
}

/// How columns merge into supernodes.
#[derive(Debug, Clone, PartialEq)]
pub struct AmalgamationSettings {
    /// Supernodes narrower than this merge into their parent when the parent
    /// is narrower too. Default `16`; `1` turns the size rule off.
    pub nemin: usize,
    /// Relaxed amalgamation: merge beyond the size rule while the merged
    /// supernode stays within `max_width` columns and adds at most
    /// `max_extra_rows` rows of explicit zeros. Wider fronts run their
    /// updates at a higher GEMM rank for a little fill; on grid classes the
    /// padded zeros cost more than that buys, so it is off by default.
    pub relax: Option<RelaxAmalgamation>,
    /// Relaxed amalgamation applies from this many unknowns. Default `1024`.
    pub relax_min_n: usize,
    /// How merges reach non-adjacent children. Default
    /// [`Auto`](AmalgamationStrategy::Auto).
    pub strategy: AmalgamationStrategy,
    /// `Auto` treats a tree as path-like, and does not renumber, when fewer
    /// than this share of its internal nodes have several children. Default
    /// `0.05`.
    pub path_like_fraction: f64,
    /// Merges into a root supernode stop at `root_cap_fraction * n` columns,
    /// at most `root_cap_max`, from `root_cap_min_n` unknowns: a wide
    /// top-level Schur complement (interior-point KKT systems) otherwise
    /// grows one dense root block. Defaults `1024`, `0.05`, `2048`.
    pub root_cap_min_n: usize,
    /// See [`root_cap_min_n`](Self::root_cap_min_n).
    pub root_cap_fraction: f64,
    /// See [`root_cap_min_n`](Self::root_cap_min_n).
    pub root_cap_max: usize,
}

impl Default for AmalgamationSettings {
    fn default() -> Self {
        Self {
            nemin: NEMIN,
            relax: None,
            relax_min_n: RELAX_MIN_N,
            strategy: AmalgamationStrategy::Auto,
            path_like_fraction: PATH_LIKE_FRACTION,
            root_cap_min_n: ROOT_CAP_MIN_N,
            root_cap_fraction: ROOT_CAP_FRACTION,
            root_cap_max: ROOT_CAP_MAX,
        }
    }
}

/// Blocking and scheduling of the dense kernels, in flops of the update
/// they gate unless noted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSettings {
    /// Bunch-Kaufman panel width. Changes the pivot search window (a
    /// different, equally valid factor). Default `64`, at least 8.
    pub panel_nb: usize,
    /// Below this an update runs as a scalar loop instead of a SIMD GEMM.
    /// Default `4096`.
    pub scalar_gate: usize,
    /// From this an update GEMM takes the parallel path (its kernel and
    /// blocking, which fix the rounding of complex products), on as many
    /// threads as its work is worth there, one included
    /// ([`crate::parallel`]). Default `1e6`.
    pub par_gemm: usize,
    /// From this a panel's trailing update takes the parallel path, on as
    /// many threads as it is worth. Default `8e6`.
    pub par_cdiv: usize,
    /// From this the work of one supernode takes the forking path: its
    /// updates (cmod flops) in column slabs and its panel factorization
    /// (cdiv work `nrow * ncol^2`) in parallel sections, each on as many
    /// threads as it is worth. A small node that forks pays the join-steal
    /// latency: its waiting thread steals other work, often a whole sibling
    /// subtree, and the node's chain stalls (74 ms of cmod at 0.03 Gflop
    /// measured on a 1046x170 node). The path fixes the rounding, so this
    /// is a structural gate, never one of time. Default `1e8`.
    pub fork_min_flops: usize,
    /// Column tile of the lower-triangular Schur GEMM. Default `256`.
    pub schur_tile: usize,
    /// Columns per sub-block of the deferred trailing sweep in the
    /// Bunch-Kaufman panel (the rank of its GEMMs). Default `16`.
    pub trailing_block: usize,
    /// A complex product runs as real products on split planes when its
    /// flops `m n k` are at least this many times its plane copies
    /// `m k + k n + m n`; thin products lose on the copies. Default `64`.
    pub complex_split_min_ratio: usize,
    /// Tile edge of that split, which bounds its scratch. Default `256`.
    pub complex_split_tile: usize,
    /// Use the SIMD GEMM for the LDL^T Schur update (vs the scalar loop).
    /// Default `true`.
    pub use_gemm_schur: bool,
    /// Nodes with at least this many columns factor in panels of at least
    /// [`wide_panel_nb`](Self::wide_panel_nb) columns, on both paths: the
    /// deferred Schur GEMM's inner dimension is the panel width, too thin
    /// at the default width to reach peak on root-class panels. Changes the
    /// pivot window of those nodes. Default `512`.
    pub wide_panel_min_ncol: usize,
    /// The least panel width of those wide nodes. Default `128`.
    pub wide_panel_nb: usize,
    /// Panel width of the LU path's panel factorization (the LDL^T path's
    /// is [`panel_nb`](Self::panel_nb)). Changes the pivot window. Default
    /// `32`, at least 1.
    pub lu_panel_nb: usize,
    /// A parallel update of a node splits its columns into about this many
    /// tiles of [`cmod_tile_min`](Self::cmod_tile_min) to
    /// [`cmod_tile_max`](Self::cmod_tile_max) columns. Default `16`.
    pub cmod_tiles: usize,
    /// The narrowest such tile. Default `32`, at least 1.
    pub cmod_tile_min: usize,
    /// The widest such tile. Default `256`.
    pub cmod_tile_max: usize,
    /// Rows per task of the parallel `U12` triangular solve in the LU
    /// panel factorization. Default `256`.
    pub trsm_rows: usize,
}

impl KernelSettings {
    /// The panel width of a node of `ncol` columns whose kernel's own width
    /// is `base`: at least [`wide_panel_nb`](Self::wide_panel_nb) from
    /// [`wide_panel_min_ncol`](Self::wide_panel_min_ncol) columns.
    pub(crate) fn panel_width(&self, base: usize, ncol: usize) -> usize {
        if ncol >= self.wide_panel_min_ncol {
            base.max(self.wide_panel_nb)
        } else {
            base
        }
    }

    /// The parallel threshold of the panel factorization of an `nrow x ncol`
    /// node: [`par_cdiv`](Self::par_cdiv) from
    /// [`fork_min_flops`](Self::fork_min_flops) of work `nrow * ncol^2`,
    /// else never, so a small node does not fork inside its cdiv either.
    pub(crate) fn cdiv_par(&self, nrow: usize, ncol: usize) -> usize {
        if nrow * ncol * ncol >= self.fork_min_flops {
            self.par_cdiv
        } else {
            usize::MAX
        }
    }

    /// The column tile of a parallel update over `width` columns.
    pub(crate) fn cmod_tile(&self, width: usize) -> usize {
        (width / self.cmod_tiles.max(1))
            .max(self.cmod_tile_min)
            .min(self.cmod_tile_max)
    }
}

impl Default for KernelSettings {
    fn default() -> Self {
        Self {
            panel_nb: PANEL_NB,
            scalar_gate: SCALAR_GATE,
            par_gemm: PAR_GEMM_MIN_FLOPS,
            par_cdiv: PAR_CDIV_MIN_FLOPS,
            fork_min_flops: FORK_MIN_FLOPS,
            schur_tile: SCHUR_TILE,
            trailing_block: TRAILING_BLOCK,
            complex_split_min_ratio: COMPLEX_SPLIT_MIN_RATIO,
            complex_split_tile: COMPLEX_SPLIT_TILE,
            use_gemm_schur: true,
            wide_panel_min_ncol: WIDE_PANEL_MIN_NCOL,
            wide_panel_nb: WIDE_PANEL_NB,
            lu_panel_nb: LU_PANEL_NB,
            cmod_tiles: CMOD_TILES,
            cmod_tile_min: CMOD_TILE_MIN,
            cmod_tile_max: CMOD_TILE_MAX,
            trsm_rows: TRSM_ROWS,
        }
    }
}

/// Scheduling of the supernodal triangular solves. The result does not
/// depend on the thread count; these change at most the rounding. The nodes'
/// products run as GEMMs, in row chunks on the pool where their work is
/// worth it there ([`crate::parallel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveSettings {
    /// Independent leaf subtrees the tree is cut into for the parallel
    /// sweeps. Default `128`.
    pub leaf_subtrees: usize,
    /// A fixed gate in place of the time gate: the work of a triangular
    /// sweep from which it runs on the thread pool, in the factor's panel
    /// entries times the right-hand sides, up to four (not the fmadds of
    /// [`KluSettings::par_min_work`](crate::KluSettings::par_min_work)).
    /// Default `None`: a sweep runs on the pool where its work, at the
    /// rate the sweeps have measured, is worth more than one thread there
    /// ([`crate::parallel`]); a smaller one on the calling thread, where
    /// handing it to the pool costs more than it saves. The same bits
    /// either way.
    pub par_min_work: Option<usize>,
}

impl Default for SolveSettings {
    fn default() -> Self {
        Self {
            leaf_subtrees: SOLVE_LEAF_SUBTREES,
            par_min_work: None,
        }
    }
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            pivoting: PivotSettings::default(),
            scaling: ScalingStrategy::OnePassInfNorm,
            matching: MatchingSettings::default(),
            ordering: OrderingSettings::default(),
            amalgamation: AmalgamationSettings::default(),
            kernels: KernelSettings::default(),
            solve: SolveSettings::default(),
            threads: Threads::default(),
            interrupt: None,
        }
    }
}

impl SolverSettings {
    /// Exact, complete factorization (the default): fail on a singular pivot.
    pub fn exact() -> Self {
        Self::default()
    }

    /// Robust never-fail preconditioner: static pivoting lifts any pivot below
    /// `abs_floor` (typically `eps_rel * ||A||`) to it, so the factorization
    /// always succeeds.
    pub fn preconditioner(abs_floor: f64) -> Self {
        Self::default().with_zero_pivot(ZeroPivotAction::PerturbToEps { abs_floor })
    }

    /// Set the near-zero pivot policy.
    pub fn with_zero_pivot(mut self, action: ZeroPivotAction) -> Self {
        self.pivoting.on_zero_pivot = action;
        self
    }

    /// Set the LU pivot threshold `u`, clamped to `[0, 1]` (see
    /// [`PivotSettings::threshold`]).
    pub fn with_pivot_threshold(mut self, u: f64) -> Self {
        self.pivoting.threshold = u.clamp(0.0, 1.0);
        self
    }

    /// Set the worker-thread policy; a number is a fixed count (`0` = all
    /// logical cores).
    pub fn with_threads(mut self, threads: impl Into<Threads>) -> Self {
        self.threads = threads.into();
        self
    }

    /// Set the ordering method. Analyze-time.
    pub fn with_ordering(mut self, method: OrderingMethod) -> Self {
        self.ordering.method = method;
        self
    }

    /// Analyze with this ordering instead of computing one (see
    /// [`OrderingSettings::permutation`]).
    pub fn with_permutation(mut self, perm: Arc<[usize]>) -> Self {
        self.ordering.permutation = Some(perm);
        self
    }

    /// Run the nested-dissection seed ensemble (see
    /// [`RaceSettings::ensemble`]).
    pub fn with_nd_ensemble(mut self, on: bool) -> Self {
        self.ordering.race.ensemble = on;
        self
    }

    /// Set the amalgamation `nemin`. Analyze-time.
    pub fn with_nemin(mut self, nemin: usize) -> Self {
        self.amalgamation.nemin = nemin;
        self
    }

    /// Set the relaxed amalgamation (`None` for none). Analyze-time.
    pub fn with_relax(mut self, relax: Option<RelaxAmalgamation>) -> Self {
        self.amalgamation.relax = relax;
        self
    }

    /// Set the symmetric equilibration of the LDL^T path.
    pub fn with_scaling(mut self, scaling: ScalingStrategy) -> Self {
        self.scaling = scaling;
        self
    }

    /// Allow or forbid the LU row matching (see [`MatchingSettings`]).
    pub fn with_matching(mut self, on: bool) -> Self {
        self.matching.enabled = on;
        self
    }

    /// Arm the numeric factorization with a caller-owned cancellation flag
    /// (see [`interrupt`](Self::interrupt)).
    pub fn with_interrupt(mut self, flag: Arc<AtomicBool>) -> Self {
        self.interrupt = Some(flag);
        self
    }

    /// These settings with the worker count resolved to `threads` (an
    /// `Ambient` policy stays ambient), so every stage of one factorization
    /// uses the same count.
    pub(crate) fn pinned(&self, threads: usize) -> Self {
        Self {
            threads: match self.threads {
                Threads::Ambient => Threads::Ambient,
                _ => Threads::Fixed(threads),
            },
            ..self.clone()
        }
    }

    /// The kernel knobs as the `Copy` bundle the kernels take.
    pub(crate) fn kernel(&self) -> crate::numeric::gemm_tuning::KernelTuning<'_> {
        crate::numeric::gemm_tuning::KernelTuning {
            k: KernelSettings {
                panel_nb: self.kernels.panel_nb.max(PANEL_NB_MIN),
                lu_panel_nb: self.kernels.lu_panel_nb.max(1),
                cmod_tile_min: self.kernels.cmod_tile_min.max(1),
                cmod_tile_max: self.kernels.cmod_tile_max.max(1),
                trsm_rows: self.kernels.trsm_rows.max(1),
                ..self.kernels
            },
            pivot_threshold: self.pivoting.threshold.clamp(0.0, 1.0),
            interrupt: self.interrupt.as_deref(),
        }
    }

    /// A static upper bound on the worker count for *reporting*, without the
    /// structural predictor: a fixed count resolves exactly; an
    /// [`Auto`](Threads::Auto) policy reports its cap (all cores for `0`). The
    /// concrete count actually used is resolved at factor time and recorded in
    /// the [`Diagnostics`](crate::Diagnostics).
    pub fn resolved_threads(&self) -> usize {
        match self.threads {
            _ if crate::parallel::lent().is_some() => crate::parallel::threads(),
            Threads::Fixed(0) | Threads::Auto { max: 0 } => all_cores(),
            Threads::Fixed(n) | Threads::Auto { max: n } => n,
            Threads::Ambient => rayon::current_num_threads().max(1),
        }
    }

    /// The settings set to a non-default value that `path` does not read, each
    /// as one sentence naming the field and why. A factorization logs them as
    /// `Warning` records and carries them in its
    /// [`Diagnostics::warnings`](crate::Diagnostics::warnings), so a setting
    /// with no effect is never silent.
    pub(crate) fn ignored_on(&self, path: FactorPath) -> Vec<String> {
        let d = SolverSettings::default();
        let mut out = Vec::new();
        match path {
            FactorPath::Ldlt => {
                if self.pivoting.threshold != d.pivoting.threshold {
                    out.push(format!(
                        "pivoting.threshold = {} is ignored by the LDL^T path (Bunch-Kaufman \
                         pivots the fully-summed block)",
                        self.pivoting.threshold
                    ));
                }
                if self.matching != d.matching {
                    out.push("matching is ignored by the LDL^T path (an LU setting)".to_string());
                }
                if self.kernels.lu_panel_nb != d.kernels.lu_panel_nb {
                    out.push(format!(
                        "kernels.lu_panel_nb = {} is ignored by the LDL^T path (an LU kernel knob)",
                        self.kernels.lu_panel_nb
                    ));
                }
                if self.kernels.trsm_rows != d.kernels.trsm_rows {
                    out.push(format!(
                        "kernels.trsm_rows = {} is ignored by the LDL^T path (an LU kernel knob)",
                        self.kernels.trsm_rows
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
                if self.kernels.panel_nb != d.kernels.panel_nb {
                    out.push(format!(
                        "kernels.panel_nb = {} is ignored by the LU path (an LDL^T kernel knob)",
                        self.kernels.panel_nb
                    ));
                }
                if self.kernels.use_gemm_schur != d.kernels.use_gemm_schur {
                    out.push(
                        "kernels.use_gemm_schur is ignored by the LU path (an LDL^T kernel knob)"
                            .to_string(),
                    );
                }
            }
        }
        out
    }
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
    /// validated policy: thin / tiny systems stay low where they would only regress, big
    /// BLAS-3-rich systems use the cores), **capped at `max`** (`0` = all logical
    /// cores). The single-solve default: best throughput without oversubscribing
    /// the matrices that do not scale.
    Auto {
        /// Upper bound on the predicted worker count (`0` = all logical cores).
        max: usize,
    },
    /// Use the **current** rayon pool as-is, without building a scoped pool. The
    /// solver-in-the-loop path: build **one** bounded rayon pool (e.g. 4 workers)
    /// and run the factorization *and* every
    /// iterative solve inside it, so both phases share the same capped pool with no
    /// per-call thread spawn. The numeric factor is unchanged.
    ///
    /// Called from the tasks of a parallel sweep on that pool, the sections
    /// share the sweep's workers, and a worker waiting in one takes the
    /// sweep's other tasks meanwhile, as rayon's nested parallelism does. The
    /// other policies run in pools of their own, entered from such a worker
    /// through a thread that waits for it, so a sweep's tasks never nest.
    Ambient,
}

impl Default for Threads {
    fn default() -> Self {
        Threads::Auto {
            max: DEFAULT_MAX_THREADS,
        }
    }
}

/// A fixed worker count (`0` = all logical cores).
impl From<usize> for Threads {
    fn from(n: usize) -> Self {
        Threads::Fixed(n)
    }
}

/// All logical cores (the `0` sentinel resolution).
pub(crate) fn all_cores() -> usize {
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
            // A lent pool's workers, whatever the policy asked for.
            _ if crate::parallel::lent().is_some() => crate::parallel::threads(),
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
            // On a lent pool (see `crate::parallel`), or the caller's rayon pool.
            _ if crate::parallel::lent().is_some() => f(),
            Threads::Ambient => f(),
            policy => in_scoped_pool(policy.resolve(recommend), stack_bytes, f),
        }
    }
}

/// Run `f` inside a **scoped** rayon thread pool of `threads` workers, so this
/// factorization's parallelism is bounded and concurrent solves coexist instead of
/// each grabbing the global pool. Falls back to running on the current pool if the
/// build fails. `threads == 0` means all logical cores.
///
/// The pool is the calling thread's own, built on its first use and kept (see
/// [`POOLS`]): a factorization per Newton step would otherwise start and join
/// its workers every time, tens of microseconds per worker.
///
/// Work that needs no stack of its own (`stack_bytes == 0`, the solves)
/// called on a worker of one of these pools runs there: it is part of the
/// work the pool was entered for. Called on a worker of any other rayon
/// pool, it enters a pool of its own like the factorizations do (see
/// [`enter`]): run on the caller's pool, its sections would make the worker
/// wait inside the caller's task and take the caller's other tasks meanwhile.
pub(crate) fn in_scoped_pool<R: Send>(
    threads: usize,
    stack_bytes: usize,
    f: impl FnOnce() -> R + Send,
) -> R {
    if crate::parallel::lent().is_some() || (stack_bytes == 0 && on_own_worker()) {
        return f();
    }
    let n = if threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        threads
    };
    match scoped_pool(n, stack_bytes) {
        // The pool's section cost is measured on its first use, before any
        // work keeps its workers busy.
        Some(pool) => enter(&pool, || {
            crate::parallel::prepare();
            f()
        }),
        None => f(),
    }
}

thread_local! {
    /// Whether this thread is a worker of a pool of rslab's own
    /// ([`scoped_pool`]).
    static OWN_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the calling thread is a worker of a pool of rslab's own.
fn on_own_worker() -> bool {
    OWN_WORKER.with(std::cell::Cell::get)
}

/// Run `f` on `pool`, the calling thread waiting for it. A worker of another
/// rayon pool waiting in `install` keeps taking its own pool's jobs (rayon's
/// cross-pool wait), and each of those may enter here again: a parallel
/// sweep of factorizations then nests without bound and overflows the
/// worker's stack. Such a caller enters through a thread of its own
/// ([`Hop`]), which waits without taking work, while the caller blocks.
pub(crate) fn enter<R: Send>(pool: &rayon::ThreadPool, f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_none() || pool.current_thread_index().is_some() {
        return pool.install(f);
    }
    let (mut f, mut r) = (Some(f), None);
    let entered = HOP.with(|h| {
        h.run(&mut || {
            if let Some(f) = f.take() {
                r = Some(pool.install(f));
            }
        })
    });
    match (entered, r, f) {
        (true, Some(r), _) => r,
        // No thread to enter through: the waiting worker may take work.
        (false, _, Some(f)) => pool.install(f),
        _ => unreachable!("a hop runs the job it takes, or panics"),
    }
}

thread_local! {
    /// The calling thread's [`Hop`], its thread started on the first entry.
    static HOP: Hop = const { Hop { shared: std::cell::OnceCell::new() } };
}

/// A thread a worker of a foreign rayon pool enters rslab's pools through:
/// it takes one job at a time from its worker and runs it, the worker
/// blocked until it is done. Kept for the worker's life, so an entry costs
/// a hand-over and back, no thread start and no allocation.
struct Hop {
    shared: std::cell::OnceCell<Option<Arc<HopShared>>>,
}

/// What a worker and its hop thread share: the slot of the job in hand and
/// the signal both wait on.
struct HopShared {
    slot: std::sync::Mutex<HopSlot>,
    signal: std::sync::Condvar,
}

/// A job handed to a hop thread: the worker's closure, valid until the
/// worker sees [`HopSlot::Done`] (it blocks until then).
struct HopJob(*mut (dyn FnMut() + Send + 'static));
// SAFETY: the closure is `Send`; the worker blocks while the hop runs it.
unsafe impl Send for HopJob {}

/// The state of a hop's slot.
enum HopSlot {
    Empty,
    Job(HopJob),
    Done(std::thread::Result<()>),
    Quit,
}

impl Hop {
    /// Run `job` on this thread's hop, blocking until it ran (its panic
    /// resumed here); `false` if no hop thread could be started.
    fn run(&self, job: &mut (dyn FnMut() + Send)) -> bool {
        let Some(shared) = self.shared.get_or_init(HopShared::start) else {
            return false;
        };
        // SAFETY: only the lifetime is erased. This thread blocks below
        // until the hop reports the job done, so the closure outlives every
        // use the hop makes of it.
        let job: *mut (dyn FnMut() + Send + 'static) = unsafe { std::mem::transmute(job) };
        let mut slot = lock(&shared.slot);
        *slot = HopSlot::Job(HopJob(job));
        shared.signal.notify_all();
        loop {
            match std::mem::replace(&mut *slot, HopSlot::Empty) {
                HopSlot::Done(Ok(())) => return true,
                HopSlot::Done(Err(panic)) => {
                    drop(slot);
                    std::panic::resume_unwind(panic)
                }
                other => *slot = other,
            }
            slot = shared.signal.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Drop for Hop {
    fn drop(&mut self) {
        if let Some(Some(shared)) = self.shared.get() {
            *lock(&shared.slot) = HopSlot::Quit;
            shared.signal.notify_all();
        }
    }
}

impl HopShared {
    /// The shared slot with its hop thread running; `None` if the thread
    /// does not start.
    fn start() -> Option<Arc<HopShared>> {
        let shared = Arc::new(HopShared {
            slot: std::sync::Mutex::new(HopSlot::Empty),
            signal: std::sync::Condvar::new(),
        });
        let hop = shared.clone();
        std::thread::Builder::new()
            .name("rslab-hop".into())
            .stack_size(HOP_STACK)
            .spawn(move || hop.serve())
            .ok()?;
        Some(shared)
    }

    /// The hop thread: run each job handed over and report it done, until
    /// told to quit.
    fn serve(&self) {
        let mut slot = lock(&self.slot);
        loop {
            match std::mem::replace(&mut *slot, HopSlot::Empty) {
                HopSlot::Job(HopJob(job)) => {
                    drop(slot);
                    // SAFETY: see `Hop::run`; the worker waits for `Done`.
                    let done = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                        (*job)()
                    }));
                    slot = lock(&self.slot);
                    *slot = HopSlot::Done(done);
                    self.signal.notify_all();
                }
                HopSlot::Quit => return,
                other => *slot = other,
            }
            slot = self.signal.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// `m` locked: a lock is poisoned only by a panic, which its holder carries
/// on.
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

thread_local! {
    /// The scoped pools of this thread by worker count and stack size, most
    /// recently used last. Per thread, so concurrent callers keep separate
    /// pools as before; a pool with a larger stack serves a smaller need.
    static POOLS: std::cell::RefCell<Vec<(usize, usize, std::sync::Arc<rayon::ThreadPool>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// This thread's pool of `n` workers with at least `stack_bytes` of stack
/// (`0` for rayon's default), built when it has none; `None` if the build
/// fails. Its workers know they are rslab's ([`on_own_worker`]).
pub(crate) fn scoped_pool(
    n: usize,
    stack_bytes: usize,
) -> Option<std::sync::Arc<rayon::ThreadPool>> {
    // Stacks by powers of two, so a few pools serve every depth.
    let stack = if stack_bytes == 0 {
        0
    } else {
        stack_bytes
            .checked_next_power_of_two()
            .unwrap_or(stack_bytes)
    };
    POOLS.with(|pools| {
        let mut pools = pools.borrow_mut();
        let fits =
            |&(w, s, _): &(usize, usize, _)| w == n && (s >= stack) && (s == 0) == (stack == 0);
        if let Some(i) = pools.iter().position(fits) {
            let entry = pools.remove(i);
            let pool = entry.2.clone();
            pools.push(entry);
            return Some(pool);
        }
        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .start_handler(|_| OWN_WORKER.with(|w| w.set(true)));
        if stack > 0 {
            builder = builder.stack_size(stack);
        }
        let pool = std::sync::Arc::new(builder.build().ok()?);
        if pools.len() == POOLS_KEPT {
            pools.remove(0);
        }
        pools.push((n, stack, pool.clone()));
        Some(pool)
    })
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
    // Always set an explicit, depth-proportional stack - never fall back to the
    // small rayon default, which a moderate depth (a few hundred supernodes, as a
    // banded matrix amalgamates to) already overflows.
    depth
        .saturating_mul(STACK_PER_LEVEL)
        .clamp(STACK_MIN, STACK_MAX)
}
