//! Deterministic resource diagnostics: an **a-priori** peak-memory estimate
//! (computed from the symbolic factorization, before any numeric work) and
//! the account of a factorization: its decisions, counts and pivots. The
//! times of the phases are the scopes of the `rsprobe` record.
//!
//! The estimate is a pure function of the analyzed structure, so it is fully
//! reproducible and lets a solver-in-the-loop scheduler decide *before* allocating
//! whether a factorization fits the memory budget (fail-fast / pick approximation).

use crate::constants::{GIGA, MEGA, MS_PER_S, RATE_FLOOR};
use std::fmt;

/// A-priori estimate of the memory a factorization will use, in bytes. All fields
/// are deterministic functions of the symbolic structure and the scalar size.
/// The itemized account behind [`transient_peak_bytes`](Self::transient_peak_bytes)
/// is the [`MemoryPlan`](crate::MemoryPlan) of the path's `memory_plan`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryEstimate {
    /// Scalar size in bytes (`16` for `Complex<f64>`, `8` for `f64`, ...).
    pub value_bytes: usize,
    /// Structural nonzeros in the factor (`L`+`U` for LU, `L` for LDL^T) - an upper
    /// bound on the emitted factor (numeric cancellation can only lower it).
    pub factor_nnz: u64,
    /// Heap bytes the factor holds: its values, pivots and solve schedule.
    pub factor_bytes: u64,
    /// The dense supernode panels (`0` for KLU).
    pub panels_all_bytes: u64,
    /// The panels live at once. The left-looking drivers factor into one
    /// arena holding every panel, so this equals
    /// [`panels_all_bytes`](Self::panels_all_bytes).
    pub panel_live_peak_bytes: u64,
    /// Heap peak from the analysis through a one-column solve, on all cores
    /// (the most kernel scratch): [`MemoryPlan::peak_bytes`](crate::MemoryPlan::peak_bytes)
    /// of the path's `memory_plan`. The number to compare against RAM.
    pub transient_peak_bytes: u64,
    /// Geometric factorization work proxy `sum nrow^2*ncol` over supernodes (type-
    /// independent). Divide by a calibrated geometric-flops/s rate for a runtime
    /// estimate - see [`est_runtime_ms`](Self::est_runtime_ms).
    pub factor_flops: u64,
    /// Critical-path geom-flops: the longest serial chain of front work from a
    /// leaf to a root of the assembly tree (`front_flops(s) + max child`). This is
    /// the Amdahl lower bound on parallel factor time --- even with unlimited
    /// workers the tree cannot factor below `critical_path_flops / rate`, since a
    /// front depends on its children. The v2 thread-aware time model uses it to
    /// decide the worker count (a memory-bound or critical-path-bound matrix gains
    /// nothing, and may lose, from more threads). `0` until the tree pass fills it.
    pub critical_path_flops: u64,
    /// Peak assembly-tree width: the most supernodes at any one level, i.e. the
    /// maximum node-level parallelism available. Caps the useful worker count.
    pub max_tree_width: u64,
}

impl MemoryEstimate {
    pub fn transient_peak_mb(&self) -> f64 {
        self.transient_peak_bytes as f64 / MEGA
    }
    pub fn factor_mb(&self) -> f64 {
        self.factor_bytes as f64 / MEGA
    }
    /// Does the estimated transient peak fit in `available` bytes?
    pub fn fits_in(&self, available_bytes: u64) -> bool {
        self.transient_peak_bytes <= available_bytes
    }

    /// Estimated factor wall-clock in ms: `factor_flops` divided by a calibrated
    /// geometric-flops/s rate (`gflops` = giga-geom-flops/s on one thread) scaled by
    /// the measured `parallel_speedup` at the chosen thread count. Both come from
    /// the calibration (`tuning` feature); pass machine defaults otherwise.
    pub fn est_runtime_ms(&self, gflops: f64, parallel_speedup: f64) -> f64 {
        let rate = (gflops.max(RATE_FLOOR) * parallel_speedup.max(RATE_FLOOR)) * GIGA;
        (self.factor_flops as f64 / rate) * MS_PER_S
    }

    /// Thread-aware runtime estimate (the v2 model): the parallel time cannot fall
    /// below the **critical path** of the assembly tree (Amdahl), so it is the max
    /// of the serial critical-path floor and the work divided by the achieved
    /// parallel rate. `gflops` is the one-thread geom-flops/s rate and
    /// `parallel_speedup` the achieved speedup at the chosen worker count (from the
    /// calibration). Unlike [`est_runtime_ms`](Self::est_runtime_ms) this does not
    /// let more threads drive the estimate below the tree's serial dependency, so
    /// argmin over the worker count correctly stops adding threads once the
    /// critical path (or, in the full v2 model, memory bandwidth) dominates.
    pub fn est_runtime_ms_threaded(&self, gflops: f64, parallel_speedup: f64) -> f64 {
        let rate1 = gflops.max(RATE_FLOOR) * GIGA; // one-thread geom-flops/s
        let serial_floor = self.critical_path_flops as f64 / rate1;
        let parallel = self.factor_flops as f64 / (rate1 * parallel_speedup.max(RATE_FLOOR));
        serial_floor.max(parallel) * MS_PER_S
    }
}

impl fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "peak <= {:.0} MB (factor {:.0} MB, ~{} nnz)",
            self.transient_peak_bytes as f64 / MEGA,
            self.factor_bytes as f64 / MEGA,
            self.factor_nnz,
        )
    }
}

/// The choices the solver made on its own for one factorization: what the
/// `Auto` settings resolved to and what the structure looked like. Plain
/// strings (the `Debug` form of the enums) so a host can display or serialise
/// them without the enum types.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Decisions {
    /// The ordering the caller asked for (`Auto` included).
    pub ordering_requested: String,
    /// The ordering actually dispatched after `Auto` resolution.
    pub ordering_used: String,
    /// The supernode amalgamation strategy actually used.
    pub amalgamation: String,
    /// The equilibration applied before factoring (the symmetric path); the
    /// unsymmetric paths name their built-in scaling.
    pub scaling: String,
    /// The numeric kernel (`LeftLooking`, `Klu`).
    pub method: String,
    pub n_supernodes: usize,
    /// Largest front (rows) after amalgamation.
    pub max_front: usize,
    /// Depth of the assembly tree.
    pub tree_levels: usize,
    /// KLU: number of BTF blocks (0 when BTF is off or not a KLU factor).
    pub btf_blocks: usize,
}

/// What the numeric phase did to the pivots.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NumericReport {
    /// Pivots lifted by the static-pivoting floor (`n_perturbed`).
    pub perturbed: usize,
    /// Bunch-Kaufman 2x2 pivots (`None` for the LU paths).
    pub two_by_two: Option<usize>,
    /// Pivots a front of the supernodal LU passed to its parent: a column
    /// none of whose fully-summed rows passes the threshold over the whole
    /// column (one delayed twice counts twice). `None` for the other paths.
    pub delayed: Option<usize>,
    /// Pivots of the supernodal LU that failed that threshold and stayed in
    /// their front (a root, or the delay allowance spent): the pivot growth
    /// the threshold bounds is not bounded there. `None` for the other paths.
    pub forced: Option<usize>,
    /// Inertia `(positive, negative, zero)` of a symmetric factor.
    pub inertia: Option<(usize, usize, usize)>,
    /// The smallest pivot magnitude of the factor, on the equilibrated
    /// matrix the static floor is compared with: the 1x1 pivots and the
    /// smallest singular value of each 2x2 block. A lifted pivot shows as
    /// the floor. `None` without a factor.
    pub min_pivot: Option<f64>,
}

/// Solve-phase accumulators (updated by every `solve*` call on the factor).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SolveStats {
    /// Right-hand sides solved (a `solve_many` counts each column).
    pub rhs: usize,
    /// `solve*` calls.
    pub calls: usize,
    /// Iterative-refinement steps taken over all calls.
    pub refine_steps: usize,
}

impl SolveStats {
    pub fn record(&mut self, rhs: usize, refine_steps: usize) {
        self.rhs += rhs;
        self.calls += 1;
        self.refine_steps += refine_steps;
    }
}

/// [`SolveStats`] behind a mutex, so a factor handle records its solves
/// through `&self`; cloning snapshots the counters (a cloned handle starts a
/// separate account).
#[derive(Debug, Default)]
pub(crate) struct SolveCounter(std::sync::Mutex<SolveStats>);

impl SolveCounter {
    pub fn record(&self, rhs: usize, refine_steps: usize) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(rhs, refine_steps);
    }
    pub fn snapshot(&self) -> SolveStats {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Clone for SolveCounter {
    fn clone(&self) -> Self {
        SolveCounter(std::sync::Mutex::new(self.snapshot()))
    }
}

/// [`MemoryEstimate`]s of one analysis, keyed by scalar size and complexity.
/// The estimate is a pure function of the structure and the scalar type, but
/// computing it walks every node's updaters, expensive enough to pay only
/// once per scalar type.
#[derive(Debug, Default)]
pub(crate) struct EstimateCache(std::sync::Mutex<Vec<((usize, bool), MemoryEstimate)>>);

impl EstimateCache {
    pub fn get<T: crate::scalar::Scalar>(
        &self,
        estimate: impl FnOnce() -> MemoryEstimate,
    ) -> MemoryEstimate {
        let key = (std::mem::size_of::<T>(), T::COMPLEX);
        if let Ok(cache) = self.0.lock() {
            if let Some(&(_, est)) = cache.iter().find(|&&(k, _)| k == key) {
                return est;
            }
        }
        let est = estimate();
        if let Ok(mut cache) = self.0.lock() {
            if !cache.iter().any(|&(k, _)| k == key) {
                cache.push((key, est));
            }
        }
        est
    }
}

/// Everything one factorization can tell about itself: the decisions taken,
/// the numeric outcome, the settings that had no effect on the chosen path,
/// and the solve-phase accumulators. Per-call and concurrency-safe (no global
/// state), so a solver-in-the-loop with many concurrent solves gets correct
/// per-solve numbers. Carries the a-priori [`MemoryEstimate`] next to the
/// factor's actual size. Emitted as one DEBUG event per factorization (see
/// [`summary`](Self::summary)) and readable from the factor handle.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub threads: usize,
    pub n: usize,
    /// Stored nonzeros of the input pattern.
    pub nnz_a: u64,
    pub factor_nnz: u64,
    /// Heap bytes of the numeric factor: its values and indices.
    pub factor_bytes: u64,
    pub estimate: Option<MemoryEstimate>,
    pub decisions: Decisions,
    pub numeric: NumericReport,
    /// Settings the caller set to a non-default value that the chosen factor
    /// path does not read (also emitted as WARN events).
    pub warnings: Vec<String>,
    pub solves: SolveStats,
}

impl Diagnostics {
    /// `nnz(L) / nnz(A)` (`0` without an input count).
    pub fn fill_ratio(&self) -> f64 {
        if self.nnz_a == 0 {
            0.0
        } else {
            self.factor_nnz as f64 / self.nnz_a as f64
        }
    }
    /// The counts of this factorization as measures at the current scope,
    /// and its [`summary`](Self::summary) as a DEBUG event there: a caller
    /// that factors per step gets one per factorization.
    pub(crate) fn probe(&self) {
        rsprobe::add!("fill", self.factor_nnz);
        if let Some(e) = &self.estimate {
            rsprobe::add!("flops", e.factor_flops);
        }
        rsprobe::add!("perturbed", self.numeric.perturbed);
        if let Some(k) = self.numeric.two_by_two {
            rsprobe::add!("two_by_two", k);
        }
        if let Some(k) = self.numeric.delayed {
            rsprobe::add!("delayed", k);
        }
        if let Some(k) = self.numeric.forced {
            rsprobe::add!("forced", k);
        }
        for w in &self.warnings {
            rsprobe::warn!("settings: {w}");
        }
        rsprobe::debug!("{}", self.summary());
    }

    /// The one-line account of the DEBUG event per factorization.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} n={} nnz(A)={} nnz(L)={} fill={:.2} threads={} ordering={}",
            self.decisions.method,
            self.n,
            self.nnz_a,
            self.factor_nnz,
            self.fill_ratio(),
            self.threads,
            self.decisions.ordering_used,
        );
        if self.decisions.ordering_requested != self.decisions.ordering_used
            && !self.decisions.ordering_requested.is_empty()
        {
            s.push_str(&format!(
                " (requested {})",
                self.decisions.ordering_requested
            ));
        }
        if self.decisions.n_supernodes > 0 {
            s.push_str(&format!(
                " supernodes={} max_front={} levels={}",
                self.decisions.n_supernodes, self.decisions.max_front, self.decisions.tree_levels
            ));
        }
        if self.decisions.btf_blocks > 0 {
            s.push_str(&format!(" btf_blocks={}", self.decisions.btf_blocks));
        }
        if self.numeric.perturbed > 0 {
            s.push_str(&format!(" perturbed={}", self.numeric.perturbed));
        }
        if let Some(k) = self.numeric.two_by_two {
            s.push_str(&format!(" pivots2x2={k}"));
        }
        if let Some(k) = self.numeric.delayed.filter(|&k| k > 0) {
            s.push_str(&format!(" delayed={k}"));
        }
        if let Some(k) = self.numeric.forced.filter(|&k| k > 0) {
            s.push_str(&format!(" forced={k}"));
        }
        if let Some(p) = self.numeric.min_pivot {
            s.push_str(&format!(" min_pivot={p:.1e}"));
        }
        if let Some(e) = &self.estimate {
            s.push_str(&format!(" est_peak={:.0}MB", e.transient_peak_mb()));
        }
        s
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "factorization diagnostics: {}", self.summary())?;
        writeln!(
            f,
            "  decisions: amalgamation={} scaling={}",
            self.decisions.amalgamation, self.decisions.scaling
        )?;
        if let Some((p, n, z)) = self.numeric.inertia {
            writeln!(f, "  inertia: +{p} -{n} 0:{z}")?;
        }
        writeln!(f, "  factor: {:.1} MB", self.factor_bytes as f64 / MEGA)?;
        if self.solves.calls > 0 {
            writeln!(
                f,
                "  solves: {} calls, {} rhs, {} refinement steps",
                self.solves.calls, self.solves.rhs, self.solves.refine_steps
            )?;
        }
        for w in &self.warnings {
            writeln!(f, "  warning: {w}")?;
        }
        Ok(())
    }
}
