//! Deterministic resource diagnostics: an **a-priori** peak-memory estimate
//! (computed from the symbolic factorization, before any numeric work) and a
//! per-stage runtime/memory report collected during factorization.
//!
//! The estimate is a pure function of the analyzed structure, so it is fully
//! reproducible and lets a solver-in-the-loop scheduler decide *before* allocating
//! whether a factorization fits the memory budget (fail-fast / pick approximation).

use std::fmt;

/// A-priori estimate of the memory a factorization will use, in bytes. All fields
/// are deterministic functions of the symbolic structure and the scalar size.
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    /// Scalar size in bytes (`16` for `Complex<f64>`, `8` for `f64`, …).
    pub value_bytes: usize,
    /// Structural nonzeros in the factor (`L`+`U` for LU, `L` for LDLᵀ) — an upper
    /// bound on the emitted factor (numeric cancellation can only lower it).
    pub factor_nnz: u64,
    /// Bytes of the resident factor (the CSC output): `factor_nnz·(value+index)`.
    pub factor_bytes: u64,
    /// Dense supernode panels if **all** were held at once (the naive left-looking
    /// peak, i.e. without panel-freeing).
    pub panels_all_bytes: u64,
    /// Peak of the **live** dense panels under the refcount free-schedule — what
    /// the left-looking path actually holds at once.
    pub panel_live_peak_bytes: u64,
    /// Estimated overall transient peak: live panels + accumulated compact factor +
    /// the equilibrated input copy/copies. The number to compare against RAM.
    pub transient_peak_bytes: u64,
}

impl MemoryEstimate {
    pub fn transient_peak_mb(&self) -> f64 {
        self.transient_peak_bytes as f64 / 1e6
    }
    pub fn factor_mb(&self) -> f64 {
        self.factor_bytes as f64 / 1e6
    }
    /// Does the estimated transient peak fit in `available` bytes?
    pub fn fits_in(&self, available_bytes: u64) -> bool {
        self.transient_peak_bytes <= available_bytes
    }
}

impl fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "transient-peak ≤ {:.0} MB (panels {:.0} + factor {:.0} + input/scratch); \
             factor ~{} nnz; panel-freed floor {:.0} MB",
            self.transient_peak_bytes as f64 / 1e6,
            self.panels_all_bytes as f64 / 1e6,
            self.factor_bytes as f64 / 1e6,
            self.factor_nnz,
            self.panel_live_peak_bytes as f64 / 1e6,
        )
    }
}

/// Core left-looking memory estimator. `panel_bytes(s)` is supernode `s`'s dense
/// panel size; `compact_bytes(s)` its CSC-fragment size; `update_list[s]` its
/// factored descendants (consumers). Simulates the refcount free-schedule in
/// elimination/postorder (supernodes are numbered in postorder) to get the live
/// panel peak and the accumulating compact factor — the same schedule the numeric
/// path runs, so the estimate matches what it allocates.
pub(crate) fn estimate_left_looking(
    nsuper: usize,
    panel_bytes: &dyn Fn(usize) -> u64,
    compact_bytes: &dyn Fn(usize) -> u64,
    update_list: &[Vec<usize>],
    value_bytes: usize,
    input_bytes: u64,
) -> MemoryEstimate {
    let mut refc = vec![0usize; nsuper];
    for ul in update_list {
        for &k in ul {
            refc[k] += 1;
        }
    }
    let panels_all: u64 = (0..nsuper).map(panel_bytes).sum();
    let factor_bytes: u64 = (0..nsuper).map(compact_bytes).sum();

    let mut live_panels: i64 = 0;
    let mut compact: i64 = 0;
    let mut peak: i64 = 0;
    for s in 0..nsuper {
        live_panels += panel_bytes(s) as i64;
        for &k in &update_list[s] {
            refc[k] -= 1;
            if refc[k] == 0 {
                live_panels -= panel_bytes(k) as i64;
                compact += compact_bytes(k) as i64;
            }
        }
        if refc[s] == 0 {
            live_panels -= panel_bytes(s) as i64;
            compact += compact_bytes(s) as i64;
        }
        peak = peak.max(live_panels + compact);
    }
    let panel_live_peak = peak.max(0) as u64;
    // Conservative transient upper bound. At many threads the parallel frontier of
    // a top-heavy tree holds nearly all panels at once, and the emit builds the full
    // factor CSC on top — so the safe estimate is all-resident panels + the factor +
    // the input copies + a per-thread scratch margin (cmod/cdiv buffers, gloc). This
    // is the number to compare against RAM for a fail-fast / scheduling decision; the
    // panel-freeing path makes the *actual* peak lower (down to `panel_live_peak`),
    // so this never under-predicts.
    // Per-thread scratch (cmod/cdiv buffers, gloc, the emit double-buffer) plus a
    // small absolute floor — tuned so the bound stays ≥ the measured peak across
    // sizes (validated: est/measured ≈ 1.0–1.2×), never under-predicting.
    let scratch = (panels_all + factor_bytes) / 4 + 32_000_000;
    MemoryEstimate {
        value_bytes,
        factor_nnz: factor_bytes / (value_bytes as u64 + 8).max(1),
        factor_bytes,
        panels_all_bytes: panels_all,
        panel_live_peak_bytes: panel_live_peak,
        transient_peak_bytes: panels_all + factor_bytes + input_bytes + scratch,
    }
}

// ---------------------------------------------------------------------------
// Per-stage runtime/memory report, collected during a factorization.
// ---------------------------------------------------------------------------

/// One factorization stage's cost. `flops`/`bytes` are deterministic (structural);
/// `wall_ms` is observability (varies with load/threads).
#[derive(Debug, Clone)]
pub struct StageReport {
    pub name: &'static str,
    pub wall_ms: f64,
    pub flops: u64,
    pub bytes: u64,
}

/// Per-stage diagnostics for one factorization. Per-call and concurrency-safe (no
/// global state), so a solver-in-the-loop with many concurrent solves gets correct
/// per-solve numbers. Carries the a-priori [`MemoryEstimate`] alongside the
/// measured factor time for estimate-vs-actual feedback.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub stages: Vec<StageReport>,
    pub threads: usize,
    pub factor_nnz: u64,
    pub estimate: Option<MemoryEstimate>,
}

impl Diagnostics {
    pub fn total_ms(&self) -> f64 {
        self.stages.iter().map(|s| s.wall_ms).sum()
    }
    pub fn push(&mut self, name: &'static str, wall_ms: f64, flops: u64, bytes: u64) {
        self.stages.push(StageReport { name, wall_ms, flops, bytes });
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "factorization diagnostics (threads={}, factor_nnz={}):", self.threads, self.factor_nnz)?;
        let tot = self.total_ms().max(1e-9);
        for s in &self.stages {
            writeln!(
                f,
                "  {:<10} {:8.1} ms ({:4.0}%)  {:>10} Mflop  {:>8.0} MB",
                s.name,
                s.wall_ms,
                100.0 * s.wall_ms / tot,
                s.flops / 1_000_000,
                s.bytes as f64 / 1e6,
            )?;
        }
        Ok(())
    }
}
