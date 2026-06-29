//! Process-wide GEMM scheduling thresholds - the kernel-level parallelism levers.
//!
//! The factorization kernels switch between a scalar triple loop and a SIMD
//! GEMM, and between a serial and a rayon-parallel GEMM, at fixed flop-count
//! thresholds. These decide *where* node-parallelism kicks in - in particular,
//! near the assembly-tree root, where tree-parallelism has dried up and a few
//! huge fronts must pick up idle cores (the "top-of-tree" parallelism lever).
//!
//! They are exposed as a **process-wide** tunable, mirroring the existing
//! [`set_use_gemm_schur`](crate::set_use_gemm_schur) kernel A/B knob, not as
//! per-call [`FactorOptions`](crate::FactorOptions): they are micro-scheduling
//! knobs for the auto-tuning sweep and benchmarks, read on the hot path as a
//! cheap relaxed atomic load. The defaults reproduce the historically-tuned
//! behaviour, so leaving them untouched changes nothing. A reasonable predictor
//! sets them once per process before factoring.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Default: below `4096` flops a contribution update is a scalar triple loop.
pub const DEFAULT_SCALAR_GATE: usize = 4096;
/// Default: at/above `1_000_000` flops a cmod-class GEMM goes rayon-parallel.
pub const DEFAULT_PAR_GEMM: usize = 1_000_000;
/// Default: at/above `8_000_000` flops a cdiv / Schur / LU-front GEMM goes
/// rayon-parallel.
pub const DEFAULT_PAR_CDIV: usize = 8_000_000;

static SCALAR_GATE: AtomicUsize = AtomicUsize::new(DEFAULT_SCALAR_GATE);
static PAR_GEMM: AtomicUsize = AtomicUsize::new(DEFAULT_PAR_GEMM);
static PAR_CDIV: AtomicUsize = AtomicUsize::new(DEFAULT_PAR_CDIV);

/// Scalar-vs-GEMM cutoff (flops) for contribution updates.
#[inline]
pub(crate) fn scalar_gate() -> usize {
    SCALAR_GATE.load(Ordering::Relaxed)
}

/// Serial-vs-parallel cutoff (flops) for cmod-class GEMM updates.
#[inline]
pub(crate) fn par_gemm() -> usize {
    PAR_GEMM.load(Ordering::Relaxed)
}

/// Serial-vs-parallel cutoff (flops) for cdiv / Schur / LU-front trailing GEMM.
#[inline]
pub(crate) fn par_cdiv() -> usize {
    PAR_CDIV.load(Ordering::Relaxed)
}

/// The three GEMM scheduling thresholds (flop counts), the kernel-level
/// parallelism levers. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmThresholds {
    /// Below this flop count (`rows * cols * k`) a contribution update runs as a
    /// scalar triple loop instead of a SIMD GEMM.
    pub scalar_gate: usize,
    /// At/above this flop count a cmod-class GEMM runs rayon-parallel.
    pub par_gemm: usize,
    /// At/above this flop count the panel-trailing / Schur / LU-front GEMM runs
    /// rayon-parallel. Lowering it makes large fronts near the tree root engage
    /// node-parallelism earlier (the top-of-tree lever).
    pub par_cdiv: usize,
}

impl Default for GemmThresholds {
    fn default() -> Self {
        Self {
            scalar_gate: DEFAULT_SCALAR_GATE,
            par_gemm: DEFAULT_PAR_GEMM,
            par_cdiv: DEFAULT_PAR_CDIV,
        }
    }
}

/// Set the process-wide GEMM scheduling thresholds (auto-tuning / benchmarks).
/// Affects every subsequent factorization in the process. The numeric result is
/// unchanged - only the serial/parallel and scalar/GEMM kernel selection moves.
pub fn set_gemm_thresholds(t: GemmThresholds) {
    SCALAR_GATE.store(t.scalar_gate, Ordering::Relaxed);
    PAR_GEMM.store(t.par_gemm, Ordering::Relaxed);
    PAR_CDIV.store(t.par_cdiv, Ordering::Relaxed);
}

/// The current process-wide GEMM scheduling thresholds.
pub fn gemm_thresholds() -> GemmThresholds {
    GemmThresholds {
        scalar_gate: scalar_gate(),
        par_gemm: par_gemm(),
        par_cdiv: par_cdiv(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CscMatrix, FactorOptions, LdltSolver};
    use num_complex::Complex;

    /// Moving the thresholds changes only the serial/parallel and scalar/GEMM
    /// kernel selection - never the answer. Factor the same matrix under
    /// all-scalar/serial, all-GEMM/parallel, and the default, and confirm the
    /// solutions agree to working precision.
    #[test]
    fn thresholds_do_not_change_the_result() {
        let c = |re, im| Complex::new(re, im);
        let m = 10;
        let n = m * m;
        let idx = |r: usize, cc: usize| r * m + cc;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 0.5));
                for (dr, dc) in [(1usize, 0usize), (0, 1)] {
                    if r + dr < m && cc + dc < m {
                        let q = idx(r + dr, cc + dc);
                        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                        rows.push(hi);
                        cols.push(lo);
                        vals.push(c(-1.0, 0.1));
                    }
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 50.0, 1.0)).collect();
        let solve = || {
            LdltSolver::factor_with(&a, &FactorOptions::default().with_threads(0))
                .unwrap()
                .solve(&b)
                .unwrap()
        };

        set_gemm_thresholds(GemmThresholds::default());
        let x_def = solve();
        // Force every update through the scalar / serial path.
        set_gemm_thresholds(GemmThresholds {
            scalar_gate: usize::MAX,
            par_gemm: usize::MAX,
            par_cdiv: usize::MAX,
        });
        let x_scalar = solve();
        // Force every update through the parallel GEMM path.
        set_gemm_thresholds(GemmThresholds {
            scalar_gate: 0,
            par_gemm: 0,
            par_cdiv: 0,
        });
        let x_par = solve();
        set_gemm_thresholds(GemmThresholds::default());

        for i in 0..n {
            assert!((x_def[i] - x_scalar[i]).norm() < 1e-9, "scalar path diverged at {i}");
            assert!((x_def[i] - x_par[i]).norm() < 1e-9, "parallel path diverged at {i}");
        }
    }
}
