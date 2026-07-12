//! Do the settings pickers' memory backstops actually hold? The Pareto sweep
//! showed some matrices (banded + a switched ordering) measuring more fill/memory
//! than the default. This checks both real paths - the heuristic default
//! (`LdltSolver::tuned` / `LuSolver::tuned`, whose ND bakeoff must veto a
//! fill-growing MetisND pick) and the optional ML tuner (`tuned_model`) - the
//! picked factor's fill must not exceed the untuned default's beyond the
//! backstop's 2% tolerance.
//!
//! Needs the `matgen` test-matrix generators, so the whole test is gated on that
//! feature - under the default build set it compiles to nothing (run it with
//! `cargo test --features matgen`).
#![cfg(feature = "matgen")]

use num_complex::Complex;
use rslab::{
    matgen, LdltSolver, LdltSymbolic, LuSolver, LuSymbolic, SolverSettings, DEFAULT_TUNE_WEIGHT,
};

type C = Complex<f64>;

#[test]
fn ldlt_tuner_never_grows_fill_over_default() {
    // Banded is exactly the class the sweep flagged: a band ordering (AMD/default)
    // is optimal, and nested dissection (METIS) blows up the fill ~2x. The tuner
    // must not pick it — the fill backstop should veto it back to the default.
    for &(n, bw) in &[(20000usize, 30), (40000, 40)] {
        let a = matgen::structured::banded::<C>(n, bw, 1.0, 1);
        let default_fill = {
            let sym = LdltSymbolic::analyze(&a).expect("analyze");
            sym.factor(&a, &SolverSettings::default())
                .expect("factor")
                .factor_nnz()
        };
        let (sym, s) = LdltSolver::<C>::tuned_model(&a, DEFAULT_TUNE_WEIGHT).expect("tuned");
        let tuned_fill = sym.factor(&a, &s).expect("tuned factor").factor_nnz();
        assert!(
            tuned_fill as f64 <= default_fill as f64 * 1.02,
            "banded_{n}_{bw}: ML-tuned fill {tuned_fill} > 1.02x default {default_fill} \
             (ordering {:?}) — memory backstop breached",
            s.ordering
        );
        // The heuristic default path: its ND bakeoff requires fill_ok, so a
        // banded matrix (where MetisND over-separates) must keep the default.
        let (sym_h, s_h) = LdltSolver::<C>::tuned(&a).expect("heuristic tuned");
        let heur_fill = sym_h
            .factor(&a, &s_h)
            .expect("heuristic factor")
            .factor_nnz();
        assert!(
            heur_fill as f64 <= default_fill as f64 * 1.02,
            "banded_{n}_{bw}: heuristic fill {heur_fill} > 1.02x default {default_fill} \
             (ordering {:?})",
            s_h.ordering
        );
    }
}

#[test]
fn lu_tuner_never_grows_fill_over_default() {
    // The LU-path analogue on a convection-diffusion matrix.
    let a = matgen::fem::convection_diffusion::<C>(
        &[140, 140],
        0.01,
        matgen::fem::Flow::Rotating,
        true,
    );
    let default_fill = {
        let sym = LuSymbolic::analyze(&a).expect("analyze");
        sym.factor(&a, &SolverSettings::default())
            .expect("factor")
            .factor_nnz()
    };
    let (sym, s) = LuSolver::<C>::tuned_model(&a, DEFAULT_TUNE_WEIGHT).expect("tuned");
    let tuned_fill = sym.factor(&a, &s).expect("tuned factor").factor_nnz();
    assert!(
        tuned_fill as f64 <= default_fill as f64 * 1.02,
        "convdiff: ML-tuned fill {tuned_fill} > 1.02x default {default_fill} (ordering {:?})",
        s.ordering
    );
    let (sym_h, s_h) = LuSolver::<C>::tuned(&a).expect("heuristic tuned");
    let heur_fill = sym_h
        .factor(&a, &s_h)
        .expect("heuristic factor")
        .factor_nnz();
    assert!(
        heur_fill as f64 <= default_fill as f64 * 1.02,
        "convdiff: heuristic fill {heur_fill} > 1.02x default {default_fill} (ordering {:?})",
        s_h.ordering
    );
}
