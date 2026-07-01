//! Runtime tuner profile (issue #1): applying the default profile must be a
//! semantic no-op — the recommendation before and after `apply_profile` is
//! identical, because `default_profile()` bundles the embedded models with the
//! default guards. This lives in its own test binary because `apply_profile`
//! mutates a process-global (set-once), which would perturb the in-crate tuner
//! tests if run in the same process.

use rslab::{
    apply_profile, default_profile, recommend_settings_pathed, SolverPath, StructuralFeatures,
    DEFAULT_TUNE_WEIGHT,
};

#[test]
fn applying_default_profile_is_a_no_op() {
    let f = StructuralFeatures::default();
    // Recommendation with the compile-time-embedded models (no profile active).
    let before_ldlt = recommend_settings_pathed(&f, DEFAULT_TUNE_WEIGHT, 0.0, SolverPath::Ldlt);
    let before_lu = recommend_settings_pathed(&f, DEFAULT_TUNE_WEIGHT, 0.0, SolverPath::Lu);

    // Apply the default profile: same models, same guards, delivered as a config
    // artifact rather than a recompile.
    apply_profile(&default_profile()).expect("apply default profile");

    let after_ldlt = recommend_settings_pathed(&f, DEFAULT_TUNE_WEIGHT, 0.0, SolverPath::Ldlt);
    let after_lu = recommend_settings_pathed(&f, DEFAULT_TUNE_WEIGHT, 0.0, SolverPath::Lu);

    // The profile pathway must reproduce the embedded pathway bit for bit.
    assert_eq!(before_ldlt.ordering, after_ldlt.ordering);
    assert_eq!(before_ldlt.method, after_ldlt.method);
    assert_eq!(before_ldlt.nemin, after_ldlt.nemin);
    assert_eq!(before_ldlt.panel_nb, after_ldlt.panel_nb);
    assert_eq!(before_lu.ordering, after_lu.ordering);
    assert_eq!(before_lu.method, after_lu.method);
    assert_eq!(before_lu.pivot_u, after_lu.pivot_u);

    // A second apply is rejected (set-once semantics).
    assert!(apply_profile(&default_profile()).is_err());
}
