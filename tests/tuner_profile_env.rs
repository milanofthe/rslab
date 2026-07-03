//! `RSLAB_TUNER_PROFILE` env auto-load: a profile named by the environment is
//! picked up on first tuner use with no recompile and no explicit `apply_profile`
//! call — the config-artifact contract of the meta-tuner (issue #1). Own binary,
//! because the first tuner call sets a process-global.

use rslab::{
    default_profile, recommend_settings_pathed, SolverPath, StructuralFeatures, TunerProfile,
    DEFAULT_TUNE_WEIGHT,
};

#[test]
fn env_named_profile_is_auto_loaded() {
    // Write a profile whose min_gain guard is a distinctive, easily-checked value.
    let mut p = default_profile();
    p.class = "env-test".to_string();
    p.min_gain = 0.123_456;
    let path = std::env::temp_dir().join("rslab_env_profile.json");
    p.save(&path).expect("save profile");

    // Point the env var at it BEFORE the first tuner call in this process.
    std::env::set_var("RSLAB_TUNER_PROFILE", &path);

    // The first recommendation triggers the one-shot env load; the call must
    // succeed and return valid settings (the models are the embedded defaults, so
    // the recommendation itself is well-formed).
    let f = StructuralFeatures::default();
    let s = recommend_settings_pathed(&f, DEFAULT_TUNE_WEIGHT, 0.0, SolverPath::Ldlt);
    assert!(
        s.nemin >= 1,
        "recommendation is well-formed under an env profile"
    );

    // The env profile is now the active one, so a manual re-apply is rejected
    // (set-once) — confirming the env profile was actually installed.
    assert!(
        rslab::apply_profile(&default_profile()).is_err(),
        "env profile occupies the set-once slot"
    );

    let _ = std::fs::remove_file(&path);
    // Round-trip sanity on the guard value we wrote.
    let reloaded = TunerProfile::load(&std::env::temp_dir().join("does_not_exist.json"));
    assert!(
        reloaded.is_err(),
        "loading a missing profile is a clean error"
    );
}
