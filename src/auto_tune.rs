//! Auto-tuner: predict the knob config that minimizes a weighted Pareto score
//! `w·log(time) + (1-w)·log(mem)` for a matrix, from its structural features alone.
//!
//! A small MLP performance model `(features ⊕ knobs) → (log factor_ms, log peak_mb)`
//! is trained offline (`benches/train_tuner.py`) on the corpus sweep and embedded
//! here as JSON; inference is **pure Rust** (a few dense layers). At tune time the
//! model scores a candidate grid and returns the best config as [`SolverSettings`].
//! The trained model held out ~10% time / ~8% mem regret vs the oracle on unseen
//! matrices (at the default `w = 0.7`).

use serde::Deserialize;
use std::sync::OnceLock;

use crate::analysis::StructuralFeatures;
use crate::numeric::gemm_tuning::GemmThresholds;
use crate::{FactorMethod, OrderingMethod, SolverSettings};

/// Default Pareto weight: 0.7 toward speed, 0.3 toward memory.
pub const DEFAULT_TUNE_WEIGHT: f64 = 0.7;

#[derive(Deserialize)]
struct InputComp {
    kind: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    log: bool,
    #[serde(default)]
    mean: f64,
    #[serde(default)]
    std: f64,
    #[serde(default)]
    value: String,
}

#[derive(Deserialize)]
struct Layer {
    /// Row-major `(out, in)` weights.
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Deserialize)]
struct Model {
    input_spec: Vec<InputComp>,
    layers: Vec<Layer>,
    target_mean: Vec<f64>,
    target_std: Vec<f64>,
    /// Out-of-distribution flop ceiling: above the largest well-sampled training
    /// matrix the model extrapolates on the knobs, so the recommendation falls back
    /// to the default. `0` disables the guard.
    #[serde(default)]
    flops_ood_cap: f64,
}

// One model per solver path: the symmetric LDLᵀ and the unsymmetric LU paths have
// different relevant axes (e.g. `pivot_u` only affects LU) and different
// speed/memory profiles, so each is fit and selected independently.
const MODEL_LDLT_JSON: &str = include_str!("auto_tune_model_ldlt.json");
const MODEL_LU_JSON: &str = include_str!("auto_tune_model_lu.json");

/// Which factorization path the tuner selects a configuration for. Each path has
/// its own trained model and its own candidate grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolverPath {
    /// Symmetric Bunch-Kaufman LDLᵀ.
    Ldlt,
    /// Unsymmetric threshold-pivoted LU.
    Lu,
}

fn model_for(path: SolverPath) -> Option<&'static Model> {
    static LDLT: OnceLock<Option<Model>> = OnceLock::new();
    static LU: OnceLock<Option<Model>> = OnceLock::new();
    match path {
        SolverPath::Ldlt => LDLT
            .get_or_init(|| serde_json::from_str(MODEL_LDLT_JSON).ok())
            .as_ref(),
        SolverPath::Lu => LU
            .get_or_init(|| serde_json::from_str(MODEL_LU_JSON).ok())
            .as_ref(),
    }
}

/// Copy proxy for the (non-`Copy`) [`ScalingStrategy`](crate::ScalingStrategy) so
/// [`Candidate`] stays `Copy`; only the value-independent strategies are tuned.
#[derive(Clone, Copy, PartialEq)]
enum ScalingKnob {
    OnePass,
    Identity,
    InfNorm,
    Auto,
}
impl ScalingKnob {
    fn to_strategy(self) -> crate::ScalingStrategy {
        match self {
            ScalingKnob::OnePass => crate::ScalingStrategy::OnePassInfNorm,
            ScalingKnob::Identity => crate::ScalingStrategy::Identity,
            ScalingKnob::InfNorm => crate::ScalingStrategy::InfNorm,
            ScalingKnob::Auto => crate::ScalingStrategy::Auto,
        }
    }
    fn name(self) -> &'static str {
        match self {
            ScalingKnob::OnePass => "onepass",
            ScalingKnob::Identity => "identity",
            ScalingKnob::InfNorm => "infnorm",
            ScalingKnob::Auto => "auto",
        }
    }
}

/// A candidate knob config the tuner scores. Mirrors the sweep's tunable knobs
/// (the worker count is left to the thread predictor).
#[derive(Clone, Copy)]
struct Candidate {
    ordering: OrderingMethod,
    nemin: usize,
    relax_width: usize, // 0 = relaxed amalgamation off
    panel_nb: usize,
    scalar_gate: usize,
    par_gemm: usize,
    par_cdiv: usize,
    use_gemm_schur: bool,
    method: FactorMethod,
    // Exact-path tuning axes (issue #2). Only varied when the loaded model tunes
    // them (its input_spec references the axis); otherwise they stay at BASE, so a
    // model that predates these axes yields a bit-identical candidate set.
    pivot_u: f64,
    scaling: ScalingKnob,
    memory_eager: bool,
}

const BASE: Candidate = Candidate {
    ordering: OrderingMethod::Auto,
    nemin: 16,
    relax_width: 256,
    panel_nb: 64,
    scalar_gate: 4096,
    par_gemm: 1_000_000,
    par_cdiv: 8_000_000,
    use_gemm_schur: true,
    method: FactorMethod::LeftLooking,
    pivot_u: 0.1,
    scaling: ScalingKnob::OnePass,
    memory_eager: false,
};

/// Structural-feature lookup by the name used in the model's input spec.
fn feat_value(f: &StructuralFeatures, name: &str) -> f64 {
    match name {
        "n" => f.n as f64,
        "nnz" => f.nnz as f64,
        "deg_mean" => f.deg_mean,
        "deg_max" => f.deg_max as f64,
        "deg_cv" => f.deg_cv,
        "bandwidth_max" => f.bandwidth_max as f64,
        "bandwidth_mean_rel" => f.bandwidth_mean_rel,
        "diag_dominant_frac" => f.diag_dominant_frac,
        "diag_present_frac" => f.diag_present_frac,
        "n_supernodes" => f.n_supernodes as f64,
        "fill_nnz" => f.fill_nnz as f64,
        "fill_ratio" => f.fill_ratio,
        "supernode_cols_mean" => f.supernode_cols_mean,
        "front_nrow_max" => f.front_nrow_max as f64,
        "tree_depth" => f.tree_depth as f64,
        "tree_width_max" => f.tree_width_max as f64,
        "tree_width_mean" => f.tree_width_mean,
        "factor_flops" => f.factor_flops as f64,
        "arith_intensity" => f.arith_intensity,
        "flop_top1_frac" => f.flop_top1_frac,
        "flop_top1pct_frac" => f.flop_top1pct_frac,
        _ => 0.0,
    }
}

fn knob_value(c: &Candidate, name: &str) -> f64 {
    match name {
        "nemin" => c.nemin as f64,
        "relax_width" => c.relax_width as f64,
        "panel_nb" => c.panel_nb as f64,
        "scalar_gate" => c.scalar_gate as f64,
        "par_gemm" => c.par_gemm as f64,
        "par_cdiv" => c.par_cdiv as f64,
        _ => 0.0,
    }
}

fn ordering_name(o: OrderingMethod) -> &'static str {
    // Real per-variant names so the one-hot encoding matches the trainer for any
    // swept ordering. The model's `orderings` one-hot only lists a subset; a name
    // outside it encodes as all-zero (matching Python), so an ordering the tuner
    // never proposes still round-trips correctly through the parity check.
    match o {
        OrderingMethod::Amd => "amd",
        OrderingMethod::Amf => "amf",
        OrderingMethod::MetisND => "metis",
        OrderingMethod::ScotchND => "scotch",
        OrderingMethod::KahipND => "kahip",
        OrderingMethod::Rcm => "rcm",
        OrderingMethod::Auto => "auto",
        OrderingMethod::AutoRace => "auto_race",
    }
}

/// Build the standardized input vector per the model's spec (identical layout to
/// the Python training export).
fn build_input(m: &Model, f: &StructuralFeatures, c: &Candidate) -> Vec<f64> {
    m.input_spec
        .iter()
        .map(|comp| match comp.kind.as_str() {
            "feat" => {
                let v = feat_value(f, &comp.name);
                let v = if comp.log { v.ln_1p() } else { v };
                (v - comp.mean) / comp.std
            }
            "knob_log" => {
                let v = knob_value(c, &comp.name).ln_1p();
                (v - comp.mean) / comp.std
            }
            "ordering_onehot" => (ordering_name(c.ordering) == comp.value) as i32 as f64,
            "method_is_mf" => (c.method == FactorMethod::Multifrontal) as i32 as f64,
            "use_gemm_schur" => c.use_gemm_schur as i32 as f64,
            // Exact-path tuning axes (issue #2): threshold pivot `u` (linear in
            // [0,1]), equilibration strategy (one-hot), and the emit/memory mode.
            "pivot_u" => (c.pivot_u - comp.mean) / if comp.std != 0.0 { comp.std } else { 1.0 },
            "scaling_onehot" => (c.scaling.name() == comp.value) as i32 as f64,
            "memory_is_eager" => c.memory_eager as i32 as f64,
            _ => 0.0,
        })
        .collect()
}

/// Which issue-#2 axes the loaded model actually tunes, detected from its
/// `input_spec`. A model trained before an axis existed does not reference it, so
/// the axis stays pinned to [`BASE`] and the candidate grid is unchanged (auto
/// behaviour bit-identical). A retrained model that includes the axis activates it.
struct ActiveKnobs {
    pivot_u: bool,
    scaling: bool,
    memory: bool,
}

fn active_knobs(m: &Model) -> ActiveKnobs {
    let mut a = ActiveKnobs { pivot_u: false, scaling: false, memory: false };
    for comp in &m.input_spec {
        match comp.kind.as_str() {
            "pivot_u" => a.pivot_u = true,
            "scaling_onehot" => a.scaling = true,
            "memory_is_eager" => a.memory = true,
            _ => {}
        }
    }
    a
}

/// Forward pass; returns `[log factor_ms, log peak_mb]` (un-standardized).
fn predict(m: &Model, input: &[f64]) -> [f64; 2] {
    let mut a = input.to_vec();
    let last = m.layers.len().saturating_sub(1);
    for (li, layer) in m.layers.iter().enumerate() {
        let mut z = vec![0.0f64; layer.b.len()];
        for (o, row) in layer.w.iter().enumerate() {
            let mut s = layer.b[o];
            for (k, &wv) in row.iter().enumerate() {
                s += wv * a[k];
            }
            z[o] = if li < last && s < 0.0 { 0.0 } else { s }; // ReLU on hidden layers
        }
        a = z;
    }
    [
        a[0] * m.target_std[0] + m.target_mean[0],
        a[1] * m.target_std[1] + m.target_mean[1],
    ]
}

/// Candidate grid over ordering, amalgamation (`nemin`/`relax`), method, and the
/// kernel scheduling knobs (panel width, top-of-tree GEMM gate). Includes an
/// explicit `MetisND` candidate: on 3D / complex-indefinite classes (curl-curl EM)
/// the adaptive `Auto` heuristic mispicks and MetisND cuts fill ~3x, and the
/// deterministic fill backstop (a pick must not exceed the default's fill) rejects
/// MetisND on the banded classes where it over-separates -- so it wins where it
/// helps and is vetoed where it hurts. Memory stays safe by the backstop
/// regardless of which knob is picked.
fn candidates(path: SolverPath, active: &ActiveKnobs) -> Vec<Candidate> {
    let orderings = [OrderingMethod::Auto, OrderingMethod::Amd, OrderingMethod::MetisND];
    let nemins = [1usize, 16, 48, 128];
    let relaxes = [0usize, 128, 256, 512];
    let panels = [32usize, 64, 96, 128];
    let cdivs = [2_000_000usize, 8_000_000, 32_000_000];
    let methods = [FactorMethod::LeftLooking, FactorMethod::Multifrontal];
    // Issue-#2 axes: expanded only when the model tunes them; otherwise a single
    // baseline value keeps the grid (and thus the auto pick) bit-identical. The
    // threshold-pivot `u` is an LU-only knob (Bunch-Kaufman ignores it), so it is
    // pinned to the default on the LDLᵀ path regardless of the model.
    let pivot_us: &[f64] = if active.pivot_u && path == SolverPath::Lu {
        &[0.0, 0.1, 0.5, 1.0]
    } else {
        &[BASE.pivot_u]
    };
    let scalings: &[ScalingKnob] = if active.scaling {
        &[ScalingKnob::OnePass, ScalingKnob::Identity, ScalingKnob::InfNorm, ScalingKnob::Auto]
    } else {
        &[ScalingKnob::OnePass]
    };
    let memories: &[bool] = if active.memory { &[false, true] } else { &[false] };
    let mut v = Vec::new();
    for &ordering in &orderings {
        for &nemin in &nemins {
            for &relax_width in &relaxes {
                for &panel_nb in &panels {
                    for &par_cdiv in &cdivs {
                        for &method in &methods {
                            for &pivot_u in pivot_us {
                                for &scaling in scalings {
                                    for &memory_eager in memories {
                                        v.push(Candidate {
                                            ordering,
                                            nemin,
                                            relax_width,
                                            panel_nb,
                                            par_cdiv,
                                            method,
                                            pivot_u,
                                            scaling,
                                            memory_eager,
                                            ..BASE
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

fn apply(c: &Candidate) -> SolverSettings {
    let relax = (c.relax_width > 0).then_some(crate::RelaxAmalgamation {
        max_width: c.relax_width,
        max_extra_rows: 64,
    });
    SolverSettings::default()
        .with_ordering(c.ordering)
        .with_nemin(c.nemin)
        .with_relax(relax)
        .with_panel_nb(c.panel_nb)
        .with_gemm_thresholds(GemmThresholds {
            scalar_gate: c.scalar_gate,
            par_gemm: c.par_gemm,
            par_cdiv: c.par_cdiv,
        })
        .with_use_gemm_schur(c.use_gemm_schur)
        .with_method(c.method)
        .with_pivot_u(c.pivot_u)
        .with_scaling(c.scaling.to_strategy())
        .with_memory(if c.memory_eager {
            crate::MemoryMode::Eager
        } else {
            crate::MemoryMode::LowMemory
        })
}

/// Minimum predicted score gain (in `w·log(time)+(1-w)·log(mem)` units, i.e. ~ a
/// log-ratio) required to deviate from the default config at all - below this the
/// model's edge is within its own error, so keep the proven default. `0.08 ≈ 8%`.
const MIN_GAIN: f64 = 0.08;
/// Larger gain required to *flip the factorization method* (left-looking <-> mult
/// frontal): the highest-variance knob, where the model over-favours multifrontal
/// and the strong regressions live. `0.20 ≈ 22%`.
const METHOD_FLIP_GAIN: f64 = 0.20;
/// A-priori veto: reject a multifrontal pick when its exact transient-memory
/// estimate exceeds the left-looking **realistic floor** (`panel_live_peak`) by
/// more than this factor. The floor (not the loose all-panels `transient`) is the
/// reliable left-looking memory reference - the all-panels estimate is wildly
/// conservative on banded / structural patterns, so `mf_transient / ll_transient`
/// misleads there; `mf_transient / ll_floor > 1` is the memory-safe bar that cuts
/// the multifrontal CB-stack regressions.
const VETO_MF_MEM_RATIO: f64 = 1.0;
/// Hard memory constraint: a candidate's **predicted** peak memory must not exceed
/// the default's by more than this (log-space) tolerance. Memory is the critical
/// resource - it must not regress - so this is a constraint, not a soft weight: the
/// tuner only ever considers configs that do not use more memory than the default,
/// and picks the fastest among them. `ln(1.02)` ≈ 2% slack for model noise.
const MEM_TOL_LN: f64 = 0.0198;

/// Recommend a [`SolverSettings`] for a matrix with the given structural features,
/// minimizing `weight·log(time) + (1-weight)·log(mem)` over the candidate grid via
/// the embedded performance model. `weight` is clamped to `[0, 1]`
/// ([`DEFAULT_TUNE_WEIGHT`] = speed-leaning). Applies the safety + method-flip
/// guards (only deviates from the default when the predicted gain is clear); the
/// worker-thread count is left to the [`Auto`](crate::Threads::Auto) predictor.
/// Use [`recommend_settings_vetoed`] when the per-path memory estimate is known to
/// also veto memory-pathological multifrontal picks.
pub fn recommend_settings(features: &StructuralFeatures, weight: f64) -> SolverSettings {
    recommend_settings_pathed(features, weight, 1.0, SolverPath::Ldlt)
}

/// [`recommend_settings`] plus the **a-priori memory veto**: `mf_ll_mem_ratio` is
/// the exact `multifrontal / left-looking` transient-memory estimate
/// ([`MemoryEstimate`](crate::diagnostics::MemoryEstimate)); multifrontal
/// candidates are rejected when it exceeds [`VETO_MF_MEM_RATIO`], deterministically
/// cutting the multifrontal blow-up regressions. Pass `1.0` to disable the veto.
pub fn recommend_settings_vetoed(
    features: &StructuralFeatures,
    weight: f64,
    mf_ll_mem_ratio: f64,
) -> SolverSettings {
    recommend_settings_pathed(features, weight, mf_ll_mem_ratio, SolverPath::Ldlt)
}

/// [`recommend_settings_vetoed`] for a specific solver [`SolverPath`]: the LDLᵀ and
/// LU paths each use their own trained model and candidate grid (the LU grid
/// searches the threshold-pivot `u`; the LDLᵀ grid pins it). [`LdltSolver`] and
/// [`LuSolver`] call this with their path.
pub fn recommend_settings_pathed(
    features: &StructuralFeatures,
    weight: f64,
    mf_ll_mem_ratio: f64,
    path: SolverPath,
) -> SolverSettings {
    let w = weight.clamp(0.0, 1.0);
    let Some(m) = model_for(path) else {
        return SolverSettings::default();
    };
    // Out-of-distribution: above the training grid's well-sampled range the model
    // extrapolates on the knobs, so it must not be trusted. But the *ordering*
    // choice is decidable from the exact a-priori fill (not extrapolated), and on
    // large 3D / complex-indefinite classes the adaptive `Auto` heuristic mispicks
    // badly (curl-curl EM: ~3x extra fill, an order of magnitude in factor time).
    // So the OOD fallback is the deterministic ordering race (`AutoRace`): it picks
    // the minimum-exact-fill ordering (with `Auto` among the candidates, so never
    // worse than the default), a safe non-model win exactly where the model cannot
    // help. The remaining knobs stay at the proven default.
    if m.flops_ood_cap > 0.0 && features.factor_flops as f64 > m.flops_ood_cap {
        return SolverSettings::default().with_ordering(OrderingMethod::AutoRace);
    }
    let base = predict(m, &build_input(m, features, &BASE)); // [log time, log mem]
    let base_score = w * base[0] + (1.0 - w) * base[1];
    let mem_cap = base[1] + MEM_TOL_LN; // hard: never exceed the default's peak memory
    let mut best = BASE;
    let mut best_score = base_score;
    let active = active_knobs(m);
    for c in candidates(path, &active) {
        // A-priori veto: skip multifrontal picks whose exact transient estimate is
        // much worse than left-looking (catches the CB-stack memory blow-up).
        if c.method == FactorMethod::Multifrontal && mf_ll_mem_ratio > VETO_MF_MEM_RATIO {
            continue;
        }
        let p = predict(m, &build_input(m, features, &c));
        // Hard memory constraint: never pick a config predicted to use more memory
        // than the default (memory is the critical resource). Note this is on the
        // *full* config (ordering + method), so multifrontal is allowed when a
        // better ordering keeps its peak within the default's.
        if p[1] > mem_cap {
            continue;
        }
        let s = w * p[0] + (1.0 - w) * p[1];
        if s < best_score {
            best = c;
            best_score = s;
        }
    }
    // Safety + method-flip guard: only deviate from the default when the predicted
    // gain clears the margin (a larger one for a method flip - the risky knob).
    let needed = if best.method != BASE.method { METHOD_FLIP_GAIN } else { MIN_GAIN };
    if base_score - best_score < needed {
        return apply(&BASE);
    }
    apply(&best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_loads_and_predicts_finite() {
        let m = model_for(SolverPath::Ldlt).expect("embedded model parses");
        assert!(!m.layers.is_empty() && m.target_mean.len() == 2);
        // A plausible mid-size feature vector; prediction must be finite.
        let c = BASE;
        let dim: usize = m.layers[0].w[0].len();
        assert_eq!(m.input_spec.len(), dim, "input spec width matches layer 0");
        // Build a zeroed feature struct and predict - exercises the full path.
        let f = StructuralFeatures::default();
        let p = predict(m, &build_input(m, &f, &c));
        assert!(p[0].is_finite() && p[1].is_finite());
    }

    #[test]
    fn recommend_returns_valid_settings() {
        let f = StructuralFeatures::default();
        let s = recommend_settings(&f, DEFAULT_TUNE_WEIGHT);
        // A recommendation is one of the candidate orderings/methods.
        assert!(s.nemin >= 1 && s.panel_nb >= 8);
    }

    /// Numerical parity with the Python training model: for each reference sample
    /// (`benches/train_tuner.py` writes `tuner_parity.json`), the pure-Rust forward
    /// pass must reproduce the predicted log-time / log-mem. Env-gated so it only
    /// runs when the fixture path is provided (`RLA_TUNER_PARITY=<path>`).
    #[test]
    fn parity_with_python_model() {
        let Ok(path) = std::env::var("RLA_TUNER_PARITY") else {
            return;
        };
        // RLA_TUNER_PARITY_PATH selects which per-path model to check against the
        // fixture (default LDLᵀ). Run once per path with its own fixture.
        let sp = match std::env::var("RLA_TUNER_PARITY_PATH").as_deref() {
            Ok("lu") => SolverPath::Lu,
            _ => SolverPath::Ldlt,
        };
        let m = model_for(sp).expect("model");
        let txt = std::fs::read_to_string(&path).expect("parity fixture");
        let samples: serde_json::Value = serde_json::from_str(&txt).unwrap();
        let order = |s: &str| match s {
            "amd" => OrderingMethod::Amd,
            "amf" => OrderingMethod::Amf,
            "metis" => OrderingMethod::MetisND,
            "scotch" => OrderingMethod::ScotchND,
            "kahip" => OrderingMethod::KahipND,
            "rcm" => OrderingMethod::Rcm,
            "auto_race" => OrderingMethod::AutoRace,
            _ => OrderingMethod::Auto,
        };
        let g = |v: &serde_json::Value, k: &str| v[k].as_f64().unwrap() as usize;
        for s in samples.as_array().unwrap() {
            let f: StructuralFeatures = serde_json::from_value(s["features"].clone()).unwrap();
            let p = &s["params"];
            let c = Candidate {
                ordering: order(p["ordering"].as_str().unwrap()),
                nemin: g(p, "nemin"),
                relax_width: g(p, "relax_width"),
                panel_nb: g(p, "panel_nb"),
                scalar_gate: g(p, "scalar_gate"),
                par_gemm: g(p, "par_gemm"),
                par_cdiv: g(p, "par_cdiv"),
                use_gemm_schur: p["use_gemm_schur"].as_bool().unwrap(),
                method: if p["method"] == "multifrontal" {
                    FactorMethod::Multifrontal
                } else {
                    FactorMethod::LeftLooking
                },
                // Issue-#2 axes carried by the sample (default to BASE if absent).
                pivot_u: p["pivot_u"].as_f64().unwrap_or(BASE.pivot_u),
                scaling: match p["scaling"].as_str() {
                    Some("identity") => ScalingKnob::Identity,
                    Some("infnorm") => ScalingKnob::InfNorm,
                    Some("auto") => ScalingKnob::Auto,
                    _ => ScalingKnob::OnePass,
                },
                memory_eager: p["memory_eager"].as_bool().unwrap_or(false),
            };
            let pred = predict(m, &build_input(m, &f, &c));
            let (ems, emb) = (s["pred_log_ms"].as_f64().unwrap(), s["pred_log_mb"].as_f64().unwrap());
            assert!((pred[0] - ems).abs() < 1e-4, "log_ms parity: rust {} vs py {}", pred[0], ems);
            assert!((pred[1] - emb).abs() < 1e-4, "log_mb parity: rust {} vs py {}", pred[1], emb);
        }
    }

    #[test]
    fn issue2_axes_active_and_path_gated() {
        // The shipped models carry the issue-#2 axes in their input_spec, so the
        // tuner expands the candidate grid over them. Base grid 768 = 2·4·4·4·3·2;
        // scaling ×4 and memory ×2 on both paths, and pivot_u ×4 only on the LU
        // path (Bunch-Kaufman ignores it, so it is pinned on LDLᵀ).
        let ldlt = active_knobs(model_for(SolverPath::Ldlt).expect("ldlt model loads"));
        let lu = active_knobs(model_for(SolverPath::Lu).expect("lu model loads"));
        assert!(ldlt.scaling && ldlt.memory && !ldlt.pivot_u, "LDLᵀ tunes scaling+memory, not pivot_u");
        assert!(lu.scaling && lu.memory && lu.pivot_u, "LU tunes scaling+memory+pivot_u");
        // Base grid: 3 orderings·4 nemin·4 relax·4 panel·3 cdiv·2 method = 1152.
        assert_eq!(candidates(SolverPath::Ldlt, &ldlt).len(), 1152 * 8, "LDLᵀ: scaling×4·memory×2");
        assert_eq!(candidates(SolverPath::Lu, &lu).len(), 1152 * 32, "LU: +pivot_u×4");
        // Data-driven gate: a model referencing no axis leaves the grid unchanged.
        let none = ActiveKnobs { pivot_u: false, scaling: false, memory: false };
        assert_eq!(candidates(SolverPath::Ldlt, &none).len(), 1152);
        assert_eq!(candidates(SolverPath::Lu, &none).len(), 1152);
    }
}
