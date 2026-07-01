"""Train the RSLAB auto-tuner: a performance model that maps a matrix's structural
features + a knob config to (factor time, peak memory), so the solver can pick the
config minimizing a weighted Pareto score  w*log(time) + (1-w)*log(mem)  for a new
matrix from its features alone.

Pipeline:
  1. Load the sweep dataset (`benches/bench_out/sweep.jsonl`), keep valid records.
  2. Build a canonical input vector (transformed features + encoded knobs) per a
     spec that is *exported* so the pure-Rust inference builds the identical vector.
  3. Train a small MLP (sklearn) with multi-output targets [log ms, log mb].
  4. Validate by *matrix-grouped* split (test matrices unseen) - report R^2 and,
     end-to-end, the *regret*: time/mem of the model-picked config vs the oracle
     best, over held-out matrices.
  5. Retrain on all data and export a self-contained model JSON for Rust.

Run:  python benches/train_tuner.py [sweep.jsonl] [out_model.json]
"""
import json
import sys
import math
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupShuffleSplit

# Structural features used as model inputs. `log` = apply log1p before standardizing
# (wide-range, non-negative); fractions/ratios stay linear.
FEATURES = [
    ("n", True), ("nnz", True), ("deg_mean", True), ("deg_max", True),
    ("deg_cv", False), ("bandwidth_max", True), ("bandwidth_mean_rel", False),
    ("diag_dominant_frac", False), ("diag_present_frac", False),
    ("n_supernodes", True), ("fill_nnz", True), ("fill_ratio", True),
    ("supernode_cols_mean", True), ("front_nrow_max", True), ("tree_depth", True),
    ("tree_width_max", True), ("tree_width_mean", True), ("factor_flops", True),
    ("arith_intensity", True), ("flop_top1_frac", False), ("flop_top1pct_frac", False),
]
ORDERINGS = ["auto", "amd", "metis"]  # one-hot
# Numeric knobs (log1p + standardize).
KNOBS_NUM = ["nemin", "relax_width", "panel_nb", "scalar_gate", "par_gemm", "par_cdiv"]
# Linear (not log) numeric knobs, standardized. Issue #2: threshold pivot u in [0,1].
KNOBS_LIN = ["pivot_u"]
# Issue #2 equilibration strategy (one-hot) + emit/memory mode (bool memory_eager).
SCALINGS = ["onepass", "identity", "infnorm", "auto"]
# Defaults for records predating the issue-#2 axes (so old sweeps still train; a
# constant column standardizes to 0 and contributes nothing).
KNOB_DEFAULTS = {"pivot_u": 0.1, "scaling": "onepass", "memory_eager": False}


def load(path):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rs if r["metrics"]["residual"] < 1e-6
            and r["metrics"]["factor_ms"] > 0 and r["metrics"]["peak_mb"] > 0]


def raw_input_row(r):
    """The pre-standardization numeric row (features + numeric knobs), plus the
    categorical/bool parts, as a flat dict keyed by input-component name."""
    f, p = r["features"], r["params"]
    row = {}
    for name, lg in FEATURES:
        v = float(f[name])
        row[name] = math.log1p(v) if lg else v
    for name in KNOBS_NUM:
        row[name] = math.log1p(float(p[name]))
    for name in KNOBS_LIN:
        row[name] = float(p.get(name, KNOB_DEFAULTS[name]))
    return row


def build_matrix(recs):
    """Return X (N x D), the component spec (names + which are standardized), and the
    standardization stats. One-hot and bool components are not standardized."""
    # Standardized numeric components: features + log knobs + linear knobs. The
    # column order here is the exact order the Rust `build_input` reproduces from
    # `input_spec`, so X columns and `spec` entries must stay in lockstep.
    num_names = [n for n, _ in FEATURES] + KNOBS_NUM + KNOBS_LIN
    raw = np.array([[raw_input_row(r)[n] for n in num_names] for r in recs], float)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-12] = 1.0
    num = (raw - mean) / std
    # One-hot ordering + bool method(MF=1)/use_gemm_schur + issue-#2 scaling one-hot
    # + memory_eager bool.
    oh = np.array([[1.0 if r["params"]["ordering"] == o else 0.0 for o in ORDERINGS]
                   for r in recs], float)
    meth = np.array([[1.0 if r["params"]["method"] == "multifrontal" else 0.0] for r in recs], float)
    schur = np.array([[1.0 if r["params"]["use_gemm_schur"] else 0.0] for r in recs], float)
    scal = np.array([[1.0 if r["params"].get("scaling", KNOB_DEFAULTS["scaling"]) == s else 0.0
                      for s in SCALINGS] for r in recs], float)
    mem = np.array([[1.0 if r["params"].get("memory_eager", KNOB_DEFAULTS["memory_eager"]) else 0.0]
                    for r in recs], float)
    X = np.hstack([num, oh, meth, schur, scal, mem])
    # Component spec, in column order, for the Rust side to reproduce the vector.
    spec = []
    for i, (name, lg) in enumerate(FEATURES):
        spec.append({"kind": "feat", "name": name, "log": lg,
                     "mean": mean[i], "std": std[i]})
    for j, name in enumerate(KNOBS_NUM):
        i = len(FEATURES) + j
        spec.append({"kind": "knob_log", "name": name, "mean": mean[i], "std": std[i]})
    for j, name in enumerate(KNOBS_LIN):
        i = len(FEATURES) + len(KNOBS_NUM) + j
        # Rust matches this axis by `kind`; `name` is advisory. `pivot_u` is the
        # only linear knob today.
        spec.append({"kind": name, "name": name, "mean": mean[i], "std": std[i]})
    for o in ORDERINGS:
        spec.append({"kind": "ordering_onehot", "value": o})
    spec.append({"kind": "method_is_mf"})
    spec.append({"kind": "use_gemm_schur"})
    for s in SCALINGS:
        spec.append({"kind": "scaling_onehot", "value": s})
    spec.append({"kind": "memory_is_eager"})
    return X, spec


def targets(recs):
    t = np.array([[math.log(r["metrics"]["factor_ms"]),
                   math.log(r["metrics"]["peak_mb"])] for r in recs], float)
    return t


def make_mlp():
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                        solver="adam", alpha=1e-4, max_iter=2000,
                        early_stopping=True, n_iter_no_change=30, random_state=0)


def grouped_eval(recs):
    """Matrix-grouped split: train on some matrices, test on unseen ones. Reports
    R^2 per target and the end-to-end regret of the model-picked config."""
    X, _ = build_matrix(recs)
    y = targets(recs)
    ymean, ystd = y.mean(axis=0), y.std(axis=0)
    yn = (y - ymean) / ystd
    groups = [r["matrix"] for r in recs]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    tr, te = next(gss.split(X, yn, groups))
    model = make_mlp().fit(X[tr], yn[tr])
    pred = model.predict(X[te]) * ystd + ymean
    true = y[te]
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot
    print(f"  held-out R^2:  log(ms)={r2[0]:.3f}  log(mb)={r2[1]:.3f}  "
          f"({len(set(groups[i] for i in tr))} train / {len(set(groups[i] for i in te))} test matrices)")
    # Regret: per held-out matrix, model-picked config (min weighted score) vs oracle.
    te_recs = [recs[i] for i in te]
    by_mat = {}
    for idx, r in zip(te, te_recs):
        by_mat.setdefault(r["matrix"], []).append(idx)
    for w in (0.7, 1.0, 0.0):
        t_reg, m_reg = [], []
        for mat, idxs in by_mat.items():
            p = model.predict(X[idxs]) * ystd + ymean  # [log ms, log mb]
            score = w * p[:, 0] + (1 - w) * p[:, 1]
            pick = idxs[int(np.argmin(score))]
            tms = np.array([recs[i]["metrics"]["factor_ms"] for i in idxs])
            mmb = np.array([recs[i]["metrics"]["peak_mb"] for i in idxs])
            t_reg.append(recs[pick]["metrics"]["factor_ms"] / tms.min())
            m_reg.append(recs[pick]["metrics"]["peak_mb"] / mmb.min())
        g = lambda xs: math.exp(np.mean(np.log(xs)))
        print(f"    w={w:.1f}: picked/oracle  time x{g(t_reg):.3f}  mem x{g(m_reg):.3f}  (geomean over {len(by_mat)} matrices)")


def export(recs, out_path, parity_path=None, n_parity=8):
    X, spec = build_matrix(recs)
    y = targets(recs)
    ymean, ystd = y.mean(axis=0), y.std(axis=0)
    yn = (y - ymean) / ystd
    model = make_mlp().fit(X, yn)
    # Parity fixture from the *same* fitted model (no drift) for the Rust check.
    if parity_path is not None:
        idxs = list(range(0, len(recs), max(1, len(recs) // n_parity)))[:n_parity]
        pred = model.predict(X[idxs]) * ystd + ymean
        samples = [{"features": recs[i]["features"], "params": recs[i]["params"],
                    "pred_log_ms": float(pred[k][0]), "pred_log_mb": float(pred[k][1])}
                   for k, i in enumerate(idxs)]
        json.dump(samples, open(parity_path, "w"))
        print(f"  wrote {parity_path}  ({len(samples)} parity samples)")
    layers = [{"w": c.T.tolist(), "b": b.tolist()}  # w: (out,in) row-major for Rust
              for c, b in zip(model.coefs_, model.intercepts_)]
    # Out-of-distribution guard: above the largest *well-sampled* matrix (one that
    # got the full knob grid, not just the flop-gated baseline) the model has no
    # knob-variation signal and extrapolates badly, so the caller falls back to the
    # default there. Use the max factor_flops among matrices with many configs.
    from collections import Counter
    cfg_count = Counter(r["matrix"] for r in recs)
    well = [r["features"]["factor_flops"] for r in recs if cfg_count[r["matrix"]] >= 10]
    flops_ood_cap = max(well) if well else 0.0
    out = {
        "input_spec": spec,
        "orderings": ORDERINGS,
        "layers": layers,             # relu between, identity on the last
        "target_mean": ymean.tolist(),  # [log ms, log mb]
        "target_std": ystd.tolist(),
        "flops_ood_cap": float(flops_ood_cap),
        "n_records": len(recs),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    print(f"  wrote {out_path}  ({len(spec)} inputs, layers {[len(c.intercepts_) if False else len(l['b']) for l in layers]})")


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/sweep.jsonl")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benches/bench_out/tuner_model.json")
    recs = load(src)
    print(f"loaded {len(recs)} valid records over {len(set(r['matrix'] for r in recs))} matrices")
    print("matrix-grouped generalization:")
    grouped_eval(recs)
    print("final model (all data):")
    export(recs, out, parity_path=out.parent / "tuner_parity.json")


if __name__ == "__main__":
    main()
