#!/usr/bin/env python3
"""Fit the learned residual for the v2 analytical cost model (issue #62).

The v2 thread-aware time model predicts factor time from the a-priori cost
triple (factor_flops, critical_path_flops) and the hardware calibration:

    est_runtime(threads) ~ max( critical_path_flops / rate,          # Amdahl floor
                                factor_flops / (rate * hw_speedup) )  # parallel work

The absolute `rate` cancels in the *speedup* s(threads) = t(1)/t(threads), so the
model's structural quality is judged on speedup alone (hardware-agnostic):

    s_pred(threads) = factor_flops / max(critical_path_flops,
                                         factor_flops / hw_speedup(threads))

This caps s_pred at the Amdahl ceiling factor_flops/critical_path_flops, which is
exactly why curl-curl (crit/flops ~ 0.82 -> ceiling ~ 1.2x) barely scales. The
question #62 asks: does a small learned residual on log(s_meas / s_pred), a
function of dimensionless structural features, measurably reduce the error over
the bare analytical model? We fit a ridge regression and judge it by
leave-one-matrix-out cross-validation, so a residual is only recommended if it
generalizes to held-out matrices.

Usage: python benches/fit_residual.py <sweep.jsonl> [out_residual.json]
       hw_speedup peak + threads default to this machine's calibration; override
       with RLA_HW_SPEEDUP / RLA_HW_SPEEDUP_THREADS.
"""
import json
import math
import os
import sys
from collections import defaultdict


def hw_speedup(threads, peak, peak_threads):
    """The calibrated speedup curve (mirrors Calibration::speedup_for)."""
    if threads <= 1:
        return 1.0
    if threads >= peak_threads:
        return peak
    return 1.0 + (peak - 1.0) * (threads - 1) / max(peak_threads - 1, 1)


def load(path):
    """Group sweep records by matrix -> list of (threads, ms, cost)."""
    by_matrix = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cost = r.get("cost")
        if not cost:
            continue  # only the thread-ladder records carry the cost block
        ms = r["metrics"]["factor_ms"]
        if ms is None or ms <= 0:
            continue
        by_matrix[r["matrix"]].append((cost["threads"], ms, cost, r.get("features", {})))
    return by_matrix


def features(cost, threads):
    """Dimensionless structural inputs for the *additive* residual on log(s_pred).

    Deliberately excludes s_pred itself: including it lets the fit cancel the
    analytical base (weight -> -1) and become a pure learned model. Keeping it out
    forces a genuine residual that refines the hardware-calibrated base rather than
    replacing it, which transfers across machines. `amdahl_frac` (the critical-path
    fraction from issue #60) carries almost all of the signal."""
    flops = max(float(cost["factor_flops"]), 1.0)
    crit = max(float(cost["critical_path_flops"]), 1.0)
    width = max(float(cost["max_tree_width"]), 1.0)
    return [
        1.0,                       # intercept
        crit / flops,              # amdahl_frac in (0,1]: how serial the tree is
        math.log(threads),         # ladder position (per-thread overhead trend)
        math.log(width),           # tree width (parallelism available)
    ]


def ridge_fit(X, y, lam=1e-2):
    """Closed-form ridge regression (normal equations); pure Python."""
    m = len(X[0])
    # A = XᵀX + lam I ; b = Xᵀy
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) + (lam if i == j else 0.0)
          for j in range(m)] for i in range(m)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(m)]
    return solve(A, b)


def solve(A, b):
    """Gaussian elimination with partial pivoting."""
    m = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(m):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, m + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][m] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(m)]


def build_rows(by_matrix, peak, peak_threads):
    """Per (matrix, threads): dimensionless features, target log(s_meas/s_pred)."""
    rows = []
    for matrix, recs in by_matrix.items():
        recs = sorted(recs, key=lambda t: t[0])
        t1 = next((ms for th, ms, *_ in recs if th == 1), None)
        if t1 is None:
            continue
        for threads, ms, cost, feat in recs:
            if threads == 1:
                continue
            flops = float(cost["factor_flops"])
            crit = float(cost["critical_path_flops"])
            denom = max(crit, flops / hw_speedup(threads, peak, peak_threads))
            s_pred = flops / denom if denom > 0 else 1.0
            s_meas = t1 / ms
            if s_pred <= 0 or s_meas <= 0:
                continue
            x = features(cost, threads)
            y = math.log(s_meas / s_pred)  # residual target
            rows.append((matrix, x, y, s_pred, s_meas))
    return rows


def rmse(errs):
    return math.sqrt(sum(e * e for e in errs) / max(len(errs), 1))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sweep = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    peak = float(os.environ.get("RLA_HW_SPEEDUP", "4.25"))
    peak_threads = int(os.environ.get("RLA_HW_SPEEDUP_THREADS", "12"))

    by_matrix = load(sweep)
    rows = build_rows(by_matrix, peak, peak_threads)
    if not rows:
        print("no thread-ladder rows with cost fields found")
        sys.exit(1)
    matrices = sorted({r[0] for r in rows})
    print(f"{len(rows)} ladder points over {len(matrices)} matrices "
          f"(hw_speedup peak {peak:.2f} @ {peak_threads} threads)\n")

    # Baseline: the bare analytical model (residual == 0).
    base_err = [r[2] for r in rows]  # y = log(s_meas/s_pred); model predicts 0
    print(f"analytical-only  RMSE(log speedup) = {rmse(base_err):.4f}")

    # Leave-one-matrix-out CV of the ridge residual: fit on the other matrices,
    # predict the held-out one. This is the honest test of generalization.
    cv_err = []
    for held in matrices:
        tr = [(x, y) for (mx, x, y, *_ ) in rows if mx != held]
        te = [(x, y) for (mx, x, y, *_ ) in rows if mx == held]
        w = ridge_fit([x for x, _ in tr], [y for _, y in tr])
        for x, y in te:
            pred = sum(wi * xi for wi, xi in zip(w, x))
            cv_err.append(y - pred)
    print(f"with residual    RMSE(log speedup) = {rmse(cv_err):.4f}  (leave-one-matrix-out)")

    improvement = 1.0 - rmse(cv_err) / max(rmse(base_err), 1e-9)
    print(f"\nresidual reduces held-out speedup error by {improvement*100:.1f}%")

    # Fit the final residual on all rows for shipping (if it generalized).
    w = ridge_fit([r[1] for r in rows], [r[2] for r in rows])
    names = ["intercept", "amdahl_frac", "log_threads", "log_tree_width"]
    print("\nfitted residual weights:")
    for nm, wi in zip(names, w):
        print(f"  {nm:<16} {wi:+.4f}")

    ship = improvement >= 0.10  # ship only a meaningful (>=10%) held-out gain
    print(f"\n=> {'SHIP' if ship else 'DO NOT SHIP'} "
          f"(threshold 10% held-out improvement)")
    if out and ship:
        json.dump({"feature_names": names, "weights": w,
                   "hw_speedup_peak": peak, "hw_speedup_threads": peak_threads,
                   "cv_rmse": rmse(cv_err), "base_rmse": rmse(base_err)},
                  open(out, "w"), indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
