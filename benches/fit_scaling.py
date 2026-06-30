"""Corpus scaling study: factor time vs problem size for all five solvers, with
a power-law fit through each solver's corpus points.

Reads the merged corpus JSONL (RSLAB left-looking / multifrontal, faer, MKL
PARDISO, SuperLU) and plots `fac_ms` against `nnz` on log-log axes - one scatter
of corpus points per solver plus a least-squares power-law fit
`fac = C * nnz^alpha` (the fitted exponent is the empirical scaling order,
printed in the legend). Garbage points (relative residual > 0.1) are excluded
from the fit so a solver that silently fails does not distort its trend.

Run:  python benches/fit_scaling.py [corpus.jsonl]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {  # solver -> (label, color, marker)
    "ll": ("RSLAB left-looking", "#3b82f6", "o"),
    "mf": ("RSLAB multifrontal", "#06b6d4", "s"),
    "faer": ("faer LU", "#f59e0b", "^"),
    "pardiso": ("MKL PARDISO", "#22c55e", "D"),
    "superlu": ("SuperLU (scipy)", "#ef4444", "P"),
}
ORDER = ["ll", "mf", "faer", "pardiso", "superlu"]
GRAY = "#808080"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 11,
    })


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/corpus.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get("metric") == "time" and r.get("fac_ms", 0) > 0]

    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    summary = []
    for s in ORDER:
        pts = [(r["nnz"], r["fac_ms"]) for r in recs
               if r["solver"] == s and r["nnz"] > 0 and r.get("res", 1.0) < 0.1]
        if len(pts) < 2:
            continue
        label, color, marker = STYLE[s]
        x = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        ax.scatter(x, y, s=42, c=color, marker=marker, alpha=0.8, edgecolors="none", zorder=3)
        # Power-law fit in log-log space: log y = alpha*log x + log C.
        alpha, logc = np.polyfit(np.log(x), np.log(y), 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, np.exp(logc) * xs ** alpha, color=color, lw=2, alpha=0.9, zorder=2,
                label=f"{label}  (n={len(pts)}, $\\alpha$={alpha:.2f})")
        summary.append((label, alpha, len(pts)))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nonzeros (nnz)")
    ax.set_ylabel("factor time [ms]")
    ax.set_title("Corpus scaling: factor time vs problem size (power-law fits)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    out = path.parent / "corpus_scaling_fit.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, transparent=True)
    print(f"wrote {out}")
    print("fitted scaling exponents (fac ~ nnz^alpha):")
    for label, alpha, n in summary:
        print(f"  {label:<22} alpha={alpha:.2f}  ({n} points)")


if __name__ == "__main__":
    main()
