"""Corpus scaling study: factor time vs problem size for all five solvers, with
a power-law fit through each solver's corpus points.

Reads the corpus JSONL (RSLAB auto-tuned default, faer, MKL PARDISO, SuperLU -
all measured in one run) and plots `fac_ms` against `nnz` on log-log axes - one scatter
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

import bench_style
from bench_style import GRAY, SOLVERS

# The auto-tuned default (`LdltSolver::factor` / `LuSolver::factor`) is the product
# RSLAB ships: per matrix it picks left-looking or multifrontal under the memory /
# OOD guards. We plot that single curve against the external solvers rather than the
# two raw kernels (those are compared internally in corpus_breakdown.py).
ORDER = ["auto", "faer", "pardiso", "superlu"]


def plot_metric(recs, metric, value_key, ylabel, title, out):
    rows = [r for r in recs if r.get("metric") == metric and r.get(value_key, 0) > 0]
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    summary = []
    for s in ORDER:
        pts = [(r["nnz"], r[value_key]) for r in rows
               if r["solver"] == s and r["nnz"] > 0 and r.get("res", 1.0) < 0.1]
        if len(pts) < 2:
            continue
        label, color, marker = SOLVERS[s]
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
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    bench_style.legend_below(fig, ax=ax)
    fig.savefig(out, dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {out}")
    for label, alpha, n in summary:
        print(f"  {label:<22} alpha={alpha:.2f}  ({n} points)")


def plot_residual(recs, out):
    """Per-solver accuracy: relative residual vs problem size over the corpus."""
    rows = [r for r in recs if r.get("metric") == "time"]
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    for s in ORDER:
        pts = [(r["nnz"], max(r.get("res", 0.0), 1e-18)) for r in rows
               if r["solver"] == s and r["nnz"] > 0]
        if not pts:
            continue
        label, color, marker = SOLVERS[s]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.scatter(x, y, s=42, c=color, marker=marker, alpha=0.8, edgecolors="none",
                   label=f"{label} (n={len(pts)})")
    ax.axhline(1e-8, color=GRAY, ls="--", lw=1.2, alpha=0.7)
    ax.text(ax.get_xlim()[0], 1.3e-8, "  accuracy target 1e-8", color=GRAY, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nonzeros (nnz)")
    ax.set_ylabel("relative residual ‖Ax-b‖/‖b‖")
    ax.set_title("Corpus accuracy: relative residual per solver")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    bench_style.legend_below(fig, ax=ax)
    fig.savefig(out, dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/corpus.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    bench_style.setup()
    print("factor time ~ nnz^alpha:")
    plot_metric(recs, "time", "fac_ms", "factor time [ms]",
                "Corpus scaling: factor time vs problem size (power-law fits)",
                path.parent / "corpus_scaling_fit.png")
    print("peak memory ~ nnz^alpha:")
    plot_metric(recs, "mem", "mem_mb", "peak memory [MB]",
                "Corpus scaling: peak factor memory vs problem size (power-law fits)",
                path.parent / "corpus_memory_fit.png")
    plot_residual(recs, path.parent / "corpus_residual.png")


if __name__ == "__main__":
    main()
