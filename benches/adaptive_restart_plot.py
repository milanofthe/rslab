"""Adaptive GMRES restart under a memory budget.

Reads the JSONL from the ``adaptive_restart`` bench and produces one two-panel figure:

* left  - the up-front FGMRES basis memory (2*n*(restart+1) scalars) vs problem size
  for fixed restart lengths, and the adaptive-policy curve (analytic): it rides at the
  maximum restart until the basis would exceed the 1 GiB budget, then declines to hold
  memory flat. The measured sizes are marked;
* right - measured GMRES iterations vs restart length per problem size: a longer
  restart cuts iterations (with diminishing return), the trade-off the budget navigates
  by picking the longest restart that fits.

Run:  python benches/adaptive_restart_plot.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_style
from bench_style import GRAY, RED, BLUE_SHADES

OUT = Path("benches/bench_out")
IN = OUT / "adaptive_restart.jsonl"

BUDGET = 1 << 30          # 1 GiB
RESTART_MIN, RESTART_MAX = 20, 80
BYTES = 16                # Complex<f64>
BASES = 2                 # single-RHS FGMRES V + Z


def adaptive(n):
    per_layer = n * BYTES * BASES
    cap = max(BUDGET // per_layer - 1, 0)
    return min(max(cap, RESTART_MIN), RESTART_MAX)


def basis_mb(n, restart):
    return BASES * n * (restart + 1) * BYTES / (1024.0 * 1024.0)


def main():
    if not IN.exists():
        print("no adaptive_restart.jsonl in", OUT)
        return
    recs = [json.loads(l) for l in open(IN) if l.strip()]
    if not recs:
        return
    ns = sorted({r["n"] for r in recs})
    restarts = sorted({r["restart"] for r in recs})

    bench_style.setup()
    fig, (axL, axR) = bench_style.two_panel()

    # --- Left: analytic basis memory vs n, fixed restarts + adaptive policy. ---
    grid = np.logspace(4, np.log10(4e6), 200)
    for j, r in enumerate(restarts):
        col = BLUE_SHADES[min(j, len(BLUE_SHADES) - 1)]
        axL.plot(grid, [basis_mb(n, r) for n in grid], color=col, lw=1.6,
                 alpha=0.9, label=f"restart={r}")
    axL.plot(grid, [basis_mb(n, adaptive(n)) for n in grid], color=RED, lw=2.6,
             label="adaptive policy")
    axL.axhline(BUDGET / (1024.0 * 1024.0), ls="--", color=GRAY, lw=1.3)
    axL.text(grid[0], BUDGET / (1024.0 * 1024.0), " 1 GiB budget", color=GRAY,
             va="bottom", fontsize=8)
    # Mark the measured sizes.
    for n in ns:
        axL.axvline(n, ls=":", color=GRAY, lw=0.8, alpha=0.5)
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("problem size  n")
    axL.set_ylabel("FGMRES basis memory  [MB]")
    axL.grid(True, ls=":", alpha=0.4)
    axL.set_title("Basis memory vs restart policy")

    # --- Right: measured iterations vs restart, per n. ---
    for j, n in enumerate(ns):
        rows = sorted((r for r in recs if r["n"] == n), key=lambda r: r["restart"])
        col = BLUE_SHADES[min(j, len(BLUE_SHADES) - 1)]
        axR.plot([r["restart"] for r in rows], [r["iters"] for r in rows],
                 color=col, marker="o", lw=2.0, ms=6, label=f"n={n}")
    axR.set_xlabel("restart length")
    axR.set_ylabel("GMRES iterations")
    axR.grid(True, ls=":", alpha=0.4)
    axR.set_title("Iterations vs restart")

    fig.suptitle("Adaptive GMRES restart under a 1 GiB basis budget", color=GRAY)
    # Combined legend: memory-panel restart curves + policy, and the size lines.
    bench_style.legend_below(fig, ax=axL)
    bench_style.save(fig, OUT / "adaptive_restart.png")


if __name__ == "__main__":
    main()
