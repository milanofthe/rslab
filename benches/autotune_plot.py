"""Auto-tuner end-to-end figures: the tuned config (from the embedded MLP, picked
from structural features alone) vs the default settings, measured across the corpus.

Reads `autotune.jsonl` (one record per (matrix, config) where config is
default / tuned_balanced / tuned_speed / tuned_memory) and emits:

  1. `autotune_vs_size.png`  - speedup (default/tuned time) and peak-memory ratio
     (tuned/default) of the balanced tuner vs problem size (nnz). Answers whether
     the benefit holds / grows for larger matrices.
  2. `autotune_modes.png`    - per-matrix time-vs-memory relative to default for the
     three Pareto weights, with geomean markers - the speed/memory trade-off knob.

Run:  python benches/autotune_plot.py [autotune.jsonl]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bench_style
from bench_style import GRAY, BLUE, CYAN, AMBER, GREEN

MODES = [("tuned_balanced", "balanced (w=0.7)", BLUE),
         ("tuned_speed", "speed (w=1)", AMBER),
         ("tuned_memory", "memory (w=0)", GREEN)]


def geomean(xs):
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def load(path):
    """matrix -> {config: {ms, mb, nnz}}, keeping only valid factorizations."""
    by = defaultdict(dict)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = r["metrics"]
        # Sanity: drop invalid residuals and absurd peaks (a metering underflow
        # would wrap to ~2^44 MB - guard even though the allocator now saturates).
        if not (m["residual"] < 1e-6 and 0 < m["factor_ms"] and 0 < m["peak_mb"] < 1e6):
            continue
        by[r["matrix"]][r["config"]] = {"ms": m["factor_ms"], "mb": m["peak_mb"], "nnz": r["nnz"]}
    # keep matrices with a default + at least the balanced tuned config
    return {k: v for k, v in by.items() if "default" in v and "tuned_balanced" in v}


def plot_vs_size(data, outdir, suffix="", n=None):
    """Two panels (factor speedup, peak-memory ratio) vs problem size, with the
    three tuner modes (balanced / speed / memory) overlaid - shows each mode's
    time and memory effect and how it tracks problem size."""
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
    handles = []
    print("tuned vs default (geomean over corpus; small/large = below/above median nnz):")
    for cfg, label, color in MODES:
        rows = sorted((d for d in data.values() if cfg in d), key=lambda d: d["default"]["nnz"])
        nnz = np.array([d["default"]["nnz"] for d in rows], float)
        speedup = np.array([d["default"]["ms"] / d[cfg]["ms"] for d in rows])
        memratio = np.array([d[cfg]["mb"] / d["default"]["mb"] for d in rows])
        axes[0].scatter(nnz, speedup, s=34, c=color, edgecolors="none")
        axes[0].axhline(geomean(speedup), color=color, ls=":", lw=1.4)
        axes[1].scatter(nnz, memratio, s=34, c=color, edgecolors="none")
        axes[1].axhline(geomean(memratio), color=color, ls=":", lw=1.4)
        handles.append(Line2D([], [], color=color, marker="o", ls="",
                              label=f"{label}: {geomean(speedup):.2f}x time, {geomean(memratio):.2f}x mem"))
        med = np.median(nnz)
        sm, lg = nnz < med, nnz >= med
        print(f"  {label:<18} speedup {geomean(speedup):.3f}x  mem {geomean(memratio):.3f}x  "
              f"| small {geomean(speedup[sm]):.2f}x/{geomean(memratio[sm]):.2f}x  "
              f"large {geomean(speedup[lg]):.2f}x/{geomean(memratio[lg]):.2f}x  ({len(rows)} mat)")
    axes[0].axhline(1.0, color=GRAY, ls="--", lw=1.2)
    axes[0].set_ylabel("factor speedup\n(default / tuned)")
    cnt = f" ({n} matrices)" if n else ""
    axes[0].set_title(f"Auto-tuner vs default by problem size{cnt} (dotted = geomean)")
    axes[1].axhline(1.0, color=GRAY, ls="--", lw=1.2)
    axes[1].set_ylabel("peak-memory ratio\n(tuned / default)")
    axes[1].set_xlabel("nonzeros (nnz)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.4)
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, outdir / f"autotune_vs_size{suffix}.png")


def plot_modes(data, outdir, suffix="", title="Auto-tuner Pareto modes vs default"):
    fig, ax = plt.subplots(figsize=(8, 7))
    handles = []
    for cfg, label, color in MODES:
        t = np.array([d[cfg]["ms"] / d["default"]["ms"] for d in data.values() if cfg in d])
        m = np.array([d[cfg]["mb"] / d["default"]["mb"] for d in data.values() if cfg in d])
        ax.scatter(t, m, s=26, c=color, edgecolors="none")
        gt, gm = geomean(t), geomean(m)
        handles.append(Line2D([], [], color=color, marker="o", ls="", label=f"{label}  (geomean {gt:.2f}x time, {gm:.2f}x mem)"))
    ax.axvline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.set_xlabel("factor time relative to default")
    ax.set_ylabel("peak memory relative to default")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, outdir / f"autotune_modes{suffix}.png")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/autotune.jsonl")
    # Optional path label: `autotune_plot.py <jsonl> <slug> <title>` writes
    # `autotune_modes_<slug>.png` / `autotune_vs_size_<slug>.png` for a per-path plot.
    slug = sys.argv[2] if len(sys.argv) > 2 else ""
    title = sys.argv[3] if len(sys.argv) > 3 else "Auto-tuner Pareto modes vs default"
    suffix = f"_{slug}" if slug else ""
    bench_style.setup()
    data = load(path)
    if not data:
        print(f"[autotune] no valid records in {path}")
        return
    n = len(data)
    plot_vs_size(data, path.parent, suffix, n)
    plot_modes(data, path.parent, suffix, title)
    print(f"[autotune] {n} matrices -> autotune_modes{suffix}.png")


if __name__ == "__main__":
    main()
