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
        if not (m["residual"] < 1e-6 and m["factor_ms"] > 0 and m["peak_mb"] > 0):
            continue
        by[r["matrix"]][r["config"]] = {"ms": m["factor_ms"], "mb": m["peak_mb"], "nnz": r["nnz"]}
    # keep matrices with a default + at least the balanced tuned config
    return {k: v for k, v in by.items() if "default" in v and "tuned_balanced" in v}


def plot_vs_size(data, outdir):
    rows = sorted(data.values(), key=lambda d: d["default"]["nnz"])
    nnz = np.array([d["default"]["nnz"] for d in rows], float)
    speedup = np.array([d["default"]["ms"] / d["tuned_balanced"]["ms"] for d in rows])
    memratio = np.array([d["tuned_balanced"]["mb"] / d["default"]["mb"] for d in rows])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    axes[0].scatter(nnz, speedup, s=40, c=BLUE, alpha=0.8, edgecolors="none")
    axes[0].axhline(1.0, color=GRAY, ls="--", lw=1.2)
    axes[0].axhline(geomean(speedup), color=BLUE, ls=":", lw=1.5,
                    label=f"geomean {geomean(speedup):.2f}x")
    axes[0].set_ylabel("factor speedup\n(default / tuned)")
    axes[0].set_title("Auto-tuner (balanced) vs default over the corpus, by problem size")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)

    axes[1].scatter(nnz, memratio, s=40, c=CYAN, alpha=0.8, edgecolors="none")
    axes[1].axhline(1.0, color=GRAY, ls="--", lw=1.2)
    axes[1].axhline(geomean(memratio), color=CYAN, ls=":", lw=1.5,
                    label=f"geomean {geomean(memratio):.2f}x")
    axes[1].set_ylabel("peak-memory ratio\n(tuned / default)")
    axes[1].set_xlabel("nonzeros (nnz)")
    axes[1].set_xscale("log")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "autotune_vs_size.png", dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {outdir / 'autotune_vs_size.png'}")
    print(f"  balanced: speedup geomean {geomean(speedup):.3f}x, mem geomean {geomean(memratio):.3f}x  ({len(rows)} matrices)")
    # size split
    med = np.median(nnz)
    for lbl, mask in (("small (<median nnz)", nnz < med), ("large (>=median nnz)", nnz >= med)):
        print(f"    {lbl}: speedup {geomean(speedup[mask]):.3f}x  mem {geomean(memratio[mask]):.3f}x")


def plot_modes(data, outdir):
    fig, ax = plt.subplots(figsize=(8, 7))
    handles = []
    for cfg, label, color in MODES:
        t = np.array([d[cfg]["ms"] / d["default"]["ms"] for d in data.values() if cfg in d])
        m = np.array([d[cfg]["mb"] / d["default"]["mb"] for d in data.values() if cfg in d])
        ax.scatter(t, m, s=26, c=color, alpha=0.45, edgecolors="none")
        gt, gm = geomean(t), geomean(m)
        ax.scatter([gt], [gm], s=220, c=color, marker="*", edgecolors="white", linewidths=1.2, zorder=5)
        handles.append(Line2D([], [], color=color, marker="o", ls="", label=f"{label}  (geomean {gt:.2f}x time, {gm:.2f}x mem)"))
    ax.axvline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1.0)
    ax.scatter([1.0], [1.0], s=160, c=GRAY, marker="X", zorder=5)
    ax.set_xlabel("factor time relative to default")
    ax.set_ylabel("peak memory relative to default")
    ax.set_title("Auto-tuner Pareto modes vs default (★ = geomean, ✕ = default)")
    ax.grid(True, ls=":", alpha=0.4)
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    fig.savefig(outdir / "autotune_modes.png", dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {outdir / 'autotune_modes.png'}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/autotune.jsonl")
    bench_style.setup()
    data = load(path)
    if not data:
        print(f"[autotune] no valid records in {path}")
        return
    plot_vs_size(data, path.parent)
    plot_modes(data, path.parent)


if __name__ == "__main__":
    main()
