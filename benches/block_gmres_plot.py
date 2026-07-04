"""Block-GMRES (BCGS2) scaling figures for the report and README.

Reads the JSONL emitted by the ``block_gmres_scaling`` bench (``RLA_JSON=...``):
``grid`` records (per-RHS cost over the full thread ladder x RHS count) and
``rhs`` records (per-RHS cost over the RHS count at a fixed thread count), tagged
by ``variant`` - ``bcgs2`` (the current, regenerated build) and ``mgs`` (the
committed v0.11 pre-BCGS2 reference). Produces three figures:

* ``block_gmres_scaling``    - strong-scaling speedup vs threads, per RHS count.
* ``block_gmres_per_rhs``    - per-RHS cost vs RHS count at 12 threads.
* ``block_gmres_efficiency`` - parallel efficiency vs threads, per RHS count.

BCGS2 is drawn solid, the MGS reference dashed; the RHS count is encoded by shade.

Run:  python benches/block_gmres_plot.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import bench_style
from bench_style import GRAY

OUT = Path("benches/bench_out")
# RHS count -> color (a blue family: darker = more RHS).
S_COLOR = {1: "#93c5fd", 4: "#3b82f6", 16: "#1d4ed8", 2: "#bfdbfe", 8: "#2563eb", 32: "#1e3a8a"}
VARIANT = {  # variant -> (label, linestyle, marker, alpha)
    "bcgs2": ("BCGS2 (v0.15)", "-", "o", 1.0),
    "mgs": ("MGS (v0.11)", "--", "x", 0.6),
}


def load():
    recs = []
    for name in ("block_gmres_bcgs2.jsonl", "block_gmres_mgs_ref.jsonl"):
        p = OUT / name
        if p.exists():
            recs += [json.loads(l) for l in open(p) if l.strip()]
    return recs


def grouped(recs, mode, value):
    """(variant, s) -> {threads: value} for the given record mode."""
    g = defaultdict(dict)
    for r in recs:
        if r.get("mode") != mode:
            continue
        g[(r["variant"], r["s"])][int(r["threads"])] = r[value]
    return g


def variant_legend(present):
    return [Line2D([], [], color=GRAY, ls=VARIANT[v][1], marker=VARIANT[v][2],
                   lw=2.0, label=VARIANT[v][0]) for v in present]


def s_legend(s_vals):
    return [Line2D([], [], color=S_COLOR[s], lw=2.4, label=f"{s} RHS") for s in s_vals]


def plot_scaling(recs):
    """Speedup vs threads (from grid), per RHS count, BCGS2 vs MGS."""
    g = grouped(recs, "grid", "speedup")
    if not g:
        return
    s_vals = sorted({s for _, s in g})
    variants = [v for v in ("bcgs2", "mgs") if any(vr == v for vr, _ in g)]
    bench_style.setup()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    tmax = max(t for d in g.values() for t in d)
    ax.plot([1, tmax], [1, tmax], ls=":", color=GRAY, lw=1.2, alpha=0.7, zorder=1)
    for (variant, s), d in sorted(g.items()):
        _, ls, mk, al = VARIANT[variant]
        th = sorted(d)
        ax.plot(th, [d[t] for t in th], color=S_COLOR[s], ls=ls, marker=mk,
                lw=2.0, alpha=al, ms=6, zorder=3)
    ax.set_xlabel("worker threads")
    ax.set_ylabel("block-solve speedup vs 1 thread")
    ax.set_title("Block GMRES strong scaling: BCGS2 vs MGS")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xticks(sorted({t for d in g.values() for t in d}))
    handles = s_legend(s_vals) + variant_legend(variants) + [
        Line2D([], [], ls=":", color=GRAY, lw=1.2, label="ideal (linear)")]
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, OUT / "block_gmres_scaling.png")


def plot_efficiency(recs):
    """Parallel efficiency (%) vs threads (speedup / threads), per RHS count."""
    g = grouped(recs, "grid", "speedup")
    if not g:
        return
    s_vals = sorted({s for _, s in g})
    variants = [v for v in ("bcgs2", "mgs") if any(vr == v for vr, _ in g)]
    bench_style.setup()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for (variant, s), d in sorted(g.items()):
        _, ls, mk, al = VARIANT[variant]
        th = sorted(d)
        ax.plot(th, [100.0 * d[t] / t for t in th], color=S_COLOR[s], ls=ls,
                marker=mk, lw=2.0, alpha=al, ms=6)
    ax.axhline(100, ls=":", color=GRAY, lw=1.2, alpha=0.7)
    ax.set_xlabel("worker threads")
    ax.set_ylabel("parallel efficiency  (speedup / threads)  [%]")
    ax.set_title("Block GMRES parallel efficiency")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xticks(sorted({t for d in g.values() for t in d}))
    handles = s_legend(s_vals) + variant_legend(variants)
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, OUT / "block_gmres_efficiency.png")


def plot_per_rhs(recs, threads=12):
    """Per-RHS cost vs RHS count at a fixed thread count, BCGS2 vs MGS."""
    g = grouped(recs, "rhs", "per_rhs_ms")
    # (variant, s) -> {threads: per_rhs}; pull the fixed-thread slice.
    series = defaultdict(dict)  # variant -> {s: per_rhs}
    for (variant, s), d in g.items():
        if threads in d:
            series[variant][s] = d[threads]
    if not series:
        return
    variants = [v for v in ("bcgs2", "mgs") if v in series]
    bench_style.setup()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for variant in variants:
        label, ls, mk, al = VARIANT[variant]
        s_vals = sorted(series[variant])
        color = "#3b82f6" if variant == "bcgs2" else "#f59e0b"
        ax.plot(s_vals, [series[variant][s] for s in s_vals], color=color, ls=ls,
                marker=mk, lw=2.2, alpha=al, ms=7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({s for v in series for s in series[v]}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("right-hand sides  (block width s)")
    ax.set_ylabel(f"time per RHS  [ms]   (at {threads} threads)")
    ax.set_title("Block GMRES per-RHS cost vs block width")
    ax.grid(True, ls=":", alpha=0.4)
    bench_style.legend_below(fig, ax=ax)
    bench_style.save(fig, OUT / "block_gmres_per_rhs.png")


def main():
    recs = load()
    if not recs:
        print("no block-GMRES JSONL found in", OUT)
        return
    plot_scaling(recs)
    plot_efficiency(recs)
    plot_per_rhs(recs, threads=12)


if __name__ == "__main__":
    main()
