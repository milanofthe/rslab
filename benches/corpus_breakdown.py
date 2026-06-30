"""Corpus-based breakdown figures: where each solver spends its wall-clock, and
the a-priori factor-memory composition of RSLAB.

* `wct_breakdown.png` - per solver, the mean fraction of time in analyze / factor
  / solve over the corpus (so the composition is comparable across solvers; faer
  has no separate analyze phase).
* `memory_breakdown.png` - RSLAB's a-priori memory estimate (dense panels +
  compact factor + input/scratch, with the panel-freed live floor) per corpus
  matrix, sorted by size; read from a `RLA_BENCH_ESTIMATE=1` corpus run.

Run:  python benches/corpus_breakdown.py [corpus.jsonl] [corpus_estimate.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "ll": ("RSLAB left-looking", "#3b82f6"),
    "mf": ("RSLAB multifrontal", "#06b6d4"),
    "faer": ("faer LU", "#f59e0b"),
    "pardiso": ("MKL PARDISO", "#22c55e"),
}
ORDER = ["ll", "mf", "faer", "pardiso"]
GRAY = "#808080"


def setup():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 10,
    })


def wct_breakdown(corpus_path, outdir):
    recs = [json.loads(l) for l in open(corpus_path) if l.strip()]
    recs = [r for r in recs if r.get("metric") == "time" and r.get("res", 1.0) < 0.1]
    # Mean analyze/factor/solve fraction of total wall-clock, per solver.
    frac = {}
    for s in ORDER:
        rows = [r for r in recs if r["solver"] == s]
        parts = []
        for r in rows:
            tot = r["ana_ms"] + r["fac_ms"] + r["slv_ms"]
            if tot > 0:
                parts.append((r["ana_ms"] / tot, r["fac_ms"] / tot, r["slv_ms"] / tot))
        if parts:
            frac[s] = (np.mean(parts, axis=0), len(parts))
    present = [s for s in ORDER if s in frac]
    x = np.arange(len(present))
    ana = [frac[s][0][0] for s in present]
    fac = [frac[s][0][1] for s in present]
    slv = [frac[s][0][2] for s in present]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, ana, 0.6, label="analyze", color=GRAY, alpha=0.55)
    ax.bar(x, fac, 0.6, bottom=ana, label="factor", color="#3b82f6")
    ax.bar(x, slv, 0.6, bottom=[a + f for a, f in zip(ana, fac)], label="solve", color="#f59e0b")
    ax.set_ylabel("mean fraction of wall-clock")
    ax.set_title("Where each solver spends its time (corpus mean)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{STYLE[s][0]}\n(n={frac[s][1]})" for s in present], fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "wct_breakdown.png", dpi=140, transparent=True)
    print(f"wrote {outdir / 'wct_breakdown.png'}")


def memory_breakdown(est_path, outdir):
    if not Path(est_path).exists():
        print(f"[memory] {est_path} missing - run bench_suite with RLA_BENCH_ESTIMATE=1 family=corpus")
        return
    recs = [json.loads(l) for l in open(est_path) if l.strip()]
    recs = sorted(recs, key=lambda r: r["n"])
    names = [r["name"] for r in recs]
    x = np.arange(len(names))
    panels = [r["panels_mb"] for r in recs]
    factor = [r["factor_mb"] for r in recs]
    scratch = [r["scratch_mb"] for r in recs]
    floor = [r["freed_floor_mb"] for r in recs]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, panels, 0.7, label="dense panels", color="#3b82f6")
    ax.bar(x, factor, 0.7, bottom=panels, label="compact factor (CSC)", color="#06b6d4")
    ax.bar(x, scratch, 0.7, bottom=[p + f for p, f in zip(panels, factor)],
           label="input + scratch", color=GRAY, alpha=0.55)
    ax.plot(x, floor, "o--", color="#f59e0b", label="panel-freed live floor", lw=1.4, ms=4)
    ax.set_title("RSLAB a-priori factor-memory estimate over the corpus (by size)")
    ax.set_ylabel("estimated memory (MB)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in names], rotation=60, ha="right", fontsize=7)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "memory_breakdown.png", dpi=140, transparent=True)
    print(f"wrote {outdir / 'memory_breakdown.png'}")


def main():
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/corpus.jsonl")
    est = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benches/bench_out/corpus_estimate.jsonl")
    setup()
    wct_breakdown(corpus, corpus.parent)
    memory_breakdown(est, corpus.parent)


if __name__ == "__main__":
    main()
