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


def memory_breakdown(est_path, corpus_path, outdir):
    """Grouped (not stacked - this is a log axis) factor-memory comparison per
    matrix: the conservative a-priori upper bound (all panels resident), the
    a-priori panel-freed estimate (what the low-memory path should hold), and the
    measured RSLAB left-looking peak. The measured bar should sit between the two
    estimates - validating that the estimate brackets reality and never
    under-predicts."""
    if not Path(est_path).exists():
        print(f"[memory] {est_path} missing - run bench_suite with RLA_BENCH_ESTIMATE=1 family=corpus")
        return
    est = {r["name"]: r for r in (json.loads(l) for l in open(est_path) if l.strip())}
    # Measured RSLAB left-looking peak from the memory pass.
    measured = {}
    if Path(corpus_path).exists():
        for r in (json.loads(l) for l in open(corpus_path) if l.strip()):
            if r.get("solver") == "ll" and r.get("metric") == "mem" and r.get("mem_mb", 0) > 0:
                measured[r["name"]] = r["mem_mb"]
    # Keep matrices with all three numbers, sorted by size.
    rows = [est[n] for n in est if n in measured]
    rows.sort(key=lambda r: r["n"])
    if not rows:
        print("[memory] no matrices with estimate + measured peak")
        return
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    worst = [r["transient_mb"] for r in rows]
    floor = [r["freed_floor_mb"] for r in rows]
    meas = [measured[r["name"]] for r in rows]
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w, worst, w, label="worst-case estimate (all panels)", color=GRAY, alpha=0.65)
    ax.bar(x, floor, w, label="panel-freed estimate", color="#3b82f6")
    ax.bar(x + w, meas, w, label="measured peak (RSLAB left-looking)", color="#22c55e")
    ax.set_title("RSLAB factor memory: a-priori estimate vs measured (per matrix, by size)")
    ax.set_ylabel("memory (MB)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "memory_breakdown.png", dpi=140, transparent=True)
    print(f"wrote {outdir / 'memory_breakdown.png'}")
    # Console: measured vs the two estimates (the bracketing check).
    import math
    over = [w_ / m for w_, m in zip(worst, meas) if m > 0]
    print(f"  worst-case / measured: geomean {math.exp(sum(map(math.log, over))/len(over)):.2f}x "
          f"(should be >= 1; conservative upper bound, {len(rows)} matrices)")


def main():
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/corpus.jsonl")
    est = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benches/bench_out/corpus_estimate.jsonl")
    setup()
    wct_breakdown(corpus, corpus.parent)
    memory_breakdown(est, corpus, corpus.parent)


if __name__ == "__main__":
    main()
