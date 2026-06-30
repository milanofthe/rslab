"""RSLAB-only breakdown figures over the corpus (LL = left-looking, MF =
multifrontal; the cross-solver comparison lives in the scaling/thread figures).

1. `memory_breakdown.png`   - factor memory: a-priori estimate (conservative
   upper bound + panel-freed estimate) vs the measured peak of *both* RSLAB
   paths. Grouped bars (log axis - never stacked on a log scale).
2. `memory_composition.png` - the a-priori estimate's composition (dense panels /
   compact factor / input+scratch) as a **normalized, linear** stacked bar per
   matrix - what fraction of the estimate each part is.
3. `runtime_stage_breakdown.png` - the analyze / factor / solve split per matrix,
   **normalized, linear**, for LL and MF - where each path spends its time.

Run:  python benches/corpus_breakdown.py [corpus.jsonl] [corpus_estimate.jsonl]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAY = "#808080"


def setup():
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 10,
    })


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def measured_peak(corpus, solver):
    return {r["name"]: r["mem_mb"] for r in corpus
            if r.get("solver") == solver and r.get("metric") == "mem" and r.get("mem_mb", 0) > 0}


def memory_breakdown(est, corpus, outdir):
    """Grouped (log axis): conservative upper bound + panel-freed estimate vs the
    measured peak of both RSLAB paths."""
    ll, mf = measured_peak(corpus, "ll"), measured_peak(corpus, "mf")
    rows = sorted((r for r in est if r["name"] in ll or r["name"] in mf), key=lambda r: r["n"])
    if not rows:
        print("[memory] no matrices with estimate + measured peak")
        return
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    worst = [r["transient_mb"] for r in rows]
    floor = [r["freed_floor_mb"] for r in rows]
    m_ll = [ll.get(r["name"], np.nan) for r in rows]
    m_mf = [mf.get(r["name"], np.nan) for r in rows]
    w = 0.21
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - 1.5 * w, worst, w, label="worst-case estimate (all panels)", color=GRAY, alpha=0.6)
    ax.bar(x - 0.5 * w, floor, w, label="panel-freed estimate", color="#a855f7", alpha=0.8)
    ax.bar(x + 0.5 * w, m_ll, w, label="measured peak - LL (left-looking)", color="#3b82f6")
    ax.bar(x + 1.5 * w, m_mf, w, label="measured peak - MF (multifrontal)", color="#06b6d4")
    ax.set_title("RSLAB factor memory: a-priori estimate vs measured (LL & MF)")
    ax.set_ylabel("memory (MB)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "memory_breakdown.png", dpi=140, transparent=True)
    print(f"wrote {outdir / 'memory_breakdown.png'}")
    g = lambda xs: math.exp(sum(math.log(v) for v in xs) / len(xs))
    ov_ll = [w_ / ll[r["name"]] for w_, r in zip(worst, rows) if r["name"] in ll]
    print(f"  worst-case / measured-LL: geomean {g(ov_ll):.2f}x ({len(ov_ll)} matrices, >=1 = conservative)")


def memory_composition(est, outdir):
    """Normalized, linear: each part's fraction of the a-priori estimate."""
    rows = sorted(est, key=lambda r: r["n"])
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    tot = [max(r["panels_mb"] + r["factor_mb"] + r["scratch_mb"], 1e-9) for r in rows]
    panels = [r["panels_mb"] / t for r, t in zip(rows, tot)]
    factor = [r["factor_mb"] / t for r, t in zip(rows, tot)]
    scratch = [r["scratch_mb"] / t for r, t in zip(rows, tot)]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, panels, 0.8, label="dense panels", color="#3b82f6")
    ax.bar(x, factor, 0.8, bottom=panels, label="compact factor (CSC)", color="#06b6d4")
    ax.bar(x, scratch, 0.8, bottom=[p + f for p, f in zip(panels, factor)],
           label="input + scratch", color=GRAY, alpha=0.6)
    ax.set_title("RSLAB a-priori factor-memory estimate: composition (normalized)")
    ax.set_ylabel("fraction of estimate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "memory_composition.png", dpi=140, transparent=True)
    print(f"wrote {outdir / 'memory_composition.png'}")


def runtime_stage_breakdown(corpus, outdir):
    """Normalized, linear analyze/factor/solve split per matrix, for LL and MF."""
    def split(solver):
        rows = [r for r in corpus if r.get("solver") == solver and r.get("metric") == "time"
                and r.get("res", 1.0) < 0.1]
        rows.sort(key=lambda r: r["n"])
        out = []
        for r in rows:
            tot = r["ana_ms"] + r["fac_ms"] + r["slv_ms"]
            if tot > 0:
                out.append((r["name"], r["ana_ms"] / tot, r["fac_ms"] / tot, r["slv_ms"] / tot))
        return out

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    for ax, (solver, title) in zip(axes, [("ll", "left-looking"), ("mf", "multifrontal")]):
        rows = split(solver)
        names = [r[0] for r in rows]
        x = np.arange(len(names))
        ana = [r[1] for r in rows]
        fac = [r[2] for r in rows]
        slv = [r[3] for r in rows]
        ax.bar(x, ana, 0.8, label="analyze", color=GRAY, alpha=0.6)
        ax.bar(x, fac, 0.8, bottom=ana, label="factor", color="#3b82f6")
        ax.bar(x, slv, 0.8, bottom=[a + f for a, f in zip(ana, fac)], label="solve", color="#f59e0b")
        ax.set_title(f"RSLAB {title}: analyze / factor / solve (normalized)")
        ax.set_ylabel("fraction of wall-clock")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "runtime_stage_breakdown.png", dpi=140, transparent=True)
    print(f"wrote {outdir / 'runtime_stage_breakdown.png'}")


def main():
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/corpus.jsonl")
    est_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benches/bench_out/corpus_estimate.jsonl")
    setup()
    corpus = load(corpus_path)
    est = load(est_path)
    if est:
        memory_breakdown(est, corpus, corpus_path.parent)
        memory_composition(est, corpus_path.parent)
    else:
        print(f"[breakdown] {est_path} missing - run bench_suite RLA_BENCH_ESTIMATE=1 family=corpus")
    runtime_stage_breakdown(corpus, corpus_path.parent)


if __name__ == "__main__":
    main()
