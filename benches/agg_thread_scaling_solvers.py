"""Aggregated thread scaling per solver: mean and min-max band over the corpus,
for RSLAB left-looking / multifrontal, faer and MKL PARDISO.

Reads a thread-sweep JSONL emitted by `bench_suite` run at several
`RAYON_NUM_THREADS` values (so each record carries its `threads`), forms the
per-matrix speedup curve `t(1)/t(p)` for every solver, and aggregates across the
corpus into a mean curve with a min-max band - one panel per solver, so each
solver's typical scaling and its spread (including the matrices that *regress*
under more threads) are directly comparable.

Run:  python benches/agg_thread_scaling_solvers.py [corpus_threads.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import bench_style
from bench_style import GRAY, SOLVERS

ORDER = ["ll", "mf", "faer", "pardiso"]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/corpus_threads.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get("metric") == "time" and r.get("fac_ms", 0) > 0
            and r.get("res", 1.0) < 0.1]

    # (solver, matrix) -> {threads: ms}
    cur = defaultdict(dict)
    for r in recs:
        cur[(r["solver"], r["name"])][int(r["threads"])] = r["fac_ms"]

    bench_style.setup()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    print(f"{'solver':<20}{'thr':>5}{'mean':>8}{'min':>7}{'max':>7}{'n':>5}")
    for ax, s in zip(axes.flat, ORDER):
        label, color = SOLVERS[s][0], SOLVERS[s][1]
        per_t = defaultdict(list)
        n_curves = 0
        for (sv, _name), d in cur.items():
            if sv != s or 1 not in d or len(d) < 3:
                continue
            n_curves += 1
            t1 = d[1]
            for t, ms in d.items():
                per_t[t].append(t1 / ms)
        if not per_t:
            ax.set_title(f"{label}  (no data)")
            continue
        threads = sorted(t for t in per_t if len(per_t[t]) >= 0.9 * n_curves)
        mean = np.array([np.mean(per_t[t]) for t in threads])
        lo = np.array([np.min(per_t[t]) for t in threads])
        hi = np.array([np.max(per_t[t]) for t in threads])
        tmax = max(threads)
        ax.plot([1, tmax], [1, tmax], ls="--", color=GRAY, lw=1.2, alpha=0.7)
        ax.fill_between(threads, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(threads, mean, color=color, lw=2.2, marker="o", zorder=3)
        ax.set_title(f"{label}  (n={n_curves})")
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_xticks(threads)
        for t, m, l, h in zip(threads, mean, lo, hi):
            print(f"{label:<20}{t:>5}{m:>8.2f}{l:>7.2f}{h:>7.2f}{len(per_t[t]):>5}")
    for ax in axes[-1]:
        ax.set_xlabel("worker threads")
    for ax in axes[:, 0]:
        ax.set_ylabel("speedup vs 1 thread")
    fig.suptitle("Thread scaling per solver over the corpus (mean, min-max band)", color=GRAY)
    handles = [
        Line2D([], [], color=GRAY, lw=2.2, marker="o", label="mean speedup"),
        Patch(facecolor=GRAY, alpha=0.3, label="min-max over corpus"),
        Line2D([], [], ls="--", color=GRAY, lw=1.2, label="ideal (linear)"),
    ]
    out = path.parent / "thread_scaling_solvers.png"
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, out)


if __name__ == "__main__":
    main()
