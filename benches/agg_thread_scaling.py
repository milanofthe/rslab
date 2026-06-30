"""Aggregated thread-scaling across the corpus: mean +/- std speedup vs cores.

Reads the thread-scaling sweep (RLA_SWEEP_THREADS_ONLY output: per matrix a
factor time at each worker count) and aggregates it into a single curve - the
mean parallel speedup over the corpus at each thread count, with a +/- one
standard-deviation band - so the *typical* scaling (and its spread) is visible
at a glance, against the ideal linear line.

Run:  python benches/agg_thread_scaling.py [sweep_threads.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_style
from bench_style import GRAY, BLUE


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/sweep_threads.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    # Suspend-artifact guard: no real factor here exceeds tens of seconds.
    recs = [r for r in recs if r["metrics"]["factor_ms"] < 1_000_000]

    by = defaultdict(dict)  # matrix -> {threads: ms}
    for r in recs:
        by[r["matrix"]][r["params"]["threads"]] = r["metrics"]["factor_ms"]

    # Per-matrix speedup curves, then aggregate per thread count.
    per_t = defaultdict(list)  # threads -> [speedup over matrices]
    n_curves = 0
    for d in by.values():
        if 1 not in d or len(d) < 3:
            continue
        n_curves += 1
        t1 = d[1]
        for t, ms in d.items():
            if ms > 0:
                per_t[t].append(t1 / ms)

    # Keep only thread counts present in (almost) every matrix, so each point
    # aggregates the same population - otherwise the reduced ladder of the big
    # matrices (1,4,8,12 only) biases the intermediate points downward.
    threads = sorted(t for t in per_t if len(per_t[t]) >= 0.9 * n_curves)
    mean = np.array([np.mean(per_t[t]) for t in threads])
    lo = np.array([np.min(per_t[t]) for t in threads])
    hi = np.array([np.max(per_t[t]) for t in threads])
    counts = [len(per_t[t]) for t in threads]

    bench_style.setup()
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    tmax = max(threads)
    ax.plot([1, tmax], [1, tmax], ls="--", color=GRAY, lw=1.3, alpha=0.7, label="ideal (linear)")
    ax.fill_between(threads, lo, hi, color=BLUE, alpha=0.18, lw=0,
                    label="min-max over corpus")
    ax.plot(threads, lo, color=BLUE, lw=1, alpha=0.5, ls=":")
    ax.plot(threads, hi, color=BLUE, lw=1, alpha=0.5, ls=":")
    ax.plot(threads, mean, color=BLUE, lw=2.4, marker="o", zorder=3,
            label=f"mean speedup (n={n_curves} matrices)")
    ax.set_xlabel("worker threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("Aggregated thread scaling over the corpus (mean, min-max band)")
    ax.set_xticks(threads)
    ax.grid(True, ls=":", alpha=0.4)
    out = path.parent / "thread_scaling_agg.png"
    bench_style.legend_below(fig, ax=ax)
    fig.savefig(out, dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"{'threads':>8}{'mean':>8}{'min':>7}{'max':>7}{'n':>5}")
    for t, m, l, h, c in zip(threads, mean, lo, hi, counts):
        print(f"{t:>8}{m:>8.2f}{l:>7.2f}{h:>7.2f}{c:>5}")


if __name__ == "__main__":
    main()
