"""Phased-reuse figure: the analyze-once / factor-many amortization.

For each matrix the `phased_reuse` bench records the analyze and factor times.
A frequency sweep / Newton iteration factors K value sets that share one
sparsity pattern, so the cost is:

* analyze-once (RSLAB phased): `analyze + K * factor`
* analyze-each (naive):        `K * (analyze + factor)`

This plots cumulative wall-clock vs K for both, per matrix - the gap is the
amortized analysis, the reason RSLAB exposes a reusable `LdltSymbolic`.

Run:  python benches/phased_reuse.py [phased_reuse.jsonl]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_style
from bench_style import GRAY, BLUE, CYAN, PURPLE

COLORS = [BLUE, CYAN, PURPLE]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/phased_reuse.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    # Three representative matrices spanning the analyze fraction (low/mid/high).
    recs.sort(key=lambda r: r["analyze_ms"] / (r["analyze_ms"] + r["factor_ms"]))
    sel = [recs[0], recs[len(recs) // 2], recs[-1]] if len(recs) >= 3 else recs

    bench_style.setup()
    ks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    for r, c in zip(sel, COLORS):
        a, f = r["analyze_ms"], r["factor_ms"]
        once = a + ks * f
        each = ks * (a + f)
        label = f"{r['name']} ({100*a/(a+f):.0f}% analyze)"
        ax.plot(ks, each / 1000, ls="--", color=c, lw=1.6, alpha=0.7)
        ax.plot(ks, once / 1000, ls="-", color=c, lw=2.4, label=label)
    ax.plot([], [], ls="-", color=GRAY, lw=2.4, label="analyze once (phased)")
    ax.plot([], [], ls="--", color=GRAY, lw=1.6, label="analyze each (naive)")
    ax.set_xlabel("factorizations sharing the pattern (e.g. frequency points)")
    ax.set_ylabel("cumulative wall-clock [s]")
    ax.set_title("Phased reuse: analyze once, factor many")
    ax.grid(True, ls=":", alpha=0.4)
    out = path.parent / "phased_reuse.png"
    bench_style.legend_below(fig, ax=ax)
    fig.savefig(out, dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {out}")
    for r in recs:
        a, f = r["analyze_ms"], r["factor_ms"]
        sp = (a + f) / f  # asymptotic speedup of phased vs naive (K -> inf)
        print(f"  n={r['n']:>7}  analyze {a:>8.2f}ms  factor {f:>9.2f}ms  "
              f"asymptotic phased speedup x{sp:.2f}")


if __name__ == "__main__":
    main()
