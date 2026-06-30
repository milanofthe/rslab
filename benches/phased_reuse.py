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
from bench_style import GRAY, BLUE_SHADES

COLORS = BLUE_SHADES


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/phased_reuse.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    # Three representative matrices spanning the analyze fraction (low/mid/high).
    recs.sort(key=lambda r: r["analyze_ms"] / (r["analyze_ms"] + r["factor_ms"]))
    sel = [recs[0], recs[len(recs) // 2], recs[-1]] if len(recs) >= 3 else recs

    bench_style.setup()
    ks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    for r, c in zip(sel, COLORS):
        a, f = r["analyze_ms"], r["factor_ms"]
        # Speedup of reusing the analysis vs re-analyzing each of K factorizations:
        # K*(a+f) / (a + K*f), rising from 1 to the asymptote 1 + a/f.
        speedup = ks * (a + f) / (a + ks * f)
        asymp = (a + f) / f
        ax.plot(ks, speedup, color=c, lw=2.4, marker="o", ms=4,
                label=f"{r['name']}  (analyze {100*a/(a+f):.0f}%  →  max {asymp:.2f}x)")
        ax.axhline(asymp, color=c, ls=":", lw=1.2, alpha=0.5)
    ax.axhline(1.0, color=GRAY, lw=1, alpha=0.5)
    ax.set_xlabel("factorizations sharing the pattern  (K, e.g. frequency points)")
    ax.set_ylabel("speedup: re-analyze each  /  analyze once")
    ax.set_title("Phased reuse: speedup from reusing the analysis over K factorizations")
    ax.set_xlim(1, ks[-1])
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
