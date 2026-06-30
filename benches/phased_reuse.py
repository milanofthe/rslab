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

GRAY = "#808080"
COLORS = ["#3b82f6", "#06b6d4", "#a855f7"]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out/phased_reuse.jsonl")
    recs = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["n"])
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 11,
    })
    ks = np.arange(1, 21)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for r, c in zip(recs, COLORS):
        a, f = r["analyze_ms"], r["factor_ms"]
        once = a + ks * f
        each = ks * (a + f)
        label = f"n={r['n']} (analyze {100*a/(a+f):.0f}% of one solve)"
        ax.plot(ks, each / 1000, ls="--", color=c, lw=1.6, alpha=0.7)
        ax.plot(ks, once / 1000, ls="-", color=c, lw=2.4, label=label)
    # Legend for the two line styles.
    ax.plot([], [], ls="-", color=GRAY, lw=2.4, label="analyze once (RSLAB phased)")
    ax.plot([], [], ls="--", color=GRAY, lw=1.6, label="analyze each (naive)")
    ax.set_xlabel("factorizations sharing the pattern (e.g. frequency points)")
    ax.set_ylabel("cumulative wall-clock [s]")
    ax.set_title("Phased reuse: analyze once, factor many (Helmholtz 3D sweep)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    out = path.parent / "phased_reuse.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, transparent=True)
    print(f"wrote {out}")
    for r in recs:
        a, f = r["analyze_ms"], r["factor_ms"]
        sp = (a + f) / f  # asymptotic speedup of phased vs naive (K -> inf)
        print(f"  n={r['n']:>7}  analyze {a:>8.2f}ms  factor {f:>9.2f}ms  "
              f"asymptotic phased speedup x{sp:.2f}")


if __name__ == "__main__":
    main()
