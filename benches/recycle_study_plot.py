"""GCRO-DR Krylov subspace recycling on a sequence of related solves.

Reads the JSONL from the ``recycle_study`` bench (one record per method/k/step of a
slowly-varying operator + rotating RHS sequence) and produces one two-panel figure:

* left  - cumulative wall-clock time vs solve number;
* right - cumulative GMRES iterations vs solve number,

for cold (x0=0), warm (x0=prev), and GCRO-DR(k) recycling (warm + a Recycle handle
carried across the sequence). The iteration panel shows the headline recycling win;
the wall-clock panel shows the (smaller) net time win after the per-iteration
recycling overhead.

Run:  python benches/recycle_study_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_style
from bench_style import GRAY, AMBER, BLUE_SHADES

OUT = Path("benches/bench_out")
IN = OUT / "recycle_study.jsonl"


def series(recs, method, k=0):
    rows = sorted((r for r in recs if r["method"] == method and r.get("k", 0) == k),
                  key=lambda r: r["step"])
    x = [r["step"] + 1 for r in rows]
    return x, [r["cum_ms"] for r in rows], [r["cum_iters"] for r in rows]


def main():
    if not IN.exists():
        print("no recycle_study.jsonl in", OUT)
        return
    recs = [json.loads(l) for l in open(IN) if l.strip()]
    if not recs:
        return
    n = recs[0]["n"]
    ks = sorted({r["k"] for r in recs if r["method"] == "gcrodr"})

    bench_style.setup()
    fig, (axL, axR) = bench_style.two_panel()

    # (label, color, marker, cumulative-ms, cumulative-iters, x)
    lines = []
    xc, mc, ic = series(recs, "cold")
    lines.append(("cold (x0=0)", GRAY, "x", xc, mc, ic))
    xw, mw, iw = series(recs, "warm")
    lines.append(("warm start", AMBER, "s", xw, mw, iw))
    for j, k in enumerate(ks):
        xk, mk, ik = series(recs, "gcrodr", k)
        color = BLUE_SHADES[min(j, len(BLUE_SHADES) - 1)]
        lines.append((f"GCRO-DR(k={k})", color, "o", xk, mk, ik))

    for label, color, marker, x, ms, it in lines:
        axL.plot(x, ms, color=color, marker=marker, lw=2.0, ms=6, label=label)
        axR.plot(x, it, color=color, marker=marker, lw=2.0, ms=6, label=label)

    axL.set_xlabel("solve number in sequence")
    axL.set_ylabel("cumulative wall-clock  [ms]")
    axL.grid(True, ls=":", alpha=0.4)
    axL.set_title("Wall-clock over the sequence")

    axR.set_xlabel("solve number in sequence")
    axR.set_ylabel("cumulative GMRES iterations")
    axR.grid(True, ls=":", alpha=0.4)
    axR.set_title("Iterations over the sequence")

    fig.suptitle(f"GCRO-DR subspace recycling  (stagnation spectrum, n={n})", color=GRAY)
    bench_style.legend_below(fig, ax=axL)
    bench_style.save(fig, OUT / "recycle_study.png")


if __name__ == "__main__":
    main()
