"""Block-GMRES within-cycle deflation: the shrinking-panel effect.

Reads the JSONL from the ``deflation_study`` bench and produces one two-panel figure:

* left  - total operator column-applies with within-cycle deflation vs the full-width
  bound (``calls x s``, the work a schedule with no mid-cycle compaction would do),
  grouped by block width s;
* right - the per-step active-panel width for the widest block (the "staircase":
  columns compact out of the batched applies the instant they converge, so later
  steps run narrower).

Run:  python benches/deflation_study_plot.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_style
from bench_style import GRAY, BLUE, CYAN

OUT = Path("benches/bench_out")
IN = OUT / "deflation_study.jsonl"


def main():
    if not IN.exists():
        print("no deflation_study.jsonl in", OUT)
        return
    recs = [json.loads(l) for l in open(IN) if l.strip()]
    summ = sorted((r for r in recs if r.get("kind") == "summary"), key=lambda r: r["s"])
    widths_rec = next((r for r in recs if r.get("kind") == "widths"), None)
    if not summ:
        return
    n = summ[0]["n"]

    bench_style.setup()
    fig, (axL, axR) = bench_style.two_panel()

    # --- Left: measured column-applies vs full-width bound, per block width s. ---
    ss = [r["s"] for r in summ]
    meas = [r["op_cols"] for r in summ]
    full = [r["op_full"] for r in summ]
    x = np.arange(len(ss))
    w = 0.38
    axL.bar(x - w / 2, full, w, color=GRAY, alpha=0.7, label="full-width bound (calls x s)")
    axL.bar(x + w / 2, meas, w, color=BLUE, label="with within-cycle deflation")
    for xi, m, f in zip(x, meas, full):
        axL.text(xi + w / 2, m, f"{m/f:.2f}x", ha="center", va="bottom", fontsize=8, color=BLUE)
    axL.set_xticks(x)
    axL.set_xticklabels([str(s) for s in ss])
    axL.set_xlabel("block width  s  (right-hand sides)")
    axL.set_ylabel("operator column-applies")
    axL.grid(True, axis="y", ls=":", alpha=0.4)
    axL.set_title("Batched-apply work saved")

    # --- Right: the shrinking-panel staircase for the widest block. ---
    if widths_rec is not None:
        ws = widths_rec["widths"]
        axR.step(range(len(ws)), ws, where="post", color=CYAN, lw=2.0)
        axR.fill_between(range(len(ws)), ws, step="post", color=CYAN, alpha=0.18)
        axR.set_xlabel("block apply-call index")
        axR.set_ylabel("active panel width")
        axR.set_ylim(0, widths_rec["s"] + 0.5)
        axR.grid(True, ls=":", alpha=0.4)
        axR.set_title(f"Panel drains mid-cycle  (s={widths_rec['s']})")

    fig.suptitle(f"Block-GMRES within-cycle deflation  (multi-rate, n={n})", color=GRAY)
    bench_style.legend_below(fig, ax=axL)
    bench_style.save(fig, OUT / "deflation_study.png")


if __name__ == "__main__":
    main()
