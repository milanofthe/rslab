"""Incomplete-factor preconditioner + GMRES trade-off (the `pc` path).

Reads the JSONL from the ``precond_gmres`` bench: one record per drop tolerance
with the resulting factor fill, GMRES iteration count, and factor / solve / total
time. Produces one two-panel figure:

* left  - factor fill (memory) and GMRES iterations vs ``drop_tol`` (the core
  memory <-> iterations trade-off), with the exact factor as a reference line.
* right - factor / GMRES / total wall time vs ``drop_tol``, exposing the sweet
  spot where a lighter factor still solves in comparable total time.

Run:  python benches/precond_gmres_plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_style
from bench_style import GRAY, PURPLE, AMBER, GREEN

OUT = Path("benches/bench_out")
IN = OUT / "precond_gmres.jsonl"


def main():
    if not IN.exists():
        print("no precond_gmres.jsonl in", OUT)
        return
    recs = [json.loads(l) for l in open(IN) if l.strip()]
    if not recs:
        return
    exact = next((r for r in recs if r["drop_tol"] == 0.0), None)
    ilu = sorted((r for r in recs if r["drop_tol"] > 0.0), key=lambda r: r["drop_tol"])
    tau = [r["drop_tol"] for r in ilu]

    bench_style.setup()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # --- Left: fill (memory) vs GMRES iterations ---
    axL.plot(tau, [r["fill_mb"] for r in ilu], color=PURPLE, marker="v", lw=2.2, ms=7,
             label="factor fill")
    axL.set_xscale("log")
    axL.set_xlabel("drop tolerance")
    axL.set_ylabel("factor fill  [MB]", color=PURPLE)
    axL.tick_params(axis="y", labelcolor=PURPLE)
    axL.grid(True, ls=":", alpha=0.4)
    if exact:
        axL.axhline(exact["fill_mb"], ls="--", color=GRAY, lw=1.3, alpha=0.8)
        axL.text(tau[0], exact["fill_mb"], "  exact factor", color=GRAY, va="bottom", fontsize=8)
    axi = axL.twinx()
    axi.plot(tau, [r["iters"] for r in ilu], color=AMBER, marker="o", lw=2.2, ms=7,
             label="GMRES iters")
    axi.set_ylabel("GMRES iterations", color=AMBER)
    axi.tick_params(axis="y", labelcolor=AMBER)
    axL.set_title("Memory vs iterations")

    # --- Right: wall-time breakdown ---
    axR.plot(tau, [r["fac_ms"] for r in ilu], color=PURPLE, marker="v", lw=1.8, ms=6,
             alpha=0.65, label="factor")
    axR.plot(tau, [r["slv_ms"] for r in ilu], color=AMBER, marker="o", lw=1.8, ms=6,
             alpha=0.65, label="GMRES")
    axR.plot(tau, [r["total_ms"] for r in ilu], color=GREEN, marker="s", lw=2.4, ms=7,
             label="total")
    axR.set_xscale("log")
    axR.set_xlabel("drop tolerance")
    axR.set_ylabel("wall time  [ms]")
    axR.grid(True, ls=":", alpha=0.4)
    if exact:
        axR.axhline(exact["total_ms"], ls="--", color=GRAY, lw=1.3, alpha=0.8)
        axR.text(tau[-1], exact["total_ms"], "exact total  ", color=GRAY, va="bottom",
                 ha="right", fontsize=8)
    axR.set_title("Wall-time trade-off")

    n = recs[0]["n"]
    fig.suptitle(f"Incomplete-factor preconditioner + GMRES  (conv-diffusion, n={n})", color=GRAY)
    # One combined legend below.
    handles = [
        plt.Line2D([], [], color=PURPLE, marker="v", lw=2.2, label="factor fill / time"),
        plt.Line2D([], [], color=AMBER, marker="o", lw=2.2, label="GMRES iters / time"),
        plt.Line2D([], [], color=GREEN, marker="s", lw=2.4, label="total time"),
        plt.Line2D([], [], ls="--", color=GRAY, lw=1.3, label="exact factor"),
    ]
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, OUT / "precond_gmres.png")


if __name__ == "__main__":
    main()
