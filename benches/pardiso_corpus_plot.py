"""Figures for RSLAB against MKL PARDISO on the real-system corpus.

Reads ``bench_out/pardiso_corpus.jsonl`` (``benches/pardiso_corpus.py``) and writes
into ``docs/figures/``:

* ``wct_breakdown.png``   - wall time per stage, per system and solver,
* ``wct_breakdown_social.png`` - its share card on a few systems,
* ``estimate_accuracy.png`` - RSLAB's memory estimate against the measurement,

and prints the per-class table of the README: wall time divided by PARDISO's,
geomean per matrix class, for factor, refactor, solve and one-shot.

Usage: ``python benches/pardiso_corpus_plot.py [bench_out/pardiso_corpus.jsonl]``
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import bench_style as st

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "figures"
RSLAB, PARDISO = st.SOLVERS["auto"][1], st.SOLVERS["pardiso"][1]
CLASSES = [("fem_", "FEM, curl-curl (rapidfem)"), ("sane_", "power grid (SANE)"),
           ("mom_", "MoM near field (rapidmom)"), ("ss_", "circuit, KLU path (SuiteSparse)")]
METRICS = [("factor", "factor"), ("refactor", "refactor (new values)"), ("solve", "solve"),
           ("oneshot", "one-shot (analysis + factor + solve)")]
# The share card of the breakdown: one or two systems per class, small to large.
CARD_SYSTEMS = ["fem_iris_filter_f00", "fem_microstrip_line_f00", "fem_patch_antenna_f00",
                "sane_ibmpg2_dc", "mom_xformer_D350", "mom_opamp_c350n"]
STAGES = [("analyze", "analysis", st.DARKGRAY), ("scale", "scaling", st.AMBER),
          ("factor", "factorization", st.BLUE), ("solve-layout", "solve layout", st.CYAN),
          ("solve", "solve", st.GREEN)]


def load(path):
    """{system: {solver: row}} over the rows without errors, last row wins."""
    out = {}
    for line in open(path):
        row = json.loads(line)
        if "error" not in row:
            out.setdefault(row["system"], {})[row["solver"]] = row
    return {s: v for s, v in out.items() if {"rslab", "pardiso"} <= v.keys()}


def med(row, key):
    if key == "oneshot":
        return statistics.median(r["analyze"] + r["factor"] + r["solve"] for r in row["runs"])
    return statistics.median(r[key] for r in row["runs"])


def cls(system):
    return next(i for i, (p, _) in enumerate(CLASSES) if system.startswith(p))


def short(system):
    return system.removeprefix("fem_").removeprefix("sane_").removeprefix("mom_") \
        .removeprefix("ss_").removesuffix("_f00").removesuffix("_dc")


def ordered(data):
    return sorted(data, key=lambda s: (cls(s), data[s]["rslab"]["n"]))


def geomean(xs):
    return float(np.exp(np.mean(np.log(xs))))


def class_table(data):
    """{(class, metric): geomean over the class of RSLAB / PARDISO wall time}."""
    return {(c, key): geomean([med(data[s]["rslab"], key) / med(data[s]["pardiso"], key)
                               for s in data if cls(s) == c])
            for c in sorted({cls(s) for s in data}) for key, _ in METRICS}


def breakdown(data, ax, names):
    """Per system two stacked bars normalized to PARDISO's one-shot time."""
    h = 0.36
    for yi, s in enumerate(names):
        ref = med(data[s]["pardiso"], "oneshot")
        rs = data[s]["rslab"]
        stages = rs["report"].get("stages", {})
        # Stage times come from the last run; scale them to the median factor call.
        in_factor = sum(stages.get(k, 0.0) for k in ("scale", "factor", "solve-layout")) or 1.0
        f = med(rs, "factor") / in_factor
        parts = {"analyze": med(rs, "analyze"), "scale": stages.get("scale", 0.0) * f,
                 "factor": stages.get("factor", 0.0) * f, "solve-layout": stages.get("solve-layout", 0.0) * f,
                 "solve": med(rs, "solve")}
        pa = data[s]["pardiso"]
        pparts = {"analyze": med(pa, "analyze"), "factor": med(pa, "factor"), "solve": med(pa, "solve")}
        for y, pieces in ((yi - h / 2 - 0.02, parts), (yi + h / 2 + 0.02, pparts)):
            left = 0.0
            for key, _, color in STAGES:
                if key in pieces:
                    ax.barh(y, pieces[key] / ref, h, left=left, color=color)
                    left += pieces[key] / ref
            ax.text(left + 0.02, y, f"{left * ref:.2f} s", va="center", fontsize=6.5, color=st.GRAY)
    ax.set_yticks([v for i in range(len(names)) for v in (i - h / 2 - 0.02, i + h / 2 + 0.02)])
    ax.set_yticklabels([f"{short(s)}  {lab}" for s in names for lab in ("RSLAB", "PARDISO")], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(1.0, color=PARDISO, linewidth=1.0, alpha=0.6, zorder=0)
    ax.set_xlabel("wall time / PARDISO one-shot (analysis + factorization + solve)", fontsize=9)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    st.despine(ax)
    return [Patch(facecolor=c, label=l) for _, l, c in STAGES]


def estimates(data):
    """RSLAB's analysis-time memory estimate against the measurement."""
    fig, (ax_f, ax_p) = st.two_panel(figsize=(10.0, 4.2))
    ratios = {"factor": [], "peak": []}
    seen, lo, hi = set(), np.inf, 0.0
    for s in ordered(data):
        rep = data[s]["rslab"]["report"]
        est = rep.get("estimate")
        if not est or not rep.get("factor_bytes"):
            continue
        color = [RSLAB, st.CYAN, st.AMBER, st.GREEN][cls(s)]
        for ax, key, e, m in ((ax_f, "factor", est["factor_bytes"], rep["factor_bytes"]),
                              (ax_p, "peak", est["transient_peak_bytes"], data[s]["rslab"]["peak_mb"] * 2**20)):
            ax.scatter(m / 1e9, e / 1e9, color=color, s=22, zorder=3)
            ratios[key].append(e / m)
            lo, hi = min(lo, m / 1e9, e / 1e9), max(hi, m / 1e9, e / 1e9)
        seen.add(cls(s))
    for ax, key, what in ((ax_f, "factor", "factor storage"), (ax_p, "peak", "peak memory")):
        lo, hi = lo / 2, hi * 2
        ax.plot([lo, hi], [lo, hi], color=st.GRAY, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"measured {what} [GB]", fontsize=9)
        ax.set_ylabel(f"estimated {what} [GB]", fontsize=9)
        r = ratios[key]
        if r:
            ax.text(0.04, 0.94, f"estimate / measured: geomean {geomean(r):.2f}, range {min(r):.2f} to {max(r):.2f}",
                    transform=ax.transAxes, fontsize=8, color=st.GRAY, va="top")
        ax.grid(alpha=0.3, linewidth=0.5)
        st.despine(ax)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=CLASSES[i][1])
               for i, c in enumerate([RSLAB, st.CYAN, st.AMBER, st.GREEN])
               if i in seen]
    st.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    return fig, ratios


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bench_out" / "pardiso_corpus.jsonl"
    data = load(path)
    st.setup()
    OUT.mkdir(parents=True, exist_ok=True)

    names = [s for s in ordered(data) if not s.startswith("ss_")]  # KLU reports no stages
    for subset, size, out in ((names, (8.0, 0.84 * len(names) + 1.2), None),
                              ([s for s in CARD_SYSTEMS if s in data], (9.0, 5.2), "social")):
        fig, ax = plt.subplots(figsize=size)
        handles = breakdown(data, ax, subset)
        st.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles], fontsize=8)
        if out:
            st.card(fig, OUT / "wct_breakdown_social.png")
        else:
            st.save(fig, OUT / "wct_breakdown.png")

    fig, ratios = estimates(data)
    st.save(fig, OUT / "estimate_accuracy.png")

    table = class_table(data)
    print("\n| class | " + " | ".join(l for _, l in METRICS) + " |")
    print("|---" * (len(METRICS) + 1) + "|")
    for c in sorted({cls(s) for s in data}):
        print(f"| {CLASSES[c][1]} | " + " | ".join(f"{table[(c, k)]:.2f}" for k, _ in METRICS) + " |")
    for key, r in ratios.items():
        if r:
            print(f"estimate/measured {key}: geomean {geomean(r):.2f}, min {min(r):.2f}, max {max(r):.2f}")


if __name__ == "__main__":
    main()
