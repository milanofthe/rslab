"""Figures for the real-circuit KLU head-to-head (klu_realworld bench).

Reads the JSONL written by `RLA_BENCH_OUT=... cargo bench --bench
klu_realworld --features matgen-download` and renders:

* ``bench_out/h2h_klu_realworld.png`` — house-style two-panel (factor /
  20-point refactor+solve sweep) grouped bars, RSLAB KLU vs SuiteSparse KLU
  on the SuiteSparse-collection circuit corpus. ``RSLAB_REPORT=1`` redirects
  to ``docs/report/figures/h2h_klu_realworld.pdf`` (paper skin).
* ``bench_out/klu_social.png`` — standalone dark share-card: per-matrix
  speedup of RSLAB KLU over SuiteSparse KLU (factor phase), opaque
  background, self-explanatory annotations. Not affected by RSLAB_REPORT.

Usage: ``python benches/klu_realworld_plot.py [bench_out/klu_realworld.jsonl]``
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bench_style as st

TEAL = "#14b8a6"      # RSLAB KLU (house solver color)
TEAL_DARK = "#0f766e" # RSLAB KLU parallel blocks
AMBER = st.AMBER      # SuiteSparse KLU (reference competitor)

HERE = Path(__file__).resolve().parent
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "bench_out" / "klu_realworld.jsonl"
OUT = HERE / "bench_out"


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["group"], r["name"]), {})[r["solver"]] = r
    # keep matrices with both solvers, ascending n
    pairs = {k: v for k, v in by.items() if "rslab" in v and "ss-klu" in v}
    return dict(sorted(pairs.items(), key=lambda kv: kv[1]["rslab"]["n"]))


def two_panel_bars(data):
    st.setup()
    fig, (ax_fac, ax_sweep) = st.two_panel()
    def knum(n):
        return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}k" if n >= 1e3 else str(n))

    names = [f"{name} ({knum(v['rslab']['n'])})" for (_, name), v in data.items()]
    xs = np.arange(len(data))
    w = 0.38
    for ax, key, title in (
        (ax_fac, "fac_s", "factor (one pivoting factorization)"),
        (ax_sweep, "sweep_s", "20-point sweep (refactor + solve)"),
    ):
        rs = [v["rslab"][key] * 1e3 for v in data.values()]
        ss = [v["ss-klu"][key] * 1e3 for v in data.values()]
        ax.bar(xs - w / 2, rs, w, color=TEAL, label="RSLAB KLU (pure Rust)")
        ax.bar(xs + w / 2, ss, w, color=AMBER, label="SuiteSparse KLU (C)")
        for x, (a, b) in zip(xs, zip(rs, ss)):
            r = b / a
            ax.text(x, max(a, b) * 1.12, f"{r:.1f}x" if r >= 1 else f"{r:.2f}x",
                    ha="center", va="bottom", fontsize=7,
                    color=TEAL if r >= 1 else AMBER)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=7, rotation=50, ha="right")
        ax.set_ylabel("wall time [ms]")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_ylim(top=max(max(rs), max(ss)) * 3)
    st.legend_below(fig, ax=ax_fac)
    return st.save(fig, OUT / "h2h_klu_realworld.png")


def social_card(data):
    """Standalone dark share-card, independent of the house rcParams."""
    plt.rcdefaults()
    bg, fg, dim = "#0b1220", "#e2e8f0", "#64748b"
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": dim,
        "xtick.color": dim, "ytick.color": fg, "font.size": 11,
    })
    names = [name for (_, name) in data.keys()]
    ratio = np.array([v["ss-klu"]["fac_s"] / v["rslab"]["fac_s"] for v in data.values()])
    order = np.argsort(ratio)
    names = [names[i] for i in order]
    ratio = ratio[order]

    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(names) + 2.2))
    ys = np.arange(len(names))
    colors = [TEAL if r >= 1 else "#334155" for r in ratio]
    ax.barh(ys, ratio, 0.62, color=colors)
    ax.axvline(1.0, color=fg, linewidth=1.0, alpha=0.8)
    for y, r in zip(ys, ratio):
        ax.text(r * 1.04, y, f"{r:.2f}x", va="center", fontsize=9,
                color=TEAL if r >= 1 else dim)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    gm = float(np.exp(np.log(ratio).mean()))
    ax.set_xlabel("factor speedup vs SuiteSparse KLU  (log scale, >1x = RSLAB faster)")
    ax.set_title(
        "RSLAB KLU (pure Rust) vs SuiteSparse KLU (C)\n"
        f"real circuit matrices, SuiteSparse collection - geomean {gm:.2f}x",
        fontsize=12, loc="left", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    out = OUT / "klu_social.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    data = load(PATH)
    if not data:
        sys.exit(f"no paired rslab/ss-klu records in {PATH}")
    two_panel_bars(data)
    social_card(data)
