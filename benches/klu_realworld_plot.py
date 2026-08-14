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
    """Share card: the factor panel of the head-to-head in house style
    (same grouped wall-time bars, no speedup axis), on an opaque background
    so it renders on any feed."""
    st.setup()

    def knum(n):
        return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}k" if n >= 1e3 else str(n))

    names = [f"{name} ({knum(v['rslab']['n'])})" for (_, name), v in data.items()]
    xs = np.arange(len(data))
    w = 0.38
    rs = [v["rslab"]["fac_s"] * 1e3 for v in data.values()]
    ss = [v["ss-klu"]["fac_s"] * 1e3 for v in data.values()]
    gm = float(np.exp(np.mean([np.log(b / a) for a, b in zip(rs, ss)])))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(xs - w / 2, rs, w, color=TEAL, label="RSLAB KLU (pure Rust)")
    ax.bar(xs + w / 2, ss, w, color=AMBER, label="SuiteSparse KLU (C)")
    for x, (a, b) in zip(xs, zip(rs, ss)):
        r = b / a
        ax.text(x, max(a, b) * 1.12, f"{r:.1f}x" if r >= 1 else f"{r:.2f}x",
                ha="center", va="bottom", fontsize=7.5,
                color=TEAL if r >= 1 else AMBER)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7, rotation=50, ha="right")
    ax.set_ylabel("factor wall time [ms]")
    ax.set_title(
        "RSLAB KLU (pure Rust) vs SuiteSparse KLU (C) - real circuit matrices, "
        f"SuiteSparse collection - geomean {gm:.2f}x",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_ylim(top=max(max(rs), max(ss)) * 3)
    st.legend_below(fig, ax=ax)
    out = OUT / "klu_social.png"
    fig.savefig(out, dpi=200, transparent=False, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    data = load(PATH)
    if not data:
        sys.exit(f"no paired rslab/ss-klu records in {PATH}")
    two_panel_bars(data)
    social_card(data)
