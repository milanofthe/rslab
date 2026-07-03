"""Shared plotting style for all RSLAB benchmark figures.

One definition of the palette + rcParams so every figure looks the same, and one
helper that places the legend in a single horizontal row **below** the plot -
import this instead of re-defining colors per script.

Two render modes, selected by the ``RSLAB_REPORT`` environment variable:

* default (unset)  - transparent background, neutral-gray axes/text for the
  README (readable on light *and* dark GitHub themes); figures save as ``.png``.
* ``RSLAB_REPORT=1`` - **paper mode**: white page, **black axes/text**, serif
  font (Computer Modern math), in-figure titles stripped (the LaTeX caption
  carries them); figures are redirected to ``docs/report/figures/<stem>.pdf``.

Both modes share the same palette and legend placement, so the report and the
README are the same plots in two skins. Route every save through :func:`save`.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt

GRAY = "#808080"

# Paper mode: set by the report figure driver (`RSLAB_REPORT=1`).
REPORT = os.environ.get("RSLAB_REPORT") == "1"
REPORT_FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "report" / "figures"

# Canonical solver palette: key -> (label, color, marker).
SOLVERS = {
    "default": ("RSLAB (untuned default)", "#94a3b8", "x"),
    "auto": ("RSLAB (auto-tuned)", "#3b82f6", "o"),
    "ll": ("RSLAB left-looking", "#60a5fa", "o"),
    "mf": ("RSLAB multifrontal", "#06b6d4", "s"),
    "faer": ("faer LU", "#f59e0b", "^"),
    "pardiso": ("MKL PARDISO", "#22c55e", "D"),
    "superlu": ("SuperLU (scipy)", "#ef4444", "P"),
    "pc": ("RSLAB precond+GMRES", "#a855f7", "v"),
}

# Named data colors for breakdown stages / estimate parts. The two grays are for
# *neutral / reference* series (estimates, the analyze stage); the saturated
# blue/cyan stay tied to the two RSLAB paths (LL / MF) so meaning is consistent
# across figures. PURPLE is reserved for the `pc` solver only - never reused.
BLUE = "#3b82f6"
CYAN = "#06b6d4"
AMBER = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#a855f7"
DARKGRAY = "#4b5563"
# Sequential blue shades for several series of the *same* kind (e.g. example
# matrices in one RSLAB plot) - reads as "all RSLAB", not as different solvers.
BLUE_SHADES = ["#93c5fd", "#3b82f6", "#1d4ed8"]


def setup():
    """Apply the shared rcParams. Paper mode (``RSLAB_REPORT=1``): white page,
    black axes/text, serif/CM font. Default: transparent bg, gray axes/text."""
    if REPORT:
        plt.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
            "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
            "xtick.color": "black", "ytick.color": "black", "grid.color": "#c0c0c0",
            "axes.titlecolor": "black", "font.size": 9, "font.family": "serif",
            "mathtext.fontset": "cm", "legend.frameon": False,
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
            "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
            "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
            "axes.titlecolor": GRAY, "font.size": 11, "legend.frameon": False,
        })


def save(fig, out_path):
    """Save `fig` honoring the render mode. Paper mode redirects to
    ``docs/report/figures/<stem>.pdf`` (white, opaque, in-figure titles stripped
    so the LaTeX caption is the single source of the caption); default writes the
    given path as a transparent PNG. Returns the path written."""
    out_path = Path(out_path)
    if REPORT:
        for a in fig.axes:
            a.set_title("")
        if getattr(fig, "_suptitle", None) is not None:
            fig.suptitle("")
        REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        dest = REPORT_FIG_DIR / (out_path.stem + ".pdf")
        fig.savefig(dest, bbox_inches="tight", facecolor="white")
        print(f"wrote {dest}")
        return dest
    fig.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {out_path}")
    return out_path


def two_panel(figsize=(11.0, 4.6)):
    """House-style two-panel figure for a wall-clock-time / peak-memory pair of the
    *same* experiment: one PDF/PNG instead of two separate files. Returns
    ``(fig, (ax_wct, ax_mem))`` with **wall-clock time in the left panel and peak
    memory in the right**, sharing the same x-axis convention (the caller sets the
    identical x-scale/label on both). Draw the two metrics into the two axes, place a
    single shared legend with :func:`legend_below`, and route the save through
    :func:`save` so both render modes and the report redirect are honored. The
    ``(11.0, 4.6)`` default is the report's established full-text-width two-panel size
    (matching ``precond_gmres``)."""
    fig, (ax_wct, ax_mem) = plt.subplots(1, 2, figsize=figsize)
    return fig, (ax_wct, ax_mem)


def legend_below(fig, handles=None, labels=None, ax=None, ncol=None, fontsize=9):
    """Place the figure legend in a compact block just **below** the plot: a
    single horizontal row, wrapping to two rows only when there are many entries.
    `bbox_inches="tight"` at save time then crops to include it. Pass explicit
    `handles`/`labels`, or an `ax` to pull them from."""
    if handles is None:
        src = ax if ax is not None else fig.axes[0]
        handles, labels = src.get_legend_handles_labels()
    n = len(labels)
    if ncol is None:
        ncol = n if n <= 4 else (n + 1) // 2  # 2 rows past 4 entries
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=ncol, frameon=False, fontsize=fontsize, columnspacing=1.6,
               handletextpad=0.5)
