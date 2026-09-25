"""Shared plotting style for all RSLAB benchmark figures.

One definition of the palette + rcParams so every figure looks the same, and one
helper that places the legend in a single horizontal row **below** the plot -
import this instead of re-defining colors per script.

Figures have a transparent background and neutral-gray axes and text, readable
on light *and* dark GitHub themes, and save as ``.png``. Route every save
through :func:`save`.
"""
from pathlib import Path

import matplotlib.pyplot as plt

GRAY = "#808080"

# Canonical solver palette: key -> (label, color, marker).
SOLVERS = {
    "default": ("RSLAB (fixed default cfg)", "#94a3b8", "x"),
    "auto": ("RSLAB (heuristic pick)", "#3b82f6", "o"),
    "ll": ("RSLAB left-looking", "#60a5fa", "o"),
    "mf": ("RSLAB multifrontal", "#06b6d4", "s"),
    "faer": ("faer LU", "#f59e0b", "^"),
    "pardiso": ("MKL PARDISO", "#22c55e", "D"),
    "superlu": ("SuperLU (scipy)", "#ef4444", "P"),
    "pc": ("RSLAB precond+GMRES", "#a855f7", "v"),
    "accel": ("Apple Accelerate", "#ec4899", "h"),
    "klu": ("RSLAB KLU", "#14b8a6", "p"),
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
    """Apply the shared rcParams: transparent background, gray axes and text."""
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 11, "legend.frameon": False,
    })


def save(fig, out_path):
    """Save `fig` as a transparent PNG at `out_path` and return the path."""
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight")
    print(f"wrote {out_path}")
    return out_path


def card(fig, out_path):
    """Share-card skin of the same figure: opaque white page at 200 dpi, so it
    renders on any feed. Returns the path written."""
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, transparent=False, facecolor="white", bbox_inches="tight")
    print(f"wrote {out_path}")
    return out_path


def two_panel(figsize=(11.0, 4.6)):
    """House-style two-panel figure for a wall-clock-time / peak-memory pair of the
    *same* experiment: one PDF/PNG instead of two separate files. Returns
    ``(fig, (ax_wct, ax_mem))`` with **wall-clock time in the left panel and peak
    memory in the right**, sharing the same x-axis convention (the caller sets the
    identical x-scale/label on both). Draw the two metrics into the two axes, place a
    single shared legend with :func:`legend_below`, and route the save through
    :func:`save`."""
    fig, (ax_wct, ax_mem) = plt.subplots(1, 2, figsize=figsize)
    return fig, (ax_wct, ax_mem)


def despine(*axes):
    """Drop the top and right spines: the frame carries no information once the
    grid is there, and the open corner leaves room for callouts."""
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


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
