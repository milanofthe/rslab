"""Shared plotting style for all RSLAB benchmark figures.

One definition of the palette + rcParams so every figure looks the same
(transparent background, neutral-gray axes for light/dark pages, saturated data
colors), and one helper that places the legend in a single horizontal row
**below** the plot - import this instead of re-defining colors per script.
"""
import matplotlib.pyplot as plt

GRAY = "#808080"

# Canonical solver palette: key -> (label, color, marker).
SOLVERS = {
    "ll": ("RSLAB left-looking", "#3b82f6", "o"),
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
    """Apply the shared rcParams (transparent bg, gray axes/text/grid)."""
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 11, "legend.frameon": False,
    })


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
