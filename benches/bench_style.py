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

# Named data colors for breakdown stages / estimate parts (drawn from the same
# saturated set as the solver palette so nothing drifts).
BLUE = "#3b82f6"
CYAN = "#06b6d4"
AMBER = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#a855f7"


def setup():
    """Apply the shared rcParams (transparent bg, gray axes/text/grid)."""
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": GRAY, "axes.labelcolor": GRAY, "axes.edgecolor": GRAY,
        "xtick.color": GRAY, "ytick.color": GRAY, "grid.color": GRAY,
        "axes.titlecolor": GRAY, "font.size": 11, "legend.frameon": False,
    })


def legend_below(fig, handles=None, labels=None, ax=None, ncol=None,
                 fontsize=9, bottom=0.14):
    """Lay the figure's legend out as one horizontal row centered **below** the
    plot, reserving `bottom` of the figure height for it (so it never overlaps
    the axes - bump the figure height to keep the plot area). Pass explicit
    `handles`/`labels`, or an `ax` to pull them from."""
    if handles is None:
        src = ax if ax is not None else fig.axes[0]
        handles, labels = src.get_legend_handles_labels()
    ncol = ncol or max(1, len(labels))
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=fontsize, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6,
               handletextpad=0.5)
