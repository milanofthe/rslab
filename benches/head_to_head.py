"""Per-path head-to-head: RSLAB vs faer vs MKL PARDISO, one figure per solver path.

LDLᵀ (symmetric) and LU (unsymmetric) are separate solvers a caller dispatches to
explicitly, so each is benchmarked and plotted against *its own* PARDISO mtype
(6 for complex-symmetric, 13 for unsymmetric) and faer. Each path is one **two-panel**
figure — factor wall-clock time (left) and peak memory (right) vs nonzeros on log-log
axes, with a least-squares power-law fit `y = C·nnz^alpha` per solver (the exponent,
the empirical scaling order, is annotated on each fit line) and one shared legend.

The RSLAB curve is the auto-tuned default (`LdltSolver`/`LuSolver::tuned`) — the
per-path learned tuner as shipped, the subject of this whole study.

Run: python benches/head_to_head.py <sym.jsonl> <unsym.jsonl>
"""
import json
import sys
from pathlib import Path

from matplotlib.lines import Line2D

import bench_style
from fit_scaling import plot_metric

# RSLAB's untuned default is drawn alongside the auto-tuned curve, so the gap the
# learned tuner closes (default -> auto) is visible against the external solvers.
ORDER = ["default", "auto", "faer", "pardiso"]


def run(path, title, slug, mtype):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = path.parent
    print(f"== {title} ==")
    # One figure, two panels: wall-clock time (left) + peak memory (right), same x-axis.
    fig, (ax_wct, ax_mem) = bench_style.two_panel()
    print("  factor time ~ nnz^alpha:")
    present = plot_metric(recs, "time", "fac_ms", "factor time [ms]", None,
                          order=ORDER, ax=ax_wct)
    print("  peak memory ~ nnz^alpha:")
    plot_metric(recs, "mem", "mem_mb", "peak memory [MB]", None, order=ORDER, ax=ax_mem)
    fig.suptitle(
        f"{title}: factor time (left) and peak memory (right) vs size "
        f"— RSLAB (default & tuned) vs faer vs PARDISO (mtype {mtype})",
        color=bench_style.GRAY)
    # One shared legend for both panels: solver identity only (the per-panel scaling
    # exponents differ and are annotated on the fit lines).
    handles = [Line2D([], [], color=c, marker=mk, ls="", label=lbl)
               for _, lbl, c, mk in present]
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, out / f"h2h_{slug}.png")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    bench_style.setup()
    run(Path(sys.argv[1]), "LDLt path (symmetric)", "ldlt", 6)
    run(Path(sys.argv[2]), "LU path (unsymmetric)", "lu", 13)


if __name__ == "__main__":
    main()
