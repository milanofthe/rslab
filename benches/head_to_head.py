"""Per-path head-to-head: RSLAB vs faer vs MKL PARDISO, one figure per solver path.

LDLᵀ (symmetric) and LU (unsymmetric) are separate solvers a caller dispatches to
explicitly, so each is benchmarked and plotted against *its own* PARDISO mtype
(6 for complex-symmetric, 13 for unsymmetric) and faer. For each path: factor time
and peak memory vs nonzeros on log-log axes, with a least-squares power-law fit
`y = C·nnz^alpha` per solver (the exponent is the empirical scaling order).

The RSLAB curve is the auto-tuned default (`LdltSolver`/`LuSolver::tuned`) — the
per-path learned tuner as shipped, the subject of this whole study.

Run: python benches/head_to_head.py <sym.jsonl> <unsym.jsonl>
"""
import json
import sys
from pathlib import Path

import bench_style
from fit_scaling import plot_metric, MEM_ORDER


def run(path, title, slug, mtype):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = path.parent
    print(f"== {title} ==")
    print("  factor time ~ nnz^alpha:")
    plot_metric(
        recs, "time", "fac_ms", "factor time [ms]",
        f"{title}: factor time vs size — RSLAB vs faer vs PARDISO (mtype {mtype})",
        out / f"h2h_{slug}_time.png",
    )
    print("  peak memory ~ nnz^alpha:")
    plot_metric(
        recs, "mem", "mem_mb", "peak memory [MB]",
        f"{title}: peak factor memory vs size — RSLAB vs faer vs PARDISO (mtype {mtype})",
        out / f"h2h_{slug}_mem.png", order=MEM_ORDER,
    )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    bench_style.setup()
    run(Path(sys.argv[1]), "LDLt path (symmetric)", "ldlt", 6)
    run(Path(sys.argv[2]), "LU path (unsymmetric)", "lu", 13)


if __name__ == "__main__":
    main()
