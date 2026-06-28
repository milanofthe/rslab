"""Benchmark driver: orchestrates the scaling + thread sweeps over the Rust
`bench_suite` engine and renders the matplotlib plots.

  * Scaling sweep  — one process per (family, metric), looping sizes.
  * Thread sweep   — one process per (family, thread-count) at a fixed size,
                     with RAYON / MKL / OMP thread counts pinned.

Produces: scaling_factor.png, scaling_memory.png, scaling_solve.png,
          thread_scaling.png  (each a 2-panel symmetric / unsymmetric figure).

Run:  python benches/run_bench.py
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(r"C:\Repositories\RLA")
MKL = r"C:\Users\milan\anaconda3\Library\bin"
OUTDIR = Path(os.environ.get("RLA_BENCH_DIR", REPO / "benches" / "bench_out"))
OUTDIR.mkdir(parents=True, exist_ok=True)
SCALING = OUTDIR / "scaling.jsonl"
THREADS = OUTDIR / "threads.jsonl"
REAL = OUTDIR / "real.jsonl"
ESTIMATE = OUTDIR / "estimate.jsonl"

# Problem sizes (≈ nodes). faer factors the symmetric family as full LU (2× work),
# so the symmetric max is kept tractable. The unsymmetric BEM kernel now keeps a
# constant ≈120 nnz/row (density-matched cutoff), so it scales to larger n.
SYM_SIZES = [4000, 8000, 16000, 32000, 64000]
UNSYM_SIZES = [2000, 4000, 8000, 16000, 32000]
THREAD_COUNTS = [1, 2, 4, 6, 8, 12, 16, 24]
SYM_FIXED, UNSYM_FIXED = 32000, 16000

STYLE = {  # solver -> (label, color, marker)
    "ll": ("RSLAB left-looking", "#3b82f6", "o"),
    "mf": ("RSLAB multifrontal", "#06b6d4", "s"),
    "faer": ("faer LU", "#f59e0b", "^"),
    "pardiso": ("MKL PARDISO", "#22c55e", "D"),
}
ORDER = ["ll", "mf", "faer", "pardiso"]

# Neutral gray for axes/text/grid so figures read on both light and dark pages;
# data colours stay saturated (visible on either). Backgrounds are transparent.
GRAY = "#808080"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": GRAY,
        "axes.edgecolor": GRAY,
        "axes.labelcolor": GRAY,
        "axes.titlecolor": GRAY,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "grid.color": GRAY,
        "legend.framealpha": 0.0,
        "legend.labelcolor": GRAY,
        "font.size": 10,
    })


def env_for(family, sizes, mem, out, threads=None):
    e = dict(os.environ)
    e["PATH"] = MKL + os.pathsep + e.get("PATH", "")
    e["RLA_BENCH_FAMILY"] = family
    e["RLA_BENCH_SIZES"] = ",".join(map(str, sizes))
    e["RLA_BENCH_MEM"] = "1" if mem else ""
    e["RLA_BENCH_SOLVERS"] = "ll,mf,faer,pardiso"
    e["RLA_BENCH_OUT"] = str(out)
    if threads is not None:
        for k in ("RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            e[k] = str(threads)
    else:
        for k in ("RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            e.pop(k, None)
    return e


def build_engine():
    print("building bench_suite ...", flush=True)
    subprocess.run(
        ["cargo", "bench", "--bench", "bench_suite", "--features", "matgen", "--no-run"],
        cwd=REPO, check=True,
    )
    exes = [
        e for e in glob.glob(str(REPO / "target" / "release" / "deps" / "bench_suite-*.exe"))
        if not e.endswith(".d")
    ]
    if not exes:
        sys.exit("bench_suite executable not found")
    return max(exes, key=os.path.getmtime)


def run(exe, env):
    subprocess.run([exe], env=env, check=True)


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def collect():
    exe = build_engine()
    for p in (SCALING, THREADS, REAL, ESTIMATE):
        p.unlink(missing_ok=True)
    # Scaling: time + memory passes, default (all) threads.
    for family, sizes in (("sym", SYM_SIZES), ("unsym", UNSYM_SIZES)):
        for mem in (False, True):
            print(f"scaling {family} metric={'mem' if mem else 'time'} ...", flush=True)
            run(exe, env_for(family, sizes, mem, SCALING))
    # Real precond_matrices (realism anchor) — time + memory.
    for mem in (False, True):
        print(f"real metric={'mem' if mem else 'time'} ...", flush=True)
        run(exe, env_for("real", [], mem, REAL))
    # A-priori memory-estimate breakdown (instant, no factoring) over the sym sweep.
    print("estimate sweep ...", flush=True)
    e = env_for("sym", SYM_SIZES, False, ESTIMATE)
    e["RLA_BENCH_ESTIMATE"] = "1"
    run(exe, e)
    # Thread sweep at a fixed size, time only.
    for family, fixed in (("sym", SYM_FIXED), ("unsym", UNSYM_FIXED)):
        for t in THREAD_COUNTS:
            print(f"threads {family} t={t} ...", flush=True)
            run(exe, env_for(family, [fixed], False, THREADS, threads=t))


def series(recs, family, metric, xkey, ykey):
    """{solver: ([x...],[y...])} sorted by x, for one family/metric."""
    out = {}
    for r in recs:
        if r["family"] != family or r["metric"] != metric:
            continue
        out.setdefault(r["solver"], []).append((r[xkey], r[ykey]))
    return {s: tuple(zip(*sorted(v))) for s, v in out.items()}


def panel(ax, data, title, xlabel, ylabel, logx=True, logy=True):
    for s in ORDER:
        if s not in data or not data[s][0]:
            continue
        label, color, marker = STYLE[s]
        xs, ys = data[s]
        ax.plot(xs, ys, marker=marker, color=color, label=label, lw=1.8, ms=6)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)


def fig_scaling(recs, ykey, ylabel, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panel(axes[0], series(recs, "sym", "time" if ykey != "mem_mb" else "mem", "n", ykey),
          f"{title} — symmetric (LDLᵀ / 3D Helmholtz)", "n (DOFs)", ylabel)
    panel(axes[1], series(recs, "unsym", "time" if ykey != "mem_mb" else "mem", "n", ykey),
          f"{title} — unsymmetric (LU / MoM near-field)", "n (DOFs)", ylabel)
    fig.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=130)
    print("wrote", OUTDIR / fname)


def fig_threads(recs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, family, fixed in ((axes[0], "sym", SYM_FIXED), (axes[1], "unsym", UNSYM_FIXED)):
        data = series(recs, family, "time", "threads", "fac_ms")
        tmax = 0
        for s in ORDER:
            if s not in data or not data[s][0]:
                continue
            label, color, marker = STYLE[s]
            ts, fac = data[s]
            base = dict(zip(ts, fac)).get(1)
            if not base:
                continue
            speed = [base / f for f in fac]
            ax.plot(ts, speed, marker=marker, color=color, label=label, lw=1.8, ms=6)
            tmax = max(tmax, max(ts))
        if tmax:
            ax.plot([1, tmax], [1, tmax], ls="--", color="gray", alpha=0.6, label="ideal")
        ax.set_title(f"thread scaling — {family} (n={fixed})")
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup vs 1 thread (factor)")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "thread_scaling.png", dpi=130)
    print("wrote", OUTDIR / "thread_scaling.png")


def fig_real(recs):
    """Grouped bars per real matrix: factor time and factor memory, per solver."""
    names = sorted({r["name"] for r in recs},
                   key=lambda nm: next(r["n"] for r in recs if r["name"] == nm))
    short = [nm.replace("_D300_N3", "").replace("_D350_N3", "").replace("_D280_N4_w16", "")
             for nm in names]
    x = np.arange(len(names))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ykey, title, ylab in (
        (axes[0], "time", "fac_ms", "Real MoM matrices — factor time", "factor wall-clock (ms)"),
        (axes[1], "mem", "mem_mb", "Real MoM matrices — factor memory", "factor memory (MB)"),
    ):
        for i, s in enumerate(ORDER):
            label, color, _ = STYLE[s]
            vals = []
            for nm in names:
                v = [r[ykey] for r in recs
                     if r["name"] == nm and r["solver"] == s and r["metric"] == metric]
                vals.append(v[0] if v else 0)
            ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha="right", fontsize=7)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "real_matrices.png", dpi=130)
    print("wrote", OUTDIR / "real_matrices.png")


def fig_wct_breakdown(recs):
    """Stacked analyze / factor / solve wall-clock per solver, at the largest size
    of each family — where each solver spends its time."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, family in ((axes[0], "sym"), (axes[1], "unsym")):
        rows = [r for r in recs if r["family"] == family and r["metric"] == "time"]
        if not rows:
            continue
        nmax = max(r["n"] for r in rows)
        rows = [r for r in rows if r["n"] == nmax]
        present = [s for s in ORDER if any(r["solver"] == s for r in rows)]
        x = np.arange(len(present))
        ana = [next((r["ana_ms"] for r in rows if r["solver"] == s), 0) for s in present]
        fac = [next((r["fac_ms"] for r in rows if r["solver"] == s), 0) for s in present]
        slv = [next((r["slv_ms"] for r in rows if r["solver"] == s), 0) for s in present]
        ax.bar(x, ana, 0.6, label="analyze", color=GRAY, alpha=0.55)
        ax.bar(x, fac, 0.6, bottom=ana, label="factor", color="#3b82f6")
        ax.bar(x, slv, 0.6, bottom=[a + f for a, f in zip(ana, fac)], label="solve", color="#f59e0b")
        ax.set_yscale("log")
        ax.set_title(f"WCT breakdown — {family} (n={nmax})")
        ax.set_ylabel("wall-clock (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[s][0] for s in present], rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "wct_breakdown.png", dpi=130, transparent=True)
    print("wrote", OUTDIR / "wct_breakdown.png")


def fig_memory_breakdown(recs):
    """Stacked a-priori memory estimate (dense panels / compact factor / scratch)
    vs DOFs, with the panel-freed live floor marked."""
    recs = sorted(recs, key=lambda r: r["n"])
    ns = [r["n"] for r in recs]
    x = np.arange(len(ns))
    panels = [r["panels_mb"] for r in recs]
    factor = [r["factor_mb"] for r in recs]
    scratch = [r["scratch_mb"] for r in recs]
    floor = [r["freed_floor_mb"] for r in recs]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x, panels, 0.6, label="dense panels", color="#3b82f6")
    ax.bar(x, factor, 0.6, bottom=panels, label="compact factor (CSC)", color="#06b6d4")
    ax.bar(x, scratch, 0.6, bottom=[p + f for p, f in zip(panels, factor)],
           label="input + scratch", color=GRAY, alpha=0.55)
    ax.plot(x, floor, "o--", color="#f59e0b", label="panel-freed live floor", lw=1.6, ms=5)
    ax.set_title("A-priori factor-memory breakdown — symmetric (3D)")
    ax.set_xlabel("n (DOFs)")
    ax.set_ylabel("estimated memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n//1000}k" for n in ns])
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "memory_breakdown.png", dpi=130, transparent=True)
    print("wrote", OUTDIR / "memory_breakdown.png")


def main():
    setup_style()
    if "--plot-only" not in sys.argv:
        collect()
    sc = load(SCALING)
    th = load(THREADS)
    rl = load(REAL)
    es = load(ESTIMATE)
    if not sc:
        sys.exit("no scaling records collected")
    fig_scaling(sc, "fac_ms", "factor wall-clock (ms)", "scaling_factor.png", "Factor time")
    fig_scaling(sc, "slv_ms", "solve wall-clock (ms)", "scaling_solve.png", "Solve time")
    fig_scaling(sc, "mem_mb", "factor memory (MB)", "scaling_memory.png", "Factor memory")
    fig_wct_breakdown(sc)
    if th:
        fig_threads(th)
    if rl:
        fig_real(rl)
    if es:
        fig_memory_breakdown(es)
    print("done.")


if __name__ == "__main__":
    main()
