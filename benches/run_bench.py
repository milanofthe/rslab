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

REPO = Path(r"C:\Repositories\RLA")
MKL = r"C:\Users\milan\anaconda3\Library\bin"
OUTDIR = Path(os.environ.get("RLA_BENCH_DIR", REPO / "benches" / "bench_out"))
OUTDIR.mkdir(parents=True, exist_ok=True)
SCALING = OUTDIR / "scaling.jsonl"
THREADS = OUTDIR / "threads.jsonl"

# Problem sizes (≈ nodes). faer factors the symmetric family as full LU (2× work),
# so the symmetric max is kept tractable.
SYM_SIZES = [4000, 8000, 16000, 32000, 64000]
UNSYM_SIZES = [1000, 2000, 4000, 8000, 16000]
THREAD_COUNTS = [1, 2, 4, 6, 8, 12, 16, 24]
SYM_FIXED, UNSYM_FIXED = 32000, 8000

STYLE = {  # solver -> (label, color, marker)
    "ll": ("RLA left-looking", "#1f77b4", "o"),
    "mf": ("RLA multifrontal", "#17becf", "s"),
    "faer": ("faer LU", "#ff7f0e", "^"),
    "pardiso": ("MKL PARDISO", "#2ca02c", "D"),
}
ORDER = ["ll", "mf", "faer", "pardiso"]


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
    for p in (SCALING, THREADS):
        p.unlink(missing_ok=True)
    # Scaling: time + memory passes, default (all) threads.
    for family, sizes in (("sym", SYM_SIZES), ("unsym", UNSYM_SIZES)):
        for mem in (False, True):
            print(f"scaling {family} metric={'mem' if mem else 'time'} ...", flush=True)
            run(exe, env_for(family, sizes, mem, SCALING))
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


def main():
    if "--plot-only" not in sys.argv:
        collect()
    sc = load(SCALING)
    th = load(THREADS)
    if not sc:
        sys.exit("no scaling records collected")
    fig_scaling(sc, "fac_ms", "factor wall-clock (ms)", "scaling_factor.png", "Factor time")
    fig_scaling(sc, "slv_ms", "solve wall-clock (ms)", "scaling_solve.png", "Solve time")
    fig_scaling(sc, "mem_mb", "factor memory (MB)", "scaling_memory.png", "Factor memory")
    if th:
        fig_threads(th)
    print("done.")


if __name__ == "__main__":
    main()
