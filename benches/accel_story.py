"""Head-to-head story figures: RSLAB vs Apple Accelerate, now and over the releases.

Reads ``bench_out/accel_history.jsonl`` (written by ``benches/run_accel_history.sh``:
the same matrices through the shipped path of every release binary, with Accelerate
measured in the same rounds) and renders

* ``bench_out/accel_classes.png`` - speedup over Accelerate per matrix class at the
  current version, factor-only and one-shot end-to-end, parity line at 1x,
* ``bench_out/accel_timeline.png`` - geomean speedup over Accelerate per release for
  the three shipped paths, annotated with what each release brought,
* ``bench_out/accel_scaling.png`` - speedup vs problem size: where the vendor's AMX
  kernels win and where RSLAB's structure exploitation takes over,
* ``bench_out/accel_social_{classes,timeline,scaling}.png`` - the same stories on an
  opaque page, share-card skin (same idiom as ``klu_social.png``).

Two metrics, because they answer different questions:
  * **factor** - the repeated-factorization cost (same pattern, new values); this is
    what the README head-to-head table reports.
  * **end-to-end** - one-shot analyze + factor + solve, i.e. what a caller who solves
    a system once actually waits for. Both solvers race orderings in their analyze
    (RSLAB its exact race, Accelerate the AMD-vs-Metis bakeoff), so this is a fair
    like-for-like number, not an artifact of a missing symbolic phase.

``RSLAB_REPORT=1`` redirects the non-card figures to ``docs/report/figures/*.pdf``.

Usage: ``python benches/accel_story.py [bench_out/accel_history.jsonl]``
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import bench_style as st

HERE = Path(__file__).resolve().parent
OUT = HERE / "bench_out"
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "accel_history.jsonl"

# The three shipped paths, in figure order: (family, rslab solver key, label, color).
PATHS = [
    ("sym", "auto", "LDLT path (EM/FEM)", st.BLUE),
    ("unsym", "auto", "LU path (CFD, MoM)", st.CYAN),
    ("circuit", "klu", "KLU path (circuit MNA)", st.SOLVERS["klu"][1]),
]

# Matrix classes as they appear in the bench names: prefix -> (label, family, solver).
CLASSES = {
    "circuit": ("circuit MNA", "circuit", "klu"),
    "curlcurl": ("curl-curl Maxwell", "sym", "auto"),
    "helmholtz": ("Helmholtz 3D", "sym", "auto"),
    "saddle": ("Stokes saddle point", "sym", "auto"),
    "convdiff3d": ("convection-diffusion 3D", "unsym", "auto"),
    "convdiff2d": ("convection-diffusion 2D", "unsym", "auto"),
    "mom": ("BEM/MoM near field", "unsym", "auto"),
}

# One short headline per release for the timeline callouts.
RELEASE_NOTES = {
    "v0.20.0": "mixed precision,\nadaptive BLR tail",
    "v0.22.0": "bit-identity fix,\nAutoRace prefixes",
    "v0.23.0": "Accelerate bench,\ncircuit family",
    "v0.24.0": "clean-room ND ordering,\nKLU 32-bit kernels",
    "v0.25.0": "KLU parallel\nrefactor + splice",
    "v0.26.0": "KLU parallel\nby default",
    "v0.27.0": "Hopcroft-Karp BTF,\nmatching bakeoff",
    "v0.28.0": "consolidation, KLU pipeline,\nexact ordering race",
}

METRICS = {
    "factor": ("factor only (repeated factorizations)", lambda r: r["fac_ms"]),
    "e2e": ("one-shot analyze + factor + solve",
            lambda r: r["ana_ms"] + r["fac_ms"] + r["slv_ms"]),
}


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit(f"no records in {path}")
    return rows


def versions(rows):
    return sorted({r["version"] for r in rows},
                  key=lambda v: tuple(int(p) for p in v.lstrip("v").split(".")))


def accel_ref(rows, metric):
    """(family, name) -> Accelerate time under `metric`, from the newest run."""
    newest = versions(rows)[-1]
    cost = METRICS[metric][1]
    return {(r["family"], r["name"]): cost(r) for r in rows
            if r["version"] == newest and r["solver"] == "accel" and r["res"] < 0.1}


def speedups(rows, ref, metric, version, family, solver):
    """name -> Accelerate / RSLAB ratio (>1 means RSLAB is faster)."""
    cost = METRICS[metric][1]
    return {r["name"]: ref[(family, r["name"])] / cost(r) for r in rows
            if r["version"] == version and r["family"] == family
            and r["solver"] == solver and r["res"] < 0.1
            and (family, r["name"]) in ref and cost(r) > 0}


def geomean(xs):
    xs = [x for x in xs if x > 0 and not np.isnan(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def per_class(rows, metric, version):
    """[(label, geomean speedup, matrices, color)] for one version, one metric."""
    ref = accel_ref(rows, metric)
    color = {fam: c for fam, _, _, c in PATHS}
    out = []
    for prefix, (label, fam, solv) in CLASSES.items():
        sp = speedups(rows, ref, metric, version, fam, solv)
        vals = [v for n, v in sp.items() if n.split("_")[0] == prefix]
        if vals:
            out.append((label, geomean(vals), len(vals), color[fam]))
    return out


def class_bars(rows, ax):
    """Per-class speedup at the newest version: factor-only vs one-shot end-to-end."""
    latest = versions(rows)[-1]
    fac = per_class(rows, "factor", latest)
    e2e = {b[0]: b[1] for b in per_class(rows, "e2e", latest)}
    fac.sort(key=lambda b: e2e.get(b[0], b[1]))

    ys = np.arange(len(fac))
    h = 0.36
    ax.barh(ys + h / 2, [e2e[b[0]] for b in fac], h, color=[b[3] for b in fac])
    ax.barh(ys - h / 2, [b[1] for b in fac], h, color=[b[3] for b in fac], alpha=0.45)
    ax.axvline(1.0, color=st.GRAY, lw=1.0, ls="--")
    ax.text(1.02, len(fac) - 0.4, "Accelerate parity", color=st.GRAY, fontsize=8,
            va="center", ha="left")
    for y, b in zip(ys, fac):
        ax.text(e2e[b[0]] + 0.08, y + h / 2, f"{e2e[b[0]]:.2f}x", va="center",
                fontsize=8.5, color=b[3])
        ax.text(b[1] + 0.08, y - h / 2, f"{b[1]:.2f}x", va="center", fontsize=8.5,
                color=b[3], alpha=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{b[0]}  ({b[2]})" for b in fac], fontsize=9)
    ax.set_xlabel("speedup over Apple Accelerate (higher is better)")
    ax.set_xlim(0, max(list(e2e.values()) + [b[1] for b in fac]) * 1.18)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_title(f"RSLAB {latest} vs Apple Accelerate, per matrix class", fontsize=10)
    # Bar color carries the solver path, bar shade the metric - so the legend is
    # two neutral patches for the metric plus one line per path.
    handles = [Patch(facecolor=st.DARKGRAY, label="one-shot (analyze + factor + solve)"),
               Patch(facecolor=st.DARKGRAY, alpha=0.45, label="factor only (repeated)")]
    handles += [Patch(facecolor=c, label=lbl) for _, _, lbl, c in PATHS]
    return fac, e2e, handles


def timeline(rows, ax, metric, annotate=True):
    """Geomean speedup per release, one line per shipped path plus the mean."""
    ref = accel_ref(rows, metric)
    vers = versions(rows)
    xs = np.arange(len(vers))
    series = {}
    for fam, solv, label, color in PATHS:
        ys = [geomean(list(speedups(rows, ref, metric, v, fam, solv).values()))
              for v in vers]
        series[label] = ys
        ax.plot(xs, ys, marker="o", ms=4.5, color=color, lw=1.6, label=label)
    mean = [geomean([series[l][i] for _, _, l, _ in PATHS]) for i in range(len(vers))]
    ax.plot(xs, mean, color=st.DARKGRAY, lw=2.6, ls=":", label="mean over all paths")
    ax.axhline(1.0, color=st.GRAY, lw=1.0, ls="--")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.yaxis.set_major_locator(plt.FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    lo = np.nanmin([y for ys in series.values() for y in ys])
    hi = np.nanmax([y for ys in series.values() for y in ys])

    if annotate:
        # Callouts staggered in two rows above the highest line, and pinned inside
        # the axes at the two ends so nothing runs off the page.
        for i, v in enumerate(vers):
            note = RELEASE_NOTES.get(v)
            if not note:
                continue
            y = np.nanmax([series[l][i] for _, _, l, _ in PATHS])
            ha = "left" if i == 0 else ("right" if i == len(vers) - 1 else "center")
            ax.annotate(note, (i, y), xytext=(0, 11 if i % 2 == 0 else 30),
                        textcoords="offset points", ha=ha, fontsize=6.5,
                        color="black" if st.REPORT else st.GRAY)
        hi *= 2.6
    ax.set_ylim(lo / 1.35, hi * 1.05)

    ax.set_xticks(xs)
    ax.set_xticklabels(vers, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("geomean speedup over Accelerate")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title(f"Every release, same matrices: {METRICS[metric][0]}", fontsize=10)
    return vers, series, mean


def scaling(rows, ax, metric="factor"):
    """Speedup vs problem size - the vendor's AMX kernels own the small end, RSLAB's
    structure exploitation takes over at scale. One line per matrix *class*: a family
    cycles through its classes size by size, so a per-family line would zigzag between
    unrelated problems."""
    ref = accel_ref(rows, metric)
    latest = versions(rows)[-1]
    cost = METRICS[metric][1]
    color = {fam: c for fam, _, _, c in PATHS}
    for prefix, (label, fam, solv) in CLASSES.items():
        pts = sorted((r["nnz"], ref[(fam, r["name"])] / cost(r)) for r in rows
                     if r["version"] == latest and r["family"] == fam
                     and r["solver"] == solv and r["name"].split("_")[0] == prefix
                     and (fam, r["name"]) in ref and r["res"] < 0.1)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4.0,
                color=color[fam], lw=1.4)
        ax.annotate(f" {label}", pts[-1], fontsize=7, color=color[fam],
                    va="center", ha="left")
    ax.axhline(1.0, color=st.GRAY, lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}x"))
    ax.yaxis.set_major_locator(plt.FixedLocator([0.125, 0.25, 0.5, 1, 2, 4]))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
    ax.set_xlabel("nonzeros in A")
    ax.set_ylabel(f"speedup over Accelerate ({METRICS[metric][0].split(' (')[0]})")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_title("Speedup vs problem size: the crossover", fontsize=10)


def card(fig, name):
    out = OUT / name
    fig.savefig(out, dpi=200, transparent=False, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def main():
    rows = load(PATH)
    if not accel_ref(rows, "factor"):
        sys.exit("no Accelerate reference records (solver=accel) in the history file")
    st.setup()

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fac, e2e, handles = class_bars(rows, ax)
    st.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    st.save(fig, OUT / "accel_classes.png")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    vers, series, mean = timeline(rows, ax_l, "factor")
    _, series_e2e, mean_e2e = timeline(rows, ax_r, "e2e", annotate=False)
    st.legend_below(fig, ax=ax_l)
    st.save(fig, OUT / "accel_timeline.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    scaling(rows, ax)
    st.save(fig, OUT / "accel_scaling.png")

    # Share cards (opaque, standalone).
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _, _, handles = class_bars(rows, ax)
    st.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    card(fig, "accel_social_classes.png")

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    timeline(rows, ax, "factor")
    st.legend_below(fig, ax=ax)
    card(fig, "accel_social_timeline.png")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    scaling(rows, ax)
    card(fig, "accel_social_scaling.png")

    print("\n== geomean speedup over Accelerate (factor / end-to-end) ==")
    print("| version | " + " | ".join(l for _, _, l, _ in PATHS) + " | mean |")
    print("|---" * (len(PATHS) + 2) + "|")
    for i, v in enumerate(vers):
        cells = " | ".join(
            f"{series[l][i]:.2f}x / {series_e2e[l][i]:.2f}x"
            if not np.isnan(series[l][i]) else "-" for _, _, l, _ in PATHS)
        print(f"| {v} | {cells} | {mean[i]:.2f}x / {mean_e2e[i]:.2f}x |")
    print("\n== per class, current version (factor / end-to-end) ==")
    for label, val, n, _ in sorted(fac, key=lambda b: -e2e[b[0]]):
        print(f"  {label:<26} {val:.2f}x / {e2e[label]:.2f}x  ({n} matrices)")


if __name__ == "__main__":
    main()
