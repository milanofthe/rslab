"""RSLAB vs Apple Accelerate: wall time normalized to Accelerate, now and per release.

Reads ``bench_out/accel_history.jsonl`` (written by ``benches/run_accel_history.sh``:
the same matrices through the shipped path of every release binary, with Accelerate
measured in the same rounds) and renders

* ``bench_out/accel_classes.png``  - per matrix class, current version,
* ``bench_out/accel_timeline.png`` - per release, one line per shipped path,
* ``bench_out/accel_scaling.png``  - vs problem size,
* ``bench_out/accel_social_*.png`` - the same three on an opaque page.

Every figure plots RSLAB wall time divided by Accelerate wall time, so Accelerate
is the 1.0 line and lower is faster. Two metrics, because they answer different
questions: factor only is the repeated-factorization cost, one-shot is analyze +
factor + solve, what a caller solving a system once waits for. Both solvers race
orderings inside their analyze, so the one-shot number is like for like.

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

# Short note per release for the timeline callouts. Releases without an entry stay
# unlabeled (patch releases, wasm and CI fixes).
RELEASE_NOTES = {
    "v0.18.0": "ND bakeoff\nin tuned()",
    "v0.19.0": "heuristic default pick,\nLL throughput work",
    "v0.20.0": "mixed precision,\nBLR tail",
    "v0.22.0": "bit-identity fix,\nAutoRace prefixes",
    "v0.23.0": "circuit family",
    "v0.24.0": "clean-room ND,\nKLU 32-bit kernels",
    "v0.25.0": "KLU parallel\nrefactor",
    "v0.26.0": "KLU parallel\nby default",
    "v0.27.0": "Hopcroft-Karp BTF",
    "v0.28.0": "consolidation,\nordering race",
}

METRICS = {
    "factor": ("factor only", lambda r: r["fac_ms"]),
    "e2e": ("analyze + factor + solve", lambda r: r["ana_ms"] + r["fac_ms"] + r["slv_ms"]),
}
YLABEL = "wall time / Accelerate (lower is faster)"
TICKS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit(f"no records in {path}")
    return rows


def versions(rows):
    return sorted({r["version"] for r in rows},
                  key=lambda v: tuple(int(p) for p in v.lstrip("v").split(".")))


def accel_ref(rows, metric, version=None):
    """(family, name) -> Accelerate time under `metric`, for one RSLAB version.

    Accelerate is measured by the newest binary of each measuring session, so the
    file carries one accel block per session. A version is normalized against the
    accel of its own session: the smallest accel-carrying version that is not
    older than it. Mixing sessions would compare an RSLAB time from a heat-soaked
    machine against an Accelerate time from a cool one - on this fanless machine
    that is a factor of two, enough to invent regressions in paths that did not
    change."""
    cost = METRICS[metric][1]
    key = lambda v: tuple(int(p) for p in v.lstrip("v").split("."))
    carriers = sorted({r["version"] for r in rows if r["solver"] == "accel"}, key=key)
    if not carriers:
        return {}
    if version is None:
        version = carriers[-1]
    ref_v = next((c for c in carriers if key(c) >= key(version)), carriers[-1])
    return {(r["family"], r["name"]): cost(r) for r in rows
            if r["version"] == ref_v and r["solver"] == "accel" and r["res"] < 0.1}


def normalized(rows, ref, metric, version, family, solver):
    """name -> RSLAB / Accelerate wall time (1.0 = Accelerate, lower is faster)."""
    cost = METRICS[metric][1]
    return {r["name"]: cost(r) / ref[(family, r["name"])] for r in rows
            if r["version"] == version and r["family"] == family
            and r["solver"] == solver and r["res"] < 0.1
            and (family, r["name"]) in ref and cost(r) > 0}


def geomean(xs):
    xs = [x for x in xs if x > 0 and not np.isnan(x)]
    return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")


def log_axis(ax, axis="y"):
    getattr(ax, f"set_{axis}scale")("log")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(plt.FixedLocator(TICKS))
    a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(plt.NullFormatter())


def per_class(rows, metric, version):
    """[(label, geomean normalized time, matrices, color)] for one version."""
    ref = accel_ref(rows, metric, version)
    color = {fam: c for fam, _, _, c in PATHS}
    out = []
    for prefix, (label, fam, solv) in CLASSES.items():
        vals = [v for n, v in normalized(rows, ref, metric, version, fam, solv).items()
                if n.split("_")[0] == prefix]
        if vals:
            out.append((label, geomean(vals), len(vals), color[fam]))
    return out


def class_bars(rows, ax):
    """Per class at the newest version: factor only vs one-shot, normalized."""
    latest = versions(rows)[-1]
    fac = per_class(rows, "factor", latest)
    e2e = {b[0]: b[1] for b in per_class(rows, "e2e", latest)}
    fac.sort(key=lambda b: -e2e.get(b[0], b[1]))

    ys = np.arange(len(fac))
    h = 0.36
    vals = list(e2e.values()) + [b[1] for b in fac]
    # Linear axis: on a log axis a bar's length is not proportional to its value.
    ax.barh(ys + h / 2, [e2e[b[0]] for b in fac], h, color=[b[3] for b in fac])
    ax.barh(ys - h / 2, [b[1] for b in fac], h, color=[b[3] for b in fac], alpha=0.45)
    ax.axvline(1.0, color=st.GRAY, lw=1.0, ls="--")
    ax.text(1.0, len(fac) - 0.4, " Accelerate", color=st.GRAY, fontsize=8,
            va="center", ha="left")
    for y, b in zip(ys, fac):
        for val, dy, alpha in ((e2e[b[0]], h / 2, 1.0), (b[1], -h / 2, 0.8)):
            ax.text(val + max(vals) * 0.015, y + dy, f"{val:.2f}", va="center",
                    fontsize=8, color=b[3], alpha=alpha)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{b[0]} ({b[2]})" for b in fac], fontsize=8)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel(YLABEL, fontsize=9)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_title(f"RSLAB {latest} vs Accelerate, per matrix class (M3, 8 threads)",
                 fontsize=9)
    handles = [Patch(facecolor=st.DARKGRAY, label="one-shot"),
               Patch(facecolor=st.DARKGRAY, alpha=0.45, label="factor only")]
    handles += [Patch(facecolor=c, label=lbl) for _, _, lbl, c in PATHS]
    return fac, e2e, handles


def timeline(rows, ax, metric, annotate=True):
    """Geomean normalized wall time per release, one line per shipped path."""
    ref = accel_ref(rows, metric)
    vers = versions(rows)
    xs = np.arange(len(vers))
    series = {}
    for fam, solv, label, color in PATHS:
        ys = [geomean(list(normalized(rows, accel_ref(rows, metric, v), metric, v,
                                      fam, solv).values()))
              for v in vers]
        series[label] = ys
        ax.plot(xs, ys, marker="o", ms=3.5, color=color, lw=1.4, label=label)
    mean = [geomean([series[l][i] for _, _, l, _ in PATHS]) for i in range(len(vers))]
    ax.plot(xs, mean, color=st.DARKGRAY, lw=2.2, ls=":", label="mean over all paths")
    ax.axhline(1.0, color=st.GRAY, lw=1.0, ls="--")
    ax.text(len(vers) - 0.5, 1.0, "Accelerate", color=st.GRAY, fontsize=7,
            va="bottom", ha="right")

    log_axis(ax, "y")
    lo = np.nanmin([y for ys in series.values() for y in ys])
    hi = np.nanmax([y for ys in series.values() for y in ys])
    if annotate:
        for i, v in enumerate(vers):
            note = RELEASE_NOTES.get(v)
            if not note:
                continue
            ha = "left" if i == 0 else ("right" if i == len(vers) - 1 else "center")
            ax.annotate(note, (i, hi), xytext=(0, 12 if i % 2 == 0 else 30),
                        textcoords="offset points", ha=ha, fontsize=6,
                        color="black" if st.REPORT else st.GRAY)
        hi *= 2.6
    ax.set_ylim(lo / 1.3, hi * 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([v.lstrip("v") for v in vers], fontsize=7, rotation=60,
                       ha="right")
    ax.set_ylabel(YLABEL, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title(f"Per release, same matrices ({METRICS[metric][0]})", fontsize=9)
    return vers, series, mean


def scaling(rows, ax, metric="factor"):
    """Normalized wall time vs problem size, one line per matrix class (a family
    cycles through its classes size by size, so a per-family line would zigzag
    between unrelated problems)."""
    ref = accel_ref(rows, metric)
    latest = versions(rows)[-1]
    cost = METRICS[metric][1]
    color = {fam: c for fam, _, _, c in PATHS}
    for prefix, (label, fam, solv) in CLASSES.items():
        pts = sorted((r["nnz"], cost(r) / ref[(fam, r["name"])]) for r in rows
                     if r["version"] == latest and r["family"] == fam
                     and r["solver"] == solv and r["name"].split("_")[0] == prefix
                     and (fam, r["name"]) in ref and r["res"] < 0.1)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.5,
                color=color[fam], lw=1.3)
        ax.annotate(f" {label}", pts[-1], fontsize=6.5, color=color[fam],
                    va="center", ha="left")
    ax.axhline(1.0, color=st.GRAY, lw=1.0, ls="--")
    ax.text(ax.get_xlim()[0], 1.0, " Accelerate", color=st.GRAY, fontsize=7,
            va="bottom", ha="left")
    ax.set_xscale("log")
    log_axis(ax, "y")
    ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
    ax.set_xlabel("nonzeros in A", fontsize=9)
    ax.set_ylabel(YLABEL, fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_title("Wall time vs problem size (factor only)", fontsize=9)


def card(fig, name):
    out = OUT / name
    fig.savefig(out, dpi=200, transparent=False, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


def main():
    rows = load(PATH)
    if not accel_ref(rows, "factor"):
        sys.exit("no Accelerate reference records (solver=accel) in the history file")
    st.setup()

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    fac, e2e, handles = class_bars(rows, ax)
    st.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles],
                    fontsize=8)
    st.save(fig, OUT / "accel_classes.png")
    card(fig, "accel_social_classes.png")

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    vers, series, mean = timeline(rows, ax, "factor")
    st.legend_below(fig, ax=ax, fontsize=8)
    st.save(fig, OUT / "accel_timeline.png")
    card(fig, "accel_social_timeline.png")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    timeline(rows, ax_l, "factor")
    _, series_e2e, mean_e2e = timeline(rows, ax_r, "e2e", annotate=False)
    st.legend_below(fig, ax=ax_l, fontsize=8)
    st.save(fig, OUT / "accel_timeline_both.png")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    scaling(rows, ax)
    st.save(fig, OUT / "accel_scaling.png")
    card(fig, "accel_social_scaling.png")

    print("\n== wall time / Accelerate, geomean (factor / one-shot) ==")
    print("| version | " + " | ".join(l for _, _, l, _ in PATHS) + " | mean |")
    print("|---" * (len(PATHS) + 2) + "|")
    for i, v in enumerate(vers):
        cells = " | ".join(
            f"{series[l][i]:.2f} / {series_e2e[l][i]:.2f}"
            if not np.isnan(series[l][i]) else "-" for _, _, l, _ in PATHS)
        print(f"| {v} | {cells} | {mean[i]:.2f} / {mean_e2e[i]:.2f} |")
    print("\n== per class, current version (factor / one-shot) ==")
    for label, val, n, _ in sorted(fac, key=lambda b: e2e[b[0]]):
        print(f"  {label:<26} {val:.2f} / {e2e[label]:.2f}  ({n} matrices)")


if __name__ == "__main__":
    main()
