"""Apple-Silicon bench: RSLAB shipped default vs Apple Accelerate Sparse Solvers.

Reads the three same-run datasets produced by `benches/run_apple_silicon.sh`
(`apple_sym.jsonl`, `apple_unsym.jsonl`, `apple_corpus.jsonl`) and produces

* one two-panel figure per generator family - factor wall-clock time (left) and
  live-bytes peak memory (right) vs nonzeros, log-log, with a power-law fit per
  solver (`h2h_apple_ldlt.png` / `h2h_apple_lu.png`),
* the corpus residual scatter (`apple_corpus_residual.png`),
* head-to-head geomean ratios (Accelerate / RSLAB, over the matrices both solve
  to `< 0.1` residual), printed as a markdown table and written to
  `apple_geomean.json` for the report.

The Accelerate variant per matrix (native LDLT for real-symmetric, sparse LU on
the full matrix for complex-symmetric, sparse LU for unsymmetric) is recovered
from the run log (`apple_run.log`) when present.

Run: python benches/apple_silicon.py [bench_out_dir]
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from matplotlib.lines import Line2D

import bench_style
from bench_style import SOLVERS
from fit_scaling import plot_metric, plot_residual

ORDER = ["auto", "accel"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def variants(log_path):
    """name -> accel variant (chol / ldlt / lu-full / lu), parsed from the run log."""
    out = {}
    if log_path.exists():
        for m in re.finditer(
                r"\[accel\] (\S+): (chol|ldlt|lu-full|lu)(?: \((\w+)\))?,",
                log_path.read_text()):
            out[m.group(1)] = m.group(2)
    return out


def family_figure(recs, title, slug, out_dir, order=ORDER):
    print(f"== {title} ==")
    fig, (ax_wct, ax_mem) = bench_style.two_panel()
    print("  factor time ~ nnz^alpha:")
    present = plot_metric(recs, "time", "fac_ms", "factor time [ms]", None,
                          order=order, ax=ax_wct)
    print("  peak memory ~ nnz^alpha:")
    plot_metric(recs, "mem", "mem_mb", "peak memory [MB]", None, order=order, ax=ax_mem)
    fig.suptitle(f"{title} — factor time & peak memory (Apple M3)",
                 color=bench_style.GRAY)
    handles = [Line2D([], [], color=c, marker=mk, ls="", label=lbl)
               for _, lbl, c, mk in present]
    bench_style.legend_below(fig, handles=handles, labels=[h.get_label() for h in handles])
    bench_style.save(fig, out_dir / f"h2h_apple_{slug}.png")


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def head_to_head(recs, key_metric, baseline="auto"):
    """(accel / baseline) ratios per matrix for one metric pass; only matrices
    where both solvers produced a record with residual < 0.1."""
    by = defaultdict(dict)
    for r in recs:
        if r.get("metric") == key_metric and r.get("res", 1.0) < 0.1:
            by[r["name"]][r["solver"]] = r
    val = "fac_ms" if key_metric == "time" else "mem_mb"
    ratios = {name: d["accel"][val] / d[baseline][val]
              for name, d in by.items()
              if "accel" in d and baseline in d
              and d[baseline][val] > 0 and d["accel"][val] > 0}
    return ratios


def solve_counts(recs):
    """Per solver: matrices attempted / solved to < 1e-8 (time pass)."""
    seen, ok = defaultdict(set), defaultdict(set)
    names = set()
    for r in recs:
        if r.get("metric") != "time":
            continue
        names.add(r["name"])
        seen[r["solver"]].add(r["name"])
        if r.get("res", 1.0) < 1e-8:
            ok[r["solver"]].add(r["name"])
    return names, seen, ok


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/bench_out")
    bench_style.setup()
    sym = load(out_dir / "apple_sym.jsonl")
    unsym = load(out_dir / "apple_unsym.jsonl")
    circuit = load(out_dir / "apple_circuit.jsonl")
    corpus = load(out_dir / "apple_corpus.jsonl")
    var = variants(out_dir / "apple_run.log")

    family_figure(sym, "LDLt path (sym)", "ldlt", out_dir)
    family_figure(unsym, "LU path (unsym)", "lu", out_dir)
    family_figure(circuit, "KLU path (circuit)", "klu", out_dir,
                  order=["auto", "klu", "accel"])
    plot_residual(corpus, out_dir / "apple_corpus_residual.png", order=ORDER)

    summary = {}
    print("\n== head-to-head geomean (Apple Accelerate / RSLAB shipped path) ==")
    print("| corpus | factor time | peak memory | matrices |")
    print("|---|---|---|---|")
    for label, recs, base in [("sym (LDLt path)", sym, "auto"),
                              ("unsym (LU path)", unsym, "auto"),
                              ("circuit (KLU path)", circuit, "klu"),
                              ("SuiteSparse corpus", corpus, "auto")]:
        t = head_to_head(recs, "time", base)
        m = head_to_head(recs, "mem", base)
        gt, gm = geomean(t.values()), geomean(m.values())
        summary[label] = {"time_ratio": gt, "mem_ratio": gm,
                          "n_time": len(t), "n_mem": len(m), "baseline": base}
        print(f"| {label} | {gt:.2f}x | {gm:.2f}x | {len(t)} |")

    names, seen, ok = solve_counts(corpus)
    summary["corpus_accuracy"] = {
        s: {"solved_1e8": len(ok[s]), "attempted": len(seen[s])} for s in ORDER}
    summary["corpus_total"] = len(names)
    summary["accel_variants"] = {v: sorted(k for k, vv in var.items() if vv == v)
                                 for v in ("chol", "ldlt", "lu-full", "lu")}
    print(f"\ncorpus accuracy (< 1e-8): " + ", ".join(
        f"{s}: {len(ok[s])}/{len(seen[s])}" for s in ORDER))
    for v, ks in summary["accel_variants"].items():
        print(f"accel variant {v}: {len(ks)} matrices")

    with open(out_dir / "apple_geomean.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_dir / 'apple_geomean.json'}")


if __name__ == "__main__":
    main()
