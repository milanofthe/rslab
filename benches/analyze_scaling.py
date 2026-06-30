"""Analyze the thread-scaling sweep (RLA_SWEEP_THREADS_ONLY output).

Per matrix the sweep records factor time at a ladder of worker counts at fixed
production-default knobs. This builds the speedup curve speedup(t)=ms@1/ms@t,
the parallel efficiency, the saturation point, and correlates the achieved
scaling with structural features - so we can later *predict* how well a system
will scale from its fingerprint.

Usage: python benches/analyze_scaling.py benches/bench_out/sweep_threads.jsonl
"""
import json, sys, math
from collections import defaultdict

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
# Suspend-artifact guard: no real factor here exceeds tens of seconds.
recs = [r for r in recs if r["metrics"]["factor_ms"] < 1_000_000]

by = defaultdict(dict)        # matrix -> {threads: ms}
feat = {}
meta = {}
for r in recs:
    by[r["matrix"]][r["params"]["threads"]] = r["metrics"]["factor_ms"]
    feat[r["matrix"]] = r["features"]
    meta[r["matrix"]] = (r["n"], r["nnz"])

PHYS = 12  # physical cores

def speedup_curve(d):
    if 1 not in d:
        return None
    t1 = d[1]
    return {t: t1 / ms for t, ms in sorted(d.items()) if ms > 0}

rows = []
for m, d in by.items():
    cur = speedup_curve(d)
    if not cur or len(cur) < 3:
        continue
    sp_phys = cur.get(PHYS) or cur.get(max(t for t in cur if t <= PHYS), 1.0)
    sp_max = max(cur.values())
    t_best = max(cur, key=cur.get)
    eff_phys = sp_phys / PHYS
    f = feat[m]
    n, nnz = meta[m]
    rows.append({
        "matrix": m, "n": n, "flops": f["factor_flops"],
        "tree_width_max": f["tree_width_max"], "flop_top1": f["flop_top1_frac"],
        "front_nrow_max": f["front_nrow_max"], "fill_ratio": f["fill_ratio"],
        "t1_ms": d[1], "sp_phys": sp_phys, "sp_max": sp_max, "t_best": t_best,
        "eff_phys": eff_phys, "curve": cur,
    })

rows.sort(key=lambda r: -r["sp_max"])
print(f"matrices with a usable curve: {len(rows)}\n")
print(f'{"matrix":<20}{"n":>8}{"flops":>9}{"t1_ms":>9}{"sp@12":>7}{"sp_max":>8}{"t_best":>7}{"eff@12":>7}')
for r in rows:
    print(f'{r["matrix"]:<20}{r["n"]:>8}{r["flops"]:>9.1e}{r["t1_ms"]:>9.1f}'
          f'{r["sp_phys"]:>7.2f}{r["sp_max"]:>8.2f}{r["t_best"]:>7}{r["eff_phys"]:>7.2f}')

def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

print(f'\ngeomean speedup @12 cores: {gm([r["sp_phys"] for r in rows]):.2f}')
print(f'geomean max speedup:      {gm([r["sp_max"] for r in rows]):.2f}')

# What predicts scaling? Correlate sp_max with candidate features (Spearman-ish
# via rank correlation, dependency-free).
def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    for pos, i in enumerate(order):
        rk[i] = pos
    return rk

def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 3:
        return float("nan")
    ma = sum(ra) / n; mb = sum(rb) / n
    cov = sum((ra[i]-ma)*(rb[i]-mb) for i in range(n))
    va = math.sqrt(sum((x-ma)**2 for x in ra)); vb = math.sqrt(sum((x-mb)**2 for x in rb))
    return cov/(va*vb) if va*vb else float("nan")

sp = [r["sp_max"] for r in rows]
print("\nrank-correlation of max-speedup with features:")
for k in ["flops", "n", "tree_width_max", "flop_top1", "front_nrow_max", "fill_ratio"]:
    print(f'  {k:<16} {spearman([r[k] for r in rows], sp):+.2f}')
print("\n(positive = larger feature -> better scaling; flops/front size usually dominate:"
      " more BLAS-3 work -> more parallelism to exploit.)")
