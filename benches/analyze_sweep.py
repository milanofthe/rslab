import json, sys
from collections import defaultdict
from statistics import median

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
# index by matrix
by_m = defaultdict(list)
for r in recs:
    by_m[r["matrix"]].append(r)

def key(p):
    return (p["ordering"], p["nemin"], p["par_cdiv"], p["threads"])

def gm(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs: return float('nan')
    import math
    return math.exp(sum(math.log(x) for x in xs)/len(xs))

# 1) par_cdiv lever: hold (ordering,nemin,threads), compare 8M vs 2M factor_ms.
print("=== par_cdiv lever (8M -> 2M), threads=0 (all cores) ===")
print(f'{"matrix":<18}{"front_nrow_max":>14}{"flop_top1":>10}{"ms@8M":>9}{"ms@2M":>9}{"speedup":>9}')
speeds = []
for m, rs in by_m.items():
    d = {key(r["params"]): r for r in rs}
    # pick amd, nemin16, threads0
    base = ("amd", 16, 8_000_000, 0)
    low  = ("amd", 16, 2_000_000, 0)
    if base in d and low in d:
        t8 = d[base]["metrics"]["factor_ms"]; t2 = d[low]["metrics"]["factor_ms"]
        sp = t8/t2 if t2 else float('nan')
        speeds.append(sp)
        f = d[base]["features"]
        print(f'{m:<18}{f["front_nrow_max"]:>14}{f["flop_top1_frac"]:>10.3f}{t8:>9.2f}{t2:>9.2f}{sp:>9.2f}')
print(f'  geomean speedup (8M/2M): {gm(speeds):.3f}  (>1 means lowering par_cdiv helped)')

# 2) nemin lever: amd, par_cdiv 8M, threads0, 16 vs 48
print("\n=== nemin lever (16 -> 48), amd, threads=0 ===")
print(f'{"matrix":<18}{"ms@16":>9}{"ms@48":>9}{"fill@16":>10}{"fill@48":>10}{"t_speedup":>10}')
tsp=[];
for m, rs in by_m.items():
    d = {key(r["params"]): r for r in rs}
    a=("amd",16,8_000_000,0); b=("amd",48,8_000_000,0)
    if a in d and b in d:
        ta=d[a]["metrics"]["factor_ms"]; tb=d[b]["metrics"]["factor_ms"]
        fa=d[a]["metrics"]["factor_nnz"]; fb=d[b]["metrics"]["factor_nnz"]
        s=ta/tb if tb else float('nan'); tsp.append(s)
        print(f'{m:<18}{ta:>9.2f}{tb:>9.2f}{fa:>10}{fb:>10}{s:>10.2f}')
print(f'  geomean time speedup (16/48): {gm(tsp):.3f}')

# 3) ordering: amd vs metis (nemin16, 8M, threads0)
print("\n=== ordering: amd vs metis (nemin16, 8M, threads0) ===")
print(f'{"matrix":<18}{"amd_ms":>9}{"metis_ms":>10}{"amd_fill":>10}{"metis_fill":>11}{"win":>8}')
for m, rs in by_m.items():
    d = {key(r["params"]): r for r in rs}
    a=("amd",16,8_000_000,0); b=("metis",16,8_000_000,0)
    if a in d and b in d:
        ta=d[a]["metrics"]["factor_ms"]; tb=d[b]["metrics"]["factor_ms"]
        fa=d[a]["metrics"]["factor_nnz"]; fb=d[b]["metrics"]["factor_nnz"]
        win = "amd" if ta<tb else "metis"
        print(f'{m:<18}{ta:>9.2f}{tb:>10.2f}{fa:>10}{fb:>11}{win:>8}')

# 4) best combo vs baseline (amd,16,8M,0)
print("\n=== headroom: best-in-grid vs baseline (amd,16,8M,thr0) by factor_ms ===")
gains=[]
for m, rs in by_m.items():
    d = {key(r["params"]): r for r in rs}
    base=("amd",16,8_000_000,0)
    if base not in d: continue
    bt=d[base]["metrics"]["factor_ms"]
    best=min(rs, key=lambda r:r["metrics"]["factor_ms"])
    g=bt/best["metrics"]["factor_ms"] if best["metrics"]["factor_ms"] else float('nan')
    gains.append(g)
    bp=best["params"]
    print(f'{m:<18} base {bt:>8.2f}ms  best {best["metrics"]["factor_ms"]:>8.2f}ms  x{g:>5.2f}  via {bp["ordering"]}/nemin{bp["nemin"]}/cdiv{bp["par_cdiv"]//1000000}M/thr{bp["threads"]}')
print(f'  geomean headroom: {gm(gains):.3f}')

# residual sanity
bad=[r for r in recs if not (r["metrics"]["residual"]<1e-6)]
print(f'\nrecords with residual >= 1e-6: {len(bad)} / {len(recs)}')
