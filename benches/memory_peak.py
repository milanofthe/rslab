"""RSLAB's memory estimate against the exact heap peak of each phase.

    python benches/memory_peak.py <corpus-dir or .npz files> [--threads 12]

Needs the extension built with the `alloc-stats` feature, which counts every
byte the Rust side allocates:

    cargo build --release --features alloc-stats   (in python/)

Per system (the `pardiso_corpus.py` format) it measures the heap peak of the
analysis, of the factorization with the analysis held, and of a solve with the
factor held, each above what was live when the phase began, plus what stays
live after each. These are the Rust allocations only: the input arrays belong
to NumPy, and allocator caching and thread stacks come on top in the process's
working set, which is reported alongside. Results append to
`benches/bench_out/memory_peak.jsonl`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pardiso_corpus import load, partner  # noqa: E402

OUT = Path(__file__).resolve().parent / 'bench_out' / 'memory_peak.jsonl'


def phase(work):
    """Runs `work`; returns (result, heap peak above the start, live change)."""
    from rslab import _rslab
    live0, _ = _rslab._alloc_stats()
    _rslab._reset_alloc_peak()
    result = work()
    live1, peak = _rslab._alloc_stats()
    return result, peak - live0, live1 - live0


def measure(path, threads):
    import rslab
    a, b, meta = load(path)
    route = meta['path']
    settings = meta.get('settings', {}).get('rslab', {})
    if route != 'klu':
        settings = {'threads': threads, **settings}
    numeric = {k: v for k, v in settings.items() if k != 'ordering'}
    b = np.ascontiguousarray(b)
    sym, analyze_peak, analyze_live = phase(lambda: rslab.analyze(a, path=route, **settings))
    # The preflight: the plan before any numeric work (it may build the
    # schedule, which then belongs to the analysis the factorization starts on).
    plan = sym.memory_plan(str(a.dtype), nrhs=b.shape[1], **numeric)
    analyzed_held = sym.heap_bytes
    factor, factor_peak, factor_live = phase(lambda: sym.factor(a, **numeric))
    sym_held = sym.heap_bytes
    _, solve_peak, _ = phase(lambda: factor.solve_many(b))
    estimate = sym.estimate_memory(str(a.dtype))
    return {
        'n': a.shape[0], 'nnz': int(a.nnz), 'dtype': str(a.dtype), 'path': route,
        'nrhs': b.shape[1], 'threads': threads,
        'analyze_peak': analyze_peak, 'analyze_live': analyze_live,
        'factor_peak': factor_peak, 'factor_live': factor_live,
        'solve_peak': solve_peak, 'estimate': estimate, 'plan': plan,
        'analyzed_held': analyzed_held, 'symbolic_held': sym_held, 'factor_held': factor.heap_bytes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', type=Path, nargs='+')
    p.add_argument('--threads', type=int, default=12)
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    from rslab import _rslab
    if not hasattr(_rslab, '_alloc_stats'):
        sys.exit('rslab was built without the alloc-stats feature')
    files = sorted(f for i in args.inputs for f in (i.rglob('*.npz') if i.is_dir() else [i]))
    files = [f for f in files if not any(partner(g) == f and g != f for g in files)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mb = 2**20
    print(f"{'system':26s} {'path':4s} {'peak':>8s} {'plan':>6s} {'factor':>8s} {'plan':>6s} "
          f"{'held':>8s} {'plan':>6s} {'solve':>7s} {'plan':>6s}   MB, plan / measured")
    for f in files:
        row = {'system': f.stem, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **measure(f, args.threads)}
        with args.out.open('a') as out:
            out.write(json.dumps(row) + '\n')
        plan = row['plan']
        # From the analysis on: the factorization, then a solve with the factor held.
        peak = row['analyzed_held'] + max(row['factor_peak'], row['factor_live'] + row['solve_peak'])
        pairs = [(peak, plan['peak_bytes']),
                 (row['factor_peak'], plan['factor_peak_bytes']),
                 (row['factor_live'], plan['analysis_growth_bytes'] + plan['factor_bytes']),
                 (row['solve_peak'], plan['solve_bytes'])]
        cols = ' '.join(f"{m / mb:8.1f} {p / max(m, 1):6.2f}" for m, p in pairs)
        print(f"{f.stem:26s} {row['path']:4s} {cols}", flush=True)


if __name__ == '__main__':
    main()
