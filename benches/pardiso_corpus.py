"""RSLAB against MKL PARDISO on a corpus of real systems, one worker process per solver and system.

    python benches/pardiso_corpus.py <corpus-dir or .npz files> [--threads 12] [--repeats 3]

A corpus file is an `.npz` with the matrix as CSC (`indptr`, `indices`, `data`,
both triangles), right-hand sides `b` (n x nrhs) and a JSON `meta` naming the
path (`ldlt`, `lu` or `klu`) and optionally `settings` per solver family, the
options the system's source uses in production. A file `<name>_fNN.npz` takes
its refactorization values from `<name>_f<NN+1>.npz` when that exists.

Per repeat a worker times analysis, factorization, solve and refactorization,
then refines the refactorization to a relative residual of 1e-10 (plain
iterative refinement, the same for both solvers; `to_target` runs from new
values to that answer). It records the peak working
set above the loaded matrix, PARDISO's own memory report and RSLAB's
diagnostics (stage times, memory estimate). Both solvers run with the same
thread count. `pardiso` runs `pardisoinit`'s defaults for the matrix type,
`pardiso-2l` adds the two-level factorization (`iparm(24) = 1`). Results append
to `benches/bench_out/pardiso_corpus.jsonl`.

PARDISO comes from the MKL runtime of `pip install mkl` in the running
environment, or from the library `MKL_RT` names.
"""
import argparse
import ctypes
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

OUT = Path(__file__).resolve().parent / 'bench_out' / 'pardiso_corpus.jsonl'
TARGET = 1e-10


def load(path):
    """(A as CSC, b as n x nrhs, meta) for one corpus file."""
    with np.load(path) as f:
        n = len(f['indptr']) - 1
        a = sp.csc_matrix((f['data'], f['indices'], f['indptr']), shape=(n, n))
        b, meta = f['b'].reshape(n, -1).astype(a.dtype), json.loads(str(f['meta']))
    return a, b, meta


def partner(path):
    """The next sweep point sharing the pattern, else the file itself."""
    m = re.fullmatch(r'(.*_f)(\d+)\.npz', path.name)
    if m:
        nxt = path.with_name(f'{m[1]}{int(m[2]) + 1:0{len(m[2])}d}.npz')
        if nxt.exists():
            return nxt
    return path


def residual(a, x, b):
    """Largest relative residual over the columns."""
    return float(np.max(np.linalg.norm(b - a @ x, axis=0) / np.linalg.norm(b, axis=0)))


class Rslab:
    def __init__(self, path, settings, threads):
        import rslab
        self.rslab, self.path, self.factor = rslab, path, None
        # KLU parallelizes over its BTF blocks on its own and takes no count.
        self.settings = settings if path == 'klu' else {'threads': threads, **settings}

    def analyze(self, a):
        self.sym = self.rslab.analyze(a, path=self.path, **self.settings)

    def factorize(self, a):
        if self.factor is not None and self.path == 'klu':
            self.factor.refactor(a)  # KLU keeps its pivot sequence across a sweep
            return
        numeric = {k: v for k, v in self.settings.items() if k != 'ordering'}
        self.factor = None  # free the previous factor first, as PARDISO reuses its storage
        self.factor = self.sym.factor(a, **numeric)

    def solve(self, b):
        return np.asarray(self.factor.solve_many(np.ascontiguousarray(b))).reshape(b.shape)

    def report(self):
        d = self.factor.diagnostics()
        est = self.sym.estimate_memory(str(np.dtype(self.dtype)))
        return {'stages': {s['name']: s['wall_ms'] / 1e3 for s in d.get('stages', [])},
                'factor_nnz': d.get('factor_nnz'), 'decisions': d.get('decisions'),
                'factor_bytes': next((s['bytes'] for s in d.get('stages', []) if s['name'] == 'factor'), None),
                'estimate': est}


def load_mkl():
    """The MKL runtime: `MKL_RT` if set, else the one `pip install mkl` put
    into this environment, else the system's."""
    names = [os.environ['MKL_RT']] if os.environ.get('MKL_RT') else []
    names += glob.glob(str(Path(sys.prefix) / 'Library' / 'bin' / 'mkl_rt*.dll'))
    names += glob.glob(str(Path(sys.prefix) / 'lib' / 'libmkl_rt.so*')) + ['libmkl_rt.so.2']
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise OSError('MKL runtime not found: `pip install mkl` into this environment '
                  'or set MKL_RT to the mkl_rt library')


class Pardiso:
    def __init__(self, path, settings, threads, two_level=False):
        self.two_level = two_level
        self.mkl = load_mkl()
        self.kind, self.settings = ('ldlt' if path == 'ldlt' else 'lu'), settings
        self.pt, self.iparm, self.mtype = np.zeros(64, np.int64), np.zeros(64, np.int32), None

    def _call(self, phase, values, b=None, x=None):
        i32, err = ctypes.c_int32, ctypes.c_int32(0)
        nrhs = 1 if b is None else b.shape[0]
        dummy = np.zeros(1, self.dtype)
        self.mkl.pardiso(self.pt.ctypes, ctypes.byref(i32(1)), ctypes.byref(i32(1)), ctypes.byref(i32(self.mtype)),
                         ctypes.byref(i32(phase)), ctypes.byref(i32(self.n)), values.ctypes, self.ia.ctypes,
                         self.ja.ctypes, ctypes.byref(i32(0)), ctypes.byref(i32(nrhs)), self.iparm.ctypes,
                         ctypes.byref(i32(0)), (dummy if b is None else b).ctypes,
                         (dummy if x is None else x).ctypes, ctypes.byref(err))
        if err.value:
            raise RuntimeError(f'PARDISO phase {phase} returned error {err.value}')

    def prepare(self, a):
        """PARDISO's input: the upper triangle when symmetric, one-based CSR."""
        stored = sp.triu(a, format='csr') if self.kind == 'ldlt' else a.tocsr()
        stored.sort_indices()
        return stored

    def analyze(self, stored):
        real = not np.iscomplexobj(stored.data)
        self.dtype = np.float64 if real else np.complex128
        self.mtype = (-2 if real else 6) if self.kind == 'ldlt' else (11 if real else 13)
        self.mkl.pardisoinit(self.pt.ctypes, ctypes.byref(ctypes.c_int32(self.mtype)), self.iparm.ctypes)
        for index, value in self.settings.get('iparm', {}).items():
            self.iparm[int(index)] = value  # zero-based overrides from the corpus
        if self.two_level:
            self.iparm[23] = 1  # iparm(24): the two-level factorization, Intel's advice for many threads
        self.n = stored.shape[0]
        self.ia = (stored.indptr + 1).astype(np.int32)
        self.ja = (stored.indices + 1).astype(np.int32)
        self._call(11, np.ascontiguousarray(stored.data))

    def factorize(self, stored):
        self.values = np.ascontiguousarray(stored.data)
        self._call(22, self.values)

    def solve(self, b):
        rhs = np.ascontiguousarray(b.T, dtype=self.dtype)
        x = np.zeros_like(rhs)
        self._call(33, self.values, rhs, x)
        return x.T

    def report(self):
        # iparm(15..18), one-based: peak analysis, permanent and factor memory in KB, nnz of the factors.
        return {'peak_analysis_kb': int(self.iparm[14]), 'permanent_kb': int(self.iparm[15]),
                'factor_kb': int(self.iparm[16]), 'factor_nnz': int(self.iparm[17]),
                'mkl_threads': int(self.mkl.mkl_get_max_threads()), 'mtype': self.mtype,
                'iparm24': int(self.iparm[23])}

    def close(self):
        self._call(-1, np.zeros(1, self.dtype))


SOLVERS = {'rslab': Rslab, 'pardiso': Pardiso,
           'pardiso-2l': lambda path, settings, threads: Pardiso(path, settings, threads, two_level=True)}


def working_set_mb():
    """(current, peak) working set of this process in MB (peak RSS on Linux)."""
    import psutil
    m = psutil.Process().memory_info()
    return m.rss / 2**20, getattr(m, 'peak_wset', m.rss) / 2**20


def measure(name, first, second, repeats, warmup, threads):
    a, b, meta = load(first)
    a2, b2, _ = load(second)
    path = meta['path']
    runs, report = [], {}
    base_mb = working_set_mb()[0]
    for round_ in range(warmup + repeats):
        # A variant takes the production settings of its family (`pardiso-2l`: `pardiso`).
        solver = SOLVERS[name](path, meta.get('settings', {}).get(name.split('-')[0], {}), threads)
        solver.dtype = a.dtype
        prep = getattr(solver, 'prepare', lambda m: m)
        native, native2 = prep(a), prep(a2)
        clock = time.perf_counter
        t0 = clock(); solver.analyze(native)
        t1 = clock(); solver.factorize(native)
        t2 = clock(); x = solver.solve(b)
        t3 = clock(); solver.factorize(native2)
        t4 = clock(); x2 = solver.solve(b2)
        t5 = clock()
        steps = 0
        while residual(a2, x2, b2) > TARGET and steps < 50:
            x2 = x2 + solver.solve(b2 - a2 @ x2)
            steps += 1
        t6 = clock()
        run = {'analyze': t1 - t0, 'factor': t2 - t1, 'solve': t3 - t2, 'refactor': t4 - t3,
               'residual': residual(a, x, b), 'to_target': t6 - t3, 'refine_steps': steps,
               'refined_residual': residual(a2, x2, b2)}
        report = solver.report()
        if hasattr(solver, 'close'):
            solver.close()
        if round_ >= warmup:
            runs.append(run)
    peak_mb = working_set_mb()[1]
    return {'path': path, 'n': a.shape[0], 'nnz': int(a.nnz), 'dtype': str(a.dtype), 'nrhs': b.shape[1],
            'threads': threads, 'runs': runs, 'report': report, 'peak_mb': peak_mb - base_mb}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', type=Path, nargs='*')
    p.add_argument('--solvers', nargs='+', choices=list(SOLVERS), default=list(SOLVERS))
    p.add_argument('--threads', type=int, default=0, help='worker threads, 0 for the physical cores')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        name, first, second = args.worker
        # Before MKL loads: it reads the count once.
        os.environ['MKL_NUM_THREADS'] = str(args.threads)
        try:
            result = measure(name, Path(first), Path(second), args.repeats, args.warmup, args.threads)
        except Exception as exc:
            result = {'error': f'{type(exc).__name__}: {exc}'}
        print('RESULT ' + json.dumps(result), flush=True)
        os._exit(0)

    import psutil
    threads = args.threads or psutil.cpu_count(logical=False)
    files = sorted(f for i in args.inputs for f in (i.rglob('*.npz') if i.is_dir() else [i]))
    # Sweep partners are refactorization values, not systems of their own.
    files = [f for f in files if not any(partner(g) == f and g != f for g in files)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for f in files:
        for name in args.solvers:
            cmd = [sys.executable, __file__, '--worker', name, str(f), str(partner(f)), '--threads', str(threads),
                   '--repeats', str(args.repeats), '--warmup', str(args.warmup)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            lines = [l for l in done.stdout.splitlines() if l.startswith('RESULT ')]
            result = json.loads(lines[-1][7:]) if lines else {'error': (done.stderr or done.stdout).strip()[-300:]}
            row = {'system': f.stem, 'group': f.parent.name, 'solver': name,
                   'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **result}
            with args.out.open('a') as out:
                out.write(json.dumps(row) + '\n')
            if 'error' in row:
                print(f'{f.stem:<26}{name:<9}failed: {row["error"][:90]}', flush=True)
            else:
                med = {k: statistics.median(r[k] for r in row['runs']) for k in ('analyze', 'factor', 'solve')}
                print(f'{f.stem:<26}{name:<9}analyze {med["analyze"]:7.3f}  factor {med["factor"]:7.3f}  '
                      f'solve {med["solve"]:6.3f}  res {row["runs"][-1]["residual"]:.1e}  peak {row["peak_mb"]:7.0f} MB',
                      flush=True)


if __name__ == '__main__':
    main()
