"""RSLAB against MKL PARDISO on a corpus of real systems, one worker process per solver and system.

    python benches/pardiso_corpus.py <corpus-dir or .npz files> [--threads 12] [--repeats 3]

A corpus file is an `.npz` with the matrix as CSC (`indptr`, `indices`, `data`,
both triangles), right-hand sides `b` (n x nrhs) and a JSON `meta` naming the
path (`ldlt`, `lu` or `klu`) and optionally `settings` per solver family, the
options the system's source uses in production. A file `<name>_fNN.npz` takes
its refactorization values from `<name>_f<NN+1>.npz` when that exists.

Per repeat a worker times analysis, factorization, solve and refactorization,
then refines the refactorization to a relative residual of 1e-10 (plain
iterative refinement, the same for both solvers, stopped where a step no
longer lowers the residual; `to_target` runs from new values to that answer). It records the peak working set above the loaded
matrix and solver library, PARDISO's own memory report and RSLAB's
diagnostics (stage times, memory estimate). Both solvers run with the same
thread count. `pardiso` runs `pardisoinit`'s defaults for the matrix type,
`pardiso-2l` adds the two-level factorization (`iparm(24) = 1`). Results append
to `benches/bench_out/pardiso_corpus.jsonl`.

PARDISO comes from the MKL runtime of `pip install mkl` in the running
environment, or from the library `MKL_RT` names.

A/B of RSLAB revisions: `--rslab A B` runs every system on two (or more)
revisions of RSLAB side by side with PARDISO, on one machine:

    python benches/pardiso_corpus.py <corpus> --rslab main v1.7.0 --rounds 3 [--family mom fem]

Each revision is a git ref of this repository or `LABEL=DIR`, a source tree
(the pool entry `benches/corpus/pool.sh` ships them so); its wheel is built
the way the published one is (`maturin build --release` in `python/`) into a
virtual environment of its own under `--build-dir`, once per commit, every
build through one cargo target directory. The workers of one system run
interleaved, `--rounds` times each (A, B, A, B, ...), so a drift of the
machine reaches every side alike; PARDISO, no side of the comparison, runs in
the first round only. An older revision runs without the settings of a system
it does not know (`settings_dropped` in its report). The comparison against the
first revision follows the runs (or `--compare FILE` prints it for a results
file): per system and metric the median ratio and the noise, and a verdict
per corpus family.

Noise criterion: the samples of a side are its timed repeats over all rounds
(times) or one value per round (peak working set). A side's spread is
`(max - min) / median` of its samples, the noise of a comparison the larger
spread of its two sides: a ratio beyond `1 + noise` (or below its inverse)
is a difference beyond noise. A family's verdict compares the geometric mean
of its systems' ratios with the median of their noise, which is conservative
(the mean of N ratios scatters less than one ratio). Without two samples per
side there is no noise and no verdict. Factor entries and pivot counts are
deterministic and compared exactly.
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
        # An older revision runs without the settings it does not know, and says which.
        kind = rslab.KluSettings if path == 'klu' else rslab.Settings
        self.dropped = []
        for key, value in settings.items():
            try:
                kind(**{key: value})
            except (TypeError, ValueError):
                self.dropped.append(key)
        settings = {k: v for k, v in settings.items() if k not in self.dropped}
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
                'estimate': est, 'numeric': d.get('numeric'), 'warnings': d.get('warnings'),
                'settings_dropped': self.dropped, 'version': self.rslab.__version__}


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
        # iparm(14..18), one-based: perturbed pivots, peak analysis, permanent and factor memory in
        # KB, nnz of the factors.
        return {'peak_analysis_kb': int(self.iparm[14]), 'permanent_kb': int(self.iparm[15]),
                'factor_kb': int(self.iparm[16]), 'factor_nnz': int(self.iparm[17]),
                'numeric': {'perturbed': int(self.iparm[13])},
                'mkl_threads': int(self.mkl.mkl_get_max_threads()), 'mtype': self.mtype,
                'iparm24': int(self.iparm[23])}

    def close(self):
        self._call(-1, np.zeros(1, self.dtype))


SOLVERS = {'rslab': Rslab, 'pardiso': Pardiso,
           'pardiso-2l': lambda path, settings, threads: Pardiso(path, settings, threads, two_level=True)}


def working_set_mb():
    """(current, peak) working set of this process in MB: the peak the OS keeps (Windows' peak
    working set, Linux' VmHWM, else getrusage's resident maximum, bytes on macOS)."""
    import psutil
    m = psutil.Process().memory_info()
    peak = getattr(m, 'peak_wset', None)
    if peak is None:
        try:
            with open('/proc/self/status') as fh:
                peak = next(int(line.split()[1]) * 1024 for line in fh if line.startswith('VmHWM:'))
        except OSError:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m.rss / 2**20, peak / 2**20


def measure(name, first, second, repeats, warmup, threads):
    a, b, meta = load(first)
    a2, b2, _ = load(second)
    path = meta['path']
    runs, report = [], {}
    # The solver's library loaded before the baseline: the peak is the solver's work alone,
    # not the size of its binary (two revisions' wheels differ in it).
    SOLVERS[name.split('@')[0]](path, {}, threads)
    base_mb = working_set_mb()[0]
    for round_ in range(warmup + repeats):
        # A variant takes the production settings of its family (`pardiso-2l`: `pardiso`), a
        # revision those of its solver (`rslab@main`: `rslab`).
        kind = name.split('@')[0]
        solver = SOLVERS[kind](path, meta.get('settings', {}).get(kind.split('-')[0], {}), threads)
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
        steps, res = 0, residual(a2, x2, b2)
        # Until the target, or until a step no longer lowers the residual (refinement that
        # diverges or stalls where the factor cannot get further); that step is not kept.
        while res > TARGET and steps < 50:
            x_next = x2 + solver.solve(b2 - a2 @ x2)
            res_next = residual(a2, x_next, b2)
            steps += 1
            if res_next >= res:
                break
            x2, res = x_next, res_next
        t6 = clock()
        run = {'analyze': t1 - t0, 'factor': t2 - t1, 'solve': t3 - t2, 'refactor': t4 - t3,
               'residual': residual(a, x, b), 'to_target': t6 - t3, 'refine_steps': steps,
               'refined_residual': res}
        report = solver.report()
        if hasattr(solver, 'close'):
            solver.close()
        if round_ >= warmup:
            runs.append(run)
    peak_mb = working_set_mb()[1]
    return {'path': path, 'family': meta.get('family'), 'n': a.shape[0], 'nnz': int(a.nnz),
            'dtype': str(a.dtype), 'nrhs': b.shape[1], 'threads': threads, 'runs': runs, 'report': report,
            'peak_mb': peak_mb - base_mb}


def build_revision(spec, build_dir):
    """(label, python, revision) of one RSLAB revision: `REF`, a git ref of this repository, or
    `LABEL=DIR`, a source tree (its `REVISION` file names the commit). The wheel is built as
    the published one is, `maturin build --release` in `python/` (its `.cargo/config.toml`
    pins the target CPU), into a virtual environment of its own: once per revision, every
    build through the one target directory `build_dir/target`."""
    import tarfile
    label, _, src = spec.partition('=')
    src = src or label
    if Path(src).is_dir():
        tree = Path(src).resolve()
        stamp = tree / 'REVISION'
        rev = stamp.read_text().split()[0] if stamp.exists() else tree.name
    else:
        repo = Path(__file__).resolve().parent.parent
        rev = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', f'{src}^{{commit}}'],
                             capture_output=True, text=True, check=True).stdout.strip()
        tree = build_dir / rev / 'src'
        if not (tree / 'python').is_dir():
            tree.mkdir(parents=True, exist_ok=True)
            archive = subprocess.Popen(['git', '-C', str(repo), 'archive', '--format=tar', rev],
                                       stdout=subprocess.PIPE)
            with tarfile.open(fileobj=archive.stdout, mode='r|') as tar:
                tar.extractall(tree)
            archive.wait()
    home = build_dir / rev
    bindir = home / 'venv' / ('Scripts' if os.name == 'nt' else 'bin')
    python = bindir / ('python.exe' if os.name == 'nt' else 'python')
    built = home / 'built'
    if not built.exists():
        print(f'== building rslab {label} ({rev[:12]}) from {tree}', flush=True)
        t0 = time.perf_counter()
        subprocess.run(['uv', 'venv', '-q', '--allow-existing', str(home / 'venv')], check=True)
        subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(python),
                        'numpy', 'scipy', 'psutil', 'maturin'], check=True)
        env = {**os.environ, 'CARGO_TARGET_DIR': str((build_dir / 'target').resolve())}
        subprocess.run([str(bindir / 'maturin'), 'build', '--release', '--interpreter', str(python),
                        '--out', str(home / 'dist')], cwd=tree / 'python', env=env, check=True)
        wheel = max((home / 'dist').glob('rslab-*.whl'), key=lambda w: w.stat().st_mtime)
        subprocess.run(['uv', 'pip', 'install', '-q', '--reinstall', '--python', str(python), str(wheel)],
                       check=True)
        version = subprocess.run([str(python), '-c', 'import rslab; print(rslab.__version__)'],
                                 capture_output=True, text=True, check=True).stdout.strip()
        built.write_text(f'{rev} {version}\n')
        print(f'== built rslab {version} ({label}, {rev[:12]}) in {time.perf_counter() - t0:.0f} s', flush=True)
    return label, python, rev


TIMES = ('analyze', 'factor', 'refactor', 'solve')
METRICS = TIMES + ('peak_mb',)


def samples(rows, metric):
    """The samples of one side: every timed repeat of every round, or one peak per round."""
    if metric == 'peak_mb':
        return [r['peak_mb'] for r in rows]
    return [run[metric] for r in rows for run in r['runs']]


def spread(xs):
    """`(max - min) / median` of the samples, None below two samples or a median of zero."""
    if len(xs) < 2:
        return None
    m = statistics.median(xs)
    return (max(xs) - min(xs)) / m if m > 0 else None


def versus(base, other):
    """(ratio of the medians, noise), the noise None where a side has no spread to measure."""
    mb, mo = statistics.median(base), statistics.median(other)
    ratio = mo / mb if mb > 0 and mo > 0 else None
    sb, so = spread(base), spread(other)
    return ratio, (None if sb is None or so is None else max(sb, so))


def beyond(ratio, noise):
    """+1 above the noise, -1 below, 0 within, None without a measure."""
    if ratio is None or noise is None:
        return None
    return 1 if ratio > 1 + noise else -1 if ratio < 1 / (1 + noise) else 0


def exact(rows, key):
    """The values a side reported for a deterministic count (factor entries, pivots)."""
    out = set()
    for r in rows:
        rep = r.get('report', {})
        v = rep.get('factor_nnz') if key == 'factor_nnz' else (rep.get('numeric') or {}).get(key)
        if v is not None:
            out.add(v)
    return out


def fmt_set(vals):
    return '-' if not vals else '/'.join(str(v) for v in sorted(vals))


def compare(rows, base):
    """Per system the medians of every side and the ratios against `base`, then a verdict per
    family and side (module docstring: noise criterion)."""
    ok = [r for r in rows if 'error' not in r]
    sides = list(dict.fromkeys(r['solver'] for r in ok))
    if base not in sides or len(sides) < 2:
        return
    systems = list(dict.fromkeys(r['system'] for r in ok))
    by = {}
    for r in ok:
        by.setdefault((r['system'], r['solver']), []).append(r)
    family = {r['system']: r.get('family') or r.get('group') for r in ok}
    print(f'\n== Comparison against {base}: medians; ratio other/{base}, noise in brackets, '
          f'* beyond noise', flush=True)
    head = (f'{"system / side":<34}' + ''.join(f'{m:>17}' for m in ('analyze', 'factor', 'refactor', 'solve'))
            + f'{"peak MB":>17}{"factor nnz":>14}{"perturbed":>11}{"delayed":>9}{"res":>9}{"refined":>9}')
    print(head)
    verdict = {}
    for sysname in systems:
        if (sysname, base) not in by:
            continue
        print(f'{sysname} [{family[sysname]}]')
        b_rows = by[(sysname, base)]
        for side in sides:
            rows_s = by.get((sysname, side))
            if not rows_s:
                continue
            res = [run['residual'] for r in rows_s for run in r['runs']]
            ref = [run['refined_residual'] for r in rows_s for run in r['runs']]
            line = f'  {side:<32}' + ''.join(f'{statistics.median(samples(rows_s, m)):17.4g}' for m in METRICS)
            line += (f'{fmt_set(exact(rows_s, "factor_nnz")):>14}{fmt_set(exact(rows_s, "perturbed")):>11}'
                     f'{fmt_set(exact(rows_s, "delayed")):>9}{max(res):9.1e}{max(ref):9.1e}')
            print(line)
            if side == base:
                continue
            cells = []
            for m in METRICS:
                ratio, noise = versus(samples(b_rows, m), samples(rows_s, m))
                mark = beyond(ratio, noise)
                verdict.setdefault((family[sysname], side), {}).setdefault(m, []).append((ratio, noise, sysname))
                cell = '-' if ratio is None else f'{ratio:.3f}{"*" if mark else " "}'
                cell += '' if noise is None else f' [{100 * noise:.0f}%]'
                cells.append(f'{cell:>17}')
            diffs = [k for k in ('factor_nnz', 'perturbed', 'delayed') if exact(b_rows, k) != exact(rows_s, k)]
            print(f'  {"ratio " + side:<32}' + ''.join(cells) + ('   differs: ' + ', '.join(diffs) if diffs else ''))
    print('\n== Verdict per family: geometric mean of the ratios against the median noise')
    for (fam, side), per in verdict.items():
        parts, worse, better = [], [], []
        for m in METRICS:
            vals = [(r, n, s) for r, n, s in per.get(m, []) if r is not None]
            if not vals:
                continue
            gm = float(np.exp(np.mean([np.log(r) for r, _, _ in vals])))
            noises = [n for _, n, _ in vals if n is not None]
            noise = statistics.median(noises) if len(noises) == len(vals) else None
            mark = beyond(gm, noise)
            parts.append(f'{m} {gm:.3f}' + ('' if noise is None else f' [{100 * noise:.0f}%]'))
            singles = [s for r, n, s in vals if beyond(r, n)]
            if mark == 1:
                worse.append(m)
            elif mark == -1:
                better.append(m)
            if singles and not mark:
                parts[-1] += f' ({len(singles)} single beyond)'
        if not any(n is not None for v in per.values() for _, n, _ in v):
            text = 'one sample a side, no verdict'
        else:
            text = ('regression beyond noise: ' + ', '.join(worse) if worse else 'no regression beyond noise')
            text += ('; improvement: ' + ', '.join(better)) if better else ''
        print(f'{fam:<10}{side} / {base}, {len(set(s for v in per.values() for *_, s in v))} systems: '
              f'{"; ".join(parts)}  ->  {text}', flush=True)


def selected(files, families):
    """The corpus files of the given families (the `family` of their meta), all without."""
    if not families:
        return files
    keep = []
    for f in files:
        with np.load(f) as z:
            if json.loads(str(z['meta'])).get('family') in families:
                keep.append(f)
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', type=Path, nargs='*')
    p.add_argument('--solvers', nargs='*', choices=list(SOLVERS),
                   help='default: all; with --rslab the reference solvers beside it, default pardiso')
    p.add_argument('--rslab', nargs='+', metavar='REV',
                   help='RSLAB revisions to compare, the first the base: git refs or LABEL=DIR')
    p.add_argument('--build-dir', type=Path, default=OUT.parent / 'revisions',
                   help='wheels, environments and the target directory of the revisions')
    p.add_argument('--family', nargs='+', help='only the corpus files of these families')
    p.add_argument('--rounds', type=int, default=1, help='worker processes per solver and system, interleaved')
    p.add_argument('--threads', type=int, default=0, help='worker threads, 0 for the physical cores')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--compare', type=Path, metavar='JSONL',
                   help='no runs: the comparison of the last run in this results file')
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

    if args.compare:
        rows = [json.loads(line) for line in args.compare.open() if line.strip()]
        last = rows[-1].get('run')
        rows = [r for r in rows if r.get('run') == last]
        sides = list(dict.fromkeys(r['solver'] for r in rows))
        compare(rows, next((s for s in sides if s.startswith('rslab')), sides[0]))
        return

    import psutil
    threads = args.threads or psutil.cpu_count(logical=False)
    pythons = {}
    if args.rslab:
        labels = []
        for spec in args.rslab:
            label, python, rev = build_revision(spec, args.build_dir)
            # The same revision twice (A against A, the noise) gets a second label.
            while label in labels:
                label += "'"
            labels.append(label)
            pythons[f'rslab@{label}'] = python
        solvers = list(pythons) + (args.solvers if args.solvers is not None else ['pardiso'])
    else:
        solvers = args.solvers if args.solvers is not None else list(SOLVERS)
    files = sorted(f for i in args.inputs for f in (i.rglob('*.npz') if i.is_dir() else [i]))
    # Sweep partners are refactorization values, not systems of their own.
    files = [f for f in files if not any(partner(g) == f and g != f for g in files)]
    files = selected(files, args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y-%m-%dT%H:%M:%S')
    print(f'== {len(files)} systems, {", ".join(solvers)}, {threads} threads, {args.rounds} rounds of '
          f'{args.warmup} + {args.repeats} repeats, run {run_id}', flush=True)
    rows = []
    for f in files:
        for round_ in range(args.rounds):
            # With revisions to compare, the reference solvers run once: they are no side of it.
            for name in [s for s in solvers if round_ == 0 or s in pythons or not pythons]:
                cmd = [str(pythons.get(name, sys.executable)), __file__, '--worker', name, str(f),
                       str(partner(f)), '--threads', str(threads), '--repeats', str(args.repeats),
                       '--warmup', str(args.warmup)]
                done = subprocess.run(cmd, capture_output=True, text=True)
                lines = [l for l in done.stdout.splitlines() if l.startswith('RESULT ')]
                result = json.loads(lines[-1][7:]) if lines else {'error': (done.stderr or done.stdout).strip()[-300:]}
                row = {'system': f.stem, 'group': f.parent.name, 'solver': name, 'round': round_, 'run': run_id,
                       'time': time.strftime('%Y-%m-%dT%H:%M:%S'), **result}
                rows.append(row)
                with args.out.open('a') as out:
                    out.write(json.dumps(row) + '\n')
                if 'error' in row:
                    print(f'{f.stem:<34}{name:<18}failed: {row["error"][:90]}', flush=True)
                    continue
                med = {k: statistics.median(r[k] for r in row['runs']) for k in TIMES}
                num = row['report'].get('numeric') or {}
                print(f'{f.stem:<34}{name:<18}r{round_} analyze {med["analyze"]:7.3f}  factor {med["factor"]:7.3f}  '
                      f'refactor {med["refactor"]:7.3f}  solve {med["solve"]:6.3f}  '
                      f'res {row["runs"][-1]["residual"]:.1e}  refined {row["runs"][-1]["refined_residual"]:.1e}  '
                      f'peak {row["peak_mb"]:7.0f} MB  fnnz {row["report"].get("factor_nnz")}  '
                      f'perturbed {num.get("perturbed")}  delayed {num.get("delayed")}', flush=True)
    compare(rows, solvers[0])


if __name__ == '__main__':
    main()
