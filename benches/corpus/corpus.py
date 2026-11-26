"""RSLAB's standing benchmark corpus: real systems of the codes RSLAB serves, rebuilt from
pinned commits of their sources.

    python benches/corpus/corpus.py build SOURCE --src DIR --out DIR [--work DIR] [--only NAME ...]
    python benches/corpus/corpus.py manifest DIR
    python benches/corpus/corpus.py pins SOURCE

`build` runs inside a pool job (`benches/corpus/pool.sh corpus SOURCE`, which ships the
sources at their pinned commits): it builds the source, runs it with its own dump switch,
and writes every system as `DIR/<family>/<name>.npz` in the format `benches/pardiso_corpus.py`
reads (CSC with both triangles, right-hand sides `b`, a JSON `meta`), then the manifest
`DIR/manifest.json`. A system with refactorization values (the next frequency of a sweep,
the next Newton iterate) has them as `<name>_f01.npz` next to its `<name>_f00.npz`.

Sources (SOURCES, the pins), each feeding families (mom, fem, circuit):

  rapidfem   the complex-symmetric indefinite systems of frequency sweeps
             (`RAPIDFEM_DUMP_DIR`), the shipped examples as they are
  suitesparse  the SuiteSparse Matrix Collection: the KLU paper's circuits (the list of
             benches/klu_realworld.rs) and FEM matrices of the classes the FEM codes do
             not factor today (real SPD, real symmetric indefinite, real and complex
             unsymmetric)
  legacy     systems from the pool cache bench-mtx whose generator is not in any
             repository (named in the manifest as such)

Sources from repositories that are not public come from `sources.local.toml` next to this
script (not versioned; `sources.example.toml` shows the form): per source its repository,
commit, paths, the families it feeds and a local module whose `build` writes its systems.

`meta` of every file: family, path (the solver path the source uses: lu, ldlt, klu),
source, repo, commit, command, symmetry, definiteness where known, the production
`settings` per solver and the source's own notes.
"""
import argparse
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
LOCAL = HERE / 'sources.local.toml'
FAMILIES = {'mom', 'circuit', 'fem'}

PLACEMENT = {'mom': ('euler', '13G'), 'circuit': ('kutta', '8G'), 'fem': ('runge', '8G')}
SUITESPARSE_MIRRORS = ['https://sparse.tamu.edu/MM', 'https://www.cise.ufl.edu/research/sparse/MM']

FEM_EXAMPLES = ['wr90', 'wr90_pml', 'pyramidal_horn', 'patch_antenna', 'microstrip_line']
FEM_DUMPS = 2

SUITESPARSE_FEM = [
    ('Boeing/bcsstk36', 'spd'), ('Boeing/pwtk', 'spd'),
    ('GHS_indef/helm3d01', 'indefinite'), ('GHS_indef/dawson5', 'indefinite'),
    ('Simon/raefsky3', None), ('Norris/torso3', None), ('Lee/fem_filter', None),
]

LEGACY = [
    ('mom_sym_46428', 'sym_46428_1_n32460_nd15284_np17176_aux0.mtx', 'ldlt',
     'complex-symmetric near-pattern form of a 32k-unknown A-EFIE system (n = nd + np, '
     'no auxiliary block); the MoM code factors no complex-symmetric sparse matrix in production'),
]


def log(msg):
    print(f'[corpus {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd, **kw):
    """A command with its output in the job's log as it comes."""
    log('$ ' + ' '.join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)




def symmetry(a):
    """'symmetric' when A equals its transpose (no conjugate) exactly, else 'unsymmetric'."""
    d = (a - a.T).tocsc()
    d.eliminate_zeros()
    return 'symmetric' if d.nnz == 0 else 'unsymmetric'


def save(out, family, name, a, b, meta):
    """`out/<family>/<name>.npz`, the format of benches/pardiso_corpus.py, written atomically."""
    if family not in FAMILIES:
        return
    a = sp.csc_matrix(a)
    a.sum_duplicates()
    a.sort_indices()
    n = a.shape[0]
    b = np.asarray(b, dtype=a.dtype).reshape(n, -1)
    meta = {'family': family, 'n': n, 'nnz': int(a.nnz), 'dtype': str(a.dtype), 'nrhs': b.shape[1],
            'symmetry': symmetry(a), **meta}
    path = out / family / f'{name}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{name}.tmp.npz')
    np.savez(tmp, indptr=a.indptr, indices=a.indices, data=a.data, b=b, meta=json.dumps(meta))
    os.replace(tmp, path)
    log(f'{family}/{name}: n {n}, nnz {a.nnz}, {a.dtype}, {meta["symmetry"]}, path {meta["path"]}, '
        f'{path.stat().st_size / 2**20:.0f} MB')


def from_lower(lower):
    """The full matrix of its lower triangle, pattern for pattern: entries that are zero at
    one frequency stay, so the systems of a sweep share their pattern as in the source."""
    lower = sp.coo_matrix(lower)
    off = lower.row != lower.col
    rows = np.concatenate([lower.row, lower.col[off]])
    cols = np.concatenate([lower.col, lower.row[off]])
    return sp.csc_matrix((np.concatenate([lower.data, lower.data[off]]), (rows, cols)), shape=lower.shape)


def read_mtx(path):
    """A Matrix Market file as CSC; a symmetric file with both triangles (scipy expands it)."""
    return sp.csc_matrix(scipy.io.mmread(str(path)))


def manifest(out):
    """`out/manifest.json`: every system's meta, its files and their size, from the files."""
    systems = []
    for f in sorted(out.glob('*/*.npz')):
        if f.name.startswith('.'):
            continue
        with np.load(f) as z:
            meta = json.loads(str(z['meta']))
        systems.append({'name': f.stem, 'file': str(f.relative_to(out)), 'bytes': f.stat().st_size, **meta})
    doc = {'generator': 'benches/corpus/corpus.py (rslab)', 'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'sources': {name: {k: s[k] for k in ('repo', 'commit', 'paths')} for name, s in SOURCES.items()
                       if 'repo' in s},
           'systems': systems}
    tmp = out / '.manifest.json.tmp'
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out / 'manifest.json')
    log(f'manifest: {len(systems)} files, {sum(s["bytes"] for s in systems) / 2**30:.1f} GB')




def venv(work, name, *packages):
    """A virtual environment with `packages`, its python."""
    home = work / f'venv-{name}'
    python = home / 'bin' / 'python'
    if not python.exists():
        run(['uv', 'venv', '-q', home])
    if packages:
        run(['uv', 'pip', 'install', '-q', '--python', python, *packages])
    return python


def wheel(python, project, work, name, *extra):
    """Builds and installs the maturin project `project` (its pyproject's directory)."""
    dist = work / f'dist-{name}'
    env = {**os.environ, 'CARGO_TARGET_DIR': str(work / f'target-{name}')}
    run([python.parent / 'maturin', 'build', '--release', '--interpreter', python, '--out', dist, *extra],
        cwd=project, env=env)
    run(['uv', 'pip', 'install', '-q', '--reinstall', '--python', python,
         max(dist.glob('*.whl'), key=lambda w: w.stat().st_mtime)])


def pinned(source, extra=''):
    s = SOURCES[source]
    return {'source': source, 'repo': s['repo'], 'commit': s['commit'], **({'note': extra} if extra else {})}


def build_rapidfem(src, out, work, only, reuse):
    tree = src / 'rapidfem'
    python = venv(work, 'rapidfem', 'numpy', 'scipy', 'maturin')
    wheel(python, tree / 'python', work, 'rapidfem')
    examples = tree / 'python' / 'python_src' / 'rapidfem' / 'examples'
    raw = work / 'raw-rapidfem'
    for ex in FEM_EXAMPLES:
        name = f'fem_{ex}'
        if only and name not in only:
            continue
        env = {**os.environ, 'RAPIDFEM_DUMP_DIR': str(raw), 'RAPIDFEM_DUMP_TAG': name,
               'RAPIDFEM_DUMP_LIMIT': str(FEM_DUMPS)}
        t0 = time.perf_counter()
        try:
            run([python, '-u', examples / f'fd_{ex}.py'], cwd=examples, env=env)
        except subprocess.CalledProcessError as exc:
            log(f'{name}: the example failed ({exc.returncode}), skipped')
            continue
        log(f'{name}: dumped in {time.perf_counter() - t0:.0f} s')
        for k in range(FEM_DUMPS):
            f = raw / f'{name}_f{k:02d}.sparta'
            if not f.exists():
                continue
            a, b, k0 = read_sparta(f)
            save(out, 'fem', f'{name}_f{k:02d}', a, b, {
                'path': 'ldlt', 'definiteness': 'indefinite', 'settings': {}, 'k0': k0,
                'command': f'RAPIDFEM_DUMP_DIR=DIR RAPIDFEM_DUMP_TAG={name} RAPIDFEM_DUMP_LIMIT={FEM_DUMPS} '
                           f'python python/python_src/rapidfem/examples/fd_{ex}.py',
                **pinned('rapidfem', 'driven sweep, curl-curl Nedelec with ports/PML/ABC as the example '
                                     'sets them; rapidfem factors it with LDLT, default settings')})
            f.unlink()


def read_sparta(path):
    """(A, b, k0) of rapidfem's SPARTA02 container: the lower triangle in CSC, u64 indices,
    complex values, then the right-hand sides (crates/rapidfem-fd/src/dump.rs)."""
    with open(path, 'rb') as fh:
        assert fh.read(8) == b'SPARTA02', path
        n, nnz, nrhs, _ = np.frombuffer(fh.read(32), '<u8')
        k0 = float(np.frombuffer(fh.read(8), '<f8')[0])
        indptr = np.frombuffer(fh.read(8 * (n + 1)), '<u8').astype(np.int64)
        indices = np.frombuffer(fh.read(8 * nnz), '<u8').astype(np.int64)
        data = np.frombuffer(fh.read(16 * nnz), '<c16')
        b = np.frombuffer(fh.read(16 * n * nrhs), '<c16').reshape(nrhs, n).T
    lower = sp.csc_matrix((data, indices, indptr), shape=(int(n), int(n)))
    return from_lower(lower), b, k0


def fetch_suitesparse(group, name, dest):
    """`<name>.mtx` (and `<name>_b.mtx` where the collection has one) into dest, from the first
    mirror that serves it."""
    for base in SUITESPARSE_MIRRORS:
        url = f'{base}/{group}/{name}.tar.gz'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'rslab-corpus'})
            data = urllib.request.urlopen(req, timeout=600).read()
        except Exception as exc:  # noqa: BLE001 - the next mirror
            log(f'{group}/{name}: {url}: {exc}')
            continue
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            for member in tar.getmembers():
                stem = Path(member.name).name
                if stem in (f'{name}.mtx', f'{name}_b.mtx'):
                    (dest / stem).write_bytes(tar.extractfile(member).read())
        log(f'{group}/{name}: {len(data) / 2**20:.1f} MB from {base}')
        return url
    return None


def klu_paper_set():
    """The SuiteSparse circuits of benches/klu_realworld.rs (its CORPUS), the one list."""
    text = (HERE.parent / 'klu_realworld.rs').read_text()
    block = text[text.index('const CORPUS: '):]
    block = block[:block.index('];')]
    return re.findall(r'\("(\w+)", "(\w+)"\)', block)


def build_suitesparse(src, out, work, only, reuse):
    raw = work / 'raw-suitesparse'
    raw.mkdir(parents=True, exist_ok=True)
    todo = [(g, n, 'circuit', None) for g, n in klu_paper_set()] + \
           [(g, n, 'fem', kind) for (gn, kind) in SUITESPARSE_FEM for g, n in [gn.split('/')]]
    for group, name, family, kind in todo:
        sysname = f'ss_{name}'
        if family not in FAMILIES or (only and sysname not in only):
            continue
        url = fetch_suitesparse(group, name, raw)
        if not url:
            log(f'{sysname}: no mirror serves {group}/{name}, skipped')
            continue
        a = read_mtx(raw / f'{name}.mtx')
        fb = raw / f'{name}_b.mtx'
        b = np.asarray(scipy.io.mmread(str(fb))).reshape(a.shape[0], -1) if fb.exists() else np.ones(a.shape[0])
        sym = symmetry(a)
        path = 'klu' if family == 'circuit' else ('ldlt' if sym == 'symmetric' else 'lu')
        save(out, family, sysname, a, b, {
            'path': path, 'settings': {}, 'definiteness': kind, 'source': 'suitesparse',
            'repo': f'SuiteSparse Matrix Collection {group}/{name}', 'commit': None, 'command': f'GET {url}',
            'note': 'right-hand side from the collection' if fb.exists() else 'right-hand side all ones'})
        for f in raw.glob(f'{name}*.mtx'):
            f.unlink()


def build_legacy(src, out, work, only, reuse):
    for name, file, path, note in LEGACY:
        if only and name not in only:
            continue
        f = reuse / file
        if not f.exists():
            log(f'{name}: {f} missing, skipped')
            continue
        a = read_mtx(f)
        save(out, 'mom', name, a, np.ones(a.shape[0]), {
            'path': path, 'settings': {}, 'definiteness': 'indefinite', 'source': 'legacy',
            'repo': None, 'commit': None, 'command': f'pool cache bench-mtx: {file}',
            'note': note + '; right-hand side all ones'})


SOURCES = {
    'rapidfem': {'repo': 'milanofthe/rapidfem', 'commit': 'e917b8018022a1e8994d48accead0262d05aa8cd',
                 'paths': ['.', ':(exclude)python/python_src/rapidfem/ui/frontend-src/static/demo'],
                 'feeds': ['fem'], 'build': build_rapidfem},
    'circuits': {'repo': 'milanofthe/circuit-reference-suite', 'commit': '6a97c8c6d3d24f5dbd3709a0ca479be3054c5449',
                 'paths': ['cases/vacask_c6288_psp103']},
    'suitesparse': {'feeds': ['circuit', 'fem'], 'build': build_suitesparse},
    'legacy': {'feeds': ['mom'], 'build': build_legacy},
}
if LOCAL.exists():
    for _name, _source in tomllib.loads(LOCAL.read_text())['sources'].items():
        assert _name not in SOURCES, f'{LOCAL}: {_name} is a versioned source'
        SOURCES[_name] = _source


def builder(source):
    """The `build` of a source: its own, or that of the local module it names (which imports
    this script as `corpus`)."""
    s = SOURCES[source]
    if 'build' in s:
        return s['build']
    sys.modules.setdefault('corpus', sys.modules[__name__])
    spec = importlib.util.spec_from_file_location(f'corpus_{source}', LOCAL.parent / s['module'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='build one source into the corpus (in a pool job)')
    built = [name for name, s in SOURCES.items() if 'feeds' in s]
    b.add_argument('source', choices=built)
    b.add_argument('--src', type=Path, default=Path('src'), help='the shipped sources, one directory each')
    b.add_argument('--out', type=Path, required=True, help='the corpus directory')
    b.add_argument('--work', type=Path, default=Path(os.environ.get('FALLLOW_SCRATCH', '/scratch')) / 'corpus')
    b.add_argument('--only', nargs='+', help='only these systems (their names without _fNN)')
    b.add_argument('--reuse', type=Path, help='earlier output the source reads (legacy: bench-mtx)')
    b.add_argument('--family', nargs='+', help='only the systems of these families')
    m = sub.add_parser('manifest', help='rewrite the manifest from the corpus files')
    m.add_argument('out', type=Path)
    pins = sub.add_parser('pins', help='the sources a build ships, one line each: name repo commit pathspecs')
    pins.add_argument('source', choices=built)
    place = sub.add_parser('placement', help='family machine memory, one line each: of SOURCE, else all')
    place.add_argument('source', nargs='?', choices=built)
    args = p.parse_args()
    if args.cmd == 'pins':
        s = SOURCES[args.source]
        for name in ([args.source] if 'repo' in s else []) + s.get('ships', []):
            print(name, SOURCES[name]['repo'], SOURCES[name]['commit'], *SOURCES[name]['paths'])
        return
    if args.cmd == 'placement':
        for family in (SOURCES[args.source]['feeds'] if args.source else PLACEMENT):
            print(family, *PLACEMENT[family])
        return
    if args.cmd == 'manifest':
        manifest(args.out)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    FAMILIES.intersection_update(args.family or FAMILIES)
    only = set(args.only or [])
    t0 = time.perf_counter()
    builder(args.source)(args.src.resolve(), args.out.resolve(), args.work.resolve(), only, args.reuse)
    log(f'{args.source}: done in {time.perf_counter() - t0:.0f} s')
    manifest(args.out.resolve())


if __name__ == '__main__':
    sys.exit(main())
