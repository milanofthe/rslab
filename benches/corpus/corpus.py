"""RSLAB's standing benchmark corpus: real systems of the codes RSLAB serves, rebuilt from
pinned commits of their sources.

    python benches/corpus/corpus.py build SOURCE --src DIR --out DIR [--work DIR] [--only NAME ...]
    python benches/corpus/corpus.py manifest DIR
    python benches/corpus/corpus.py pins SOURCE

`build` runs inside a pool job (`benches/corpus/pool.sh corpus SOURCE`, which ships the
sources at the commits below): it builds the source, runs it with its own dump switch, and
writes every system as `DIR/<family>/<name>.npz` in the format `benches/pardiso_corpus.py`
reads (CSC with both triangles, right-hand sides `b`, a JSON `meta`), then the manifest
`DIR/manifest.json`. A system with refactorization values (the next frequency of a sweep,
the next Newton iterate) has them as `<name>_f01.npz` next to its `<name>_f00.npz`.

Sources (SOURCES, the pins):

  rapidmom   MoM near-field matrices the preconditioner LU factors, unsymmetric complex, at
             a low and a high frequency (`analysis.condlab.dump`, branch exp/cond-lu); the
             two project cases come from `inputs/mom` in the corpus directory
  rapidfem   the complex-symmetric indefinite systems of frequency sweeps
             (`RAPIDFEM_DUMP_DIR`), the shipped examples as they are
  sane       DC Newton and transient Jacobians of SANE (`SANE_DUMP_MATRIX`) on ISCAS-85,
             VACASK c6288, the IBM power grids and SANE's own generated circuits; the c6288
             decks come from circuit-reference-suite
  suitesparse  the SuiteSparse Matrix Collection: the KLU paper's circuits (the list of
             benches/klu_realworld.rs) and FEM matrices of the classes the FEM codes do
             not factor today (real SPD, real symmetric indefinite, real and complex
             unsymmetric)
  legacy     systems from the pool cache bench-mtx whose generator is not in any
             repository (named in the manifest as such)

`meta` of every file: family, path (the solver path the source uses: lu, ldlt, klu),
source, repo, commit, command, symmetry, definiteness where known, the production
`settings` per solver and the source's own notes.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
# The families a build writes (`build --family`), all by default.
FAMILIES = {'mom', 'circuit', 'fem'}

# The pinned sources: repository, commit and the paths of it a job needs (git pathspecs).
SOURCES = {
    'rapidmom': {'repo': 'milanofthe/rapidmom-dev', 'commit': '2c1544466358086a06fee6c0657abe85647e1aed',
                 'paths': ['.', ':(exclude)docs', ':(exclude,glob)analysis/out/**/*.pdf',
                           ':(exclude,glob)analysis/out/**/*.png']},
    'rapidfem': {'repo': 'milanofthe/rapidfem', 'commit': 'e917b8018022a1e8994d48accead0262d05aa8cd',
                 'paths': ['.', ':(exclude)python/python_src/rapidfem/ui/frontend-src/static/demo']},
    'sane': {'repo': 'milanofthe/sane-dev', 'commit': '8d686a28846538e33f6e419cf77dc70f77d65e83',
             'paths': ['.']},
    # the VACASK c6288 decks SANE's ISCAS and c6288 circuits read (SANE_VACASK_UPSTREAM)
    'circuits': {'repo': 'milanofthe/circuit-reference-suite', 'commit': '6a97c8c6d3d24f5dbd3709a0ca479be3054c5449',
                 'paths': ['cases/vacask_c6288_psp103']},
}
# The sources a build ships, and the families it writes, per corpus source.
NEEDS = {'rapidmom': ['rapidmom'], 'rapidfem': ['rapidfem'], 'sane': ['sane', 'circuits'],
         'suitesparse': [], 'legacy': []}
FEEDS = {'rapidmom': ['mom'], 'legacy': ['mom'], 'rapidfem': ['fem'], 'sane': ['circuit'],
         'suitesparse': ['circuit', 'fem']}
# Where a family lives (the pool cache rslab-corpus of that machine) and the memory its A/B
# job asks for: the peak falllow measured for main against main with headroom (mom 10.1 GB,
# circuit 1.1 GB; fem not yet measured, a strato machine's room).
PLACEMENT = {'mom': ('milan-box-01', '13G'), 'circuit': ('strato-2', '2G'), 'fem': ('strato-1', '18G')}
# SuiteSparse's file server first, the old University of Florida mirror after it.
SUITESPARSE_MIRRORS = ['https://sparse.tamu.edu/MM', 'https://www.cise.ufl.edu/research/sparse/MM']

# rapidmom's near LU in production (crates/rapidmom/src/problem/precond/mod.rs, settings()):
# static pivots lifted to the floor PC_PIVOT_FLOOR, iterated Ruiz scaling, MC64 matching,
# METIS. PARDISO beside it perturbs at the same order, iparm(10) = 6 (zero-based index 9).
MOM_SETTINGS = {'rslab': {'preconditioner': 1e-6, 'scaling': 'inf_norm', 'ordering': 'metis', 'matching': True},
                'pardiso': {'iparm': {'9': 6}}}
# rapidmom's GF table store of the IHP SG13G2 stack the project cases share.
MOM_STORE = 'cache/condlab_sg13g2.gf'
# (name, case, frequency in Hz, dump file stem of analysis.condlab.dump, store)
MOM = [
    ('mom_spiral_1GHz', 'kit:spiral_D130_N3_w10_s3', 1e9, 'spiral_D130_N3_w10_s3_shared_faces_1GHz', None),
    ('mom_momstack_1GHz', 'kit:momstack_N9_L3_w2', 1e9, 'momstack_N9_L3_w2_shared_faces_1GHz', None),
    ('mom_tia130a_1GHz', 'inputs/mom/tia130a/tia130a.toml', 1e9, 'tia130a_1GHz', MOM_STORE),
    ('mom_tia130a_170GHz', 'inputs/mom/tia130a/tia130a.toml', 1.7e11, 'tia130a_170GHz', MOM_STORE),
    ('mom_pa180a_1GHz', 'inputs/mom/pa180a/pa180a.toml', 1e9, 'pa180a_1GHz', MOM_STORE),
    ('mom_pa180a_180GHz', 'inputs/mom/pa180a/pa180a.toml', 1.8e11, 'pa180a_180GHz', MOM_STORE),
]

# rapidfem's driven-sweep examples (python/python_src/rapidfem/examples/fd_<name>.py), each
# up to its second frequency: the first system and its refactorization values. A spread of
# sizes and terminations: waveguide ports (28k), PML (51k), horn with PML and ABC (140k),
# lumped port with PML (186k), numerical wave ports (262k). The other examples repeat these
# classes at sizes up to 5.6M DOF, beyond what the corpus machines factor in minutes.
FEM_EXAMPLES = ['wr90', 'wr90_pml', 'pyramidal_horn', 'patch_antenna', 'microstrip_line']
# Systems a sweep dumps per example: the first frequency and the next one.
FEM_DUMPS = 2

# SANE circuits (benchmarks/scaling.py cases) and analyses. A DC run dumps its Newton
# Jacobians from the first on, a transient (uic) its consistent start and then the stage
# matrices of the Rosenbrock steps; SANE_DUMP_LIMIT bounds the files of one run.
SANE = [('c1908', 'dc'), ('c1908', 'tran'), ('c6288', 'tran'), ('c7552', 'dc'), ('c7552', 'tran'),
        ('ibmpg1', 'dc'), ('grid300', 'tran'), ('ibmpg2', 'dc')]
SANE_DUMP_LIMIT = {'dc': 8, 'tran': 40}

# SuiteSparse FEM matrices for the classes rapidfem does not factor: (group/name, kind).
SUITESPARSE_FEM = [
    ('Boeing/bcsstk36', 'spd'), ('Boeing/pwtk', 'spd'),
    ('GHS_indef/helm3d01', 'indefinite'), ('GHS_indef/dawson5', 'indefinite'),
    ('Simon/raefsky3', None), ('Norris/torso3', None), ('Lee/fem_filter', None),
]

# Systems of the pool cache bench-mtx (dumped by hand from rapidmom in October 2026, the
# writer in no repository): (name, file, path, note).
LEGACY = [
    ('mom_sym_46428', 'sym_46428_1_n32460_nd15284_np17176_aux0.mtx', 'ldlt',
     'complex-symmetric near-pattern form of a 32k-unknown A-EFIE system (n = nd + np, '
     'no auxiliary block); rapidmom factors no complex-symmetric sparse matrix in production'),
]


def log(msg):
    print(f'[corpus {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd, **kw):
    """A command with its output in the job's log as it comes."""
    log('$ ' + ' '.join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# --- the corpus format ----------------------------------------------------------------------


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
    """The full matrix of its lower triangle."""
    lower = sp.csc_matrix(lower)
    return (lower + lower.T - sp.diags(lower.diagonal())).tocsc()


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
           'sources': SOURCES, 'systems': systems}
    tmp = out / '.manifest.json.tmp'
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out / 'manifest.json')
    log(f'manifest: {len(systems)} files, {sum(s["bytes"] for s in systems) / 2**30:.1f} GB')


# --- building a source --------------------------------------------------------------------


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


def build_rapidmom(src, out, work, only, reuse):
    """The near matrices: from `reuse` (a directory of analysis.condlab.dump output of the
    same commit) where it holds them, else dumped here."""
    tree = src / 'rapidmom'
    raw = work / 'raw-rapidmom'
    raw.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MOM if not only or m[0] in only]
    missing = [m for m in todo if not (reuse and (reuse / f'{m[3]}.npz').exists())]
    if missing:
        python = venv(work, 'rapidmom', 'numpy', 'scipy', 'shapely', 'gdstk', 'pyyaml', 'mpmath', 'pillow',
                      'maturin')
        env = {**os.environ, 'CARGO_TARGET_DIR': str(work / 'target-rapidmom')}
        # the package as rapidmom's own pool runner installs it, without the notebook viewer
        run(['uv', 'pip', 'install', '--python', python, '-e', '.', '-C', 'viewer=skip'], cwd=tree, env=env)
        inputs = tree / 'inputs'
        if not inputs.exists():
            inputs.symlink_to(out / 'inputs')
        for name, case, freq, stem, store in missing:
            run([python, '-u', '-m', 'analysis.condlab.dump', case, f'{freq:g}', '--out', raw,
                 '--threads', os.environ.get('FALLLOW_CORES', '0'), *(['--store', store] if store else [])],
                cwd=tree)
    for name, case, freq, stem, store in todo:
        f = (reuse / f'{stem}.npz') if reuse and (reuse / f'{stem}.npz').exists() else raw / f'{stem}.npz'
        with np.load(f) as z:
            n = int(z['n'])
            a = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=(n, n))
            b, nd, npq = z['b'], int(z['nd']), int(z['np'])
        cmd = (f'python -m analysis.condlab.dump {case} {freq:g} --out DIR'
               + (f' --store {store}' if store else ''))
        save(out, 'mom', name, a, b.reshape(n, -1), {
            'path': 'lu', 'definiteness': 'indefinite', 'settings': MOM_SETTINGS, 'command': cmd,
            'freq_hz': freq, 'nd': nd, 'np': npq,
            **pinned('rapidmom', 'the near matrix the preconditioner factors, production scaling J A J '
                                 '(branch exp/cond-lu)')})


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


def build_sane(src, out, work, only, reuse):
    tree = src / 'sane'
    python = venv(work, 'sane', 'numpy', 'scipy', 'maturin')
    wheel(python, tree / 'crates' / 'py', work, 'sane')
    # SANE reads the VACASK c6288 decks from <upstream>/c6288/ngspice
    upstream = work / 'vacask'
    (upstream / 'c6288').mkdir(parents=True, exist_ok=True)
    link = upstream / 'c6288' / 'ngspice'
    if not link.exists():
        link.symlink_to(src / 'circuits' / 'cases' / 'vacask_c6288_psp103' / 'ngspice')
    for case, analysis in SANE:
        name = f'sane_{case}_{analysis}'
        if only and name not in only:
            continue
        raw = work / 'raw-sane' / name
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        env = {**os.environ, 'SANE_DUMP_MATRIX': str(raw), 'SANE_DUMP_LIMIT': str(SANE_DUMP_LIMIT[analysis]),
               'SANE_VACASK_UPSTREAM': str(upstream)}
        cmd = ['benchmarks/scaling.py', '--one', case, analysis, os.environ.get('FALLLOW_CORES', '1'), '1']
        try:
            run([python, '-u', *cmd], cwd=tree, env=env)
        except subprocess.CalledProcessError as exc:
            log(f'{name}: SANE failed ({exc.returncode}); keeping what it dumped')
        picked = pick_sane(raw, analysis)
        if not picked:
            log(f'{name}: nothing dumped, skipped')
            continue
        route = None
        for k, (f, fb) in enumerate(picked):
            a = read_mtx(f)
            b = np.asarray(scipy.io.mmread(str(fb))).reshape(a.shape[0], -1) if fb.exists() else None
            consistent = b is None or not np.any(b)
            if consistent:
                b = a @ np.ones(a.shape[0])  # no right-hand side to measure against: x = 1
            if route is None:
                import rslab
                route = rslab.analyze(a, path='auto').route
            path = 'klu' if route == 'klu' else ('ldlt' if symmetry(a) == 'symmetric' else 'lu')
            save(out, 'circuit', f'{name}_f{k:02d}', a, b, {
                'path': path, 'settings': {}, 'auto_route': route, 'dump_file': f.name,
                'rhs': 'A @ ones (every dumped one zero)' if consistent else 'dumped',
                'command': f'SANE_DUMP_MATRIX=DIR SANE_DUMP_LIMIT={SANE_DUMP_LIMIT[analysis]} '
                           f'SANE_VACASK_UPSTREAM=<circuits>/cases/vacask_c6288_psp103/.. python '
                           f'{" ".join(cmd[:4])} <threads> 1',
                **pinned('sane', f'{"DC Newton Jacobian" if analysis == "dc" else "transient stage matrix"}; '
                                 f'SANE factors through rslab AutoSolver, route {route}')})


def pick_sane(raw, analysis):
    """The (matrix, rhs) files of a SANE dump the corpus keeps. SANE writes a file per solve,
    so one matrix comes with several right-hand sides (the six stages of a Rosenbrock step
    share it; the first stage's is zero at rest): per distinct matrix the right-hand side of
    the largest norm. DC: the first two distinct Jacobians (Newton iterates 0 and 1);
    transient: the last two distinct ones among the dumped, consecutive stage matrices past
    the consistent start."""
    files = [f for f in raw.glob('sane_*_*.mtx') if not f.stem.endswith('_b')]
    files.sort(key=lambda f: int(f.stem.rsplit('_', 1)[1]))
    groups = {}
    for f in files:
        groups.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), []).append(f)

    def norm(f):
        fb = f.with_name(f.stem + '_b.mtx')
        return float(np.linalg.norm(np.asarray(scipy.io.mmread(str(fb))))) if fb.exists() else 0.0

    distinct = [max(g, key=norm) for g in groups.values()]
    keep = distinct[:2] if analysis == 'dc' else distinct[-2:]
    return [(f, f.with_name(f.stem + '_b.mtx')) for f in keep]


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
            'repo': 'milanofthe/rapidmom-dev', 'commit': None, 'command': f'pool cache bench-mtx: {file}',
            'note': note + '; right-hand side all ones'})


BUILDERS = {'rapidmom': build_rapidmom, 'rapidfem': build_rapidfem, 'sane': build_sane,
            'suitesparse': build_suitesparse, 'legacy': build_legacy}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='build one source into the corpus (in a pool job)')
    b.add_argument('source', choices=list(BUILDERS))
    b.add_argument('--src', type=Path, default=Path('src'), help='the shipped sources, one directory each')
    b.add_argument('--out', type=Path, required=True, help='the corpus directory')
    b.add_argument('--work', type=Path, default=Path(os.environ.get('FALLLOW_SCRATCH', '/scratch')) / 'corpus')
    b.add_argument('--only', nargs='+', help='only these systems (their names without _fNN)')
    b.add_argument('--reuse', type=Path, help='rapidmom: earlier dumps of the same commit; legacy: bench-mtx')
    b.add_argument('--family', nargs='+', help='only the systems of these families')
    m = sub.add_parser('manifest', help='rewrite the manifest from the corpus files')
    m.add_argument('out', type=Path)
    pins = sub.add_parser('pins', help='the sources a build ships, one line each: name repo commit pathspecs')
    pins.add_argument('source', choices=list(BUILDERS))
    place = sub.add_parser('placement', help='family machine memory, one line each: of SOURCE, else all')
    place.add_argument('source', nargs='?', choices=list(BUILDERS))
    args = p.parse_args()
    if args.cmd == 'pins':
        for name in NEEDS[args.source]:
            print(name, SOURCES[name]['repo'], SOURCES[name]['commit'], *SOURCES[name]['paths'])
        return
    if args.cmd == 'placement':
        for family in (FEEDS[args.source] if args.source else PLACEMENT):
            print(family, *PLACEMENT[family])
        return
    if args.cmd == 'manifest':
        manifest(args.out)
        return
    args.work.mkdir(parents=True, exist_ok=True)
    FAMILIES.intersection_update(args.family or FAMILIES)
    only = set(args.only or [])
    t0 = time.perf_counter()
    BUILDERS[args.source](args.src.resolve(), args.out.resolve(), args.work.resolve(), only, args.reuse)
    log(f'{args.source}: done in {time.perf_counter() - t0:.0f} s')
    manifest(args.out.resolve())


if __name__ == '__main__':
    sys.exit(main())
