"""SuperLU (scipy.sparse.linalg.splu) over the SuiteSparse corpus, with a
per-matrix wall-clock cap.

Adds a 5th solver to the corpus scaling study: loads the same cached `.mtx`
files the Rust `bench_suite` corpus uses (populate them by running the Rust
corpus bench first), runs an exact SuperLU factor + solve per matrix, and
appends JSONL records in the `bench_suite` schema (`solver="superlu"`, a `time`
and a `mem` record per matrix), so `fit_scaling.py` plots all solvers together.

SuperLU does not exploit symmetry, so on 3D FEM matrices its fill (and factor
time) can blow up to minutes. Each matrix is therefore factored in a **child
process with a timeout** (`RLA_SUPERLU_TIMEOUT`, default 60 s); a matrix that
exceeds it is killed and skipped (logged), so the run stays bounded.

Run:  python benches/superlu_corpus.py            # orchestrator
Env:  RLA_BENCH_OUT (default benches/bench_out/corpus.jsonl), RLA_MATGEN_CACHE,
      RLA_BENCH_CORPUS, RLA_SUPERLU_TIMEOUT (seconds).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import splu

DEFAULT = (
    "HB/bcsstk27,HB/bcsstk14,HB/bcsstk16,HB/bcsstk17,HB/bcsstk18,HB/bcsstk25,"
    "Cylshell/s3rmt3m3,Boeing/msc10848,Boeing/crystk03,Boeing/bcsstk39,"
    "GHS_psdef/wathen100,GHS_psdef/wathen120,GHS_psdef/oilpan,GHS_psdef/s3dkt3m2,"
    "Williams/pdb1HYS,Williams/cant,Nasa/nasasrb,Rothberg/cfd1,Schmid/thermal1,"
    "Um/2cubes_sphere,"
    "GHS_indef/stokes64,GHS_indef/bratu3d,GHS_indef/copter2,GHS_indef/dixmaanl,"
    "GHS_indef/cont-201,"
    "HB/sherman5,HB/sherman3,FIDAP/ex11,Hamm/memplus,Simon/raefsky3,Wang/wang3,"
    "Bai/af23560,Mallya/lhr34,Goodwin/rim,"
    "Bai/qc2534,Bai/mhd4800b"
)

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # noqa: BLE001
    _PROC = None


def cache_dir() -> Path:
    d = os.environ.get("RLA_MATGEN_CACHE")
    return Path(d) if d else Path(tempfile.gettempdir()) / "rla-matgen"


def rhs(n: int) -> np.ndarray:
    i = np.arange(n)
    return (i % 7 - 3).astype(np.complex128) + 1j * ((i % 5 - 2) * 0.5)


def _rss_mb() -> float:
    return _PROC.memory_info().rss / 1048576.0 if _PROC else 0.0


def factor_one(name: str) -> None:
    """Child-process worker: factor one matrix, print its two JSONL records to
    stdout (sampling RSS for the peak), or print nothing on failure."""
    mtx = cache_dir() / f"{name}.mtx"
    A = sio.mmread(str(mtx)).tocsc().astype(np.complex128)
    n, nnz = A.shape[0], A.nnz
    b = rhs(n)
    base, peak, stop = _rss_mb(), [_rss_mb()], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], _rss_mb())
            time.sleep(0.002)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t = time.perf_counter()
    lu = splu(A)
    fac = (time.perf_counter() - t) * 1e3
    stop.set()
    th.join()
    peak_mb = max(0.0, peak[0] - base)
    t = time.perf_counter()
    x = lu.solve(b)
    slv = (time.perf_counter() - t) * 1e3
    res = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    rec = {
        "solver": "superlu", "family": "corpus", "name": name,
        "n": int(n), "nnz": int(nnz), "threads": 1, "ana_ms": 0.0,
        "fac_ms": fac, "slv_ms": slv, "fill": int(lu.L.nnz + lu.U.nnz), "res": res,
    }
    print(json.dumps({**rec, "metric": "time", "mem_mb": 0.0}))
    print(json.dumps({**rec, "metric": "mem", "mem_mb": peak_mb}))


def main() -> None:
    timeout = float(os.environ.get("RLA_SUPERLU_TIMEOUT", "60"))
    out_path = os.environ.get("RLA_BENCH_OUT", "benches/bench_out/corpus.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    names = [gn.split("/")[-1] for gn in os.environ.get("RLA_BENCH_CORPUS", DEFAULT).split(",")]
    cache = cache_dir()
    print(f"[superlu] corpus from {cache}  timeout {timeout:.0f}s  ->  {out_path}")

    done = skipped = 0
    with open(out_path, "a") as out:
        for name in names:
            if not (cache / f"{name}.mtx").exists():
                print(f"[superlu] skip {name}: not cached (run the Rust corpus bench first)")
                continue
            try:
                t = time.perf_counter()
                r = subprocess.run([sys.executable, __file__, "--one", name],
                                   capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[superlu] CAP {name}: exceeded {timeout:.0f}s, skipped")
                skipped += 1
                continue
            lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if len(lines) != 2:
                print(f"[superlu] skip {name}: failed ({r.stderr.strip()[:80]})")
                continue
            for l in lines:
                out.write(l + "\n")
            rec = json.loads(lines[0])
            done += 1
            wall = (time.perf_counter() - t)
            print(f"[superlu] {name:<14} n={rec['n']:>7} nnz={rec['nnz']:>9}  "
                  f"fac {rec['fac_ms']:>9.1f}ms  res {rec['res']:.1e}  ({wall:.1f}s)")
    print(f"[superlu] done: {done} appended, {skipped} capped")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        factor_one(sys.argv[2])
    else:
        main()
