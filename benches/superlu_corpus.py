"""SuperLU (scipy.sparse.linalg.splu) over the SuiteSparse corpus.

Adds a 5th solver to the corpus scaling study: loads the same cached `.mtx`
files the Rust `bench_suite` corpus uses (populate them by running the Rust
corpus bench first, which downloads via `matgen::download::fetch`), runs an
exact SuperLU factor + solve per matrix, and appends JSONL records in the same
schema as `bench_suite` (`solver="superlu"`), so `fit_scaling.py` can plot all
solvers together.

Run:  python benches/superlu_corpus.py            # default corpus list
      RLA_BENCH_CORPUS=HB/bcsstk16,... python benches/superlu_corpus.py
Env:  RLA_BENCH_OUT (default benches/bench_out/corpus.jsonl), RLA_MATGEN_CACHE.
"""
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# The same default list as benches/bench_suite.rs `build_corpus`.
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


def cache_dir() -> Path:
    d = os.environ.get("RLA_MATGEN_CACHE")
    return Path(d) if d else Path(tempfile.gettempdir()) / "rla-matgen"


def rhs(n: int) -> np.ndarray:
    # Matches the Rust corpus RHS so residuals are comparable.
    i = np.arange(n)
    return (i % 7 - 3).astype(np.complex128) + 1j * ((i % 5 - 2) * 0.5)


def main() -> None:
    out_path = os.environ.get("RLA_BENCH_OUT", "benches/bench_out/corpus.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    names = [gn.split("/")[-1] for gn in os.environ.get("RLA_BENCH_CORPUS", DEFAULT).split(",")]
    cache = cache_dir()
    print(f"[superlu] corpus from {cache}  ->  {out_path}")

    done = 0
    with open(out_path, "a") as out:
        for name in names:
            mtx = cache / f"{name}.mtx"
            if not mtx.exists():
                print(f"[superlu] skip {name}: {mtx} not cached (run the Rust corpus bench first)")
                continue
            try:
                A = sio.mmread(str(mtx)).tocsc().astype(np.complex128)
            except Exception as e:  # noqa: BLE001
                print(f"[superlu] skip {name}: read {type(e).__name__} {str(e)[:60]}")
                continue
            n, nnz = A.shape[0], A.nnz
            b = rhs(n)
            try:
                t = time.perf_counter()
                lu = splu(A)
                fac = (time.perf_counter() - t) * 1e3
                t = time.perf_counter()
                x = lu.solve(b)
                slv = (time.perf_counter() - t) * 1e3
                res = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
                fill = int(lu.L.nnz + lu.U.nnz)
            except Exception as e:  # noqa: BLE001
                print(f"[superlu] {name}: FACTOR/SOLVE FAILED {type(e).__name__} {str(e)[:60]}")
                continue
            rec = {
                "solver": "superlu", "family": "corpus", "name": name,
                "n": int(n), "nnz": int(nnz), "threads": 1, "metric": "time",
                "ana_ms": 0.0, "fac_ms": fac, "slv_ms": slv, "mem_mb": 0.0,
                "fill": fill, "res": res,
            }
            out.write(json.dumps(rec) + "\n")
            done += 1
            print(f"[superlu] {name:<14} n={n:>7} nnz={nnz:>9}  fac {fac:>9.1f}ms  res {res:.1e}")
    print(f"[superlu] done: {done} matrices appended")


if __name__ == "__main__":
    main()
