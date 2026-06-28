"""SuperLU (scipy.sparse.linalg.splu) reference for the `vs_all` head-to-head.

Reads the same complex MoM matrices as `benches/vs_all.rs`, builds the **same**
right-hand side, runs an exact SuperLU factor + solve, and prints factor/solve
time and the true relative residual - one line per matrix, to sit beside the
Rust table (RLA-LL / RLA-MF / faer / PARDISO).

Run:  python benches/superlu_mom.py  [substring-filter]
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import splu

DIR = Path(r"C:\Repositories\rapidmom\precond_matrices")


def rhs(n: int) -> np.ndarray:
    i = np.arange(n)
    return (i % 7 - 3).astype(np.complex128) + 1j * ((i % 5 - 2) * 0.5)


def run(path: Path) -> None:
    A = sio.mmread(str(path)).tocsc().astype(np.complex128)
    n = A.shape[0]
    b = rhs(n)
    t = time.perf_counter()
    try:
        lu = splu(A)  # SuperLU: combined analyze + factor
    except Exception as e:  # noqa: BLE001
        print(f"  SuperLU  FACTOR FAILED: {type(e).__name__} {str(e)[:80]}")
        return
    fac = (time.perf_counter() - t) * 1e3
    t = time.perf_counter()
    x = lu.solve(b)
    slv = (time.perf_counter() - t) * 1e3
    res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    fill = lu.L.nnz + lu.U.nnz
    print(f"{path.name}  n={n}  SuperLU  fac {fac:8.1f}  slv {slv:6.1f}  res {res:.1e}  fill {fill}")


def main() -> None:
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(DIR.glob("*.mtx"), key=lambda p: p.stat().st_size)
    print(f"SuperLU (scipy {sp.__version__ if hasattr(sp, '__version__') else ''}) exact direct solve\n")
    for f in files:
        if not flt or flt in f.name:
            run(f)


if __name__ == "__main__":
    main()
