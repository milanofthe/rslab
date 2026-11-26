import numpy as np
import scipy.sparse as sp

from constants import U32, U64, UC32, UC64, recomputed_residual, recursive_gap


def unit_roundoff(dtype):
    single = np.finfo(dtype).bits == 32
    if np.dtype(dtype).kind == "c":
        return UC32 if single else UC64
    return U32 if single else U64


def inf_norm(A):
    return float(np.max(abs(A).sum(axis=1)))


def backward_error(A, x, b):
    r = np.max(np.abs(A @ x - b))
    return float(r / (inf_norm(A) * np.max(np.abs(x)) + np.max(np.abs(b))))


def residual_norm(A, x, b):
    return float(np.linalg.norm(A @ x - b))


def row_length(A):
    return int(np.max(np.diff(sp.csr_matrix(A).indptr)))


def krylov_bound(A, x, b, tol):
    k = row_length(A)
    abs_norm = np.linalg.norm(np.abs(b) + abs(A) @ np.abs(x))
    return recomputed_residual(tol, np.linalg.norm(b), abs_norm, A.shape[0], k, U64)


def recursive_drift(solve, A, b, iters):
    b_norm = np.linalg.norm(b)
    ax_max, r_max = 0.0, b_norm
    for j in range(1, iters + 1):
        step = solve(j)
        ax_max = max(ax_max, np.linalg.norm(abs(A) @ np.abs(step.x)))
        r_max = max(r_max, step.final_res * b_norm)
    return recursive_gap(iters, row_length(A), U64, ax_max, r_max)
