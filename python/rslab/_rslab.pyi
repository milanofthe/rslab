"""Type stubs of the compiled extension ``rslab._rslab``.

The docstrings live in the Rust sources (python/src/*.rs) and in the
runtime module; see docs/api.md for the rendered reference.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

__version__: str

Operator = tuple[int, NDArray[np.int64], NDArray[np.int64], NDArray[Any]]

class Interrupt:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    def reset(self) -> None: ...
    @property
    def is_set(self) -> bool: ...

class Settings:
    def __init__(
        self,
        *,
        threads: int | str | tuple[str, int] | None = ...,
        preconditioner: float | None = ...,
        force_accept: bool = ...,
        drop_tol: float | None = ...,
        ordering: str | None = ...,
        race_candidates: Sequence[str] | None = ...,
        scaling: str | NDArray[Any] | None = ...,
        relax: bool | tuple[int, int] | None = ...,
        amalgamation: str | None = ...,
        pivot_threshold: float | None = ...,
        matching_negligible_diagonal: float | None = ...,
        compress_max_ratio: float | None = ...,
        nd_two_hop_ratio: float | None = ...,
        nd_max_imbalance: float | None = ...,
        nd_max_overshoot: float | None = ...,
        amd_dense_alpha: float | None = ...,
        amf_dense_alpha: float | None = ...,
        path_like_fraction: float | None = ...,
        root_cap_fraction: float | None = ...,
        matching: bool | None = ...,
        nd_ensemble: bool | None = ...,
        amd_aggressive: bool | None = ...,
        use_gemm_schur: bool | None = ...,
        race_nd_min_n: int | None = ...,
        race_nd_min_work: int | None = ...,
        race_assumed_workers: int | None = ...,
        race_eager_nd_min_nnz: int | None = ...,
        race_ensemble_size: int | None = ...,
        race_ensemble_min_flops: int | None = ...,
        nd_seed: int | None = ...,
        nd_init_trials: int | None = ...,
        nd_coarsen_floor: int | None = ...,
        nd_leaf_size: int | None = ...,
        nd_fm_passes: int | None = ...,
        nd_move_limit: int | None = ...,
        nd_parallel_min_vertices: int | None = ...,
        nd_parallel_min_edges: int | None = ...,
        nemin: int | None = ...,
        relax_min_n: int | None = ...,
        root_cap_min_n: int | None = ...,
        root_cap_max: int | None = ...,
        panel_nb: int | None = ...,
        scalar_gate: int | None = ...,
        par_gemm: int | None = ...,
        par_cdiv: int | None = ...,
        fork_min_flops: int | None = ...,
        schur_tile: int | None = ...,
        trailing_block: int | None = ...,
        complex_split_min_ratio: int | None = ...,
        complex_split_tile: int | None = ...,
        solve_leaf_subtrees: int | None = ...,
        solve_block: int | None = ...,
        solve_ancestor_chunk: int | None = ...,
        solve_apex_min_work: int | None = ...,
        interrupt: Interrupt | None = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class KluSettings:
    def __init__(
        self,
        *,
        pivot_threshold: float | None = ...,
        row_scaling: bool | None = ...,
        btf: bool | None = ...,
        parallel: bool | None = ...,
        matching: bool | None = ...,
        par_min_nnz: int | None = ...,
        par_min_work: int | None = ...,
        par_min_ratio: float | None = ...,
        interrupt: Interrupt | None = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class Recycle:
    @property
    def k(self) -> int: ...
    @property
    def active(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def clear(self) -> None: ...

class KrylovResult(Sequence[Any]):
    x: NDArray[Any]
    converged: bool
    iters: int
    final_res: float | NDArray[np.float64]
    stop: str
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Any: ...  # type: ignore[override]
    def __iter__(self) -> Iterator[Any]: ...

class _Factor:
    @property
    def n(self) -> int: ...
    @property
    def factor_nnz(self) -> int: ...
    @property
    def n_perturbed(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def diagnostics(self) -> dict[str, Any]: ...
    def solve(
        self,
        b: NDArray[Any],
        refine: int = ...,
        target: float | None = ...,
        measure: str = ...,
    ) -> NDArray[Any]: ...
    def solve_many(self, b: NDArray[Any]) -> NDArray[Any]: ...
    def gmres(
        self,
        b: NDArray[Any],
        tol: float = ...,
        maxit: int = ...,
        restart: int | None = ...,
        x0: NDArray[Any] | None = ...,
        recycle: Recycle | None = ...,
        operator: Operator | None = ...,
    ) -> KrylovResult: ...
    def gmres_block(
        self,
        b: NDArray[Any],
        tol: float = ...,
        maxit: int = ...,
        restart: int | None = ...,
        x0: NDArray[Any] | None = ...,
        operator: Operator | None = ...,
    ) -> KrylovResult: ...
    def cocg(
        self, b: NDArray[Any], tol: float = ..., maxit: int = ..., operator: Operator | None = ...
    ) -> KrylovResult: ...
    def cocr(
        self, b: NDArray[Any], tol: float = ..., maxit: int = ..., operator: Operator | None = ...
    ) -> KrylovResult: ...
    def recycle(self, k: int) -> Recycle: ...

class Ldlt(_Factor):
    @property
    def inertia(self) -> tuple[int, int, int]: ...

class Lu(_Factor): ...

class Klu(_Factor):
    @property
    def n_blocks(self) -> int: ...
    @property
    def L(self) -> sp.csc_matrix: ...
    @property
    def U(self) -> sp.csc_matrix: ...
    @property
    def F(self) -> sp.csc_matrix: ...
    @property
    def perm_r(self) -> NDArray[np.int64]: ...
    @property
    def perm_c(self) -> NDArray[np.int64]: ...
    @property
    def block_ptr(self) -> NDArray[np.int64]: ...
    @property
    def row_scale(self) -> NDArray[np.float64]: ...
    def solve_transpose(self, b: NDArray[Any]) -> NDArray[Any]: ...
    def refactor(self, data: NDArray[Any] | Any) -> None: ...

class _Symbolic:
    @property
    def n(self) -> int: ...
    @property
    def factor_nnz(self) -> int: ...
    def estimate_memory(self, dtype: str = ...) -> dict[str, Any]: ...

class LdltSymbolic(_Symbolic):
    @property
    def n_levels(self) -> int: ...
    @property
    def level_widths(self) -> list[int]: ...
    @property
    def front_dims(self) -> list[tuple[int, int]]: ...
    @property
    def settings(self) -> Settings: ...
    def factor(self, data: NDArray[Any] | Any, settings: Settings | None = ..., **kwargs: Any) -> Ldlt: ...

class LuSymbolic(_Symbolic):
    @property
    def n_levels(self) -> int: ...
    @property
    def level_widths(self) -> list[int]: ...
    @property
    def front_dims(self) -> list[tuple[int, int]]: ...
    @property
    def settings(self) -> Settings: ...
    def factor(self, data: NDArray[Any] | Any, settings: Settings | None = ..., **kwargs: Any) -> Lu: ...

class KluSymbolic(_Symbolic):
    @property
    def n_blocks(self) -> int: ...
    @property
    def max_block_size(self) -> int: ...
    @property
    def block_ptr(self) -> list[int]: ...
    @property
    def settings(self) -> KluSettings: ...
    def factor(self, data: NDArray[Any] | Any, settings: KluSettings | None = ..., **kwargs: Any) -> Klu: ...

def ldlt_factor(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    settings: Settings | None = ..., **kwargs: Any,
) -> Ldlt: ...
def lu_factor(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    settings: Settings | None = ..., **kwargs: Any,
) -> Lu: ...
def klu_factor(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    settings: KluSettings | None = ..., **kwargs: Any,
) -> Klu: ...
def analyze_ldlt(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    settings: Settings | None = ..., **kwargs: Any,
) -> LdltSymbolic: ...
def analyze_lu(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    settings: Settings | None = ..., **kwargs: Any,
) -> LuSymbolic: ...
def analyze_klu(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    settings: KluSettings | None = ..., **kwargs: Any,
) -> KluSymbolic: ...
def gmres_plain(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    b: NDArray[Any], tol: float = ..., maxit: int = ..., restart: int | None = ...,
    x0: NDArray[Any] | None = ...,
) -> KrylovResult: ...
def gmres_block_plain(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    b: NDArray[Any], tol: float = ..., maxit: int = ..., restart: int | None = ...,
    x0: NDArray[Any] | None = ...,
) -> KrylovResult: ...
def cocg_plain(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    b: NDArray[Any], tol: float = ..., maxit: int = ...,
) -> KrylovResult: ...
def cocr_plain(
    n: int, indptr: NDArray[np.int64], indices: NDArray[np.int64], data: NDArray[Any],
    b: NDArray[Any], tol: float = ..., maxit: int = ...,
) -> KrylovResult: ...
def install_diagnose() -> dict[str, Any]: ...
def set_log_level(level: str) -> None: ...
def log_level() -> str: ...
def set_log_sink(sink: Callable[[str, str], None] | None) -> None: ...
