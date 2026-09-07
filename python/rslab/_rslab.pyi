"""Type stubs of the compiled extension ``rslab._rslab``.

The docstrings live in the Rust sources (python/src/*.rs) and in the
runtime module; see docs/api.md for the rendered reference.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

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
        threads: int | str | None = ...,
        preconditioner: float | None = ...,
        force_accept: bool = ...,
        drop_tol: float | None = ...,
        method: str = ...,
        memory: str = ...,
        ordering: str | None = ...,
        scaling: str | None = ...,
        pivot_u: float | None = ...,
        matching: bool | None = ...,
        nemin: int | None = ...,
        relax: bool | tuple[int, int] | None = ...,
        reorder: str | None = ...,
        blr: float | bool | None = ...,
        panel_nb: int | None = ...,
        scalar_gate: int | None = ...,
        par_gemm: int | None = ...,
        par_cdiv: int | None = ...,
        use_gemm_schur: bool | None = ...,
        interrupt: Interrupt | None = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class KluSettings:
    def __init__(
        self,
        *,
        pivot_tol: float | None = ...,
        row_scaling: bool | None = ...,
        btf: bool | None = ...,
        parallel: bool | None = ...,
        matching: bool | None = ...,
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
