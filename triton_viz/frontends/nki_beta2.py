from collections.abc import Callable
from typing import Any

from triton_viz.core.data import Allocate, Dot, Op, ProgramId
from triton_viz.frontends.base import AdapterResult, Frontend, OPERATION_REGISTRY

HAS_NKI_BETA2 = False
nisa = None
nl = None
NDArray = None
nki_builder = None
try:
    from triton_viz.core import nki_beta2 as _nki_beta2  # type: ignore

    NDArray = _nki_beta2.NDArray
    nki_builder = _nki_beta2.nki_builder
    nisa = _nki_beta2.nisa
    nl = _nki_beta2.nl

    HAS_NKI_BETA2 = True
except ModuleNotFoundError:
    pass


def program_id_adapter(axis: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(axis)


def _nki_beta2_allocate_adapter(*_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult()


def _nki_beta2_dot_adapter(
    dst: Any, stationary: Any, moving: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    del dst
    assert NDArray is not None
    x = NDArray(value=stationary.data.T)
    return AdapterResult(x, moving)


NKI_BETA2_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_BETA2_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI_BETA2:
    assert nl is not None
    assert nisa is not None

    NKI_BETA2_NAMESPACES = {
        nl: {
            "program_id": ProgramId,
            "ndarray": Allocate,
        },
        nisa: {
            "nc_matmul": Dot,
        },
    }

    NKI_BETA2_ADAPTERS = {
        ProgramId: program_id_adapter,
        Allocate: _nki_beta2_allocate_adapter,
        Dot: _nki_beta2_dot_adapter,
    }

NKI_BETA2_FRONTEND = Frontend.from_namespaces(
    builder=nki_builder,
    namespaces=NKI_BETA2_NAMESPACES,
    adapters=NKI_BETA2_ADAPTERS,
)

OPERATION_REGISTRY["nki_beta2"] = NKI_BETA2_FRONTEND
