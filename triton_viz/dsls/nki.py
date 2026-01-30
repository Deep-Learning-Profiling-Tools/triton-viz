from collections.abc import Callable
from typing import Any, Optional

from triton_viz.core.data import (
    Allocate,
    Dot,
    Load,
    MakeRange,
    Op,
    ProgramId,
    ReduceSum,
    Store,
    UnaryOp,
)
from triton_viz.dsls.base import (
    AdapterResult,
    Frontend,
    OPERATION_REGISTRY,
)

HAS_NKI = False
nki_builder = None
try:
    from triton_viz.core.nki import nki_builder  # type: ignore

    HAS_NKI = True
except ModuleNotFoundError:
    pass


def program_id_adapter(axis: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(axis)


def _nki_allocate_adapter(*_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult()


def _nki_load_adapter(
    src: Any, keys: Any, *, mask: Optional[Any] = None, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(src, mask, keys)


def _nki_store_adapter(
    dst: Any,
    keys: Any,
    value: Any,
    *,
    mask: Optional[Any] = None,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(dst, mask, keys)


def _nki_dot_adapter(x: Any, y: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(x, y)


def _nki_reduce_sum_adapter(
    input_tensor: Any, *args: Any, mask: Any = None, **kwargs: Any
) -> AdapterResult:
    axis = args[0] if args else kwargs.get("axis")
    keep_dims = kwargs.get("keep_dims", kwargs.get("keepdims", False))
    return AdapterResult(input_tensor, axis, keep_dims)


NKI_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI:
    assert nki_builder is not None

    NKI_NAMESPACES = {
        nki_builder: {
            "program_id": ProgramId,
            "ndarray": Allocate,
            "masked_load": Load,
            "masked_store": Store,
            "matmul": Dot,
            "_unary_op": UnaryOp,
            "sum": ReduceSum,
            "arange": MakeRange,
        }
    }

    NKI_ADAPTERS = {
        ProgramId: program_id_adapter,
        Allocate: _nki_allocate_adapter,
        Load: _nki_load_adapter,
        Store: _nki_store_adapter,
        Dot: _nki_dot_adapter,
        ReduceSum: _nki_reduce_sum_adapter,
    }

NKI_FRONTEND = Frontend.from_namespaces(
    builder=nki_builder,
    namespaces=NKI_NAMESPACES,
    adapters=NKI_ADAPTERS,
)

OPERATION_REGISTRY["nki"] = NKI_FRONTEND
