from collections.abc import Callable
from typing import Any

from triton_viz.core.data import (
    Allocate,
    Dot,
    NkiDmaCopy,
    NkiTensorCopy,
    NkiTensorScalar,
    NkiTensorTensor,
    Op,
    ProgramId,
)
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


def _nki_beta2_dma_copy_adapter(
    dst: Any, src: Any, dst_rmw_op: Any = None, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(dst, src, dst_rmw_op=dst_rmw_op)


def _nki_beta2_tensor_copy_adapter(
    dst: Any, src: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(dst, src)


def _nki_beta2_tensor_scalar_adapter(
    dst: Any,
    data: Any,
    op0: Any,
    operand0: Any,
    reverse0: bool = False,
    op1: Any = None,
    operand1: Any = None,
    reverse1: bool = False,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    del operand0, reverse0, operand1, reverse1
    return AdapterResult(dst, data, op0=op0, op1=op1)


def _nki_beta2_tensor_tensor_adapter(
    dst: Any, data1: Any, data2: Any, op: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(dst, data1, data2, op=op)


NKI_BETA2_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
NKI_BETA2_NAMESPACES: dict[Any, dict[str, type[Op]]] = {}
if HAS_NKI_BETA2:
    assert nki_builder is not None

    NKI_BETA2_NAMESPACES = {
        _nki_beta2: {
            "program_id": ProgramId,
            "ndarray": Allocate,
            "nc_matmul": Dot,
            "dma_copy": NkiDmaCopy,
            "tensor_copy": NkiTensorCopy,
            "tensor_scalar": NkiTensorScalar,
            "tensor_tensor": NkiTensorTensor,
        },
    }

    NKI_BETA2_ADAPTERS = {
        ProgramId: program_id_adapter,
        Allocate: _nki_beta2_allocate_adapter,
        Dot: _nki_beta2_dot_adapter,
        NkiDmaCopy: _nki_beta2_dma_copy_adapter,
        NkiTensorCopy: _nki_beta2_tensor_copy_adapter,
        NkiTensorScalar: _nki_beta2_tensor_scalar_adapter,
        NkiTensorTensor: _nki_beta2_tensor_tensor_adapter,
    }

NKI_BETA2_FRONTEND = Frontend.from_namespaces(
    builder=nki_builder,
    namespaces=NKI_BETA2_NAMESPACES,
    adapters=NKI_BETA2_ADAPTERS,
)

OPERATION_REGISTRY["nki_beta2"] = NKI_BETA2_FRONTEND
