from collections.abc import Callable
from typing import Any

import triton.language as tl
from triton.runtime.interpreter import interpreter_builder

from triton_viz.core.data import (
    AddPtr,
    Advance,
    Ashr,
    AtomicCas,
    AtomicRMW,
    BinaryOp,
    Bitcast,
    Broadcast,
    CastImpl,
    CumSum,
    Dot,
    ExpandDims,
    Fabs,
    FpToFp,
    Idiv,
    Join,
    Load,
    MakeBlockPointer,
    MakeRange,
    Op,
    ProgramId,
    RawLoad,
    RawStore,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Reshape,
    Rsqrt,
    Splat,
    Store,
    TensorPointerLoad,
    TensorPointerStore,
    TernaryOp,
    Trans,
    Umulhi,
    UnaryOp,
)
from triton_viz.frontends.base import (
    AdapterResult,
    Frontend,
    OPERATION_REGISTRY,
)


def program_id_adapter(axis: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(axis)


def _triton_raw_store_adapter(
    ptr: Any, value: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, value)


def _triton_store_adapter(
    ptr: Any, _value: Any, mask: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    keys = kwargs.get("keys")
    return AdapterResult(ptr, mask, keys)


def _triton_raw_load_adapter(ptr: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(ptr)


def _triton_load_adapter(
    ptr: Any, mask: Any, _other: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, mask, kwargs.get("keys"))


def _triton_dot_adapter(a: Any, b: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(a, b)


def _triton_reduce_sum_adapter(
    input_tensor: Any,
    axis: Any = None,
    keep_dims: bool = False,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(input_tensor, axis, keep_dims)


def _triton_addptr_adapter(
    ptr: Any, offset: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, offset)


TRITON_NAMESPACES: dict[Any, dict[str, type[Op]]] = {
    interpreter_builder: {
        "create_get_program_id": ProgramId,
        "create_store": RawStore,
        "create_masked_store": Store,
        "create_load": RawLoad,
        "create_masked_load": Load,
        "create_dot": Dot,
        "unary_op": UnaryOp,
        "binary_op": BinaryOp,
        "ternary_op": TernaryOp,
        "create_make_range": MakeRange,
        "create_addptr": AddPtr,
        "create_expand_dims": ExpandDims,
        "create_broadcast": Broadcast,
        "create_splat": Splat,
        "create_make_block_ptr": MakeBlockPointer,
        "create_tensor_pointer_load": TensorPointerLoad,
        "create_tensor_pointer_store": TensorPointerStore,
        "create_idiv": Idiv,
        "create_rsqrt": Rsqrt,
        "cast_impl": CastImpl,
        "create_reshape": Reshape,
        "create_join": Join,
        "create_fabs": Fabs,
        "create_ashr": Ashr,
        "create_advance": Advance,
        "create_fp_to_fp": FpToFp,
        "create_umulhi": Umulhi,
        "create_bitcast": Bitcast,
        "create_atomic_cas": AtomicCas,
        "create_atomic_rmw": AtomicRMW,
    },
    tl: {
        "max": ReduceMax,
        "min": ReduceMin,
        "sum": ReduceSum,
        "cumsum": CumSum,
        "trans": Trans,
    },
    tl.math: {
        "umulhi": Umulhi,
    },
}
TRITON_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: program_id_adapter,
    RawStore: _triton_raw_store_adapter,
    Store: _triton_store_adapter,
    RawLoad: _triton_raw_load_adapter,
    Load: _triton_load_adapter,
    Dot: _triton_dot_adapter,
    ReduceSum: _triton_reduce_sum_adapter,
    AddPtr: _triton_addptr_adapter,
}

TRITON_FRONTEND = Frontend.from_namespaces(
    builder=interpreter_builder,
    namespaces=TRITON_NAMESPACES,
    adapters=TRITON_ADAPTERS,
)

OPERATION_REGISTRY["triton"] = TRITON_FRONTEND
