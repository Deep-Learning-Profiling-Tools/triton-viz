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
    program_id_adapter,
    register_frontend,
)


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


TRITON_NAMESPACES: dict[Any, dict[type[Op], str]] = {
    interpreter_builder: {
        ProgramId: "create_get_program_id",
        RawStore: "create_store",
        Store: "create_masked_store",
        RawLoad: "create_load",
        Load: "create_masked_load",
        Dot: "create_dot",
        UnaryOp: "unary_op",
        BinaryOp: "binary_op",
        TernaryOp: "ternary_op",
        MakeRange: "create_make_range",
        AddPtr: "create_addptr",
        ExpandDims: "create_expand_dims",
        Broadcast: "create_broadcast",
        Splat: "create_splat",
        MakeBlockPointer: "create_make_block_ptr",
        TensorPointerLoad: "create_tensor_pointer_load",
        TensorPointerStore: "create_tensor_pointer_store",
        Idiv: "create_idiv",
        Rsqrt: "create_rsqrt",
        CastImpl: "cast_impl",
        Reshape: "create_reshape",
        Join: "create_join",
        Fabs: "create_fabs",
        Ashr: "create_ashr",
        Advance: "create_advance",
        FpToFp: "create_fp_to_fp",
        Umulhi: "create_umulhi",
        Bitcast: "create_bitcast",
        AtomicCas: "create_atomic_cas",
        AtomicRMW: "create_atomic_rmw",
    },
    tl: {
        ReduceMax: "max",
        ReduceMin: "min",
        ReduceSum: "sum",
        CumSum: "cumsum",
        Trans: "trans",
    },
    tl.math: {
        Umulhi: "umulhi",
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
    primary_namespace=interpreter_builder,
)
TRITON_OP_LIST = TRITON_FRONTEND.op_list
TRITON_ORIGINAL_OPS = TRITON_FRONTEND.original_ops
TRITON_OP_ATTR_NAMES = TRITON_FRONTEND.op_attr_names

register_frontend("triton", TRITON_FRONTEND)
