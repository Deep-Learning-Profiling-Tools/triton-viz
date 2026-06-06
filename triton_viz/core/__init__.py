from __future__ import annotations

from typing import Any

from .data import (
    AddPtr,
    Broadcast,
    CastImpl,
    Dot,
    ExpandDims,
    IntToPtr,
    Idiv,
    Load,
    MakeBlockPointer,
    MakeRange,
    Op,
    ProgramId,
    PtrToInt,
    RawLoad,
    RawStore,
    Reduce,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Rsqrt,
    Splat,
    Store,
    TensorPointerLoad,
    TensorPointerStore,
    TernaryOp,
    UnaryOp,
    BinaryOp,
)
from .masked_load_store import masked_load, masked_store


__all__ = [
    "trace",
    "clear",
    "Op",
    "ProgramId",
    "RawStore",
    "Store",
    "RawLoad",
    "Load",
    "UnaryOp",
    "BinaryOp",
    "TernaryOp",
    "Dot",
    "MakeRange",
    "AddPtr",
    "ExpandDims",
    "Broadcast",
    "Reduce",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "Splat",
    "MakeBlockPointer",
    "TensorPointerLoad",
    "TensorPointerStore",
    "Idiv",
    "Rsqrt",
    "CastImpl",
    "PtrToInt",
    "IntToPtr",
    "masked_load",
    "masked_store",
]


def trace(*args: Any, **kwargs: Any) -> Any:
    from .trace import trace as _trace

    return _trace(*args, **kwargs)


def clear(*args: Any, **kwargs: Any) -> Any:
    from .trace import clear as _clear

    return _clear(*args, **kwargs)
