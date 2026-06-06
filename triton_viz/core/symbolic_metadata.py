from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import numpy as np

from triton_viz.utils.dtypes import STORAGE_DTYPES


@dataclass(frozen=True)
class SymbolicScalarDType:
    name: str
    primitive_bitwidth: int
    np_dtype: np.dtype[Any]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class SymbolicPointerDType:
    element_ty: SymbolicScalarDType

    @property
    def name(self) -> str:
        return f"pointer<{self.element_ty}>"

    @property
    def np_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.uint64)

    def __str__(self) -> str:
        return self.name


SymbolicDType: TypeAlias = SymbolicScalarDType | SymbolicPointerDType


@dataclass(frozen=True)
class SymbolicTypeSpec:
    dtype: SymbolicDType
    shape: tuple[int, ...] = ()

    def __str__(self) -> str:
        if not self.shape:
            return str(self.dtype)
        return f"<{list(self.shape)}, {self.dtype}>"


@dataclass
class SymbolicTensorValue:
    data: np.ndarray
    dtype: SymbolicDType
    attr: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.data.shape)


INT1 = SymbolicScalarDType("int1", 1, np.dtype(bool))
INT8 = SymbolicScalarDType("int8", 8, np.dtype(np.int8))
INT16 = SymbolicScalarDType("int16", 16, np.dtype(np.int16))
INT32 = SymbolicScalarDType("int32", 32, np.dtype(np.int32))
INT64 = SymbolicScalarDType("int64", 64, np.dtype(np.int64))
UINT8 = SymbolicScalarDType("uint8", 8, np.dtype(np.uint8))
UINT16 = SymbolicScalarDType("uint16", 16, np.dtype(np.uint16))
UINT32 = SymbolicScalarDType("uint32", 32, np.dtype(np.uint32))
UINT64 = SymbolicScalarDType("uint64", 64, np.dtype(np.uint64))
FLOAT16 = SymbolicScalarDType("fp16", 16, np.dtype(np.float16))
FLOAT32 = SymbolicScalarDType("fp32", 32, np.dtype(np.float32))
FLOAT64 = SymbolicScalarDType("fp64", 64, np.dtype(np.float64))

BFLOAT16 = SymbolicScalarDType(
    "bf16", 16, cast(np.dtype[Any], STORAGE_DTYPES["bfloat16"])
)
FLOAT8_E4M3 = SymbolicScalarDType(
    "float8_e4m3", 8, cast(np.dtype[Any], STORAGE_DTYPES["float8_e4m3"])
)
FLOAT8_E5M2 = SymbolicScalarDType(
    "float8_e5m2", 8, cast(np.dtype[Any], STORAGE_DTYPES["float8_e5m2"])
)
FLOAT8_E4M3FN = SymbolicScalarDType(
    "float8_e4m3fn", 8, cast(np.dtype[Any], STORAGE_DTYPES["float8_e4m3fn"])
)
FLOAT4_E2M1FN = SymbolicScalarDType(
    "float4_e2m1fn", 4, cast(np.dtype[Any], STORAGE_DTYPES["float4_e2m1fn"])
)


_DTYPES: tuple[SymbolicScalarDType, ...] = (
    INT1,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT8_E5M2,
    FLOAT8_E4M3FN,
    FLOAT4_E2M1FN,
)

DTYPE_BY_NAME: dict[str, SymbolicScalarDType] = {dtype.name: dtype for dtype in _DTYPES}
DTYPE_BY_NAME.update(
    {
        "bool": INT1,
        "fp8e4nv": FLOAT8_E4M3,
        "fp8e5": FLOAT8_E5M2,
        "float16": FLOAT16,
        "half": FLOAT16,
        "float": FLOAT32,
        "float32": FLOAT32,
        "double": FLOAT64,
        "float64": FLOAT64,
        "bfloat16": BFLOAT16,
    }
)


def pointer_type(element_ty: SymbolicDType | str) -> SymbolicPointerDType:
    dtype = normalize_dtype(element_ty)
    if isinstance(dtype, SymbolicPointerDType):
        raise TypeError("Nested symbolic pointer dtypes are not supported")
    return SymbolicPointerDType(dtype)


def block_type(
    dtype: SymbolicDType | str, shape: Sequence[int]
) -> SymbolicTypeSpec:
    return SymbolicTypeSpec(
        normalize_dtype(dtype),
        tuple(int(dim) for dim in shape),
    )


def type_spec(
    dtype: SymbolicDType | str | SymbolicTypeSpec,
    shape: Sequence[int] | None = None,
) -> SymbolicTypeSpec:
    if isinstance(dtype, SymbolicTypeSpec):
        if shape is not None:
            return SymbolicTypeSpec(dtype.dtype, tuple(int(dim) for dim in shape))
        return dtype
    return SymbolicTypeSpec(normalize_dtype(dtype), tuple(int(dim) for dim in shape or ()))


def normalize_dtype(dtype: SymbolicDType | str) -> SymbolicDType:
    if isinstance(dtype, (SymbolicScalarDType, SymbolicPointerDType)):
        return dtype
    try:
        return DTYPE_BY_NAME[str(dtype)]
    except KeyError as exc:
        raise TypeError(f"Unsupported symbolic dtype: {dtype}") from exc


def normalize_symbolic_value(value: Any) -> Any:
    if isinstance(
        value,
        (
            SymbolicScalarDType,
            SymbolicPointerDType,
            SymbolicTypeSpec,
            SymbolicTensorValue,
        ),
    ):
        return value
    return value


def unpack_type_spec(
    dtype: SymbolicDType | SymbolicTypeSpec,
    fallback_shape: Sequence[int] = (),
) -> tuple[SymbolicDType, tuple[int, ...]]:
    if isinstance(dtype, SymbolicTypeSpec):
        return dtype.dtype, dtype.shape
    return dtype, tuple(int(dim) for dim in fallback_shape)


def dtype_to_numpy(dtype: SymbolicDType) -> np.dtype[Any]:
    if isinstance(dtype, SymbolicPointerDType):
        return np.dtype(np.uint64)
    return dtype.np_dtype


def is_pointer_dtype(dtype: SymbolicDType | None) -> bool:
    return isinstance(dtype, SymbolicPointerDType)


def pointee_dtype(dtype: SymbolicDType | None) -> SymbolicDType | None:
    if isinstance(dtype, SymbolicPointerDType):
        return dtype.element_ty
    return dtype


def element_bytewidth(dtype: SymbolicDType | None) -> int:
    if isinstance(dtype, SymbolicPointerDType):
        bitwidth = dtype.element_ty.primitive_bitwidth
    elif isinstance(dtype, SymbolicScalarDType):
        bitwidth = dtype.primitive_bitwidth
    else:
        bitwidth = 8
    return max(1, int(bitwidth) // 8)
