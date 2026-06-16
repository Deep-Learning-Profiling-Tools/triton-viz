"""NumPy-backed execution for Gluon kernels.

The simulator reuses Triton's interpreter tensors, memory operations, and
host/device argument copyback machinery, while adapting Gluon's language
signatures to Triton's interpreter semantic layer.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
from typing import Any

import numpy as np
import triton
import triton.language as tl
import triton.language.extra.libdevice as triton_libdevice  # type: ignore
from triton.experimental.gluon import language as gl  # type: ignore
from triton.experimental.gluon.language import _core as gluon_core  # type: ignore
from triton.experimental.gluon.language import _math as gluon_math  # type: ignore
from triton.experimental.gluon.language import _standard as gluon_standard  # type: ignore
from triton.experimental.gluon.language import amd as gluon_amd  # type: ignore
from triton.experimental.gluon.language.amd import cdna3 as gluon_amd_cdna3  # type: ignore
from triton.experimental.gluon.language.amd import cdna4 as gluon_amd_cdna4  # type: ignore
from triton.experimental.gluon.language.amd import rdna3 as gluon_amd_rdna3  # type: ignore
from triton.experimental.gluon.language.amd import rdna4 as gluon_amd_rdna4  # type: ignore
from triton.experimental.gluon.language.amd.cdna4 import (  # type: ignore
    async_copy as gluon_amd_cdna4_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    async_copy as gluon_amd_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    cluster as gluon_amd_cluster,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    mbarrier as gluon_amd_mbarrier,
)
from triton.experimental.gluon.language.amd import gfx1250 as gluon_amd_gfx1250  # type: ignore
from triton.experimental.gluon.language.nvidia.ampere import (  # type: ignore
    async_copy as gluon_ampere_async_copy,
)
from triton.experimental.gluon.language.amd.gfx1250 import (  # type: ignore
    tdm as gluon_amd_tdm,
)
from triton.experimental.gluon.language.nvidia import blackwell as gluon_blackwell  # type: ignore
from triton.experimental.gluon.language.nvidia import hopper as gluon_hopper  # type: ignore
from triton.experimental.gluon.language.nvidia.blackwell import clc as gluon_clc  # type: ignore
from triton.experimental.gluon.language.nvidia.blackwell import (  # type: ignore
    tma as gluon_blackwell_tma,
)
from triton.experimental.gluon.language.nvidia.hopper import (  # type: ignore
    mbarrier as gluon_hopper_mbarrier,
)
from triton.experimental.gluon.language.nvidia.hopper import (  # type: ignore
    tma as gluon_hopper_tma,
)
from triton.runtime.interpreter import (  # type: ignore
    GridExecutor,
    InterpreterError,
    TensorHandle,
    _convert_float,
    _mxfp_value_handle_to_float32,
    _implicit_cvt,
    _get_np_dtype,
    _patch_lang_core,
    _patch_lang_tensor,
    _unpack_e2m1,
    interpreter_builder,
)
from triton.language.semantic import TritonSemantic

from ..frontend.base import AdapterResult, _LangPatchScope
from ..symbolic_metadata import TensorDescriptorAccess

_POINTER_TENSORS: dict[int, Any] = {}
_CURRENT_NUM_WARPS: int | None = None
_CURRENT_NUM_CTAS: int | None = None
_CURRENT_GRID_DIM: tuple[int, int, int] | None = None
_CURRENT_GRID_IDX: tuple[int, int, int] | None = None
_HOST_TENSOR_DESCRIPTOR_TYPES = {"TensorDescriptor", "TensorDescriptorIm2Col"}
_MISSING = object()
_MBARRIER_STATES: dict[int, "_MBarrierState"] = {}
_PENDING_CLC_BARRIERS: set[int] = set()


class _MBarrierState:
    def __init__(self, ready: bool = True, phase: int = 1) -> None:
        self.ready = ready
        self.phase = phase
        self.pending_bytes = 0
        self.condition = threading.Condition()


class _WarpSpecializeScheduler:
    def __init__(self, count: int) -> None:
        self._active = list(range(count))
        self._turn_index = 0
        self._condition = threading.Condition()
        self._local = threading.local()

    def bind(self, partition_id: int) -> None:
        self._local.partition_id = partition_id

    def yield_point(self) -> None:
        partition_id = getattr(self._local, "partition_id", None)
        if partition_id is None:
            return
        with self._condition:
            while (
                partition_id in self._active
                and self._active
                and self._active[self._turn_index] != partition_id
            ):
                self._condition.wait()
            if partition_id not in self._active or not self._active:
                return
            self._turn_index = (self._turn_index + 1) % len(self._active)
            self._condition.notify_all()

    def finish(self, partition_id: int) -> None:
        with self._condition:
            if partition_id in self._active:
                removed_index = self._active.index(partition_id)
                self._active.pop(removed_index)
                if self._active:
                    if removed_index < self._turn_index:
                        self._turn_index -= 1
                    self._turn_index %= len(self._active)
                else:
                    self._turn_index = 0
            self._condition.notify_all()


class GluonSemantic(TritonSemantic):
    """Triton interpreter semantic with Gluon-compatible builtin signatures."""

    def arange(self, start: int, end: int, layout: Any = None):
        return super().arange(start, end)

    def full(self, shape: list[int], value: Any, dtype: tl.dtype, layout: Any = None):
        return super().full(shape, value, dtype)

    def dot_fma(self, a: tl.tensor, b: tl.tensor, acc: tl.tensor):
        return super().dot(
            a,
            b,
            acc,
            input_precision=None,
            max_num_imprecise_acc=0,
            out_dtype=None,
        )

    def convert_layout(self, value: Any, layout: Any, assert_trivial: bool = False):
        return value


gluon_semantic = GluonSemantic(interpreter_builder)


def _unwrap_constexpr(value: Any) -> Any:
    if isinstance(value, tl.constexpr):
        return value.value
    return value


def _to_python_scalar(value: Any) -> Any:
    value = _unwrap_constexpr(value)
    if isinstance(value, tl.tensor):
        data = value.handle.data
        if np.asarray(data).size == 1:
            return np.asarray(data).reshape(-1)[0].item()
        return data
    if isinstance(value, TensorHandle):
        data = value.data
        if np.asarray(data).size == 1:
            return np.asarray(data).reshape(-1)[0].item()
        return data
    return value


def _to_int(value: Any) -> int:
    return int(_to_python_scalar(value))


def _dtype_from_base(base: Any) -> tl.dtype:
    if isinstance(base, tl.tensor):
        dtype = base.dtype
        return dtype.element_ty if isinstance(dtype, tl.pointer_type) else dtype
    if isinstance(base, TensorHandle):
        dtype = base.dtype
        return dtype.element_ty if isinstance(dtype, tl.pointer_type) else dtype
    if getattr(base, "dtype", _MISSING) is not _MISSING:
        dtype = tl.str_to_ty(triton.runtime.jit.mangle_type(base), None)
        return dtype.element_ty if isinstance(dtype, tl.pointer_type) else dtype
    raise TypeError(f"Cannot infer descriptor dtype from {type(base)}")


def _numpy_array(base: Any) -> np.ndarray:
    detach = getattr(base, "detach", None)
    cpu = getattr(base, "cpu", None)
    if callable(detach) and callable(cpu):
        return detach().cpu().numpy()
    numpy = getattr(base, "numpy", None)
    if callable(numpy):
        return numpy()
    return np.asarray(base)


class SimulatedBlockType:
    def __init__(self, element_ty: tl.dtype, shape: list[int]) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]

    @property
    def nbytes(self) -> int:
        bitwidth = getattr(self.element_ty, "primitive_bitwidth", 8)
        return int(np.prod(self.shape, dtype=np.int64)) * max(1, int(bitwidth) // 8)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SimulatedBlockType)
            and self.element_ty == other.element_ty
            and self.shape == list(other.shape)
        )


class SimulatedTensorDescriptorType:
    def __init__(self, block_type: SimulatedBlockType) -> None:
        self.block_type = block_type


class SimulatedSharedMemoryDescriptorType:
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        alloc_shape: list[int] | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = tuple(int(dim) for dim in shape)
        self.layout = layout
        self.alloc_shape = (
            tuple(int(dim) for dim in alloc_shape)
            if alloc_shape is not None
            else self.shape
        )

    @property
    def nbytes_per_cta(self) -> int:
        bitwidth = getattr(self.element_ty, "primitive_bitwidth", 8)
        return int(np.prod(self.shape, dtype=np.int64)) * max(1, int(bitwidth) // 8)


class SimulatedTensorDescriptor(gluon_hopper_tma.tensor_descriptor):
    def __init__(
        self,
        base: Any,
        shape: list[int],
        strides: list[int],
        block_shape: list[int],
        layout: Any,
        padding: str = "zero",
        element_strides: list[int] | None = None,
        pixel_box_lower_corner: list[int] | None = None,
        pixel_box_upper_corner: list[int] | None = None,
        origin: list[int] | None = None,
    ) -> None:
        self.base = base
        self.shape = tuple(int(dim) for dim in shape)
        self.strides = tuple(int(stride) for stride in strides)
        self.block_shape = tuple(int(dim) for dim in block_shape)
        self.layout = layout
        self.padding = padding
        self.element_strides = (
            tuple(int(stride) for stride in element_strides)
            if element_strides is not None
            else tuple(1 for _ in self.shape)
        )
        self.pixel_box_lower_corner = (
            tuple(int(corner) for corner in pixel_box_lower_corner)
            if pixel_box_lower_corner is not None
            else None
        )
        self.pixel_box_upper_corner = (
            tuple(int(corner) for corner in pixel_box_upper_corner)
            if pixel_box_upper_corner is not None
            else None
        )
        self.origin = (
            tuple(int(offset) for offset in origin)
            if origin is not None
            else tuple(0 for _ in self.shape)
        )
        self.dtype = _dtype_from_base(base)
        self.block_type = SimulatedBlockType(self.dtype, list(self.block_shape))
        self.type = SimulatedTensorDescriptorType(self.block_type)
        self.nbytes_per_cta = self.block_type.nbytes
        self.mode = "im2col" if self.pixel_box_lower_corner is not None else "tiled"

    @property
    def block_shape(self):
        return self.__dict__["_block_shape"]

    @block_shape.setter
    def block_shape(self, value: Any) -> None:
        self.__dict__["_block_shape"] = tuple(int(dim) for dim in value)

    @property
    def layout(self):
        return self.__dict__["_layout"]

    @layout.setter
    def layout(self, value: Any) -> None:
        self.__dict__["_layout"] = value

    @property
    def dtype(self):
        return self.__dict__["_dtype"]

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.__dict__["_dtype"] = value

    @property
    def block_type(self):
        return self.__dict__["_block_type"]

    @block_type.setter
    def block_type(self, value: Any) -> None:
        self.__dict__["_block_type"] = value

    @property
    def nbytes_per_cta(self) -> int:
        return self.__dict__["_nbytes_per_cta"]

    @nbytes_per_cta.setter
    def nbytes_per_cta(self, value: Any) -> None:
        self.__dict__["_nbytes_per_cta"] = int(value)

    @property
    def _array(self) -> np.ndarray:
        return _numpy_array(self.base)

    @property
    def data(self) -> np.ndarray:
        return self._array

    def data_ptr(self) -> int:
        data_ptr = getattr(self.base, "data_ptr", None)
        if callable(data_ptr):
            return int(data_ptr())
        return int(self.data.__array_interface__["data"][0])

    def load_block(self, coord: Any) -> np.ndarray:
        coord = _normalize_coord(coord, len(self.shape))
        result = np.zeros(self.block_shape, dtype=_get_np_dtype(self.dtype))
        src = self._array
        for block_idx in np.ndindex(self.block_shape):
            src_idx = tuple(
                self.origin[dim] + coord[dim] + block_idx[dim]
                for dim in range(len(coord))
            )
            if all(0 <= src_idx[dim] < self.shape[dim] for dim in range(len(coord))):
                result[block_idx] = src[src_idx]
        return result

    def store_block(self, coord: Any, values: np.ndarray) -> None:
        coord = _normalize_coord(coord, len(self.shape))
        dst = self._array
        for block_idx in np.ndindex(self.block_shape):
            dst_idx = tuple(
                self.origin[dim] + coord[dim] + block_idx[dim]
                for dim in range(len(coord))
            )
            if all(0 <= dst_idx[dim] < self.shape[dim] for dim in range(len(coord))):
                dst[dst_idx] = values[block_idx]

    def reduce_block(self, coord: Any, values: np.ndarray, kind: str) -> None:
        coord = _normalize_coord(coord, len(self.shape))
        dst = self._array
        for block_idx in np.ndindex(self.block_shape):
            dst_idx = tuple(
                self.origin[dim] + coord[dim] + block_idx[dim]
                for dim in range(len(coord))
            )
            if not all(
                0 <= dst_idx[dim] < self.shape[dim] for dim in range(len(coord))
            ):
                continue
            value = values[block_idx]
            if kind == "add":
                dst[dst_idx] += value
            elif kind == "min":
                dst[dst_idx] = np.minimum(dst[dst_idx], value)
            elif kind == "max":
                dst[dst_idx] = np.maximum(dst[dst_idx], value)
            elif kind == "and":
                dst[dst_idx] = np.bitwise_and(dst[dst_idx], value)
            elif kind == "or":
                dst[dst_idx] = np.bitwise_or(dst[dst_idx], value)
            elif kind == "xor":
                dst[dst_idx] = np.bitwise_xor(dst[dst_idx], value)
            else:
                raise ValueError(f"Unsupported TMA atomic reduction: {kind}")

    def load_im2col_block(self, coord: Any, offsets: Any) -> np.ndarray:
        if len(self.shape) != 4:
            raise NotImplementedError(
                "Simulated TMA im2col currently supports rank-4 NHWC"
            )
        if self.pixel_box_lower_corner is None or self.pixel_box_upper_corner is None:
            raise TypeError("TMA im2col descriptor requires pixel box corners")
        coord = _normalize_coord(coord, len(self.shape))
        offsets = _normalize_coord(offsets, len(self.shape) - 2)
        result = np.zeros(self.block_shape, dtype=_get_np_dtype(self.dtype))
        if self.padding == "nan":
            result.fill(np.nan)
        src = self._array
        n_dim, h_dim, w_dim, c_dim = self.shape
        lower_h = self.pixel_box_lower_corner[0] + offsets[0]
        lower_w = self.pixel_box_lower_corner[1] + offsets[1]
        upper_h = h_dim - 1 + self.pixel_box_upper_corner[0] + offsets[0]
        upper_w = w_dim - 1 + self.pixel_box_upper_corner[1] + offsets[1]
        box_h = upper_h - lower_h + 1
        box_w = upper_w - lower_w + 1
        start_h = self.origin[1] + coord[1] + offsets[0]
        start_w = self.origin[2] + coord[2] + offsets[1]
        start_linear = (start_h - lower_h) * box_w + (start_w - lower_w)
        start_batch = self.origin[0] + coord[0]
        coord_c = self.origin[3] + coord[3]
        stride_c = self.element_strides[3]
        for pixel in range(self.block_shape[0]):
            linear = start_linear + pixel
            batch = start_batch + linear // (box_h * box_w)
            within_batch = linear % (box_h * box_w)
            h = lower_h + within_batch // box_w
            w = lower_w + within_batch % box_w
            for channel in range(self.block_shape[1]):
                c = coord_c + channel * stride_c
                if (
                    0 <= batch < n_dim
                    and 0 <= h < h_dim
                    and 0 <= w < w_dim
                    and 0 <= c < c_dim
                ):
                    result[pixel, channel] = src[batch, h, w, c]
        return result


class SimulatedCLCResult:
    def is_canceled(self, _semantic: Any = None):
        return _to_tensor(False)

    def program_id(self, dim: Any, _semantic: Any = None):
        del dim
        return _to_tensor(0)


def _normalize_coord(coord: Any, ndim: int) -> tuple[int, ...]:
    coord = _unwrap_constexpr(coord)
    if isinstance(coord, (list, tuple)):
        values = tuple(_to_int(item) for item in coord)
    else:
        values = (_to_int(coord),)
    if len(values) != ndim:
        raise ValueError(f"Expected {ndim} TMA coordinates, got {len(values)}")
    return values


def _descriptor_from_host(value: Any) -> SimulatedTensorDescriptor:
    if isinstance(value, SimulatedTensorDescriptor):
        return value
    if type(value).__name__ in _HOST_TENSOR_DESCRIPTOR_TYPES:
        return SimulatedTensorDescriptor(
            value.base,
            list(value.shape),
            list(value.strides),
            list(value.block_shape),
            value.layout,
            getattr(value, "padding", "zero"),
            getattr(value, "element_strides", None),
            getattr(value, "pixel_box_lower_corner", None),
            getattr(value, "pixel_box_upper_corner", None),
        )
    raise TypeError(f"Unsupported TMA descriptor type: {type(value)}")


def _implicit_gluon_cvt(name: str, value: Any, constexprs: set[str]) -> Any:
    if name in constexprs:
        return value
    if isinstance(value, SimulatedTensorDescriptor) or (
        type(value).__name__ in _HOST_TENSOR_DESCRIPTOR_TYPES
    ):
        return _descriptor_from_host(value)
    return _implicit_cvt(value)


class SharedMemoryHandle(TensorHandle, gluon_core.shared_memory_descriptor):
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        data: np.ndarray | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = tuple(int(dim) for dim in shape)
        self.layout = layout
        data = (
            np.zeros(self.shape, dtype=_get_np_dtype(element_ty))
            if data is None
            else data
        )
        TensorHandle.__init__(self, data, element_ty)
        self.type = SimulatedSharedMemoryDescriptorType(
            self.element_ty,
            list(self.shape),
            self.layout,
        )

    @property
    def shape(self):
        if "type" not in self.__dict__:
            return self.__dict__["_shape"]
        return self.type.shape

    @shape.setter
    def shape(self, value: Any) -> None:
        self.__dict__["_shape"] = tuple(int(dim) for dim in value)

    @property
    def dtype(self):
        return self.__dict__["dtype"]

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.__dict__["dtype"] = value
        self.element_ty = value

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def layout(self):
        if "type" not in self.__dict__:
            return self.__dict__["_layout"]
        return self.type.layout

    @layout.setter
    def layout(self, value: Any) -> None:
        self.__dict__["_layout"] = value

    @property
    def nbytes_per_cta(self) -> int:
        return self.type.nbytes_per_cta

    def data_ptr(self) -> int:
        return int(self.data.__array_interface__["data"][0])

    def load(self, layout: Any = None, _semantic: Any = None) -> tl.tensor:
        return tl.tensor(
            TensorHandle(self.data.copy(), self.element_ty),
            tl.block_type(self.element_ty, list(self.shape)),
        )

    def store(self, value: Any, _semantic: Any = None) -> None:
        value = gluon_semantic.to_tensor(value)
        self.data[...] = np.asarray(value.handle.data, dtype=self.data.dtype)
        return None

    def index(self, idx: Any) -> "SharedMemoryHandle":
        idx_value = _to_int(idx)
        return SharedMemoryHandle(
            self.element_ty,
            list(self.shape[1:]),
            self.layout,
            self.data[idx_value],
        )

    def slice(
        self,
        start: Any,
        length: Any,
        dim: Any = 0,
        _semantic: Any = None,
    ) -> "SharedMemoryHandle":
        start_value = _to_int(start)
        length_value = _to_int(length)
        dim_value = _to_int(dim)
        slices = [slice(None)] * len(self.shape)
        slices[dim_value] = slice(start_value, start_value + length_value)
        shape = list(self.shape)
        shape[dim_value] = length_value
        return SharedMemoryHandle(
            self.element_ty,
            shape,
            self.layout,
            self.data[tuple(slices)],
        )

    def reshape(
        self,
        shape: Any,
        _semantic: Any = None,
    ) -> "SharedMemoryHandle":
        shape = tuple(int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape))
        return SharedMemoryHandle(
            self.element_ty,
            list(shape),
            self.layout,
            self.data.reshape(shape),
        )

    def _reinterpret(
        self,
        shape: Any,
        layout: Any,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> "SharedMemoryHandle":
        del _semantic, _generator
        shape = tuple(int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape))
        return SharedMemoryHandle(
            self.element_ty,
            list(shape),
            _unwrap_constexpr(layout),
            self.data.reshape(shape),
        )

    def permute(self, order: Any, _semantic: Any = None) -> "SharedMemoryHandle":
        order = tuple(int(_unwrap_constexpr(item)) for item in _unwrap_constexpr(order))
        return SharedMemoryHandle(
            self.element_ty,
            [self.shape[dim] for dim in order],
            self.layout,
            np.transpose(self.data, order),
        )


class SimulatedTensorMemoryDescriptorType:
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        alloc_shape: list[int] | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]
        self.layout = layout
        self.alloc_shape = (
            [int(dim) for dim in alloc_shape] if alloc_shape is not None else self.shape
        )

    def get_reg_layout(self, num_warps: Any = None, instr_variant: str = "32x32b"):
        del num_warps, instr_variant
        return self.layout


class TensorMemoryHandle(TensorHandle, gluon_blackwell.tensor_memory_descriptor):
    def __init__(
        self,
        element_ty: tl.dtype,
        shape: list[int],
        layout: Any,
        data: np.ndarray | None = None,
    ) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]
        self.layout = layout
        data = (
            np.zeros(self.shape, dtype=_get_np_dtype(element_ty))
            if data is None
            else data
        )
        TensorHandle.__init__(self, data, element_ty)
        self.type = SimulatedTensorMemoryDescriptorType(
            self.element_ty,
            self.shape,
            self.layout,
        )

    @property
    def shape(self):
        return self.__dict__["_shape"]

    @shape.setter
    def shape(self, value: Any) -> None:
        self.__dict__["_shape"] = [int(dim) for dim in value]

    @property
    def layout(self):
        return self.__dict__["_layout"]

    @layout.setter
    def layout(self, value: Any) -> None:
        self.__dict__["_layout"] = value

    @property
    def dtype(self):
        return self.__dict__["dtype"]

    @dtype.setter
    def dtype(self, value: Any) -> None:
        self.__dict__["dtype"] = value
        self.element_ty = value

    @property
    def rank(self) -> int:
        return len(self.shape)

    def data_ptr(self) -> int:
        return int(self.data.__array_interface__["data"][0])

    def get_reg_layout(
        self,
        num_warps: Any = None,
        instr_variant: str = "32x32b",
        _semantic: Any = None,
        _generator: Any = None,
    ):
        del _semantic, _generator
        return self.type.get_reg_layout(num_warps, instr_variant)

    def load(
        self,
        layout: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tl.tensor:
        del layout, _semantic, _generator
        return tl.tensor(
            TensorHandle(self.data.copy(), self.element_ty),
            tl.block_type(self.element_ty, list(self.shape)),
        )

    def _load_reduction(
        self,
        np_func: Callable,
        layout: Any = None,
        abs: Any = False,
        propagate_nan: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tuple[tl.tensor, tl.tensor]:
        del propagate_nan
        values = self.load(layout, _semantic, _generator)
        data = np.asarray(self.data)
        if bool(_unwrap_constexpr(abs)):
            data = np.abs(data)
        reduced_data = np_func(data, axis=1)
        reduced_data = np.asarray(reduced_data, dtype=data.dtype)
        reduced = tl.tensor(
            TensorHandle(reduced_data, self.element_ty),
            tl.block_type(self.element_ty, list(reduced_data.shape)),
        )
        return values, reduced

    def load_min(
        self,
        layout: Any = None,
        abs: Any = False,
        propagate_nan: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tuple[tl.tensor, tl.tensor]:
        return self._load_reduction(
            np.min,
            layout,
            abs,
            propagate_nan,
            _semantic,
            _generator,
        )

    def load_max(
        self,
        layout: Any = None,
        abs: Any = False,
        propagate_nan: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tuple[tl.tensor, tl.tensor]:
        return self._load_reduction(
            np.max,
            layout,
            abs,
            propagate_nan,
            _semantic,
            _generator,
        )

    def store(self, value: Any, pred: Any = True, _semantic: Any = None) -> None:
        if not bool(_to_python_scalar(pred)):
            return None
        value = gluon_semantic.to_tensor(value)
        self.data[...] = np.asarray(value.handle.data, dtype=self.data.dtype)
        return None

    def index(self, idx: Any, _semantic: Any = None) -> "TensorMemoryHandle":
        idx_value = _to_int(idx)
        return TensorMemoryHandle(
            self.element_ty,
            self.shape[1:],
            self.layout,
            self.data[idx_value],
        )

    def slice(
        self,
        start: Any,
        length: Any,
        _semantic: Any = None,
    ) -> "TensorMemoryHandle":
        start_value = _to_int(start)
        length_value = _to_int(length)
        return TensorMemoryHandle(
            self.element_ty,
            [self.shape[0], length_value],
            self.layout,
            self.data[:, start_value : start_value + length_value],
        )

    def reshape(
        self,
        shape: Any,
        _semantic: Any = None,
    ) -> "TensorMemoryHandle":
        shape = [int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape)]
        return TensorMemoryHandle(
            self.element_ty,
            shape,
            self.layout,
            self.data.reshape(shape),
        )


def _allocate_shared_memory(
    element_ty: Any,
    shape: Any,
    layout: Any,
    value: Any = None,
    _semantic: Any = None,
):
    element_ty = _unwrap_constexpr(element_ty)
    shape = [int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape)]
    desc = SharedMemoryHandle(element_ty, shape, _unwrap_constexpr(layout))
    if value is not None:
        desc.store(value)
    return desc


def _allocate_tensor_memory(
    element_ty: Any,
    shape: Any,
    layout: Any,
    value: Any = None,
    _semantic: Any = None,
):
    element_ty = _unwrap_constexpr(element_ty)
    shape = [int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(shape)]
    desc = TensorMemoryHandle(element_ty, shape, _unwrap_constexpr(layout))
    if value is not None:
        desc.store(value)
    return desc


def _allocate_mbarrier(
    batch: Any = None,
    two_ctas: Any = False,
    _semantic: Any = None,
) -> SharedMemoryHandle:
    num_ctas = 1 if _CURRENT_NUM_CTAS is None else int(_CURRENT_NUM_CTAS)
    two_ctas = bool(_unwrap_constexpr(two_ctas))
    barrier_count = num_ctas // 2 if two_ctas else num_ctas
    barrier_count = max(1, barrier_count)
    if batch is None:
        shape = [barrier_count]
    else:
        shape = [_to_int(batch), barrier_count]
    return _allocate_shared_memory(gl.int64, shape, None)


def _mbarrier_key(barrier: Any) -> int:
    if isinstance(barrier, SharedMemoryHandle):
        return int(barrier.data.__array_interface__["data"][0])
    return id(barrier)


def _mbarrier_state(barrier: Any) -> _MBarrierState:
    key = _mbarrier_key(barrier)
    state = _MBARRIER_STATES.get(key)
    if state is None:
        state = _MBarrierState()
        _MBARRIER_STATES[key] = state
    return state


def _mbarrier_signal(barrier: Any) -> None:
    state = _mbarrier_state(barrier)
    with state.condition:
        state.pending_bytes = 0
        state.phase ^= 1
        state.ready = True
        state.condition.notify_all()


def _mbarrier_complete_bytes(barrier: Any, nbytes: int) -> None:
    state = _mbarrier_state(barrier)
    with state.condition:
        if state.pending_bytes > 0:
            state.pending_bytes -= int(nbytes)
            if state.pending_bytes > 0:
                return
        state.pending_bytes = 0
    _mbarrier_signal(barrier)


def _mbarrier_init(barrier: Any, *args: Any, **kwargs: Any) -> None:
    del args, kwargs
    state = _mbarrier_state(barrier)
    with state.condition:
        state.pending_bytes = 0
        state.phase = 1
        state.ready = True
        state.condition.notify_all()


def _mbarrier_expect(barrier: Any, *args: Any, **kwargs: Any) -> None:
    expected = args[0] if args else kwargs.get("bytes", kwargs.get("tx_count", 0))
    del args, kwargs
    key = _mbarrier_key(barrier)
    state = _mbarrier_state(barrier)
    with state.condition:
        state.pending_bytes = max(0, _to_int(expected))
        state.ready = False
    if key in _PENDING_CLC_BARRIERS:
        _PENDING_CLC_BARRIERS.remove(key)
        _mbarrier_signal(barrier)


def _mbarrier_arrive(barrier: Any, *args: Any, **kwargs: Any) -> None:
    pred = kwargs.get("pred", True)
    del args, kwargs
    if bool(_to_python_scalar(pred)):
        _mbarrier_signal(barrier)


def _mbarrier_wait(barrier: Any, *args: Any, **kwargs: Any) -> None:
    from triton_viz.core.frontend import gluon as gluon_frontend

    pred = kwargs.get("pred", True)
    phase = kwargs.get("phase", args[0] if args else None)
    del args, kwargs
    if not bool(_to_python_scalar(pred)):
        return None
    phase_value = None if phase is None else (_to_int(phase) & 1)
    state = _mbarrier_state(barrier)
    while True:
        with state.condition:
            if state.ready and (phase_value is None or state.phase == phase_value):
                return None
            state.condition.wait(timeout=0.001)
        gluon_frontend._maybe_yield_warp_specialize()


def _mbarrier_invalidate(barrier: Any, *args: Any, **kwargs: Any) -> None:
    del args, kwargs
    _mbarrier_signal(barrier)


def _async_load(
    dst: SharedMemoryHandle,
    src: Any,
    *args: Any,
    mask: Any = None,
    pred: Any = None,
    **kwargs: Any,
) -> None:
    if mask is None:
        mask = pred
    if mask is None and args:
        mask = args[0]
    other = kwargs.get("other")
    value = (
        gl.load(src, mask=mask, other=other)
        if other is not None
        else gl.load(src, mask=mask)
    )
    dst.store(value)
    return None


def _buffer_load_to_shared(
    dst: SharedMemoryHandle,
    ptr: Any,
    offsets: Any,
    mask: Any = None,
    other: Any = None,
    cache_modifier: Any = "",
    _semantic: Any = None,
) -> None:
    del cache_modifier, _semantic
    value = _buffer_load(ptr, offsets, mask=mask, other=other)
    dst.store(value)
    return None


def _load_shared_relaxed(
    smem: SharedMemoryHandle,
    layout: Any,
    _semantic: Any = None,
):
    del _semantic
    return smem.load(_unwrap_constexpr(layout))


def _async_store(
    dst: Any,
    src: SharedMemoryHandle,
    mask: Any = None,
    cache_modifier: Any = "",
    _semantic: Any = None,
) -> None:
    del cache_modifier, _semantic
    _yield_warp_specialize()
    gl.store(dst, src.load(), mask=mask)
    return None


def _async_mbarrier_arrive(mbarrier: Any, _semantic: Any = None) -> None:
    del _semantic
    _mbarrier_signal(mbarrier)
    return None


def _async_commit_group(_semantic: Any = None) -> None:
    return None


def _async_wait_group(num_outstanding: Any = 0, _semantic: Any = None) -> None:
    return None


def _buffer_load(
    ptr: Any,
    offsets: Any,
    mask: Any = None,
    other: Any = None,
    cache: Any = None,
    _semantic: Any = None,
):
    del cache, _semantic
    _yield_warp_specialize()
    address = ptr + gluon_semantic.to_tensor(offsets)
    if other is not None:
        return gl.load(address, mask=mask, other=other)
    return gl.load(address, mask=mask)


def _buffer_store(
    stored_value: Any,
    ptr: Any,
    offsets: Any,
    mask: Any = None,
    cache: Any = None,
    _semantic: Any = None,
) -> None:
    del cache, _semantic
    _yield_warp_specialize()
    address = ptr + gluon_semantic.to_tensor(offsets)
    gl.store(address, stored_value, mask=mask)
    return None


def _buffer_atomic(kind: str) -> Callable:
    def atomic(
        ptr: Any,
        offsets: Any,
        value: Any,
        mask: Any = None,
        sem: Any = None,
        scope: Any = None,
        _semantic: Any = None,
    ):
        del sem, scope, _semantic
        _yield_warp_specialize()
        address = ptr + gluon_semantic.to_tensor(offsets)
        old = gl.load(address, mask=mask)
        value_tensor = gluon_semantic.to_tensor(value)
        if kind == "add":
            new_value = old + value_tensor
        elif kind == "max":
            new_value = gl.maximum(old, value_tensor)
        elif kind == "min":
            new_value = gl.minimum(old, value_tensor)
        elif kind == "and":
            new_value = old & value_tensor
        elif kind == "or":
            new_value = old | value_tensor
        elif kind == "xor":
            new_value = old ^ value_tensor
        elif kind == "xchg":
            new_value = value_tensor
        else:
            raise ValueError(f"Unsupported AMD buffer atomic: {kind}")
        gl.store(address, new_value, mask=mask)
        return old

    atomic.__name__ = f"_buffer_atomic_{kind}"
    return atomic


_buffer_atomic_add = _buffer_atomic("add")
_buffer_atomic_max = _buffer_atomic("max")
_buffer_atomic_min = _buffer_atomic("min")
_buffer_atomic_and = _buffer_atomic("and")
_buffer_atomic_or = _buffer_atomic("or")
_buffer_atomic_xor = _buffer_atomic("xor")
_buffer_atomic_xchg = _buffer_atomic("xchg")


def _cluster_arrive(_semantic: Any = None) -> None:
    return None


def _cluster_wait(_semantic: Any = None) -> None:
    return None


def _make_tensor_descriptor(
    base: Any,
    shape: list[Any],
    strides: list[Any],
    block_shape: list[Any],
    layout: Any,
    padding_option: str = "zero",
    _semantic: Any = None,
) -> SimulatedTensorDescriptor:
    host_base = base
    if isinstance(base, tl.tensor):
        ptr = int(np.asarray(base.handle.data).reshape(-1)[0])
        host_base = _POINTER_TENSORS.get(ptr, base)
    return SimulatedTensorDescriptor(
        host_base,
        [_to_int(dim) for dim in shape],
        [_to_int(stride) for stride in strides],
        [_to_int(dim) for dim in _unwrap_constexpr(block_shape)],
        _unwrap_constexpr(layout),
        str(_unwrap_constexpr(padding_option)),
    )


def _tma_async_load(
    tensor_desc: Any,
    coord: Any,
    barrier: Any,
    result: SharedMemoryHandle,
    pred: Any = True,
    multicast: Any = False,
    _semantic: Any = None,
) -> None:
    if not bool(_to_python_scalar(pred)):
        return None
    desc = _descriptor_from_host(tensor_desc)
    result.data[...] = desc.load_block(coord)
    _mbarrier_complete_bytes(barrier, desc.block_type.nbytes)
    return None


def _tma_async_load_im2col(
    tensor_desc: Any,
    coord: Any,
    offsets: Any,
    barrier: Any,
    result: SharedMemoryHandle,
    pred: Any = True,
    multicast: Any = False,
    _semantic: Any = None,
) -> None:
    del multicast
    if not bool(_to_python_scalar(pred)):
        return None
    desc = _descriptor_from_host(tensor_desc)
    result.data[...] = desc.load_im2col_block(coord, offsets)
    _mbarrier_complete_bytes(barrier, desc.block_type.nbytes)
    return None


def _tma_async_store(
    tensor_desc: Any,
    coord: Any,
    src: SharedMemoryHandle,
    _semantic: Any = None,
) -> None:
    desc = _descriptor_from_host(tensor_desc)
    desc.store_block(coord, np.asarray(src.data, dtype=_get_np_dtype(desc.dtype)))
    return None


def _tma_async_atomic(kind: str) -> Callable:
    def atomic(
        tensor_desc: Any,
        coord: Any,
        src: SharedMemoryHandle,
        _semantic: Any = None,
    ) -> None:
        desc = _descriptor_from_host(tensor_desc)
        values = np.asarray(src.data, dtype=_get_np_dtype(desc.dtype))
        desc.reduce_block(coord, values, kind)
        return None

    atomic.__name__ = f"_tma_async_atomic_{kind}"
    return atomic


_tma_async_atomic_add = _tma_async_atomic("add")
_tma_async_atomic_min = _tma_async_atomic("min")
_tma_async_atomic_max = _tma_async_atomic("max")
_tma_async_atomic_and = _tma_async_atomic("and")
_tma_async_atomic_or = _tma_async_atomic("or")
_tma_async_atomic_xor = _tma_async_atomic("xor")


def _tensor_data(value: Any) -> np.ndarray:
    if isinstance(value, tl.tensor):
        return np.asarray(value.handle.data)
    if isinstance(value, TensorHandle):
        return np.asarray(value.data)
    return np.asarray(value)


def _descriptor_pointer_args(
    descriptor: Any, coord: Any
) -> tuple[TensorHandle, TensorHandle]:
    coord = coord if isinstance(coord, (list, tuple)) else (coord,)
    coord_values = tuple(_to_int(item) for item in coord)
    block_shape = tuple(int(dim) for dim in descriptor.block_shape)
    strides = tuple(int(stride) for stride in descriptor.strides)
    base = descriptor.base
    data_ptr = getattr(base, "data_ptr", None)
    base_ptr = int(data_ptr()) if callable(data_ptr) else 0
    element_size_fn = getattr(base, "element_size", None)
    element_size = int(element_size_fn()) if callable(element_size_fn) else 1
    offsets = np.zeros(block_shape, dtype=np.uint64)
    mask = np.ones(block_shape, dtype=bool)
    for block_idx in np.ndindex(block_shape):
        element_offset = 0
        for dim, idx in enumerate(block_idx):
            element_offset += (coord_values[dim] + idx) * strides[dim]
        offsets[block_idx] = base_ptr + element_offset * element_size
    return TensorHandle(offsets, tl.pointer_type(descriptor.dtype)), TensorHandle(
        mask,
        tl.int1,
    )


def _descriptor_load_adapter(descriptor: Any, coord: Any, *_args: Any, **_kwargs: Any):
    ptr_handle, mask_handle = _descriptor_pointer_args(descriptor, coord)
    return AdapterResult(ptr_handle, mask_handle, None)


def _descriptor_store_adapter(descriptor: Any, coord: Any, *_args: Any, **_kwargs: Any):
    ptr_handle, mask_handle = _descriptor_pointer_args(descriptor, coord)
    return AdapterResult(ptr_handle, mask_handle, None)


def _register_descriptor_callback_adapters() -> None:
    from triton_viz.core.frontend import gluon as gluon_frontend

    pointer_load_adapter = gluon_frontend.GLUON_ADAPTERS[gluon_frontend.Load]
    pointer_store_adapter = gluon_frontend.GLUON_ADAPTERS[gluon_frontend.Store]

    def load_adapter(ptr: Any, *args: Any, **kwargs: Any) -> AdapterResult:
        if isinstance(ptr, SimulatedTensorDescriptor):
            return _descriptor_load_adapter(ptr, *args, **kwargs)
        return pointer_load_adapter(ptr, *args, **kwargs)

    def store_adapter(ptr: Any, *args: Any, **kwargs: Any) -> AdapterResult:
        if isinstance(ptr, SimulatedTensorDescriptor):
            return _descriptor_store_adapter(ptr, *args, **kwargs)
        return pointer_store_adapter(ptr, *args, **kwargs)

    gluon_frontend.GLUON_ADAPTERS[gluon_frontend.Load] = load_adapter
    gluon_frontend.GLUON_ADAPTERS[gluon_frontend.Store] = store_adapter


_register_descriptor_callback_adapters()


def _symbolic_numpy_op_adapter(
    default_adapter: Callable[..., AdapterResult],
    numpy_op: Callable[..., Any],
) -> Callable[..., AdapterResult]:
    return lambda *args, **kwargs: AdapterResult(
        *default_adapter(*args, **kwargs).args, numpy_op
    )


_SYMBOLIC_NUMPY_OP_BY_ATTR: dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "add": np.add,
    "ceil": np.ceil,
    "cos": np.cos,
    "exp": np.exp,
    "exp2": np.exp2,
    "floor": np.floor,
    "greater_equal": np.greater_equal,
    "greater_than": np.greater,
    "less_equal": np.less_equal,
    "less_than": np.less,
    "log": np.log,
    "log2": np.log2,
    "mul": np.multiply,
    "sin": np.sin,
    "sqrt": np.sqrt,
    "sub": np.subtract,
    "where": np.where,
}


def _register_symbolic_adapter_for_attr(
    op: Callable,
    attr: str,
    default_adapter: Callable[..., AdapterResult],
) -> None:
    numpy_op = _SYMBOLIC_NUMPY_OP_BY_ATTR.get(attr)
    if numpy_op is not None:
        from triton_viz.core.frontend import gluon as gluon_frontend

        gluon_frontend.GLUON_CALLABLE_ADAPTERS[op] = _symbolic_numpy_op_adapter(
            default_adapter,
            numpy_op,
        )


def _descriptor_symbolic_load_adapter(
    coord_index: int, pred_index: int | None = None
) -> Callable[..., AdapterResult]:
    def adapter(*args: Any, **kwargs: Any) -> AdapterResult:
        coord = args[coord_index] if len(args) > coord_index else kwargs.get("coord")
        pred = kwargs.get("pred")
        if pred is None and pred_index is not None and len(args) > pred_index:
            pred = args[pred_index]
        return AdapterResult(TensorDescriptorAccess(args[0], coord, pred), None, None)

    return adapter


def _descriptor_symbolic_store_adapter(
    coord_index: int,
) -> Callable[..., AdapterResult]:
    def adapter(*args: Any, **kwargs: Any) -> AdapterResult:
        coord = args[coord_index] if len(args) > coord_index else kwargs.get("coord")
        return AdapterResult(TensorDescriptorAccess(args[0], coord, None), 0, None)

    return adapter


def _register_simulated_callable_adapter(
    op: Any,
    adapter: Callable[..., AdapterResult],
) -> None:
    from triton_viz.core.frontend import gluon as gluon_frontend

    gluon_frontend.GLUON_CALLABLE_ADAPTERS[op] = adapter
    gluon_frontend.GLUON_REPLAY_CALLABLES.add(op)


def _register_replay_callable(op: Any) -> None:
    from triton_viz.core.frontend import gluon as gluon_frontend

    gluon_frontend.GLUON_REPLAY_CALLABLES.add(op)


def _tma_async_gather(
    tensor_desc: Any,
    x_offsets: Any,
    y_offset: Any,
    barrier: Any = None,
    result: SharedMemoryHandle | None = None,
    pred: Any = True,
    multicast: Any = False,
    _semantic: Any = None,
) -> None:
    del multicast
    if result is None:
        raise ValueError("async_gather requires a result shared-memory descriptor")
    if not bool(_to_python_scalar(pred)):
        return None
    desc = _descriptor_from_host(tensor_desc)
    rows = _tensor_data(x_offsets).astype(np.int64).reshape(-1)
    col_offset = _to_int(y_offset)
    dst = result.data
    dst.fill(0)
    src = desc._array
    for out_row, src_row in enumerate(rows):
        for out_col in range(dst.shape[1]):
            src_col = col_offset + out_col
            if 0 <= src_row < desc.shape[0] and 0 <= src_col < desc.shape[1]:
                dst[out_row, out_col] = src[int(src_row), src_col]
    if barrier is not None:
        _mbarrier_complete_bytes(barrier, dst.shape[0] * desc.block_type.nbytes)
    return None


def _tma_async_scatter(
    tensor_desc: Any,
    x_offsets: Any,
    y_offset: Any,
    src: SharedMemoryHandle,
    _semantic: Any = None,
) -> None:
    desc = _descriptor_from_host(tensor_desc)
    rows = _tensor_data(x_offsets).astype(np.int64).reshape(-1)
    col_offset = _to_int(y_offset)
    if col_offset < 0 or np.any(rows < 0):
        raise ValueError("async_scatter offsets must be non-negative")
    dst = desc._array
    values = np.asarray(src.data, dtype=_get_np_dtype(desc.dtype))
    for in_row, dst_row in enumerate(rows):
        if dst_row >= desc.shape[0]:
            continue
        for in_col in range(values.shape[1]):
            dst_col = col_offset + in_col
            if 0 <= dst_col < desc.shape[1]:
                dst[int(dst_row), dst_col] = values[in_row, in_col]
    return None


def _tdm_make_tensor_descriptor(
    base: Any,
    shape: list[Any],
    strides: list[Any],
    block_shape: list[Any],
    layout: Any,
    _semantic: Any = None,
) -> SimulatedTensorDescriptor:
    return _make_tensor_descriptor(
        base, shape, strides, block_shape, layout, "zero", _semantic
    )


def _tdm_update_tensor_descriptor(
    desc: Any,
    add_offsets: list[Any] | None = None,
    set_bounds: list[Any] | None = None,
    dest: Any = None,
    pred: Any = None,
    barrier: Any = None,
    _semantic: Any = None,
) -> SimulatedTensorDescriptor:
    del dest, pred, barrier, _semantic
    desc = _descriptor_from_host(desc)
    origin = list(desc.origin)
    shape = list(desc.shape)
    if add_offsets is not None:
        offsets = _normalize_coord(add_offsets, len(desc.shape))
        origin = [origin[dim] + offsets[dim] for dim in range(len(origin))]
    if set_bounds is not None:
        bounds = _normalize_coord(set_bounds, len(desc.shape))
        shape = [origin[dim] + bounds[dim] for dim in range(len(origin))]
    return SimulatedTensorDescriptor(
        desc.base,
        shape,
        list(desc.strides),
        list(desc.block_shape),
        desc.layout,
        desc.padding,
        list(desc.element_strides),
        (
            list(desc.pixel_box_lower_corner)
            if desc.pixel_box_lower_corner is not None
            else None
        ),
        (
            list(desc.pixel_box_upper_corner)
            if desc.pixel_box_upper_corner is not None
            else None
        ),
        origin,
    )


def _tdm_async_load(
    src: Any,
    offsets: list[Any],
    dest: SharedMemoryHandle,
    pred: Any = True,
    mbarrier: Any = None,
    warp_used_hint: Any = None,
    cache_modifier: Any = "",
    _semantic: Any = None,
) -> None:
    del warp_used_hint, cache_modifier, _semantic
    _yield_warp_specialize()
    if not bool(_to_python_scalar(pred)):
        return None
    desc = _descriptor_from_host(src)
    dest.data[...] = desc.load_block(offsets)
    if mbarrier is not None:
        _mbarrier_signal(mbarrier)
    return None


def _tdm_async_store(
    dest: Any,
    offsets: list[Any],
    src: SharedMemoryHandle,
    mbarrier: Any = None,
    cache_modifier: Any = "",
    _semantic: Any = None,
) -> None:
    del cache_modifier, _semantic
    _yield_warp_specialize()
    desc = _descriptor_from_host(dest)
    desc.store_block(offsets, np.asarray(src.data, dtype=_get_np_dtype(desc.dtype)))
    if mbarrier is not None:
        _mbarrier_signal(mbarrier)
    return None


def _tdm_async_gather(
    desc: Any,
    src_row_indices: Any,
    src_col_offset: Any,
    dst: SharedMemoryHandle,
    pred: Any = True,
    mbarrier: Any = None,
    _semantic: Any = None,
) -> None:
    _yield_warp_specialize()
    return _tma_async_gather(
        desc, src_row_indices, src_col_offset, mbarrier, dst, pred, _semantic=_semantic
    )


def _tdm_async_scatter(
    desc: Any,
    dst_row_indices: Any,
    dst_col_offset: Any,
    src: SharedMemoryHandle,
    mbarrier: Any = None,
    _semantic: Any = None,
) -> None:
    _yield_warp_specialize()
    _tma_async_scatter(desc, dst_row_indices, dst_col_offset, src, _semantic=_semantic)
    if mbarrier is not None:
        _mbarrier_signal(mbarrier)
    return None


def _tdm_async_wait(num_outstanding: Any = 0, _semantic: Any = None) -> None:
    return None


def _tdm_prefetch(
    src: Any,
    offsets: list[Any],
    pred: Any = True,
    speculative: Any = False,
    _semantic: Any = None,
) -> None:
    return None


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class _WarpPipelineStage:
    def __init__(
        self, label: Any = None, *, priority: Any = None, **_kwargs: Any
    ) -> None:
        del label, priority

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc, tb
        return False if exc_type is not None else False


def _is_gluon_jit_function(value: Any) -> bool:
    return (
        callable(value)
        and callable(getattr(value, "fn", None))
        and (getattr(value, "arg_names", None) is not None)
    )


def _device_function_wrapper(jit_fn: Any) -> Callable:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        scope = gluon_patch_lang(jit_fn.fn)
        try:
            return jit_fn.fn(*args, **kwargs)
        finally:
            scope.restore()

    wrapped.__name__ = getattr(
        jit_fn,
        "__name__",
        getattr(jit_fn.fn, "__name__", "gluon_device_fn"),
    )
    return wrapped


def _aggregate_new_wrapper(aggregate_cls: type, original_new: Callable) -> Callable:
    fields = tuple(getattr(aggregate_cls, "__aggregate_fields__", ()))
    annotations = getattr(aggregate_cls, "__annotations__", {})
    if not fields:
        fields = tuple(annotations)

    def wrapped(
        this_cls,
        *args: Any,
        _semantic: Any = None,
        _generator: Any = None,
        **kwargs: Any,
    ):
        del _semantic, _generator
        converted_args = list(args)
        for index, value in enumerate(converted_args):
            if index >= len(fields):
                break
            field = fields[index]
            if annotations.get(field) == tl.tensor and not isinstance(value, tl.tensor):
                converted_args[index] = gluon_semantic.to_tensor(
                    _unwrap_constexpr(value)
                )
        converted_kwargs = dict(kwargs)
        for field, value in list(converted_kwargs.items()):
            if annotations.get(field) == tl.tensor and not isinstance(value, tl.tensor):
                converted_kwargs[field] = gluon_semantic.to_tensor(
                    _unwrap_constexpr(value)
                )
        return original_new(this_cls, *converted_args, **converted_kwargs)

    return wrapped


def _patch_aggregate_jit_methods(value: Any, scope: _LangPatchScope) -> bool:
    if not inspect.isclass(value):
        return False
    patched = False
    if getattr(value, "__triton_aggregate__", False):
        original_new = getattr(value, "__new__", None)
        if original_new is not None:
            scope.set_attr(
                value, "__new__", _aggregate_new_wrapper(value, original_new)
            )
            patched = True
    for name, member in inspect.getmembers(value):
        if _is_gluon_jit_function(member):
            scope.set_attr(value, name, _device_function_wrapper(member))
            patched = True
    return patched


def _patch_jit_function_globals(jit_fn: Any, scope: _LangPatchScope) -> None:
    fn = getattr(jit_fn, "fn", None)
    globals_dict = getattr(fn, "__globals__", {})
    for value in list(globals_dict.values()):
        _patch_aggregate_jit_methods(value, scope)


for _simulated_tma in (
    _make_tensor_descriptor,
    _tma_async_load,
    _tma_async_load_im2col,
    _tma_async_store,
    _tma_async_atomic_add,
    _tma_async_atomic_min,
    _tma_async_atomic_max,
    _tma_async_atomic_and,
    _tma_async_atomic_or,
    _tma_async_atomic_xor,
    _tma_async_gather,
    _tma_async_scatter,
    _noop,
):
    _simulated_tma.__triton_viz_simulated__ = True  # type: ignore[attr-defined]
    _register_replay_callable(_simulated_tma)

_register_simulated_callable_adapter(
    _tma_async_load,
    _descriptor_symbolic_load_adapter(coord_index=1, pred_index=4),
)
_register_simulated_callable_adapter(
    _tma_async_load_im2col,
    _descriptor_symbolic_load_adapter(coord_index=1, pred_index=5),
)
_simulated_descriptor_stores: tuple[Callable[..., Any], ...] = (
    _tma_async_store,
    _tma_async_atomic_add,
    _tma_async_atomic_min,
    _tma_async_atomic_max,
    _tma_async_atomic_and,
    _tma_async_atomic_or,
    _tma_async_atomic_xor,
)
for _simulated_descriptor_store in _simulated_descriptor_stores:
    _register_simulated_callable_adapter(
        _simulated_descriptor_store,
        _descriptor_symbolic_store_adapter(coord_index=1),
    )
_register_simulated_callable_adapter(
    _tdm_async_load,
    _descriptor_symbolic_load_adapter(coord_index=1, pred_index=3),
)
_register_simulated_callable_adapter(
    _tdm_async_store,
    _descriptor_symbolic_store_adapter(coord_index=1),
)


def _make_gluon_wrapper(member: Callable) -> Callable:
    def wrapped(*args: Any, **kwargs: Any):
        _yield_warp_specialize()
        kwargs = {key: value for key, value in kwargs.items() if key != "_semantic"}
        return member(*args, **kwargs, _semantic=gluon_semantic)

    wrapped.__name__ = getattr(member, "__name__", "wrapped_gluon_builtin")
    wrapped.__doc__ = getattr(member, "__doc__", None)
    wrapped.__triton_viz_simulated__ = True  # type: ignore[attr-defined]
    _register_replay_callable(wrapped)
    return wrapped


def _yield_warp_specialize() -> None:
    from triton_viz.core.frontend import gluon as gluon_frontend

    gluon_frontend._maybe_yield_warp_specialize()


def _program_id(axis: Any, _semantic: Any = None):
    return gluon_semantic.program_id(int(_unwrap_constexpr(axis)))


def _arange(start: Any, end: Any, layout: Any = None, _semantic: Any = None):
    return gluon_semantic.arange(
        int(_unwrap_constexpr(start)), int(_unwrap_constexpr(end))
    )


def _full(
    shape: Any,
    value: Any,
    dtype: tl.dtype,
    layout: Any = None,
    _semantic: Any = None,
):
    shape = tuple(int(_unwrap_constexpr(dim)) for dim in shape)
    return gluon_semantic.full(
        list(shape), _unwrap_constexpr(value), _unwrap_constexpr(dtype)
    )


def _full_like(
    input: Any,
    value: Any,
    shape: Any = None,
    dtype: tl.dtype | None = None,
    layout: Any = None,
    _semantic: Any = None,
):
    del layout, _semantic
    input_tensor = gluon_semantic.to_tensor(input)
    dtype = input_tensor.dtype if dtype is None else _unwrap_constexpr(dtype)
    target_shape = input_tensor.shape if shape is None else _unwrap_constexpr(shape)
    target_shape = tuple(int(_unwrap_constexpr(dim)) for dim in target_shape)
    value_data = np.asarray(_to_python_scalar(value), dtype=_get_np_dtype(dtype))
    if value_data.shape:
        data = np.broadcast_to(value_data, target_shape).astype(
            _get_np_dtype(dtype), copy=True
        )
    else:
        data = np.full(target_shape, value_data.item(), dtype=_get_np_dtype(dtype))
    ret_ty = tl.block_type(dtype, list(data.shape)) if data.shape else dtype
    return tl.tensor(TensorHandle(data, dtype), ret_ty)


def _cdiv(x: Any, div: Any, _semantic: Any = None):
    x = _unwrap_constexpr(x)
    div = _unwrap_constexpr(div)
    return (x + div - 1) // div


def _to_tensor(x: Any, _semantic: Any = None):
    return gluon_semantic.to_tensor(_unwrap_constexpr(x))


def _num_programs(axis: Any, _semantic: Any = None):
    return gluon_semantic.num_programs(int(_unwrap_constexpr(axis)))


def _num_ctas(_semantic: Any = None):
    return 1 if _CURRENT_NUM_CTAS is None else _CURRENT_NUM_CTAS


def _barrier(*_args: Any, **_kwargs: Any) -> None:
    return None


def _static_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
    _semantic: Any = None,
) -> None:
    del _semantic
    values = tuple(_unwrap_constexpr(value) for value in values)
    print(*values, sep=sep, end=end, file=file, flush=flush)


def _set_auto_layout(value: Any, layout: Any, _semantic: Any = None):
    del layout
    return value


def _load(
    pointer: Any,
    mask: Any = None,
    other: Any = None,
    boundary_check: Any = (),
    padding_option: Any = "",
    cache_modifier: Any = "",
    eviction_policy: Any = "",
    volatile: Any = False,
    _semantic: Any = None,
):
    del _semantic
    _yield_warp_specialize()
    pointer = gluon_semantic.to_tensor(pointer)
    mask = _unwrap_constexpr(mask)
    other = _unwrap_constexpr(other)
    if mask is not None:
        mask = gluon_semantic.to_tensor(mask)
    if other is not None:
        other = gluon_semantic.to_tensor(other)
    return gluon_semantic.load(
        pointer,
        mask,
        other,
        _unwrap_constexpr(boundary_check),
        _unwrap_constexpr(padding_option),
        _unwrap_constexpr(cache_modifier),
        _unwrap_constexpr(eviction_policy),
        bool(_unwrap_constexpr(volatile)),
    )


def _store(
    pointer: Any,
    value: Any,
    mask: Any = None,
    boundary_check: Any = (),
    cache_modifier: Any = "",
    eviction_policy: Any = "",
    _semantic: Any = None,
):
    del _semantic
    _yield_warp_specialize()
    pointer = gluon_semantic.to_tensor(pointer)
    value = gluon_semantic.to_tensor(value)
    mask = _unwrap_constexpr(mask)
    if mask is not None:
        mask = gluon_semantic.to_tensor(mask)
    return gluon_semantic.store(
        pointer,
        value,
        mask,
        _unwrap_constexpr(boundary_check),
        _unwrap_constexpr(cache_modifier),
        _unwrap_constexpr(eviction_policy),
    )


def _split(a: Any, _semantic: Any = None, _generator: Any = None):
    del _semantic, _generator
    _yield_warp_specialize()
    return gluon_semantic.split(gluon_semantic.to_tensor(a))


def _shape_args(values: tuple[Any, ...]) -> list[int]:
    if len(values) == 1:
        first = _unwrap_constexpr(values[0])
        if isinstance(first, (tuple, list)):
            values = tuple(first)
    return [int(_unwrap_constexpr(value)) for value in values]


def _reshape(
    input: Any,
    *shape: Any,
    can_reorder: Any = False,
    _semantic: Any = None,
    _generator: Any = None,
):
    del _semantic, _generator
    _yield_warp_specialize()
    return gluon_semantic.reshape(
        gluon_semantic.to_tensor(input),
        _shape_args(shape),
        bool(_unwrap_constexpr(can_reorder)),
    )


def _permute(input: Any, *dims: Any, _semantic: Any = None):
    del _semantic
    _yield_warp_specialize()
    return gluon_semantic.permute(
        gluon_semantic.to_tensor(input),
        tuple(_shape_args(dims)),
    )


def _convert_layout(
    value: Any,
    layout: Any,
    assert_trivial: Any = False,
    _semantic: Any = None,
):
    del _semantic
    _yield_warp_specialize()
    return gluon_semantic.convert_layout(
        gluon_semantic.to_tensor(value),
        _unwrap_constexpr(layout),
        bool(_unwrap_constexpr(assert_trivial)),
    )


def _cast(
    input: Any,
    dtype: Any,
    fp_downcast_rounding: Any = None,
    bitcast: Any = False,
    _semantic: Any = None,
):
    del _semantic
    _yield_warp_specialize()
    input = gluon_semantic.to_tensor(input)
    dtype = _unwrap_constexpr(dtype)
    fp_downcast_rounding = _unwrap_constexpr(fp_downcast_rounding)
    if bool(_unwrap_constexpr(bitcast)):
        return gluon_semantic.bitcast(input, dtype)
    return gluon_semantic.cast(input, dtype, fp_downcast_rounding)


def _tensor_to(
    input: Any,
    dtype: Any,
    fp_downcast_rounding: Any = None,
    bitcast: Any = False,
    _semantic: Any = None,
):
    return _cast(input, dtype, fp_downcast_rounding, bitcast, _semantic=_semantic)


def _where(condition: Any, x: Any, y: Any, _semantic: Any = None):
    return gluon_semantic.where(
        gluon_semantic.to_tensor(condition),
        gluon_semantic.to_tensor(x),
        gluon_semantic.to_tensor(y),
    )


def _minimum(
    x: Any,
    y: Any,
    propagate_nan: Any = tl.PropagateNan.NONE,
    _semantic: Any = None,
):
    return gluon_semantic.minimum(
        gluon_semantic.to_tensor(x),
        gluon_semantic.to_tensor(y),
        _unwrap_constexpr(propagate_nan),
    )


def _maximum(
    x: Any,
    y: Any,
    propagate_nan: Any = tl.PropagateNan.NONE,
    _semantic: Any = None,
):
    return gluon_semantic.maximum(
        gluon_semantic.to_tensor(x),
        gluon_semantic.to_tensor(y),
        _unwrap_constexpr(propagate_nan),
    )


def _clamp(
    x: Any,
    min: Any,
    max: Any,
    propagate_nan: Any = tl.PropagateNan.NONE,
    _semantic: Any = None,
):
    return gluon_semantic.clamp(
        gluon_semantic.to_tensor(x),
        gluon_semantic.to_tensor(min),
        gluon_semantic.to_tensor(max),
        _unwrap_constexpr(propagate_nan),
    )


def _zeros(
    shape: Any,
    dtype: tl.dtype,
    layout: Any = None,
    _semantic: Any = None,
):
    return _full(shape, 0, dtype, layout)


def _reduction_tensor(
    value: Any, np_func: Callable, axis: Any = None, keep_dims: bool = False
):
    tensor = gluon_semantic.to_tensor(value)
    axis = _unwrap_constexpr(axis)
    keep_dims = bool(_unwrap_constexpr(keep_dims))
    data = np_func(tensor.handle.data, axis=axis, keepdims=keep_dims)
    data = np.asarray(data, dtype=tensor.handle.data.dtype)
    ret_ty = (
        tl.block_type(tensor.dtype, list(data.shape)) if data.shape else tensor.dtype
    )
    return tl.tensor(TensorHandle(data, tensor.dtype), ret_ty)


def _sum(value: Any, axis: Any = None, keep_dims: bool = False, _semantic: Any = None):
    return _reduction_tensor(value, np.sum, axis, keep_dims)


def _max(value: Any, axis: Any = None, keep_dims: bool = False, _semantic: Any = None):
    return _reduction_tensor(value, np.max, axis, keep_dims)


def _broadcast(input: Any, other: Any, _semantic: Any = None):
    input = gluon_semantic.to_tensor(input)
    other = gluon_semantic.to_tensor(other)
    lhs, rhs = gluon_semantic.broadcast_impl_value(input, other)
    return lhs, rhs


def _broadcast_to(input: Any, *shape: Any, _semantic: Any = None):
    del _semantic
    _yield_warp_specialize()
    return gluon_semantic.broadcast_impl_shape(
        gluon_semantic.to_tensor(input),
        _shape_args(shape),
    )


def _join(a: Any, b: Any, _semantic: Any = None):
    return gluon_semantic.join(gluon_semantic.to_tensor(a), gluon_semantic.to_tensor(b))


def _dot_fma(a: Any, b: Any, acc: Any, _semantic: Any = None):
    return gluon_semantic.dot_fma(
        gluon_semantic.to_tensor(a),
        gluon_semantic.to_tensor(b),
        gluon_semantic.to_tensor(acc),
    )


def _scalar_tensor(value: Any, dtype: tl.dtype):
    data = np.asarray(value, dtype=_get_np_dtype(dtype))
    return tl.tensor(TensorHandle(data, dtype), dtype)


def _tensor_from_data(data: Any, dtype: tl.dtype, keep_scalar: bool = False):
    data = np.asarray(data, dtype=_get_np_dtype(dtype))
    if data.shape or keep_scalar:
        ret_ty = tl.block_type(dtype, list(data.shape)) if data.shape else dtype
        return tl.tensor(TensorHandle(data, dtype), ret_ty)
    return _scalar_tensor(data.item(), dtype)


def _reduce(
    input: Any,
    axis: Any,
    combine_fn: Callable,
    keep_dims: Any = False,
    _semantic: Any = None,
    _generator: Any = None,
):
    del _semantic, _generator
    single_input = isinstance(input, tl.tensor)
    inputs = (input,) if single_input else tuple(input)
    tensors = tuple(gluon_semantic.to_tensor(item) for item in inputs)
    axis = _unwrap_constexpr(axis)
    keep_dims = bool(_unwrap_constexpr(keep_dims))
    data_inputs = [np.asarray(tensor.handle.data) for tensor in tensors]
    if axis is None:
        data_inputs = [data.reshape(-1) for data in data_inputs]
        axis = 0
        original_axis_none = True
    else:
        axis = int(_unwrap_constexpr(axis))
        if axis < 0:
            axis += len(data_inputs[0].shape)
        original_axis_none = False
    input_shape = data_inputs[0].shape
    output_shape = input_shape[:axis] + input_shape[axis + 1 :]
    outputs = [np.zeros(output_shape, dtype=data.dtype) for data in data_inputs]
    reducer = (
        _device_function_wrapper(combine_fn)
        if _is_gluon_jit_function(combine_fn)
        else combine_fn
    )

    for flat_index in range(data_inputs[0].size):
        input_index = np.unravel_index(flat_index, input_shape)
        output_index = input_index[:axis] + input_index[axis + 1 :]
        values = tuple(
            _scalar_tensor(data[input_index], tensors[index].dtype)
            for index, data in enumerate(data_inputs)
        )
        if input_index[axis] == 0:
            for out_index, value in enumerate(values):
                outputs[out_index][output_index] = value.handle.data.item()
            continue
        acc = tuple(
            _scalar_tensor(output[output_index], tensors[index].dtype)
            for index, output in enumerate(outputs)
        )
        reduced = reducer(*acc, *values)
        reduced_values = reduced if isinstance(reduced, tuple) else (reduced,)
        for out_index, value in enumerate(reduced_values):
            if isinstance(value, tl.tensor):
                outputs[out_index][output_index] = value.handle.data.item()
            else:
                outputs[out_index][output_index] = value

    result = []
    for index, output in enumerate(outputs):
        if keep_dims:
            if original_axis_none:
                for _ in range(len(input_shape)):
                    output = np.expand_dims(output, 0)
            else:
                output = np.expand_dims(output, axis)
        elif original_axis_none:
            output = output.item()
        result.append(_tensor_from_data(output, tensors[index].dtype))
    return result[0] if single_input else tuple(result)


def _map_elementwise(
    scalar_fn: Callable,
    *args: Any,
    pack: Any = 1,
    _semantic: Any = None,
    _generator: Any = None,
):
    del pack, _semantic, _generator
    arrays = []
    tensor_dtypes = []
    has_tensor = False
    for arg in args:
        if isinstance(arg, tl.tensor):
            arrays.append(np.asarray(arg.handle.data))
            tensor_dtypes.append(arg.dtype)
            has_tensor = True
        elif isinstance(arg, TensorHandle):
            arrays.append(np.asarray(arg.data))
            tensor_dtypes.append(arg.dtype)
            has_tensor = True
        else:
            arrays.append(np.asarray(_unwrap_constexpr(arg)))
    if not has_tensor:
        return scalar_fn(*args)

    broadcasted = np.broadcast_arrays(*arrays)
    out_shape = broadcasted[0].shape
    flat_results: list[list[Any]] = []
    first_result = None
    for idx in np.ndindex(out_shape):
        call_args = [array[idx].item() for array in broadcasted]
        result = scalar_fn(*call_args)
        if not isinstance(result, tuple):
            result = (result,)
        if first_result is None:
            first_result = result
            flat_results = [[] for _ in result]
        for result_idx, item in enumerate(result):
            flat_results[result_idx].append(_to_python_scalar(item))

    if first_result is None:
        return ()

    tensors = []
    for result_idx, values in enumerate(flat_results):
        first_item = first_result[result_idx]
        if isinstance(first_item, tl.tensor):
            dtype = first_item.dtype
        elif isinstance(first_item, TensorHandle):
            dtype = first_item.dtype
        else:
            dtype = tensor_dtypes[0]
        data = np.asarray(values, dtype=_get_np_dtype(dtype)).reshape(out_shape)
        tensors.append(
            tl.tensor(
                TensorHandle(data, dtype),
                tl.block_type(dtype, list(out_shape)),
            )
        )
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def _as_numpy_operand(value: Any) -> np.ndarray:
    if isinstance(value, SharedMemoryHandle):
        return value.data
    if isinstance(value, TensorMemoryHandle):
        return value.data
    if isinstance(value, tl.tensor):
        return np.asarray(value.handle.data)
    if isinstance(value, TensorHandle):
        return np.asarray(value.data)
    return np.asarray(value)
