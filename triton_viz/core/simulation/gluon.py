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
