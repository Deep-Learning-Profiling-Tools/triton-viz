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
from triton.experimental.gluon.language import _layouts as gluon_layouts  # type: ignore
from triton.experimental.gluon.language import _math as gluon_math  # type: ignore
from triton.experimental.gluon.language import _semantic as gluon_semantic_module  # type: ignore
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


class GluonSemantic(gluon_semantic_module.GluonSemantic):
    """Triton interpreter semantic with Gluon-compatible builtin signatures."""

    def convert_layout(self, value: Any, layout: Any, assert_trivial: bool = False):
        ty = value.type
        if not isinstance(ty, gluon_core.distributed_type):
            return value
        ret_ty = gluon_core.distributed_type(ty.element_ty, ty.shape, layout)
        handle = self.builder.create_convert_layout(ret_ty.to_ir(self.builder), value.handle)
        return gluon_core.tensor(handle, ret_ty)


gluon_semantic: GluonSemantic


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


class BlockType:
    def __init__(self, element_ty: tl.dtype, shape: list[int]) -> None:
        self.element_ty = element_ty
        self.shape = [int(dim) for dim in shape]

    @property
    def nbytes(self) -> int:
        bitwidth = getattr(self.element_ty, "primitive_bitwidth", 8)
        return int(np.prod(self.shape, dtype=np.int64)) * max(1, int(bitwidth) // 8)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockType)
            and self.element_ty == other.element_ty
            and self.shape == list(other.shape)
        )


class TensorDescriptorType:
    def __init__(self, block_type: BlockType, layout: Any = None) -> None:
        self.block_type = block_type
        self.layout = layout


class TensorDescriptorLayoutType:
    def __init__(self, block_type: tl.block_type, is_signed: bool, layout: Any) -> None:
        self.block_type = block_type
        self.is_signed = is_signed
        self.layout = layout


class SharedMemoryDescriptorType:
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


class TensorDescriptor(gluon_hopper_tma.tensor_descriptor):
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
        self.block_type = BlockType(self.dtype, list(self.block_shape))
        self.type = TensorDescriptorType(self.block_type)
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
                "TMA im2col currently supports rank-4 NHWC"
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


class CLCResult:
    def is_canceled(self, _semantic: Any = None):
        return _to_tensor(False)

    def program_id(self, dim: Any, _semantic: Any = None):
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


def _descriptor_from_host(value: Any) -> TensorDescriptor:
    if isinstance(value, TensorDescriptor):
        return value
    if type(value).__name__ in _HOST_TENSOR_DESCRIPTOR_TYPES:
        return TensorDescriptor(
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
    if isinstance(value, TensorDescriptor) or (
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
        self.type = SharedMemoryDescriptorType(
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


class TensorMemoryDescriptorType:
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
        self.type = TensorMemoryDescriptorType(
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
        return self.type.get_reg_layout(num_warps, instr_variant)

    def load(
        self,
        layout: Any = None,
        _semantic: Any = None,
        _generator: Any = None,
    ) -> tl.tensor:
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


def _as_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(_unwrap_constexpr(dim)) for dim in _unwrap_constexpr(value))


def _set_handle_layout(handle: TensorHandle, layout: Any) -> TensorHandle:
    handle.set_attr("gluon.layout", layout)
    return handle


def _get_handle_layout(handle: TensorHandle) -> Any:
    return handle.attr.get("gluon.layout", gluon_layouts.AutoLayout())


def _tensor_result(data: np.ndarray, dtype: tl.dtype, layout: Any = None) -> TensorHandle:
    return _set_handle_layout(TensorHandle(data, dtype), layout or gluon_layouts.AutoLayout())


def _local_indices(indices: np.ndarray, axis: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    for output_index in np.ndindex(indices.shape):
        memory_index = list(output_index)
        memory_index[axis] = int(indices[output_index])
        yield output_index, tuple(memory_index)


def _rmw_kind(rmw_op: Any) -> str:
    name = getattr(rmw_op, "name", str(rmw_op)).lower()
    if "fadd" in name or name.endswith("add") or ".add" in name:
        return "add"
    if "umin" in name or "min" in name:
        return "min"
    if "umax" in name or "max" in name:
        return "max"
    if "and" in name:
        return "and"
    if "or" in name:
        return "or"
    if "xor" in name:
        return "xor"
    if "xchg" in name:
        return "xchg"
    raise ValueError(f"Unsupported local atomic scatter RMW op: {rmw_op}")


def _apply_rmw(kind: str, old: Any, value: Any) -> Any:
    if kind == "add":
        return old + value
    if kind == "min":
        return np.minimum(old, value)
    if kind == "max":
        return np.maximum(old, value)
    if kind == "and":
        return np.bitwise_and(old, value)
    if kind == "or":
        return np.bitwise_or(old, value)
    if kind == "xor":
        return np.bitwise_xor(old, value)
    if kind == "xchg":
        return value
    raise ValueError(f"Unsupported local atomic scatter RMW kind: {kind}")


class Builder(interpreter_builder.__class__):
    """Interpreter builder with the Gluon-specific methods used by semantics."""

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self.options, "enable_iisan"):
            object.__setattr__(self.options, "enable_iisan", False)

    def get_auto_layout(self):
        return gluon_layouts.AutoLayout()

    def get_coalesced_layout(self):
        return gluon_layouts.CoalescedLayout()

    def get_blocked_layout(
        self,
        size_per_thread: Any,
        threads_per_warp: Any,
        warps_per_cta: Any,
        order: Any,
        cga_layout: Any = None,
    ):
        return gluon_layouts.BlockedLayout(
            list(size_per_thread),
            list(threads_per_warp),
            list(warps_per_cta),
            list(order),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_slice_layout(self, dim: int, parent: Any):
        return gluon_layouts.SliceLayout(int(dim), parent)

    def get_distributed_linear_layout(
        self,
        reg_bases: Any,
        lane_bases: Any,
        warp_bases: Any,
        block_bases: Any,
        shape: Any,
    ):
        return gluon_layouts.DistributedLinearLayout(
            list(reg_bases),
            list(lane_bases),
            list(warp_bases),
            list(block_bases),
            list(shape),
        )

    def get_dot_operand_layout(self, operand_index: int, parent: Any, k_width: int):
        return gluon_layouts.DotOperandLayout(int(operand_index), parent, int(k_width))

    def get_mma_layout(
        self,
        version: Any,
        warps_per_cta: Any,
        cga_layout: Any,
        instr_shape: Any,
    ):
        return gluon_layouts.NVMMADistributedLayout(
            list(version),
            list(warps_per_cta),
            list(instr_shape),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_nvmma_shared_layout(
        self,
        swizzle_byte_width: int,
        element_bitwidth: int,
        transposed: bool,
        fp4_padded: bool,
        cga_layout: Any,
        rank: int,
    ):
        return gluon_layouts.NVMMASharedLayout(
            int(swizzle_byte_width),
            int(element_bitwidth),
            int(rank),
            bool(transposed),
            bool(fp4_padded),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_swizzled_shared_layout(
        self,
        vec: int,
        per_phase: int,
        max_phase: int,
        order: Any,
        cga_layout: Any,
    ):
        return gluon_layouts.SwizzledSharedLayout(
            int(vec),
            int(per_phase),
            int(max_phase),
            list(order),
            [] if cga_layout is None else list(cga_layout),
        )

    def get_padded_shared_layout(
        self,
        intervals: Any,
        paddings: Any,
        offset_bases: Any,
        cga_layout: Any,
        shape: Any,
    ):
        return gluon_layouts.PaddedSharedLayout(
            [list(pair) for pair in zip(intervals, paddings)],
            list(offset_bases),
            [] if cga_layout is None else list(cga_layout),
            list(shape),
        )

    def get_shared_linear_layout(
        self,
        offset_bases: Any,
        block_bases: Any = None,
        alignment: int = 16,
    ):
        return gluon_layouts.SharedLinearLayout(
            list(offset_bases),
            [] if block_bases is None else list(block_bases),
            int(alignment),
        )

    def get_distributed_ty(self, element_ty: tl.dtype, shape: Any, layout: Any):
        return tl.block_type(element_ty, list(shape))

    def get_shared_mem_desc_ty(
        self,
        element_ty: tl.dtype,
        shape: Any,
        layout: Any,
        alloc_shape: Any,
    ):
        return SharedMemoryDescriptorType(
            element_ty,
            list(shape),
            layout,
            list(alloc_shape),
        )

    def get_tensor_mem_desc_ty(
        self,
        element_ty: tl.dtype,
        shape: Any,
        layout: Any,
        alloc_shape: Any,
    ):
        return TensorMemoryDescriptorType(
            element_ty,
            list(shape),
            layout,
            list(alloc_shape),
        )

    def get_tensor_descriptor_layout_type(
        self,
        block_type: tl.block_type,
        is_signed: bool,
        layout: Any,
    ):
        return TensorDescriptorLayoutType(block_type, bool(is_signed), layout)

    def get_tensor_descriptor_im2col_layout_type(
        self,
        block_type: tl.block_type,
        is_signed: bool,
        layout: Any,
    ):
        return TensorDescriptorLayoutType(block_type, bool(is_signed), layout)

    def get_gluon_layout_from_tensor(self, handle: TensorHandle):
        return _get_handle_layout(handle)

    def get_gluon_layout_from_memdesc(self, handle: Any):
        return getattr(handle, "layout", gluon_layouts.AutoLayout())

    def is_convert_layout_trivial(self, _ret_ty: Any, _value: Any) -> bool:
        return True

    def create_convert_layout(self, ret_ty: Any, value: TensorHandle):
        layout = getattr(ret_ty, "layout", gluon_layouts.AutoLayout())
        return _tensor_result(np.asarray(value.data).copy(), value.dtype, layout)

    def create_local_alloc(
        self,
        desc_ty: SharedMemoryDescriptorType,
        value: TensorHandle | None = None,
    ):
        data = np.zeros(desc_ty.alloc_shape, dtype=_get_np_dtype(desc_ty.element_ty))
        if value is not None:
            data[...] = np.asarray(value.data, dtype=data.dtype)
        return SharedMemoryHandle(desc_ty.element_ty, list(desc_ty.shape), desc_ty.layout, data)

    def create_local_load(self, _ret_ty: Any, mem_desc: SharedMemoryHandle):
        return _tensor_result(
            np.asarray(mem_desc.data).copy(),
            mem_desc.element_ty,
            mem_desc.layout,
        )

    def create_local_store(self, mem_desc: SharedMemoryHandle, value: TensorHandle):
        mem_desc.data[...] = np.asarray(value.data, dtype=mem_desc.data.dtype)
        return None

    def create_local_gather(
        self,
        _ret_ty: Any,
        mem_desc: SharedMemoryHandle,
        indices: TensorHandle,
        axis: int,
    ):
        indices_data = np.asarray(indices.data)
        result = np.empty(indices_data.shape, dtype=mem_desc.data.dtype)
        for output_index, memory_index in _local_indices(indices_data, int(axis)):
            result[output_index] = mem_desc.data[memory_index]
        return _tensor_result(result, mem_desc.element_ty, _get_handle_layout(indices))

    def create_local_scatter(
        self,
        mem_desc: SharedMemoryHandle,
        values: TensorHandle,
        indices: TensorHandle,
        axis: int,
    ):
        indices_data = np.asarray(indices.data)
        values_data = np.asarray(values.data)
        for output_index, memory_index in _local_indices(indices_data, int(axis)):
            mem_desc.data[memory_index] = values_data[output_index]
        return None

    def create_local_atomic_scatter_rmw(
        self,
        rmw_op: Any,
        mem_desc: SharedMemoryHandle,
        values: TensorHandle,
        indices: TensorHandle,
        mask: TensorHandle | None,
        axis: int,
    ):
        kind = _rmw_kind(rmw_op)
        indices_data = np.asarray(indices.data)
        values_data = np.asarray(values.data)
        mask_data = None if mask is None else np.asarray(mask.data, dtype=bool)
        old_values = np.empty(indices_data.shape, dtype=mem_desc.data.dtype)
        for output_index, memory_index in _local_indices(indices_data, int(axis)):
            old = mem_desc.data[memory_index]
            old_values[output_index] = old
            if mask_data is not None and not mask_data[output_index]:
                continue
            mem_desc.data[memory_index] = _apply_rmw(
                kind,
                old,
                values_data[output_index],
            )
        return _tensor_result(old_values, mem_desc.element_ty, _get_handle_layout(values))

    def create_local_dealloc(self, _mem_desc: SharedMemoryHandle):
        return None

    def create_memdesc_subslice(
        self,
        ret_ty: SharedMemoryDescriptorType,
        mem_desc: SharedMemoryHandle,
        offsets: Any,
    ):
        slices = []
        for dim, offset in enumerate(offsets):
            start = int(offset)
            stop = start + int(ret_ty.shape[dim])
            slices.append(slice(start, stop))
        return SharedMemoryHandle(
            ret_ty.element_ty,
            list(ret_ty.shape),
            ret_ty.layout,
            mem_desc.data[tuple(slices)],
        )

    def create_memdesc_index(
        self,
        ret_ty: SharedMemoryDescriptorType,
        mem_desc: SharedMemoryHandle,
        index: TensorHandle,
    ):
        index_value = int(np.asarray(index.data).reshape(-1)[0])
        return SharedMemoryHandle(
            ret_ty.element_ty,
            list(ret_ty.shape),
            ret_ty.layout,
            mem_desc.data[index_value],
        )

    def create_memdesc_trans(self, mem_desc: SharedMemoryHandle, order: Any):
        order = tuple(int(item) for item in order)
        return SharedMemoryHandle(
            mem_desc.element_ty,
            [mem_desc.shape[dim] for dim in order],
            mem_desc.layout,
            np.transpose(mem_desc.data, order),
        )

    def create_memdesc_reshape(self, mem_desc: SharedMemoryHandle, shape: Any):
        shape = list(_as_shape(shape))
        return SharedMemoryHandle(
            mem_desc.element_ty,
            shape,
            mem_desc.layout,
            mem_desc.data.reshape(shape),
        )

    def create_memdesc_reinterpret(
        self,
        ret_ty: SharedMemoryDescriptorType,
        mem_desc: SharedMemoryHandle,
    ):
        data = mem_desc.data.view(_get_np_dtype(ret_ty.element_ty)).reshape(ret_ty.shape)
        return SharedMemoryHandle(ret_ty.element_ty, list(ret_ty.shape), ret_ty.layout, data)

    def create_tmem_alloc(
        self,
        desc_ty: TensorMemoryDescriptorType,
        value: TensorHandle | None = None,
    ):
        data = np.zeros(desc_ty.alloc_shape, dtype=_get_np_dtype(desc_ty.element_ty))
        if value is not None:
            data[...] = np.asarray(value.data, dtype=data.dtype)
        return TensorMemoryHandle(desc_ty.element_ty, list(desc_ty.shape), desc_ty.layout, data)

    def create_tmem_load(self, _ret_ty: Any, desc: TensorMemoryHandle, *args: Any):
        if not args:
            return _tensor_result(np.asarray(desc.data).copy(), desc.element_ty, desc.layout)
        red_op = args[0]
        abs_flag = args[1] if len(args) > 1 else False
        data = np.asarray(desc.data)
        if bool(_to_python_scalar(abs_flag)):
            data = np.abs(data)
        name = getattr(red_op, "name", str(red_op)).lower()
        reduced_data = np.min(data, axis=1) if "min" in name else np.max(data, axis=1)
        return (
            _tensor_result(np.asarray(desc.data).copy(), desc.element_ty, desc.layout),
            _tensor_result(np.asarray(reduced_data, dtype=data.dtype), desc.element_ty, desc.layout),
            desc.layout,
        )

    def create_tmem_store(
        self,
        desc: TensorMemoryHandle,
        value: TensorHandle,
        pred: TensorHandle,
    ):
        if bool(np.asarray(pred.data).reshape(-1)[0]):
            desc.data[...] = np.asarray(value.data, dtype=desc.data.dtype)
        return None

    def create_tmem_subslice(
        self,
        ret_ty: TensorMemoryDescriptorType,
        desc: TensorMemoryHandle,
        start: Any,
    ):
        start_value = _to_int(start)
        return TensorMemoryHandle(
            ret_ty.element_ty,
            ret_ty.shape,
            ret_ty.layout,
            desc.data[:, start_value : start_value + ret_ty.shape[-1]],
        )

    def create_make_tensor_descriptor(
        self,
        desc_ty: TensorDescriptorLayoutType,
        base_handle: TensorHandle,
        shape_handles: list[TensorHandle],
        stride_handles: list[TensorHandle],
        padding: Any = "zero",
    ):
        shape = [_to_int(handle) for handle in shape_handles]
        strides = [_to_int(handle) for handle in stride_handles]
        block_shape = list(desc_ty.block_type.shape)
        return TensorDescriptor(
            base_handle,
            shape,
            strides,
            block_shape,
            desc_ty.layout,
            str(padding).split(".")[-1].lower(),
        )

    def create_async_tma_copy_global_to_local(
        self,
        tensor_desc: TensorDescriptor,
        coord: Any,
        barrier: Any,
        result: SharedMemoryHandle,
        pred: TensorHandle,
        *args: Any,
    ):
        if not bool(np.asarray(pred.data).reshape(-1)[0]):
            return None
        result.data[...] = tensor_desc.load_block(coord)
        return None

    def create_async_tma_copy_local_to_global(
        self,
        tensor_desc: TensorDescriptor,
        coord: Any,
        src: SharedMemoryHandle,
    ):
        tensor_desc.store_block(coord, np.asarray(src.data))
        return None

    def create_async_tma_reduce(
        self,
        kind: Any,
        tensor_desc: TensorDescriptor,
        coord: Any,
        src: SharedMemoryHandle,
    ):
        kind_name = str(kind).split(".")[-1].lower()
        tensor_desc.reduce_block(coord, np.asarray(src.data), kind_name)
        return None

    def create_clc_try_cancel(self, result: Any, barrier: Any):
        _PENDING_CLC_BARRIERS.add(_mbarrier_key(barrier))
        return None

    def create_clc_load_result(self, _src: Any):
        return CLCResult()

    def create_clc_is_canceled(self, result: CLCResult):
        return result.is_canceled().handle

    def create_clc_get_program_id(self, result: CLCResult, dim: Any):
        return result.program_id(dim).handle

    def get_int128_ty(self):
        return tl.int64

    def create_set_auto_layout(self, layout: Any, value: TensorHandle):
        return _tensor_result(np.asarray(value.data).copy(), value.dtype, layout)

    def create_histogram(
        self,
        data: TensorHandle,
        bins: int,
        mask: TensorHandle | None,
        _layout: Any = None,
    ):
        return super().create_histogram(data, bins, mask)

    def create_fp4_to_fp(self, src: TensorHandle, elem_type: tl.dtype, axis: int):
        return TensorHandle(
            _unpack_e2m1(src.data).astype(_get_np_dtype(elem_type)),
            elem_type,
        )

    def create_scaled_upcast_fp8(
        self,
        _ret_ty: Any,
        src: TensorHandle,
        scale: TensorHandle,
    ):
        return TensorHandle(src.data.astype(np.float32) * scale.data, tl.float32)

    def create_scaled_upcast_fp4(
        self,
        src: TensorHandle,
        scale: TensorHandle,
        elem_type: tl.dtype,
        axis: int,
    ):
        return TensorHandle(
            _unpack_e2m1(src.data).astype(_get_np_dtype(elem_type)) * scale.data,
            elem_type,
        )

    def _buffer_ptrs(self, ptr: TensorHandle, offsets: TensorHandle):
        return self.create_addptr(ptr, offsets)

    def create_buffer_load(
        self,
        _ret_ty: Any,
        ptr: TensorHandle,
        offsets: TensorHandle,
        mask: TensorHandle | None,
        other: TensorHandle | None,
        cache_modifier: Any = None,
    ):
        ptrs = self._buffer_ptrs(ptr, offsets)
        if mask is None:
            mask = TensorHandle(np.ones_like(ptrs.data, dtype=bool), tl.int1)
        return self.create_masked_load(ptrs, mask, other, None, None, False)

    def create_buffer_store(
        self,
        stored_value: TensorHandle,
        ptr: TensorHandle,
        offsets: TensorHandle,
        mask: TensorHandle | None,
        cache_modifier: Any = None,
    ):
        ptrs = self._buffer_ptrs(ptr, offsets)
        if mask is None:
            mask = TensorHandle(np.ones_like(ptrs.data, dtype=bool), tl.int1)
        return self.create_masked_store(ptrs, stored_value, mask, None, None)

    def create_buffer_atomic_rmw(
        self,
        rmw_op: Any,
        ptr: TensorHandle,
        offsets: TensorHandle,
        value: TensorHandle,
        sem: Any,
        scope: Any,
        mask: TensorHandle | None,
    ):
        ptrs = self._buffer_ptrs(ptr, offsets)
        if mask is None:
            mask = TensorHandle(np.ones_like(ptrs.data, dtype=bool), tl.int1)
        old = self.create_masked_load(ptrs, mask, None, None, None, False)
        new_data = _apply_rmw(_rmw_kind(rmw_op), old.data, value.data)
        new_value = TensorHandle(new_data.astype(old.data.dtype), old.dtype)
        self.create_masked_store(ptrs, new_value, mask, None, None)
        return old

    def create_buffer_load_to_local(
        self,
        dest: SharedMemoryHandle,
        ptr: TensorHandle,
        offsets: TensorHandle,
        mask: TensorHandle | None,
        other: TensorHandle | None,
        *args: Any,
    ):
        value = self.create_buffer_load(None, ptr, offsets, mask, other)
        dest.data[...] = np.asarray(value.data, dtype=dest.data.dtype)
        return None

    def create_async_copy_global_to_local(
        self,
        dest: SharedMemoryHandle,
        ptr: TensorHandle,
        mask: TensorHandle | None,
        other: TensorHandle | None = None,
        *args: Any,
    ):
        value = self.create_buffer_load(None, ptr, TensorHandle(np.array([0]), tl.int32), mask, other)
        dest.data[...] = np.asarray(value.data, dtype=dest.data.dtype)
        return None

    def create_async_copy_local_to_global(
        self,
        src: SharedMemoryHandle,
        ptr: TensorHandle,
        mask: TensorHandle | None,
        *args: Any,
    ):
        value = TensorHandle(np.asarray(src.data), src.element_ty)
        if mask is None:
            mask = TensorHandle(np.ones_like(value.data, dtype=bool), tl.int1)
        return self.create_masked_store(ptr, value, mask, None, None)

    def create_async_copy_mbarrier_arrive(self, mbarrier: Any, *args: Any):
        _mbarrier_signal(mbarrier)
        return None

    def create_async_copy_lds_barrier_arrive(self, mbarrier: Any, *args: Any):
        self.create_async_copy_mbarrier_arrive(mbarrier, *args)
        return None

    def create_async_commit_group(self):
        return None

    def create_async_wait_group(self, num_outstanding: Any = 0):
        return None

    def create_mbarrier_init(self, barrier: Any, *args: Any):
        _mbarrier_init(barrier, *args)
        return None

    def create_mbarrier_inval(self, barrier: Any):
        _mbarrier_invalidate(barrier)
        return None

    def create_mbarrier_expect(self, barrier: Any, *args: Any):
        _mbarrier_expect(barrier, *args)
        return None

    def create_mbarrier_arrive(self, barrier: Any, *args: Any):
        _mbarrier_arrive(barrier, *args)
        return None

    def create_mbarrier_wait(self, barrier: Any, phase: Any = None, pred: Any = True, *args: Any):
        _mbarrier_wait(barrier, phase, pred=pred)
        return None

    def create_lds_barrier_init(self, barrier: Any, *args: Any):
        return self.create_mbarrier_init(barrier, *args)

    def create_lds_barrier_wait(self, barrier: Any, phase: Any = None, *args: Any):
        return self.create_mbarrier_wait(barrier, phase, *args)

    def create_lds_barrier_arrive(self, barrier: Any, *args: Any):
        _mbarrier_arrive(barrier, *args)
        return TensorHandle(np.array([0], dtype=np.int32), tl.int32)

    def create_fence_mbarrier_init_release_cluster(self):
        return None

    def create_fence_async_shared(self, cluster: Any = False):
        return None

    def create_async_shared_store(
        self,
        dst: SharedMemoryHandle,
        value: TensorHandle,
        mbarrier: Any,
    ):
        dst.data[...] = np.asarray(value.data, dtype=dst.data.dtype)
        _mbarrier_signal(mbarrier)
        return None

    def create_cluster_arrive(self, *args: Any):
        return None

    def create_cluster_wait(self):
        return None

    def create_cluster_barrier(self, *args: Any):
        return None

    def create_amd_cluster_arrive(self):
        return None

    def create_amd_cluster_wait(self):
        return None

    def create_async_tdm_copy_global_to_local(
        self,
        src: TensorDescriptor,
        offsets: list[TensorHandle],
        dest: SharedMemoryHandle,
        pred: TensorHandle,
        *args: Any,
    ):
        if bool(np.asarray(pred.data).reshape(-1)[0]):
            coord = tuple(_to_int(offset) for offset in offsets)
            dest.data[...] = src.load_block(coord)
        return None

    def create_async_tdm_copy_local_to_global(
        self,
        dest: TensorDescriptor,
        offsets: list[TensorHandle],
        src: SharedMemoryHandle,
        mbarrier: Any = None,
        *args: Any,
    ):
        coord = tuple(_to_int(offset) for offset in offsets)
        dest.store_block(coord, np.asarray(src.data))
        if mbarrier is not None:
            _mbarrier_signal(mbarrier)
        return None

    def create_async_tdm_wait(self, num_outstanding: Any = 0):
        return None

    def create_async_tdm_scatter(
        self,
        desc: TensorDescriptor,
        row_indices: TensorHandle,
        col_offset: TensorHandle,
        src: SharedMemoryHandle,
        *args: Any,
    ):
        for row, row_index in enumerate(np.asarray(row_indices.data).reshape(-1)):
            desc.store_block((int(row_index), _to_int(col_offset)), src.data[row])
        return None

    def create_async_tdm_gather(
        self,
        desc: TensorDescriptor,
        row_indices: TensorHandle,
        col_offset: TensorHandle,
        dst: SharedMemoryHandle,
        *args: Any,
    ):
        for row, row_index in enumerate(np.asarray(row_indices.data).reshape(-1)):
            dst.data[row] = desc.load_block((int(row_index), _to_int(col_offset)))
        return None

    def create_tdm_prefetch(self, *args: Any):
        return CLCResult()

    def create_tmem_copy(self, src: SharedMemoryHandle, dst: TensorMemoryHandle):
        dst.data[...] = np.asarray(src.data, dtype=dst.data.dtype)
        return None

    def create_tcgen05_mma(self, *args: Any):
        return None

    def create_tcgen05_mma_scaled(self, *args: Any):
        return None

    def create_tcgen05_commit(self, barrier: Any, pred: TensorHandle, *args: Any):
        if bool(np.asarray(pred.data).reshape(-1)[0]):
            _mbarrier_signal(barrier)
        return None

    def create_warpgroup_mma(self, a: Any, b: Any, acc: TensorHandle, *args: Any):
        return acc

    def create_warpgroup_mma_wait(self, deps: list[TensorHandle], num_outstanding: Any):
        return deps

    def create_warp_pipeline_border(self, marker: Any, priority: Any):
        return None

    def create_warp_yield(self, values: list[Any]):
        return values

    def create_warp_return(self):
        return None


gluon_builder = Builder()
gluon_semantic = GluonSemantic(gluon_builder)


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
    state = _mbarrier_state(barrier)
    with state.condition:
        state.pending_bytes = 0
        state.phase = 1
        state.ready = True
        state.condition.notify_all()


def _mbarrier_expect(barrier: Any, *args: Any, **kwargs: Any) -> None:
    expected = args[0] if args else kwargs.get("bytes", kwargs.get("tx_count", 0))
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
    if bool(_to_python_scalar(pred)):
        _mbarrier_signal(barrier)


def _mbarrier_wait(barrier: Any, *args: Any, **kwargs: Any) -> None:
    from triton_viz.core.frontend import gluon as gluon_frontend

    pred = kwargs.get("pred", True)
    phase = kwargs.get("phase", args[0] if args else None)
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
    _mbarrier_signal(barrier)


_GLUON_BUILTIN_MODULES: tuple[Any, ...] = (
    gluon_core,
    gl,
    gluon_math,
    gluon_standard,
    gluon_amd,
    gluon_amd_cdna3,
    gluon_amd_cdna4,
    gluon_amd_cdna4_async_copy,
    gluon_amd_gfx1250,
    gluon_amd_async_copy,
    gluon_amd_cluster,
    gluon_amd_mbarrier,
    gluon_amd_tdm,
    gluon_amd_rdna3,
    gluon_amd_rdna4,
    gluon_ampere_async_copy,
    gluon_blackwell,
    gluon_blackwell_tma,
    gluon_clc,
    gluon_hopper,
    gluon_hopper_mbarrier,
    gluon_hopper_tma,
)


_GLUON_BUILTIN_CLASSES: tuple[Any, ...] = (
    gluon_core.shared_memory_descriptor,
    gluon_blackwell.tensor_memory_descriptor,
    gluon_clc.clc_result,
)


def _is_gluon_builtin(member: Any) -> bool:
    return callable(member) and bool(
        getattr(member, gluon_core.GLUON_BUILTIN, False)
        or tl.core.is_builtin(member)
    )


def _patch_gluon_attr(
    obj: Any,
    name: str,
    member: Callable,
    scope: _LangPatchScope,
) -> None:
    def new_member(*args: Any, member: Callable = member, **kwargs: Any):
        kwargs = {key: value for key, value in kwargs.items() if key != "_semantic"}
        return member(*args, **kwargs, _semantic=gluon_semantic)

    scope.set_attr(obj, name, new_member)


def _patch_gluon_builtins(pkg: Any, scope: _LangPatchScope) -> None:
    for name, member in inspect.getmembers(pkg):
        if _is_gluon_builtin(member):
            _patch_gluon_attr(pkg, name, member, scope)


def patch_lang(fn: Callable) -> _LangPatchScope:
    """Patch Gluon builtins to execute through the NumPy interpreter builder."""

    scope = _LangPatchScope()
    for module in _GLUON_BUILTIN_MODULES:
        _patch_gluon_builtins(module, scope)
    for cls in _GLUON_BUILTIN_CLASSES:
        _patch_gluon_builtins(cls, scope)
    _patch_lang_tensor(gluon_core.tensor, scope)
    _patch_lang_core(gluon_core, scope)
    return scope
