from __future__ import annotations

import inspect
import itertools

import numpy as np
from typing import Any, Callable, Protocol, Sequence, TypeAlias, cast
from triton_viz.core.client import ClientManager

from triton_viz.utils.quantize import STORAGE_DTYPES, quantize_float

import nki.isa as nisa
import nki.language as nl


ScalarLike: TypeAlias = bool | int | float | np.integer | np.floating | np.bool_
TensorInput: TypeAlias = "NDArray | np.ndarray | ScalarLike | Sequence[ScalarLike]"
BufferLike: TypeAlias = "Buffer | str"
ShapeLike: TypeAlias = Sequence[int] | tuple[int, ...]
PatternLike: TypeAlias = Sequence[Sequence[int | np.integer]]
DTypeLike: TypeAlias = (
    str | np.dtype | None
)  # e.g. np.float32, "bfloat16", None (default)
OpLike: TypeAlias = "NKIOp | Callable[..., object] | None"
UnaryOpLike: TypeAlias = "NKIOp | Callable[..., object]"
BinaryCallable: TypeAlias = Callable[[Any, Any], Any]
IndexKey: TypeAlias = "int | slice | np.ndarray | NDArray"


class PatchScope(Protocol):
    def set_attr(self, obj: object, name: str, value: object) -> None:
        ...

    def restore(self) -> None:
        ...


def _as_scalar(value: TensorInput) -> int:
    # unwrap NDArray scalar inputs before validating scalar shape
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError("Expected scalar value")
    return int(array.reshape(-1)[0])


def _shape_tuple(shape: ShapeLike) -> tuple[int, ...]:
    """Convert arbitrary shape sequences into canonical integer tuples."""
    return tuple(int(dim) for dim in shape)


def _normalize_pattern(pattern: PatternLike) -> list[tuple[int, int]]:
    if not pattern:
        raise ValueError("pattern must not be empty")
    normalized = []
    for pair in pattern:
        if len(pair) != 2:
            raise ValueError(f"pattern pair must have two values, got {pair}")
        step, count = pair
        normalized.append((int(step), int(count)))
    return normalized


def _compute_ap_indices(pattern: PatternLike, offset: TensorInput) -> np.ndarray:
    pattern_pairs = _normalize_pattern(pattern)
    shape = [count for _, count in pattern_pairs]
    indices = np.zeros(shape, dtype=np.int64)
    for axis, (step, count) in enumerate(pattern_pairs):
        axis_shape = [1] * len(pattern_pairs)
        axis_shape[axis] = count
        indices += step * np.arange(count, dtype=np.int64).reshape(axis_shape)
    return indices + _as_scalar(offset)


def _to_numpy(value: TensorInput) -> np.ndarray:
    return value.data if isinstance(value, NDArray) else np.asarray(value)


def _dtype_spec(dtype: DTypeLike) -> tuple[int, int, bool] | None:
    if isinstance(dtype, str):
        return STORAGE_DTYPES.get(dtype)
    dtype_name = getattr(dtype, "name", None)
    if isinstance(dtype_name, str):
        return STORAGE_DTYPES.get(dtype_name)
    return None


def _cast_dtype(value: TensorInput, dtype: DTypeLike) -> np.ndarray:
    # casts are only needed at the final write boundary to emulate low-precision storage semantics.
    value = _to_numpy(value)
    spec = _dtype_spec(dtype)
    if spec is None:
        return np.asarray(value, dtype=dtype)
    exp_bits, mant_bits, finite_only = spec
    casted = quantize_float(value, exp_bits, mant_bits, finite_only=finite_only)
    return np.asarray(casted, dtype=np.float64)


class NDArray:
    """Lightweight NumPy-backed tensor used by the NKI beta2 interpreter."""

    def __init__(
        self: NDArray,
        buffer: BufferLike | None = None,
        name: str | None = "",
        shape: ShapeLike | None = None,
        dtype: DTypeLike = None,
        value: TensorInput | None = None,
    ) -> None:
        self.buffer = buffer
        self.name = name
        self.dtype = dtype
        # normalize user shape into a concrete tuple for storage allocation
        storage_shape = _shape_tuple(shape) if shape is not None else None
        storage_dtype = (
            np.float64 if _dtype_spec(dtype) is not None or dtype is None else dtype
        )
        if value is None:
            assert storage_shape
            self.data = np.ndarray(storage_shape, dtype=storage_dtype)
        else:
            array = _cast_dtype(value, dtype)
            if storage_shape is not None and array.shape != storage_shape:
                array = array.reshape(storage_shape)
            self.data = array
        if self.dtype is None:
            self.dtype = self.data.dtype
        self._data_ptr: int | None = None

    @property
    def shape(self: NDArray) -> tuple[int, ...]:
        return self.data.shape

    @property
    def address(self: NDArray) -> int:
        if self._data_ptr is None:
            self._data_ptr = int(self.data.ctypes.data)
        return self._data_ptr

    @property
    def offset(self: NDArray) -> NDArray:
        """Return element offsets for this tensor."""
        return self.get_offsets()

    @property
    def pattern(self: NDArray) -> list[list[int]]:
        """Return the default linear access pattern for this tensor."""
        itemsize = max(self.element_size(), 1)
        steps = [stride // itemsize for stride in self.stride()]
        return [[step, dim] for step, dim in zip(steps, self.shape)]

    def data_ptr(self: NDArray) -> int:  # alias for self.address for triton-viz compat
        return self.address

    def stride(self: NDArray) -> tuple[int, ...]:
        return self.data.strides

    def element_size(self: NDArray) -> int:
        return getattr(self.dtype, "itemsize", self.data.dtype.itemsize)

    def cpu(self: NDArray) -> NDArray:
        return self

    def detach(self: NDArray) -> NDArray:
        return self

    def numpy(self: NDArray) -> np.ndarray:
        return self.data

    def get_offsets(self: NDArray) -> NDArray:
        """
        Generate offset arrays for each dimension based on shape and stride.
        Given array with shape (A, ..., Z) and strides (a, ..., z), return offsets:
        a * arange(A)[:, None, ..., None] + ... + z * arange(Z)[None, None, ..., :]
        """
        offsets = 0
        for dim_size, stride in zip(self.shape, self.stride()):
            offsets = np.expand_dims(offsets, -1) + np.arange(dim_size) * stride
        return NDArray(value=offsets, name=self.name)

    def __repr__(self: NDArray) -> str:
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def __getitem__(self: NDArray, keys: IndexKey | tuple[IndexKey, ...]) -> NDArray:
        """Implement slicing operations for NDArray"""
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        sliced_value = self.data[tuple(new_keys)]
        return NDArray(
            value=sliced_value, name=f"{self.name}_slice", buffer=self.buffer
        )

    def __setitem__(
        self: NDArray, keys: IndexKey | tuple[IndexKey, ...], value: TensorInput
    ) -> NDArray:
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        self.data[tuple(new_keys)] = _cast_dtype(value, self.dtype)
        return self

    def _binary_op(
        self: NDArray,
        other: TensorInput,
        op_func: BinaryCallable,
        op_name: str,
        op_symbol: str,
    ) -> NDArray:
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(self.data, other.data),
                name=f"{self.name}_{op_name}_{other.name}",
            )
        if np.isscalar(other):
            return NDArray(
                value=op_func(self.data, other), name=f"{self.name}_{op_name}_scalar"
            )
        return NDArray(
            value=op_func(self.data, _to_numpy(other)),
            name=f"{self.name}_{op_name}_tensor",
        )

    def _rbinary_op(
        self: NDArray,
        other: TensorInput,
        op_func: BinaryCallable,
        op_name: str,
        op_symbol: str,
    ) -> NDArray:
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(other.data, self.data),
                name=f"{other.name}_{op_name}_{self.name}",
            )
        if np.isscalar(other):
            return NDArray(
                value=op_func(other, self.data), name=f"scalar_{op_name}_{self.name}"
            )
        return NDArray(
            value=op_func(_to_numpy(other), self.data),
            name=f"tensor_{op_name}_{self.name}",
        )

    def reshape(self: NDArray, *args: object, **kwargs: object) -> NDArray:
        """Return a reshaped view backed by the same NumPy data when possible."""
        return NDArray(
            value=self.data.reshape(*args, **kwargs),
            name=f"{self.name}_reshape",
            buffer=self.buffer,
        )

    def broadcast_to(self: NDArray, *args: object, **kwargs: object) -> NDArray:
        """Return a broadcasted view of this tensor."""
        return NDArray(
            value=np.broadcast_to(self.data, *args, **kwargs),
            name=f"{self.name}_broadcast_to",
            buffer=self.buffer,
        )

    def ap(
        self: NDArray,
        *pattern: Sequence[int | np.integer] | ScalarLike,
        offset: TensorInput = 0,
        dtype: DTypeLike = None,
        scalar_offset: TensorInput | None = None,
        vector_offset: TensorInput | None = None,
        indirect_dim: int = 0,
        **kwargs: object,
    ) -> NDArray:
        """
        Materialize an address-pattern access from this tensor.

        Supports both `tensor.ap(pattern=[[...], ...], offset=...)` and
        `tensor.ap([step, count], [step, count], ...)` forms.
        """
        if "pattern" in kwargs:
            pattern = tuple(cast(PatternLike, kwargs.pop("pattern")))
        if kwargs:
            raise TypeError(f"Unsupported kwargs for ap: {tuple(kwargs)}")
        if len(pattern) == 2:
            first_pattern, second_pattern = pattern
            if np.isscalar(second_pattern) and not np.isscalar(first_pattern):
                pattern = (first_pattern,)
                offset = cast(TensorInput, second_pattern)
        if len(pattern) == 1 and isinstance(pattern[0], (list, tuple)) and pattern[0]:
            first = pattern[0][0]
            if isinstance(first, (list, tuple)):
                pattern = tuple(pattern[0])
        pattern_pairs = _normalize_pattern(cast(PatternLike, pattern))
        indices = _compute_ap_indices(pattern_pairs, offset)
        if scalar_offset is not None:
            indices += _as_scalar(scalar_offset)
        if vector_offset is not None:
            # unwrap NDArray vector offsets before computing indirect AP addressing
            vector_source = (
                vector_offset.data
                if isinstance(vector_offset, NDArray)
                else vector_offset
            )
            vector = np.asarray(vector_source, dtype=np.int64).reshape(-1)
            axis = int(indirect_dim)
            if indices.shape[axis] != vector.shape[0]:
                raise ValueError(
                    f"vector_offset length {vector.shape[0]} does not match AP dim {indices.shape[axis]}"
                )
            step = pattern_pairs[axis][0]
            vector_shape = [1] * indices.ndim
            vector_shape[axis] = vector.shape[0]
            static_axis = np.arange(indices.shape[axis], dtype=np.int64).reshape(
                vector_shape
            )
            indices -= step * static_axis
            indices += step * vector.reshape(vector_shape)
        base = self.data if dtype is None else self.data.view(dtype)
        values = base.reshape(-1)[indices]
        return NDArray(value=values, buffer=self.buffer, name=f"{self.name}_ap")


for method, op_name, symbol, op_func, reverse_operands in (
    ("__add__", "add", "+", np.add, False),
    ("__radd__", "add", "+", np.add, True),
    ("__sub__", "sub", "-", np.subtract, False),
    ("__rsub__", "sub", "-", np.subtract, True),
    ("__mul__", "mul", "*", np.multiply, False),
    ("__rmul__", "mul", "*", np.multiply, True),
    ("__truediv__", "div", "/", np.divide, False),
    ("__rtruediv__", "div", "/", np.divide, True),
    ("__lt__", "lt", "<", np.less, False),
    ("__gt__", "gt", ">", np.greater, False),
    ("__le__", "le", "<=", np.less_equal, False),
    ("__ge__", "ge", ">=", np.greater_equal, False),
    ("__and__", "and", "&", np.bitwise_and, False),
    ("__or__", "or", "|", np.bitwise_or, False),
):
    if reverse_operands:
        b_func = (
            lambda self,
            other,
            _op=op_func,
            _name=op_name,
            _symbol=symbol: self._rbinary_op(other, _op, _name, _symbol)
        )
    else:
        b_func = (
            lambda self,
            other,
            _op=op_func,
            _name=op_name,
            _symbol=symbol: self._binary_op(other, _op, _name, _symbol)
        )
    setattr(NDArray, method, b_func)


class Buffer:
    """NKI memory region abstraction used by `sbuf`, `hbm`, and `psum`."""

    def __init__(
        self: Buffer,
        buffer: str,
        size: ShapeLike | None = None,
        data: np.ndarray | None = None,
    ) -> None:
        self.buffer = buffer
        self.size = size
        self.data = data
        if size is not None and data is None:
            self.data = np.empty(shape=size, dtype=np.uint8)

    def ptr(
        self: Buffer, size: ShapeLike, offset: Sequence[int] | None = None
    ) -> Buffer:
        """Return a sub-buffer view with the requested shape and optional offset."""
        if offset is None:
            return Buffer(self.buffer, size)
        assert self.data, "Cannot slice a buffer pointer without backing data"
        coords = tuple(
            slice(off, off + dim_size) for off, dim_size in zip(offset, size)
        )
        return Buffer(self.buffer, size, data=self.data[coords])

    def view(self: Buffer, dtype: DTypeLike, size: ShapeLike) -> NDArray:
        """Materialize a typed NDArray view over this buffer region."""
        probe = NDArray(buffer=self.buffer, dtype=dtype, shape=_shape_tuple(size))
        if self.data is None:
            return probe
        if self.data.nbytes != probe.data.nbytes:
            raise ValueError(
                f"Buffer size mismatch: have {self.data.nbytes} bytes, expected {probe.data.nbytes}"
            )
        if self.data.dtype == np.uint8:
            data = self.data.view(probe.data.dtype).reshape(probe.data.shape)
        else:
            data = np.asarray(self.data, dtype=probe.data.dtype).reshape(
                probe.data.shape
            )
        return NDArray(buffer=self.buffer, dtype=dtype, value=data)


sbuf = Buffer("sbuf")
hbm = Buffer("hbm")
psum = Buffer("psum")


def _default_shared_hbm_name() -> str:
    frame = inspect.currentframe()
    caller = (
        frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    )
    if caller is None:
        return "shared_hbm"
    return f"{caller.f_code.co_filename}_{caller.f_code.co_name}_{caller.f_lineno}"


def _buffer_name(buffer: BufferLike | object | None) -> str:
    if isinstance(buffer, Buffer):
        return buffer.buffer
    if isinstance(buffer, str):
        return buffer
    return getattr(buffer, "name", "")


def _resolve_buffer(buffer: BufferLike | object | None) -> tuple[Buffer, bool]:
    if isinstance(buffer, Buffer):
        return buffer, False
    if buffer is None:
        return sbuf, False

    name = _buffer_name(buffer)
    aliases = {
        "shared_hbm": (hbm, True),
        "private_hbm": (hbm, False),
        "hbm": (hbm, False),
        "sbuf": (sbuf, False),
        "psum": (psum, False),
    }
    if name in aliases:
        return aliases[name]
    raise TypeError(f"Unsupported buffer type: {type(buffer).__name__}")


def ndarray(
    shape: ShapeLike,
    dtype: DTypeLike,
    *,
    buffer: BufferLike | None = None,
    name: str | None = None,
    **kwargs: object,
) -> NDArray:
    """Create an NDArray on a requested NKI buffer."""
    resolved_buffer, is_shared_hbm = _resolve_buffer(buffer)
    if is_shared_hbm:
        shared_name = name or _default_shared_hbm_name()
        cached = nki_builder.shared_hbm_arrays.get(shared_name)
        if cached is not None:
            # validate shared_hbm tensor reuses the same logical shape
            if cached.data.shape != tuple(shape):
                raise ValueError(f"shared_hbm shape mismatch for {shared_name}")
            if "value" in kwargs:
                cached.data[...] = _cast_dtype(kwargs["value"], cached.dtype)
            return cached
        created = NDArray(
            buffer=resolved_buffer.buffer,
            name=shared_name,
            shape=_shape_tuple(shape),
            dtype=dtype,
            value=kwargs.get("value"),
        )
        nki_builder.shared_hbm_arrays[shared_name] = created
        return created

    if "value" in kwargs:
        return NDArray(
            buffer=resolved_buffer.buffer,
            name=name,
            shape=_shape_tuple(shape),
            dtype=dtype,
            value=kwargs["value"],
        )
    ret = resolved_buffer.view(dtype, shape)
    ret.name = name
    return ret


def zeros(
    shape: ShapeLike,
    dtype: DTypeLike,
    *,
    buffer: BufferLike | None = None,
    name: str | None = None,
    **kwargs: object,
) -> NDArray:
    """Create a zero-initialized tensor."""
    ret = ndarray(shape, dtype, buffer=buffer, name=name, **kwargs)
    ret.data.fill(0)
    return ret


def _to_scalar(
    value: TensorInput, cast: Callable[[Any], int | float] = float
) -> int | float:
    array = _to_numpy(value)
    if array.size != 1:
        raise ValueError("Expected scalar value")
    return cast(array.reshape(-1)[0].item())


def _write_dst(dst: NDArray, value: TensorInput) -> NDArray:
    dst.data[...] = _cast_dtype(value, dst.dtype)
    return dst


def _op_key(op: OpLike) -> str:
    if op is None:
        return ""
    np_op = op.op if isinstance(op, NKIOp) else op
    return getattr(np_op, "__name__", str(np_op)).lower()


def _apply_binary(
    op: OpLike, lhs: TensorInput, rhs: TensorInput, reverse: bool = False
) -> TensorInput:
    lhs_value = _to_numpy(lhs)
    rhs_value = _to_numpy(rhs)
    left, right = (rhs_value, lhs_value) if reverse else (lhs_value, rhs_value)
    np_op = op.op if isinstance(op, NKIOp) else op
    if np_op is None:
        return lhs_value
    # if isinstance(op, str):
    #    op_name = op.lower().lstrip(".")
    #    op_map: dict[str, BinaryCallable] = {
    #        "add": np.add,
    #        "subtract": np.subtract,
    #        "sub": np.subtract,
    #        "multiply": np.multiply,
    #        "mul": np.multiply,
    #        "divide": np.divide,
    #        "maximum": np.maximum,
    #        "max": np.maximum,
    #        "minimum": np.minimum,
    #        "min": np.minimum,
    #        "bitwise_and": np.bitwise_and,
    #        "bitwise_or": np.bitwise_or,
    #        "bitwise_xor": np.bitwise_xor,
    #        "equal": np.equal,
    #        "not_equal": np.not_equal,
    #        "greater": np.greater,
    #        "greater_equal": np.greater_equal,
    #        "less": np.less,
    #        "less_equal": np.less_equal,
    #        "logical_and": np.logical_and,
    #        "logical_or": np.logical_or,
    #        "logical_xor": np.logical_xor,
    #        "power": np.power,
    #    }
    #    if op_name not in op_map:
    #        raise KeyError(f"Unsupported binary op: {op}")
    #    return _to_numpy(op_map[op_name](left, right))
    # try:
    #    return _to_numpy(op(left, right))
    # except TypeError:
    #    return _to_numpy(op(left, y=right))
    return np_op(left, right)


def _apply_unary(op: UnaryOpLike, value: TensorInput) -> TensorInput:
    array = _to_numpy(value)
    np_op = op.op if isinstance(op, NKIOp) else op
    # if isinstance(op, str):
    #    op_name = op.lower().lstrip(".")
    #    op_map: dict[str, Callable[[Any], Any]] = {
    #        "abs": np.abs,
    #        "invert": np.invert,
    #        "logical_not": np.logical_not,
    #        "reciprocal": np.reciprocal,
    #        "rsqrt": lambda x: 1 / np.sqrt(x),
    #        "gelu_apprx_sigmoid": lambda x: x / (1 + np.exp(-1.702 * x)),
    #    }
    #    if op_name not in op_map:
    #        raise KeyError(f"Unsupported unary op: {op}")
    #    return np.asarray(op_map[op_name](array))
    # try:
    #    return np.asarray(op(array))
    # except Exception:
    #    return np.asarray(op(NDArray(value=array)))
    return np_op(array)


def _apply_reduce(
    op: OpLike, value: TensorInput, axis: int | Sequence[int], keepdims: bool
) -> np.ndarray:
    # reducers = {
    #    "add": np.sum,
    #    "subtract": np.subtract.reduce,
    #    ".sub": np.subtract.reduce,
    #    "multiply": np.prod,
    #    "mul": np.prod,
    #    "maximum": np.max,
    #    ".max": np.max,
    #    "minimum": np.min,
    #    ".min": np.min,
    #    "bitwise_or": np.bitwise_or.reduce,
    #    "bitwise_and": np.bitwise_and.reduce,
    #    "bitwise_xor": np.bitwise_xor.reduce,
    # }
    value_array = _to_numpy(value)
    # key = _op_key(op).lstrip(".")
    np_op = op.op if isinstance(op, NKIOp) else op
    assert isinstance(np_op, np.ufunc)
    # if key == "":
    #    return np.sum(value_array, axis=axis, keepdims=keepdims)
    # if key in reducers:
    #    return np.asarray(reducers[key](value_array, axis=axis, keepdims=keepdims))
    # if callable(np_op):
    #    try:
    #        return np.asarray(np_op(value_array, axis=axis, keepdims=keepdims))
    #    except TypeError:
    #        reducer = getattr(np_op, "reduce", None)
    #        if callable(reducer):
    #            return np.asarray(reducer(value_array, axis=axis, keepdims=keepdims))
    # raise TypeError(f"Unsupported reduce op: {op!r}")
    return np.asarray(np_op.reduce(value_array, axis=axis, keepdims=keepdims))


def _free_axes(value: TensorInput) -> tuple[int, ...]:
    return tuple(range(1, _to_numpy(value).ndim))


def _range_args(
    start: TensorInput, stop: TensorInput | None = None, step: TensorInput = 1
) -> range:
    start_v = int(_to_scalar(start, int))
    if stop is None:
        return range(start_v)
    stop_v = int(_to_scalar(stop, int))
    step_v = int(_to_scalar(step, int))
    return range(start_v, stop_v, step_v)


def shared_constant(constant: TensorInput, dtype: DTypeLike = None) -> NDArray:
    """Create a tensor from trace-time constant data."""
    if dtype is None:
        raise ValueError("dtype must be specified")
    array = _to_numpy(constant)
    return NDArray(value=_cast_dtype(array, dtype), dtype=dtype, buffer=hbm.buffer)


def shared_identity_matrix(n: int | np.integer, dtype: DTypeLike = "uint8") -> NDArray:
    """Create a shared identity matrix tensor."""
    return shared_constant(np.eye(int(n)), dtype=dtype)


def affine_range(
    start: TensorInput, stop: TensorInput | None = None, step: TensorInput = 1
) -> range:
    """Create a parallel iterator range."""
    return _range_args(start, stop, step)


def ds(start: TensorInput, size: int | ShapeLike) -> slice:
    """Build a dynamic slice object."""
    start_v = _to_scalar(start, int)
    size_v = _to_scalar(size, int)
    return slice(start_v, start_v + size_v, None)


def sequential_range(
    start: TensorInput, stop: TensorInput | None, step: TensorInput
) -> range:
    """Create a sequential iterator range."""
    return _range_args(start, stop, step)


def static_range(
    start: TensorInput, stop: TensorInput | None = None, step: TensorInput = 1
) -> range:
    """Create a fully unrolled iterator range for tracing."""
    return _range_args(start, stop, step)


class tile_size:
    """Tile size constants used by the interpreter."""

    pmax = 128
    psum_fmax = 512
    gemm_stationary_fmax = 128
    gemm_moving_fmax = 512
    bn_stats_fmax = 512
    psum_min_align = 16
    sbuf_min_align = 16
    total_available_sbuf_size = 24 * 1024 * 1024


def device_print(print_prefix: str, tensor: TensorInput) -> None:
    """Print a tensor value from interpreted kernels."""
    print(print_prefix, _to_numpy(tensor))


def num_programs(axes: int | Sequence[int] | None = None) -> int | tuple[int, ...]:
    """Return launch grid extents."""
    if axes is None:
        return int(np.prod(nki_builder.grid_dims))
    if isinstance(axes, Sequence):
        return tuple(nki_builder.grid_dims[int(axis)] for axis in axes)
    return nki_builder.grid_dims[int(axes)]


def program_id(axis: int | np.integer) -> int:
    """Return the current launch index for one axis."""
    axis_v = int(axis)
    if 0 <= axis_v < len(nki_builder.grid_idx):
        return nki_builder.grid_idx[axis_v]
    raise ValueError(f"Invalid axis {axis_v} for {len(nki_builder.grid_idx)}D grid")


def program_ndim() -> int:
    """Return launch grid dimensionality."""
    return len(nki_builder.grid_dims)


_REDUCE_CMD = getattr(nisa, "reduce_cmd", None)
_REDUCE_IDLE = getattr(_REDUCE_CMD, "idle", None)
_REDUCE_RESET = getattr(_REDUCE_CMD, "reset", None)
_REDUCE_REDUCE = getattr(_REDUCE_CMD, "reduce", None)
_REDUCE_RESET_REDUCE = getattr(_REDUCE_CMD, "reset_reduce", None)
_REDUCE_LOAD_REDUCE = getattr(_REDUCE_CMD, "load_reduce", None)

_ENGINE = getattr(nisa, "engine", None)
_ENGINE_UNKNOWN = getattr(_ENGINE, "unknown", None)

_DGE_MODE = getattr(nisa, "dge_mode", None)
_DGE_UNKNOWN = getattr(_DGE_MODE, "unknown", None)

_OOB_MODE = getattr(nisa, "oob_mode", None)
_OOB_ERROR = getattr(_OOB_MODE, "error", None)

_MATMUL_PERF_MODE = getattr(nisa, "matmul_perf_mode", None)
_MATMUL_PERF_NONE = getattr(_MATMUL_PERF_MODE, "none", None)


def _cmd_name(cmd: object | None) -> str:
    if cmd is None:
        return "none"
    return getattr(cmd, "name", str(cmd)).lower()


def _is_cmd(cmd: object | None, expected_name: str) -> bool:
    return expected_name in _cmd_name(cmd)


def _partition_reduce(value: TensorInput, op: OpLike) -> np.ndarray:
    axes = _free_axes(value)
    if not axes:
        return np.asarray(value).reshape(np.asarray(value).shape[0], 1)
    reduced = _apply_reduce(op, value, axis=axes, keepdims=True)
    return np.asarray(reduced).reshape(np.asarray(value).shape[0], 1)


def _ensure_regs(attr: str, size: int, init_value: ScalarLike) -> np.ndarray:
    """Create or resize reduction register state on demand."""
    regs = getattr(nki_builder, attr)
    if regs is None or regs.shape[0] != size:
        regs = np.full((size, 1), init_value, np.float64)
        setattr(nki_builder, attr, regs)
    return regs


def nc_transpose(
    dst: NDArray,
    data: TensorInput,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> NDArray:
    """Compute a transpose between partition and flattened free axes."""
    del engine, name
    value = _to_numpy(data)
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    return _write_dst(dst, transposed.reshape(dst.shape))


def nc_matmul(
    dst: NDArray,
    stationary: TensorInput,
    moving: TensorInput,
    is_stationary_onezero: bool = False,
    is_moving_onezero: bool = False,
    is_transpose: bool = False,
    tile_position: Sequence[int] | None = (),
    tile_size: Sequence[int] | None = (),
    perf_mode: object | None = _MATMUL_PERF_NONE,
    name: str | None = None,
) -> NDArray:
    """Compute `dst = stationary.T @ moving`."""
    del (
        is_stationary_onezero,
        is_moving_onezero,
        is_transpose,
        tile_position,
        tile_size,
        perf_mode,
        name,
    )
    result = _to_numpy(stationary).T @ _to_numpy(moving)
    return _write_dst(dst, result)


def _mx_dequantize(data: NDArray, scale: TensorInput) -> np.ndarray:
    # unwrap NDArray scales before dequantization
    scale_value = _to_numpy(scale)
    if scale_value.ndim > 0 and data.shape[0] != scale_value.shape[0]:
        if data.shape[0] % scale_value.shape[0] == 0:
            repeats = data.shape[0] // scale_value.shape[0]
            scale_value = np.repeat(scale_value, repeats, axis=0)
    return data.data * scale_value


def nc_matmul_mx(
    dst: NDArray,
    stationary: NDArray,
    moving: NDArray,
    stationary_scale: TensorInput,
    moving_scale: TensorInput,
    tile_position: Sequence[int] | None = None,
    tile_size: Sequence[int] | None = None,
    name: str | None = None,
) -> NDArray:
    """Compute MX matmul by dequantizing operands then applying matmul."""
    del tile_position, tile_size, name
    stationary_dequant = _mx_dequantize(stationary, stationary_scale)
    moving_dequant = _mx_dequantize(moving, moving_scale)
    result = stationary_dequant.T @ moving_dequant
    return _write_dst(dst, result)


def reciprocal(dst: NDArray, data: TensorInput, name: str | None = None) -> NDArray:
    """Compute elementwise reciprocal into destination tensor."""
    del name
    result = np.reciprocal(_to_numpy(data))
    return _write_dst(dst, result)


def register_alloc(x: TensorInput | None = None) -> int | NDArray:
    """Allocate a register value for scalar loop/control flow."""
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    return NDArray("register", value=np.array(_as_scalar(x), dtype=np.int32))


def register_move(dst: NDArray, imm: TensorInput) -> int | NDArray:
    """Move an immediate integer value into a register."""
    value = _as_scalar(imm)
    if isinstance(dst, NDArray):
        dst.data[...] = value
        return dst
    return value


def register_load(dst: NDArray, src: TensorInput) -> int | NDArray:
    """Load a scalar tensor value into a register."""
    value = _as_scalar(src)
    if isinstance(dst, NDArray):
        dst.data[...] = value
        return dst
    return value


def register_store(dst: NDArray, src: TensorInput) -> NDArray:
    """Store a register value into a scalar tensor."""
    return _write_dst(dst, _as_scalar(src))


def dma_copy(
    dst: NDArray,
    src: TensorInput,
    dst_rmw_op: OpLike | None = None,
    oob_mode: object | None = _OOB_ERROR,
    dge_mode: object | None = _DGE_UNKNOWN,
    unique_indices: bool = True,
    name: str | None = None,
) -> NDArray:
    """Copy data from source tensor to destination tensor."""
    del oob_mode, dge_mode, unique_indices, name
    src_value = _to_numpy(src)
    if dst_rmw_op is None:
        return _write_dst(dst, src_value)
    merged = _apply_binary(dst_rmw_op, dst.data, src_value)
    return _write_dst(dst, merged)


def tensor_copy(
    dst: NDArray,
    src: TensorInput,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> NDArray:
    """Copy tensor data within on-chip memory."""
    del engine, name
    return _write_dst(dst, _to_numpy(src))


def tensor_tensor(
    dst: NDArray,
    data1: TensorInput,
    data2: TensorInput,
    op: OpLike,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> NDArray:
    """Apply a binary tensor op and store the result in `dst`."""
    del engine, name
    return _write_dst(dst, _apply_binary(op, _to_numpy(data1), _to_numpy(data2)))


def quantize_mx(
    dst: NDArray, src: TensorInput, dst_scale: NDArray, name: str | None = None
) -> NDArray:
    """Quantize source values with a simple global scale approximation."""
    del name
    src_fp32 = _to_numpy(src)
    if src_fp32.size == 0:
        _write_dst(dst, src_fp32)
        _write_dst(dst_scale, 1)
        return dst
    max_abs = float(np.max(np.abs(src_fp32)))
    scale = 1.0 if max_abs == 0 else max_abs / 127.0
    _write_dst(dst_scale, scale)
    quantized = np.round(src_fp32 / scale)
    return _write_dst(dst, quantized)


def memset(
    dst: NDArray,
    value: TensorInput,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> NDArray:
    """Fill destination tensor with a constant value."""
    del engine, name
    return _write_dst(dst, value)


def iota(
    dst: NDArray,
    pattern: PatternLike,
    offset: TensorInput,
    channel_multiplier: TensorInput = 0,
    name: str | None = None,
) -> NDArray:
    """Generate index pattern values into destination tensor."""
    del name
    pattern_pairs = _normalize_pattern(pattern)
    counts = [count for _, count in pattern_pairs]
    base_indices = _compute_ap_indices(pattern_pairs, offset)
    partition_count = dst.shape[0]
    channel_multiplier_v = _as_scalar(channel_multiplier)
    out = np.empty((partition_count, *counts), dtype=np.int64)
    for channel_id in range(partition_count):
        out[channel_id] = base_indices + channel_id * channel_multiplier_v
    return _write_dst(dst, out.reshape(dst.shape))


def activation(
    dst: NDArray,
    op: UnaryOpLike,
    data: TensorInput,
    bias: TensorInput | None = None,
    scale: TensorInput = 1.0,
    reduce_op: OpLike | None = None,
    reduce_res: NDArray | None = None,
    reduce_cmd: object | None = _REDUCE_IDLE,
    name: str | None = None,
) -> NDArray:
    """Apply activation with optional scale, bias, and reduction registers."""
    del name
    input_value = _to_numpy(data)
    scale_value = _to_numpy(scale)
    bias_value = 0.0 if bias is None else _to_numpy(bias)
    activated = _apply_unary(op, input_value * scale_value + bias_value)
    activated = cast(np.ndarray, activated)
    _write_dst(dst, activated)

    if reduce_op is None:
        reduce_op = np.add
    part_reduce = _partition_reduce(activated, reduce_op).astype(np.float64, copy=False)
    regs = _ensure_regs("scalar_reduce_regs", activated.shape[0], 0.0)
    if _is_cmd(reduce_cmd, "reset_reduce"):
        regs[...] = 0.0
        regs[...] += part_reduce
    elif _is_cmd(reduce_cmd, "reset"):
        regs[...] = 0.0
    elif _is_cmd(reduce_cmd, "reduce"):
        regs[...] += part_reduce
    if reduce_res is not None:
        _write_dst(reduce_res, regs)
    return dst


def activation_reduce(
    dst: NDArray,
    op: UnaryOpLike,
    data: TensorInput,
    reduce_op: OpLike | None,
    reduce_res: NDArray | None,
    bias: TensorInput | None = None,
    scale: TensorInput = 1.0,
    name: str | None = None,
) -> NDArray:
    """Run activation with reset-then-reduce behavior."""
    return activation(
        dst=dst,
        op=op,
        data=data,
        bias=bias,
        scale=scale,
        reduce_op=reduce_op,
        reduce_res=reduce_res,
        reduce_cmd=_REDUCE_RESET_REDUCE,
        name=name,
    )


def affine_select(
    dst: NDArray,
    pattern: PatternLike,
    offset: TensorInput,
    channel_multiplier: TensorInput,
    on_true_tile: TensorInput,
    on_false_value: TensorInput,
    cmp_op: OpLike = np.equal,
    name: str | None = None,
) -> NDArray:
    """Select between tile values and scalar using an affine predicate."""
    del name
    pattern_pairs = _normalize_pattern(pattern)
    counts = [count for _, count in pattern_pairs]
    affine = _compute_ap_indices(pattern_pairs, offset)
    out = np.empty((dst.shape[0], *counts))
    on_true_value = _to_numpy(on_true_tile)
    channel_multiplier_v = _as_scalar(channel_multiplier)
    for channel in range(dst.shape[0]):
        affine_value = affine + channel * channel_multiplier_v
        pred = _apply_binary(cmp_op, affine_value, 0, reverse=False)
        out[channel] = np.where(pred, on_true_value[channel], on_false_value)
    return _write_dst(dst, out.reshape(dst.shape))


def bn_stats(dst: NDArray, data: NDArray, name: str | None = None) -> NDArray:
    """Compute even/odd count, mean, and variance*count per partition."""
    del name
    flat = _to_numpy(data).reshape(data.shape[0], -1)
    even = flat[:, 0::2]
    odd = flat[:, 1::2]
    count_even = even.shape[1]
    count_odd = odd.shape[1]
    mean_even = (
        np.mean(even, axis=1, keepdims=True)
        if count_even
        else np.zeros((flat.shape[0], 1))
    )
    mean_odd = (
        np.mean(odd, axis=1, keepdims=True)
        if count_odd
        else np.zeros((flat.shape[0], 1))
    )
    var_count_even = (
        np.var(even, axis=1, keepdims=True) * count_even
        if count_even
        else np.zeros((flat.shape[0], 1))
    )
    var_count_odd = (
        np.var(odd, axis=1, keepdims=True) * count_odd
        if count_odd
        else np.zeros((flat.shape[0], 1))
    )
    out = np.concatenate(
        [
            np.full_like(mean_even, count_even),
            mean_even,
            var_count_even,
            np.full_like(mean_odd, count_odd),
            mean_odd,
            var_count_odd,
        ],
        axis=1,
    )
    return _write_dst(dst, out.reshape(dst.shape))


def bn_aggr(dst: NDArray, data: NDArray, name: str | None = None) -> NDArray:
    """Aggregate tuples of (count, mean, var*count) per partition."""
    del name
    flat = _to_numpy(data).reshape(data.shape[0], -1)
    if flat.shape[1] % 3 != 0:
        raise ValueError(
            "bn_aggr expects data elements per partition as a multiple of 3"
        )
    tuples = flat.reshape(flat.shape[0], -1, 3)
    counts = tuples[:, :, 0]
    means = tuples[:, :, 1]
    var_counts = tuples[:, :, 2]
    total_count = np.sum(counts, axis=1, keepdims=True)
    safe_total = np.where(total_count == 0, 1.0, total_count)
    combined_mean = np.sum(counts * means, axis=1, keepdims=True) / safe_total
    centered = means - combined_mean
    combined_var = (
        np.sum(var_counts + counts * centered * centered, axis=1, keepdims=True)
        / safe_total
    )
    out = np.concatenate([combined_mean, combined_var], axis=1)
    return _write_dst(dst, out.reshape(dst.shape))


def core_barrier(
    data: TensorInput,
    cores: TensorInput,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> None:
    """Synchronize cores in interpreter mode as a no-op."""
    del data, cores, engine, name
    return None


def dma_compute(
    dst: NDArray,
    srcs: Sequence[NDArray],
    scales: Sequence[ScalarLike],
    reduce_op: OpLike | None,
    name: str | None = None,
) -> NDArray:
    """Compute scaled elementwise reduction across source tensors."""
    del name
    if not srcs:
        return dst
    result = _to_numpy(srcs[0]) * float(_to_scalar(scales[0], float))
    for src, scale in zip(srcs[1:], scales[1:]):
        result = _apply_binary(
            reduce_op,
            result,
            _to_numpy(src) * float(_to_scalar(scale, float)),
        )
    return _write_dst(dst, result)


def dma_transpose(
    dst: NDArray,
    src: TensorInput,
    axes: int | Sequence[int] | None = None,
    dge_mode: object | None = _DGE_UNKNOWN,
    oob_mode: object | None = _OOB_ERROR,
    name: str | None = None,
) -> NDArray:
    """Transpose input tensor with supported default axis permutations."""
    del dge_mode, oob_mode, name
    src_value = _to_numpy(src)
    if axes is None:
        default_axes = {2: (1, 0), 3: (2, 1, 0), 4: (3, 1, 2, 0)}
        axes = default_axes.get(src_value.ndim, tuple(reversed(range(src_value.ndim))))
    return _write_dst(dst, np.transpose(src_value, axes=axes))


def _rng_key(engine: object | None) -> str:
    if engine is None:
        return "unknown"
    return getattr(engine, "name", str(engine))


def _get_rng(engine: object | None) -> np.random.Generator:
    key = _rng_key(engine)
    rng = nki_builder.rng_generators.get(key)
    if rng is None:
        rng = np.random.default_rng(0)
        nki_builder.rng_generators[key] = rng
    return rng


def dropout(
    dst: NDArray, data: TensorInput, prob: TensorInput, name: str | None = None
) -> NDArray:
    """Apply per-element dropout with scalar or vector probability."""
    del name
    data_value = _to_numpy(data)
    prob_value = np.asarray(_to_numpy(prob))
    if prob_value.size == 1:
        prob_broadcast = prob_value
    else:
        shape = [data_value.shape[0]] + [1] * (data_value.ndim - 1)
        prob_broadcast = prob_value.reshape(shape)
    mask = _get_rng(_ENGINE_UNKNOWN).random(data_value.shape) >= prob_broadcast
    return _write_dst(dst, np.where(mask, data_value, 0))


def local_gather(
    dst: NDArray,
    src_buffer: NDArray,
    index: NDArray,
    num_elem_per_idx: TensorInput = 1,
    num_valid_indices: TensorInput | None = None,
    name: str | None = None,
) -> NDArray:
    """Gather within 16-partition groups using flattened indices."""
    del name
    src_flat = _to_numpy(src_buffer).reshape(src_buffer.shape[0], -1)
    index_flat = np.asarray(_to_numpy(index), dtype=np.int64).reshape(
        index.shape[0], -1
    )
    num_elem_per_idx_v = _as_scalar(num_elem_per_idx)
    num_valid_indices_v = (
        _as_scalar(num_valid_indices) if num_valid_indices is not None else None
    )
    out = np.zeros_like(_to_numpy(dst).reshape(dst.shape[0], -1))
    for start in range(0, src_flat.shape[0], 16):
        src_group = src_flat[start : start + 16].reshape(-1)
        idx_group = index_flat[start : start + 16].reshape(-1)
        if num_valid_indices_v is not None:
            idx_group = idx_group[:num_valid_indices_v]
        values = []
        for idx in idx_group:
            base = int(idx) * num_elem_per_idx_v
            values.extend(src_group[base : base + num_elem_per_idx_v])
        group_out = np.asarray(values, dtype=out.dtype)
        size = out[start : start + 16].size
        tmp = np.zeros(size, dtype=out.dtype)
        tmp[: min(size, group_out.size)] = group_out[: min(size, group_out.size)]
        out[start : start + 16] = tmp.reshape(out[start : start + 16].shape)
    return _write_dst(dst, out.reshape(dst.shape))


def max8(dst: NDArray, src: NDArray, name: str | None = None) -> NDArray:
    """Select top-8 values per partition in descending order."""
    del name
    src_flat = _to_numpy(src).reshape(src.shape[0], -1)
    sorted_desc = -np.sort(-src_flat, axis=1)
    return _write_dst(dst, sorted_desc[:, :8].reshape(dst.shape))


def nc_find_index8(
    dst: NDArray, data: NDArray, vals: NDArray, name: str | None = None
) -> NDArray:
    """Find first occurrence indices for 8 values per partition."""
    del name
    data_flat = _to_numpy(data).reshape(data.shape[0], -1)
    vals_flat = _to_numpy(vals).reshape(vals.shape[0], -1)
    out = np.zeros((data.shape[0], 8), dtype=np.uint32)
    for p in range(data.shape[0]):
        for i in range(8):
            matches = np.nonzero(data_flat[p] == vals_flat[p, i])[0]
            out[p, i] = matches[0] if matches.size else np.uint32(0)
    return _write_dst(dst, out.reshape(dst.shape))


def nc_match_replace8(
    dst: NDArray,
    data: NDArray,
    vals: NDArray,
    imm: TensorInput,
    dst_idx: NDArray | None = None,
    name: str | None = None,
) -> NDArray:
    """Replace first matches and optionally write matched indices."""
    del name
    data_flat = _to_numpy(data).reshape(data.shape[0], -1)
    vals_flat = _to_numpy(vals).reshape(vals.shape[0], -1)
    out = data_flat.copy()
    idx_out = np.full((data.shape[0], 8), -1, dtype=np.int32)
    imm_v = float(_to_scalar(imm, float))
    for p in range(data.shape[0]):
        for i in range(8):
            matches = np.nonzero(out[p] == vals_flat[p, i])[0]
            if matches.size:
                first = int(matches[0])
                idx_out[p, i] = first
                out[p, first] = imm_v
    _write_dst(dst, out.reshape(dst.shape))
    if dst_idx is not None:
        _write_dst(dst_idx, idx_out.reshape(dst_idx.shape))
    return dst


def nc_n_gather(
    dst: NDArray, data: NDArray, indices: NDArray, name: str | None = None
) -> NDArray:
    """Gather flattened free-dimension elements per partition."""
    del name
    data_flat = _to_numpy(data).reshape(data.shape[0], -1)
    idx = np.asarray(_to_numpy(indices), dtype=np.int64).reshape(indices.shape[0], -1)
    gathered = np.take_along_axis(data_flat, idx, axis=1)
    return _write_dst(dst, gathered.reshape(dst.shape))


def nc_stream_shuffle(
    dst: NDArray,
    src: TensorInput,
    shuffle_mask: Sequence[int] | np.ndarray,
    name: str | None = None,
) -> NDArray:
    """Shuffle partitions within each 32-partition quadrant."""
    del name
    src_value = _to_numpy(src)
    out = _to_numpy(dst).copy()
    for start in range(0, src_value.shape[0], 32):
        for i, src_idx in enumerate(shuffle_mask[:32]):
            if src_idx == 255:
                continue
            out[start + i] = src_value[start + int(src_idx)]
    return _write_dst(dst, out)


def nonzero_with_count(
    dst: NDArray,
    src: NDArray,
    index_offset: TensorInput = 0,
    padding_val: TensorInput = -1,
) -> NDArray:
    """Write nonzero indices, padding, and count to destination tile."""
    src_flat = _to_numpy(src).reshape(src.shape[0], -1)
    dst_flat = _to_numpy(dst).reshape(dst.shape[0], -1)
    index_offset_v = _as_scalar(index_offset)
    padding_val_v = _as_scalar(padding_val)
    for p in range(0, src_flat.shape[0], 16):
        nz = np.nonzero(src_flat[p] != 0)[0] + index_offset_v
        out = np.full(src_flat.shape[1] + 1, padding_val_v, dtype=np.int32)
        count = min(nz.size, src_flat.shape[1])
        out[:count] = nz[:count]
        out[-1] = count
        dst_flat[p, : out.size] = out
    return _write_dst(dst, dst_flat.reshape(dst.shape))


def rand2(
    dst: NDArray, min: TensorInput, max: TensorInput, name: str | None = None
) -> NDArray:
    """Generate uniform random numbers in [min, max]."""
    del name
    min_value = _to_numpy(min)
    max_value = _to_numpy(max)
    random = _get_rng(_ENGINE_UNKNOWN).random(dst.shape)
    return _write_dst(dst, min_value + random * (max_value - min_value))


def rand_get_state(
    dst: NDArray, engine: object | None = _ENGINE_UNKNOWN, name: str | None = None
) -> NDArray:
    """Write cached RNG state seeds for the requested engine."""
    del name
    key = _rng_key(engine)
    state = nki_builder.rng_states.get(key)
    if state is None:
        state = np.zeros(dst.shape, dtype=np.uint32)
    return _write_dst(dst, np.asarray(state, dtype=np.uint32).reshape(dst.shape))


def _seed_from_values(values: TensorInput) -> int:
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    if flat.size == 0:
        return 0
    weights = np.arange(1, flat.size + 1, dtype=np.uint64)
    seed = int(np.bitwise_xor.reduce(flat * weights))
    return seed & 0x7FFFFFFF


def rand_set_state(
    src_seeds: TensorInput,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> None:
    """Set RNG state seeds for the requested engine."""
    del name
    key = _rng_key(engine)
    seeds = np.asarray(_to_numpy(src_seeds), dtype=np.uint32)
    nki_builder.rng_states[key] = seeds.copy()
    nki_builder.rng_generators[key] = np.random.default_rng(_seed_from_values(seeds))
    return None


def range_select(
    dst: NDArray,
    on_true_tile: TensorInput,
    comp_op0: OpLike,
    comp_op1: OpLike,
    bound0: TensorInput,
    bound1: TensorInput,
    reduce_cmd: object | None = _REDUCE_RESET_REDUCE,
    reduce_res: NDArray | None = None,
    reduce_op: OpLike | None = np.maximum,
    range_start: TensorInput = 0.0,
    on_false_value: TensorInput = -3.4028235e38,
    name: str | None = None,
) -> NDArray:
    """Select values by index-range predicate and optionally reduce."""
    del reduce_cmd, on_false_value, name
    on_true = _to_numpy(on_true_tile)
    idx_shape = [1] + list(on_true.shape[1:])
    index_grid = np.arange(on_true.reshape(on_true.shape[0], -1).shape[1]).reshape(
        idx_shape
    )
    index_grid = index_grid + float(_to_scalar(range_start, float))
    bound0_v = _to_numpy(bound0).reshape(on_true.shape[0], *([1] * (on_true.ndim - 1)))
    bound1_v = _to_numpy(bound1).reshape(on_true.shape[0], *([1] * (on_true.ndim - 1)))
    comp_lhs = cast(np.ndarray, _apply_binary(comp_op0, index_grid, bound0_v))
    comp_rhs = cast(np.ndarray, _apply_binary(comp_op1, index_grid, bound1_v))
    pred = comp_lhs & comp_rhs
    fill = np.float64(np.finfo(np.float64).min)
    out = np.where(pred, on_true, fill)
    _write_dst(dst, out)
    if reduce_res is not None:
        _write_dst(reduce_res, _partition_reduce(out, reduce_op))
    return dst


def rng(
    dst: NDArray, engine: object | None = _ENGINE_UNKNOWN, name: str | None = None
) -> NDArray:
    """Generate random integer bits into destination tensor."""
    del name
    random = _get_rng(engine).integers(0, 2**32, size=dst.shape, dtype=np.uint32)
    return _write_dst(dst, random)


def scalar_tensor_tensor(
    dst: NDArray,
    data: TensorInput,
    op0: OpLike,
    operand0: TensorInput,
    op1: OpLike | None,
    operand1: TensorInput | None,
    reverse0: bool = False,
    reverse1: bool = False,
    name: str | None = None,
) -> NDArray:
    """Apply scalar then tensor operator sequence."""
    del name
    tmp = _apply_binary(op0, data, operand0, reverse=reverse0)
    out = _apply_binary(op1, tmp, operand1, reverse=reverse1)
    return _write_dst(dst, out)


def select_reduce(
    dst: NDArray,
    predicate: TensorInput,
    on_true: TensorInput,
    on_false: TensorInput,
    reduce_res: NDArray | None = None,
    reduce_cmd: object | None = _REDUCE_IDLE,
    reduce_op: OpLike | None = np.maximum,
    reverse_pred: bool = False,
    name: str | None = None,
) -> NDArray:
    """Select by predicate and optionally update vector reduce registers."""
    del name
    pred = _to_numpy(predicate) != 0
    if reverse_pred:
        pred = ~pred
    out = np.where(pred, _to_numpy(on_true), _to_numpy(on_false))
    _write_dst(dst, out)
    regs = _ensure_regs("vector_reduce_regs", out.shape[0], -np.inf)
    if _is_cmd(reduce_cmd, "reset_reduce"):
        regs[...] = -np.inf
        regs[...] = _apply_binary(reduce_op, regs, _partition_reduce(out, reduce_op))
    elif _is_cmd(reduce_cmd, "reduce"):
        regs[...] = _apply_binary(reduce_op, regs, _partition_reduce(out, reduce_op))
    if reduce_res is not None:
        _write_dst(reduce_res, regs)
    return dst


def sendrecv(
    src: TensorInput,
    dst: NDArray,
    send_to_rank: TensorInput,
    recv_from_rank: TensorInput,
    pipe_id: TensorInput,
    name: str | None = None,
) -> NDArray:
    """Simulate point-to-point send/recv as local copy."""
    del send_to_rank, recv_from_rank, pipe_id, name
    return _write_dst(dst, _to_numpy(src))


def sequence_bounds(
    dst: NDArray, segment_ids: NDArray, name: str | None = None
) -> NDArray:
    """Compute [start, end] segment bounds for each element."""
    del name
    seg = np.asarray(_to_numpy(segment_ids), dtype=np.int32).reshape(
        segment_ids.shape[0], -1
    )
    out = np.empty((seg.shape[0], 2, seg.shape[1]), dtype=np.int32)
    n = seg.shape[1]
    for p in range(seg.shape[0]):
        for i in range(n):
            sid = seg[p, i]
            if sid == 0:
                out[p, 0, i] = n
                out[p, 1, i] = -1
                continue
            matches = np.nonzero(seg[p] == sid)[0]
            out[p, 0, i] = int(matches[0]) if matches.size else n
            out[p, 1, i] = int(matches[-1]) if matches.size else -1
    return _write_dst(dst, out.reshape(dst.shape))


def set_rng_seed(src_seeds: TensorInput, name: str | None = None) -> None:
    """Set vector-engine RNG seed state."""
    del name
    return rand_set_state(src_seeds, engine=_ENGINE_UNKNOWN)


def tensor_copy_predicated(
    dst: NDArray,
    src: TensorInput,
    predicate: TensorInput,
    reverse_pred: bool = False,
    name: str | None = None,
) -> NDArray:
    """Conditionally copy from src where predicate is true."""
    del name
    pred = _to_numpy(predicate) != 0
    if reverse_pred:
        pred = ~pred
    src_value = _to_numpy(src)
    out = np.where(pred, src_value, dst.data)
    return _write_dst(dst, out)


def tensor_partition_reduce(
    dst: NDArray, op: OpLike, data: TensorInput, name: str | None = None
) -> NDArray:
    """Reduce tensor across partition axis."""
    del name
    reduced = _apply_reduce(op, _to_numpy(data), axis=0, keepdims=True)
    return _write_dst(dst, np.asarray(reduced).reshape(dst.shape))


def tensor_reduce(
    dst: NDArray,
    op: OpLike,
    data: TensorInput,
    axis: int | Sequence[int],
    negate: bool = False,
    keepdims: bool = False,
    name: str | None = None,
) -> NDArray:
    """Reduce tensor across selected free axes."""
    del name
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    reduced = _apply_reduce(op, _to_numpy(data), axis=axes, keepdims=keepdims)
    if negate:
        reduced = -np.asarray(reduced)
    return _write_dst(dst, np.asarray(reduced).reshape(dst.shape))


def tensor_scalar(
    dst: NDArray,
    data: TensorInput,
    op0: OpLike,
    operand0: TensorInput,
    reverse0: bool = False,
    op1: OpLike | None = None,
    operand1: TensorInput | None = None,
    reverse1: bool = False,
    engine: object | None = _ENGINE_UNKNOWN,
    name: str | None = None,
) -> NDArray:
    """Apply one or two tensor-scalar operators."""
    del engine, name
    out = _apply_binary(op0, data, operand0, reverse=reverse0)
    if op1 is not None:
        out = _apply_binary(op1, out, operand1, reverse=reverse1)
    return _write_dst(dst, out)


def tensor_scalar_cumulative(
    dst: NDArray,
    src: NDArray,
    op0: OpLike,
    op1: OpLike | None,
    imm0: TensorInput,
    imm1: TensorInput | None = None,
    reduce_cmd: object | None = _REDUCE_RESET_REDUCE,
) -> NDArray:
    """Apply scalar op then cumulative reduction along free dimension."""
    src_flat = _to_numpy(src).reshape(src.shape[0], -1)
    imm0_value = _to_numpy(imm0)
    imm1_value = 0.0 if imm1 is None else _to_numpy(imm1)
    op_key = _op_key(op1)
    if _is_cmd(reduce_cmd, "load_reduce"):
        reg = np.broadcast_to(imm1_value.reshape(-1, 1), (src.shape[0], 1)).reshape(-1)
    elif "mul" in op_key:
        reg = np.ones(src.shape[0])
    elif "max" in op_key:
        reg = np.full(src.shape[0], -np.inf)
    elif "min" in op_key:
        reg = np.full(src.shape[0], np.inf)
    else:
        reg = np.zeros(src.shape[0])
    out = np.empty_like(src_flat)
    for i in range(src_flat.shape[1]):
        step = _to_numpy(_apply_binary(op0, src_flat[:, i], imm0_value, reverse=False))
        reg = _to_numpy(_apply_binary(op1, step, reg, reverse=False)).reshape(-1)
        out[:, i] = reg
    nki_builder.vector_reduce_regs = reg.reshape(-1, 1)
    return _write_dst(dst, out.reshape(dst.shape))


def tensor_scalar_reduce(
    dst: NDArray,
    data: TensorInput,
    op0: OpLike,
    operand0: TensorInput,
    reduce_op: OpLike | None,
    reduce_res: NDArray | None,
    reverse0: bool = False,
    name: str | None = None,
) -> NDArray:
    """Apply tensor-scalar op then reduce over free dimensions."""
    del name
    out = _apply_binary(op0, data, operand0, reverse=reverse0)
    _write_dst(dst, out)
    if reduce_res is not None:
        _write_dst(reduce_res, _partition_reduce(out, reduce_op))
    return dst


def tensor_tensor_scan(
    dst: NDArray,
    data0: NDArray,
    data1: NDArray,
    # initial: TensorInput,
    initial: NDArray,
    op0: OpLike,
    op1: OpLike | None,
    reverse0: bool = False,
    reverse1: bool = False,
    name: str | None = None,
) -> NDArray:
    """Run tensor scan recurrence along flattened free dimensions."""
    del name
    lhs = _to_numpy(data0).reshape(data0.shape[0], -1)
    rhs = _to_numpy(data1).reshape(data1.shape[0], -1)
    init = np.asarray(_to_numpy(initial)).reshape(data0.shape[0])
    out = np.empty_like(lhs)
    prev = _to_numpy(_apply_binary(op0, lhs[:, 0], init, reverse=reverse0))
    out[:, 0] = _to_numpy(_apply_binary(op1, prev, rhs[:, 0], reverse=reverse1))
    for i in range(1, lhs.shape[1]):
        prev = _to_numpy(_apply_binary(op0, lhs[:, i], out[:, i - 1], reverse=reverse0))
        out[:, i] = _to_numpy(_apply_binary(op1, prev, rhs[:, i], reverse=reverse1))
    return _write_dst(dst, out.reshape(dst.shape))


class Builder:
    """Tracks grid dimensions/index for the active interpreted kernel."""

    def __init__(self: Builder, grid_dims: Sequence[int] | None = None) -> None:
        self.grid_dims: tuple[int, ...] = (
            tuple(int(dim) for dim in grid_dims) if grid_dims is not None else (1,)
        )
        self.grid_idx: list[int] = [0] * len(self.grid_dims)
        self.fn: Callable[..., object] | None = None
        self.shared_hbm_arrays: dict[str, NDArray] = {}
        self.scalar_reduce_regs: np.ndarray | None = None
        self.vector_reduce_regs: np.ndarray | None = None
        self.rng_states: dict[str, np.ndarray] = {}
        self.rng_generators: dict[str, np.random.Generator] = {}


nki_builder = Builder()


class NKIOp:
    """Wrap a NumPy op and preserve NDArray input/output semantics for `nl.*` ops."""

    def __init__(self: NKIOp, op: Callable[..., np.ndarray]) -> None:
        self.op: Callable[..., np.ndarray] = op

    def __call__(
        self: NKIOp,
        *values: TensorInput,
        dtype: DTypeLike = None,
        **kwargs: object,
    ) -> NDArray:
        """Apply wrapped op on NumPy views of inputs, then wrap result as NDArray."""
        result = self.op(*(_to_numpy(value) for value in values), **kwargs).astype(
            dtype
        )
        buffer = None  # infer memory buffer based on inputs
        for value in values:
            if isinstance(result, NDArray):
                buffer = result.buffer
                break
        return NDArray(value=result, buffer=buffer)


def nki_patch_lang(scope: PatchScope | None = None) -> None:
    """Patch `nl` and `nisa` APIs to point at beta2 interpreter implementations."""

    def gelu_apprx_sigmoid_dx(x: TensorInput) -> np.ndarray:
        x_value = _to_numpy(x)
        sig = 1.0 / (1.0 + np.exp(-1.702 * x_value))
        return sig + x_value * (1.702 * sig * (1.0 - sig))

    nl_bindings = {
        "ndarray": ndarray,
        "zeros": zeros,
        "shared_constant": shared_constant,
        "shared_identity_matrix": shared_identity_matrix,
        "affine_range": affine_range,
        "ds": ds,
        "sequential_range": sequential_range,
        "static_range": static_range,
        "tile_size": tile_size,
        "abs": NKIOp(np.abs),
        "add": NKIOp(np.add),
        "bitwise_and": NKIOp(np.bitwise_and),
        "bitwise_or": NKIOp(np.bitwise_or),
        "bitwise_xor": NKIOp(np.bitwise_xor),
        "divide": NKIOp(np.divide),
        "equal": NKIOp(np.equal),
        "gelu_apprx_sigmoid": NKIOp(lambda x: x / (1 + np.exp(-1.702 * x))),
        "gelu_apprx_sigmoid_dx": NKIOp(gelu_apprx_sigmoid_dx),
        "greater": NKIOp(np.greater),
        "greater_equal": NKIOp(np.greater_equal),
        "invert": NKIOp(np.invert),
        "left_shift": NKIOp(np.left_shift),
        "less": NKIOp(np.less),
        "less_equal": NKIOp(np.less_equal),
        "logical_and": NKIOp(np.logical_and),
        "logical_not": NKIOp(np.logical_not),
        "logical_or": NKIOp(np.logical_or),
        "logical_xor": NKIOp(np.logical_xor),
        "maximum": NKIOp(np.maximum),
        "minimum": NKIOp(np.minimum),
        "multiply": NKIOp(np.multiply),
        "not_equal": NKIOp(np.not_equal),
        "power": NKIOp(np.power),
        "reciprocal": NKIOp(np.reciprocal),
        "right_shift": NKIOp(np.right_shift),
        "rsqrt": NKIOp(lambda x: 1 / np.sqrt(x)),
        "subtract": NKIOp(np.subtract),
        "device_print": device_print,
        "num_programs": num_programs,
        "program_id": program_id,
        "program_ndim": program_ndim,
    }
    nisa_bindings = {
        "activation": activation,
        "activation_reduce": activation_reduce,
        "affine_select": affine_select,
        "bn_aggr": bn_aggr,
        "bn_stats": bn_stats,
        "core_barrier": core_barrier,
        "dma_compute": dma_compute,
        "dma_copy": dma_copy,
        "dma_transpose": dma_transpose,
        "dropout": dropout,
        "iota": iota,
        "local_gather": local_gather,
        "max8": max8,
        "memset": memset,
        "nc_find_index8": nc_find_index8,
        "nc_match_replace8": nc_match_replace8,
        "nc_matmul": nc_matmul,
        "nc_matmul_mx": nc_matmul_mx,
        "nc_n_gather": nc_n_gather,
        "nc_stream_shuffle": nc_stream_shuffle,
        "nc_transpose": nc_transpose,
        "nonzero_with_count": nonzero_with_count,
        "quantize_mx": quantize_mx,
        "rand2": rand2,
        "rand_get_state": rand_get_state,
        "rand_set_state": rand_set_state,
        "range_select": range_select,
        "reciprocal": reciprocal,
        "register_alloc": register_alloc,
        "register_move": register_move,
        "register_load": register_load,
        "register_store": register_store,
        "rng": rng,
        "scalar_tensor_tensor": scalar_tensor_tensor,
        "select_reduce": select_reduce,
        "sendrecv": sendrecv,
        "sequence_bounds": sequence_bounds,
        "set_rng_seed": set_rng_seed,
        "tensor_copy": tensor_copy,
        "tensor_copy_predicated": tensor_copy_predicated,
        "tensor_partition_reduce": tensor_partition_reduce,
        "tensor_reduce": tensor_reduce,
        "tensor_scalar": tensor_scalar,
        "tensor_scalar_cumulative": tensor_scalar_cumulative,
        "tensor_scalar_reduce": tensor_scalar_reduce,
        "tensor_tensor": tensor_tensor,
        "tensor_tensor_scan": tensor_tensor_scan,
    }
    set_attr = setattr if scope is None else scope.set_attr
    # for name, fn in nl_bindings.items():
    #    #export_name = "nl_reciprocal" if name == "reciprocal" else name
    #    globals()[name] = fn
    for name, fn in nl_bindings.items():
        set_attr(nl, name, fn)
    for name, fn in nisa_bindings.items():
        set_attr(nisa, name, fn)


# restore any patch scope used for nki_patch_lang
nki_unpatch_lang = (
    lambda scope=None: scope.restore()
    if scope is not None and hasattr(scope, "restore")
    else None
)


class NKIInterpretedFunction:
    """Callable wrapper that executes NKI kernels with interpreter semantics."""

    def __init__(self: NKIInterpretedFunction, fn: Callable[..., object]) -> None:
        """Store the original kernel function."""
        self.fn = fn

    def run(self: NKIInterpretedFunction, *args: object, **kwargs: object) -> None:
        """Run the wrapped kernel over a launch grid with optional tracing callbacks."""
        grid_dims_kw: Sequence[int] | None = cast(
            Sequence[int], kwargs.pop("grid", (1,))
        )
        if isinstance(grid_dims_kw, Sequence):
            grid_dims = tuple(grid_dims_kw)
        else:
            raise TypeError("grid must be an int or a sequence of ints")
        # set active interpreted launch grid shape and reset current grid index cursor
        nki_builder.grid_dims = grid_dims
        nki_builder.grid_idx = [0] * len(nki_builder.grid_dims)
        nki_builder.shared_hbm_arrays = {}
        nki_builder.scalar_reduce_regs = None
        nki_builder.vector_reduce_regs = None
        nki_builder.rng_states = {}
        nki_builder.rng_generators = {}
        nki_builder.fn = self.fn

        kwargs.pop("warmup", None)
        client_manager = cast(ClientManager | None, kwargs.pop("client_manager", None))

        if client_manager is not None:
            client_manager.grid_callback(grid_dims)

        assert hasattr(self.fn, "__globals__")
        exec_globals = cast(dict[str, object], self.fn.__globals__)
        exec_globals["sbuf"] = sbuf
        exec_globals["hbm"] = hbm
        exec_globals["shared_hbm"] = getattr(nl, "shared_hbm", hbm)
        exec_globals["private_hbm"] = getattr(nl, "private_hbm", hbm)
        exec_globals["psum"] = psum

        runtime_args = [
            NDArray("hbm", value=arg) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]

        sig = inspect.signature(self.fn)
        bound = sig.bind(*runtime_args, **kwargs)
        bound.apply_defaults()
        for name, arg in bound.arguments.items():
            assert arg is None or isinstance(arg, (bool, int, float, str, NDArray))
            if client_manager is not None:
                client_manager.arg_callback(name, arg, arg)

        for grid_idx in itertools.product(*[range(dim) for dim in grid_dims]):
            if len(grid_idx) != len(nki_builder.grid_dims):
                raise ValueError(
                    f"Grid index rank mismatch: got {len(grid_idx)}, expected {len(nki_builder.grid_dims)}"
                )
            # set active interpreted program index for nl.program_id() queries
            nki_builder.grid_idx = [int(idx) for idx in grid_idx]
            if client_manager is not None:
                client_manager.grid_idx_callback(grid_idx)
                if not client_manager.pre_run_callback(self.fn):
                    return
            self.fn(*runtime_args, **kwargs)
            if client_manager is not None and not client_manager.post_run_callback(
                self.fn
            ):
                return
