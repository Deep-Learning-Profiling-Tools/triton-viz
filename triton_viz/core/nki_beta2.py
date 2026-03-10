from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Iterable
from typing import Any, Optional, TypeAlias, cast

import numpy as np

from triton_viz.utils.dtypes import STORAGE_DTYPES, DTypeLike
from triton_viz.utils.traceback_utils import CODE_KEYS, get_code_key

try:
    import nki.isa as nisa
    import nki.language as nl
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - only hit when optional deps missing
    raise ModuleNotFoundError(
        "NKI dependencies are missing. Install triton-viz[nki] to enable the "
        "NKI Beta 2 interpreter."
    ) from exc


_ERR_BINARY_TENSOR_OP = (
    "binary operators on tensors not supported. Use nki.isa directly."
)
_ERR_TENSOR_SUBSCRIPT = "subscript not supported, for 'tensor'"
_ERR_TENSOR_MUTATION = "mutation not supported"
_ERR_UNDEFINED_USE = "Illegal IR, encountered undefined use"
_ERR_AP_MISMATCH = "Expect AP same number of elements"

ScalarLike: TypeAlias = bool | int | float
ArrayLike: TypeAlias = np.ndarray[Any, Any]
TensorLike: TypeAlias = "NDArray | ArrayLike"
TensorOrScalar: TypeAlias = "NDArray | ArrayLike | ScalarLike"
BufferLike: TypeAlias = Any


def _storage_dtype(dtype: DTypeLike) -> np.dtype[Any] | None:
    """Resolve a logical dtype to concrete NumPy storage dtype."""
    try:
        return STORAGE_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype: {dtype}") from exc


def _cast_value(value: TensorOrScalar, dtype: DTypeLike) -> ArrayLike:
    """Cast value to destination dtype storage using shared ml_dtypes mapping."""
    return np.asarray(value).astype(_storage_dtype(dtype))


def _mark_defined(value: Any) -> None:
    """Mark NDArray values as initialized after writes."""
    while isinstance(value, NDArray):
        value._defined = True
        value = value._parent


def _store(dst: "NDArray", value: TensorOrScalar) -> "NDArray":
    """Write a computed value into a destination tensor."""
    dst.data[...] = _cast_value(value, dst.dtype)
    _mark_defined(dst)
    return dst


def _tensor_value(value: TensorOrScalar | None) -> TensorOrScalar | None:
    """Unwrap NDArray inputs and validate they are initialized."""
    if isinstance(value, NDArray):
        if not value._defined:
            raise RuntimeError(_ERR_UNDEFINED_USE)
        return value.data
    return value


def _as_scalar(value: ScalarLike | NDArray) -> int:
    """Convert scalar-like value into int."""
    array = np.asarray(_tensor_value(value))
    if array.size != 1:
        raise ValueError("Expected scalar value")
    return int(array.reshape(-1)[0])


def _dtype_str(dtype: DTypeLike) -> str:
    """Return a stable string name for logical or storage dtypes."""
    return np.dtype(_storage_dtype(dtype)).name


def _buffer_name(value: BufferLike) -> str:
    """Return the interpreter buffer name for values with storage."""
    return getattr(value, "buffer", None) or "sbuf"


def _require(condition: bool, message: str) -> None:
    """Raise ValueError with a stable message when condition is false."""
    if not condition:
        raise ValueError(message)


def _is_integer_dtype(dtype: DTypeLike) -> bool:
    """Return whether dtype is integer/bool-like."""
    resolved = _storage_dtype(dtype)
    return np.issubdtype(np.dtype(resolved), np.integer)


def _partition_and_free(value: TensorLike) -> tuple[int, int]:
    """Return partition size and flattened free size."""
    shape = (
        cast(tuple[int, ...], value.shape)
        if isinstance(value, np.ndarray)
        else cast(tuple[int, ...], value.data.shape)
    )
    free = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    return shape[0], free


def _is_access_view(value: Any) -> bool:
    """Return whether a tensor is a sliced access view."""
    return isinstance(value, NDArray) and value._origin == "access"


def _nc_version_value(version: Any) -> int:
    """Normalize NC version tokens into plain integers for comparisons."""
    raw = getattr(version, "value", version)
    if isinstance(raw, int):
        return raw
    return int(cast(Any, raw))


def _activation_operand(
    value: TensorOrScalar | None, data_shape: tuple[int, ...], label: str
) -> float | ArrayLike:
    """Normalize activation scale/bias into broadcastable arrays."""
    if value is None:
        return 0.0
    arr = np.asarray(_tensor_value(value), dtype=np.float32)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    _require(
        arr.shape == (data_shape[0], 1),
        f"{label} must be a scalar or shape ({data_shape[0]}, 1)",
    )
    return np.broadcast_to(
        arr.reshape(data_shape[0], 1), (data_shape[0], int(np.prod(data_shape[1:])))
    ).reshape(data_shape)


class NDArray:
    """Lightweight NumPy-backed tensor used by the NKI beta2 interpreter."""

    def __init__(
        self,
        buffer: str | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike = None,
        value: TensorOrScalar | None = None,
        origin: str = "raw",
        parent_ndim: int | None = None,
        defined: bool | None = None,
        parent: NDArray | None = None,
    ) -> None:
        """Initialize an interpreter tensor backed by NumPy storage.

        Args:
            buffer: Logical buffer name such as ``"sbuf"``, ``"hbm"``, or
                ``"psum"``.
            shape: Requested storage shape when allocating a new tensor.
            dtype: Logical dtype token used to choose storage dtype.
            value: Optional initial value. When provided, the tensor uses this
                payload instead of allocating fresh storage.
            origin: Interpreter provenance tag used for mutation rules.
            parent_ndim: Rank of the parent tensor when this value is a view.
            defined: Explicit initialized-state override for undefined-use
                tracking.
            parent: Parent tensor when this value is a sliced view.
        """
        self.buffer = buffer
        self.dtype = dtype
        self._origin = origin
        self._parent = parent
        storage_shape = tuple(shape) if shape is not None else None
        storage_dtype = _storage_dtype(dtype)
        if value is None:
            assert storage_shape is not None
            self.data = np.zeros(storage_shape, dtype=storage_dtype)
        else:
            array = (
                _cast_value(value, dtype) if dtype is not None else np.asarray(value)
            )
            if storage_shape is not None and array.shape != storage_shape:
                array = array.reshape(storage_shape)
            self.data = array
        self._defined = (
            value is not None or origin != "tensor" if defined is None else defined
        )
        self._parent_ndim = self.data.ndim if parent_ndim is None else parent_ndim
        if self.dtype is None:
            self.dtype = self.data.dtype
        self._data_ptr = None

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return tensor shape."""
        return self.data.shape if self.data is not None else None

    @property
    def address(self) -> int:
        """Return host pointer integer for underlying storage."""
        if self._data_ptr is None:
            self._data_ptr = self.data.ctypes.data
        return cast(int, self._data_ptr)

    def data_ptr(self) -> int:
        """Return pointer address alias for frontend compatibility."""
        return self.address

    def stride(self) -> tuple[int, ...]:
        """Return byte strides."""
        return self.data.strides

    def element_size(self) -> int:
        """Return item size in bytes."""
        return getattr(self.dtype, "itemsize", self.data.dtype.itemsize)

    def cpu(self) -> "NDArray":
        """Return self to mimic framework tensor api."""
        return self

    def detach(self) -> "NDArray":
        """Return self to mimic framework tensor api."""
        return self

    def numpy(self) -> ArrayLike:
        """Return underlying NumPy array."""
        return self.data

    def __repr__(self) -> str:
        return f"NDArray(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, keys: Any) -> "NDArray":
        """Implement slicing operations for NDArray."""
        # if self._origin == "tensor":
        #    raise TypeError(_ERR_TENSOR_SUBSCRIPT)
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = []
        for axis, key in enumerate(keys):
            value = key.data if isinstance(key, NDArray) else key
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.reshape(-1)[0]
            new_keys.append(value)
        sliced_value = self.data[tuple(new_keys)]
        return NDArray(
            value=sliced_value,
            buffer=self.buffer,
            origin="access",
            parent_ndim=self._parent_ndim,
            defined=self._defined,
            parent=self,
        )

    def __setitem__(self, keys: Any, value: TensorOrScalar) -> "NDArray":
        """Assign values into a sliced region."""
        if self._origin == "tensor":
            raise RuntimeError(_ERR_TENSOR_MUTATION)
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        target = self.data[tuple(new_keys)]
        unwrapped = _tensor_value(value)
        if isinstance(value, NDArray):
            if (
                value.data.size != target.size
                or value._parent_ndim != self._parent_ndim
            ):
                raise ValueError(_ERR_AP_MISMATCH)
        target[...] = _cast_value(cast(TensorOrScalar, unwrapped), self.dtype)
        _mark_defined(self)
        return self

    def reshape(self, *args: Any, **kwargs: Any) -> "NDArray":
        """Return a reshaped tensor view."""
        return NDArray(value=self.data.reshape(*args), buffer=self.buffer, **kwargs)

    def __neg__(self) -> "NDArray":
        raise TypeError("cannot negate values of this type")

    def _raise_binary_op(self, *_args: Any, **_kwargs: Any) -> "NDArray":
        raise TypeError(_ERR_BINARY_TENSOR_OP)

    __and__ = __or__ = __xor__ = __lshift__ = __rshift__ = _raise_binary_op
    __rand__ = __ror__ = __rxor__ = __rlshift__ = __rrshift__ = _raise_binary_op
    __iand__ = __ior__ = __ixor__ = __ilshift__ = __irshift__ = _raise_binary_op


class Buffer:
    """NKI memory region abstraction used by sbuf/hbm/psum."""

    def __init__(
        self,
        buffer: str,
        shape: tuple[int, ...] | None = None,
        data: ArrayLike | None = None,
    ) -> None:
        """Initialize a raw buffer region.

        Args:
            buffer: Logical buffer name.
            shape: Optional byte-storage shape used when allocating a raw backing
                buffer.
            data: Optional existing raw or typed backing array.
        """
        self.buffer = buffer
        self.shape = shape
        self.data = data
        if shape is not None and data is None:
            self.data = np.empty(shape=shape, dtype=np.uint8)

    def view(
        self,
        dtype: DTypeLike,
        shape: tuple[int, ...],
        *,
        origin: str = "raw",
    ) -> NDArray:
        """Materialize a typed tensor view over this buffer region.

        Args:
            dtype: Logical dtype token for the returned view.
            shape: Tensor shape for the returned view.
            origin: Interpreter provenance tag for the returned tensor.
        """
        probe = NDArray(
            buffer=self.buffer,
            dtype=dtype,
            shape=shape,
            origin=origin,
            parent_ndim=len(shape),
        )
        if self.data is None:
            return probe
        if self.data.nbytes != probe.data.nbytes:
            raise ValueError(
                "Buffer shape mismatch: have "
                f"{self.data.nbytes} bytes, expected {probe.data.nbytes}"
            )
        if self.data.dtype == np.uint8:
            data = self.data.view(probe.data.dtype).reshape(probe.data.shape)
        else:
            data = np.asarray(self.data, dtype=probe.data.dtype).reshape(
                probe.data.shape
            )
        return NDArray(
            buffer=self.buffer,
            dtype=dtype,
            value=data,
            origin=origin,
            parent_ndim=len(shape),
        )


sbuf = Buffer("sbuf")
hbm = Buffer("hbm")
psum = Buffer("psum")


def _resolve_buffer(buffer: BufferLike) -> Buffer:
    """Resolve a buffer token into one of the interpreter buffers."""
    if isinstance(buffer, Buffer):
        return buffer
    if buffer is None:
        return sbuf
    if isinstance(buffer, str):
        name = buffer
    else:
        name = getattr(buffer, "name", "")
    if name == "hbm":
        return hbm
    if name == "sbuf":
        return sbuf
    if name == "psum":
        return psum
    raise TypeError(f"Unsupported buffer type: {type(buffer).__name__}")


def ndarray(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    *,
    buffer: BufferLike = None,
    **kwargs: TensorOrScalar,
) -> NDArray:
    """Create an interpreter tensor on the requested NKI buffer.

    Args:
        shape: Requested tensor shape.
        dtype: Logical dtype token for the returned tensor.
        buffer: Buffer token or buffer-like object naming ``sbuf``, ``hbm``, or
            ``psum``.
        **kwargs: Optional tensor construction overrides, such as ``value``.
    """
    resolved_buffer = _resolve_buffer(buffer)
    if "value" in kwargs:
        return NDArray(
            buffer=resolved_buffer.buffer,
            shape=shape,
            dtype=dtype,
            value=kwargs["value"],
            origin="tensor",
            parent_ndim=len(tuple(shape)),
        )
    ret = resolved_buffer.view(dtype, shape, origin="tensor")
    ret._defined = False
    return ret


def zeros(
    shape: tuple[int, ...],
    dtype: DTypeLike,
    *,
    buffer: BufferLike = None,
    **kwargs: TensorOrScalar,
) -> NDArray:
    """Create a zero-initialized tensor.

    Args:
        shape: Requested tensor shape.
        dtype: Logical dtype token for the returned tensor.
        buffer: Unsupported beta2 argument kept for signature parity.
        **kwargs: Additional tensor construction overrides.
    """
    if buffer is not None:
        raise TypeError(
            "unexpected keyword argument 'buffer' in builtinfunction 'builtin_lang_zeros'"
        )
    ret = ndarray(shape, dtype, **kwargs)
    ret.data.fill(0)
    _mark_defined(ret)
    return ret


def nc_transpose(
    dst: NDArray,
    data: NDArray,
    engine: Any = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Compute a transpose between partition and flattened free axes.

    Args:
        dst: Destination tensor receiving the transpose result.
        data: Source tensor to transpose.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    del name
    value = np.asarray(_tensor_value(data))
    dst_shape = cast(tuple[int, ...], dst.shape)
    data_par, data_free = _partition_and_free(value)
    _require(max(data_par, data_free) <= 128, "tile too large for nc_transpose")
    src_buffer = _buffer_name(data)
    dst_buffer = _buffer_name(dst)
    if engine == nisa.engine.unknown:
        engine = (
            nisa.vector_engine if max(data_par, data_free) <= 32 else nisa.tensor_engine
        )
    if engine == nisa.vector_engine:
        _require(
            src_buffer in ("sbuf", "psum") and dst_buffer in ("sbuf", "psum"),
            "Vector Engine nc_transpose only supports SBUF/PSUM",
        )
    elif engine == nisa.tensor_engine:
        _require(
            src_buffer == "sbuf" and dst_buffer == "psum",
            "Tensor Engine nc_transpose requires SBUF -> PSUM",
        )
        _require(
            dst_shape == (data_free, data_par),
            "Tensor Engine nc_transpose requires dst shape (free, partition)",
        )
    data_dtype = getattr(data, "dtype", np.asarray(_tensor_value(data)).dtype)
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    _require(
        _dtype_str(dst_dtype) == _dtype_str(data_dtype)
        or (
            _dtype_str(dst_dtype) == "bfloat16" and _dtype_str(data_dtype) == "float32"
        ),
        "dst dtype must match data dtype",
    )
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    return _store(dst, transposed.reshape(dst.shape))


def nc_matmul(
    dst: NDArray,
    stationary: NDArray,
    moving: NDArray,
    is_stationary_onezero: bool = False,
    is_moving_onezero: bool = False,
    is_transpose: bool = False,
    tile_position: tuple[int, int] | tuple[()] = (),
    tile_size: tuple[int, int] | tuple[()] = (),
    perf_mode: Any = nisa.matmul_perf_mode.none,
    name: str | None = None,
) -> NDArray:
    """Compute ``dst += stationary.T @ moving``.

    Args:
        dst: PSUM destination tensor to accumulate into.
        stationary: Stationary SBUF operand.
        moving: Moving SBUF operand.
        is_stationary_onezero: Unused beta2 compatibility flag.
        is_moving_onezero: Unused beta2 compatibility flag.
        is_transpose: Unused beta2 compatibility flag.
        tile_position: Optional tile origin for documented matmul tiling.
        tile_size: Optional tile shape paired with ``tile_position``.
        perf_mode: Unused beta2 performance-mode token.
        name: Optional kernel op name retained for API parity.
    """
    del (
        is_stationary_onezero,
        is_moving_onezero,
        is_transpose,
        perf_mode,
        name,
    )
    _require(_buffer_name(dst) == "psum", "nc_matmul requires dst in PSUM")
    _require(
        _buffer_name(stationary) == "sbuf",
        "nc_matmul requires stationary in SBUF",
    )
    _require(_buffer_name(moving) == "sbuf", "nc_matmul requires moving in SBUF")
    stationary_value = np.asarray(_tensor_value(stationary))
    moving_value = np.asarray(_tensor_value(moving))
    dst_shape = cast(tuple[int, ...], dst.shape)
    _require(stationary_value.ndim == 2, "nc_matmul stationary must be rank-2")
    stationary_par, stationary_free = _partition_and_free(stationary_value)
    moving_par, moving_free = _partition_and_free(moving_value)
    _require(
        stationary_par == moving_par,
        "partition dims of stationary and moving must match",
    )
    _require(stationary_par <= 128, "partition dim of stationary > 128")
    _require(stationary_free <= 128, "free dim of stationary > 128")
    _require(moving_free <= 512, "free dim of moving > 512")
    stationary_dtype = getattr(stationary, "dtype", stationary_value.dtype)
    moving_dtype = getattr(moving, "dtype", moving_value.dtype)
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    stationary_is_f32 = _dtype_str(stationary_dtype) == "float32"
    moving_is_f32 = _dtype_str(moving_dtype) == "float32"
    _require(
        stationary_is_f32 == moving_is_f32,
        "if one matmul operand is float32, both must be float32",
    )
    _require(
        _dtype_str(dst_dtype) in {"float32", "bfloat16"},
        "dst must be float32 or bfloat16",
    )
    _require(
        dst_shape == (stationary_free, *moving_value.shape[1:]),
        "nc_matmul dst must preserve stationary free dim and moving free dims",
    )
    has_tile_pos = bool(tile_position)
    has_tile_size = bool(tile_size)
    _require(
        has_tile_pos == has_tile_size,
        "both tile_position and tile_size must be supplied",
    )
    if has_tile_size:
        row_size, col_size = map(int, tile_size)
        start_row, start_col = map(int, tile_position)
        _require(
            row_size in (32, 64, 128) and col_size in (32, 64, 128),
            "tile_size dims must be 32, 64, or 128",
        )
        _require(
            row_size <= 128 and col_size <= 128, "tile_size larger than (128, 128)"
        )
        _require(
            stationary_par <= row_size and stationary_free <= col_size,
            "stationary tile size exceeds tile_size",
        )
        _require(
            start_row % row_size == 0, "start row must be a multiple of row tile size"
        )
        _require(
            start_col % col_size == 0, "start col must be a multiple of col tile size"
        )
        _require(
            start_row + row_size <= 128,
            "matmul tile (row_start + row_size) must not exceed 128",
        )
        _require(
            start_col + col_size <= 128,
            "matmul tile (col_start + col_size) must not exceed 128",
        )
    stationary_matrix = stationary_value.reshape(stationary_par, stationary_free)
    moving_matrix = moving_value.reshape(moving_par, moving_free)
    result = stationary_matrix.T @ moving_matrix
    return _store(
        dst,
        np.asarray(dst.data) + _cast_value(result.reshape(dst.shape), dst.dtype),
    )


def reciprocal(dst: NDArray, data: NDArray, name: str | None = None) -> NDArray:
    """Compute elementwise reciprocal into the destination tensor.

    Args:
        dst: Destination tensor receiving reciprocal values.
        data: Source tensor providing reciprocal inputs.
        name: Optional kernel op name retained for API parity.
    """
    del name
    result = np.reciprocal(np.asarray(_tensor_value(data), dtype=np.float32))
    return _store(dst, result)


def exponential(
    dst,
    src: NDArray,
    max_value: TensorOrScalar = 0.0,
    reduce_res: NDArray | None = None,
    reduce_cmd=nisa.reduce_cmd.idle,
    reduce_init=0.0,
) -> NDArray:
    """Compute ``exp(src - max_value)`` into the destination tensor.

    Args:
        dst: Destination tensor receiving the exponentiated output.
        src: Source tensor to exponentiate.
        max_value: Scalar or tensor broadcast operand subtracted before ``exp``.
        reduce_res: Optional destination tensor for row-wise output sums.
        reduce_cmd: Unused beta2 reduction token kept for API parity.
        reduce_init: Unused beta2 reduction init value kept for API parity.
    """
    del reduce_cmd, reduce_init
    current_nc = _nc_version_value(nki_builder.nc_version)
    min_nc = _nc_version_value(nisa.nc_version.gen4)
    _require(current_nc >= min_nc, "exponential only supports >= NeuronCore-v4")
    src_value = _tensor_value(src)
    max_value_data = _tensor_value(max_value)
    output = np.exp(
        np.asarray(src_value, dtype=np.float32)
        - np.asarray(max_value_data, dtype=np.float32)
    )
    _store(dst, output)
    if reduce_res is not None:
        reduced = np.sum(
            np.asarray(output).reshape(output.shape[0], -1), axis=1, keepdims=True
        )
        _store(reduce_res, reduced)
    return dst


def affine_range(
    start: int | NDArray, stop: int | NDArray | None = None, step: int | NDArray = 1
) -> range:
    """Create a Python range from static or register-backed bounds.

    Args:
        start: Range start or stop when ``stop`` is omitted.
        stop: Optional range stop.
        step: Optional range step.
    """
    start_v = _as_scalar(start) if isinstance(start, NDArray) else int(start)
    if stop is None:
        return range(start_v)
    stop_v = _as_scalar(stop) if isinstance(stop, NDArray) else int(stop)
    step_v = _as_scalar(step) if isinstance(step, NDArray) else int(step)
    return range(start_v, stop_v, step_v)


def ds(start: int | NDArray, size: int | NDArray) -> slice:
    """Return an NKI-style dynamic slice using start/extent semantics.

    Args:
        start: Slice start offset.
        size: Slice extent.
    """
    begin = _as_scalar(start)
    extent = _as_scalar(size)
    return slice(begin, begin + extent)


class TileSize:
    """Minimal tile size constants used by beta2 example kernels."""

    pmax = 128
    gemm_stationary_fmax = 128
    gemm_moving_fmax = 512


tile_size = TileSize()


def _broadcast_tensor_scalar_operand(
    data: TensorLike, operand: TensorOrScalar
) -> ArrayLike:
    """Broadcast tensor_scalar operand over free dimensions only."""
    data_arr = np.asarray(data)
    operand_arr = np.asarray(operand)
    if operand_arr.size == 1:
        return operand_arr
    _require(
        operand_arr.shape[0] == data_arr.shape[0],
        "tensor_scalar only broadcasts free dimensions",
    )
    _require(
        all(dim == 1 for dim in operand_arr.shape[1:]),
        "tensor_scalar only broadcasts free dimensions",
    )
    target_shape = (data_arr.shape[0],) + (1,) * (data_arr.ndim - 1)
    return np.broadcast_to(operand_arr.reshape(target_shape), data_arr.shape)


class SubOp:
    """
    NKI ISA ops like tensor_tensor have parameters to modify which "sub-operation" each element does (e.g. nl.add).
    There are a couple of NKI quirks with these ops (e.g. unary ops can be used in nisa ops that pass in two args)
    so we use this instead of numpy ops.
    """

    def __init__(
        self,
        np_op: Callable[..., Any],
        is_bitwise: bool,
        is_reducible: bool,
        name: str | None = None,
    ) -> None:
        """Initialize a patched NKI operator token.

        Args:
            np_op: Underlying NumPy callable implementing the op.
            is_bitwise: Whether the op follows integer bitwise rules.
            is_reducible: Whether the op can be used in ``tensor_reduce``.
            name: Optional stable display name override.
        """
        self._op = np_op  # NOTE: Only NKI ISA ops should use self._op/self._run
        self.is_bitwise = is_bitwise
        self.is_reducible = is_reducible
        self.num_args = len(inspect.signature(self._op).parameters)
        if name is None:
            self.name = getattr(np_op, "__name__", type(np_op).__name__)
        else:
            self.name = name

    def _run(self, *args: Any) -> Any:
        return self._op(*args[: self.num_args])

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(_ERR_BINARY_TENSOR_OP)

    def __repr__(self):
        return f"SubOp({self.name})"


def dma_copy(
    dst: NDArray,
    src: NDArray,
    dst_rmw_op: SubOp | None = None,
    oob_mode: Any = nisa.oob_mode.error,
    dge_mode: Any = nisa.dge_mode.unknown,
    unique_indices: bool = True,
    name: str | None = None,
) -> NDArray:
    """Copy data from a source tensor into a destination tensor.

    Args:
        dst: Destination tensor in HBM or SBUF.
        src: Source tensor in HBM or SBUF.
        dst_rmw_op: Optional read-modify-write op. Only ``nl.add`` is supported.
        oob_mode: Unused beta2 out-of-bounds mode token kept for API parity.
        dge_mode: Unused beta2 DMA gather engine token kept for API parity.
        unique_indices: Unused beta2 uniqueness flag kept for API parity.
        name: Optional kernel op name retained for API parity.
    """
    del oob_mode, dge_mode, unique_indices, name
    _require(
        _buffer_name(dst) in ("hbm", "sbuf") and _buffer_name(src) in ("hbm", "sbuf"),
        "dma_copy only supports HBM/SBUF source and destination",
    )
    src_value = np.asarray(_tensor_value(src))
    if isinstance(dst, NDArray) and isinstance(src, NDArray):
        _require(dst.shape == src.shape, _ERR_AP_MISMATCH)
    if dst_rmw_op is None:
        return _store(dst, src_value.reshape(dst.shape))
    _require(dst_rmw_op == nl.add, "only nl.add is supported for dma_copy dst_rmw_op")
    merged = dst_rmw_op._op(dst.data, src_value)
    return _store(dst, merged.data if isinstance(merged, NDArray) else merged)


def tensor_copy(
    dst: NDArray,
    src: NDArray,
    engine: Any = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Copy tensor data within on-chip memory.

    Args:
        dst: Destination tensor in SBUF or PSUM.
        src: Source tensor in SBUF or PSUM.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    del name
    dst_buffer = _buffer_name(dst)
    src_buffer = _buffer_name(src)
    _require(
        dst_buffer in ("sbuf", "psum") and src_buffer in ("sbuf", "psum"),
        "tensor_copy is on-chip only",
    )
    current_nc = _nc_version_value(nki_builder.nc_version)
    gen2 = _nc_version_value(nisa.nc_version.gen2)
    if engine == nisa.scalar_engine and current_nc <= gen2:
        raise ValueError("Scalar Engine tensor_copy is unsupported on NeuronCore-v2")
    if engine == nisa.gpsimd_engine and "psum" in (dst_buffer, src_buffer):
        raise ValueError("GpSimd tensor_copy cannot access PSUM")
    src_value = np.asarray(_tensor_value(src))
    _require(
        _partition_and_free(src_value) == _partition_and_free(dst),
        _ERR_AP_MISMATCH,
    )
    return _store(dst, src_value.reshape(dst.shape))


def tensor_tensor(
    dst: NDArray,
    data1: NDArray,
    data2: NDArray,
    op: SubOp,
    engine: Optional[Any] = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Apply a binary tensor op and store the result in ``dst``.

    Args:
        dst: Destination tensor for the computed output.
        data1: Left-hand tensor operand.
        data2: Right-hand tensor operand.
        op: Patched beta2 operator token such as ``nl.add``.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    del name
    if not isinstance(op, SubOp):
        raise ValueError(f"Unsupported op: {op}")
    dst_buffer = _buffer_name(dst)
    lhs_buffer = _buffer_name(data1)
    rhs_buffer = _buffer_name(data2)
    _require(
        not (lhs_buffer == "psum" and rhs_buffer == "psum"),
        "tensor_tensor cannot read both inputs from PSUM",
    )
    if op.name == "power":
        engine = nisa.gpsimd_engine

    if engine == nisa.gpsimd_engine and "psum" in (dst_buffer, lhs_buffer, rhs_buffer):
        raise ValueError(
            "attempted to access PSUM with explicit (e.g. non-default engine arg)/implicit (e.g. op == nl.power) use of GpSimd Engine"
        )
    _require(
        (lhs_buffer, rhs_buffer)
        in (("sbuf", "sbuf"), ("sbuf", "psum"), ("psum", "sbuf")),
        "buffer combination for data1/data2 must be in sbuf/sbuf, sbuf/psum, or psum/sbuf",
    )
    lhs = np.asarray(_tensor_value(data1))
    rhs = np.asarray(_tensor_value(data2))
    if lhs.ndim < 2 or rhs.ndim < 2:
        raise ValueError(
            "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar"
        )
    lhs_pf_size = _partition_and_free(lhs)
    rhs_pf_size = _partition_and_free(rhs)
    dst_pf_size = _partition_and_free(dst)
    _require(
        lhs_pf_size == rhs_pf_size == dst_pf_size,
        "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar",
    )
    lhs = lhs.reshape(*lhs_pf_size)
    rhs = rhs.reshape(*rhs_pf_size)
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    lhs_dtype = getattr(data1, "dtype", lhs.dtype)
    rhs_dtype = getattr(data2, "dtype", rhs.dtype)
    if op.is_bitwise:
        _require(
            all(
                _is_integer_dtype(dtype) for dtype in (lhs_dtype, rhs_dtype, dst_dtype)
            ),
            "data and dst must be integer dtypes for bitvec operators",
        )
    result = op._run(lhs, rhs)
    result_value = result.data if isinstance(result, NDArray) else result
    return _store(dst, result_value.reshape(dst.shape))


def tensor_reduce(
    dst: NDArray,
    op: SubOp,
    data: NDArray,
    axis: int | list[int] | tuple[int, ...],
    negate: bool = False,
    keepdims: bool = False,
    name: str | None = None,
) -> NDArray:
    """Reduce a tensor along axes using add/max/min-like ops.

    Args:
        dst: Destination tensor receiving the reduced output.
        op: Patched reduction operator token.
        data: Source tensor to reduce.
        axis: Reduction axis or axes.
        negate: Whether to negate the reduced result.
        keepdims: Whether to preserve reduced dimensions as size-1 axes.
        name: Optional kernel op name retained for API parity.
    """
    del name
    if not isinstance(op, SubOp):
        raise ValueError(f"Unsupported op: {op}")
    source = np.asarray(_tensor_value(data))
    if isinstance(axis, list):
        axis = tuple(axis)
    axis_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
    if axis_tuple != (1,):
        raise ValueError("not a valid dim")
    data_par, _ = _partition_and_free(source)
    dst_par, _ = _partition_and_free(dst)
    _require(data_par == dst_par, "tensor_reduce partition dim mismatch")
    _require(data_par <= 128, "tensor_reduce partition dim > 128")
    exp_pf_size = _partition_and_free(np.sum(source, axis=axis_tuple))
    dst_pf_size = _partition_and_free(dst)
    if exp_pf_size != dst_pf_size:
        raise ValueError("cannot reduce free dim of data into free dim of dst")
    if not op.is_reducible:
        raise RuntimeError("Op is not a legal reduction op")
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    data_dtype = getattr(data, "dtype", source.dtype)
    if op.is_bitwise and not (
        _is_integer_dtype(data_dtype) and _is_integer_dtype(dst_dtype)
    ):
        raise ValueError("data and dst must be integer dtypes for bitvec operators")
    if negate and op.is_bitwise:
        raise ValueError("can only negate when reducing with an arithmetic operator")
    reduced = cast(Any, op._op).reduce(source, axis=axis_tuple, keepdims=keepdims)
    if negate:
        reduced = -np.asarray(reduced)
    reduced = np.asarray(reduced)
    if reduced.shape != dst.shape:
        _require(
            reduced.ndim == 1 and dst.shape == (reduced.shape[0], 1),
            "cannot reduce free dim of data into free dim of dst",
        )
        reduced = reduced.reshape(dst.shape)
    return _store(dst, reduced)


def _apply_tensor_scalar_op(
    data: TensorLike, op: SubOp, operand: TensorOrScalar, reverse: bool
) -> ArrayLike:
    """Apply tensor_scalar op using SubOp dispatch."""
    lhs, rhs = (operand, data) if reverse else (data, operand)
    applied = op._run(lhs, rhs)
    return applied.data if isinstance(applied, NDArray) else np.asarray(applied)


def tensor_scalar(
    dst: NDArray,
    data: NDArray,
    op0: SubOp,
    operand0: TensorOrScalar,
    reverse0: bool = False,
    op1: SubOp | None = None,
    operand1: TensorOrScalar | None = None,
    reverse1: bool = False,
    engine: Any = nisa.engine.unknown,
    name: str | None = None,
) -> NDArray:
    """Apply one or two tensor-scalar operations and write into ``dst``.

    Args:
        dst: Destination tensor for the computed output.
        data: Source tensor operand.
        op0: First patched beta2 operator token.
        operand0: Scalar or tensor operand paired with ``op0``.
        reverse0: Whether to reverse the argument order for ``op0``.
        op1: Optional second patched beta2 operator token.
        operand1: Optional scalar or tensor operand paired with ``op1``.
        reverse1: Whether to reverse the argument order for ``op1``.
        engine: Requested execution engine token.
        name: Optional kernel op name retained for API parity.
    """
    del name
    if not isinstance(op0, SubOp):
        raise ValueError(f"Unsupported op: {op0}")
    if not (op1 is None or isinstance(op1, SubOp)):
        raise ValueError(f"Unsupported op: {op1}")
    op0_bitwise = op0.is_bitwise
    op1_bitwise = op1.is_bitwise if op1 is not None else op0_bitwise
    _require(
        op1 is None or op0_bitwise == op1_bitwise,
        "bitvec and arithmetic ops can't be mixed",
    )
    _require(
        not (op0_bitwise and engine in (nisa.scalar_engine, nisa.gpsimd_engine)),
        "bitvec ops must run on Vector Engine",
    )
    _require(
        not (op0.name == "rsqrt" and engine == nisa.vector_engine),
        "rsqrt can only run on GpSimd/Scalar Engine",
    )
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    data_dtype = getattr(data, "dtype", np.asarray(_tensor_value(data)).dtype)
    if op0_bitwise:
        _require(
            _is_integer_dtype(data_dtype) and _is_integer_dtype(dst_dtype),
            "data must be int dtype for bitvec ops",
        )
    result = np.asarray(_tensor_value(data))
    result_pf_size = _partition_and_free(result)
    _require(
        result_pf_size == _partition_and_free(dst),
        "tensor_scalar dst must preserve partition/free shape",
    )
    for idx, (op, operand, reverse) in enumerate(
        (
            (op0, operand0, reverse0),
            (op1, operand1, reverse1),
        ),
    ):
        if op is None:
            continue
        _require(
            not (
                op0_bitwise
                and isinstance(operand, NDArray)
                and not _is_integer_dtype(getattr(operand, "dtype", operand.data.dtype))
            ),
            "operand0/1 dtype must match bitvec integer requirements",
        )
        assert operand is not None
        if isinstance(operand, NDArray):
            operand_arr = np.asarray(_tensor_value(operand))
            if operand_arr.ndim > 1 and operand_arr.shape[0] == result.shape[0]:
                operand_free = int(np.prod(operand_arr.shape[1:], dtype=np.int64))
                idx_str = "1st" if idx == 0 else "2nd"
                _require(
                    operand_free == 1,
                    f"{idx_str} Immediate pointer's number of elements per partition must be 1",
                )
        operand_value = _broadcast_tensor_scalar_operand(
            result,
            cast(TensorOrScalar, _tensor_value(operand)),
        )
        result = _apply_tensor_scalar_op(result, op, operand_value, reverse)
    return _store(dst, np.asarray(result).reshape(dst.shape))


def activation(
    dst: NDArray,
    op: SubOp | None,
    data: NDArray,
    bias: TensorOrScalar | None = None,
    scale: TensorOrScalar = 1.0,
    reduce_op: SubOp | None = None,
    reduce_res: NDArray | None = None,
    reduce_cmd: Any = nisa.reduce_cmd.idle,
    name: str | None = None,
) -> NDArray:
    """Apply an activation epilogue into ``dst``.

    Args:
        dst: Destination tensor receiving the activation output.
        op: Patched unary activation operator token.
        data: Source tensor to transform.
        bias: Optional scalar or per-partition bias operand.
        scale: Optional scalar or per-partition scale operand.
        reduce_op: Optional reduction operator token. Only ``nl.add`` is legal.
        reduce_res: Optional destination tensor for a companion reduction result.
        reduce_cmd: Unused beta2 reduction command token kept for API parity.
        name: Optional kernel op name retained for API parity.
    """
    del name
    if not isinstance(op, SubOp):
        raise ValueError(f"Unsupported op: {op}")
    _require(
        _buffer_name(dst) in ("sbuf", "psum")
        and _buffer_name(data) in ("sbuf", "psum"),
        "activation only supports SBUF/PSUM",
    )
    data_value = np.asarray(_tensor_value(data), dtype=np.float32)
    data_par, _ = _partition_and_free(data_value)
    dst_par, dst_free = _partition_and_free(dst)
    data_par, data_free = _partition_and_free(data_value)
    _require(dst_par == data_par, "dst/data partition dim mismatch")
    _require(dst_free == data_free, "dst/data free dim mismatch")
    _require(data_par <= 128, "data partition dim > 128")
    _require(
        reduce_op is None or reduce_op.name == "add",
        "activation reduce_op must be None or nl.add",
    )
    scale_value = _activation_operand(scale, data_value.shape, "scale")
    bias_value = _activation_operand(bias, data_value.shape, "bias")
    pre_act = data_value * scale_value + bias_value
    output = op._run(pre_act)
    output_value = output.data if isinstance(output, NDArray) else output
    _store(dst, np.asarray(output_value).reshape(dst.shape))
    if reduce_res is not None and reduce_op is not None:
        _require(reduce_res.shape == (data_par, 1), "reduce_res must have shape (P, 1)")
        reduced = np.sum(
            np.asarray(output_value, dtype=np.float32).reshape(data_par, -1),
            axis=1,
            keepdims=True,
        )
        _store(reduce_res, reduced)
    return dst


class Builder:
    """Tracks grid dimensions/index for the active interpreted kernel."""

    def __init__(self, grid_dims: Iterable[int] | None = None) -> None:
        """Initialize grid-tracking state for interpreted kernel launches.

        Args:
            grid_dims: Optional initial launch grid dimensions.
        """
        self.grid_dims = tuple(grid_dims) if grid_dims is not None else (1,)
        self.grid_idx = [0] * len(self.grid_dims)
        self.nc_version = nisa.nc_version.gen2
        self.fn: Callable[..., Any] | None = None

    def set_grid_dim(self, *grid_dims: int) -> None:
        """Update the active launch grid dimensions.

        Args:
            *grid_dims: New launch grid dimensions.
        """
        _require(
            len(grid_dims) == 1, "NKI Beta 2 currently only supports 1D SPMD grids"
        )
        self.grid_dims = tuple(grid_dims)
        self.grid_idx = [0] * len(self.grid_dims)

    def set_grid_idx(self, x: int, y: int = 0, z: int = 0) -> None:
        """Update the current program-id coordinates.

        Args:
            x: Grid index for axis 0.
            y: Grid index for axis 1.
            z: Grid index for axis 2.
        """
        del y, z
        self.grid_idx = [x]


nki_builder = Builder()


def program_id(axis: int) -> int:
    """Return the current program-id component for the requested axis.

    Args:
        axis: Program-id axis to query.
    """
    axis = int(axis)
    if 0 <= axis < len(nki_builder.grid_idx):
        return nki_builder.grid_idx[axis]
    raise ValueError(f"Invalid axis {axis} for {len(nki_builder.grid_idx)}D grid")


def nki_patch_lang(scope: Any = None) -> None:
    """Patch ``nl`` and ``nisa`` to use beta2 interpreter implementations.

    Args:
        scope: Optional patch scope that exposes ``set_attr`` for reversible
            patching.
    """
    set_attr = setattr if scope is None else scope.set_attr
    set_attr(nl, "ndarray", ndarray)
    set_attr(nl, "zeros", zeros)
    set_attr(nl, "copy", SubOp(np.copy, False, False))
    set_attr(nl, "add", SubOp(np.add, False, True))
    set_attr(nl, "subtract", SubOp(np.subtract, False, True))
    set_attr(nl, "multiply", SubOp(np.multiply, False, True))
    set_attr(nl, "maximum", SubOp(np.maximum, False, True))
    set_attr(nl, "minimum", SubOp(np.minimum, False, True))
    set_attr(nl, "power", SubOp(np.power, False, False))
    set_attr(nl, "logical_and", SubOp(np.logical_and, True, True))
    set_attr(nl, "logical_or", SubOp(np.logical_or, True, True))
    set_attr(nl, "logical_xor", SubOp(np.logical_xor, True, True))
    set_attr(nl, "bitwise_and", SubOp(np.bitwise_and, True, True))
    set_attr(nl, "bitwise_or", SubOp(np.bitwise_or, True, True))
    set_attr(nl, "bitwise_xor", SubOp(np.bitwise_xor, True, True))
    set_attr(nl, "sqrt", SubOp(np.sqrt, False, False))
    set_attr(nl, "rsqrt", SubOp(lambda v: 1.0 / np.sqrt(v), False, False, name="rsqrt"))
    set_attr(nl, "exp", SubOp(np.exp, False, False))
    set_attr(nl, "reciprocal", SubOp(np.reciprocal, False, False))
    set_attr(nl, "tile_size", tile_size)
    set_attr(nl, "sbuf", sbuf)
    set_attr(nl, "hbm", hbm)
    set_attr(nl, "psum", psum)
    set_attr(nl, "program_id", program_id)
    set_attr(nl, "affine_range", affine_range)
    set_attr(nl, "ds", ds)

    set_attr(nisa, "nc_matmul", nc_matmul)
    set_attr(nisa, "nc_transpose", nc_transpose)
    set_attr(nisa, "activation", activation)
    set_attr(nisa, "exponential", exponential)
    set_attr(nisa, "reciprocal", reciprocal)
    set_attr(nisa, "dma_copy", dma_copy)
    set_attr(nisa, "tensor_copy", tensor_copy)
    set_attr(nisa, "tensor_tensor", tensor_tensor)
    set_attr(nisa, "tensor_reduce", tensor_reduce)
    set_attr(nisa, "tensor_scalar", tensor_scalar)


def nki_unpatch_lang(scope: Any = None) -> None:
    """Restore beta2 language patches when the scope supports it.

    Args:
        scope: Optional patch scope that exposes ``restore``.
    """
    if scope is not None and hasattr(scope, "restore"):
        scope.restore()


class NKIBeta2InterpretedFunction:
    """Callable wrapper that executes NKI Beta 2 kernels with interpreter semantics."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        """Store the original kernel function.

        Args:
            fn: Kernel function to execute under beta2 interpreter semantics.
        """
        self.fn = fn

    def run(self, *args: Any, **kwargs: Any) -> None:
        """Run the wrapped kernel over a launch grid with tracing callbacks.

        Args:
            *args: Positional kernel arguments.
            **kwargs: Kernel keyword arguments, including optional ``grid``,
                ``warmup``, and ``client_manager`` controls.
        """
        grid_dims = kwargs.pop("grid", (1,))
        if isinstance(grid_dims, int):
            grid_dims = (grid_dims,)
        else:
            grid_dims = tuple(grid_dims)
        if not grid_dims:
            raise ValueError("Grid must have at least one dimension")

        nki_builder.grid_dims = tuple(int(dim) for dim in grid_dims)
        nki_builder.grid_idx = [0] * len(nki_builder.grid_dims)
        nki_builder.fn = self.fn

        kwargs.pop("warmup", None)
        client_manager = kwargs.pop("client_manager", None)
        if client_manager is not None:
            client_manager.grid_callback(grid_dims)

        exec_globals = cast(Any, self.fn).__globals__
        exec_globals["sbuf"] = sbuf
        exec_globals["hbm"] = hbm
        exec_globals["psum"] = psum
        CODE_KEYS.add(get_code_key(self.fn))

        kernel_args = tuple(
            arg
            if isinstance(arg, (NDArray, bool, int, float, str)) or arg is None
            else NDArray(value=arg)
            for arg in args
        )

        sig = inspect.signature(self.fn)
        bound = sig.bind(*kernel_args, **kwargs)
        bound.apply_defaults()
        for name, arg in bound.arguments.items():
            assert arg is None or isinstance(arg, (bool, int, float, str, NDArray))
            if client_manager is not None:
                client_manager.arg_callback(name, arg, arg)

        for grid_idx in itertools.product(*[range(dim) for dim in grid_dims]):
            if len(grid_idx) != len(nki_builder.grid_dims):
                raise ValueError(
                    "Grid index rank mismatch: got "
                    f"{len(grid_idx)}, expected {len(nki_builder.grid_dims)}"
                )
            nki_builder.grid_idx = [int(idx) for idx in grid_idx]
            if client_manager is not None:
                client_manager.grid_idx_callback(
                    tuple(int(idx) for idx in grid_idx) + (0,) * (3 - len(grid_idx))
                )
                if not client_manager.pre_run_callback(self.fn):
                    return
            self.fn(*kernel_args, **kwargs)
            if client_manager is not None and not client_manager.post_run_callback(
                self.fn
            ):
                return
