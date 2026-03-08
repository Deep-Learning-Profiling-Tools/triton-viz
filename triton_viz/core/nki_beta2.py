import inspect
import itertools
from typing import Optional

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


def _storage_dtype(dtype: DTypeLike):
    """Resolve a logical dtype to concrete NumPy storage dtype."""
    try:
        return STORAGE_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype: {dtype}") from exc


def _cast_value(value, dtype: DTypeLike):
    """Cast value to destination dtype storage using shared ml_dtypes mapping."""
    array = np.asarray(value)
    if dtype is None:
        return array
    return array.astype(_storage_dtype(dtype))


def _mark_defined(value):
    """Mark NDArray values as initialized after writes."""
    if isinstance(value, NDArray):
        value._defined = True


def _tensor_value(value):
    """Unwrap NDArray inputs and validate they are initialized."""
    if isinstance(value, NDArray):
        if not value._defined:
            raise RuntimeError(_ERR_UNDEFINED_USE)
        return value.data
    return value


def _as_scalar(value):
    """Convert scalar-like value into int."""
    array = np.asarray(_tensor_value(value))
    if array.size != 1:
        raise ValueError("Expected scalar value")
    return int(array.reshape(-1)[0])


def _dtype_str(dtype):
    """Return a stable string name for logical or storage dtypes."""
    if dtype is None:
        return "None"
    try:
        return np.dtype(_storage_dtype(dtype)).name
    except (TypeError, ValueError):
        try:
            return np.dtype(dtype).name
        except (TypeError, ValueError):
            return str(dtype)


def _buffer_name(value):
    """Return the interpreter buffer name for values with storage."""
    return getattr(value, "buffer", None) or "sbuf"


def _is_float32_dtype(dtype):
    """Return whether dtype is float32-like."""
    return _dtype_str(dtype) == "float32"


def _is_bfloat16_dtype(dtype):
    """Return whether dtype is bfloat16-like."""
    return _dtype_str(dtype) == "bfloat16"


def _is_integer_dtype(dtype):
    """Return whether dtype is integer/bool-like."""
    if _dtype_str(dtype) == "bool":
        return True
    try:
        try:
            resolved = _storage_dtype(dtype)
        except TypeError:
            resolved = np.dtype(dtype)
        return np.issubdtype(np.dtype(resolved), np.integer)
    except (TypeError, ValueError):
        return False


def _normalize_axis(axis):
    """Normalize scalar or list-like axis arguments into ints/tuples."""
    if isinstance(axis, list):
        return tuple(axis)
    return axis


def _partition_and_free(shape):
    """Return partition size and flattened free size."""
    shape = tuple(shape)
    free = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    return shape[0], free


def _op_name(op):
    """Return the normalized operation name."""
    return getattr(op, "__name__", str(op))


def _is_bitwise_op_name(name):
    """Return whether an op name is a documented bitvec operator."""
    prefixes = ("bitwise_", "logical_")
    return name.startswith(prefixes)


def _is_arithmetic_op_name(name):
    """Return whether an op name is a documented arithmetic operator."""
    return name in {
        "add",
        "subtract",
        "multiply",
        "maximum",
        "minimum",
        "power",
        "rsqrt",
    }


def _activation_operand(value, data_shape, label):
    """Normalize activation scale/bias into broadcastable arrays."""
    if value is None:
        return 0.0
    arr = np.asarray(_tensor_value(value), dtype=np.float32)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    if arr.shape != (data_shape[0], 1):
        raise ValueError(f"{label} must be a scalar or shape ({data_shape[0]}, 1)")
    return np.broadcast_to(
        arr.reshape(data_shape[0], 1), (data_shape[0], int(np.prod(data_shape[1:])))
    ).reshape(data_shape)


class NDArray:
    """Lightweight NumPy-backed tensor used by the NKI beta2 interpreter."""

    def __init__(
        self,
        buffer=None,
        shape=None,
        dtype=None,
        value=None,
        origin="raw",
        parent_ndim=None,
    ):
        self.buffer = buffer
        self.dtype = dtype
        self._origin = origin
        storage_shape = tuple(shape) if shape is not None else None
        storage_dtype = _storage_dtype(dtype)
        if value is None:
            assert storage_shape is not None
            self.data = np.zeros(storage_shape, dtype=storage_dtype)
            self._defined = True
        else:
            array = (
                _cast_value(value, dtype) if dtype is not None else np.asarray(value)
            )
            if storage_shape is not None and array.shape != storage_shape:
                array = array.reshape(storage_shape)
            self.data = array
            self._defined = True
        self._parent_ndim = self.data.ndim if parent_ndim is None else parent_ndim
        if self.dtype is None:
            self.dtype = self.data.dtype
        self._data_ptr = None

    @property
    def shape(self):
        """Return tensor shape."""
        return self.data.shape if self.data is not None else None

    @property
    def address(self):
        """Return host pointer integer for underlying storage."""
        if self._data_ptr is None:
            self._data_ptr = self.data.ctypes.data
        return self._data_ptr

    def data_ptr(self):
        """Return pointer address alias for frontend compatibility."""
        return self.address

    def stride(self):
        """Return byte strides."""
        return self.data.strides

    def element_size(self):
        """Return item size in bytes."""
        return getattr(self.dtype, "itemsize", self.data.dtype.itemsize)

    def cpu(self):
        """Return self to mimic framework tensor api."""
        return self

    def detach(self):
        """Return self to mimic framework tensor api."""
        return self

    def numpy(self):
        """Return underlying NumPy array."""
        return self.data

    def __repr__(self):
        return f"NDArray(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, keys):
        """Implement slicing operations for NDArray."""
        # if self._origin == "tensor":
        #    raise TypeError(_ERR_TENSOR_SUBSCRIPT)
        if self.data is None:
            raise AttributeError("NDArray has no value to slice")
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = []
        for axis, key in enumerate(keys):
            value = key.data if isinstance(key, NDArray) else key
            if isinstance(value, np.ndarray) and value.size == 1:
                value = value.reshape(-1)[0]
            if isinstance(value, (int, np.integer)):
                dim = self.data.shape[axis]
                index = int(value)
                if index < 0:
                    index += dim
                value = slice(index, index + 1)
            new_keys.append(value)
        sliced_value = self.data[tuple(new_keys)]
        return NDArray(
            value=sliced_value,
            buffer=self.buffer,
            origin="access",
            parent_ndim=self._parent_ndim,
        )

    def __setitem__(self, keys, value):
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
        target[...] = _cast_value(unwrapped, self.dtype)
        _mark_defined(self)
        return self

    def reshape(self, *args, **kwargs):
        """Return a reshaped tensor view."""
        return NDArray(value=self.data.reshape(*args), buffer=self.buffer, **kwargs)

    def __neg__(self):
        raise TypeError("cannot negate values of this type")

    def _raise_binary_op(self, *_args, **_kwargs):
        raise TypeError(_ERR_BINARY_TENSOR_OP)

    __and__ = __or__ = __xor__ = __lshift__ = __rshift__ = _raise_binary_op
    __rand__ = __ror__ = __rxor__ = __rlshift__ = __rrshift__ = _raise_binary_op
    __iand__ = __ior__ = __ixor__ = __ilshift__ = __irshift__ = _raise_binary_op


class Buffer:
    """NKI memory region abstraction used by sbuf/hbm/psum."""

    def __init__(self, buffer: str, shape=None, data=None):
        self.buffer = buffer
        self.shape = shape
        self.data = data
        if shape is not None and data is None:
            self.data = np.empty(shape=shape, dtype=np.uint8)

    def view(self, dtype, shape, *, origin="raw"):
        """Materialize a typed NDArray view over this buffer region."""
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


def _resolve_buffer(buffer):
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


def ndarray(shape, dtype, *, buffer=None, **kwargs):
    """Create an NDArray on a requested NKI buffer."""
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
    return resolved_buffer.view(dtype, shape, origin="tensor")


def zeros(shape, dtype, *, buffer=None, **kwargs):
    """Create a zero-initialized tensor."""
    if buffer is not None:
        raise TypeError(
            "unexpected keyword argument 'buffer' in builtinfunction 'builtin_lang_zeros'"
        )
    ret = ndarray(shape, dtype, **kwargs)
    ret.data.fill(0)
    _mark_defined(ret)
    return ret


def nc_transpose(dst, data, engine=nisa.engine.unknown, name=None):
    """Compute a transpose between partition and flattened free axes."""
    del name
    src_buffer = _buffer_name(data)
    dst_buffer = _buffer_name(dst)
    if engine == nisa.vector_engine:
        if src_buffer not in ("sbuf", "psum") or dst_buffer not in ("sbuf", "psum"):
            raise ValueError("vector engine nc_transpose only supports sbuf/psum")
    elif engine == nisa.tensor_engine:
        if src_buffer != "sbuf" or dst_buffer != "psum":
            raise ValueError("tensor engine nc_transpose requires sbuf -> psum")
    data_dtype = getattr(data, "dtype", np.asarray(_tensor_value(data)).dtype)
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    if _dtype_str(dst_dtype) != _dtype_str(data_dtype) and not (
        _is_bfloat16_dtype(dst_dtype) and _is_float32_dtype(data_dtype)
    ):
        raise ValueError("dst dtype must match data dtype")
    value = np.asarray(_tensor_value(data))
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    dst.data[...] = _cast_value(transposed.reshape(dst.shape), dst.dtype)
    _mark_defined(dst)
    return dst


def nc_matmul(
    dst,
    stationary,
    moving,
    is_stationary_onezero: bool = False,
    is_moving_onezero: bool = False,
    is_transpose: bool = False,
    tile_position=(),
    tile_size=(),
    perf_mode=nisa.matmul_perf_mode.none,
    name=None,
):
    """Compute dst += stationary.T @ moving."""
    del (
        is_stationary_onezero,
        is_moving_onezero,
        is_transpose,
        perf_mode,
        name,
    )
    if getattr(dst, "buffer", None) != "psum":
        raise ValueError("nc_matmul requires dst in psum")
    if getattr(stationary, "buffer", None) != "sbuf":
        raise ValueError("nc_matmul requires stationary in sbuf")
    if getattr(moving, "buffer", None) != "sbuf":
        raise ValueError("nc_matmul requires moving in sbuf")
    stationary_value = np.asarray(_tensor_value(stationary))
    moving_value = np.asarray(_tensor_value(moving))
    if stationary_value.shape[0] != moving_value.shape[0]:
        raise ValueError("partition dims of stationary and moving must match")
    if stationary_value.shape[0] > 128:
        raise ValueError("partition dim of stationary > 128")
    if stationary_value.shape[1] > 128:
        raise ValueError("free dim of stationary > 128")
    if moving_value.shape[1] > 512:
        raise ValueError("free dim of moving > 512")
    stationary_dtype = getattr(stationary, "dtype", stationary_value.dtype)
    moving_dtype = getattr(moving, "dtype", moving_value.dtype)
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    stationary_is_f32 = _is_float32_dtype(stationary_dtype)
    moving_is_f32 = _is_float32_dtype(moving_dtype)
    if stationary_is_f32 != moving_is_f32:
        raise ValueError("if one matmul operand is float32, both must be float32")
    if _dtype_str(dst_dtype) not in {"float32", "bfloat16"}:
        raise ValueError("dst must be float32 or bfloat16")
    has_tile_position = bool(tile_position)
    has_tile_size = bool(tile_size)
    if has_tile_position != has_tile_size:
        raise ValueError("both tile_position and tile_size must be supplied")
    if has_tile_size:
        row_size, col_size = map(int, tile_size)
        start_row, start_col = map(int, tile_position)
        if row_size > 128 or col_size > 128:
            raise ValueError("tile_size max size is (128, 128)")
        if stationary_value.shape[0] > row_size or stationary_value.shape[1] > col_size:
            raise ValueError("stationary tile size exceeds tile_size")
        if start_row % row_size != 0:
            raise ValueError("start row must be a multiple of row tile size")
        if start_col % col_size != 0:
            raise ValueError("start col must be a multiple of col tile size")
        if start_row > 128:
            raise ValueError("start row must not exceed 128")
        if start_col > 128:
            raise ValueError("start col must not exceed 128")
    result = stationary_value.T @ moving_value
    dst.data[...] = np.asarray(dst.data) + _cast_value(result, dst.dtype)
    _mark_defined(dst)
    return dst


def reciprocal(dst, data, name=None):
    """Compute elementwise reciprocal into destination tensor."""
    del name
    result = np.reciprocal(np.asarray(_tensor_value(data), dtype=np.float32))
    dst.data[...] = _cast_value(result, dst.dtype)
    _mark_defined(dst)
    return dst


def exponential(
    dst,
    src,
    max_value=0.0,
    reduce_res=None,
    reduce_cmd=nisa.reduce_cmd.idle,
    reduce_init=0.0,
):
    """Compute ``exp(src - max_value)`` into destination tensor."""
    del reduce_cmd, reduce_init
    current_nc = getattr(nki_builder.nc_version, "value", nki_builder.nc_version)
    min_nc = getattr(nisa.nc_version.gen4, "value", nisa.nc_version.gen4)
    if current_nc < min_nc:
        raise RuntimeError("exponential only supports neuron-core-v4 or newer")
    src_value = _tensor_value(src)
    max_value_data = _tensor_value(max_value)
    output = np.exp(
        np.asarray(src_value, dtype=np.float32)
        - np.asarray(max_value_data, dtype=np.float32)
    )
    dst.data[...] = _cast_value(output, dst.dtype)
    _mark_defined(dst)
    if reduce_res is not None:
        reduced = np.sum(
            np.asarray(output).reshape(output.shape[0], -1), axis=1, keepdims=True
        )
        reduce_res.data[...] = _cast_value(reduced, reduce_res.dtype)
        _mark_defined(reduce_res)
    return dst


def affine_range(start, stop=None, step=1):
    """Create a Python range from static or register-backed bounds."""
    start_v = _as_scalar(start) if isinstance(start, NDArray) else int(start)
    if stop is None:
        return range(start_v)
    stop_v = _as_scalar(stop) if isinstance(stop, NDArray) else int(stop)
    step_v = _as_scalar(step) if isinstance(step, NDArray) else int(step)
    return range(start_v, stop_v, step_v)


def ds(start, size):
    """Return an NKI-style dynamic slice using start/extent semantics."""
    begin = _as_scalar(start)
    extent = _as_scalar(size)
    return slice(begin, begin + extent)


class TileSize:
    """Minimal tile size constants used by beta2 example kernels."""

    pmax = 128
    gemm_stationary_fmax = 128
    gemm_moving_fmax = 512


tile_size = TileSize()


def _unwrap_operand(value):
    """Return NumPy/scalar payload from NDArray/scalar inputs."""
    return _tensor_value(value)


def _broadcast_tensor_scalar_operand(data, operand):
    """Broadcast tensor_scalar operand over free dimensions only."""
    data_arr = np.asarray(data)
    operand_arr = np.asarray(operand)
    if operand_arr.size == 1:
        return operand_arr
    if operand_arr.shape == data_arr.shape:
        return operand_arr
    if operand_arr.ndim != data_arr.ndim:
        raise ValueError("tensor_scalar only broadcasts free dimensions")
    if operand_arr.shape[0] != data_arr.shape[0]:
        raise ValueError("tensor_scalar only broadcasts free dimensions")
    target_shape = []
    for axis, (op_dim, data_dim) in enumerate(zip(operand_arr.shape, data_arr.shape)):
        if axis == 0:
            target_shape.append(op_dim)
            continue
        if op_dim not in (1, data_dim):
            raise ValueError("tensor_scalar only broadcasts free dimensions")
        target_shape.append(data_dim)
    return np.broadcast_to(operand_arr, tuple(target_shape))


def _wrap_result(result, *operands):
    """Wrap NumPy outputs as NDArray when needed."""
    if isinstance(result, np.ndarray) or any(
        isinstance(operand, NDArray) for operand in operands
    ):
        return NDArray(value=result)
    return result


def _raise_direct_tensor_math(x, y):
    """Reject direct nl binary math on beta2 tensors."""
    if isinstance(x, NDArray) or isinstance(y, NDArray):
        raise TypeError(_ERR_BINARY_TENSOR_OP)


def _binary_math_op(x, y, op, dtype=None):
    """Apply a binary math operation with NDArray-aware coercion."""
    _raise_direct_tensor_math(x, y)
    x_value = _unwrap_operand(x)
    y_value = _unwrap_operand(y)
    result = op(x_value, y_value)
    if dtype is not None:
        result = _cast_value(result, dtype)
    return _wrap_result(result, x, y)


def _unary_math_op(x, op, dtype=None):
    """Apply a unary math operation with NDArray-aware coercion."""
    x_value = _unwrap_operand(x)
    result = op(x_value)
    if dtype is not None:
        result = _cast_value(result, dtype)
    return _wrap_result(result, x)


def add(x, y, dtype=None):
    """Compute elementwise addition."""
    return _binary_math_op(x, y, np.add, dtype=dtype)


def subtract(x, y, dtype=None):
    """Compute elementwise subtraction."""
    return _binary_math_op(x, y, np.subtract, dtype=dtype)


def multiply(x, y, dtype=None):
    """Compute elementwise multiplication."""
    return _binary_math_op(x, y, np.multiply, dtype=dtype)


def maximum(x, y, dtype=None):
    """Compute elementwise maximum."""
    return _binary_math_op(x, y, np.maximum, dtype=dtype)


def minimum(x, y, dtype=None):
    """Compute elementwise minimum."""
    return _binary_math_op(x, y, np.minimum, dtype=dtype)


def power(x, y, dtype=None):
    """Compute elementwise power."""
    return _binary_math_op(x, y, np.power, dtype=dtype)


def logical_and(x, y, dtype=None):
    """Compute elementwise logical and."""
    return _binary_math_op(x, y, np.logical_and, dtype=dtype)


def logical_or(x, y, dtype=None):
    """Compute elementwise logical or."""
    return _binary_math_op(x, y, np.logical_or, dtype=dtype)


def logical_xor(x, y, dtype=None):
    """Compute elementwise logical xor."""
    return _binary_math_op(x, y, np.logical_xor, dtype=dtype)


def bitwise_and(x, y, dtype=None):
    """Compute elementwise bitwise and."""
    return _binary_math_op(x, y, np.bitwise_and, dtype=dtype)


def bitwise_or(x, y, dtype=None):
    """Compute elementwise bitwise or."""
    return _binary_math_op(x, y, np.bitwise_or, dtype=dtype)


def bitwise_xor(x, y, dtype=None):
    """Compute elementwise bitwise xor."""
    return _binary_math_op(x, y, np.bitwise_xor, dtype=dtype)


def sqrt(x, dtype=None):
    """Compute elementwise square root."""
    return _unary_math_op(x, np.sqrt, dtype=dtype)


def rsqrt(x, dtype=None):
    """Compute elementwise reciprocal square root."""
    return _unary_math_op(x, lambda v: 1.0 / np.sqrt(v), dtype=dtype)


def exp(x, dtype=None):
    """Compute elementwise maximum."""
    return _unary_math_op(x, np.exp, dtype=dtype)


def dma_copy(
    dst,
    src,
    dst_rmw_op=None,
    oob_mode=nisa.oob_mode.error,
    dge_mode=nisa.dge_mode.unknown,
    unique_indices=True,
    name=None,
) -> None:
    """Copy data from source tensor to destination tensor."""
    del oob_mode, dge_mode, unique_indices, name
    dst_buffer = dst.buffer or "sbuf"
    src_buffer = src.buffer or "sbuf"
    if dst_buffer not in ("hbm", "sbuf") or src_buffer not in ("hbm", "sbuf"):
        raise ValueError("dma_copy only supports hbm/sbuf source and destination")
    src_value = np.asarray(_tensor_value(src))
    if isinstance(dst, NDArray) and isinstance(src, NDArray):
        if dst.data.size != src.data.size:
            raise ValueError(_ERR_AP_MISMATCH)
        if (
            dst._origin == "access"
            and src._origin == "access"
            and dst._parent_ndim != src._parent_ndim
        ):
            raise ValueError(_ERR_AP_MISMATCH)
    if dst_rmw_op is None:
        dst.data[...] = _cast_value(src_value.reshape(dst.shape), dst.dtype)
        _mark_defined(dst)
        return dst
    if _op_name(dst_rmw_op) != "add":
        raise ValueError("only nl.add is supported for dma_copy dst_rmw_op")
    try:
        merged = dst_rmw_op(dst.data, src_value)
    except TypeError:
        merged = dst_rmw_op(dst.data, y=src_value)
    merged_value = merged.data if isinstance(merged, NDArray) else merged
    dst.data[...] = _cast_value(merged_value, dst.dtype)
    _mark_defined(dst)
    return dst


def tensor_copy(dst, src, engine=nisa.engine.unknown, name=None):
    """Copy tensor data within on-chip memory."""
    del name
    dst_buffer = _buffer_name(dst)
    src_buffer = _buffer_name(src)
    if dst_buffer not in ("sbuf", "psum") or src_buffer not in ("sbuf", "psum"):
        raise ValueError("tensor_copy is on-chip only")
    current_nc = getattr(nki_builder.nc_version, "value", nki_builder.nc_version)
    gen2 = getattr(nisa.nc_version.gen2, "value", nisa.nc_version.gen2)
    if engine == nisa.scalar_engine and current_nc <= gen2:
        raise ValueError("scalar engine tensor_copy is unsupported on nc-v2")
    if engine == nisa.gpsimd_engine and "psum" in (dst_buffer, src_buffer):
        raise ValueError("gpsimd tensor_copy cannot access psum")
    src_value = np.asarray(_tensor_value(src))
    if src_value.size != dst.data.size:
        raise ValueError(_ERR_AP_MISMATCH)
    dst.data[...] = _cast_value(src_value.reshape(dst.shape), dst.dtype)
    _mark_defined(dst)
    return dst


def tensor_tensor(
    dst,
    data1,
    data2,
    op,
    engine: Optional[nisa.engine] = nisa.engine.unknown,
    name=None,
):
    """Apply a binary tensor op and store the result in dst."""
    del name
    # if isinstance(dst, NDArray) and dst._origin == "tensor":
    #    raise TypeError(
    #        "failed to resolve an argument 'dst', expecting tensor access, got 'tensor'"
    #    )
    # if isinstance(data2, NDArray) and data2._origin == "tensor":
    #    raise TypeError(
    #        "failed to resolve an argument 'data2', expecting tensor access, got 'tensor'"
    #    )
    op_name = _op_name(op)
    dst_buffer = _buffer_name(dst)
    lhs_buffer = _buffer_name(data1)
    rhs_buffer = _buffer_name(data2)
    if lhs_buffer == "psum" and rhs_buffer == "psum":
        raise ValueError("tensor_tensor cannot read both inputs from psum")
    if op_name == "power" and (
        engine == nisa.gpsimd_engine
        and "psum" in (dst_buffer, lhs_buffer, rhs_buffer)
        or dst_buffer == "psum"
    ):
        raise ValueError("nl.power uses Gpsimd, which cannot access PSUM")
    lhs = np.asarray(_tensor_value(data1))
    rhs = np.asarray(_tensor_value(data2))
    if lhs.ndim < 2 or rhs.ndim < 2:
        raise ValueError(
            "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar"
        )
    if lhs.size != rhs.size:
        raise ValueError(_ERR_AP_MISMATCH)
    if lhs.shape != dst.shape:
        if (
            lhs.ndim < 2
            or lhs.shape[0] != dst.shape[0]
            or lhs.size != np.prod(dst.shape)
        ):
            raise ValueError(
                "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar"
            )
        lhs = lhs.reshape(dst.shape)
    if rhs.shape != dst.shape:
        if (
            rhs.ndim < 2
            or rhs.shape[0] != dst.shape[0]
            or rhs.size != np.prod(dst.shape)
        ):
            raise ValueError(
                "tensor_tensor doesn't broadcast scalars/vectors; use tensor_scalar"
            )
        rhs = rhs.reshape(dst.shape)
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    lhs_dtype = getattr(data1, "dtype", lhs.dtype)
    rhs_dtype = getattr(data2, "dtype", rhs.dtype)
    if _is_bitwise_op_name(op_name):
        if not all(
            _is_integer_dtype(dtype) for dtype in (lhs_dtype, rhs_dtype, dst_dtype)
        ):
            raise ValueError("data and dst must be integer dtypes for bitvec operators")
    try:
        result = op(lhs, rhs)
    except TypeError:
        try:
            result = op(lhs, y=rhs)
        except Exception:
            result = None
    except Exception:
        result = None
    if result is None:
        fallback_ops = {
            "add": np.add,
            "subtract": np.subtract,
            "sub": np.subtract,
            "multiply": np.multiply,
            "mul": np.multiply,
            "divide": np.divide,
            "div": np.divide,
            "maximum": np.maximum,
            "minimum": np.minimum,
            "power": np.power,
            "bitwise_and": np.bitwise_and,
            "bitwise_or": np.bitwise_or,
            "bitwise_xor": np.bitwise_xor,
            "logical_and": np.logical_and,
            "logical_or": np.logical_or,
            "logical_xor": np.logical_xor,
        }
        matched = next((fn for key, fn in fallback_ops.items() if key in op_name), None)
        if matched is None:
            raise RuntimeError(
                f"Unsupported tensor op outside kernel context: {op_name}"
            )
        result = matched(lhs, rhs)
    result_value = result.data if isinstance(result, NDArray) else result
    dst.data[...] = _cast_value(result_value, dst.dtype)
    _mark_defined(dst)
    return dst


def tensor_reduce(
    dst,
    op,
    data,
    axis,
    negate: bool = False,
    keepdims: bool = False,
    name=None,
):
    """Reduce a tensor along axes using add/max/min-like ops."""
    del name
    source = np.asarray(_tensor_value(data))
    axis = _normalize_axis(axis)
    axis_tuple = (axis,) if isinstance(axis, int) else tuple(axis)
    if not axis_tuple or axis_tuple[0] != 1:
        raise ValueError("tensor_reduce must start reducing over axis 1")
    if any(b - a != 1 for a, b in zip(axis_tuple, axis_tuple[1:])):
        raise ValueError("tensor_reduce axes must increase consecutively")
    if len(axis_tuple) > 4:
        raise ValueError("tensor_reduce must reduce on <= 4 dimensions")
    data_par, _ = _partition_and_free(source.shape)
    dst_par, _ = _partition_and_free(dst.shape)
    if data_par != dst_par:
        raise ValueError("tensor_reduce partition dim mismatch")
    if data_par > 128:
        raise ValueError("tensor_reduce partition dim > 128")
    expected_shape = np.sum(source, axis=axis_tuple, keepdims=keepdims).shape
    if keepdims:
        if expected_shape != dst.shape:
            raise ValueError("cannot reduce free dim of data into free dim of dst")
    else:
        exp_par, exp_free = _partition_and_free(expected_shape)
        dst_par, dst_free = _partition_and_free(dst.shape)
        if exp_par != dst_par or exp_free != dst_free:
            raise ValueError("cannot reduce free dim of data into free dim of dst")
    op_name = _op_name(op)
    reduce_ops = {
        "add": np.add.reduce,
        "subtract": np.subtract.reduce,
        "multiply": np.multiply.reduce,
        "maximum": np.maximum.reduce,
        "minimum": np.minimum.reduce,
        "bitwise_and": np.bitwise_and.reduce,
        "bitwise_or": np.bitwise_or.reduce,
        "bitwise_xor": np.bitwise_xor.reduce,
        "logical_and": np.logical_and.reduce,
        "logical_or": np.logical_or.reduce,
        "logical_xor": np.logical_xor.reduce,
    }
    reducer = reduce_ops.get(op_name)
    if reducer is None:
        raise RuntimeError(f"Unsupported tensor_reduce op: {op_name}")
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    data_dtype = getattr(data, "dtype", source.dtype)
    if _is_bitwise_op_name(op_name) and not (
        _is_integer_dtype(data_dtype) and _is_integer_dtype(dst_dtype)
    ):
        raise ValueError("data and dst must be integer dtypes for bitvec operators")
    if negate and _is_bitwise_op_name(op_name):
        raise ValueError("can only negate when reducing with an arithmetic operator")
    reduced = reducer(source, axis=axis_tuple, keepdims=keepdims)
    if negate:
        reduced = -np.asarray(reduced)
    if reduced.shape != dst.shape:
        reduced = np.asarray(reduced).reshape(dst.shape)
    dst.data[...] = _cast_value(reduced, dst.dtype)
    _mark_defined(dst)
    return dst


def _apply_tensor_scalar_op(data, op, operand, reverse):
    """Apply tensor_scalar op, including unary ops such as nl.rsqrt."""
    op_name = _op_name(op)
    param_names = tuple(inspect.signature(op).parameters)
    if param_names == ("x", "dtype"):
        applied = op(operand if reverse else data)
        return applied.data if isinstance(applied, NDArray) else np.asarray(applied)

    lhs, rhs = (operand, data) if reverse else (data, operand)
    try:
        applied = op(lhs, rhs)
    except TypeError:
        try:
            applied = op(lhs, y=rhs)
        except Exception:
            applied = None
    except Exception:
        applied = None
    if applied is None:
        if "mul" in op_name:
            applied = np.multiply(lhs, rhs)
        elif "add" in op_name:
            applied = np.add(lhs, rhs)
        elif "sub" in op_name:
            applied = np.subtract(lhs, rhs)
        elif "div" in op_name:
            applied = np.divide(lhs, rhs)
        elif "logical_and" in op_name:
            applied = np.logical_and(lhs, rhs)
        elif "logical_or" in op_name:
            applied = np.logical_or(lhs, rhs)
        elif "logical_xor" in op_name:
            applied = np.logical_xor(lhs, rhs)
        elif "bitwise_and" in op_name:
            applied = np.bitwise_and(lhs, rhs)
        elif "bitwise_or" in op_name:
            applied = np.bitwise_or(lhs, rhs)
        elif "bitwise_xor" in op_name:
            applied = np.bitwise_xor(lhs, rhs)
        else:
            raise RuntimeError(f"Unsupported tensor_scalar op: {op_name}")
    return applied.data if isinstance(applied, NDArray) else np.asarray(applied)


def tensor_scalar(
    dst,
    data,
    op0,
    operand0,
    reverse0: bool = False,
    op1=None,
    operand1=None,
    reverse1: bool = False,
    engine=nisa.engine.unknown,
    name=None,
):
    """Apply one or two tensor-scalar operations and write into dst."""
    del name
    op0_name = _op_name(op0)
    op1_name = _op_name(op1) if op1 is not None else None
    op0_bitwise = _is_bitwise_op_name(op0_name)
    op1_bitwise = _is_bitwise_op_name(op1_name) if op1_name is not None else op0_bitwise
    if op1 is not None and op0_bitwise != op1_bitwise:
        raise ValueError("bitvec and arithmetic ops can't be mixed")
    if op0_bitwise and engine in (nisa.scalar_engine, nisa.gpsimd_engine):
        raise ValueError("bitvec ops must run on vector engine")
    if op0_name == "rsqrt" and engine == nisa.vector_engine:
        raise ValueError("rsqrt can only run on GpSimd/Scalar Engine")
    dst_dtype = getattr(dst, "dtype", dst.data.dtype)
    data_dtype = getattr(data, "dtype", np.asarray(_tensor_value(data)).dtype)
    if op0_bitwise:
        if not (_is_integer_dtype(data_dtype) and _is_integer_dtype(dst_dtype)):
            raise ValueError("data must be int dtype for bitvec ops")
    result = np.asarray(_tensor_value(data))
    for op, operand, reverse in (
        (op0, operand0, reverse0),
        (op1, operand1, reverse1),
    ):
        if op is None:
            continue
        if (
            op0_bitwise
            and isinstance(operand, NDArray)
            and not _is_integer_dtype(getattr(operand, "dtype", operand.data.dtype))
        ):
            raise ValueError("operand0/1 dtype must match bitvec integer requirements")
        operand_value = _broadcast_tensor_scalar_operand(
            result, _unwrap_operand(operand)
        )
        result = _apply_tensor_scalar_op(result, op, operand_value, reverse)
    dst.data[...] = _cast_value(result, dst.dtype)
    _mark_defined(dst)
    return dst


def activation(
    dst,
    op,
    data,
    bias=None,
    scale=1.0,
    reduce_op=None,
    reduce_res=None,
    reduce_cmd=nisa.reduce_cmd.idle,
    name=None,
):
    """Apply an activation epilogue into dst."""
    del name
    if _buffer_name(data) not in ("sbuf", "psum") or _buffer_name(dst) not in (
        "sbuf",
        "psum",
    ):
        raise ValueError("activation only supports sbuf/psum")
    data_value = np.asarray(_tensor_value(data), dtype=np.float32)
    data_par, data_free = _partition_and_free(data_value.shape)
    dst_par, dst_free = _partition_and_free(dst.shape)
    if data_par != dst_par:
        raise ValueError("dst/data partition dim mismatch")
    if data_free != dst_free:
        raise ValueError("dst/data free dim mismatch")
    if data_par > 128:
        raise ValueError("data partition dim > 128")
    if reduce_op is not None and _op_name(reduce_op) != "add":
        raise ValueError("activation reduce_op must be None or nl.add")
    scale_value = _activation_operand(scale, data_value.shape, "scale")
    bias_value = _activation_operand(bias, data_value.shape, "bias")
    pre_act = data_value * scale_value + bias_value
    op_name = _op_name(op)
    if op_name in {"copy", "_identity"}:
        output = pre_act
    else:
        output = op(pre_act)
    output_value = output.data if isinstance(output, NDArray) else output
    dst.data[...] = _cast_value(np.asarray(output_value).reshape(dst.shape), dst.dtype)
    _mark_defined(dst)
    if reduce_res is not None:
        reduced = np.sum(
            np.asarray(output_value).reshape(data_par, -1), axis=1, keepdims=True
        )
        reduce_res.data[...] = _cast_value(reduced, reduce_res.dtype)
        _mark_defined(reduce_res)
    return dst


class Builder:
    """Tracks grid dimensions/index for the active interpreted kernel."""

    def __init__(self, grid_dims=None):
        self.grid_dims = tuple(grid_dims) if grid_dims is not None else (1,)
        self.grid_idx = [0] * len(self.grid_dims)
        self.nc_version = nisa.nc_version.gen4
        self.fn = None

    def set_grid_dim(self, *grid_dims):
        """Update active grid dimensions."""
        self.grid_dims = tuple(grid_dims)
        self.grid_idx = [0] * len(self.grid_dims)

    def set_grid_idx(self, x, y, z):
        """Update current program id coordinates."""
        self.grid_idx = [x, y, z][: len(self.grid_dims)]


nki_builder = Builder()


def program_id(axis):
    """Return current program id component for the requested axis."""
    axis = int(axis)
    if 0 <= axis < len(nki_builder.grid_idx):
        return nki_builder.grid_idx[axis]
    raise ValueError(f"Invalid axis {axis} for {len(nki_builder.grid_idx)}D grid")


def nki_patch_lang(scope=None):
    """Patch nl/nisa APIs to point at beta2 interpreter implementations."""
    set_attr = setattr if scope is None else scope.set_attr
    set_attr(nl, "ndarray", ndarray)
    set_attr(nl, "zeros", zeros)
    set_attr(nl, "add", add)
    set_attr(nl, "subtract", subtract)
    set_attr(nl, "multiply", multiply)
    set_attr(nl, "maximum", maximum)
    set_attr(nl, "minimum", minimum)
    set_attr(nl, "power", power)
    set_attr(nl, "logical_and", logical_and)
    set_attr(nl, "logical_or", logical_or)
    set_attr(nl, "logical_xor", logical_xor)
    set_attr(nl, "bitwise_and", bitwise_and)
    set_attr(nl, "bitwise_or", bitwise_or)
    set_attr(nl, "bitwise_xor", bitwise_xor)
    set_attr(nl, "sqrt", sqrt)
    set_attr(nl, "rsqrt", rsqrt)
    set_attr(nl, "exp", exp)
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


nki_unpatch_lang = (
    lambda scope=None: scope.restore()
    if scope is not None and hasattr(scope, "restore")
    else None
)


class NKIBeta2InterpretedFunction:
    """Callable wrapper that executes NKI Beta 2 kernels with interpreter semantics."""

    def __init__(self, fn):
        """Store the original kernel function."""
        self.fn = fn

    def run(self, *args, **kwargs):
        """Run the wrapped kernel over a launch grid with tracing callbacks."""
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

        exec_globals = self.fn.__globals__
        exec_globals["sbuf"] = sbuf
        exec_globals["hbm"] = hbm
        exec_globals["psum"] = psum
        CODE_KEYS.add(get_code_key(self.fn))

        args = [
            arg
            if isinstance(arg, (NDArray, bool, int, float, str)) or arg is None
            else NDArray(value=arg)
            for arg in args
        ]

        sig = inspect.signature(self.fn)
        bound = sig.bind(*args, **kwargs)
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
            self.fn(*args, **kwargs)
            if client_manager is not None and not client_manager.post_run_callback(
                self.fn
            ):
                return
