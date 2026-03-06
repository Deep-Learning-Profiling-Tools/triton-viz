import inspect
import itertools

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
        if self._origin == "tensor":
            raise TypeError(_ERR_TENSOR_SUBSCRIPT)
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

    __add__ = __sub__ = __mul__ = __truediv__ = _raise_binary_op
    __radd__ = __rsub__ = __rmul__ = __rtruediv__ = _raise_binary_op
    __iadd__ = __isub__ = __imul__ = __itruediv__ = _raise_binary_op


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


def nc_transpose(dst, data, engine=None, name=None):
    """Compute a transpose between partition and flattened free axes."""
    del engine, name
    value = np.asarray(_tensor_value(data))
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    dst.data[...] = _cast_value(transposed.reshape(dst.shape), dst.dtype)
    _mark_defined(dst)
    return dst


def nc_matmul(
    dst,
    stationary,
    moving,
    is_stationary_onezero=False,
    is_moving_onezero=False,
    is_transpose=False,
    tile_position=(),
    tile_size=(),
    psum_accumulate_flag=3,
    perf_mode=None,
    name=None,
):
    """Compute dst += stationary.T @ moving."""
    del (
        is_stationary_onezero,
        is_moving_onezero,
        is_transpose,
        tile_position,
        tile_size,
        psum_accumulate_flag,
        perf_mode,
        name,
    )
    if getattr(dst, "buffer", None) != "psum":
        raise ValueError("nc_matmul requires dst in psum")
    if getattr(stationary, "buffer", None) != "sbuf":
        raise ValueError("nc_matmul requires stationary in sbuf")
    if getattr(moving, "buffer", None) != "sbuf":
        raise ValueError("nc_matmul requires moving in sbuf")
    result = np.asarray(_tensor_value(stationary)).T @ np.asarray(_tensor_value(moving))
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


def _binary_math_op(lhs, rhs, op):
    """Apply a binary math operation with NDArray-aware coercion."""
    lhs_value = _unwrap_operand(lhs)
    rhs_value = _unwrap_operand(rhs)
    result = op(lhs_value, rhs_value)
    return _wrap_result(result, lhs, rhs)


def _unary_math_op(x, op):
    """Apply a unary math operation with NDArray-aware coercion."""
    x_value = _unwrap_operand(x)
    result = op(x_value)
    return _wrap_result(result, x)


def add(lhs, rhs):
    """Compute elementwise addition."""
    return _binary_math_op(lhs, rhs, np.add)


def subtract(lhs, rhs):
    """Compute elementwise subtraction."""
    return _binary_math_op(lhs, rhs, np.subtract)


def multiply(lhs, rhs):
    """Compute elementwise multiplication."""
    return _binary_math_op(lhs, rhs, np.multiply)


def maximum(lhs, rhs):
    """Compute elementwise maximum."""
    return _binary_math_op(lhs, rhs, np.maximum)


def sqrt(x):
    """Compute elementwise square root."""
    return _unary_math_op(x, np.sqrt)


def rsqrt(x):
    """Compute elementwise reciprocal square root."""
    return _unary_math_op(x, lambda v: 1.0 / np.sqrt(v))


def dma_copy(dst, src, dst_rmw_op=None, oob_mode=None, dge_mode=None, name=None):
    """Copy data from source tensor to destination tensor."""
    del oob_mode, dge_mode, name
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
    try:
        merged = dst_rmw_op(dst.data, src_value)
    except TypeError:
        merged = dst_rmw_op(dst.data, y=src_value)
    merged_value = merged.data if isinstance(merged, NDArray) else merged
    dst.data[...] = _cast_value(merged_value, dst.dtype)
    _mark_defined(dst)
    return dst


def tensor_copy(dst, src, engine=None, name=None):
    """Copy tensor data within on-chip memory."""
    del engine, name
    src_value = np.asarray(_tensor_value(src))
    if src_value.size != dst.data.size:
        raise ValueError(_ERR_AP_MISMATCH)
    dst.data[...] = _cast_value(src_value.reshape(dst.shape), dst.dtype)
    _mark_defined(dst)
    return dst


def tensor_tensor(dst, data1, data2, op, engine=None, name=None):
    """Apply a binary tensor op and store the result in dst."""
    del engine, name
    if isinstance(dst, NDArray) and dst._origin == "tensor":
        raise TypeError(
            "failed to resolve an argument 'dst', expecting tensor access, got 'tensor'"
        )
    if isinstance(data2, NDArray) and data2._origin == "tensor":
        raise TypeError(
            "failed to resolve an argument 'data2', expecting tensor access, got 'tensor'"
        )
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
    op_name = getattr(op, "__name__", str(op))
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


def tensor_reduce(dst, op, data, axis=None, keepdims=False, engine=None, name=None):
    """Reduce a tensor along axes using add/max/min-like ops."""
    del engine, name
    source = np.asarray(_tensor_value(data))
    op_name = getattr(op, "__name__", str(op))
    if "max" in op_name:
        reduced = np.max(source, axis=axis, keepdims=keepdims)
    elif "min" in op_name:
        reduced = np.min(source, axis=axis, keepdims=keepdims)
    elif "add" in op_name or "sum" in op_name:
        reduced = np.sum(source, axis=axis, keepdims=keepdims)
    else:
        try:
            reduced = op.reduce(source, axis=axis, keepdims=keepdims)
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError(f"Unsupported tensor_reduce op: {op_name}") from exc
    dst.data[...] = _cast_value(reduced, dst.dtype)
    _mark_defined(dst)
    return dst


def tensor_scalar(
    dst,
    data,
    op0,
    operand0,
    op1=None,
    operand1=None,
    engine=None,
    name=None,
):
    """Apply one or two tensor-scalar operations and write into dst."""
    del engine, name
    result = np.asarray(_tensor_value(data))
    for op, operand in ((op0, operand0), (op1, operand1)):
        if op is None:
            continue
        operand_value = _broadcast_tensor_scalar_operand(
            result, _unwrap_operand(operand)
        )
        op_name = getattr(op, "__name__", str(op))
        try:
            applied = op(result, operand_value)
        except TypeError:
            try:
                applied = op(result, y=operand_value)
            except Exception:
                applied = None
        except Exception:
            applied = None
        if applied is None:
            if "mul" in op_name:
                applied = np.multiply(result, operand_value)
            elif "add" in op_name:
                applied = np.add(result, operand_value)
            elif "sub" in op_name:
                applied = np.subtract(result, operand_value)
            elif "div" in op_name:
                applied = np.divide(result, operand_value)
            else:
                raise RuntimeError(f"Unsupported tensor_scalar op: {op_name}")
        result = applied.data if isinstance(applied, NDArray) else np.asarray(applied)
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
    reduce_cmd=None,
    name=None,
):
    """Apply an activation epilogue into dst."""
    del reduce_op, reduce_res, reduce_cmd, name
    if bias is not None and not isinstance(bias, NDArray):
        raise TypeError("activation bias must be a tensor")
    in_data = _tensor_value(data)
    in_bias = 0 if bias is None else _tensor_value(bias)
    output = op(in_data * scale + in_bias)
    output_value = output.data if isinstance(output, NDArray) else output
    dst.data[...] = _cast_value(output_value, dst.dtype)
    _mark_defined(dst)
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
    set_attr(nl, "sqrt", sqrt)
    set_attr(nl, "rsqrt", rsqrt)
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
                client_manager.grid_idx_callback(grid_idx)
                if not client_manager.pre_run_callback(self.fn):
                    return
            self.fn(*args, **kwargs)
            if client_manager is not None and not client_manager.post_run_callback(
                self.fn
            ):
                return
