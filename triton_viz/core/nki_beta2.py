import inspect
import itertools

import numpy as np

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


FLOAT32_STORAGE_DTYPES = {
    "bfloat16": (8, 7, False),
    "float8_e4m3": (4, 3, False),
    "float8_e5m2": (5, 2, False),
    "float8_e4m3fn": (4, 3, True),
    "float8_e5m2fn": (5, 2, True),
    "float4_e2m1fn": (2, 1, True),
}
for _dtype_name, _spec in tuple(FLOAT32_STORAGE_DTYPES.items()):
    if hasattr(nl, _dtype_name):
        FLOAT32_STORAGE_DTYPES[getattr(nl, _dtype_name)] = _spec


def _quantize_binary_float(value, exp_bits, mant_bits, finite_only=False):
    """Quantize float32 data to a reduced binary format."""
    x = np.asarray(value, dtype=np.float32)
    abs_x = np.abs(x)
    bias = (1 << (exp_bits - 1)) - 1
    emax = ((1 << exp_bits) - (1 if finite_only else 2)) - bias
    emin = 1 - bias
    max_finite = np.float32((2.0 - 2.0**-mant_bits) * 2.0**emax)
    out = abs_x.copy()
    mask = np.isfinite(x) & (abs_x > 0)
    if np.any(mask):
        exp = np.maximum(np.floor(np.log2(abs_x[mask])), emin)
        step = np.exp2(exp - mant_bits).astype(np.float32)
        out[mask] = np.rint(abs_x[mask] / step) * step
    if finite_only:
        out = np.minimum(out, max_finite)
    return np.copysign(out, x)


def _dtype_spec(dtype):
    """Return quantization spec for logical float32-backed dtypes."""
    return FLOAT32_STORAGE_DTYPES.get(
        dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dtype, "name", None))
    )


def _cast_value(value, dtype):
    """Cast/quantize value to destination logical dtype storage."""
    spec = _dtype_spec(dtype)
    if spec is not None:
        exp_bits, mant_bits, finite_only = spec
        quantized = _quantize_binary_float(
            value, exp_bits, mant_bits, finite_only=finite_only
        )
        return np.asarray(quantized, dtype=np.float32)
    return np.asarray(value, dtype=dtype)


def _as_scalar(value):
    """Convert scalar-like value into int."""
    value = value.data if isinstance(value, NDArray) else value
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected scalar value")
    return int(array.reshape(-1)[0])


class NDArray:
    """Lightweight NumPy-backed tensor used by the NKI beta2 interpreter."""

    def __init__(self, buffer=None, name="", shape=None, dtype=None, value=None):
        self.buffer = buffer
        self.name = name
        self.dtype = dtype
        storage_shape = tuple(shape) if shape is not None else None
        storage_dtype = np.float32 if _dtype_spec(dtype) is not None else dtype
        if value is None:
            assert storage_shape
            self.data = np.ndarray(storage_shape, dtype=storage_dtype)
        else:
            array = (
                _cast_value(value, dtype) if dtype is not None else np.asarray(value)
            )
            if storage_shape is not None and array.shape != storage_shape:
                array = array.reshape(storage_shape)
            self.data = array
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
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def __getitem__(self, keys):
        """Implement slicing operations for NDArray."""
        if self.data is None:
            raise AttributeError("NDArray has no value to slice")
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        sliced_value = self.data[tuple(new_keys)]
        return NDArray(
            value=sliced_value, name=f"{self.name}_slice", buffer=self.buffer
        )

    def __setitem__(self, keys, value):
        """Assign values into a sliced region."""
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        unwrapped = value.data if isinstance(value, NDArray) else value
        self.data[tuple(new_keys)] = _cast_value(unwrapped, self.dtype)
        return self

    def reshape(self, *args, **kwargs):
        """Return a reshaped tensor view."""
        return NDArray(
            value=self.data.reshape(*args),
            name=f"{self.name}_reshape",
            buffer=self.buffer,
            **kwargs,
        )

    def broadcast_to(self, *args, **kwargs):
        """Return a broadcasted tensor view."""
        return NDArray(
            value=np.broadcast_to(self.data, *args),
            name=f"{self.name}_broadcast_to",
            buffer=self.buffer,
            **kwargs,
        )


class Buffer:
    """NKI memory region abstraction used by sbuf/hbm/psum."""

    def __init__(self, buffer: str, size=None, data=None):
        self.buffer = buffer
        self.size = size
        self.data = data
        if size is not None and data is None:
            self.data = np.empty(shape=size, dtype=np.uint8)

    def view(self, dtype, size):
        """Materialize a typed NDArray view over this buffer region."""
        probe = NDArray(buffer=self.buffer, dtype=dtype, shape=size)
        if self.data is None:
            return probe
        if self.data.nbytes != probe.data.nbytes:
            raise ValueError(
                "Buffer size mismatch: have "
                f"{self.data.nbytes} bytes, expected {probe.data.nbytes}"
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


def ndarray(shape, dtype, *, buffer=None, name=None, **kwargs):
    """Create an NDArray on a requested NKI buffer."""
    resolved_buffer = _resolve_buffer(buffer)
    if "value" in kwargs:
        return NDArray(
            buffer=resolved_buffer.buffer,
            name=name,
            shape=shape,
            dtype=dtype,
            value=kwargs["value"],
        )
    ret = resolved_buffer.view(dtype, shape)
    ret.name = name
    return ret


def zeros(shape, dtype, *, buffer=None, name=None, **kwargs):
    """Create a zero-initialized tensor."""
    ret = ndarray(shape, dtype, buffer=buffer, name=name, **kwargs)
    ret.data.fill(0)
    return ret


def nc_transpose(dst, data, engine=None, name=None):
    """Compute a transpose between partition and flattened free axes."""
    del engine, name
    value = np.asarray(data.data)
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    dst.data[...] = _cast_value(transposed.reshape(dst.shape), dst.dtype)
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
    result = np.asarray(stationary.data).T @ np.asarray(moving.data)
    dst.data += _cast_value(result, dst.dtype)
    return dst


def reciprocal(dst, data, name=None):
    """Compute elementwise reciprocal into destination tensor."""
    del name
    result = np.reciprocal(np.asarray(data.data, dtype=np.float32))
    dst.data[...] = _cast_value(result, dst.dtype)
    return dst


def affine_range(*args, **kwargs):
    """Create a Python range from static or register-backed bounds."""
    del kwargs
    normalized = [
        _as_scalar(arg) if isinstance(arg, NDArray) else int(arg) for arg in args
    ]
    return range(*normalized)


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
    return value.data if isinstance(value, NDArray) else value


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


def exp(x):
    """Compute elementwise exponential."""
    return _unary_math_op(x, np.exp)


def sqrt(x):
    """Compute elementwise square root."""
    return _unary_math_op(x, np.sqrt)


def rsqrt(x):
    """Compute elementwise reciprocal square root."""
    return _unary_math_op(x, lambda v: 1.0 / np.sqrt(v))


def dma_copy(dst, src, dst_rmw_op=None, oob_mode=None, dge_mode=None, name=None):
    """Copy data from source tensor to destination tensor."""
    del oob_mode, dge_mode, name
    src_value = np.asarray(src.data)
    if dst_rmw_op is None:
        dst.data[...] = _cast_value(src_value, dst.dtype)
        return dst
    try:
        merged = dst_rmw_op(dst.data, src_value)
    except TypeError:
        merged = dst_rmw_op(dst.data, y=src_value)
    merged_value = merged.data if isinstance(merged, NDArray) else merged
    dst.data[...] = _cast_value(merged_value, dst.dtype)
    return dst


def tensor_copy(dst, src, engine=None, name=None):
    """Copy tensor data within on-chip memory."""
    del engine, name
    dst.data[...] = _cast_value(src.data, dst.dtype)
    return dst


def tensor_tensor(dst, data1, data2, op, engine=None, name=None):
    """Apply a binary tensor op and store the result in dst."""
    del engine, name
    lhs = np.asarray(data1.data)
    rhs = np.asarray(data2.data)
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
    return dst


def tensor_reduce(dst, op, data, axis=None, keepdims=False, engine=None, name=None):
    """Reduce a tensor along axes using add/max/min-like ops."""
    del engine, name
    source = np.asarray(data.data)
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
    result = np.asarray(data.data)
    for op, operand in ((op0, operand0), (op1, operand1)):
        if op is None:
            continue
        operand_value = _unwrap_operand(operand)
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
    in_data = data.data if isinstance(data, NDArray) else data
    in_bias = 0 if bias is None else (bias.data if isinstance(bias, NDArray) else bias)
    output = op(in_data * scale + in_bias)
    output_value = output.data if isinstance(output, NDArray) else output
    dst.data[...] = _cast_value(output_value, dst.dtype)
    return dst


class Builder:
    """Tracks grid dimensions/index for the active interpreted kernel."""

    def __init__(self, grid_dims=None):
        self.grid_dims = tuple(grid_dims) if grid_dims is not None else (1,)
        self.grid_idx = [0] * len(self.grid_dims)
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
    set_attr(nl, "exp", exp)
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
