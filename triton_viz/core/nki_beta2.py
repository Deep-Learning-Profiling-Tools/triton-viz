import inspect
import itertools
import textwrap

import numpy as np

from triton_viz.utils.traceback_utils import CODE_KEYS, get_code_key

try:
    import neuronxcc.nki.language as nl
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - only hit when optional deps missing
    raise ModuleNotFoundError(
        "NeuronX dependencies are missing. Install triton-viz[nki] to enable the NKI interpreter."
    ) from exc

import nki.isa as nisa

from ..transformers.nki_extract_slice import transform_code
from .masked_load_store import masked_load, masked_store


def _buffer_name(buffer):
    if isinstance(buffer, Buffer):
        return buffer.buffer
    if isinstance(buffer, str):
        return buffer
    return getattr(buffer, "name", "")


FLOAT32_STORAGE_DTYPES = {
    "bfloat16": (8, 7, False),
    "tfloat32": (8, 10, False),
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
    x = np.asarray(value, dtype=np.float32)
    abs_x = np.abs(x)

    # Calculate IEEE-like format parameters
    bias = (1 << (exp_bits - 1)) - 1
    emax = ((1 << exp_bits) - (1 if finite_only else 2)) - bias
    emin = 1 - bias
    max_finite = np.float32((2.0 - 2.0**-mant_bits) * 2.0**emax)

    # Initialize output with magnitudes; this naturally propagates 0, inf, and NaN
    out = abs_x.copy()
    mask = np.isfinite(x) & (abs_x > 0)

    if np.any(mask):
        # Clamping exponent to emin gracefully handles both normal and subnormal numbers
        exp = np.maximum(np.floor(np.log2(abs_x[mask])), emin)
        step = np.exp2(exp - mant_bits).astype(np.float32)
        out[mask] = np.rint(abs_x[mask] / step) * step

    # Handle clipping for finite bounds
    if finite_only:
        out = np.minimum(out, max_finite)

    return np.copysign(out, x)


def _as_scalar(value):
    # unwrap NDArray scalar inputs before validating scalar shape
    value = value.data if isinstance(value, NDArray) else value
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected scalar value")
    return int(array.reshape(-1)[0])


def _normalize_pattern(pattern):
    if not pattern:
        raise ValueError("pattern must not be empty")
    normalized = []
    for pair in pattern:
        if len(pair) != 2:
            raise ValueError(f"pattern pair must have two values, got {pair}")
        step, count = pair
        normalized.append((int(step), int(count)))
    return normalized


def _compute_ap_indices(pattern, offset):
    pattern_pairs = _normalize_pattern(pattern)
    shape = [count for _, count in pattern_pairs]
    indices = np.zeros(shape, dtype=np.int64)
    for axis, (step, count) in enumerate(pattern_pairs):
        axis_shape = [1] * len(pattern_pairs)
        axis_shape[axis] = count
        indices += step * np.arange(count, dtype=np.int64).reshape(axis_shape)
    return indices + int(offset)


class NDArray:
    """Lightweight NumPy-backed tensor used by the NKI beta2 interpreter."""

    def __init__(self, buffer=None, name="", shape=None, dtype=None, value=None):
        self.buffer = buffer
        self.name = name
        self.dtype = dtype
        # normalize user shape into a concrete tuple for storage allocation
        storage_shape = tuple(shape) if shape is not None else None
        # force float32 backing for logical dtypes listed in FLOAT32_STORAGE_DTYPES
        storage_dtype = (
            np.float32
            if FLOAT32_STORAGE_DTYPES.get(
                dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dtype, "name", None))
            )
            is not None
            else dtype
        )
        if value is None:
            self.data = np.ndarray(storage_shape, dtype=storage_dtype)
        else:
            spec = FLOAT32_STORAGE_DTYPES.get(
                dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dtype, "name", None))
            )
            if spec is not None:
                # quantize logical low-precision dtype values, then store as fp32
                exp_bits, mant_bits, finite_only = spec
                array = _quantize_binary_float(
                    value, exp_bits, mant_bits, finite_only=finite_only
                )
                array = np.asarray(array, dtype=np.float32)
            else:
                array = np.asarray(value, dtype=storage_dtype)
            if storage_shape is not None and array.shape != storage_shape:
                array = array.reshape(storage_shape)
            self.data = array
        if self.dtype is None:
            self.dtype = self.data.dtype
        self._data_ptr = None

    @property
    def shape(self):
        return self.data.shape if self.data is not None else None

    @property
    def address(self):
        if self._data_ptr is None:
            self._data_ptr = self.data.ctypes.data
        return self._data_ptr

    @property
    def offset(self):
        """Return element offsets for this tensor."""
        return self.get_offsets()

    @property
    def pattern(self):
        """Return the default linear access pattern for this tensor."""
        itemsize = max(self.element_size(), 1)
        steps = [stride // itemsize for stride in self.stride()]
        return [[step, dim] for step, dim in zip(steps, self.shape)]

    def data_ptr(self):  # alias for self.address for triton-viz compat
        return self.address

    def stride(self):
        return self.data.strides

    def element_size(self):
        return getattr(self.dtype, "itemsize", self.data.dtype.itemsize)

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.data

    def get_offsets(self):
        """
        Generate offset arrays for each dimension based on shape and stride.
        Given array with shape (A, ..., Z) and strides (a, ..., z), return offsets:
        a * arange(A)[:, None, ..., None] + ... + z * arange(Z)[None, None, ..., :]
        """
        offsets = 0
        for dim_size, stride in zip(self.shape, self.stride()):
            offsets = np.expand_dims(offsets, -1) + np.arange(dim_size) * stride
        return NDArray(value=offsets, name=self.name)

    def __repr__(self):
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def __getitem__(self, keys):
        """Implement slicing operations for NDArray"""
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
        if not isinstance(keys, tuple):
            keys = (keys,)
        new_keys = [k.data if isinstance(k, NDArray) else k for k in keys]
        # unwrap NDArray sources before coercing into destination logical dtype
        unwrapped = value.data if isinstance(value, NDArray) else value
        spec = FLOAT32_STORAGE_DTYPES.get(
            self.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(self.dtype, "name", None))
        )
        if spec is not None:
            # quantize logical low-precision dtype values, then store as fp32
            exp_bits, mant_bits, finite_only = spec
            casted = _quantize_binary_float(
                unwrapped, exp_bits, mant_bits, finite_only=finite_only
            )
            casted = np.asarray(casted, dtype=np.float32)
        else:
            casted = np.asarray(unwrapped, dtype=self.dtype)
        self.data[tuple(new_keys)] = casted
        return self

    def _binary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(self.data, other.data),
                name=f"{self.name}_{op_name}_{other.name}",
            )
        if np.isscalar(other):
            return NDArray(
                value=op_func(self.data, other), name=f"{self.name}_{op_name}_scalar"
            )
        raise TypeError(
            f"Unsupported operand type(s) for {op_symbol}: 'NDArray' and '{type(other).__name__}'"
        )

    def _rbinary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(
                value=op_func(other.data, self.data),
                name=f"{other.name}_{op_name}_{self.name}",
            )
        if np.isscalar(other):
            return NDArray(
                value=op_func(other, self.data), name=f"scalar_{op_name}_{self.name}"
            )
        raise TypeError(
            f"Unsupported operand type(s) for {op_symbol}: '{type(other).__name__}' and 'NDArray'"
        )

    def reshape(self, *args, **kwargs):
        """Return a reshaped view backed by the same NumPy data when possible."""
        return NDArray(
            value=self.data.reshape(*args),
            name=f"{self.name}_reshape",
            buffer=self.buffer,
            **kwargs,
        )

    def broadcast_to(self, *args, **kwargs):
        """Return a broadcasted view of this tensor."""
        return NDArray(
            value=np.broadcast_to(self.data, *args),
            name=f"{self.name}_broadcast_to",
            buffer=self.buffer,
            **kwargs,
        )

    def ap(
        self,
        *pattern,
        offset=0,
        dtype=None,
        scalar_offset=None,
        vector_offset=None,
        indirect_dim=0,
        **kwargs,
    ):
        """
        Materialize an address-pattern access from this tensor.

        Supports both `tensor.ap(pattern=[[...], ...], offset=...)` and
        `tensor.ap([step, count], [step, count], ...)` forms.
        """
        if "pattern" in kwargs:
            pattern = tuple(kwargs.pop("pattern"))
        if kwargs:
            raise TypeError(f"Unsupported kwargs for ap: {tuple(kwargs)}")
        if (
            len(pattern) == 2
            and np.isscalar(pattern[1])
            and not np.isscalar(pattern[0])
        ):
            pattern, offset = (pattern[0],), pattern[1]
        if len(pattern) == 1 and isinstance(pattern[0], (list, tuple)) and pattern[0]:
            first = pattern[0][0]
            if isinstance(first, (list, tuple)):
                pattern = tuple(pattern[0])
        pattern_pairs = _normalize_pattern(pattern)
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


# install NDArray binary/comparison/bitwise dunder ops without one-line wrappers
for _method, _op_name, _symbol, _op_func, _reverse in (
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
    if _reverse:
        # dispatch to NDArray reverse binary op helper for reflected dunder ops
        setattr(
            NDArray,
            _method,
            lambda self,
            other,
            op_name=_op_name,
            symbol=_symbol,
            op_func=_op_func: self._rbinary_op(other, op_func, op_name, symbol),
        )
    else:
        # dispatch to NDArray binary op helper for forward dunder ops
        setattr(
            NDArray,
            _method,
            lambda self,
            other,
            op_name=_op_name,
            symbol=_symbol,
            op_func=_op_func: self._binary_op(other, op_func, op_name, symbol),
        )


class Buffer:
    """NKI memory region abstraction used by `sbuf`, `hbm`, and `psum`."""

    def __init__(self, buffer: str, size=None, data=None):
        self.buffer = buffer
        self.size = size
        self.data = data
        if size is not None and data is None:
            self.data = np.empty(shape=size, dtype=np.uint8)

    def ptr(self, size, offset=None):
        """Return a sub-buffer view with the requested shape and optional offset."""
        if offset is None:
            return Buffer(self.buffer, size)
        if self.data is None:
            raise ValueError("Cannot slice a buffer pointer without backing data")
        coords = tuple(
            slice(off, off + dim_size) for off, dim_size in zip(offset, size)
        )
        return Buffer(self.buffer, size, data=self.data[coords])

    def view(self, dtype, size):
        """Materialize a typed NDArray view over this buffer region."""
        probe = NDArray(buffer=self.buffer, dtype=dtype, shape=size)
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


def _default_shared_hbm_name():
    frame = inspect.currentframe()
    caller = (
        frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    )
    if caller is None:
        return "shared_hbm"
    return f"{caller.f_code.co_filename}_{caller.f_code.co_name}_{caller.f_lineno}"


def _resolve_buffer(buffer):
    if isinstance(buffer, Buffer):
        return buffer, False
    if buffer is None:
        return sbuf, False

    name = _buffer_name(buffer)
    if name == "shared_hbm":
        return hbm, True
    if name in {"private_hbm", "hbm"}:
        return hbm, False
    if name == "sbuf":
        return sbuf, False
    if name == "psum":
        return psum, False
    raise TypeError(f"Unsupported buffer type: {type(buffer).__name__}")


def ndarray(shape, dtype, *, buffer=None, name=None, **kwargs):
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
                # assign and cast shared_hbm initialization/update data into cached tensor storage
                spec = FLOAT32_STORAGE_DTYPES.get(
                    cached.dtype,
                    FLOAT32_STORAGE_DTYPES.get(getattr(cached.dtype, "name", None)),
                )
                if spec is not None:
                    # quantize logical low-precision dtype values, then store as fp32
                    exp_bits, mant_bits, finite_only = spec
                    casted = _quantize_binary_float(
                        kwargs["value"], exp_bits, mant_bits, finite_only=finite_only
                    )
                    casted = np.asarray(casted, dtype=np.float32)
                else:
                    casted = np.asarray(kwargs["value"], dtype=cached.dtype)
                cached.data[...] = casted
            return cached
        created = NDArray(
            buffer=resolved_buffer.buffer,
            name=shared_name,
            shape=shape,
            dtype=dtype,
            value=kwargs.get("value"),
        )
        nki_builder.shared_hbm_arrays[shared_name] = created
        return created

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


def arange(*args):
    """Create a 1D NDArray range."""
    return NDArray(value=np.arange(*args))


def zeros(shape, dtype, *, buffer=None, name=None, **kwargs):
    """Create a zero-initialized tensor."""
    ret = ndarray(shape, dtype, buffer=buffer, name=name, **kwargs)
    ret.data.fill(0)
    return ret


def _convert_keys_to_numpy(keys):
    if isinstance(keys, (tuple, list)):
        return tuple(_convert_keys_to_numpy(k) for k in keys)
    if isinstance(keys, NDArray):
        return keys.data
    return keys


def load(src, keys=None, *, mask=None, **kwargs):
    """Load array elements with optional mask semantics."""
    if keys is None:
        value = np.copy(src.data)
        if mask is not None:
            # unwrap NDArray masks before applying masked load behavior
            mask_value = mask.data if isinstance(mask, NDArray) else mask
            value = np.where(mask_value, value, np.zeros((), dtype=value.dtype))
        return NDArray(
            value=value, name=f"{src.name}_load", buffer=src.buffer, **kwargs
        )
    # unwrap NDArray masks before passing to masked_load
    mask_value = (
        (mask.data if isinstance(mask, NDArray) else mask) if mask is not None else None
    )
    result = masked_load(src.data, _convert_keys_to_numpy(keys), mask=mask_value)
    return NDArray(value=result, name=f"{src.name}_load", buffer=src.buffer, **kwargs)


def store(dst, keys=None, value=None, *, mask=None, **kwargs):
    """Store values into a destination tensor with optional masking."""
    if value is None:
        raise ValueError("value must be provided")
    # unwrap NDArray store payload before coercing into destination storage dtype
    value_source = value.data if isinstance(value, NDArray) else value
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        value_array = _quantize_binary_float(
            value_source, exp_bits, mant_bits, finite_only=finite_only
        )
        value_array = np.asarray(value_array, dtype=np.float32)
    else:
        value_array = np.asarray(value_source, dtype=dst.dtype)
    if keys is None:
        if mask is None:
            # full-tensor store relies on numpy assignment broadcasting semantics
            dst.data[...] = value_array
        else:
            # unwrap NDArray masks before applying masked full-tensor store
            mask_value = mask.data if isinstance(mask, NDArray) else mask
            # masked full-tensor store relies on numpy assignment broadcasting semantics
            dst.data[...] = np.where(mask_value, value_array, dst.data)
        return dst
    # unwrap NDArray masks before passing to masked_store
    mask_value = (
        (mask.data if isinstance(mask, NDArray) else mask) if mask is not None else None
    )
    masked_store(dst.data, _convert_keys_to_numpy(keys), value_array, mask=mask_value)
    return dst


def sum(x, *args, mask=None, **kwargs):
    """Reduce a tensor by summing across selected axes."""
    if mask is not None:
        # unwrap NDArray masks into NumPy `where` masks for reduction
        kwargs["where"] = mask.data if isinstance(mask, NDArray) else mask
    return NDArray(
        value=x.data.sum(*args, **kwargs), name=f"{x.name}_sum", buffer=x.buffer
    )


def copy(x, dtype=None, **kwargs):
    """Copy tensor values, optionally casting to a different dtype."""
    if dtype is None:
        value = np.copy(x.data)
    else:
        spec = FLOAT32_STORAGE_DTYPES.get(
            dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dtype, "name", None))
        )
        if spec is not None:
            # quantize logical low-precision dtype values, then store as fp32
            exp_bits, mant_bits, finite_only = spec
            value = _quantize_binary_float(
                x.data, exp_bits, mant_bits, finite_only=finite_only
            )
            value = np.asarray(value, dtype=np.float32)
        else:
            value = np.asarray(x.data, dtype=dtype)
    return NDArray(value=value, name=f"{x.name}_copy", buffer=x.buffer, **kwargs)


def nc_transpose(dst, data, engine=None, name=None):
    """Compute a transpose between partition and flattened free axes."""
    value = np.asarray(data.data)
    transposed = value if value.ndim < 2 else value.reshape(value.shape[0], -1).T
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            transposed.reshape(dst.shape), exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(transposed.reshape(dst.shape), dtype=dst.dtype)
    # transpose write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
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
    """Compute `dst = stationary.T @ moving`."""
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
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            result, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(result, dtype=dst.dtype)
    # matmul write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def _mx_dequantize(data, scale):
    value = np.asarray(data.data, dtype=np.float32)
    # unwrap NDArray scales before dequantization
    scale_source = scale.data if isinstance(scale, NDArray) else scale
    scale_value = np.asarray(scale_source, dtype=np.float32)
    if scale_value.ndim > 0 and value.shape[0] != scale_value.shape[0]:
        if value.shape[0] % scale_value.shape[0] == 0:
            repeats = value.shape[0] // scale_value.shape[0]
            scale_value = np.repeat(scale_value, repeats, axis=0)
    return value * scale_value


def nc_matmul_mx(
    dst,
    stationary,
    moving,
    stationary_scale,
    moving_scale,
    tile_position=None,
    tile_size=None,
    psum_accumulate_flag=3,
    name=None,
):
    """Compute MX matmul by dequantizing operands then applying matmul."""
    del tile_position, tile_size, psum_accumulate_flag, name
    stationary_dequant = _mx_dequantize(stationary, stationary_scale)
    moving_dequant = _mx_dequantize(moving, moving_scale)
    result = stationary_dequant.T @ moving_dequant
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            result, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(result, dtype=dst.dtype)
    # dequantized matmul write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def reciprocal(dst, data, name=None):
    """Compute elementwise reciprocal into destination tensor."""
    del name
    result = np.reciprocal(np.asarray(data.data, dtype=np.float32))
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            result, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(result, dtype=dst.dtype)
    # reciprocal write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


# create Python ranges from static or register-backed bounds
dynamic_range = sequential_range = static_range = affine_range = (
    lambda *args, **kwargs: range(
        *[_as_scalar(arg) if isinstance(arg, NDArray) else int(arg) for arg in args]
    )
)


def register_alloc(x=None):
    """Allocate a register value for scalar loop/control flow."""
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    return NDArray("register", value=np.array(_as_scalar(x), dtype=np.int32))


def register_move(dst, imm):
    """Move an immediate integer value into a register."""
    if isinstance(dst, NDArray):
        dst.data[...] = int(imm)
        return dst
    return int(imm)


def register_load(dst, src):
    """Load a scalar tensor value into a register."""
    value = _as_scalar(src)
    if isinstance(dst, NDArray):
        dst.data[...] = value
        return dst
    return value


def register_store(dst, src):
    """Store a register value into a scalar tensor."""
    value = _as_scalar(src)
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            value, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(value, dtype=dst.dtype)
    # register store relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def dma_copy(dst, src, dst_rmw_op=None, oob_mode=None, dge_mode=None, name=None):
    """Copy data from source tensor to destination tensor."""
    del oob_mode, dge_mode, name
    src_value = np.asarray(src.data)
    if dst_rmw_op is None:
        spec = FLOAT32_STORAGE_DTYPES.get(
            dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
        )
        if spec is not None:
            # quantize logical low-precision dtype values, then store as fp32
            exp_bits, mant_bits, finite_only = spec
            casted = _quantize_binary_float(
                src_value, exp_bits, mant_bits, finite_only=finite_only
            )
            casted = np.asarray(casted, dtype=np.float32)
        else:
            casted = np.asarray(src_value, dtype=dst.dtype)
        # dma copy write relies on numpy assignment broadcasting semantics
        dst.data[...] = casted
        return dst
    try:
        merged = dst_rmw_op(dst.data, src_value)
    except TypeError:
        merged = dst_rmw_op(dst.data, y=src_value)
    # unwrap NDArray rmw results and cast into destination storage
    merged_value = merged.data if isinstance(merged, NDArray) else merged
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            merged_value, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(merged_value, dtype=dst.dtype)
    # dma rmw write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def tensor_copy(dst, src, engine=None, name=None):
    """Copy tensor data within on-chip memory."""
    del engine, name
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            src.data, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(src.data, dtype=dst.dtype)
    # tensor copy write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def tensor_tensor(dst, data1, data2, op, engine=None, name=None):
    """Apply a binary tensor op and store the result in `dst`."""
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
            "minimum": np.minimum,
            "power": np.power,
            "pow": np.power,
        }
        matched = next((fn for key, fn in fallback_ops.items() if key in op_name), None)
        if matched is None:
            raise RuntimeError(
                f"Unsupported tensor op outside kernel context: {op_name}"
            )
        result = matched(lhs, rhs)
    # unwrap NDArray op results and cast into destination storage
    result_value = result.data if isinstance(result, NDArray) else result
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            result_value, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(result_value, dtype=dst.dtype)
    # tensor op write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def quantize_mx(dst, src, dst_scale, name=None):
    """Quantize source values with a simple global scale approximation."""
    del name
    src_fp32 = np.asarray(src.data, dtype=np.float32)
    if src_fp32.size == 0:
        dst_spec = FLOAT32_STORAGE_DTYPES.get(
            dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
        )
        if dst_spec is not None:
            # quantize logical low-precision dtype values, then store as fp32
            exp_bits, mant_bits, finite_only = dst_spec
            dst_casted = _quantize_binary_float(
                src_fp32, exp_bits, mant_bits, finite_only=finite_only
            )
            dst_casted = np.asarray(dst_casted, dtype=np.float32)
        else:
            dst_casted = np.asarray(src_fp32, dtype=dst.dtype)
        scale_spec = FLOAT32_STORAGE_DTYPES.get(
            dst_scale.dtype,
            FLOAT32_STORAGE_DTYPES.get(getattr(dst_scale.dtype, "name", None)),
        )
        if scale_spec is not None:
            # quantize logical low-precision dtype values, then store as fp32
            exp_bits, mant_bits, finite_only = scale_spec
            scale_casted = _quantize_binary_float(
                1, exp_bits, mant_bits, finite_only=finite_only
            )
            scale_casted = np.asarray(scale_casted, dtype=np.float32)
        else:
            scale_casted = np.asarray(1, dtype=dst_scale.dtype)
        # quantize_mx empty writes rely on numpy assignment broadcasting semantics
        dst.data[...] = dst_casted
        dst_scale.data[...] = scale_casted
        return dst
    max_abs = float(np.max(np.abs(src_fp32)))
    scale = 1.0 if max_abs == 0 else max_abs / 127.0
    scale_spec = FLOAT32_STORAGE_DTYPES.get(
        dst_scale.dtype,
        FLOAT32_STORAGE_DTYPES.get(getattr(dst_scale.dtype, "name", None)),
    )
    if scale_spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = scale_spec
        scale_casted = _quantize_binary_float(
            scale, exp_bits, mant_bits, finite_only=finite_only
        )
        scale_casted = np.asarray(scale_casted, dtype=np.float32)
    else:
        scale_casted = np.asarray(scale, dtype=dst_scale.dtype)
    # quantization scale write relies on numpy assignment broadcasting semantics
    dst_scale.data[...] = scale_casted
    quantized = np.round(src_fp32 / scale)
    dst_spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if dst_spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = dst_spec
        dst_casted = _quantize_binary_float(
            quantized, exp_bits, mant_bits, finite_only=finite_only
        )
        dst_casted = np.asarray(dst_casted, dtype=np.float32)
    else:
        dst_casted = np.asarray(quantized, dtype=dst.dtype)
    # quantized payload write relies on numpy assignment broadcasting semantics
    dst.data[...] = dst_casted
    return dst


def memset(dst, value, engine=None, name=None):
    """Fill destination tensor with a constant value."""
    del engine, name
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            value, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(value, dtype=dst.dtype)
    # memset write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


def iota(dst, pattern, offset, channel_multiplier=0, name=None):
    """Generate index pattern values into destination tensor."""
    del name
    pattern_pairs = _normalize_pattern(pattern)
    counts = [count for _, count in pattern_pairs]
    base_indices = _compute_ap_indices(pattern_pairs, offset)
    partition_count = dst.shape[0]
    out = np.empty((partition_count, *counts), dtype=np.int64)
    for channel_id in range(partition_count):
        out[channel_id] = base_indices + channel_id * int(channel_multiplier)
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            out.reshape(dst.shape), exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(out.reshape(dst.shape), dtype=dst.dtype)
    # iota write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
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
    """Apply a simple activation epilogue into dst."""
    del reduce_op, reduce_res, reduce_cmd, name
    # unwrap NDArray inputs before evaluating activation op
    in_data = data.data if isinstance(data, NDArray) else data
    in_bias = 0 if bias is None else (bias.data if isinstance(bias, NDArray) else bias)
    output = op(in_data * scale + in_bias)
    # unwrap NDArray activation results and cast into destination storage
    output_value = output.data if isinstance(output, NDArray) else output
    spec = FLOAT32_STORAGE_DTYPES.get(
        dst.dtype, FLOAT32_STORAGE_DTYPES.get(getattr(dst.dtype, "name", None))
    )
    if spec is not None:
        # quantize logical low-precision dtype values, then store as fp32
        exp_bits, mant_bits, finite_only = spec
        casted = _quantize_binary_float(
            output_value, exp_bits, mant_bits, finite_only=finite_only
        )
        casted = np.asarray(casted, dtype=np.float32)
    else:
        casted = np.asarray(output_value, dtype=dst.dtype)
    # activation write relies on numpy assignment broadcasting semantics
    dst.data[...] = casted
    return dst


class Builder:
    """Tracks grid dimensions/index for the active interpreted kernel."""

    def __init__(self, grid_dims=None):
        self.grid_dims = tuple(grid_dims) if grid_dims is not None else (1,)
        self.grid_idx = [0] * len(self.grid_dims)
        self.fn = None
        self.shared_hbm_arrays = {}


nki_builder = Builder()


def nki_patch_lang(scope=None):
    """Patch `nl` and `nisa` APIs to point at beta2 interpreter implementations."""
    # choose attribute patch callback based on whether a patch scope was provided
    set_attr = setattr if scope is None else scope.set_attr

    set_attr(nl, "ndarray", ndarray)
    set_attr(nl, "zeros", zeros)
    set_attr(nl, "arange", arange)
    set_attr(nl, "load", load)
    set_attr(nl, "store", store)
    set_attr(nl, "copy", copy)
    set_attr(nl, "sum", sum)
    # map nl.program_id onto the active interpreted grid index
    set_attr(
        nl,
        "program_id",
        lambda axis: (
            nki_builder.grid_idx[int(axis)]
            if 0 <= int(axis) < len(nki_builder.grid_idx)
            else (_ for _ in ()).throw(
                ValueError(
                    f"Invalid axis {int(axis)} for {len(nki_builder.grid_idx)}D grid"
                )
            )
        ),
    )
    # map nl.num_programs onto interpreted grid dimensions
    set_attr(
        nl,
        "num_programs",
        lambda axes=None: (
            int(np.prod(nki_builder.grid_dims))
            if axes is None
            else (
                tuple(nki_builder.grid_dims[int(axis)] for axis in axes)
                if isinstance(axes, (tuple, list))
                else nki_builder.grid_dims[int(axes)]
            )
        ),
    )
    # map nl.program_ndim onto interpreted grid rank
    set_attr(nl, "program_ndim", lambda: len(nki_builder.grid_dims))
    set_attr(nl, "affine_range", affine_range)
    set_attr(nl, "static_range", static_range)
    set_attr(nl, "sequential_range", sequential_range)
    set_attr(nl, "dynamic_range", dynamic_range)
    set_attr(nl, "device_print", print)
    set_attr(nl, "ds", slice)
    set_attr(nl, "shared_constant", lambda tensor: tensor)

    set_attr(nisa, "nc_matmul", nc_matmul)
    set_attr(nisa, "nc_matmul_mx", nc_matmul_mx)
    set_attr(nisa, "nc_transpose", nc_transpose)
    set_attr(nisa, "activation", activation)
    set_attr(nisa, "reciprocal", reciprocal)
    set_attr(nisa, "memset", memset)
    set_attr(nisa, "iota", iota)
    set_attr(nisa, "dma_copy", dma_copy)
    set_attr(nisa, "tensor_copy", tensor_copy)
    set_attr(nisa, "tensor_tensor", tensor_tensor)
    set_attr(nisa, "quantize_mx", quantize_mx)
    set_attr(nisa, "register_alloc", register_alloc)
    set_attr(nisa, "register_move", register_move)
    set_attr(nisa, "register_load", register_load)
    set_attr(nisa, "register_store", register_store)


# restore any patch scope used for nki_patch_lang
nki_unpatch_lang = (
    lambda scope=None: scope.restore()
    if scope is not None and hasattr(scope, "restore")
    else None
)


class NKIInterpretedFunction:
    """Callable wrapper that executes NKI kernels with interpreter semantics."""

    def __init__(self, fn):
        """Store the original kernel function."""
        self.fn = fn

    def run(self, *args, **kwargs):
        """Run the wrapped kernel over a launch grid with optional tracing callbacks."""
        grid_dims = kwargs.pop("grid", (1,))
        if isinstance(grid_dims, int):
            grid_dims = (grid_dims,)
        else:
            grid_dims = tuple(grid_dims)
        if not grid_dims:
            raise ValueError("Grid must have at least one dimension")
        # set active interpreted launch grid shape and reset current grid index cursor
        nki_builder.grid_dims = tuple(int(dim) for dim in grid_dims)
        nki_builder.grid_idx = [0] * len(nki_builder.grid_dims)
        nki_builder.shared_hbm_arrays = {}
        nki_builder.fn = self.fn

        kwargs.pop("warmup", None)
        client_manager = kwargs.pop("client_manager", None)

        if client_manager is not None:
            client_manager.grid_callback(grid_dims)

        if hasattr(self.fn, "__code__"):
            source_code = textwrap.dedent(inspect.getsource(self.fn))
            transformed_code = transform_code(source_code)
            exec_globals = self.fn.__globals__.copy()

            import os
            import random
            import string

            rand_str = "".join(
                random.choices(string.ascii_letters + string.digits, k=16)
            )
            os.makedirs("/tmp/triton-viz", exist_ok=True)
            filename = f"/tmp/triton-viz/{rand_str}.py"
            with open(filename, "w") as f:
                f.write(transformed_code)
            code_obj = compile(transformed_code, filename=filename, mode="exec")
            exec(code_obj, exec_globals)
            self.fn = exec_globals[self.fn.__name__]
            CODE_KEYS.add(get_code_key(self.fn))

        exec_globals = self.fn.__globals__
        exec_globals["sbuf"] = sbuf
        exec_globals["hbm"] = hbm
        exec_globals["shared_hbm"] = getattr(nl, "shared_hbm", hbm)
        exec_globals["private_hbm"] = getattr(nl, "private_hbm", hbm)
        exec_globals["psum"] = psum
        exec_globals["dynamic_range"] = dynamic_range

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
                    f"Grid index rank mismatch: got {len(grid_idx)}, expected {len(nki_builder.grid_dims)}"
                )
            # set active interpreted program index for nl.program_id() queries
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
