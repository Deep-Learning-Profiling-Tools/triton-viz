import numpy as np

import neuronxcc.nki.language as nl
import inspect


# Q1: slicing semantic is weird
# Q2: why cannot we execute the .func function?

# Multi-dimensional slice class
class NLSlice:
    def __init__(self, start=None, stop=None, step: int = 1):
        self.start = [start] if isinstance(start, int) else start
        self.stop = [stop] if isinstance(stop, int) else stop
        self.step = [step] if isinstance(step, int) else step

    def __repr__(self):
        repr = ""
        for start, stop, step in zip(self.start, self.stop, self.step):
            if start is None:
                start = "None"
            if stop is None:
                stop = "None"
            if step is None:
                step = "None"
            repr += f"(start={start}, stop={stop}, step={step}) " 
        return repr

    def __add__(self, other):
        new_start = []
        new_stop = []
        new_step = []
        if isinstance(other, NLSlice):
            for start, stop, step in zip(self.start, self.stop, self.step):
                new_start.append(start + other.start if start is not None else None)
                new_stop.append(stop + other.stop if stop is not None else None)
                new_step.append(step + other.step if step is not None else None)
            return NLSlice(start=new_start, stop=new_stop, step=new_step)
        elif isinstance(other, int):
            for start, stop, step in zip(self.start, self.stop, self.step):
                new_start.append(start + other if start is not None else None)
                new_stop.append(stop + other if stop is not None else None)
                new_step.append(step)
            return NLSlice(start=new_start, stop=new_stop, step=new_step)
        raise TypeError(f"Unsupported operand type(s) for +: 'NLSlice' and '{type(other).__name__}'")

    def __radd__(self, other):
        new_start = []
        new_stop = []
        new_step = []
        if isinstance(other, NLSlice):
            for start, stop, step in zip(self.start, self.stop, self.step):
                new_start.append(other.start + start if start is not None else None)
                new_stop.append(other.stop + stop if stop is not None else None)
                new_step.append(other.step + step if step is not None else None)
            return NLSlice(start=new_start, stop=new_stop, step=new_step)
        elif isinstance(other, int):
            for start, stop, step in zip(self.start, self.stop, self.step):
                new_start.append(other + start if start is not None else None)
                new_stop.append(other + stop if stop is not None else None)
                new_step.append(step)
            return NLSlice(start=new_start, stop=new_stop, step=new_step)
        raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'NLSlice'")

    def __getitem__(self, keys):
        new_start = []
        new_stop = []
        new_step = []
        idx = 0
        for k in keys:
            # check if k is None:
            if k is None:
                new_start.append(None)
                new_stop.append(None)
                new_step.append(None)
            elif isinstance(k, slice):
                assert k.start is None and k.stop is None and k.step is None, "Slice must be complete"
                new_start.append(self.start[idx])
                new_stop.append(self.stop[idx])
                new_step.append(self.step[idx])
                idx += 1
            else:
                raise TypeError(f"Unsupported key type: {type(k)}")

        return NLSlice(start=new_start, stop=new_stop, step=new_step)

class NDArray:
    def __init__(self, buffer=None, name="", **kwargs):
        self.buffer = buffer
        self.name = name
        self.kwargs = kwargs
        value = None
        if "shape" in kwargs and "dtype" in kwargs:
            shape = kwargs.pop("shape")
            dtype = kwargs.pop("dtype")
            value = np.ndarray(shape, dtype=dtype)
        if "value" in kwargs:
            assert value is None or value.shape == kwargs["value"].shape
            value = kwargs["value"]
        self._value = value

    @property
    def shape(self):
        return self._value.shape if self._value is not None else None

    @property
    def dtype(self):
        return self._value.dtype if self._value is not None else None

    @property
    def value(self):
        return self._value
    
    @property
    def data(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    def data_ptr(self):
        return self.ctypes.data

    def __repr__(self):
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def __getitem__(self, keys):
        """Implement slicing operations for NDArray"""
        if self._value is None:
            raise AttributeError("NDArray has no value to slice")
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Apply the slicing to the underlying numpy array
        new_keys = []
        arr_dim = 0
        for k in keys:
            if isinstance(k, NDArray):
                dim_len = self._value.shape[arr_dim]
                new_keys.append(k._value.clip(0, dim_len - 1))
            elif isinstance(k, NLSlice):
                new_keys.append(slice(k.start, k.stop, k.step))
            elif k is None:
                new_keys.append(k)
                arr_dim -= 1 # add new dim -> revisit arr_dim for next key
            else:
                new_keys.append(k)
            arr_dim += 1
        
        sliced_value = self._value[tuple(new_keys)]

        # Create a new NDArray with the sliced data
        return NDArray(value=sliced_value, name=f"{self.name}_slice")


    def _binary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(value=op_func(self._value, other._value), name=f"{self.name}_{op_name}_{other.name}")
        elif np.isscalar(other):
            return NDArray(value=op_func(self._value, other), name=f"{self.name}_{op_name}_scalar")
        raise TypeError(f"Unsupported operand type(s) for {op_symbol}: 'NDArray' and '{type(other).__name__}'")

    def _rbinary_op(self, other, op_func, op_name, op_symbol):
        if isinstance(other, NDArray):
            return NDArray(value=op_func(other._value, self._value), name=f"{other.name}_{op_name}_{self.name}")
        elif np.isscalar(other):
            return NDArray(value=op_func(other, self._value), name=f"scalar_{op_name}_{self.name}")
        raise TypeError(f"Unsupported operand type(s) for {op_symbol}: '{type(other).__name__}' and 'NDArray'")

    # Define operator +/-/*//
    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b, "add", "+")

    def __radd__(self, other):
        return self._rbinary_op(other, lambda a, b: a + b, "add", "+")

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b, "sub", "-")

    def __rsub__(self, other):
        return self._rbinary_op(other, lambda a, b: a - b, "sub", "-")

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b, "mul", "*")

    def __rmul__(self, other):
        return self._rbinary_op(other, lambda a, b: a * b, "mul", "*")

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b, "div", "/")

    def __rtruediv__(self, other):
        return self._rbinary_op(other, lambda a, b: a / b, "div", "/")

    def __lt__(self, other):
        return self._binary_op(other, lambda a, b: a < b, "lt", "<")

    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: a > b, "gt", ">")

    def __le__(self, other):
        return self._binary_op(other, lambda a, b: a <= b, "le", "<=")

    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b, "ge", ">=")

    def __and__(self, other):
        return self._binary_op(other, lambda a, b: a & b, "and", "&")

    def __or__(self, other):
        return self._binary_op(other, lambda a, b: a | b, "or", "|")


class Builder:
    def __init__(self, grid_dims=None):
        # TODO: infinite grid dims for NKI
        self.grid_dims = grid_dims if grid_dims is not None else (1, 1, 1)
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.fn = None
        self.shared_hbm_arrays = {}

    def set_grid_dim(self, grid_dims):
        self.grid_dims = grid_dims

    def set_grid_idx(self, x, y, z):
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

    def ndarray(self, shape, dtype, *, buffer=None, name=None, **kwargs):
        if buffer == nl.shared_hbm:
            if name is None:
                # file name + function name + line number
                frame = inspect.currentframe().f_back
                file_name = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                name = f"{file_name}_{function_name}_{line_number}"
            if name in self.shared_hbm_arrays:
                # Return the existing shared HBM array
                ret = self.shared_hbm_arrays[name]
            else:
                # Create a new shared HBM array and store it
                ret = NDArray(buffer=buffer, name=name, shape=shape, dtype=dtype, **kwargs)
                self.shared_hbm_arrays[name] = ret
        else:
            ret = NDArray(buffer=buffer, name=name, shape=shape, dtype=dtype, **kwargs)
        return ret
    
    def zeros(self, shape, dtype, *, buffer=None, name=None, **kwargs):
        value = np.zeros(shape, dtype=dtype)
        return self.ndarray(
            shape, dtype, buffer=buffer, name=name, value=value, **kwargs
        )

    def arange(self, *args):
        if len(args) == 1:
            start = 0
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        else:
            raise ValueError("arange expects 1 or 2 arguments")
        return NDArray(value=np.arange(*args))

    def program_id(self, axis: int):
        if axis == 0:
            return self.grid_x
        elif axis == 1:
            return self.grid_y
        elif axis == 2:
            return self.grid_z
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    def load(self, src: NDArray, *, mask=None, dtype=None, **kwargs):
        value = src._value
        if isinstance(mask, NDArray):
            value = value[mask._value]
        elif mask is not None:
            value = value[mask]
        if dtype is not None:
            value = value.astype(dtype)

        mask_value = getattr(mask, "_value", np.ones_like(src))
        new_shape = []
        for i, _v in enumerate(mask_value.shape):
            assert np.unique(mask_value.sum(i)).size <= 2
            new_dim = mask_value.sum(i).flatten()[0]
            new_shape.append(new_dim)

        value = value.reshape(new_shape)
        return NDArray(value=value, name=src.name, **kwargs)

    def load_transpose2d(self, src: NDArray, *, mask=None, dtype=None, **kwargs):
        # THTODO
        value = src._value
        return self.load(NDArray(value=value.T, name=src.name), mask=mask, dtype=dtype, **kwargs)

    def store(self, dst: NDArray, value: NDArray, *, mask=None, **kwargs):
        dst._value[mask._value] = value._value.ravel()
        return dst

    def unary_op(self, x: NDArray, np_func, op_name, **kwargs):
        return NDArray(value=np_func(x._value), name=f"{x.name}_{op_name}", **kwargs)

    # Elementwise operator implementations
    def exp(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.exp, "exp", **kwargs)

    def relu(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: np.maximum(v, 0), "relu", **kwargs)

    def sigmoid(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: 1 / (1 + np.exp(-v)), "sigmoid", **kwargs)

    def tanh(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.tanh, "tanh", **kwargs)

    def silu(self, x: NDArray, **kwargs):
        # SiLU(x) = x * sigmoid(x)
        sigmoid_x = 1 / (1 + np.exp(-x._value))
        return NDArray(value=x._value * sigmoid_x, name=f"{x.name}_silu", **kwargs)

    def gelu(self, x: NDArray, **kwargs):
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = np.sqrt(2 / np.pi)
        inner = sqrt_2_pi * (x._value + 0.044715 * np.power(x._value, 3))
        return NDArray(value=0.5 * x._value * (1 + np.tanh(inner)), name=f"{x.name}_gelu", **kwargs)

    def sqrt(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.sqrt, "sqrt", **kwargs)

    def abs(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.abs, "abs", **kwargs)

    def log(self, x: NDArray, **kwargs):
        return self._unary_op(x, np.log, "log", **kwargs)

    def pow(self, x: NDArray, exponent, **kwargs):
        if isinstance(exponent, NDArray):
            return NDArray(value=np.power(x._value, exponent._value), name=f"{x.name}_pow_{exponent.name}", **kwargs)
        elif np.isscalar(exponent):
            return NDArray(value=np.power(x._value, exponent), name=f"{x.name}_pow_{exponent}", **kwargs)
        else:
            raise TypeError(f"Unsupported exponent type: {type(exponent)}")

    def reciprocal(self, x: NDArray, **kwargs):
        return self._unary_op(x, lambda v: 1 / v, "reciprocal", **kwargs)

    def matmul(self, x: NDArray, y: NDArray, transpose_x=False, mask=None, **kwargs):
        x_value = x._value 
        if transpose_x:
            x_value = x_value.T
        y_value = y._value 
        return NDArray(value=(x_value @ y_value), name=f"{x.name}_{y.name}_matmul", **kwargs)

    def copy(self, x: NDArray, **kwargs):
        return self.unary_op(x, np.copy, "copy", **kwargs)

    def range(self, stop):
        return range(stop)

nki_builder = Builder()


def patch():
    nl.ndarray = nki_builder.ndarray
    nl.program_id = nki_builder.program_id
    nl.arange = nki_builder.arange
    nl.load = nki_builder.load
    nl.store = nki_builder.store
    # see https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html

    # TODO: implement
    # matmul-specific
    # nl.shared_hbm
    # nl.psum
    nl.affine_range = nki_builder.range
    nl.par_dim
    nl.zeros = nki_builder.zeros
    nl.mgrid = NDArray(value=np.mgrid, buffer=nl.sbuf, name='mgrid')
    nl.matmul = nki_builder.matmul
    nl.copy = nki_builder.copy

    # attention-specific
    nl.load_transpose2d = nki_builder.load_transpose2d
    #nisa.affine_select
    #nl.tensor_reduce
    #nisa.activation
    nl.broadcast_to
    #nisa.nc_transpose

    # Elementwise operators
    nl.exp = nki_builder.exp
    nl.relu = nki_builder.relu
    nl.sigmoid = nki_builder.sigmoid
    nl.tanh = nki_builder.tanh
    nl.silu = nki_builder.silu
    nl.gelu = nki_builder.gelu
    nl.sqrt = nki_builder.sqrt
    nl.abs = nki_builder.abs
    nl.log = nki_builder.log
    nl.pow = nki_builder.pow
    nl.reciprocal = nki_builder.reciprocal


def unpatch():
    # reload the original functions
    import importlib

    importlib.reload(nl)


class NKIInterpretedFunction:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        grid_dims = kwargs.pop("grid", (1, 1, 1))  # Remove grid from kwargs to avoid passing it to the function
        # make it 3d if not
        if len(grid_dims) == 1:
            grid_dims = (grid_dims[0], 1, 1)
        elif len(grid_dims) == 2:
            grid_dims = (grid_dims[0], grid_dims[1], 1)
        elif len(grid_dims) != 3:
            raise ValueError(f"Grid must be 1, 2, or 3 dimensions, got {len(grid_dims)}")
        nki_builder.set_grid_dim(grid_dims)
        nki_builder.shared_hbm_arrays = {}
        nki_builder.fn = self.fn

        kwargs.pop("warmup", None)  # Remove warmup from kwargs if it exists
        kwargs.pop("client_manager", None)  # Remove client_manager from kwargs if it exists

        patch()

        # convert args to NDArray if they are not already
        args = [arg if isinstance(arg, NDArray) else NDArray(value=arg) for arg in args]
        for x in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                for z in range(grid_dims[2]):
                    nki_builder.set_grid_idx(x, y, z)
                    result = self.fn(*args, **kwargs)
        unpatch()
        return result.value
