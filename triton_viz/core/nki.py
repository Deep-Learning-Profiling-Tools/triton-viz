import numpy as np

import neuronxcc.nki.language as nl
import inspect


class NDArray:
    def __init__(self, buffer=None, name="", **kwargs):
        self.buffer = buffer
        self.name = name
        self.kwargs = kwargs
        if "shape" in kwargs and "dtype" in kwargs:
            shape = kwargs.pop("shape")
            dtype = kwargs.pop("dtype")
            self._value = np.ndarray(shape, dtype=dtype)
        elif "value" in kwargs:
            self._value = kwargs["value"]
        else:
            self._value = None

    @property
    def shape(self):
        return self._value.shape if self._value is not None else None

    @property
    def dtype(self):
        return self._value.dtype if self._value is not None else None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    def __repr__(self):
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, name={self.name})"

    def __getitem__(self, keys):
        """Implement slicing operations for NDArray"""
        if self._value is None:
            raise AttributeError("NDArray has no value to slice")

        # Apply the slicing to the underlying numpy array
        new_keys = []
        if isinstance(keys, tuple):
            for k in keys:
                if isinstance(k, NDArray):
                    new_keys.append(k._value)
                else:
                    new_keys.append(k)
        sliced_value = self._value[tuple(new_keys)]

        # Create a new NDArray with the sliced data
        return NDArray(value=sliced_value, name=f"{self.name}_slice")


    # Define operator +/-/*//
    def __add__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value + other._value, name=f"{self.name}_add_{other.name}")
        elif np.isscalar(other):
            return NDArray(value=self._value + other, name=f"{self.name}_add_scalar")
        raise TypeError(f"Unsupported operand type(s) for +: 'NDArray' and '{type(other).__name__}'")

    def __radd__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=other._value + self._value, name=f"{other.name}_add_{self.name}")
        elif np.isscalar(other):
            return NDArray(value=other + self._value, name=f"scalar_add_{self.name}")
        raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'NDArray'")

    def __sub__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value - other._value, name=f"{self.name}_sub_{other.name}")
        elif np.isscalar(other):
            return NDArray(value=self._value - other, name=f"{self.name}_sub_scalar")
        raise TypeError(f"Unsupported operand type(s) for -: 'NDArray' and '{type(other).__name__}'")

    def __rsub__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=other._value - self._value, name=f"{other.name}_sub_{self.name}")
        elif np.isscalar(other):
            return NDArray(value=other - self._value, name=f"scalar_sub_{self.name}")
        raise TypeError(f"Unsupported operand type(s) for -: '{type(other).__name__}' and 'NDArray'")

    def __mul__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value * other._value, name=f"{self.name}_mul_{other.name}")
        elif np.isscalar(other):
            return NDArray(value=self._value * other, name=f"{self.name}_mul_scalar")
        raise TypeError(f"Unsupported operand type(s) for *: 'NDArray' and '{type(other).__name__}'")

    def __rmul__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=other._value * self._value, name=f"{other.name}_mul_{self.name}")
        elif np.isscalar(other):
            return NDArray(value=other * self._value, name=f"scalar_mul_{self.name}")
        raise TypeError(f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'NDArray'")

    def __truediv__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value / other._value, name=f"{self.name}_div_{other.name}")
        elif np.isscalar(other):
            return NDArray(value=self._value / other, name=f"{self.name}_div_scalar")
        raise TypeError(f"Unsupported operand type(s) for /: 'NDArray' and '{type(other).__name__}'")

    def __rtruediv__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=other._value / self._value, name=f"{other.name}_div_{self.name}")
        elif np.isscalar(other):
            return NDArray(value=other / self._value, name=f"scalar_div_{self.name}")
        raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'NDArray'")


class Builder:
    def __init__(self, fn, grid_dims=None):
        self.fn = fn
        self.grid_dims = grid_dims if grid_dims is not None else (1, 1, 1)
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.shared_hbm_arrays = {}

    def set_grid_dims(self, grid_dims):
        self.grid_dims = grid_dims

    def set_grid_idx(self, x, y, z):
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

    def ndarray(self, shape, dtype, *, buffer=None, name="", **kwargs):
        if name is None:
            # file name + function name + line number
            frame = inspect.currentframe().f_back
            file_name = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            name = f"{file_name}_{function_name}_{line_number}"
        if buffer == nl.shared_hbm and name in self.shared_hbm_arrays:
            return self.shared_hbm_arrays[name]
        return NDArray(buffer=buffer, name=name, shape=shape, dtype=dtype, **kwargs)

    def arange(self, *args):
        if len(args) == 1:
            value = np.arange(args[0])
        elif len(args) == 2:
            value = np.arange(args[0], args[1])
        else:
            raise ValueError("arange expects 1 or 2 arguments")
        return NDArray(value=value, name="arange")

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
        if mask is not None:
            value = value[mask]
        if dtype is not None:
            value = value.astype(dtype)
        return NDArray(value=value, name=src.name, **kwargs)

    def store(self, dst: NDArray, value: NDArray, *, mask=None, **kwargs):
        if mask is not None:
            value = value[mask]
        dst._value = value._value
        return dst


nki_builder = Builder()


def patch():
    nl.ndarray = lambda *args, **kwargs: nki_builder.ndarray(*args, **kwargs)
    nl.program_id = lambda axis: nki_builder.program_id(axis)
    nl.arange = lambda *args: nki_builder.arange(*args)
    nl.load = lambda src, **kwargs: nki_builder.load(src, **kwargs)
    nl.store = lambda dst, value, **kwargs: nki_builder.store(dst, value, **kwargs)


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
        nki_builder.set_grid_dims(grid_dims)

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
