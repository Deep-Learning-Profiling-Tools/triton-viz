import numpy as np


class NDArray:
    def __init__(self, buffer=None, name="", **kwargs):
        self.buffer = buffer
        self.name = name
        self.kwargs = kwargs
        if "shape" in kwargs and "dtype" in kwargs:
            shape = kwargs.pop("shape")
            dtype = kwargs.pop("dtype")
            self._value = np.ndarray(shape, dtype=dtype, buffer=buffer, **kwargs)
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

    def __getitem__(self, key):
        """Implement slicing operations for NDArray"""
        if self._value is None:
            raise AttributeError("NDArray has no value to slice")

        # Apply the slicing to the underlying numpy array
        sliced_value = self._value[key]

        # Create a new NDArray with the sliced data
        return NDArray(value=sliced_value, name=f"{self.name}_slice")

    # Define operator +/-/*//
    def __add__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value + other._value, name=f"{self.name}_add_{other.name}")
        raise TypeError(f"Unsupported operand type(s) for +: 'NDArray' and '{type(other).__name__}'")

    def __sub__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value - other._value, name=f"{self.name}_sub_{other.name}")
        raise TypeError(f"Unsupported operand type(s) for -: 'NDArray' and '{type(other).__name__}'")

    def __mul__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value * other._value, name=f"{self.name}_mul_{other.name}")
        raise TypeError(f"Unsupported operand type(s) for *: 'NDArray' and '{type(other).__name__}'")

    def __truediv__(self, other):
        if isinstance(other, NDArray):
            return NDArray(value=self._value / other._value, name=f"{self.name}_div_{other.name}")
        raise TypeError(f"Unsupported operand type(s) for /: 'NDArray' and '{type(other).__name__}'")


class Builder:
    def __init__(self, grid_dims=None):
        self.grid_dims = grid_dims if grid_dims is not None else (1, 1, 1)
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None

    def set_grid_dims(self, grid_dims):
        self.grid_dims = grid_dims

    def set_grid_idx(self, x, y, z):
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

    def ndarray(self, shape, dtype, *, buffer=None, name="", **kwargs):
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


class NKIInterpreterFunction:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        grid_dims = kwargs.pop("grid", (1, 1, 1))  # Remove grid from kwargs to avoid passing it to the function
        nki_builder.set_grid_dims(grid_dims)
        
        for x in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                for z in range(grid_dims[2]):
                    nki_builder.set_grid_idx(x, y, z)
                    result = self.fn(*args, **kwargs)
        return result
