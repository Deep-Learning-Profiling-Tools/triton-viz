import inspect

import numpy as np

from .data import Launch, Grid, Load, Tensor
from triton.runtime.interpreter import (
    GridExecutor,
    _unwrap,
    _implicit_cvt,
    RESERVED_KWS,
    builder,
    _patch_lang,
)
from typing import Tuple, List, Optional
from contextlib import contextmanager
from functools import wraps


def _unpatch_lang():
    import importlib
    import sys
    import triton.language

    modules = [
        triton.language,
        triton.language.core,
        triton.language.standard,
        triton.language.math,
    ]
    for module in modules:
        if module.__name__ in sys.modules:
            importlib.reload(module)


class RecordBuilder:
    def __init__(self) -> None:
        self._launches: List[Launch] = []
        self._sampling_grid_idx: Optional[Tuple] = None
        self._grid_idx = (0, 0, 0)
        self._grid_dim = (1, 1, 1)

    @property
    def launches(self):
        return self._launches

    def set_sampling_grid_idx(self, idx: Tuple):
        self._sampling_grid_idx = idx

    def set_grid_dim(self, nx, ny, nz):
        self._grid_dim = (nx, ny, nz)
        self._launches.append(Launch((nx, ny, nz), [], []))

    def set_grid_idx(self, x, y, z):
        assert x < self._grid_dim[0]
        assert y < self._grid_dim[1]
        assert z < self._grid_dim[2]
        self._grid_idx = (x, y, z)
        grid_record = Grid(self._grid_idx)
        self.add_record(grid_record)

    def add_tensor(self, data, dtype, shape=None, stride=None):
        tensor = Tensor(data, shape, stride, dtype)
        self._launches[-1].tensors.append(tensor)

    def add_tensors(self, tensors):
        self._launches[-1].tensors.extend(tensors)

    def sort_tensor_handles(self):
        # Sort tensor handles based on ptr
        launch = self._launches[-1]
        launch.tensors = sorted(launch.tensors, key=lambda x: x.ptr)

    def get_tensor_ptr(self, ptr):
        # From a give ptr, get where the original tensor is stored
        # Tensors have been sorted by ptr
        ret_idx = 0
        for i in range(len(self._launches[-1].tensors)):
            if ptr < self._launches[-1].tensors[i].ptr:
                break
            ret_idx = i
        return self._launches[-1].tensors[ret_idx]

    def add_record(self, record):
        def _to_1d_grid(idx: Tuple):
            # Assuming originally 1d, 2d, or 3d input
            if len(idx) == 1:
                return idx[0]
            elif len(idx) == 2:
                return idx[0] * self._grid_dim[1] + idx[1]
            elif len(idx) == 3:
                return (
                    idx[0] * self._grid_dim[1] * self._grid_dim[2]
                    + idx[1] * self._grid_dim[2]
                    + idx[2]
                )

        if not self._sampling_grid_idx or _to_1d_grid(
            self._sampling_grid_idx
        ) == _to_1d_grid(self._grid_idx):
            self._launches[-1].records.append(record)


record_builder = RecordBuilder()


def _grid_executor_call(self, *args_dev, **kwargs):
    args_hst = [
        _unwrap(arg).cpu() if hasattr(arg, "data_ptr") else arg for arg in args_dev
    ]
    # Removes reserved keywords from kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS}
    if kwargs.pop("warmup", False):
        return
    # Remaps core language functions to interpreted ones
    _patch_lang(self.fn)
    # We need to copy arguments to the host for the interpreter.
    # Implicitly convert tensor arguments to their base pointers
    args = inspect.getcallargs(self.fn, *args_hst, **kwargs)
    call_args = {}
    tensors = []
    for name, arg in args.items():
        if name in self.constexprs:
            call_args[name] = arg
        else:
            ret = _implicit_cvt(arg)
            if isinstance(arg, int):
                tensors.append(
                    Tensor(ret.handle.data, ret.dtype, stride=None, shape=None)
                )
            elif hasattr(arg, "data_ptr"):
                tensors.append(
                    Tensor(ret.handle.data, ret.dtype, arg.stride(), arg.shape)
                )
            call_args[name] = ret
    # Flatten kwargs
    call_args.pop("self", None)
    # Iterate through grid
    grid = self.grid(call_args) if callable(self.grid) else self.grid
    assert len(grid) <= 3
    grid = grid + (1,) * (3 - len(grid))
    builder.set_grid_dim(*grid)
    record_builder.set_grid_dim(*grid)
    record_builder.add_tensors(tensors)
    record_builder.sort_tensor_handles()
    for x in range(grid[0]):
        for y in range(grid[1]):
            for z in range(grid[2]):
                builder.set_grid_idx(x, y, z)
                record_builder.set_grid_idx(x, y, z)
                self.fn(**call_args)
    # Copy arguments back to propagate side-effects
    for arg_dev, arg_hst in zip(args_dev, args_hst):
        if hasattr(arg_dev, "data_ptr"):
            _unwrap(arg_dev).copy_(arg_hst.to(arg_dev.device))
    _unpatch_lang()


def _create_masked_load(fn):
    @wraps(fn)
    def wrapper(ptrs, mask, other, cache_modifier, eviction_policy, is_volatile):
        tensor_ptr = record_builder.get_tensor_ptr(np.reshape(ptrs.data, (-1))[0])
        load_record = Load(
            ptr=tensor_ptr.ptr,
            shape=ptrs.data.shape,
            offsets=ptrs.data - tensor_ptr.ptr,
            masks=mask.data,
        )
        record_builder.add_record(load_record)
        return fn(
            ptrs,
            mask,
            other,
            cache_modifier,
            eviction_policy,
            is_volatile,
        )

    return wrapper


@contextmanager
def patch():
    old_grid_executor_call = GridExecutor.__call__
    old_create_masked_load = builder.create_masked_load
    GridExecutor.__call__ = _grid_executor_call
    builder.create_masked_load = _create_masked_load(builder.create_masked_load)
    yield
    GridExecutor.__call__ = old_grid_executor_call
    builder.create_masked_load = old_create_masked_load
