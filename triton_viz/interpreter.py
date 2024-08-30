import inspect
import triton.language as tl
import numpy as np

from .data import (
    Launch,
    Grid,
    Tensor,
    Load,
    Store,
    BinaryOp,
    MakeRange,
    ExpandDims,
    Dot,
    Reduce,
)
from triton.runtime.interpreter import (
    GridExecutor,
    _implicit_cvt,
    RESERVED_KWS,
    interpreter_builder,
    InterpretedFunction,
)
from triton.runtime.interpreter import _patch_lang as triton_patch_lang
from triton.runtime import JITFunction
from typing import Tuple, List, Optional
from contextlib import contextmanager
from functools import wraps


def _patch_lang(fn):
    triton_patch_lang(fn)
    tl.sum = _create_reduce(tl.reduce, "sum")
    tl.min = _create_reduce(tl.reduce, "min")
    tl.max = _create_reduce(tl.reduce, "max")


def _unpatch_lang():
    import importlib
    import sys

    if tl.__name__ in sys.modules:
        importlib.reload(tl)


class RecordBuilder:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
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


def _check_storage_contiguous(tensor):
    # Note that this is different from if a tensor is accessed contiguously, so we cannot use tensor.is_contiguous()
    # 1. Sort strides from smallest to largest
    # 2. If the tensor is contiguous, the stride product should be the same of the shape product of all previous dimensions
    shape_prod = 1
    indices = sorted(range(len(tensor.stride())), key=tensor.stride().__getitem__)
    for i, index in enumerate(indices):
        stride = tensor.stride(index)
        shape = tensor.shape[index]
        if i == 0 and stride != 1:
            return False
        if i != 0 and stride != shape_prod:
            return False
        shape_prod *= shape
    return True


def _grid_executor_call(self, *args_dev, **kwargs):
    # Removes reserved keywords from kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS}
    if kwargs.pop("warmup", False):
        return
    args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
    # Remaps core language functions to interpreted ones
    _patch_lang(self.fn)
    # Prepare call arguments
    args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
    call_args = {}
    tensors = []
    for name, arg in args.items():
        if name in self.constexprs:
            call_args[name] = arg
        else:
            ret = _implicit_cvt(arg)
            if hasattr(arg, "data_ptr"):
                assert _check_storage_contiguous(
                    arg
                ), "triton-viz only supports contiguouly stored tensors for now"
                tensors.append(
                    Tensor(
                        ret.handle.data[0],
                        ret.dtype,
                        arg.stride(),
                        arg.shape,
                        arg.element_size(),
                    )
                )
            call_args[name] = ret
    call_args.pop("self", None)
    # Iterate through grid
    grid = self.grid(call_args) if callable(self.grid) else self.grid
    assert len(grid) <= 3
    grid = grid + (1,) * (3 - len(grid))
    interpreter_builder.set_grid_dim(*grid)
    record_builder.set_grid_dim(*grid)
    record_builder.add_tensors(tensors)
    record_builder.sort_tensor_handles()
    for x in range(grid[0]):
        for y in range(grid[1]):
            for z in range(grid[2]):
                interpreter_builder.set_grid_idx(x, y, z)
                record_builder.set_grid_idx(x, y, z)
                self.fn(**call_args)
    # Copy arguments back to propagate side-effects
    self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
    _unpatch_lang()


def _jit_function_call(self, *args, **kwargs):
    triton_patch_lang(self.fn)
    return self.fn(*args, **kwargs)


def check_out_of_bounds_access(ptrs, masks):
    first_ptr = np.reshape(ptrs.data, (-1))[0]
    tensor_ptr = record_builder.get_tensor_ptr(first_ptr)
    offsets = ptrs.data - tensor_ptr.ptr
    max_valid_offset = np.prod(tensor_ptr.shape) * tensor_ptr.element_size
    valid_access_masks = (offsets >= 0) & (offsets < max_valid_offset)
    invalid_access_masks = (~valid_access_masks) & masks.data
    corrected_offsets = np.where(valid_access_masks, offsets, 0)
    return (
        tensor_ptr,
        valid_access_masks & masks.data,
        invalid_access_masks,
        corrected_offsets,
        offsets,
    )


def _create_masked_load(fn):
    @wraps(fn)
    def wrapper(ptrs, masks, other, cache_modifier, eviction_policy, is_volatile):
        (
            tensor_ptr,
            valid_access_masks,
            invalid_access_masks,
            corrected_offsets,
            original_offsets,
        ) = check_out_of_bounds_access(ptrs, masks)
        load_record = Load(
            ptr=tensor_ptr.ptr,
            shape=ptrs.data.shape,
            offsets=corrected_offsets,
            access_masks=valid_access_masks,
            invalid_access_masks=invalid_access_masks,
            original_offsets=original_offsets,
            original_masks=masks.data,
        )
        record_builder.add_record(load_record)

        return fn(
            ptrs,
            masks,
            other,
            cache_modifier,
            eviction_policy,
            is_volatile,
        )

    return wrapper


def _create_masked_store(fn):
    @wraps(fn)
    def wrapper(ptrs, value, masks, cache_modifier, eviction_policy):
        (
            tensor_ptr,
            valid_access_masks,
            invalid_access_masks,
            corrected_offsets,
            original_offsets,
        ) = check_out_of_bounds_access(ptrs, masks)
        store_record = Store(
            ptr=tensor_ptr.ptr,
            shape=ptrs.data.shape,
            offsets=corrected_offsets,
            access_masks=valid_access_masks,
            invalid_access_masks=invalid_access_masks,
            original_offsets=original_offsets,
            original_masks=masks.data,
        )
        record_builder.add_record(store_record)

        return fn(ptrs, value, valid_access_masks, cache_modifier, eviction_policy)

    return wrapper


def _create_make_range(fn):
    @wraps(fn)
    def wrapper(start, stop):
        range_record = MakeRange(start=start, end=stop)
        record_builder.add_record(range_record)
        return fn(start, stop)

    return wrapper


def _create_binary_op(fn):
    @wraps(fn)
    def wrapper(lhs, rhs, op):
        ret = fn(lhs, rhs, op)
        binary_op_record = BinaryOp(
            op=op.__name__, input_shape=(lhs.data.shape), output_shape=ret.data.shape
        )
        record_builder.add_record(binary_op_record)
        return ret

    return wrapper


def _create_dot(fn):
    @wraps(fn)
    def wrapper(a, b, d, allow_tf32, maxNumImpreciseAcc):
        ret = fn(a, b, d, allow_tf32, maxNumImpreciseAcc)
        dot_record = Dot(
            input_shape=(a.data.shape, b.data.shape),
            other_shape=d.data.shape,
            output_shape=ret.data.shape,
        )
        record_builder.add_record(dot_record)
        return ret

    return wrapper


def _create_expand_dims(fn):
    @wraps(fn)
    def wrapper(arg, axis):
        ret = fn(arg, axis)
        expand_dims_record = ExpandDims(
            input_shape=arg.data.shape, index=axis, output_shape=ret.data.shape
        )
        record_builder.add_record(expand_dims_record)
        return ret

    return wrapper


def _create_reduce(fn, op_name: str):
    @wraps(fn)
    def wrapper(input, axis=None, keep_dims=False):
        mapping = {
            "max": tl.standard._elementwise_max,
            "min": tl.standard._elementwise_min,
            "sum": tl.standard._sum_combine,
        }
        ret = fn(input, axis=axis, combine_fn=mapping[op_name], keep_dims=keep_dims)
        reduce_record = Reduce(
            input_shape=input.handle.data.shape,
            index=axis,
            op=op_name,
            keep_dims=keep_dims,
            output_shape=ret.handle.data.shape,
        )
        record_builder.add_record(reduce_record)
        return ret

    return wrapper


@contextmanager
def patch():
    old_grid_executor_call = GridExecutor.__call__
    old_jit_function_call = JITFunction.__call__
    old_create_make_range = interpreter_builder.create_make_range
    old_create_masked_load = interpreter_builder.create_masked_load
    old_create_expand_dims = interpreter_builder.create_expand_dims
    old_binary_op = interpreter_builder.binary_op
    old_create_dot = interpreter_builder.create_dot
    old_create_masked_store = interpreter_builder.create_masked_store
    GridExecutor.__call__ = _grid_executor_call
    JITFunction.__call__ = _jit_function_call
    InterpretedFunction._rewrite_ast = lambda self: self.fn
    interpreter_builder.create_make_range = _create_make_range(
        interpreter_builder.create_make_range
    )
    interpreter_builder.create_masked_load = _create_masked_load(
        interpreter_builder.create_masked_load
    )
    interpreter_builder.create_expand_dims = _create_expand_dims(
        interpreter_builder.create_expand_dims
    )
    interpreter_builder.binary_op = _create_binary_op(interpreter_builder.binary_op)
    interpreter_builder.create_dot = _create_dot(interpreter_builder.create_dot)
    interpreter_builder.create_masked_store = _create_masked_store(
        interpreter_builder.create_masked_store
    )
    try:
        yield
    finally:
        GridExecutor.__call__ = old_grid_executor_call
        JITFunction.__call__ = old_jit_function_call
        interpreter_builder.create_make_range = old_create_make_range
        interpreter_builder.create_masked_load = old_create_masked_load
        interpreter_builder.create_expand_dims = old_create_expand_dims
        interpreter_builder.binary_op = old_binary_op
        interpreter_builder.create_dot = old_create_dot
        interpreter_builder.create_masked_store = old_create_masked_store
