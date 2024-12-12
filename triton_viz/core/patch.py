import triton.language as tl
from contextlib import contextmanager
from typing import Callable, Type, Dict
from tqdm import tqdm

from .config import report_grid_execution_progress
from .data import Op, Store, Load, Dot, BinaryOp, ExpandDims, MakeRange, ReduceMax, ReduceMin, ReduceSum
import inspect
from triton.runtime.interpreter import (
    GridExecutor,
    _implicit_cvt,
    interpreter_builder,
)
from triton.runtime.interpreter import _patch_lang as triton_patch_lang
from triton.runtime import JITFunction

op_list = [Store, Load, Dot, BinaryOp, ExpandDims, MakeRange, ReduceMax, ReduceMin, ReduceSum]
original_ops = {
    Store: interpreter_builder.create_masked_store,
    Load: interpreter_builder.create_masked_load,
    Dot: interpreter_builder.create_dot,
    BinaryOp: interpreter_builder.binary_op,
    ExpandDims: interpreter_builder.create_expand_dims,
    MakeRange: interpreter_builder.create_make_range,
}
reduce_map: Dict[Type[Op], Callable] = {
    ReduceMax: tl.max,
    ReduceMin: tl.min,
    ReduceSum: tl.sum
}


class PatchOp:
    def __init__(self, op, before_callback, after_callback, op_overrider):
        self.op = op
        self.before_callback = before_callback
        self.after_callback = after_callback
        self.op_overrider = op_overrider

    def __call__(self, *args, **kwargs):
        if self.before_callback:
            self.before_callback(*args, **kwargs)
        if self.op_overrider:
            ret = self.op_overrider(*args, **kwargs)
        else:
            ret = self.op(*args, **kwargs)
        if self.after_callback:
            # Pass ret so that we don't have to derive output shape from args
            self.after_callback(ret, *args, **kwargs)
        return ret


def patch_op(op_type: Type[Op], before_callback: Callable, after_callback: Callable, op_overrider: Callable):
    """
    Register a callback to be called before and after an operator is executed.

    :param op_name: The name of the operator to register the callback for.
    :param before_callback: The callback to be called before the operator is executed.
    :param after_callback: The callback to be called after the operator is executed.
    """
    if op_type in original_ops:
        # create a new function that calls the before_callback, the original op and the after_callback
        op_name = original_ops[op_type].__name__
        current_op = getattr(interpreter_builder, op_name)
        patched_op = PatchOp(current_op, before_callback, after_callback, op_overrider)
        setattr(interpreter_builder, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    elif op_type in [ReduceMax, ReduceMin, ReduceSum]:
        op_name = reduce_map[op_type].__name__
        current_op = getattr(tl, op_name)
        patched_op = PatchOp(current_op, before_callback, after_callback, op_overrider)
        setattr(tl, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    else:
        raise ValueError(f"Patching operator {op_type} not supported")


def unpatch_op(op_type: Type[Op]):
    """
    Unregister a callback for an operator.

    :param op_name: The name of the operator to unregister the callback for.
    """
    if op_type in original_ops:
        op_name = original_ops[op_type].__name__
        setattr(interpreter_builder, op_name, original_ops[op_type])


def _patch_lang(fn):
    triton_patch_lang(fn)


def _unpatch_lang():
    import importlib
    import sys

    if tl.__name__ in sys.modules:
        importlib.reload(tl)


def _grid_executor_call(self, *args_dev, **kwargs):
    if kwargs.pop("warmup", False):
        return
    # Removes not used reserved keywords from kwargs
    # Triton doesn't support keyword-only, variable positional or variable keyword arguments
    # It's safe to inspect only positional or keyword arguments (i.e., argspec.args)
    argspec = inspect.getfullargspec(self.fn)
    triton_viz_args = ["client_manager"]
    kwargs = {k: v for k, v in kwargs.items() if k in argspec.args or k in triton_viz_args}
    client_manager = kwargs.pop("client_manager")
    args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
    # Remaps core language functions to interpreted ones
    _patch_lang(self.fn)
    # Prepare call arguments
    args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
    call_args = {}
    for name, arg in args.items():
        if name in self.constexprs:
            call_args[name] = arg
        else:
            ret = _implicit_cvt(arg)
            client_manager.arg_callback(arg, ret)
            call_args[name] = ret
    call_args.pop("self", None)
    # Iterate through grid
    grid = self.grid(call_args) if callable(self.grid) else self.grid
    assert len(grid) <= 3
    grid = grid + (1,) * (3 - len(grid))
    interpreter_builder.set_grid_dim(*grid)
    client_manager.grid_callback(grid)
    for x in tqdm(range(grid[0]), desc='Grid X', leave=False, disable=not report_grid_execution_progress):
        for y in tqdm(range(grid[1]), desc='Grid Y', leave=False, disable=not (report_grid_execution_progress and grid[1] > 1)):
            for z in tqdm(range(grid[2]), desc='Grid Z', leave=False, disable=not (report_grid_execution_progress and grid[2] > 1)):
                interpreter_builder.set_grid_idx(x, y, z)
                client_manager.grid_idx_callback((x, y, z))
                self.fn(**call_args)
    # Copy arguments back to propagate side-effects
    self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
    _unpatch_lang()


def _jit_function_call(self, *args, **kwargs):
    triton_patch_lang(self.fn)
    return self.fn(*args, **kwargs)


@contextmanager
def patch_calls():
    old_grid_executor_call = GridExecutor.__call__
    old_jit_function_call = JITFunction.__call__
    GridExecutor.__call__ = _grid_executor_call
    JITFunction.__call__ = _jit_function_call
    try:
        yield
    finally:
        GridExecutor.__call__ = old_grid_executor_call
        JITFunction.__call__ = old_jit_function_call
