import triton.language as tl
from contextlib import contextmanager
from collections.abc import Callable
from typing import Any, Optional
from tqdm import tqdm

from .config import config as cfg
from dataclasses import dataclass
from .callbacks import OpCallbacks, ForLoopCallbacks

from .data import (
    Op,
    RawLoad,
    Load,
    RawStore,
    Store,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    ProgramId,
    Dot,
    MakeRange,
    AddPtr,
    ReduceSum,
    Splat,
    ExpandDims,
    Broadcast,
    ReduceMax,
    ReduceMin,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
    Idiv,
    Rsqrt,
    CastImpl,
    Reshape,
    Join,
    Fabs,
    Ashr,
    Advance,
    FpToFp,
    Umulhi,
    Trans,
    CumSum,
    Bitcast,
    AtomicCas,
    AtomicRMW,
)
import inspect
import ast
from triton.runtime.interpreter import (
    GridExecutor,
    _implicit_cvt,
    interpreter_builder,
)
from triton.runtime.interpreter import _patch_lang as triton_patch_lang
from triton.runtime.interpreter import ASTTransformer as _OrigASTTransformer
from triton.runtime.interpreter import _tuple_create, _unwrap_tensor, _rewrap_tensor
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.runtime import JITFunction

op_list = [
    ProgramId,
    RawStore,
    Store,
    RawLoad,
    Load,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    Dot,
    MakeRange,
    AddPtr,
    Splat,
    ExpandDims,
    Broadcast,
    ReduceMax,
    ReduceMin,
    ReduceSum,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
    Idiv,
    Rsqrt,
    CastImpl,
    Reshape,
    Join,
    Fabs,
    Ashr,
    Advance,
    FpToFp,
    Umulhi,
    Trans,
    CumSum,
    Bitcast,
    AtomicCas,
    AtomicRMW,
]

# Hardcoded operation attribute names to avoid issues with lambda functions
_OP_ATTR_NAMES = {
    ProgramId: "create_get_program_id",
    RawStore: "create_store",
    Store: "create_masked_store",
    RawLoad: "create_load",
    Load: "create_masked_load",
    Dot: "create_dot",
    UnaryOp: "unary_op",
    BinaryOp: "binary_op",
    TernaryOp: "ternary_op",
    MakeRange: "create_make_range",
    AddPtr: "create_addptr",
    ExpandDims: "create_expand_dims",
    Broadcast: "create_broadcast",
    Splat: "create_splat",
    MakeBlockPointer: "create_make_block_ptr",
    TensorPointerLoad: "create_tensor_pointer_load",
    TensorPointerStore: "create_tensor_pointer_store",
    Idiv: "create_idiv",
    Rsqrt: "create_rsqrt",
    CastImpl: "cast_impl",
    Reshape: "create_reshape",
    Join: "create_join",
    Fabs: "create_fabs",
    Ashr: "create_ashr",
    Advance: "create_advance",
    FpToFp: "create_fp_to_fp",
    Umulhi: "create_umulhi",
    Trans: "create_trans",
    Bitcast: "create_bitcast",
    AtomicCas: "create_atomic_cas",
    AtomicRMW: "create_atomic_rmw",
}

original_ops = {
    ProgramId: interpreter_builder.create_get_program_id,
    RawStore: interpreter_builder.create_store,
    Store: interpreter_builder.create_masked_store,
    RawLoad: interpreter_builder.create_load,
    Load: interpreter_builder.create_masked_load,
    Dot: interpreter_builder.create_dot,
    UnaryOp: interpreter_builder.unary_op,
    BinaryOp: interpreter_builder.binary_op,
    TernaryOp: interpreter_builder.ternary_op,
    MakeRange: interpreter_builder.create_make_range,
    AddPtr: interpreter_builder.create_addptr,
    ExpandDims: interpreter_builder.create_expand_dims,
    Broadcast: interpreter_builder.create_broadcast,
    Splat: interpreter_builder.create_splat,
    MakeBlockPointer: interpreter_builder.create_make_block_ptr,
    TensorPointerLoad: interpreter_builder.create_tensor_pointer_load,
    TensorPointerStore: interpreter_builder.create_tensor_pointer_store,
    Idiv: interpreter_builder.create_idiv,
    Rsqrt: interpreter_builder.create_rsqrt,
    CastImpl: interpreter_builder.cast_impl,
    Reshape: interpreter_builder.create_reshape,
    Join: interpreter_builder.create_join,
    Fabs: interpreter_builder.create_fabs,
    Ashr: interpreter_builder.create_ashr,
    Advance: interpreter_builder.create_advance,
    FpToFp: interpreter_builder.create_fp_to_fp,
    Umulhi: interpreter_builder.create_umulhi,
    Trans: interpreter_builder.create_trans,
    Bitcast: interpreter_builder.create_bitcast,
    AtomicCas: interpreter_builder.create_atomic_cas,
    AtomicRMW: interpreter_builder.create_atomic_rmw,
}
reduce_map: dict[type[Op], Callable] = {
    ReduceMax: tl.max,
    ReduceMin: tl.min,
    ReduceSum: tl.sum,
}
scan_map: dict[type[Op], Callable] = {
    CumSum: tl.cumsum,
}
math_map: dict[type[Op], Callable] = {
    Umulhi: tl.math.umulhi,
}


class PatchOp:
    def __init__(
        self,
        op: Callable,
        op_type: type[Op],
        callbacks: OpCallbacks,
    ):
        self.op = op
        self.op_type = op_type
        self.callbacks = callbacks

    def __call__(self, *args, **kwargs):
        if self.callbacks.before_callback:
            self.callbacks.before_callback(*args, **kwargs)
        if self.callbacks.op_overrider:
            if self.op_type in reduce_map or self.op_type in scan_map:
                # see triton.runtime.interpreter:ReduceOps.sum
                # First, convert input from tl.tensor to TensorHandle. Here, input tensor is args[0]
                # Then, convert return value from TensorHandle to tl.tensor
                ret = tl.core.tensor(
                    self.callbacks.op_overrider(args[0].handle, *args[1:], **kwargs),
                    args[0].dtype,
                )
            elif self.op_type in math_map:
                raise NotImplementedError()
            else:
                ret = self.callbacks.op_overrider(*args, **kwargs)
                from ..clients.sanitizer.sanitizer import SymbolicExpr

                if isinstance(ret, SymbolicExpr):
                    ret.concrete_fn = self.op
        else:
            ret = self.op(*args, **kwargs)
        if self.callbacks.after_callback:
            # Pass ret so that we don't have to derive output shape from args
            self.callbacks.after_callback(ret, *args, **kwargs)
        return ret


def patch_op(op_type: type[Op], callbacks: OpCallbacks):
    """
    Register a callback to be called before and after an operator is executed.

    :param op_type: The type of the operator to register the callback for.
    :param callbacks: The OpCallbacks object containing before_callback, after_callback, and op_overrider.
    """
    if op_type in original_ops:
        # create a new function that calls the before_callback, the original op and the after_callback
        op_name = _OP_ATTR_NAMES[op_type]
        original_op = original_ops[op_type]
        patched_op = PatchOp(original_op, op_type, callbacks)
        setattr(
            interpreter_builder,
            op_name,
            lambda *args, **kwargs: patched_op(*args, **kwargs),
        )
    elif op_type in reduce_map or op_type in scan_map:
        if op_type in reduce_map:
            op_name = reduce_map[op_type].__name__
        elif op_type in scan_map:
            op_name = scan_map[op_type].__name__
        original_op = getattr(tl, op_name)
        patched_op = PatchOp(original_op, op_type, callbacks)
        setattr(tl, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    elif op_type in math_map:
        op_name = math_map[op_type].__name__
        original_op = getattr(tl.math, op_name)
        patched_op = PatchOp(original_op, op_type, callbacks)
        setattr(tl.math, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    else:
        raise ValueError(f"Patching operator {op_type} not supported")


def unpatch_op(op_type: type[Op]):
    """
    Unregister a callback for an operator.

    :param op_type: The type of the operator to unregister the callback for.
    """
    if op_type in original_ops:
        original_op = original_ops[op_type]
        # Use hardcoded name from _OP_ATTR_NAMES
        op_name = _OP_ATTR_NAMES[op_type]
        setattr(interpreter_builder, op_name, original_op)


class _LoopIter:
    def __init__(self, hooks, iterable, lineno, range_type):
        self._it = iter(iterable)
        self._lineno = lineno
        self._hooks = hooks
        # triggering range_type
        self._hooks.range_type(self._lineno, range_type)
        # triggering before_loop
        if self._hooks.before_loop:
            self._hooks.before_loop(self._lineno, iterable)

    def __iter__(self):
        return self

    def __next__(self):
        idx = None
        try:
            idx = next(self._it)
        except StopIteration:
            # Exiting the loop and triggering after_loop
            if self._hooks.after_loop:
                self._hooks.after_loop(self._lineno)
            raise

        # trigger loop overriders and loop listeners
        idx = self._hooks.loop_iter(self._lineno, idx)
        return idx


class _CombinedLoopHooks:
    """
    Combine for_loop callbacks from all clients.
    """

    def __init__(self):
        self._range_type: list[Callable] = []
        self._before: list[Callable] = []
        self._iter_listeners: list[Callable] = []
        self._iter_overrider: Optional[Callable] = None
        self._after: list[Callable] = []

    # Register hooks
    def add_range_type_callback(self, hook: Callable) -> None:
        self._range_type.append(hook)

    def add_before(self, hook: Callable) -> None:
        self._before.append(hook)

    def add_iter_listener(self, hook: Callable) -> None:
        self._iter_listeners.append(hook)

    def set_iter_overrider(self, hook: Callable) -> None:
        if self._iter_overrider is not None:
            raise RuntimeError("Only one loop_iter overrider allowed")
        self._iter_overrider = hook

    def add_after(self, hook: Callable) -> None:
        self._after.append(hook)

    # Call combined hooks
    def range_type(self, lineno: int, range_type: str) -> None:
        for hook in self._range_type:
            hook(lineno, range_type)

    def before_loop(self, lineno: int, iterable: Any) -> None:
        for hook in self._before:
            hook(lineno, iterable)

    def loop_iter(self, lineno: int, idx: Any) -> Any:
        # override iteration index
        if self._iter_overrider is not None:
            new_idx = self._iter_overrider(lineno, idx)
            if new_idx is not None:
                idx = new_idx

        # call all iteration listeners
        for hook in self._iter_listeners:
            hook(lineno, idx)

        return idx

    def after_loop(self, lineno: int) -> None:
        for hook in self._after:
            hook(lineno)

    def loop_iter_wrapper(
        self, iterable: Any, lineno: int, range_type: str
    ) -> "_LoopIter":
        return _LoopIter(self, iterable, lineno, range_type)

    def clear(self) -> None:
        self._before.clear()
        self._iter_listeners.clear()
        self._iter_overrider = None
        self._after.clear()


class _LoopPatcher:
    """Manages loop patching state and hooks."""

    def __init__(self):
        self.hooks = _CombinedLoopHooks()
        self._orig_visit_for: Optional[Callable] = None
        self._patched: bool = False

    def patch(self) -> None:
        """Apply loop patching."""
        if not self._patched:
            self._orig_visit_for = getattr(_OrigASTTransformer, "visit_For", None)
            _OrigASTTransformer.visit_For = _visit_For  # type: ignore[assignment]
            self._patched = True

    def unpatch(self) -> None:
        """Remove loop patching."""
        if not self._patched:
            return

        if self._orig_visit_for is not None:
            _OrigASTTransformer.visit_For = self._orig_visit_for

        self.hooks.clear()
        self._patched = False


_loop_patcher = _LoopPatcher()


def _visit_For(self, node: ast.For):  # type: ignore[override]
    """
    for i in R:
        ...
    ==>
    for i in _triton_viz_loop_patcher.hooks.loop_iter_wrapper(R, lineno, range_type):
        ...
    where _triton_viz_loop_patcher.hooks.loop_iter_wrapper returns a _LoopIter object.
    """
    self.generic_visit(node)

    # Detect range type
    range_type = "unknown"
    if isinstance(node.iter, ast.Call):
        func = node.iter.func
        if isinstance(func, ast.Name) and func.id == "range":
            range_type = "python_range"
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "tl"
        ):
            if func.attr == "range":
                range_type = "tl_range"
            elif func.attr == "static_range":
                range_type = "tl_static_range"

    # _triton_viz_loop_patcher.hooks.loop_iter(range(...), lineno, range_type)
    new_iter = ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="_triton_viz_loop_patcher", ctx=ast.Load()),
                attr="hooks",
                ctx=ast.Load(),
            ),
            attr="loop_iter_wrapper",
            ctx=ast.Load(),
        ),
        args=[
            node.iter,
            ast.Constant(value=node.lineno),
            ast.Constant(value=range_type),
        ],
        keywords=[],
    )

    new_for = ast.For(
        target=node.target,
        iter=new_iter,
        body=node.body,
        orelse=node.orelse,
        type_comment=node.type_comment,
    )
    return ast.fix_missing_locations(new_for)


def patch_for_loop(loop_callbacks: ForLoopCallbacks):
    _loop_patcher.patch()

    # Registering hooks
    if loop_callbacks.range_type_callback is not None:
        _loop_patcher.hooks.add_range_type_callback(loop_callbacks.range_type_callback)
    if loop_callbacks.before_loop_callback is not None:
        _loop_patcher.hooks.add_before(loop_callbacks.before_loop_callback)
    if loop_callbacks.loop_iter_overrider is not None:
        _loop_patcher.hooks.set_iter_overrider(loop_callbacks.loop_iter_overrider)
    if loop_callbacks.loop_iter_listener is not None:
        _loop_patcher.hooks.add_iter_listener(loop_callbacks.loop_iter_listener)
    if loop_callbacks.after_loop_callback is not None:
        _loop_patcher.hooks.add_after(loop_callbacks.after_loop_callback)


def unpatch_for_loop():
    _loop_patcher.unpatch()


def patch_lang(fn):
    triton_patch_lang(fn)
    fn.__globals__["_triton_viz_loop_patcher"] = _loop_patcher


def unpatch_lang():
    import importlib
    import sys

    if tl.__name__ in sys.modules:
        importlib.reload(tl)


@dataclass(frozen=True)
class FakeTensor:
    _data_ptr: int
    dtype: str
    shape: tuple[int, ...] = ()
    _stride: tuple[int, ...] = ()
    _is_contiguous: bool = True
    _element_size: int = 1

    def data_ptr(self) -> int:
        return self._data_ptr

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def is_contiguous(self) -> bool:
        return self._is_contiguous

    def numel(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def element_size(self) -> int:
        return self._element_size


def _init_args_hst(args_dev, kwargs):
    def _to_cpu(arg):
        if isinstance(arg, tuple):
            return _tuple_create(arg, map(_to_cpu, arg))
        elif isinstance(arg, TensorDescriptor):
            return TensorDescriptor(
                _to_cpu(arg.base),
                arg.shape,
                arg.strides,
                arg.block_shape,
            )
        elif not hasattr(arg, "data_ptr"):
            return arg

        unwrapped_arg = _unwrap_tensor(arg)
        cpu_arg = FakeTensor(
            _data_ptr=unwrapped_arg.data_ptr(),
            dtype=unwrapped_arg.dtype,
            shape=unwrapped_arg.shape,
            _stride=unwrapped_arg.stride(),
            _is_contiguous=unwrapped_arg.is_contiguous(),
            _element_size=unwrapped_arg.element_size(),
        )
        cpu_arg = _rewrap_tensor(cpu_arg, original_tensor=arg)
        return cpu_arg

    args_hst = [_to_cpu(arg) for arg in args_dev]

    # Process keyword arguments
    kwargs_hst = {}
    for key, value in kwargs.items():
        kwargs_hst[key] = _to_cpu(value)
    return args_hst, kwargs_hst


def _grid_executor_call(self, *args_dev, **kwargs):
    if kwargs.pop("warmup", False):
        return

    def run_grid_loops(grid):
        for x in tqdm(
            range(grid[0]),
            desc="Grid X",
            leave=False,
            disable=not cfg.report_grid_execution_progress,
        ):
            for y in tqdm(
                range(grid[1]),
                desc="Grid Y",
                leave=False,
                disable=not (cfg.report_grid_execution_progress and grid[1] > 1),
            ):
                for z in tqdm(
                    range(grid[2]),
                    desc="Grid Z",
                    leave=False,
                    disable=not (cfg.report_grid_execution_progress and grid[2] > 1),
                ):
                    interpreter_builder.set_grid_idx(x, y, z)
                    client_manager.grid_idx_callback((x, y, z))
                    if not client_manager.pre_run_callback(self.fn):
                        continue  # Skip this block
                    self.fn(**call_args)
                    if not client_manager.post_run_callback(self.fn):
                        return

    # Removes not used reserved keywords from kwargs
    # Triton doesn't support keyword-only, variable positional or variable keyword arguments
    # It's safe to inspect only positional or keyword arguments (i.e., argspec.args)
    argspec = inspect.getfullargspec(self.fn)
    triton_viz_args = ["client_manager", "jit_fn"]
    kwargs = {
        k: v for k, v in kwargs.items() if k in argspec.args or k in triton_viz_args
    }
    client_manager = kwargs.pop("client_manager")
    kwargs.pop("jit_fn")
    if cfg.virtual_memory:
        args_hst, kwargs_hst = _init_args_hst(args_dev, kwargs)
    else:
        args_hst, kwargs_hst = self._init_args_hst(args_dev, kwargs)
    # Prepare call arguments
    args = inspect.getcallargs(self.fn, *args_hst, **kwargs_hst)
    call_args = {}
    for name, arg in args.items():
        if name in self.constexprs:
            call_args[name] = arg
            ret = arg
        else:
            ret = _implicit_cvt(arg)
        client_manager.arg_callback(name, arg, ret)
        call_args[name] = ret
    call_args.pop("self", None)
    # Iterate through grid
    grid = self.grid(call_args) if callable(self.grid) else self.grid
    assert len(grid) <= 3
    grid = grid + (1,) * (3 - len(grid))
    interpreter_builder.set_grid_dim(*grid)
    client_manager.grid_callback(grid)
    run_grid_loops(grid)
    # Copy arguments back to propagate side-effects
    if not cfg.virtual_memory:
        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)


def _jit_function_call(self, *args, **kwargs):
    patch_lang(self.fn)
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
