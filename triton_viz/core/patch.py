import triton.language as tl
from contextlib import contextmanager
from collections.abc import Callable
from tqdm import tqdm

from . import config as cfg
from .callbacks import OpCallbacks
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
]
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
}
reduce_map: dict[type[Op], Callable] = {
    ReduceMax: tl.max,
    ReduceMin: tl.min,
    ReduceSum: tl.sum,
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
            if self.op_type == ReduceSum:
                # see triton.runtime.interpreter:ReduceOps.sum
                # First, convert input from tl.tensor to TensorHandle. Here, input tensor is args[0]
                # Then, convert return value from TensorHandle to tl.tensor
                ret = tl.core.tensor(
                    self.callbacks.op_overrider(args[0].handle, *args[1:], **kwargs),
                    args[0].dtype,
                )
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
        op_name = original_ops[op_type].__name__
        current_op = getattr(interpreter_builder, op_name)
        patched_op = PatchOp(current_op, op_type, callbacks)
        setattr(
            interpreter_builder,
            op_name,
            lambda *args, **kwargs: patched_op(*args, **kwargs),
        )
    elif op_type in reduce_map:
        op_name = reduce_map[op_type].__name__
        current_op = getattr(tl, op_name)
        patched_op = PatchOp(current_op, op_type, callbacks)
        setattr(tl, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    else:
        raise ValueError(f"Patching operator {op_type} not supported")


def unpatch_op(op_type: type[Op]):
    """
    Unregister a callback for an operator.

    :param op_name: The name of the operator to unregister the callback for.
    """
    if op_type in original_ops:
        op_name = original_ops[op_type].__name__
        setattr(interpreter_builder, op_name, original_ops[op_type])


class _LoopIter:
    def __init__(self, iterable, lineno, hooks):
        self._it = iter(iterable)
        self._lineno = lineno
        self._hooks = hooks
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
        self._before: list[Callable] = []
        self._iter_listeners: list[Callable] = []
        self._iter_overrider: Callable | None = None
        self._after: list[Callable] = []

    # Register hooks
    def add_before(self, hook):
        self._before.append(hook)

    def add_iter_listener(self, hook):
        self._iter_listeners.append(hook)

    def set_iter_overrider(self, hook):
        if self._iter_overrider is not None:
            raise RuntimeError("Only one loop_iter overrider allowed")
        self._iter_overrider = hook

    def add_after(self, hook):
        self._after.append(hook)

    # Call combined hooks
    def before_loop(self, lineno, iterable):
        for hook in self._before:
            hook(lineno, iterable)

    def loop_iter(self, lineno, idx):
        # override iteration index
        if self._iter_overrider is not None:
            new_idx = self._iter_overrider(lineno, idx)
            if new_idx is not None:
                idx = new_idx

        # call all iteration listeners
        for hook in self._iter_listeners:
            hook(lineno, idx)

        return idx

    def after_loop(self, lineno):
        for hook in self._after:
            hook(lineno)

    def loop_iter_wrapper(self, iterable, lineno):
        return _LoopIter(iterable, lineno, self)

    def clear(self):
        self._before.clear()
        self._iter_listeners.clear()
        self._iter_overrider = None
        self._after.clear()


__triton_viz_hooks = _CombinedLoopHooks()
_orig_visit_for: Callable | None = None
_orig_patch_lang: Callable | None = None
_loops_patched: bool = False


def _visit_For(self, node: ast.For):  # type: ignore[override]
    """
    for i in R:
        ...
    ==>
    for i in __triton_viz_hooks_loop(R, lineno):
        ...
    where __triton_viz_hooks_loop returns a _LoopIter object.
    """
    self.generic_visit(node)

    # __triton_viz_hooks.loop_iter(range(...), lineno)
    new_iter = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="__triton_viz_hooks", ctx=ast.Load()),
            attr="loop_iter_wrapper",
            ctx=ast.Load(),
        ),
        args=[node.iter, ast.Constant(value=node.lineno)],
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


def patch_for_loop(
    before_loop_callback: Callable | None,
    loop_iter_overrider: Callable | None,
    loop_iter_listener: Callable | None,
    after_loop_callback: Callable | None,
):
    global _orig_visit_for, _orig_patch_lang, _loops_patched
    if not _loops_patched:
        _orig_visit_for = getattr(_OrigASTTransformer, "visit_For", None)
        _OrigASTTransformer.visit_For = _visit_For  # type: ignore[assignment]
        _loops_patched = True

    # Registering hooks
    if before_loop_callback is not None:
        __triton_viz_hooks.add_before(before_loop_callback)
    if loop_iter_overrider is not None:
        __triton_viz_hooks.set_iter_overrider(loop_iter_overrider)
    if loop_iter_listener is not None:
        __triton_viz_hooks.add_iter_listener(loop_iter_listener)
    if after_loop_callback is not None:
        __triton_viz_hooks.add_after(after_loop_callback)


def unpatch_for_loop():
    global _loops_patched
    if not _loops_patched:
        return

    if _orig_visit_for is not None:
        _OrigASTTransformer.visit_For = _orig_visit_for

    __triton_viz_hooks.clear()

    _loops_patched = False


def _patch_lang(fn):
    triton_patch_lang(fn)
    fn.__globals__["__triton_viz_hooks"] = __triton_viz_hooks


def _unpatch_lang():
    import importlib
    import sys

    if tl.__name__ in sys.modules:
        importlib.reload(tl)


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
                    self.fn(**call_args)
                    # if symbolic execution, only do one iteration
                    if (
                        cfg.sanitizer_backend == "symexec"
                        and not client_manager.get_client("sanitizer").need_full_grid
                    ):
                        return

    # Removes not used reserved keywords from kwargs
    # Triton doesn't support keyword-only, variable positional or variable keyword arguments
    # It's safe to inspect only positional or keyword arguments (i.e., argspec.args)
    argspec = inspect.getfullargspec(self.fn)
    triton_viz_args = ["client_manager"]
    kwargs = {
        k: v for k, v in kwargs.items() if k in argspec.args or k in triton_viz_args
    }
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
    run_grid_loops(grid)
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
