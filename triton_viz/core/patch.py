import triton.language as tl
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional
from functools import partialmethod
from tqdm import tqdm

from .config import config as cfg
from .callbacks import OpCallbacks, ForLoopCallbacks
from .data import (
    Op,
    Allocate,
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
from .data import Flip  # separate import to avoid reordering noise
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
from triton_viz.core.nki import nki_builder


@dataclass
class AdapterResult:
    """
    For each backend, ops may have slightly different function signatures
    which we run through (backend, function)-specific adapters to return
    standardized args/kwargs for client callbacks.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def passthrough_adapter(*args: Any, **kwargs: Any) -> AdapterResult:
    """Return arguments unchanged for clients that expect the original signature."""
    return AdapterResult(*args, **kwargs)


def _program_id_adapter(axis: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(axis)


def _triton_raw_store_adapter(
    ptr: Any, value: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, value)


def _triton_store_adapter(
    ptr: Any, _value: Any, mask: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    keys = kwargs.get("keys")
    return AdapterResult(ptr, mask, keys)


def _triton_raw_load_adapter(ptr: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(ptr)


def _triton_load_adapter(
    ptr: Any, mask: Any, _other: Any, *_args: Any, **kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, mask, kwargs.get("keys"))


def _triton_dot_adapter(a: Any, b: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(a, b)


def _triton_reduce_sum_adapter(
    input_tensor, axis=None, keep_dims=False, *_args, **_kwargs
) -> AdapterResult:
    return AdapterResult(input_tensor, axis, keep_dims)


def _triton_addptr_adapter(
    ptr: Any, offset: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, offset)


def _nki_allocate_adapter(*_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult()


def _nki_load_adapter(
    src: Any, keys: Any, *, mask: Optional[Any] = None, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(src, mask, keys)


def _nki_store_adapter(
    dst: Any, keys: Any, value: Any, *, mask: Optional[Any] = None, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(dst, mask, keys)


def _nki_dot_adapter(x: Any, y: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
    return AdapterResult(x, y)


TRITON_OP_LIST = [
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

TRITON_ORIGINAL_OPS = {
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

TRITON_OP_ATTR_NAMES = {
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

TRITON_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: _program_id_adapter,
    RawStore: _triton_raw_store_adapter,
    Store: _triton_store_adapter,
    RawLoad: _triton_raw_load_adapter,
    Load: _triton_load_adapter,
    Dot: _triton_dot_adapter,
    ReduceSum: _triton_reduce_sum_adapter,
    AddPtr: _triton_addptr_adapter,
}

for op_type in TRITON_OP_LIST:
    TRITON_ADAPTERS.setdefault(op_type, passthrough_adapter)


NKI_OP_LIST = [
    Allocate,
    ProgramId,
    Load,
    Store,
    Dot,
    UnaryOp,
    MakeRange,
]

NKI_ORIGINAL_OPS = {
    ProgramId: nki_builder.program_id,
    Allocate: nki_builder.ndarray,
    Load: nki_builder.masked_load,
    Store: nki_builder.masked_store,
    Dot: nki_builder.matmul,
    UnaryOp: nki_builder._unary_op,
    MakeRange: nki_builder.arange,
}

NKI_OP_ATTR_NAMES = {
    ProgramId: "program_id",
    Allocate: "ndarray",
    Load: "masked_load",
    Store: "masked_store",
    Dot: "matmul",
    UnaryOp: "_unary_op",
    MakeRange: "arange",
}

NKI_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: _program_id_adapter,
    Allocate: _nki_allocate_adapter,
    Load: _nki_load_adapter,
    Store: _nki_store_adapter,
    Dot: _nki_dot_adapter,
}

for op_type in NKI_OP_LIST:
    NKI_ADAPTERS.setdefault(op_type, passthrough_adapter)


OPERATION_REGISTRY: dict[str, dict[str, Any]] = {
    "triton": {
        "op_list": TRITON_OP_LIST,
        "original_ops": TRITON_ORIGINAL_OPS,
        "op_attr_names": TRITON_OP_ATTR_NAMES,
        "adapters": TRITON_ADAPTERS,
    },
    "nki": {
        "op_list": NKI_OP_LIST,
        "original_ops": NKI_ORIGINAL_OPS,
        "op_attr_names": NKI_OP_ATTR_NAMES,
        "adapters": NKI_ADAPTERS,
    },
}

BUILDER = interpreter_builder
current_backend = "triton"
op_list: list[Op] = OPERATION_REGISTRY[current_backend]["op_list"]
original_ops: dict[Op, Callable] = OPERATION_REGISTRY[current_backend]["original_ops"]
_OP_ATTR_NAMES: dict[Op, str] = OPERATION_REGISTRY[current_backend]["op_attr_names"]


def get_builder_for_backend(backend: str):
    """Get the appropriate builder for a given backend."""
    return nki_builder if backend == "nki" else interpreter_builder


def detect_current_backend():
    """Detect the current backend based on which operations have been patched."""
    # Check if Triton operations have been patched first (since this example uses Triton)
    if hasattr(interpreter_builder, "create_get_program_id") and hasattr(
        interpreter_builder.create_get_program_id, "__wrapped__"
    ):
        return "triton"
    # Check if NKI operations have been patched
    elif hasattr(nki_builder, "program_id") and hasattr(
        nki_builder.program_id, "__wrapped__"
    ):
        return "nki"
    # Default to checking BUILDER global
    else:
        return "nki" if BUILDER == nki_builder else "triton"


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
        adapter: Callable[..., AdapterResult],
    ):
        self.op = op
        self.op_type = op_type
        self.callbacks = callbacks
        self.adapter = adapter

    def __call__(self, *args, **kwargs):
        if self.callbacks.before_callback:
            before_args = self.adapter(*args, **kwargs)
            self.callbacks.before_callback(*before_args.args, **before_args.kwargs)
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
            after_args = self.adapter(*args, **kwargs)
            self.callbacks.after_callback(ret, *after_args.args, **after_args.kwargs)
        return ret


def patch_op(op_type: type[Op], callbacks: OpCallbacks, backend: Optional[str] = None):
    """
    Register a callback to be called before and after an operator is executed.

    :param op_type: The type of the operator to register the callback for.
    :param callbacks: The OpCallbacks object containing before_callback, after_callback, and op_overrider.
    :param backend: The backend to use ('triton', 'nki', or None for current backend).
    """
    if backend is None:
        backend = current_backend

    if backend not in OPERATION_REGISTRY:
        raise ValueError(f"Unknown backend: {backend}")

    backend_ops = OPERATION_REGISTRY[backend]["original_ops"]
    backend_attr_names = OPERATION_REGISTRY[backend]["op_attr_names"]
    backend_adapters = OPERATION_REGISTRY[backend]["adapters"]
    backend_builder = nki_builder if backend == "nki" else interpreter_builder

    if op_type in backend_ops:
        op_name = backend_attr_names[op_type]
        original_op = backend_ops[op_type]
        adapter = backend_adapters[op_type]
        patched_op = PatchOp(original_op, op_type, callbacks, adapter)
        setattr(
            backend_builder,
            op_name,
            lambda *args, **kwargs: patched_op(*args, **kwargs),
        )
    elif backend == "triton" and (op_type in reduce_map or op_type in scan_map):
        if op_type in reduce_map:
            op_name = reduce_map[op_type].__name__
        elif op_type in scan_map:
            op_name = scan_map[op_type].__name__
        original_op = getattr(tl, op_name)
        adapter = backend_adapters[op_type]
        patched_op = PatchOp(original_op, op_type, callbacks, adapter)
        setattr(tl, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    elif backend == "triton" and op_type in math_map:
        op_name = math_map[op_type].__name__
        original_op = getattr(tl.math, op_name)
        adapter = backend_adapters[op_type]
        patched_op = PatchOp(original_op, op_type, callbacks, adapter)
        setattr(tl.math, op_name, lambda *args, **kwargs: patched_op(*args, **kwargs))
    else:
        raise ValueError(f"Patching operator {op_type} not supported")


def unpatch_op(op_type: type[Op]):
    """
    Unregister a callback for an operator.

    :param op_type: The type of the operator to unregister the callback for.
    """
    if op_type in original_ops:
        original_op = original_ops[op_type]  # type: ignore
        # Use hardcoded name from _OP_ATTR_NAMES
        op_name = _OP_ATTR_NAMES[op_type]  # type: ignore
        setattr(BUILDER, op_name, original_op)


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


def patch_lang(fn, backend):
    if backend == "triton":
        triton_patch_lang(fn)
    elif backend == "nki":
        from triton_viz.core.nki import nki_patch_lang

        nki_patch_lang()
    else:
        raise ValueError(
            f"Unsupported backend {backend} received. Triton-viz only supports one of ('triton', 'nki')."
        )

    fn.__globals__["_triton_viz_loop_patcher"] = _loop_patcher
    # Wrap tl.flip to emit a Flip record after computing result
    try:
        _orig_flip = getattr(tl, "flip", None)
        if _orig_flip is not None and not getattr(
            _orig_flip, "__triton_viz_wrapped__", False
        ):

            def _viz_flip(x, *args, **kwargs):
                # Call original flip implementation
                ret = _orig_flip(x, *args, **kwargs)
                # Best-effort extract dim
                dim = None
                if args:
                    dim = args[0]
                if "dim" in kwargs:
                    dim = kwargs.get("dim")
                # Best-effort shapes
                in_shape = None
                out_shape = None
                x_arr = None
                r_arr = None
                try:
                    # interpreter tensors may expose .data or .handle.data
                    x_data = getattr(x, "data", None)
                    if x_data is None and hasattr(x, "handle"):
                        x_data = getattr(x.handle, "data", None)
                    if x_data is not None:
                        in_shape = tuple(x_data.shape)
                        x_arr = x_data
                except Exception:
                    pass
                try:
                    r_data = getattr(ret, "data", None)
                    if r_data is None and hasattr(ret, "handle"):
                        r_data = getattr(ret.handle, "data", None)
                    if r_data is not None:
                        out_shape = tuple(r_data.shape)
                        r_arr = r_data
                except Exception:
                    pass

                # Emit a Flip record to the active tracer, if available
                try:
                    global _current_client_manager
                    cm = _current_client_manager
                    if cm is not None and hasattr(cm, "clients"):
                        tracer = cm.get_client("tracer")
                        if tracer is not None:
                            input_payload = None
                            output_payload = None
                            try:
                                # Avoid huge payloads: cap to 64k elements
                                def _maybe_list(arr):
                                    import numpy as _np

                                    if arr is None:
                                        return None
                                    try:
                                        if arr.size <= 65536:
                                            return _np.asarray(arr).tolist()
                                    except Exception:
                                        pass
                                    return None

                                input_payload = _maybe_list(x_arr)
                                output_payload = _maybe_list(r_arr)
                            except Exception:
                                pass

                            rec = Flip(
                                input_shape=in_shape or tuple(),
                                output_shape=out_shape or (in_shape or tuple()),
                                dim=int(dim) if dim is not None else 0,
                                input_data=input_payload,
                                output_data=output_payload,
                            )
                            # attach call path already handled by Flip.__post_init__
                            tracer.records.append(rec)
                except Exception:
                    # Never fail kernel execution due to viz
                    pass
                return ret

            # mark wrapper to avoid double-wrapping on subsequent patch_lang calls
            setattr(_viz_flip, "__triton_viz_wrapped__", True)
            tl.flip = _viz_flip  # type: ignore[assignment]
    except Exception:
        # If wrapping fails, continue without Flip records
        pass


def unpatch_lang(backend):
    import importlib
    import sys

    if backend == "triton":
        if tl.__name__ in sys.modules:
            importlib.reload(tl)
    elif backend == "nki":
        from triton_viz.core.nki import nki_unpatch_lang

        nki_unpatch_lang()


def _grid_executor_call(self, *args_dev, backend=None, **kwargs):
    assert backend is not None
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
                    builder.set_grid_idx(x, y, z)
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
    # Expose client_manager to tl.flip wrapper via a module-global
    global _current_client_manager
    _current_client_manager = client_manager
    # drop compatibility arg if present
    kwargs.pop("jit_fn", None)
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

    # Use the correct builder based on current backend
    builder = get_builder_for_backend(backend)

    if backend == "nki":
        builder.set_grid_dim(grid)
    else:  # triton
        builder.set_grid_dim(*grid)
    client_manager.grid_callback(grid)
    run_grid_loops(grid)
    # Copy arguments back to propagate side-effects
    self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)


def _jit_function_call(
    self, *args, backend=None, **kwargs
):  # NOTE: is this ever called?
    assert backend is not None
    patch_lang(self.fn, backend)
    return self.fn(*args, **kwargs)


@contextmanager
def patch_calls(backend):
    old_grid_executor_call = GridExecutor.__call__
    old_jit_function_call = JITFunction.__call__
    GridExecutor.__call__ = partialmethod(_grid_executor_call, backend=backend)
    JITFunction.__call__ = partialmethod(_jit_function_call, backend=backend)
    try:
        yield
    finally:
        GridExecutor.__call__ = old_grid_executor_call
        JITFunction.__call__ = old_jit_function_call
