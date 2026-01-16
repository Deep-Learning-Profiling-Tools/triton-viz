import triton.language as tl
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, cast
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue, Empty
import threading
import time
from functools import partialmethod

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

from .flip_patch import patch_flip

HAS_NKI = False
nki_builder = None
try:
    from triton_viz.core.nki import nki_builder  # type: ignore

    HAS_NKI = True
except ModuleNotFoundError:
    pass


_MISSING = object()


class _LangPatchScope:
    """Tracks patched attributes so they can be restored."""

    def __init__(self) -> None:
        self._changes: list[tuple[object, str, object]] = []

    def set_attr(self, obj: object, name: str, value: object) -> None:
        original = getattr(obj, name, _MISSING)
        self._changes.append((obj, name, original))
        setattr(obj, name, value)

    def restore(self) -> None:
        while self._changes:
            obj, name, original = self._changes.pop()
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)


_LANG_PATCH_SCOPES: dict[str, list[Any]] = {"triton": [], "nki": []}


def _push_lang_patch_scope(backend: str, scope: Any) -> None:
    _LANG_PATCH_SCOPES.setdefault(backend, []).append(scope)


def _pop_lang_patch_scope(backend: str) -> Optional[Any]:
    scopes = _LANG_PATCH_SCOPES.get(backend)
    if not scopes:
        return None
    return scopes.pop()


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


def _nki_reduce_sum_adapter(
    input_tensor: Any, *args: Any, mask: Any = None, **kwargs: Any
) -> AdapterResult:
    axis = args[0] if args else kwargs.get("axis")
    keep_dims = kwargs.get("keep_dims", kwargs.get("keepdims", False))
    return AdapterResult(input_tensor, axis, keep_dims)


TRITON_NAMESPACES: dict[Any, dict[type[Op], str]] = {
    interpreter_builder: {
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
        Bitcast: "create_bitcast",
        AtomicCas: "create_atomic_cas",
        AtomicRMW: "create_atomic_rmw",
    },
    tl: {
        ReduceMax: "max",
        ReduceMin: "min",
        ReduceSum: "sum",
        CumSum: "cumsum",
        Trans: "trans",
    },
    tl.math: {
        Umulhi: "umulhi",
    },
}
TRITON_ORIGINAL_OPS: dict[Any, dict[str, Op]] = {}
for namespace, attrs in TRITON_NAMESPACES.items():
    for op, attr in attrs.items():
        if op not in TRITON_ORIGINAL_OPS:  # umulhi use interpreter_builder, not tl.math
            TRITON_ORIGINAL_OPS[op] = getattr(namespace, attr)
TRITON_OP_LIST: set[type[Op]] = set()
for op_to_attrs in TRITON_NAMESPACES.values():
    TRITON_OP_LIST |= op_to_attrs.keys()

TRITON_OP_ATTR_NAMES = TRITON_NAMESPACES[interpreter_builder]

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

NKI_OP_LIST: set[type[Op]] = set()
NKI_ORIGINAL_OPS: dict[Any, dict[str, type[Op]]] = {}
NKI_OP_ATTR_NAMES: dict[type[Op], str] = {}
NKI_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {}
if HAS_NKI:
    assert nki_builder is not None

    NKI_NAMESPACES: dict[Any, dict[type[Op], str]] = {
        nki_builder: {
            ProgramId: "program_id",
            Allocate: "ndarray",
            Load: "masked_load",
            Store: "masked_store",
            Dot: "matmul",
            UnaryOp: "_unary_op",
            ReduceSum: "sum",
            MakeRange: "arange",
        }
    }
    for op_to_attrs in NKI_NAMESPACES.values():
        NKI_OP_LIST |= op_to_attrs.keys()

    NKI_OP_ATTR_NAMES = NKI_NAMESPACES[nki_builder]
    for namespace, attrs in NKI_NAMESPACES.items():
        for op, attr in attrs.items():
            if (
                op not in NKI_ORIGINAL_OPS
            ):  # umulhi use interpreter_builder, not tl.math
                NKI_ORIGINAL_OPS[op] = getattr(namespace, attr)

    NKI_ADAPTERS = {
        ProgramId: _program_id_adapter,
        Allocate: _nki_allocate_adapter,
        Load: _nki_load_adapter,
        Store: _nki_store_adapter,
        Dot: _nki_dot_adapter,
        ReduceSum: _nki_reduce_sum_adapter,
    }

    for op_type in NKI_OP_LIST:
        NKI_ADAPTERS.setdefault(op_type, passthrough_adapter)


OPERATION_REGISTRY: dict[str, dict[str, Any]] = {
    "triton": {
        "builder": interpreter_builder,
        "op_list": TRITON_OP_LIST,
        "original_ops": TRITON_ORIGINAL_OPS,
        "op_attr_names": TRITON_OP_ATTR_NAMES,
        "adapters": TRITON_ADAPTERS,
        "namespaces": TRITON_NAMESPACES,
    },
    "nki": {
        "builder": nki_builder,
        "op_list": NKI_OP_LIST,
        "original_ops": NKI_ORIGINAL_OPS,
        "op_attr_names": NKI_OP_ATTR_NAMES,
        "adapters": NKI_ADAPTERS,
        "namespaces": NKI_NAMESPACES,
    },
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
reshape_map: dict[type[Op], Callable] = {
    Trans: tl.trans,
}

_thread_local_interpreter_state = threading.local()
_thread_local_interpreter_state.grid_idx = None  # just set a default


def _set_thread_grid_idx(self, x: int, y: int, z: int) -> None:
    _thread_local_interpreter_state.grid_idx = (x, y, z)


# Bind to the builder class so attribute access uses thread-local storage.
_interp_cls = interpreter_builder.__class__
_interp_cls.set_grid_idx = _set_thread_grid_idx  # type: ignore[attr-defined]
_interp_cls.grid_idx = property(lambda self: _thread_local_interpreter_state.grid_idx)  # type: ignore[attr-defined]


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
        if cfg.num_sms > 1:
            # periodically sleep briefly so other worker threads can run
            _YIELD_INTERVAL_SEC = 0.005  # Request GIL handoff roughly every 5ms
            _YIELD_SLEEP_SEC = (
                0.0005  # Small positive sleep to encourage OS-level yield
            )
            now = time.perf_counter()
            last = getattr(_thread_local_interpreter_state, "last_yield_ts", 0.0)
            if now - last >= _YIELD_INTERVAL_SEC:
                _thread_local_interpreter_state.last_yield_ts = now
                time.sleep(_YIELD_SLEEP_SEC)

        if self.callbacks.before_callback:
            before_args = self.adapter(*args, **kwargs)
            self.callbacks.before_callback(*before_args.args, **before_args.kwargs)
        if self.callbacks.op_overrider:
            if self.op_type in math_map:
                raise NotImplementedError("Patching math ops not yet supported")
            elif self.op_type in {**reduce_map, **scan_map, **reshape_map}:
                # see triton.runtime.interpreter:ReduceOps.sum
                # First, convert input from tl.tensor to TensorHandle. Here, input tensor is args[0]
                # Then, convert return value from TensorHandle to tl.tensor
                ret = tl.core.tensor(
                    self.callbacks.op_overrider(args[0].handle, *args[1:], **kwargs),
                    args[0].dtype,
                )
                fn = cast(Any, ret.handle)
                if fn is not None:
                    fn.concrete_fn = self.op
            else:
                ret = self.callbacks.op_overrider(*args, **kwargs)
                if ret is not None:
                    if self.op_type == RawLoad:
                        ret.concrete_fn = TRITON_ORIGINAL_OPS[Load]
                    elif self.op_type == RawStore:
                        ret.concrete_fn = TRITON_ORIGINAL_OPS[Store]
                    else:
                        ret.concrete_fn = self.op
        else:
            ret = self.op(*args, **kwargs)
        if self.callbacks.after_callback:
            # Pass ret so that we don't have to derive output shape from args
            after_args = self.adapter(*args, **kwargs)
            self.callbacks.after_callback(ret, *after_args.args, **after_args.kwargs)
        return ret


def patch_op(op_type: type[Op], callbacks: OpCallbacks, backend: str):
    """
    Register a callback to be called before and after an operator is executed.

    :param op_type: The type of the operator to register the callback for.
    :param callbacks: The OpCallbacks object containing before_callback, after_callback, and op_overrider.
    :param backend: The backend to use ('triton', 'nki', or None for current backend).
    """
    if backend not in OPERATION_REGISTRY:
        raise ValueError(f"Unknown backend: {backend}")

    backend_ops = OPERATION_REGISTRY[backend]["original_ops"]
    backend_adapters = OPERATION_REGISTRY[backend]["adapters"]
    namespaces = OPERATION_REGISTRY[backend]["namespaces"]
    for namespace, ops in namespaces.items():
        if op_type in ops:
            attr = ops[op_type]
            original_op = backend_ops[op_type]
            adapter = backend_adapters[op_type]
            patched_op = PatchOp(original_op, op_type, callbacks, adapter)
            setattr(namespace, attr, patched_op)
            break
    else:
        raise ValueError(f"Patching operator {op_type} not supported")


def unpatch_op(op_type: type[Op], backend: str):
    """
    Unregister a callback for an operator.

    :param op_type: The type of the operator to unregister the callback for.
    """
    backend_ops = OPERATION_REGISTRY[backend]["original_ops"]
    namespaces = OPERATION_REGISTRY[backend]["namespaces"]
    for namespace, ops in namespaces.items():
        if op_type in ops:
            attr = ops[op_type]
            original_op = backend_ops[op_type]
            setattr(namespace, attr, original_op)
            break
    else:
        raise ValueError(f"Patching operator {op_type} not supported")


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
        self._range_wrapper_factory: Optional[Callable] = None
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

    def set_range_wrapper_factory(self, hook: Callable) -> None:
        if self._range_wrapper_factory is not None:
            raise RuntimeError("Only one range_wrapper_factory allowed")
        self._range_wrapper_factory = hook

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
        self,
        iterable_callable: Callable,
        iter_args,
        iter_kwargs,
        lineno: int,
        range_type: str,
    ) -> "_LoopIter":
        args = tuple(iter_args) if iter_args is not None else ()
        kwargs = dict(iter_kwargs) if iter_kwargs is not None else {}

        if self._range_wrapper_factory is not None:
            wrapped = self._range_wrapper_factory(
                None, lineno, range_type, args, kwargs, iterable_callable
            )
            if wrapped is not None:
                iterable = wrapped
            else:
                iterable = iterable_callable(*args, **kwargs)
        else:
            iterable = iterable_callable(*args, **kwargs)
        return _LoopIter(self, iterable, lineno, range_type)

    def clear(self) -> None:
        self._range_type.clear()
        self._before.clear()
        self._iter_listeners.clear()
        self._iter_overrider = None
        self._range_wrapper_factory = None
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
    for i in _triton_viz_loop_patcher.hooks.loop_iter_wrapper(iter_callable, args, kwargs, lineno, range_type):
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

    if isinstance(node.iter, ast.Call):
        iter_callable = node.iter.func
        iter_args = ast.Tuple(elts=node.iter.args, ctx=ast.Load())
        kw_keys = []
        kw_vals = []
        for kw in node.iter.keywords:
            if kw.arg is None:  # skip **kwargs for simplicity
                continue
            kw_keys.append(ast.Constant(value=kw.arg))
            kw_vals.append(kw.value)
        iter_kwargs = ast.Dict(keys=kw_keys, values=kw_vals)
    else:
        iter_callable = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=node.iter,
        )
        iter_args = ast.Tuple(elts=[], ctx=ast.Load())
        iter_kwargs = ast.Dict(keys=[], values=[])

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
            iter_callable,
            iter_args,
            iter_kwargs,
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
    if loop_callbacks.range_wrapper_factory is not None:
        _loop_patcher.hooks.set_range_wrapper_factory(
            loop_callbacks.range_wrapper_factory
        )
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
        scope = triton_patch_lang(fn)
    elif backend == "nki":
        from triton_viz.core.nki import nki_patch_lang

        scope = _LangPatchScope()
        nki_patch_lang(scope)
    else:
        raise ValueError(
            f"Unsupported backend {backend} received. Triton-viz only supports one of ('triton', 'nki')."
        )

    _push_lang_patch_scope(backend, scope)

    fn.__globals__["_triton_viz_loop_patcher"] = _loop_patcher
    patch_flip(scope, lambda: _current_client_manager)


def unpatch_lang(backend):
    scope = _pop_lang_patch_scope(backend)
    if scope is not None and hasattr(scope, "restore"):
        scope.restore()


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


def _grid_executor_call(self, *args_dev, backend=None, **kwargs):
    assert backend is not None
    if kwargs.pop("warmup", False):
        return

    builder = OPERATION_REGISTRY[backend]["builder"]

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

    builder.set_grid_dim(*grid)
    client_manager.grid_callback(grid)
    total_blocks = grid[0] * grid[1] * grid[2]
    max_workers = min(cfg.num_sms, total_blocks)

    def run_grid_loops(grid):
        tasks: SimpleQueue = SimpleQueue()
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    tasks.put((x, y, z))

        stop_event = threading.Event()

        def _worker():
            while not stop_event.is_set():
                try:
                    x, y, z = tasks.get_nowait()
                except Empty:
                    return
                interpreter_builder.set_grid_idx(x, y, z)
                client_manager.grid_idx_callback((x, y, z))
                if not client_manager.pre_run_callback(self.fn):
                    continue
                self.fn(**call_args)
                if not client_manager.post_run_callback(self.fn):
                    stop_event.set()
                    return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker) for _ in range(max_workers)]
            for fut in futures:
                fut.result()

    def run_grid_loops_1thread(grid):
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    interpreter_builder.set_grid_idx(x, y, z)
                    client_manager.grid_idx_callback((x, y, z))
                    if not client_manager.pre_run_callback(self.fn):
                        continue
                    self.fn(**call_args)
                    if not client_manager.post_run_callback(self.fn):
                        return

    if cfg.enable_timing:
        import time

        start_time = time.time()

    if max_workers == 1:
        run_grid_loops_1thread(grid)
    else:
        run_grid_loops(grid)

    if cfg.enable_timing:
        end_time = time.time()
        elapsed_time = end_time - start_time
        name = self.fn.__name__
        print(f"Triton-Viz: execution time for {name}: {elapsed_time * 1000:.3f} ms")
    # Copy arguments back to propagate side-effects
    if not cfg.virtual_memory:
        self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)


def _jit_function_call(self, *args, backend=None, **kwargs):
    assert backend is not None
    patch_lang(self.fn, backend)
    try:
        return self.fn(*args, **kwargs)
    finally:
        unpatch_lang(backend)


patch_calls_scope = 0


@contextmanager
def patch_calls(backend):
    global patch_calls_scope
    if patch_calls_scope == 0:
        # only patch at top-level scope (e.g. with patch_calls_top(): with patch_calls_bottom())
        old_grid_executor_call = GridExecutor.__call__
        old_jit_function_call = JITFunction.__call__
        GridExecutor.__call__ = partialmethod(_grid_executor_call, backend=backend)
        JITFunction.__call__ = partialmethod(_jit_function_call, backend=backend)
    patch_calls_scope += 1
    try:
        yield
    finally:
        patch_calls_scope -= 1
        if patch_calls_scope == 0:
            # only unpatch at top-level scope (e.g. in above example, don't
            # unpatch in between patch_calls_bottom() and patch_calls_top() scopes
            GridExecutor.__call__ = old_grid_executor_call
            JITFunction.__call__ = old_jit_function_call
