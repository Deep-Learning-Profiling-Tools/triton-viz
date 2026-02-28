import triton.language as tl
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue, Empty
import threading
import time
from functools import partialmethod

from .config import config as cfg
from .callbacks import OpCallbacks, ForLoopCallbacks

from .data import (
    Op,
    RawLoad,
    RawStore,
)
import inspect
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
from ..transformers.for_loop_patcher import _visit_For as triton_viz_visit_For

from .flip_patch import patch_flip
from ..frontends.base import AdapterResult, OPERATION_REGISTRY


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


def _triton_snapshot_scope(fn: Callable[..., Any]) -> _LangPatchScope:
    """
    Stores Triton attributes into a LangPatchScope for later unpatching.
    This is to be run before patching with the interpreter.
    This is equivalent to what triton>=3.6.0 does natively
    but also works for triton<3.6.0.
    """

    def _capture_builtin_attrs(scope: _LangPatchScope, obj: Any) -> None:
        for name, member in inspect.getmembers(obj):
            if tl.core.is_builtin(member):
                scope.set_attr(obj, name, member)

    scope = _LangPatchScope()
    tensor_attrs = ("__index__", "__bool__", "__repr__", "__str__", "T")
    lang_attrs = (
        "range",
        "static_range",
        "static_assert",
        "static_print",
        "multiple_of",
        "max_contiguous",
        "max_constancy",
        "reduce",
        "associative_scan",
    )
    langs = [
        value
        for value in fn.__globals__.values()
        if inspect.ismodule(value) and value in [tl, tl.core]
    ]
    for lang in langs:
        _capture_builtin_attrs(scope, lang)
        _capture_builtin_attrs(scope, lang.tensor)
        if lang == tl:
            _capture_builtin_attrs(scope, lang.math)

        for attr in tensor_attrs:
            if hasattr(lang.tensor, attr):
                scope.set_attr(lang.tensor, attr, getattr(lang.tensor, attr))

        for attr in lang_attrs:
            if hasattr(lang, attr):
                scope.set_attr(lang, attr, getattr(lang, attr))
        scope.set_attr(lang.dtype, "to_ir", lang.dtype.to_ir)

    if hasattr(tl.core, "tensor_descriptor_base"):
        _capture_builtin_attrs(scope, tl.core.tensor_descriptor_base)

    return scope


def _pop_lang_patch_scope(backend: str) -> Any | None:
    scopes = _LANG_PATCH_SCOPES.get(backend)
    if not scopes:
        return None
    return scopes.pop()


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
            if (
                self.op_type
                in OPERATION_REGISTRY["triton"].namespaces[tl.math].values()
            ):
                raise NotImplementedError("Patching math ops not yet supported")
            elif self.op_type in OPERATION_REGISTRY["triton"].namespaces[tl].values():
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
            else:  # interpreter_builder
                ret = self.callbacks.op_overrider(*args, **kwargs)
                if ret is not None:
                    original_ops = OPERATION_REGISTRY["triton"].original_ops
                    if self.op_type == RawLoad:
                        ret.concrete_fn = original_ops[interpreter_builder][
                            "create_masked_load"
                        ]
                    elif self.op_type == RawStore:
                        ret.concrete_fn = original_ops[interpreter_builder][
                            "create_masked_store"
                        ]
                    else:
                        ret.concrete_fn = self.op
        else:
            ret = self.op(*args, **kwargs)
        if self.callbacks.after_callback:
            # Pass ret so that we don't have to derive output shape from args
            after_args = self.adapter(*args, **kwargs)
            self.callbacks.after_callback(ret, *after_args.args, **after_args.kwargs)
        return ret


def patch_op(namespace: Any, attr: str, callbacks: OpCallbacks, backend: str):
    """
    Register a callback to be called before and after an operator is executed.

    :param namespace: The namespace object that owns the operator.
    :param attr: The attribute name for the operator on the namespace.
    :param callbacks: The OpCallbacks object containing before_callback, after_callback, and op_overrider.
    :param backend: The backend to use ('triton', 'nki', or None for current backend).
    """
    if backend not in OPERATION_REGISTRY:
        raise ValueError(f"Unknown backend: {backend}")

    registry = OPERATION_REGISTRY[backend]
    op_type = registry.namespaces[namespace][attr]
    original_op = registry.original_ops[namespace][attr]
    adapter = registry.adapters[op_type]
    patched_op = PatchOp(original_op, op_type, callbacks, adapter)
    setattr(namespace, attr, patched_op)


def unpatch_op(namespace: Any, attr: str, backend: str):
    """
    Unregister a callback for an operator.

    :param namespace: The namespace object that owns the operator.
    :param attr: The attribute name for the operator on the namespace.
    """
    registry = OPERATION_REGISTRY[backend]
    original_op = registry.original_ops[namespace][attr]
    setattr(namespace, attr, original_op)


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
        self._iter_overrider: Callable | None = None
        self._range_wrapper_factory: Callable | None = None
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
        self._orig_visit_for: Callable | None = None
        self._patched: bool = False

    def patch(self) -> None:
        """Apply loop patching."""
        if not self._patched:
            self._orig_visit_for = getattr(_OrigASTTransformer, "visit_For", None)
            _OrigASTTransformer.visit_For = triton_viz_visit_For  # type: ignore[assignment]
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
        scope = _triton_snapshot_scope(fn)
        triton_patch_lang(fn)
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

    builder = OPERATION_REGISTRY[backend].builder

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
