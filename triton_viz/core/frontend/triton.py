from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partialmethod
import inspect
from queue import Empty, SimpleQueue
import threading
import time
from typing import Any, cast
import warnings

import triton.language as tl
from triton import knobs
from triton.runtime import JITFunction
from triton.runtime.interpreter import (
    ASTTransformer as _OrigASTTransformer,
    GridExecutor,
    _implicit_cvt,
    _patch_builtin,
    _patch_lang as triton_patch_lang,
    _rewrap_tensor,
    _tuple_create,
    _unwrap_tensor,
    interpreter_builder,
    interpreter_semantic,
)
from triton.tools.tensor_descriptor import TensorDescriptor

from ..callbacks import OpCallbacks
from ..config import config as cfg
from ..data import (
    AddPtr,
    Advance,
    Ashr,
    AtomicCas,
    AtomicRMW,
    BinaryOp,
    Bitcast,
    Broadcast,
    CastImpl,
    CumSum,
    Dot,
    ExpandDims,
    Fabs,
    Fma,
    FpToFp,
    Idiv,
    Join,
    Load,
    MakeBlockPointer,
    MakeRange,
    Op,
    ProgramId,
    RawLoad,
    RawStore,
    ReduceMax,
    ReduceMin,
    ReduceOr,
    ReduceSum,
    ReduceXor,
    Reshape,
    Rsqrt,
    Sort,
    Splat,
    Split,
    Store,
    TensorPointerLoad,
    TensorPointerStore,
    TernaryOp,
    Trans,
    Umulhi,
    UnaryOp,
)
from ..flip_patch import patch_flip
from .base import (
    AdapterResult,
    Frontend,
    LoopPatcher,
    _LangPatchScope,
    _MISSING,
    register_frontend,
)
from ...transformers.for_loop_patcher import _visit_For as triton_viz_visit_For


def program_id_adapter(axis: Any, *_args: Any, **_kwargs: Any) -> AdapterResult:
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
    input_tensor: Any,
    axis: Any = None,
    keep_dims: bool = False,
    *_args: Any,
    **_kwargs: Any,
) -> AdapterResult:
    return AdapterResult(input_tensor, axis, keep_dims)


def _triton_addptr_adapter(
    ptr: Any, offset: Any, *_args: Any, **_kwargs: Any
) -> AdapterResult:
    return AdapterResult(ptr, offset)


TRITON_NAMESPACES: dict[Any, dict[str, type[Op]]] = {
    interpreter_builder: {
        "create_get_program_id": ProgramId,
        "create_store": RawStore,
        "create_masked_store": Store,
        "create_load": RawLoad,
        "create_masked_load": Load,
        "create_dot": Dot,
        "unary_op": UnaryOp,
        "binary_op": BinaryOp,
        "ternary_op": TernaryOp,
        "create_fma": Fma,
        "create_make_range": MakeRange,
        "create_addptr": AddPtr,
        "create_expand_dims": ExpandDims,
        "create_broadcast": Broadcast,
        "create_splat": Splat,
        "create_make_block_ptr": MakeBlockPointer,
        "create_tensor_pointer_load": TensorPointerLoad,
        "create_tensor_pointer_store": TensorPointerStore,
        "create_idiv": Idiv,
        "create_rsqrt": Rsqrt,
        "cast_impl": CastImpl,
        "create_reshape": Reshape,
        "create_trans": Trans,
        "create_join": Join,
        "create_split": Split,
        "create_fabs": Fabs,
        "create_ashr": Ashr,
        "create_advance": Advance,
        "create_fp_to_fp": FpToFp,
        "create_umulhi": Umulhi,
        "create_bitcast": Bitcast,
        "create_ptr_to_int": CastImpl,
        "create_int_to_ptr": CastImpl,
        "create_atomic_cas": AtomicCas,
        "create_atomic_rmw": AtomicRMW,
    },
    tl: {
        "max": ReduceMax,
        "min": ReduceMin,
        "sum": ReduceSum,
        "xor_sum": ReduceXor,
        "reduce_or": ReduceOr,
        "sort": Sort,
        "cumsum": CumSum,
    },
    tl.standard: {
        "max": ReduceMax,
        "min": ReduceMin,
        "sum": ReduceSum,
        "xor_sum": ReduceXor,
        "reduce_or": ReduceOr,
    },
    tl.math: {
        "umulhi": Umulhi,
    },
}


TRITON_ADAPTERS: dict[type[Op], Callable[..., AdapterResult]] = {
    ProgramId: program_id_adapter,
    RawStore: _triton_raw_store_adapter,
    Store: _triton_store_adapter,
    RawLoad: _triton_raw_load_adapter,
    Load: _triton_load_adapter,
    Dot: _triton_dot_adapter,
    ReduceSum: _triton_reduce_sum_adapter,
    AddPtr: _triton_addptr_adapter,
}


def _triton_language_attr_targets() -> list[tuple[Any, tuple[str, ...]]]:
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

    targets: list[tuple[Any, tuple[str, ...]]] = []
    for lang in (tl, tl.core):
        targets.append((lang.tensor, tensor_attrs))
        targets.append((lang, lang_attrs))
        targets.append((lang.dtype, ("to_ir",)))
        if lang == tl:
            targets.append((lang.math, ()))
    if hasattr(tl.core, "tensor_descriptor_base"):
        targets.append((tl.core.tensor_descriptor_base, ()))
    return targets


def _triton_snapshot_scope(fn: Callable[..., Any]) -> _LangPatchScope:
    """
    Stores Triton attributes into a LangPatchScope for later unpatching.
    This is to be run before patching with the interpreter.
    This is equivalent to what triton>=3.6.0 does natively
    but also works for triton<3.6.0.
    """

    scope = _LangPatchScope()
    # Triton's interpreter patch mutates global tl/tl.core objects. Snapshot
    # both unconditionally so nested or generated JIT functions without a
    # direct `tl` global still restore the language state after tracing.
    for obj, attrs in _triton_language_attr_targets():
        for name, member in inspect.getmembers(obj):
            if tl.core.is_builtin(member):
                scope.set_attr(obj, name, member)
        for attr in attrs:
            if hasattr(obj, attr):
                scope.set_attr(obj, attr, getattr(obj, attr))

    return scope


def _triton_extra_builtin_modules() -> tuple[Any, ...]:
    modules = []
    extra = getattr(tl, "extra", None)
    if extra is None:
        return ()
    for name in ("cuda", "hip"):
        module = getattr(extra, name, None)
        if module is not None:
            modules.append(module)
    return tuple(modules)


def _patch_triton_inline_asm(scope: _LangPatchScope) -> None:
    warned = False

    def _warn_inline_asm_approximation_once() -> None:
        nonlocal warned
        if warned:
            return
        warned = True
        warnings.warn(
            "Triton inline assembly is approximated in trace mode by returning "
            "input tensor values; traced values may differ from the numeric result "
            "of the real inline assembly.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _inline_asm_elementwise_fallback(
        asm,
        constraints,
        args,
        dtype,
        is_pure,
        pack,
        **kwargs,
    ):
        _warn_inline_asm_approximation_once()
        if isinstance(dtype, (list, tuple)):
            return tuple(args[0].to(_dtype) for _dtype in dtype)
        return args[0].to(dtype)

    scope.set_attr(tl, "inline_asm_elementwise", _inline_asm_elementwise_fallback)


def _patch_triton_semantic_to_tensor(scope: _LangPatchScope) -> None:
    original_to_tensor = interpreter_semantic.to_tensor

    def _to_tensor_symbolic_aware(x, check_type=True):
        unwrapped = x.value if isinstance(x, tl.constexpr) else x
        if isinstance(unwrapped, tl.core.tensor):
            return unwrapped
        return original_to_tensor(x, check_type)

    scope.set_attr(interpreter_semantic, "to_tensor", _to_tensor_symbolic_aware)


def _patch_triton_interpret_knob(scope: _LangPatchScope) -> None:
    scope.set_attr(knobs.runtime, "interpret", True)


_thread_local_interpreter_state = threading.local()
_thread_local_interpreter_state.grid_idx = None  # just set a default
_current_client_manager = None


def _get_current_client_manager():
    return _current_client_manager


def _set_thread_grid_idx(self, x: int, y: int, z: int) -> None:
    _thread_local_interpreter_state.grid_idx = (x, y, z)


# Bind to the builder class so attribute access uses thread-local storage.
_interp_cls = interpreter_builder.__class__
_interp_cls.set_grid_idx = _set_thread_grid_idx  # type: ignore[attr-defined]
_interp_cls.grid_idx = property(lambda self: _thread_local_interpreter_state.grid_idx)  # type: ignore[attr-defined]


def _maybe_yield_for_multism() -> None:
    if cfg.num_sms <= 1:
        return

    # Periodically sleep briefly so other worker threads can run.
    yield_interval_sec = 0.005  # Request GIL handoff roughly every 5ms.
    yield_sleep_sec = 0.0005  # Small positive sleep to encourage OS-level yield.
    now = time.perf_counter()
    last = getattr(_thread_local_interpreter_state, "last_yield_ts", 0.0)
    if now - last >= yield_interval_sec:
        _thread_local_interpreter_state.last_yield_ts = now
        time.sleep(yield_sleep_sec)


def _run_op_overrider(
    op: Callable,
    op_type: type[Op],
    callbacks: OpCallbacks,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    if op_type in frontend.namespaces[tl.math].values():
        raise NotImplementedError("Patching math ops not yet supported")
    elif op_type in frontend.namespaces[tl].values():
        # see triton.runtime.interpreter:ReduceOps.sum
        # First, convert input from tl.tensor to TensorHandle. Here, input tensor is args[0]
        # Then, convert return value from TensorHandle to tl.tensor
        symbolic_ret = callbacks.op_overrider(args[0].handle, *args[1:], **kwargs)
        if isinstance(symbolic_ret, tuple):
            ret_parts = []
            for sym_elem in symbolic_ret:
                _shape = getattr(sym_elem, "shape", ())
                _dtype = getattr(sym_elem, "dtype", None)
                if _shape and _dtype:
                    elem_dtype = tl.block_type(_dtype, list(_shape))
                else:
                    elem_dtype = _dtype or args[0].dtype
                elem_tensor = tl.core.tensor(sym_elem, elem_dtype)
                fn = cast(Any, elem_tensor.handle)
                if fn is not None:
                    fn.concrete_fn = op
                ret_parts.append(elem_tensor)
            return tuple(ret_parts)

        _shape = getattr(symbolic_ret, "shape", ())
        _dtype = getattr(symbolic_ret, "dtype", None)
        if _shape and _dtype:
            ret_dtype = tl.block_type(_dtype, list(_shape))
        else:
            ret_dtype = _dtype or args[0].dtype
        ret = tl.core.tensor(symbolic_ret, ret_dtype)
        fn = cast(Any, ret.handle)
        if fn is not None:
            fn.concrete_fn = op
        return ret

    ret = callbacks.op_overrider(*args, **kwargs)
    if ret is not None:
        original_ops = frontend.original_ops
        if op_type == RawLoad:
            ret.concrete_fn = original_ops[interpreter_builder]["create_masked_load"]
        elif op_type == RawStore:
            ret.concrete_fn = original_ops[interpreter_builder]["create_masked_store"]
        elif op_type == Ashr:
            ret.concrete_fn = original_ops[interpreter_builder]["binary_op"]
        else:
            ret_parts = ret if isinstance(ret, tuple) else (ret,)
            for elem in ret_parts:
                elem.concrete_fn = op
    return ret


_loop_patcher = LoopPatcher(_OrigASTTransformer, triton_viz_visit_For)


def _patch_for_loop() -> None:
    _loop_patcher.patch()


def _unpatch_for_loop() -> None:
    _loop_patcher.unpatch()


def _patch_lang(fn, client_manager=None) -> _LangPatchScope:
    scope = _triton_snapshot_scope(fn)
    triton_patch_lang(fn)
    for module in _triton_extra_builtin_modules():
        _patch_builtin(module, interpreter_builder, scope)
    _patch_triton_inline_asm(scope)
    _patch_triton_semantic_to_tensor(scope)
    _patch_triton_interpret_knob(scope)

    if client_manager is not None:
        scope.set_item(fn.__globals__, "_triton_viz_loop_patcher", client_manager)
    patch_flip(scope, _get_current_client_manager)
    return scope


@dataclass(frozen=True)
class FakeTensor:
    _data_ptr: int
    dtype: str
    shape: tuple[int, ...] = ()
    _stride: tuple[int, ...] = ()
    _is_contiguous: bool = True
    _element_size: int = 1
    device: str = "fake_tensor"

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


def _grid_executor_call(self, *args_dev, frontend_name=None, **kwargs):
    assert frontend_name == "triton"
    if kwargs.pop("warmup", False):
        return

    builder = frontend.builder
    launch_num_warps = kwargs.get("num_warps", 4)

    # Removes not used reserved keywords from kwargs
    # Triton doesn't support keyword-only, variable positional or variable keyword arguments
    # It's safe to inspect only positional or keyword arguments (i.e., argspec.args)
    argspec = inspect.getfullargspec(self.fn)
    triton_viz_args = ["client_manager", "jit_fn"]
    kwargs = {
        k: v for k, v in kwargs.items() if k in argspec.args or k in triton_viz_args
    }
    client_manager = kwargs.pop("client_manager")

    # Expose client_manager to tl.flip wrapper and nested traced calls.
    global _current_client_manager
    previous_client_manager = _current_client_manager
    _current_client_manager = client_manager
    try:
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
            start_time = time.time()

        old_num_warps = getattr(builder.options, "num_warps", _MISSING)
        object.__setattr__(builder.options, "num_warps", launch_num_warps)
        try:
            if max_workers == 1:
                run_grid_loops_1thread(grid)
            else:
                run_grid_loops(grid)
        finally:
            if old_num_warps is _MISSING:
                object.__delattr__(builder.options, "num_warps")
            else:
                object.__setattr__(builder.options, "num_warps", old_num_warps)

        if cfg.enable_timing:
            end_time = time.time()
            elapsed_time = end_time - start_time
            name = self.fn.__name__
            print(
                f"Triton-Viz: execution time for {name}: "
                f"{elapsed_time * 1000:.3f} ms"
            )
        # Copy arguments back to propagate side-effects
        if not cfg.virtual_memory:
            self._restore_args_dev(args_dev, args_hst, kwargs, kwargs_hst)
    finally:
        _current_client_manager = previous_client_manager


def _jit_function_call(self, *args, frontend_name=None, **kwargs):
    assert frontend_name == "triton"
    scope = _patch_lang(self.fn, client_manager=_current_client_manager)
    try:
        return self.fn(*args, **kwargs)
    finally:
        scope.restore()


patch_calls_scope = 0


@contextmanager
def _patch_calls():
    global patch_calls_scope
    if patch_calls_scope == 0:
        # only patch at top-level scope (e.g. with patch_calls_top(): with patch_calls_bottom())
        old_grid_executor_call = GridExecutor.__call__
        old_jit_function_call = JITFunction.__call__
        GridExecutor.__call__ = partialmethod(
            _grid_executor_call,
            frontend_name="triton",
        )
        JITFunction.__call__ = partialmethod(
            _jit_function_call,
            frontend_name="triton",
        )
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


class TritonFrontend(Frontend):
    def __init__(self):
        definition = Frontend.from_namespaces(
            name="triton",
            builder=interpreter_builder,
            namespaces=TRITON_NAMESPACES,
            adapters=TRITON_ADAPTERS,
        )
        super().__init__(
            name=definition.name,
            builder=definition.builder,
            original_ops=definition.original_ops,
            adapters=definition.adapters,
            namespaces=definition.namespaces,
        )

    def current_client_manager(self):
        return _get_current_client_manager()

    def maybe_yield_for_multism(self) -> None:
        _maybe_yield_for_multism()

    def run_op_overrider(
        self,
        op: Callable,
        op_type: type[Op],
        callbacks: OpCallbacks,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ):
        return _run_op_overrider(op, op_type, callbacks, args, kwargs)

    def patch_for_loop(self) -> None:
        _patch_for_loop()

    def unpatch_for_loop(self) -> None:
        _unpatch_for_loop()

    def patch_lang(self, fn, client_manager=None) -> _LangPatchScope:
        return _patch_lang(fn, client_manager=client_manager)

    def patch_calls(self):
        return _patch_calls()


frontend = register_frontend(TritonFrontend())
