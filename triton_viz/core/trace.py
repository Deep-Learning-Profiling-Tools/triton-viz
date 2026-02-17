from copy import deepcopy
from contextlib import contextmanager, nullcontext
from functools import wraps
import threading
from triton.runtime import KernelInterface, Autotuner
from triton.runtime.autotuner import Heuristics
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from .config import config as cfg
from ..clients import Sanitizer, Profiler, Tracer
from .client import Client
from .data import Launch
from .patch import (
    patch_calls,
    patch_for_loop,
    patch_lang,
    patch_op,
    unpatch_for_loop,
    unpatch_lang,
    unpatch_op,
    OPERATION_REGISTRY,
)
from typing import Callable, Optional, Union


launches: list[Launch] = []


class TraceRuntime:
    """
    Internal tracing runtime shared by traced kernel wrappers.

    Users do not instantiate this directly. They call `triton_viz.trace(...)`,
    which returns a `TritonTrace` (or `NKITrace`) object. This mixin provides
    the common lifecycle used by those user-facing wrappers:
    patching interpreter ops, forwarding callbacks to the selected client,
    and collecting the final `Launch` record.
    """

    def __init__(self, client: Union[str, Client]) -> None:
        self.client = self._normalize_client(client)
        self.launch = Launch()
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_client(client: Union[str, Client]) -> Client:
        if isinstance(client, str):
            name = client.lower()
            if name == "sanitizer":
                return Sanitizer()
            if name == "profiler":
                return Profiler()
            if name == "tracer":
                return Tracer()
            raise ValueError(f"Unknown client: {client}")
        elif isinstance(client, Client):
            return client
        else:
            raise TypeError(f"Expected str or Client, got {type(client)}")

    def finalize(self):
        with self._lock_context():
            self.launch.records = self.client.finalize()
        launches.append(self.launch)

    def _lock_context(self):
        if cfg.num_sms > 1:
            return self._lock
        return nullcontext()

    def get_client(self, name: str) -> Optional[Client]:
        if self.client.NAME == name:
            return self.client
        return None

    @contextmanager
    def patch_warmup(self, jit_fn):
        if not hasattr(jit_fn, "warmup"):
            yield
            return

        def patcher(fn):
            @wraps(fn)
            def wrapped(*args, **kwargs):
                if not self.client.pre_warmup_callback(jit_fn, *args, **kwargs):
                    return None
                kwargs.pop("warmup", None)
                ret = fn(*args, **kwargs)
                self.client.post_warmup_callback(jit_fn, ret)
                return ret

            return wrapped

        jit_fn.warmup = patcher(jit_fn.warmup)
        try:
            yield
        finally:
            jit_fn.warmup = jit_fn.warmup.__wrapped__

    @contextmanager
    def patch_run(self, fn, backend: str):
        namespaces = OPERATION_REGISTRY[backend].namespaces
        with patch_calls(backend):
            for namespace, attrs in namespaces.items():
                for attr, op in attrs.items():
                    callbacks = self.client.register_op_callback(op)
                    patch_op(namespace, attr, callbacks, backend=backend)
            patch_for_loop(self.client.register_for_loop_callback())
            patch_lang(fn, backend)
            try:
                yield
            finally:
                for namespace, attrs in namespaces.items():
                    for attr in attrs:
                        unpatch_op(namespace, attr, backend)
                unpatch_for_loop()
                unpatch_lang(backend)

    def pre_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            return self.client.pre_run_callback(fn)

    def post_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            return self.client.post_run_callback(fn)

    def arg_callback(self, name, arg, arg_cvt):
        with self._lock_context():
            if hasattr(arg, "data_ptr"):
                self.launch.tensors.add(arg)
            self.client.arg_callback(name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int]):
        with self._lock_context():
            self.launch.grid = grid
            self.client.grid_callback(grid)

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        with self._lock_context():
            self.client.grid_idx_callback(grid_idx)


class TritonTrace(KernelInterface, TraceRuntime):
    def __init__(
        self,
        runner: Union[JITFunction, InterpretedFunction, Autotuner, Heuristics],
        client: Union[str, Client],
    ) -> None:
        self.jit_fn: Optional[JITFunction] = None
        self.base_fn: Optional[Callable] = None
        self.interpreted_fn: Optional[InterpretedFunction] = None

        def unpack_kernel(
            source: Union["TritonTrace", JITFunction, InterpretedFunction, Heuristics],
        ) -> tuple[
            Optional[JITFunction], Optional[Callable], Optional[InterpretedFunction]
        ]:
            if isinstance(source, TritonTrace):
                return source.jit_fn, source.base_fn, source.interpreted_fn
            if isinstance(source, JITFunction):
                base_fn = source.fn
                return source, base_fn, InterpretedFunction(base_fn)
            if isinstance(source, InterpretedFunction):
                return None, source.fn, source
            if isinstance(source, Heuristics):
                # Heuristics wraps another kernel, recursively unpack it
                return unpack_kernel(source.fn)
            raise TypeError(f"Unsupported runner type: {type(source)}")

        if isinstance(runner, Autotuner):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)

            # Kernel Cache: replace the benchmark with a dummy to skip performance testing.
            def dummy_benchmarker(fn, quantiles):
                fn()
                return (1.0, 1.0, 1.0)

            runner._do_bench = dummy_benchmarker
            runner.fn = self.interpreted_fn
            self.runner = runner
        elif isinstance(runner, Heuristics):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)
            runner.fn = self.interpreted_fn
            self.runner = runner
        else:
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner)
            self.runner = self.interpreted_fn

        if isinstance(runner, (Autotuner, Heuristics)):
            if self.jit_fn is not None:
                warmup_runner = deepcopy(runner)
                warmup_runner.fn = self.jit_fn
                self.warmup_runner = warmup_runner
            else:
                self.warmup_runner = None
        else:
            self.warmup_runner = self.jit_fn

        self.arg_names = runner.arg_names

        self.fn = runner

        TraceRuntime.__init__(self, client)

        # Preserve common function attributes for compatibility
        # with code that expects to access these attributes on the kernel
        if hasattr(runner, "__name__"):
            self.__name__ = runner.__name__
        elif self.base_fn and hasattr(self.base_fn, "__name__"):
            self.__name__ = self.base_fn.__name__
        else:
            self.__name__ = "<unknown>"

        if hasattr(runner, "__module__"):
            self.__module__ = runner.__module__
        elif self.base_fn and hasattr(self.base_fn, "__module__"):
            self.__module__ = self.base_fn.__module__

        if hasattr(runner, "__doc__"):
            self.__doc__ = runner.__doc__
        elif self.base_fn and hasattr(self.base_fn, "__doc__"):
            self.__doc__ = self.base_fn.__doc__

        if hasattr(runner, "__qualname__"):
            self.__qualname__ = runner.__qualname__
        elif self.base_fn and hasattr(self.base_fn, "__qualname__"):
            self.__qualname__ = self.base_fn.__qualname__

        # Preserve Triton-specific attributes (like src for JITFunction)
        if hasattr(runner, "src"):
            self.src = runner.src
        elif self.jit_fn and hasattr(self.jit_fn, "src"):
            self.src = self.jit_fn.src

    def run(self, *args, **kwargs):
        with self.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)

        with self.patch_run(self.base_fn, backend="triton"):
            kwargs.update({"trace_runtime": self})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly
        return self.interpreted_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        with self.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)


class NKITrace(KernelInterface, TraceRuntime):
    """User-facing traced wrapper for NKI kernels."""

    def __init__(self, kernel, client: str | Client) -> None:
        from neuronxcc.nki.compile import GenericKernel
        from .nki import NKIInterpretedFunction

        if isinstance(kernel, GenericKernel):
            self.interpreter_fn = NKIInterpretedFunction(kernel.func)
            self.func = kernel.func
        elif isinstance(kernel, NKIInterpretedFunction):
            self.interpreter_fn = kernel
            self.func = kernel.fn
        else:
            self.interpreter_fn = NKIInterpretedFunction(kernel)
            self.func = kernel

        TraceRuntime.__init__(self, client)

    def __getattr__(self, name):
        # Forward any missing attributes to the underlying runner
        # This allows TraceRuntime to transparently proxy attributes like 'src', 'hash', etc.
        # Use object.__getattribute__ to avoid infinite recursion
        try:
            fn = object.__getattribute__(self, "fn")
            if hasattr(fn, name):
                return getattr(fn, name)
        except AttributeError:
            pass

        try:
            jit_fn = object.__getattribute__(self, "jit_fn")
            if hasattr(jit_fn, name):
                return getattr(jit_fn, name)
        except AttributeError:
            pass

        try:
            base_fn = object.__getattribute__(self, "base_fn")
            if hasattr(base_fn, name):
                return getattr(base_fn, name)
        except AttributeError:
            pass

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, *grid):
        return KernelInterface.__getitem__(self, tuple(*grid))

    def __call__(self, *args, **kwargs):
        return self[(1, 1, 1)](*args, **kwargs)

    def run(self, *args, **kwargs):
        with self.patch_run(self.func, backend="nki"):
            kwargs.update({"trace_runtime": self})
            ret = self.interpreter_fn.run(*args, **kwargs)
            self.finalize()
            return ret


def trace(client: Union[str, Client, None] = None, backend: str = "triton"):
    """
    Create a trace object that can be used to run a kernel with instrumentation client(s).

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel. Defaults to Tracer() if not specified.
    """
    if client is None:
        client = Tracer()

    if not isinstance(client, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(client)}")

    def _is_sanitizer_client(selected: Union[str, Client]) -> bool:
        if isinstance(selected, str):
            return selected.lower() == "sanitizer"
        return isinstance(selected, Sanitizer)

    def decorator(kernel) -> TritonTrace | NKITrace | KernelInterface:
        if _is_sanitizer_client(client) and not cfg.enable_sanitizer:
            # when dry-running triton-sanitizer CLI (i.e. wrap kernels with sanitizer
            # tracing but don't actually sanitize), don't actually trace the kernel
            return kernel

        # multi-client stacking is not supported
        if isinstance(kernel, TraceRuntime):
            raise ValueError(
                "Kernel is already traced. Stacking multiple @trace decorators is not supported."
            )

        # First-time wrapping
        # Triton backend need JIT/Interpreter/Autotuner；
        # NKI allow Python function（ NKIInterpretedFunction）
        if backend == "nki":
            return NKITrace(kernel, client)
        if isinstance(
            kernel, (JITFunction, InterpretedFunction, Autotuner, Heuristics)
        ):
            if backend == "triton":
                return TritonTrace(kernel, client)
            else:
                raise ValueError(f"Unknown backend: {backend}")

        raise TypeError(
            f"Expected JITFunction, InterpretedFunction or TraceRuntime, got {type(kernel)}"
        )

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches.clear()
