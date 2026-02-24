from copy import deepcopy
from ..utils.traceback_utils import CODE_KEYS, get_code_key
from triton.runtime import KernelInterface, Autotuner
from triton.runtime.autotuner import Heuristics
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from .config import config as cfg
from ..clients import Sanitizer, Profiler, Tracer, RaceDetector
from .client import ClientManager, Client
from .data import Launch
from . import patch
from typing import Callable, Optional, Union
import types


launches: list[Launch] = []


class TraceInterface:
    def __init__(self, client: Union[str, Client]) -> None:
        self.client_manager = ClientManager()
        self.add_client(client)

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
            if name == "race_detector":
                return RaceDetector()
            raise ValueError(f"Unknown client: {client}")
        elif isinstance(client, Client):
            return client
        else:
            raise TypeError(f"Expected str or Client, got {type(client)}")

    def add_client(self, new_client: Union[str, Client]) -> None:
        self.client_manager.add_clients([self._normalize_client(new_client)])

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


class TritonTrace(KernelInterface, TraceInterface):
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

        TraceInterface.__init__(self, client)

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
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)

        with self.client_manager.patch_run(self.base_fn, backend="triton"):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly

        # check that client sets match for calling and called functions
        outer_client_manager = getattr(patch, "_current_client_manager", None)
        if outer_client_manager is not None:
            outer_clients = set(outer_client_manager.clients)
            inner_clients = set(self.client_manager.clients)
            if outer_clients != inner_clients:
                raise RuntimeError(
                    "nested traced calls require matching clients; "
                    f"outer={outer_clients}, inner={inner_clients}"
                )

        return self.interpreted_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)


class NKITrace(KernelInterface, TraceInterface):
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

        TraceInterface.__init__(self, client)

    def __getattr__(self, name):
        # Forward any missing attributes to the underlying runner
        # This allows Trace to transparently proxy attributes like 'src', 'hash', etc.
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
        with self.client_manager.patch_run(self.func, backend="nki"):
            kwargs.update({"client_manager": self.client_manager})
            ret = self.interpreter_fn.run(*args, **kwargs)
            self.finalize()
            return ret


def trace_source(kernel):
    """
    Add the kernel code to be traceable within stack traces for clients
    (e.g. to capture source code to display with visualizer/client).
    You can also use this to decorate other functions that a kernel calls.
    """
    base_fn = kernel
    while not isinstance(base_fn, types.FunctionType):
        # base_fn may be a raw function but also a JITFunction, Autotuner, InterpretedFunction, ...
        # we want to strip away the wrappers until we get to the python function
        base_fn = base_fn.fn
    CODE_KEYS.add(get_code_key(base_fn))
    return kernel


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
        if cfg.cli_active and isinstance(kernel, TraceInterface):
            raise RuntimeError(
                "@triton_viz.trace() decorator cannot be used together with "
                "CLI wrapper (e.g., triton-sanitizer / triton-profiler). "
                "Please remove the @triton_viz.trace() decorator from your code "
                "when using CLI tools."
            )

        if _is_sanitizer_client(client) and not cfg.enable_sanitizer:
            # when dry-running triton-sanitizer CLI (i.e. wrap kernels with sanitizer
            # tracing but don't actually sanitize), don't actually trace the kernel
            return kernel

        # If the object is already initialized as a TraceInterface, just append the new client(s)
        if isinstance(kernel, TraceInterface):
            trace = kernel
            trace.add_client(client)
            return trace

        trace_source(kernel)

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
            f"Expected JITFunction, InterpretedFunction or Trace, got {type(kernel)}"
        )

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches.clear()
