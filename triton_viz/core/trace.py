from copy import deepcopy
from triton.runtime import KernelInterface, Autotuner
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from .config import config as cfg
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch
from typing import Callable, Optional, Union, TypeVar


launches: list[Launch] = []

T = TypeVar("T")


def dummy_benchmarker(fn, quantiles):
    fn()
    return (1.0, 1.0, 1.0)


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
        runner: Union[JITFunction, InterpretedFunction, Autotuner],
        client: str | Client,
    ) -> None:
        self.jit_fn: Optional[JITFunction] = None
        self.base_fn: Optional[Callable] = None
        self.interpreted_fn: Optional[InterpretedFunction] = None

        def unpack_kernel(
            source: Union["TritonTrace", JITFunction, InterpretedFunction],
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
            raise TypeError(f"Unsupported runner type: {type(source)}")

        if isinstance(runner, Autotuner):
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner.fn)
            # replace the benchmark with a dummy that just calls the function once
            runner._do_bench = dummy_benchmarker
            # replace the fn with an InterpretedFunction to avoid re-jitting
            runner.fn = self.interpreted_fn
            # make a deepcopy of the runner for warmup
            warmup_runner = deepcopy(runner)
            warmup_runner.fn = self.jit_fn
            self.runner = runner
            self.warmup_runner = warmup_runner
        else:
            self.jit_fn, self.base_fn, self.interpreted_fn = unpack_kernel(runner)
            self.runner = self.interpreted_fn
            self.warmup_runner = self.jit_fn

        self.arg_names = runner.arg_names

        self.fn = runner

        TraceInterface.__init__(self, client)

    def warmup(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)

    def run(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)

        with self.client_manager.patch_run(self.base_fn):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly
        if self.base_fn is None:
            raise RuntimeError("No base function to call!")
        return self.base_fn(*args, **kwargs)


class NKITrace(KernelInterface, TraceInterface):
    def __init__(self, kernel, client: str | Client) -> None:
        from neuronxcc.nki.compile import GenericKernel
        from .nki import NKIInterpretedFunction

        if isinstance(kernel, GenericKernel):
            # This is wrong
            self.interpreter_fn = NKIInterpretedFunction(kernel.func)
            self.func = kernel.func
        elif isinstance(kernel, NKIInterpretedFunction):
            self.interpreter_fn = kernel
            self.func = kernel.fn
        else:
            self.interpreter_fn = NKIInterpretedFunction(kernel)
            self.func = kernel

        TraceInterface.__init__(self, client)

    def __getitem__(self, *grid):
        return KernelInterface.__getitem__(self, tuple(*grid))

    def __call__(self, *args, **kwargs):
        return self[(1, 1, 1)](*args, **kwargs)

    def run(self, *args, **kwargs):
        kwargs.update({"client_manager": self.client_manager})
        ret = self.interpreter_fn.run(*args, **kwargs)
        self.finalize()
        return ret


def trace(clients: Union[str, Client, None] = None, backend: str = "triton"):
    """
    Create a trace object that can be used to run a kernel with instrumentation clients.

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel. Defaults to Tracer() if not specified.
    """
    if clients is None:
        clients = Tracer()

    if not isinstance(clients, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(clients)}")

    def decorator(kernel) -> TraceInterface:
        # When sanitizer is disabled, skip tracing and return the original kernel unchanged
        if cfg.disable_sanitizer:
            return kernel

        # First-time wrapping
        if isinstance(kernel, (JITFunction, InterpretedFunction, Autotuner)):
            if backend == "triton":
                return TritonTrace(kernel, clients)
            elif backend == "nki":
                return NKITrace(kernel, clients)
            else:
                raise ValueError(f"Unknown backend: {backend}")

        # Handle NKI functions specifically
        if backend == "nki":
            from .nki import NKIInterpretedFunction

            if isinstance(kernel, NKIInterpretedFunction):
                return NKITrace(kernel, clients)
            else:
                # Wrap plain functions as NKIInterpretedFunction
                interpreted_fn = NKIInterpretedFunction(kernel)
                return NKITrace(interpreted_fn, clients)

        # If the object is already initialized as a TraceInterface, just append the new client(s)
        if isinstance(kernel, TraceInterface):
            trace = kernel
            trace.add_client(clients)
            return trace

        raise TypeError(
            f"Expected JITFunction, InterpretedFunction or Trace, got {type(kernel)}"
        )

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches = []
