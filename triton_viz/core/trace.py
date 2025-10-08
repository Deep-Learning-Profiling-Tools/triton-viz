from copy import deepcopy

from triton.runtime import KernelInterface, Autotuner
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from . import config as cfg
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch
from typing import Callable, Optional, Union


launches: list[Launch] = []


def dummy_benchmarker(fn, quantiles):
    fn()
    return (1.0, 1.0, 1.0)


class Trace(KernelInterface):
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

    def __init__(
        self,
        kernel: Union[JITFunction, InterpretedFunction, Autotuner],
        client: Union[str, Client],
    ) -> None:
        self.fn = kernel

        def unpack_kernel(
            source: Union["Trace", JITFunction, InterpretedFunction]
        ) -> tuple[Optional[JITFunction], Callable, InterpretedFunction]:
            if isinstance(source, Trace):
                return source.jit_fn, source.base_fn, source.interpreter_fn
            if isinstance(source, JITFunction):
                base_fn = source.fn
                return source, base_fn, InterpretedFunction(base_fn)
            if isinstance(source, InterpretedFunction):
                return None, source.fn, source
            raise TypeError(f"Unsupported kernel type: {type(source)}")

        if isinstance(kernel, Autotuner):
            self.jit_fn, self.base_fn, interpreter_fn = unpack_kernel(kernel.fn)
            # replace the benchmark with a dummy that just calls the function once
            kernel._do_bench = dummy_benchmarker
            # replace the fn with an InterpretedFunction to avoid re-jitting
            kernel.fn = interpreter_fn
            self.interpreter_fn = kernel
        else:
            self.jit_fn, self.base_fn, self.interpreter_fn = unpack_kernel(kernel)
        self.arg_names = kernel.arg_names
        self.client_manager = ClientManager()
        self.add_client(client)

    def run(self, *args, **kwargs):
        with self.client_manager.patch(self.base_fn):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.interpreter_fn.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly
        return self.base_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        raise NotImplementedError

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


def trace(clients: Union[str, Client, None] = None):
    """
    Create a trace object that can be used to run a kernel with instrumentation clients.

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel. Defaults to Tracer() if not specified.
    """
    if clients is None:
        clients = Tracer()

    if not isinstance(clients, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(clients)}")

    def decorator(kernel) -> Trace:
        # When sanitizer is disabled, skip tracing and return the original kernel unchanged
        if cfg.disable_sanitizer:
            return kernel

        # First-time wrapping
        if isinstance(kernel, (JITFunction, InterpretedFunction, Autotuner)):
            return Trace(kernel, clients)

        # If the object is already a Trace, just append the new client(s)
        if isinstance(kernel, Trace):
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
