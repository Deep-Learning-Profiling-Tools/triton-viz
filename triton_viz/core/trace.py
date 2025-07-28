from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from . import config as cfg
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch
from typing import TypeVar, Generic


launches: list[Launch] = []

T = TypeVar("T")


class TraceInterface(Generic[T]):
    def __init__(self, client: str | Client, interpreter_fn: T) -> None:
        self.client_manager = ClientManager()
        self.add_client(client)
        self.interpreter_fn: T = interpreter_fn

    @staticmethod
    def _normalize_client(client: str | Client) -> Client:
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

    def add_client(self, new_client: str | Client) -> None:
        self.client_manager.add_clients([self._normalize_client(new_client)])

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)

    def run(self, *args, **kwargs):
        with self.client_manager.patch():
            kwargs.update({"client_manager": self.client_manager})
            ret = self.interpreter_fn.run(*args, **kwargs)
            self.finalize()
            return ret


class TritonTrace(KernelInterface, TraceInterface):
    def __init__(
        self,
        kernel: JITFunction | InterpretedFunction,
        client: str | Client,
    ) -> None:
        if isinstance(kernel, InterpretedFunction):
            interpreter_fn = kernel
        elif isinstance(kernel, JITFunction):
            interpreter_fn = InterpretedFunction(kernel.fn)
        else:
            raise TypeError(f"Kernel must be JITFunction or InterpretedFunction, got {type(kernel)}")
        TraceInterface.__init__(self, client, interpreter_fn)
        self.fn = (kernel,)
        self.arg_names = kernel.arg_names

    def warmup(self, *args, **kwargs):
        raise NotImplementedError


class NKITrace(KernelInterface, TraceInterface):
    def __init__(self, kernel, client: str | Client) -> None:
        from neuronxcc.nki.compile import GenericKernel
        from .nki import NKIInterpretedFunction

        if isinstance(kernel, GenericKernel):
            interpreter_fn = NKIInterpretedFunction(kernel.func)
        elif isinstance(kernel, NKIInterpretedFunction):
            interpreter_fn = kernel
        TraceInterface.__init__(self, client, interpreter_fn)

    def __getitem__(self, *grid):
        return KernelInterface.__getitem__(self, tuple(*grid))


def trace(clients: str | Client | None = None, backend: str = "triton"):
    """
    Create a trace object that can be used to run a kernel with instrumentation clients.

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel. Defaults to Tracer() if not specified.
    """
    if clients is None:
        clients = Tracer()

    if not isinstance(clients, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(clients)}")

    def decorator(kernel) -> TritonTrace:
        # When sanitizer is off, skip tracing and return the original kernel unchanged
        if cfg.sanitizer_backend == "off":
            return kernel

        # First-time wrapping
        if backend == "triton":
            if isinstance(kernel, (JITFunction, InterpretedFunction)):
                return TritonTrace(kernel, clients)

            # If the object is already a TritonTrace, just append the new client(s)
            if isinstance(kernel, TritonTrace):
                trace = kernel
                trace.add_client(clients)
                return trace
        elif backend == "nki":
            return NKITrace(kernel, clients)

        raise TypeError(f"Expected JITFunction, InterpretedFunction or Trace, got {type(kernel)}")

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches = []
