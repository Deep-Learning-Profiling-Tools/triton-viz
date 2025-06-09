from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from typing import Union

from . import config as cfg
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch


launches: list[Launch] = []


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

    def add_client(self, new_client: Union[Client, str]) -> None:
        new_client_instance = self._normalize_client(new_client)
        self.client_manager.add_clients([new_client_instance])

    def __init__(self, kernel: JITFunction, client: Union[str, Client]) -> None:
        self.fn = kernel
        if isinstance(kernel, InterpretedFunction):
            self.interpreter_fn = kernel
        elif isinstance(kernel, JITFunction):
            self.interpreter_fn = InterpretedFunction(kernel.fn)
        else:
            raise TypeError(
                f"Kernel must be JITFunction or InterpretedFunction, got {type(kernel)}"
            )
        self.arg_names = kernel.arg_names
        self.client_manager = ClientManager()
        self.add_client(client)

    def run(self, *args, **kwargs):
        with self.client_manager.patch():
            kwargs.update({"client_manager": self.client_manager})
            ret = self.interpreter_fn.run(*args, **kwargs)
            self.finalize()
            return ret

    def warmup(self, *args, **kwargs):
        raise NotImplementedError

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


def trace(clients: Union[str, Client]):
    """
    Create a trace object that can be used to run a kernel with instrumentation clients.

    :param kernel: The kernel to run.
    :param client: A client to run with the kernel.
    """
    if not clients:
        raise ValueError("At least one client must be specified!")

    if not isinstance(clients, (str, Client)):
        raise TypeError(f"Expected str or Client, got {type(clients)}")

    def decorator(kernel) -> Trace:
        # When sanitizer is off, skip tracing and return the original kernel unchanged
        if cfg.sanitizer_backend == "off":
            return kernel

        # First-time wrapping
        if isinstance(kernel, (JITFunction, InterpretedFunction)):
            return Trace(kernel, clients)

        # If the object is already a Trace, just append the new client(s)
        if isinstance(kernel, Trace):
            trace = kernel
            trace.add_client(clients)
            return trace

        raise TypeError(f"Expected JITFunction, InterpretedFunction or Trace, got {type(kernel)}")

    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches = []
