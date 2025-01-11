from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

import os
from typing import Tuple, Union

from .config import sanitizer_backend
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch


launches: list[Launch] = []


class Trace(KernelInterface):

    def __init__(self, kernel: JITFunction, clients: Union[Tuple[Union[str, Client], ...], Union[str, Client]]) -> None:
        assert isinstance(kernel, JITFunction), "Kernel must be a JITFunction"
        self.interpreter = InterpretedFunction(kernel.fn)
        self.fn = kernel
        self.arg_names = kernel.arg_names
        init_clients: list[Client] = []
        clients = (clients,) if not isinstance(clients, tuple) else clients
        for client in clients:
            if isinstance(client, str):
                if client.lower() == "sanitizer":
                    init_clients.append(Sanitizer())
                elif client.lower() == "profiler":
                    init_clients.append(Profiler())
                elif client.lower() == "tracer":
                    init_clients.append(Tracer())
                else:
                    raise ValueError(f"Unknown client: {client}")
            else:
                init_clients.append(client)
        self.client_manager = ClientManager(init_clients)

    def run(self, *args, **kwargs):
        with self.client_manager.patch():
            kwargs.update({"client_manager": self.client_manager})
            ret = self.interpreter.run(*args, **kwargs)
            self.finalize()
            return ret

    def warmup(self, *args, **kwargs):
        raise NotImplementedError

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


def trace(clients: Union[Tuple[Union[str, Client], ...], Union[str, Client]] = ("sanitizer", "profiler")):
    """
    Create a trace object that can be used to run a kernel with instrumentation clients.

    :param kernel: The kernel to run.
    :param clients: A tuple of clients to run with the kernel.
    """
    def decorator(kernel: JITFunction) -> Trace:
        if sanitizer_backend == "off":
            return kernel
        return Trace(kernel, clients)
    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches = []
