from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction
from triton.runtime import Autotuner

from typing import Tuple, Union
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch


launches: list[Launch] = []


class Trace(KernelInterface):

    def __init__(self, kernel: Union[JITFunction, Autotuner], clients: Union[Tuple[Union[str, Client], ...], Union[str, Client]]) -> None:
        if isinstance(kernel, Autotuner):
            self.is_autotuner = True
            self.autotuner_config = kernel.configs[0]
            self.autotuner = kernel
            self.autotuner_cached = False
            kernel = kernel.fn
        elif isinstance(kernel, JITFunction):
            self.is_autotuner = False
        else:
            raise ValueError("Kernel must be a JITFunction or Autotuner")
        self.fn = InterpretedFunction(kernel.fn)
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
        # run autotuner once
        if self.is_autotuner and not self.autotuner_cached:
            ret = self.autotuner.run(*args, **kwargs)
            self.autotuner_cached = True
            self.autotuner_config = self.autotuner.best_config
        with self.client_manager.patch():
            kwargs.update({"client_manager": self.client_manager})
            if self.is_autotuner:
                ret = self.fn.run(*args, **kwargs, **self.autotuner_config.all_kwargs())
            else:
                ret = self.fn.run(*args, **kwargs)
            self.finalize()
            return ret

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)


def trace(clients: Union[Tuple[Union[str, Client], ...], Union[str, Client]] = ("sanitizer", "profiler")):
    """
    Create a trace object that can be used to run a kernel with instrumentation clients.

    :param kernel: The kernel to run.
    :param clients: A tuple of clients to run with the kernel.
    """
    def decorator(kernel: Union[JITFunction, Autotuner]) -> Trace:
        return Trace(kernel, clients)
    return decorator


def clear() -> None:
    """
    Clear all traces.
    """
    global launches
    launches = []
