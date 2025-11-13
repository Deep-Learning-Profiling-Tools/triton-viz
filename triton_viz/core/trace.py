from copy import deepcopy
from triton.runtime import KernelInterface, Autotuner
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from .config import config as cfg
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
        runner: Union[JITFunction, InterpretedFunction, Autotuner],
        client: Union[str, Client],
    ) -> None:
        self.fn = runner

        def unpack_kernel(
            source: Union["Trace", JITFunction, InterpretedFunction],
        ) -> tuple[
            Optional[JITFunction], Optional[Callable], Optional[InterpretedFunction]
        ]:
            if isinstance(source, Trace):
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
        self.client_manager = ClientManager()
        self.add_client(client)

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

        with self.client_manager.patch_run(self.base_fn):
            kwargs.update({"client_manager": self.client_manager})
            kwargs.update({"jit_fn": self.jit_fn})
            ret = self.runner.run(*args, **kwargs)
            self.finalize()
            return ret

    def __call__(self, *args, **kwargs):
        # When a traced JIT function is called from within another JIT function,
        # we need to execute the underlying function directly
        return self.base_fn(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        with self.client_manager.patch_warmup(self.jit_fn):
            if self.warmup_runner:
                self.warmup_runner.warmup(*args, **kwargs)

    def finalize(self):
        self.client_manager.finalize()
        launches.append(self.client_manager.launch)

    def __getattr__(self, name):
        # Forward any missing attributes to the underlying runner
        # This allows Trace to transparently proxy attributes like 'src', 'hash', etc.
        # First check if runner has it
        if hasattr(self.fn, name):
            return getattr(self.fn, name)
        # Then check jit_fn
        if hasattr(self.jit_fn, name):
            return getattr(self.jit_fn, name)
        # Finally check base_fn
        if hasattr(self.base_fn, name):
            return getattr(self.base_fn, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


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
