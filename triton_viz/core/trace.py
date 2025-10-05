from triton.runtime import KernelInterface, Autotuner
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from . import config as cfg
from ..clients import Sanitizer, Profiler, Tracer
from .client import ClientManager, Client
from .data import Launch
from typing import Union


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
        kernel: Union[JITFunction, InterpretedFunction],
        client: Union[str, Client],
    ) -> None:
        self.fn = kernel
        if isinstance(kernel, Autotuner):
            self.base_fn = kernel.base_fn
            kernel._do_bench = dummy_benchmarker
            kernel.fn = InterpretedFunction(kernel.base_fn)
            self.interpreter_fn = kernel
        elif isinstance(kernel, InterpretedFunction):
            self.base_fn = kernel.fn
            self.interpreter_fn = kernel
        elif isinstance(kernel, JITFunction):
            self.base_fn = kernel.fn
            self.interpreter_fn = InterpretedFunction(kernel.fn)
        else:
            raise TypeError(
                f"Kernel must be JITFunction or InterpretedFunction, got {type(kernel)}"
            )
        self.arg_names = kernel.arg_names
        self.client_manager = ClientManager()
        self.add_client(client)

    def run(self, *args, **kwargs):
        # Check if any client needs ASM information
        if self.client_manager.needs_asm():
            # Run warmup to get ASM information from kernel compilation
            # This doesn't actually execute the kernel, just compiles it
            try:
                # Get the original function for warmup
                if hasattr(self.fn, "warmup"):
                    # Remove client_manager and warmup from kwargs for warmup call
                    warmup_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if k not in ("client_manager", "warmup")
                    }
                    warmup_result = self.fn.warmup(*args, **warmup_kwargs)

                    if warmup_result:
                        if hasattr(warmup_result, "asm"):
                            # Distribute ASM information to all clients that need it
                            self.client_manager.distribute_asm_info(warmup_result.asm)
                        elif hasattr(warmup_result, "__getitem__"):
                            # Sometimes warmup returns a tuple (kernel, asm_info)
                            # Try to extract ASM from the result
                            for item in warmup_result:
                                if hasattr(item, "asm"):
                                    self.client_manager.distribute_asm_info(item.asm)
                                    break
            except Exception as e:
                # If warmup fails, log warning but continue execution
                import warnings

                warnings.warn(f"Failed to collect ASM information: {e}")

        with self.client_manager.patch(self.base_fn):
            kwargs.update({"client_manager": self.client_manager})
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
