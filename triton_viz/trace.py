from triton.runtime import KernelInterface
from triton import JITFunction

from .interpreter import InterpretedFunction


class Trace(KernelInterface):
    def __init__(
        self, kernel: JITFunction
    ) -> None:
        assert isinstance(kernel, JITFunction), "Kernel must be a JITFunction"
        self._fn = InterpretedFunction(kernel.fn)

    def run(self, *args, **kwargs):
        # Take out warmpup from kwargs
        kwargs.pop("warmup", 0)
        return self._fn.run(*args, **kwargs)


def trace(kernel):
    return Trace(kernel)
