from triton.runtime import KernelInterface
from triton.runtime.interpreter import InterpretedFunction
from triton import JITFunction

from .interpreter import patch, record_builder
from typing import Tuple


class Trace(KernelInterface):
    def __init__(self, kernel: JITFunction) -> None:
        assert isinstance(kernel, JITFunction), "Kernel must be a JITFunction"
        self._fn = InterpretedFunction(kernel.fn)

    def run(self, *args, **kwargs):
        with patch():
            return self._fn.run(*args, **kwargs)


def trace(kernel):
    return Trace(kernel)


def dump(path: str):
    # TODO: Dump the record_builder to a file
    for launch in record_builder.launches:
        print(launch)


def sample(idx: Tuple):
    record_builder.set_sampling_grid_idx(idx)
