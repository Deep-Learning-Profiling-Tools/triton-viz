from triton.runtime import KernelInterface
from triton import JITFunction

from .interpreter import InterpretedFunction, builder
from typing import Tuple


class Trace(KernelInterface):
    def __init__(self, kernel: JITFunction) -> None:
        assert isinstance(kernel, JITFunction), "Kernel must be a JITFunction"
        self._fn = InterpretedFunction(kernel.fn)

    def run(self, *args, **kwargs):
        # Take out warmpup from kwargs
        kwargs.pop("warmup", 0)
        return self._fn.run(*args, **kwargs)


def trace(kernel):
    return Trace(kernel)


def dump(path: str):
    launch_records = builder.launch_records
    for record in launch_records:
        print(record)


def sample(idx: Tuple):
    builder.set_sampling_grid_idx(idx)
