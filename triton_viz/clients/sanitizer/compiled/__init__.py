"""Compiled-mode sanitizer: static out-of-bounds checking over TritonGPU TTIR.

The eager mode (``SymbolicSanitizer``) checks each global memory access as
the interpreter executes the kernel. The compiled mode analyzes the kernel's
TTIR once per specialization and instantiates the check per launch with the
concrete tensor metadata and scalar argument values — proving in-boundedness
for ALL inputs consistent with those scalars and the grid, with no
interpreted execution. Selected via ``Sanitizer(compile=True)``.

Data-dependent (gather/indirect) addressing and block-pointer kernels are
marked unsupported; the eager ``Sanitizer()`` covers those.
"""

from .client import CompiledSanitizer
from .oob import CompiledOOB, LaunchContext, TensorMeta, check_graph
from .ttir_reader import AccessGraph, UnsupportedTTIR, parse_ttir

__all__ = [
    "AccessGraph",
    "CompiledOOB",
    "CompiledSanitizer",
    "LaunchContext",
    "TensorMeta",
    "UnsupportedTTIR",
    "check_graph",
    "parse_ttir",
]
