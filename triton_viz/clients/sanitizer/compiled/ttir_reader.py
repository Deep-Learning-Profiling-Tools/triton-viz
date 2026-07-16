"""Back-compat shim: the TTIR reader lives in ``triton_viz.clients.common``.

The reader is mechanism shared with the race detector's TTIR front-end, so
it moved to ``triton_viz.clients.common.ttir_reader``. This module only
re-exports the public surface; import from the shared module in new code.
"""

from ...common.ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    AtomicInfo,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    FuncArg,
    IterArgInfo,
    IterArgOffset,
    LoopInfo,
    LoopVar,
    Not,
    Param,
    Pid,
    PtrValue,
    Select,
    SourceLoc,
    Term,
    UnsupportedTTIR,
    parse_ttir,
)

__all__ = [
    "AccessEvent",
    "AccessGraph",
    "Arange",
    "AtomicInfo",
    "Bin",
    "BoolBin",
    "Cmp",
    "Const",
    "DataDep",
    "FuncArg",
    "IterArgInfo",
    "IterArgOffset",
    "LoopInfo",
    "LoopVar",
    "Not",
    "Param",
    "Pid",
    "PtrValue",
    "Select",
    "SourceLoc",
    "Term",
    "UnsupportedTTIR",
    "parse_ttir",
]
