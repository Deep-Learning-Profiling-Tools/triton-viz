from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import numpy.typing as npt

from ...utils.traceback_utils import TracebackInfo


class AccessType(Enum):
    LOAD = auto()
    STORE = auto()
    ATOMIC = auto()


class RaceType(Enum):
    RAW = auto()  # Read-After-Write
    WAR = auto()  # Write-After-Read
    WAW = auto()  # Write-After-Write


@dataclass
class MemoryAccess:
    """A single memory access performed by a block."""

    access_type: AccessType
    ptr: int
    offsets: npt.NDArray[np.int_]
    masks: npt.NDArray[np.bool_]
    grid_idx: tuple[int, ...]
    call_path: list[TracebackInfo] = field(default_factory=list)
    epoch: int = 0
    event_id: int = 0
    atomic_op: str | None = None


@dataclass
class SymbolicMemoryAccess:
    """A symbolic memory access recorded during the symbolic phase."""

    access_type: AccessType
    ptr_expr: Any  # SymbolicExpr for the pointer
    mask_expr: Any  # Optional SymbolicExpr for the mask
    is_data_dependent: bool
    call_path: list[TracebackInfo] = field(default_factory=list)
    epoch: int = 0
    event_id: int = 0
    atomic_op: str | None = None


@dataclass
class RaceRecord:
    """A detected data race between two blocks."""

    race_type: RaceType
    address_offset: int
    access_a: MemoryAccess
    access_b: MemoryAccess
