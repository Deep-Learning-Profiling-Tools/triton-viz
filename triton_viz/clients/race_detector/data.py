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
    atomic_sem: str | None = None
    atomic_scope: str | None = None
    atomic_cmp: npt.NDArray | None = None
    atomic_val: npt.NDArray | None = None
    atomic_old: npt.NDArray | None = None
    read_mask: npt.NDArray[np.bool_] | None = None
    write_mask: npt.NDArray[np.bool_] | None = None
    legacy_atomic: bool = False


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
    atomic_sem: str | None = None
    atomic_scope: str | None = None
    atomic_cmp_expr: Any = None
    atomic_val_expr: Any = None


@dataclass
class RaceRecord:
    """A detected data race between two blocks."""

    race_type: RaceType
    address_offset: int
    access_a: MemoryAccess
    access_b: MemoryAccess


def effects_at_addr(access: MemoryAccess, addr: int) -> tuple[bool, bool]:
    """Return (reads, writes) for a given access at a specific byte address.

    For accesses without read_mask/write_mask (legacy), falls back to
    AccessType-based classification:
      LOAD -> (True, False), STORE -> (False, True), ATOMIC -> (True, True)
    """
    if access.read_mask is None and access.write_mask is None:
        # Legacy fallback based on AccessType
        if access.access_type == AccessType.LOAD:
            return (True, False)
        elif access.access_type == AccessType.STORE:
            return (False, True)
        else:  # ATOMIC
            return (True, True)

    # Find the lane index for this address
    lane_indices = np.where(access.masks & (access.offsets == addr))[0]
    if len(lane_indices) == 0:
        return (False, False)

    if access.read_mask is not None:
        reads = bool(np.any(access.read_mask[lane_indices]))
    else:
        reads = access.access_type in (AccessType.LOAD, AccessType.ATOMIC)

    if access.write_mask is not None:
        writes = bool(np.any(access.write_mask[lane_indices]))
    else:
        writes = access.access_type in (AccessType.STORE, AccessType.ATOMIC)

    return (reads, writes)
