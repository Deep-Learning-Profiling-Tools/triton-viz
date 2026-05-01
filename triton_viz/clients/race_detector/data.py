from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

import torch

from ...core.data import Op


MemorySem = Literal["plain", "relaxed", "acquire", "release", "acq_rel"]
AtomicKind = Literal["none", "cas", "rmw"]


class RaceType(Enum):
    RAW = auto()  # Read-After-Write
    WAR = auto()  # Write-After-Read
    WAW = auto()  # Write-After-Write


@dataclass
class AccessEventRecord:
    op_type: type[Op]
    access_mode: Literal["read", "write"]
    tensor: torch.Tensor | None = None
    tensor_name: str | None = None
    symbolic_expr: Any = None
    addr_expr: Any = None
    premises: tuple[Any, ...] = field(default_factory=tuple)
    local_constraints: tuple[Any, ...] = field(default_factory=tuple)
    source_location: tuple[str, int, str] | None = None
    grid_idx: tuple[int, ...] | None = None
    program_seq: int = -1
    debug_name: str | None = None
    active: Any = True
    reads: Any = None
    writes: Any = None
    is_atomic: bool = False
    atomic_kind: AtomicKind = "none"
    sem: MemorySem = "plain"
    scope: str | None = None
    old_value: Any = None
    written_value: Any = None
