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

    # Two-copy solver fields.
    event_id: int = -1  # stable dedup key (per launch)
    elem_size: int = 1  # for byte-overlap when > 1

    # CAS-specific raw symbolic pieces. None for non-CAS records.
    cas_cmp_value: Any = None
    cas_new_value: Any = None

    # Z3 vars representing per-program-instance nondeterminism for THIS record
    # (the fresh CAS return var, this record's loop iterator vars). The two-copy
    # solver collects these across all records and alpha-renames each ORIGINAL
    # var exactly once per copy (launch-level), so downstream records that
    # reference the same var get the same _a/_b rename. Tensor base pointers,
    # kernel scalar args, and global constants are explicitly NOT included.
    copy_local_vars: tuple[Any, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RaceReport:
    """Race detected between two memory events.

    ``first`` and ``second`` are deliberately untyped ``Any`` so the same
    record type can be produced by both ``HBSolver`` (single-copy
    ``ScalarMemoryEvent``) and ``TwoCopySymbolicHBSolver`` (two-copy
    ``SymbolicMemoryEvent``).
    """

    first: Any
    second: Any
    model: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    race_type_value: RaceType | None = None
    witness_addr: int | None = None
    witness_grid_a: tuple[int, int, int] | None = None
    witness_grid_b: tuple[int, int, int] | None = None

    @property
    def first_record(self) -> AccessEventRecord:
        return self.first.record

    @property
    def second_record(self) -> AccessEventRecord:
        return self.second.record

    @property
    def race_type(self) -> RaceType:
        if self.race_type_value is not None:
            return self.race_type_value
        # Legacy fallback used ONLY by HBSolver synthetic non-CAS reports.
        # TwoCopySymbolicHBSolver always populates race_type_value.
        first_writes = self.first.record.access_mode == "write"
        second_writes = self.second.record.access_mode == "write"
        if first_writes and second_writes:
            return RaceType.WAW
        if first_writes:
            return RaceType.RAW
        return RaceType.WAR


__all__ = [
    "AccessEventRecord",
    "AtomicKind",
    "MemorySem",
    "RaceReport",
    "RaceType",
]
