from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from z3 import And, BoolVal, IntVal, Not, Or, Solver, sat
from z3.z3 import BoolRef, IntNumRef, ModelRef

from .data import AccessEventRecord, RaceReport
from .hb_common import (
    as_bool,
    build_transitive_hb,
    conflicting_access_modes,
    is_acquire_sem,
    is_release_sem,
    iter_constraints,
    lane_value,
    minimal_atomic_read_from,
)


@dataclass(frozen=True)
class ScalarMemoryEvent:
    idx: int
    record: AccessEventRecord
    name: str
    lane: int
    grid_idx: tuple[int, ...] | None
    program_seq: int
    addr: Any
    active: BoolRef
    reads: BoolRef
    writes: BoolRef
    is_atomic: bool
    atomic_kind: str
    sem: str
    scope: str | None
    old_value: Any = None
    written_value: Any = None


@dataclass(frozen=True)
class RaceCheckResult:
    possible: bool
    model: dict[str, str] = field(default_factory=dict)


class HBSolver:
    """Small event-graph happens-before solver for the PR-A demo.

    This solver is intentionally isolated from the production capture path. It
    consumes synthetic ``AccessEventRecord`` inputs and checks whether
    conflicting accesses can remain unordered after program-order and
    release/acquire synchronization edges are applied.
    """

    def __init__(
        self,
        records: list[AccessEventRecord],
        extra_assumptions: tuple[Any, ...] = (),
    ) -> None:
        self.records = list(records)
        self.extra_assumptions = tuple(extra_assumptions)
        self.events = self._lower_records()
        self.hb = build_transitive_hb(self.events, self._edge)

    def find_races(self) -> list[RaceReport]:
        reports: list[RaceReport] = []

        for first, second in combinations(self.events, 2):
            result = self.check_race_possible(first, second)
            if result.possible:
                reports.append(
                    RaceReport(
                        first=first,
                        second=second,
                        model=result.model,
                        reason=(
                            "unordered conflicting memory accesses under "
                            "the current symbolic assumptions"
                        ),
                    )
                )

        return reports

    def check_race_possible(
        self,
        first: ScalarMemoryEvent,
        second: ScalarMemoryEvent,
    ) -> RaceCheckResult:
        solver = self._new_solver()
        solver.add(self._race_expr(first, second))

        if solver.check() != sat:
            return RaceCheckResult(possible=False)

        return RaceCheckResult(
            possible=True,
            model=self._model_to_dict(solver.model()),
        )

    def _lower_records(self) -> list[ScalarMemoryEvent]:
        events: list[ScalarMemoryEvent] = []

        for record in self.records:
            addrs = self._iter_addrs(record.addr_expr)
            n_lanes = len(addrs)

            for lane, addr in enumerate(addrs):
                active_terms = [
                    as_bool(lane_value(record.active, lane, n_lanes)),
                    *(as_bool(c) for c in iter_constraints(record.local_constraints)),
                ]
                active = And(*active_terms)

                raw_reads = lane_value(record.reads, lane, n_lanes)
                if raw_reads is None:
                    reads = active if record.access_mode == "read" else BoolVal(False)
                else:
                    reads = And(active, as_bool(raw_reads))

                raw_writes = lane_value(record.writes, lane, n_lanes)
                if raw_writes is None:
                    writes = active if record.access_mode == "write" else BoolVal(False)
                else:
                    writes = And(active, as_bool(raw_writes))

                name = record.debug_name or f"e{len(events)}"
                if n_lanes > 1:
                    name = f"{name}.lane{lane}"

                events.append(
                    ScalarMemoryEvent(
                        idx=len(events),
                        record=record,
                        name=name,
                        lane=lane,
                        grid_idx=record.grid_idx,
                        program_seq=record.program_seq,
                        addr=self._as_z3_value(addr),
                        active=active,
                        reads=reads,
                        writes=writes,
                        is_atomic=record.is_atomic,
                        atomic_kind=record.atomic_kind,
                        sem=record.sem,
                        scope=record.scope,
                        old_value=self._as_z3_value(
                            lane_value(record.old_value, lane, n_lanes)
                        ),
                        written_value=self._as_z3_value(
                            lane_value(record.written_value, lane, n_lanes)
                        ),
                    )
                )

        return events

    @staticmethod
    def _iter_addrs(addr_expr: Any) -> list[Any]:
        if addr_expr is None:
            raise ValueError("AccessEventRecord.addr_expr is required for HB solving")
        if isinstance(addr_expr, (list, tuple)):
            return list(addr_expr)
        return [addr_expr]

    def _same_addr(
        self, first: ScalarMemoryEvent, second: ScalarMemoryEvent
    ) -> BoolRef:
        return self._as_z3_value(first.addr) == self._as_z3_value(second.addr)

    @staticmethod
    def _scope_ok(first: ScalarMemoryEvent, second: ScalarMemoryEvent) -> BoolRef:
        if first.scope == "cta" or second.scope == "cta":
            return BoolVal(
                first.grid_idx is not None
                and second.grid_idx is not None
                and first.grid_idx == second.grid_idx
            )
        return BoolVal(True)

    @staticmethod
    def _program_order(first: ScalarMemoryEvent, second: ScalarMemoryEvent) -> BoolRef:
        if first.grid_idx is None or second.grid_idx is None:
            return BoolVal(False)
        if first.grid_idx != second.grid_idx:
            return BoolVal(False)
        if first.program_seq < 0 or second.program_seq < 0:
            return BoolVal(False)
        return BoolVal(first.program_seq < second.program_seq)

    def _synchronizes_with(
        self,
        writer: ScalarMemoryEvent,
        reader: ScalarMemoryEvent,
    ) -> BoolRef:
        return And(
            BoolVal(is_release_sem(writer.sem)),
            BoolVal(is_acquire_sem(reader.sem)),
            self._scope_ok(writer, reader),
            minimal_atomic_read_from(
                writer, reader, same_atomic_addr_fn=self._same_addr
            ),
        )

    def _initial_atomic_value(self, event: ScalarMemoryEvent) -> Any:
        tensor = event.record.tensor
        if tensor is None or event.old_value is None:
            return None
        try:
            if tensor.numel() != 1:
                return None
            addr = event.addr
            if isinstance(addr, IntNumRef):
                addr = addr.as_long()
            if not isinstance(addr, int) or addr != tensor.data_ptr():
                return None
            return IntVal(int(tensor.item()))
        except Exception:
            return None

    def _atomic_old_value_has_source(self, reader: ScalarMemoryEvent) -> BoolRef:
        if not reader.is_atomic or reader.old_value is None:
            return BoolVal(True)

        initial_value = self._initial_atomic_value(reader)
        if initial_value is None:
            return BoolVal(True)

        candidate_sources = [
            minimal_atomic_read_from(
                writer, reader, same_atomic_addr_fn=self._same_addr
            )
            for writer in self.events
            if writer.idx != reader.idx
        ]
        if candidate_sources:
            return Or(reader.old_value == initial_value, *candidate_sources)
        return reader.old_value == initial_value

    def _edge(self, first: ScalarMemoryEvent, second: ScalarMemoryEvent) -> BoolRef:
        if first.idx == second.idx:
            return BoolVal(False)
        return Or(
            self._program_order(first, second),
            self._synchronizes_with(first, second),
        )

    def _conflict(self, first: ScalarMemoryEvent, second: ScalarMemoryEvent) -> BoolRef:
        return And(
            first.active,
            second.active,
            self._same_addr(first, second),
            conflicting_access_modes(first, second),
        )

    def _race_expr(
        self, first: ScalarMemoryEvent, second: ScalarMemoryEvent
    ) -> BoolRef:
        return And(
            self._conflict(first, second),
            Not(self.hb[first.idx][second.idx]),
            Not(self.hb[second.idx][first.idx]),
        )

    def _new_solver(self) -> Solver:
        solver = Solver()

        for c in iter_constraints(self.extra_assumptions):
            solver.add(as_bool(c))

        for record in self.records:
            for c in iter_constraints(record.premises):
                solver.add(as_bool(c))

        for event in self.events:
            solver.add(self._atomic_old_value_has_source(event))

        return solver

    @staticmethod
    def _as_z3_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return BoolVal(value)
        if isinstance(value, int):
            return IntVal(value)
        return value

    @staticmethod
    def _model_to_dict(model: ModelRef) -> dict[str, str]:
        return {decl.name(): str(model[decl]) for decl in model.decls()}


# Backwards-compat: a few existing imports still reach for RaceReport from this
# module. RaceReport's canonical home is now data.py.
__all__ = [
    "HBSolver",
    "RaceCheckResult",
    "RaceReport",
    "ScalarMemoryEvent",
]
