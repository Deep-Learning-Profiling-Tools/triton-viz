from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

from z3 import And, BoolVal, IntVal, Not, Or, Solver, sat
from z3.z3 import BoolRef, IntNumRef, ModelRef

from .data import AccessEventRecord


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


@dataclass(frozen=True)
class RaceReport:
    first: ScalarMemoryEvent
    second: ScalarMemoryEvent
    model: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    @property
    def first_record(self) -> AccessEventRecord:
        return self.first.record

    @property
    def second_record(self) -> AccessEventRecord:
        return self.second.record


class HBSolver:
    """Small event-graph happens-before solver for the PR-A demo.

    This solver is intentionally isolated from the current Triton race-detector
    capture path. It consumes synthetic ``AccessEventRecord`` inputs and checks
    whether conflicting accesses can remain unordered after program-order and
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
        self.hb = self._build_hb()

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

            for lane, addr in enumerate(addrs):
                active_terms = [
                    self._as_bool(self._lane_value(record.active, lane)),
                    *(
                        self._as_bool(constraint)
                        for constraint in self._iter_constraints(
                            record.local_constraints
                        )
                    ),
                ]
                active = And(*active_terms)

                raw_reads = self._lane_value(record.reads, lane)
                if raw_reads is None:
                    reads = active if record.access_mode == "read" else BoolVal(False)
                else:
                    reads = And(active, self._as_bool(raw_reads))

                raw_writes = self._lane_value(record.writes, lane)
                if raw_writes is None:
                    writes = active if record.access_mode == "write" else BoolVal(False)
                else:
                    writes = And(active, self._as_bool(raw_writes))

                name = record.debug_name or f"e{len(events)}"
                if len(addrs) > 1:
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
                            self._lane_value(record.old_value, lane)
                        ),
                        written_value=self._as_z3_value(
                            self._lane_value(record.written_value, lane)
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

    @staticmethod
    def _lane_value(value: Any, lane: int) -> Any:
        if isinstance(value, (list, tuple)):
            return value[lane]
        return value

    def _same_addr(
        self, first: ScalarMemoryEvent, second: ScalarMemoryEvent
    ) -> BoolRef:
        return self._as_z3_value(first.addr) == self._as_z3_value(second.addr)

    @staticmethod
    def _is_release(event: ScalarMemoryEvent) -> bool:
        return event.sem in ("release", "acq_rel")

    @staticmethod
    def _is_acquire(event: ScalarMemoryEvent) -> bool:
        return event.sem in ("acquire", "acq_rel")

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

    def _minimal_atomic_read_from(
        self,
        writer: ScalarMemoryEvent,
        reader: ScalarMemoryEvent,
    ) -> BoolRef:
        """PR-A synthetic relation only.

        This is intentionally *not* a full coherence/read-from model. It does
        not prove unique writers, coherence order, same-value disambiguation,
        ABA exclusion, or must-alias properties. It only models the minimal
        event fact needed for the solver-only CAS demo:

            writer.written_value == reader.old_value
        """

        if writer.written_value is None or reader.old_value is None:
            return BoolVal(False)

        return And(
            BoolVal(writer.is_atomic),
            BoolVal(reader.is_atomic),
            writer.writes,
            reader.reads,
            self._same_addr(writer, reader),
            writer.written_value == reader.old_value,
        )

    def _synchronizes_with(
        self,
        writer: ScalarMemoryEvent,
        reader: ScalarMemoryEvent,
    ) -> BoolRef:
        return And(
            BoolVal(self._is_release(writer)),
            BoolVal(self._is_acquire(reader)),
            self._scope_ok(writer, reader),
            self._minimal_atomic_read_from(writer, reader),
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
            self._minimal_atomic_read_from(writer, reader)
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

    def _build_hb(self) -> list[list[BoolRef]]:
        n_events = len(self.events)
        reach: list[list[BoolRef]] = [
            [self._edge(self.events[i], self.events[j]) for j in range(n_events)]
            for i in range(n_events)
        ]

        for k in range(n_events):
            reach = [
                [
                    Or(reach[i][j], And(reach[i][k], reach[k][j]))
                    for j in range(n_events)
                ]
                for i in range(n_events)
            ]

        return reach

    def _conflict(self, first: ScalarMemoryEvent, second: ScalarMemoryEvent) -> BoolRef:
        at_least_one_non_atomic = BoolVal(
            (not first.is_atomic) or (not second.is_atomic)
        )

        return And(
            first.active,
            second.active,
            self._same_addr(first, second),
            Or(
                And(first.writes, Or(second.reads, second.writes)),
                And(second.writes, Or(first.reads, first.writes)),
            ),
            at_least_one_non_atomic,
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

        for constraint in self._iter_constraints(self.extra_assumptions):
            solver.add(self._as_bool(constraint))

        for record in self.records:
            for constraint in self._iter_constraints(record.premises):
                solver.add(self._as_bool(constraint))

        for event in self.events:
            solver.add(self._atomic_old_value_has_source(event))

        return solver

    @classmethod
    def _iter_constraints(cls, value: Any) -> Iterable[Any]:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._iter_constraints(item)
            return
        yield value

    @staticmethod
    def _as_bool(value: Any) -> BoolRef:
        if isinstance(value, bool):
            return BoolVal(value)
        return value

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


__all__ = [
    "HBSolver",
    "RaceCheckResult",
    "RaceReport",
    "ScalarMemoryEvent",
]
