"""Two-copy symbolic HB solver.

Production race-finder for ``SymbolicRaceDetector``. The solver duplicates each
recorded access into two symbolic program-instances ``A`` and ``B`` (alpha-
renaming PIDs, ``tl.arange`` lane vars, and per-record copy-local vars such as
the CAS return), then asks Z3 whether any pair of cross-copy events is
unordered, in conflict, and aliasing.

Model boundary (closed-world atomic source assumption):
  When an initial scalar source is identifiable, source choices are closed
  over: (initial source) + (modeled CAS writers in the two selected copies).
  When the initial source is not identifiable, ``rf_unknown_R`` is introduced
  but does NOT enable a synchronizes-with edge. Synchronization through a
  third program instance is therefore not modeled. The guarded acquire/release
  CAS no-race result depends on this closed-world assumption.

Address-domain invariant:
  ``record.addr_expr`` consumed by this solver MUST be a byte address matching
  ``tensor.data_ptr()``. ``byte_overlap`` and ``initial_atomic_source`` rely on
  this. Capture-side normalisation must convert element / tensor-relative
  offsets to byte addresses BEFORE the records reach the solver.

Limitations (current):
  - **Initial atomic source is identifiable only for scalar tensors** —
    ``_initial_atomic_source`` requires ``tensor.numel() == 1``. Multi-element
    flag arrays and pointer-arithmetic flag indexing fall through to
    ``rf_unknown_R``, which deliberately does NOT enable synchronizes-with;
    guarded acq/rel CAS over flag arrays will be reported as races
    conservatively. See ``test_flag_array_cas_acq_rel_guarded_is_not_racy_xfail``
    for the documented gap.
  - **Two program instances only** — synchronization that travels through a
    third block (writer-via-third-block CAS chains) is not modeled directly.
  - **AtomicRMW value semantics not modeled** — the RMW return is wrapped in
    a sentinel that triggers ``UnsupportedSymbolicRaceQuery`` if used
    downstream (e.g. ``mask = old == 0``). The launch is marked unsupported
    via ``SymbolicRaceDetector._mark_unsupported`` rather than racing.
  - **Atomic CAS/RMW inside loops are unsupported** — they are eagerly
    captured today (no integration with the loop-pending path), so the
    handlers mark the launch unsupported instead of recording phantom
    events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from z3 import (
    And,
    AtMost,
    Bool,
    BoolVal,
    Const,
    If,
    Implies,
    Int,
    IntVal,
    Not,
    Or,
    Solver,
    is_true,
    sat,
)
from z3.z3 import BoolRef, ModelRef

from .data import AccessEventRecord, RaceReport, RaceType
from .hb_common import (
    UnsupportedSymbolicRaceQuery,
    apply_sub,
    as_bool,
    build_transitive_hb,
    conflicting_access_modes,
    is_acquire_sem,
    is_release_sem,
    iter_constraints,
    lane_value,
    minimal_atomic_read_from,
    normalize_copy_local_vars,
    to_lanes,
)


@dataclass(frozen=True)
class CopyContext:
    label: str  # "a" or "b"
    pid: tuple[Any, Any, Any]
    pid_substitutions: tuple[tuple[Any, Any], ...]
    arange_substitutions: tuple[tuple[Any, Any], ...]
    arange_constraints: tuple[Any, ...]
    copy_local_substitutions: tuple[tuple[Any, Any], ...]  # launch-level


@dataclass(frozen=True)
class SymbolicMemoryEvent:
    idx: int
    copy: str
    record: AccessEventRecord
    name: str
    lane: int
    event_id: int
    program_seq: int
    pid: tuple[Any, Any, Any]
    addr: Any
    elem_size: int
    active: BoolRef
    reads: BoolRef
    writes: BoolRef
    is_atomic: bool
    atomic_kind: str
    sem: str
    scope: str | None
    old_value: Any = None
    written_value: Any = None


def _import_symbolic_expr_pids():
    # Local import: SymbolicExpr is a heavy module; keep tests cheap.
    from ..symbolic_engine import SymbolicExpr

    return (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)


class TwoCopySymbolicHBSolver:
    """Two-copy symbolic happens-before solver.

    See the module docstring for the model boundary and address-domain
    invariants.
    """

    def __init__(
        self,
        records: list[AccessEventRecord],
        *,
        grid: tuple[int, ...],
        arange_dict: dict[Any, Any] | None = None,
        extra_assumptions: tuple[Any, ...] = (),
    ) -> None:
        self.records = list(records)
        self.grid = self._normalize_grid(grid)
        self.arange_dict = dict(arange_dict or {})
        self.extra_assumptions = tuple(extra_assumptions)

        # 1. PID vars + substitutions for both copies.
        pid_a, pid_b = self._make_pid_vars()
        pid_subs_a, pid_subs_b = self._make_pid_subs(pid_a, pid_b)

        # 2. Grid bounds + different-block constraints.
        (
            self.grid_constraints,
            self.different_blocks,
        ) = self._make_grid_and_diff_block_constraints(pid_a, pid_b)

        # 3. Arange substitutions + range constraints from the snapshot.
        (
            arange_subs_a,
            arange_subs_b,
            arange_consts_a,
            arange_consts_b,
        ) = self._make_arange_subs_and_constraints()
        self.arange_constraints_a = tuple(arange_consts_a)
        self.arange_constraints_b = tuple(arange_consts_b)

        # 4. LAUNCH-LEVEL copy-local substitutions (union over all records).
        copy_local_subs_a, copy_local_subs_b = self._make_launch_copy_local_subs()

        # 5. Build the two CopyContexts (frozen).
        self.ctx_a = CopyContext(
            label="a",
            pid=tuple(pid_a),
            pid_substitutions=tuple(pid_subs_a),
            arange_substitutions=tuple(arange_subs_a),
            arange_constraints=self.arange_constraints_a,
            copy_local_substitutions=tuple(copy_local_subs_a),
        )
        self.ctx_b = CopyContext(
            label="b",
            pid=tuple(pid_b),
            pid_substitutions=tuple(pid_subs_b),
            arange_substitutions=tuple(arange_subs_b),
            arange_constraints=self.arange_constraints_b,
            copy_local_substitutions=tuple(copy_local_subs_b),
        )

        # 6. Lower every record under both contexts.
        self.events: list[SymbolicMemoryEvent] = self._lower_two_copies()

        # 7. Allocate RF source booleans BEFORE building HB closure.
        self.rf_source: dict[tuple[int, int], BoolRef] = {}
        self.rf_constraints: list[BoolRef] = []
        self._build_read_from_choices()

        # 8. Build HB transitive closure (synchronizes_with reads rf_source).
        self.hb = build_transitive_hb(self.events, self._edge)

    # ──────────────────────── Public API ────────────────────────

    def find_races(self) -> list[RaceReport]:
        events_a = [e for e in self.events if e.copy == "a"]
        events_b = [e for e in self.events if e.copy == "b"]

        candidates: list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef]]
        candidates = []
        for a in events_a:
            for b in events_b:
                solver = self._new_solver()
                solver.add(self._race_expr(a, b))
                if solver.check() == sat:
                    candidates.append((a, b, solver.model()))

        return self._dedupe_reports(candidates)

    # ──────────────────────── Construction ────────────────────────

    @staticmethod
    def _normalize_grid(grid: tuple[int, ...]) -> tuple[int, int, int]:
        dims = [int(d) for d in grid]
        while len(dims) < 3:
            dims.append(1)
        return (dims[0], dims[1], dims[2])

    @staticmethod
    def _make_pid_vars():
        pid_a = [Int(f"pid_a_{i}") for i in range(3)]
        pid_b = [Int(f"pid_b_{i}") for i in range(3)]
        return pid_a, pid_b

    @staticmethod
    def _make_pid_subs(pid_a, pid_b):
        orig = _import_symbolic_expr_pids()
        sub_a = tuple((orig[i], pid_a[i]) for i in range(3))
        sub_b = tuple((orig[i], pid_b[i]) for i in range(3))
        return sub_a, sub_b

    def _make_grid_and_diff_block_constraints(self, pid_a, pid_b):
        grid_constraints = And(
            *[And(pid_a[i] >= 0, pid_a[i] < self.grid[i]) for i in range(3)],
            *[And(pid_b[i] >= 0, pid_b[i] < self.grid[i]) for i in range(3)],
        )
        different_blocks = Or(
            pid_a[0] != pid_b[0],
            pid_a[1] != pid_b[1],
            pid_a[2] != pid_b[2],
        )
        return grid_constraints, different_blocks

    def _make_arange_subs_and_constraints(self):
        sub_a, sub_b = [], []
        cons_a, cons_b = [], []
        for key, value in self.arange_dict.items():
            # ARANGE_DICT entry shape: key=(start, end), value=(orig_var, _).
            try:
                start, end = key
                orig_var = value[0] if isinstance(value, (list, tuple)) else value
            except Exception:
                continue
            var_a = Int(f"arange_a_{start}_{end}")
            var_b = Int(f"arange_b_{start}_{end}")
            sub_a.append((orig_var, var_a))
            sub_b.append((orig_var, var_b))
            cons_a.append(And(var_a >= start, var_a < end))
            cons_b.append(And(var_b >= start, var_b < end))
        return sub_a, sub_b, cons_a, cons_b

    def _make_launch_copy_local_subs(self):
        all_vars = normalize_copy_local_vars(
            v for r in self.records for v in r.copy_local_vars
        )
        subs_a, subs_b = [], []
        for i, v in enumerate(all_vars):
            base = f"{v.decl().name()}__{i}__{v.hash()}"
            subs_a.append((v, Const(f"{base}__a", v.sort())))
            subs_b.append((v, Const(f"{base}__b", v.sort())))
        return tuple(subs_a), tuple(subs_b)

    # ──────────────────────── Lowering ────────────────────────

    def _lower_two_copies(self) -> list[SymbolicMemoryEvent]:
        events: list[SymbolicMemoryEvent] = []
        for ctx in (self.ctx_a, self.ctx_b):
            for record in self.records:
                events.extend(self._lower_record(record, ctx, len(events)))
        return events

    def _lower_record(
        self,
        record: AccessEventRecord,
        ctx: CopyContext,
        start_idx: int,
    ) -> list[SymbolicMemoryEvent]:
        if record.addr_expr is None:
            raise UnsupportedSymbolicRaceQuery(
                "AccessEventRecord.addr_expr is required for two-copy lowering"
            )

        sub = (
            ctx.pid_substitutions
            + ctx.arange_substitutions
            + ctx.copy_local_substitutions
        )

        addr_all = apply_sub(record.addr_expr, sub)
        active_all = apply_sub(record.active, sub)
        local_all = apply_sub(record.local_constraints, sub)
        prem_all = apply_sub(record.premises, sub)

        addr_lanes = to_lanes(addr_all)
        n_lanes = len(addr_lanes) or 1

        # Per-record CAS substitutions are needed in cas_cmp/new/old.
        cas_old_all: Any = None
        cas_cmp_all: Any = None
        cas_new_all: Any = None
        if record.atomic_kind == "cas":
            cas_old_all = apply_sub(record.old_value, sub)
            cas_cmp_all = apply_sub(record.cas_cmp_value, sub)
            cas_new_all = apply_sub(record.cas_new_value, sub)

        # local_constraints / premises are FLAT lists of globally applicable
        # constraints (mask conditions, address-validity, loop iterators); they
        # are NOT lane-indexed. Per-lane variation lives in record.active /
        # record.reads / record.writes.
        local_terms = tuple(as_bool(c) for c in iter_constraints(local_all))
        prem_terms = tuple(as_bool(c) for c in iter_constraints(prem_all))

        out: list[SymbolicMemoryEvent] = []
        for lane, addr in enumerate(addr_lanes):
            active = And(
                as_bool(lane_value(active_all, lane, n_lanes)),
                *local_terms,
                *prem_terms,
            )

            if record.atomic_kind == "cas":
                old = lane_value(cas_old_all, lane, n_lanes)
                cmp_ = lane_value(cas_cmp_all, lane, n_lanes)
                new = lane_value(cas_new_all, lane, n_lanes)
                if old is None or cmp_ is None or new is None:
                    raise UnsupportedSymbolicRaceQuery(
                        "CAS record missing old_value / cas_cmp_value / cas_new_value"
                    )
                success = old == cmp_
                reads = active
                writes = And(active, success)
                written = If(success, new, old)
                old_value: Any = old
                written_value: Any = written
            elif record.is_atomic:
                # AtomicRMW: always reads and always writes when active.
                reads = active
                writes = active
                old_value = None
                written_value = None
            else:
                if record.reads is None:
                    read_cond: Any = record.access_mode == "read"
                else:
                    read_cond = lane_value(apply_sub(record.reads, sub), lane, n_lanes)
                if record.writes is None:
                    write_cond: Any = record.access_mode == "write"
                else:
                    write_cond = lane_value(
                        apply_sub(record.writes, sub), lane, n_lanes
                    )
                reads = And(active, as_bool(read_cond))
                writes = And(active, as_bool(write_cond))
                old_value = None
                written_value = None

            name = record.debug_name or f"e{record.event_id}"
            if n_lanes > 1:
                name = f"{name}.lane{lane}"
            name = f"{name}.{ctx.label}"

            out.append(
                SymbolicMemoryEvent(
                    idx=start_idx + len(out),
                    copy=ctx.label,
                    record=record,
                    name=name,
                    lane=lane,
                    event_id=record.event_id,
                    program_seq=record.program_seq,
                    pid=ctx.pid,
                    addr=addr,
                    elem_size=max(1, int(record.elem_size)),
                    active=active,
                    reads=reads,
                    writes=writes,
                    is_atomic=record.is_atomic,
                    atomic_kind=record.atomic_kind,
                    sem=record.sem,
                    scope=record.scope,
                    old_value=old_value,
                    written_value=written_value,
                )
            )
        return out

    # ──────────────────────── Edges & HB closure ────────────────────────

    @staticmethod
    def _program_order(e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent) -> BoolRef:
        if e1.copy != e2.copy:
            return BoolVal(False)
        if e1.program_seq < 0 or e2.program_seq < 0:
            return BoolVal(False)
        if e1.program_seq >= e2.program_seq:
            return BoolVal(False)
        return And(e1.active, e2.active)  # active-gated

    @staticmethod
    def _exact_atomic_addr(w: SymbolicMemoryEvent, r: SymbolicMemoryEvent) -> BoolRef:
        if w.atomic_kind != "cas" or r.atomic_kind != "cas":
            return BoolVal(False)
        if w.elem_size != r.elem_size:
            return BoolVal(False)
        return w.addr == r.addr

    @staticmethod
    def _scope_ok(w: SymbolicMemoryEvent, r: SymbolicMemoryEvent) -> BoolRef:
        if w.scope == "cta" or r.scope == "cta":
            return And(
                w.pid[0] == r.pid[0],
                w.pid[1] == r.pid[1],
                w.pid[2] == r.pid[2],
            )
        return BoolVal(True)

    def _synchronizes_with(
        self, w: SymbolicMemoryEvent, r: SymbolicMemoryEvent
    ) -> BoolRef:
        rf = self.rf_source.get((w.idx, r.idx))
        if rf is None:
            return BoolVal(False)
        return And(
            BoolVal(is_release_sem(w.sem)),
            BoolVal(is_acquire_sem(r.sem)),
            self._scope_ok(w, r),
            rf,
        )

    def _edge(self, e1: SymbolicMemoryEvent, e2: SymbolicMemoryEvent) -> BoolRef:
        if e1.idx == e2.idx:
            return BoolVal(False)
        return Or(
            self._program_order(e1, e2),
            self._synchronizes_with(e1, e2),
        )

    # ──────────────────────── Read-from / RF source choices ────────────────

    def _can_be_rf_candidate(
        self, w: SymbolicMemoryEvent, r: SymbolicMemoryEvent
    ) -> bool:
        if w.idx == r.idx:
            return False
        # Same program instance cannot read from a future write.
        if w.copy == r.copy and w.program_seq >= r.program_seq:
            return False
        return True

    @staticmethod
    def _initial_atomic_source(r: SymbolicMemoryEvent) -> Any:
        t = r.record.tensor
        if t is None or r.old_value is None:
            return None
        try:
            if t.numel() != 1:
                return None
            ptr = int(t.data_ptr())
            item = t.detach().cpu().item() if hasattr(t, "detach") else t.item()
            init = int(item)
        except Exception:
            return None
        return And(r.addr == IntVal(ptr), r.old_value == IntVal(init))

    def _build_read_from_choices(self) -> None:
        # Closed-world atomic source model.
        # If the initial scalar source is identifiable, source choices are
        # closed over: (initial source) + (modeled CAS writers). If the
        # initial source is not identifiable, rf_unknown is introduced and
        # does NOT enable synchronizes-with. This is intentionally NOT a full
        # coherence/read-from model over all program instances; the guarded
        # acq_rel CAS no-race result depends on this closed-world assumption.
        cas_writers = [e for e in self.events if e.atomic_kind == "cas"]
        for r in self.events:
            if r.atomic_kind != "cas":
                continue
            choices: list[BoolRef] = []
            init_pred = self._initial_atomic_source(r)

            if init_pred is not None:
                rf_init = Bool(f"rf_init_{r.idx}")
                choices.append(rf_init)
                self.rf_constraints.append(Implies(rf_init, And(r.reads, init_pred)))
            else:
                rf_unknown = Bool(f"rf_unknown_{r.idx}")
                choices.append(rf_unknown)
                self.rf_constraints.append(Implies(rf_unknown, r.reads))

            for w in cas_writers:
                if not self._can_be_rf_candidate(w, r):
                    continue
                rf = Bool(f"rf_{w.idx}_to_{r.idx}")
                choices.append(rf)
                self.rf_source[(w.idx, r.idx)] = rf
                self.rf_constraints.append(
                    Implies(
                        rf,
                        minimal_atomic_read_from(
                            w, r, same_atomic_addr_fn=self._exact_atomic_addr
                        ),
                    )
                )

            if choices:
                self.rf_constraints.append(Implies(r.reads, Or(*choices)))
                self.rf_constraints.append(
                    Implies(Not(r.reads), And(*(Not(c) for c in choices)))
                )
                if len(choices) > 1:
                    self.rf_constraints.append(AtMost(*choices, 1))

    # ──────────────────────── Conflict / race query ────────────────────────

    @staticmethod
    def _byte_overlap(a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> BoolRef:
        if a.elem_size == 1 and b.elem_size == 1:
            return a.addr == b.addr
        return And(
            a.addr < b.addr + b.elem_size,
            b.addr < a.addr + a.elem_size,
        )

    def _conflict(self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> BoolRef:
        return And(
            a.active,
            b.active,
            self._byte_overlap(a, b),
            conflicting_access_modes(a, b),
        )

    def _race_expr(self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent) -> BoolRef:
        return And(
            self._conflict(a, b),
            Not(self.hb[a.idx][b.idx]),
            Not(self.hb[b.idx][a.idx]),
        )

    def _new_solver(self) -> Solver:
        solver = Solver()
        solver.add(self.grid_constraints)
        solver.add(self.different_blocks)
        for c in self.arange_constraints_a:
            solver.add(c)
        for c in self.arange_constraints_b:
            solver.add(c)
        for c in self.rf_constraints:
            solver.add(c)
        for c in self.extra_assumptions:
            solver.add(as_bool(c))
        return solver

    # ──────────────────────── Reports ────────────────────────

    @staticmethod
    def _canonical_pair(
        a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> tuple[SymbolicMemoryEvent, SymbolicMemoryEvent]:
        if (a.event_id, a.lane) <= (b.event_id, b.lane):
            return a, b
        return b, a

    def _dedupe_reports(
        self,
        candidates: list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef]],
    ) -> list[RaceReport]:
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        reports: list[RaceReport] = []
        for a, b, model in candidates:
            first, second = self._canonical_pair(a, b)
            key = (
                (first.event_id, first.lane),
                (second.event_id, second.lane),
            )
            if key in seen:
                continue
            seen.add(key)
            reports.append(self._make_report(first, second, model))
        return reports

    def _make_report(
        self,
        first: SymbolicMemoryEvent,
        second: SymbolicMemoryEvent,
        model: ModelRef,
    ) -> RaceReport:
        fw = bool(is_true(model.evaluate(first.writes, model_completion=True)))
        sw = bool(is_true(model.evaluate(second.writes, model_completion=True)))
        if fw and sw:
            race_type: RaceType = RaceType.WAW
        elif fw:
            race_type = RaceType.RAW
        else:
            race_type = RaceType.WAR

        addr_a_val = model.evaluate(first.addr, model_completion=True).as_long()
        addr_b_val = model.evaluate(second.addr, model_completion=True).as_long()
        if first.elem_size > 1 or second.elem_size > 1:
            witness_addr = max(addr_a_val, addr_b_val)
        else:
            witness_addr = addr_a_val

        witness_grid_a = tuple(
            model.evaluate(first.pid[i], model_completion=True).as_long()
            for i in range(3)
        )
        witness_grid_b = tuple(
            model.evaluate(second.pid[i], model_completion=True).as_long()
            for i in range(3)
        )

        assert race_type is not None, "race_type_value must always be populated"
        return RaceReport(
            first=first,
            second=second,
            model=self._model_to_dict(model),
            reason=(
                "unordered conflicting memory accesses across two symbolic "
                "program instances under the current symbolic assumptions"
            ),
            race_type_value=race_type,
            witness_addr=int(witness_addr),
            witness_grid_a=witness_grid_a,
            witness_grid_b=witness_grid_b,
        )

    @staticmethod
    def _model_to_dict(model: ModelRef) -> dict[str, str]:
        return {decl.name(): str(model[decl]) for decl in model.decls()}


__all__ = [
    "CopyContext",
    "SymbolicMemoryEvent",
    "TwoCopySymbolicHBSolver",
]
