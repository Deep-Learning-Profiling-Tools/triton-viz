"""Two-copy symbolic HB solver.

Production race-finder for ``SymbolicRaceDetector``. The solver duplicates each
recorded access into two symbolic program-instances ``A`` and ``B`` (alpha-
renaming PIDs, ``tl.arange`` lane vars, and per-record copy-local vars such as
the CAS return), then asks Z3 whether any pair of cross-copy events is
unordered, in conflict, and aliasing.

Model boundary (closed-world atomic source assumption):
  When an initial scalar source is identifiable AND no unmodeled write (plain
  store or atomic RMW) can overlap the location, source choices are closed
  over: (initial source) + (modeled CAS writers in the two selected copies).
  Otherwise ``rf_unknown_R`` is introduced but does NOT enable a
  synchronizes-with edge — an overlapping plain-store/RMW can publish a value
  outside the closed world (e.g. a flag set via ``tl.atomic_xchg``), so the
  reader's old value must not be over-constrained or every conflict gated on
  it silently disappears. Synchronization through a third program instance is
  not modeled. The guarded acquire/release CAS no-race result depends on the
  closed world, which holds whenever the flag is only ever written by modeled
  CAS.

Address-domain invariant:
  ``record.addr_expr`` consumed by this solver MUST be a byte address matching
  ``tensor.data_ptr()``. ``byte_overlap`` and ``initial_atomic_source`` rely on
  this. Capture-side normalisation must convert element / tensor-relative
  offsets to byte addresses BEFORE the records reach the solver.

Intra-instance duplicate lanes:
  Cross-copy queries assert ``different_blocks``, so they can never witness
  two lanes of one store colliding inside a single program instance — and a
  grid=(1,1,1) launch makes ``different_blocks`` UNSAT outright. A separate
  same-instance query pins ``pid_a == pid_b`` and every launch-level
  copy-local var equal, leaving the arange lane vars as the only
  alpha-difference between the copies; requiring a lane-identity difference
  then asks whether two DISTINCT lanes of the same dynamic access conflict.

Z3 ``unknown`` policy:
  ``unknown`` is never treated as unsat. A race query that comes back
  undecided raises :class:`UnsupportedSymbolicRaceQuery` (the launch reports
  ``unsupported`` instead of a silent clean verdict), and an undecided
  overlap in the closed-world escape check opens the ``rf_unknown`` escape.

RMW value modeling (spec part B):
  An integer AtomicRMW with a modeled op (``add``/``max``/``min``/``xchg``)
  gets a fresh observation symbol ``o_r`` (``record.old_value``, alpha-
  renamed per copy exactly like the CAS return) and a modeled write part
  ``f_op(o_r, v)`` (``written_value``). ``o_r`` is justified like a CAS
  observation: rf candidates are the value-modeled atomic writers, the
  initial-value source, and the unknown source when an unmodeled writer can
  overlap. Value-modeled atomics join the per-location atomic order, which
  makes RMW **atomicity/immediacy** structural: one order position covers
  the read AND the write part, and the existing "no successful same-address
  writer strictly between rf source and reader" constraint is exactly the
  immediacy axiom. Two extensions build on it:

  - **Reads-through (release sequences over RMW chains)**: sw(w, r) also
    holds when r reads-through w via a bounded chain of modeled RMW write
    parts (each link an rf edge). Chains through UNMODELED grid instances
    are covered only by the counting axiom below.
  - **Counting axiom** (guarded): for a location L touched by EXACTLY ONE
    always-active scalar constant-increment ``add`` RMW record with a known
    initial value and no other possible writer, per-copy rank variables
    r ∈ [0, |G|) satisfy ``o = init + c·rank``, rank equality iff same
    block, and ``co ⇔ rank<``. Under those guards every write to L between
    two instances is a link of the same RMW chain, so reads-through is
    equivalent to coherence order — giving last-block-done and work-queue
    disjointness. A counted reader also gets an ``rf_chain`` source choice
    (an unmodeled instance at rank-1) so real races at non-adjacent ranks
    are not squeezed out of the closed world; ``rf_chain`` itself never
    enables synchronizes-with. If ANY guard fails the axiom is omitted
    entirely (over-report direction), and an observation variable used in
    an ADDRESS position without counting support raises
    ``UnsupportedSymbolicRaceQuery`` instead of widening silently.

Limitations (current):
  - **Initial atomic source covers scalar tensors and small contiguous flag
    arrays** (``numel <= _MAX_INITIAL_ATOMIC_ELEMENTS = 1024``). Larger or
    non-contiguous tensors fall through to ``rf_unknown_R``, which
    deliberately does NOT enable synchronizes-with; guarded acq/rel CAS over
    them is reported as races conservatively.
  - **Two program instances only** — synchronization that travels through a
    third block (writer-via-third-block CAS chains) is not modeled directly;
    the counting axiom is the one guarded exception.
  - **Float-typed RMW returns and bitwise/unsigned RMW ops are not value-
    modeled** (``fadd``, float max, ``and``/``or``/``xor``, ``umax``/
    ``umin``): the capture keeps the downstream-use sentinel / DataDep
    fallback, and the record's write stays in the unmodeled-writer set.
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
    ExprRef,
    If,
    Implies,
    Int,
    IntVal,
    Not,
    Or,
    Solver,
    is_true,
    sat,
    simplify,
    unsat,
)
from z3.z3 import BoolRef, IntNumRef, ModelRef

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
    modeled_atomic_read_from,
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
    # The substituted RMW operand v (None for non-RMW / unmodeled RMW);
    # the counting axiom reads the constant increment off it.
    rmw_operand: Any = None


def _import_symbolic_expr_pids():
    # Local import: SymbolicExpr is a heavy module; keep tests cheap.
    from ..symbolic_engine import SymbolicExpr

    return (SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2)


def _is_symbolic_dim(d: Any) -> bool:
    """A grid dim that is a Z3 expression rather than a Python int.

    Must be an isinstance check: duck-typing on ``sort`` misfires on numpy
    scalars (ndarray.sort) and would leave them un-coerced where the old
    ``int(d)`` handled them."""
    return isinstance(d, ExprRef)


@dataclass(frozen=True)
class _CountingInfo:
    """The counting axiom (spec B.1.5) fired for one RMW record: per-copy
    rank variables tied to the observation values and the atomic order."""

    idx_a: int
    idx_b: int
    rank_a: Any
    rank_b: Any
    init: int
    inc: int
    loc: int


def _rmw_written_value(op: str | None, old: Any, v: Any) -> Any:
    """The modeled write part f_op(old, v) of an RMW, or ``None`` when the
    op has no Int-sort model: bitwise and/or/xor need bitvectors, unsigned
    umax/umin diverge from the signed Int order, and float ops are outside
    the integer model. ``None`` keeps the record's write in the
    unmodeled-writer set (rf_unknown escape) — the over-report direction."""
    if op is None or old is None or v is None:
        return None
    if op == "add":
        return old + v
    if op == "max":
        return If(old >= v, old, v)
    if op == "min":
        return If(old <= v, old, v)
    if op == "xchg":
        return v
    return None


def _as_numeral(v: Any) -> int | None:
    """``v`` as a concrete int if it simplifies to one, else None."""
    if v is None:
        return None
    if isinstance(v, (bool, int)):
        return int(v)
    try:
        s = simplify(v)
    except Exception:
        return None
    if isinstance(s, IntNumRef):
        return s.as_long()
    return None


def _z3_var_key(v: Any) -> tuple[int, str, str]:
    # Mirrors the dedup key used by hb_common.normalize_copy_local_vars.
    return (v.hash(), str(v.sort()), v.decl().name())


def _collect_z3_var_keys(values: tuple[Any, ...]) -> set[tuple[int, str, str]]:
    """Keys of every 0-ary leaf in ``values``. Numeral leaves are included
    but can never collide with a variable's key."""
    seen: set[tuple[int, str, str]] = set()
    stack: list[Any] = list(values)
    while stack:
        v = stack.pop()
        if v is None or isinstance(v, (bool, int, float, str)):
            continue
        if isinstance(v, (list, tuple)):
            stack.extend(v)
            continue
        if not hasattr(v, "num_args"):
            continue
        if v.num_args() == 0:
            try:
                seen.add(_z3_var_key(v))
            except Exception:
                pass
            continue
        stack.extend(v.children())
    return seen


class TwoCopySymbolicHBSolver:
    """Two-copy symbolic happens-before solver.

    See the module docstring for the model boundary and address-domain
    invariants.
    """

    # Ablation switches recognized by the RQ5 study (evaluation/ablation.py).
    # Both default OFF; production semantics are the empty set.
    #   "hb"        — assert NO happens-before at all (skip the transitive
    #                 closure): isolates how much of the verdict quality is
    #                 carried by ordering edges.
    #   "coherence" — drop the per-location atomic order (coherence AND the
    #                 counting axiom, which is a coherence-order axiom):
    #                 isolates the immediacy/single-winner machinery.
    ABLATIONS: tuple[str, ...] = ("hb", "coherence")

    def __init__(
        self,
        records: list[AccessEventRecord],
        *,
        grid: tuple[Any, ...],
        arange_dict: dict[Any, Any] | None = None,
        extra_assumptions: tuple[Any, ...] = (),
        ablations: tuple[str, ...] = (),
    ) -> None:
        self.records = list(records)
        self.grid = self._normalize_grid(grid)
        self.arange_dict = dict(arange_dict or {})
        self.extra_assumptions = tuple(extra_assumptions)
        unknown = set(ablations) - set(self.ABLATIONS)
        if unknown:
            raise ValueError(f"unknown ablations: {sorted(unknown)}")
        self.ablations = frozenset(ablations)

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

        # 7. Atomic-order vars, counting axioms, RF source booleans, and the
        # reads-through relation, BEFORE building the HB closure (HB reads
        # self.reads_through for synchronizes_with). Order matters: the
        # counting axiom's applicability is consulted while building rf
        # choices (it adds the rf_chain source), and the reads-through
        # closure folds in both the direct rf edges and the counting pairs.
        self.atomic_order: dict[int, Any] = self._make_atomic_order_vars()
        self.rf_source: dict[tuple[int, int], BoolRef] = {}
        self.rf_init_source: dict[int, BoolRef] = {}
        self.rf_unknown_source: dict[int, BoolRef] = {}
        self.rf_chain_source: dict[int, BoolRef] = {}
        self.rf_constraints: list[BoolRef] = []
        self.atomic_coherence_constraints: list[BoolRef] = []
        self.counting_constraints: list[BoolRef] = []
        # RQ5 ablation "coherence": no per-location atomic order — the
        # counting axiom (a coherence-order axiom) is omitted with it.
        self._counting: dict[int, _CountingInfo] = (
            {} if "coherence" in self.ablations else self._build_counting_axioms()
        )
        self._build_read_from_choices()
        if "coherence" not in self.ablations:
            self._build_atomic_coherence_constraints()
        self._assert_no_uncounted_observation_addresses()
        self.reads_through: dict[tuple[int, int], BoolRef] = self._build_reads_through()

        # 8. Build HB transitive closure (synchronizes_with reads
        # reads_through). RQ5 ablation "hb": no ordering edges exist at all
        # — every conflicting aliasing pair becomes a report.
        n_events = len(self.events)
        self.hb = (
            [[BoolVal(False) for _ in range(n_events)] for _ in range(n_events)]
            if "hb" in self.ablations
            else build_transitive_hb(self.events, self._edge)
        )

    # ──────────────────────── Public API ────────────────────────

    _CROSS_INSTANCE_REASON: str = (
        "unordered conflicting memory accesses across two symbolic "
        "program instances under the current symbolic assumptions"
    )
    _INTRA_INSTANCE_REASON: str = (
        "conflicting lanes of a single program instance touch the same "
        "bytes with no defined intra-instance order"
    )

    def find_races(self) -> list[RaceReport]:
        import time as _time

        events_a = [e for e in self.events if e.copy == "a"]
        events_b = [e for e in self.events if e.copy == "b"]
        # (kind, seconds, sat) per executed query — the RQ3 scaling sweep
        # reads these for per-query mean/median/p95 and the SAT/UNSAT split.
        self.query_stats: list[tuple[str, float, bool]] = []

        candidates: list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]]
        candidates = []
        for a in events_a:
            for b in events_b:
                solver = self._new_solver()
                solver.add(self._race_expr(a, b))
                t0 = _time.perf_counter()
                is_sat = self._race_query_is_sat(solver, a, b)
                self.query_stats.append(("cross", _time.perf_counter() - t0, is_sat))
                if is_sat:
                    candidates.append(
                        (a, b, solver.model(), self._CROSS_INSTANCE_REASON)
                    )

        candidates.extend(self._find_intra_instance_candidates(events_a, events_b))
        return self._dedupe_reports(candidates)

    @staticmethod
    def _race_query_is_sat(
        solver: Solver, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> bool:
        """``solver.check()`` with Z3 ``unknown`` made conservative.

        ``unknown`` (timeout, nonlinear give-up) must not collapse into
        unsat: dropping the pair would turn an undecided query into a
        silent clean "ok" verdict. There is no witness model to report
        either, so escalate to :class:`UnsupportedSymbolicRaceQuery` —
        ``SymbolicRaceDetector.finalize`` then reports the launch as
        unsupported instead of race-free.
        """
        result = solver.check()
        if result == sat:
            return True
        if result == unsat:
            return False
        detail = solver.reason_unknown()
        raise UnsupportedSymbolicRaceQuery(
            f"Z3 could not decide the race query for {a.name} vs {b.name}"
            + (f" ({detail})" if detail else "")
        )

    def _find_intra_instance_candidates(
        self,
        events_a: list[SymbolicMemoryEvent],
        events_b: list[SymbolicMemoryEvent],
    ) -> list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]]:
        """Duplicate-lane conflicts inside a single program instance.

        See the module docstring: cross-copy queries assert
        ``different_blocks`` and therefore cannot witness these (under
        grid=(1,1,1) they are vacuously unsat). Distinct ops within an
        instance are program-ordered and an atomic op's lanes serialize, so
        the intra-instance hazard is duplicate addresses across the lanes
        of a single non-atomic store — plus record pairs the capture left
        genuinely unordered (equal or unset sequence numbers).
        """
        import time as _time

        same_instance = self._same_instance_constraints()
        out: list[tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]]
        out = []
        for a in events_a:
            for b in events_b:
                lane_cond = self._intra_pair_lane_condition(a, b)
                if lane_cond is None:
                    continue
                solver = self._base_solver()
                for c in same_instance:
                    solver.add(c)
                solver.add(lane_cond)
                solver.add(self._race_expr(a, b))
                t0 = _time.perf_counter()
                is_sat = self._race_query_is_sat(solver, a, b)
                if hasattr(self, "query_stats"):
                    self.query_stats.append(
                        ("intra", _time.perf_counter() - t0, is_sat)
                    )
                if is_sat:
                    out.append((a, b, solver.model(), self._INTRA_INSTANCE_REASON))
        return out

    def _intra_pair_lane_condition(
        self, a: SymbolicMemoryEvent, b: SymbolicMemoryEvent
    ) -> BoolRef | None:
        """Lane-identity constraint for an intra-instance pair, or ``None``
        when the pair cannot race within one instance (program-ordered,
        serialized, never writes, or the symmetric duplicate of an
        already-queried pair).
        """
        if a.record is b.record:
            # Lanes of one atomic op serialize against each other; a
            # load's duplicate lanes read-read and cannot conflict.
            if a.record.is_atomic or a.record.access_mode != "write":
                return None
            if a.lane > b.lane:
                return None  # symmetric duplicate
            if a.lane < b.lane:
                return BoolVal(True)  # explicitly distinct lanes
            return self._lane_identity_differs(a)
        # Distinct ops within one instance are program-ordered; only pairs
        # the capture left without an order (equal or unset sequence
        # numbers) can be concurrently in flight.
        if a.event_id > b.event_id:
            return None  # symmetric duplicate
        if a.program_seq >= 0 and b.program_seq >= 0 and a.program_seq != b.program_seq:
            return None
        return BoolVal(True)

    def _lane_identity_differs(self, e: SymbolicMemoryEvent) -> BoolRef | None:
        """Constraint that the a/b copies of ``e`` denote two DIFFERENT
        lanes of its record, or ``None`` for a true scalar access (no
        second lane exists).

        Each arange summary var is injective in the lane index, so any one
        of the record's arange vars differing across the copies witnesses
        two distinct lanes. Vars in the activity condition count too: a
        store whose address ignores the lane still has its lanes
        distinguished by the mask.
        """
        occurring = _collect_z3_var_keys((e.addr, e.active, e.writes))
        diffs = [
            var_a != var_b
            for (_, var_a), (_, var_b) in zip(
                self.ctx_a.arange_substitutions, self.ctx_b.arange_substitutions
            )
            if _z3_var_key(var_a) in occurring or _z3_var_key(var_b) in occurring
        ]
        if not diffs:
            return None
        return diffs[0] if len(diffs) == 1 else Or(*diffs)

    def _same_instance_constraints(self) -> tuple[BoolRef, ...]:
        """Pin the b copy onto the a copy's program instance.

        ``pid_a == pid_b`` makes the two copies the same block. Copy-local
        vars (loop iterators, CAS returns) are pinned equal because within
        one instance the two lane roles share each dynamic op's iteration
        and return value — leaving them free would let an ordered
        cross-iteration pair masquerade as an intra-instance lane conflict.
        """
        cons: list[BoolRef] = [self.ctx_a.pid[i] == self.ctx_b.pid[i] for i in range(3)]
        for (_, var_a), (_, var_b) in zip(
            self.ctx_a.copy_local_substitutions, self.ctx_b.copy_local_substitutions
        ):
            cons.append(var_a == var_b)
        return tuple(cons)

    # ──────────────────────── Construction ────────────────────────

    @staticmethod
    def _normalize_grid(grid: tuple[Any, ...]) -> tuple[Any, Any, Any]:
        """Concrete launches pass ints; the T1 static front-end passes Z3
        Ints so the verdict covers EVERY grid (each symbolic dim gets a
        ``>= 1`` bound in the grid constraints)."""
        dims = [d if _is_symbolic_dim(d) else int(d) for d in grid]
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
            # A symbolic dim needs its own lower bound or a zero/negative
            # grid would make every pid constraint vacuously unsat and turn
            # any query into a false proof.
            *[d >= 1 for d in self.grid if _is_symbolic_dim(d)],
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
            # ARANGE_DICT entry shape: key=(start, end) or
            # (start, end, filename, lineno) — the engine keys interned vars
            # by creation site so independent same-range arange instances stay
            # distinct. value=(orig_var, _). Per-copy names derive from the
            # original var's name so every dict entry renames uniquely.
            try:
                start, end = key[0], key[1]
                orig_var = value[0] if isinstance(value, (list, tuple)) else value
                base_name = orig_var.decl().name()
            except Exception:
                continue
            var_a = Int(f"{base_name}__a")
            var_b = Int(f"{base_name}__b")
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

            rmw_operand: Any = None
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
                if record.old_value is not None:
                    # Value-modeled RMW (spec B.1): the observation symbol
                    # o_r lives in copy_local_vars, so `sub` already alpha-
                    # renamed it per copy; the write part is f_op(o_r, v).
                    old_value = lane_value(
                        apply_sub(record.old_value, sub), lane, n_lanes
                    )
                    if record.rmw_operand is not None:
                        rmw_operand = lane_value(
                            apply_sub(record.rmw_operand, sub), lane, n_lanes
                        )
                    written_value = _rmw_written_value(
                        record.rmw_op, old_value, rmw_operand
                    )
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
                    rmw_operand=rmw_operand,
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
        """Same-location predicate for the VALUE-MODELED atomic machinery
        (rf, coherence order). Membership is presence of an observation
        symbol: CAS always has one, an RMW exactly when its value semantics
        are modeled (spec part B)."""
        if not (w.is_atomic and r.is_atomic):
            return BoolVal(False)
        if w.old_value is None or r.old_value is None:
            return BoolVal(False)
        if w.elem_size != r.elem_size:
            return BoolVal(False)
        return w.addr == r.addr

    def _same_dynamic_op(
        self, e: SymbolicMemoryEvent, f: SymbolicMemoryEvent
    ) -> BoolRef:
        """Predicate: the two events denote ONE dynamic operation — the
        cross-copy alpha-twins of a record (same record, same lane) with
        both copies pinned to the same block. The same-instance query pins
        pids equal, so without this identification the coherence machinery
        would demand two distinct order positions and two distinct rf
        sources for what is a single real operation, making e.g. the
        rank-0 winner's world structurally UNSAT (adversarial finding:
        single-winner duplicate-lane WAW silently proved clean)."""
        if e.record is not f.record or e.lane != f.lane or e.copy == f.copy:
            return BoolVal(False)
        pid_a, pid_b = self.ctx_a.pid, self.ctx_b.pid
        return And(*[pid_a[i] == pid_b[i] for i in range(3)])

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
        """sw goes through READS-THROUGH (B.1.4), not bare rf: a release
        write also synchronizes with an acquire read that reads the tail of
        an RMW chain rooted at it. ``rf_unknown`` / ``rf_chain`` sources are
        deliberately absent from reads_through — an unmodeled source must
        never manufacture an hb edge."""
        rt = self.reads_through.get((w.idx, r.idx))
        if rt is None:
            return BoolVal(False)
        return And(
            BoolVal(is_release_sem(w.sem)),
            BoolVal(is_acquire_sem(r.sem)),
            self._scope_ok(w, r),
            rt,
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

    # Cap on initial-source disjunction size. Above this, the solver falls
    # back to rf_unknown (no synchronizes-with) — keeps formulas tractable.
    _MAX_INITIAL_ATOMIC_ELEMENTS: int = 1024

    @classmethod
    def _initial_atomic_source(cls, r: SymbolicMemoryEvent) -> Any:
        """Predicate that ``r`` reads the launch-time initial value.

        Supports scalar tensors and small contiguous flag arrays. For large or
        non-contiguous tensors, returns ``None`` so the solver falls back to
        ``rf_unknown`` without synchronizes-with.
        """
        t = r.record.tensor
        if t is None or r.old_value is None:
            return None
        # Mirror the capture-side dtype guard (_is_modelable_dtype): the
        # model is integer-only, so a float-valued flag must fall back to
        # rf_unknown rather than be silently truncated — int(0.7) == 0
        # would let the modeled CAS succeed where the real one fails (or
        # mask a real race behind a fabricated single-winner lock).
        dtype = getattr(t, "dtype", None)
        if dtype is not None and (
            bool(getattr(dtype, "is_floating_point", False))
            or bool(getattr(dtype, "is_complex", False))
        ):
            return None
        try:
            numel = int(t.numel())
            if numel <= 0 or numel > cls._MAX_INITIAL_ATOMIC_ELEMENTS:
                return None
            if hasattr(t, "is_contiguous") and not bool(t.is_contiguous()):
                return None
            base = int(t.data_ptr())
            elem_size = (
                int(t.element_size()) if hasattr(t, "element_size") else r.elem_size
            )
            elem_size = max(1, elem_size)
            tensor_for_read = t.detach() if hasattr(t, "detach") else t
            tensor_for_read = (
                tensor_for_read.cpu()
                if hasattr(tensor_for_read, "cpu")
                else tensor_for_read
            )
            values = tensor_for_read.reshape(-1).tolist()
        except Exception:
            return None

        clauses = []
        for i, value in enumerate(values):
            # bool is an int subclass; anything else (a duck-typed tensor
            # without a dtype attribute yielding floats) must not be
            # truncated — fall back to rf_unknown.
            if not isinstance(value, int):
                return None
            clauses.append(
                And(
                    r.addr == IntVal(base + i * elem_size),
                    r.old_value == IntVal(int(value)),
                )
            )

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return Or(*clauses)

    def _template_solver(self) -> Solver:
        """A solver holding only the template bounds (grid + arange) — used
        for guard checks that must stay CONSERVATIVE (fewer assumptions →
        more guard failures → axioms omitted, never fabricated)."""
        solver = Solver()
        solver.add(self.grid_constraints)
        for c in self.arange_constraints_a:
            solver.add(c)
        for c in self.arange_constraints_b:
            solver.add(c)
        return solver

    def _has_unmodeled_overlapping_writer(self, r: SymbolicMemoryEvent) -> bool:
        """True when a write the rf model does not include — a plain store,
        or an atomic whose WRITTEN VALUE is not modeled (float/bitwise/
        unsigned RMW) — can overlap the location ``r`` reads.

        Such a writer can publish a value the closed-world choice set
        excludes; without an escape hatch the reader's ``old_value`` would be
        over-constrained and every conflict gated on it silently vanishes
        (e.g. a guard flag set via an unmodeled ``tl.atomic_xchg``). Overlap
        is decided by Z3 on the symbolic addresses under grid/arange bounds
        AND both events' activity (``e.writes`` / ``r.reads`` — masks, path
        conditions, in-bounds premises): a writer only publishes a value
        when it actually writes, and only reads that actually happen need a
        source. Without the activity gate, an address expression drifting
        arithmetically past its tensor under an unbounded symbolic grid
        would open the escape for writers of a DIFFERENT tensor. Writers to
        other tensors (distinct concrete bases) never weaken the closed
        world. A VALUE-MODELED atomic is excluded from this set only to the
        extent its overlap is COVERED by rf candidacy — exact address at
        equal width (_exact_atomic_addr). A torn overlap (different widths,
        or same width at a misaligned unequal address) can publish bytes
        the rf model never delivers, so it must open the escape like any
        unmodeled write (adversarial finding: an 8-byte modeled xchg over a
        4-byte reader was excluded from BOTH channels and pinned the reader
        to the initial value — a false proof).
        """
        candidates = [
            e
            for e in self.events
            if ((not e.is_atomic and e.record.access_mode == "write") or e.is_atomic)
            and e.idx != r.idx
            and self._can_be_rf_candidate(e, r)
        ]
        if not candidates:
            return False
        solver = self._template_solver()
        solver.add(r.reads)
        for e in candidates:
            solver.push()
            solver.add(e.writes)
            solver.add(self._byte_overlap(e, r))
            if e.is_atomic and e.written_value is not None:
                # An ADD write's modeled value old+v is unbounded-Int
                # arithmetic, but the machine wraps: unless the record
                # carries the counting certificate (guard (f) bounded every
                # reachable value inside the dtype range) or provably adds
                # zero, the published value may be one the model cannot
                # express — treat the write as unmodeled (adversarial
                # finding: an INT32_MAX counter's wrap-gated store was
                # proved dead through the closed-world value chain).
                # max/min never leave the operands' range and xchg/CAS
                # write user values, so only `add` is wrap-capable.
                wrap_capable = (
                    (e.record.rmw_op or "") == "add"
                    and _as_numeral(e.rmw_operand) != 0
                    and id(e.record) not in self._counting
                )
                if not wrap_capable and e.elem_size == r.elem_size:
                    # rf covers exactly the equal-width same-address
                    # overlap; only the residual TORN overlap counts as
                    # unmodeled. (Width mismatch: no rf edge exists at all
                    # — any overlap is uncovered.)
                    solver.add(e.addr != r.addr)
            # Z3 ``unknown`` must open the escape: keeping the closed world
            # on an undecided overlap would over-constrain the reader's old
            # value and silently hide every conflict gated on it.
            feasible = solver.check() != unsat
            solver.pop()
            if feasible:
                return True
        return False

    def _build_read_from_choices(self) -> None:
        # Closed-world atomic source model.
        # If the initial scalar source is identifiable, source choices are
        # closed over: (initial source) + (value-modeled atomic writers —
        # CAS and modeled RMW write parts, spec B.1.2). If the initial
        # source is not identifiable — or an UNMODELED write (plain store,
        # float/bitwise RMW) can overlap the location, publishing a value
        # the closed world does not contain — rf_unknown is introduced and
        # does NOT enable synchronizes-with. This is intentionally NOT a
        # full coherence/read-from model over all program instances; the
        # guarded acq_rel CAS no-race result depends on this closed-world
        # assumption holding whenever the flag is only ever written by
        # modeled atomics. For a COUNTED record (B.1.5) the closed world is
        # widened by rf_chain: the reader may take its value from an
        # unmodeled grid instance at rank-1 (the counting equations already
        # pin the value), so real races at non-adjacent ranks are not
        # squeezed out. rf_chain never enables synchronizes-with either —
        # the counted sw edge rides on coherence order via reads_through.
        modeled_writers = [
            e for e in self.events if e.is_atomic and e.written_value is not None
        ]
        for r in self.events:
            if not r.is_atomic or r.old_value is None:
                continue
            choices: list[BoolRef] = []
            init_pred = self._initial_atomic_source(r)

            if init_pred is not None:
                rf_init = Bool(f"rf_init_{r.idx}")
                self.rf_init_source[r.idx] = rf_init
                choices.append(rf_init)
                self.rf_constraints.append(Implies(rf_init, And(r.reads, init_pred)))
            if init_pred is None or self._has_unmodeled_overlapping_writer(r):
                rf_unknown = Bool(f"rf_unknown_{r.idx}")
                self.rf_unknown_source[r.idx] = rf_unknown
                choices.append(rf_unknown)
                self.rf_constraints.append(Implies(rf_unknown, r.reads))

            counting = self._counting.get(id(r.record))
            if counting is not None:
                rank_r = counting.rank_a if r.copy == "a" else counting.rank_b
                rf_chain = Bool(f"rf_chain_{r.idx}")
                self.rf_chain_source[r.idx] = rf_chain
                choices.append(rf_chain)
                # Source = the unmodeled instance at rank-1; rank >= 1 or
                # the only source with the matching value is the initial
                # one, which rf_init already covers.
                self.rf_constraints.append(Implies(rf_chain, And(r.reads, rank_r >= 1)))

            for w in modeled_writers:
                if not self._can_be_rf_candidate(w, r):
                    continue
                rf = Bool(f"rf_{w.idx}_to_{r.idx}")
                choices.append(rf)
                self.rf_source[(w.idx, r.idx)] = rf
                self.rf_constraints.append(
                    Implies(
                        rf,
                        modeled_atomic_read_from(
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

    # ──────────────────── modeled-atomic coherence ────────────────────

    def _modeled_atomic_events(self) -> list[SymbolicMemoryEvent]:
        """Events in the per-location atomic order: every VALUE-MODELED
        atomic — CAS always, RMW exactly when its observation is modeled."""
        return [e for e in self.events if e.is_atomic and e.old_value is not None]

    def _make_atomic_order_vars(self) -> dict[int, Any]:
        """One symbolic atomic-order position per value-modeled atomic.

        The variable denotes the position of the WHOLE operation in the
        per-location atomic order — the read and the write part share it,
        which is what makes RMW atomicity/immediacy (spec B.1.3)
        structural: no write can sit between an RMW's read and its write
        because they occupy one position, and the rf constraints below
        forbid a writer strictly between the source and the reader.
        """
        return {
            e.idx: Int(f"atomic_order_{e.idx}")
            for e in self.events
            if e.is_atomic and e.old_value is not None
        }

    def _build_atomic_coherence_constraints(self) -> None:
        """Closed-world atomic coherence for the two modeled program copies.

        Without these constraints, two CAS try-locks at the same flag could
        both read the initial value and both succeed — producing a false WAW
        on guarded stores; likewise two modeled RMWs could both observe the
        initial counter. The coherence model (over value-modeled atomics):

          * Active actions get bounded atomic-order positions.
          * Same-address active actions are distinct in the per-location
            order.
          * Same-copy program order is preserved for same-address actions.
          * If a reader reads the initial source, no modeled successful
            writer at the same address may precede it.
          * If r reads from modeled writer w, w must be before r in the order
            and no modeled same-address successful writer may sit between
            them — with the shared read/write position this is exactly the
            RMW immediacy axiom (B.1.3): what makes two successful lock
            acquisitions of the same "0" unsatisfiable.

        This is not a full GPU memory model, but it suffices to suppress the
        most obvious unsoundness around atomic synchronization patterns.
        """
        atomic_events = self._modeled_atomic_events()
        if not atomic_events:
            return

        n_orders = max(1, len(atomic_events))
        cons = self.atomic_coherence_constraints

        for e in atomic_events:
            ord_e = self.atomic_order[e.idx]
            cons.append(Implies(e.reads, And(ord_e >= 0, ord_e < n_orders)))

        for i, e in enumerate(atomic_events):
            for f in atomic_events[i + 1 :]:
                same_addr = self._exact_atomic_addr(e, f)
                both_active_same_addr = And(e.reads, f.reads, same_addr)
                ord_e = self.atomic_order[e.idx]
                ord_f = self.atomic_order[f.idx]

                # Distinct positions apply to distinct dynamic operations;
                # a record's cross-copy twins under pinned-equal pids are
                # ONE operation and share the position instead.
                same_op = self._same_dynamic_op(e, f)
                cons.append(
                    Implies(And(both_active_same_addr, Not(same_op)), ord_e != ord_f)
                )
                cons.append(Implies(same_op, ord_e == ord_f))

                if e.copy == f.copy and e.program_seq >= 0 and f.program_seq >= 0:
                    if e.program_seq < f.program_seq:
                        cons.append(Implies(both_active_same_addr, ord_e < ord_f))
                    elif f.program_seq < e.program_seq:
                        cons.append(Implies(both_active_same_addr, ord_f < ord_e))

        # rf_init: no modeled successful writer at the same address may
        # precede the reader in the per-location order. The reader's own
        # cross-copy twin (same dynamic op when pids coincide) is not a
        # DISTINCT writer and must not block rf_init.
        for r in atomic_events:
            rf_init = self.rf_init_source.get(r.idx)
            if rf_init is None:
                continue
            ord_r = self.atomic_order[r.idx]
            for w in atomic_events:
                if w.idx == r.idx:
                    continue
                ord_w = self.atomic_order[w.idx]
                cons.append(
                    Implies(
                        And(
                            rf_init,
                            w.writes,
                            self._exact_atomic_addr(w, r),
                            Not(self._same_dynamic_op(w, r)),
                        ),
                        ord_r < ord_w,
                    )
                )

        # rf from modeled writer w to reader r: w precedes r and no modeled
        # same-address successful writer sits strictly between w and r
        # (immediacy, B.1.3 — the reader's write part shares its position).
        # Twins of w or r (same dynamic op) share the endpoints' positions
        # and are not "between".
        for r in atomic_events:
            ord_r = self.atomic_order[r.idx]
            for w in atomic_events:
                rf = self.rf_source.get((w.idx, r.idx))
                if rf is None:
                    continue
                ord_w = self.atomic_order[w.idx]
                cons.append(Implies(rf, ord_w < ord_r))
                # An operation never reads-from itself: forbid the rf edge
                # between cross-copy twins when they denote one op.
                cons.append(Implies(rf, Not(self._same_dynamic_op(w, r))))

                for v in atomic_events:
                    if v.idx in (w.idx, r.idx):
                        continue
                    ord_v = self.atomic_order[v.idx]
                    cons.append(
                        Implies(
                            And(
                                rf,
                                v.writes,
                                self._exact_atomic_addr(v, r),
                                Not(self._same_dynamic_op(v, r)),
                                Not(self._same_dynamic_op(v, w)),
                            ),
                            Or(ord_v < ord_w, ord_r < ord_v),
                        )
                    )

    # ──────────────────── counting axiom (B.1.5) ────────────────────

    def _build_counting_axioms(self) -> dict[int, _CountingInfo]:
        """Per-copy rank variables for each RMW record whose guards hold.

        Guards, all checked on the TEMPLATE (Z3 ``unknown`` fails a guard —
        the axiom is then OMITTED, never approximated):
          (a) the record is a value-modeled ``add`` RMW with exactly one
              lane per copy, and its address simplifies to ONE numeral in
              both copies (a fixed scalar location L);
          (b) its activity is provably true for every instance;
          (c) the increment is a constant c > 0;
          (d) L's initial value is known via the rf-init machinery's own
              side conditions;
          (e) no OTHER event's write part can overlap L (which also keeps
              the rf_unknown escape closed for the counted readers).
        The grid-size product must stay linear: at most one symbolic dim.

        Justification of the emitted constraints: under (a)-(e) every write
        to L is an instance of this one always-active RMW, so the
        per-location coherence order is exactly the arrival order; the k-th
        arriver observes init + c·k (induction over the chain, using
        atomicity/immediacy), distinct instances have distinct ranks, and
        coherence order agrees with rank order. Consequences the solver
        derives for free: distinct observations across blocks (work-queue
        disjointness) and "o = init + c·(|G|-1) forces coherence-after-
        everyone" (last-block-done).
        """
        out: dict[int, _CountingInfo] = {}
        by_record: dict[int, list[SymbolicMemoryEvent]] = {}
        for e in self.events:
            if e.is_atomic and e.atomic_kind == "rmw" and e.old_value is not None:
                by_record.setdefault(id(e.record), []).append(e)
        if not by_record:
            return out
        # |G| must stay linear (rank < g0·g1·g2): >1 symbolic dim is
        # Z3-unknown bait, so the axiom is omitted for such grids.
        if sum(1 for d in self.grid if _is_symbolic_dim(d)) > 1:
            return out
        grid_size = self.grid[0] * self.grid[1] * self.grid[2]

        for rec_key, evts in by_record.items():
            if len(evts) != 2:
                continue  # multi-lane record (or a missing copy)
            e_a = next((e for e in evts if e.copy == "a"), None)
            e_b = next((e for e in evts if e.copy == "b"), None)
            if e_a is None or e_b is None:
                continue
            if (e_a.record.rmw_op or "") != "add":
                continue
            info = self._counting_guards(e_a, e_b, grid_size)
            if info is not None:
                out[rec_key] = info
        return out

    def _counting_guards(
        self,
        e_a: SymbolicMemoryEvent,
        e_b: SymbolicMemoryEvent,
        grid_size: Any,
    ) -> _CountingInfo | None:
        # (a) fixed scalar location: both copies' addresses are the SAME
        # numeral (no pid/arange/copy-local dependence survives simplify).
        loc = _as_numeral(e_a.addr)
        if loc is None or _as_numeral(e_b.addr) != loc:
            return None
        # (c) constant increment > 0, identical in both copies.
        inc = _as_numeral(e_a.rmw_operand)
        if inc is None or inc <= 0 or _as_numeral(e_b.rmw_operand) != inc:
            return None
        # (b) activity provably true for every instance of the grid.
        solver = self._template_solver()
        for e in (e_a, e_b):
            solver.push()
            solver.add(Not(e.active))
            always_active = solver.check() == unsat
            solver.pop()
            if not always_active:
                return None
        # (d) known initial value, via the rf-init machinery's own side
        # conditions (so rf_init is guaranteed to exist as a choice).
        if self._initial_atomic_source(e_a) is None:
            return None
        init = self._initial_value_at(e_a, loc)
        if init is None:
            return None
        # (f) no machine-integer wraparound on any LAUNCHABLE grid: the
        # value model is unbounded-Int, so pinning o = init + c·rank is
        # sound only while the largest reachable observation stays inside
        # the element type's signed range (adversarial finding: an
        # INT32_MAX-initialized counter really wraps on hardware while the
        # model proved its wrap-gated store dead). Symbolic dims are
        # bounded by the CUDA launch caps (2^31-1 on x, 65535 on y/z) —
        # no real launch exceeds them, so certifying up to the cap
        # certifies every execution the claim covers.
        elem = max(1, e_a.elem_size)
        max_g = 1
        caps = (2**31 - 1, 65535, 65535)
        for axis, d in enumerate(self.grid):
            max_g *= caps[axis] if _is_symbolic_dim(d) else max(1, int(d))
        signed_max = (1 << (8 * elem - 1)) - 1
        # The largest reachable value is the LAST WRITE, init + inc·|G|
        # (the last observation plus one increment) — the certificate must
        # cover it, since it is what the escape hatch trusts to stay
        # representable.
        if init + inc * max_g > signed_max:
            return None
        # (e) no other event's write part can overlap [L, L+elem). The
        # check runs under the axiom's own PROVISIONAL observation bounds
        # (init <= o <= init + c·(|G|-1)) — sound by first-violation
        # induction over L's coherence order: before the first foreign
        # write, every observation is a bounded chain value, so a foreign
        # write whose address needs an out-of-bound o cannot be first
        # (work-queue stores addressed by the observation would otherwise
        # alias everything and kill the axiom that pins them).
        own = {e_a.idx, e_b.idx}
        lo = IntVal(init)
        hi = IntVal(init) + IntVal(inc) * (grid_size - 1)
        for e in self.events:
            if e.idx in own:
                continue
            if not e.is_atomic and e.record.access_mode != "write":
                continue
            solver.push()
            for o in (e_a.old_value, e_b.old_value):
                solver.add(o >= lo, o <= hi)
            solver.add(e.writes)
            solver.add(e.addr < IntVal(loc + elem))
            solver.add(IntVal(loc) < e.addr + e.elem_size)
            overlap_possible = solver.check() != unsat
            solver.pop()
            if overlap_possible:
                return None

        rank_a = Int(f"rmw_rank_{e_a.idx}_a")
        rank_b = Int(f"rmw_rank_{e_b.idx}_b")
        pid_a, pid_b = self.ctx_a.pid, self.ctx_b.pid
        same_block = And(*[pid_a[i] == pid_b[i] for i in range(3)])
        ord_a = self.atomic_order[e_a.idx]
        ord_b = self.atomic_order[e_b.idx]
        cons = self.counting_constraints
        cons.append(And(rank_a >= 0, rank_a < grid_size))
        cons.append(And(rank_b >= 0, rank_b < grid_size))
        # rank is the instance's position in L's coherence order: equal iff
        # the two copies denote the same block.
        cons.append(same_block == (rank_a == rank_b))
        cons.append(e_a.old_value == IntVal(init) + IntVal(inc) * rank_a)
        cons.append(e_b.old_value == IntVal(init) + IntVal(inc) * rank_b)
        cons.append((ord_b < ord_a) == (rank_b < rank_a))
        return _CountingInfo(
            idx_a=e_a.idx,
            idx_b=e_b.idx,
            rank_a=rank_a,
            rank_b=rank_b,
            init=init,
            inc=inc,
            loc=loc,
        )

    @classmethod
    def _initial_value_at(cls, r: SymbolicMemoryEvent, loc: int) -> int | None:
        """The launch-time integer value stored at byte address ``loc`` in
        the reader's tensor, under the same side conditions as
        ``_initial_atomic_source`` (integer contents, aligned, in range)."""
        t = r.record.tensor
        if t is None:
            return None
        try:
            base = int(t.data_ptr())
            elem = max(
                1, int(t.element_size()) if hasattr(t, "element_size") else r.elem_size
            )
            off = loc - base
            if off < 0 or off % elem != 0:
                return None
            idx = off // elem
            if idx >= int(t.numel()):
                return None
            tensor_for_read = t.detach() if hasattr(t, "detach") else t
            tensor_for_read = (
                tensor_for_read.cpu()
                if hasattr(tensor_for_read, "cpu")
                else tensor_for_read
            )
            value = tensor_for_read.reshape(-1).tolist()[idx]
        except Exception:
            return None
        if not isinstance(value, int):  # bool is an int subclass; floats fail
            return None
        return int(value)

    # ──────────────────── reads-through (B.1.4) ────────────────────

    def _build_reads_through(self) -> dict[tuple[int, int], BoolRef]:
        """rt(w, r): r reads-through w — directly (rf), via a bounded chain
        of value-modeled RMW write parts (each link an rf edge, so each
        intermediate's write part is implied active), or — for a counted
        record — via coherence order between its two copies (justified in
        ``_build_counting_axioms``: every write to the counted location is a
        link of the same RMW chain). Chain length over MODELED events is
        bounded by their count; chains through unmodeled grid instances are
        exactly what the counting pairs cover.
        """
        rt: dict[tuple[int, int], BoolRef] = dict(self.rf_source)
        n_modeled = len(self._modeled_atomic_events())
        frontier: dict[tuple[int, int], BoolRef] = dict(rt)
        for _ in range(max(0, n_modeled - 1)):
            grown: dict[tuple[int, int], BoolRef] = {}
            for (w_idx, m_idx), pred in frontier.items():
                for (m2_idx, r_idx), rf2 in self.rf_source.items():
                    if m2_idx != m_idx or r_idx == w_idx:
                        continue
                    step = And(pred, rf2)
                    key = (w_idx, r_idx)
                    grown[key] = Or(grown[key], step) if key in grown else step
            if not grown:
                break
            for key, pred in grown.items():
                rt[key] = Or(rt[key], pred) if key in rt else pred
            frontier = grown
        for info in self._counting.values():
            ord_a = self.atomic_order[info.idx_a]
            ord_b = self.atomic_order[info.idx_b]
            for w_idx, r_idx, before in (
                (info.idx_b, info.idx_a, ord_b < ord_a),
                (info.idx_a, info.idx_b, ord_a < ord_b),
            ):
                key = (w_idx, r_idx)
                rt[key] = Or(rt[key], before) if key in rt else before
        return rt

    def _assert_no_uncounted_observation_addresses(self) -> None:
        """Spec B.5 boundary with the B.1.5 carve-out: an observation
        variable may feed an ADDRESS only when its record's counting axiom
        fired (the address is then affine in the pinned rank — the
        work-queue pattern). Anything else raises: an unconstrained
        observation in an address would alias everything, and silently
        widening is the forbidden failure mode — unsupported is the honest
        verdict."""
        counted: set[int] = set()
        for info in self._counting.values():
            counted.update((info.idx_a, info.idx_b))
        obs_keys: dict[tuple[int, str, str], SymbolicMemoryEvent] = {}
        for e in self._modeled_atomic_events():
            if e.idx in counted:
                continue
            for key in _collect_z3_var_keys((e.old_value,)):
                obs_keys[key] = e
        if not obs_keys:
            return
        for e in self.events:
            hit = _collect_z3_var_keys((e.addr,)) & set(obs_keys)
            if hit:
                src = obs_keys[next(iter(hit))]
                raise UnsupportedSymbolicRaceQuery(
                    f"the observation of atomic {src.name} feeds the address "
                    f"of {e.name}, and the counting axiom's guards do not "
                    "hold for it — an atomic return in address position is "
                    "only modeled under the counting axiom (spec B.1.5)"
                )

    def _base_solver(self) -> Solver:
        """Assertions shared by every race query; the caller adds the
        cross-instance (``different_blocks``) or same-instance constraints.
        """
        solver = Solver()
        solver.add(self.grid_constraints)
        for c in self.arange_constraints_a:
            solver.add(c)
        for c in self.arange_constraints_b:
            solver.add(c)
        for c in self.rf_constraints:
            solver.add(c)
        for c in self.atomic_coherence_constraints:
            solver.add(c)
        for c in self.counting_constraints:
            solver.add(c)
        for c in self.extra_assumptions:
            solver.add(as_bool(c))
        return solver

    def _new_solver(self) -> Solver:
        solver = self._base_solver()
        solver.add(self.different_blocks)
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
        candidates: list[
            tuple[SymbolicMemoryEvent, SymbolicMemoryEvent, ModelRef, str]
        ],
    ) -> list[RaceReport]:
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        reports: list[RaceReport] = []
        for a, b, model, reason in candidates:
            first, second = self._canonical_pair(a, b)
            key = (
                (first.event_id, first.lane),
                (second.event_id, second.lane),
            )
            if key in seen:
                continue
            seen.add(key)
            reports.append(self._make_report(first, second, model, reason))
        return reports

    def _make_report(
        self,
        first: SymbolicMemoryEvent,
        second: SymbolicMemoryEvent,
        model: ModelRef,
        reason: str,
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
            reason=reason,
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
