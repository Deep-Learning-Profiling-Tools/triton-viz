from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)

import numpy as np
from z3.z3 import BoolRef
from triton.runtime.interpreter import TensorHandle, interpreter_builder

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Load,
    RawLoad,
    Store,
    RawStore,
    AtomicCas,
    AtomicRMW,
)
from ...frontends.base import OPERATION_REGISTRY
from ..symbolic_engine import (
    SymbolicExpr,
    SymbolicClient,
    NullSymbolicClient,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
    AccessMode,
)
from .data import (
    AccessEventRecord,
    RaceCandidate,
    RaceReport,
    RaceType,
    VALID_RMW_OPS,
    active_mask_for,
    apply_rmw,
    broadcast_lane_operand,
    effects_at_addr,
    flatten_np,
    infer_elem_size,
    maybe_concretize,
    normalize_sem_scope,
    resolve_tensor_from_pointer,
)
from .hb_solver import HBSolver
from ...utils.traceback_utils import capture_current_source_location
from ...core.config import config as cfg


def _normalize_rmw_op(rmw_op: Any) -> str:
    """Map a Triton RMWOp (enum or string) into the PR2 vocabulary.

    Collapses ``FADD`` → ``"add"``, ``UMAX`` → ``"max"``, ``UMIN`` → ``"min"``
    since the type signage is carried separately by the operand dtype.
    Raises ``NotImplementedError`` for anything outside ``VALID_RMW_OPS``
    after normalization.
    """
    name = getattr(rmw_op, "name", None)
    if name is None:
        name = str(rmw_op)
    if "." in name:
        name = name.rsplit(".", 1)[1]
    low = name.lower()
    low = {"fadd": "add", "umax": "max", "umin": "min"}.get(low, low)
    if low not in VALID_RMW_OPS:
        raise NotImplementedError(f"Unsupported atomic_rmw op: {rmw_op!r}")
    return low


RaceDetectorT = TypeVar("RaceDetectorT", bound="RaceDetector")


def _make_event_signature(
    access_mode: AccessMode,
    source_location: tuple[str, int, str] | None,
    addr_expr: Z3Expr,
    local_constraints: ConstraintConjunction,
) -> int:
    """Signature used to dedupe repeated access events within a single loop.

    Distinct from sanitizer's ``_make_signature``: ``access_mode`` and
    ``source_location`` are part of the key so a ``load`` and a ``store`` at
    the same address inside the same loop stay as separate events (different
    program-order nodes for future HB analysis).
    """
    if isinstance(addr_expr, list):
        if len(addr_expr) == 1:
            addr_hash = hash(addr_expr[0])
        else:
            addr_hash = hash(tuple(hash(e) for e in addr_expr))
    else:
        addr_hash = hash(addr_expr)

    constr_hash = 0 if local_constraints is None else hash(local_constraints)
    return hash((access_mode, source_location, addr_hash, constr_hash))


@dataclass
class PendingEvent(PendingCheck):
    """Extends ``PendingCheck`` with access-mode and op_type metadata.

    ``LoopContext.pending_checks`` is typed as ``list[PendingCheck]`` but is
    fine accepting subclass instances at runtime. Keeping the extension local
    to this module avoids widening the shared schema used by sanitizer.
    """

    access_mode: AccessMode = "read"
    op_type: type[Op] = Load


def _wrap_atomic_old_as_symbolic(old_handle: TensorHandle) -> SymbolicExpr:
    """Wrap the concrete ``old`` TensorHandle returned by a real atomic in
    a ``const`` SymbolicExpr so downstream code sees a SymbolicExpr-like
    object (matching what the rest of the symbolic pipeline produces) while
    still being able to ``concretize()`` back to the hardware result.

    ``ConstSymbolicExpr.concretize()`` returns the wrapped ``TensorHandle``
    directly, and ``SymbolicExpr.concrete_fn`` is a writable class
    attribute — ``PatchOp.__call__`` (``core/patch.py:L221``) assigns it
    after the overrider returns. That satisfies the ``interpreter_builder``
    branch's contract with no extra wrapping.
    """
    return SymbolicExpr.create("const", old_handle, old_handle.dtype)


def _iter_event_bytes(event: AccessEventRecord):
    """Yield every byte address covered by an active lane of ``event``.

    Byte granularity matches ``effects_at_addr`` (upstream #344 fix):
    single-lane-only would miss multi-lane writes to the same element.
    Iteration is per-active-lane because hits are aggregated via
    ``np.any`` over every lane covering a given byte.
    """
    if event.lane_addrs is None or event.elem_size is None:
        return
    active = (
        event.active_mask
        if event.active_mask is not None
        else np.ones_like(event.lane_addrs, dtype=bool)
    )
    elem_size = int(event.elem_size)
    for idx, hit in enumerate(active):
        if not hit:
            continue
        base = int(event.lane_addrs[idx])
        for k in range(elem_size):
            yield base + k


def _canonical_pair(
    a: AccessEventRecord, b: AccessEventRecord
) -> tuple[AccessEventRecord, AccessEventRecord]:
    """Order two conflicting events by ``(grid_idx, program_seq, event_id)``.

    Fixing the direction here (a -> first, b -> second) is what lets Step 4
    emit *directional* RAW vs. WAR — without a canonical order, upstream's
    WAR enum stays dead code (#351).
    """

    def _key(e: AccessEventRecord) -> tuple:
        return (
            e.grid_idx if e.grid_idx is not None else (),
            e.program_seq if e.program_seq is not None else -1,
            e.event_id,
        )

    return (a, b) if _key(a) <= _key(b) else (b, a)


_BARRIER_ATOMIC_OPS: frozenset[str] = frozenset({"add", "cas", "xchg"})
_BARRIER_VALID_SCOPES: frozenset[str | None] = frozenset({"gpu", "sys", None})


def _iter_element_bases(event: AccessEventRecord):
    """Per-active-lane element base addresses — mirrors ``hb_solver._iter_element_bases``.

    Duplicated here (rather than imported) to keep module layering simple:
    race_detector.py already imports HBSolver, but this helper is used by
    the barrier/epoch driver which shouldn't cross-depend on the solver
    module's internals.
    """
    if event.lane_addrs is None:
        return
    active = (
        event.active_mask
        if event.active_mask is not None
        else np.ones_like(event.lane_addrs, dtype=bool)
    )
    for idx, hit in enumerate(active):
        if not hit:
            continue
        yield int(event.lane_addrs[idx])


def _is_barrier_atomic(event: AccessEventRecord) -> bool:
    """Identifies *candidate* barrier atomics — addresses to consider as a
    global phase counter / flag. Not a phase-advance check; don't exclude
    ``relaxed`` here. The phase-advance gate is ``_is_phase_advancing_write``.
    """
    if event.atomic_op not in _BARRIER_ATOMIC_OPS:
        return False
    if event.atomic_scope not in _BARRIER_VALID_SCOPES:
        return False
    # Needs at least one active lane that reads or writes.
    active = (
        event.active_mask
        if event.active_mask is not None
        else (
            np.ones_like(event.lane_addrs, dtype=bool)
            if event.lane_addrs is not None
            else None
        )
    )
    if active is None or not np.any(active):
        return False
    return True


_PHASE_ADVANCE_WRITER_SEMS: frozenset[str] = frozenset({"release", "acq_rel"})


def _is_phase_advancing_write(event: AccessEventRecord, elem_base: int) -> bool:
    """Did ``event`` move the barrier forward? Must satisfy three things:

    1. ``event.atomic_sem`` is ``release`` or ``acq_rel`` — an acquire-
       only write that happens to change the value is consumption, not
       phase advancement; counting it would split epochs even when the
       producer's release never synchronized.
    2. ``event`` actually writes the element (``write_mask`` has True
       on a lane covering ``elem_base``).
    3. ``written_value != atomic_old`` — failed CAS, ``cas(1, 1)``
       polling, ``atomic_add(0)`` all have written == old, and must NOT
       split epochs (upstream #350 same-value ambiguity, mirrored in
       ``_has_ambiguous_writer``).
    """
    if event.atomic_sem not in _PHASE_ADVANCE_WRITER_SEMS:
        return False
    if event.lane_addrs is None:
        return False
    active = (
        event.active_mask
        if event.active_mask is not None
        else np.ones_like(event.lane_addrs, dtype=bool)
    )
    hit_mask = active & (event.lane_addrs == elem_base)
    if not np.any(hit_mask):
        return False
    if event.write_mask is None:
        return False
    if not np.any(event.write_mask[hit_mask]):
        return False
    if event.atomic_old is None or event.written_value is None:
        return False
    old = event.atomic_old[hit_mask]
    new = event.written_value[hit_mask]
    return bool(np.any(new != old))


def _classify_candidate(
    first: AccessEventRecord,
    second: AccessEventRecord,
    byte_addr: int,
) -> RaceType | None:
    """Classify a concrete-event pair at ``byte_addr`` via the effect model.

    The classification goes through ``effects_at_addr`` (not op-type
    hardcoding) so plain load/store, CAS success/failure lanes, and RMW
    events are all treated uniformly. Upstream #345 explicitly flagged
    the op-type hardcoding as a correctness bug.
    """
    r1, w1 = effects_at_addr(first, byte_addr)
    r2, w2 = effects_at_addr(second, byte_addr)
    if w1 and w2:
        return RaceType.WAW
    if w1 and r2:
        return RaceType.RAW
    if r1 and w2:
        return RaceType.WAR
    return None


class RaceDetector(Client):
    """Factory class that returns the concrete race-detector implementation
    based on the value of ``cfg.enable_race_detector``.
    """

    NAME = "race_detector"
    LOG_TAG: ClassVar[str] = "RaceDetector"
    LOG_VERB: ClassVar[str] = "recording"

    def __new__(cls: type[RaceDetectorT], *args: Any, **kwargs: Any) -> RaceDetectorT:
        if cls is RaceDetector:
            target_cls = cast(
                type["RaceDetector"],
                SymbolicRaceDetector if cfg.enable_race_detector else NullRaceDetector,
            )
            obj = object.__new__(target_cls)
            cast(Any, target_cls).__init__(obj, *args, **kwargs)
            return cast(RaceDetectorT, obj)
        return cast(RaceDetectorT, object.__new__(cls))

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.abort_on_error: bool = abort_on_error

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        raise NotImplementedError

    def finalize(self) -> list:
        raise NotImplementedError

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        raise NotImplementedError

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        raise NotImplementedError

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        raise NotImplementedError

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        raise NotImplementedError


class SymbolicRaceDetector(RaceDetector, SymbolicClient):
    def __init__(self, abort_on_error: bool = False):
        super().__init__(abort_on_error=abort_on_error)
        # Step 1 symbolic path
        self.symbolic_events: list[AccessEventRecord] = []
        # ``records`` is Step 1's historical name — kept as an alias for the
        # symbolic path so pre-existing tests keep working.
        self.records = self.symbolic_events
        self._next_event_id = 0

        # Concrete atomic + plain event path
        self.concrete_events: list[AccessEventRecord] = []

        # Tripped by any atomic capture. Step 4/5 consumers must fall back
        # to concrete-only reasoning (or conservatively not suppress) when
        # True. Reset at the START of the next launch (grid_callback), not
        # at finalize — a consumer inspecting the detector immediately
        # after the launch completes must still see True.
        self.atomic_symbolic_escape: bool = False

        # Per-launch state. ``concrete_events`` / ``symbolic_events`` stay
        # client-lifetime accumulations (preserves pre-PR3 test style); race
        # detection slices from these ``*_start`` indices each launch.
        self._launch_symbolic_start: int = 0
        self._launch_concrete_start: int = 0
        self._launch_id: int = 0

        # Per-grid_idx program-order counter. event_id is a unique monotonic
        # id, but under ``num_sms > 1`` blocks race to append so append order
        # is not program order — ``po`` must key on ``(grid_idx, program_seq)``.
        self._program_seq_by_grid: defaultdict[tuple[int, ...], int] = defaultdict(int)

        # Stubs populated by the Step 4/5/6 pipeline (wired in commits 3–4).
        self._last_candidates: list = []
        self._last_reports: list = []
        self._last_barrier_addrs: set[int] = set()

        # Cache the un-patched atomic ops so the overrider below can execute
        # the real hardware atomic on concrete inputs without re-entering
        # ``PatchOp`` (which would loop back into ourselves).
        triton_frontend = OPERATION_REGISTRY["triton"]
        self._original_atomic_cas = triton_frontend.original_ops[interpreter_builder][
            "create_atomic_cas"
        ]
        self._original_atomic_rmw = triton_frontend.original_ops[interpreter_builder][
            "create_atomic_rmw"
        ]

    def _next_program_seq(self, grid_idx: tuple[int, ...]) -> int:
        """Allocate the next program-order index for ``grid_idx`` within
        the current launch. Both the atomic overriders and the plain
        load/store before-callbacks MUST call this when emitting concrete
        events — po, epoch partitioning, and same-value-ambiguity checks
        all depend on it."""
        n = self._program_seq_by_grid[grid_idx]
        self._program_seq_by_grid[grid_idx] = n + 1
        return n

    def _new_event_id(self) -> int:
        """Client-lifetime monotonic unique ID. NOT an execution-order
        indicator under multi-SM concurrency — blocks race to append, so
        append order != execution order. Consumers that need execution
        order must key on grid_idx + program_seq, not event_id."""
        eid = self._next_event_id
        self._next_event_id += 1
        return eid

    # Explicit forwarders to SymbolicClient: the RaceDetector factory
    # carries concrete stubs (NotImplementedError or ``return True``) to
    # satisfy Client's @abstractmethod contract, and those stubs would
    # otherwise shadow SymbolicClient's impls in the subclass MRO.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicClient.grid_idx_callback(self, grid_idx)

    def finalize(self) -> list:
        # NOTE: atomic_symbolic_escape is intentionally NOT reset here —
        # Step 4/5 consumers inspect the detector right after the launch
        # completes and must still see True. The flag is reset at the
        # next launch's grid_callback.
        SymbolicClient.finalize(self)

        launch_events = [
            ev
            for ev in self.concrete_events[self._launch_concrete_start :]
            if not ev.legacy_atomic
        ]
        if not launch_events:
            self._last_candidates = []
            self._last_reports = []
            self._last_barrier_addrs = set()
            return []

        # Step 6: coarse epoch partitioning on phase-advancing barrier writes.
        total_blocks = 1
        if self.grid is not None:
            total_blocks = int(self.grid[0]) * int(self.grid[1]) * int(self.grid[2])
        barrier_addrs = self._detect_global_barrier_addresses(
            launch_events, total_blocks
        )
        self._assign_concrete_epochs(launch_events, barrier_addrs)

        # Step 4: byte-bucket candidate pairing (now respects per-event epoch).
        candidates = self._find_race_candidates(launch_events)

        # Step 5: one HBSolver per epoch; drop candidates that po | sw orders.
        epochs: dict[int, list[AccessEventRecord]] = {}
        for ev in launch_events:
            epochs.setdefault(ev.epoch, []).append(ev)
        solvers = {epoch: HBSolver(evs) for epoch, evs in epochs.items()}

        reports: list[RaceReport] = []
        for cand in candidates:
            solver = solvers.get(cand.epoch)
            if solver is None:
                reports.append(self._candidate_to_report(cand))
                continue
            hb = solver.check_candidate(cand)
            if hb.ordered:
                continue
            reports.append(self._candidate_to_report(cand, hb_reason=hb.reason))

        self._last_candidates = candidates
        self._last_reports = reports
        self._last_barrier_addrs = barrier_addrs
        return reports

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        SymbolicClient.arg_callback(self, name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        # Per-launch reset. ``symbolic_events`` / ``concrete_events`` stay
        # client-lifetime; the race pipeline slices from these start
        # indices. ``atomic_symbolic_escape`` is reset HERE, not in
        # ``finalize()``, so consumers inspecting the detector immediately
        # after a launch ends still see True.
        self.atomic_symbolic_escape = False
        self._launch_id += 1
        self._launch_symbolic_start = len(self.symbolic_events)
        self._launch_concrete_start = len(self.concrete_events)
        self._program_seq_by_grid.clear()
        self._last_candidates = []
        self._last_reports = []
        self._last_barrier_addrs = set()
        SymbolicClient.grid_callback(self, grid)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        if op_type is AtomicCas:
            return OpCallbacks(
                op_overrider=self.lock_fn(self._op_atomic_cas_overrider),
            )
        if op_type is AtomicRMW:
            return OpCallbacks(
                op_overrider=self.lock_fn(self._op_atomic_rmw_overrider),
            )
        # Plain load/store: stack a concrete before-callback on top of the
        # symbolic overrider SymbolicClient provides. The before-callback
        # records a concrete event (lane addresses, active mask, read/write
        # effect) alongside the symbolic path that keeps producing
        # SymbolicExpr nodes for downstream analysis.
        if op_type is Load:
            base = SymbolicClient.register_op_callback(self, op_type)
            return OpCallbacks(
                before_callback=self.lock_fn(self._before_load_concrete),
                after_callback=base.after_callback,
                op_overrider=base.op_overrider,
            )
        if op_type is RawLoad:
            base = SymbolicClient.register_op_callback(self, op_type)
            return OpCallbacks(
                before_callback=self.lock_fn(self._before_raw_load_concrete),
                after_callback=base.after_callback,
                op_overrider=base.op_overrider,
            )
        if op_type is Store:
            base = SymbolicClient.register_op_callback(self, op_type)
            return OpCallbacks(
                before_callback=self.lock_fn(self._before_store_concrete),
                after_callback=base.after_callback,
                op_overrider=base.op_overrider,
            )
        if op_type is RawStore:
            base = SymbolicClient.register_op_callback(self, op_type)
            return OpCallbacks(
                before_callback=self.lock_fn(self._before_raw_store_concrete),
                after_callback=base.after_callback,
                op_overrider=base.op_overrider,
            )
        return SymbolicClient.register_op_callback(self, op_type)

    def pre_run_callback(self, fn: Callable) -> bool:
        # Race detection fundamentally needs full-grid coverage — cross-block
        # conflicts can't be reasoned about if blocks get skipped. Force
        # every block to run; let SymbolicClient's pre_run still do its
        # per-block setup.
        SymbolicClient.pre_run_callback(self, fn)
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        # Same reasoning — PatchOp terminates the grid iteration when any
        # client's post_run returns False (see ``core/patch.py:L488-507``).
        # SymbolicClient returns False for non-data-dependent launches to
        # skip redundant blocks; for the race detector every block is
        # meaningful, so force True and let the base still clean caches.
        SymbolicClient.post_run_callback(self, fn)
        return True

    # ── Event recording (symbolic) ────────────────────────────────────────

    def _emit_symbolic_event(
        self,
        access_mode: AccessMode,
        op_type: type[Op],
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        # Full solver-assertion snapshot at the moment this event is recorded.
        # When called from _loop_hook_after it runs between solver.push() and
        # solver.pop(), so the snapshot captures:
        #   addr_ok, pid_ok, innermost loop iterator, all outer loop iterators.
        # plus the expression's own local constraints — giving Step 2 enough
        # context to run alias queries without replaying callbacks.
        solver_snapshot: tuple[Any, ...] = (
            tuple(self.solver.assertions()) if self.solver is not None else ()
        )
        if expr_constraints is None:
            local: tuple[Any, ...] = ()
        elif isinstance(expr_constraints, (list, tuple)):
            local = tuple(expr_constraints)
        else:
            local = (expr_constraints,)

        self.symbolic_events.append(
            AccessEventRecord(
                event_id=self._new_event_id(),
                op_type=op_type,
                access_mode=access_mode,
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=access_addr,
                premises=solver_snapshot + local,
                local_constraints=local,
                source_location=source_location,
                grid_idx=self.grid_idx,
            )
        )

    def _handle_access_check(
        self,
        expr: SymbolicExpr,
        op_type: type[Op],
        access_mode: AccessMode,
    ) -> None:
        """Capture a memory access expression.

        Outside any loop: recorded immediately. Inside a loop: deferred to the
        enclosing loop's flush point, with ``_make_event_signature`` used to
        dedupe events that repeat across iterations of the same loop.
        """
        z3_addr, z3_constraints = expr.eval()
        source_location = capture_current_source_location()

        if not self.loop_stack:
            self._emit_symbolic_event(
                access_mode,
                op_type,
                z3_addr,
                z3_constraints,
                expr,
                source_location,
            )
            return

        ctx = self.loop_stack[-1]
        signature = _make_event_signature(
            access_mode, source_location, z3_addr, z3_constraints
        )
        pending_idx = ctx.signature_cache.get(signature)
        if pending_idx is None:
            ctx.signature_cache[signature] = len(ctx.pending_checks)
            ctx.pending_checks.append(
                PendingEvent(
                    symbolic_expr=expr,
                    addr_expr=z3_addr,
                    constraints=z3_constraints,
                    source_location=source_location,
                    access_mode=access_mode,
                    op_type=op_type,
                )
            )
        else:
            if cfg.verbose:
                print(f"[{self.LOG_TAG}]  ↪ skip duplicated addr in loop")

    # ── Per-pending handler invoked from SymbolicClient's loop template

    def _process_pending_check(
        self,
        ctx: LoopContext,
        pending: PendingCheck,
        iter_constraints: list[BoolRef],
    ) -> None:
        del ctx, iter_constraints  # verbose logging is handled by the base
        # Items enqueued by _handle_access_check are PendingEvent instances
        # (subclass of PendingCheck) — narrow so attribute accesses are
        # type-safe under Literal["read", "write"].
        assert isinstance(pending, PendingEvent)
        self._emit_symbolic_event(
            pending.access_mode,
            pending.op_type,
            pending.addr_expr,
            pending.constraints,
            pending.symbolic_expr,
            pending.source_location,
        )

    # ── Concrete atomic capture (concretize-and-execute overriders) ───────
    #
    # Each atomic overrider runs in PatchOp's interpreter_builder branch.
    # Order of operations:
    #   1. Concretize inputs (``maybe_concretize`` — no-op on already
    #      concrete TensorHandles, walks the symbolic tree otherwise).
    #   2. Build per-lane concrete address / mask / element-size metadata
    #      from the concrete inputs.
    #   3. Call the real ``interpreter_builder.create_atomic_*`` captured at
    #      init time — this actually writes memory and returns a TensorHandle
    #      carrying the hardware ``old``.
    #   4. Compute the effect-model (read_mask / write_mask / written_value)
    #      and append an AccessEventRecord to ``concrete_events``.
    #   5. Return a ``const`` SymbolicExpr wrapping the TensorHandle so
    #      downstream symbolic consumers keep working; PatchOp's
    #      interpreter_builder branch then assigns ``.concrete_fn = self.op``
    #      which makes a later ``concretize()`` on this node return the
    #      cached old value without re-executing the atomic.

    def _op_atomic_cas_overrider(self, ptr, cmp, val, sem, scope):
        self.atomic_symbolic_escape = True
        grid_idx = self.grid_idx
        assert grid_idx is not None, "atomic overrider fired before grid_idx_callback"

        c_ptr = maybe_concretize(ptr)
        c_cmp = maybe_concretize(cmp)
        c_val = maybe_concretize(val)

        lane_addrs = flatten_np(c_ptr).astype(np.int64, copy=False)
        nlanes = lane_addrs.shape[0]
        cmp_np = broadcast_lane_operand(c_cmp, nlanes)
        val_np = broadcast_lane_operand(c_val, nlanes)
        active = active_mask_for(None, nlanes)  # CAS has no mask parameter
        sem_norm, scope_norm = normalize_sem_scope(sem, scope)
        elem_size = infer_elem_size(c_val, c_ptr)
        tensor = resolve_tensor_from_pointer(
            c_ptr, active, elem_size, self.tensor_addrs
        )

        ret_handle = self._original_atomic_cas(c_ptr, c_cmp, c_val, sem, scope)
        old_np = flatten_np(ret_handle)

        success = active & np.equal(old_np, cmp_np)
        read_mask = active.copy()
        write_mask = success
        written_value = np.where(success, val_np, old_np).astype(
            old_np.dtype, copy=False
        )

        self.concrete_events.append(
            AccessEventRecord(
                event_id=self._new_event_id(),
                op_type=AtomicCas,
                source_location=capture_current_source_location(),
                grid_idx=grid_idx,
                tensor=tensor,
                tensor_name=(
                    self._get_tensor_name(tensor) if tensor is not None else None
                ),
                lane_addrs=lane_addrs,
                active_mask=active,
                elem_size=elem_size,
                read_mask=read_mask,
                write_mask=write_mask,
                atomic_op="cas",
                atomic_sem=sem_norm,
                atomic_scope=scope_norm,
                atomic_cmp=cmp_np,
                atomic_val=val_np,
                atomic_old=old_np,
                written_value=written_value,
                program_seq=self._next_program_seq(grid_idx),
                launch_id=self._launch_id,
            )
        )

        return _wrap_atomic_old_as_symbolic(ret_handle)

    def _op_atomic_rmw_overrider(self, rmwOp, ptr, val, mask, sem, scope):
        self.atomic_symbolic_escape = True
        grid_idx = self.grid_idx
        assert grid_idx is not None, "atomic overrider fired before grid_idx_callback"

        atomic_op = _normalize_rmw_op(rmwOp)
        c_ptr = maybe_concretize(ptr)
        c_val = maybe_concretize(val)
        c_mask = maybe_concretize(mask)

        lane_addrs = flatten_np(c_ptr).astype(np.int64, copy=False)
        nlanes = lane_addrs.shape[0]
        val_np = broadcast_lane_operand(c_val, nlanes)
        active = active_mask_for(c_mask, nlanes)
        sem_norm, scope_norm = normalize_sem_scope(sem, scope)
        elem_size = infer_elem_size(c_val, c_ptr)
        tensor = resolve_tensor_from_pointer(
            c_ptr, active, elem_size, self.tensor_addrs
        )

        ret_handle = self._original_atomic_rmw(rmwOp, c_ptr, c_val, c_mask, sem, scope)
        old_np = flatten_np(ret_handle)

        read_mask = active.copy()
        write_mask = active.copy()
        written_value = apply_rmw(atomic_op, old_np, val_np)

        self.concrete_events.append(
            AccessEventRecord(
                event_id=self._new_event_id(),
                op_type=AtomicRMW,
                source_location=capture_current_source_location(),
                grid_idx=grid_idx,
                tensor=tensor,
                tensor_name=(
                    self._get_tensor_name(tensor) if tensor is not None else None
                ),
                lane_addrs=lane_addrs,
                active_mask=active,
                elem_size=elem_size,
                read_mask=read_mask,
                write_mask=write_mask,
                atomic_op=atomic_op,
                atomic_sem=sem_norm,
                atomic_scope=scope_norm,
                atomic_cmp=None,
                atomic_val=val_np,
                atomic_old=old_np,
                written_value=written_value,
                program_seq=self._next_program_seq(grid_idx),
                launch_id=self._launch_id,
            )
        )

        return _wrap_atomic_old_as_symbolic(ret_handle)

    # ── Concrete plain load / store capture ───────────────────────────────
    #
    # Stacked on top of SymbolicClient's symbolic overrider, so the
    # symbolic pipeline keeps producing SymbolicExpr nodes for Step 1
    # analysis while the race pipeline gets the concrete lane geometry it
    # needs. Order in PatchOp.__call__: before → overrider → after.
    # TensorPointerLoad / TensorPointerStore are NOT captured here — block
    # pointer element-address extraction is deferred.

    # Adapter-driven signatures (see ``frontends/triton.py``):
    #   * ``_triton_load_adapter``      -> ``(ptr, mask, keys)``
    #   * ``_triton_raw_load_adapter``  -> ``(ptr,)``
    #   * ``_triton_store_adapter``     -> ``(ptr, mask, keys)`` — ``value`` is
    #                                     stripped by design; elem_size falls
    #                                     through ``ptr.dtype.element_ty`` in
    #                                     ``_pointer_elem_size``.
    #   * ``_triton_raw_store_adapter`` -> ``(ptr, value)``

    def _before_load_concrete(self, ptr, mask=None, keys=None) -> None:
        del keys
        self._emit_concrete_plain_event(Load, "read", ptr, mask=mask, value=None)

    def _before_raw_load_concrete(self, ptr) -> None:
        self._emit_concrete_plain_event(RawLoad, "read", ptr, mask=None, value=None)

    def _before_store_concrete(self, ptr, mask=None, keys=None) -> None:
        del keys
        self._emit_concrete_plain_event(Store, "write", ptr, mask=mask, value=None)

    def _before_raw_store_concrete(self, ptr, value) -> None:
        self._emit_concrete_plain_event(RawStore, "write", ptr, mask=None, value=value)

    def _emit_concrete_plain_event(
        self,
        op_type: type[Op],
        access_mode: str,  # "read" | "write"
        ptr: Any,
        mask: Any = None,
        value: Any = None,
    ) -> None:
        grid_idx = self.grid_idx
        if grid_idx is None:
            # Defensive: a plain load/store firing before grid_idx_callback
            # would mean a client dispatch happened outside the normal
            # launch lifecycle. Skip rather than crash.
            return

        c_ptr = maybe_concretize(ptr)
        c_mask = maybe_concretize(mask)
        c_value = maybe_concretize(value) if value is not None else None

        try:
            lane_addrs = flatten_np(c_ptr).astype(np.int64, copy=False)
        except Exception:
            # Non-pointer shapes the interpreter handles via symbolic-only
            # paths — skip concrete capture rather than raise.
            return
        nlanes = lane_addrs.shape[0]
        if nlanes == 0:
            return

        active = active_mask_for(c_mask, nlanes)
        try:
            elem_size = infer_elem_size(c_value, c_ptr)
        except ValueError:
            return
        tensor = resolve_tensor_from_pointer(
            c_ptr, active, elem_size, self.tensor_addrs
        )

        if access_mode == "read":
            read_mask = active.copy()
            write_mask = np.zeros_like(active)
        else:
            read_mask = np.zeros_like(active)
            write_mask = active.copy()

        self.concrete_events.append(
            AccessEventRecord(
                event_id=self._new_event_id(),
                op_type=op_type,
                source_location=capture_current_source_location(),
                grid_idx=grid_idx,
                tensor=tensor,
                tensor_name=(
                    self._get_tensor_name(tensor) if tensor is not None else None
                ),
                lane_addrs=lane_addrs,
                active_mask=active,
                elem_size=elem_size,
                read_mask=read_mask,
                write_mask=write_mask,
                program_seq=self._next_program_seq(grid_idx),
                launch_id=self._launch_id,
            )
        )

    # ── Candidate finder (Step 4) ─────────────────────────────────────────
    #
    # Bucket concrete events by ``(epoch, byte_addr)`` and pair everything
    # in each bucket. Skip same-grid_idx pairs (intra-block is po-ordered,
    # not PR3's target) and dedupe on ``(min(event_id), max(event_id))``
    # so a multi-byte conflict between the same two events emits a single
    # candidate, not ``elem_size`` copies.

    def _find_race_candidates(
        self, events: list[AccessEventRecord]
    ) -> list[RaceCandidate]:
        addr_buckets: dict[tuple[int, int], list[AccessEventRecord]] = {}
        for ev in events:
            for byte_addr in _iter_event_bytes(ev):
                addr_buckets.setdefault((ev.epoch, byte_addr), []).append(ev)

        candidates: list[RaceCandidate] = []
        seen_pairs: set[tuple[int, int, int]] = set()

        for (epoch, byte_addr), bucket in addr_buckets.items():
            n = len(bucket)
            for i in range(n):
                ev_a = bucket[i]
                for j in range(i + 1, n):
                    ev_b = bucket[j]
                    if ev_a is ev_b:
                        continue
                    if ev_a.grid_idx == ev_b.grid_idx:
                        # Intra-block is po-ordered — not a race.
                        continue
                    key = (
                        epoch,
                        min(ev_a.event_id, ev_b.event_id),
                        max(ev_a.event_id, ev_b.event_id),
                    )
                    if key in seen_pairs:
                        continue
                    first, second = _canonical_pair(ev_a, ev_b)
                    race_type = _classify_candidate(first, second, byte_addr)
                    if race_type is None:
                        continue
                    seen_pairs.add(key)
                    candidates.append(
                        RaceCandidate(
                            first=first,
                            second=second,
                            race_type=race_type,
                            witness_addr=byte_addr,
                            epoch=epoch,
                        )
                    )
        return candidates

    # ── Barrier detection & epoch partitioning (Step 6) ───────────────────

    def _detect_global_barrier_addresses(
        self, events: list[AccessEventRecord], total_blocks: int
    ) -> set[int]:
        """Return element base addresses that qualify as global barrier keys.

        Conservative — an address qualifies only when:
          * every event on it passes ``_is_barrier_atomic``;
          * the set of distinct ``grid_idx`` hitting it equals
            ``total_blocks`` (every block participates);
          * ``total_blocks >= 2``.

        Over-identifying barrier keys splits phases wrongly and drops real
        races, so the bar is high — a single non-barrier atomic on the
        same address disqualifies it.
        """
        if total_blocks < 2:
            return set()
        # atomic_events_by_addr[addr] = (grids, all_barrier)
        by_addr: dict[int, tuple[set, bool]] = {}
        for ev in events:
            if ev.atomic_op is None or ev.grid_idx is None:
                continue
            is_barrier = _is_barrier_atomic(ev)
            for base in _iter_element_bases(ev):
                grids, all_barrier = by_addr.get(base, (set(), True))
                grids.add(ev.grid_idx)
                all_barrier = all_barrier and is_barrier
                by_addr[base] = (grids, all_barrier)
        result: set[int] = set()
        for addr, (grids, all_barrier) in by_addr.items():
            if all_barrier and len(grids) == total_blocks:
                result.add(addr)
        return result

    def _assign_concrete_epochs(
        self, events: list[AccessEventRecord], barrier_addrs: set[int]
    ) -> None:
        """Walk each grid_idx in program_seq order, bumping the epoch when
        the current event is a phase-advancing write on a barrier address.

        The barrier event itself stays in the pre-barrier epoch;
        subsequent events in the same grid get the next epoch. If no
        barrier addresses were identified, every event stays in epoch 0.
        """
        if not barrier_addrs:
            for ev in events:
                ev.epoch = 0
            return

        by_grid: dict[Any, list[AccessEventRecord]] = {}
        for ev in events:
            by_grid.setdefault(ev.grid_idx, []).append(ev)

        for grid_events in by_grid.values():
            grid_events.sort(
                key=lambda e: (
                    e.program_seq if e.program_seq is not None else -1,
                    e.event_id,
                )
            )
            epoch = 0
            for ev in grid_events:
                ev.epoch = epoch
                # Check every active element base of this event.
                for base in _iter_element_bases(ev):
                    if base not in barrier_addrs:
                        continue
                    if _is_phase_advancing_write(ev, base):
                        epoch += 1
                        break

    def _candidate_to_report(
        self, cand: RaceCandidate, hb_reason: str | None = None
    ) -> RaceReport:
        """Flatten a RaceCandidate into the public RaceReport shape. HB
        reason is None until Step 5 (HBSolver) is wired in the next commit."""
        tensor_name = cand.first.tensor_name or cand.second.tensor_name
        return RaceReport(
            race_type=cand.race_type,
            witness_addr=cand.witness_addr,
            epoch=cand.epoch,
            tensor_name=tensor_name,
            grid_a=cand.first.grid_idx,
            grid_b=cand.second.grid_idx,
            source_a=cand.first.source_location,
            source_b=cand.second.source_location,
            hb_reason=hb_reason,
        )


class NullRaceDetector(NullSymbolicClient, RaceDetector):
    """A do-nothing object returned when the race-detector backend is 'off'.
    Every callback raises via ``NullSymbolicClient`` so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)
