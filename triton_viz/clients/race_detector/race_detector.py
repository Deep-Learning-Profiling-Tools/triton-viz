import threading
from collections import defaultdict, deque
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

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Load,
    AtomicCas,
    AtomicRMW,
)
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
    PendingAtomicEvent,
    VALID_RMW_OPS,
    active_mask_for,
    apply_rmw,
    broadcast_lane_operand,
    flatten_np,
    infer_elem_size,
    normalize_sem_scope,
    resolve_tensor_from_pointer,
)
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
        # symbolic path so pre-existing tests (test_race_detector.py) keep
        # working. Commit 2 adds the sibling ``concrete_events`` list for
        # the atomic path.
        self.records = self.symbolic_events
        self._next_event_id = 0

        # PR2 concrete atomic path
        self.concrete_events: list[AccessEventRecord] = []
        self._pending_atomic_by_grid: defaultdict[
            tuple[int, ...], deque[PendingAtomicEvent]
        ] = defaultdict(deque)
        self._pending_atomic_lock = threading.Lock()

        # Tripped by any atomic capture. Step 4/5 consumers must fall back
        # to concrete-only reasoning (or conservatively not suppress) when
        # True. Reset at the START of the next launch (grid_callback), not
        # at finalize — a consumer inspecting the detector immediately
        # after the launch completes must still see True.
        self.atomic_symbolic_escape: bool = False

    def _new_event_id(self) -> int:
        """Client-lifetime monotonic unique ID. NOT an execution-order
        indicator under multi-SM concurrency — blocks race to append, so
        append order != execution order. Consumers that need execution
        order must key on grid_idx + queue position, not event_id."""
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
        with self._pending_atomic_lock:
            dangling = {k: len(v) for k, v in self._pending_atomic_by_grid.items() if v}
            self._pending_atomic_by_grid.clear()
        if dangling:
            raise RuntimeError(
                f"Dangling pending atomic events at finalize: {dangling}. "
                "A before_atomic_* callback enqueued without a matching after_*."
            )
        return SymbolicClient.finalize(self)

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        SymbolicClient.arg_callback(self, name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        # Per-launch reset BEFORE the base sets up its solver/premises.
        # Defensive clear in case a prior launch errored out of finalize()
        # mid-assertion.
        self.atomic_symbolic_escape = False
        with self._pending_atomic_lock:
            self._pending_atomic_by_grid.clear()
        SymbolicClient.grid_callback(self, grid)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        if op_type is AtomicCas:
            return OpCallbacks(
                before_callback=self.lock_fn(self._before_atomic_cas),
                after_callback=self.lock_fn(self._after_atomic_cas),
                op_overrider=None,  # required — PatchOp feeds overrider ret to after
            )
        if op_type is AtomicRMW:
            return OpCallbacks(
                before_callback=self.lock_fn(self._before_atomic_rmw),
                after_callback=self.lock_fn(self._after_atomic_rmw),
                op_overrider=None,
            )
        return SymbolicClient.register_op_callback(self, op_type)

    def pre_run_callback(self, fn: Callable) -> bool:
        return SymbolicClient.pre_run_callback(self, fn)

    def post_run_callback(self, fn: Callable) -> bool:
        return SymbolicClient.post_run_callback(self, fn)

    # ── Event recording ───────────────────────────────────────────────────

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

    # ── Concrete atomic capture (before/after callback pairs) ────────────
    #
    # PatchOp.__call__ runs before_callback → (overrider | original op) →
    # after_callback. We register op_overrider=None so the original hardware
    # atomic runs; after_callback then sees the real ``old`` value from the
    # op's return. An overrider here would feed its (symbolic) return into
    # after_callback, and we'd lose the hardware old.
    #
    # Side effect: the symbolic overrider for atomic_cas/atomic_rmw (defined
    # on SymbolicClient) is bypassed for race_detector, so any post-atomic
    # code that uses the atomic's return value to compute addresses/masks/
    # control flow loses symbolic fidelity past that point. We track this
    # via ``atomic_symbolic_escape`` — Step 4/5 consumers must gate on it.

    def _before_atomic_cas(
        self,
        ptr,
        cmp,
        val,
        sem=None,
        scope=None,
        mask=None,
        **_kwargs,
    ) -> None:
        self.atomic_symbolic_escape = True
        grid_idx = self.grid_idx
        assert grid_idx is not None, "atomic callback fired before grid_idx_callback"
        lane_addrs = flatten_np(ptr).astype(np.int64, copy=False)
        nlanes = lane_addrs.shape[0]
        cmp_np = broadcast_lane_operand(cmp, nlanes)
        val_np = broadcast_lane_operand(val, nlanes)
        active = active_mask_for(mask, nlanes)
        sem_norm, scope_norm = normalize_sem_scope(sem, scope)
        elem_size = infer_elem_size(val, ptr)
        tensor = resolve_tensor_from_pointer(ptr, active, elem_size, self.tensor_addrs)
        pending = PendingAtomicEvent(
            event_id=self._new_event_id(),
            op_type=AtomicCas,
            atomic_op="cas",
            grid_idx=grid_idx,
            source_location=capture_current_source_location(),
            tensor=tensor,
            tensor_name=(self._get_tensor_name(tensor) if tensor is not None else None),
            lane_addrs=lane_addrs,
            active_mask=active,
            elem_size=elem_size,
            atomic_sem=sem_norm,
            atomic_scope=scope_norm,
            atomic_cmp=cmp_np,
            atomic_val=val_np,
        )
        with self._pending_atomic_lock:
            self._pending_atomic_by_grid[grid_idx].append(pending)

    def _after_atomic_cas(
        self,
        ret,
        ptr,
        cmp,
        val,
        sem=None,
        scope=None,
        mask=None,
        **_kwargs,
    ) -> None:
        del ptr, cmp, val, sem, scope, mask  # all already snapshotted in pending
        grid_idx = self.grid_idx
        assert grid_idx is not None, "atomic callback fired before grid_idx_callback"
        with self._pending_atomic_lock:
            pending = self._pending_atomic_by_grid[grid_idx].popleft()
        old_np = flatten_np(ret)
        success = pending.active_mask & np.equal(old_np, pending.atomic_cmp)
        read_mask = pending.active_mask.copy()
        write_mask = success
        written_value = np.where(success, pending.atomic_val, old_np).astype(
            old_np.dtype, copy=False
        )
        self.concrete_events.append(
            AccessEventRecord(
                event_id=pending.event_id,
                op_type=pending.op_type,
                source_location=pending.source_location,
                grid_idx=pending.grid_idx,
                tensor=pending.tensor,
                tensor_name=pending.tensor_name,
                lane_addrs=pending.lane_addrs,
                active_mask=pending.active_mask,
                elem_size=pending.elem_size,
                read_mask=read_mask,
                write_mask=write_mask,
                atomic_op="cas",
                atomic_sem=pending.atomic_sem,
                atomic_scope=pending.atomic_scope,
                atomic_cmp=pending.atomic_cmp,
                atomic_val=pending.atomic_val,
                atomic_old=old_np,
                written_value=written_value,
            )
        )

    def _before_atomic_rmw(
        self,
        rmwOp,
        ptr,
        val,
        mask=None,
        sem=None,
        scope=None,
        **_kwargs,
    ) -> None:
        self.atomic_symbolic_escape = True
        grid_idx = self.grid_idx
        assert grid_idx is not None, "atomic callback fired before grid_idx_callback"
        atomic_op = _normalize_rmw_op(rmwOp)
        lane_addrs = flatten_np(ptr).astype(np.int64, copy=False)
        nlanes = lane_addrs.shape[0]
        val_np = broadcast_lane_operand(val, nlanes)
        active = active_mask_for(mask, nlanes)
        sem_norm, scope_norm = normalize_sem_scope(sem, scope)
        elem_size = infer_elem_size(val, ptr)
        tensor = resolve_tensor_from_pointer(ptr, active, elem_size, self.tensor_addrs)
        pending = PendingAtomicEvent(
            event_id=self._new_event_id(),
            op_type=AtomicRMW,
            atomic_op=atomic_op,
            grid_idx=grid_idx,
            source_location=capture_current_source_location(),
            tensor=tensor,
            tensor_name=(self._get_tensor_name(tensor) if tensor is not None else None),
            lane_addrs=lane_addrs,
            active_mask=active,
            elem_size=elem_size,
            atomic_sem=sem_norm,
            atomic_scope=scope_norm,
            atomic_cmp=None,
            atomic_val=val_np,
        )
        with self._pending_atomic_lock:
            self._pending_atomic_by_grid[grid_idx].append(pending)

    def _after_atomic_rmw(
        self,
        ret,
        rmwOp,
        ptr,
        val,
        mask=None,
        sem=None,
        scope=None,
        **_kwargs,
    ) -> None:
        del rmwOp, ptr, val, mask, sem, scope  # already snapshotted in pending
        grid_idx = self.grid_idx
        assert grid_idx is not None, "atomic callback fired before grid_idx_callback"
        with self._pending_atomic_lock:
            pending = self._pending_atomic_by_grid[grid_idx].popleft()
        old_np = flatten_np(ret)
        read_mask = pending.active_mask.copy()
        write_mask = pending.active_mask.copy()
        written_value = apply_rmw(pending.atomic_op, old_np, pending.atomic_val)
        self.concrete_events.append(
            AccessEventRecord(
                event_id=pending.event_id,
                op_type=pending.op_type,
                source_location=pending.source_location,
                grid_idx=pending.grid_idx,
                tensor=pending.tensor,
                tensor_name=pending.tensor_name,
                lane_addrs=pending.lane_addrs,
                active_mask=pending.active_mask,
                elem_size=pending.elem_size,
                read_mask=read_mask,
                write_mask=write_mask,
                atomic_op=pending.atomic_op,
                atomic_sem=pending.atomic_sem,
                atomic_scope=pending.atomic_scope,
                atomic_cmp=None,
                atomic_val=pending.atomic_val,
                atomic_old=old_np,
                written_value=written_value,
            )
        )


class NullRaceDetector(NullSymbolicClient, RaceDetector):
    """A do-nothing object returned when the race-detector backend is 'off'.
    Every callback raises via ``NullSymbolicClient`` so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)
