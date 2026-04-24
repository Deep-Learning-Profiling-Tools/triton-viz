from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)

from z3 import If, IntVal, substitute
from z3.z3 import BoolRef

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    AtomicCas,
    Load,
)
from ..symbolic_engine import (
    SymbolicExpr,
    AtomicCasSymbolicExpr,
    SymbolicClient,
    NullSymbolicClient,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
    AccessMode,
)
from .data import AccessEventRecord, MemorySem
from .hb_solver import HBSolver
from ...utils.traceback_utils import capture_current_source_location
from ...core.config import config as cfg

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
        self.records: list[AccessEventRecord] = []
        self.last_reports: list[Any] = []
        self._program_seq: int = 0
        self._expected_blocks: int = 0
        self._completed_blocks: int = 0

    # Explicit forwarders to SymbolicClient: the RaceDetector factory
    # carries concrete stubs (NotImplementedError or ``return True``) to
    # satisfy Client's @abstractmethod contract, and those stubs would
    # otherwise shadow SymbolicClient's impls in the subclass MRO.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicClient.grid_idx_callback(self, grid_idx)
        self._program_seq = 0

    def finalize(self) -> list:
        reports = HBSolver(self.records).find_races()
        self.last_reports = reports
        self._clear_launch_runtime()
        return reports

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        SymbolicClient.arg_callback(self, name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self.records = []
        self.last_reports = []
        self._program_seq = 0
        self._expected_blocks = prod(int(dim) for dim in grid)
        self._completed_blocks = 0
        SymbolicClient.grid_callback(self, grid)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        return SymbolicClient.register_op_callback(self, op_type)

    def pre_run_callback(self, fn: Callable) -> bool:
        # v1 guarantees standalone race-detector semantics only: always run the
        # full launch grid instead of relying on SymbolicClient's lazy sampling.
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        self._completed_blocks += 1
        return True

    # ── Event recording ───────────────────────────────────────────────────

    def _clear_launch_runtime(self) -> None:
        self._clear_cache()
        self._clear_symbolic_launch_state()
        self.need_full_grid = None
        self.solver = None
        self.addr_ok = None
        self.pid_ok = None
        self.addr_sym = None
        self.grid = None
        self.grid_idx = None
        self.last_grid = None
        self._program_seq = 0
        self._expected_blocks = 0
        self._completed_blocks = 0

    @staticmethod
    def _normalize_constraints(
        constraints: ConstraintConjunction,
    ) -> tuple[Any, ...]:
        if constraints is None:
            return ()
        if isinstance(constraints, (list, tuple)):
            return tuple(constraints)
        return (constraints,)

    def _pid_substitutions(self) -> tuple[tuple[Any, Any], ...]:
        if self.grid_idx is None:
            return ()
        return (
            (SymbolicExpr.PID0, IntVal(int(self.grid_idx[0]))),
            (SymbolicExpr.PID1, IntVal(int(self.grid_idx[1]))),
            (SymbolicExpr.PID2, IntVal(int(self.grid_idx[2]))),
        )

    def _concretize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, list):
            return [self._concretize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._concretize_value(item) for item in value)

        substitutions = self._pid_substitutions()
        if not substitutions:
            return value
        return substitute(value, *substitutions)

    @staticmethod
    def _debug_name(
        op_type: type[Op],
        source_location: tuple[str, int, str] | None,
    ) -> str:
        base = getattr(op_type, "name", op_type.__name__.lower())
        if source_location is None:
            return base
        _, lineno, func_name = source_location
        if func_name:
            return f"{func_name}:{lineno}:{base}"
        return f"{base}:{lineno}"

    def _next_program_seq(self) -> int:
        seq = self._program_seq
        self._program_seq += 1
        return seq

    @staticmethod
    def _zip_lanes(*values: Any) -> list[tuple[Any, ...]] | None:
        lane_values = [value for value in values if isinstance(value, (list, tuple))]
        if not lane_values:
            return None
        lane_count = len(lane_values[0])
        if any(len(value) != lane_count for value in lane_values):
            raise ValueError("Lane-wise atomic_cas values must have matching lengths")

        def lane_value(value: Any, lane: int) -> Any:
            if isinstance(value, (list, tuple)):
                return value[lane]
            return value

        return [
            tuple(lane_value(value, lane) for value in values)
            for lane in range(lane_count)
        ]

    @classmethod
    def _eq_by_lane(cls, lhs: Any, rhs: Any) -> Any:
        lanes = cls._zip_lanes(lhs, rhs)
        if lanes is None:
            return lhs == rhs
        return [left == right for left, right in lanes]

    @classmethod
    def _if_by_lane(cls, cond: Any, on_true: Any, on_false: Any) -> Any:
        lanes = cls._zip_lanes(cond, on_true, on_false)
        if lanes is None:
            return If(cond, on_true, on_false)
        return [If(c, t, f) for c, t, f in lanes]

    def _record_access_event(
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
        local = self._normalize_constraints(expr_constraints)
        access_addr = self._concretize_value(access_addr)
        solver_snapshot = tuple(
            self._concretize_value(item) for item in solver_snapshot
        )
        local = tuple(self._concretize_value(item) for item in local)

        self.records.append(
            AccessEventRecord(
                op_type=op_type,
                access_mode=access_mode,
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=access_addr,
                premises=solver_snapshot,
                local_constraints=local,
                source_location=source_location,
                grid_idx=self.grid_idx,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(op_type, source_location),
                active=True,
                reads=access_mode == "read",
                writes=access_mode == "write",
            )
        )

    @staticmethod
    def _normalize_sem(sem: str | None) -> MemorySem:
        if sem is None:
            return "acq_rel"
        name = getattr(sem, "name", sem)
        normalized = str(name).lower()
        if normalized == "relaxed":
            return "relaxed"
        if normalized == "acquire":
            return "acquire"
        if normalized == "release":
            return "release"
        if normalized in ("acquire_release", "acq_rel"):
            return "acq_rel"
        if normalized == "plain":
            return "plain"
        return cast(MemorySem, normalized)

    @staticmethod
    def _normalize_scope(scope: str | None) -> str:
        if scope is None:
            return "gpu"
        name = getattr(scope, "name", scope)
        normalized = str(name).lower()
        return {
            "gpu": "gpu",
            "cta": "cta",
            "system": "sys",
            "sys": "sys",
        }.get(normalized, normalized)

    def _record_atomic_cas_event(
        self,
        symbolic_expr: SymbolicExpr,
        addr_expr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        cmp_value: Any,
        value: Any,
        old_value: Any,
        sem: str | None,
        scope: str | None,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None
        solver_snapshot: tuple[Any, ...] = (
            tuple(self.solver.assertions()) if self.solver is not None else ()
        )
        local = self._normalize_constraints(expr_constraints)

        addr_expr = self._concretize_value(addr_expr)
        cmp_value = self._concretize_value(cmp_value)
        value = self._concretize_value(value)
        old_value = self._concretize_value(old_value)
        solver_snapshot = tuple(
            self._concretize_value(item) for item in solver_snapshot
        )
        local = tuple(self._concretize_value(item) for item in local)

        success = self._eq_by_lane(old_value, cmp_value)
        written_value = self._if_by_lane(success, value, old_value)

        self.records.append(
            AccessEventRecord(
                op_type=AtomicCas,
                access_mode="read",
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=addr_expr,
                premises=solver_snapshot,
                local_constraints=local,
                source_location=source_location,
                grid_idx=self.grid_idx,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(AtomicCas, source_location),
                active=True,
                reads=True,
                writes=success,
                is_atomic=True,
                atomic_kind="cas",
                sem=self._normalize_sem(sem),
                scope=self._normalize_scope(scope),
                old_value=old_value,
                written_value=written_value,
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
            self._record_access_event(
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

    def _handle_atomic_cas_check(
        self,
        expr: SymbolicExpr,
        sem: str | None,
        scope: str | None,
    ) -> None:
        expr_atomic = cast(AtomicCasSymbolicExpr, expr)
        old_value, expr_constraints = expr.eval()
        addr_expr, _ = expr_atomic.ptr.eval()
        cmp_value, _ = expr_atomic.cmp.eval()
        value, _ = expr_atomic.val.eval()
        source_location = capture_current_source_location()

        if self.loop_stack and cfg.verbose:
            print(
                f"[{self.LOG_TAG}] atomic_cas inside loops is recorded eagerly; "
                "loop dedupe is not supported yet"
            )

        self._record_atomic_cas_event(
            symbolic_expr=expr,
            addr_expr=addr_expr,
            expr_constraints=expr_constraints,
            cmp_value=cmp_value,
            value=value,
            old_value=old_value,
            sem=sem,
            scope=scope,
            source_location=source_location,
        )

    def _op_atomic_cas_overrider(
        self,
        ptr: Any,
        cmp: Any,
        val: Any,
        sem: str | None = None,
        scope: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SymbolicExpr:
        ptr_sym = SymbolicExpr.from_value(ptr)
        cmp_sym = SymbolicExpr.from_value(cmp)
        val_sym = SymbolicExpr.from_value(val)
        ret = SymbolicExpr.create("atomic_cas", ptr_sym, cmp_sym, val_sym)
        self._handle_atomic_cas_check(ret, sem=sem, scope=scope)
        return ret

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
        self._record_access_event(
            pending.access_mode,
            pending.op_type,
            pending.addr_expr,
            pending.constraints,
            pending.symbolic_expr,
            pending.source_location,
        )


class NullRaceDetector(NullSymbolicClient, RaceDetector):
    """A do-nothing object returned when the race-detector backend is 'off'.
    Every callback raises via ``NullSymbolicClient`` so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)
