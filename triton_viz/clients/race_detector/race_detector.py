from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)

from z3.z3 import BoolRef

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    AtomicCas,
    AtomicRMW,
    Load,
)
from ..symbolic_engine import (
    SymbolicExpr,
    AtomicCasSymbolicExpr,
    AtomicRmwSymbolicExpr,
    SymbolicClient,
    NullSymbolicClient,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
    AccessMode,
)
from .data import AccessEventRecord, MemorySem
from .hb_common import (
    UnsupportedSymbolicRaceQuery,
    normalize_copy_local_vars,
)
from .two_copy_symbolic_hb_solver import TwoCopySymbolicHBSolver
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
        self._event_seq: int = 0
        self._launch_grid: tuple[int, int, int] = (1, 1, 1)
        self._captured_symbolic_template: bool = False
        self._unsupported_capture: bool = False
        self._arange_dict_snapshot: dict[Any, Any] = {}

    # Explicit forwarders to SymbolicClient: the RaceDetector factory
    # carries concrete stubs (NotImplementedError or ``return True``) to
    # satisfy Client's @abstractmethod contract, and those stubs would
    # otherwise shadow SymbolicClient's impls in the subclass MRO.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicClient.grid_idx_callback(self, grid_idx)
        # Capture is one-shot: program_seq spans a single symbolic pass over
        # all records, so we only reset it on grid_callback, not per block.

    def finalize(self) -> list:
        if not self._captured_symbolic_template or self._unsupported_capture:
            self.last_reports = []
            self._clear_launch_runtime()
            return []
        try:
            reports = TwoCopySymbolicHBSolver(
                self.records,
                grid=self._launch_grid,
                arange_dict=self._arange_dict_snapshot,
            ).find_races()
        except UnsupportedSymbolicRaceQuery:
            if self.abort_on_error:
                raise
            reports = []  # NO concrete fallback
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
        self._event_seq = 0
        normalized = tuple(int(dim) for dim in grid)
        while len(normalized) < 3:
            normalized = normalized + (1,)
        self._launch_grid = cast(tuple[int, int, int], normalized[:3])
        self._captured_symbolic_template = False
        self._unsupported_capture = False
        self._arange_dict_snapshot = {}
        SymbolicExpr.ARANGE_DICT.clear()
        SymbolicClient.grid_callback(self, grid)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        return SymbolicClient.register_op_callback(self, op_type)

    def pre_run_callback(self, fn: Callable) -> bool:
        # One-shot capture: capture symbolic templates from a single
        # representative block; the two-copy solver reasons over all blocks.
        return not self._captured_symbolic_template

    def post_run_callback(self, fn: Callable) -> bool:
        self._unsupported_capture = False
        try:
            self._force_eval_record_templates()
        except UnsupportedSymbolicRaceQuery:
            if self.abort_on_error:
                raise
            self._unsupported_capture = True
            self.records = []  # do not feed half-baked records to solver
        # Snapshot ARANGE_DICT after templates are evaluated so the two-copy
        # solver's arange substitutions are independent of subsequent launches.
        self._arange_dict_snapshot = dict(SymbolicExpr.ARANGE_DICT)
        self._captured_symbolic_template = True
        return False

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
        self._event_seq = 0

    @staticmethod
    def _normalize_constraints(
        constraints: ConstraintConjunction,
    ) -> tuple[Any, ...]:
        if constraints is None:
            return ()
        if isinstance(constraints, (list, tuple)):
            return tuple(constraints)
        return (constraints,)

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

    def _next_event_id(self) -> int:
        seq = self._event_seq
        self._event_seq += 1
        return seq

    @staticmethod
    def _infer_elem_size(expr: SymbolicExpr | None) -> int:
        """Best-effort byte-width of an access's element type.

        Falls back to 1 when the dtype isn't introspectable; the solver's
        byte-overlap predicate then degenerates to ``addr ==``, which is the
        same conservative behaviour as the previous single-copy solver.
        """
        if expr is None:
            return 1
        try:
            dtype = getattr(expr, "dtype", None)
            if dtype is None:
                return 1
            elem_ty = getattr(dtype, "element_ty", dtype)
            bw = getattr(elem_ty, "primitive_bitwidth", None)
            if bw is None:
                return 1
            return max(1, int(bw) // 8)
        except Exception:
            return 1

    def _current_loop_iter_vars(self) -> tuple[Any, ...]:
        return tuple(c.idx_z3 for c in self.loop_stack)

    def _force_eval_record_templates(self) -> None:
        """Ensure record template fields are Z3-ish, not unevaluated SymbolicExpr.

        Triggers ``.eval()`` on captured ``SymbolicExpr`` fields so the
        snapshotted ``ARANGE_DICT`` is complete before the solver consumes
        the records. Does NOT re-eval ``record.old_value`` /
        ``record.symbolic_expr`` themselves: re-evaluating an
        ``AtomicCasSymbolicExpr`` would not change anything thanks to caching,
        but we keep the rule explicit so downstream maintainers don't
        accidentally invalidate launch-level CAS-return identity.
        """

        def force(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (bool, int, str)):
                return value
            if isinstance(value, list):
                return [force(v) for v in value]
            if isinstance(value, tuple):
                return tuple(force(v) for v in value)
            if isinstance(value, SymbolicExpr):
                z3_value, _ = value.eval(simplify_constraints=False)
                return z3_value
            return value

        for record in self.records:
            try:
                record.addr_expr = force(record.addr_expr)
                record.local_constraints = self._normalize_constraints(
                    force(record.local_constraints)
                )
                record.premises = self._normalize_constraints(force(record.premises))
                if record.cas_cmp_value is not None:
                    record.cas_cmp_value = force(record.cas_cmp_value)
                if record.cas_new_value is not None:
                    record.cas_new_value = force(record.cas_new_value)
            except Exception as exc:  # pragma: no cover - defensive
                raise UnsupportedSymbolicRaceQuery(
                    f"failed to normalize record templates: {exc}"
                ) from exc

    def _record_access_event(
        self,
        access_mode: AccessMode,
        op_type: type[Op],
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
        *,
        semantic_constraints: tuple[Any, ...] = (),
        copy_local_vars: tuple[Any, ...] = (),
    ) -> None:
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        # Two-copy capture: keep raw symbolic templates (PID0/1/2 preserved)
        # rather than snapshotting solver assertions, which can carry sampled
        # PID equalities that pin pid_a == pid_b == sampled_pid after alpha-
        # renaming and break the two-copy alias query.
        local = self._normalize_constraints(expr_constraints)
        premises = self._normalize_constraints(semantic_constraints)

        self.records.append(
            AccessEventRecord(
                op_type=op_type,
                access_mode=access_mode,
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=access_addr,
                premises=premises,
                local_constraints=local,
                source_location=source_location,
                grid_idx=None,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(op_type, source_location),
                active=True,
                reads=access_mode == "read",
                writes=access_mode == "write",
                event_id=self._next_event_id(),
                elem_size=self._infer_elem_size(symbolic_expr),
                copy_local_vars=normalize_copy_local_vars(copy_local_vars),
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
        *,
        semantic_constraints: tuple[Any, ...] = (),
    ) -> None:
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        local = self._normalize_constraints(expr_constraints)
        premises = self._normalize_constraints(semantic_constraints)
        loop_vars = self._current_loop_iter_vars()

        # Raw symbolic templates: writes / written_value are recomputed by the
        # two-copy solver per copy from cas_cmp_value / cas_new_value /
        # old_value. Storing them here as None keeps the per-copy CAS return
        # rename in lockstep with the substitution applied to old_value.
        self.records.append(
            AccessEventRecord(
                op_type=AtomicCas,
                access_mode="read",
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=addr_expr,
                premises=premises,
                local_constraints=local,
                source_location=source_location,
                grid_idx=None,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(AtomicCas, source_location),
                active=True,
                reads=True,
                writes=None,
                is_atomic=True,
                atomic_kind="cas",
                sem=self._normalize_sem(sem),
                scope=self._normalize_scope(scope),
                old_value=old_value,
                written_value=None,
                event_id=self._next_event_id(),
                elem_size=self._infer_elem_size(symbolic_expr),
                cas_cmp_value=cmp_value,
                cas_new_value=value,
                copy_local_vars=normalize_copy_local_vars((old_value,) + loop_vars),
            )
        )

    def _record_atomic_rmw_event(
        self,
        symbolic_expr: SymbolicExpr,
        addr_expr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        sem: str | None,
        scope: str | None,
        source_location: tuple[str, int, str] | None = None,
        *,
        semantic_constraints: tuple[Any, ...] = (),
    ) -> None:
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        local = self._normalize_constraints(expr_constraints)
        premises = self._normalize_constraints(semantic_constraints)
        loop_vars = self._current_loop_iter_vars()

        self.records.append(
            AccessEventRecord(
                op_type=AtomicRMW,
                access_mode="read",
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=addr_expr,
                premises=premises,
                local_constraints=local,
                source_location=source_location,
                grid_idx=None,
                program_seq=self._next_program_seq(),
                debug_name=self._debug_name(AtomicRMW, source_location),
                active=True,
                reads=True,
                writes=True,  # RMW always writes when active
                is_atomic=True,
                atomic_kind="rmw",
                sem=self._normalize_sem(sem),
                scope=self._normalize_scope(scope),
                old_value=None,
                written_value=None,
                event_id=self._next_event_id(),
                elem_size=self._infer_elem_size(symbolic_expr),
                cas_cmp_value=None,
                cas_new_value=None,
                copy_local_vars=normalize_copy_local_vars(loop_vars),
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

    def _handle_atomic_rmw_check(
        self,
        expr: SymbolicExpr,
        sem: str | None,
        scope: str | None,
    ) -> None:
        expr_rmw = cast(AtomicRmwSymbolicExpr, expr)
        addr_expr, addr_constraints = expr_rmw.ptr.eval()
        # AtomicRmw doesn't define its own _to_z3_impl; we don't need a Z3
        # value for the access itself, only its address and any pointer-side
        # constraints. The RMW captures reads=True / writes=True regardless.
        source_location = capture_current_source_location()

        if self.loop_stack and cfg.verbose:
            print(
                f"[{self.LOG_TAG}] atomic_rmw inside loops is recorded eagerly; "
                "loop dedupe is not supported yet"
            )

        self._record_atomic_rmw_event(
            symbolic_expr=expr,
            addr_expr=addr_expr,
            expr_constraints=addr_constraints,
            sem=sem,
            scope=scope,
            source_location=source_location,
        )

    def _op_atomic_rmw_overrider(
        self,
        rmwOp: Any,
        ptr: Any,
        val: Any,
        mask: Any,
        sem: str | None = None,
        scope: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> SymbolicExpr:
        ptr_sym = SymbolicExpr.from_value(ptr)
        val_sym = SymbolicExpr.from_value(val)
        mask_sym = SymbolicExpr.from_value(mask)
        ret = SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)
        self._handle_atomic_rmw_check(ret, sem=sem, scope=scope)
        return ret

    # ── Per-pending handler invoked from SymbolicClient's loop template

    def _process_pending_check(
        self,
        ctx: LoopContext,
        pending: PendingCheck,
        iter_constraints: list[BoolRef],
    ) -> None:
        del ctx
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
            semantic_constraints=tuple(iter_constraints),
            copy_local_vars=self._current_loop_iter_vars(),
        )


class NullRaceDetector(NullSymbolicClient, RaceDetector):
    """A do-nothing object returned when the race-detector backend is 'off'.
    Every callback raises via ``NullSymbolicClient`` so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)
