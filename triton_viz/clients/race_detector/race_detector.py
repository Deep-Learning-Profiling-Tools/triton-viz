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

    # Surface the public detector interface as class-level annotations so
    # callers (e.g. example scripts) and static type-checkers can read these
    # off the factory base without downcasting to a concrete subclass.
    # Concrete subclasses populate these public attributes at runtime.
    #
    # ``last_status`` values:
    #   "ok"          — solver ran (last_reports holds the verdict)
    #   "unsupported" — a feature the solver doesn't model fired during
    #                   capture (atomic-in-loop, RMW return downstream,
    #                   data-dependent address, etc.); see unsupported_reason
    #   "disabled"    — race detector backend is off (NullRaceDetector)
    last_reports: list[Any]
    last_status: str
    unsupported_reason: str | None

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


class _UnsupportedRMWReturn(SymbolicExpr):
    """Sentinel SymbolicExpr returned by ``SymbolicRaceDetector`` for an
    atomic-RMW result. The RMW return value's symbolic semantics are not
    modeled by the two-copy solver; if a kernel consumes the return
    downstream (e.g. ``mask = old == 0``), the eventual ``_to_z3_impl``
    call raises :class:`UnsupportedSymbolicRaceQuery`, which the wrapping
    ``_safe_eval`` in ``_handle_*_check`` converts into a clean
    ``_mark_unsupported`` so the launch finishes without raising.

    The op string MUST be ``"atomic_rmw"`` because
    ``SymbolicExpr.__init__`` asserts ``op in self.SUPPORTED_OPS``;
    coining a new op here would assert-fail at construction time.
    """

    def __init__(self, *, dtype: Any = None, shape: Any = ()) -> None:
        super().__init__("atomic_rmw")
        self.dtype = dtype
        self.shape = shape

    def _to_z3_impl(self) -> tuple[Any, Any]:
        raise UnsupportedSymbolicRaceQuery(
            "atomic_rmw return value used downstream is not modeled"
        )


class SymbolicRaceDetector(RaceDetector, SymbolicClient):
    def __init__(self, abort_on_error: bool = False):
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[AccessEventRecord] = []
        self.last_reports: list[Any] = []
        # Status of the most recent finalize(): "ok" means the solver ran;
        # "unsupported" means the launch hit a feature the solver doesn't
        # model (atomic-in-loop, RMW return downstream, data-dependent
        # address, etc.). last_reports being empty does NOT imply "no race"
        # unless last_status == "ok".
        self.last_status: str = "ok"
        self._program_seq: int = 0
        self._event_seq: int = 0
        self._launch_grid: tuple[int, int, int] = (1, 1, 1)
        self._captured_symbolic_template: bool = False
        self._unsupported_capture: bool = False
        self.unsupported_reason: str | None = None
        self._arange_dict_snapshot: dict[Any, Any] = {}

    # ── Unsupported-launch plumbing ──────────────────────────────────────

    def _mark_unsupported(self, reason: str) -> None:
        """Mark the current launch as unsupported by the two-copy solver.

        Discards any partial records so finalize() can't accidentally feed
        them to the solver. Callers MUST return after invoking this — kernel
        tracing may still execute, and the early-return guards in
        ``_handle_*_check`` / ``_record_*_event`` keep further events from
        leaking back into ``self.records``.
        """
        self._unsupported_capture = True
        self.unsupported_reason = reason
        self.last_status = "unsupported"
        self.records = []

    def _safe_eval(self, expr: "SymbolicExpr", reason: str) -> tuple[Any, Any] | None:
        """Eval a SymbolicExpr, marking the launch unsupported on
        :class:`UnsupportedSymbolicRaceQuery`. Returns ``None`` when
        unsupported so callers can ``if result is None: return``.
        """
        try:
            return expr.eval()
        except UnsupportedSymbolicRaceQuery as exc:
            if self.abort_on_error:
                raise
            self._mark_unsupported(str(exc) or reason)
            return None

    @staticmethod
    def _combine_constraints(*constraints: Any) -> tuple[Any, ...]:
        """Flat tuple of non-None constraints; the two-copy solver's
        ``iter_constraints`` recursively flattens nested tuples/lists.
        """
        return tuple(c for c in constraints if c is not None)

    @staticmethod
    def _expr_contains_load(expr: SymbolicExpr | None) -> bool:
        """True when ``expr`` (typically a pointer expression) embeds
        ``tl.load``. Such expressions encode data-dependent addressing —
        scatter/histogram patterns where the destination index comes from a
        loaded value — which the current symbolic model conflates with the
        load's pointer rather than its loaded value. Flag these as
        unsupported until value semantics are properly modeled.
        """
        if expr is None:
            return False
        try:
            return bool(expr.has_op("load"))
        except Exception:
            return False

    def _reject_data_dependent_address(self, ptr_expr: SymbolicExpr | None) -> bool:
        """If ``ptr_expr`` depends on a loaded value, mark the launch
        unsupported (or raise under abort_on_error) and return True; callers
        should ``return`` immediately on True.
        """
        if not self._expr_contains_load(ptr_expr):
            return False
        reason = (
            "data-dependent memory address through tl.load is unsupported "
            "by the current symbolic race detector"
        )
        if self.abort_on_error:
            raise UnsupportedSymbolicRaceQuery(reason)
        self._mark_unsupported(reason)
        return True

    # Explicit forwarders to SymbolicClient: the RaceDetector factory
    # carries concrete stubs (NotImplementedError or ``return True``) to
    # satisfy Client's @abstractmethod contract, and those stubs would
    # otherwise shadow SymbolicClient's impls in the subclass MRO.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicClient.grid_idx_callback(self, grid_idx)
        # Capture is one-shot: program_seq spans a single symbolic pass over
        # all records, so we only reset it on grid_callback, not per block.

    def finalize(self) -> list:
        """Run the two-copy symbolic HB solver and return any detected races.

        Returns an empty list when the launch was marked unsupported during
        tracing (e.g. atomic CAS/RMW inside a loop, AtomicRMW return used
        downstream). Callers that need to distinguish "no race" from
        "unsupported" / "disabled" should read :attr:`last_status` and
        :attr:`unsupported_reason`. ``last_status == "disabled"`` is set by
        :class:`NullRaceDetector` when the backend is off.

        Limitations carried by the underlying ``TwoCopySymbolicHBSolver``:
          - Initial atomic source covers scalar tensors and small contiguous
            flag arrays (``numel <= 1024``); larger or non-contiguous
            tensors are conservatively reported as races.
          - Synchronization through a third program instance is not modeled.
          - AtomicRMW return value semantics are not modeled — downstream
            use of the return marks the launch unsupported.
          - Atomic CAS/RMW inside loops are not modeled — the launch is
            marked unsupported instead of recording phantom events.
        """
        if not self._captured_symbolic_template or self._unsupported_capture:
            if self._unsupported_capture and cfg.verbose:
                print(
                    f"[{self.LOG_TAG}] launch unsupported by two-copy solver: "
                    f"{self.unsupported_reason}"
                )
            self.last_reports = []
            self.last_status = "unsupported" if self._unsupported_capture else "ok"
            self._clear_launch_runtime()
            return []
        try:
            reports = TwoCopySymbolicHBSolver(
                self.records,
                grid=self._launch_grid,
                arange_dict=self._arange_dict_snapshot,
            ).find_races()
            self.last_status = "ok"
        except UnsupportedSymbolicRaceQuery as exc:
            if self.abort_on_error:
                raise
            self._mark_unsupported(str(exc))
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
        self.last_status = "ok"
        self._program_seq = 0
        self._event_seq = 0
        normalized = tuple(int(dim) for dim in grid)
        while len(normalized) < 3:
            normalized = normalized + (1,)
        self._launch_grid = cast(tuple[int, int, int], normalized[:3])
        self._captured_symbolic_template = False
        # Reset of unsupported state lives ONLY in grid_callback. post_run_callback
        # must NOT zero these — handlers within the same launch may have set them.
        self._unsupported_capture = False
        self.unsupported_reason = None
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
        # If a handler already marked the launch unsupported, don't try to
        # force-eval half-baked record state — short-circuit cleanly.
        if self._unsupported_capture:
            self._captured_symbolic_template = True
            return False
        try:
            self._force_eval_record_templates()
        except UnsupportedSymbolicRaceQuery as exc:
            if self.abort_on_error:
                raise
            self._mark_unsupported(str(exc))
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

        Prefer the pointer element type — store expressions historically
        didn't set their own dtype, so reading from the access dtype could
        silently degrade ``elem_size`` to 1 and degrade the solver's
        byte-overlap predicate to ``addr ==``. Fallback chain:
        ``expr.ptr.dtype`` → ``expr.dtype`` → ``expr.value.dtype`` → 1.
        """
        if expr is None:
            return 1

        def dtype_to_size(dtype: Any) -> int | None:
            if dtype is None:
                return None
            elem_ty = getattr(dtype, "element_ty", dtype)
            bw = getattr(elem_ty, "primitive_bitwidth", None)
            if bw is None:
                return None
            try:
                return max(1, int(bw) // 8)
            except Exception:
                return None

        try:
            ptr = getattr(expr, "ptr", None)
            size = dtype_to_size(getattr(ptr, "dtype", None))
            if size is not None:
                return size
            size = dtype_to_size(getattr(expr, "dtype", None))
            if size is not None:
                return size
            value = getattr(expr, "value", None)
            size = dtype_to_size(getattr(value, "dtype", None))
            if size is not None:
                return size
        except Exception:
            pass
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
        if self._unsupported_capture:
            return
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
        if self._unsupported_capture:
            return
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
        active: Any = True,
    ) -> None:
        if self._unsupported_capture:
            return
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
                active=active,
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
        if self._unsupported_capture:
            return
        # Reject scatter/histogram-style addressing where the pointer itself
        # depends on a loaded value — the current model conflates the load's
        # pointer with its loaded value.
        if self._reject_data_dependent_address(getattr(expr, "ptr", None)):
            return
        eval_result = self._safe_eval(expr, "load/store eval")
        if eval_result is None:
            return
        z3_addr, z3_constraints = eval_result
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
        if self._unsupported_capture:
            return
        # Loop check FIRST — before any .eval() can produce side effects
        # (ARANGE_DICT entries, fresh CAS-old vars, downstream sentinels).
        if self.loop_stack:
            if self.abort_on_error:
                raise UnsupportedSymbolicRaceQuery(
                    "atomic_cas inside loop is unsupported by the two-copy solver"
                )
            self._mark_unsupported(
                "atomic_cas inside loop is unsupported by the two-copy solver"
            )
            return

        expr_atomic = cast(AtomicCasSymbolicExpr, expr)
        if self._reject_data_dependent_address(expr_atomic.ptr):
            return
        result = self._safe_eval(expr, "atomic_cas eval")
        if result is None:
            return
        old_value, expr_constraints = result
        result = self._safe_eval(expr_atomic.ptr, "atomic_cas ptr eval")
        if result is None:
            return
        addr_expr, _ = result
        result = self._safe_eval(expr_atomic.cmp, "atomic_cas cmp eval")
        if result is None:
            return
        cmp_value, _ = result
        result = self._safe_eval(expr_atomic.val, "atomic_cas val eval")
        if result is None:
            return
        value, _ = result

        source_location = capture_current_source_location()
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
        if self._unsupported_capture:
            return
        # Loop check FIRST — see _handle_atomic_cas_check for rationale.
        if self.loop_stack:
            if self.abort_on_error:
                raise UnsupportedSymbolicRaceQuery(
                    "atomic_rmw inside loop is unsupported by the two-copy solver"
                )
            self._mark_unsupported(
                "atomic_rmw inside loop is unsupported by the two-copy solver"
            )
            return

        expr_rmw = cast(AtomicRmwSymbolicExpr, expr)
        if self._reject_data_dependent_address(expr_rmw.ptr):
            return
        ptr_result = self._safe_eval(expr_rmw.ptr, "atomic_rmw ptr eval")
        if ptr_result is None:
            return
        addr_expr, addr_constraints = ptr_result

        if expr_rmw.mask is not None:
            mask_result = self._safe_eval(expr_rmw.mask, "atomic_rmw mask eval")
            if mask_result is None:
                return
            mask_z3, mask_constraints = mask_result
        else:
            mask_z3, mask_constraints = None, None

        expr_constraints = self._combine_constraints(addr_constraints, mask_constraints)
        active = mask_z3 if mask_z3 is not None else True
        source_location = capture_current_source_location()

        self._record_atomic_rmw_event(
            symbolic_expr=expr,
            addr_expr=addr_expr,
            expr_constraints=expr_constraints,
            sem=sem,
            scope=scope,
            source_location=source_location,
            active=active,
        )

    @staticmethod
    def _atomic_rmw_return_dtype(ptr_sym: SymbolicExpr, val_sym: SymbolicExpr) -> Any:
        ptr_dtype = getattr(ptr_sym, "dtype", None)
        elem_ty = getattr(ptr_dtype, "element_ty", None)
        if elem_ty is not None:
            return elem_ty
        return getattr(val_sym, "dtype", None)

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
        mask_sym = None if mask is None else SymbolicExpr.from_value(mask)
        event_expr = SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)
        self._handle_atomic_rmw_check(event_expr, sem=sem, scope=scope)
        # Return a sentinel rather than the event expr: the RMW return value's
        # symbolic semantics are NOT modeled. Downstream use (mask = old == 0)
        # triggers UnsupportedSymbolicRaceQuery via the sentinel's _to_z3_impl,
        # which the wrapping _safe_eval translates into _mark_unsupported.
        return _UnsupportedRMWReturn(
            dtype=self._atomic_rmw_return_dtype(ptr_sym, val_sym),
            shape=getattr(ptr_sym, "shape", ()),
        )

    # ── Per-pending handler invoked from SymbolicClient's loop template

    def _process_pending_check(
        self,
        ctx: LoopContext,
        pending: PendingCheck,
        iter_constraints: list[BoolRef],
    ) -> None:
        del ctx
        if self._unsupported_capture:
            return
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
        # Distinguish "no race" from "race detector wasn't running": a Null
        # detector reports last_status == "disabled" so callers don't read
        # last_reports == [] as a clean pass.
        self.last_reports = []
        self.last_status = "disabled"
        self.unsupported_reason = "race detector disabled"
