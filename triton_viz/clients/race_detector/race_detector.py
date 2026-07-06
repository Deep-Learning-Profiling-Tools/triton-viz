import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    ClassVar,
    TypeVar,
    cast,
)

import torch
from z3 import (
    If,
    Implies,
    IntSort,
    IntVal,
    K,
    Or,
    Select,
    Store as Z3ArrayStore,
)
from z3.z3 import BoolRef

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    AtomicCas,
    AtomicRMW,
    Load,
    Store,
)
from ..symbolic_engine import (
    SymbolicExpr,
    LoadSymbolicExpr,
    AtomicCasSymbolicExpr,
    AtomicRmwSymbolicExpr,
    TensorPointerSymbolicExpr,
    SymbolicClient,
    NullSymbolicClient,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
    AccessMode,
    _and_constraints,
    _constraint_to_bool,
    scalar_truthiness_from_user_code,
)
from .data import AccessEventRecord, MemorySem
from .hb_common import (
    UnsupportedSymbolicRaceQuery,
    apply_sub,
    normalize_copy_local_vars,
)
from .two_copy_symbolic_hb_solver import TwoCopySymbolicHBSolver
from ...utils.traceback_utils import capture_current_source_location
from ...core.config import config as cfg

RaceDetectorT = TypeVar("RaceDetectorT", bound="RaceDetector")


def _hash_signature_part(x: Any) -> int:
    """Hash one component of an event signature, tolerating Z3 expressions,
    nested lists/tuples, and unhashable values."""
    if isinstance(x, (list, tuple)):
        return hash(tuple(_hash_signature_part(v) for v in x))
    if x is None:
        return 0
    try:
        return hash(x)
    except TypeError:
        return hash(repr(x))


def _make_event_signature(
    access_mode: AccessMode,
    source_location: tuple[str, int, str] | None,
    addr_expr: Z3Expr,
    local_constraints: ConstraintConjunction,
    active_expr: Any = True,
) -> int:
    """Signature used to dedupe repeated access events within a single loop.

    Distinct from sanitizer's ``_make_signature``: ``access_mode`` and
    ``source_location`` are part of the key so a ``load`` and a ``store`` at
    the same address inside the same loop stay as separate events (different
    program-order nodes for future HB analysis). ``active_expr`` is part of
    the key because mask conditions now flow into ``record.active`` instead
    of ``local_constraints``; without it, two iterations with the same
    address but different masks would collide.
    """
    return hash(
        (
            access_mode,
            source_location,
            _hash_signature_part(addr_expr),
            _hash_signature_part(local_constraints),
            _hash_signature_part(active_expr),
        )
    )


@dataclass
class PendingEvent(PendingCheck):
    """Extends ``PendingCheck`` with access-mode and op_type metadata.

    ``LoopContext.pending_checks`` is typed as ``list[PendingCheck]`` but is
    fine accepting subclass instances at runtime. Keeping the extension local
    to this module avoids widening the shared schema used by sanitizer.
    """

    access_mode: AccessMode = "read"
    op_type: type[Op] = Load
    active: Any = True
    # Event-local Z3 vars the two-copy solver must alpha-rename per program
    # copy (block-pointer tile index vars); the flushed loop's iterators are
    # appended at flush time by _process_pending_check.
    copy_local_vars: tuple[Any, ...] = ()


class RaceDetector(Client):
    """Factory class that returns the concrete race-detector implementation.

    Backend selection (only when the public ``RaceDetector(...)`` factory is
    instantiated directly — explicit subclass instances bypass this):

      * ``cfg.enable_race_detector`` off  → :class:`NullRaceDetector`
      * ``compile=True``                  → ``CompiledRaceDetector`` (static
        shared-memory analysis over the compiled TTGIR)
      * otherwise                         → :class:`SymbolicRaceDetector`
        (dynamic cross-CTA global-memory analysis)

    The two backends are complementary: the dynamic one reasons about global
    memory from an interpreter-driven symbolic capture, the compiled one about
    shared memory introduced by the TritonGPU pipeliner. Pick the compiled
    backend with ``RaceDetector(compile=True)``; extra keywords flow to the
    chosen backend's ``__init__`` (e.g. ``RaceDetector(compile=True,
    collect_smtlib=True)``).

    Note: the compiled backend runs STANDALONE — it skips the interpreted run,
    so it cannot be composed with other clients (Tracer, the dynamic detector,
    Profiler) in one ``@triton_viz.trace``; ClientManager raises if you try.
    Trace the dynamic and compiled detectors as separate decorations.
    """

    NAME = "race_detector"
    LOG_TAG: ClassVar[str] = "RaceDetector"
    LOG_VERB: ClassVar[str] = "recording"

    def __new__(cls: type[RaceDetectorT], *args: Any, **kwargs: Any) -> RaceDetectorT:
        if cls is RaceDetector:
            # ``compile`` selects the backend here; pop it so it does not reach
            # the backend __init__ via the manual call below. (It still reaches
            # SymbolicRaceDetector via Python re-invoking __init__ on the
            # factory-returned instance, which is why that __init__ tolerates
            # it. The compiled backend is not a RaceDetector subclass, so no
            # such re-invocation happens for it.)
            compiled = bool(kwargs.pop("compile", False))
            if not cfg.enable_race_detector:
                target_cls: type[RaceDetector] = NullRaceDetector
            elif compiled:
                from .compiled import CompiledRaceDetector

                target_cls = cast(type["RaceDetector"], CompiledRaceDetector)
            else:
                target_cls = SymbolicRaceDetector
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
    #   "aborted"     — no verdict exists: set pessimistically at
    #                   construction and re-armed at every launch's
    #                   arg_callback/grid_callback; only finalize() upgrades
    #                   it, so a launch that dies before finalize runs can
    #                   never be read as a clean "ok"
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
    # Upper bound on numel for an input tensor used as a tl.load value source.
    # Mirrors _MAX_INITIAL_ATOMIC_ELEMENTS in two_copy_symbolic_hb_solver.py;
    # larger sources are marked unsupported rather than blowing up the Z3
    # array snapshot.
    _MAX_LOAD_SOURCE_ELEMENTS: ClassVar[int] = 1024

    def __init__(self, abort_on_error: bool = False, *, compile: bool = False):
        # ``compile`` is consumed by the RaceDetector factory (__new__) to pick
        # the backend; it only reaches this __init__ because Python re-invokes
        # __init__ on the factory-returned instance with the original kwargs.
        # The symbolic backend ignores it (compile=True dispatches to
        # CompiledRaceDetector instead).
        del compile
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[AccessEventRecord] = []
        self.last_reports: list[Any] = []
        # Status of the most recent finalize(): "ok" means the solver ran;
        # "unsupported" means the launch hit a feature the solver doesn't
        # model (atomic-in-loop, RMW return downstream, data-dependent
        # address, etc.); "aborted" means the launch produced no verdict.
        # Initialized pessimistically and re-armed to "aborted" at every
        # launch's arg_callback/grid_callback; only finalize() upgrades it,
        # so neither a never-launched detector nor a launch that dies
        # mid-kernel can ever be read as clean. last_reports being empty
        # does NOT imply "no race" unless last_status == "ok".
        self.last_status: str = "aborted"
        self._program_seq: int = 0
        self._event_seq: int = 0
        self._launch_grid: tuple[int, int, int] = (1, 1, 1)
        self._captured_symbolic_template: bool = False
        # One-shot capture slot: claimed atomically by the first admitted
        # block in pre_run_callback, sealed by that block's
        # post_run_callback. Blocks running outside the slot (need_full_grid
        # reruns for a co-attached client, sibling workers under
        # TRITON_VIZ_NUM_SMS >= 2) execute the kernel but must not mutate
        # the shared per-launch record state.
        self._capture_claimed: bool = False
        self._capture_thread_id: int | None = None
        self._unsupported_capture: bool = False
        self.unsupported_reason: str | None = None
        self._arange_dict_snapshot: dict[Any, Any] = {}
        # Load-value modelling: per-launch Z3 array cache and bidirectional
        # region tracking. Keys are (base, elem_size, numel, str(dtype));
        # values are (z3_array_const, [IntVal(base+i*es) for i in range(n)]).
        self._load_array_cache: dict[
            tuple[int, int, int, str], tuple[Any, list[Any]]
        ] = {}
        self._load_value_regions: list[tuple[int, int, Any]] = []
        self._written_regions: list[tuple[int, int, Any]] = []
        # Set when a write event's target tensor cannot be resolved; the
        # load-value provider raises unsupported on subsequent loads because
        # the unknown write may alias a snapshotted load source.
        self._unknown_written_region_seen: bool = False
        # Finished-loop iterator substitutions, keyed by loop hook lineno:
        # (idx_z3, IntVal(final iteration value)). See _apply_finished_iter_subs
        # for why leftover iterator references must be concretized at record time.
        self._finished_loop_iter_subs: dict[Any, tuple[Any, Any]] = {}
        # Stash of the substitution entry popped when a loop re-enters,
        # restored on a zero-iteration exit (a zero-trip loop leaves the
        # leftover Python variable — and thus its final value — unchanged).
        self._suspended_iter_subs: list[tuple[Any, tuple[Any, Any] | None]] = []
        # Every iterator var created this launch, plus the ones whose final
        # value varied across activations under an active outer loop
        # (substituting any single constant for those would be wrong).
        self._known_iter_var_keys: set[tuple[int, str, str]] = set()
        self._unstable_iter_var_keys: set[tuple[int, str, str]] = set()
        # Launch-level signatures of records flushed from loops. A loop
        # nested under an outer loop re-enters once per outer iteration with
        # a fresh LoopContext, so its per-activation signature_cache cannot
        # dedupe across activations; without this cache every outer iteration
        # appends another structurally identical record.
        self._loop_flush_signatures: set[int] = set()

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

    @contextmanager
    def _load_value_semantics(self):
        """Install the ``tl.load`` value provider around one detector-
        triggered evaluation.

        The provider slot is class-global on ``SymbolicExpr``; leaving it
        installed for the whole launch (as ``grid_callback`` once did)
        hijacks every other client's ``expr.eval()`` — a co-attached
        sanitizer's masked load without ``other`` would raise
        :class:`UnsupportedSymbolicRaceQuery` inside sanitizer code, and its
        unmasked loads would silently get ``Select(arr, addr)`` value
        semantics instead of the pointer-as-value lowering its OOB checks
        expect. Save/restore keeps nested detector evaluations re-entrant
        and exception-safe.
        """
        prev = SymbolicExpr._load_value_provider
        prev_owner = SymbolicExpr._load_value_provider_owner
        SymbolicExpr._load_value_provider = self._load_value_provider_impl
        SymbolicExpr._load_value_provider_owner = id(self)
        try:
            yield
        finally:
            SymbolicExpr._load_value_provider = prev
            SymbolicExpr._load_value_provider_owner = prev_owner

    def _safe_eval(self, expr: "SymbolicExpr", reason: str) -> tuple[Any, Any] | None:
        """Eval a SymbolicExpr under the detector's load-value semantics,
        marking the launch unsupported on
        :class:`UnsupportedSymbolicRaceQuery`. Returns ``None`` when
        unsupported so callers can ``if result is None: return``. The mark
        is recorded even when re-raising under ``abort_on_error`` so the
        aborted launch never reads as a clean verdict.

        ``NotImplementedError`` is how the symbolic engine signals a missing
        Z3 lowering (cumsum, dot, block-pointer descriptors, ...); it gets
        the same unsupported treatment so a lowering gap surfaces as a
        verdict instead of crashing the launch.
        """
        try:
            with self._load_value_semantics():
                return expr.eval()
        except UnsupportedSymbolicRaceQuery as exc:
            self._mark_unsupported(str(exc) or reason)
            if self.abort_on_error:
                raise
            return None
        except NotImplementedError as exc:
            message = f"{reason}: {exc}" if str(exc) else reason
            self._mark_unsupported(message)
            if self.abort_on_error:
                raise UnsupportedSymbolicRaceQuery(message) from exc
            return None

    @staticmethod
    def _combine_constraints(*constraints: Any) -> tuple[Any, ...]:
        """Flat tuple of non-None constraints; the two-copy solver's
        ``iter_constraints`` recursively flattens nested tuples/lists.
        """
        return tuple(c for c in constraints if c is not None)

    # Ops whose Z3 lowering depends on runtime memory contents or is not
    # value-faithful: loads/atomics conflate the pointer with the loaded
    # value, cumsum/sort are scans over runtime data (sort even lowers as
    # the identity of its input). An address embedding any of these would
    # record a wrong or unmodelable footprint — scatter/histogram patterns
    # where the destination index comes from a runtime value. Flag these as
    # unsupported until value semantics are properly modeled.
    _VALUE_DEPENDENT_ADDRESS_OPS: ClassVar[tuple[str, ...]] = (
        "load",
        "tensor_pointer_load",
        "atomic_cas",
        "atomic_rmw",
        "sort",
        "cumsum",
    )

    @classmethod
    def _find_value_dependent_op(cls, expr: SymbolicExpr | None) -> str | None:
        """Name of the first value-dependent op embedded in ``expr`` (a
        pointer expression), or None when the address is value-independent.
        """
        if expr is None:
            return None
        for op in cls._VALUE_DEPENDENT_ADDRESS_OPS:
            try:
                if expr.has_op(op):
                    return op
            except Exception:
                continue
        return None

    def _reject_data_dependent_address(self, ptr_expr: SymbolicExpr | None) -> bool:
        """If ``ptr_expr`` depends on a runtime value, mark the launch
        unsupported (or raise under abort_on_error) and return True; callers
        should ``return`` immediately on True.
        """
        op = self._find_value_dependent_op(ptr_expr)
        if op is None:
            return False
        self._raise_or_mark(
            f"data-dependent memory address through {op} is unsupported "
            "by the current symbolic race detector"
        )
        return True

    # Ops whose value differs across program instances (pid, arange lanes)
    # or depends on runtime memory contents (loads, sorts/scans over loaded
    # tensors, atomic returns). A scalar built only from constants — or from
    # enclosing loop iterators, which concretize per iteration — takes the
    # same value in every block, so host-side use of it stays sound.
    _PER_INSTANCE_OPS: ClassVar[tuple[str, ...]] = (
        "pid",
        "arange",
        "load",
        "tensor_pointer_load",
        "atomic_cas",
        "atomic_rmw",
        "sort",
        "cumsum",
    )

    @classmethod
    def _expr_varies_per_instance(cls, expr: SymbolicExpr | None) -> bool:
        if expr is None:
            return False
        try:
            return any(expr.has_op(op) for op in cls._PER_INSTANCE_OPS)
        except Exception:
            return False

    def _scalar_concretize_observer_impl(self, expr: SymbolicExpr) -> None:
        """Policy for the engine's scalar-concretization hook.

        Host-side control flow (``if pid == 0:``, ``while flag:``) forces a
        scalar symbolic value to the capture block's concrete value. Under
        one-shot capture that bakes block (0,0,0)'s branch decisions into
        the template for every PID: events inside a pid-guarded branch
        become unconditional for all pids (false positives) and branches
        the capture block doesn't take record nothing (false negatives) —
        with last_status still "ok". No path condition is modeled, so the
        only sound verdict is unsupported.
        """
        if self._unsupported_capture:
            return
        if not self._expr_varies_per_instance(expr):
            return
        if not scalar_truthiness_from_user_code():
            return
        self._raise_or_mark(
            "host-side control flow on a value that varies per program "
            "instance (program id, tl.arange, or loaded data) is "
            "unsupported by one-shot symbolic capture"
        )

    def _on_data_dependent_value(self, expr: Any = None) -> None:
        """Loop bounds / materialized operands that depend on loads or pids
        are concretized to the capture block's values (the sanitizer
        compensates by running the full grid via need_full_grid; one-shot
        capture cannot), so the launch verdict would be silently wrong.

        A bound built only from enclosing loop iterators is exempt: it
        concretizes to the right value on every iteration, and the
        finished-iterator machinery models the leftovers.
        """
        SymbolicClient._on_data_dependent_value(self, expr)
        if self._unsupported_capture:
            return
        if isinstance(expr, SymbolicExpr) and not self._expr_varies_per_instance(expr):
            return
        self._raise_or_mark(
            "loop bound or operand depending on per-instance or loaded "
            "data was concretized; unsupported by one-shot symbolic "
            "capture"
        )

    @staticmethod
    def _tensor_region(tensor: Any) -> tuple[int, int, Any]:
        """Byte-region (base, end_exclusive, tensor). ``end`` is exclusive.

        Using regions instead of ``data_ptr()`` equality so views/slices that
        share storage but differ in offset are still detected as overlapping.
        """
        base = int(tensor.data_ptr())
        elem_size = int(tensor.element_size()) if hasattr(tensor, "element_size") else 1
        end = base + int(tensor.numel()) * elem_size
        return (base, end, tensor)

    @staticmethod
    def _regions_overlap(a: tuple[int, int, Any], b: tuple[int, int, Any]) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])

    def _raise_or_mark(self, reason: str) -> None:
        # Mark before raising: the abort_on_error exception unwinds out of
        # the launch, and any finalize() that runs afterwards — whether the
        # harness routes the abort through it or the caller invokes it —
        # could not otherwise tell an unsupported launch from a clean one.
        self._mark_unsupported(reason)
        if self.abort_on_error:
            raise UnsupportedSymbolicRaceQuery(reason)

    def _note_written_tensor(self, tensor: Any) -> bool:
        """Register a write target. Returns True if the caller may proceed
        with recording the event, False if the launch was marked unsupported.

        - Unknown target: set ``_unknown_written_region_seen``. If a load
          snapshot already exists, also mark unsupported (the unknown write
          may alias it). Otherwise allow the event through; the flag gates
          any *subsequent* load via the provider's first check.
        - Known target overlapping an existing load source: mark unsupported.
        """
        if tensor is None:
            self._unknown_written_region_seen = True
            if self._load_value_regions:
                self._raise_or_mark(
                    "write to unknown region after tl.load value snapshot "
                    "is unsupported"
                )
                return False
            return True

        region = self._tensor_region(tensor)
        for snap in self._load_value_regions:
            if self._regions_overlap(region, snap):
                self._raise_or_mark(
                    "tl.store/atomic into a tensor previously read as a "
                    "tl.load value source is unsupported"
                )
                return False
        # Dedup writes to the same region — no need to track multiple
        # equal entries.
        for existing in self._written_regions:
            if existing[0] == region[0] and existing[1] == region[1]:
                return True
        self._written_regions.append(region)
        return True

    def _note_load_source_or_raise(self, tensor: Any) -> None:
        """Register a tensor as a load-value source. Raises
        :class:`UnsupportedSymbolicRaceQuery` when the source overlaps a
        region this kernel has already written to.
        """
        region = self._tensor_region(tensor)
        for written in self._written_regions:
            if self._regions_overlap(region, written):
                raise UnsupportedSymbolicRaceQuery(
                    "tl.load value from a tensor written by this kernel is "
                    "unsupported"
                )
        for existing in self._load_value_regions:
            if existing[0] == region[0] and existing[1] == region[1]:
                return
        self._load_value_regions.append(region)

    # ── Load-value provider (tl.load value semantics in Z3) ────────────────

    @staticmethod
    def _is_modelable_dtype(dtype: Any) -> bool:
        """v1 only models integer-valued or bool input tensors. Floats and
        complex dtypes raise unsupported because the Z3 model is integer-
        only and silently downcasting would mask real value-dependent
        behaviour.
        """
        if dtype is None:
            return False
        try:
            if dtype == torch.bool:
                return True
            if hasattr(dtype, "is_floating_point") and dtype.is_floating_point:
                return False
            if hasattr(dtype, "is_complex") and dtype.is_complex:
                return False
            # Treat anything else with a finite integer representation as
            # modelable. Triton's int8/int16/int32/int64/uint8 all qualify.
            return getattr(dtype, "is_signed", None) is not None or hasattr(
                dtype, "itemsize"
            )
        except Exception:
            return False

    def _snapshot_array_for_tensor(self, tensor: Any) -> tuple[Any, list[Any]]:
        """Build (or fetch from cache) a Z3 Array representing the tensor's
        current contents, plus the list of known IntVal addresses for every
        element. Raises :class:`UnsupportedSymbolicRaceQuery` on any input
        the v1 model can't faithfully represent.
        """
        if not hasattr(tensor, "numel") or not hasattr(tensor, "data_ptr"):
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value modelling requires a torch tensor source"
            )
        if hasattr(tensor, "is_contiguous") and not bool(tensor.is_contiguous()):
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value from a non-contiguous tensor is unsupported"
            )
        numel = int(tensor.numel())
        if numel <= 0:
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value from an empty tensor is unsupported"
            )
        if numel > self._MAX_LOAD_SOURCE_ELEMENTS:
            raise UnsupportedSymbolicRaceQuery(
                f"tl.load value source tensor exceeds size cap "
                f"({numel} > {self._MAX_LOAD_SOURCE_ELEMENTS})"
            )
        if not self._is_modelable_dtype(getattr(tensor, "dtype", None)):
            raise UnsupportedSymbolicRaceQuery(
                f"tl.load value with dtype {getattr(tensor, 'dtype', '?')} "
                "is unsupported (v1 models integer/bool only)"
            )

        base = int(tensor.data_ptr())
        elem_size = int(tensor.element_size()) if hasattr(tensor, "element_size") else 1
        elem_size = max(1, elem_size)
        cache_key = (base, elem_size, numel, str(tensor.dtype))
        cached = self._load_array_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            host = tensor.detach() if hasattr(tensor, "detach") else tensor
            host = host.cpu() if hasattr(host, "cpu") else host
            flat = host.reshape(-1).tolist()
            int_values = [int(v) for v in flat]
        except Exception as exc:
            raise UnsupportedSymbolicRaceQuery(
                f"tl.load value tensor snapshot failed: {exc}"
            ) from exc

        arr = K(IntSort(), IntVal(0))
        known_addrs: list[Any] = []
        for i, value in enumerate(int_values):
            addr_i = IntVal(base + i * elem_size)
            arr = Z3ArrayStore(arr, addr_i, IntVal(value))
            known_addrs.append(addr_i)

        self._load_array_cache[cache_key] = (arr, known_addrs)
        return arr, known_addrs

    @staticmethod
    def _to_lane_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _broadcast_lanes(lanes: list[Any], n: int) -> list[Any]:
        if len(lanes) == n:
            return lanes
        if len(lanes) == 1:
            return [lanes[0]] * n
        raise UnsupportedSymbolicRaceQuery(
            f"tl.load lane count mismatch (expected {n}, got {len(lanes)})"
        )

    def _load_value_provider_impl(
        self, load_expr: LoadSymbolicExpr
    ) -> tuple[Z3Expr, ConstraintConjunction]:
        """Compute Z3 value semantics for ``tl.load``.

        Address-of-event recording stays with ``expr.ptr._to_z3()`` (see
        ``_handle_access_check``); this provider only kicks in when the
        loaded value is consumed as part of a downstream expression (e.g.
        ``(v == 0)`` in a store mask). The returned Z3 expression evaluates
        to the actual loaded value via ``Select(arr, addr)`` over a
        per-launch tensor snapshot.

        Boundaries that raise :class:`UnsupportedSymbolicRaceQuery`:
          - launch has already seen a write to an unresolved tensor region;
          - source pointer cannot be resolved to a known tensor;
          - source overlaps a tensor this kernel writes to;
          - source is non-contiguous, too large, or has unsupported dtype;
          - masked load without an explicit ``other`` (no sound default for
            inactive lanes in v1).
        """
        if self._unknown_written_region_seen:
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value snapshot is unsupported after a write to an "
                "unknown kernel region"
            )

        ptr_z3, ptr_constraints = load_expr.ptr._to_z3()

        tensor = self._resolve_tensor(load_expr.ptr)
        if tensor is None:
            raise UnsupportedSymbolicRaceQuery(
                "tl.load value from unknown tensor is unsupported"
            )

        self._note_load_source_or_raise(tensor)
        arr, known_addrs = self._snapshot_array_for_tensor(tensor)

        addr_lanes = self._to_lane_list(ptr_z3)
        lane_count = len(addr_lanes)

        if load_expr.mask is None:
            values = [Select(arr, a) for a in addr_lanes]
            domain_terms = [Or(*(a == k for k in known_addrs)) for a in addr_lanes]
            extra_constraints: tuple[Any, ...] = (_and_constraints(*domain_terms),)
        else:
            if load_expr.other is None:
                raise UnsupportedSymbolicRaceQuery(
                    "masked tl.load without explicit `other` is unsupported"
                )
            mask_z3, mask_constraints = load_expr.mask._to_z3()
            other_z3, other_constraints = load_expr.other._to_z3()
            mask_lanes = self._broadcast_lanes(self._to_lane_list(mask_z3), lane_count)
            other_lanes = self._broadcast_lanes(
                self._to_lane_list(other_z3), lane_count
            )

            values = []
            domain_terms = []
            for a, m, o in zip(addr_lanes, mask_lanes, other_lanes):
                m_bool = _constraint_to_bool(m)
                values.append(If(m_bool, Select(arr, a), o))
                domain_terms.append(Implies(m_bool, Or(*(a == k for k in known_addrs))))
            extra_constraints = (
                mask_constraints,
                other_constraints,
                _and_constraints(*domain_terms),
            )

        result: Z3Expr = values[0] if lane_count == 1 else values
        constraints = _and_constraints(ptr_constraints, *extra_constraints)
        return result, constraints

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
        "unsupported" / "aborted" / "disabled" should read
        :attr:`last_status` and :attr:`unsupported_reason`.
        ``last_status == "aborted"`` means an exception cut the launch short
        before the capture (or the solver) completed, so no verdict exists.
        ``last_status == "disabled"`` is set by :class:`NullRaceDetector`
        when the backend is off.

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
        try:
            if not self._captured_symbolic_template or self._unsupported_capture:
                if self._unsupported_capture and cfg.verbose:
                    print(
                        f"[{self.LOG_TAG}] launch unsupported by two-copy solver: "
                        f"{self.unsupported_reason}"
                    )
                self.last_reports = []
                if self._unsupported_capture:
                    self.last_status = "unsupported"
                else:
                    # The capture was never sealed: an exception aborted the
                    # launch mid-block. No analysis ran, so reporting "ok"
                    # here would be a silent false no-race verdict. This
                    # matches the pessimistic "aborted" grid_callback set at
                    # launch start, whether or not the harness routed the
                    # abort through finalize.
                    self.last_status = "aborted"
                return []
            try:
                reports = TwoCopySymbolicHBSolver(
                    self.records,
                    grid=self._launch_grid,
                    arange_dict=self._arange_dict_snapshot,
                ).find_races()
                self.last_status = "ok"
            except UnsupportedSymbolicRaceQuery as exc:
                self._mark_unsupported(str(exc))
                if self.abort_on_error:
                    raise
                reports = []  # NO concrete fallback
            except BaseException:
                # Solver-internal failures (z3 errors, lowering bugs) abort
                # the analysis; re-assert the launch-start "aborted" so no
                # partial upgrade can ever read as a clean verdict.
                self.last_reports = []
                self.last_status = "aborted"
                raise
            self.last_reports = reports
            return reports
        finally:
            # Unconditional: an exception escaping finalize (abort_on_error
            # re-raise, z3 error) must still release the launch runtime —
            # the class-level scalar-concretize observer would otherwise
            # leak into later launches of other clients.
            self._clear_launch_runtime()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        # Arm the pessimistic verdict as early as the launch becomes
        # observable: arg conversion and the user's grid lambda run BEFORE
        # grid_callback (core/frontend/triton.py), and a crash there must
        # not leave the previous launch's "ok" readable as this launch's
        # verdict. Idempotent across the per-argument calls.
        self.last_status = "aborted"
        self.last_reports = []
        SymbolicClient.arg_callback(self, name, arg, arg_cvt)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self.records = []
        self.last_reports = []
        # Pessimistic until finalize() proves otherwise: if the launch dies
        # before finalize runs (a mid-kernel exception with no
        # finalize-on-error routing in the harness), a pre-set "ok" here
        # would read as a clean no-race verdict. Every finalize() path
        # overwrites this with the real outcome.
        self.last_status = "aborted"
        self._program_seq = 0
        self._event_seq = 0
        normalized = tuple(int(dim) for dim in grid)
        while len(normalized) < 3:
            normalized = normalized + (1,)
        self._launch_grid = cast(tuple[int, int, int], normalized[:3])
        self._captured_symbolic_template = False
        self._capture_claimed = False
        self._capture_thread_id = None
        # Reset of unsupported state lives ONLY in grid_callback. post_run_callback
        # must NOT zero these — handlers within the same launch may have set them.
        self._unsupported_capture = False
        self.unsupported_reason = None
        self._arange_dict_snapshot = {}
        self._load_array_cache = {}
        self._load_value_regions = []
        self._written_regions = []
        self._unknown_written_region_seen = False
        self._finished_loop_iter_subs = {}
        self._suspended_iter_subs = []
        self._known_iter_var_keys = set()
        self._unstable_iter_var_keys = set()
        self._loop_flush_signatures = set()
        SymbolicExpr.ARANGE_DICT.clear()
        # SymbolicClient.grid_callback also clears loop_stack, so a launch
        # that aborted mid-loop cannot poison this one.
        SymbolicClient.grid_callback(self, grid)
        # The scalar-concretize observer must span the whole kernel run
        # (host-side truthiness fires from interpreter code, not from
        # detector-triggered evals), so it is installed launch-wide with an
        # owner token; _clear_launch_runtime only uninstalls while we still
        # own the slot. The load-value provider is NOT installed here — it
        # is scoped to the detector's own evaluations via
        # _load_value_semantics so other clients' expr.eval() never sees it.
        SymbolicExpr._scalar_concretize_observer = self._scalar_concretize_observer_impl
        SymbolicExpr._scalar_concretize_observer_owner = id(self)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        return SymbolicClient.register_op_callback(self, op_type)

    # ── Race-detector-only overrides for load/store overriders ────────────
    # The shared SymbolicClient versions concretise nested loads in ``ptr``
    # and ``mask`` via ``replace_subtree("load")`` and flip
    # ``need_full_grid``. Under one-shot symbolic capture that turns the
    # first block's concrete mask value into a template for every PID — the
    # source of the load-dependent-mask false negative. The race detector
    # has its own ``LoadSymbolicExpr`` value semantics (per-PID Z3
    # ``Select(arr, addr)``) plus a pointer-side ``_reject_data_dependent_
    # address`` gate, so it doesn't need (and must not use) the
    # concretisation path. Sanitizer keeps the shared behaviour.

    def _op_load_overrider(self, ptr, mask=None, other=None, *args, **kwargs):
        ptr_sym = SymbolicExpr.from_value(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        other_sym = SymbolicExpr.from_value(other) if other is not None else None
        ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)
        self._handle_access_check(ret, Load, "read")
        return ret

    def _op_store_overrider(self, ptr, value, mask=None, *args, **kwargs):
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        ret = SymbolicExpr.create("store", ptr_sym, value_sym, mask_sym)
        self._handle_access_check(ret, Store, "write")
        return ret

    def pre_run_callback(self, fn: Callable) -> bool:
        # Block scheduling stays with the shared SymbolicClient machinery:
        # it honors need_full_grid (so a co-attached client is not starved
        # of blocks when the engine concretized per-block values) and counts
        # _active_blocks for the deferred launch-state clear. One-shot
        # capture is layered on top: the first admitted block claims the
        # capture slot atomically, so sibling workers under
        # TRITON_VIZ_NUM_SMS >= 2 never interleave captures into the shared
        # per-launch record state.
        with self._lock_context():
            should_run = SymbolicClient.pre_run_callback(self, fn)
            if should_run and not self._capture_claimed:
                self._capture_claimed = True
                self._capture_thread_id = threading.get_ident()
            return should_run

    def post_run_callback(self, fn: Callable) -> bool:
        with self._lock_context():
            if self._capture_thread_id == threading.get_ident():
                try:
                    self._seal_capture()
                finally:
                    self._capture_thread_id = None
            return SymbolicClient.post_run_callback(self, fn)

    def _capture_active(self) -> bool:
        """True when events observed on the current thread belong to the
        one-shot capture.

        Blocks executed outside the capture slot (need_full_grid reruns for
        a co-attached client, sibling workers under TRITON_VIZ_NUM_SMS >= 2)
        still run the kernel, but their events must not reach the shared
        per-launch record/loop state — the two-copy solver already reasons
        over all blocks from the single captured template. Direct handler
        calls with no claimed slot (unit-level use) count as capturing.
        """
        if self._captured_symbolic_template:
            return False
        tid = self._capture_thread_id
        return tid is None or tid == threading.get_ident()

    def _seal_capture(self) -> None:
        """Seal the one-shot capture once the capturing block finishes.

        Runs exactly once per launch, on the thread that claimed the capture
        slot in pre_run_callback. The seal flag is set even on the
        unsupported/raising paths so finalize() can tell a completed capture
        from a launch aborted mid-block.
        """
        try:
            # If a handler already marked the launch unsupported, don't try
            # to force-eval half-baked record state — short-circuit cleanly.
            if self._unsupported_capture:
                return
            # Defense in depth: a LoopContext that survives to launch end
            # was never flushed (its teardown hook did not fire), so its
            # deferred accesses are missing from the records — the capture
            # is incomplete and an "ok" verdict would be silently wrong.
            if self.loop_stack:
                self._raise_or_mark(
                    "loop context left open at launch end; loop-deferred "
                    "accesses were never recorded"
                )
                return
            try:
                self._force_eval_record_templates()
            except UnsupportedSymbolicRaceQuery as exc:
                self._mark_unsupported(str(exc))
                if self.abort_on_error:
                    raise
                return
            # Defensive sweep: load-side / write-side checks at record time
            # should already catch overlaps, but loop-deferred events can
            # re-order tensors through `_process_pending_check`. Cross-
            # product check the two region lists once before sealing.
            for src in self._load_value_regions:
                for dst in self._written_regions:
                    if self._regions_overlap(src, dst):
                        self._raise_or_mark(
                            "tl.load value source overlaps a tensor written "
                            "by this kernel"
                        )
                        return
            # Snapshot ARANGE_DICT after templates are evaluated so the
            # two-copy solver's arange substitutions are independent of
            # subsequent launches.
            self._arange_dict_snapshot = dict(SymbolicExpr.ARANGE_DICT)
        finally:
            self._captured_symbolic_template = True

    # ── Event recording ───────────────────────────────────────────────────

    def _clear_launch_runtime(self) -> None:
        self._clear_cache()
        self._clear_symbolic_launch_state()
        self.need_full_grid = None
        self.solver = None
        self.addr_ok = None
        self.pid_ok = None
        # addr_sym is instance-lifetime, not launch-scoped: it is created once
        # in SymbolicClient.__init__ and never recreated, and the next launch's
        # grid_callback dereferences it via _addr_ok_premise(). Clearing it
        # here would crash the second launch of any traced kernel.
        self.grid = None
        self.grid_idx = None
        self.last_grid = None
        self._capture_claimed = False
        self._capture_thread_id = None
        self._program_seq = 0
        self._event_seq = 0
        self._load_array_cache = {}
        self._load_value_regions = []
        self._written_regions = []
        self._unknown_written_region_seen = False
        self._finished_loop_iter_subs = {}
        self._suspended_iter_subs = []
        self._known_iter_var_keys = set()
        self._unstable_iter_var_keys = set()
        self._loop_flush_signatures = set()
        if SymbolicExpr._load_value_provider_owner == id(self):
            SymbolicExpr._load_value_provider = None
            SymbolicExpr._load_value_provider_owner = None
        if SymbolicExpr._scalar_concretize_observer_owner == id(self):
            SymbolicExpr._scalar_concretize_observer = None
            SymbolicExpr._scalar_concretize_observer_owner = None

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

    # ── Finished-loop iterator concretization ─────────────────────────────
    # After a range loop exits, the leftover Python loop variable (and any
    # value derived from it) still lowers to the loop's symbolic loop_i_*
    # var, yet every program instance concretely holds the same final
    # iteration value. Keeping the symbolic var in post-loop records is
    # unsound either way: un-renamed it pins both solver copies together,
    # and renamed (which happens launch-wide once any in-loop record lists
    # the var in copy_local_vars) it roams unbounded because post-loop
    # records carry no range premise for it. Substituting the concrete
    # final value at record time models the actual semantics — but only
    # when that value is well defined:
    #   * a zero-trip re-activation leaves the leftover variable (and so
    #     its substitution) unchanged, hence the stash/restore;
    #   * a loop whose final value varies across activations under a still-
    #     active outer loop has no single correct constant (deferred records
    #     dedupe across those activations), so its var is marked unstable
    #     and any record still referencing it is rejected as unsupported by
    #     _refs_unresolved_iter_var.

    # The loop hooks fire for every executed block, but the loop bookkeeping
    # (loop_stack, suspended/finished iterator substitutions) belongs to the
    # one-shot capture: blocks running outside the capture slot iterate
    # concretely and must not touch it — a sibling worker popping the
    # capture thread's stash is exactly the kind of corruption the slot
    # exists to prevent. _wrap_range stays ungated: every running block
    # still needs its loop bounds materialized.

    def _loop_hook_iter_overrider(self, lineno: Any, idx: Any) -> Any:
        if not self._capture_active():
            return idx
        return SymbolicClient._loop_hook_iter_overrider(self, lineno, idx)

    def _loop_hook_before(self, lineno: int, iterable: Any) -> None:
        if not self._capture_active():
            return
        SymbolicClient._loop_hook_before(self, lineno, iterable)
        if self.loop_stack and self.loop_stack[-1].lineno == lineno:
            ctx = self.loop_stack[-1]
            self._known_iter_var_keys.add(self._iter_var_key(ctx.idx_z3))
            # Reactivation: suspend the finished-value substitution so this
            # loop's own deferred records keep the symbolic var for per-copy
            # renaming. Restored on a zero-iteration exit.
            self._suspended_iter_subs.append(
                (lineno, self._finished_loop_iter_subs.pop(lineno, None))
            )

    def _loop_hook_after(self, lineno: int) -> None:
        if not self._capture_active():
            return
        ctx = (
            self.loop_stack[-1]
            if self.loop_stack and self.loop_stack[-1].lineno == lineno
            else None
        )
        SymbolicClient._loop_hook_after(self, lineno)
        if ctx is None:
            return
        stashed_lineno, stashed = self._suspended_iter_subs.pop()
        assert stashed_lineno == lineno
        var_key = self._iter_var_key(ctx.idx_z3)
        if var_key in self._unstable_iter_var_keys:
            return
        if ctx.current_value is None:
            # Zero-trip activation: the leftover variable still holds the
            # previous activation's final value — restore its substitution.
            if stashed is not None:
                self._finished_loop_iter_subs[lineno] = stashed
            return
        # Register AFTER the super() flush so the loop's own records (which
        # are recorded during the flush) keep their symbolic iterator; only
        # records created after this point see the concrete final value.
        final = IntVal(int(ctx.current_value))
        if stashed is not None and not stashed[1].eq(final) and self.loop_stack:
            # The final value varies across activations while an outer loop
            # is active: pendings deferred in that outer loop dedupe across
            # the differing activations, so no constant is correct.
            self._unstable_iter_var_keys.add(var_key)
            return
        self._finished_loop_iter_subs[lineno] = (ctx.idx_z3, final)

    def _apply_finished_iter_subs(self, value: Any) -> Any:
        if not self._finished_loop_iter_subs:
            return value
        return apply_sub(value, tuple(self._finished_loop_iter_subs.values()))

    @staticmethod
    def _iter_var_key(v: Any) -> tuple[int, str, str]:
        # Mirrors the dedup key used by hb_common.normalize_copy_local_vars.
        return (v.hash(), str(v.sort()), v.decl().name())

    def _refs_unresolved_iter_var(
        self, values: tuple[Any, ...], allowed_vars: tuple[Any, ...]
    ) -> bool:
        """True if any Z3 expr in ``values`` references a loop iterator var
        that is neither substituted away nor legitimately symbolic here.

        ``allowed_vars`` are the iterators the record is allowed to keep
        symbolic: the still-active outer loops' plus (for deferred records)
        the flushed loop's own — exactly the ones the two-copy solver
        alpha-renames via copy_local_vars. Anything else is a finished
        iterator whose substitution was skipped (unstable final value or a
        lifecycle corner); recording it would be silently wrong, so the
        caller marks the launch unsupported instead.
        """
        disallowed = self._known_iter_var_keys - {
            self._iter_var_key(v) for v in allowed_vars
        }
        if not disallowed:
            return False
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
                if self._iter_var_key(v) in disallowed:
                    return True
                continue
            stack.extend(v.children())
        return False

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

        with self._load_value_semantics():
            for record in self.records:
                try:
                    record.addr_expr = force(record.addr_expr)
                    record.local_constraints = self._normalize_constraints(
                        force(record.local_constraints)
                    )
                    record.premises = self._normalize_constraints(
                        force(record.premises)
                    )
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
        active: Any = True,
        loop_flush: bool = False,
    ) -> None:
        if self._unsupported_capture:
            return
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        # Bidirectional self-write guard. For writes (and atomics), the
        # target region must not overlap a tensor we've already snapshotted
        # as a load-value source — that would mean the snapshot represents
        # stale memory. Likewise, an unresolved write target may alias an
        # existing snapshot, so we set a flag the load-value provider checks
        # before snapshotting any further tensors.
        if access_mode == "write" or op_type in (AtomicCas, AtomicRMW):
            if not self._note_written_tensor(tensor):
                return

        # Two-copy capture: keep raw symbolic templates (PID0/1/2 preserved)
        # rather than snapshotting solver assertions, which can carry sampled
        # PID equalities that pin pid_a == pid_b == sampled_pid after alpha-
        # renaming and break the two-copy alias query.
        access_addr = self._apply_finished_iter_subs(access_addr)
        active = self._apply_finished_iter_subs(active)
        local = self._normalize_constraints(
            self._apply_finished_iter_subs(expr_constraints)
        )
        premises = self._normalize_constraints(
            self._apply_finished_iter_subs(semantic_constraints)
        )
        allowed_iter_vars = copy_local_vars + self._current_loop_iter_vars()
        if self._refs_unresolved_iter_var(
            (access_addr, active, local, premises), allowed_iter_vars
        ):
            self._raise_or_mark(
                "access references a finished loop iterator with no stable "
                "final value"
            )
            return

        normalized_copy_vars = normalize_copy_local_vars(copy_local_vars)
        if loop_flush:
            # A loop nested under another loop re-enters once per outer
            # iteration, re-flushing structurally identical pending events
            # each time (fresh LoopContext, fresh signature_cache). One
            # record already covers every iteration pair through its symbolic
            # iterators, so dedupe at launch scope — AFTER the finished-
            # iterator substitutions and the unresolved-var rejection above,
            # which are the parts that can legitimately differ (or become
            # unsupported) across activations.
            signature = hash(
                (
                    _make_event_signature(
                        access_mode, source_location, access_addr, local, active
                    ),
                    op_type.__name__,
                    _hash_signature_part(premises),
                    tuple(self._iter_var_key(v) for v in normalized_copy_vars),
                )
            )
            if signature in self._loop_flush_signatures:
                return
            self._loop_flush_signatures.add(signature)

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
                active=active,
                reads=access_mode == "read",
                writes=access_mode == "write",
                event_id=self._next_event_id(),
                elem_size=self._infer_elem_size(symbolic_expr),
                copy_local_vars=normalized_copy_vars,
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
        if not self._note_written_tensor(tensor):
            return

        addr_expr = self._apply_finished_iter_subs(addr_expr)
        cmp_value = self._apply_finished_iter_subs(cmp_value)
        value = self._apply_finished_iter_subs(value)
        local = self._normalize_constraints(
            self._apply_finished_iter_subs(expr_constraints)
        )
        premises = self._normalize_constraints(
            self._apply_finished_iter_subs(semantic_constraints)
        )
        loop_vars = self._current_loop_iter_vars()
        if self._refs_unresolved_iter_var(
            (addr_expr, cmp_value, value, local, premises), loop_vars
        ):
            self._raise_or_mark(
                "atomic_cas references a finished loop iterator with no "
                "stable final value"
            )
            return

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
        if not self._note_written_tensor(tensor):
            return

        addr_expr = self._apply_finished_iter_subs(addr_expr)
        active = self._apply_finished_iter_subs(active)
        local = self._normalize_constraints(
            self._apply_finished_iter_subs(expr_constraints)
        )
        premises = self._normalize_constraints(
            self._apply_finished_iter_subs(semantic_constraints)
        )
        loop_vars = self._current_loop_iter_vars()
        if self._refs_unresolved_iter_var(
            (addr_expr, active, local, premises), loop_vars
        ):
            self._raise_or_mark(
                "atomic_rmw references a finished loop iterator with no "
                "stable final value"
            )
            return

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

        The pointer and mask expressions are evaluated separately: the
        address-of-the-event is ``expr.ptr._to_z3()`` (independent of any
        load-value provider that may give ``LoadSymbolicExpr`` value
        semantics), and the mask becomes the event's ``active`` condition so
        ``_lower_record`` can take per-lane lane-values rather than ``And``-
        collapsing a vector mask into a scalar local constraint. Block-
        pointer accesses are the exception: their ``ptr`` is an unlowerable
        descriptor, so the access expr itself supplies the tile footprint.
        """
        if self._unsupported_capture or not self._capture_active():
            return
        # Reject scatter/histogram-style addressing where the pointer itself
        # depends on a runtime value — the current model conflates e.g. a
        # load's pointer with its loaded value.
        ptr_attr = getattr(expr, "ptr", None)
        if self._reject_data_dependent_address(ptr_attr):
            return
        if ptr_attr is None:
            return
        if isinstance(expr, TensorPointerSymbolicExpr):
            # Block pointers: expr.ptr is a make_block_ptr/advance descriptor
            # with no address lowering of its own — the access expr itself
            # lowers the tile footprint (base + (offset_d + k_d) * stride_d
            # with each k_d range-bound in the constraints). The k_d vars are
            # copy-local: without per-copy renaming the two program copies
            # would share one tile coordinate and overlapping tiles could
            # only collide at equal k.
            addr_attr: SymbolicExpr = expr
            tile_vars = expr.tile_index_vars()
        else:
            addr_attr = ptr_attr
            tile_vars = ()
        ptr_result = self._safe_eval(addr_attr, f"{op_type.__name__} ptr eval")
        if ptr_result is None:
            return
        z3_addr, ptr_constraints = ptr_result

        active_expr: Any = True
        mask_constraints: ConstraintConjunction = None
        mask_attr = getattr(expr, "mask", None)
        if mask_attr is not None:
            mask_result = self._safe_eval(mask_attr, f"{op_type.__name__} mask eval")
            if mask_result is None:
                return
            active_expr, mask_constraints = mask_result

        z3_constraints = _and_constraints(ptr_constraints, mask_constraints)
        source_location = capture_current_source_location()

        if not self.loop_stack:
            self._record_access_event(
                access_mode,
                op_type,
                z3_addr,
                z3_constraints,
                expr,
                source_location,
                copy_local_vars=tile_vars,
                active=active_expr,
            )
            return

        ctx = self.loop_stack[-1]
        signature = _make_event_signature(
            access_mode, source_location, z3_addr, z3_constraints, active_expr
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
                    active=active_expr,
                    copy_local_vars=tile_vars,
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
        if self._unsupported_capture or not self._capture_active():
            return
        # Loop check FIRST — before any .eval() can produce side effects
        # (ARANGE_DICT entries, fresh CAS-old vars, downstream sentinels).
        if self.loop_stack:
            self._raise_or_mark(
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
        if self._unsupported_capture or not self._capture_active():
            return
        # Loop check FIRST — see _handle_atomic_cas_check for rationale.
        if self.loop_stack:
            self._raise_or_mark(
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
    def _atomic_rmw_return_dtype(
        ptr_sym: SymbolicExpr | tuple[SymbolicExpr, ...],
        val_sym: SymbolicExpr | tuple[SymbolicExpr, ...],
    ) -> Any:
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
        if self._unsupported_capture:
            return
        # Items enqueued by _handle_access_check are PendingEvent instances
        # (subclass of PendingCheck) — narrow so attribute accesses are
        # type-safe under Literal["read", "write"].
        assert isinstance(pending, PendingEvent)
        # _loop_hook_after pops the flushed loop off loop_stack BEFORE calling
        # this, so _current_loop_iter_vars() only sees still-active outer
        # loops — the flushed loop's own iterator must come from ctx. Without
        # it the two-copy solver would not alpha-rename loop_i_* per copy,
        # pinning both program instances to the same iteration and missing
        # every cross-iteration cross-block race. The iterator's range
        # constraint travels alongside in iter_constraints -> premises, so
        # the renamed var stays bounded in each copy.
        self._record_access_event(
            pending.access_mode,
            pending.op_type,
            pending.addr_expr,
            pending.constraints,
            pending.symbolic_expr,
            pending.source_location,
            semantic_constraints=tuple(iter_constraints),
            copy_local_vars=(
                *pending.copy_local_vars,
                *self._current_loop_iter_vars(),
                ctx.idx_z3,
            ),
            active=pending.active,
            loop_flush=True,
        )


class NullRaceDetector(NullSymbolicClient, RaceDetector):
    """A do-nothing object returned when the race detector is off.
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
