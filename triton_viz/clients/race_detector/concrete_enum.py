"""Route 1: per-instance concrete footprint enumeration (ladder level L1).

The BOTTOM rung of the concretization ladder. It runs only when every
symbolic rung has refused (the harness's third-track gate, see
``evaluation/harness.run_one``) and the launch's tensor arguments and
concrete grid are available. It evaluates the kernel CONCRETELY for every
program instance of the grid under the Triton interpreter (no Z3, no
symbols): every branch takes the truth value the instance's data gives it,
every loop runs its concrete trip count, every indirect address is
computed from the actual loaded values, every mask evaluates per lane.

What is recorded (``ConcreteFootprintRecorder``): per instance and per
executed load / store / atomic OPERATION, the active lanes' byte intervals
``[addr, addr + elem)`` with lane multiplicity preserved (contiguous lanes
coalesce; a duplicate position inside one plain store is the A1 shape and
is reported), plus the operation's kind, element width, atomic scope and
source location.

What is decided (``analyze``): the model's conflict predicate, mirrored
from ``hb_common.conflicting_access_modes`` and the two-copy solver's
``_byte_overlap``:

* across DISTINCT instances, two accesses race when their byte intervals
  overlap, at least one writes (a plain store, or an atomic), and they are
  not a compatible atomic pair (both atomic, same width, exactly the same
  start address, and neither side ``cta``-scoped: a cta-scoped atomic is
  never compatible across instances). A plain access overlapping another
  instance's atomic IS a race. No happens-before edge exists for this
  rung: every cross-instance synchronization shape the model knows is a
  disqualifier (below).
* within ONE instance, program order orders every access of an earlier
  operation before every access of a later one, so cross-operation
  overlaps are never races; the element accesses of ONE tile operation are
  mutually unordered, so two lanes of one plain store at overlapping bytes
  race (the duplicate-position query). Duplicate lanes of one atomic
  serialize, duplicate lanes of one load read-read: neither races.

The claim: ``proved`` / ``races`` at the ANALYZED-LAUNCH extent (these
scalar arguments, this grid, these tensor contents), the same strength as
the interpreter frontend's ``proved@interp`` / ``race@interp``, under the
value-source premise (A2, extended): every load whose value reaches a
footprint-determining position (an address, a mask, a host-side branch, a
loop bound) must read bytes no OTHER instance writes. The premise is
enforced by CONCRETE TAINT: every interpreter value carries the set of
load operations it derives from (plus an ``atomic-return`` marker), and
taint also flows THROUGH MEMORY within an instance: a store records the
taint of the value it writes, and a later same-instance load of those
bytes inherits it (an atomic return relayed through scratch memory still
refuses; a relayed loaded value makes the original load a value source
too). A value-source load whose interval overlaps another instance's
write footprint refuses by name after the run. Instances execute
sequentially on one cloned copy of the tensors, which is exact under
that premise: a footprint can only depend on memory through a
value-source load, and such a load reads either the initial contents or
its own instance's program-ordered earlier writes, in the sequential run
exactly as in every real execution. Plain-data loads are unrestricted,
and their cross-instance overlaps with writes are reported as races.

Disqualifiers, each refusing BY NAME (``"<kind>: detail"``), never
silently:

  atomic-return     an atomic return value reaches an address, a mask, a
                    host-side branch, or a loop bound (ticket and
                    last-block idioms, spins on atomic polls): footprints
                    are not per-instance determined.
  value-source      a value-source load overlaps another instance's
                    write footprint (the A2 premise), including a spin
                    on a plain-loaded flag.
  projected-cost    after the first instance and a grace period of
                    ``ENUM_PROJECTION_GRACE_S``, the running mean
                    per-instance time (first instance excluded) times
                    the remaining instances, plus the time already
                    spent, exceeds ``ENUM_PROJECTION_FACTOR`` times the
                    caller's budget. A heuristic that trades a possible
                    proof for a fast abstention (a heavy first stretch
                    mis-projects a light remainder); the factor keeps
                    modest over-estimates running; never a verdict, and
                    the watchdog stays the bound when the projection
                    under-estimates.
  instance-ceiling  the grid has more than ``ENUM_MAX_INSTANCES``
                    instances (refused before executing anything: per-
                    instance execution cannot be vectorized across
                    instances, so the ceiling is a structural fact stated
                    by name, like the solver's ``ENUM_MAX_CASES``).
  no-grid           the launch grid is not a concrete tuple.
  no-contents       fake-tensor storage (no memory to evaluate against).
  scope             an atomic carries a memory scope outside cta/gpu/sys.
  timeout           the watchdog fired (a spin the taint did not see, or
                    a launch too slow for the budget).
  interpreter-error the interpreter raised inside the kernel.

There is no time budget in the rung itself beyond the watchdog the caller
configures; the per-row harness budget is evaluation protocol.
"""

from __future__ import annotations

import heapq
import importlib
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ...core.callbacks import ForLoopCallbacks, OpCallbacks
from ...core.client import Client
from ...core.config import config as cfg
from ...core.data import AtomicCas, AtomicRMW, Load, Store
from ...core.patch import PatchOp
from ...utils.traceback_utils import (
    _is_framework_frame,
    capture_current_source_location,
)
from .data import RaceType

# Structural instance ceiling (paper repo design doc section 4): 488 of the
# pinned run's 492 abstaining rows have at most this many instances; the
# four above it (up to 2,031,616) refuse by name in the time it takes to
# read the grid.
ENUM_MAX_INSTANCES = 65536
# Wall-clock watchdog for one enumeration run (defense in depth against a
# spin the taint could not see; the disqualifiers are the deterministic
# protection). Same value as the C2 replay watchdog.
ENUM_TIMEOUT_S = 60
# Distinct (site, site, race type) witnesses reported before the sweep
# stops early; mirrors REPLAY_MAX_REPORTS.
ENUM_MAX_REPORTS = 8
# Projected-cost refusal (Hao, 2026-09-04): no refusal before this much
# of the run has elapsed, and the first instance never enters the mean
# (warm-up: AST rewrite caching, first allocations).
ENUM_PROJECTION_GRACE_S = 5.0
ENUM_PROJECTION_SKIP_FIRST = 1
# The projection refuses only when it exceeds this multiple of the
# budget (Hao, 2026-09-04): a run projected between one and two budgets
# keeps going and the watchdog stays the bound, so a modest
# over-estimate (a heavy first stretch, a light remainder) cannot lose a
# proof that would have finished; only a run projected far beyond the
# budget abstains early.
ENUM_PROJECTION_FACTOR = 2.0

_ATOMIC = -1  # taint marker: derived from an atomic return value
_TAINT_ATTR = "_tilerace_taint"

_KIND_LOAD = 0
_KIND_STORE = 1
_KIND_RMW = 2
_KIND_CAS = 3
_KIND_NAMES = ("load", "store", "atomic_rmw", "atomic_cas")
_SCOPE_CODES = {"gpu": 0, "sys": 0, "cta": 1}  # gpu and sys are mutually inclusive


class ConcreteEnumRefusal(Exception):
    """Raised inside the run when a disqualifier fires; carries the
    ``"<kind>: detail"`` reason string."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail

    @property
    def reason(self) -> str:
        return f"{self.kind}: {self.detail}"


# ─────────────────────────── reports ───────────────────────────


@dataclass(frozen=True)
class ConcreteAccess:
    """The record-like endpoint of a concrete witness (the harness reads
    ``.source_location`` and ``.access_mode`` exactly as it does from an
    ``AccessEventRecord``)."""

    source_location: tuple[str, int, str] | None
    access_mode: str  # "read" | "write"
    kind: str  # load | store | atomic_rmw | atomic_cas
    is_atomic: bool
    elem_size: int


@dataclass(frozen=True)
class _Endpoint:
    record: ConcreteAccess


@dataclass(frozen=True)
class ConcreteRaceReport:
    """Shape-compatible with ``RaceReport`` where the harness and the
    tests look (``first_record``, ``second_record``, ``race_type``,
    ``witness_addr``, ``witness_grid_a/b``), plus the overlapping byte
    range."""

    first: _Endpoint
    second: _Endpoint
    race_type_value: RaceType
    witness_addr: int
    witness_grid_a: tuple[int, int, int]
    witness_grid_b: tuple[int, int, int]
    byte_range: tuple[int, int]
    reason: str = ""
    model: dict[str, str] = field(default_factory=dict)

    @property
    def first_record(self) -> ConcreteAccess:
        return self.first.record

    @property
    def second_record(self) -> ConcreteAccess:
        return self.second.record

    @property
    def race_type(self) -> RaceType:
        return self.race_type_value


_CROSS_INSTANCE_REASON = (
    "concrete footprints of two program instances overlap with no "
    "happens-before edge (per-instance enumeration at the analyzed launch)"
)
_INTRA_OP_REASON = (
    "conflicting lanes of a single tile store touch the same bytes with no "
    "defined intra-instance order"
)


@dataclass
class EnumOutcome:
    """Result of ``enumerate_launch``. ``status`` mirrors the static
    track's vocabulary: ``"ok"`` (race-free at the analyzed launch),
    ``"races"`` (concrete witnesses in ``reports``), ``"unsupported"``
    (refused by name in ``reason``)."""

    status: str
    reason: str | None = None
    reports: list[ConcreteRaceReport] = field(default_factory=list)
    grid: tuple[int, int, int] | None = None
    n_instances: int = 0
    n_ops: int = 0
    n_value_source_loads: int = 0
    time_s: float = 0.0
    run_s: float = 0.0
    analyze_s: float = 0.0
    instance_s: float | None = None  # median per-instance interpreter time
    max_instance_s: float | None = None


# ─────────────────────────── taint helpers ───────────────────────────


def _tensor_handle_cls() -> type:
    from triton.runtime.interpreter import TensorHandle

    return TensorHandle


def _composite_handle_classes() -> tuple[type, ...]:
    """Interpreter handles that carry TensorHandles as components (block
    pointers, tensor descriptors): taint flows through their fields."""
    from triton.runtime import interpreter as interp_mod

    return tuple(
        cls
        for cls in (
            getattr(interp_mod, "BlockPointerHandle", None),
            getattr(interp_mod, "TensorDescHandle", None),
        )
        if cls is not None
    )


def _handle_of(value: Any) -> Any | None:
    """The interpreter ``TensorHandle`` behind a value (a ``tl.tensor`` or
    a bare handle), else None."""
    cls = _tensor_handle_cls()
    if isinstance(value, cls):
        return value
    try:
        inner = getattr(value, "handle", None)
    except Exception:  # noqa: BLE001  (tl.tuple raises ValueError on unknown attrs)
        return None
    if isinstance(inner, cls):
        return inner
    return None


def _taint_of(handle: Any) -> frozenset[int] | None:
    attr = getattr(handle, "attr", None)
    if not isinstance(attr, dict):
        return None
    return attr.get(_TAINT_ATTR)


def _tag(handle: Any, taint: frozenset[int]) -> None:
    attr = getattr(handle, "attr", None)
    if not isinstance(attr, dict):
        return
    existing = attr.get(_TAINT_ATTR)
    attr[_TAINT_ATTR] = taint if existing is None else (existing | taint)


def _iter_handles(obj: Any, depth: int = 0):
    """Yield every TensorHandle reachable from ``obj`` (through tl.tensor
    wrappers and list/tuple nesting)."""
    if obj is None or depth > 3:
        return
    h = _handle_of(obj)
    if h is not None:
        yield h
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_handles(item, depth + 1)
        return
    if isinstance(obj, _composite_handle_classes()):
        for attr in ("base", "shape", "strides", "offsets"):
            yield from _iter_handles(getattr(obj, attr, None), depth + 1)


def _collect_taint(objs) -> tuple[frozenset[int], bool]:
    """(union of known taints, whether some handle carried no tag)."""
    taint: frozenset[int] = frozenset()
    unknown = False
    for obj in objs:
        for h in _iter_handles(obj):
            t = _taint_of(h)
            if t is None:
                unknown = True
            elif t:
                taint = taint | t
    return taint, unknown


# ─────────────────────────── projected cost ───────────────────────────


def projected_cost_refusal(
    elapsed_s: float,
    instance_times: list[float],
    n_total: int,
    budget_s: float | None,
    *,
    grace_s: float = ENUM_PROJECTION_GRACE_S,
    skip_first: int = ENUM_PROJECTION_SKIP_FIRST,
    factor: float = ENUM_PROJECTION_FACTOR,
) -> str | None:
    """The projected-cost decision, pure so it can be pinned without a
    kernel: None to keep running, else the refusal detail. The mean is
    over the instances completed so far EXCLUDING the first
    ``skip_first`` (warm-up); nothing is decided before ``grace_s`` of
    run time, so a heavy leader instance is diluted by the light ones
    that follow it before the projection is trusted; and the refusal
    fires only when the projection exceeds ``factor`` times the budget
    (a projection between one and ``factor`` budgets keeps running with
    the watchdog as the bound)."""
    if budget_s is None:
        return None
    done = len(instance_times)
    if elapsed_s < grace_s or done <= skip_first:
        return None
    remaining = max(0, n_total - done)
    if remaining == 0:
        return None  # the run is complete; there is nothing to project
    sample = instance_times[skip_first:]
    mean = sum(sample) / len(sample)
    projected = elapsed_s + mean * remaining
    if projected <= factor * budget_s:
        return None
    return (
        f"{done} of {n_total} instances in {elapsed_s:.1f}s, mean "
        f"{mean * 1000:.1f} ms per instance after the first; projected "
        f"{projected:.0f}s exceeds {factor:g}x the {budget_s:.0f}s budget"
    )


# ─────────────────────────── the recorder ───────────────────────────


class _IntervalBuffer:
    """Append-only int64 columns (start, end, op) with doubling growth."""

    def __init__(self) -> None:
        self._cap = 1024
        self.starts = np.empty(self._cap, dtype=np.int64)
        self.ends = np.empty(self._cap, dtype=np.int64)
        self.ops = np.empty(self._cap, dtype=np.int64)
        self.n = 0

    def append(self, starts: np.ndarray, ends: np.ndarray, op_id: int) -> None:
        k = int(starts.size)
        if k == 0:
            return
        while self.n + k > self._cap:
            self._cap *= 2
            for name in ("starts", "ends", "ops"):
                old = getattr(self, name)
                new = np.empty(self._cap, dtype=np.int64)
                new[: self.n] = old[: self.n]
                setattr(self, name, new)
        self.starts[self.n : self.n + k] = starts
        self.ends[self.n : self.n + k] = ends
        self.ops[self.n : self.n + k] = op_id
        self.n += k

    def view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.starts[: self.n], self.ends[: self.n], self.ops[: self.n]


class ConcreteFootprintRecorder(Client):
    """Interpreter client for the L1 rung: runs EVERY block sequentially,
    records per-operation byte footprints with lane multiplicity, carries
    concrete taint through every builder op, and refuses by name at the
    first disqualifier. Use it through ``enumerate_launch``."""

    NAME = "concrete_footprint_recorder"

    def __init__(self, budget_s: float | None = None) -> None:
        super().__init__()
        # the caller's wall-clock budget for the run (the watchdog's
        # value); drives the projected-cost refusal, None disables it
        self.budget_s = budget_s
        # per-op metadata, parallel lists indexed by op id
        self.op_pid_index: list[int] = []
        self.op_seq: list[int] = []
        self.op_kind: list[int] = []
        self.op_elem: list[int] = []
        self.op_scope: list[int] = []  # _SCOPE_CODES; 0 for plain accesses
        self.op_site: list[int] = []  # interned site id
        self.op_lanes: list[int] = []
        self.op_value_source: list[bool] = []
        # taint of the value a store wrote (atomics: the atomic marker;
        # loads: None): taint through memory within an instance
        self.op_store_taint: list[frozenset[int] | None] = []
        self.intervals = _IntervalBuffer()
        self.sites: list[tuple[str, int, str] | None] = []
        self._site_ids: dict[Any, int] = {}
        self.pids: list[tuple[int, int, int]] = []
        # intra-operation duplicate-position witnesses: (op_id, addr)
        self.intra_dups: list[tuple[int, int]] = []
        self.instance_times: list[float] = []
        self.grid: tuple[int, int, int] | None = None
        self.n_tensor_args = 0
        # per-instance state
        self._pid_index = -1
        self._seq = 0
        self._loads_in_instance: list[int] = []
        self._atomic_seen = False
        self._last_load_op_id: int | None = None
        self._synthesized_mask_pending = False
        self._pending_store_taint: frozenset[int] | None = None
        self._instance_t0 = 0.0
        self._run_t0 = 0.0
        # patch bookkeeping
        self._lang_patch_installed = False
        self._saved_attrs: list[tuple[Any, str, Any, bool]] = []
        self._builder_patch_installed = False

    # ── lifecycle ──────────────────────────────────────────────────
    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        if hasattr(arg, "data_ptr"):
            self.n_tensor_args += 1
        for h in _iter_handles(arg_cvt):
            _tag(h, frozenset())

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        g = tuple(int(d) for d in grid) + (1,) * (3 - len(grid))
        self.grid = (g[0], g[1], g[2])
        self._run_t0 = time.perf_counter()
        self._install_builder_patch()

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        pid = tuple(int(i) for i in grid_idx) + (0,) * (3 - len(grid_idx))
        self.pids.append((pid[0], pid[1], pid[2]))
        self._pid_index = len(self.pids) - 1
        self._seq = 0
        self._loads_in_instance = []
        self._atomic_seen = False
        self._last_load_op_id = None

    def pre_run_callback(self, fn: Callable) -> bool:
        self._instance_t0 = time.perf_counter()
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        now = time.perf_counter()
        self.instance_times.append(now - self._instance_t0)
        if self.budget_s is not None and self.grid is not None:
            detail = projected_cost_refusal(
                now - self._run_t0,
                self.instance_times,
                self.grid[0] * self.grid[1] * self.grid[2],
                self.budget_s,
            )
            if detail is not None:
                raise ConcreteEnumRefusal("projected-cost", detail)
        return True

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return False  # interpreter only; no real compile

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def finalize(self) -> list:
        self.cleanup()
        return []

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks(range_wrapper_factory=self._range_wrapper)

    def register_op_callback(
        self, op_type: type, *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        # register_op_callback runs BEFORE the language patch of this
        # launch, so the tensor-dunder hooks are installed here.
        self._install_lang_patch()
        table: dict[type, Callable[..., Any]] = {
            Load: self._pre_load,
            Store: self._pre_store,
            AtomicRMW: self._pre_atomic_rmw,
            AtomicCas: self._pre_atomic_cas,
        }
        # RawLoad/RawStore deliberately absent: the interpreter's
        # create_load/create_store synthesize an all-True mask and
        # delegate to the masked variants (which fire Load/Store), so
        # recording the raw hooks would double-count every unmasked
        # access; the builder wrapper tags the synthesized mask instead.
        cb = table.get(op_type)
        if cb is None:
            return OpCallbacks()
        return OpCallbacks(before_callback=cb)

    # ── patches: builder taint propagation and tensor dunders ─────────
    def _wrap_attr(self, obj: Any, name: str, kind: str | None) -> None:
        fn = getattr(obj, name)
        had_own = name in getattr(obj, "__dict__", {})
        recorder = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if name in ("create_load", "create_store"):
                # the interpreter synthesizes an all-True mask handle and
                # delegates to the masked variant: that mask is a
                # constant, not an unknown-provenance value
                recorder._synthesized_mask_pending = True
            elif name in ("create_masked_load", "create_masked_store"):
                mask_pos = 1 if name == "create_masked_load" else 2
                if recorder._synthesized_mask_pending and len(args) > mask_pos:
                    h = _handle_of(args[mask_pos])
                    if h is not None and _taint_of(h) is None:
                        _tag(h, frozenset())
                recorder._synthesized_mask_pending = False
            if kind == "store" and len(args) > 1:
                # the value about to be written: the Store callback
                # (fired inside fn) records it as the op's memory taint
                vt, vunknown = _collect_taint([args[1]])
                if vunknown:
                    vt = vt | recorder._unknown_taint()
                recorder._pending_store_taint = vt
            ret = fn(*args, **kwargs)
            taint, unknown = _collect_taint(list(args) + list(kwargs.values()))
            if unknown:
                taint = taint | recorder._unknown_taint()
            if kind == "load" and recorder._last_load_op_id is not None:
                taint = taint | frozenset((recorder._last_load_op_id,))
            elif kind == "atomic":
                taint = taint | frozenset((_ATOMIC,))
            for h in _iter_handles(ret):
                _tag(h, taint)
            return ret

        wrapper.__name__ = getattr(fn, "__name__", name)
        wrapper._tilerace_taint_wrapper = True  # type: ignore[attr-defined]
        setattr(obj, name, wrapper)
        self._saved_attrs.append((obj, name, fn, had_own))

    _LOAD_METHODS = frozenset(
        (
            "create_masked_load",
            "create_load",
            "create_tensor_pointer_load",
            "create_descriptor_load",
        )
    )
    _ATOMIC_METHODS = frozenset(("create_atomic_rmw", "create_atomic_cas"))
    _STORE_METHODS = frozenset(
        (
            "create_masked_store",
            "create_store",
            "create_tensor_pointer_store",
            "create_descriptor_store",
        )
    )

    def _install_builder_patch(self) -> None:
        """Wrap every public callable of the interpreter builder (bound
        methods AND the PatchOps triton_viz installed) plus the tl-level
        reduce/scan entry points so that taint(result) = union of the
        inputs' taints. Installed after the op patches (grid_callback),
        removed in cleanup()."""
        if self._builder_patch_installed:
            return
        import triton.language as tl
        from triton.runtime.interpreter import interpreter_builder

        for name in dir(interpreter_builder):
            if name.startswith("_") or name in ("set_grid_idx", "set_grid_dim"):
                continue
            try:
                attr = getattr(interpreter_builder, name)
            except Exception:  # noqa: BLE001
                continue
            if not callable(attr) or isinstance(attr, type):
                continue
            kind = None
            if name in self._LOAD_METHODS:
                kind = "load"
            elif name in self._ATOMIC_METHODS:
                kind = "atomic"
            elif name in self._STORE_METHODS:
                kind = "store"
            self._wrap_attr(interpreter_builder, name, kind)
        for cls in _composite_handle_classes():
            if callable(getattr(cls, "materialize_pointers", None)):
                self._wrap_attr(cls, "materialize_pointers", None)
        for ns in (tl, tl.core):
            for name in ("reduce", "associative_scan"):
                if callable(getattr(ns, name, None)):
                    self._wrap_attr(ns, name, None)
        for name in dir(tl):
            if isinstance(getattr(tl, name, None), PatchOp):
                self._wrap_attr(tl, name, None)
        self._builder_patch_installed = True

    def _install_lang_patch(self) -> None:
        """Hook ``tl.tensor.__bool__`` / ``__index__`` through the
        interpreter's own language patcher so the hooks survive the
        re-patch a tl.core-using helper triggers mid-launch."""
        if self._lang_patch_installed:
            return
        import triton.runtime.interpreter as interp_mod

        from ...core.frontend import triton as frontend_mod

        recorder = self
        orig_patch_tensor = interp_mod._patch_lang_tensor
        orig_index_patch = frontend_mod.TritonFrontend.__dict__[
            "_patch_numpy2_scalar_index"
        ]

        def _patch_lang_tensor(tensor: Any, scope: Any) -> None:
            orig_patch_tensor(tensor, scope)
            prev_bool = tensor.__bool__

            def __bool__(self_t: Any) -> bool:
                recorder._on_host_use(self_t, "a host-side branch", sys._getframe(1))
                return prev_bool(self_t)

            scope.set_attr(tensor, "__bool__", __bool__)

        def _patch_numpy2_scalar_index(scope: Any) -> None:
            orig_index_patch.__func__(scope)
            for tensor_cls in {tl_mod.tensor, tl_mod.core.tensor}:
                prev_index = tensor_cls.__index__

                def __index__(self_t: Any, _prev: Any = prev_index) -> int:
                    recorder._on_host_use(
                        self_t, "a host-side integer conversion", sys._getframe(1)
                    )
                    return _prev(self_t)

                scope.set_attr(tensor_cls, "__index__", __index__)

        import triton.language as tl_mod

        setattr(interp_mod, "_patch_lang_tensor", _patch_lang_tensor)  # noqa: B010
        setattr(  # noqa: B010
            frontend_mod.TritonFrontend,
            "_patch_numpy2_scalar_index",
            staticmethod(_patch_numpy2_scalar_index),
        )
        self._saved_attrs.append(
            (interp_mod, "_patch_lang_tensor", orig_patch_tensor, True)
        )
        self._saved_attrs.append(
            (
                frontend_mod.TritonFrontend,
                "_patch_numpy2_scalar_index",
                orig_index_patch,
                True,
            )
        )
        self._lang_patch_installed = True

    def cleanup(self) -> None:
        """Restore every attribute this recorder patched (idempotent)."""
        while self._saved_attrs:
            obj, name, original, had_own = self._saved_attrs.pop()
            try:
                if had_own:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
            except Exception:  # noqa: BLE001
                pass
        self._lang_patch_installed = False
        self._builder_patch_installed = False

    # ── taint sinks ────────────────────────────────────────────────
    def _unknown_taint(self) -> frozenset[int]:
        """Conservative taint for a value the wrappers did not see
        (constructed outside the builder): every load of this instance so
        far, plus the atomic marker when an atomic has executed."""
        taint = frozenset(self._loads_in_instance)
        if self._atomic_seen:
            taint = taint | frozenset((_ATOMIC,))
        return taint

    def _sink(self, handles: list[Any], position: str) -> None:
        taint, unknown = _collect_taint(handles)
        if unknown:
            taint = taint | self._unknown_taint()
        if _ATOMIC in taint:
            site = capture_current_source_location()
            raise ConcreteEnumRefusal(
                "atomic-return",
                f"an atomic return value reaches {position} at {_fmt_site(site)} "
                f"(instance {self._current_pid()}): the footprint is not "
                "per-instance determined",
            )
        for op_id in taint:
            if op_id >= 0:
                self.op_value_source[op_id] = True

    def _on_host_use(self, tensor: Any, position: str, caller: Any) -> None:
        if self._pid_index < 0 or _is_framework_frame(caller):
            return
        h = _handle_of(tensor)
        if h is None:
            return
        self._sink([h], position)

    def _range_wrapper(
        self,
        iterable: Any,
        loop_site: Any,
        range_type: str,
        args: tuple,
        kwargs: dict,
        iterable_callable: Callable,
    ) -> Any:
        handles = [
            h for a in list(args) + list(kwargs.values()) for h in _iter_handles(a)
        ]
        if handles:
            self._sink(handles, "a loop bound")
        return None  # evaluate the original iterable

    # ── recording ──────────────────────────────────────────────────
    def _current_pid(self) -> tuple[int, int, int]:
        return self.pids[self._pid_index] if self._pid_index >= 0 else (0, 0, 0)

    def _site_id(self, site: Any) -> int:
        sid = self._site_ids.get(site)
        if sid is None:
            sid = len(self.sites)
            self.sites.append(site)
            self._site_ids[site] = sid
        return sid

    @staticmethod
    def _normalize_scope(scope: Any) -> int:
        if scope is None:
            return _SCOPE_CODES["gpu"]
        name = str(getattr(scope, "name", scope)).lower()
        name = {"system": "sys"}.get(name, name)
        if name not in _SCOPE_CODES:
            raise ConcreteEnumRefusal("scope", f"unsupported memory scope {name!r}")
        return _SCOPE_CODES[name]

    def _record(self, kind: int, ptr: Any, mask: Any, scope: Any = None) -> None:
        if self._pid_index < 0:
            return
        sink_handles = [ptr]
        if _handle_of(mask) is not None:
            sink_handles.append(mask)
        position = "a memory address" if kind != _KIND_LOAD else "a load address"
        self._sink(sink_handles, position if mask is None else "an address or mask")
        data = np.asarray(ptr.data).reshape(-1)
        if mask is not None:
            raw = (
                mask.data if hasattr(mask, "data") and hasattr(mask, "dtype") else mask
            )
            m = np.broadcast_to(
                np.asarray(raw, dtype=bool), np.shape(ptr.data)
            ).reshape(-1)
            data = data[m]
        elem = max(1, int(ptr.get_element_ty().primitive_bitwidth) // 8)
        scope_code = (
            self._normalize_scope(scope) if kind in (_KIND_RMW, _KIND_CAS) else 0
        )
        op_id = len(self.op_kind)
        self.op_pid_index.append(self._pid_index)
        self.op_seq.append(self._seq)
        self._seq += 1
        self.op_kind.append(kind)
        self.op_elem.append(elem)
        self.op_scope.append(scope_code)
        self.op_site.append(self._site_id(capture_current_source_location()))
        self.op_lanes.append(int(data.size))
        self.op_value_source.append(False)
        if kind == _KIND_STORE:
            pending = self._pending_store_taint
            self._pending_store_taint = None
            self.op_store_taint.append(
                pending if pending is not None else self._unknown_taint()
            )
        elif kind in (_KIND_RMW, _KIND_CAS):
            self.op_store_taint.append(frozenset((_ATOMIC,)))
        else:
            self.op_store_taint.append(None)
        if kind == _KIND_LOAD:
            self._loads_in_instance.append(op_id)
            self._last_load_op_id = op_id
        else:
            self._last_load_op_id = None
        if kind in (_KIND_RMW, _KIND_CAS):
            self._atomic_seen = True
        if data.size:
            addrs = np.sort(data.astype(np.int64, copy=False))
            if kind == _KIND_STORE and addrs.size > 1:
                gaps = np.diff(addrs)
                dup = np.nonzero(gaps < elem)[0]
                if dup.size:
                    self.intra_dups.append((op_id, int(addrs[dup[0] + 1])))
            uniq = np.unique(addrs)
            if kind in (_KIND_RMW, _KIND_CAS) or uniq.size == 1:
                # atomics stay one interval per lane: the compatible-pair
                # judgment is per exact (address, width), so lanes must
                # never coalesce into a multi-element interval
                starts = uniq
                ends = uniq + elem
            else:
                brk = np.nonzero(uniq[1:] != uniq[:-1] + elem)[0]
                starts = uniq[np.concatenate(([0], brk + 1))]
                ends = uniq[np.concatenate((brk, [uniq.size - 1]))] + elem
            self.intervals.append(starts, ends, op_id)

    def _pre_load(
        self, ptr: Any, mask: Any, keys: Any = None, *a: Any, **k: Any
    ) -> None:
        if keys is not None:  # NKI frontend
            return
        self._record(_KIND_LOAD, ptr, mask)

    def _pre_store(
        self, ptr: Any, mask: Any, keys: Any = None, *a: Any, **k: Any
    ) -> None:
        if keys is not None:
            return
        self._record(_KIND_STORE, ptr, mask)

    def _pre_atomic_rmw(
        self,
        rmw_op: Any,
        ptr: Any,
        val: Any,
        mask: Any,
        sem: Any = None,
        scope: Any = None,
        *a: Any,
        **k: Any,
    ) -> None:
        self._record(_KIND_RMW, ptr, mask, scope=scope)

    def _pre_atomic_cas(
        self,
        ptr: Any,
        cmp: Any,
        val: Any,
        sem: Any = None,
        scope: Any = None,
        *a: Any,
        **k: Any,
    ) -> None:
        self._record(_KIND_CAS, ptr, None, scope=scope)


def _fmt_site(site: Any) -> str:
    if not site:
        return "<unknown site>"
    return f"{site[0]}:{site[1]}"


# ─────────────────────────── analysis ───────────────────────────


def _access_mode(kind: int) -> str:
    return "read" if kind == _KIND_LOAD else "write"


def _race_type(first_writes: bool, second_writes: bool) -> RaceType:
    if first_writes and second_writes:
        return RaceType.WAW
    if first_writes:
        return RaceType.RAW
    return RaceType.WAR


def _writes(kind: int) -> bool:
    return kind != _KIND_LOAD


def _is_atomic(kind: int) -> bool:
    return kind in (_KIND_RMW, _KIND_CAS)


class _Analyzer:
    def __init__(self, rec: ConcreteFootprintRecorder, max_reports: int) -> None:
        self.rec = rec
        self.max_reports = max_reports
        self.starts, self.ends, self.ops = rec.intervals.view()
        self.kind = np.asarray(rec.op_kind, dtype=np.int64)
        self.pid_index = np.asarray(rec.op_pid_index, dtype=np.int64)
        self.reports: list[ConcreteRaceReport] = []
        self._seen: set[tuple[int, int, str, bool]] = set()

    # ── witness construction ──
    def _access(self, op: int) -> ConcreteAccess:
        rec = self.rec
        k = rec.op_kind[op]
        return ConcreteAccess(
            source_location=rec.sites[rec.op_site[op]],
            access_mode=_access_mode(k),
            kind=_KIND_NAMES[k],
            is_atomic=_is_atomic(k),
            elem_size=rec.op_elem[op],
        )

    def _report(self, op_a: int, op_b: int, lo: int, hi: int, reason: str) -> bool:
        """Record a witness; return True when the report cap is reached."""
        rec = self.rec
        # program order first: (pid_index, seq)
        if (rec.op_pid_index[op_a], rec.op_seq[op_a]) > (
            rec.op_pid_index[op_b],
            rec.op_seq[op_b],
        ):
            op_a, op_b = op_b, op_a
        rt = _race_type(_writes(rec.op_kind[op_a]), _writes(rec.op_kind[op_b]))
        same_instance = rec.op_pid_index[op_a] == rec.op_pid_index[op_b]
        key = (rec.op_site[op_a], rec.op_site[op_b], rt.name, same_instance)
        if key in self._seen:
            return len(self.reports) >= self.max_reports
        self._seen.add(key)
        self.reports.append(
            ConcreteRaceReport(
                first=_Endpoint(self._access(op_a)),
                second=_Endpoint(self._access(op_b)),
                race_type_value=rt,
                witness_addr=int(lo),
                witness_grid_a=rec.pids[rec.op_pid_index[op_a]],
                witness_grid_b=rec.pids[rec.op_pid_index[op_b]],
                byte_range=(int(lo), int(hi)),
                reason=reason,
            )
        )
        return len(self.reports) >= self.max_reports

    # ── the value-source premise (A2) ──
    def value_source_violation(self) -> str | None:
        """The A2 premise, cross-instance: a value-source load must not
        overlap bytes another instance writes. Same-instance writes are
        program-ordered and deterministic; an EARLIER same-instance
        write relays its value's taint into the load (an atomic return
        refuses, a relayed loaded value makes the original load a value
        source, checked in turn); a LATER one cannot affect the value."""
        rec = self.rec
        worklist = [op for op, flag in enumerate(rec.op_value_source) if flag]
        if not worklist:
            return None
        write_mask = self.kind[self.ops] != _KIND_LOAD
        ws, we, wo = (
            self.starts[write_mask],
            self.ends[write_mask],
            self.ops[write_mask],
        )
        if ws.size == 0:
            return None
        order = np.argsort(ws, kind="stable")
        ws, we, wo = ws[order], we[order], wo[order]
        prefix_max_end = np.maximum.accumulate(we)
        # intervals are appended per operation in op-id order, so an op's
        # intervals are one contiguous slice of the buffer: locate them by
        # bisection instead of a full scan per load (the scan made the
        # check quadratic in the number of value-source loads)
        processed: set[int] = set()
        while worklist:
            load = worklist.pop()
            if load in processed:
                continue
            processed.add(load)
            rec.op_value_source[load] = True
            lo = int(np.searchsorted(self.ops, load, side="left"))
            hi = int(np.searchsorted(self.ops, load, side="right"))
            lpid, lseq = rec.op_pid_index[load], rec.op_seq[load]
            for s, e in zip(self.starts[lo:hi], self.ends[lo:hi]):
                hi = int(np.searchsorted(ws, e, side="left"))  # writes with start < e
                j = hi - 1
                while j >= 0 and prefix_max_end[j] > s:
                    if we[j] > s:
                        other = int(wo[j])
                        if rec.op_pid_index[other] != lpid:
                            return (
                                f"value-source: the load at {_fmt_site(rec.sites[rec.op_site[load]])} "
                                f"(instance {rec.pids[lpid]}) feeds an address, mask, "
                                f"branch, or loop bound and overlaps bytes written by the "
                                f"{_KIND_NAMES[rec.op_kind[other]]} at "
                                f"{_fmt_site(rec.sites[rec.op_site[other]])} (instance "
                                f"{rec.pids[rec.op_pid_index[other]]}): the read-only-inputs premise fails"
                            )
                        if rec.op_seq[other] < lseq:
                            relayed = rec.op_store_taint[other] or frozenset()
                            if _ATOMIC in relayed:
                                return (
                                    f"atomic-return: an atomic return value reaches a footprint "
                                    f"position through memory: stored by the "
                                    f"{_KIND_NAMES[rec.op_kind[other]]} at "
                                    f"{_fmt_site(rec.sites[rec.op_site[other]])}, loaded at "
                                    f"{_fmt_site(rec.sites[rec.op_site[load]])} (instance "
                                    f"{rec.pids[lpid]}): the footprint is not per-instance determined"
                                )
                            for src in relayed:
                                if src >= 0 and src not in processed:
                                    worklist.append(src)
                    j -= 1
        return None

    # ── intra-operation duplicate positions (the A1 shape) ──
    def intra_op_duplicates(self) -> bool:
        for op, addr in self.rec.intra_dups:
            elem = self.rec.op_elem[op]
            if self._report(op, op, addr, addr + elem, _INTRA_OP_REASON):
                return True
        return False

    # ── cross-instance sweep ──
    def cross_instance(self) -> None:
        if self.starts.size == 0 or len(self.reports) >= self.max_reports:
            return
        rec = self.rec
        order = np.lexsort((self.ends, self.starts))
        starts, ends, ops = self.starts[order], self.ends[order], self.ops[order]
        kinds = self.kind[ops]
        pids = self.pid_index[ops]
        # active plain stores / loads: heaps of (end, op, start)
        active_w: list[tuple[int, int]] = []
        active_r: list[tuple[int, int]] = []
        # active atomic buckets keyed by (start, elem): [end, pid_set, has_cta, ops]
        buckets: dict[tuple[int, int], list[Any]] = {}
        bucket_heap: list[tuple[int, tuple[int, int]]] = []
        n = int(starts.size)
        for i in range(n):
            s, e, op, k, pid = (
                int(starts[i]),
                int(ends[i]),
                int(ops[i]),
                int(kinds[i]),
                int(pids[i]),
            )
            while active_w and active_w[0][0] <= s:
                heapq.heappop(active_w)
            while active_r and active_r[0][0] <= s:
                heapq.heappop(active_r)
            while bucket_heap and bucket_heap[0][0] <= s:
                _, key = heapq.heappop(bucket_heap)
                buckets.pop(key, None)
            if k == _KIND_STORE:
                for end_j, op_j in active_w:
                    if pids_differ(pid, int(self.pid_index[op_j])):
                        if self._report(
                            op_j, op, s, min(e, end_j), _CROSS_INSTANCE_REASON
                        ):
                            return
                for end_j, op_j in active_r:
                    if pids_differ(pid, int(self.pid_index[op_j])):
                        if self._report(
                            op_j, op, s, min(e, end_j), _CROSS_INSTANCE_REASON
                        ):
                            return
                for (bs, be_elem), bucket in buckets.items():
                    if bs < e and bucket[0] > s:
                        for op_j in bucket[3]:
                            if pids_differ(pid, int(self.pid_index[op_j])):
                                if self._report(
                                    op_j,
                                    op,
                                    max(s, bs),
                                    min(e, bucket[0]),
                                    _CROSS_INSTANCE_REASON,
                                ):
                                    return
                                break
                heapq.heappush(active_w, (e, op))
            elif k == _KIND_LOAD:
                for end_j, op_j in active_w:
                    if pids_differ(pid, int(self.pid_index[op_j])):
                        if self._report(
                            op_j, op, s, min(e, end_j), _CROSS_INSTANCE_REASON
                        ):
                            return
                for (bs, be_elem), bucket in buckets.items():
                    if bs < e and bucket[0] > s:
                        for op_j in bucket[3]:
                            if pids_differ(pid, int(self.pid_index[op_j])):
                                if self._report(
                                    op_j,
                                    op,
                                    max(s, bs),
                                    min(e, bucket[0]),
                                    _CROSS_INSTANCE_REASON,
                                ):
                                    return
                                break
                heapq.heappush(active_r, (e, op))
            else:  # atomic
                elem = rec.op_elem[op]
                scope_cta = rec.op_scope[op] == _SCOPE_CODES["cta"]
                for end_j, op_j in active_w:
                    if pids_differ(pid, int(self.pid_index[op_j])):
                        if self._report(
                            op_j, op, s, min(e, end_j), _CROSS_INSTANCE_REASON
                        ):
                            return
                for end_j, op_j in active_r:
                    if pids_differ(pid, int(self.pid_index[op_j])):
                        if self._report(
                            op_j, op, s, min(e, end_j), _CROSS_INSTANCE_REASON
                        ):
                            return
                key = (s, elem)
                for bkey, bucket in buckets.items():
                    if bucket[0] <= s or bkey[0] >= e:
                        continue
                    if bkey == key:
                        # same address and width: compatible unless a cta
                        # scope is involved across distinct instances
                        if (scope_cta or bucket[2]) and any(
                            p != pid for p in bucket[1]
                        ):
                            for op_j in bucket[3]:
                                if pids_differ(pid, int(self.pid_index[op_j])):
                                    if self._report(
                                        op_j, op, s, e, _CROSS_INSTANCE_REASON
                                    ):
                                        return
                                    break
                        continue
                    # different start or width: torn atomics race like writes
                    for op_j in bucket[3]:
                        if pids_differ(pid, int(self.pid_index[op_j])):
                            if self._report(
                                op_j,
                                op,
                                max(s, bkey[0]),
                                min(e, bucket[0]),
                                _CROSS_INSTANCE_REASON,
                            ):
                                return
                            break
                existing = buckets.get(key)
                if existing is None:
                    buckets[key] = [e, {pid}, scope_cta, [op]]
                    heapq.heappush(bucket_heap, (e, key))
                else:
                    existing[1].add(pid)
                    existing[2] = existing[2] or scope_cta
                    if len(existing[3]) < 4:
                        existing[3].append(op)


def pids_differ(a: int, b: int) -> bool:
    return a != b


def analyze(
    rec: ConcreteFootprintRecorder, max_reports: int = ENUM_MAX_REPORTS
) -> EnumOutcome:
    """Decide the recorded launch: value-source premise first (a violation
    refuses the whole launch), then the duplicate-position query, then
    the cross-instance sweep."""
    t0 = time.perf_counter()
    an = _Analyzer(rec, max_reports)
    violation = an.value_source_violation()
    outcome = EnumOutcome(
        status="unsupported",
        grid=rec.grid,
        n_instances=len(rec.pids),
        n_ops=len(rec.op_kind),
        n_value_source_loads=sum(1 for f in rec.op_value_source if f),
    )
    if violation is not None:
        outcome.reason = violation
    else:
        if not an.intra_op_duplicates():
            an.cross_instance()
        outcome.reports = an.reports
        outcome.status = "races" if an.reports else "ok"
    if rec.instance_times:
        outcome.instance_s = statistics.median(rec.instance_times)
        outcome.max_instance_s = max(rec.instance_times)
    outcome.analyze_s = time.perf_counter() - t0
    return outcome


# ─────────────────────────── driver ───────────────────────────


def _concrete_grid(grid: Any) -> tuple[int, int, int] | None:
    if callable(grid) or grid is None:
        return None
    try:
        g = tuple(int(d) for d in grid)
    except (TypeError, ValueError):
        return None
    if not 1 <= len(g) <= 3 or any(d < 0 for d in g):
        return None
    g = g + (1,) * (3 - len(g))
    return (g[0], g[1], g[2])


def _translate_addr(addr: int, spans: list[tuple[int, int, int]]) -> int:
    for lo, hi, orig in spans:
        if lo <= addr < hi:
            return orig + (addr - lo)
    return addr


def _translate_report(
    rep: ConcreteRaceReport, spans: list[tuple[int, int, int]]
) -> ConcreteRaceReport:
    if not spans:
        return rep
    lo, hi = rep.byte_range
    return ConcreteRaceReport(
        first=rep.first,
        second=rep.second,
        race_type_value=rep.race_type_value,
        witness_addr=_translate_addr(rep.witness_addr, spans),
        witness_grid_a=rep.witness_grid_a,
        witness_grid_b=rep.witness_grid_b,
        byte_range=(_translate_addr(lo, spans), _translate_addr(lo, spans) + (hi - lo)),
        reason=rep.reason,
        model=rep.model,
    )


def enumerate_launch(
    jit_fn: Any,
    args: tuple,
    kwargs: dict,
    grid: Any,
    *,
    max_instances: int = ENUM_MAX_INSTANCES,
    timeout_s: float | None = ENUM_TIMEOUT_S,
    max_reports: int = ENUM_MAX_REPORTS,
) -> EnumOutcome:
    """Run the L1 rung on one launch. Every tensor argument is CLONED
    (the caller's tensors are never touched); instances execute
    sequentially (``cfg.num_sms`` forced to 1 for the run). Never raises:
    every failure is a named refusal in the outcome."""
    from .compiled.replay import _replay_watchdog

    t_start = time.perf_counter()
    g = _concrete_grid(grid)
    if g is None:
        return EnumOutcome(
            "unsupported", "no-grid: the launch grid is not a concrete tuple"
        )
    n = g[0] * g[1] * g[2]
    if n > max_instances:
        return EnumOutcome(
            "unsupported",
            f"instance-ceiling: {n} program instances exceed ENUM_MAX_INSTANCES={max_instances}",
            grid=g,
            n_instances=n,
        )
    if cfg.virtual_memory:
        return EnumOutcome(
            "unsupported",
            "no-contents: fake tensor storage (SANITIZER_ENABLE_FAKE_TENSOR) has no memory contents",
            grid=g,
        )

    trace_mod = importlib.import_module("triton_viz.core.trace")

    # Clone PER STORAGE, not per argument: aliased arguments (an in-place
    # kernel passes the same tensor, or two views of one storage, as two
    # parameters) must keep aliasing on the clones, or the enumeration
    # would evaluate a launch that never existed. Every argument becomes a
    # view on its storage's clone with the same offset/size/stride, which
    # is how the interpreter itself materializes host copies.
    # clone_spans: (clone storage base, clone storage end, original base)
    # translate witnesses back to the caller's addresses.
    clone_spans: list[tuple[int, int, int]] = []
    storage_clones: dict[int, Any] = {}

    def _clone(v: Any) -> Any:
        if not (hasattr(v, "data_ptr") and hasattr(v, "untyped_storage")):
            return v
        try:
            storage = v.untyped_storage()
            key = int(storage.data_ptr())
            cloned_storage = storage_clones.get(key)
            if cloned_storage is None:
                cloned_storage = storage.clone()
                storage_clones[key] = cloned_storage
                clone_spans.append(
                    (
                        int(cloned_storage.data_ptr()),
                        int(cloned_storage.data_ptr()) + int(storage.nbytes()),
                        key,
                    )
                )
            c = v.detach().new_empty(0)
            c.set_(cloned_storage, v.storage_offset(), v.size(), v.stride())
            return c
        except Exception:  # noqa: BLE001
            c = v.detach().clone()
            try:
                nbytes = int(c.numel()) * int(c.element_size())
                clone_spans.append(
                    (int(c.data_ptr()), int(c.data_ptr()) + nbytes, int(v.data_ptr()))
                )
            except Exception:  # noqa: BLE001
                pass
            return c

    cloned_args = tuple(_clone(a) for a in args)
    cloned_kwargs = {k: _clone(v) for k, v in kwargs.items()}

    recorder = ConcreteFootprintRecorder(budget_s=timeout_s)
    saved_num_sms = cfg.num_sms
    cfg.num_sms = 1
    n_before = len(trace_mod.launches)
    reason: str | None = None
    t_run = time.perf_counter()
    try:
        # register the kernel as the user-frame boundary so recorded
        # source locations are the kernel's own absolute lines (what the
        # trace() decorator does for the symbolic frontends)
        trace_mod.trace_source(jit_fn)
        traced = trace_mod.TritonTrace(jit_fn, recorder)
        if timeout_s is not None:
            with _replay_watchdog(timeout_s):
                traced[g](*cloned_args, **cloned_kwargs)
        else:
            traced[g](*cloned_args, **cloned_kwargs)
    except ConcreteEnumRefusal as r:
        reason = r.reason
    except TimeoutError as e:
        reason = f"timeout: concrete enumeration exceeded {timeout_s}s ({e})"
    except Exception as e:  # noqa: BLE001
        reason = f"interpreter-error: {type(e).__name__}: {e}"
    finally:
        run_s = time.perf_counter() - t_run
        del trace_mod.launches[n_before:]
        recorder.cleanup()
        cfg.num_sms = saved_num_sms

    if reason is None:
        # the analysis runs under the remaining budget too: a slow sweep
        # must end in a named refusal, never in a row-level timeout
        try:
            remaining = None if timeout_s is None else max(1.0, timeout_s - run_s)
            if remaining is not None:
                with _replay_watchdog(remaining):
                    outcome = analyze(recorder, max_reports=max_reports)
            else:
                outcome = analyze(recorder, max_reports=max_reports)
            outcome.reports = [
                _translate_report(r, clone_spans) for r in outcome.reports
            ]
        except TimeoutError as e:
            reason = f"timeout: footprint analysis exceeded the budget ({e})"
        except Exception as e:  # noqa: BLE001
            reason = f"analysis-error: {type(e).__name__}: {e}"
    if reason is not None:
        outcome = EnumOutcome(
            "unsupported",
            reason,
            grid=g,
            n_instances=len(recorder.pids),
            n_ops=len(recorder.op_kind),
        )
    if recorder.instance_times:
        outcome.instance_s = statistics.median(recorder.instance_times)
        outcome.max_instance_s = max(recorder.instance_times)
    outcome.run_s = run_s
    outcome.time_s = time.perf_counter() - t_start
    return outcome


__all__ = [
    "ENUM_MAX_INSTANCES",
    "ENUM_MAX_REPORTS",
    "ENUM_PROJECTION_FACTOR",
    "ENUM_PROJECTION_GRACE_S",
    "ENUM_PROJECTION_SKIP_FIRST",
    "ENUM_TIMEOUT_S",
    "ConcreteAccess",
    "ConcreteEnumRefusal",
    "ConcreteFootprintRecorder",
    "ConcreteRaceReport",
    "EnumOutcome",
    "analyze",
    "enumerate_launch",
    "projected_cost_refusal",
]
