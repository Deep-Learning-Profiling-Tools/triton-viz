"""Trace client for the compiled-mode sanitizer (``Sanitizer(compile=True)``).

Statically checks global-memory out-of-bounds from the kernel's TTIR,
acquired through the real compilation warmup, instantiated per launch with
the concrete tensor metadata and scalar argument values seen in
``arg_callback`` / ``grid_callback``. UNSAT over every access is a proof
that the launch is in-bounds for all inputs consistent with those scalar
values; SAT is a witness reported as an :class:`OutOfBoundsRecordZ3`,
matching the eager sanitizer's record/abort contract.

Data-dependent (indirect/gather) addressing, block pointers,
non-contiguous tensors, and nested loops are reported as
``last_status="unsupported"`` with empty records — NOT a silent "ok"
proof, but also NOT an automatic fallback. v1 does not run the
interpreter on unsupported kernels; to check them, run the eager
``Sanitizer()`` (which executes indices concretely) on the same kernel.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from ....core.callbacks import ForLoopCallbacks, OpCallbacks
from ....core.client import Client
from ....core.config import config as cfg
from ....core.data import Load, Op, Store
from ....utils.traceback_utils import location_to_traceback_info
from ..data import OutOfBoundsRecordZ3
from ..report import print_oob_record
from .oob import LaunchContext, TensorMeta, check_graph
from .ttir_reader import AccessGraph, UnsupportedTTIR, parse_ttir


class CompiledSanitizer(Client):
    """Static OOB sanitizer over compiled TTIR.

    Public surface mirrors the eager ``SymbolicSanitizer``:
      * ``records``: list of :class:`OutOfBoundsRecordZ3` (when
        ``abort_on_error=False``)
      * ``last_status``: ``"ok"`` (analysis ran; empty records is a proof) |
        ``"unsupported"`` (data-dependent / block-ptr / no driver) with
        ``unsupported_reason``
    """

    NAME = "sanitizer_compiled"
    LOG_TAG: ClassVar[str] = "CompiledSanitizer"
    LOG_VERB: ClassVar[str] = "analyzing"

    def __init__(self, abort_on_error: bool = True, **_ignored: Any) -> None:
        super().__init__()
        self.abort_on_error = abort_on_error
        self.records: list[OutOfBoundsRecordZ3] = []
        self.last_status: str = "ok"
        self.unsupported_reason: str | None = None
        # TTIR-hash -> parsed AccessGraph (or None if unsupported).
        self._graph_cache: dict[int, AccessGraph | None] = {}
        self._reset_launch()  # also initializes self._pending_ttir = None

    def _reset_launch(self) -> None:
        self._tensor_meta: dict[str, TensorMeta] = {}
        self._tensor_obj: dict[str, torch.Tensor] = {}
        self._params: dict[str, int] = {}
        self._grid: tuple[int, int, int] = (1, 1, 1)
        # _pending_ttir is the CURRENT launch's captured TTIR — it must not
        # survive into the next launch. The parsed-graph cache (_graph_cache,
        # keyed by TTIR hash) is what persists across launches; the pending
        # input does not. Without this, a later launch whose warmup produces
        # no TTIR would re-analyze a previous kernel's graph against the
        # current launch's metadata (wrong locs, or a wrong-graph false
        # verdict). Cleared at launch teardown (finalize's finally) so a
        # no-TTIR launch correctly falls to "unsupported".
        self._pending_ttir: str | None = None

    # ── compilation hooks: grab the runtime's own TTIR ────────────────

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        # Start each launch's TTIR capture fresh (belt-and-suspenders with the
        # finalize teardown): if this warmup yields no TTIR, finalize sees
        # None and reports unsupported instead of reusing a stale graph.
        self._pending_ttir = None
        return True

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        asm = getattr(ret, "asm", None)
        if asm and "ttir" in asm:
            self._pending_ttir = asm["ttir"]

    # ── per-launch metadata collection ────────────────────────────────

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        # arg_callback runs BEFORE grid_callback (frontend/triton.py loops
        # args, then resolves+reports the grid), so tensor/param metadata is
        # already collected by now — must NOT clear it here. Per-launch state
        # is cleared at the end of finalize() instead.
        self.records = []
        self.last_status = "ok"
        self.unsupported_reason = None
        g = tuple(int(x) for x in grid) + (1, 1, 1)
        self._grid = (g[0], g[1], g[2])

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        pass

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        # Constexpr: arg_cvt passes through unchanged (already folded into
        # TTIR constants, not a tt.func arg).
        if arg_cvt is arg and not hasattr(arg, "data_ptr"):
            return
        # Unwrap descriptor-style wrappers (mirrors SymbolicClient).
        if hasattr(arg, "base") and hasattr(arg.base, "data_ptr"):
            arg = arg.base
        if hasattr(arg, "data_ptr"):
            self._tensor_obj[name] = arg
            self._tensor_meta[name] = TensorMeta(
                numel=int(arg.numel()),
                elem_bits=int(arg.element_size()) * 8,
                data_ptr=int(arg.data_ptr()),
                contiguous=bool(arg.is_contiguous()),
            )
        elif isinstance(arg, bool):
            self._params[name] = int(arg)
        elif isinstance(arg, int):
            self._params[name] = arg
        # Non-int scalars (floats) are not address structure; ignore.

    def pre_run_callback(self, fn: Callable) -> bool:
        # Static analysis needs no interpreted execution; skip every block.
        # CAVEAT (same as the compiled race detector): pre_run is all()-
        # combined, so run this client standalone, not alongside an
        # interpreting client in one trace().
        return False

    def post_run_callback(self, fn: Callable) -> bool:
        return False

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        return OpCallbacks()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks()

    # ── analysis ──────────────────────────────────────────────────────

    def _graph(self) -> AccessGraph | None:
        if self._pending_ttir is None:
            return None
        key = hash(self._pending_ttir)
        if key not in self._graph_cache:
            try:
                self._graph_cache[key] = parse_ttir(self._pending_ttir)
            except UnsupportedTTIR as exc:
                self._graph_cache[key] = None
                self.unsupported_reason = str(exc)
        return self._graph_cache[key]

    def finalize(self) -> list:
        try:
            return self._finalize_inner()
        finally:
            self._reset_launch()

    def _finalize_inner(self) -> list:
        if self._pending_ttir is None:
            self.last_status = "unsupported"
            self.unsupported_reason = (
                "no TTIR captured from warmup; compiled-mode analysis did "
                "not run (driverless environment?)"
            )
            return []

        graph = self._graph()
        if graph is None:
            self.last_status = "unsupported"
            if self.unsupported_reason is None:
                self.unsupported_reason = "TTIR could not be analyzed"
            return []

        ctx = LaunchContext(
            grid=self._grid, params=self._params, tensors=self._tensor_meta
        )
        try:
            violations = check_graph(graph, ctx)
        except UnsupportedTTIR as exc:
            self.last_status = "unsupported"
            self.unsupported_reason = str(exc)
            return []

        self.last_status = "ok"
        for v in violations:
            self._emit(graph, v)
        return list(self.records)

    def _emit(self, graph: AccessGraph, v: Any) -> None:
        op_type: type[Load] | type[Store] = Store if v.kind == "store" else Load
        tensor = self._tensor_obj.get(v.base_param)
        if v.loc_file is not None and v.loc_line is not None:
            tb = [
                location_to_traceback_info((v.loc_file, v.loc_line, graph.kernel_name))
            ]
        else:
            tb = []
        record = OutOfBoundsRecordZ3(
            op_type=op_type,
            tensor=tensor,
            user_code_tracebacks=tb,
            constraints=None,
            violation_address=v.violation_address,
            symbolic_expr=None,
            tensor_name=v.base_param,
        )
        if self.abort_on_error:
            print_oob_record(record)
            sys.exit(1)
        if cfg.verbose:
            print_oob_record(record)
        self.records.append(record)
