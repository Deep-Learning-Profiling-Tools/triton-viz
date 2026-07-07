"""Trace client for the compiled-mode race detector.

Acquires TTGIR through the REAL compilation warmup (``pre_warmup_callback``
returns True; ``post_warmup_callback`` receives the ``CompiledKernel`` whose
``.asm["ttgir"]`` is the runtime's own specialization — never a hand-built
ASTSource, which would miss the divisibility specialization and silently
analyze unpipelined IR). Analysis is cached per compiled-kernel hash.

The same warmup also captures ``.asm["ttir"]`` and parses it into the shared
:class:`AccessGraph` (global-memory access footprints, atomic RMW metadata).
This is the Track 2 capture front-end: graphs are parsed and cached per
specialization but not yet encoded — the global-memory race queries land in
a later step, and a TTIR parse failure never affects the TTGIR shared-memory
verdict (``last_status``).

The client registers no op overriders and needs nothing from the interpreted
grid run: ``pre_run_callback`` returns False to skip each block's body
entirely (the static analysis works off the warmup TTGIR alone). Because
``ClientManager.pre_run_callback`` all()-combines every client's vote, that
False would suppress a co-registered client's capture — so this client is
STANDALONE: ``ClientManager.add_clients`` rejects composing it with any other
client (see ``STANDALONE`` below). Run the dynamic and compiled detectors as
separate ``@triton_viz.trace`` decorations.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any, ClassVar

from ....core.callbacks import ForLoopCallbacks, OpCallbacks
from ....core.client import Client
from ....core.config import config as cfg
from ....core.data import Op
from ...common.ttir_reader import AccessGraph, UnsupportedTTIR, parse_ttir
from .smt_encoder import AnalysisResult, analyze_ttgir

_RE_TTGIR_FUNC = re.compile(r"tt\.func\s+\w+\s+@(\w+)\(")


def _kernel_name(ttgir: str) -> str:
    m = _RE_TTGIR_FUNC.search(ttgir)
    return m.group(1) if m else "<kernel>"


class CompiledRaceDetector(Client):
    """Static shared-memory race detector over compiled TTGIR.

    Public surface mirrors the dynamic ``RaceDetector``:
      * ``last_reports``: list of :class:`CompiledRaceReport`
      * ``last_status``: ``"ok"`` (analysis ran; an empty list means no
        wait-coverage violation was found for the analyzed specializations,
        within the model boundary documented in ``hb.py``) |
        ``"unsupported"`` | ``"no_ttgir"``
      * ``unsupported_reason``

    Standalone-only: skips the interpreted run, so ``ClientManager`` refuses to
    compose it with other clients (see the module docstring).
    """

    NAME = "race_detector_compiled"
    LOG_TAG: ClassVar[str] = "CompiledRaceDetector"
    LOG_VERB: ClassVar[str] = "analyzing"
    # Skips the interpreted run via pre_run_callback() == False; ClientManager
    # enforces that this is the only client in the trace.
    STANDALONE: ClassVar[bool] = True
    # The analysis consumes only the warmup compilation artifact (TTGIR), so
    # TritonTrace.run skips the interpreter machinery entirely and executes
    # the real kernel — the host script keeps its true semantics.
    WARMUP_ONLY: ClassVar[bool] = True

    def __init__(self, collect_smtlib: bool = False) -> None:
        super().__init__()
        self.collect_smtlib = collect_smtlib
        self.last_reports: list[Any] = []
        self.last_status: str = "ok"
        self.unsupported_reason: str | None = None
        self.smtlib: list[str] = []
        self._pending_ttgir: list[str] = []
        # ttgir content digest -> AnalysisResult; one analysis per
        # specialization. Keyed by a stable SHA-256 of the TTGIR text rather
        # than Python's built-in hash(): the cache holds proof/witness verdicts
        # (a soundness boundary), so it must be collision-resistant and
        # reproducible, not process-randomized.
        self._analysis_cache: dict[str, AnalysisResult] = {}
        # Track 2 (global memory over TTIR) capture: per-launch pending texts,
        # a per-specialization parse cache (same SHA-256 rationale — parsed
        # footprints will back race verdicts), and the last launch's parse
        # results as parallel lists (graph, or None + the unsupported reason).
        self._pending_ttir: list[str] = []
        self._ttir_graph_cache: dict[str, tuple[AccessGraph | None, str | None]] = {}
        self.last_ttir_graphs: list[AccessGraph | None] = []
        self.last_ttir_unsupported: list[str | None] = []

    # ── compilation hooks ─────────────────────────────────────────────

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return True  # force the real compile so TTGIR exists

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        asm = getattr(ret, "asm", None)
        if not asm:
            return
        if "ttgir" in asm:
            self._pending_ttgir.append(asm["ttgir"])
        if "ttir" in asm:
            self._pending_ttir.append(asm["ttir"])

    # ── interpreted-run hooks (analysis needs none of this) ───────────

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        pass

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        # NOTE: the warmup-only production path never runs the interpreted
        # grid loop, so this callback is NOT a reliable per-launch reset
        # point — finalize() owns the resets. These stay only for the
        # composed/interpreted path's mid-launch consistency.
        self.last_reports = []
        self.last_status = "ok"
        self.unsupported_reason = None

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        pass

    def pre_run_callback(self, fn: Callable) -> bool:
        # The static analysis never needs the interpreted kernel body, and
        # executing it concretely can fail on constructs only the symbolic
        # clients' loop machinery handles (e.g. range(tl.cdiv(...)) bounds).
        # Returning False skips every block's body. Because pre_run is
        # all()-combined across clients, this would suppress a co-registered
        # client's capture — which is exactly why STANDALONE is set and
        # ClientManager.add_clients refuses to compose this client with others.
        return False

    def post_run_callback(self, fn: Callable) -> bool:
        # any()-combined: False lets the grid loop stop early when the
        # interpreter consults it.
        return False

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        return OpCallbacks()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks()

    # ── analysis ──────────────────────────────────────────────────────

    def _consume_pending_ttir(self) -> None:
        """Parse this launch's TTIR into AccessGraphs (Track 2 capture).

        Resets ``last_ttir_*`` first: finalize() is the per-launch reset
        point (grid_callback never fires on the warmup-only path). Failures
        are recorded per kernel in ``last_ttir_unsupported`` and never
        escalate to ``last_status`` — the TTGIR shared-memory verdict is
        independent of the global-memory front-end. Nothing raised here may
        escape: finalize runs in the trace teardown of the user's real
        launch.
        """
        self.last_ttir_graphs = []
        self.last_ttir_unsupported = []
        for text in self._pending_ttir:
            # errors="replace": a hash key must never raise (lone surrogates
            # in a hostile string would otherwise escape finalize).
            key = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            if key not in self._ttir_graph_cache:
                try:
                    self._ttir_graph_cache[key] = (parse_ttir(text), None)
                except UnsupportedTTIR as e:
                    # "kind: message" — the stable kind prefix is what the
                    # hybrid tier selector will route on (indirect-address →
                    # interpreter front-end) and what the evaluation buckets.
                    self._ttir_graph_cache[key] = (None, f"{e.kind}: {e}")
                except Exception as e:  # noqa: BLE001
                    # Reader bug or printer drift: degrade to unsupported,
                    # never crash the launch.
                    self._ttir_graph_cache[key] = (None, f"{type(e).__name__}: {e}")
            graph, reason = self._ttir_graph_cache[key]
            self.last_ttir_graphs.append(graph)
            self.last_ttir_unsupported.append(reason)
        self._pending_ttir = []

    def finalize(self) -> list:
        # Per-launch reset point (see grid_callback note): smtlib is extended
        # below, so it must be cleared here or it accumulates across launches
        # on the warmup-only path.
        self.smtlib = []
        self._consume_pending_ttir()
        if not self._pending_ttgir:
            # Warmup never delivered IR (e.g. driverless environment where
            # JITFunction.run could not bind a device). Distinguish from a
            # genuine proof.
            self.last_status = "no_ttgir"
            self.unsupported_reason = (
                "no TTGIR captured from warmup; compiled-mode analysis " "did not run"
            )
            self.last_reports = []
            if cfg.cli_active:
                print(f"[{self.LOG_TAG}] no TTGIR captured from warmup")
            return []

        reports: list[Any] = []
        status = "ok"
        reason: str | None = None
        for text in self._pending_ttgir:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            result = self._analysis_cache.get(key)
            if result is None:
                result = analyze_ttgir(text, collect_smtlib=self.collect_smtlib)
                self._analysis_cache[key] = result
            if result.status == "unsupported" and status == "ok":
                status = "unsupported"
                reason = result.unsupported_reason
            reports.extend(result.reports)
            self.smtlib.extend(result.smtlib)
            if cfg.cli_active:
                self._report_cli(_kernel_name(text), result)
        self._pending_ttgir = []

        self.last_reports = reports
        self.last_status = status
        self.unsupported_reason = reason
        return list(reports)

    def _report_cli(self, name: str, result: AnalysisResult) -> None:
        """Print a one-line verdict per analyzed kernel for the CLI tool."""
        if result.status == "unsupported":
            print(f"[{self.LOG_TAG}] {name}: UNSUPPORTED — {result.unsupported_reason}")
        elif result.reports:
            print(f"[{self.LOG_TAG}] {name}: RACE — {len(result.reports)} report(s)")
            for rep in result.reports:
                print(f"    {rep.render()}")
        else:
            print(f"[{self.LOG_TAG}] {name}: race-free (proof)")
