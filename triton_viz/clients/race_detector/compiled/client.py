"""Trace client for the compiled-mode race detector.

Acquires TTGIR through the REAL compilation warmup (``pre_warmup_callback``
returns True; ``post_warmup_callback`` receives the ``CompiledKernel`` whose
``.asm["ttgir"]`` is the runtime's own specialization — never a hand-built
ASTSource, which would miss the divisibility specialization and silently
analyze unpipelined IR). Analysis is cached per compiled-kernel hash.

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

from collections.abc import Callable
from typing import Any, ClassVar

from ....core.callbacks import ForLoopCallbacks, OpCallbacks
from ....core.client import Client
from ....core.data import Op
from .smt_encoder import AnalysisResult, analyze_ttgir


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

    def __init__(self, collect_smtlib: bool = False) -> None:
        super().__init__()
        self.collect_smtlib = collect_smtlib
        self.last_reports: list[Any] = []
        self.last_status: str = "ok"
        self.unsupported_reason: str | None = None
        self.smtlib: list[str] = []
        self._pending_ttgir: list[str] = []
        # ttgir-hash -> AnalysisResult; one analysis per specialization.
        self._analysis_cache: dict[int, AnalysisResult] = {}

    # ── compilation hooks ─────────────────────────────────────────────

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return True  # force the real compile so TTGIR exists

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        asm = getattr(ret, "asm", None)
        if not asm or "ttgir" not in asm:
            return
        self._pending_ttgir.append(asm["ttgir"])

    # ── interpreted-run hooks (analysis needs none of this) ───────────

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        pass

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self.last_reports = []
        self.last_status = "ok"
        self.unsupported_reason = None
        self.smtlib = []

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

    def finalize(self) -> list:
        if not self._pending_ttgir:
            # Warmup never delivered IR (e.g. driverless environment where
            # JITFunction.run could not bind a device). Distinguish from a
            # genuine proof.
            self.last_status = "no_ttgir"
            self.unsupported_reason = (
                "no TTGIR captured from warmup; compiled-mode analysis " "did not run"
            )
            self.last_reports = []
            return []

        reports: list[Any] = []
        status = "ok"
        reason: str | None = None
        for text in self._pending_ttgir:
            key = hash(text)
            result = self._analysis_cache.get(key)
            if result is None:
                result = analyze_ttgir(text, collect_smtlib=self.collect_smtlib)
                self._analysis_cache[key] = result
            if result.status == "unsupported" and status == "ok":
                status = "unsupported"
                reason = result.unsupported_reason
            reports.extend(result.reports)
            self.smtlib.extend(result.smtlib)
        self._pending_ttgir = []

        self.last_reports = reports
        self.last_status = status
        self.unsupported_reason = reason
        return list(reports)
