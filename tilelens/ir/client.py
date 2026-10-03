"""The IR client base (the L5 layer): lifecycle only, no analysis defaults.

An ``IRClient`` takes no part in the interpreted run: its interpreter-path
methods are inert and it declares ``NEEDS_INTERPRETER = False``, so the core
hands it compiled kernels through ``before_launch`` / ``compile_failed``,
which fill its per-launch ``ArtifactLog``. ``finalize`` is a template:

1. the D10b version gate: outside the tested Triton window (and without
   ``TILELENS_IR_ALLOW_UNTESTED_TRITON=1``), ``on_refusal`` gets a
   ``Refusal`` of kind ``"untested-triton-version"``;
2. otherwise ``analyze_launch(log)`` returns the reports and the verdict,
   also for a launch nothing could be captured for (see analyze_launch);
3. an ``Exception`` from either goes to ``on_analysis_error``, which returns
   the verdict instead (an interrupt or ``SystemExit`` propagates).

It returns the reports followed by the verdict, which ``ClientManager``
puts into ``Launch.records``, and keeps the verdict as ``last_verdict``
(None until a launch finalizes). A subclass declares ``NAME``,
``IR_STAGES`` and ``LAUNCH``, may set ``ir_target`` (the target its kernels
are compiled for on the host, D26; the configured default otherwise) and
implements the three hooks; statuses, refusal meanings, caches and report
printing are all its own.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from ..core.callbacks import ForLoopCallbacks, OpCallbacks
from ..core.client import Client, LaunchCall, LaunchEvent
from ..core.config import TESTED_TRITON_VERSIONS, untested_triton_version
from ..core.data import Op
from .capture import ArtifactLog
from .verdict import IRVerdict, Refusal


class IRClient(Client):
    NEEDS_INTERPRETER: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.artifacts = ArtifactLog(self.IR_STAGES)
        # Compatibility view of the last finalized launch's verdict.
        self.last_verdict: IRVerdict | None = None

    # ── the client's analysis ────────────────────────────────────────

    @abstractmethod
    def analyze_launch(self, log: ArtifactLog) -> tuple[list, IRVerdict]:
        """Analyze one traced launch: the reports and the verdict.

        ``log.call`` is the launch's LaunchCall. When ``log.call.capture`` is
        False, nothing was compiled or recorded: the trace has no JITFunction
        (``log.call.jit_fn`` is None: TRITON_INTERPRET, an InterpretedFunction
        runner, Gluon, NKI; the other cause, an untested Triton, is refused by
        the version gate first). An empty log then says nothing about the
        kernel; what such a launch gets is the subclass's call.
        """

    @abstractmethod
    def on_analysis_error(self, exc: Exception) -> IRVerdict:
        """The verdict for a launch whose analysis raised ``exc``."""

    @abstractmethod
    def on_refusal(self, refusal: Refusal) -> IRVerdict:
        """The verdict for a launch the base refused to analyze (kind
        ``"untested-triton-version"``)."""

    # ── launch lifecycle ─────────────────────────────────────────────
    # A subclass overriding one of these calls super().

    def begin_launch(self, call: LaunchCall) -> None:
        self.artifacts.reset(call)
        self.last_verdict = None

    def abort_launch(self, exc: BaseException) -> None:
        self.artifacts.reset()

    def before_launch(self, event: LaunchEvent) -> None:
        self.artifacts.record(event)

    def compile_failed(self, event: LaunchEvent) -> None:
        self.artifacts.record_failure(event)

    def finalize(self) -> list:
        try:
            reports, verdict = self._verdict()
        finally:
            # The log holds compile exceptions (and their frames); the next
            # launch starts from a fresh one anyway.
            self.artifacts.reset()
        self.last_verdict = verdict
        return [*reports, verdict]

    def _verdict(self) -> tuple[list, IRVerdict]:
        try:
            version = untested_triton_version()
            if version is not None:
                tested = ", ".join(f"{v}.x" for v in TESTED_TRITON_VERSIONS)
                return [], self.on_refusal(
                    Refusal(
                        kind="untested-triton-version",
                        message=(
                            f"IR mode is tested on Triton {tested}, not {version}; "
                            "set TILELENS_IR_ALLOW_UNTESTED_TRITON=1 to run it anyway"
                        ),
                    )
                )
            reports, verdict = self.analyze_launch(self.artifacts)
            return list(reports), verdict
        except Exception as exc:
            return [], self.on_analysis_error(exc)

    # ── inert interpreter path: the core calls none of these for an IR
    # client except the warmup vote, which declines (IR compiles go through
    # ir_capture) ──

    def pre_run_callback(self, fn: Callable) -> bool:
        return False

    def post_run_callback(self, fn: Callable) -> bool:
        return False

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        pass

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        pass

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        return OpCallbacks()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks()

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass
