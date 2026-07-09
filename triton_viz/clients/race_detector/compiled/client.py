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
from ..hb_common import UnsupportedSymbolicRaceQuery
from ..two_copy_symbolic_hb_solver import TwoCopySymbolicHBSolver
from .global_records import (
    GlobalTensor,
    encode_graph,
    encode_graph_t0,
    symbolic_grid,
    t0_linearity_gate,
)
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

    def __init__(
        self,
        collect_smtlib: bool = False,
        confirm_races: bool = True,
        differential_check: bool = False,
    ) -> None:
        super().__init__()
        self.collect_smtlib = collect_smtlib
        # C2: replay SAT witnesses under the interpreter to classify them
        # confirmed/unconfirmed. Costs a pre-launch tensor snapshot (capped)
        # and, only when a SAT exists, an interpreter run of two blocks.
        self.confirm_races = confirm_races
        # C3: opt-in per-launch differential cross-check — the static
        # model's concrete footprint vs the interpreter's, for a couple of
        # program ids. Results land in last_differential ([] = the lowering
        # and the interpreter agree).
        self.differential_check = differential_check
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
        # Global-memory verdict (independent of the TTGIR shared-memory
        # last_status): "ok" = proved race-free (see last_global_provenance
        # for the rung); "races" = definite reports in last_global_reports;
        # "unsupported"; "no_ttir".
        self.last_global_status: str = "ok"
        self.last_global_reason: str | None = None
        self.last_global_reports: list[Any] = []
        # The proof rung when status is "ok": "proved@T0" = for ANY scalar
        # params, any grid along the read axes — scoped to this
        # specialization and accepted only after the launch's captured
        # tensor intervals verified the non-aliasing premise;
        # "proved@T1" = for this launch's params, any grid.
        self.last_global_provenance: str | None = None
        # C2 aggregate over the replayed reports when status is "races":
        # "confirmed" | "unconfirmed" | "partial" | None (replay off or
        # unavailable).
        self.last_global_confirmation: str | None = None
        # C3 mismatches when differential_check is on ([] = agreement);
        # None when the check did not run (flag off, no graph, no snapshot).
        self.last_differential: list[str] | None = None
        # Concrete launch capture (pre_warmup is the only hook that sees the
        # real args on the warmup-only path).
        self._launch_params: dict[str, int] = {}
        self._launch_tensors: dict[str, GlobalTensor] = {}
        self._launch_grid: tuple[Any, ...] | None = None
        self._warmup_count: int = 0
        self._capture_error: str | None = None
        # C2 snapshot: PRE-launch clones (finalize runs after the real
        # kernel already mutated the originals).
        self._replay_jit_fn: Any = None
        self._snapshot_args: tuple | None = None
        self._snapshot_kwargs: dict | None = None
        self._snapshot_skipped: str | None = None
        self._snapshot_tensors: dict[str, GlobalTensor] | None = None

    # ── compilation hooks ─────────────────────────────────────────────

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        # The warmup-only path never runs arg_callback/grid_callback, so this
        # is the only hook that sees the concrete launch — capture the scalar
        # params and tensor bases the T1 global-memory encoder needs.
        self._capture_launch(jit_fn, args, kwargs)
        return True  # force the real compile so TTGIR/TTIR exist

    def _capture_launch(self, jit_fn: Any, args: tuple, kwargs: dict) -> None:
        self._warmup_count += 1
        if self._warmup_count > 1:
            return  # ambiguous params; _analyze_global abstains
        try:
            names = list(getattr(jit_fn, "arg_names", None) or [])
            bound: list[tuple[str, Any]] = list(zip(names, args))
            bound += [(k, v) for k, v in kwargs.items() if k in names]
            self._launch_grid = kwargs.get("grid")
            for name, value in bound:
                if hasattr(value, "data_ptr"):
                    # contiguous defaults to False when unverifiable: the
                    # in-bounds premise is only sound for contiguous storage
                    # (numel·elem understates a strided view's extent).
                    is_contig = getattr(value, "is_contiguous", None)
                    self._launch_tensors[name] = GlobalTensor(
                        data_ptr=int(value.data_ptr()),
                        elem_size=int(value.element_size()),
                        numel=int(value.numel()),
                        contiguous=bool(is_contig()) if is_contig else False,
                    )
                elif isinstance(value, bool):
                    self._launch_params[name] = int(value)
                elif isinstance(value, int):
                    self._launch_params[name] = value
                # floats / other objects: not representable in the integer
                # model; a Param lookup on one aborts to unsupported.
            if self.confirm_races or self.differential_check:
                self._snapshot_launch(jit_fn, args, kwargs)
        except Exception as e:  # noqa: BLE001
            self._capture_error = f"{type(e).__name__}: {e}"

    # Replay snapshot cap: cloning the launch tensors is the price of
    # confirmable witnesses; past this total the replay is marked
    # unavailable instead of surprising the user with a giant copy.
    SNAPSHOT_CAP_BYTES: ClassVar[int] = 256 * 1024 * 1024
    # At most this many reports are replayed per launch (same-pid-pair
    # replays are cached).
    REPLAY_MAX_REPORTS: ClassVar[int] = 8

    def _snapshot_launch(self, jit_fn: Any, args: tuple, kwargs: dict) -> None:
        values = list(args) + [
            v for k, v in kwargs.items() if k not in ("grid", "warmup")
        ]
        total = sum(
            int(v.numel()) * int(v.element_size())
            for v in values
            if hasattr(v, "data_ptr") and hasattr(v, "numel")
        )
        if total > self.SNAPSHOT_CAP_BYTES:
            self._snapshot_skipped = (
                f"tensor snapshot over cap ({total} bytes > "
                f"{self.SNAPSHOT_CAP_BYTES})"
            )
            return

        def clone(v: Any) -> Any:
            if hasattr(v, "data_ptr") and hasattr(v, "clone"):
                return v.detach().clone()
            return v

        self._replay_jit_fn = jit_fn
        self._snapshot_args = tuple(clone(v) for v in args)
        self._snapshot_kwargs = {
            k: clone(v) for k, v in kwargs.items() if k not in ("grid", "warmup")
        }
        # C3 needs name → SNAPSHOT-clone bases: the diff compares the static
        # enumeration against the replay, and both sides must speak clone
        # addresses (the originals are mutated by the real launch).
        names = list(getattr(jit_fn, "arg_names", None) or [])
        snap_bound = list(zip(names, self._snapshot_args))
        snap_bound += [(k, v) for k, v in self._snapshot_kwargs.items() if k in names]
        self._snapshot_tensors = {
            name: GlobalTensor(
                data_ptr=int(v.data_ptr()),
                elem_size=int(v.element_size()),
                numel=int(v.numel()),
            )
            for name, v in snap_bound
            if hasattr(v, "data_ptr")
        }

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

    def _analyze_global(self) -> None:
        """Global-memory race verdict over this launch's parsed TTIR.

        One solver, second capture front-end: graphs lower to the same
        record shape the dynamic mode produces; pid/grid/arange/loop stay
        symbolic. The tier selector (_solve_one_graph) picks the least
        concretization per kernel: T0 (params symbolic too — race-free for
        ANY input) behind the linearity gate, else T1 (this launch's
        params); the rung lands in last_global_provenance. Nothing raised
        in here may escape (finalize runs in the launch teardown). Consumes
        and resets the per-launch capture state.
        """
        params, tensors = self._launch_params, self._launch_tensors
        warmups, capture_error = self._warmup_count, self._capture_error
        replay_jit_fn = self._replay_jit_fn
        snapshot_args, snapshot_kwargs = self._snapshot_args, self._snapshot_kwargs
        snapshot_skipped = self._snapshot_skipped
        snapshot_tensors = self._snapshot_tensors
        launch_grid = self._launch_grid
        self._launch_params, self._launch_tensors = {}, {}
        self._launch_grid = None
        self._warmup_count, self._capture_error = 0, None
        self._replay_jit_fn = None
        self._snapshot_args, self._snapshot_kwargs = None, None
        self._snapshot_skipped = None
        self._snapshot_tensors = None

        self.last_global_reports = []
        self.last_global_status = "ok"
        self.last_global_reason = None
        self.last_global_provenance = None
        self.last_global_confirmation = None
        self.last_differential = None
        if not self.last_ttir_graphs:
            self.last_global_status = "no_ttir"
            self.last_global_reason = "no TTIR captured from warmup"
            return
        if warmups > 1:
            self.last_global_status = "unsupported"
            self.last_global_reason = (
                f"{warmups} warmups in one launch: parameter capture is " "ambiguous"
            )
            return
        if capture_error is not None:
            self.last_global_status = "unsupported"
            self.last_global_reason = f"launch capture failed: {capture_error}"
            return

        reports: list[Any] = []
        widened_all: list[Any] = []
        status, reason = "ok", None
        rungs: list[str] = []
        for graph, parse_reason in zip(
            self.last_ttir_graphs, self.last_ttir_unsupported
        ):
            if graph is None:
                status, reason = "unsupported", parse_reason
                continue
            outcome = self._solve_one_graph(graph, params, tensors)
            if outcome[0] == "proved":
                rungs.append(outcome[1])
            elif outcome[0] == "races":
                _, exact, widened = outcome
                reports.extend(exact)
                widened_all.extend(widened)
            else:
                status, reason = "unsupported", outcome[1]

        # ── C2: replay SAT witnesses under the interpreter ──
        confirmation: str | None = None
        replay_note: str | None = None
        if (reports or widened_all) and self.confirm_races:
            if snapshot_args is None:
                replay_note = snapshot_skipped or "replay snapshot unavailable"
            else:
                # Foci resolve against the SNAPSHOT clones' bases: the
                # replay's base_map is keyed by them (originals were cloned
                # at pre_warmup, and run_replay clones once more).
                confirmation, upgraded, widened_unclassified = self._confirm_reports(
                    replay_jit_fn,
                    snapshot_args,
                    snapshot_kwargs or {},
                    reports,
                    widened_all,
                    snapshot_tensors or {},
                    launch_grid,
                    self._ambiguous_focus_buckets(),
                )
                # A CONFIRMED widened report is a real race on this launch's
                # data: it graduates from the uncertain channel.
                reports = reports + upgraded
                upgraded_ids = {id(r) for r in upgraded}
                widened_all = [w for w in widened_all if id(w) not in upgraded_ids]

        self.last_global_reports = reports
        if reports:
            self.last_global_status = "races"
            self.last_global_confirmation = confirmation
            notes = []
            if widened_all:
                notes.append(
                    "additional possible races under over-approximation "
                    "were withheld"
                )
            if replay_note:
                notes.append(f"replay: {replay_note}")
            self.last_global_reason = "; ".join(notes) or None
        elif widened_all:
            self.last_global_status = "unsupported"
            if confirmation is not None and widened_unclassified == 0:
                # EVERY widened SAT was replayed and none reproduced: the
                # race-unconfirmed terminal state (potential, never
                # definite). The claim is only made when the replay actually
                # established it for all of them — a capped or unavailable
                # replay keeps the generic abstention below.
                self.last_global_reason = (
                    "race-unconfirmed: possible race under over-approximation "
                    "(data-dependent mask / unmodeled branch); the interpreter "
                    "replay did not reproduce it on this launch's data"
                )
            else:
                self.last_global_reason = (
                    "possible race under over-approximation (data-dependent "
                    "mask / unmodeled branch) — not a certifiable witness"
                    + (f" (replay: {replay_note})" if replay_note else "")
                )
        elif status != "ok":
            self.last_global_status = status
            self.last_global_reason = reason
        else:
            self.last_global_provenance = (
                "proved@T0" if rungs and all(r == "T0" for r in rungs) else "proved@T1"
            )

        # ── C3: opt-in differential cross-check ──
        if self.differential_check:
            # NOTE: pass the SNAPSHOT tensor bases, not the originals — the
            # replay runs on the snapshot clones, so both sides of the diff
            # must speak clone addresses.
            self._run_differential(
                params,
                snapshot_tensors or {},
                replay_jit_fn,
                snapshot_args,
                snapshot_kwargs,
                launch_grid,
            )
        if cfg.cli_active:
            self._report_global_cli()

    def _run_differential(
        self,
        params: dict,
        tensors: dict,
        jit_fn: Any,
        snapshot_args: tuple | None,
        snapshot_kwargs: dict | None,
        launch_grid: Any,
    ) -> None:
        """C3 (plan §I.4): compare the static model's concrete footprint
        against the interpreter's for the first block(s) of the launch grid.
        Requires the single-kernel case, a snapshot, and a concrete grid;
        otherwise last_differential stays None. Never raises."""
        try:
            graphs = [g for g in self.last_ttir_graphs if g is not None]
            if (
                len(graphs) != 1
                or snapshot_args is None
                or jit_fn is None
                or not isinstance(launch_grid, (tuple, list))
            ):
                return
            grid = tuple(int(d) for d in launch_grid)
            grid = grid + (1,) * (3 - len(grid))
            pids: list[tuple[int, int, int]] = [(0, 0, 0)]
            if grid[0] > 1:
                pids.append((grid[0] - 1, 0, 0))
            from .replay import cross_check

            self.last_differential = cross_check(
                graphs[0],
                params,
                tensors,
                jit_fn,
                snapshot_args,
                snapshot_kwargs or {},
                pids,
                grid,
            )
        except Exception as e:  # noqa: BLE001
            self.last_differential = [f"differential check failed: {e}"]

    @staticmethod
    def _report_focus(record: Any, tensors: dict) -> tuple[int, str] | None:
        """(original tensor base, footprint kind bucket) for one record of a
        race report — the replay's overlap check is restricted to the
        report's own access pair (a whole-block check would fabricate
        confirmations for unrelated widened reports; adversarial repro in
        test_replay_channels)."""
        meta = tensors.get(record.tensor_name)
        if meta is None:
            return None
        if record.atomic_kind == "rmw":
            kind = "atomic_rmw"
        elif record.atomic_kind == "cas":
            kind = "atomic_cas"
        elif record.access_mode == "write":
            kind = "store"
        else:
            kind = "load"
        return (meta.data_ptr, kind)

    def _ambiguous_focus_buckets(self) -> set[tuple[str, str]]:
        """(tensor_name, kind bucket) pairs with MORE THAN ONE access site
        across this launch's graphs. Replay footprints merge all same-kind
        accesses to one tensor into a single bucket, so a report in an
        ambiguous bucket cannot be classified: an unrelated site's real
        overlap would confirm a widened report whose own access never
        executes (adversarial repro in test_replay_channels)."""
        from .differential import KIND_BUCKET

        counts: dict[tuple[str, str], int] = {}
        for graph in self.last_ttir_graphs:
            if graph is None:
                continue
            for a in graph.accesses:
                key = (a.base_param, KIND_BUCKET[a.kind])
                counts[key] = counts.get(key, 0) + 1
        return {k for k, n in counts.items() if n > 1}

    def _confirm_reports(
        self,
        jit_fn: Any,
        args: tuple,
        kwargs: dict,
        exact: list[Any],
        widened: list[Any],
        tensors: dict,
        launch_grid: Any,
        ambiguous: set[tuple[str, str]],
    ) -> tuple[str | None, list[Any], int]:
        """C2 (plan §I.4): replay each report's witness block pair on the
        snapshot clones and classify. WIDENED reports replay first — they
        are what the channel exists to classify; exact reports are already
        definite and only gain a label. Returns (aggregate, upgraded
        widened reports, number of widened reports left UNCLASSIFIED by
        cap/ambiguity/unavailability — the race-unconfirmed claim is only
        honest when that count is zero)."""
        from .replay import confirm_witness

        widened_ids = {id(w) for w in widened}
        cache: dict[tuple, tuple[str, str | None]] = {}
        confirmed = unconfirmed = 0
        upgraded: list[Any] = []
        widened_classified = 0
        for rep in (widened + exact)[: self.REPLAY_MAX_REPORTS]:
            pids = (tuple(rep.witness_grid_a), tuple(rep.witness_grid_b))
            focus_a = self._report_focus(rep.first_record, tensors)
            focus_b = self._report_focus(rep.second_record, tensors)
            names = (
                (rep.first_record.tensor_name, focus_a[1]) if focus_a else None,
                (rep.second_record.tensor_name, focus_b[1]) if focus_b else None,
            )
            if any(n is not None and n in ambiguous for n in names):
                continue  # unclassifiable: shared footprint bucket
            key = (pids, focus_a, focus_b)
            if key not in cache:
                cache[key] = confirm_witness(
                    jit_fn,
                    args,
                    kwargs,
                    *pids,
                    launch_grid,
                    focus_a=focus_a,
                    focus_b=focus_b,
                )
            verdict, _why = cache[key]
            if verdict == "confirmed":
                confirmed += 1
                if id(rep) in widened_ids:
                    widened_classified += 1
                    upgraded.append(rep)
            elif verdict == "unconfirmed":
                unconfirmed += 1
                if id(rep) in widened_ids:
                    widened_classified += 1
        widened_unclassified = len(widened) - widened_classified
        if confirmed and unconfirmed:
            return ("partial", upgraded, widened_unclassified)
        if confirmed:
            return ("confirmed", upgraded, widened_unclassified)
        if unconfirmed:
            return ("unconfirmed", upgraded, widened_unclassified)
        return (None, upgraded, widened_unclassified)

    # T0 backstop: the linearity gate should keep queries decidable, but an
    # unexpected hard query must cost bounded time before falling to T1.
    T0_TIMEOUT_MS: ClassVar[int] = 10_000
    _Z3_DEFAULT_TIMEOUT: ClassVar[int] = 4294967295  # z3's own default

    def _solve_one_graph(self, graph: AccessGraph, params: dict, tensors: dict):
        """The tier selector (plan §I.3) for one kernel specialization.

        Returns ``("proved", "T0"|"T1")``, ``("races", exact, widened)``, or
        ``("unsupported", reason)``. T0 (params symbolic — race-free for ANY
        input) is attempted only behind the syntactic linearity gate; any T0
        SAT falls through to T1 because a T0 witness carries parameter
        values that need not match this launch."""
        try:
            t0_proved = (
                self._t0_premises_hold_for_launch(graph, tensors)
                and t0_linearity_gate(graph)
                and self._try_t0(graph)
            )
        except Exception:  # noqa: BLE001
            # Even the gate walk must not escape finalize (deep-but-legal
            # term chains can exhaust recursion); T1 has its own guards.
            t0_proved = False
        if t0_proved:
            return ("proved", "T0")
        try:
            enc = encode_graph(graph, params, tensors)
            solver = TwoCopySymbolicHBSolver(
                enc.records, grid=symbolic_grid(enc), arange_dict=enc.arange_dict
            )
            found = solver.find_races()
        except UnsupportedTTIR as e:
            return ("unsupported", f"{e.kind}: {e}")
        except UnsupportedSymbolicRaceQuery as e:
            return ("unsupported", f"solver: {e}")
        except Exception as e:  # noqa: BLE001
            return ("unsupported", f"{type(e).__name__}: {e}")
        # Uncertainty discipline: a report touching a widened record
        # (dropped mask / unmodeled branch) is not a certifiable witness —
        # same rule as the sanitizer's check_graph. Widened reports are
        # KEPT (not just counted): C2 replays them, and a reproduced one
        # graduates to a definite race.
        exact: list[Any] = []
        widened: list[Any] = []
        for rep in found:
            ids = {rep.first.event_id, rep.second.event_id}
            if ids & enc.uncertain_event_ids:
                widened.append(rep)
            else:
                exact.append(rep)
        if exact or widened:
            return ("races", exact, widened)
        return ("proved", "T1")

    @staticmethod
    def _t0_premises_hold_for_launch(graph: AccessGraph, tensors: dict) -> bool:
        """A T0 proof partitions accesses per base pointer — the
        NON-ALIASING premise. It may stand in for THIS launch's verdict only
        when the launch demonstrably satisfies it: every accessed base has
        captured, contiguous metadata and the allocation intervals
        [data_ptr, data_ptr + numel·elem) are pairwise disjoint. An aliased
        (e.g. in-place) or unverifiable launch falls through to T1, which
        uses the real bases — reporting the aliased race — or fails closed."""
        intervals = []
        for name in {a.base_param for a in graph.accesses}:
            meta = tensors.get(name)
            if meta is None or not meta.contiguous:
                return False
            intervals.append(
                (meta.data_ptr, meta.data_ptr + meta.numel * meta.elem_size)
            )
        intervals.sort()
        return all(s2 >= e1 for (_, e1), (s2, _) in zip(intervals, intervals[1:]))

    def _try_t0(self, graph: AccessGraph) -> bool:
        """True only when EVERY per-tensor T0 group is UNSAT under symbolic
        params. Any SAT, unknown, timeout, or encoding limit (e.g. a loop
        bound referencing a param) falls back to T1 — never a report."""
        from z3 import set_param

        try:
            t0_groups = encode_graph_t0(graph)
        except Exception:  # noqa: BLE001
            return False
        set_param("timeout", self.T0_TIMEOUT_MS)
        try:
            for _name, enc in t0_groups:
                solver = TwoCopySymbolicHBSolver(
                    enc.records, grid=symbolic_grid(enc), arange_dict=enc.arange_dict
                )
                if solver.find_races():
                    return False
        except Exception:  # noqa: BLE001
            return False
        finally:
            set_param("timeout", self._Z3_DEFAULT_TIMEOUT)
        return True

    def _report_global_cli(self) -> None:
        s = self.last_global_status
        if s == "races":
            conf = self.last_global_confirmation
            suffix = f", replay: {conf}" if conf else ""
            print(
                f"[{self.LOG_TAG}] global memory: RACE — "
                f"{len(self.last_global_reports)} report(s){suffix}"
            )
        elif s == "ok":
            claim = (
                "race-free for ANY scalar params (this specialization, "
                "non-aliased args)"
                if self.last_global_provenance == "proved@T0"
                else "race-free for this input"
            )
            print(
                f"[{self.LOG_TAG}] global memory: {claim} on every grid "
                f"along the axes read ({self.last_global_provenance})"
            )
        else:
            print(f"[{self.LOG_TAG}] global memory: {s} — {self.last_global_reason}")

    def finalize(self) -> list:
        # Per-launch reset point (see grid_callback note): smtlib is extended
        # below, so it must be cleared here or it accumulates across launches
        # on the warmup-only path.
        self.smtlib = []
        self._consume_pending_ttir()
        self._analyze_global()
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
