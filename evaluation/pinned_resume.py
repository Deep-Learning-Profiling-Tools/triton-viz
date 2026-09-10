"""Resumable pinned main/retry scheduling; all durable writes are outside rows."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from evaluation import pinned_manifest as identity
from evaluation.pinned_state import RunStore, atomic_write, exclusive_lock


def _json(path: Path, value):
    atomic_write(path, identity.canonical(value) + b"\n")


def _note_cleanup_error(primary: BaseException, message: str) -> None:
    if hasattr(primary, "add_note"):
        primary.add_note(message)
    else:
        print(f"[pinned] {message}", file=sys.stderr, flush=True)


class PauseRequested(Exception):
    pass


class Control:
    def __init__(self, path: Path, consumed: int = 0):
        self.path, self.consumed = path, consumed
        self.sequence = consumed
        self.request = None
        self.last_poll = 0.0
        self.signaled = False

    def poll(self, force=False):
        if self.signaled:
            self.request = "now"
            return self.request
        now = time.monotonic()
        if not force and now - self.last_poll < 0.5:
            return self.request
        self.last_poll = now
        path = self.path / "control.json"
        if path.exists():
            item = json.loads(path.read_text())
            if item["sequence"] > self.consumed:
                self.sequence = item["sequence"]
                self.request = item["mode"]
        if self.signaled:
            self.request = "now"
        return self.request

    def immediate(self):
        return self.poll() == "now"

    def between(self):
        if self.poll(force=True):
            raise PauseRequested

    @contextlib.contextmanager
    def signals(self):
        old = {}

        def stop(signum, frame):
            self.signaled = True

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                old[signum] = signal.signal(signum, stop)
            yield
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)


def request_pause(run_dir: Path, *, immediate=False) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    with exclusive_lock(run_dir / "control.lock"):
        path = run_dir / "control.json"
        old = json.loads(path.read_text()) if path.exists() else {"sequence": 0}
        item = {
            "run_id": manifest["run_id"],
            "sequence": old["sequence"] + 1,
            "mode": "now" if immediate else "drain",
            "requested_at": time.time(),
        }
        # A second graceful request cannot downgrade an outstanding immediate one.
        if old.get("mode") == "now":
            item["mode"] = "now"
        _json(path, item)
    return dict(
        item, status="request saved; await paused status and an empty process domain"
    )


def _load_guard(control: Control, enabled: bool, label: str):
    from evaluation.pinned_run import _foreign_evaluation_processes, LOAD_MAX

    if not enabled:
        control.between()
        return
    next_sample = 0.0
    while True:
        control.between()
        now = time.monotonic()
        if now >= next_sample:
            foreign = _foreign_evaluation_processes()
            load = os.getloadavg()[0]
            if not foreign and load < LOAD_MAX:
                return
            print(
                f"[pinned] waiting before {label}: load={load:.2f}, foreign={len(foreign)}",
                file=sys.stderr,
                flush=True,
            )
            next_sample = now + 15
        time.sleep(0.25)


def official_config(
    level,
    corpora,
    seed,
    row_timeout,
    retry_timeout,
    rehearsal,
    guard,
    purpose,
    dynamic_launcher="spawn",
):
    from evaluation.pinned_run import ALL_CORPORA, RETRY_TIMEOUT_S
    from evaluation.frontend_policy import frontend_policy
    from evaluation.runner import row_timeout_s
    from triton_viz.core.config import config as cfg

    budget = row_timeout_s(level) if row_timeout is None else row_timeout
    if min(budget, retry_timeout) <= 0:
        raise ValueError("budgets must be positive")
    if not rehearsal:
        if budget != row_timeout_s(level):
            raise SystemExit("row budget override is rehearsal-only")
        if not cfg.race_detector_fence_order:
            raise SystemExit("a pinned run is fence-ordered")
        if (
            seed != 0
            or tuple(corpora) != ALL_CORPORA
            or retry_timeout != RETRY_TIMEOUT_S
        ):
            raise SystemExit(
                "formal runs require all corpora, seed 0 and retry budget 320"
            )
        if not guard or os.environ.get("TRITON_VIZ_FENCE_ORDER") is not None:
            raise SystemExit(
                "formal runs require load guard and an unset fence-order environment"
            )
        if os.environ.get("TRITON_VIZ_PINNED_STATE_DIR") is not None:
            raise SystemExit("host-state directory overrides are rehearsal-only")
        if purpose == "definitive" and level.name != "L2":
            raise SystemExit(
                "the definitive run is L2; lower levels require purpose=attribution"
            )
        if identity.git("status", "--porcelain", "--untracked-files=no"):
            raise SystemExit("the tracked execution tree is dirty")
    return {
        **identity.launcher_metadata(dynamic_launcher),
        "ladder_level": level.name,
        "frontend_policy": frontend_policy(level),
        "corpora": list(corpora),
        "seed": seed,
        "row_timeout_s": budget,
        "retry_timeout_s": retry_timeout,
        "rehearsal": rehearsal,
        "load_guard": guard,
        "purpose": purpose,
        "fence_order": bool(cfg.race_detector_fence_order),
        "jobs": 1,
        "worker_reuse": False,
        "mutate": False,
        "retry_policy": "terminal-timeout-or-full-wall-v1",
        "merge_policy": "637f57f",
        "statistics_policy": "main-overhead-selected-verdict-v1",
    }


def start_run(
    level,
    corpora,
    seed=0,
    row_timeout=None,
    retry_timeout=320,
    rehearsal=False,
    guard=True,
    *,
    purpose="definitive",
    run_dir=None,
    foreground=False,
    only_names=None,
    dynamic_launcher="spawn",
    prepare_only=False,
):
    from evaluation.runner import RESULTS_DIR
    from evaluation.pinned_service import launch, admission

    config = official_config(
        level,
        corpora,
        seed,
        row_timeout,
        retry_timeout,
        rehearsal,
        guard,
        purpose,
        dynamic_launcher,
    )
    if prepare_only and foreground:
        raise ValueError("prepare-only and foreground execution are mutually exclusive")
    if only_names is not None and not rehearsal:
        raise ValueError("a subset is rehearsal-only")
    if foreground and not rehearsal:
        raise ValueError(
            "foreground execution is rehearsal-only; formal runs use owned services"
        )
    run_id = uuid.uuid4().hex
    run_dir = Path(run_dir or RESULTS_DIR / "pinned-runs" / run_id).resolve()
    # Preflight performs substantial import/hashing work. It participates in
    # host exclusion too, before there is an executable experiment session.
    with admission(None, run_id, "preflight", rehearsal=True):
        _load_guard(Control(run_dir), guard, "manifest preflight")
        if dynamic_launcher == "preload":
            identity.prepare_preload_environment(run_dir)
        manifest, _ = identity.build_manifest(
            config, run_id=run_id, only_names=only_names
        )
        with RunStore.create(run_dir, manifest):
            pass
    print(f"[pinned] created {run_id}: {run_dir}", file=sys.stderr, flush=True)
    if prepare_only:
        print(
            "[pinned] prepared only; no execution service launched",
            file=sys.stderr,
            flush=True,
        )
        return run_dir
    if foreground:
        return execute_run(run_dir)
    unit = launch(run_dir, manifest)
    print(
        f"[pinned] service {unit}; status/pause/resume use --run-dir {run_dir}",
        file=sys.stderr,
        flush=True,
    )
    return run_dir


def _complete_sets(store):
    from evaluation.pinned_run import budget_reached

    expected = {(r["corpus"], r["name"]) for r in store.manifest["rows"]}
    main, retry = store.results("main"), store.results("retry")
    if set(main) != expected:
        raise ValueError(
            f"incomplete main set: missing={sorted(expected - set(main))}, extra={sorted(set(main) - expected)}"
        )
    required = {
        key
        for key, row in main.items()
        if budget_reached(row, store.manifest["config"]["row_timeout_s"])
    }
    if set(retry) != required:
        raise ValueError(
            f"incomplete retry set: missing={sorted(required - set(retry))}, extra={sorted(set(retry) - required)}"
        )
    return main, retry


def _jsonl(path, header, rows):
    atomic_write(path, b"".join(identity.canonical(x) + b"\n" for x in [header, *rows]))


@contextlib.contextmanager
def _preload_session_cost(broker, run_dir: Path, session: str, manifest_hash: str):
    """Measure enter, all rows/gaps and verified close, including broker fsyncs."""
    started = time.perf_counter()
    primary = None
    try:
        with broker:
            yield broker
    except BaseException as exc:
        primary = exc
        raise
    finally:
        elapsed = time.perf_counter() - started
        try:
            broker.run_dir.mkdir(parents=True, exist_ok=True)
            cost_path = broker.run_dir / "run-cost.json"
            _json(
                broker.run_dir / "session-cost.json",
                {
                    "schema": "pinned-preload-session-cost-v1",
                    "session_id": session,
                    "manifest_hash": manifest_hash,
                    "observed_session_wall_s": elapsed,
                    "boundary": "before BrokerRun.__enter__ through BrokerRun.__exit__; includes broker audit fsyncs, excludes this checkpoint write",
                    "broker_closed": broker.closed,
                    "broker_cost_file": str(cost_path.relative_to(run_dir)),
                    "broker_cost_sha256": identity.file_hash(cost_path)
                    if cost_path.is_file()
                    else None,
                    "error": repr(primary) if primary is not None else None,
                },
            )
        except BaseException as cleanup_error:
            if primary is None:
                raise
            _note_cleanup_error(
                primary, f"session cost checkpoint also failed: {cleanup_error!r}"
            )


def _preload_publication_audit(
    store, run_dir: Path, records: dict
) -> tuple[list[Path], list[dict]]:
    """Verify bound receipts and charge every started broker session, including failures."""
    artifacts = set((run_dir / "preload").rglob("*.json"))

    def checked_file(name, expected):
        path = (run_dir / name).resolve()
        if (
            not path.is_relative_to(run_dir.resolve())
            or identity.file_hash(path) != expected
        ):
            raise ValueError(f"preload audit hash mismatch: {name}")
        artifacts.add(path)
        return path

    for slot_records in records.values():
        for record in slot_records.values():
            row = record["row"]
            audit = row.get("dynamic_preload")
            if not isinstance(audit, dict):
                raise ValueError("preload completion is missing its row cleanup audit")
            path = checked_file(audit["audit_file"], audit["audit_sha256"])
            bound = json.loads(path.read_text())
            if (
                bound["attempt_id"] != record["attempt_id"]
                or bound["session_id"] != record["session_id"]
                or bound["manifest_hash"] != store.manifest_hash
            ):
                raise ValueError(
                    "preload row audit belongs to a different attempt/session/manifest"
                )
            receipt = bound["receipt"]
            for name, expected in receipt["files"].items():
                checked_file(name, expected)
            checked_file(receipt["audit_file"], receipt["audit_sha256"])

    costs = []
    for session in store.status()["sessions"]:
        directory = run_dir / "preload" / session["session_id"]
        if not directory.exists():
            if store.get_metadata(f"preload_session:{session['session_id']}"):
                raise ValueError(
                    f"started preload session audit is missing: {session['session_id']}"
                )
            # A pause can precede broker construction. No preload cost exists.
            costs.append(
                {
                    "session_id": session["session_id"],
                    "broker_started": False,
                    "session_reason": session["reason"],
                }
            )
            continue
        path = directory / "run-cost.json"
        if not path.is_file():
            raise ValueError(
                f"preload session cost is missing: {session['session_id']}"
            )
        cost = json.loads(path.read_text())
        session_path = directory / "session-cost.json"
        if not session_path.is_file():
            raise ValueError(
                f"preload outer session cost is missing: {session['session_id']}"
            )
        observed = json.loads(session_path.read_text())
        if (
            observed.get("session_id") != session["session_id"]
            or observed.get("manifest_hash") != store.manifest_hash
            or observed.get("broker_closed") is not True
            or observed.get("broker_cost_file") != str(path.relative_to(run_dir))
            or observed.get("broker_cost_sha256") != identity.file_hash(path)
        ):
            raise ValueError(
                f"preload outer session cost is unverified: {session['session_id']}"
            )
        if (
            cost.get("protocol")
            != store.manifest["config"]["dynamic_launcher_protocol"]
            or cost.get("closed") is not True
            or cost.get("broker_reaped") is not True
            or cost.get("remaining_children") != []
            or cost.get("cleanup_issues") != []
        ):
            raise ValueError(
                f"preload session cleanup/cost is unverified: {session['session_id']}"
            )
        for key in ("run_wall_s", "shared_setup_wall_s", "shared_shutdown_wall_s"):
            value = cost.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"preload session has invalid {key}: {session['session_id']}"
                )
        elapsed = observed.get("observed_session_wall_s")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(
                f"preload outer session elapsed is invalid: {session['session_id']}"
            )
        artifacts.add(path)
        artifacts.add(session_path)
        failed = cost.get("error") is not None or cost.get("broker_returncode") != 0
        costs.append(
            {
                "session_id": session["session_id"],
                "broker_started": True,
                "session_reason": session["reason"],
                "error": cost.get("error"),
                "status": "interrupted"
                if session["reason"] == "operator pause"
                else "failed"
                if failed
                else "closed",
                "cost_file": str(path.relative_to(run_dir)),
                "cost_sha256": identity.file_hash(path),
                "observed_session_wall_s": elapsed,
                "session_cost_file": str(session_path.relative_to(run_dir)),
                "session_cost_sha256": identity.file_hash(session_path),
                "cost_boundary": observed["boundary"],
                "broker_run_wall_s": cost["run_wall_s"],
                "shared_setup_wall_s": cost["shared_setup_wall_s"],
                "shared_shutdown_wall_s": cost["shared_shutdown_wall_s"],
            }
        )
    return sorted(artifacts), costs


def publish(store, run_dir: Path) -> Path:
    from evaluation import pinned_run as pr
    from triton_viz.clients.race_detector.ladder import parse_ladder_level

    main, retry = _complete_sets(store)
    manifest, config = store.manifest, store.manifest["config"]
    root = run_dir / "exports"
    root.mkdir(exist_ok=True)
    store.set_phase("FINALIZING")
    files = {}
    manifest_hash = identity.digest(manifest)
    extra = {
        "run_id": manifest["run_id"],
        "manifest_hash": manifest_hash,
        "protocol_version": identity.PROTOCOL_VERSION,
        "rehearsal": config["rehearsal"],
        **identity.launcher_metadata(config.get("dynamic_launcher", "spawn")),
    }
    for corpus in config["corpora"]:
        rows = [
            main[(r["corpus"], r["name"])]
            for r in manifest["rows"]
            if r["corpus"] == corpus
        ]
        files[corpus] = root / f"{corpus}_main.jsonl"
        _jsonl(files[corpus], dict(manifest["headers"][corpus], **extra), rows)
    level = parse_ladder_level(config["ladder_level"])
    header, merged = pr.merge(
        files,
        retry,
        manifest["execution_commit"],
        level,
        config["row_timeout_s"],
        config["retry_timeout_s"],
        config["seed"],
        config["fence_order"],
        frontend_policy=config.get("frontend_policy", "all"),
    )
    header.update(extra, retried_rows=len(retry), jobs=1, purpose=config["purpose"])
    records = {slot: store.result_records(slot) for slot in ("main", "retry")}
    for row in merged:
        record = records[row["pinned_pass"]][(row["corpus"], row["name"])]
        row.update(
            extra, attempt_id=record["attempt_id"], session_id=record["session_id"]
        )
    tag = "REHEARSAL" if config["rehearsal"] else "PINNED"
    suffix = "" if level.name == "L0" else "_" + level.name
    out = root / f"{tag}_{manifest['execution_commit'][:7]}{suffix}.jsonl"
    _jsonl(out, header, merged)
    retry_file = root / "retry.jsonl"
    _jsonl(retry_file, dict(header, artifact="raw-retries"), list(retry.values()))
    stats = pr.overhead_stats(files, config["row_timeout_s"])
    table = pr.verdict_table(merged)
    summary = out.with_name(out.stem + "_SUMMARY.md")
    atomic_write(summary, pr.summary_markdown(header, merged, stats, table).encode())
    artifacts = [*files.values(), retry_file, out, summary]
    session_costs = None
    if config.get("dynamic_launcher", "spawn") == "preload":
        audits, session_costs = _preload_publication_audit(store, run_dir, records)
        artifacts.extend(audits)
    receipt = dict(
        extra,
        main_rows=len(main),
        retry_rows=len(retry),
        dataset=str(out.relative_to(run_dir)),
        files={str(p.relative_to(run_dir)): identity.file_hash(p) for p in artifacts},
    )
    if session_costs is not None:
        receipt["preload_session_costs"] = session_costs
    # Each file is synced and atomically replaced; COMPLETE makes the bundle visible.
    _json(run_dir / "COMPLETE.json", receipt)
    store.set_metadata("publication", receipt)
    store.set_phase("COMPLETE")
    return out


def verify_publication(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    receipt = json.loads((run_dir / "COMPLETE.json").read_text())
    if receipt["run_id"] != manifest["run_id"] or receipt[
        "manifest_hash"
    ] != identity.digest(manifest):
        raise ValueError("publication identity mismatch")
    for name, expected in receipt["files"].items():
        path = (run_dir / name).resolve()
        if (
            not path.is_relative_to(run_dir.resolve())
            or identity.file_hash(path) != expected
        ):
            raise ValueError(f"publication hash mismatch: {name}")
    return receipt


def verify_dataset(path: Path):
    """Require a receipt for v1, retain explicit legacy dataset compatibility."""
    with path.open() as f:
        header = json.loads(f.readline())
    if header.get("protocol_version") == identity.PROTOCOL_VERSION:
        run_dir = path.resolve().parent.parent
        receipt = verify_publication(run_dir)
        relative = str(path.resolve().relative_to(run_dir))
        if relative not in receipt["files"]:
            raise ValueError("dataset is not an artifact of its completed run")
    return header


def reconcile_complete(run_dir: Path) -> Path:
    """A receipt may be durable before the final ledger bookkeeping."""
    with RunStore.open(run_dir) as store:
        _complete_sets(store)
        receipt = verify_publication(run_dir)
        store.recover_interrupted("publication already durably complete")
        store.set_metadata("publication", receipt)
        store.set_phase("COMPLETE")
        return run_dir / receipt["dataset"]


def execute_run(
    run_dir: Path, *, unit=None, session_token=None, consumed_pause_sequence=None
):
    from evaluation.pinned_service import admission, assert_quiescent
    from evaluation.runner import _run_one, RowInterrupted
    from evaluation.pinned_run import budget_reached
    from triton_viz.clients.race_detector.ladder import parse_ladder_level

    manifest = json.loads((run_dir / "manifest.json").read_text())
    if consumed_pause_sequence is None:
        # Explicit foreground resume boundary; subsequent requests stay pending
        # even if admission or fingerprint validation takes a long time.
        with exclusive_lock(run_dir / "control.lock"):
            control_file = run_dir / "control.json"
            consumed_pause_sequence = (
                json.loads(control_file.read_text())["sequence"]
                if control_file.exists()
                else 0
            )
    if (run_dir / "COMPLETE.json").exists():
        return reconcile_complete(run_dir)
    token = session_token or uuid.uuid4().hex
    with admission(
        unit, manifest["run_id"], token, rehearsal=manifest["config"]["rehearsal"]
    ):
        with RunStore.open(run_dir) as store:
            corpora = identity.validate_manifest(store.manifest)
            config = store.manifest["config"]
            level = parse_ladder_level(config["ladder_level"])
            official_config(
                level,
                config["corpora"],
                config["seed"],
                config["row_timeout_s"],
                config["retry_timeout_s"],
                config["rehearsal"],
                config["load_guard"],
                config["purpose"],
                config.get("dynamic_launcher", "spawn"),
            )
            store.recover_interrupted(
                "previous controller ended before durable completion"
            )
            with exclusive_lock(run_dir / "control.lock"):
                control_path = run_dir / "control.json"
                prior = (
                    json.loads(control_path.read_text())
                    if control_path.exists()
                    else {"sequence": 0}
                )
                consumed = consumed_pause_sequence
                # Persist acknowledgment without deleting a possibly newer request.
                store.set_metadata("consumed_pause_sequence", consumed)
                if control_path.exists() and prior["sequence"] <= consumed:
                    _json(control_path, dict(prior, mode=None))
            control = Control(run_dir, consumed)
            session = store.new_session(
                {
                    "token": token,
                    "unit": unit,
                    "pid": os.getpid(),
                    "started_at": time.time(),
                    **identity.launcher_metadata(
                        config.get("dynamic_launcher", "spawn")
                    ),
                }
            )
            broker = None
            row_cleanup_failed = False
            try:
                with control.signals():
                    _load_guard(control, config["load_guard"], "session")
                    with contextlib.ExitStack() as resources:
                        if config.get("dynamic_launcher", "spawn") == "preload":
                            from evaluation.dynamic_preload.broker_adapter import (
                                BrokerRun,
                            )

                            broker = BrokerRun(
                                identity.ROOT, run_dir / "preload" / session
                            )
                            store.set_metadata(
                                f"preload_session:{session}",
                                {
                                    "broker_dir": str(
                                        broker.run_dir.relative_to(run_dir)
                                    ),
                                    "launch_intent": True,
                                },
                            )
                            resources.enter_context(
                                _preload_session_cost(
                                    broker, run_dir, session, store.manifest_hash
                                )
                            )
                        for slot in ("main", "retry"):
                            # On export-only recovery never move the phase backward.
                            if slot == "retry":
                                if len(store.results("main")) != len(manifest["rows"]):
                                    raise ValueError("retry before complete main pass")
                                if store.status().get("phase") not in (
                                    "FINALIZING",
                                    "COMPLETE",
                                ):
                                    store.set_phase("RETRY")
                            completed = store.results(slot)
                            main = store.results("main")
                            last_corpus = None
                            for item in manifest["rows"]:
                                key = item["corpus"], item["name"]
                                if key in completed or (
                                    slot == "retry"
                                    and not budget_reached(
                                        main[key], config["row_timeout_s"]
                                    )
                                ):
                                    continue
                                control.between()
                                if (
                                    broker is not None
                                    or slot == "retry"
                                    or key[0] != last_corpus
                                ):
                                    _load_guard(
                                        control,
                                        config["load_guard"],
                                        f"{slot} {key[0]}/{key[1]}",
                                    )
                                budget = (
                                    config["row_timeout_s"]
                                    if slot == "main"
                                    else config["retry_timeout_s"]
                                )
                                with exclusive_lock(run_dir / "control.lock"):
                                    control.between()
                                    attempt = store.begin_attempt(
                                        *key, slot, session, budget
                                    )
                                last_corpus = key[0]
                                attempt_dir = run_dir / "attempts" / attempt
                                attempt_dir.mkdir(parents=True, exist_ok=True)
                                options: dict[str, Any] = {
                                    "cancel_requested": control.immediate,
                                    "output_dir": attempt_dir,
                                }
                                if broker is not None:
                                    broker.row_process = broker.row_identity = None
                                    options["dynamic_broker"] = broker
                                primary = None
                                audit = None
                                started = time.perf_counter()
                                try:
                                    row = _run_one(
                                        corpora[key[0]][key[1]],
                                        key[0],
                                        config["seed"],
                                        budget,
                                        False,
                                        level,
                                        **options,
                                    )
                                except BaseException as exc:
                                    primary = exc
                                    raise
                                finally:
                                    elapsed = time.perf_counter() - started
                                    if (
                                        broker is not None
                                        and broker.row_process is not None
                                    ):
                                        try:
                                            receipt = broker.finish_row(
                                                attempt, elapsed
                                            )
                                            if primary is None:
                                                broker.validate_row(row, receipt)
                                            audit = {
                                                "attempt_id": attempt,
                                                "session_id": session,
                                                "manifest_hash": store.manifest_hash,
                                                "receipt": receipt,
                                            }
                                            audit_path = (
                                                attempt_dir / "preload-row.json"
                                            )
                                            _json(audit_path, audit)
                                            audit = {
                                                "audit_file": str(
                                                    audit_path.relative_to(run_dir)
                                                ),
                                                "audit_sha256": identity.file_hash(
                                                    audit_path
                                                ),
                                            }
                                        except BaseException as cleanup_error:
                                            row_cleanup_failed = True
                                            if primary is None:
                                                raise
                                            _note_cleanup_error(
                                                primary,
                                                f"preload row cleanup also failed: {cleanup_error!r}",
                                            )
                                if broker is not None:
                                    if audit is None:
                                        raise RuntimeError(
                                            "preload row returned without a registered/reaped process"
                                        )
                                    row["dynamic_preload"] = audit
                                    assert_quiescent(
                                        unit, allowed_resident=broker.broker_identity
                                    )
                                else:
                                    assert_quiescent(unit)
                                row.update(
                                    identity.launcher_metadata(
                                        config.get("dynamic_launcher", "spawn")
                                    )
                                )
                                metrics = store.commit_result(attempt, row)
                                print(
                                    f"[pinned] saved {slot} {key[0]}/{key[1]} "
                                    f"checkpoint={metrics['total_s']:.6f}s",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                control.between()
                    # No live broker remains during final validation/publication.
                    assert_quiescent(unit)
                    control.between()
                    identity.validate_manifest(store.manifest)
                    control.between()
                    assert_quiescent(unit)
                    result = publish(store, run_dir)
                    store.end_session(session, "complete")
                    return result
            except BaseException as exc:
                paused = (
                    isinstance(exc, (PauseRequested, RowInterrupted))
                    and not row_cleanup_failed
                )
                try:
                    if broker is not None and not broker.closed:
                        raise RuntimeError(
                            "broker cleanup is incomplete; session remains recoverable"
                        )
                    assert_quiescent(unit)
                    store.recover_interrupted(
                        "operator pause" if paused else "controller error"
                    )
                except BaseException as cleanup_error:
                    _note_cleanup_error(
                        exc, f"could not safely record interruption: {cleanup_error!r}"
                    )
                    raise exc
                if paused:
                    store.set_metadata(
                        "pause_acknowledged",
                        {"sequence": control.sequence, "mode": control.request},
                    )
                    print(f"[pinned] paused: {run_dir}", file=sys.stderr, flush=True)
                    return run_dir
                raise


def main(argv=None):
    from evaluation.pinned_run import ALL_CORPORA
    from evaluation.pinned_service import domain_status, launch
    from triton_viz.clients.race_detector.ladder import parse_ladder_level

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--ladder-level", choices=("L0", "L1", "L2"), default="L2")
    start.add_argument(
        "--corpora", nargs="+", default=list(ALL_CORPORA), choices=ALL_CORPORA
    )
    start.add_argument("--seed", type=int, default=0)
    start.add_argument("--row-timeout", type=float)
    start.add_argument("--retry-timeout", type=float, default=320)
    start.add_argument("--rehearsal", action="store_true")
    start.add_argument("--no-load-guard", action="store_true")
    start.add_argument(
        "--purpose", choices=("definitive", "attribution"), default="definitive"
    )
    start.add_argument("--foreground", action="store_true")
    start.add_argument("--prepare-only", action="store_true")
    start.add_argument(
        "--dynamic-launcher", choices=("spawn", "preload"), default="spawn"
    )
    start.add_argument("--run-dir", type=Path)
    for name in ("resume", "status", "verify", "pause", "_execute"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "pause":
            command.add_argument("--now", action="store_true")
        if name == "resume":
            command.add_argument("--foreground", action="store_true")
        if name == "_execute":
            command.add_argument("--unit", required=True)
            command.add_argument("--session-token", required=True)
            command.add_argument("--pause-sequence", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "start":
        result = start_run(
            parse_ladder_level(args.ladder_level),
            tuple(args.corpora),
            args.seed,
            args.row_timeout,
            args.retry_timeout,
            args.rehearsal,
            not args.no_load_guard,
            purpose=args.purpose,
            run_dir=args.run_dir,
            foreground=args.foreground,
            dynamic_launcher=args.dynamic_launcher,
            prepare_only=args.prepare_only,
        )
    else:
        directory = args.run_dir.resolve()
        if args.command == "pause":
            result = request_pause(directory, immediate=args.now)
        elif args.command == "status":
            result = RunStore.inspect(directory)
            result["domain"] = domain_status(directory)
        elif args.command == "verify":
            with RunStore.open(directory) as store:
                _complete_sets(store)
                result = verify_publication(directory)
        elif args.command == "_execute":
            result = execute_run(
                directory,
                unit=args.unit,
                session_token=args.session_token,
                consumed_pause_sequence=args.pause_sequence,
            )
        else:
            manifest = json.loads((directory / "manifest.json").read_text())
            if (directory / "COMPLETE.json").exists():
                result = reconcile_complete(directory)
            elif args.foreground:
                if not manifest["config"]["rehearsal"]:
                    parser.error("foreground is rehearsal-only")
                result = execute_run(directory)
            else:
                result = launch(directory, manifest)
    print(json.dumps(result, indent=2) if isinstance(result, dict) else result)
    return 0
