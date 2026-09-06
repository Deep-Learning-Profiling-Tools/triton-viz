"""Durable, single-writer attempt ledger for resumable pinned experiments.

Only this ledger accepts results; child output and JSONL exports are diagnostic
or derived artifacts.  A transaction never spans measured worker execution.
Timing returned by begin/commit is controller overhead, not raw ``wall_s``.
Use a persistent local filesystem with working fsync, not a network filesystem.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PHASES = {"MAIN", "RETRY", "FINALIZING", "COMPLETE"}
_SCHEMA = """
CREATE TABLE run (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    manifest_json TEXT NOT NULL, manifest_hash TEXT NOT NULL,
    phase TEXT NOT NULL CHECK(phase IN ('MAIN','RETRY','FINALIZING','COMPLETE'))
);
CREATE TABLE specs (
    corpus TEXT NOT NULL, name TEXT NOT NULL, position INTEGER UNIQUE NOT NULL,
    spec_hash TEXT NOT NULL, PRIMARY KEY(corpus,name)
);
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL,
    started_at REAL NOT NULL, ended_at REAL, reason TEXT
);
CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions,
    corpus TEXT NOT NULL, name TEXT NOT NULL,
    slot TEXT NOT NULL CHECK(slot IN ('main','retry')),
    execution_number INTEGER NOT NULL, budget_s REAL NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('STARTED','COMMITTED','INTERRUPTED')),
    started_at REAL NOT NULL, ended_at REAL, reason TEXT, start_overhead_s REAL,
    FOREIGN KEY(corpus,name) REFERENCES specs(corpus,name),
    UNIQUE(corpus,name,slot,execution_number)
);
CREATE TABLE results (
    corpus TEXT NOT NULL, name TEXT NOT NULL, slot TEXT NOT NULL,
    attempt_id TEXT UNIQUE NOT NULL REFERENCES attempts,
    manifest_hash TEXT NOT NULL, row_json TEXT NOT NULL, row_hash TEXT NOT NULL,
    wall_s REAL NOT NULL, serialization_s REAL NOT NULL,
    PRIMARY KEY(corpus,name,slot),
    FOREIGN KEY(corpus,name) REFERENCES specs(corpus,name)
);
CREATE TABLE events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
    payload_json TEXT NOT NULL, recorded_at REAL NOT NULL
);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TRIGGER immutable_results_update BEFORE UPDATE ON results BEGIN
    SELECT RAISE(ABORT,'accepted results are immutable');
END;
CREATE TRIGGER immutable_results_delete BEFORE DELETE ON results BEGIN
    SELECT RAISE(ABORT,'accepted results are immutable');
END;
"""


class StateError(RuntimeError):
    """The ledger or requested transition violates the experiment protocol."""


class RunLocked(StateError):
    """Another controller currently owns this run."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _digest(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path | str, data: bytes) -> None:
    """Replace one file durably, syncing the file before the directory entry."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def exclusive_lock(path: Path | str):
    """Hold an OS lock; never delete the lock file or inherit it in children."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLocked(f"another controller holds {path}") from exc
        yield
    finally:
        os.close(descriptor)


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    mode = "ro" if readonly else "rw"
    con = sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    if not readonly:
        if con.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
            con.close()
            raise StateError("SQLite refused DELETE journal mode")
        con.execute("PRAGMA synchronous=EXTRA")
        if con.execute("PRAGMA synchronous").fetchone()[0] != 3:
            con.close()
            raise StateError("SQLite refused EXTRA synchronization")
    if con.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        con.close()
        raise StateError("SQLite refused foreign key enforcement")
    return con


def _validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise StateError("manifest must be an object")
    for key in ("protocol_version", "run_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise StateError(f"manifest requires {key}")
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("ladder_level") not in (
        "L0",
        "L1",
        "L2",
    ):
        raise StateError("manifest requires config.ladder_level")
    if not isinstance(config.get("fence_order"), bool):
        raise StateError("manifest requires config.fence_order")
    for key in ("row_timeout_s", "retry_timeout_s"):
        _positive_number(config.get(key), key)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise StateError("manifest requires a nonempty ordered roster")
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k]
            for k in ("corpus", "name", "spec_hash")
        ):
            raise StateError("manifest roster contains an invalid spec")
        row_key = row["corpus"], row["name"]
        if row_key in keys:
            raise StateError(f"duplicate manifest row: {row_key}")
        keys.add(row_key)
    if not isinstance(manifest.get("fingerprints"), dict):
        raise StateError("manifest requires fingerprints")
    try:
        canonical_json(manifest)
    except (TypeError, ValueError) as exc:
        raise StateError(f"manifest is not canonical JSON: {exc}") from exc


def _positive_number(value: Any, label: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        raise StateError(
            f"{label} must be a finite {'nonnegative' if allow_zero else 'positive'} number"
        )


def _schema_identity(con: sqlite3.Connection) -> list:
    return [
        tuple(row)
        for row in con.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
        )
    ]


class RunStore:
    """An experiment ledger whose lifetime holds its exclusive controller lock."""

    def __init__(self, path: Path, con: sqlite3.Connection, lock):
        self.path = path
        self._con = con
        self._closed = False
        self._lock = lock
        self._manifest: dict[str, Any] = {}
        self._manifest_digest = ""
        self.last_begin_metrics: dict = {}
        self._start_metrics: dict[str, float] = {}

    @classmethod
    def create(cls, path: Path | str, manifest: dict) -> "RunStore":
        _validate_manifest(manifest)
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(path.parent / f".{path.name}.initialize.lock"):
            if path.exists():
                raise StateError(f"run directory already exists: {path}")
            temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
            try:
                manifest_json = canonical_json(manifest)
                atomic_write(temporary / "manifest.json", manifest_json.encode("utf-8"))
                db = temporary / "checkpoint.sqlite"
                db.touch()
                con = _connect(db)
                try:
                    con.executescript(_SCHEMA)
                    con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    with con:
                        con.execute(
                            "INSERT INTO run VALUES (1,?,?, 'MAIN')",
                            (manifest_json, _digest(manifest_json)),
                        )
                        con.executemany(
                            "INSERT INTO specs VALUES (?,?,?,?)",
                            [
                                (row["corpus"], row["name"], i, row["spec_hash"])
                                for i, row in enumerate(manifest["rows"])
                            ],
                        )
                finally:
                    con.close()
                for directory in ("control", "attempts", "sessions", "exports"):
                    (temporary / directory).mkdir()
                    _fsync_directory(temporary / directory)
                (temporary / "writer.lock").touch()
                (temporary / "control.lock").touch()
                _fsync_directory(temporary)
                os.rename(temporary, path)
                _fsync_directory(path.parent)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return cls.open(path)

    @classmethod
    def open(cls, path: Path | str) -> "RunStore":
        path = Path(path).resolve()
        if not path.is_dir():
            raise StateError(f"run directory does not exist: {path}")
        lock = exclusive_lock(path / "writer.lock")
        lock.__enter__()
        con = None
        try:
            con = _connect(path / "checkpoint.sqlite")
            store = cls(path, con, lock)
            store._validate()
            return store
        except Exception as exc:
            if con is not None:
                con.close()
            lock.__exit__(None, None, None)
            if isinstance(exc, (sqlite3.DatabaseError, json.JSONDecodeError, OSError)):
                raise StateError(f"cannot open checkpoint: {exc}") from exc
            raise

    @classmethod
    def inspect(cls, path: Path | str) -> dict:
        """Read consistent progress while a writer is live, without owning it."""
        path = Path(path).resolve()
        try:
            con = _connect(path / "checkpoint.sqlite", readonly=True)
            store = cls(path, con, None)
            try:
                con.execute("BEGIN")
                store._validate()
                return store.status()
            finally:
                con.close()
        except (sqlite3.DatabaseError, json.JSONDecodeError, OSError) as exc:
            raise StateError(f"cannot inspect checkpoint: {exc}") from exc

    @property
    def manifest(self) -> dict:
        # Callers cannot mutate the experiment by changing a nested dictionary.
        return json.loads(canonical_json(self._manifest))

    @property
    def manifest_hash(self) -> str:
        return self._manifest_digest

    def _validate(self) -> None:
        con = self._con
        if con.execute("PRAGMA integrity_check").fetchall()[0][0] != "ok":
            raise StateError("checkpoint integrity check failed")
        if con.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise StateError("checkpoint foreign key check failed")
        if con.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise StateError("unsupported checkpoint schema version")
        expected = sqlite3.connect(":memory:")
        try:
            expected.executescript(_SCHEMA)
            if _schema_identity(con) != _schema_identity(expected):
                raise StateError("checkpoint schema differs from the supported schema")
        finally:
            expected.close()
        run = con.execute("SELECT * FROM run").fetchall()
        if len(run) != 1:
            raise StateError("checkpoint requires exactly one run identity")
        raw = (self.path / "manifest.json").read_text()
        if raw != run[0]["manifest_json"] or _digest(raw) != run[0]["manifest_hash"]:
            raise StateError("manifest bytes or digest do not match the checkpoint")
        self._manifest = json.loads(raw)
        self._manifest_digest = _digest(raw)
        _validate_manifest(self._manifest)
        if canonical_json(self._manifest) != raw:
            raise StateError("manifest is not canonically serialized")
        specs = [
            dict(row) for row in con.execute("SELECT * FROM specs ORDER BY position")
        ]
        expected_specs = [
            dict(
                corpus=row["corpus"],
                name=row["name"],
                position=i,
                spec_hash=row["spec_hash"],
            )
            for i, row in enumerate(self._manifest["rows"])
        ]
        if specs != expected_specs:
            raise StateError("checkpoint roster differs from manifest")
        accepted = set()
        for result in con.execute("SELECT * FROM results"):
            attempt = con.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (result["attempt_id"],)
            ).fetchone()
            row = json.loads(result["row_json"])
            if (
                result["manifest_hash"] != self.manifest_hash
                or _digest(result["row_json"]) != result["row_hash"]
                or canonical_json(row) != result["row_json"]
            ):
                raise StateError("saved result hash or canonical bytes mismatch")
            if (
                attempt is None
                or attempt["state"] != "COMMITTED"
                or (attempt["corpus"], attempt["name"], attempt["slot"])
                != (result["corpus"], result["name"], result["slot"])
            ):
                raise StateError("result and committed attempt disagree")
            self._validate_row(attempt, row)
            if row["wall_s"] != result["wall_s"]:
                raise StateError("saved wall time differs from raw row")
            accepted.add(result["attempt_id"])
        for attempt in con.execute("SELECT * FROM attempts"):
            self._validate_budget(attempt["slot"], attempt["budget_s"])
            if (attempt["state"] == "STARTED") != (attempt["ended_at"] is None):
                raise StateError("attempt state and completion time disagree")
            if attempt["execution_number"] < 1:
                raise StateError("attempt execution number must be positive")
            owner = con.execute(
                "SELECT ended_at FROM sessions WHERE session_id=?",
                (attempt["session_id"],),
            ).fetchone()
            if attempt["state"] == "STARTED" and owner[0] is not None:
                raise StateError("active attempt belongs to an ended session")
            if (attempt["state"] == "COMMITTED") != (attempt["attempt_id"] in accepted):
                raise StateError("committed attempt has no unique accepted result")
            if attempt["slot"] == "retry":
                self._require_retry(attempt["corpus"], attempt["name"])
        for table, column in (
            ("sessions", "metadata_json"),
            ("events", "payload_json"),
            ("metadata", "value_json"),
        ):
            for saved in con.execute(f"SELECT {column} FROM {table}"):
                value = json.loads(saved[0])
                if canonical_json(value) != saved[0]:
                    raise StateError(f"invalid canonical JSON in {table}")
        if (
            con.execute(
                "SELECT COUNT(*) FROM attempts WHERE state='STARTED'"
            ).fetchone()[0]
            > 1
        ):
            raise StateError("checkpoint contains more than one active attempt")
        if (
            con.execute(
                "SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL"
            ).fetchone()[0]
            > 1
        ):
            raise StateError("checkpoint contains more than one active session")

    def _event(self, kind: str, payload: dict) -> None:
        self._con.execute(
            "INSERT INTO events(kind,payload_json,recorded_at) VALUES (?,?,?)",
            (kind, canonical_json(payload), time.time()),
        )

    def new_session(self, metadata: dict) -> str:
        if self._con.execute(
            "SELECT 1 FROM sessions WHERE ended_at IS NULL"
        ).fetchone():
            raise StateError("recover the previous session before starting another")
        session_id = str(uuid.uuid4())
        with self._con:
            self._con.execute(
                "INSERT INTO sessions VALUES (?,?,?,NULL,NULL)",
                (session_id, canonical_json(metadata), time.time()),
            )
            self._event("session_started", {"session_id": session_id})
        return session_id

    def end_session(self, session_id: str, reason: str) -> None:
        if self._con.execute(
            "SELECT 1 FROM attempts WHERE session_id=? AND state='STARTED'",
            (session_id,),
        ).fetchone():
            raise StateError(
                "interrupt or commit the active attempt before ending a session"
            )
        with self._con:
            cur = self._con.execute(
                "UPDATE sessions SET ended_at=?,reason=? WHERE session_id=? AND ended_at IS NULL",
                (time.time(), reason, session_id),
            )
            if cur.rowcount != 1:
                raise StateError("session is absent or already ended")
            self._event("session_ended", {"session_id": session_id, "reason": reason})

    def recover_interrupted(self, reason: str) -> int:
        """Call only after the former controller's process domain is empty."""
        with self._con:
            cur = self._con.execute(
                "UPDATE attempts SET state='INTERRUPTED',ended_at=?,reason=? WHERE state='STARTED'",
                (time.time(), reason),
            )
            count = cur.rowcount
            sessions = self._con.execute(
                "UPDATE sessions SET ended_at=?,reason=? WHERE ended_at IS NULL",
                (time.time(), reason),
            ).rowcount
            if count or sessions:
                self._event(
                    "recovery",
                    {
                        "interrupted_attempts": count,
                        "closed_sessions": sessions,
                        "reason": reason,
                    },
                )
        return count

    def _validate_budget(self, slot: str, budget_s: float) -> None:
        if slot not in ("main", "retry"):
            raise StateError("attempt slot must be main or retry")
        _positive_number(budget_s, "attempt budget")
        expected = self._manifest["config"][
            "row_timeout_s" if slot == "main" else "retry_timeout_s"
        ]
        if budget_s != expected:
            raise StateError("attempt budget differs from the manifest")

    def _require_retry(self, corpus: str, name: str) -> None:
        main = self._con.execute(
            "SELECT row_json FROM results WHERE corpus=? AND name=? AND slot='main'",
            (corpus, name),
        ).fetchone()
        if main is None:
            raise StateError("retry requires a committed main result")
        row = json.loads(main[0])
        if not (
            row["terminal"] == "timeout"
            or row["wall_s"] >= self._manifest["config"]["row_timeout_s"]
        ):
            raise StateError("main result is not eligible for retry")

    def begin_attempt(
        self, corpus: str, name: str, slot: str, session_id: str, budget_s: float
    ) -> str:
        start = time.perf_counter()
        self._validate_budget(slot, budget_s)
        if self._con.execute("SELECT phase FROM run").fetchone()[0] == "COMPLETE":
            raise StateError("cannot execute a completed run")
        if not self._con.execute(
            "SELECT 1 FROM specs WHERE corpus=? AND name=?", (corpus, name)
        ).fetchone():
            raise StateError("attempt is outside the manifest roster")
        if not self._con.execute(
            "SELECT 1 FROM sessions WHERE session_id=? AND ended_at IS NULL",
            (session_id,),
        ).fetchone():
            raise StateError("attempt requires an active session")
        if self._con.execute("SELECT 1 FROM attempts WHERE state='STARTED'").fetchone():
            raise StateError("only one attempt can run at a time")
        if self._con.execute(
            "SELECT 1 FROM results WHERE corpus=? AND name=? AND slot=?",
            (corpus, name, slot),
        ).fetchone():
            raise StateError("slot already has an accepted completion")
        if slot == "retry":
            self._require_retry(corpus, name)
        attempt_id = str(uuid.uuid4())
        number = self._con.execute(
            "SELECT COALESCE(MAX(execution_number),0)+1 FROM attempts WHERE corpus=? AND name=? AND slot=?",
            (corpus, name, slot),
        ).fetchone()[0]
        with self._con:
            self._con.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,?,?,'STARTED',?,NULL,NULL,NULL)",
                (
                    attempt_id,
                    session_id,
                    corpus,
                    name,
                    slot,
                    number,
                    budget_s,
                    time.time(),
                ),
            )
            self._event("attempt_started", {"attempt_id": attempt_id})
        elapsed = time.perf_counter() - start
        self.last_begin_metrics = {"total_s": elapsed}
        self._start_metrics[attempt_id] = elapsed
        return attempt_id

    def _validate_row(self, attempt, row: dict) -> None:
        if not isinstance(row, dict):
            raise StateError("raw result must be an object")
        for key in ("name", "corpus"):
            if row.get(key) != attempt[key]:
                raise StateError(f"raw result {key} differs from the attempt")
        config = self._manifest["config"]
        if row.get("ladder_level") != config["ladder_level"]:
            raise StateError("raw result ladder level differs from the manifest")
        if (
            not isinstance(row.get("fence_order"), bool)
            or row["fence_order"] != config["fence_order"]
        ):
            raise StateError("raw result fence order differs from the manifest")
        _positive_number(row.get("wall_s"), "raw wall_s", allow_zero=True)
        if row.get("verdict") not in ("error", "abstain", "race", "race-free"):
            raise StateError("raw result has no recognized verdict")
        if not isinstance(row.get("terminal"), str) or not row["terminal"]:
            raise StateError("raw result requires a nonempty terminal")
        if "budget_s" in row and row["budget_s"] != attempt["budget_s"]:
            raise StateError("raw result budget differs from the attempt")

    def commit_result(self, attempt_id: str, row: dict) -> dict:
        started = time.perf_counter()
        attempt = self._con.execute(
            "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if attempt is None:
            raise StateError("unknown attempt")
        self._validate_row(attempt, row)
        serialize_start = time.perf_counter()
        try:
            raw = canonical_json(row)
        except (TypeError, ValueError) as exc:
            raise StateError(f"raw result is not canonical JSON: {exc}") from exc
        digest = _digest(raw)
        serialization_s = time.perf_counter() - serialize_start
        previous = self._con.execute(
            "SELECT * FROM results WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if previous is not None:
            if previous["row_json"] != raw or previous["row_hash"] != digest:
                raise StateError("accepted completion cannot be replaced")
            return {
                "serialization_s": serialization_s,
                "commit_s": 0.0,
                "total_s": time.perf_counter() - started,
                "replayed": True,
            }
        if attempt["state"] != "STARTED":
            raise StateError("only a started attempt can commit a result")
        commit_start = time.perf_counter()
        try:
            with self._con:
                self._con.execute(
                    "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        attempt["corpus"],
                        attempt["name"],
                        attempt["slot"],
                        attempt_id,
                        self.manifest_hash,
                        raw,
                        digest,
                        row["wall_s"],
                        serialization_s,
                    ),
                )
                self._con.execute(
                    "UPDATE attempts SET state='COMMITTED',ended_at=?,start_overhead_s=? WHERE attempt_id=?",
                    (time.time(), self._start_metrics.get(attempt_id), attempt_id),
                )
                self._event(
                    "result_committed", {"attempt_id": attempt_id, "row_hash": digest}
                )
        except sqlite3.IntegrityError as exc:
            raise StateError(f"result transaction rejected: {exc}") from exc
        commit_s = time.perf_counter() - commit_start
        return {
            "serialization_s": serialization_s,
            "commit_s": commit_s,
            "total_s": time.perf_counter() - started,
            "replayed": False,
        }

    def results(self, slot: str) -> dict:
        return {key: record["row"] for key, record in self.result_records(slot).items()}

    def result_records(self, slot: str) -> dict:
        if slot not in ("main", "retry"):
            raise StateError("result slot must be main or retry")
        return {
            (r["corpus"], r["name"]): {
                "row": json.loads(r["row_json"]),
                "attempt_id": r["attempt_id"],
                "session_id": r["session_id"],
            }
            for r in self._con.execute(
                "SELECT r.*,a.session_id FROM results r JOIN attempts a USING(attempt_id) "
                "JOIN specs s ON r.corpus=s.corpus AND r.name=s.name "
                "WHERE r.slot=? ORDER BY s.position",
                (slot,),
            )
        }

    def set_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise StateError("unknown run phase")
        old = self._con.execute("SELECT phase FROM run").fetchone()[0]
        order = ["MAIN", "RETRY", "FINALIZING", "COMPLETE"]
        if order.index(phase) < order.index(old):
            raise StateError("run phase cannot move backwards")
        if phase == old:
            return
        with self._con:
            self._con.execute("UPDATE run SET phase=?", (phase,))
            self._event("phase_changed", {"phase": phase})

    def get_metadata(self, key: str, default=None):
        row = self._con.execute(
            "SELECT value_json FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row is not None else default

    def set_metadata(self, key: str, value: Any) -> None:
        with self._con:
            self._con.execute(
                "INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) "
                "DO UPDATE SET value_json=excluded.value_json",
                (key, canonical_json(value)),
            )
            self._event("metadata_changed", {"key": key})

    def status(self) -> dict:
        counts = {
            r["slot"]: r["n"]
            for r in self._con.execute(
                "SELECT slot,COUNT(*) AS n FROM results GROUP BY slot"
            )
        }
        active = [
            dict(r)
            for r in self._con.execute("SELECT * FROM attempts WHERE state='STARTED'")
        ]
        sessions = [
            dict(r)
            for r in self._con.execute("SELECT * FROM sessions ORDER BY started_at")
        ]
        for session in sessions:
            session["metadata"] = json.loads(session.pop("metadata_json"))
        return {
            "run_id": self._manifest["run_id"],
            "manifest_hash": self.manifest_hash,
            "phase": self._con.execute("SELECT phase FROM run").fetchone()[0],
            "total_rows": len(self._manifest["rows"]),
            "main_committed": counts.get("main", 0),
            "retry_committed": counts.get("retry", 0),
            "active_attempts": active,
            "sessions": sessions,
            "interrupted_attempts": self._con.execute(
                "SELECT COUNT(*) FROM attempts WHERE state='INTERRUPTED'"
            ).fetchone()[0],
            "last_event_sequence": self._con.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM events"
            ).fetchone()[0],
        }

    def close(self) -> None:
        if not self._closed:
            self._con.close()
            self._closed = True
        if self._lock is not None:
            self._lock.__exit__(None, None, None)
            self._lock = None

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
