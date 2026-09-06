"""Owned Linux service domains and host admission for pinned experiments."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from evaluation.pinned_manifest import ENV_KEYS, ROOT
from evaluation.pinned_state import atomic_write, exclusive_lock


def host_state() -> Path:
    return Path(
        os.environ.get(
            "TRITON_VIZ_PINNED_STATE_DIR",
            str(Path.home() / ".local/state/triton-viz/pinned"),
        )
    )


def _show(unit: str) -> dict:
    proc = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "MainPID",
            "-p",
            "ControlGroup",
            "-p",
            "ActiveState",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(
            f"cannot inspect owned service {unit}: {proc.stderr.strip()}"
        )
    return dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def _members(info: dict) -> list[int]:
    group = info.get("ControlGroup", "")
    if not group:
        return []
    path = Path("/sys/fs/cgroup") / group.lstrip("/")
    members: set[int] = set()
    for procs in path.rglob("cgroup.procs"):
        try:
            members.update(map(int, procs.read_text().split()))
        except FileNotFoundError:
            pass
    return sorted(members)


def _boot() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


@contextlib.contextmanager
def admission(unit: str | None, run_id: str, session_id: str, *, rehearsal=False):
    """Hold admission through execution, checking old domains after owner loss."""
    root = host_state()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "evaluation.lock"):
        registry_path = root / "domains.json"
        registry = (
            json.loads(registry_path.read_text()) if registry_path.exists() else []
        )
        for record in registry:
            if record["boot_id"] != _boot():
                continue
            info = _show(record["unit"])
            if not _members(info):
                continue
            if int(info.get("MainPID", 0)):
                raise RuntimeError(
                    f"previous pinned domain is still active: {record['unit']}"
                )
            # Only authenticated registry domains whose controller is gone.
            subprocess.run(
                ["systemctl", "--user", "stop", record["unit"]], check=True, timeout=30
            )
            if _members(_show(record["unit"])):
                raise RuntimeError(
                    f"previous pinned workers are still stopping: {record['unit']}"
                )
        if unit:
            expected = f"triton-pinned-{run_id}-{session_id}.service"
            if unit != expected or int(_show(unit).get("MainPID", 0)) != os.getpid():
                raise RuntimeError(
                    "driver is not the controller of its declared service"
                )
            registry.append(
                {
                    "unit": unit,
                    "run_id": run_id,
                    "session_id": session_id,
                    "boot_id": _boot(),
                }
            )
            atomic_write(registry_path, json.dumps(registry, sort_keys=True).encode())
        elif not rehearsal:
            raise ValueError(
                "formal pinned execution requires the owned service launcher"
            )
        yield


def launch(run_dir: Path, manifest: dict) -> str:
    """Dispatch asynchronously; progress and pause requests live on disk."""
    with exclusive_lock(run_dir / "launch.lock"):
        current = domain_status(run_dir)
        if current.get("active") or current.get("service_state") in (
            "activating",
            "active",
            "deactivating",
        ):
            raise RuntimeError(f"run already has an active service: {current['unit']}")
        return _launch_locked(run_dir, manifest)


def _launch_locked(run_dir: Path, manifest: dict) -> str:
    session_id = uuid.uuid4().hex
    unit = f"triton-pinned-{manifest['run_id']}-{session_id}.service"
    with exclusive_lock(run_dir / "control.lock"):
        control = run_dir / "control.json"
        sequence = (
            json.loads(control.read_text())["sequence"] if control.exists() else 0
        )
        launch_record = {
            "unit": unit,
            "session_id": session_id,
            "boot_id": _boot(),
            "consumed_pause_sequence": sequence,
        }
        atomic_write(
            run_dir / "launch.json", json.dumps(launch_record, sort_keys=True).encode()
        )
    log = run_dir / "service.log"
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--service-type=exec",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=TimeoutStopSec=5",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{log}",
        f"--property=StandardError=append:{log}",
    ]
    environment = manifest["fingerprints"]["environment"]
    unset = []
    for key in ENV_KEYS:
        value = environment.get(key)
        if value is not None:
            command.append(f"--setenv={key}={value}")
        else:
            unset.append(key)
    if unset:
        command.append("--property=UnsetEnvironment=" + " ".join(unset))
    # This location controls host exclusion only, never solver configuration.
    if "TRITON_VIZ_PINNED_STATE_DIR" in os.environ:
        command.append(
            "--setenv=TRITON_VIZ_PINNED_STATE_DIR="
            + os.environ["TRITON_VIZ_PINNED_STATE_DIR"]
        )
    command += [
        sys.executable,
        "-m",
        "evaluation.pinned_run",
        "_execute",
        "--run-dir",
        str(run_dir),
        "--unit",
        unit,
        "--session-token",
        session_id,
        "--pause-sequence",
        str(sequence),
    ]
    subprocess.run(command, check=True)
    return unit


def assert_quiescent(unit: str | None):
    if unit:
        remaining = set(_members(_show(unit))) - {os.getpid()}
        if remaining:
            raise RuntimeError(
                f"row descendants remain in owned domain: {sorted(remaining)}"
            )


def domain_status(run_dir: Path) -> dict:
    path = run_dir / "launch.json"
    if not path.exists():
        return {"unit": None, "active": False}
    record = json.loads(path.read_text())
    if record["boot_id"] != _boot():
        return {"unit": record["unit"], "active": False, "previous_boot": True}
    info = _show(record["unit"])
    return {
        "unit": record["unit"],
        "active": bool(_members(info)),
        "controller_pid": int(info.get("MainPID", 0)),
        "service_state": info.get("ActiveState"),
    }
