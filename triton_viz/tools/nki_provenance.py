"""Versioned provenance and compatibility checks for NKI model artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_viz.tools.nki_region_ir import REGION_IR_SCHEMA_VERSION

FINGERPRINT_SCHEMA_VERSION = 1
PACKAGE_NAMES = (
    "aws-neuronx-runtime-discovery",
    "neuronx-cc",
    "torch-neuronx",
    "triton",
)
TOOL_COMMANDS = {
    "neuron-explorer": ("neuron-explorer", "--version"),
    "neuron-ls": ("neuron-ls", "--version"),
    "neuronx-cc": ("neuronx-cc", "--version"),
}


def _stable_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def make_compiler_fingerprint(
    *,
    packages: Mapping[str, str],
    tools: Mapping[str, str],
    hardware: Mapping[str, str],
    repository_revision: str,
    repository_dirty: bool = False,
    repository_diff_digest: str = "",
    region_ir_schema_version: int = REGION_IR_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a canonical fingerprint from already collected environment facts."""
    identity = {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "packages": dict(sorted(packages.items())),
        "tools": dict(sorted(tools.items())),
        "hardware": dict(sorted(hardware.items())),
        "repository_revision": repository_revision,
        "repository_dirty": bool(repository_dirty),
        "repository_diff_digest": repository_diff_digest,
        "region_ir_schema_version": int(region_ir_schema_version),
    }
    return {**identity, "fingerprint": _stable_digest(identity)}


def _command_version(command: tuple[str, ...]) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable:{type(exc).__name__}"
    output = (result.stdout or result.stderr).strip()
    return (
        output
        if result.returncode == 0 and output
        else f"error:{result.returncode}:{output}"
    )


def _repository_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _repository_worktree(root: Path) -> tuple[bool, str]:
    """Hash tracked changes and untracked contents without recording source text."""
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=root,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True, "unavailable"
    if status.returncode != 0 or diff.returncode != 0:
        return True, "unavailable"
    status_text = status.stdout
    if not status_text:
        return False, ""
    digest = hashlib.sha256(diff.stdout)
    digest.update(status_text.encode())
    for line in status_text.splitlines():
        if not line.startswith("?? "):
            continue
        path = root / line[3:]
        if path.is_file():
            digest.update(line[3:].encode())
            digest.update(path.read_bytes())
    return True, digest.hexdigest()[:20]


def _hardware_identity() -> dict[str, str]:
    identity = {
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    try:
        result = subprocess.run(
            ("neuron-ls", "--json-output"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        devices = json.loads(result.stdout) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        devices = []
    if devices:
        identity.update(
            {
                "instance_type": str(devices[0].get("instance_type", "unknown")),
                "device_count": str(len(devices)),
                "neuroncore_count": str(
                    sum(int(device.get("nc_count", 0)) for device in devices)
                ),
                "device_memory_bytes": str(
                    sum(int(device.get("memory_size", 0)) for device in devices)
                ),
            }
        )
    return identity


def collect_compiler_fingerprint(repository_root: Path | None = None) -> dict[str, Any]:
    """Collect a fingerprint without importing accelerator packages."""
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    tools = {name: _command_version(command) for name, command in TOOL_COMMANDS.items()}
    hardware = _hardware_identity()
    root = repository_root or Path(__file__).resolve().parents[2]
    repository_dirty, repository_diff_digest = _repository_worktree(root)
    return make_compiler_fingerprint(
        packages=packages,
        tools=tools,
        hardware=hardware,
        repository_revision=_repository_revision(root),
        repository_dirty=repository_dirty,
        repository_diff_digest=repository_diff_digest,
    )


def compare_fingerprints(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify provenance differences; this does not infer lowering stability."""
    changed: dict[str, Any] = {}
    for field in (
        "packages",
        "tools",
        "hardware",
        "repository_revision",
        "repository_dirty",
        "repository_diff_digest",
        "region_ir_schema_version",
    ):
        if reference.get(field) != candidate.get(field):
            changed[field] = {
                "reference": reference.get(field),
                "candidate": candidate.get(field),
            }
    if not changed:
        status = "exact"
    elif set(changed) <= {
        "repository_revision",
        "repository_dirty",
        "repository_diff_digest",
    }:
        status = "repository_changed"
    elif "hardware" in changed:
        status = "hardware_changed"
    else:
        status = "compiler_stack_changed"
    return {"status": status, "changed": changed, "requires_canary": status != "exact"}


def write_experiment_manifest(
    output_dir: Path,
    *,
    experiment: str,
    config: Mapping[str, Any],
    repository_root: Path | None = None,
) -> Path:
    """Write reproducible experiment inputs and environment identity."""
    normalized_config = json.loads(json.dumps(config, sort_keys=True, default=str))
    manifest = {
        "manifest_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "config": normalized_config,
        "config_hash": _stable_digest(normalized_config),
        "compiler_fingerprint": collect_compiler_fingerprint(repository_root),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
