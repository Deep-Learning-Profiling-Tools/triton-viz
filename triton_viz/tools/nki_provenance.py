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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_manifest(
    calibration_dir: Path,
    *,
    calibration_files: list[Path],
    source_manifests: list[Path],
    split_file: Path | None = None,
    payload_definition: str = "unspecified",
) -> Path:
    """Create a frozen calibration bundle manifest with strict provenance."""
    fingerprints = []
    sources = []
    for manifest_path in source_manifests:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = data.get("compiler_fingerprint") or (
            data.get("versions") or {}
        ).get("compiler_fingerprint")
        if not fingerprint:
            raise ValueError(f"Manifest has no compiler_fingerprint: {manifest_path}")
        fingerprints.append(fingerprint)
        sources.append(
            {"path": str(manifest_path), "sha256": _file_sha256(manifest_path)}
        )
    reference = fingerprints[0]
    source_compatibility = []
    for candidate in fingerprints[1:]:
        comparison = compare_fingerprints(reference, candidate)
        if comparison["status"] not in {"exact", "repository_changed"}:
            raise ValueError(
                "Calibration sources have incompatible compiler fingerprints: "
                f"{comparison}"
            )
        source_compatibility.append(comparison)
    files = {}
    for path in calibration_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        files[path.name] = _file_sha256(path)
    payload = {
        "schema": "triton-viz.nki-model-bundle-v1",
        "compatibility_policy": "compiler_hardware_exact_model_builder_exact",
        # Hardware/control artifacts must agree with each other. The code that
        # fits and consumes those frozen artifacts is a separate compatibility
        # boundary: a trace/evaluation fix after collection must not require
        # recompiling hardware, but evaluate must exactly match the builder.
        "calibration_source_fingerprint": reference,
        # Source manifests may span repository-only changes when the hardware
        # kernel artifacts have been audited as unchanged. Preserve those
        # differences instead of pretending all collection happened from one
        # worktree. Compiler stack, hardware and Region IR schema differences
        # remain fatal above.
        "calibration_source_compatibility": source_compatibility,
        "model_builder_fingerprint": collect_compiler_fingerprint(
            Path(__file__).resolve().parents[2]
        ),
        # Compatibility alias for v1 readers.
        "compiler_fingerprint": reference,
        "region_ir_schema_version": REGION_IR_SCHEMA_VERSION,
        "engine_payload_definition": payload_definition,
        "calibration_files": files,
        "source_manifests": sources,
    }
    if split_file is not None:
        payload["train_split"] = {
            "path": str(split_file),
            "sha256": _file_sha256(split_file),
        }
    path = calibration_dir / "model_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def validate_model_manifest(
    path: Path,
    *,
    calibration_files: list[Path],
    current_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject missing, tampered, or compiler-incompatible calibration bundles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "triton-viz.nki-model-bundle-v1":
        raise ValueError(f"Unsupported model manifest schema: {path}")
    expected = data.get("calibration_files") or {}
    for file_path in calibration_files:
        digest = expected.get(file_path.name)
        if not digest:
            raise ValueError(f"{file_path.name} is not declared in {path}")
        if _file_sha256(file_path) != digest:
            raise ValueError(f"Calibration hash mismatch: {file_path}")
    if current_fingerprint is not None:
        reference = data.get("model_builder_fingerprint") or data[
            "compiler_fingerprint"
        ]
        comparison = compare_fingerprints(reference, current_fingerprint)
        if comparison["status"] != "exact":
            raise ValueError(f"Incompatible model bundle: {comparison}")
    return data
