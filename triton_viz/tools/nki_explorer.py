"""Shared Neuron Explorer export helpers for auditable NKI experiments."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

REQUIRED_MAPPING_TABLES = ("Instruction.parquet", "ActiveTime.parquet")


def _required_tables_ready(parquet_dir: Path) -> bool:
    return all(
        (parquet_dir / name).is_file() and (parquet_dir / name).stat().st_size > 0
        for name in REQUIRED_MAPPING_TABLES
    )


def export_parquet(hardware_dir: Path, timeout_s: float = 45.0) -> Path:
    """Export Explorer parquet without waiting on its known post-flush hang."""
    parquet_dir = hardware_dir / "explorer_parquet"
    if _required_tables_ready(parquet_dir):
        return parquet_dir
    if parquet_dir.exists():
        shutil.rmtree(parquet_dir)

    command = [
        "neuron-explorer",
        "view",
        "-n",
        str(hardware_dir / "file.neff"),
        "-s",
        str(hardware_dir / "profile.ntff"),
        "--output-format",
        "parquet",
        "--output-file",
        str(parquet_dir),
        "--disable-ui",
        "--ignore-event-trace",
    ]
    process = subprocess.Popen(
        command,
        cwd=hardware_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout_s
    ready_since: float | None = None
    while time.monotonic() < deadline:
        ready = _required_tables_ready(parquet_dir)
        if ready:
            ready_since = ready_since or time.monotonic()
            if time.monotonic() - ready_since >= 2.0:
                break
        returncode = process.poll()
        if returncode is not None:
            stdout, stderr = process.communicate()
            if returncode != 0 or not ready:
                raise RuntimeError(
                    f"neuron-explorer failed ({returncode}): "
                    f"stdout={stdout!r} stderr={stderr!r}"
                )
            return parquet_dir
        time.sleep(0.25)

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    stdout, stderr = process.communicate()
    if not _required_tables_ready(parquet_dir):
        raise TimeoutError(
            f"Explorer parquet incomplete after {timeout_s}s: "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
    return parquet_dir
