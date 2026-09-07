"""Admit real dynamic transport through READY, without authorizing analysis.

Run from the detector checkout with the evaluation dependencies installed::

    python -m evaluation.dynamic_transport_admission \
        --row golden_smoke/smoke_add_no --out admission.json

Repeat --row for named corpus launches. --builder-config accepts a JSON object
with module, module_sha256, function, and kwargs, for example the paper module
baselines.hybrid_scaling_cases, function build_case, kwargs {"case": <one complete
list_cases() entry>}. Put that repository on PYTHONPATH. The builder may return a
LaunchSpec or (LaunchSpec, metadata). Its verified Python source is executed
explicitly; argument factories and imported corpus modules are trusted code.

The exact production child imports dependencies, reconstructs tensors and JIT
functions, installs configuration, and constructs its detector before READY.
This helper never creates GO, so tracing, interpreter analysis, and profiling
begin() never run. A passed receipt is transport admission, not a verdict,
performance measurement, or evidence that subsequent interpretation succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from evaluation import dynamic_subprocess as transport

PROTOCOL = "dynamic-transport-admission-v1"
LOG_TAIL_BYTES = 16384


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _helper_identity() -> dict:
    filename = Path(__file__).resolve()
    root = filename.parent.parent
    return {
        "file": str(filename),
        "sha256": _file_hash(filename),
        "checkout": str(root),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tracked_changes": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
        ).splitlines(),
    }


def _log_receipt(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tail": raw[-LOG_TAIL_BYTES:].decode(errors="replace"),
        "tail_truncated": len(raw) > LOG_TAIL_BYTES,
    }


def _verify_ready(ready: dict, request: dict) -> None:
    if (
        ready.get("input_sha256") != transport._hash(request["inputs"])
        or ready.get("kernel_sha256") != transport._hash(request["kernel"])
        or ready.get("source") != request["source"]
    ):
        raise transport.DynamicSubprocessError("child READY identity mismatch")


def admit(spec, *, seed=0, level=0, setup_timeout_s=60.0) -> dict[str, Any]:
    """Use production transport, preserve errors, and always stop/reap the child.

    setup_timeout_s bounds spawn-to-READY waiting, like the production setup
    clock. Parent tensor creation/serialization precedes it and is separately
    included in full_wall_s. No solver-stage timeout or result is measured.
    """
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "error",
        "name": spec.name,
        "seed": seed,
        "level": int(level),
        "setup_timeout_s": setup_timeout_s,
        "ready_verified": False,
        "go_written": False,
        "analysis_ran": False,
        "fallback_used": False,
        "errors": [],
    }
    proc = None
    try:
        if not math.isfinite(setup_timeout_s) or setup_timeout_s <= 0:
            raise ValueError("setup timeout must be finite and positive")
        if int(level) not in (0, 1, 2):
            raise ValueError("level must be 0, 1, or 2")
        if spec.frontend != "triton":
            raise ValueError("transport admission requires a Triton launch")
        import torch
        from triton_viz.core.config import config

        receipt["helper"] = _helper_identity()
        with tempfile.TemporaryDirectory(
            prefix="tilerace-transport-admission-"
        ) as directory:
            path = Path(directory)
            try:
                # Preserve production ordering and serialization, including the
                # actual CPU storage/view objects. Never serialize make_args.
                args = spec.make_args(seed)
                identity = transport.input_identity(spec, args)
                receipt["inputs"] = identity
                spec_bytes = transport._serialize_spec(spec)
                (path / "spec.pkl").write_bytes(spec_bytes)
                torch.save(args, path / "inputs.pt")
                del args
                request = {
                    "protocol": transport.PROTOCOL,
                    "budget_s": 60.0,  # Required by child; no GO uses this budget.
                    "seed": seed,
                    "level": int(level),
                    "hooks": [],
                    "inputs": identity,
                    "kernel": transport._kernel_identity(spec.kernel_fn),
                    "config": dict(vars(config)),
                    "source": transport.source_identity(),
                    "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                }
                receipt["request"] = request
                receipt["hashes"] = {
                    key: transport._hash(request[key])
                    for key in ("inputs", "kernel", "config", "source")
                }
                receipt["hashes"].update(
                    request=transport._hash(request),
                    spec=request["spec_sha256"],
                    serialized_inputs=_file_hash(path / "inputs.pt"),
                )
                transport._json(path / "request.json", request)
                environment = dict(os.environ)
                environment["PYTHONPATH"] = os.pathsep.join(
                    str(p) for p in sys.path if p
                )
                command = [
                    sys.executable,
                    "-m",
                    "evaluation.dynamic_subprocess",
                    "--child",
                    str(path),
                ]
                receipt["child_module"] = "evaluation.dynamic_subprocess"
                receipt["python_executable"] = sys.executable
                spawned = time.perf_counter()
                receipt["parent_preparation_s"] = spawned - started
                with (path / "stdout.log").open("wb") as stdout, (
                    path / "stderr.log"
                ).open("wb") as stderr:
                    proc = subprocess.Popen(
                        command, env=environment, stdout=stdout, stderr=stderr
                    )
                    receipt["child_pid"] = proc.pid
                    try:
                        while True:
                            if (path / "ready.json").exists():
                                receipt["ready"] = transport._read(path / "ready.json")
                                _verify_ready(receipt["ready"], request)
                                if proc.poll() is not None:
                                    raise transport.DynamicSubprocessError(
                                        "child exited at READY without GO"
                                    )
                                receipt["ready_verified"] = True
                                receipt["spawn_to_ready_s"] = (
                                    time.perf_counter() - spawned
                                )
                                if receipt["spawn_to_ready_s"] >= setup_timeout_s:
                                    raise transport.DynamicSubprocessError(
                                        "child READY exceeded setup timeout"
                                    )
                                receipt["status"] = "admitted"
                                break
                            if proc.poll() is not None:
                                raise transport.DynamicSubprocessError(
                                    "child exited before READY"
                                )
                            if time.perf_counter() - spawned >= setup_timeout_s:
                                raise transport.DynamicSubprocessError(
                                    "child setup timeout before READY"
                                )
                            time.sleep(0.005)
                    finally:
                        stop_started = time.perf_counter()
                        transport._stop(proc)
                        receipt["stop_and_reap_s"] = time.perf_counter() - stop_started
                        receipt["spawn_to_reap_s"] = time.perf_counter() - spawned
                        receipt["child_exit_code"] = proc.returncode
                        receipt["child_reaped"] = True
                        receipt[
                            "stop_policy"
                        ] = "production _stop: kill if alive, then wait"
            finally:
                for kind in ("stdout", "stderr"):
                    if (path / f"{kind}.log").exists():
                        receipt[kind] = _log_receipt(path / f"{kind}.log")
                for name in ("error", "observer-error"):
                    if (path / f"{name}.json").exists():
                        receipt[name] = transport._read(path / f"{name}.json")
                forbidden = [
                    name
                    for name in ("go.json", "live.json", "result.json")
                    if (path / name).exists()
                ]
                receipt["analysis_artifacts"] = forbidden
                if forbidden:
                    receipt[
                        "analysis_ran"
                    ] = None  # Protocol violation; cannot certify absence.
                    raise transport.DynamicSubprocessError(
                        f"unexpected analysis authorization/output: {forbidden}"
                    )
    except Exception as exc:
        receipt["status"] = "error"
        receipt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        receipt["full_wall_s"] = time.perf_counter() - started
    return receipt


def _external_builder(filename: Path):
    """Load hash-bound source bytes, avoiding an inherited patch or stale pyc."""
    description = transport._read(filename)
    if set(description) != {"module", "module_sha256", "function", "kwargs"}:
        raise ValueError(
            "builder config requires module, module_sha256, function, kwargs"
        )
    found = importlib.util.find_spec(description["module"])
    if found is None or found.origin is None or not found.origin.endswith(".py"):
        raise ValueError("external builder must resolve to a Python source module")
    source = Path(found.origin).resolve()
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != description["module_sha256"]:
        raise ValueError("external builder source hash mismatch")
    module = importlib.util.module_from_spec(found)
    sys.modules[description["module"]] = module
    exec(compile(raw, str(source), "exec"), module.__dict__)
    result = getattr(module, description["function"])(**description["kwargs"])
    spec, metadata = result if isinstance(result, tuple) else (result, None)
    return spec, {
        "config": description,
        "config_sha256": _file_hash(filename),
        "source_file": str(source),
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", action="append", default=[], metavar="CORPUS/NAME")
    parser.add_argument("--builder-config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--setup-timeout-s", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args()
    if not ns.row and ns.builder_config is None:
        parser.error("at least one --row or --builder-config is required")
    if ns.out.exists():
        parser.error(
            "--out must be a new file; existing admission receipts are immutable"
        )
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "helper": _helper_identity(),
        "rows": [],
    }
    selectors = [("row", value) for value in ns.row]
    if ns.builder_config:
        selectors.append(("builder", ns.builder_config))
    for kind, selected in selectors:
        selection = {kind: str(selected)}
        try:
            if kind == "row":
                from evaluation.kernels import load

                corpus, name = str(selected).split("/", 1)
                population = load(corpus)
                spec = next(s for s in population.specs if s.name == name)
                selection["corpus_provenance"] = population.provenance
            else:
                spec, metadata = _external_builder(Path(selected))
                selection["builder_receipt"] = metadata
            row = admit(
                spec,
                seed=ns.seed,
                level=int(ns.level[1]),
                setup_timeout_s=ns.setup_timeout_s,
            )
        except Exception as exc:
            row = {
                "status": "error",
                "analysis_ran": False,
                "fallback_used": False,
                "errors": [{"type": type(exc).__name__, "message": str(exc)}],
            }
        row["selection"] = selection
        report["rows"].append(row)
        report["all_admitted"] = all(r["status"] == "admitted" for r in report["rows"])
        transport._json(ns.out, report)
    return 0 if report["all_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
