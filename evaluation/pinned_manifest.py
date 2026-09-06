"""Content identity for a resumable pinned experiment (not a row timer)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROTOCOL_VERSION = "pinned-resume-v1"
ROOT = Path(__file__).resolve().parents[1]
ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "TRITON_CACHE_DIR",
    "TRITON_INTERPRET",
    "TRITON_VIZ_FENCE_ORDER",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_HOME",
    "PYTHONDONTWRITEBYTECODE",
    "TRITON_VIZ_NUM_SMS",
    "ENABLE_RACE_DETECTOR",
    "ENABLE_SANITIZER",
    "ENABLE_PROFILER",
    "ENABLE_TIMING",
    "SANITIZER_ENABLE_FAKE_TENSOR",
    "TILEBENCH_ROOT",
    "FLA_USE_TMA",
    "TRITON_VIZ_VERBOSE",
    "REPORT_GRID_EXECUTION_PROGRESS",
    "PROFILER_ENABLE_LOAD_STORE_SKIPPING",
    "PROFILER_ENABLE_BLOCK_SAMPLING",
    "PROFILER_DISABLE_BUFFER_LOAD_CHECK",
    "SYMBOLIC_PER_ELEMENT_WARN_THRESHOLD",
    "SANITIZER_REPORT_MAX_SEGMENTS",
)
PACKAGES = (
    "triton",
    "torch",
    "numpy",
    "z3-solver",
    "fla-core",
    "liger-kernel",
    "flag_attn",
    "flag_gems",
    "torchao",
    "tritonbench",
    "cuda-tile",
)


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _stable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in value.items()}
    # Captured constexpr dtypes have stable names; arbitrary objects do not.
    if type(value).__module__.startswith(("torch", "triton.language")):
        return {
            "type": type(value).__module__ + "." + type(value).__name__,
            "value": str(value),
        }
    raise ValueError(f"cannot canonically identify {type(value)!r}")


def spec_identity(spec) -> dict:
    values = {
        f.name: _stable(getattr(spec, f.name))
        for f in dataclasses.fields(spec)
        if f.name not in ("kernel_fn", "make_args")
    }
    kernel = spec.kernel_fn
    if kernel is not None:
        fn = getattr(kernel, "fn", kernel)
        values["kernel"] = {
            "module": getattr(fn, "__module__", ""),
            "name": getattr(fn, "__qualname__", getattr(fn, "__name__", "")),
            "source": getattr(kernel, "src", None) or inspect.getsource(fn),
        }
    maker = spec.make_args
    values["make_args"] = {
        "module": maker.__module__,
        "name": maker.__qualname__,
        "source": inspect.getsource(maker),
    }
    return values


def _tree_files(root: Path) -> dict:
    out = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            x in path.parts for x in ("__pycache__", ".git", "results")
        ):
            continue
        if (
            path.suffix in (".py", ".json", ".npz", ".ttir", ".ttgir", ".mlir", ".so")
            or ".so." in path.name
        ):
            out[str(path.relative_to(root))] = file_hash(path)
    return out


def fingerprints(*, packages: bool = True) -> dict:
    """Hash actual installed code, not only version labels or wheel RECORDs."""
    source = {name: _tree_files(ROOT / name) for name in ("evaluation", "triton_viz")}
    runtime: dict = {}
    if packages:
        for name in PACKAGES:
            try:
                dist = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                runtime[name] = None
                continue
            files = {}
            for rel in sorted(dist.files or [], key=str):
                path = Path(dist.locate_file(rel))
                if path.is_file() and (
                    path.suffix in (".py", ".so", ".pth", ".json")
                    or ".so." in path.name
                    or path.name in ("RECORD", "METADATA")
                ):
                    files[str(rel)] = file_hash(path)
            # Editable imports can live outside the distribution's file list.
            direct = dist.read_text("direct_url.json")
            if direct:
                info = json.loads(direct)
                if info.get("dir_info", {}).get("editable"):
                    from urllib.parse import unquote, urlparse

                    location = Path(unquote(urlparse(info["url"]).path))
                    files["editable-source"] = digest(_tree_files(location))
            runtime[name] = {"version": dist.version, "files_hash": digest(files)}
    nvidia = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if __import__("shutil").which("nvidia-smi")
        else None
    )
    machine_id = Path("/etc/machine-id")
    return {
        "source": source,
        "packages": runtime,
        "python": {
            "path": sys.executable,
            "version": sys.version,
            "binary": file_hash(Path(sys.executable).resolve()),
        },
        "host": {
            "machine_id": file_hash(machine_id)
            if machine_id.exists()
            else platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "gpu": nvidia.stdout.strip() if nvidia and nvidia.returncode == 0 else None,
        },
        "environment": {k: os.environ.get(k) for k in ENV_KEYS},
        "libc": list(platform.libc_ver()),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def build_manifest(config: dict, *, run_id: str, only_names=None) -> tuple[dict, dict]:
    from evaluation.kernels import load
    from evaluation.runner import results_header
    from triton_viz.clients.race_detector.ladder import parse_ladder_level

    corpora = {}
    roster: list[dict] = []
    headers = {}
    level = parse_ladder_level(config["ladder_level"])
    for name in config["corpora"]:
        corpus = load(name)
        selected = [
            s
            for s in corpus.specs
            if only_names is None or (name, s.name) in only_names
        ]
        corpora[name] = {s.name: s for s in selected}
        if len(corpora[name]) != len(selected) or not selected:
            raise ValueError(f"{name}: duplicate or empty row roster")
        roster.extend(
            {"corpus": name, "name": s.name, "spec_hash": digest(spec_identity(s))}
            for s in selected
        )
        headers[name] = results_header(
            name, config["seed"], corpus.provenance, level, config["row_timeout_s"]
        )
    if (
        only_names is not None
        and {(r["corpus"], r["name"]) for r in roster} != only_names
    ):
        raise ValueError("requested rehearsal row is missing")
    if not config["rehearsal"] and len(roster) != 1242:
        raise ValueError(f"expected 1242 frozen rows, enumerated {len(roster)}")
    # Referenced gitignored sidecars must exist before any worker runs.
    for corpus in config["corpora"]:
        specfile = ROOT / "evaluation" / "kernels" / f"{corpus}_specs.json"
        if specfile.exists() and '"values_ref"' in specfile.read_text():
            sidecar = specfile.with_name(f"{corpus}_values.npz")
            if not sidecar.is_file():
                raise ValueError(f"missing frozen values sidecar: {sidecar}")
            from evaluation.capture_common import ValueStore, referenced_values

            values = ValueStore.beside(specfile)
            for reference in referenced_values(json.loads(specfile.read_text())):
                values.get(reference)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "config": config,
        "rows": roster,
        "headers": headers,
        "execution_commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "fingerprints": fingerprints(),
    }, corpora


def validate_manifest(saved: dict) -> dict:
    """Re-enumerate and compare every identity before launching a child."""
    names = {(r["corpus"], r["name"]) for r in saved["rows"]}
    actual, corpora = build_manifest(
        saved["config"],
        run_id=saved["run_id"],
        only_names=names if saved["config"]["rehearsal"] else None,
    )
    if canonical(actual) != canonical(saved):
        changed = [k for k in actual if canonical(actual[k]) != canonical(saved.get(k))]
        raise ValueError(f"resume identity mismatch: {', '.join(changed)}")
    return corpora
