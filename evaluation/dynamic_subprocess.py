"""Fresh-process containment for the evaluation interpreter stage.

Only private, locally created transport files are deserialized. A completion is
credited only after its child exits within the parent's READY/GO-to-reap budget.
Startup, input transport, child cleanup and actual timeout slack remain visible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import math
import numbers
from pathlib import Path
import resource
import signal
import subprocess
import struct
import sys
import tempfile
import threading
import time
import types
from typing import Any

PROTOCOL = "dynamic-spawn-v1"
CANCEL_GRACE_S = 0.05
SETUP_TIMEOUT_S = 60.0
SNAPSHOT_INTERVAL_S = 0.1


class DynamicSubprocessError(RuntimeError):
    pass


def _json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _read(path: Path):
    return json.loads(path.read_text())


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def source_identity() -> dict:
    import cloudpickle
    import triton_viz
    from evaluation import harness

    root = Path(triton_viz.__file__).resolve().parent
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    dependencies = {}
    for module_name, distribution in (
        ("torch", "torch"),
        ("triton", "triton"),
        ("z3", "z3-solver"),
        ("numpy", "numpy"),
        ("cloudpickle", "cloudpickle"),
    ):
        module = importlib.import_module(module_name)
        if module.__file__ is None:
            raise DynamicSubprocessError(
                f"dependency has no source origin: {module_name}"
            )
        filename = Path(module.__file__).resolve()
        dependencies[module_name] = {
            "version": importlib.metadata.version(distribution),
            "module_file": str(filename),
            "module_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
        }
    return {
        "detector_root": str(root),
        "detector_tree_sha256": _hash(files),
        "harness_sha256": hashlib.sha256(
            Path(harness.__file__).read_bytes()
        ).hexdigest(),
        "transport_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cloudpickle_version": cloudpickle.__version__,
        "python_version": sys.version,
        "dependencies": dependencies,
    }


def _kernel_identity(kernel) -> dict:
    fn = getattr(kernel, "fn", kernel)
    result: dict[str, Any] = {
        "source": getattr(kernel, "src", None),
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", None),
        "filename": getattr(getattr(fn, "__code__", None), "co_filename", None),
        "firstlineno": getattr(getattr(fn, "__code__", None), "co_firstlineno", None),
        "arg_names": list(getattr(kernel, "arg_names", ())),
    }
    dependencies = {}
    seen = set()

    def visit(current, prefix):
        if id(current) in seen:
            return
        seen.add(id(current))
        function = getattr(current, "fn", current)
        code = getattr(function, "__code__", None)
        if code is None:
            return
        closure = {
            name: cell.cell_contents
            for name, cell in zip(code.co_freevars, function.__closure__ or ())
        }
        bindings = {**getattr(function, "__globals__", {}), **closure}
        for name in (*code.co_names, *code.co_freevars):
            value = bindings.get(name)
            key = prefix + "." + name
            if hasattr(value, "src") and hasattr(value, "fn"):
                dependencies[key] = {
                    "source": value.src,
                    "module": value.fn.__module__,
                    "qualname": value.fn.__qualname__,
                }
                visit(value, key)
            elif isinstance(value, types.ModuleType):
                filename = getattr(value, "__file__", None)
                dependencies[key] = {
                    "module": value.__name__,
                    "file": filename,
                    "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                    if filename
                    else None,
                }
            elif isinstance(value, (bool, int, str)) or value is None:
                if name in bindings:
                    dependencies[key] = {"constant": value}
            elif isinstance(value, float):
                dependencies[key] = {"constant_float": repr(value)}

    visit(kernel, "kernel")
    result["dependency_sources"] = dependencies
    return result


def input_identity(spec, args) -> dict:
    """Actual named CPU bytes, layout, reinterpret dtype and storage aliases."""
    import torch
    from evaluation.harness import _launch_binding

    groups: dict[int, int] = {}
    storage_hashes = {}
    result: dict[str, Any] = {}
    for name, value in _launch_binding(spec, args).items():
        interpreted_dtype = None
        if hasattr(value, "base") and isinstance(value.base, torch.Tensor):
            interpreted_dtype = str(value.dtype)
            value = value.base
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise DynamicSubprocessError("dynamic transport requires CPU inputs")
            storage = value.untyped_storage()
            group = groups.setdefault(storage._cdata, len(groups))
            if group not in storage_hashes:
                backing = (
                    torch.empty(0, dtype=torch.uint8)
                    .set_(storage, 0, (storage.nbytes(),), (1,))
                    .numpy()
                )
                storage_hashes[group] = hashlib.sha256(memoryview(backing)).hexdigest()
            raw = (
                value.detach()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            result[name] = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "reinterpret_dtype": interpreted_dtype,
                "storage_offset": value.storage_offset(),
                "alias_group": group,
                "storage_nbytes": storage.nbytes(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "storage_sha256": storage_hashes[group],
            }
        elif isinstance(value, bool):
            result[name] = value
        elif isinstance(value, numbers.Real):
            scalar = int(value) if isinstance(value, numbers.Integral) else float(value)
            result[name] = (
                scalar
                if isinstance(scalar, int) or math.isfinite(scalar)
                else {"float64_bits": struct.pack("!d", scalar).hex()}
            )
        elif value is None or isinstance(value, str):
            result[name] = value
        else:
            import triton.language as tl

            if name not in spec.constexprs or not isinstance(
                value, (torch.dtype, tl.dtype)
            ):
                raise DynamicSubprocessError(
                    f"unsupported transport argument type for {name}: {type(value).__name__}"
                )
            result[name] = {
                "type": type(value).__module__ + "." + type(value).__name__,
                "value": str(value),
            }
    return result


def _rebuild_jit(fn, options, source):
    from triton.runtime.jit import JITFunction

    kernel = JITFunction(fn, **options)
    if kernel.src != source:
        raise DynamicSubprocessError("kernel source differs after callable transport")
    return kernel


def _serialize_spec(spec) -> bytes:
    try:
        import cloudpickle
    except ImportError as exc:
        raise DynamicSubprocessError(
            "install the evaluation extra (cloudpickle==3.1.1)"
        ) from exc
    from triton.runtime.jit import JITFunction

    class KernelPickler(cloudpickle.CloudPickler):
        def reducer_override(self, obj):
            if isinstance(obj, JITFunction):
                options = {
                    key: getattr(obj, key, None)
                    for key in (
                        "version",
                        "do_not_specialize",
                        "do_not_specialize_on_alignment",
                        "debug",
                        "noinline",
                    )
                }
                options["repr"] = getattr(obj, "_repr", None)
                options["launch_metadata"] = getattr(obj, "launch_metadata", None)
                # No compiled device cache, lock or runtime handle crosses.
                return _rebuild_jit, (obj.fn, options, obj.src)
            return super().reducer_override(obj)

    bound = copy.copy(spec)
    object.__setattr__(bound, "make_args", None)
    stream = io.BytesIO()
    KernelPickler(stream).dump(bound)
    return stream.getvalue()


def _rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def _stop(proc) -> None:
    try:
        if proc.poll() is None:
            proc.kill()
    except ProcessLookupError:
        pass
    proc.wait()


def run_dynamic(spec, seed, level, budget_s, hooks=()) -> dict[str, Any]:
    import torch
    from triton_viz.core.config import config

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tilerace-dynamic-") as directory:
        path = Path(directory)
        args = spec.make_args(seed)
        identity = input_identity(spec, args)
        spec_bytes = _serialize_spec(spec)
        (path / "spec.pkl").write_bytes(spec_bytes)
        torch.save(args, path / "inputs.pt")
        del args
        request = {
            "protocol": PROTOCOL,
            "budget_s": float(budget_s),
            "seed": seed,
            "level": int(level),
            "hooks": list(hooks),
            "inputs": identity,
            "kernel": _kernel_identity(spec.kernel_fn),
            "config": dict(vars(config)),
            "source": source_identity(),
            "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        }
        _json(path / "request.json", request)
        environment = dict(os.environ)
        # Match actual import paths, including explicit evaluation helpers and
        # development-only dependencies; no monkeypatched objects are inherited.
        environment["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
        process_started = time.perf_counter()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "evaluation.dynamic_subprocess",
                "--child",
                str(path),
            ],
            env=environment,
        )
        ready_started = None
        cancellation_sent = None
        kill_sent = None
        child_peak = 0
        concurrent_peak = 0
        next_sample = 0.0
        try:
            while proc.poll() is None:
                now = time.perf_counter()
                if now >= next_sample:
                    child_rss = _rss(proc.pid)
                    child_peak = max(child_peak, child_rss)
                    concurrent_peak = max(
                        concurrent_peak, child_rss + _rss(os.getpid())
                    )
                    next_sample = now + SNAPSHOT_INTERVAL_S
                if ready_started is None and (path / "ready.json").exists():
                    ready = _read(path / "ready.json")
                    if (
                        ready["input_sha256"] != _hash(identity)
                        or ready["kernel_sha256"] != _hash(request["kernel"])
                        or ready["source"] != request["source"]
                    ):
                        raise DynamicSubprocessError("child launch identity mismatch")
                    ready_started = time.perf_counter()
                    _json(path / "go.json", {"parent_started": ready_started})
                if ready_started is None:
                    if now - process_started >= SETUP_TIMEOUT_S:
                        raise DynamicSubprocessError("dynamic child setup exceeded 60s")
                elif now - ready_started >= budget_s:
                    if cancellation_sent is None:
                        cancellation_sent = now
                        # Child's own soft watchdog may already be unwinding.
                        # Request cancellation without an async exception.
                        try:
                            if hasattr(signal, "SIGALRM"):
                                proc.send_signal(signal.SIGALRM)
                            else:
                                proc.terminate()
                        except ProcessLookupError:
                            break
                    if now - cancellation_sent >= CANCEL_GRACE_S:
                        kill_sent = time.perf_counter()
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        break
                time.sleep(0.005)
            proc.wait()
        except BaseException:
            _stop(proc)
            raise
        reaped = time.perf_counter()
        if ready_started is None:
            detail = (
                _read(path / "error.json")
                if (path / "error.json").exists()
                else {"exit_code": proc.returncode}
            )
            raise DynamicSubprocessError(f"dynamic child failed before READY: {detail}")
        elapsed = reaped - ready_started
        complete = (path / "result.json").exists() and proc.returncode == 0
        timed_out = elapsed >= budget_s or cancellation_sent is not None
        if not complete and not timed_out:
            detail = (
                _read(path / "error.json")
                if (path / "error.json").exists()
                else {"exit_code": proc.returncode}
            )
            raise DynamicSubprocessError(
                f"dynamic child failed during analysis: {detail}"
            )
        child_result = _read(path / "result.json") if complete else None
        live = _read(path / "live.json") if (path / "live.json").exists() else {}
        result = dict(child_result["result"]) if complete else {}
        if timed_out:
            result.update(
                status="timeout",
                reason=None,
                n_reports=0,
                premises=[],
                witnesses=[],
                error=f"dynamic track exceeded {budget_s}s",
            )
        child_internal = result.get("time_s")
        result["time_s"] = round(elapsed, 4)
        observed_hooks = child_result["hooks"] if complete else live.get("hooks", {})
        result["execution"] = {
            "protocol": PROTOCOL,
            "budget_s": budget_s,
            "startup_s": ready_started - started,
            "parent_ready_to_reap_s": elapsed,
            "return_slack_s": elapsed - budget_s,
            "cancellation_requested_s": None
            if cancellation_sent is None
            else cancellation_sent - ready_started,
            "kill_sent_s": None if kill_sent is None else kill_sent - ready_started,
            "cancel_grace_s": CANCEL_GRACE_S,
            "child_exit_code": proc.returncode,
            "child_completed": complete,
            "child_internal_time_s": child_internal,
            "child_peak_rss_kib": (child_result if complete else live).get(
                "peak_rss_kib"
            ),
            "child_peak_rss_sample_kind": "before result serialization"
            if complete
            else "last periodic sample before reap",
            "sampled_child_peak_rss_kib": child_peak,
            "sampled_parent_plus_child_peak_rss_kib": concurrent_peak,
            "rss_sampling_interval_s": SNAPSHOT_INTERVAL_S,
            "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "input_sha256": _hash(identity),
            "kernel_sha256": _hash(request["kernel"]),
            "spec_sha256": request["spec_sha256"],
            "source": ready["source"],
            "config": request["config"],
            "child_result_available": (path / "result.json").exists(),
            "observer_error": _read(path / "observer-error.json")
            if (path / "observer-error.json").exists()
            else None,
            "hooks": {
                hook["name"]: {
                    "complete": complete and not timed_out,
                    "sampled_at_s": child_result["sampled_at_s"]
                    if complete
                    else live.get("sampled_at_s"),
                    "payload": observed_hooks.get(hook["name"]),
                    "reason": None
                    if hook["name"] in observed_hooks
                    else "no complete child hook snapshot available",
                }
                for hook in hooks
            },
        }
    result["execution"]["full_wall_s"] = time.perf_counter() - started
    return result


def _child(path: Path) -> None:
    import cloudpickle
    import torch
    from evaluation import harness
    from triton_viz.clients.race_detector.ladder import LadderLevel
    from triton_viz.core.config import config

    request = _read(path / "request.json")
    if request["protocol"] != PROTOCOL:
        raise DynamicSubprocessError("dynamic child protocol mismatch")
    spec_bytes = (path / "spec.pkl").read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest() != request["spec_sha256"]:
        raise DynamicSubprocessError("dynamic callable transport hash mismatch")
    spec = cloudpickle.loads(spec_bytes)
    args = torch.load(path / "inputs.pt", map_location="cpu", weights_only=False)
    identity = input_identity(spec, args)
    kernel = _kernel_identity(spec.kernel_fn)
    if identity != request["inputs"] or kernel != request["kernel"]:
        raise DynamicSubprocessError("dynamic child input/kernel identity mismatch")
    object.__setattr__(spec, "make_args", lambda seed: args)
    if set(vars(config)) != set(request["config"]):
        raise DynamicSubprocessError("dynamic child configuration schema differs")
    for key, value in request["config"].items():
        setattr(config, key, value)
    imported_source = source_identity()
    if imported_source != request["source"]:
        raise DynamicSubprocessError("dynamic child imported source differs")
    harness.DYNAMIC_TIMEOUT_S = request["budget_s"]
    observers = {}
    for hook in request["hooks"]:
        if hook["name"] in observers:
            raise DynamicSubprocessError("duplicate dynamic child hook name")
        factory = getattr(importlib.import_module(hook["module"]), hook["factory"])
        observers[hook["name"]] = factory(**hook.get("kwargs", {}))
    running = threading.Event()
    stopped = threading.Event()
    analysis_started = None
    snapshot_error = []

    def snapshot():
        if not running.is_set():
            return
        payload = {name: observer.snapshot() for name, observer in observers.items()}
        _json(
            path / "live.json",
            {
                "hooks": payload,
                "sampled_at_s": time.perf_counter() - analysis_started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )

    def monitor():
        while not stopped.wait(SNAPSHOT_INTERVAL_S):
            try:
                snapshot()
            except BaseException as exc:
                snapshot_error.append(f"{type(exc).__name__}: {exc}")
                _json(path / "observer-error.json", {"errors": snapshot_error})
                return

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def ready():
        nonlocal analysis_started
        _json(
            path / "ready.json",
            {
                "input_sha256": _hash(identity),
                "kernel_sha256": _hash(kernel),
                "source": imported_source,
            },
        )
        while not (path / "go.json").exists():
            time.sleep(0.001)
        analysis_started = time.perf_counter()
        for observer in observers.values():
            observer.begin()
        running.set()

    try:
        result = harness._dynamic_track_local(
            spec, request["seed"], LadderLevel(request["level"]), ready=ready
        )
    finally:
        stopped.set()
        monitor_thread.join()
    if snapshot_error:
        raise DynamicSubprocessError(
            f"dynamic observer snapshot failed: {snapshot_error}"
        )
    finished = {name: observer.finish() for name, observer in observers.items()}
    if analysis_started is None:
        raise DynamicSubprocessError("dynamic child returned without READY/GO")
    _json(
        path / "result.json",
        {
            "result": result,
            "hooks": finished,
            "sampled_at_s": time.perf_counter() - analysis_started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path, required=True)
    args = parser.parse_args()
    try:
        _child(args.child)
    except BaseException as exc:
        _json(
            args.child / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
