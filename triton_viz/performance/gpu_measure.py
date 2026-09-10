"""Audited GPU timing for shared machines; never changes device settings."""

from __future__ import annotations

import os
import statistics
import threading
import time
from functools import lru_cache


class BusyGPU(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _nvml():
    import pynvml

    pynvml.nvmlInit()
    return pynvml


def snapshot(index=0):
    # Query NVML directly: spawning nvidia-smi during sub-microsecond kernel
    # timing perturbs shared ARM hosts enough to invalidate otherwise idle runs.
    nvml = _nvml()
    device = nvml.nvmlDeviceGetHandleByIndex(index)
    processes = nvml.nvmlDeviceGetComputeRunningProcesses(device)
    graphics = nvml.nvmlDeviceGetGraphicsRunningProcesses(device)

    def optional(function, *args):
        try:
            return function(*args)
        except nvml.NVMLError_NotSupported:
            return "N/A"

    return {
        "time": time.time(),
        "index": index,
        "uuid": nvml.nvmlDeviceGetUUID(device),
        "driver": nvml.nvmlSystemGetDriverVersion(),
        "utilization_pct": nvml.nvmlDeviceGetUtilizationRates(device).gpu,
        "sm_clock_mhz": optional(
            nvml.nvmlDeviceGetClockInfo, device, nvml.NVML_CLOCK_SM
        ),
        "temperature_c": optional(
            nvml.nvmlDeviceGetTemperature, device, nvml.NVML_TEMPERATURE_GPU
        ),
        "power_mw": optional(nvml.nvmlDeviceGetPowerUsage, device),
        "processes": [{"pid": pid} for pid in sorted({p.pid for p in processes})],
        "graphics_processes": [
            {"pid": pid} for pid in sorted({p.pid for p in graphics})
        ],
        "host_load": list(os.getloadavg()),
    }


def assert_available(sample, *, own_pid=None, allowed_graphics=()):
    foreign = [p for p in sample["processes"] if p["pid"] != own_pid]
    if foreign:
        raise BusyGPU(f"Foreign GPU processes present: {foreign}")
    graphics = [
        p
        for p in sample.get("graphics_processes", [])
        if p["pid"] not in allowed_graphics
    ]
    if graphics:
        raise BusyGPU(f"Unapproved graphics processes present: {graphics}")
    # A busy GPU with no visible compute PID may be running graphics or another
    # PID namespace. Refuse that ambiguous baseline before creating our context.
    if own_pid is None:
        try:
            utilization = float(sample["utilization_pct"])
        except ValueError as exc:
            raise BusyGPU("Cannot audit initial GPU utilization") from exc
        if utilization > 5:
            raise BusyGPU(f"GPU baseline is active ({utilization}%)")


def measure(fn, *, device=0, samples=11, graph_size=1024, allowed_graphics=()):
    """Time individual kernels averaged over CUDA graph replays.

    Compilation/capture/warmup are excluded. Inputs remain cache-resident when
    they fit: this metric must not be described as cold-memory or host latency.
    Concurrent load or excessive sample variation invalidates the whole batch.
    """
    import torch

    if samples < 5 or graph_size < 1:
        raise ValueError("Need at least five samples and a positive graph size")
    before = snapshot(device)
    assert_available(before, own_pid=os.getpid(), allowed_graphics=allowed_graphics)
    fn()
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_size):
            fn()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        graph.replay()
        torch.cuda.synchronize(device)

    telemetry = []
    errors = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                item = snapshot(device)
                telemetry.append(item)
                assert_available(
                    item, own_pid=os.getpid(), allowed_graphics=allowed_graphics
                )
            except Exception as exc:
                errors.append(str(exc))
            stop.wait(0.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    timings = []
    try:
        for _ in range(samples):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            # One sufficiently long graph avoids CPU/GIL gaps between small
            # replays being charged as device execution on shared ARM hosts.
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / graph_size)
    finally:
        stop.set()
        thread.join(timeout=25)
    after = snapshot(device)
    try:
        assert_available(after, own_pid=os.getpid(), allowed_graphics=allowed_graphics)
    except BusyGPU as exc:
        errors.append(str(exc))
    median = statistics.median(timings)
    relative_span = (max(timings) - min(timings)) / median
    if relative_span > 0.15:
        errors.append(f"unstable_sample_span:{relative_span:.4f}")
    if not telemetry or thread.is_alive():
        errors.append("incomplete_monitoring")
    return {
        "latency_us": median,
        "samples_us": timings,
        "contaminated": bool(errors),
        "rejection_reasons": errors,
        "telemetry": {"before": before, "during": telemetry, "after": after},
        "isolation": "monitored_shared_desktop"
        if allowed_graphics
        else "monitored_no_other_processes",
        "metric": "cuda_graph_steady_cache_kernel_us",
        "graph_size": graph_size,
    }
