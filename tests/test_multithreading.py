import threading

import pytest
import torch
import importlib

import triton
import triton.language as tl

import triton_viz

trace_state = importlib.import_module("triton_viz.core.trace")


@pytest.fixture(autouse=True)
def _force_interpreter(monkeypatch):
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    yield


@pytest.fixture
def two_sms():
    previous = triton_viz.config.num_sms
    triton_viz.config.num_sms = 2
    yield
    triton_viz.config.num_sms = previous


@pytest.fixture(autouse=True)
def _clear_traces():
    triton_viz.core.clear()
    yield
    triton_viz.core.clear()


@triton_viz.trace("tracer")
@triton.jit
def _producer_consumer(x, out):
    pid = tl.program_id(0)
    if pid == 0:
        guard = 0
        while tl.atomic_cas(out, 0.0, 0.0) == 0.0 and guard < 1_000_000:
            guard += 1
        tl.atomic_add(out, 1.0)
    else:
        tl.atomic_add(out, 1.0)


def test_producer_consumer_converges(two_sms):
    out = torch.zeros((), dtype=torch.float32)
    x = torch.zeros((1,), dtype=torch.float32)

    thread = threading.Thread(target=lambda: _producer_consumer[(2,)](x, out))
    thread.start()
    thread.join(timeout=5)

    assert (
        not thread.is_alive()
    ), "producer_consumer should complete when blocks run concurrently"
    assert out.item() == pytest.approx(2.0)


@triton_viz.trace("tracer")
@triton.jit
def _racing_threads(x, out):
    pid = tl.program_id(0)
    arange = tl.arange(0, 8)
    if pid == 0:
        s = 0.0
        for i in range(256):
            s += tl.sum(tl.load(x + i * 8 + arange))
        tl.atomic_cas(out, 0.0, 256.0 + s)
    else:
        s = tl.sum(tl.load(x + arange))
        tl.atomic_cas(out, 0.0, 1.0 + s)


def test_racing_threads_prefers_fast_block(two_sms):
    x = torch.zeros((8 * 256,), dtype=torch.float32)
    out = torch.zeros((), dtype=torch.float32)
    _racing_threads[(2,)](x, out)
    # PID 1 should win the race in a concurrent interpreter run.
    assert out.item() == pytest.approx(1.0)


@triton_viz.trace("tracer")
@triton.jit
def _write_pid(out):
    pid = tl.program_id(0)
    tl.store(out + pid, pid)


def test_tracer_records_all_blocks(two_sms):
    out = torch.zeros((4,), dtype=torch.float32)
    _write_pid[(4,)](out)
    assert trace_state.launches, "trace launch should be recorded"
    grid_records = [r for r in trace_state.launches[-1].records if hasattr(r, "idx")]
    seen = {tuple(record.idx) for record in grid_records}
    assert seen == {(i, 0, 0) for i in range(4)}
