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
def one_sm():
    previous = triton_viz.config.num_sms
    triton_viz.config.num_sms = 1
    yield
    triton_viz.config.num_sms = previous


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
        counter = 0.0
        max_spin = 10_000.0
        # Spin until PID1 updates the output, but bail out after max_spin to avoid hangs.
        while tl.atomic_cas(out, 0.0, 0.0) == 0.0 and counter < max_spin:
            counter += 1.0
        tl.atomic_add(out, counter)
    else:
        # Do a bit of work, then publish a non-zero value for PID0 to observe.
        arange = tl.arange(0, 32)
        s = 0.0
        for i in range(4):
            s += tl.sum(tl.load(x + i * 32 + arange))
        tl.atomic_add(out, s + 1.0)


def test_producer_consumer_hang(one_sm):
    """With only one SM, PID0 should spin to the cap or hang because blocks run sequentially."""
    out = torch.zeros((), dtype=torch.float32)
    x = torch.zeros((128,), dtype=torch.float32)

    thread = threading.Thread(target=lambda: _producer_consumer[(2,)](x, out))
    thread.start()
    thread.join(timeout=2)
    assert (
        thread.is_alive() or out.item() >= 10_000
    ), "producer_consumer should hang or PID 0 spin to max limit (10K) when blocks run sequentially"


def test_producer_consumer_converges(two_sms):
    """With two SMs, PID1 should run concurrently so PID0 exits the spin loop before the cap."""
    out = torch.zeros((), dtype=torch.float32)
    x = torch.zeros((128,), dtype=torch.float32)

    thread = threading.Thread(target=lambda: _producer_consumer[(2,)](x, out))
    thread.start()
    thread.join(timeout=2)
    assert (
        not thread.is_alive() and 0 < out.item() < 10_000
    ), "producer_consumer should complete and PID 0 shouldn't spin to max limit (10K) when blocks run concurrently"


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


def test_racing_threads_prefers_first_block(one_sm):
    """Sequential execution: slower PID0 wins the atomic race when blocks are serialized."""
    x = torch.zeros((8 * 256,), dtype=torch.float32)
    out = torch.zeros((), dtype=torch.float32)
    _racing_threads[(2,)](x, out)
    assert out.item() == pytest.approx(
        256.0
    ), "PID 0 in racing_threads should win when blocks run sequentially"


def test_racing_threads_prefers_fast_block(two_sms):
    """Concurrent execution: faster PID1 should win the atomic race when blocks run in parallel."""
    x = torch.zeros((8 * 256,), dtype=torch.float32)
    out = torch.zeros((), dtype=torch.float32)
    _racing_threads[(2,)](x, out)
    assert out.item() == pytest.approx(
        1.0
    ), "PID 1 in racing_threads should win when blocks run concurrently"


@triton_viz.trace("tracer")
@triton.jit
def _write_pid(out):
    pid = tl.program_id(0)
    tl.store(out + pid, pid)


def test_tracer_records_all_blocks(two_sms):
    """Tracer should record a Grid entry for every launched block when multiple blocks run."""
    out = torch.zeros((4,), dtype=torch.float32)
    _write_pid[(4,)](out)
    assert trace_state.launches, "trace launch should be recorded"
    grid_records = [r for r in trace_state.launches[-1].records if hasattr(r, "idx")]
    seen = {tuple(record.idx) for record in grid_records}
    assert seen == {(i, 0, 0) for i in range(4)}
