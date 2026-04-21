import pytest
import torch

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.core.config import config as cfg
from triton_viz.core.data import Load, Store


@pytest.fixture
def _isolate_race_detector_cfg():
    saved = cfg.enable_race_detector
    cfg.enable_race_detector = True
    yield
    cfg.enable_race_detector = saved


@triton.jit
def _basic_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v + 1)


@triton.jit
def _load_store_loop_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for _i in tl.range(0, 4):
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v + 1)


@triton.jit
def _loop_dedup_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _i in tl.range(0, 4):
        acc += tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, acc)


@triton.jit
def _loop_premises_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for _i in tl.range(0, 4):
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v)


@triton.jit
def _dispatch_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs))


def test_basic_capture(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_basic_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    modes = [r.access_mode for r in detector.records]
    op_types = [r.op_type for r in detector.records]
    assert modes == ["read", "write"], f"expected [read, write], got {modes}"
    assert op_types == [Load, Store], f"expected [Load, Store], got {op_types}"

    for record in detector.records:
        assert record.premises, "premises must include addr_ok + pid_ok at minimum"
        assert record.tensor is not None, "base tensor should be resolvable"
        assert record.tensor_name == "x_ptr"
        assert isinstance(record, AccessEventRecord)


def test_loop_load_and_store_same_addr_not_merged(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_load_store_loop_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    modes = sorted(r.access_mode for r in detector.records)
    assert modes == [
        "read",
        "write",
    ], f"load+store at same addr should stay as two events; got {modes}"


def test_loop_repeated_access_deduped(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_loop_dedup_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    reads = [r for r in detector.records if r.access_mode == "read"]
    assert (
        len(reads) == 1
    ), f"loop-body load should dedupe to a single event, got {len(reads)}"


def test_loop_event_premises_include_iterator(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_loop_premises_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    assert detector.records
    for record in detector.records:
        premise_strs = " ".join(str(p) for p in record.premises)
        assert "loop_i_" in premise_strs, (
            f"in-loop event premises must carry a loop iterator constraint; "
            f"got premises: {premise_strs}"
        )


def test_string_dispatch_and_manager_lookup(_isolate_race_detector_cfg):
    traced = triton_viz.trace("race_detector")(_dispatch_kernel)

    x = torch.zeros(8, dtype=torch.float32)
    traced[(1,)](x, BLOCK=8)

    rd = traced.client_manager.clients["race_detector"]
    assert isinstance(rd, SymbolicRaceDetector)
    assert len(rd.records) == 2


def test_flag_off_does_not_swallow_explicit_instance():
    """An explicitly constructed detector instance reflects a deliberate user
    choice and must keep tracing regardless of the global flag."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        detector = SymbolicRaceDetector()
        traced = triton_viz.trace(client=detector)(_dispatch_kernel)

        from triton_viz.core.trace import TritonTrace

        assert isinstance(
            traced, TritonTrace
        ), "explicit instance must be traced even when the env flag is off"

        x = torch.zeros(8, dtype=torch.float32)
        traced[(1,)](x, BLOCK=8)
        assert len(detector.records) == 2
    finally:
        cfg.enable_race_detector = saved


def test_repeat_launches_are_consistent(_isolate_race_detector_cfg):
    """Launching the same traced kernel twice must produce deterministic,
    identical block-execution counts. The previous grid-enumeration cache
    could skip block 0 on a cache hit and then run block 1 instead —
    different blocks run, different side effects, events captured from a
    non-deterministic representative. This regression asserts every launch
    behaves the same."""
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_dispatch_kernel)

    x1 = torch.zeros(8, dtype=torch.float32)
    traced[(2,)](x1, BLOCK=8)
    after_first = list(detector.records)

    detector.records.clear()
    x2 = torch.zeros(8, dtype=torch.float32)
    traced[(2,)](x2, BLOCK=8)
    after_second = list(detector.records)

    assert len(after_first) == len(after_second), (
        f"repeat launches must capture the same number of events; "
        f"got {len(after_first)} then {len(after_second)}"
    )
    assert [r.access_mode for r in after_first] == [r.access_mode for r in after_second]
    assert [r.source_location for r in after_first] == [
        r.source_location for r in after_second
    ]
