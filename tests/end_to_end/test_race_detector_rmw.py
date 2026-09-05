"""End-to-end dynamic-mode tests for RMW-return modeling (spec part B).

Interpreter capture mirror of the CAS machinery: an integer RMW's return
becomes a modeled observation (the record's ``old_value``), downstream
masks reference the same variable, and the solver's counting/reads-through
axioms prove the synchronization patterns. Float RMW returns keep the
sentinel boundary; addresses derived from any RMW return keep failing stop.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.core.config import config as cfg


NBLK = 4


@pytest.fixture
def _isolate_cfg():
    saved_enable = cfg.enable_race_detector
    saved_num_sms = cfg.num_sms
    cfg.enable_race_detector = True
    cfg.num_sms = 1
    triton_viz.clear()
    yield
    triton_viz.clear()
    cfg.enable_race_detector = saved_enable
    cfg.num_sms = saved_num_sms


def _run(kernel, grid, *args, **kwargs):
    triton_viz.clear()
    detector = RaceDetector()
    traced = triton_viz.trace(client=detector)(kernel)
    traced[grid](*args, **kwargs)
    return detector


def _line_no(kernel, needle: str) -> int:
    import inspect

    source_fn = kernel.fn if hasattr(kernel, "fn") else kernel
    lines, start = inspect.getsourcelines(source_fn)
    for idx, line in enumerate(lines):
        if needle in line:
            return start + idx
    raise AssertionError(f"Could not find source line containing: {needle}")


# ──────────────────── last-block-done ────────────────────


@triton.jit
def _lbd_acq_rel_kernel(partial_ptr, counter_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.store(partial_ptr + pid, 1)
    old = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    done = old == 3
    p = tl.load(partial_ptr + 0, mask=done, other=0)
    tl.store(out_ptr, p, mask=done)


@triton.jit
def _lbd_relaxed_kernel(partial_ptr, counter_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.store(partial_ptr + pid, 1)
    old = tl.atomic_add(counter_ptr, 1, sem="relaxed")
    done = old == 3
    p = tl.load(partial_ptr + 0, mask=done, other=0)
    tl.store(out_ptr, p, mask=done)


def _lbd_args():
    return (
        torch.zeros(NBLK, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


def test_last_block_done_acq_rel_proved(_isolate_cfg):
    detector = _run(_lbd_acq_rel_kernel, (NBLK,), *_lbd_args())
    assert detector.last_status == "ok"
    assert detector.last_reports == []
    rmw = [r for r in detector.records if r.atomic_kind == "rmw"]
    assert len(rmw) == 1
    assert rmw[0].rmw_op == "add"
    assert rmw[0].old_value is not None
    assert rmw[0].rmw_operand is not None
    assert rmw[0].sem == "acq_rel"


def test_last_block_done_relaxed_reports_race(_isolate_cfg):
    detector = _run(_lbd_relaxed_kernel, (NBLK,), *_lbd_args())
    assert detector.last_status == "ok"
    assert detector.last_reports, "relaxed counter must expose the race"
    lines = {
        loc[1]
        for rep in detector.last_reports
        for loc in (
            rep.first.record.source_location,
            rep.second.record.source_location,
        )
    }
    assert _line_no(_lbd_relaxed_kernel, "tl.store(partial_ptr + pid, 1)") in lines
    assert _line_no(_lbd_relaxed_kernel, "p = tl.load(partial_ptr + 0") in lines


# ──────────────────── single winner ────────────────────


@triton.jit
def _single_winner_kernel(flag_ptr, out_ptr):
    old = tl.atomic_add(flag_ptr, 1, sem="relaxed")
    win = old == 0
    tl.store(out_ptr, 1, mask=win)


@triton.jit
def _two_winner_kernel(flag_ptr, out_ptr):
    old = tl.atomic_add(flag_ptr, 1, sem="relaxed")
    win = old <= 1
    tl.store(out_ptr, 1, mask=win)


def test_single_winner_store_proved(_isolate_cfg):
    detector = _run(
        _single_winner_kernel,
        (NBLK,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_two_winner_store_races(_isolate_cfg):
    """Mutation twin: old <= 1 admits two writers of the same slot."""
    detector = _run(
        _two_winner_kernel,
        (NBLK,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    assert detector.last_status == "ok"
    assert detector.last_reports


# ──────────────────── preserved boundaries ────────────────────


@triton.jit
def _work_queue_kernel(head_ptr, buf_ptr):
    pid = tl.program_id(0)
    idx = tl.atomic_add(head_ptr, 1, sem="relaxed")
    tl.store(buf_ptr + idx, pid)


def test_rmw_return_in_address_stays_unsupported(_isolate_cfg):
    """B.2: the interpreter's address-position boundary is untouched — the
    static track is the one that proves the work-queue pattern."""
    detector = _run(
        _work_queue_kernel,
        (NBLK,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(NBLK, dtype=torch.int32),
    )
    assert detector.last_status == "unsupported"


@triton.jit
def _float_rmw_gated_kernel(fcounter_ptr, out_ptr):
    old = tl.atomic_add(fcounter_ptr, 1.0, sem="acq_rel")
    done = old == 3.0
    tl.store(out_ptr, 1.0, mask=done)


def test_float_rmw_return_downstream_stays_unsupported(_isolate_cfg):
    """B.5: float-typed RMW returns keep the sentinel."""
    detector = _run(
        _float_rmw_gated_kernel,
        (NBLK,),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )
    assert detector.last_status == "unsupported"


@triton.jit
def _float_rmw_unused_kernel(fcounter_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.atomic_add(fcounter_ptr, 1.0, sem="relaxed")
    tl.store(out_ptr + pid, 1.0)


def test_float_rmw_unused_return_still_checked(_isolate_cfg):
    """A float RMW whose return is unused keeps the plain footprint model:
    the launch is analyzable and per-pid stores stay race-free."""
    detector = _run(
        _float_rmw_unused_kernel,
        (NBLK,),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(NBLK, dtype=torch.float32),
    )
    assert detector.last_status == "ok"
    assert detector.last_reports == []


# ──────────────────── atomic_max in mask position ────────────────────


@triton.jit
def _atomic_max_scale_kernel(mx_ptr, out_ptr):
    pid = tl.program_id(0)
    m = tl.atomic_max(mx_ptr, pid, sem="relaxed")
    keep = m >= 0
    tl.store(out_ptr + pid, m, mask=keep)


def test_atomic_max_return_in_mask_proved(_isolate_cfg):
    """B.4 atomic_max_scale: the return feeds a mask; per-pid stores stay
    disjoint whatever the observation, so the launch is proved clean."""
    detector = _run(
        _atomic_max_scale_kernel,
        (NBLK,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(NBLK, dtype=torch.int32),
    )
    assert detector.last_status == "ok"
    assert detector.last_reports == []
    rmw = [r for r in detector.records if r.atomic_kind == "rmw"]
    assert len(rmw) == 1 and rmw[0].rmw_op == "max"


def test_backend_is_symbolic(_isolate_cfg):
    detector = _run(
        _single_winner_kernel,
        (2,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )
    assert isinstance(detector, SymbolicRaceDetector)
