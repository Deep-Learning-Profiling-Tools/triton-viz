"""Small footprint and ordering twins for real frontend disagreements."""

import pytest
import torch
import triton
import triton.language as tl
import triton_viz

from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.core.config import config as cfg


@pytest.fixture(autouse=True)
def _config():
    old = (cfg.enable_race_detector, cfg.num_sms, cfg.race_detector_fence_order)
    cfg.enable_race_detector = True
    cfg.num_sms = 1
    cfg.race_detector_fence_order = True
    triton_viz.clear()
    yield
    triton_viz.clear()
    cfg.enable_race_detector, cfg.num_sms, cfg.race_detector_fence_order = old


def _run(kernel, *args, grid=(2,)):
    detector = SymbolicRaceDetector()
    triton_viz.trace(detector)(kernel)[grid](*args)
    assert detector.last_status == "ok", detector.unsupported_reason
    return detector


@triton.jit
def _separate_aranges(out, ALIAS: tl.constexpr):
    pid = tl.program_id(0)
    mask = tl.arange(0, 8) < 4
    col = tl.arange(0, 8)
    offs = pid * 8 + (col % 2 if ALIAS else col)
    tl.store(out + offs, 1, mask)


def test_same_axis_aranges_do_not_fabricate_intra_instance_waw():
    assert _run(_separate_aranges, torch.zeros(16), False).last_reports == []


def test_same_axis_coupling_retains_real_duplicate_lane_waw():
    assert _run(_separate_aranges, torch.zeros(16), True).last_reports


@triton.jit
def _nonzero_range_start(out):
    mask = tl.arange(8, 16) < 12
    col = tl.arange(0, 8)
    tl.store(out + tl.program_id(0) * 8 + col, 1, mask)


def test_same_axis_coupling_uses_coordinates_not_arange_values():
    assert _run(_nonzero_range_start, torch.zeros(16)).last_reports == []


@triton.jit
def _two_axes(out):
    rows = tl.arange(0, 8)
    cols = tl.arange(0, 8)
    # The diagonal addresses are unique; off-diagonal lanes (i,j) and
    # (j,i) collide. Equating equal-extent row/column axes hides this WAW.
    addr = rows[:, None] + cols[None, :]
    tl.store(out + addr, 1)


def test_equal_extent_independent_axes_remain_independent():
    assert _run(_two_axes, torch.zeros(16), grid=(1,)).last_reports


@triton.jit
def _loop_barriers(temp, out, FENCED: tl.constexpr):
    p = tl.program_id(0)
    offsets = p * 8 + tl.arange(0, 8)
    for stage in range(2):
        tl.store(temp + offsets, stage)
        if FENCED:
            tl.debug_barrier()
        value = tl.load(temp + p * 8 + (tl.arange(0, 8) ^ 1))
        if FENCED:
            tl.debug_barrier()
        tl.store(out + offsets + stage * 16, value)


def test_loop_fences_order_store_and_permuted_load():
    assert (
        _run(_loop_barriers, torch.zeros(16), torch.zeros(32), True).last_reports == []
    )


def test_unfenced_loop_still_reports_conflict():
    assert _run(_loop_barriers, torch.zeros(16), torch.zeros(32), False).last_reports


@triton.jit
def _nested_loop_fence(temp, out, BETWEEN: tl.constexpr):
    p = tl.program_id(0)
    offs = p * 8 + tl.arange(0, 8)
    for i in range(2):
        if not BETWEEN:
            tl.debug_barrier()
        tl.store(temp + offs, i)
        if BETWEEN:
            tl.debug_barrier()
        for j in range(2):
            value = tl.load(temp + p * 8 + (tl.arange(0, 8) ^ 1))
            tl.store(out + offs + (i * 2 + j) * 16, value)
        tl.debug_barrier()


@pytest.mark.parametrize("between", [True, False])
def test_nested_flush_preserves_fence_position_not_merely_presence(between):
    detector = _run(_nested_loop_fence, torch.zeros(16), torch.zeros(64), between)
    assert bool(detector.last_reports) is (not between)


def test_fences_do_not_hide_cross_instance_conflicts():
    @triton.jit
    def same_region(temp, out):
        p = tl.program_id(0)
        offs = tl.arange(0, 8)
        for stage in range(2):
            tl.store(temp + offs, stage)
            tl.debug_barrier()
            value = tl.load(temp + (offs ^ 1))
            tl.debug_barrier()
            tl.store(out + p * 16 + stage * 8 + offs, value)

    assert _run(same_region, torch.zeros(8), torch.zeros(32)).last_reports


@triton.jit
def _boolean_mask(out, USE_AND: tl.constexpr):
    pid = tl.program_id(0)
    col = tl.arange(0, 8)
    row_mask = pid < 2
    if USE_AND:
        mask = col < 4 and row_mask
    else:
        mask = (col < 4) & row_mask
    tl.store(out + pid * 4 + col, 1, mask)


@pytest.mark.parametrize("use_and", [True, False])
def test_boolean_mask_retains_both_tensor_operands(use_and):
    assert _run(_boolean_mask, torch.zeros(12), use_and).last_reports == []


@pytest.mark.parametrize("use_and", [True, False])
def test_concrete_enumeration_uses_same_boolean_mask_semantics(use_and):
    from triton_viz.clients.race_detector.concrete_enum import enumerate_launch

    result = enumerate_launch(
        _boolean_mask, (torch.zeros(12),), {"USE_AND": use_and}, (2,)
    )
    assert result.status == "ok", result.reason
    assert result.reports == []


@triton.jit
def _boolean_or_collision(out):
    col = tl.arange(0, 8)
    mask = (col < 0) or (col < 4)
    tl.store(out + col % 2, 1, mask)


def test_tensor_or_cannot_drop_the_active_collision_arm():
    from triton_viz.clients.race_detector.concrete_enum import enumerate_launch

    assert _run(_boolean_or_collision, torch.zeros(8), grid=(1,)).last_reports
    triton_viz.clear()
    result = enumerate_launch(_boolean_or_collision, (torch.zeros(8),), {}, (1,))
    assert result.status == "races", result.reason


def test_boolean_helper_preserves_constexpr_short_circuit():
    from triton_viz.core.frontend.triton import _triton_boolean_operator

    def forbidden():
        raise AssertionError("short-circuited operand executed")

    assert _triton_boolean_operator(True, lambda: False, forbidden) is False
    assert _triton_boolean_operator(False, lambda: True, forbidden) is True


def test_boolean_helper_evaluates_each_nonshortcircuited_operand_once():
    from triton_viz.core.frontend.triton import _triton_boolean_operator

    seen = []

    def operand(value):
        seen.append(value)
        return value

    assert _triton_boolean_operator(True, lambda: operand(1), lambda: operand(2)) == 2
    assert seen == [1, 2]
