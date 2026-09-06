"""Concrete enumeration must preserve the enabled fence/dependency model."""

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceType
from triton_viz.clients.race_detector.concrete_enum import enumerate_launch
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.core.config import config as cfg


@pytest.fixture(autouse=True)
def fence_order(monkeypatch):
    monkeypatch.setattr(cfg, "race_detector_fence_order", True)
    monkeypatch.setattr(cfg, "enable_race_detector", True)
    monkeypatch.setattr(cfg, "num_sms", 1)
    triton_viz.clear()
    yield
    triton_viz.clear()


def run(kernel, *args, **kwargs):
    triton_viz.clear()
    return enumerate_launch(kernel, args, kwargs, (1,))


@triton.jit
def conflict_pair(x, out, KIND: tl.constexpr, FENCE: tl.constexpr):
    if FENCE == "before":
        tl.debug_barrier()
    if KIND == "WAR":
        value = tl.load(x)
        if FENCE == "between":
            tl.debug_barrier()
        tl.store(x, 7)
        tl.store(out, value)
    elif KIND == "RAW":
        tl.store(x, 7)
        if FENCE == "between":
            tl.debug_barrier()
        value = tl.load(x)
        tl.store(out, value)
    else:
        tl.store(x, 7)
        if FENCE == "between":
            tl.debug_barrier()
        tl.store(x, 9)
    if FENCE == "after":
        tl.debug_barrier()


@pytest.mark.parametrize("kind", ["RAW", "WAR", "WAW"])
@pytest.mark.parametrize("fence", ["none", "before", "between", "after"])
def test_order_matches_symbolic_frontend_and_requires_intervening_fence(kind, fence):
    x, out = torch.ones(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    result = run(conflict_pair, x, out, KIND=kind, FENCE=fence)
    assert x.tolist() == [1] and out.tolist() == [0]
    triton_viz.clear()
    detector = SymbolicRaceDetector()
    triton_viz.trace(client=detector)(conflict_pair)[(1,)](
        x, out, KIND=kind, FENCE=fence
    )
    expected = set() if fence == "between" else {getattr(RaceType, kind)}
    assert {r.race_type for r in detector.last_reports} == expected
    assert {r.race_type for r in result.reports} == expected
    assert result.status == ("ok" if not expected else "races")
    assert all(
        r.witness_grid_a == r.witness_grid_b == (0, 0, 0) for r in result.reports
    )


@pytest.mark.parametrize("kind", ["RAW", "WAR", "WAW"])
def test_legacy_program_order_is_explicit_and_unchanged(monkeypatch, kind):
    monkeypatch.setattr(cfg, "race_detector_fence_order", False)
    result = run(
        conflict_pair,
        torch.ones(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        KIND=kind,
        FENCE="none",
    )
    assert result.status == "ok"
    assert not result.reports


@triton.jit
def dependent_update(x, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets)
    tl.store(x + offsets + SHIFT, value * 2 + 1)


def test_same_position_elementwise_dependency_orders_in_place_update():
    result = run(dependent_update, torch.arange(8, dtype=torch.int32), SHIFT=0, BLOCK=8)
    assert result.status == "ok"
    assert not result.reports


def test_dependency_does_not_order_aliases_at_different_positions():
    result = run(dependent_update, torch.arange(9, dtype=torch.int32), SHIFT=1, BLOCK=8)
    assert result.status == "races"
    assert any(r.race_type == RaceType.WAR for r in result.reports)


@triton.jit
def reduce_update(x, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets)
    total = tl.sum(value, 0)
    tl.store(x + offsets, total)


@triton.jit
def transpose_update(x, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)[:, None]
    col = tl.arange(0, BLOCK)[None, :]
    offsets = row * BLOCK + col
    value = tl.load(x + offsets)
    tl.store(x + offsets, tl.trans(value))


@pytest.mark.parametrize(
    "kernel,block,size", [(reduce_update, 4, 4), (transpose_update, 2, 4)]
)
def test_non_elementwise_taint_never_certifies_same_position_order(kernel, block, size):
    result = run(kernel, torch.arange(size, dtype=torch.float32), BLOCK=block)
    assert result.status in ("races", "unsupported")
    if result.status == "unsupported":
        assert result.reason


@triton.jit
def masked_dependency(x, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets, mask=offsets != 0, other=0)
    tl.store(x + offsets + SHIFT, value + 1, mask=offsets != 0)


@pytest.mark.parametrize("shift", [0, 1])
def test_active_lane_filtering_preserves_original_positions(shift):
    result = run(
        masked_dependency, torch.arange(9, dtype=torch.int32), SHIFT=shift, BLOCK=8
    )
    assert result.status == ("races" if shift else "ok")


@triton.jit
def written_index(index, out, BEFORE: tl.constexpr, FENCED: tl.constexpr):
    if BEFORE:
        tl.store(index, 1)
        if FENCED:
            tl.debug_barrier()
    value = tl.load(index)
    tl.store(out + value, 7)
    if not BEFORE:
        if FENCED:
            tl.debug_barrier()
        tl.store(index, 1)


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("fenced", [False, True])
def test_value_source_requires_order_against_earlier_and_later_writers(before, fenced):
    index, out = torch.zeros(1, dtype=torch.int32), torch.zeros(4, dtype=torch.int32)
    result = run(written_index, index, out, BEFORE=before, FENCED=fenced)
    if fenced:
        assert result.status == "ok"
    else:
        assert result.status == "unsupported"
        assert result.reason.startswith("value-source-order:")
    assert index.tolist() == [0] and out.tolist() == [0, 0, 0, 0]


@triton.jit
def competing_writers(index, out, ORDER_WRITERS: tl.constexpr):
    tl.store(index, 0)
    if ORDER_WRITERS:
        tl.debug_barrier()
    tl.store(index, 1)
    tl.debug_barrier()
    value = tl.load(index)
    tl.store(out + value, 7)


@pytest.mark.parametrize("ordered", [False, True])
def test_fence_after_two_writers_does_not_select_a_deterministic_winner(ordered):
    result = run(
        competing_writers,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        ORDER_WRITERS=ordered,
    )
    if ordered:
        assert result.status == "ok"
    else:
        assert result.status == "unsupported"
        assert result.reason.startswith("value-source-order:")


@triton.jit
def duplicate_writer_index(index, out):
    offsets = tl.arange(0, 4)
    tl.store(index + offsets % 1, offsets)
    tl.debug_barrier()
    value = tl.load(index)
    tl.store(out + value, 7)


def test_fenced_duplicate_lane_writes_do_not_supply_a_stable_index():
    result = run(
        duplicate_writer_index,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )
    assert result.status == "unsupported"
    assert result.reason.startswith("value-source-order:")


@triton.jit
def unstable_relay(index, scratch, out):
    tl.store(index, 1)
    value = tl.load(index)
    tl.store(scratch, value)
    tl.debug_barrier()
    relayed = tl.load(scratch)
    tl.store(out + relayed, 7)


def test_memory_relay_checks_the_original_unordered_value_source():
    result = run(
        unstable_relay,
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )
    assert result.status == "unsupported"
    assert result.reason.startswith("value-source-order:")


@triton.jit
def inactive_intermediate_anchor(x, y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets)
    intermediate = tl.load(y + offsets, mask=value < 0, other=1)
    tl.store(x + offsets, intermediate)


def test_inactive_intermediate_load_does_not_certify_transitive_order():
    result = run(
        inactive_intermediate_anchor,
        torch.ones(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        BLOCK=4,
    )
    assert result.status == "unsupported"
    assert result.reason.startswith("value-source-order:")


@triton.jit
def cast_update(x, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(x + offsets).to(tl.float32)
    tl.store(x + offsets, (value + 1).to(tl.int32))


def test_elementwise_cast_retains_positional_dependency():
    result = run(cast_update, torch.arange(4, dtype=torch.int32), BLOCK=4)
    assert result.status == "ok"
