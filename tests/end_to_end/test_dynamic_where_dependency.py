"""Four-element interpreter controls for positional select dependencies."""

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


@triton.jit
def _selected_store(
    x,
    condition,
    DIRECT_FIRST: tl.constexpr,
    ALTERNATIVE: tl.constexpr,
    LOADED_CONDITION: tl.constexpr,
):
    offsets = tl.arange(0, 4)
    value = tl.load(x + offsets)
    if ALTERNATIVE == 0:
        alternative = tl.reshape(tl.trans(tl.reshape(value, (2, 2))), (4,))
    elif ALTERNATIVE == 1:
        alternative = tl.full((4,), 0, tl.float32)
    else:
        alternative = value + 1
    if LOADED_CONDITION:
        mask = value > 0
    else:
        mask = tl.load(condition + offsets) > 0
    if DIRECT_FIRST:
        output = tl.where(mask, value, alternative)
    else:
        output = tl.where(mask, alternative, value)
    tl.store(x + offsets, output)


def _run(direct_first, alternative, loaded_condition=False):
    detector = SymbolicRaceDetector()
    x = torch.arange(1, 5, dtype=torch.float32)
    # Choose the alternative arm; the direct arm is inactive at every lane.
    condition = torch.full((4,), 0 if direct_first else 1, dtype=torch.int32)
    triton_viz.trace(detector)(_selected_store)[(1,)](
        x, condition, direct_first, alternative, loaded_condition
    )
    assert detector.last_status == "ok", detector.unsupported_reason
    return detector


@pytest.mark.parametrize("direct_first", [False, True])
@pytest.mark.parametrize("alternative", [0, 1])
def test_inactive_direct_arm_cannot_hide_the_unordered_read_write(
    direct_first, alternative
):
    detector = _run(direct_first, alternative)
    assert (
        detector.last_reports
    ), "the inactive direct arm incorrectly ordered the read and store"


@pytest.mark.parametrize("direct_first", [False, True])
def test_both_direct_value_arms_retain_order(direct_first):
    assert _run(direct_first, 2).last_reports == []


@pytest.mark.parametrize("direct_first", [False, True])
def test_loaded_condition_supplies_its_own_positional_order(direct_first):
    assert _run(direct_first, 0, loaded_condition=True).last_reports == []


def _enumerate(direct_first, alternative, loaded_condition=False):
    from triton_viz.clients.race_detector.concrete_enum import enumerate_launch

    return enumerate_launch(
        _selected_store,
        (
            torch.arange(1, 5, dtype=torch.float32),
            torch.full((4,), 0 if direct_first else 1, dtype=torch.int32),
        ),
        {
            "DIRECT_FIRST": direct_first,
            "ALTERNATIVE": alternative,
            "LOADED_CONDITION": loaded_condition,
        },
        (1,),
    )


@pytest.mark.parametrize("direct_first", [False, True])
@pytest.mark.parametrize("alternative", [0, 1])
def test_enum_inactive_direct_arm_never_becomes_a_proof(direct_first, alternative):
    result = _enumerate(direct_first, alternative)
    assert result.status in ("races", "unsupported"), result
    if result.status == "unsupported":
        assert result.reason.startswith("dependency-order:"), result.reason


@pytest.mark.parametrize("direct_first", [False, True])
def test_enum_both_value_arms_keep_their_common_dependency(direct_first):
    result = _enumerate(direct_first, 2)
    assert result.status == "ok", result.reason
    assert result.reports == []


@pytest.mark.parametrize("direct_first", [False, True])
def test_enum_loaded_condition_keeps_its_dependency(direct_first):
    result = _enumerate(direct_first, 0, loaded_condition=True)
    assert result.status == "ok", result.reason
    assert result.reports == []
