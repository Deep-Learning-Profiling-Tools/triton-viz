"""End-to-end test for the A2 gate through the real warmup capture.

Traces a release/acquire atomic kernel with the compiled-mode race
detector and checks (a) the lowered artifacts were captured from the
warmup (the S1 exit of impl-spec-a2-gate) and (b) the gate's verdict.

The suite's pinned triton 3.6.0 PREDATES triton PR #10816 (the CTA
barrier insertion for atomic memory semantics), so on this pin the gate
must report a VIOLATION: the A2-class defect is live in the toolchain
the corpus runs on. If the pin is ever moved past the fix, this test's
expectation flips to "verified" (see tests/golden/a2gate/).
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.core.config import config

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warmup compilation needs a CUDA driver"
)


@pytest.fixture
def _enable_race_detector():
    saved = config.enable_race_detector
    config.enable_race_detector = True
    try:
        yield
    finally:
        config.enable_race_detector = saved


@requires_cuda
def test_gate_flags_release_acquire_on_prefix_pin(_enable_race_detector):
    detector = RaceDetector(compile=True)
    assert isinstance(detector, CompiledRaceDetector)

    @triton_viz.trace(detector)
    @triton.jit
    def publish(data_ptr, flag_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, 1.0)
        tl.atomic_add(flag_ptr, 1, sem="release", scope="gpu")

    data = torch.zeros(256, dtype=torch.float32, device="cuda")
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    publish[(2,)](data, flag, BLOCK=128)

    # S1 exit: the warmup delivered the lowered artifacts.
    assert detector.last_lowering_status != "no_lowered"
    # Triton 3.6.0 is pre-#10816: the release atomic has no CTA barrier
    # before it, and the gate says so with the site named.
    assert detector.last_lowering_status == "violation", (
        detector.last_lowering_status,
        detector.last_lowering_reason,
    )
    assert any(
        "release" in r and "before" in r for r in detector.last_lowering_reports
    ), detector.last_lowering_reports


@requires_cuda
def test_gate_vacuous_on_atomic_free_kernel(_enable_race_detector):
    detector = RaceDetector(compile=True)

    @triton_viz.trace(detector)
    @triton.jit
    def add(x_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) + 1)

    x = torch.zeros(256, dtype=torch.float32, device="cuda")
    out = torch.zeros(256, dtype=torch.float32, device="cuda")
    add[(2,)](x, out, BLOCK=128)

    assert detector.last_lowering_status == "verified"
    assert detector.last_lowering_reports == []
