import triton
import triton.language as tl

import tilelens
from tilelens.clients import RaceDetector
from tilelens.clients.race_detector.race_detector import (
    SymbolicRaceDetector,
    NullRaceDetector,
)
from tilelens.core.config import config as cfg


# ======== Factory Test ========


def test_race_detector_factory_toggle():
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = True
        assert isinstance(RaceDetector(), SymbolicRaceDetector)

        cfg.enable_race_detector = False
        assert isinstance(RaceDetector(), NullRaceDetector)
    finally:
        cfg.enable_race_detector = saved


@triton.jit
def _dispatch_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs))


# ======== Flag-off escape hatch ========


def test_flag_off_returns_raw_kernel():
    """With ENABLE_RACE_DETECTOR=0 a string-dispatched trace decorator must
    leave the kernel uninstrumented so opting in has literally zero impact
    when the flag is off."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        traced = tilelens.trace("race_detector")(_dispatch_kernel)

        # Should be the raw JIT kernel, not a TritonTrace wrapper.
        from tilelens.core.trace import TritonTrace

        assert not isinstance(traced, TritonTrace)
        assert traced is _dispatch_kernel
    finally:
        cfg.enable_race_detector = saved


def test_flag_off_returns_raw_kernel_for_factory_instance():
    """ENABLE_RACE_DETECTOR=0 + trace(client=RaceDetector()) must take the
    flag-off fast path. The factory's ``__new__`` already returned a
    NullRaceDetector; the trace decorator must recognize that and leave the
    kernel untraced, otherwise it would wrap the kernel and then crash at
    callback-registration time with NullSymbolicClient's raising methods.

    Identity-check alone is sufficient to prove the fix: without it, the
    predicate would miss the factory-returned instance and ``traced`` would
    be a ``TritonTrace`` wrapper.
    """
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        # The factory call happens after the flag flip, so __new__ dispatches
        # to NullRaceDetector.
        traced = tilelens.trace(client=RaceDetector())(_dispatch_kernel)

        from tilelens.core.trace import TritonTrace

        assert not isinstance(traced, TritonTrace)
        assert traced is _dispatch_kernel
    finally:
        cfg.enable_race_detector = saved
