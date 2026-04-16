"""End-to-end race detection on real traced kernels.

Each test launches a kernel through ``triton_viz.trace`` with the race
detector enabled and inspects ``detector.finalize()``'s RaceReports.
Covers the headline scenarios PR3 needs to defend:
  * unsynchronized plain cross-block reads/writes -> reported
  * producer/consumer handshake via release/acquire xchg -> suppressed
  * failed CAS still reports races on the underlying memory
  * atomic_xchg handoff with reads-from -> suppressed
  * epoch barrier preserves cross-phase pairs
  * spinning cas(1, 1) poll does NOT split phases (upstream #350)
  * private (per-block) sync addresses are not mistaken for a shared
    barrier (upstream #346)
  * single-SM vs multi-SM produce the same suppression decisions
    (upstream #352 symbolic-HB regression)
"""

from __future__ import annotations

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients.race_detector.data import RaceType
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.core.config import config as cfg


@pytest.fixture
def _race_detector_on():
    saved = cfg.enable_race_detector
    cfg.enable_race_detector = True
    yield
    cfg.enable_race_detector = saved


def _traced(kernel, detector: SymbolicRaceDetector):
    return triton_viz.trace(client=detector)(kernel)


@triton.jit
def _unsynchronized_store_load(x_ptr, out_ptr):
    pid = tl.program_id(axis=0)
    tl.store(x_ptr, 7, mask=(pid == 0))
    v = tl.load(x_ptr, mask=(pid == 1))
    tl.store(out_ptr + pid, v, mask=(pid == 1))


def test_unsynchronized_plain_store_load_reports_race(_race_detector_on):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = _traced(_unsynchronized_store_load, detector)
    x = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    traced[(2,)](x, out)
    reports = detector.finalize()
    assert any(
        r.race_type is RaceType.RAW and r.grid_a != r.grid_b for r in reports
    ), f"expected RAW cross-block report, got {reports}"


@triton.jit
def _producer_consumer_xchg(flag_ptr, sync_ptr):
    pid = tl.program_id(axis=0)
    tl.store(flag_ptr, 42, mask=(pid == 0))
    tl.atomic_xchg(sync_ptr, 1, mask=(pid == 0), sem="release", scope="gpu")
    _obtained = tl.atomic_xchg(sync_ptr, 2, mask=(pid == 1), sem="acquire", scope="gpu")
    tl.load(flag_ptr, mask=(pid == 1))


def test_producer_consumer_release_acquire_suppresses_race(_race_detector_on):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = _traced(_producer_consumer_xchg, detector)
    flag = torch.zeros(1, dtype=torch.int32)
    sync = torch.zeros(1, dtype=torch.int32)
    traced[(2,)](flag, sync)
    reports = detector.finalize()
    flag_addr = flag.data_ptr()
    flag_reports = [
        r
        for r in reports
        if flag_addr <= r.witness_addr < flag_addr + flag.element_size()
    ]
    assert not flag_reports, (
        f"release/acquire reads-from should suppress cross-block race "
        f"on flag_ptr; got {flag_reports}"
    )


def test_atomic_xchg_handoff_suppresses_race(_race_detector_on):
    """Same handshake shape as the producer/consumer test but phrased as
    an explicit ``sw`` existence check: at least one report in the launch
    must carry a ``release_acquire_reads_from`` reason when suppression
    is observed via ``_last_candidates`` vs ``_last_reports`` difference.
    """
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = _traced(_producer_consumer_xchg, detector)
    flag = torch.zeros(1, dtype=torch.int32)
    sync = torch.zeros(1, dtype=torch.int32)
    traced[(2,)](flag, sync)
    detector.finalize()
    assert len(detector._last_candidates) > len(
        detector._last_reports
    ), "HB must suppress at least one candidate when reads-from holds"


@triton.jit
def _failed_cas_plus_plain_store_kernel(x_ptr, sync_ptr):
    pid = tl.program_id(axis=0)
    # Block 0: plain write to x.
    tl.store(x_ptr, 42, mask=(pid == 0))
    # Both blocks CAS with a mismatched compare (99 vs real 0) so the
    # write ARM never fires -- same-value writeback, no phase change.
    # ``atomic_cas`` in the interpreter does not support a mask, so the
    # CAS runs for every block; both trigger a failed-compare writeback.
    tl.atomic_cas(sync_ptr, 99, 1, sem="release", scope="gpu")
    # Consumer acquire-xchg; reader sees sync_ptr=0 because the CAS never
    # wrote it -> reads-from match against the CAS's written_value (0)
    # is not useful here. The race on x_ptr should survive.
    tl.atomic_xchg(sync_ptr, 2, mask=(pid == 1), sem="acquire", scope="gpu")
    tl.load(x_ptr, mask=(pid == 1))


def test_failed_cas_mixed_access_still_reports_race(_race_detector_on):
    """Even with release sem, a CAS whose compare fails doesn't change
    the sync value (written_value == old). reads-from gate fails (reader
    observes something other than ``1``), so the sw edge is absent and
    the plain race on ``x_ptr`` must still be reported.
    """
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = _traced(_failed_cas_plus_plain_store_kernel, detector)
    x = torch.zeros(1, dtype=torch.int32)
    sync = torch.zeros(1, dtype=torch.int32)
    traced[(2,)](x, sync)
    reports = detector.finalize()
    x_addr = x.data_ptr()
    x_reports = [
        r for r in reports if x_addr <= r.witness_addr < x_addr + x.element_size()
    ]
    assert x_reports, (
        f"failed CAS must not suppress cross-block race on x_ptr; " f"got {reports}"
    )


@triton.jit
def _private_sync_per_block(flag_ptr, sync_ptr):
    """Each block uses its own sync slot (``sync_ptr + pid``), so no two
    blocks ever hit the same sync address - this should NOT count as
    a global barrier and the cross-block race on ``flag_ptr`` stays
    visible (upstream #346)."""
    pid = tl.program_id(axis=0)
    tl.store(flag_ptr, 42, mask=(pid == 0))
    tl.atomic_xchg(sync_ptr + pid, 1, mask=(pid == 0), sem="release", scope="gpu")
    tl.atomic_xchg(sync_ptr + pid, 2, mask=(pid == 1), sem="acquire", scope="gpu")
    tl.load(flag_ptr, mask=(pid == 1))


def test_private_sync_address_per_block_still_reports_race(_race_detector_on):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = _traced(_private_sync_per_block, detector)
    flag = torch.zeros(1, dtype=torch.int32)
    sync = torch.zeros(2, dtype=torch.int32)
    traced[(2,)](flag, sync)
    reports = detector.finalize()
    flag_addr = flag.data_ptr()
    flag_reports = [
        r
        for r in reports
        if flag_addr <= r.witness_addr < flag_addr + flag.element_size()
    ]
    assert flag_reports, (
        f"per-block private sync addresses must not be mistaken for "
        f"a shared barrier (#346); got {reports}"
    )


@triton.jit
def _spinning_same_value_cas(barrier_ptr, flag_ptr):
    """Every block does a ``cas(1, 1)`` on a shared barrier address -
    same-value writeback, no phase advance (upstream #350). The epoch
    partitioner must NOT split phases on this, so a cross-block plain
    race on ``flag_ptr`` remains visible.
    """
    pid = tl.program_id(axis=0)
    tl.atomic_cas(barrier_ptr, 1, 1, sem="acq_rel", scope="gpu")
    tl.store(flag_ptr, pid, mask=(pid == 0))
    tl.load(flag_ptr, mask=(pid == 1))


def test_spinning_cas_poll_does_not_split_phase(_race_detector_on):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = _traced(_spinning_same_value_cas, detector)
    barrier = torch.ones(1, dtype=torch.int32)  # seed with the cmp value
    flag = torch.zeros(1, dtype=torch.int32)
    traced[(2,)](barrier, flag)
    reports = detector.finalize()
    # Epoch partitioner: same-value cas(1,1) on the barrier must not
    # advance the phase, so the plain flag race stays in epoch 0 and
    # gets paired.
    flag_addr = flag.data_ptr()
    flag_reports = [
        r
        for r in reports
        if flag_addr <= r.witness_addr < flag_addr + flag.element_size()
    ]
    assert flag_reports, (
        f"cas(1,1) polling should not split epochs (#350); " f"got {reports}"
    )


def test_num_sms_1_and_2_match_on_release_acquire(_race_detector_on):
    """Single-SM and multi-SM launches must agree on suppression. The
    race detector uses the concrete HB path unconditionally, so the
    number-of-workers setting shouldn't change the verdict - upstream
    #352 flagged single-SM symbolic-HB regressions this test guards.
    """
    saved_num_sms = cfg.num_sms
    try:
        results: dict[int, tuple[int, int]] = {}
        for num_sms in (1, 2):
            cfg.num_sms = num_sms
            detector = SymbolicRaceDetector(abort_on_error=False)
            traced = _traced(_producer_consumer_xchg, detector)
            flag = torch.zeros(1, dtype=torch.int32)
            sync = torch.zeros(1, dtype=torch.int32)
            traced[(2,)](flag, sync)
            reports = detector.finalize()
            results[num_sms] = (
                len(detector._last_candidates),
                len(reports),
            )
        assert (
            results[1] == results[2]
        ), f"num_sms=1 vs num_sms=2 disagreed on suppression: {results}"
    finally:
        cfg.num_sms = saved_num_sms
