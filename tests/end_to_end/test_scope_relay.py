"""Per-hop moral strength across reads-through relays (PTX observation order).

The release-sequence chain w -rf-> a^R -rmw-> a^W -rf-> r carries
synchronizes-with only when EVERY rf hop is morally strong: PTX's
observation order recurses over ``morally_strong ∩ rf`` and scoped-RC11
release sequences use ``incl ∩ rf`` at every relay. Judging scope only
between the chain's head and its final reader is unsound (external-expert
counterexample, 2026-08-26): a relay whose scope excludes the head lets
the release publish through it anyway.

The litmus pair below is the smallest shape that reaches the two-copy
encoding's chain machinery: the relay CAS runs in the CONSUMER's own
instance, between its producer and its final acquire. (A third-party
relay instance is already handled conservatively: it is unmodeled, its
write reaches the reader only as ``rf_unknown``, and ``rf_unknown``
never yields synchronizes-with. A gate-synchronized three-party version
folded to two instances orders the data pair through the gate itself and
proves race-free for a legitimate reason.)

With the relay at ``cta`` scope, the hop from the producer's release
(block 0) into the relay (block 1) is morally weak: the chain breaks, and
in the execution where the final acquire reads the relay's write the data
pair is unordered and must be reported. With the relay at ``gpu`` scope
every hop is morally strong and the launch is race-free.
"""

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.core.config import config as cfg


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


@triton.jit
def _scope_relay_cta_kernel(data_ptr, flag_ptr, out_ptr):
    pid = tl.program_id(0)
    prod = pid == 0
    cons = pid == 1
    tl.store(data_ptr, 1, mask=prod)
    tl.debug_barrier()
    w_cmp = tl.where(prod, 0, -1)
    tl.atomic_cas(flag_ptr, w_cmp, 1, sem="release", scope="gpu")
    a_cmp = tl.where(cons, 1, -1)
    tl.atomic_cas(flag_ptr, a_cmp, 1, sem="relaxed", scope="cta")
    r_cmp = tl.where(cons, 1, -1)
    r = tl.atomic_cas(flag_ptr, r_cmp, 1, sem="acquire", scope="gpu")
    tl.debug_barrier()
    ok = cons & (r == 1)
    x = tl.load(data_ptr, mask=ok, other=0)
    tl.store(out_ptr, x, mask=ok)


@triton.jit
def _scope_relay_gpu_kernel(data_ptr, flag_ptr, out_ptr):
    pid = tl.program_id(0)
    prod = pid == 0
    cons = pid == 1
    tl.store(data_ptr, 1, mask=prod)
    tl.debug_barrier()
    w_cmp = tl.where(prod, 0, -1)
    tl.atomic_cas(flag_ptr, w_cmp, 1, sem="release", scope="gpu")
    a_cmp = tl.where(cons, 1, -1)
    tl.atomic_cas(flag_ptr, a_cmp, 1, sem="relaxed", scope="gpu")
    r_cmp = tl.where(cons, 1, -1)
    r = tl.atomic_cas(flag_ptr, r_cmp, 1, sem="acquire", scope="gpu")
    tl.debug_barrier()
    ok = cons & (r == 1)
    x = tl.load(data_ptr, mask=ok, other=0)
    tl.store(out_ptr, x, mask=ok)


def _args():
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


def test_cta_relay_breaks_chain_and_reports_data_race(_isolate_cfg):
    detector = _run(_scope_relay_cta_kernel, (2,), *_args())
    assert detector.last_status == "ok"
    lines = {
        loc[1]
        for rep in detector.last_reports
        for loc in (
            rep.first.record.source_location,
            rep.second.record.source_location,
        )
    }
    store_line = _line_no(_scope_relay_cta_kernel, "tl.store(data_ptr, 1, mask=prod)")
    load_line = _line_no(_scope_relay_cta_kernel, "x = tl.load(data_ptr")
    assert store_line in lines and load_line in lines, (
        "the cta-scoped relay must break the release chain: in the "
        "execution where the final acquire reads the relay's write the "
        "data pair is unordered and must be reported, "
        f"got report lines {sorted(lines)}"
    )


def test_gpu_relay_keeps_chain_and_proves(_isolate_cfg):
    detector = _run(_scope_relay_gpu_kernel, (2,), *_args())
    assert detector.last_status == "ok"
    assert detector.last_reports == [], (
        "with every hop morally strong the relayed release sequence "
        "orders the data pair; the launch is race-free"
    )
