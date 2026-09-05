"""Fence-ordered intra-instance semantics (paper design-fence-order.md,
option A), behind ``cfg.race_detector_fence_order``.

Legacy reading (flag off): every earlier tile operation of an instance is
program-ordered before every later one, so a same-instance store-then-load
of one address is never a race. Fence-ordered reading (flag on): only a
tile-level fence (``tl.debug_barrier``) orders distinct operations of one
instance, so the unfenced pair races and the fenced pair does not; the
guarded producer/consumer idiom needs a fence between its data access and
its atomic on each side.
"""

import inspect

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector
from triton_viz.core.config import config as cfg

N = 64


@pytest.fixture
def isolate_cfg():
    saved = (cfg.enable_race_detector, cfg.num_sms, cfg.race_detector_fence_order)
    cfg.enable_race_detector = True
    cfg.num_sms = 1
    triton_viz.clear()
    yield
    triton_viz.clear()
    cfg.enable_race_detector, cfg.num_sms, cfg.race_detector_fence_order = saved


@pytest.fixture
def fence_order_on(isolate_cfg):
    cfg.race_detector_fence_order = True
    yield


@pytest.fixture
def fence_order_off(isolate_cfg):
    cfg.race_detector_fence_order = False
    yield


def _run(kernel, grid, *args):
    triton_viz.clear()
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(kernel)
    traced[grid](*args)
    return detector


def _line_no(kernel, needle: str) -> int:
    source_fn = kernel.fn if hasattr(kernel, "fn") else kernel
    lines, start = inspect.getsourcelines(source_fn)
    for idx, line in enumerate(lines):
        if needle in line:
            return start + idx
    raise AssertionError(f"no source line contains {needle!r}")


def _report_lines(detector) -> set[frozenset[int]]:
    out = set()
    for report in detector.last_reports:
        out.add(
            frozenset(
                (
                    report.first.record.source_location[1],
                    report.second.record.source_location[1],
                )
            )
        )
    return out


# ── Figure 1 shape: store the whole tile, then load own slot ─────────


@triton.jit
def _store_then_load_kernel(hist_ptr, out_ptr, N: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


@triton.jit
def _store_fence_load_kernel(hist_ptr, out_ptr, N: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    tl.debug_barrier()
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


def _fig1_args():
    return torch.zeros(N, dtype=torch.int32), torch.zeros(N, dtype=torch.int32)


def test_legacy_reading_never_reports_the_intra_instance_pair(fence_order_off):
    detector = _run(_store_then_load_kernel, (1,), *_fig1_args(), N)
    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_unfenced_store_then_load_races_within_one_instance(fence_order_on):
    detector = _run(_store_then_load_kernel, (1,), *_fig1_args(), N)
    assert detector.last_status == "ok"
    expected = frozenset(
        (
            _line_no(_store_then_load_kernel, "tl.store(hist_ptr + offs, offs)"),
            _line_no(_store_then_load_kernel, "n = tl.load(hist_ptr + p)"),
        )
    )
    assert expected in _report_lines(detector)


def test_fence_records_are_captured(fence_order_on):
    detector = _run(_store_fence_load_kernel, (1,), *_fig1_args(), N)
    assert len(detector.fence_seqs) == 1
    seqs = sorted(r.program_seq for r in detector.records)
    assert seqs[0] < detector.fence_seqs[0] < seqs[-1]


def test_fenced_store_then_load_is_race_free(fence_order_on):
    detector = _run(_store_fence_load_kernel, (1,), *_fig1_args(), N)
    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_grid_launch_keeps_the_cross_instance_race_either_way(fence_order_on):
    # Every instance stores the whole tile: the cross-instance WAW/RAW is
    # independent of intra-instance ordering, so the fenced kernel still
    # reports it while the intra-instance pair is gone.
    detector = _run(_store_fence_load_kernel, (4,), *_fig1_args(), N)
    assert detector.last_status == "ok"
    store_line = _line_no(_store_fence_load_kernel, "tl.store(hist_ptr + offs, offs)")
    assert any(store_line in pair for pair in _report_lines(detector))


# ── Guarded producer/consumer: the fences the sync edge needs ─────────


@triton.jit
def _guarded_unfenced_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def _guarded_fenced_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    tl.debug_barrier()
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


def _guarded_args():
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )


def test_guarded_idiom_is_race_free_under_the_legacy_reading(fence_order_off):
    detector = _run(_guarded_unfenced_kernel, (2,), *_guarded_args())
    assert detector.last_status == "ok"
    assert detector.last_reports == []


def test_guarded_idiom_without_fences_races_under_fence_order(fence_order_on):
    detector = _run(_guarded_unfenced_kernel, (2,), *_guarded_args())
    assert detector.last_status == "ok"
    expected = frozenset(
        (
            _line_no(_guarded_unfenced_kernel, "tl.store(data_ptr, 1, mask=is_prod)"),
            _line_no(_guarded_unfenced_kernel, "x = tl.load(data_ptr, mask=cons_mask"),
        )
    )
    assert expected in _report_lines(detector)


def test_guarded_idiom_with_both_fences_is_race_free(fence_order_on):
    detector = _run(_guarded_fenced_kernel, (2,), *_guarded_args())
    assert detector.last_status == "ok"
    assert detector.last_reports == []


# ── Static (TTIR) track: the same litmus through the compiled reader ─────
# The TTIR reader records ``gpu.barrier`` (tl.debug_barrier's lowering) as a
# fence position and the compiled client hands it to the same two-copy
# solver, so the static track follows the flag exactly like the dynamic one.


def _static(kernel, grid, signature, constexprs, make_args):
    from evaluation.harness import _host_compile_ttir, _static_track
    from evaluation.spec import LaunchSpec

    spec = LaunchSpec(
        name="fence-order-litmus",
        kernel_fn=kernel,
        signature=signature,
        constexprs=constexprs,
        make_args=make_args,
        grid=grid,
    )
    ttir = _host_compile_ttir(spec)
    return _static_track(spec, ttir, seed=0)


_FIG1_SIG = {"hist_ptr": "*i32", "out_ptr": "*i32", "N": "constexpr"}


def test_static_ttir_reader_records_the_fence():
    from evaluation.harness import _host_compile_ttir
    from evaluation.spec import LaunchSpec
    from triton_viz.clients.common.ttir_reader import parse_ttir

    spec = LaunchSpec(
        name="x", kernel_fn=_store_fence_load_kernel, signature=_FIG1_SIG,
        constexprs={"N": N}, make_args=lambda seed: _fig1_args(), grid=(1,),
    )  # fmt: skip
    graph = parse_ttir(_host_compile_ttir(spec))
    assert graph.fences == [0.5]  # after the tile store, before the load
    assert len(graph.accesses) == 3


def test_static_unfenced_store_then_load_races_under_fence_order(fence_order_on):
    res = _static(
        _store_then_load_kernel, (1,), _FIG1_SIG, {"N": N}, lambda seed: _fig1_args()
    )
    assert res["status"] == "races", res
    lines = {(w["first"][1], w["second"][1]) for w in res["witnesses"]}
    store = _line_no(_store_then_load_kernel, "tl.store(hist_ptr + offs, offs)")
    load = _line_no(_store_then_load_kernel, "n = tl.load(hist_ptr + p)")
    assert any({a, b} == {store, load} for a, b in lines), lines


def test_static_fenced_store_then_load_is_race_free(fence_order_on):
    res = _static(
        _store_fence_load_kernel, (1,), _FIG1_SIG, {"N": N}, lambda seed: _fig1_args()
    )
    assert res["status"] == "ok", res


def test_static_legacy_reading_unchanged(fence_order_off):
    res = _static(
        _store_then_load_kernel, (1,), _FIG1_SIG, {"N": N}, lambda seed: _fig1_args()
    )
    assert res["status"] == "ok", res
