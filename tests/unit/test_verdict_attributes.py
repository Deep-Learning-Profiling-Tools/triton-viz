"""Verdict-attribute emission (paper sec:verdicts) and the RQ5 ablation
switches: the taxonomy rides the verdict directly, and each ablation
degrades exactly the machinery it names (defaults = production semantics).
"""


import pytest
import torch

from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)

from .test_await_abstraction import _drive_client, _prod_cons_ttir
from .test_t1_rmw_static import _lbd_ttir, _module


def _lbd_tensors():
    return (
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.float32),
    )


_LBD_NAMES = ["partial_ptr", "counter_ptr", "out_ptr"]


def _run(ttir, tensors, names, grid=(4,), **det_kwargs):
    det = CompiledRaceDetector(confirm_races=False, **det_kwargs)
    _drive_client(det, ttir, tensors, names, grid=grid)
    return det


# ─────────────────── attribute taxonomy ───────────────────


def test_race_free_attrs_t1():
    det = _run(_lbd_ttir("acq_rel"), _lbd_tensors(), _LBD_NAMES)
    v = det.last_global_verdict
    assert v is not None
    assert v["verdict"] == "race-free"
    assert v["proved_scope"] == "this-params-any-grid"
    assert v["conservative"] is False
    assert v["conditional"] == ()
    assert v["unsupported_kind"] is None


def test_race_free_attrs_t0_scope():
    text = _module(
        "%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>",
        "%c64 = arith.constant 64 : i32",
        "%pid = tt.get_program_id x : i32",
        "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "%b = arith.muli %pid, %c64 : i32",
        "%sb = tt.splat %b : i32 -> tensor<64xi32>",
        "%offs = arith.addi %sb, %r : tensor<64xi32>",
        "%op = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%oa = tt.addptr %op, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "%xp = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>",
        "%xa = tt.addptr %xp, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>",
        "%v = tt.load %xa : tensor<64x!tt.ptr<f32>>",
        "tt.store %oa, %v : tensor<64x!tt.ptr<f32>>",
    )
    det = _run(
        text,
        (torch.zeros(256), torch.zeros(256)),
        ["x_ptr", "out_ptr"],
    )
    v = det.last_global_verdict
    assert v["verdict"] == "race-free"
    assert v["proved_scope"] == "any-params-any-grid"


def test_race_attrs_exact_and_conditional():
    det = _run(
        _prod_cons_ttir(spin_sem="relaxed"),
        (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(64, dtype=torch.int32),
        ),
        ["flag_ptr", "data_ptr", "out_ptr"],
        grid=(2,),
    )
    v = det.last_global_verdict
    assert v["verdict"] == "race"
    # await pre-guard makes replay unavailable → evidence stays "exact"
    assert v["race_evidence"] == "exact"
    assert v["conditional"] == ("termination",)


def test_conditional_rides_race_free_too():
    det = _run(
        _prod_cons_ttir(),
        (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(64, dtype=torch.int32),
        ),
        ["flag_ptr", "data_ptr", "out_ptr"],
        grid=(2,),
    )
    v = det.last_global_verdict
    assert v["verdict"] == "race-free"
    assert v["conditional"] == ("termination",)


def test_abstain_attrs_carry_kind():
    text = _module(
        "%idx_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%c1 = arith.constant 1 : i32",
        "%i = tt.load %idx_ptr : !tt.ptr<i32>",
        "%oa = tt.addptr %out_ptr, %i : !tt.ptr<i32>, i32",
        "tt.store %oa, %c1 : !tt.ptr<i32>",
    )
    det = _run(
        text,
        (torch.zeros(4, dtype=torch.int32), torch.zeros(64, dtype=torch.int32)),
        ["idx_ptr", "out_ptr"],
    )
    v = det.last_global_verdict
    assert v["verdict"] == "abstain"
    assert v["unsupported_kind"] == "indirect-address"
    assert v["conservative"] is True


def test_potential_race_attrs():
    """A widened-only SAT (data-dependent mask) is a potential race —
    conservative, widened evidence."""
    text = _module(
        "%flag_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%c1 = arith.constant 1 : i32",
        "%c0 = arith.constant 0 : i32",
        "%pid = tt.get_program_id x : i32",
        "%fp = tt.addptr %flag_ptr, %pid : !tt.ptr<i32>, i32",
        "%f = tt.load %fp : !tt.ptr<i32>",
        "%m = arith.cmpi sgt, %f, %c0 : i32",
        "tt.store %out_ptr, %c1, %m : !tt.ptr<i32>",
    )
    det = _run(
        text,
        (torch.ones(4, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)),
        ["flag_ptr", "out_ptr"],
    )
    v = det.last_global_verdict
    assert v["verdict"] == "potential-race"
    assert v["race_evidence"] == "widened"
    assert v["conservative"] is True


# ─────────────────── ablation switches ───────────────────


def test_unknown_ablation_rejected():
    with pytest.raises(ValueError, match="unknown ablations"):
        TwoCopySymbolicHBSolver([], grid=(2, 1, 1), ablations=("bogus",))


def test_no_hb_flips_ordering_proof():
    """The last-block-done proof rides sw ordering: no-hb must flip it,
    while the baseline still proves."""
    tensors = _lbd_tensors()
    tensors[1][0] = 0
    base = _run(_lbd_ttir("acq_rel"), _lbd_tensors(), _LBD_NAMES)
    assert base.last_global_status == "ok"
    abl = _run(_lbd_ttir("acq_rel"), _lbd_tensors(), _LBD_NAMES, ablations=("hb",))
    assert abl.last_global_status == "races"


def test_no_hb_keeps_footprint_proof():
    """Disjoint per-pid tiles owe nothing to ordering: no-hb must NOT
    flip a pure footprint proof (the ablation is surgical)."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>",
        "%c64 = arith.constant 64 : i32",
        "%cst = arith.constant dense<1> : tensor<64xi32>",
        "%pid = tt.get_program_id x : i32",
        "%r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>",
        "%b = arith.muli %pid, %c64 : i32",
        "%sb = tt.splat %b : i32 -> tensor<64xi32>",
        "%offs = arith.addi %sb, %r : tensor<64xi32>",
        "%op = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>",
        "%oa = tt.addptr %op, %offs : tensor<64x!tt.ptr<i32>>, tensor<64xi32>",
        "tt.store %oa, %cst : tensor<64x!tt.ptr<i32>>",
    )
    args = (torch.zeros(256, dtype=torch.int32),)
    assert _run(text, args, ["out_ptr"]).last_global_status == "ok"
    assert _run(text, args, ["out_ptr"], ablations=("hb",)).last_global_status == "ok"


def test_no_coherence_flips_counting_proof():
    """The single-winner work-queue proof is pure coherence/counting: the
    no-coherence ablation must flip it (here to the observation-address
    gate's abstention), while no-hb keeps it."""
    text = _module(
        "%head_ptr: !tt.ptr<i32>, %buf_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "%old = tt.atomic_rmw add, relaxed, gpu, %head_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%b = tt.addptr %buf_ptr, %old : !tt.ptr<i32>, i32",
        "tt.store %b, %c1 : !tt.ptr<i32>",
    )

    def args():
        return (torch.zeros(1, dtype=torch.int32), torch.zeros(64, dtype=torch.int32))

    names = ["head_ptr", "buf_ptr"]
    assert _run(text, args(), names).last_global_status == "ok"
    assert _run(text, args(), names, ablations=("hb",)).last_global_status == "ok"
    abl = _run(text, args(), names, ablations=("coherence",))
    assert abl.last_global_status != "ok", "counting proof must not survive"


# ─────────── the no-pid atomic grid-pinning regression ───────────


def test_no_pid_atomic_kernel_not_falsely_proved():
    """A kernel with NO pid read but an atomic: blocks are distinguished
    by their observations, so pinning unread grid axes to 1 (the
    identical-behavior rule for non-atomic kernels) would erase the real
    cross-block WAW of the narrow-slot work queue. Atomic-bearing graphs
    size unread axes from the REAL launch instead."""
    text = _module(
        "%head_ptr: !tt.ptr<i32>, %buf_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "%c2 = arith.constant 2 : i32",
        "%old = tt.atomic_rmw add, relaxed, gpu, %head_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%h = arith.divsi %old, %c2 : i32",
        "%b = tt.addptr %buf_ptr, %h : !tt.ptr<i32>, i32",
        "tt.store %b, %c1 : !tt.ptr<i32>",
    )
    det = _run(
        text,
        (torch.zeros(1, dtype=torch.int32), torch.zeros(64, dtype=torch.int32)),
        ["head_ptr", "buf_ptr"],
        grid=(4,),
    )
    assert (
        det.last_global_status != "ok"
    ), "adjacent ranks share buf[old//2] — a grid-(4,) launch races"


def test_no_pid_atomic_kernel_good_version_still_proves():
    """The disjoint-slot twin at the same launch must still prove (the fix
    is surgical: launch-sized unread axes, not a blanket abstention)."""
    text = _module(
        "%head_ptr: !tt.ptr<i32>, %buf_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "%old = tt.atomic_rmw add, relaxed, gpu, %head_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%b = tt.addptr %buf_ptr, %old : !tt.ptr<i32>, i32",
        "tt.store %b, %c1 : !tt.ptr<i32>",
    )
    det = _run(
        text,
        (torch.zeros(1, dtype=torch.int32), torch.zeros(64, dtype=torch.int32)),
        ["head_ptr", "buf_ptr"],
        grid=(4,),
    )
    assert det.last_global_status == "ok"


def test_non_atomic_no_pid_kernel_reports_at_violating_launch():
    """The launch-contract premise is ENFORCED, not assumed: a no-pid
    store launched with grid (4,) parallelizes an axis the kernel never
    reads — the aiter#3091 caller-bug shape — and the identical writes
    are a real cross-instance WAW. Pinning the unread axis to 1 used to
    fabricate a race-freedom proof here while the interpreter reported
    the race (found on the ORIGINAL _sum_bitmatrix_rows_fused)."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>",
        "%c1 = arith.constant 1 : i32",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
    )
    det = _run(text, (torch.zeros(1, dtype=torch.int32),), ["out_ptr"], grid=(4,))
    assert (
        det.last_global_status != "ok"
    ), "a grid-(4,) launch of a no-pid store is a cross-instance WAW"


def test_non_atomic_no_pid_kernel_proves_at_respecting_launch():
    """The same kernel at grid (1,) respects the contract and proves; the
    unread-axis floor equals the launch extent, so nothing is invented."""
    text = _module(
        "%out_ptr: !tt.ptr<i32>",
        "%c1 = arith.constant 1 : i32",
        "tt.store %out_ptr, %c1 : !tt.ptr<i32>",
    )
    det = _run(text, (torch.zeros(1, dtype=torch.int32),), ["out_ptr"], grid=(1,))
    assert det.last_global_status == "ok"
