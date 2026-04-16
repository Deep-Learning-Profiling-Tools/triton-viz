import numpy as np
import pytest
import torch

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.clients.race_detector.race_detector import (
    SymbolicRaceDetector,
    NullRaceDetector,
)
from triton_viz.clients.race_detector.data import (
    AccessEventRecord,
    PendingAtomicEvent,
    active_mask_for,
    apply_rmw,
    effects_at_addr,
    infer_elem_size,
    normalize_sem_scope,
    resolve_tensor_from_pointer,
)
from triton_viz.core.config import config as cfg
from triton_viz.core.data import AtomicCas, AtomicRMW, Load, Store


# ======== Config Isolation ========
#
# Kernels are wrapped with @triton_viz.trace inside each test body so the
# flag-on escape hatch in trace.py (returns the raw kernel when
# ENABLE_RACE_DETECTOR is off) doesn't strip tracing at import time.


@pytest.fixture
def _isolate_race_detector_cfg():
    saved = cfg.enable_race_detector
    cfg.enable_race_detector = True
    yield
    cfg.enable_race_detector = saved


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


# ======== Raw kernel templates (re-decorated per test) ========


@triton.jit
def _basic_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, v + 1)


@triton.jit
def _load_store_loop_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for _i in tl.range(0, 4):
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v + 1)


@triton.jit
def _loop_dedup_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _i in tl.range(0, 4):
        acc += tl.load(x_ptr + offs)
    tl.store(x_ptr + offs, acc)


@triton.jit
def _loop_premises_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for _i in tl.range(0, 4):
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v)


@triton.jit
def _dispatch_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs))


# ======== Basic Capture ========


def test_basic_capture(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_basic_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    modes = [r.access_mode for r in detector.records]
    op_types = [r.op_type for r in detector.records]
    assert modes == ["read", "write"], f"expected [read, write], got {modes}"
    assert op_types == [Load, Store], f"expected [Load, Store], got {op_types}"

    for record in detector.records:
        assert record.premises, "premises must include addr_ok + pid_ok at minimum"
        assert record.tensor is not None, "base tensor should be resolvable"
        assert record.tensor_name == "x_ptr"
        assert isinstance(record, AccessEventRecord)


# ======== Loop: load+store at same address are NOT merged ========


def test_loop_load_and_store_same_addr_not_merged(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_load_store_loop_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    modes = sorted(r.access_mode for r in detector.records)
    # exactly one read event and one write event (deduped across 4 iterations,
    # but access_mode splits them so they stay distinct).
    assert modes == [
        "read",
        "write",
    ], f"load+store at same addr should stay as two events; got {modes}"


# ======== Loop: repeated access at same site is deduped ========


def test_loop_repeated_access_deduped(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_loop_dedup_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    reads = [r for r in detector.records if r.access_mode == "read"]
    assert (
        len(reads) == 1
    ), f"loop-body load should dedupe to a single event, got {len(reads)}"


# ======== Loop: event carries iterator constraint in its premises ========


def test_loop_event_premises_include_iterator(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_loop_premises_kernel)

    x = torch.zeros(16, dtype=torch.float32)
    traced[(1,)](x, BLOCK=16)

    assert detector.records
    for record in detector.records:
        premise_strs = " ".join(str(p) for p in record.premises)
        assert "loop_i_" in premise_strs, (
            f"in-loop event premises must carry a loop iterator constraint; "
            f"got premises: {premise_strs}"
        )


# ======== String dispatch + ClientManager lookup ========


def test_string_dispatch_and_manager_lookup(_isolate_race_detector_cfg):
    traced = triton_viz.trace("race_detector")(_dispatch_kernel)

    x = torch.zeros(8, dtype=torch.float32)
    traced[(1,)](x, BLOCK=8)

    rd = traced.client_manager.clients["race_detector"]
    assert isinstance(rd, SymbolicRaceDetector)
    assert len(rd.records) == 2


# ======== Flag-off escape hatch ========


def test_flag_off_returns_raw_kernel():
    """With ENABLE_RACE_DETECTOR=0 a string-dispatched trace decorator must
    leave the kernel uninstrumented so opting in has literally zero impact
    when the flag is off."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        traced = triton_viz.trace("race_detector")(_dispatch_kernel)

        # Should be the raw JIT kernel, not a TritonTrace wrapper.
        from triton_viz.core.trace import TritonTrace

        assert not isinstance(traced, TritonTrace)
        assert traced is _dispatch_kernel
    finally:
        cfg.enable_race_detector = saved


def test_flag_off_does_not_swallow_explicit_instance():
    """An explicitly constructed detector instance reflects a deliberate user
    choice and must keep tracing regardless of the global flag."""
    saved = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = False
        detector = SymbolicRaceDetector()
        traced = triton_viz.trace(client=detector)(_dispatch_kernel)

        from triton_viz.core.trace import TritonTrace

        assert isinstance(
            traced, TritonTrace
        ), "explicit instance must be traced even when the env flag is off"

        x = torch.zeros(8, dtype=torch.float32)
        traced[(1,)](x, BLOCK=8)
        assert len(detector.records) == 2
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
        traced = triton_viz.trace(client=RaceDetector())(_dispatch_kernel)

        from triton_viz.core.trace import TritonTrace

        assert not isinstance(traced, TritonTrace)
        assert traced is _dispatch_kernel
    finally:
        cfg.enable_race_detector = saved


# ======== Repeat launches stay consistent (no partial-grid cache) ========


def test_repeat_launches_are_consistent(_isolate_race_detector_cfg):
    """Launching the same traced kernel twice must produce deterministic,
    identical block-execution counts. The previous grid-enumeration cache
    could skip block 0 on a cache hit and then run block 1 instead —
    different blocks run, different side effects, events captured from a
    non-deterministic representative. This regression asserts every launch
    behaves the same."""
    detector = SymbolicRaceDetector()
    traced = triton_viz.trace(client=detector)(_dispatch_kernel)

    x1 = torch.zeros(8, dtype=torch.float32)
    traced[(2,)](x1, BLOCK=8)
    after_first = list(detector.records)

    detector.records.clear()
    x2 = torch.zeros(8, dtype=torch.float32)
    traced[(2,)](x2, BLOCK=8)
    after_second = list(detector.records)

    assert len(after_first) == len(after_second), (
        f"repeat launches must capture the same number of events; "
        f"got {len(after_first)} then {len(after_second)}"
    )
    assert [r.access_mode for r in after_first] == [r.access_mode for r in after_second]
    assert [r.source_location for r in after_first] == [
        r.source_location for r in after_second
    ]


# ======== PR2: Effect-aware schema + concrete atomic capture ========
#
# Organized by what could break:
#   * helper contracts in data.py
#   * CAS / RMW callback logic (hand-built pending, no kernel run)
#   * end-to-end kernel integration (currently xfail — see
#     ATOMIC_E2E_XFAIL_REASON below)
#   * state-lifecycle invariants (atomic_symbolic_escape, finalize, repeat)
#   * multi-SM concurrency smoke


# ======== Helper contracts (data.py) ========


def test_effects_at_addr_plain_fallback_from_access_mode():
    event = AccessEventRecord(
        event_id=0,
        op_type=AtomicCas,
        access_mode="read",
        lane_addrs=np.array([100], dtype=np.int64),
        elem_size=4,
        active_mask=np.array([True]),
    )
    assert effects_at_addr(event, 100) == (True, False)
    assert effects_at_addr(event, 96) == (False, False)
    assert effects_at_addr(event, 104) == (False, False)


def test_effects_at_addr_aggregates_across_all_matching_lanes():
    # Two lanes both covering address 100, one reads one writes. effects_at_addr
    # must report (True, True) via np.any — not just the first lane's effect.
    # Locks down upstream #344's single-lane bug.
    event = AccessEventRecord(
        event_id=0,
        op_type=AtomicCas,
        lane_addrs=np.array([100, 100], dtype=np.int64),
        elem_size=4,
        active_mask=np.array([True, True]),
        read_mask=np.array([False, True]),
        write_mask=np.array([True, False]),
    )
    assert effects_at_addr(event, 100) == (True, True)


def test_effects_at_addr_byte_range_for_multi_byte_elem():
    event = AccessEventRecord(
        event_id=0,
        op_type=AtomicCas,
        lane_addrs=np.array([100], dtype=np.int64),
        elem_size=4,
        access_mode="write",
    )
    for addr in (100, 101, 102, 103):
        assert effects_at_addr(event, addr) == (False, True)
    assert effects_at_addr(event, 104) == (False, False)
    assert effects_at_addr(event, 99) == (False, False)


def test_normalize_sem_scope_uses_triton_defaults():
    assert normalize_sem_scope(None, None) == ("acq_rel", "gpu")
    assert normalize_sem_scope("ACQUIRE", "CTA") == ("acquire", "cta")
    assert normalize_sem_scope("relaxed", None) == ("relaxed", "gpu")


def test_active_mask_for_none_and_scalar_and_tensor_and_length_mismatch():
    np.testing.assert_array_equal(active_mask_for(None, 4), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(active_mask_for(True, 3), np.ones(3, dtype=bool))
    np.testing.assert_array_equal(active_mask_for(False, 3), np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(
        active_mask_for(np.array([True, False, True]), 3),
        np.array([True, False, True]),
    )
    with pytest.raises(ValueError, match="does not match"):
        active_mask_for(np.array([True, False]), 4)


def test_infer_elem_size_prefers_pointer_dtype_then_value():
    # Fake a pointer that exposes .type.element_ty.primitive_bitwidth.
    class _ElemTy:
        primitive_bitwidth = 32  # int32

    class _PtrTy:
        element_ty = _ElemTy()

    class _Ptr:
        type = _PtrTy()

    # Case 1: ptr metadata authoritative even when val is a large python int.
    assert infer_elem_size(val=0xDEADBEEF, ptr=_Ptr()) == 4

    # Case 2: ptr has no metadata → fall back to value dtype.
    class _BarePtr:
        pass

    assert infer_elem_size(val=np.int16(0), ptr=_BarePtr()) == 2

    # Case 3: no ptr metadata AND bare python scalar → refuse.
    with pytest.raises(ValueError, match="refusing to guess"):
        infer_elem_size(val=1, ptr=_BarePtr())


def _fake_tensor_for_resolve(name: str) -> torch.Tensor:
    # Distinct Python objects are enough — resolve_tensor_from_pointer doesn't
    # touch tensor contents, only the interval registered with each tensor.
    t = torch.zeros(1, dtype=torch.float32)
    t.__dict__["_pr2_test_tag"] = name
    return t


def test_resolve_tensor_from_pointer_exact_interval():
    t = _fake_tensor_for_resolve("X")
    tensor_addrs = [(1000, 1003, t)]
    ptr = np.array([1000, 1000], dtype=np.int64)
    active = np.array([True, False])
    resolved = resolve_tensor_from_pointer(
        ptr, active, elem_size=4, tensor_addrs=tensor_addrs
    )
    assert resolved is t


def test_resolve_tensor_from_pointer_returns_none_on_multi_match_and_no_match():
    t1 = _fake_tensor_for_resolve("A")
    t2 = _fake_tensor_for_resolve("B")
    overlapping = [(1000, 2000, t1), (999, 2001, t2)]
    ptr = np.array([1000], dtype=np.int64)
    active = np.array([True])
    assert resolve_tensor_from_pointer(ptr, active, 4, overlapping) is None

    empty: list = []
    assert resolve_tensor_from_pointer(ptr, active, 4, empty) is None


def test_apply_rmw_add_xchg_and_dtype_preservation():
    old32 = np.array([10, 20], dtype=np.int32)
    val32 = np.array([5, 7], dtype=np.int32)
    add_res = apply_rmw("add", old32, val32)
    np.testing.assert_array_equal(add_res, np.array([15, 27], dtype=np.int32))
    assert add_res.dtype == np.int32  # no silent promotion to int64

    xchg_res = apply_rmw("xchg", old32, val32)
    np.testing.assert_array_equal(xchg_res, val32)
    assert xchg_res.dtype == np.int32


def test_apply_rmw_unsupported_raises():
    with pytest.raises(NotImplementedError, match="nope"):
        apply_rmw("nope", np.array([0]), np.array([0]))


# ======== Callback logic: hand-built PendingAtomicEvent ========


def _prime_detector_for_callback(
    grid_idx: tuple[int, int, int] = (0, 0, 0),
) -> SymbolicRaceDetector:
    """Construct a SymbolicRaceDetector with just enough state for a
    before/after atomic callback to run. Skips the real grid_callback
    path because these tests hand-build PendingAtomicEvent instances."""
    detector = SymbolicRaceDetector(abort_on_error=False)
    detector.grid_idx_callback(grid_idx)
    return detector


def test_atomic_cas_success_and_failure_compute_masks_correctly():
    detector = _prime_detector_for_callback()
    pending = PendingAtomicEvent(
        event_id=42,
        op_type=AtomicCas,
        atomic_op="cas",
        grid_idx=(0, 0, 0),
        source_location=None,
        tensor=None,
        tensor_name=None,
        lane_addrs=np.array([100, 104], dtype=np.int64),
        active_mask=np.array([True, True]),
        elem_size=4,
        atomic_sem="acq_rel",
        atomic_scope="gpu",
        atomic_cmp=np.array([0, 99], dtype=np.int32),  # lane1 cmp won't match
        atomic_val=np.array([7, 8], dtype=np.int32),
    )
    detector._pending_atomic_by_grid[(0, 0, 0)].append(pending)

    # lane0 old=0 matches cmp=0 → success; lane1 old=5 != 99 → fail.
    ret = np.array([0, 5], dtype=np.int32)
    detector._after_atomic_cas(ret, ptr=None, cmp=None, val=None)

    assert len(detector.concrete_events) == 1
    event = detector.concrete_events[0]
    np.testing.assert_array_equal(event.read_mask, [True, True])
    np.testing.assert_array_equal(event.write_mask, [True, False])
    np.testing.assert_array_equal(event.written_value, [7, 5])
    np.testing.assert_array_equal(event.atomic_old, [0, 5])
    assert event.atomic_op == "cas"


def test_atomic_cas_after_pops_matching_grid_queue():
    detector = _prime_detector_for_callback(grid_idx=(0, 0, 0))

    def _mk_pending(gidx: tuple[int, int, int]) -> PendingAtomicEvent:
        return PendingAtomicEvent(
            event_id=gidx[0],
            op_type=AtomicCas,
            atomic_op="cas",
            grid_idx=gidx,
            source_location=None,
            tensor=None,
            tensor_name=None,
            lane_addrs=np.array([100], dtype=np.int64),
            active_mask=np.array([True]),
            elem_size=4,
            atomic_sem="acq_rel",
            atomic_scope="gpu",
            atomic_cmp=np.array([0], dtype=np.int32),
            atomic_val=np.array([1], dtype=np.int32),
        )

    detector._pending_atomic_by_grid[(0, 0, 0)].append(_mk_pending((0, 0, 0)))
    detector._pending_atomic_by_grid[(1, 0, 0)].append(_mk_pending((1, 0, 0)))

    # detector.grid_idx is currently (0, 0, 0) from priming.
    ret = np.array([0], dtype=np.int32)
    detector._after_atomic_cas(ret, ptr=None, cmp=None, val=None)

    # (0,0,0) popped; (1,0,0) untouched.
    assert len(detector._pending_atomic_by_grid[(0, 0, 0)]) == 0
    assert len(detector._pending_atomic_by_grid[(1, 0, 0)]) == 1


def test_atomic_rmw_add_written_value_uses_old_plus_val():
    detector = _prime_detector_for_callback()
    pending = PendingAtomicEvent(
        event_id=7,
        op_type=AtomicRMW,
        atomic_op="add",
        grid_idx=(0, 0, 0),
        source_location=None,
        tensor=None,
        tensor_name=None,
        lane_addrs=np.array([100, 104], dtype=np.int64),
        active_mask=np.array([True, True]),
        elem_size=4,
        atomic_sem="acq_rel",
        atomic_scope="gpu",
        atomic_cmp=None,
        atomic_val=np.array([5, 7], dtype=np.int32),
    )
    detector._pending_atomic_by_grid[(0, 0, 0)].append(pending)

    ret = np.array([10, 20], dtype=np.int32)
    detector._after_atomic_rmw(ret, rmwOp=None, ptr=None, val=None)
    event = detector.concrete_events[0]
    np.testing.assert_array_equal(event.written_value, [15, 27])


def test_atomic_rmw_xchg_written_value_equals_val():
    detector = _prime_detector_for_callback()
    pending = PendingAtomicEvent(
        event_id=7,
        op_type=AtomicRMW,
        atomic_op="xchg",
        grid_idx=(0, 0, 0),
        source_location=None,
        tensor=None,
        tensor_name=None,
        lane_addrs=np.array([100], dtype=np.int64),
        active_mask=np.array([True]),
        elem_size=4,
        atomic_sem="acq_rel",
        atomic_scope="gpu",
        atomic_cmp=None,
        atomic_val=np.array([99], dtype=np.int32),
    )
    detector._pending_atomic_by_grid[(0, 0, 0)].append(pending)

    ret = np.array([7], dtype=np.int32)
    detector._after_atomic_rmw(ret, rmwOp=None, ptr=None, val=None)
    event = detector.concrete_events[0]
    np.testing.assert_array_equal(event.written_value, [99])


# ======== atomic_symbolic_escape lifecycle ========


def test_atomic_symbolic_escape_flag_set_after_atomic():
    # Unit-level: hand-call _before_atomic_cas with mock args so the test
    # has no coupling with kernel-launch lifecycle.
    detector = _prime_detector_for_callback()
    assert detector.atomic_symbolic_escape is False

    class _ElemTy:
        primitive_bitwidth = 32

    class _PtrTy:
        element_ty = _ElemTy()

    class _Ptr:
        type = _PtrTy()

        def __init__(self, addrs: np.ndarray):
            self.addrs = addrs

    # flatten_np walks .handle/.data; expose .data → address array.
    ptr = _Ptr(np.array([100], dtype=np.int64))
    ptr.data = ptr.addrs  # type: ignore[attr-defined]

    detector._before_atomic_cas(
        ptr,
        cmp=np.array([0], dtype=np.int32),
        val=np.array([1], dtype=np.int32),
        mask=None,
        sem=None,
        scope=None,
    )
    assert detector.atomic_symbolic_escape is True


def test_atomic_symbolic_escape_survives_finalize_and_resets_on_next_grid_callback():
    detector = _prime_detector_for_callback()
    # Forcibly set the flag as if an atomic had been captured.
    detector.atomic_symbolic_escape = True

    # finalize() must NOT reset the flag — Step 4/5 consumer will read it
    # immediately after launch end.
    ret = detector.finalize()
    assert ret == []
    assert detector.atomic_symbolic_escape is True

    # Next launch's grid_callback resets it.
    detector.grid_callback((1, 1, 1))
    assert detector.atomic_symbolic_escape is False


# ======== finalize: dangling-pending assertion ========


def test_finalize_raises_on_dangling_pending():
    detector = _prime_detector_for_callback()
    detector._pending_atomic_by_grid[(9, 9, 9)].append(
        PendingAtomicEvent(
            event_id=0,
            op_type=AtomicCas,
            atomic_op="cas",
            grid_idx=(9, 9, 9),
            source_location=None,
            tensor=None,
            tensor_name=None,
            lane_addrs=np.array([100], dtype=np.int64),
            active_mask=np.array([True]),
            elem_size=4,
            atomic_sem="acq_rel",
            atomic_scope="gpu",
            atomic_cmp=np.array([0], dtype=np.int32),
            atomic_val=np.array([1], dtype=np.int32),
        )
    )
    with pytest.raises(RuntimeError, match="Dangling pending atomic events"):
        detector.finalize()


# ======== End-to-end tests (xfail) ========
#
# These exercise the full path: traced kernel → SymbolicClient → atomic
# before/after callbacks → concrete_events. They're marked xfail because
# the race detector currently inherits SymbolicClient's symbolic
# overriders for Load/AddPtr/Splat/etc., which rewrite every expression
# along the way into a SymbolicExpr. By the time ``tl.atomic_cas(ptr,
# cmp, val)`` fires, its inputs are SymbolicExpr values that can't be
# fed to the real interpreter-builder ``create_atomic_cas`` (no
# op_overrider is registered for atomics on race_detector, so the
# original op is expected to run — but only on concrete inputs).
#
# The fix is an adapter-level concretize-on-entry for atomic ops
# (probably a race-detector-specific op_overrider that pulls out numpy
# values, simulates the CAS/RMW, records the effect event, and returns
# a SymbolicExpr const wrapping the resulting old value). That's a
# separate scope from this PR's data-model + callback skeleton. The
# callback *logic* is validated by the hand-built-pending tests above.


ATOMIC_E2E_XFAIL_REASON = (
    "Symbolic overriders upstream of tl.atomic_* turn the atomic's args "
    "into SymbolicExpr; race_detector's op_overrider=None requires real "
    "hardware execution on concrete inputs. Needs adapter-level "
    "concretize-on-entry scaffolding — follow-up PR."
)


@triton.jit
def _atomic_cas_kernel(x_ptr, cmp_ptr, val_ptr):
    old = tl.atomic_cas(x_ptr, tl.load(cmp_ptr), tl.load(val_ptr))
    tl.store(val_ptr + 1, old)  # keep ``old`` live


@triton.jit
def _atomic_add_kernel(x_ptr, val_ptr):
    old = tl.atomic_add(x_ptr, tl.load(val_ptr))
    tl.store(val_ptr + 1, old)


@pytest.mark.xfail(reason=ATOMIC_E2E_XFAIL_REASON, strict=True)
def test_traced_kernel_emits_concrete_cas_event(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_cas_kernel)

    x = torch.zeros(1, dtype=torch.int32)
    cmp = torch.zeros(1, dtype=torch.int32)
    val = torch.zeros(2, dtype=torch.int32)
    traced[(1,)](x, cmp, val)

    cas_events = [e for e in detector.concrete_events if e.atomic_op == "cas"]
    assert len(cas_events) == 1, f"expected 1 CAS concrete event, got {len(cas_events)}"
    e = cas_events[0]
    assert e.atomic_old is not None and e.atomic_old.size == 1
    assert e.atomic_old[0] == 0
    assert e.read_mask is not None and bool(e.read_mask.all())
    assert e.write_mask is not None and bool(e.write_mask.all())


@pytest.mark.xfail(reason=ATOMIC_E2E_XFAIL_REASON, strict=True)
def test_traced_kernel_emits_concrete_atomic_add_event(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_add_kernel)

    x = torch.tensor([10], dtype=torch.int32)
    val = torch.tensor([5, 0], dtype=torch.int32)
    traced[(1,)](x, val)

    add_events = [e for e in detector.concrete_events if e.atomic_op == "add"]
    assert len(add_events) == 1
    e = add_events[0]
    # written_value MUST be old+val, not val alone.
    assert e.atomic_old is not None
    assert e.atomic_val is not None
    assert e.written_value is not None
    np.testing.assert_array_equal(e.atomic_old, np.array([10], dtype=np.int32))
    np.testing.assert_array_equal(e.written_value, np.array([15], dtype=np.int32))


@pytest.mark.xfail(reason=ATOMIC_E2E_XFAIL_REASON, strict=True)
def test_finalize_still_returns_empty(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_add_kernel)
    x = torch.tensor([0], dtype=torch.int32)
    val = torch.tensor([1, 0], dtype=torch.int32)
    traced[(1,)](x, val)

    assert detector.finalize() == []


@pytest.mark.xfail(reason=ATOMIC_E2E_XFAIL_REASON, strict=True)
def test_repeat_launches_are_deterministic_for_atomics(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_add_kernel)

    def _launch(old_val: int, add_val: int) -> None:
        x = torch.tensor([old_val], dtype=torch.int32)
        val = torch.tensor([add_val, 0], dtype=torch.int32)
        traced[(1,)](x, val)

    _launch(10, 5)
    assert len(detector.concrete_events) == 1
    detector.concrete_events.clear()

    _launch(10, 5)
    assert len(detector.concrete_events) == 1
    assert detector.concrete_events[0].atomic_op == "add"

    # Pending queue must have drained between launches.
    assert all(len(q) == 0 for q in detector._pending_atomic_by_grid.values())


@pytest.mark.xfail(reason=ATOMIC_E2E_XFAIL_REASON, strict=True)
def test_concurrent_blocks_drain_pending_queue_cleanly():
    """cfg.num_sms = 2 with a 2-block kernel, each block issues one atomic
    add. Let the ThreadPoolExecutor interleave as it wants. The dangling-
    queue assertion lives in finalize() (which runs once, post-launch),
    NOT in per-block post_run_callback — so natural scheduling must not
    trigger a false RuntimeError, and the queue must be empty after the
    launch ends."""
    saved_num_sms = cfg.num_sms
    saved_flag = cfg.enable_race_detector
    try:
        cfg.enable_race_detector = True
        cfg.num_sms = 2
        detector = SymbolicRaceDetector(abort_on_error=False)

        @triton.jit
        def _per_block_atomic_kernel(out_ptr):
            pid = tl.program_id(axis=0)
            tl.atomic_add(out_ptr + pid, 1)

        traced = triton_viz.trace(client=detector)(_per_block_atomic_kernel)
        out = torch.zeros(2, dtype=torch.int32)
        traced[(2,)](out)

        assert detector.finalize() == []
        assert len(detector.concrete_events) == 2
        assert all(len(q) == 0 for q in detector._pending_atomic_by_grid.values())
    finally:
        cfg.num_sms = saved_num_sms
        cfg.enable_race_detector = saved_flag
