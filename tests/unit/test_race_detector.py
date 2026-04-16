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
    RaceReport,
    RaceType,
    active_mask_for,
    apply_rmw,
    effects_at_addr,
    infer_elem_size,
    normalize_sem_scope,
    resolve_tensor_from_pointer,
)
from triton_viz.core.config import config as cfg
from triton_viz.core.data import AtomicCas, Load, Store


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


# ======== atomic_symbolic_escape lifecycle ========


def test_atomic_symbolic_escape_survives_finalize_and_resets_on_next_grid_callback():
    detector = SymbolicRaceDetector(abort_on_error=False)
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


# ======== End-to-end tests ========
#
# Exercise the full path: traced kernel → race_detector atomic overrider →
# concrete_events. The overrider concretizes ``SymbolicExpr`` inputs, runs
# the real ``interpreter_builder.create_atomic_*`` on the concrete values,
# records the effect, and returns a ``const`` SymbolicExpr wrapping the
# hardware ``old`` so downstream symbolic consumers keep working.


@triton.jit
def _atomic_cas_kernel(x_ptr, cmp_ptr, val_ptr):
    old = tl.atomic_cas(x_ptr, tl.load(cmp_ptr), tl.load(val_ptr))
    tl.store(val_ptr + 1, old)  # keep ``old`` live


@triton.jit
def _atomic_add_kernel(x_ptr, val_ptr):
    old = tl.atomic_add(x_ptr, tl.load(val_ptr))
    tl.store(val_ptr + 1, old)


@triton.jit
def _atomic_cas_downstream_kernel(x_ptr, out_ptr):
    # Consume the atomic return value via a downstream symbolic op so the
    # Step-0c return-type decision (``const`` SymbolicExpr carrying the
    # concrete ``old``) stays honest end-to-end.
    old = tl.atomic_cas(x_ptr, 0, 1)
    tl.store(out_ptr, old + 1)


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


def test_finalize_still_returns_empty(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_add_kernel)
    x = torch.tensor([0], dtype=torch.int32)
    val = torch.tensor([1, 0], dtype=torch.int32)
    traced[(1,)](x, val)

    # Step 0 (this commit) does not yet emit RaceReports from finalize().
    assert detector.finalize() == []


def test_repeat_launches_are_deterministic_for_atomics(_isolate_race_detector_cfg):
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_add_kernel)

    def _launch(old_val: int, add_val: int) -> None:
        x = torch.tensor([old_val], dtype=torch.int32)
        val = torch.tensor([add_val, 0], dtype=torch.int32)
        traced[(1,)](x, val)

    _launch(10, 5)
    atomic_events = [e for e in detector.concrete_events if e.atomic_op is not None]
    assert len(atomic_events) == 1
    detector.concrete_events.clear()

    _launch(10, 5)
    atomic_events = [e for e in detector.concrete_events if e.atomic_op is not None]
    assert len(atomic_events) == 1
    assert atomic_events[0].atomic_op == "add"


def test_concurrent_blocks_capture_atomics_cleanly():
    """cfg.num_sms = 2 with a 2-block kernel, each block issues one atomic
    add. Let the ThreadPoolExecutor interleave as it wants — overrider-based
    capture is single-shot per op (no cross-callback pending state) so the
    expected count is exactly one event per block regardless of ordering."""
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
    finally:
        cfg.num_sms = saved_num_sms
        cfg.enable_race_detector = saved_flag


def test_concrete_events_carry_program_seq_and_launch_id(_isolate_race_detector_cfg):
    """Both plain load/store events and atomic events must be tagged with a
    per-grid ``program_seq`` and the current ``launch_id`` — po / epoch
    partitioning / same-value ambiguity all depend on this sequencing."""
    detector = SymbolicRaceDetector(abort_on_error=False)

    @triton.jit
    def _mixed_kernel(x_ptr, flag_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.store(x_ptr + offs, v + 1)
        tl.atomic_add(flag_ptr, 1)

    traced = triton_viz.trace(client=detector)(_mixed_kernel)
    x = torch.zeros(4, dtype=torch.int32)
    flag = torch.zeros(1, dtype=torch.int32)
    traced[(1,)](x, flag, BLOCK=4)

    # Three concrete events expected: load, store, atomic_add.
    assert len(detector.concrete_events) == 3
    seqs = [e.program_seq for e in detector.concrete_events]
    # Within a single grid_idx the program_seq values are 0, 1, 2 in issue
    # order — NOT event_id order under multi-SM (unused here but locks
    # the invariant that per-grid sequencing is contiguous).
    assert seqs == [0, 1, 2], f"expected [0,1,2] program_seq, got {seqs}"

    launch_ids = {e.launch_id for e in detector.concrete_events}
    assert launch_ids == {detector._launch_id}


def test_release_acquire_reads_from_suppresses_crossblock_plain_race(
    _isolate_race_detector_cfg,
):
    """Producer-consumer handshake: block 0 writes a plain flag then
    releases a barrier atomic; block 1 acquires on the same barrier and
    then reads the plain flag. With reads-from (acquire reads exactly
    what release wrote), the HBSolver orders the plain accesses via
    ``po | sw`` so the cross-block race on the plain flag goes away."""
    detector = SymbolicRaceDetector(abort_on_error=False)

    @triton.jit
    def _producer_consumer(flag_ptr, sync_ptr):
        pid = tl.program_id(axis=0)
        # Producer (pid==0): plain store then release-xchg.
        tl.store(flag_ptr, 42, mask=(pid == 0))
        tl.atomic_xchg(sync_ptr, 1, mask=(pid == 0), sem="release", scope="gpu")
        # Consumer (pid==1): acquire-xchg (reads what producer released),
        # then plain load.
        _obtained = tl.atomic_xchg(
            sync_ptr, 2, mask=(pid == 1), sem="acquire", scope="gpu"
        )
        tl.load(flag_ptr, mask=(pid == 1))

    traced = triton_viz.trace(client=detector)(_producer_consumer)
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
        f"release/acquire+reads-from should suppress the cross-block race "
        f"on flag_ptr; got {flag_reports}"
    )


def test_finalize_emits_report_for_crossblock_plain_store_load(
    _isolate_race_detector_cfg,
):
    """Smoke test for the Step 4 candidate pipeline.

    Block 0 stores to ``x_ptr``; block 1 loads from ``x_ptr``. Without
    HB suppression (Step 5 lands in the next commit), this emits one
    RAW RaceReport — canonical ordering places block 0's store first,
    so the flow is write-then-read.
    """
    detector = SymbolicRaceDetector(abort_on_error=False)

    @triton.jit
    def _crossblock_rw(x_ptr, out_ptr):
        pid = tl.program_id(axis=0)
        tl.store(x_ptr, 1, mask=(pid == 0))
        v = tl.load(x_ptr, mask=(pid == 1))
        tl.store(out_ptr + pid, v, mask=(pid == 1))

    traced = triton_viz.trace(client=detector)(_crossblock_rw)
    x = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    traced[(2,)](x, out)

    reports = detector.finalize()
    # At least one RAW report on x_ptr between block 0 and block 1.
    raw_reports = [
        r
        for r in reports
        if isinstance(r, RaceReport)
        and r.race_type is RaceType.RAW
        and r.grid_a == (0, 0, 0)
        and r.grid_b == (1, 0, 0)
    ]
    assert raw_reports, f"expected a RAW cross-block report, got {reports}"


def test_atomic_cas_return_is_consumable_downstream(_isolate_race_detector_cfg):
    """Locks the Step-0c return-type decision: the overrider returns a
    ``const`` SymbolicExpr wrapping the concrete ``old`` TensorHandle, so
    (a) the launch doesn't raise when downstream ops consume the result
    and (b) a CAS concrete event is still captured."""
    detector = SymbolicRaceDetector(abort_on_error=False)
    traced = triton_viz.trace(client=detector)(_atomic_cas_downstream_kernel)

    x = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(1, dtype=torch.int32)

    traced[(1,)](x, out)  # must not raise

    cas_events = [e for e in detector.concrete_events if e.atomic_op == "cas"]
    assert len(cas_events) == 1
    event = cas_events[0]
    assert event.atomic_old is not None
    assert int(event.atomic_old[0]) == 0
    # detector.atomic_symbolic_escape flips on first atomic capture.
    assert detector.atomic_symbolic_escape is True
