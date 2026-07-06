import os
import threading

import numpy as np
import pytest
import triton
import triton.language as tl
from z3 import If, Int, IntVal

import triton_viz
from triton_viz.clients import RaceDetector
from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.clients.race_detector.hb_common import UnsupportedSymbolicRaceQuery
from triton_viz.clients.race_detector.hb_solver import HBSolver
from triton_viz.clients.race_detector.race_detector import (
    SymbolicRaceDetector,
    NullRaceDetector,
)
from triton_viz.clients.symbolic_engine import (
    ConstSymbolicExpr,
    LoadSymbolicExpr,
    SymbolicExpr,
    _triton_frame_dirs,
)
from triton_viz.core.config import config as cfg
from triton_viz.core.data import AtomicCas, Load, Store, TensorPointerStore
from triton_viz.core.symbolic_metadata import (
    FLOAT32,
    INT1,
    INT32,
    SymbolicTensorValue,
    block_type,
    pointer_type,
)


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


FLAG = IntVal(100)
DATA = IntVal(200)


def _store_record(
    *,
    name: str,
    grid_idx: tuple[int, ...],
    program_seq: int,
    addr,
    value: int,
) -> AccessEventRecord:
    return AccessEventRecord(
        op_type=Store,
        access_mode="write",
        addr_expr=addr,
        grid_idx=grid_idx,
        program_seq=program_seq,
        active=True,
        reads=False,
        writes=True,
        is_atomic=False,
        sem="plain",
        written_value=IntVal(value),
        debug_name=name,
    )


def _load_record(
    *,
    name: str,
    grid_idx: tuple[int, ...],
    program_seq: int,
    addr,
    active=True,
) -> AccessEventRecord:
    return AccessEventRecord(
        op_type=Load,
        access_mode="read",
        addr_expr=addr,
        grid_idx=grid_idx,
        program_seq=program_seq,
        active=active,
        reads=True,
        writes=False,
        is_atomic=False,
        sem="plain",
        debug_name=name,
    )


def _atomic_cas_record(
    *,
    name: str,
    grid_idx: tuple[int, ...],
    program_seq: int,
    addr,
    cmp_value: int,
    value: int,
    sem: str,
) -> tuple[AccessEventRecord, object]:
    old = Int(f"{name}_old")
    success = old == IntVal(cmp_value)
    written_value = If(success, IntVal(value), old)

    record = AccessEventRecord(
        op_type=AtomicCas,
        access_mode="read",
        addr_expr=addr,
        grid_idx=grid_idx,
        program_seq=program_seq,
        active=True,
        reads=True,
        writes=success,
        is_atomic=True,
        atomic_kind="cas",
        sem=sem,
        scope="gpu",
        old_value=old,
        written_value=written_value,
        debug_name=name,
    )

    return record, old


def _build_cas_records(*, load_guarded_by_cas_success: bool) -> list[AccessEventRecord]:
    p0_store_data = _store_record(
        name="P0_store_data",
        grid_idx=(0,),
        program_seq=0,
        addr=DATA,
        value=1,
    )

    p0_release_cas, p0_old = _atomic_cas_record(
        name="P0_release_cas",
        grid_idx=(0,),
        program_seq=1,
        addr=FLAG,
        cmp_value=0,
        value=1,
        sem="release",
    )
    p0_release_cas.premises = (p0_old == 0,)

    p1_acquire_cas, p1_old = _atomic_cas_record(
        name="P1_acquire_cas",
        grid_idx=(1,),
        program_seq=0,
        addr=FLAG,
        cmp_value=1,
        value=1,
        sem="acquire",
    )
    p1_acquire_cas.premises = (p1_old >= 0, p1_old <= 1)

    load_active = p1_old == 1 if load_guarded_by_cas_success else True
    p1_load_data = _load_record(
        name="P1_load_data",
        grid_idx=(1,),
        program_seq=1,
        addr=DATA,
        active=load_active,
    )

    return [
        p0_store_data,
        p0_release_cas,
        p1_acquire_cas,
        p1_load_data,
    ]


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


# ======== HB solver demos ========


def test_plain_program_order_suppresses_same_grid_conflict():
    records = [
        _store_record(
            name="P0_store_data",
            grid_idx=(0,),
            program_seq=0,
            addr=DATA,
            value=1,
        ),
        _load_record(
            name="P0_load_data",
            grid_idx=(0,),
            program_seq=1,
            addr=DATA,
        ),
    ]

    reports = HBSolver(records).find_races()

    assert reports == []


def test_cross_grid_plain_store_load_is_racy():
    records = [
        _store_record(
            name="P0_store_data",
            grid_idx=(0,),
            program_seq=0,
            addr=DATA,
            value=1,
        ),
        _load_record(
            name="P1_load_data",
            grid_idx=(1,),
            program_seq=0,
            addr=DATA,
        ),
    ]

    reports = HBSolver(records).find_races()

    assert len(reports) == 1
    assert {reports[0].first.name, reports[0].second.name} == {
        "P0_store_data",
        "P1_load_data",
    }


def test_cas_release_acquire_unconditional_load_is_racy():
    records = _build_cas_records(load_guarded_by_cas_success=False)

    reports = HBSolver(records).find_races()

    assert len(reports) == 1

    report = reports[0]
    event_names = {report.first.name, report.second.name}

    assert event_names == {"P0_store_data", "P1_load_data"}
    assert report.model.get("P1_acquire_cas_old") == "0"
    assert report.reason


def test_cas_release_acquire_guarded_load_is_not_racy():
    records = _build_cas_records(load_guarded_by_cas_success=True)

    reports = HBSolver(records).find_races()

    assert reports == []


def test_relaxed_cas_does_not_synchronize_even_when_guarded_load_succeeds():
    records = _build_cas_records(load_guarded_by_cas_success=True)
    p0_store_data, p0_release_cas, p1_acquire_cas, p1_load_data = records
    del p0_store_data, p1_acquire_cas, p1_load_data

    p0_release_cas.sem = "relaxed"

    reports = HBSolver(records).find_races()

    assert len(reports) == 1

    report = reports[0]
    assert {report.first.name, report.second.name} == {
        "P0_store_data",
        "P1_load_data",
    }
    assert report.model.get("P1_acquire_cas_old") == "1"


def test_cta_scope_does_not_synchronize_across_different_grids():
    """cta scope neither synchronizes across CTAs (the data pair races) nor
    makes the CAS pair mutually atomic (block-local atomics from different
    CTAs at the same address race too)."""

    records = _build_cas_records(load_guarded_by_cas_success=True)
    p0_store_data, p0_release_cas, p1_acquire_cas, p1_load_data = records
    del p0_store_data, p1_load_data

    p0_release_cas.scope = "cta"
    p1_acquire_cas.scope = "cta"

    reports = HBSolver(records).find_races()

    assert len(reports) == 2

    data_reports = [
        r
        for r in reports
        if {r.first.name, r.second.name} == {"P0_store_data", "P1_load_data"}
    ]
    cas_reports = [
        r
        for r in reports
        if {r.first.name, r.second.name} == {"P0_release_cas", "P1_acquire_cas"}
    ]
    assert len(data_reports) == 1 and len(cas_reports) == 1
    assert data_reports[0].model.get("P1_acquire_cas_old") == "1"


def test_null_race_detector_reports_disabled_status():
    """NullRaceDetector signals disabled state via the public attributes so
    callers don't read ``last_reports == []`` as a clean pass when the
    backend is off.
    """
    detector = NullRaceDetector()
    assert detector.last_status == "disabled"
    assert detector.last_reports == []
    assert detector.unsupported_reason == "race detector disabled"


# ======== Launch lifecycle — capture slot and eval-scoped hooks ========


def _make_masked_load_without_other() -> LoadSymbolicExpr:
    ptr = ConstSymbolicExpr("const", value=1000, dtype=pointer_type(INT32))
    mask = ConstSymbolicExpr("const", value=True, dtype=INT1)
    return LoadSymbolicExpr("load", ptr, mask, None)


def test_load_value_provider_scoped_to_detector_evals():
    """The tl.load value provider is class-global on SymbolicExpr, so it
    must only be installed around the detector's own evaluations.
    Regression test: grid_callback installed it for the whole launch, so a
    co-attached client's expr.eval() dispatched through the race detector's
    provider — a masked load without `other` raised
    UnsupportedSymbolicRaceQuery inside foreign code, and unmasked loads
    silently swapped pointer-as-value semantics for Select(arr, addr).
    """
    detector = SymbolicRaceDetector()
    detector.grid_callback((4, 1, 1))
    try:
        assert SymbolicExpr._load_value_provider is None

        # Foreign eval (sanitizer-style): legacy pointer-as-value lowering,
        # no UnsupportedSymbolicRaceQuery.
        z3_val, _ = _make_masked_load_without_other().eval()
        assert str(z3_val) == "1000"

        # Detector eval: routes through the provider, whose policy rejects
        # the unresolvable probe tensor — proof the hook was installed for
        # exactly this evaluation, and uninstalled afterwards.
        assert detector._safe_eval(_make_masked_load_without_other(), "probe") is None
        assert detector.last_status == "unsupported"
        assert "tl.load value" in (detector.unsupported_reason or "")
        assert SymbolicExpr._load_value_provider is None
    finally:
        detector._clear_launch_runtime()


def test_raise_or_mark_records_unsupported_before_raising():
    """The abort_on_error exception unwinds through trace.py's
    finalize-on-error path; without the mark that path reported a clean
    'ok' for the aborted launch."""
    detector = SymbolicRaceDetector(abort_on_error=True)
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="probe reason"):
        detector._raise_or_mark("probe reason")
    assert detector.last_status == "unsupported"
    assert detector.unsupported_reason == "probe reason"


def test_scalar_truthiness_policy_covers_triton_tree_kernel_frames():
    """@jit/vendored code under the triton package tree is kernel code, not
    frontend plumbing: host-style truthiness there on a per-instance scalar
    must hit the unsupported policy instead of being silently forced True.
    The frontend's None-guards (semantic.py / core.py) stay exempt."""
    triton_pkg_dir, _, plumbing_files = _triton_frame_dirs()
    cas = SymbolicExpr.create(
        "atomic_cas",
        ConstSymbolicExpr("const", value=0, dtype=pointer_type(INT32)),
        ConstSymbolicExpr("const", value=0, dtype=INT32),
        ConstSymbolicExpr("const", value=1, dtype=INT32),
    )

    def observe_from(filename: str, detector: SymbolicRaceDetector) -> None:
        # Run the observer from a frame whose co_filename is `filename` —
        # the initiator the truthiness classifier sees.
        code = compile(
            "def probe(observer, expr):\n    observer(expr)\n", filename, "exec"
        )
        namespace: dict = {}
        exec(code, namespace)
        namespace["probe"](detector._scalar_concretize_observer_impl, cas)

    vendored = os.path.join(triton_pkg_dir, "tools", "vendored_kernel.py")
    assert vendored not in plumbing_files
    detector = SymbolicRaceDetector(abort_on_error=True)
    with pytest.raises(UnsupportedSymbolicRaceQuery, match="host-side control flow"):
        observe_from(vendored, detector)
    assert detector.last_status == "unsupported"

    for plumbing_file in plumbing_files:
        detector = SymbolicRaceDetector(abort_on_error=True)
        observe_from(plumbing_file, detector)
        assert detector._unsupported_capture is False


def test_finalize_reports_aborted_for_unsealed_capture():
    """A launch whose capture was never sealed (an exception aborted it
    mid-block; trace.py calls finalize before re-raising) must not read as
    a clean no-race verdict."""
    detector = SymbolicRaceDetector()
    detector.grid_callback((2, 1, 1))
    assert detector.finalize() == []
    assert detector.last_status == "aborted"
    assert detector.last_reports == []


def test_pre_run_claims_capture_once_and_respects_need_full_grid():
    """pre_run keeps the shared scheduling semantics — need_full_grid keeps
    the grid running so a co-attached client is not starved — while the
    one-shot capture slot admits exactly one block's events."""
    detector = SymbolicRaceDetector()
    detector.grid_callback((4, 1, 1))
    try:
        detector.grid_idx = (0, 0, 0)
        assert detector.pre_run_callback(_dispatch_kernel) is True
        assert detector._capture_active() is True
        # The engine concretized a per-block value mid-block.
        detector.need_full_grid = True
        assert detector.post_run_callback(_dispatch_kernel) is True
        # The next block still runs (no starvation of co-attached clients)
        # but its events fall outside the sealed capture.
        detector.grid_idx = (1, 0, 0)
        assert detector.pre_run_callback(_dispatch_kernel) is True
        assert detector._capture_active() is False
    finally:
        detector._clear_launch_runtime()


def test_capture_slot_excludes_other_threads():
    """Sibling workers under TRITON_VIZ_NUM_SMS >= 2 pass pre_run while the
    capture is still in flight; their events must not reach the shared
    per-launch record state (records, program_seq, loop bookkeeping)."""
    detector = SymbolicRaceDetector()
    detector.grid_callback((2, 1, 1))
    try:
        worker = threading.Thread(
            target=lambda: detector.pre_run_callback(_dispatch_kernel)
        )
        worker.start()
        worker.join()
        # This thread's block is admitted by the scheduler...
        assert detector.pre_run_callback(_dispatch_kernel) is True
        # ...but the capture slot belongs to the worker thread.
        assert detector._capture_active() is False
    finally:
        detector._clear_launch_runtime()


# ======== Unlowerable ops — backstop, address guard, block pointers ========


def _cumsum_vec_expr() -> SymbolicExpr:
    value = SymbolicTensorValue(np.ones(4, dtype=np.int32), INT32)
    vec = SymbolicExpr.create("const", value, block_type(INT32, [4]))
    return SymbolicExpr.create("cumsum", vec, 0, False, None)


def _sort_vec_expr() -> SymbolicExpr:
    value = SymbolicTensorValue(np.array([3, 1, 2, 0], dtype=np.int32), INT32)
    vec = SymbolicExpr.create("const", value, block_type(INT32, [4]))
    return SymbolicExpr.create("sort", vec, 0, False, None)


def test_safe_eval_translates_lowering_gap_into_unsupported():
    """SymbolicExpr lowering gaps surface as NotImplementedError (cumsum,
    dot, block-ptr descriptors). Regression test: _safe_eval caught only
    UnsupportedSymbolicRaceQuery, so the raw NotImplementedError escaped
    and crashed the launch instead of yielding an unsupported verdict."""
    detector = SymbolicRaceDetector()
    detector.grid_callback((2, 1, 1))
    try:
        assert detector._safe_eval(_cumsum_vec_expr(), "probe eval") is None
        assert detector.last_status == "unsupported"
        assert "probe eval" in (detector.unsupported_reason or "")
    finally:
        detector._clear_launch_runtime()


def test_safe_eval_lowering_gap_raises_usq_under_abort_on_error():
    """Under abort_on_error the backstop must raise the detector's own
    exception type (with the mark recorded first), not the raw
    NotImplementedError."""
    detector = SymbolicRaceDetector(abort_on_error=True)
    detector.grid_callback((2, 1, 1))
    try:
        with pytest.raises(UnsupportedSymbolicRaceQuery, match="probe eval"):
            detector._safe_eval(_cumsum_vec_expr(), "probe eval")
        assert detector.last_status == "unsupported"
    finally:
        detector._clear_launch_runtime()


@pytest.mark.parametrize(
    "op_name, make_expr", [("cumsum", _cumsum_vec_expr), ("sort", _sort_vec_expr)]
)
def test_value_dependent_address_is_rejected_not_mislowered(op_name, make_expr):
    """Addresses derived from value-dependent ops must mark the launch
    unsupported. Regression test: the guard only checked has_op("load"),
    so a cumsum-derived pointer crashed with NotImplementedError and a
    sort-derived pointer silently lowered as the identity of its input —
    a wrong footprint under a clean 'ok' verdict."""
    detector = SymbolicRaceDetector()
    detector.grid_callback((2, 1, 1))
    try:
        base = SymbolicExpr.create("const", 1000, pointer_type(FLOAT32))
        ptr = SymbolicExpr.create("addptr", base, make_expr())
        store = SymbolicExpr.create(
            "store", ptr, SymbolicExpr.create("const", 1, INT32), None
        )
        detector._handle_access_check(store, Store, "write")
        assert detector.last_status == "unsupported"
        assert op_name in (detector.unsupported_reason or "")
        assert detector.records == []
    finally:
        detector._clear_launch_runtime()


def _make_block_ptr_expr(offset: int = 0) -> SymbolicExpr:
    base = SymbolicExpr.create("const", 1000, pointer_type(FLOAT32))
    return SymbolicExpr.create(
        "make_block_ptr",
        base,
        [SymbolicExpr.create("const", 64, INT32)],
        [SymbolicExpr.create("const", 1, INT32)],
        [SymbolicExpr.create("const", offset, INT32)],
        [32],
        [0],
    )


def test_block_pointer_access_records_tile_footprint():
    """For TensorPointerLoad/Store the event address must come from the
    access expr itself (which lowers the tile footprint) — expr.ptr is a
    make_block_ptr/advance descriptor whose lowering raises
    NotImplementedError. Regression test: _handle_access_check evaluated
    expr.ptr, so every tl.make_block_ptr kernel crashed. The tile index
    vars must be copy-local so the two-copy solver lets each program copy
    pick its own tile element."""
    detector = SymbolicRaceDetector()
    detector.grid_callback((2, 1, 1))
    try:
        store = SymbolicExpr.create(
            "tensor_pointer_store",
            _make_block_ptr_expr(),
            SymbolicExpr.create("const", 1, FLOAT32),
            (0,),
        )
        detector._handle_access_check(store, TensorPointerStore, "write")
        # last_status is pessimistically "aborted" from grid_callback until
        # finalize() produces a verdict (which this unit test never reaches —
        # no block runs, so the capture is never sealed). A healthy
        # _handle_access_check must leave that launch-start value untouched:
        # not degraded to "unsupported" (the block-ptr lowering regression)
        # and not prematurely upgraded to "ok".
        assert detector.last_status == "aborted", detector.unsupported_reason
        assert len(detector.records) == 1
        record = detector.records[0]
        assert "blk_k_0" in str(record.addr_expr)
        assert "blk_k_0" in str(record.local_constraints)
        assert any(v.decl().name() == "blk_k_0" for v in record.copy_local_vars)
    finally:
        detector._clear_launch_runtime()
