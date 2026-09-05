"""Solver-free pins for the L1 rung's analysis (``concrete_enum.analyze``)
on synthetic recorder state, in the style of ``test_enum_fallback.py``:
one test per clause of the conflict predicate and of the value-source
premise, with hand-built footprints so each verdict is attributable.
"""

import numpy as np

from triton_viz.clients import RaceType
from triton_viz.clients.race_detector.concrete_enum import (
    _ATOMIC,
    _KIND_CAS,
    _KIND_LOAD,
    _KIND_RMW,
    _KIND_STORE,
    _SCOPE_CODES,
    ConcreteFootprintRecorder,
    analyze,
)

BASE = 1 << 20


def _rec() -> ConcreteFootprintRecorder:
    rec = ConcreteFootprintRecorder()
    rec.grid = (4, 1, 1)
    return rec


def _pid(rec: ConcreteFootprintRecorder, pid: int) -> int:
    rec.pids.append((pid, 0, 0))
    return len(rec.pids) - 1


def _op(
    rec: ConcreteFootprintRecorder,
    pid_index: int,
    kind: int,
    addrs,
    *,
    elem: int = 4,
    scope: str = "gpu",
    site: int = 1,
    value_source: bool = False,
    coalesce: bool = True,
    store_taint=None,
) -> int:
    """Append one operation with the given lane addresses (element
    starts); mirrors the recorder's interval construction. ``store_taint``
    is the taint of the value a store writes (loads: None; atomics: the
    atomic marker)."""
    op_id = len(rec.op_kind)
    if kind in (_KIND_RMW, _KIND_CAS):
        rec.op_store_taint.append(frozenset((_ATOMIC,)))
    elif kind == _KIND_STORE:
        rec.op_store_taint.append(frozenset(store_taint or ()))
    else:
        rec.op_store_taint.append(None)
    rec.op_pid_index.append(pid_index)
    rec.op_seq.append(sum(1 for p in rec.op_pid_index if p == pid_index) - 1)
    rec.op_kind.append(kind)
    rec.op_elem.append(elem)
    rec.op_scope.append(_SCOPE_CODES[scope] if kind in (_KIND_RMW, _KIND_CAS) else 0)
    rec.op_site.append(rec._site_id(("synthetic.py", site, "k")))
    rec.op_lanes.append(len(addrs))
    rec.op_value_source.append(value_source)
    a = np.sort(np.asarray(addrs, dtype=np.int64))
    if kind == _KIND_STORE and a.size > 1 and np.any(np.diff(a) < elem):
        rec.intra_dups.append((op_id, int(a[np.nonzero(np.diff(a) < elem)[0][0] + 1])))
    uniq = np.unique(a)
    if kind in (_KIND_RMW, _KIND_CAS) or not coalesce or uniq.size == 1:
        starts, ends = uniq, uniq + elem
    else:
        brk = np.nonzero(uniq[1:] != uniq[:-1] + elem)[0]
        starts = uniq[np.concatenate(([0], brk + 1))]
        ends = uniq[np.concatenate((brk, [uniq.size - 1]))] + elem
    rec.intervals.append(starts, ends, op_id)
    return op_id


def test_disjoint_stores_prove_clean():
    rec = _rec()
    for p in range(4):
        _op(rec, _pid(rec, p), _KIND_STORE, [BASE + 16 * p + 4 * i for i in range(4)])
    out = analyze(rec)
    assert out.status == "ok"
    assert out.reports == []


def test_overlapping_stores_of_distinct_instances_race_with_byte_range():
    rec = _rec()
    a = _op(rec, _pid(rec, 0), _KIND_STORE, [BASE, BASE + 4], site=10)
    b = _op(rec, _pid(rec, 1), _KIND_STORE, [BASE + 4, BASE + 8], site=11)
    out = analyze(rec)
    assert out.status == "races"
    rep = out.reports[0]
    assert rep.race_type == RaceType.WAW
    assert rep.byte_range == (BASE + 4, BASE + 8)
    assert rep.witness_addr == BASE + 4
    assert {rep.witness_grid_a, rep.witness_grid_b} == {(0, 0, 0), (1, 0, 0)}
    assert (
        rep.first_record.source_location[1],
        rep.second_record.source_location[1],
    ) == (10, 11)
    assert a != b


def test_same_instance_overlaps_are_never_races():
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_STORE, [BASE])
    _op(rec, p, _KIND_STORE, [BASE])
    _op(rec, p, _KIND_LOAD, [BASE])
    _op(rec, p, _KIND_RMW, [BASE])
    assert analyze(rec).status == "ok"


def test_read_read_overlap_is_not_a_conflict():
    rec = _rec()
    for p in range(4):
        _op(rec, _pid(rec, p), _KIND_LOAD, [BASE + 4 * i for i in range(64)])
    assert analyze(rec).status == "ok"


def test_read_write_overlap_is_a_race_in_either_program_order():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_LOAD, [BASE + 8])
    _op(rec, _pid(rec, 1), _KIND_STORE, [BASE + 8])
    out = analyze(rec)
    assert out.status == "races"
    assert (
        out.reports[0].race_type == RaceType.WAR
    )  # first (pid 0) reads, second writes
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_STORE, [BASE + 8])
    _op(rec, _pid(rec, 1), _KIND_LOAD, [BASE + 8])
    out = analyze(rec)
    assert out.reports[0].race_type == RaceType.RAW


def test_duplicate_lanes_inside_one_store_race_within_the_instance():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_STORE, [BASE, BASE + 4, BASE])
    out = analyze(rec)
    assert out.status == "races"
    rep = out.reports[0]
    assert rep.witness_grid_a == rep.witness_grid_b
    assert rep.byte_range == (BASE, BASE + 4)


def test_partially_overlapping_lanes_inside_one_store_are_duplicates_too():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_STORE, [BASE, BASE + 2], elem=4)
    assert analyze(rec).status == "races"


def test_duplicate_lanes_inside_one_atomic_or_load_do_not_race():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_RMW, [BASE, BASE, BASE])
    _op(rec, _pid(rec, 1), _KIND_LOAD, [BASE + 64, BASE + 64])
    assert analyze(rec).status == "ok"


def test_compatible_atomics_same_address_same_width_gpu_or_sys_scope():
    rec = _rec()
    for p, scope in enumerate(("gpu", "sys", "gpu", "sys")):
        _op(rec, _pid(rec, p), _KIND_RMW, [BASE, BASE + 4], scope=scope)
    _op(rec, _pid(rec, 0), _KIND_CAS, [BASE])
    assert analyze(rec).status == "ok"


def test_cta_scoped_atomic_is_never_compatible_across_instances():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_RMW, [BASE], scope="cta")
    _op(rec, _pid(rec, 1), _KIND_RMW, [BASE], scope="gpu")
    out = analyze(rec)
    assert out.status == "races"
    assert out.reports[0].race_type == RaceType.WAW
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_RMW, [BASE], scope="cta")
    _op(rec, p, _KIND_RMW, [BASE], scope="cta")
    assert analyze(rec).status == "ok"  # same instance: program order


def test_torn_atomics_different_width_or_start_race():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_RMW, [BASE], elem=8)
    _op(rec, _pid(rec, 1), _KIND_RMW, [BASE + 4], elem=4)
    out = analyze(rec)
    assert out.status == "races"
    assert out.reports[0].byte_range == (BASE + 4, BASE + 8)
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_RMW, [BASE], elem=8)
    _op(rec, _pid(rec, 1), _KIND_RMW, [BASE], elem=4)
    assert analyze(rec).status == "races"


def test_atomic_lanes_hitting_many_cells_stay_compatible_per_cell():
    # a histogram: every instance's atomic touches several adjacent cells;
    # per-lane intervals keep the (address, width) judgment exact
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_RMW, [BASE, BASE + 4, BASE + 8, BASE + 12])
    _op(rec, _pid(rec, 1), _KIND_RMW, [BASE + 4, BASE + 8])
    _op(rec, _pid(rec, 2), _KIND_RMW, [BASE + 12])
    assert analyze(rec).status == "ok"


def test_plain_access_overlapping_an_atomic_races():
    for kind in (_KIND_STORE, _KIND_LOAD):
        rec = _rec()
        _op(rec, _pid(rec, 0), _KIND_RMW, [BASE])
        _op(rec, _pid(rec, 1), kind, [BASE])
        out = analyze(rec)
        assert out.status == "races"
        assert out.reports[0].race_type == (
            RaceType.WAW if kind == _KIND_STORE else RaceType.RAW
        )


def test_value_source_load_overlapping_any_write_refuses_by_name():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_STORE, [BASE + 16])
    _op(rec, _pid(rec, 1), _KIND_LOAD, [BASE + 16], value_source=True, site=7)
    out = analyze(rec)
    assert out.status == "unsupported"
    assert out.reason.startswith("value-source:")
    assert "synthetic.py:7" in out.reason
    assert out.reports == []


def test_same_instance_earlier_store_of_plain_data_is_program_ordered():
    # A2 is cross-instance for this rung: the load reads its own
    # instance's program-ordered write, in the sequential run exactly as
    # in every real execution (the value it relays is untainted)
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_STORE, [BASE])
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True)
    assert analyze(rec).status == "ok"


def test_same_instance_later_store_cannot_affect_the_loaded_value():
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True)
    _op(rec, p, _KIND_STORE, [BASE], store_taint={_ATOMIC})  # after the load
    assert analyze(rec).status == "ok"


def test_atomic_return_relayed_through_memory_refuses():
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_STORE, [BASE], store_taint={_ATOMIC}, site=3)
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True, site=4)
    out = analyze(rec)
    assert out.status == "unsupported"
    assert out.reason.startswith("atomic-return:")
    assert "through memory" in out.reason
    # an earlier atomic on the bytes themselves relays the marker too
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_RMW, [BASE])
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True)
    assert analyze(rec).reason.startswith("atomic-return:")


def test_relayed_loaded_value_makes_the_original_load_a_value_source():
    # load A (plain data) -> store scratch -> load scratch -> address:
    # A becomes a value source transitively, and its own premise is
    # checked: clean when A's bytes are unwritten, refused when another
    # instance writes them
    rec = _rec()
    p = _pid(rec, 0)
    a = _op(rec, p, _KIND_LOAD, [BASE + 256])
    _op(rec, p, _KIND_STORE, [BASE], store_taint={a})
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True)
    out = analyze(rec)
    assert out.status == "ok"
    assert rec.op_value_source[a] is True
    assert out.n_value_source_loads == 2
    rec = _rec()
    p = _pid(rec, 0)
    a = _op(rec, p, _KIND_LOAD, [BASE + 256])
    _op(rec, p, _KIND_STORE, [BASE], store_taint={a})
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True)
    _op(rec, _pid(rec, 1), _KIND_STORE, [BASE + 256])  # a foreign write to A's bytes
    out = analyze(rec)
    assert out.status == "unsupported"
    assert out.reason.startswith("value-source:")


def test_value_source_load_of_unwritten_bytes_is_fine_even_next_to_writes():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_STORE, [BASE + 4 * i for i in range(8)])
    _op(rec, _pid(rec, 1), _KIND_LOAD, [BASE + 32], value_source=True)
    _op(rec, _pid(rec, 1), _KIND_LOAD, [BASE - 4], value_source=True)
    assert analyze(rec).status == "ok"


def test_premise_violation_takes_priority_over_race_reports():
    rec = _rec()
    _op(rec, _pid(rec, 0), _KIND_STORE, [BASE, BASE + 64])
    _op(rec, _pid(rec, 1), _KIND_STORE, [BASE])  # a genuine race...
    _op(
        rec, _pid(rec, 2), _KIND_LOAD, [BASE + 64], value_source=True
    )  # ...and a violation
    out = analyze(rec)
    assert out.status == "unsupported"
    assert out.reason.startswith("value-source:")


def test_report_cap_and_dedup_by_site_pair():
    rec = _rec()
    for p in range(6):
        _op(rec, _pid(rec, p), _KIND_STORE, [BASE], site=1)
    out = analyze(rec, max_reports=8)
    assert out.status == "races"
    # every pair shares the same (site, site, WAW, cross-instance) key
    assert len(out.reports) == 1
    rec = _rec()
    for p in range(20):
        _op(rec, _pid(rec, p), _KIND_STORE, [BASE], site=100 + p)
    out = analyze(rec, max_reports=3)
    assert len(out.reports) == 3


def test_empty_launch_proves_clean():
    rec = _rec()
    _pid(rec, 0)
    out = analyze(rec)
    assert out.status == "ok"
    assert out.n_ops == 0


# ── the projected-cost decision (pure) ─────────────────────────────

from triton_viz.clients.race_detector.concrete_enum import (  # noqa: E402
    projected_cost_refusal,
)


def test_projection_waits_for_the_grace_period():
    assert projected_cost_refusal(4.9, [1.0, 1.0, 1.0], 1000, 10.0) is None
    assert projected_cost_refusal(5.0, [1.0, 1.0, 1.0], 1000, 10.0) is not None


def test_projection_excludes_the_first_instance():
    # a heavy warm-up instance followed by light ones: the mean is over
    # the light ones only, so the projection stays under budget
    times = [4.0] + [0.01] * 100
    assert projected_cost_refusal(5.0, times, 500, 20.0) is None
    # the same heavy time on a non-first instance counts (projection
    # 5 + 0.05 * 399 = 25 s > 2 x 10 s)
    times = [0.01] + [4.0] + [0.01] * 99
    assert projected_cost_refusal(5.0, times, 500, 10.0) is not None


def test_projection_needs_more_than_the_skipped_instances():
    assert projected_cost_refusal(9.0, [9.0], 100, 10.0) is None
    assert projected_cost_refusal(9.0, [], 100, 10.0) is None


def test_projection_refuses_only_beyond_the_factor():
    # 10 done, 90 remaining at 0.5 s each = 45 s + 6 s elapsed = 51 s:
    # over a 20 s budget (2x = 40 s) refuses; over a 30 s budget
    # (2x = 60 s) keeps running although the plain budget is exceeded
    # (Hao: the 178 s-vs-150 s case must finish, not abstain)
    detail = projected_cost_refusal(6.0, [0.5] * 10, 100, 20.0)
    assert detail is not None
    assert "10 of 100 instances" in detail
    assert "500.0 ms per instance" in detail
    assert "projected 51s exceeds 2x the 20s budget" in detail
    assert projected_cost_refusal(6.0, [0.5] * 10, 100, 30.0) is None
    # exactly at the factor keeps running; no budget never refuses
    assert projected_cost_refusal(6.0, [0.5] * 10, 78, 20.0) is None
    assert projected_cost_refusal(6.0, [5.0] * 10, 10_000, None) is None
    # nothing remaining: the run is about to finish, never refuse
    assert projected_cost_refusal(60.0, [5.0] * 10, 10, 20.0) is None
    # the factor is a parameter
    assert projected_cost_refusal(6.0, [0.5] * 10, 100, 30.0, factor=1.0) is not None


def test_value_source_check_scales_to_many_loads():
    """Regression: the premise check located each value-source load's
    intervals by a full scan, making it quadratic (rope_fwd_3d: 11840
    instances x 7 ops took minutes after an 81 s run). Now bisection."""
    import time

    rec = _rec()
    n = 6000
    for p in range(n):
        pid = _pid(rec, p)
        _op(rec, pid, _KIND_LOAD, [BASE + 4 * p], value_source=True)
        _op(rec, pid, _KIND_STORE, [BASE + (1 << 24) + 4 * p])
    t0 = time.perf_counter()
    out = analyze(rec)
    assert out.status == "ok"
    assert time.perf_counter() - t0 < 5.0
