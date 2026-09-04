"""Solver-free pins for the L1 rung's analysis (``concrete_enum.analyze``)
on synthetic recorder state, in the style of ``test_enum_fallback.py``:
one test per clause of the conflict predicate and of the value-source
premise, with hand-built footprints so each verdict is attributable.
"""

import numpy as np

from triton_viz.clients import RaceType
from triton_viz.clients.race_detector.concrete_enum import (
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
) -> int:
    """Append one operation with the given lane addresses (element
    starts); mirrors the recorder's interval construction."""
    op_id = len(rec.op_kind)
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


def test_value_source_load_overlapping_its_own_instances_write_refuses_too():
    # A2 says the value-source tensors are UNWRITTEN by the kernel, in any
    # instance; a same-instance write before the load is program-ordered
    # but still outside the premise (the symbolic frontends refuse it too)
    rec = _rec()
    p = _pid(rec, 0)
    _op(rec, p, _KIND_STORE, [BASE])
    _op(rec, p, _KIND_LOAD, [BASE], value_source=True)
    assert analyze(rec).status == "unsupported"


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
