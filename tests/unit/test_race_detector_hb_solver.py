from __future__ import annotations

from z3 import If, Int, IntVal

from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.clients.race_detector.hb_solver import HBSolver
from triton_viz.core.data import AtomicCas, Load, Store


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
