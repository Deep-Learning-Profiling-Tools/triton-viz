"""The await abstraction (spec C1): scf.while spin recognition, the single
constrained read encoding, and the litmus proofs it unlocks.

Mini TTIR modules mirror the shapes triton 3.6 actually emits for
``while tl.load/atomic_*(...) != expected: pass`` (verified against
compiled output): a result-free ``scf.while`` whose condition region holds
the re-read + cmp + scf.condition, and a ``do { scf.yield }`` body.
"""

from types import SimpleNamespace

import pytest
import torch

from triton_viz.clients.common.ttir_reader import (
    Not,
    UnsupportedTTIR,
    parse_ttir,
)
from triton_viz.clients.race_detector.compiled.client import CompiledRaceDetector
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
    symbolic_grid,
)
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)


def _t(ptr, numel=64, elem=4, init=None):
    return GlobalTensor(data_ptr=ptr, elem_size=elem, numel=numel, init_values=init)


def _solve(graph, params, tensors):
    # Mirrors the client's T1 call: the launch grid sizes the unread axes
    # of atomic-bearing graphs (read axes stay symbolic).
    enc = encode_graph(graph, params, tensors)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (4, 1, 1)),
        arange_dict=enc.arange_dict,
    )
    return enc, solver.find_races()


def _module(args, *body_lines):
    body = "\n    ".join(body_lines)
    return (
        "module {\n"
        f"  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {body}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


# ─────────────────── shape recognition (C1.1) ───────────────────


def _spin_load_module(cond_lines=None, body_lines=None):
    cond = cond_lines or [
        "%o = tt.load %flag_ptr {isVolatile = true} : !tt.ptr<i32>",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c)",
    ]
    body = body_lines or ["scf.yield"]
    return _module(
        "%flag_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "scf.while : () -> () {",
        *cond,
        "} do {",
        *body,
        "}",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %op, %c1 : !tt.ptr<i32>",
    )


def test_volatile_spin_load_recognized():
    g = parse_ttir(_spin_load_module())
    (await_acc,) = [a for a in g.accesses if a.awaited]
    assert await_acc.kind == "load"
    assert isinstance(await_acc.exit_pred, Not)
    assert not await_acc.in_loop


def test_while_with_carried_values_is_spin_shape():
    text = _module(
        "%flag_ptr: !tt.ptr<i32>",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%r = scf.while (%arg = %c0) : (i32) -> i32 {",
        "%o = tt.load %flag_ptr : !tt.ptr<i32>",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c) %o : i32",
        "} do {",
        "^bb0(%a: i32):",
        "scf.yield %a : i32",
        "}",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_store_in_spin_condition_is_spin_shape():
    text = _spin_load_module(
        cond_lines=[
            "%o = tt.load %flag_ptr : !tt.ptr<i32>",
            "tt.store %flag_ptr, %c1 : !tt.ptr<i32>",
            "%c = arith.cmpi ne, %o, %c1 : i32",
            "scf.condition(%c)",
        ]
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_second_read_in_spin_condition_is_spin_shape():
    text = _spin_load_module(
        cond_lines=[
            "%o = tt.load %flag_ptr : !tt.ptr<i32>",
            "%o2 = tt.load %flag_ptr : !tt.ptr<i32>",
            "%c = arith.cmpi ne, %o, %c1 : i32",
            "scf.condition(%c)",
        ]
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_non_comparison_condition_is_spin_shape():
    text = _spin_load_module(
        cond_lines=[
            "%o = tt.load %flag_ptr : !tt.ptr<i32>",
            "scf.condition(%o)",
        ]
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_op_in_spin_body_is_spin_shape():
    text = _spin_load_module(
        body_lines=[
            "%x = tt.load %flag_ptr : !tt.ptr<i32>",
            "scf.yield",
        ]
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_float_awaited_location_is_spin_shape():
    text = _module(
        "%flag_ptr: !tt.ptr<f32>",
        "%cst = arith.constant 1.000000e+00 : f32",
        "scf.while : () -> () {",
        "%o = tt.load %flag_ptr : !tt.ptr<f32>",
        "%c = arith.cmpf one, %o, %cst : f32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_condition_outside_spin_is_control_flow():
    text = _module(
        "%flag_ptr: !tt.ptr<i32>",
        "%c1 = arith.constant 1 : i32",
        "scf.condition(%c1)",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "control-flow"


# ─────────────────── producer/consumer litmus ───────────────────


def _prod_cons_ttir(*, writer_sem="release", spin_sem="acquire", scope="gpu"):
    return _module(
        "%flag_ptr: !tt.ptr<i32>, %data_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%isp = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %isp {",
        "tt.store %data_ptr, %c1 : !tt.ptr<i32>",
        f"%x = tt.atomic_rmw exch, {writer_sem}, {scope}, %flag_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "} else {",
        "scf.while : () -> () {",
        f"%o = tt.atomic_rmw add, {spin_sem}, {scope}, %flag_ptr, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "%v = tt.load %data_ptr : !tt.ptr<i32>",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %op, %v : !tt.ptr<i32>",
        "}",
    )


_PC_TENSORS = {
    "flag_ptr": _t(0x2000, numel=1, init=(0,)),
    "data_ptr": _t(0x3000, numel=1),
    "out_ptr": _t(0x4000, numel=64),
}


def test_producer_consumer_wait_proved():
    enc, reports = _solve(parse_ttir(_prod_cons_ttir()), {}, _PC_TENSORS)
    assert enc.assumes_termination
    assert reports == []


@pytest.mark.parametrize(
    "mutation",
    [
        dict(writer_sem="relaxed"),
        dict(spin_sem="relaxed"),
        dict(scope="cta"),
    ],
    ids=["relaxed-writer", "relaxed-spinner", "cta-scope"],
)
def test_producer_consumer_mutations_race(mutation):
    """Each broken synchronization must flip the proof to a data race on
    the data cell (mutation-sensitivity, spec C1.4)."""
    _, reports = _solve(parse_ttir(_prod_cons_ttir(**mutation)), {}, _PC_TENSORS)
    data = [
        r
        for r in reports
        if {r.first_record.tensor_name, r.second_record.tensor_name} == {"data_ptr"}
    ]
    assert data


# ─────────────────── mutex via CAS loop ───────────────────


def _mutex_ttir(*, cas_sem="acquire", unlock="xchg-release"):
    if unlock == "xchg-release":
        unlock_lines = [
            "%u = tt.atomic_rmw exch, release, gpu, %lock_ptr, %c0, %true : "
            "(!tt.ptr<i32>, i32, i1) -> i32",
        ]
    else:  # plain store unlock: breaks the release chain AND the closed world
        unlock_lines = ["tt.store %lock_ptr, %c0 : !tt.ptr<i32>"]
    return _module(
        "%lock_ptr: !tt.ptr<i32>, %x_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "scf.while : () -> () {",
        f"%o = tt.atomic_cas {cas_sem}, gpu, %lock_ptr, %c0, %c1 : "
        "(!tt.ptr<i32>, i32, i32) -> i32",
        "%c = arith.cmpi ne, %o, %c0 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "%v = tt.load %x_ptr : !tt.ptr<i32>",
        "tt.store %x_ptr, %v : !tt.ptr<i32>",
        *unlock_lines,
    )


_MUTEX_TENSORS = {
    "lock_ptr": _t(0x2000, numel=1, init=(0,)),
    "x_ptr": _t(0x3000, numel=1),
}


def test_mutex_cas_proved():
    """Critical-section accesses to one shared cell, protected by a CAS
    lock: needs the awaited-CAS value model, the RMW-immediacy axiom (two
    acquisitions of the same "0" unsat) and the modeled xchg unlock."""
    enc, reports = _solve(parse_ttir(_mutex_ttir()), {}, _MUTEX_TENSORS)
    assert enc.assumes_termination
    assert reports == []


def test_mutex_plain_store_unlock_races():
    """Unlock as a plain store: the release sequence is gone (and the lock
    word gains an unmodeled writer) — the critical section must race."""
    _, reports = _solve(
        parse_ttir(_mutex_ttir(unlock="plain-store")), {}, _MUTEX_TENSORS
    )
    x = [
        r
        for r in reports
        if {r.first_record.tensor_name, r.second_record.tensor_name} == {"x_ptr"}
    ]
    assert x


def test_mutex_relaxed_cas_races():
    _, reports = _solve(parse_ttir(_mutex_ttir(cas_sem="relaxed")), {}, _MUTEX_TENSORS)
    assert reports


# ─────────────────── look-back chain ───────────────────


def _lookback_ttir(*, scope="gpu"):
    return _module(
        "%flag_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c-1_i32 = arith.constant -1 : i32",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%ispos = arith.cmpi sgt, %pid, %c0 : i32",
        "scf.if %ispos {",
        "scf.while : () -> () {",
        "%f = tt.addptr %flag_ptr, %pid : !tt.ptr<i32>, i32",
        "%fp = tt.addptr %f, %c-1_i32 : !tt.ptr<i32>, i32",
        f"%o = tt.atomic_rmw add, acquire, {scope}, %fp, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi eq, %o, %c0 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "%p = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "%pp = tt.addptr %p, %c-1_i32 : !tt.ptr<i32>, i32",
        "%prev = tt.load %pp : !tt.ptr<i32>",
        "%q = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %q, %prev : !tt.ptr<i32>",
        "} else {",
        "%q2 = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %q2, %c1 : !tt.ptr<i32>",
        "}",
        "%fm = tt.addptr %flag_ptr, %pid : !tt.ptr<i32>, i32",
        f"%pub = tt.atomic_rmw exch, release, {scope}, %fm, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
    )


_LOOKBACK_TENSORS = {
    "flag_ptr": _t(0x2000, numel=64, init=tuple([0] * 64)),
    "out_ptr": _t(0x8000, numel=64),
}


def test_lookback_chain_proved():
    """pid i spins on flag[i-1] (pid-dependent, loop-invariant address) and
    publishes flag[i] with release: adjacent-slot conflicts are ordered by
    the awaited rf edge."""
    enc, reports = _solve(parse_ttir(_lookback_ttir()), {}, _LOOKBACK_TENSORS)
    assert enc.assumes_termination
    assert reports == []


def test_lookback_wrong_scope_races():
    _, reports = _solve(parse_ttir(_lookback_ttir(scope="cta")), {}, _LOOKBACK_TENSORS)
    assert reports


# ─────────────────── client guards (C1.3) ───────────────────


def _drive_client(det, ttir, tensors_args, names, grid=(2,)):
    jit_fn = SimpleNamespace(arg_names=names)
    det.pre_warmup_callback(jit_fn, *tensors_args, grid=grid)
    det.post_warmup_callback(jit_fn, SimpleNamespace(asm={"ttir": ttir}))
    det.finalize()


def test_client_surfaces_assumes_termination():
    det = CompiledRaceDetector(confirm_races=False)
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(64, dtype=torch.int32)
    _drive_client(
        det,
        _prod_cons_ttir(),
        (flag, data, out),
        ["flag_ptr", "data_ptr", "out_ptr"],
    )
    assert det.last_global_status == "ok"
    assert det.last_global_assumes_termination
    assert det.last_global_provenance is not None
    assert det.last_global_provenance.endswith("+assumes-termination")


def test_client_await_replay_unavailable_before_execution():
    """C1.3.4 (mandatory): an await-bearing kernel with reports must be
    classified unavailable BEFORE any replay executes — the sequential
    interpreter would spin forever."""
    det = CompiledRaceDetector(confirm_races=True)
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(64, dtype=torch.int32)
    _drive_client(
        det,
        _prod_cons_ttir(spin_sem="relaxed"),
        (flag, data, out),
        ["flag_ptr", "data_ptr", "out_ptr"],
    )
    assert det.last_global_status == "races"
    assert det.last_global_confirmation is None
    assert det.last_global_assumes_termination
    assert "await-bearing" in (det.last_global_reason or "")


def test_client_await_skips_differential_symmetrically():
    det = CompiledRaceDetector(confirm_races=False, differential_check=True)
    flag = torch.zeros(1, dtype=torch.int32)
    data = torch.zeros(1, dtype=torch.int32)
    out = torch.zeros(64, dtype=torch.int32)
    _drive_client(
        det,
        _prod_cons_ttir(),
        (flag, data, out),
        ["flag_ptr", "data_ptr", "out_ptr"],
    )
    assert det.last_global_status == "ok"
    assert det.last_differential is None  # excluded, not fabricated


# ──────────── adversarial regressions (S6 verification round) ────────────


def test_mutating_rmw_spin_is_spin_shape():
    """Adversarial finding 4: `while atomic_add(flag, 1) != 1` terminates
    by observing its OWN increments; dropping the intermediate writes and
    attributing the exit value to a release writer fabricated a sw edge.
    A mutating re-read must be refused (spin-shape)."""
    text = _module(
        "%flag_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c1 = arith.constant 1 : i32",
        "scf.while : () -> () {",
        "%o = tt.atomic_rmw add, acquire, gpu, %flag_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"
    assert "MUTATES" in str(exc.value)


@pytest.mark.parametrize("op", ["exch", "max", "min"])
def test_non_identity_rmw_spin_is_spin_shape(op):
    text = _module(
        "%flag_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "scf.while : () -> () {",
        f"%o = tt.atomic_rmw {op}, acquire, gpu, %flag_ptr, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi eq, %o, %c0 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
    )
    with pytest.raises(UnsupportedTTIR) as exc:
        parse_ttir(text)
    assert exc.value.kind == "spin-shape"


def test_identity_add_zero_spin_still_accepted():
    """The add-0 fetch (the canonical acquire spin) stays in scope."""
    g = parse_ttir(_prod_cons_ttir())
    assert any(a.awaited for a in g.accesses)


# ──────────── pre-exit representative (await-collapse repair) ────────────
#
# The collapse keeps ONE poll whose termination premise pins its
# observation to the exit value; races living only on the dropped failed
# iterations survive on the value-model-free pre-exit representative.
# The existing clean canaries above (producer-consumer, mutex, lookback)
# double as the escape regression: the rep must not reopen rf_unknown for
# its own poll (hardening (b) in _has_unmodeled_overlapping_writer).


def _pc_reset_ttir(*, reset):
    """Producer/consumer wait with an extra access to the awaited flag
    po-before the release publish (the pid-0 branch)."""
    reset_lines = {
        "atomic-cta": [
            "%r = tt.atomic_rmw exch, relaxed, cta, %flag_ptr, %c0, %true : "
            "(!tt.ptr<i32>, i32, i1) -> i32",
        ],
        "atomic-gpu": [
            "%r = tt.atomic_rmw exch, relaxed, gpu, %flag_ptr, %c0, %true : "
            "(!tt.ptr<i32>, i32, i1) -> i32",
        ],
        "plain-store": ["tt.store %flag_ptr, %c0 : !tt.ptr<i32>"],
        "plain-read": ["%fv = tt.load %flag_ptr : !tt.ptr<i32>"],
    }[reset]
    return _module(
        "%flag_ptr: !tt.ptr<i32>, %data_ptr: !tt.ptr<i32>, %out_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%isp = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %isp {",
        "tt.store %data_ptr, %c1 : !tt.ptr<i32>",
        *reset_lines,
        "%x = tt.atomic_rmw exch, release, gpu, %flag_ptr, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "} else {",
        "scf.while : () -> () {",
        "%o = tt.atomic_rmw add, acquire, gpu, %flag_ptr, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "%v = tt.load %data_ptr : !tt.ptr<i32>",
        "%op = tt.addptr %out_ptr, %pid : !tt.ptr<i32>, i32",
        "tt.store %op, %v : !tt.ptr<i32>",
        "}",
    )


def _endpoint_records(report):
    return (report.first_record, report.second_record)


def test_cta_scoped_atomic_reset_before_publish_reports_via_rep():
    """Corner A: a VALUE-MODELED weak atomic write of the awaited flag —
    an equal-width relaxed cta-scoped xchg reset po-before the gpu release
    publish. Before the pre-exit representative this kernel silently
    PROVED: the reset's written value is modeled (no escape opens), and
    the termination premise forces the poll's rf from the publisher, which
    orders the reset behind the poll in every model. The rep restores the
    failed iterations: one WAW between the reset and the ``:pre-exit``
    event (cta scope voids mutual atomicity across CTAs)."""
    enc, reports = _solve(
        parse_ttir(_pc_reset_ttir(reset="atomic-cta")), {}, _PC_TENSORS
    )
    assert enc.assumes_termination
    assert len(reports) == 1
    (report,) = reports
    recs = _endpoint_records(report)
    assert any(r.pre_exit and r.debug_name.endswith(":pre-exit") for r in recs)
    (reset_rec,) = [r for r in recs if not r.pre_exit]
    assert reset_rec.scope == "cta"
    from triton_viz.clients.race_detector.data import RaceType

    assert report.race_type is RaceType.WAW


def test_plain_read_of_awaited_flag_reports_via_rep_write_half():
    """Corner B: a plain READ of the awaited location po-before the
    publish. Reads open no escape, so this was a silent under-report too;
    unrolled, the read races the identity write-backs of the failed RMW
    iterations — carried by the rep's write half as one WAR."""
    enc, reports = _solve(
        parse_ttir(_pc_reset_ttir(reset="plain-read")), {}, _PC_TENSORS
    )
    assert len(reports) == 1
    (report,) = reports
    recs = _endpoint_records(report)
    assert any(r.pre_exit for r in recs)
    (read_rec,) = [r for r in recs if not r.pre_exit]
    assert read_rec.access_mode == "read" and not read_rec.is_atomic
    from triton_viz.clients.race_detector.data import RaceType

    assert report.race_type is RaceType.WAR


def test_gpu_atomic_reset_before_publish_stays_clean():
    """The morally-strong twin: a gpu-scoped relaxed xchg reset. The
    modeled 0-writer is excluded from the poll's rf by the termination
    premise, and it is mutually atomic with both the poll and the rep
    (device scope, equal width, same address) — no conflict, no report."""
    _, reports = _solve(parse_ttir(_pc_reset_ttir(reset="atomic-gpu")), {}, _PC_TENSORS)
    assert reports == []


def test_plain_store_reset_verdict_unchanged_by_rep():
    """A plain-store reset of the awaited flag reports via the rf_unknown
    escape (the store is an unmodeled overlapping writer, so the sw bridge
    is optional): the flag pair and the ordering-dependent data pair both
    surface. The rep must keep that verdict — it may add its own flag
    endpoint but must not lose either pre-existing pair."""
    _, reports = _solve(
        parse_ttir(_pc_reset_ttir(reset="plain-store")), {}, _PC_TENSORS
    )
    pairs = [{r.first_record.tensor_name, r.second_record.tensor_name} for r in reports]
    assert {"flag_ptr"} in pairs
    assert {"data_ptr"} in pairs
    # the pre-existing pairs survive with non-rep endpoints on both sides
    non_rep_flag = [
        r
        for r in reports
        if {r.first_record.tensor_name, r.second_record.tensor_name} == {"flag_ptr"}
        and not (r.first_record.pre_exit or r.second_record.pre_exit)
    ]
    assert non_rep_flag


def test_guarded_await_rep_inherits_uncertainty():
    """A spin under an UNMODELED enclosing condition is ``guarded``: its
    poll record lands in uncertain_event_ids and its reports classify
    widened. The rep must inherit that membership (and the poll's exact
    activity), or a guarded-await rep report would surface as a definite
    race from an over-approximated record."""
    text = _module(
        "%flag_ptr: !tt.ptr<i32>, %gate_ptr: !tt.ptr<i32>",
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%g = tt.load %gate_ptr : !tt.ptr<i32>",
        "%cnd = arith.cmpi ne, %g, %c0 : i32",
        "scf.if %cnd {",
        "scf.while : () -> () {",
        "%o = tt.atomic_rmw add, acquire, gpu, %flag_ptr, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "%c = arith.cmpi ne, %o, %c1 : i32",
        "scf.condition(%c)",
        "} do {",
        "scf.yield",
        "}",
        "}",
    )
    g = parse_ttir(text)
    (poll_seq,) = [i for i, a in enumerate(g.accesses) if a.awaited]
    assert g.accesses[poll_seq].guarded
    enc = encode_graph(
        g,
        {},
        {
            "flag_ptr": _t(0x2000, numel=1, init=(0,)),
            "gate_ptr": _t(0x3000, numel=1),
        },
    )
    (rep,) = [r for r in enc.records if r.pre_exit]
    (poll_rec,) = [
        r for r in enc.records if r.program_seq == poll_seq and not r.pre_exit
    ]
    assert poll_seq in enc.uncertain_event_ids
    assert rep.event_id in enc.uncertain_event_ids
    assert rep.active is poll_rec.active


def test_cas_poll_rep_is_read_only_and_keeps_escape_closed():
    """The CAS-poll rep (mutex): a failed CAS writes nothing, so the rep
    is a read-only atomic (atomic_kind "rmw" with statically-False
    record-level writes, honored by the solver's lowering). Behavior pin:
    the rep must not open the rf_unknown escape for its own closed-world
    lock poll — today the candidate filter excludes every pre_exit record
    outright, and the statically-False writes independently guarantee it
    (the feasibility check conjoins e.writes), so the pin survives either
    mechanism."""
    from z3 import is_false, simplify

    g = parse_ttir(_mutex_ttir())
    enc = encode_graph(g, {}, _MUTEX_TENSORS)
    (rep,) = [r for r in enc.records if r.pre_exit]
    assert rep.atomic_kind == "rmw"
    assert rep.writes is False
    assert rep.old_value is None and rep.cas_cmp_value is None
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=symbolic_grid(enc, (4, 1, 1)),
        arange_dict=enc.arange_dict,
    )
    for e in solver.events:
        if e.record.pre_exit:
            assert is_false(simplify(e.writes))
    # the closed world of the lock poll stays closed: no rf_unknown choice
    assert solver.rf_unknown_source == {}
    assert solver.find_races() == []


def test_exactly_one_rep_per_await_and_never_in_graph_accesses():
    """Structural: encode_graph emits exactly one rep per awaited access,
    with a fresh event_id above the dense seq range and the poll's seq;
    graph.accesses itself is untouched (the reader-side ``(await_acc,) =``
    unpacks stay valid)."""
    g = parse_ttir(_prod_cons_ttir())
    n_awaited = sum(1 for a in g.accesses if a.awaited)
    assert n_awaited == 1
    enc = encode_graph(g, {}, _PC_TENSORS)
    reps = [r for r in enc.records if r.pre_exit]
    assert len(reps) == n_awaited
    (rep,) = reps
    (poll_seq,) = [i for i, a in enumerate(g.accesses) if a.awaited]
    assert rep.program_seq == poll_seq
    assert rep.event_id >= len(g.accesses)
    assert rep.debug_name.endswith(":pre-exit")
    assert rep.old_value is None and rep.rmw_op is None and rep.rmw_operand is None
    assert len(enc.records) == len(g.accesses) + 1
    # identity-RMW rep keeps the reads-AND-writes footprint
    assert rep.is_atomic and rep.reads is True and rep.writes is True


def test_nested_watchdog_restores_outer_timer():
    """Adversarial minor: an inner watchdog must re-arm the enclosing
    SIGALRM timer's remaining time instead of permanently defusing it."""
    import signal
    import threading

    from triton_viz.clients.race_detector.compiled.replay import (
        _replay_watchdog as _watchdog,
    )

    if not hasattr(signal, "SIGALRM") or (
        threading.current_thread() is not threading.main_thread()
    ):
        pytest.skip("watchdog not armable here")
    old_handler = signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 30.0)
    try:
        with _watchdog(5.0):
            pass
        remaining, _interval = signal.getitimer(signal.ITIMER_REAL)
        assert remaining > 0, "outer timer must survive the inner watchdog"
        assert remaining <= 30.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
