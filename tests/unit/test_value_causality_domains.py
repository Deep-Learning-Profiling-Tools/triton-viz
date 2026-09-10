"""Keep operational atomic dependencies separate from execution-domain facts."""

from dataclasses import replace

import pytest
from z3 import If, Int, IntVal, Not, Solver, sat, unsat

from triton_viz.clients.common.ttir_reader import parse_ttir
from triton_viz.clients.race_detector.compiled.global_records import (
    GlobalTensor,
    encode_graph,
)
from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.clients.symbolic_engine import SymbolicExpr
from triton_viz.core.data import AtomicCas, AtomicRMW


def _spin(op, suffix):
    return [
        "scf.while : () -> () {",
        f"%o{suffix} = tt.atomic_rmw {op}, acquire, gpu, %flag, %c0, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        f"%c{suffix} = arith.cmpi ne, %o{suffix}, %c1 : i32",
        f"scf.condition(%c{suffix})",
        "} do {",
        "scf.yield",
        "}",
    ]


def _two_awaits(*, sequential, writer_sem="release"):
    first, second = _spin("add", "a"), _spin("or", "b")
    if sequential:
        waits = first + second
    else:
        waits = [
            "%isfirst = arith.cmpi eq, %pid, %c1 : i32",
            "scf.if %isfirst {",
            *first,
            "} else {",
            *second,
            "}",
        ]
    body = [
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%isp = arith.cmpi eq, %pid, %c0 : i32",
        "scf.if %isp {",
        "tt.store %data, %c1 : !tt.ptr<i32>",
        "gpu.barrier",
        f"%pub = tt.atomic_rmw exch, {writer_sem}, gpu, %flag, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32",
        "} else {",
        *waits,
        "gpu.barrier",
        "%v = tt.load %data : !tt.ptr<i32>",
        "%op = tt.addptr %out, %pid : !tt.ptr<i32>, i32",
        "tt.store %op, %v : !tt.ptr<i32>",
        "}",
        "tt.return",
    ]
    text = (
        "module {\n"
        "tt.func public @two_awaits(%flag: !tt.ptr<i32>, "
        "%data: !tt.ptr<i32>, %out: !tt.ptr<i32>) attributes {noinline = false} {\n"
        + "\n".join(body)
        + "\n}\n}\n"
    )
    tensors = {
        "flag": GlobalTensor(data_ptr=0x2000, elem_size=4, numel=1, init_values=(0,)),
        "data": GlobalTensor(data_ptr=0x3000, elem_size=4, numel=1),
        "out": GlobalTensor(data_ptr=0x4000, elem_size=4, numel=3),
    }
    enc = encode_graph(parse_ttir(text, multipath=True), {}, tensors, multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records,
        grid=(3, 1, 1),
        arange_dict=enc.arange_dict,
        fence_order=True,
        fence_seqs=enc.fence_seqs,
        extra_assumptions=enc.assumptions,
    )
    return enc, solver


@pytest.mark.parametrize("sequential", [False, True], ids=["exclusive", "sequential"])
def test_two_awaits_preserve_feasibility_and_publication(sequential):
    enc, solver = _two_awaits(sequential=sequential)
    assert enc.assumes_termination
    assert solver.launch_premises
    assert solver._base_solver().check() == sat
    assert solver.check_feasibility()
    assert solver.find_races() == []


@pytest.mark.parametrize("sequential", [False, True], ids=["exclusive", "sequential"])
def test_two_awaits_without_release_still_report(sequential):
    _, solver = _two_awaits(sequential=sequential, writer_sem="relaxed")
    assert solver.check_feasibility()
    assert any(
        report.first_record.tensor_name == report.second_record.tensor_name == "data"
        for report in solver.find_races()
    )


def _rmw(addr, old, value, event_id):
    return AccessEventRecord(
        op_type=AtomicRMW,
        access_mode="read",
        addr_expr=addr,
        active=True,
        is_atomic=True,
        atomic_kind="rmw",
        sem="relaxed",
        scope="gpu",
        old_value=old,
        rmw_op="xchg",
        rmw_operand=value,
        event_id=event_id,
        program_seq=event_id,
        elem_size=4,
        copy_local_vars=(old,),
    )


@pytest.mark.parametrize(
    "dependency",
    [
        "value",
        "operand",
        "active",
        "legacy_mask",
        "legacy_loop",
        "loop_control",
        "cas_compare",
        "cas_new",
    ],
)
def test_real_read_from_dependency_cycle_remains_forbidden(dependency):
    """Two locations each read the other instance's dependent publication.

    Without the causality family, this fixed cyclic RF assignment is feasible;
    retaining the real dependency makes it inconsistent. This catches dropping
    all masks, RMW operands or CAS compares while separating global premises.
    """
    pid = SymbolicExpr.PID0
    observed, exchanged = Int("observed"), Int("exchanged")
    reader = _rmw(If(pid == 0, 0x2000, 0x3000), observed, 0, 0)
    reader.rmw_op = "add"
    writer = _rmw(If(pid == 0, 0x3000, 0x2000), exchanged, 1, 1)
    if dependency == "value":
        writer.rmw_operand = observed
    elif dependency == "operand":
        writer.rmw_op = "add"
        writer.rmw_operand = observed
    elif dependency == "active":
        writer.active = observed == 1
    elif dependency == "legacy_mask":
        writer.local_constraints = (observed == 1,)
    elif dependency == "legacy_loop":
        writer.premises = (observed == 1,)
    elif dependency == "loop_control":
        writer.local_constraints = writer.causal_constraints = (observed == 1,)
    else:
        writer = replace(
            writer,
            op_type=AtomicCas,
            atomic_kind="cas",
            rmw_op=None,
            rmw_operand=None,
            cas_cmp_value=observed if dependency == "cas_compare" else IntVal(0),
            cas_new_value=observed if dependency == "cas_new" else IntVal(1),
        )
    solver = TwoCopySymbolicHBSolver(
        [reader, writer], grid=(2, 1, 1), fence_order=True, arange_dict={}
    )
    a_read, a_write, b_read, b_write = solver.events
    cyclic_rf = (
        a_read.pid[0] == 0,
        b_read.pid[0] == 1,
        solver.rf_source[(a_write.idx, b_read.idx)],
        solver.rf_source[(b_write.idx, a_read.idx)],
    )
    base = solver._base_solver()
    base.add(*cyclic_rf)
    assert base.check() == unsat

    # Independent control: all other base constraints admit these choices.
    without_causality = Solver()
    without_causality.add(solver.grid_constraints)
    without_causality.add(*solver.rf_constraints)
    without_causality.add(*solver.atomic_coherence_constraints)
    without_causality.add(*solver.counting_constraints)
    without_causality.add(*[Not(solver.hb[i][i]) for i in range(len(solver.events))])
    without_causality.add(*cyclic_rf)
    assert without_causality.check() == sat


def test_explicit_domain_conditions_still_gate_reads_without_false_dependencies():
    """Domain constraints remain effective while causing no operational cycle."""
    a, b = Int("domain_a"), Int("domain_b")
    records = [
        _rmw(IntVal(0x2000), a, 1, 0),
        _rmw(IntVal(0x3000), b, 1, 1),
    ]
    for record in records:
        record.local_constraints = (a == 1, b == 1)
        record.premises = (a == 1, b == 1)
        record.causal_constraints = ()
        record.copy_local_vars = (a, b)
    solver = TwoCopySymbolicHBSolver(records, grid=(1, 1, 1), arange_dict={})
    assert solver.check_feasibility()
    for event in solver.events:
        query = solver._base_solver()
        query.add(event.reads, event.old_value != 1)
        assert query.check() == unsat
    # Deactivating records does not evade the separately asserted premises.
    query = solver._base_solver()
    query.add(*solver.launch_premises, solver.events[0].old_value != 1)
    assert query.check() == unsat


@pytest.mark.parametrize("initial", [0, 1])
def test_relaxed_await_publication_cycle_needs_an_initial_source(initial):
    """A relaxed await cannot bootstrap the other instance's later publication.

    Both instances wait for their own flag before setting the peer's flag.
    With zero initialization, only the invented causal cycle could justify
    termination. Initial ones provide independent roots and must remain feasible.
    """
    text = """module {
tt.func public @cycle(%flags: !tt.ptr<i32>) attributes {noinline = false} {
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%true = arith.constant true
%pid = tt.get_program_id x : i32
%peer = arith.subi %c1, %pid : i32
%mine = tt.addptr %flags, %pid : !tt.ptr<i32>, i32
%other = tt.addptr %flags, %peer : !tt.ptr<i32>, i32
scf.while : () -> () {
%poll = tt.atomic_rmw add, relaxed, gpu, %mine, %c0, %true : (!tt.ptr<i32>, i32, i1) -> i32
%again = arith.cmpi ne, %poll, %c1 : i32
scf.condition(%again)
} do {
scf.yield
}
%publish = tt.atomic_rmw exch, relaxed, gpu, %other, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
tt.return
}
}
"""
    tensors = {
        "flags": GlobalTensor(
            data_ptr=0x2000, elem_size=4, numel=2, init_values=(initial, initial)
        ),
    }
    enc = encode_graph(parse_ttir(text, multipath=True), {}, tensors, multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=(2, 1, 1), arange_dict=enc.arange_dict, fence_order=True
    )
    assert enc.assumes_termination
    assert solver._base_solver().check() == sat
    assert solver.check_feasibility() is bool(initial)


@pytest.mark.parametrize(
    "join", [False, True], ids=["branch-publications", "join-publication"]
)
@pytest.mark.parametrize("grid,initial", [(2, (1, 1)), (1, (1, 0))])
def test_await_control_respects_branch_execution_and_join(join, grid, initial):
    """The unselected consumer need not terminate before a branch or join."""
    publish = (
        "%pub = tt.atomic_rmw exch, relaxed, gpu, %dest, %c1, %true : "
        "(!tt.ptr<i32>, i32, i1) -> i32"
    )
    first, second = _spin("add", "a"), _spin("or", "b")
    # Only the else branch reads the second flag, which is allowed to remain
    # zero when that branch cannot execute on this launch.
    second = [line.replace("%flag,", "%flag1,") for line in second]
    body = [
        "%true = arith.constant true",
        "%c0 = arith.constant 0 : i32",
        "%c1 = arith.constant 1 : i32",
        "%pid = tt.get_program_id x : i32",
        "%zero = arith.cmpi eq, %pid, %c0 : i32",
        "%flag1 = tt.addptr %flag, %c1 : !tt.ptr<i32>, i32",
        "%dest = tt.addptr %out, %pid : !tt.ptr<i32>, i32",
        "scf.if %zero {",
        *first,
        *([] if join else [publish.replace("%pub", "%pub0")]),
        "} else {",
        *second,
        *([] if join else [publish.replace("%pub", "%pub1")]),
        "}",
        *([publish] if join else []),
        "tt.return",
    ]
    text = (
        "module {\n"
        "tt.func public @branches(%flag: !tt.ptr<i32>, %out: !tt.ptr<i32>) "
        "attributes {noinline = false} {\n" + "\n".join(body) + "\n}\n}\n"
    )
    tensors = {
        "flag": GlobalTensor(
            data_ptr=0x2000, elem_size=4, numel=2, init_values=initial
        ),
        "out": GlobalTensor(data_ptr=0x3000, elem_size=4, numel=2, init_values=(0, 0)),
    }
    enc = encode_graph(parse_ttir(text, multipath=True), {}, tensors, multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=(grid, 1, 1), arange_dict=enc.arange_dict, fence_order=True
    )
    assert solver.check_feasibility()
    assert solver.find_races() == []


@pytest.mark.parametrize("initial", [0, 1])
def test_await_expected_observation_cannot_come_from_a_future_publication(initial):
    """Both operands of an await comparison control its later publication."""
    text = """module {
tt.func public @expected(%flags: !tt.ptr<i32>, %ready: !tt.ptr<i32>) attributes {noinline = false} {
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%true = arith.constant true
%pid = tt.get_program_id x : i32
%peer = arith.subi %c1, %pid : i32
%mine = tt.addptr %flags, %pid : !tt.ptr<i32>, i32
%other = tt.addptr %flags, %peer : !tt.ptr<i32>, i32
%expected = tt.atomic_rmw add, relaxed, gpu, %other, %c0, %true : (!tt.ptr<i32>, i32, i1) -> i32
scf.while : () -> () {
%poll = tt.atomic_rmw add, relaxed, gpu, %ready, %c0, %true : (!tt.ptr<i32>, i32, i1) -> i32
%again = arith.cmpi ne, %poll, %expected : i32
scf.condition(%again)
} do {
scf.yield
}
%publish = tt.atomic_rmw exch, relaxed, gpu, %mine, %c1, %true : (!tt.ptr<i32>, i32, i1) -> i32
tt.return
}
}
"""
    tensors = {
        "flags": GlobalTensor(
            data_ptr=0x2000, elem_size=4, numel=2, init_values=(initial, initial)
        ),
        "ready": GlobalTensor(data_ptr=0x3000, elem_size=4, numel=1, init_values=(1,)),
    }
    enc = encode_graph(parse_ttir(text, multipath=True), {}, tensors, multipath=True)
    solver = TwoCopySymbolicHBSolver(
        enc.records, grid=(2, 1, 1), arange_dict=enc.arange_dict, fence_order=True
    )
    assert solver._base_solver().check() == sat
    assert solver.check_feasibility() is bool(initial)
