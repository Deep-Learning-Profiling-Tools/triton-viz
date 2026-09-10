"""Unit tests for atomic RMW/CAS parsing in the shared TTIR reader."""

from pathlib import Path

import pytest

from triton_viz.clients.common.ttir_reader import (
    Cmp,
    Const,
    Pid,
    UnsupportedTTIR,
    parse_ttir,
)
from triton_viz.clients.sanitizer.compiled.oob import (
    LaunchContext,
    TensorMeta,
    check_graph,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ttgir"


def _read(name):
    return (GOLDEN / name).read_text()


def _contains(term, cls) -> bool:
    """Recursively search a Term tree for a node type."""
    if isinstance(term, cls):
        return True
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None and _contains(sub, cls):
            return True
    return False


def _meta(numel, elem_bits=32, ptr=1000):
    return TensorMeta(numel=numel, elem_bits=elem_bits, data_ptr=ptr, contiguous=True)


@pytest.mark.parametrize("cap", ["sm80", "sm90"])
def test_atomic_rmw_events(cap):
    g = parse_ttir(_read(f"atomic_{cap}.ttir"))
    assert g.kernel_name == "atomic_kernel"
    assert [a.kind for a in g.accesses] == ["load", "atomic_rmw", "atomic_rmw", "store"]

    fadd, exch = g.accesses[1], g.accesses[2]
    assert fadd.base_param == "out_ptr"
    assert fadd.atomic is not None
    assert fadd.atomic.rmw_op == "fadd"
    assert fadd.atomic.sem == "acq_rel"
    assert fadd.atomic.scope == "gpu"
    assert isinstance(fadd.mask, Cmp)  # offs < n_elements
    assert _contains(fadd.offset, Pid)
    assert fadd.is_read and fadd.is_write

    # Unmasked tl.atomic_xchg: the printer still emits a mask operand — a
    # dense<true> constant — which parses to Const(1).
    assert exch.atomic is not None and exch.atomic.rmw_op == "exch"
    assert exch.mask == Const(1)

    load, store = g.accesses[0], g.accesses[3]
    assert load.is_read and not load.is_write and load.atomic is None
    assert store.is_write and not store.is_read and store.atomic is None


@pytest.mark.parametrize("cap", ["sm80", "sm90"])
def test_atomic_cas_scalar(cap):
    g = parse_ttir(_read(f"cas_{cap}.ttir"))
    assert [a.kind for a in g.accesses] == ["atomic_cas", "store"]
    cas = g.accesses[0]
    assert cas.base_param == "lock_ptr"
    assert cas.mask is None  # CAS has no mask operand
    assert cas.atomic is not None and cas.atomic.rmw_op is None
    assert cas.atomic.sem == "acq_rel" and cas.atomic.scope == "gpu"
    assert cas.offset == Const(0)
    assert cas.is_read and cas.is_write


@pytest.mark.parametrize("cap", ["sm80", "sm90"])
def test_float_atomic_max_fails_closed(cap):
    # tl.atomic_max on f32 lowers to a sign-trick dance: the pointer is
    # tt.bitcast to i32 and the RMW masks derive from the loaded value.
    # Both are outside the v1 vocabulary — the parse must fail closed.
    with pytest.raises(UnsupportedTTIR):
        parse_ttir(_read(f"atomic_fmax_{cap}.ttir"))


def test_atomics_in_bounds_proof():
    g = parse_ttir(_read("atomic_sm80.ttir"))
    ctx = LaunchContext(
        grid=(4, 1, 1),  # offsets reach 4*256-1 = 1023
        params={"n_elements": 1024},
        tensors={n: _meta(1024) for n in ("x_ptr", "out_ptr")},
    )
    assert check_graph(g, ctx) == []


def test_unmasked_atomic_tail_is_oob():
    """The masked accesses are guarded by offs < n_elements, but the unmasked
    atomic_xchg touches the full last block — only it may go out of bounds."""
    g = parse_ttir(_read("atomic_sm80.ttir"))
    ctx = LaunchContext(
        grid=(4, 1, 1),
        params={"n_elements": 1000},
        tensors={n: _meta(1000) for n in ("x_ptr", "out_ptr")},
    )
    v = check_graph(g, ctx)
    assert [r.kind for r in v] == ["atomic_rmw"]
    assert v[0].violation_offset >= 1000
