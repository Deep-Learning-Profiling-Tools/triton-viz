import triton.language as tl

from triton_viz.clients.symbolic_engine import SymbolicExpr


def test_bool_inferred_as_int1():
    assert SymbolicExpr.from_value(True).dtype is tl.int1
    assert SymbolicExpr.from_value(False).dtype is tl.int1


def test_constexpr_bool_inferred_as_int1():
    assert SymbolicExpr.from_value(tl.constexpr(True)).dtype is tl.int1
    assert SymbolicExpr.from_value(tl.constexpr(False)).dtype is tl.int1
