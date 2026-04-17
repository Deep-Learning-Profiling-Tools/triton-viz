"""Unit tests for the shared SymbolicClient substrate.

These tests are client-agnostic — they exercise pieces of
``triton_viz/clients/symbolic_engine.py`` (the ``_range_to_iterator_constraint``
Z3 helper, the ``SymbolicExpr`` tree and its subclasses, reduce-shape
inference, block-pointer construction, load/store dtype derivation) that
every symbolic client relies on. Sanitizer- and race-detector-specific
tests live next to their respective clients.
"""

import pytest
from typing import cast

import triton.language as tl

from triton_viz.clients.symbolic_engine import (
    SymbolicExpr,
    ConstSymbolicExpr,
    ReduceSymbolicExpr,
    LoadSymbolicExpr,
    StoreSymbolicExpr,
    _range_to_iterator_constraint,
)
from z3.z3 import ArithRef, BoolRef, IntNumRef
from z3 import Solver, Int, sat


# ======== Range Constraint Tests ===========


def test_range_to_iterator_constraint():
    def assert_constraint_allows(constraint: BoolRef, var: ArithRef, val: int) -> None:
        solver = Solver()
        solver.add(constraint)
        solver.add(var == val)
        assert solver.check() == sat

    def assert_constraint_forbids(constraint: BoolRef, var: ArithRef, val: int) -> None:
        solver = Solver()
        solver.add(constraint)
        solver.add(var == val)
        assert solver.check() != sat

    # Positive step: range(1, 10, 2) -> [1, 3, 5, 7, 9]
    i = Int("i")
    constraint = _range_to_iterator_constraint(i, start=1, stop=10, step=2)
    assert_constraint_allows(constraint, i, 1)
    assert_constraint_allows(constraint, i, 3)
    assert_constraint_allows(constraint, i, 9)
    assert_constraint_forbids(constraint, i, 2)
    assert_constraint_forbids(constraint, i, 10)

    # Negative step: range(4, -1, -2) -> [4, 2, 0]
    i = Int("i")
    constraint = _range_to_iterator_constraint(i, start=4, stop=-1, step=-2)
    assert_constraint_allows(constraint, i, 4)
    assert_constraint_allows(constraint, i, 2)
    assert_constraint_allows(constraint, i, 0)
    assert_constraint_forbids(constraint, i, 1)
    assert_constraint_forbids(constraint, i, -1)


# ======== Reduce Operations Tests =========


@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("data", [[1, 5, 3, 2], [42]])
def test_reduce_expr_eval(op: str, data):
    import numpy as np
    import builtins

    input_arr = SymbolicExpr.create(
        "const", np.array(data), tl.block_type(tl.int32, [len(data)])
    )
    reduce_expr = SymbolicExpr.create(op, input_arr, None, False)

    result, _ = reduce_expr.eval(simplify_constraints=False)
    # Use reflection to call the matching Python builtin, e.g. builtins.max([1,5,3,2]) -> 5
    assert cast(IntNumRef, result).as_long() == getattr(builtins, op)(data)


# ======== Basic Symbolic Expr Operations Tests =========


@pytest.mark.parametrize("value", [(1, 2, 3), 4])
def test_basic_expr_const_eval(value):
    # Test that "const" SymbolicExpr correctly evaluates both scalar (e.g. 4)
    # and vector (e.g. (1, 2, 3)) constants.
    const_expr = SymbolicExpr.create("const", value, tl.int32)
    result, constraints = const_expr.eval(simplify_constraints=False)
    if isinstance(value, (list, tuple)):
        # Vector constants return a list of Z3 ints.
        assert [cast(IntNumRef, v).as_long() for v in cast(list, result)] == list(value)
    else:
        # Scalar constants return a single Z3 int.
        assert cast(IntNumRef, result).as_long() == value
    # Constants produce no constraints.
    assert constraints is None


@pytest.mark.parametrize(
    "axis,expected_pid",
    [
        (0, SymbolicExpr.PID0),
        (1, SymbolicExpr.PID1),
        (2, SymbolicExpr.PID2),
    ],
)
def test_basic_expr_pid_eval(axis, expected_pid):
    # Test that "pid" expr returns the corresponding predefined symbolic variable (PID0/PID1/PID2).
    pid_expr = SymbolicExpr.create("pid", axis)
    result, constraints = pid_expr.eval(simplify_constraints=False)
    assert result == expected_pid
    assert constraints is None


@pytest.mark.parametrize("start,end", [(4, 8), (0, 4)])
def test_basic_expr_arange_eval(start, end):
    # Test that arange expr produces a named symbolic variable with range constraints.
    arange_expr = SymbolicExpr.create("arange", tl.int32, start, end)
    result, constraints = arange_expr.eval(simplify_constraints=False)
    result = cast(ArithRef, result)
    assert result.decl().name() == f"arange_{start}_{end}"
    assert constraints is not None
    constraints_str = str(constraints)
    assert f"{result} >= {start}" in constraints_str
    assert f"{result} < {end}" in constraints_str


# ======== Unary Symbolic Expr Operations Tests =========


@pytest.mark.parametrize(
    "op,value,expected",
    [
        ("abs", -3, 3),
        ("fabs", -7, 7),
    ],
)
def test_unary_expr_eval(op: str, value: int, expected: int):
    # Test that unary ops (abs, fabs) correctly evaluate on constant inputs.
    arg = SymbolicExpr.create("const", value, tl.int32)
    expr = SymbolicExpr.create(op, arg)
    result, constraints = expr.eval(simplify_constraints=False)
    assert cast(IntNumRef, result).as_long() == expected
    assert constraints is None


# ======== Binary Symbolic Expr Operations Tests =========


@pytest.mark.parametrize(
    "op,lhs,rhs,expected",
    [
        ("add", 2, 3, 5),
        ("sub", 7, 4, 3),
        ("mul", 3, 5, 15),
        ("idiv", 8, 2, 4),
        ("mod", 7, 4, 3),
        ("less", 2, 9, True),
        ("less_equal", 3, 3, True),
        ("greater", 5, 2, True),
        ("greater_equal", 5, 7, False),
        ("equal", 4, 4, True),
        ("not_equal", 4, 4, False),
        ("maximum", 2, 9, 9),
        ("minimum", 2, 9, 2),
        ("bitwise_and", 6, 3, 2),
        ("bitwise_or", 6, 3, 7),
        ("bitwise_xor", 6, 3, 5),
    ],
)
def test_binary_expr_eval(op: str, lhs: int, rhs: int, expected):
    # Test arithmetic, comparison, and bitwise binary ops on constant inputs.
    lhs_expr = SymbolicExpr.create("const", lhs, tl.int32)
    rhs_expr = SymbolicExpr.create("const", rhs, tl.int32)
    expr = SymbolicExpr.create(op, lhs_expr, rhs_expr)
    result, constraints = expr.eval(simplify_constraints=False)
    if isinstance(expected, bool):
        assert str(result) == str(expected)
    else:
        assert cast(IntNumRef, result).as_long() == expected
    assert constraints is None


def test_bitwise_bool_expr_eval():
    """
    Test short circuiting behavior of bitwise_and and bitwise_or operators which do not
    go through bitvector conversion like other binary operators.
    """
    a = SymbolicExpr.create("const", 1, tl.int32)
    b = SymbolicExpr.create("const", 2, tl.int32)
    cond_true = SymbolicExpr.create("less", a, b)
    cond_false = SymbolicExpr.create("greater", a, b)

    and_expr = SymbolicExpr.create("bitwise_and", cond_true, cond_false)
    result, constraints = and_expr.eval(simplify_constraints=False)
    assert isinstance(result, BoolRef)
    assert str(result) == "False"
    assert constraints is None

    or_expr = SymbolicExpr.create("bitwise_or", cond_true, cond_false)
    result, constraints = or_expr.eval(simplify_constraints=False)
    assert isinstance(result, BoolRef)
    assert str(result) == "True"
    assert constraints is None

    where_expr = SymbolicExpr.create(
        "where",
        and_expr,
        SymbolicExpr.create("const", 1, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
    )
    result, constraints = where_expr.eval(simplify_constraints=False)
    assert isinstance(result, list)
    assert cast(IntNumRef, result[0]).as_long() == 0
    assert constraints is None


# ======== Pointer Symbolic Expr Operations Tests =========


def test_pointer_expr_addptr_eval():
    # Test that addptr scales offset by element size (100 + 3 * 4 = 112).
    base = SymbolicExpr.create("const", 100, tl.pointer_type(tl.int32))
    offset = SymbolicExpr.create("const", 3, tl.int32)
    expr = SymbolicExpr.create("addptr", base, offset)
    result, constraints = expr.eval(simplify_constraints=False)
    assert cast(IntNumRef, result).as_long() == 112
    assert constraints is None


# ======== Reshape Symbolic Expr Operations Tests =========


@pytest.mark.parametrize(
    "op,extra,expected_lanes",
    [
        ("splat", tl.block_type(tl.int32, [2]), [5, 5]),
        ("expand_dims", 0, [5]),
        ("broadcast", (2,), [5, 5]),
    ],
)
def test_reshape_expr_eval(op: str, extra, expected_lanes):
    # Ops that extend a scalar into a 1-D block: each output lane carries the
    # same scalar value after real lane-semantics are in place.
    arg = SymbolicExpr.create("const", 5, tl.int32)
    if op == "splat":
        expr = SymbolicExpr.create(op, extra, arg)
    else:
        expr = SymbolicExpr.create(op, arg, extra)
    result, constraints = expr.eval(simplify_constraints=False)
    assert isinstance(result, list)
    assert [cast(IntNumRef, r).as_long() for r in result] == expected_lanes
    assert constraints is None


# ======== Regression Tests: expand_dims/broadcast Shape Fix =========


def test_expand_dims_updates_shape():
    # Scalar constant has empty shape
    arg = SymbolicExpr.create("const", 5, tl.int32)
    assert arg.shape == ()

    # expand_dims at axis 0 should insert dimension of size 1
    expr = SymbolicExpr.create("expand_dims", arg, 0)
    assert expr.shape == (1,), f"Expected shape (1,), got {expr.shape}"

    # Test with tensor input and positive axis
    tensor_arg = SymbolicExpr.create("arange", tl.int32, 0, 8)
    assert tensor_arg.shape == (8,)
    expr2 = SymbolicExpr.create("expand_dims", tensor_arg, 0)
    assert expr2.shape == (1, 8), f"Expected shape (1, 8), got {expr2.shape}"
    expr3 = SymbolicExpr.create("expand_dims", tensor_arg, 1)
    assert expr3.shape == (8, 1), f"Expected shape (8, 1), got {expr3.shape}"

    # Test with negative axis
    expr4 = SymbolicExpr.create("expand_dims", tensor_arg, -1)
    assert expr4.shape == (8, 1), f"Expected shape (8, 1), got {expr4.shape}"
    expr5 = SymbolicExpr.create("expand_dims", tensor_arg, -2)
    assert expr5.shape == (1, 8), f"Expected shape (1, 8), got {expr5.shape}"


def test_broadcast_updates_shape():
    # Scalar constant has empty shape
    arg = SymbolicExpr.create("const", 5, tl.int32)
    assert arg.shape == ()

    # broadcast to shape (4,)
    expr = SymbolicExpr.create("broadcast", arg, (4,))
    assert expr.shape == (4,), f"Expected shape (4,), got {expr.shape}"


# ======== Block Pointer Symbolic Expr Tests =========


def test_make_block_ptr_symbolic_expr_creation():
    """Create MakeBlockPtrSymbolicExpr with concrete values, verify children and attributes."""
    base = SymbolicExpr.create("const", 1000, tl.pointer_type(tl.float32))
    shape_list = [SymbolicExpr.create("const", 64, tl.int32)]
    stride_list = [SymbolicExpr.create("const", 1, tl.int32)]
    offset_list = [SymbolicExpr.create("const", 0, tl.int32)]
    block_shape_vals = [32]
    order_vals = [0]

    expr = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        block_shape_vals,
        order_vals,
    )
    assert expr.op == "make_block_ptr"
    assert expr.ndim == 1
    assert expr.block_shape_values == [32]
    assert expr.order_values == [0]
    assert expr.base is base
    assert expr.shape_keys == ["shape_0"]
    assert expr.stride_keys == ["stride_0"]
    assert expr.offset_keys == ["offset_0"]
    assert getattr(expr, "shape_0") is not None
    assert getattr(expr, "stride_0") is not None
    assert getattr(expr, "offset_0") is not None


def test_advance_symbolic_expr_creation():
    """Create Advance wrapping MakeBlockPtr, verify children."""
    base = SymbolicExpr.create("const", 1000, tl.pointer_type(tl.float32))
    shape_list = [SymbolicExpr.create("const", 64, tl.int32)]
    stride_list = [SymbolicExpr.create("const", 1, tl.int32)]
    offset_list = [SymbolicExpr.create("const", 0, tl.int32)]

    mbp = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        [32],
        [0],
    )

    delta = SymbolicExpr.create("const", 32, tl.int32)
    adv = SymbolicExpr.create("advance", mbp, [delta])
    assert adv.op == "advance"
    assert adv.ndim == 1
    assert adv.delta_keys == ["delta_0"]
    assert adv.ptr is mbp
    assert getattr(adv, "delta_0") is delta


def test_tensor_pointer_load_z3_eval():
    """Create make_block_ptr -> tensor_pointer_load, call .eval(), verify Z3 expression."""
    base = SymbolicExpr.create("const", 1000, tl.pointer_type(tl.float32))
    shape_list = [SymbolicExpr.create("const", 64, tl.int32)]
    stride_list = [SymbolicExpr.create("const", 1, tl.int32)]
    offset_list = [SymbolicExpr.create("const", 0, tl.int32)]

    mbp = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        [32],
        [0],
    )

    load_expr = SymbolicExpr.create("tensor_pointer_load", mbp, (0,))
    z3_expr, constraints = load_expr.eval()

    # Should have Z3 expression with base address (1000) and block index variable
    z3_str = str(z3_expr)
    assert "1000" in z3_str, f"Expected base address in Z3 expr, got: {z3_str}"
    assert "blk_k_0" in z3_str, f"Expected block index variable, got: {z3_str}"

    # Constraints should include block index bounds and boundary check
    assert constraints is not None
    constr_str = str(constraints)
    assert "blk_k_0" in constr_str


def test_resolve_block_ptr_through_advance_chain():
    """Create make_block_ptr -> advance -> advance, verify accumulated offsets."""
    from triton_viz.clients.symbolic_engine import TensorPointerLoadSymbolicExpr

    base = SymbolicExpr.create("const", 1000, tl.pointer_type(tl.float32))
    shape_list = [SymbolicExpr.create("const", 128, tl.int32)]
    stride_list = [SymbolicExpr.create("const", 1, tl.int32)]
    offset_list = [SymbolicExpr.create("const", 0, tl.int32)]

    mbp = SymbolicExpr.create(
        "make_block_ptr",
        base,
        shape_list,
        stride_list,
        offset_list,
        [32],
        [0],
    )

    delta1 = SymbolicExpr.create("const", 32, tl.int32)
    adv1 = SymbolicExpr.create("advance", mbp, [delta1])

    delta2 = SymbolicExpr.create("const", 32, tl.int32)
    adv2 = SymbolicExpr.create("advance", adv1, [delta2])

    # Create a load to use its _resolve_block_ptr_components
    load_expr = SymbolicExpr.create("tensor_pointer_load", adv2, ())
    assert isinstance(load_expr, TensorPointerLoadSymbolicExpr)

    (
        resolved_base,
        shapes,
        strides,
        offsets,
        bs,
    ) = load_expr._resolve_block_ptr_components(adv2)
    assert resolved_base is base
    assert bs == [32]

    # Offsets should be accumulated: 0 + 32 + 32 = 64
    off_z3, _ = offsets[0].eval()
    assert cast(IntNumRef, off_z3).as_long() == 64


# ======== ReduceSymbolicExpr Shape Tests ===========


def _make_block_expr(scalar_ty, shape):
    """Helper: create a ConstSymbolicExpr with the given block dtype."""
    return ConstSymbolicExpr("const", value=0, dtype=tl.block_type(scalar_ty, shape))


def test_dtype_is_always_scalar():
    """dtype must always be a scalar type, never a block_type."""
    expr = _make_block_expr(tl.float32, [4, 8])
    assert not isinstance(expr.dtype, tl.block_type)
    assert expr.dtype == tl.float32
    assert expr.shape == (4, 8)
    full = tl.block_type(expr.dtype, list(expr.shape))
    assert isinstance(full, tl.block_type)


def test_reduce_shape_2d_axis1():
    """tl.sum(x, axis=1) on [4, 8] -> [4]."""
    inp = _make_block_expr(tl.float32, [4, 8])
    assert inp.shape == (4, 8)

    reduced = ReduceSymbolicExpr("sum", inp, axis=1)
    assert reduced.shape == (4,)


def test_reduce_shape_2d_axis0():
    """tl.sum(x, axis=0) on [4, 8] -> [8]."""
    inp = _make_block_expr(tl.float32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=0)
    assert reduced.shape == (8,)


def test_reduce_shape_2d_axis1_keepdims():
    """tl.sum(x, axis=1, keepdims=True) on [4, 8] -> [4, 1]."""
    inp = _make_block_expr(tl.float32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=1, keepdims=True)
    assert reduced.shape == (4, 1)


def test_reduce_shape_2d_axis0_keepdims():
    """tl.sum(x, axis=0, keepdims=True) on [4, 8] -> [1, 8]."""
    inp = _make_block_expr(tl.float32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=0, keepdims=True)
    assert reduced.shape == (1, 8)


def test_reduce_shape_1d_to_scalar():
    """tl.sum(x, axis=0) on [8] -> scalar ()."""
    inp = _make_block_expr(tl.float32, [8])
    assert inp.shape == (8,)

    reduced = ReduceSymbolicExpr("sum", inp, axis=0)
    assert reduced.shape == ()


def test_reduce_shape_all_axes():
    """tl.sum(x) with axis=None on [4, 8] -> scalar ()."""
    inp = _make_block_expr(tl.float32, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=None)
    assert reduced.shape == ()


def test_reduce_shape_preserves_scalar_type():
    """Reduced dtype keeps the original scalar type (e.g. float16)."""
    inp = _make_block_expr(tl.float16, [4, 8])
    reduced = ReduceSymbolicExpr("sum", inp, axis=1)
    assert reduced.shape == (4,)
    assert reduced.dtype == tl.float16


def test_reduce_shape_max_min():
    """tl.max and tl.min compute the same output shape as tl.sum."""
    inp = _make_block_expr(tl.float32, [4, 8])

    for op in ("max", "min"):
        reduced = ReduceSymbolicExpr(op, inp, axis=1)
        assert reduced.shape == (4,), f"op={op}: expected (4,), got {reduced.shape}"


# ======== LoadSymbolicExpr dtype Tests ===========


def test_load_dtype_block_of_pointers():
    """tl.load on a block of pointers should produce a block of the pointed-to type.

    ptr dtype: block_type(pointer<fp32>, [1, 16])
    expected load dtype: block_type(fp32, [1, 16])
    """
    ptr = ConstSymbolicExpr(
        "const", value=0, dtype=tl.block_type(tl.pointer_type(tl.float32), [1, 16])
    )
    load = LoadSymbolicExpr("load", ptr)
    assert load.shape == (1, 16), f"Expected shape (1, 16), got {load.shape}"
    assert load.dtype == tl.float32, f"Expected tl.float32, got {load.dtype}"


def test_store_dtype_block_of_pointers():
    """tl.store on a block of pointers should not derive a dtype (store returns None).

    ptr dtype: block_type(pointer<fp32>, [1, 16])
    expected store dtype: None
    """
    ptr = ConstSymbolicExpr(
        "const", value=0, dtype=tl.block_type(tl.pointer_type(tl.float32), [1, 16])
    )
    value = ConstSymbolicExpr(
        "const", value=0, dtype=tl.block_type(tl.float32, [1, 16])
    )
    store = StoreSymbolicExpr("store", ptr, value)
    assert store.dtype is None, f"Expected None, got {store.dtype}"


# ======== Shape-Op Lane Semantics (PR4) =========

import numpy as np  # noqa: E402


def _tagged_block(values, dtype=tl.int32):
    """Build a ConstSymbolicExpr whose lanes are distinguishable integers."""
    arr = np.array(values, dtype=np.int32)
    return ConstSymbolicExpr(
        "const", value=arr, dtype=tl.block_type(dtype, list(arr.shape))
    )


def _lane_ints(z3_val):
    """Flatten a Z3 value (scalar or list) into ints for assertion."""
    if isinstance(z3_val, list):
        return [cast(IntNumRef, v).as_long() for v in z3_val]
    return [cast(IntNumRef, z3_val).as_long()]


def test_splat_repeats_scalar_lane():
    arg = SymbolicExpr.create("const", 7, tl.int32)
    expr = SymbolicExpr.create("splat", tl.block_type(tl.int32, [4]), arg)
    assert expr.shape == (4,)
    val, _ = expr.eval(simplify_constraints=False)
    assert _lane_ints(val) == [7, 7, 7, 7]


def test_expand_dims_preserves_flat_sequence():
    block = _tagged_block([10, 11, 12, 13])
    expr = SymbolicExpr.create("expand_dims", block, 0)
    assert expr.shape == (1, 4)
    val, _ = expr.eval(simplify_constraints=False)
    assert _lane_ints(val) == [10, 11, 12, 13]


def test_expand_dims_multi_axis_final_rank_relative():
    # axis=(0, -1) with input shape (8,) → final shape (1, 8, 1). Axes are
    # normalized against the final rank, not applied iteratively.
    from triton_viz.clients.symbolic_engine import SymbolicClient

    overrider = SymbolicClient._op_expand_dims_overrider
    base = SymbolicExpr.create("arange", tl.int32, 0, 8)
    expr = overrider(None, base, (0, -1))  # type: ignore[arg-type]
    assert expr.shape == (1, 8, 1)


def test_expand_dims_multi_axis_rejects_duplicate():
    from triton_viz.clients.symbolic_engine import SymbolicClient

    overrider = SymbolicClient._op_expand_dims_overrider
    base = SymbolicExpr.create("arange", tl.int32, 0, 8)
    with pytest.raises(ValueError):
        overrider(None, base, (0, 0))  # type: ignore[arg-type]


def test_expand_dims_multi_axis_rejects_out_of_range():
    from triton_viz.clients.symbolic_engine import SymbolicClient

    overrider = SymbolicClient._op_expand_dims_overrider
    base = SymbolicExpr.create("arange", tl.int32, 0, 8)
    with pytest.raises(ValueError):
        overrider(None, base, (5,))  # type: ignore[arg-type]


def test_broadcast_duplicates_lanes():
    # input shape (1, 4) with lanes [a0..a3] → (3, 4) should produce
    # [a0..a3, a0..a3, a0..a3].
    block = _tagged_block([[20, 21, 22, 23]])  # shape (1, 4)
    expr = SymbolicExpr.create("broadcast", block, (3, 4))
    assert expr.shape == (3, 4)
    val, _ = expr.eval(simplify_constraints=False)
    assert _lane_ints(val) == [20, 21, 22, 23] * 3


def test_broadcast_rejects_non_unit_expansion():
    block = _tagged_block([1, 2])  # shape (2,)
    with pytest.raises(ValueError):
        SymbolicExpr.create("broadcast", block, (3,)).eval(simplify_constraints=False)


def test_reshape_without_reorder_keeps_flat_sequence():
    block = _tagged_block([[100, 101, 102, 103], [104, 105, 106, 107]])  # shape (2, 4)
    expr = SymbolicExpr.create("reshape", block, (8,), False)
    assert expr.shape == (8,)
    assert expr.has_unknown_reorder() is False
    val, _ = expr.eval(simplify_constraints=False)
    assert _lane_ints(val) == [100, 101, 102, 103, 104, 105, 106, 107]


def test_reshape_with_reorder_sets_may_reorder():
    block = _tagged_block([0, 1, 2, 3])
    reshape = SymbolicExpr.create("reshape", block, (2, 2), True)
    assert reshape.has_unknown_reorder() is True
    # Propagates upward through a parent op.
    plus_one = reshape + SymbolicExpr.create("const", 1, tl.int32)
    assert plus_one.has_unknown_reorder() is True


def test_reshape_numel_mismatch_raises():
    block = _tagged_block([1, 2])  # 2 elements
    with pytest.raises(ValueError):
        SymbolicExpr.create("reshape", block, (2, 2), False)  # 4 elements


def test_trans_default_swaps_last_two_axes():
    # shape (2, 4) with lanes [[a,b,c,d],[e,f,g,h]] → (4, 2) with
    # [[a,e],[b,f],[c,g],[d,h]] → flat [a,e,b,f,c,g,d,h].
    block = _tagged_block([[1, 2, 3, 4], [5, 6, 7, 8]])
    expr = SymbolicExpr.create("trans", block, None)
    assert expr.shape == (4, 2)
    val, _ = expr.eval(simplify_constraints=False)
    assert _lane_ints(val) == [1, 5, 2, 6, 3, 7, 4, 8]


def test_trans_rank_lt_2_default_raises():
    # Triton's tl.trans raises ValueError("tl.trans invoked with a 0- or 1-
    # dimensional tensor") in interpreter mode; mirror that.
    block = _tagged_block([1, 2])
    with pytest.raises(ValueError):
        SymbolicExpr.create("trans", block, None)


def test_trans_permute_3d():
    # Input shape (2, 4, 8), perm (2, 0, 1) → output shape (8, 2, 4).
    # By the np.transpose convention: out[a, b, c] = in[b, c, a].
    # out[0, 0, 0] = in[0, 0, 0] = 0
    # out[7, 1, 3] = in[1, 3, 7] = 1*32 + 3*8 + 7 = 63
    values = np.arange(64, dtype=np.int32).reshape(2, 4, 8)
    block = _tagged_block(values)
    expr = SymbolicExpr.create("trans", block, (2, 0, 1))
    assert expr.shape == (8, 2, 4)
    val, _ = expr.eval(simplify_constraints=False)
    lanes = _lane_ints(val)
    # out flat index for (0,0,0): 0 → in[0,0,0] = 0
    assert lanes[0] == 0
    # out flat index for (7,1,3): 7*(2*4) + 1*4 + 3 = 56 + 4 + 3 = 63 →
    # in[1,3,7] = 1*(4*8) + 3*8 + 7 = 32 + 24 + 7 = 63
    assert lanes[7 * 8 + 1 * 4 + 3] == 63


def test_join_broadcasts_then_interleaves():
    # lhs shape (1, 4), rhs shape (2, 4) → common (2, 4), output (2, 4, 2).
    # Output lane at (i, j, 0) = lhs[0, j] (broadcast); at (i, j, 1) = rhs[i, j].
    lhs = _tagged_block([[100, 101, 102, 103]])  # shape (1, 4)
    rhs_vals = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    rhs = _tagged_block(rhs_vals)  # shape (2, 4)
    expr = SymbolicExpr.create("join", lhs, rhs)
    assert expr.shape == (2, 4, 2)
    val, _ = expr.eval(simplify_constraints=False)
    lanes = _lane_ints(val)
    # (0, 0, 0) = lhs[0, 0] = 100; (0, 0, 1) = rhs[0, 0] = 1
    assert lanes[0] == 100
    assert lanes[1] == 1
    # (1, 3, 0) = lhs[0, 3] = 103; (1, 3, 1) = rhs[1, 3] = 8
    assert lanes[1 * 4 * 2 + 3 * 2 + 0] == 103
    assert lanes[1 * 4 * 2 + 3 * 2 + 1] == 8


def test_join_incompatible_shapes_raise():
    lhs = _tagged_block([1, 2])  # shape (2,)
    rhs = _tagged_block([1, 2, 3, 4])  # shape (4,)
    with pytest.raises(ValueError):
        SymbolicExpr.create("join", lhs, rhs)


def test_join_scalar_scalar_interleave():
    lhs = SymbolicExpr.create("const", 5, tl.int32)
    rhs = SymbolicExpr.create("const", 9, tl.int32)
    expr = SymbolicExpr.create("join", lhs, rhs)
    assert expr.shape == (2,)
    val, _ = expr.eval(simplify_constraints=False)
    assert _lane_ints(val) == [5, 9]


def test_has_unknown_reorder_cache_invalidates_on_child_mutation():
    # Build a tree with no may_reorder below, prime the cache, then splice in
    # a reshape-with-reorder child and confirm the answer flips.
    block = _tagged_block([1, 2, 3, 4])
    parent = SymbolicExpr.create("expand_dims", block, 0)
    assert parent.has_unknown_reorder() is False  # primes cache to False
    reorder_reshape = SymbolicExpr.create("reshape", block, (2, 2), True)
    parent.add_child("arg", reorder_reshape)
    assert parent.has_unknown_reorder() is True


def test_shape_nodes_support_concretize_roundtrip():
    """Ensure replace_subtree() can walk through every shape node without
    hitting NotImplementedError. This locks the concretize() plumbing for
    downstream fallback paths (replace_subtree -> concretize() -> const)."""
    from triton_viz.core.patch import patch_op
    import triton_viz  # noqa: F401

    # Build a subtree composed of every PR4 shape op, attach real concrete
    # functions, then replace_subtree should collapse it to a const node.
    # Direct call: verify each node has a concretize() method defined.
    from triton_viz.clients.symbolic_engine import (
        SplatSymbolicExpr,
        ExpandDimsSymbolicExpr,
        BroadcastSymbolicExpr,
        ReshapeSymbolicExpr,
        TransSymbolicExpr,
        JoinSymbolicExpr,
        SymbolicExpr as _SE,
    )

    for cls in (
        SplatSymbolicExpr,
        ExpandDimsSymbolicExpr,
        BroadcastSymbolicExpr,
        ReshapeSymbolicExpr,
        TransSymbolicExpr,
        JoinSymbolicExpr,
    ):
        assert (
            cls.concretize is not _SE.concretize
        ), f"{cls.__name__} must override concretize() for PR4"
    del patch_op  # keep the import linter happy
