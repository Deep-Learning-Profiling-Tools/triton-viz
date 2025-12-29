import pytest
import torch
import numpy as np
from typing import Any, cast, Sequence

import triton
import triton.language as tl

import triton_viz
from triton_viz.core.config import config as cfg
from triton_viz.core.data import Load, RawLoad
from triton_viz.core.client import Client
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    ConstraintExpr,
    Z3Expr,
    NullSanitizer,
    SymbolicSanitizer,
    RangeWrapper,
    _intervals_to_constraint,
)
from triton_viz.core.callbacks import ForLoopCallbacks
from z3 import simplify, is_int_value
from z3.z3 import ArithRef, BoolRef, IntNumRef
from z3.z3util import get_vars

# ======== Helpers ===========


class LoadIndexChecker(SymbolicSanitizer):
    """
    Record all offsets, then union into a set.
    """
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._offset_lists: list[list[int]] = list()

    @property
    def observed_offsets(self) -> list[list[int]]:
        return self._offset_lists

    def register_op_callback(self, op_type):
        op_callbacks = super().register_op_callback(op_type)
        if op_type not in (Load, RawLoad) or op_callbacks.op_overrider is None:
            return op_callbacks

        orig_overrider = op_callbacks.op_overrider

        def _sum_offsets_from_addptr(expr):
            offsets = []

            cur = expr
            while cur.op == "addptr":
                off = cur.offset
                if (
                    off.op == "const"
                ):  # If any offset is not constant, we cannot sum it.
                    offsets.append(off.to_py())
                cur = cur.ptr

            if len(offsets) == 0:
                return None
            return np.sum(offsets, axis=0)

        def _new_load_overrider(ptr, *args, **kwargs):
            # exec original overrider
            load_expr = orig_overrider(ptr, *args, **kwargs)
            p = load_expr.ptr
            offs = _sum_offsets_from_addptr(p)
            if offs is not None:
                self._offset_lists.append(offs.tolist())
            return load_expr

        # Return OpCallbacks with the new overrider, preserving other callbacks
        from triton_viz.core.callbacks import OpCallbacks

        return OpCallbacks(
            before_callback=op_callbacks.before_callback,
            after_callback=op_callbacks.after_callback,
            op_overrider=_new_load_overrider,
        )


load_index_checker: LoadIndexChecker = LoadIndexChecker(abort_on_error=False)


class LoopBoundsChecker(SymbolicSanitizer):
    """
    Record concretized loop bounds.
    """
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self._bounds: list[tuple[int, int, int]] = list()

    @property
    def observed_bounds(self) -> list[tuple[int, int, int]]:
        return self._bounds

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        callbacks = super().register_for_loop_callback()
        orig_before = callbacks.before_loop_callback

        def _before_loop(lineno, iterable):
            if orig_before is not None:
                orig_before(lineno, iterable)
            if isinstance(iterable, RangeWrapper):
                self._bounds.append((iterable.start, iterable.stop, iterable.step))

        return ForLoopCallbacks(
            range_wrapper_factory=callbacks.range_wrapper_factory,
            range_type_callback=callbacks.range_type_callback,
            before_loop_callback=_before_loop,
            loop_iter_overrider=callbacks.loop_iter_overrider,
            loop_iter_listener=callbacks.loop_iter_listener,
            after_loop_callback=callbacks.after_loop_callback,
        )


loop_bounds_checker: LoopBoundsChecker = LoopBoundsChecker(abort_on_error=True)


class LoopDeferredCheckRecorder(SymbolicSanitizer):
    """
    Record when deferred checks are executed after loop exit.
    """
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.after_loop_pending: list[int] = []
        self.check_inside_loop: list[tuple[Z3Expr, Sequence[ConstraintExpr]]] = []
        self.iterator_constraints: list[ConstraintExpr] = []

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: Sequence[ConstraintExpr],
        symbolic_expr: SymbolicExpr,
    ) -> None:
        self.check_inside_loop.append((access_addr, expr_constraints))
        super()._check_range_satisfiable(access_addr, expr_constraints, symbolic_expr)

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        callbacks = super().register_for_loop_callback()
        orig_after = callbacks.after_loop_callback

        def _after_loop(lineno: int) -> None:
            pending = 0
            if self.loop_stack and self.loop_stack[-1].lineno == lineno:
                ctx = self.loop_stack[-1]
                pending = len(ctx.pending_checks)
                iterator_constraint = _intervals_to_constraint(ctx.idx_z3, ctx.values)
                if iterator_constraint is not None:
                    self.iterator_constraints.append(iterator_constraint)
            if orig_after is not None:
                orig_after(lineno)
            self.after_loop_pending.append(pending)

        return ForLoopCallbacks(
            range_wrapper_factory=callbacks.range_wrapper_factory,
            range_type_callback=callbacks.range_type_callback,
            before_loop_callback=callbacks.before_loop_callback,
            loop_iter_overrider=callbacks.loop_iter_overrider,
            loop_iter_listener=callbacks.loop_iter_listener,
            after_loop_callback=_after_loop,
        )


loop_deferred_check_recorder: LoopDeferredCheckRecorder = LoopDeferredCheckRecorder(
    abort_on_error=False
)


# ======== Init ===========
def test_init_null_sanitizer():
    cfg.enable_sanitizer = False
    s2 = Sanitizer(abort_on_error=True)
    assert isinstance(s2, NullSanitizer)


def test_init_symbolic_execution():
    try:
        cfg.enable_sanitizer = True
        s3 = Sanitizer(abort_on_error=False)
        assert isinstance(s3, SymbolicSanitizer) and s3.abort_on_error is False
    finally:
        cfg.enable_sanitizer = False


def test_init_default_sanitizer():
    try:
        cfg.enable_sanitizer = True
        s = Sanitizer()
        assert isinstance(s, Client)
    finally:
        cfg.enable_sanitizer = False


# ======== Indirect Load/Store ===========


@triton_viz.trace(client=load_index_checker)
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    indices = tl.load(idx_ptr + offsets)
    out_val = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offsets, out_val)


def test_indirect_load():
    load_index_checker.observed_offsets.clear()

    idx = torch.arange(128, dtype=torch.int32)
    src = torch.rand(128)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    indirect_load_kernel[grid](idx, src, dst, BLOCK_SIZE=32)

    expected_offsets = idx.cpu().numpy().tolist()  # Ground truth
    observed_offsets = [
        x for sublist in load_index_checker.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


# ======== Loop Tests ===========


@triton_viz.trace(client=loop_bounds_checker)
@triton.jit
def loop_bounds_kernel(start_ptr, stop_ptr, out_ptr):
    start = tl.load(start_ptr)
    stop = tl.load(stop_ptr)
    for _ in range(start, stop):
        pass
    tl.store(out_ptr, start)


@triton_viz.trace(client=loop_bounds_checker)
@triton.jit
def loop_bounds_pid_kernel(out_ptr):
    pid = tl.program_id(0)
    start = pid
    stop = pid + 2
    for _ in range(start, stop):
        pass
    tl.store(out_ptr + pid, start)


@triton_viz.trace(client=loop_deferred_check_recorder)
@triton.jit
def loop_deferred_check_kernel(out_ptr):
    for i in range(0, 4):
        idx = i + 1
        tl.store(out_ptr + idx, idx)


@triton_viz.trace(client=load_index_checker)
@triton.jit
def loop_deferred_check_simplify_kernel(out_ptr):
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0) + 1
    for i in range(0, num_blocks):
        idx = pid + 1
        tl.store(out_ptr + idx, idx)


def test_loop_bounds_from_load():
    loop_bounds_checker.observed_bounds.clear()
    start = torch.tensor([2], dtype=torch.int32)
    stop = torch.tensor([6], dtype=torch.int32)
    out = torch.empty((1,), dtype=torch.int32)

    loop_bounds_kernel[(1,)](start, stop, out)

    assert loop_bounds_checker.observed_bounds == [(2, 6, 1)]


def test_loop_bounds_from_pid():
    loop_bounds_checker.observed_bounds.clear()
    out = torch.empty((2,), dtype=torch.int32)

    loop_bounds_pid_kernel[(2,)](out)

    assert loop_bounds_checker.observed_bounds == [(0, 2, 1), (1, 3, 1)]


def test_loop_deferred_checks_after_context():
    loop_deferred_check_recorder.after_loop_pending.clear()
    loop_deferred_check_recorder.check_inside_loop.clear()
    loop_deferred_check_recorder.iterator_constraints.clear()
    loop_deferred_check_recorder.records.clear()

    out = torch.empty((2,), dtype=torch.int32)

    loop_deferred_check_kernel[(1,)](out)

    assert loop_deferred_check_recorder.after_loop_pending == [1]
    addr_expr, _ = loop_deferred_check_recorder.check_inside_loop[0]
    assert "4*loop_i_2" in str(addr_expr)
    iterator_constraints_str = " ".join(
        str(c) for c in loop_deferred_check_recorder.iterator_constraints
    )
    assert "loop_i_2 >= 0" in iterator_constraints_str
    assert "loop_i_2 < 4" in iterator_constraints_str
    assert loop_deferred_check_recorder.records


def test_loop_deferred_checks_simplify():
    load_index_checker.observed_offsets.clear()

    out = torch.empty((3,), dtype=torch.int32)

    loop_deferred_check_simplify_kernel[(2,)](out)

    assert load_index_checker.observed_offsets == []


# ======== Reduce Operations =========


@pytest.mark.parametrize("op", ["max", "min", "sum"])
@pytest.mark.parametrize("data", [[1, 5, 3, 2], [42]])
def test_reduce_expr_eval(op: str, data):
    input_arr = SymbolicExpr.create("const", np.array(data), tl.int32)
    max_expr = SymbolicExpr.create(op, input_arr, None, False)
    import builtins

    result, _ = max_expr.eval()
    assert cast(IntNumRef, result).as_long() == getattr(builtins, op)(data)


# ======== Basic Symbolic Expr Operations =========


@pytest.mark.parametrize("value", [(1, 2, 3), 4])
def test_basic_expr_const_eval(value):
    const_expr = SymbolicExpr.create("const", value, tl.int32)
    result, constraints = const_expr.eval()
    if isinstance(value, (list, tuple)):
        assert [cast(IntNumRef, v).as_long() for v in cast(list, result)] == list(value)
    else:
        assert cast(IntNumRef, result).as_long() == value
    assert constraints == []


@pytest.mark.parametrize(
    "axis,expected_pid",
    [
        (0, SymbolicExpr.PID0),
        (1, SymbolicExpr.PID1),
        (2, SymbolicExpr.PID2),
    ],
)
def test_basic_expr_pid_eval(axis, expected_pid):
    grid = (4, 5, 6)
    pid_expr = SymbolicExpr.create("pid", axis)
    result, constraints = pid_expr.eval()
    assert result == expected_pid
    assert constraints == []


@pytest.mark.parametrize("start,end", [(4, 8), (0, 4)])
def test_basic_expr_arange_eval(start, end):
    arange_expr = SymbolicExpr.create("arange", tl.int32, start, end)
    result, constraints = arange_expr.eval()
    result = cast(ArithRef, result)
    assert result.decl().name() == f"arange_{start}_{end}"
    assert len(constraints) == 2
    assert str(constraints[0]) == f"{result} >= {start}"
    assert str(constraints[1]) == f"{result} < {end}"


# ======== Unary Symbolic Expr Operations =========


@pytest.mark.parametrize(
    "op,value,expected",
    [
        ("abs", -3, 3),
        ("fabs", -7, 7),
    ],
)
def test_unary_expr_eval(op: str, value: int, expected: int):
    arg = SymbolicExpr.create("const", value, tl.int32)
    expr = SymbolicExpr.create(op, arg)
    result, constraints = expr.eval()
    assert cast(IntNumRef, result).as_long() == expected
    assert constraints == []


# ======== Binary Symbolic Expr Operations =========


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
    lhs_expr = SymbolicExpr.create("const", lhs, tl.int32)
    rhs_expr = SymbolicExpr.create("const", rhs, tl.int32)
    expr = SymbolicExpr.create(op, lhs_expr, rhs_expr)
    result, constraints = expr.eval()
    if isinstance(expected, bool):
        assert str(result) == str(expected)
    else:
        assert cast(IntNumRef, result).as_long() == expected
    assert constraints == []


# ======== Pointer Symbolic Expr Operations =========


def test_pointer_expr_addptr_eval():
    base = SymbolicExpr.create("const", 100, tl.pointer_type(tl.int32))
    offset = SymbolicExpr.create("const", 3, tl.int32)
    expr = SymbolicExpr.create("addptr", base, offset)
    result, constraints = expr.eval()
    assert cast(IntNumRef, result).as_long() == 112
    assert constraints == []


# ======== Reshape Symbolic Expr Operations =========


@pytest.mark.parametrize(
    "op,extra",
    [
        ("splat", tl.block_type(tl.int32, [2])),
        ("expand_dims", 0),
        ("broadcast", (2,)),
        ("reshape", (2,)),
        ("trans", (0,)),
    ],
)
def test_reshape_expr_eval(op: str, extra):
    arg = SymbolicExpr.create("const", 5, tl.int32)
    if op == "splat":
        expr = SymbolicExpr.create(op, extra, arg)
    else:
        expr = SymbolicExpr.create(op, arg, extra)
    result, constraints = expr.eval()
    assert cast(IntNumRef, result).as_long() == 5
    assert constraints == []
