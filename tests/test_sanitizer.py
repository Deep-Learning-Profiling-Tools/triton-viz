import pytest
import torch
import numpy as np

import triton
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

import triton_viz
from triton_viz import trace
from triton_viz.core.config import config as cfg
from triton_viz.core.data import AddPtr, Load, RawLoad, Trans
from triton_viz.core.client import Client
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    NullSanitizer,
    SymbolicSanitizer,
)


# ======== Init ===========
def test_init_null_sanitizer():
    try:
        cfg.enable_sanitizer = False
        s2 = Sanitizer(abort_on_error=True)
        assert isinstance(s2, NullSanitizer)
    finally:
        cfg.enable_sanitizer = True


def test_init_symbolic_execution():
    s3 = Sanitizer(abort_on_error=True)
    assert isinstance(s3, SymbolicSanitizer) and s3.abort_on_error is True


def test_init_default_sanitizer():
    s = Sanitizer()
    assert isinstance(s, Client)


# ======== AddPtr =========


def test_addptr_expr_eval():
    base = SymbolicExpr.from_value(1000)  # Synthetic base address
    base.dtype_tt = tl.pointer_type(
        tl.int32
    )  # Simulate a pointer type, int32 = 4 bytes
    offset = SymbolicExpr.from_value(3)
    expr = SymbolicExpr.create("addptr", base, offset)
    assert expr.eval()[0] == 1000 + 3 * 4


def test_addptr_overrider():
    # Run through sanitizer's overrider
    ptr_dtype = tl.pointer_type(tl.int32)
    ptr_th = TensorHandle(np.array([1000]), ptr_dtype)
    offset_th = TensorHandle(np.array([3]).astype(np.int32), tl.int32)
    sanitizer = SymbolicSanitizer(abort_on_error=False)
    op_callbacks = sanitizer.register_op_callback(AddPtr)
    assert op_callbacks.op_overrider is not None
    expr = op_callbacks.op_overrider(ptr_th, offset_th)  # offset = 3
    assert expr.op == "addptr"
    assert expr.eval()[0] == 1000 + 3 * 4  # element_bytewidth = 4


# ======== Trans =========


def test_trans_overrider_passthrough_and_dtype():
    base = SymbolicExpr.from_value((1, 2, 3))
    base.dtype_tt = tl.float32  # ensure dtype is preserved through trans

    sanitizer = SymbolicSanitizer(abort_on_error=False)
    op_callbacks = sanitizer.register_op_callback(Trans)
    assert op_callbacks.op_overrider is not None

    perm = (1, 0)
    trans_expr = op_callbacks.op_overrider(base, perm)

    assert isinstance(trans_expr, SymbolicExpr)
    assert trans_expr.op == "trans"
    assert trans_expr.arg is base
    assert trans_expr.permutation.op == "const"
    assert trans_expr.permutation.to_py() == perm
    assert trans_expr.dtype_tt == base.dtype_tt

    def _z3_to_int_list(values):
        vals = values if isinstance(values, list) else [values]
        return [v.as_long() if hasattr(v, "as_long") else int(str(v)) for v in vals]

    trans_vals, trans_constraints = trans_expr.eval()
    base_vals, base_constraints = base.eval()

    assert _z3_to_int_list(trans_vals) == _z3_to_int_list(base_vals)
    assert trans_constraints == base_constraints


# ======== Null Sanitizer =========


@triton_viz.trace(client=Sanitizer(abort_on_error=True))
@triton.jit
def null_sanitizer_kernel(idx_ptr):
    a = tl.load(idx_ptr)
    tl.store(idx_ptr, a + 5)


def test_null_sanitizer():
    try:
        cfg.enable_sanitizer = False
        idx = torch.arange(128, dtype=torch.int32)
        null_sanitizer_kernel[(1,)](idx)
    finally:
        cfg.enable_sanitizer = True


# ======== Indirect Load/Store =========
def test_const_dtype_inference():
    x = SymbolicExpr.from_value((1, 2, 3))
    y = SymbolicExpr.from_value((1, 2, 3))
    z = SymbolicExpr.create("add", x, y)

    assert x.dtype_tt == tl.int32
    assert y.dtype_tt == tl.int32
    assert z.dtype_tt == tl.int32

    expected_tree = (
        "\n"
        "add [dtype=int32]\n"
        "├── lhs: const=(1, 2, 3) [dtype=int32]\n"
        "└── rhs: const=(1, 2, 3) [dtype=int32]"
    )
    assert z.to_tree_str() == expected_tree


def test_loop_hook_before_materializes_symbolic_bounds():
    class _FakeRange:
        def __init__(self, start, stop, step, length):
            self.start = start
            self.stop = stop
            self.step = step
            self._len = length

        def __len__(self):
            return self._len

    sanitizer = SymbolicSanitizer()
    loop_callbacks = sanitizer.register_for_loop_callback()
    assert loop_callbacks.before_loop_callback is not None
    assert loop_callbacks.range_wrapper_factory is not None

    # Positive step: min(start), max(stop)
    start_expr = SymbolicExpr.create(
        "add", SymbolicExpr.from_value(2), SymbolicExpr.from_value(3)
    )
    stop_expr = SymbolicExpr.from_value(8)
    step_expr = SymbolicExpr.from_value(1)
    wrapped = loop_callbacks.range_wrapper_factory(
        None,
        200,
        "python_range",
        (start_expr, stop_expr, step_expr),
        {},
        range,
    )
    loop_callbacks.before_loop_callback(200, wrapped)
    ctx = sanitizer.loop_stack.pop()
    assert ctx.start == 5  # min(start) == 5
    assert ctx.stop == 8  # max(stop) == 8
    assert ctx.step == 1

    # Negative step: max(start), min(stop)
    start_expr = SymbolicExpr.from_value(10)
    stop_expr = SymbolicExpr.from_value(-2)
    step_expr = SymbolicExpr.from_value(-3)
    wrapped = loop_callbacks.range_wrapper_factory(
        None,
        201,
        "python_range",
        (start_expr, stop_expr, step_expr),
        {},
        range,
    )
    loop_callbacks.before_loop_callback(201, wrapped)
    ctx = sanitizer.loop_stack.pop()
    assert ctx.start == 10  # max(start)
    assert ctx.stop == -2  # min(stop)
    assert ctx.step == -3


def _sum_offsets_from_addptr(expr):
    """
    Traverse an addptr SymbolicExpr and sum all constant offsets.
    If any offset is not constant, return None.
    """
    offsets = []
    non_const_offset = None

    cur = expr
    while cur.op == "addptr":
        off = cur.offset
        if off.op != "const":  # If any offset is not constant, we cannot sum it.
            non_const_offset = off
            break
        offsets.append(off.to_py().tolist())
        cur = cur.ptr

    if non_const_offset:
        raise ValueError(
            f"Some non-constant offsets found ({non_const_offset}) in the addptr chain."
        )
    return np.sum(offsets, axis=0)


class LoadIndexChecker(SymbolicSanitizer):
    """
    Record all offsets, then union into a set.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._offset_lists: list[list[int]] = list()

    @property
    def observed_offsets(self):
        return self._offset_lists

    def register_op_callback(self, op_type):
        op_callbacks = super().register_op_callback(op_type)
        if op_type not in (Load, RawLoad) or op_callbacks.op_overrider is None:
            return op_callbacks

        orig_overrider = op_callbacks.op_overrider

        def new_load_overrider(ptr, *a, **k):
            # exec original overrider
            load_expr = orig_overrider(ptr, *a, **k)

            # Important: We only record pointers accessing fp32!
            # This is the case in all unittests.
            p = load_expr.ptr
            if (
                hasattr(p, "dtype_tt")
                and isinstance(p.dtype_tt, tl.pointer_type)
                and p.dtype_tt.element_ty is tl.float32
            ):  # filtering fp32 pointers
                offs = _sum_offsets_from_addptr(p)
                if offs is not None:
                    self._offset_lists.append(offs.tolist())
            return load_expr

        # Return OpCallbacks with the new overrider, preserving other callbacks
        from triton_viz.core.callbacks import OpCallbacks

        return OpCallbacks(
            before_callback=op_callbacks.before_callback,
            after_callback=op_callbacks.after_callback,
            op_overrider=new_load_overrider,
        )


@triton_viz.trace(client=(san1 := LoadIndexChecker(abort_on_error=True)))
@triton.jit
def indirect_load_kernel(idx_ptr, src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    indices = tl.load(idx_ptr + offsets)
    out_val = tl.load(src_ptr + indices)
    tl.store(dst_ptr + offsets, out_val)


def test_indirect_load():
    idx = torch.arange(128, dtype=torch.int32)
    src = torch.rand(128)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(128, META["BLOCK_SIZE"]),)
    indirect_load_kernel[grid](idx, src, dst, BLOCK_SIZE=32)

    expected_offsets = idx.cpu().numpy().tolist()  # Ground truth
    observed_offsets = [
        x for sublist in san1.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


@triton_viz.trace(LoadIndexChecker(abort_on_error=True))
@triton.jit
def triple_indirect_load_kernel(
    idx1_ptr,  # int32*
    idx2_ptr,  # int32*
    src_ptr,  # fp32*
    dst_ptr,  # fp32*
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    idx1_val = tl.load(idx1_ptr + offsets)
    idx2_val = tl.load(idx2_ptr + idx1_val)
    out_val = tl.load(src_ptr + idx2_val)

    tl.store(dst_ptr + offsets, out_val)


def test_triple_indirect_load(device):
    N = 128

    src = torch.rand(N, device=device, dtype=torch.float32)
    idx2 = torch.randint(0, N, (N,), device=device, dtype=torch.int32)
    idx1 = torch.randint(0, N, (N,), device=device, dtype=torch.int32)

    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    triple_indirect_load_kernel[grid](
        idx1,
        idx2,
        src,
        dst,
        BLOCK_SIZE=32,
    )

    expected_offsets = idx2[idx1].cpu().numpy().tolist()  # Ground Truth
    observed_offsets = [
        x for sublist in san2.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


@triton_viz.trace(client=(san3 := LoadIndexChecker(abort_on_error=True)))
@triton.jit
def dual_offset_load_kernel(
    idx_a_ptr,  # int32*
    idx_b_ptr,  # int32*
    src_ptr,  # fp32*
    dst_ptr,  # fp32*
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    a = tl.load(idx_a_ptr + offsets)
    b = tl.load(idx_b_ptr + offsets)
    out_val = tl.load(src_ptr + a + b)

    tl.store(dst_ptr + offsets, out_val)


def test_dual_offset_load(device):
    N = 128

    src = torch.rand(N, device=device, dtype=torch.float32)
    # Generate indices so that a + b is always in-range (0 ≤ a + b < N)
    idx_a = torch.randint(0, N // 2, (N,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, N // 2, (N,), device=device, dtype=torch.int32)
    dst = torch.empty_like(src)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    dual_offset_load_kernel[grid](
        idx_a,
        idx_b,
        src,
        dst,
        BLOCK_SIZE=32,
    )

    expected_offsets = (idx_a + idx_b).cpu().numpy().tolist()  # Ground Truth
    observed_offsets = [
        x for sublist in san3.observed_offsets for x in sublist
    ]  # Flatten the list of lists
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


# ======== Sanitizer Backend Tests =========
def test_switch_backend():
    """Switch back and forth at runtime."""
    original = cfg.enable_sanitizer

    cfg.enable_sanitizer = True
    assert cfg.enable_sanitizer is True

    cfg.enable_sanitizer = False
    assert cfg.enable_sanitizer is False

    cfg.enable_sanitizer = original


def test_invalid_backend_raises():
    """Setting a non-boolean value should raise TypeError."""
    with pytest.raises(TypeError):
        cfg.enable_sanitizer = "not_a_bool"


# ======== For-Loop Optimization Tests =========
@triton_viz.trace(client=Sanitizer(abort_on_error=True))
@triton.jit
def copy_row_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_tiles = tl.cdiv(N, TILE_N)
    for tile_idx in range(0, num_tiles):
        col_offsets = tile_idx * TILE_N + tl.arange(0, TILE_N)
        mask = col_offsets < N

        x = tl.load(in_ptr + pid * N + col_offsets, mask=mask, other=0.0)
        y = x  # Copy operation
        tl.store(out_ptr + pid * N + col_offsets, y, mask=mask)


def test_copy_kernel():
    torch.manual_seed(0)
    M, N = 32, 1000
    x = torch.randn((M, N), dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (M,)
    copy_row_kernel[grid](
        x,
        y,
        M,
        N,
        TILE_N=128,
    )


# ======== Atomic Operations Tests =========
@triton_viz.trace(client=Sanitizer(abort_on_error=True))
@triton.jit
def atomic_add_kernel(
    output_ptr,
    value: tl.constexpr,
):
    # Simple atomic add operation
    tl.atomic_add(output_ptr, value)


def test_atomic_add():
    """Test that atomic_add operations work with the sanitizer."""
    y = torch.zeros(1, dtype=torch.float32)
    grid = (1,)
    atomic_add_kernel[grid](y, value=5.0)
    # Note: The sanitizer analyzes symbolically, so the actual value may not be updated
    # This test verifies that the operation doesn't crash


@triton_viz.trace(client=Sanitizer(abort_on_error=True))
@triton.jit
def atomic_cas_kernel(
    output_ptr,
    cmp_value: tl.constexpr,
    new_value: tl.constexpr,
):
    # Simple atomic compare-and-swap operation
    tl.atomic_cas(output_ptr, cmp_value, new_value)


def test_atomic_cas():
    """Test that atomic_cas operations work with the sanitizer."""
    y = torch.zeros(1, dtype=torch.float32)
    grid = (1,)
    atomic_cas_kernel[grid](y, cmp_value=0.0, new_value=5.0)
    # Note: The sanitizer analyzes symbolically, so the actual value may not be updated
    # This test verifies that the operation doesn't crash


# ======== Reduce Operations (max, min) =========
def test_reduce_max_expr_eval():
    """Test that max reduce operation evaluates correctly."""
    # Test with array of values - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr.create("const", np.array([1, 5, 3, 2]), tl.int32)
    max_expr = SymbolicExpr.create("max", input_arr, None, False)
    result, _ = max_expr.eval()
    assert result == 5


def test_reduce_min_expr_eval():
    """Test that min reduce operation evaluates correctly."""
    # Test with array of values - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr.create("const", np.array([1, 5, 3, 2]), tl.int32)
    min_expr = SymbolicExpr.create("min", input_arr, None, False)
    result, _ = min_expr.eval()
    assert result == 1


def test_reduce_max_single_element():
    """Test that max reduce operation works with single element."""
    # Test with single element - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr.create("const", np.array([42]), tl.int32)
    max_expr = SymbolicExpr.create("max", input_arr, None, False)
    result, _ = max_expr.eval()
    assert result == 42


def test_reduce_min_single_element():
    """Test that min reduce operation works with single element."""
    # Test with single element - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr.create("const", np.array([42]), tl.int32)
    min_expr = SymbolicExpr.create("min", input_arr, None, False)
    result, _ = min_expr.eval()
    assert result == 42


def test_reduce_max_empty_array():
    """Test that max reduce operation raises ValueError for empty array."""
    input_arr = SymbolicExpr.create("const", np.array([]), tl.int32)
    max_expr = SymbolicExpr.create("max", input_arr, None, False)
    with pytest.raises(ValueError, match="Cannot compute max of empty array"):
        max_expr.eval()


def test_reduce_min_empty_array():
    """Test that min reduce operation raises ValueError for empty array."""
    input_arr = SymbolicExpr.create("const", np.array([]), tl.int32)
    min_expr = SymbolicExpr.create("min", input_arr, None, False)
    with pytest.raises(ValueError, match="Cannot compute min of empty array"):
        min_expr.eval()


# ======== Cache Behavior Tests =========

# ---- Symbol Cache Tests ----


def _build_cache_test_expression(grid):
    """
    Helper to build a common test expression: arange(0, 10) + pid_0 * 16
    This simulates a typical memory access pattern.
    """
    arange_expr = SymbolicExpr.create(
        "arange",
        SymbolicExpr.create("const", tl.int32, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
        SymbolicExpr.create("const", 10, tl.int32),
    )
    pid_expr = SymbolicExpr.create(
        "pid",
        SymbolicExpr.create("const", grid, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
    )
    block_size = SymbolicExpr.create("const", 16, tl.int32)
    pid_times_block = SymbolicExpr.create("mul", pid_expr, block_size)
    final_expr = SymbolicExpr.create("add", arange_expr, pid_times_block)
    return final_expr


def test_symbol_cache_consistency():
    """
    Test that Symbol Cache produces the same Z3 expressions across repeated calls.
    """
    grid = (8, 1, 1)

    SymbolicExpr.ARANGE_DICT.clear()
    expr = _build_cache_test_expression(grid)
    z3_expr_1, constraints_1 = expr._to_z3()
    z3_expr_2, constraints_2 = expr._to_z3()

    assert str(z3_expr_1) == str(
        z3_expr_2
    ), "Repeated calls should return same expression"
    assert sorted([str(c) for c in constraints_1]) == sorted(
        [str(c) for c in constraints_2]
    ), "Repeated calls should return same constraints"


def test_unary_op_cache_side_effect():
    """
    Test specifically for side effects in unary ops (e.g., abs, fabs).
    """
    SymbolicExpr.ARANGE_DICT.clear()

    arange_expr = SymbolicExpr.create(
        "arange",
        SymbolicExpr.create("const", tl.int32, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
        SymbolicExpr.create("const", 10, tl.int32),
    )
    abs_expr = SymbolicExpr.create("abs", arange_expr)
    z3_1, constraints_1 = abs_expr._to_z3()
    z3_2, constraints_2 = abs_expr._to_z3()

    assert str(z3_1) == str(
        z3_2
    ), "Unary op should return same expression on repeated calls"
    assert sorted([str(c) for c in constraints_1]) == sorted(
        [str(c) for c in constraints_2]
    ), "Unary op should return same constraints on repeated calls"


def _build_shared_index_expr(grid):
    """Build a shared index expression: pid_0 * 16 + arange(0, 16)"""
    pid_expr = SymbolicExpr.create(
        "pid",
        SymbolicExpr.create("const", grid, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
    )
    block_size = SymbolicExpr.create("const", 16, tl.int32)
    pid_times_block = SymbolicExpr.create("mul", pid_expr, block_size)

    arange_expr = SymbolicExpr.create(
        "arange",
        SymbolicExpr.create("const", tl.int32, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
        SymbolicExpr.create("const", 16, tl.int32),
    )

    index_expr = SymbolicExpr.create("add", pid_times_block, arange_expr)
    return index_expr


def test_nested_expression_reuse():
    """
    Test the case where a sub-expression is reused in multiple places.
    This is common in kernels where the same index calculation is used
    for multiple loads/stores (like in layernorm).
    """
    grid = (8, 1, 1)

    SymbolicExpr.ARANGE_DICT.clear()

    shared_index = _build_shared_index_expr(grid)

    z3_1, constraints_1 = shared_index._to_z3()
    z3_2, constraints_2 = shared_index._to_z3()

    assert str(z3_1) == str(
        z3_2
    ), "Shared index should return same expression on repeated calls"
    assert sorted([str(c) for c in constraints_1]) == sorted(
        [str(c) for c in constraints_2]
    ), "Shared index should return same constraints on repeated calls"


def test_sum_operation_side_effect():
    """
    Test for side effects in operations that assign to self.constraints.
    Specifically tests the "sum" operation which does:
        arr, self.constraints = self.input._to_z3()
    """
    SymbolicExpr.ARANGE_DICT.clear()

    arange_expr = SymbolicExpr.create(
        "arange",
        SymbolicExpr.create("const", tl.int32, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
        SymbolicExpr.create("const", 10, tl.int32),
    )
    const_2 = SymbolicExpr.create("const", 2, tl.int32)
    mul_expr = SymbolicExpr.create("mul", arange_expr, const_2)
    sum_expr = SymbolicExpr.create("sum", mul_expr, 0, False)

    z3_1, constraints_1 = sum_expr._to_z3()
    z3_2, constraints_2 = sum_expr._to_z3()

    assert str(z3_1) == str(
        z3_2
    ), "Sum operation should return same expression on repeated calls"
    assert sorted([str(c) for c in constraints_1]) == sorted(
        [str(c) for c in constraints_2]
    ), "Sum operation should return same constraints on repeated calls"

# ---- Z3 Expression Comparison Tests ----


def test_node_state_after_single_call():
    """
    Test that node.z3 and node.constraints are consistent after repeated calls.
    """

    def build_expr():
        """Build expression: abs(arange(0, 10))"""
        SymbolicExpr.ARANGE_DICT.clear()
        arange = SymbolicExpr.create(
            "arange",
            SymbolicExpr.create("const", tl.int32, tl.int32),
            SymbolicExpr.create("const", 0, tl.int32),
            SymbolicExpr.create("const", 10, tl.int32),
        )
        abs_expr = SymbolicExpr.create("abs", arange)
        return abs_expr, arange

    abs_expr, arange = build_expr()
    z3_1, constraints_1 = abs_expr._to_z3()
    arange_state_1 = (str(arange.z3), sorted([str(c) for c in arange.constraints]))
    abs_state_1 = (
        str(abs_expr.z3),
        sorted([str(c) for c in abs_expr.constraints]),
    )

    z3_2, constraints_2 = abs_expr._to_z3()
    arange_state_2 = (str(arange.z3), sorted([str(c) for c in arange.constraints]))
    abs_state_2 = (
        str(abs_expr.z3),
        sorted([str(c) for c in abs_expr.constraints]),
    )

    assert str(z3_1) == str(z3_2)
    assert sorted([str(c) for c in constraints_1]) == sorted(
        [str(c) for c in constraints_2]
    )
    assert arange_state_1 == arange_state_2
    assert abs_state_1 == abs_state_2


def test_node_state_after_multiple_calls():
    """
    Test that node state remains consistent after multiple _to_z3() calls
    on the same expression.
    """

    def build_and_call_multiple():
        """Build arange expression and call _to_z3 three times."""
        SymbolicExpr.ARANGE_DICT.clear()

        arange = SymbolicExpr.create(
            "arange",
            SymbolicExpr.create("const", tl.int32, tl.int32),
            SymbolicExpr.create("const", 0, tl.int32),
            SymbolicExpr.create("const", 10, tl.int32),
        )

        results = []
        for _ in range(3):
            z3_expr, constraints = arange._to_z3()
            results.append((str(z3_expr), sorted([str(c) for c in constraints])))

        final_state = {
            "z3": str(arange.z3),
            "constraints": sorted([str(c) for c in arange.constraints]),
        }

        return results, final_state

    results, state = build_and_call_multiple()

    for i in range(1, len(results)):
        assert results[i] == results[0], f"Call #{i+1} results differ"

    assert state["z3"] == results[0][0]
    assert state["constraints"] == results[0][1]

def _build_layernorm_like_index_expr():
    """
    Simulate layernorm-like memory access pattern:
    Index: pid_0 * BLOCK_SIZE + arange(0, BLOCK_SIZE)
    """
    grid = (128, 1, 1)
    BLOCK_SIZE = 64

    pid_0 = SymbolicExpr.create(
        "pid",
        SymbolicExpr.create("const", grid, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
    )

    block_size_const = SymbolicExpr.create("const", BLOCK_SIZE, tl.int32)
    offset = SymbolicExpr.create("mul", pid_0, block_size_const)

    arange = SymbolicExpr.create(
        "arange",
        SymbolicExpr.create("const", tl.int32, tl.int32),
        SymbolicExpr.create("const", 0, tl.int32),
        SymbolicExpr.create("const", BLOCK_SIZE, tl.int32),
    )

    # Shared index: pid_0 * BLOCK_SIZE + arange
    shared_index = SymbolicExpr.create("add", offset, arange)
    return shared_index


def test_layernorm_like_memory_access_pattern():
    """
    Test that simulates the layernorm kernel scenario with multiple loads/stores
    sharing the same index expressions. Verifies Z3 expression consistency
    across repeated calls.
    """
    def capture_results():
        """Run test and return results."""
        SymbolicExpr.ARANGE_DICT.clear()

        index_expr = _build_layernorm_like_index_expr()

        # Simulate multiple uses of the same index (like multiple tl.load/tl.store)
        results = []
        for _ in range(3):
            z3_expr, constraints = index_expr._to_z3()
            results.append((str(z3_expr), sorted([str(c) for c in constraints])))

        return results

    results = capture_results()

    for i in range(1, len(results)):
        expr_0, constraints_0 = results[0]
        expr_i, constraints_i = results[i]
        assert expr_0 == expr_i, f"Access #{i+1} expression differs"
        assert constraints_0 == constraints_i, f"Access #{i+1} constraints differ"


def test_z3_expression_capture_comparison():
    """
    Test that Z3 expressions captured from repeated runs produce identical
    final results for the same symbolic expression trees.
    """
    def capture_final_z3_expressions():
        """
        Run a series of symbolic operations and capture the FINAL Z3 expressions
        (not intermediate calls).
        """
        SymbolicExpr.ARANGE_DICT.clear()

        # Build a complex expression tree similar to real kernel patterns
        grid = (8, 1, 1)

        # Expression 1: arange + pid * block_size
        arange1 = SymbolicExpr.create(
            "arange",
            SymbolicExpr.create("const", tl.int32, tl.int32),
            SymbolicExpr.create("const", 0, tl.int32),
            SymbolicExpr.create("const", 16, tl.int32),
        )
        pid1 = SymbolicExpr.create(
            "pid",
            SymbolicExpr.create("const", grid, tl.int32),
            SymbolicExpr.create("const", 0, tl.int32),
        )
        block_size1 = SymbolicExpr.create("const", 16, tl.int32)
        offset1 = SymbolicExpr.create("add", arange1, SymbolicExpr.create("mul", pid1, block_size1))

        # Expression 2: nested operations (abs, sum)
        arange2 = SymbolicExpr.create(
            "arange",
            SymbolicExpr.create("const", tl.int32, tl.int32),
            SymbolicExpr.create("const", 0, tl.int32),
            SymbolicExpr.create("const", 8, tl.int32),
        )
        abs_expr = SymbolicExpr.create("abs", arange2)
        sum_expr = SymbolicExpr.create("sum", abs_expr, 0, False)

        # Expression 3: binary operations chain
        const_a = SymbolicExpr.create("const", 2, tl.int32)
        const_b = SymbolicExpr.create("const", 3, tl.int32)
        mul_expr = SymbolicExpr.create("mul", offset1, const_a)
        add_expr = SymbolicExpr.create("add", mul_expr, const_b)

        # Get final Z3 expressions
        results = {}

        z3_offset1, constraints_offset1 = offset1._to_z3()
        results["offset1"] = {
            "expr": str(z3_offset1),
            "constraints": sorted([str(c) for c in constraints_offset1]),
        }

        z3_sum, constraints_sum = sum_expr._to_z3()
        results["sum"] = {
            "expr": str(z3_sum),
            "constraints": sorted([str(c) for c in constraints_sum]),
        }

        z3_add, constraints_add = add_expr._to_z3()
        results["add"] = {
            "expr": str(z3_add),
            "constraints": sorted([str(c) for c in constraints_add]),
        }

        return results

    results_a = capture_final_z3_expressions()
    results_b = capture_final_z3_expressions()

    for key in results_a:
        expr_a = results_a[key]
        expr_b = results_b[key]

        assert (
            expr_a["expr"] == expr_b["expr"]
        ), f"Expression '{key}' differs: A={expr_a['expr']}, B={expr_b['expr']}"

        assert (
            expr_a["constraints"] == expr_b["constraints"]
        ), f"Constraints for '{key}' differ: A={expr_a['constraints']}, B={expr_b['constraints']}"


# ---- Kernel Cache Tests ----


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _cache_test_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_kernel_cache_enables_dummy_benchmarker():
    """Test that dummy_benchmarker is installed for autotuned kernels."""
    traced = trace(client=Sanitizer())(_cache_test_add_kernel)

    # Check if dummy benchmarker was installed
    if hasattr(traced, "runner") and hasattr(traced.runner, "_do_bench"):
        bench_fn = traced.runner._do_bench
        assert (
            bench_fn is not None and bench_fn.__name__ == "dummy_benchmarker"
        ), f"Expected dummy_benchmarker, got: {bench_fn}"



