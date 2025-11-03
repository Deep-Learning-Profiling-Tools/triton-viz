import pytest
import torch
import numpy as np

import triton
import triton.language as tl
from triton.runtime.interpreter import TensorHandle

import triton_viz
from triton_viz.core.config import config as cfg
from triton_viz.core.data import AddPtr, Load, RawLoad
from triton_viz.core.client import Client
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    NullSanitizer,
    SanitizerSymbolicExecution,
)


# ======== Init ===========
def test_init_null_sanitizer():
    try:
        cfg.disable_sanitizer = True
        s2 = Sanitizer(abort_on_error=True)
        assert isinstance(s2, NullSanitizer)
    finally:
        cfg.disable_sanitizer = False


def test_init_symbolic_execution():
    s3 = Sanitizer(abort_on_error=True)
    assert isinstance(s3, SanitizerSymbolicExecution) and s3.abort_on_error is True


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
    expr = SymbolicExpr("addptr", base, offset)
    assert expr.eval()[0] == 1000 + 3 * 4


def test_addptr_overrider():
    # Run through sanitizer's overrider
    ptr_dtype = tl.pointer_type(tl.int32)
    ptr_th = TensorHandle(np.array([1000]), ptr_dtype)
    offset_th = TensorHandle(np.array([3]), tl.int32)
    sanitizer = SanitizerSymbolicExecution(abort_on_error=False)
    op_callbacks = sanitizer.register_op_callback(AddPtr)
    assert op_callbacks.op_overrider is not None
    expr = op_callbacks.op_overrider(ptr_th, offset_th)  # offset = 3
    assert expr.op == "addptr"
    assert expr.eval()[0] == 1000 + 3 * 4  # element_bytewidth = 4


# ======== Null Sanitizer =========


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def null_sanitizer_kernel(idx_ptr):
    a = tl.load(idx_ptr)
    tl.store(idx_ptr, a + 5)


def test_null_sanitizer():
    try:
        cfg.disable_sanitizer = True
        idx = torch.arange(128, dtype=torch.int32)
        null_sanitizer_kernel[(1,)](idx)
    finally:
        cfg.disable_sanitizer = False


# ======== Indirect Load/Store =========
def test_const_dtype_inference():
    x = SymbolicExpr.from_value((1, 2, 3))
    y = SymbolicExpr.from_value((1, 2, 3))
    z = SymbolicExpr("add", x, y)

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
        offsets.append(np.asarray(off.to_py(), dtype=np.int64))
        cur = cur.ptr

    if non_const_offset:
        raise ValueError(
            f"Some non-constant offsets found ({non_const_offset}) in the addptr chain."
        )
    return np.sum(offsets, axis=0)


class LoadIndexChecker(SanitizerSymbolicExecution):
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
            # This is because fp32 is usually the outermost dtype of a "load of a load" chain.
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


@triton_viz.trace(clients=(san1 := LoadIndexChecker(abort_on_error=True)))
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
    assert expected_offsets == observed_offsets, (
        "Observed offsets do not match expected offsets."
    )


@triton_viz.trace(clients=(san2 := LoadIndexChecker(abort_on_error=True)))
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
    assert expected_offsets == observed_offsets, (
        "Observed offsets do not match expected offsets."
    )


@triton_viz.trace(clients=(san3 := LoadIndexChecker(abort_on_error=True)))
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
    assert expected_offsets == observed_offsets, (
        "Observed offsets do not match expected offsets."
    )


# ======== Sanitizer Backend Tests =========
def test_switch_backend():
    """Switch back and forth at runtime."""
    original = cfg.disable_sanitizer

    cfg.disable_sanitizer = False  # Enable sanitizer
    assert cfg.disable_sanitizer is False

    cfg.disable_sanitizer = True  # Disable sanitizer
    assert cfg.disable_sanitizer is True

    cfg.disable_sanitizer = original


def test_invalid_backend_raises():
    """Setting a non-boolean value should raise TypeError."""
    with pytest.raises(TypeError):
        cfg.disable_sanitizer = "not_a_bool"


# ======== For-Loop Optimization Tests =========
@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
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
@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
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


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
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
    input_arr = SymbolicExpr("const", np.array([1, 5, 3, 2]), tl.int32)
    max_expr = SymbolicExpr("max", input_arr, None, False)
    result, _ = max_expr.eval()
    assert result == 5


def test_reduce_min_expr_eval():
    """Test that min reduce operation evaluates correctly."""
    # Test with array of values - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([1, 5, 3, 2]), tl.int32)
    min_expr = SymbolicExpr("min", input_arr, None, False)
    result, _ = min_expr.eval()
    assert result == 1


def test_reduce_max_single_element():
    """Test that max reduce operation works with single element."""
    # Test with single element - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([42]), tl.int32)
    max_expr = SymbolicExpr("max", input_arr, None, False)
    result, _ = max_expr.eval()
    assert result == 42


def test_reduce_min_single_element():
    """Test that min reduce operation works with single element."""
    # Test with single element - create const SymbolicExpr with numpy array
    input_arr = SymbolicExpr("const", np.array([42]), tl.int32)
    min_expr = SymbolicExpr("min", input_arr, None, False)
    result, _ = min_expr.eval()
    assert result == 42


def test_reduce_max_empty_array():
    """Test that max reduce operation raises ValueError for empty array."""
    import pytest

    input_arr = SymbolicExpr("const", np.array([]), tl.int32)
    max_expr = SymbolicExpr("max", input_arr, None, False)
    with pytest.raises(ValueError, match="Cannot compute max of empty array"):
        max_expr.eval()


def test_reduce_min_empty_array():
    """Test that min reduce operation raises ValueError for empty array."""
    import pytest

    input_arr = SymbolicExpr("const", np.array([]), tl.int32)
    min_expr = SymbolicExpr("min", input_arr, None, False)
    with pytest.raises(ValueError, match="Cannot compute min of empty array"):
        min_expr.eval()
