"""End-to-end tests for sanitizer with kernel execution."""

import torch

import triton
import triton.language as tl

import triton_viz
from triton_viz.core.config import config as cfg
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    SanitizerSymbolicExecution,
)

from .conftest import LoadIndexChecker


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


# ======== Loop Hook =========


def test_loop_hook_before_materializes_symbolic_bounds():
    class _FakeRange:
        def __init__(self, start, stop, step, length):
            self.start = start
            self.stop = stop
            self.step = step
            self._len = length

        def __len__(self):
            return self._len

    sanitizer = SanitizerSymbolicExecution()
    loop_callbacks = sanitizer.register_for_loop_callback()
    assert loop_callbacks.before_loop_callback is not None
    assert loop_callbacks.range_wrapper_factory is not None

    # Positive step: min(start), max(stop)
    start_expr = SymbolicExpr(
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


# ======== Indirect Load/Store =========


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
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


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
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


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
    assert (
        expected_offsets == observed_offsets
    ), "Observed offsets do not match expected offsets."


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
