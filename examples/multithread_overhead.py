import torch
from tqdm import tqdm
import time
import numpy as np

import triton
import triton.language as tl

import triton_viz
from triton_viz.core.data import Load, RawLoad
from triton_viz.clients import Sanitizer
from triton_viz.clients.sanitizer.sanitizer import (
    SanitizerSymbolicExecution,
)


@triton_viz.trace(clients=Sanitizer(abort_on_error=True))
@triton.jit
def gemm_kernel(
    A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILE_SIZE: tl.constexpr
):
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    range_m = tl.arange(0, TILE_SIZE)
    range_n = tl.arange(0, TILE_SIZE)
    range_k = tl.arange(0, TILE_SIZE)
    range_m_block = TILE_SIZE * m_block + range_m[:, None]
    range_n_block = TILE_SIZE * n_block + range_n[None, :]
    accum = tl.zeros((TILE_SIZE, TILE_SIZE), dtype=tl.float32)
    for k_block in range(K // TILE_SIZE):
        range_k_block = TILE_SIZE * k_block + range_k
        A_off = K * range_m_block + range_k_block[None, :]
        A_tile = tl.load(A + A_off)

        B_off = N * range_k_block[:, None] + range_n_block
        B_tile = tl.load(B + B_off)

        accum += tl.dot(A_tile, B_tile, allow_tf32=False)
    C_off = N * range_m_block + range_n_block
    tl.store(C + C_off, accum)


def test_gemm():
    M, N, K = 32, 32, 32
    A = torch.randn((M, K))
    B = torch.randn((K, N))
    C = torch.empty((M, N))
    tile_size = 16

    for _ in range(warmup := 10):
        gemm_kernel[(M // tile_size, N // tile_size)](A, B, C, M, N, K, tile_size)

    tic = time.perf_counter()
    for _ in tqdm(range(runs := 100)):
        gemm_kernel[(M // tile_size, N // tile_size)](A, B, C, M, N, K, tile_size)
    print("matmul duration:", time.perf_counter() - tic)


test_gemm()


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

    # First run for assertion
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

    for _ in range(warmup := 10):
        triple_indirect_load_kernel[grid](
            idx1,
            idx2,
            src,
            dst,
            BLOCK_SIZE=32,
        )

    tic = time.perf_counter()
    for _ in tqdm(range(runs := 100)):
        triple_indirect_load_kernel[grid](
            idx1,
            idx2,
            src,
            dst,
            BLOCK_SIZE=32,
        )
    print("triple indirect load duration:", time.perf_counter() - tic)


test_triple_indirect_load("cuda")


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

    # First run for assertion
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

    for _ in range(warmup := 10):
        dual_offset_load_kernel[grid](
            idx_a,
            idx_b,
            src,
            dst,
            BLOCK_SIZE=32,
        )

    tic = time.perf_counter()
    for _ in tqdm(range(runs := 100)):
        dual_offset_load_kernel[grid](
            idx_a,
            idx_b,
            src,
            dst,
            BLOCK_SIZE=32,
        )
    print("dual offset load duration:", time.perf_counter() - tic)


test_dual_offset_load("cuda")
