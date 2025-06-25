import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz import config as cfg

from utils import CaptureSanitizer


cfg.sanitizer_backend = "symexec"


@triton_viz.trace(clients=(san1 := CaptureSanitizer(abort_on_error=True)))
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


@triton_viz.trace(clients=(san2 := CaptureSanitizer(abort_on_error=True)))
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


def test_triple_indirect_load():
    N = 128
    device = "cpu"

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


@triton_viz.trace(clients=(san3 := CaptureSanitizer(abort_on_error=True)))
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


def test_dual_offset_load():
    N = 128
    device = "cpu"

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
