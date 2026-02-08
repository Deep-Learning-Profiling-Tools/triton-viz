import torch

import triton
import triton.language as tl

import triton_viz
from triton_viz.clients.sanitizer.sanitizer import SymbolicSanitizer


# Shared sanitizer instance (non-aborting so we can inspect records)
block_sanitizer = SymbolicSanitizer(abort_on_error=False)


# ======== 1D Load Kernels ===========


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_load_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    tl.load(block_ptr, boundary_check=(0,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_load_oob_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # No boundary_check -> OOB is undefined behavior
    tl.load(block_ptr)


# ======== 1D Store Kernels ===========


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_store_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
    tl.store(block_ptr, val, boundary_check=(0,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_1d_store_oob_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    val = tl.full((BLOCK_SIZE,), value=1.0, dtype=tl.float32)
    # No boundary_check -> OOB is undefined behavior
    tl.store(block_ptr, val)


# ======== 2D Load Kernels ===========


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_load_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.load(block_ptr, boundary_check=(0, 1))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_load_oob_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # No boundary_check -> OOB is undefined behavior
    tl.load(block_ptr)


# ======== Boundary Check Masking Kernel ===========


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_boundary_check_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Block extends past tensor end but boundary_check masks it."""
    pid = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # boundary_check=(0,) means hardware masks OOB positions
    tl.load(block_ptr, boundary_check=(0,))


# ======== Loop + Advance Kernels ===========


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_loop_advance_kernel(ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    for _ in range(N // BLOCK_SIZE):
        tl.load(block_ptr, boundary_check=(0,))
        block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_loop_advance_oob_kernel(
    ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    # One extra iteration -> OOB, NO boundary_check
    for _ in range(N // BLOCK_SIZE + 1):
        tl.load(block_ptr)
        block_ptr = tl.advance(block_ptr, (BLOCK_SIZE,))


@triton_viz.trace(client=block_sanitizer)
@triton.jit
def block_tensor_2d_loop_advance_kernel(
    ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D block pointer advanced along K (N) dimension in a loop."""
    pid_m = tl.program_id(0)
    block_ptr = tl.make_block_ptr(
        base=ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    for _ in range(N // BLOCK_N):
        tl.load(block_ptr, boundary_check=(0, 1))
        block_ptr = tl.advance(block_ptr, (0, BLOCK_N))


# ======== Non-OOB Tests ===========


def test_block_tensor_1d_load_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (N // BLOCK_SIZE,)
    block_tensor_1d_load_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_1d_store_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.zeros(N, dtype=torch.float32)
    grid = (N // BLOCK_SIZE,)
    block_tensor_1d_store_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_2d_load_non_oob():
    block_sanitizer.records.clear()
    M, N = 4, 4
    BLOCK_M, BLOCK_N = 2, 2
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    grid = (M // BLOCK_M, N // BLOCK_N)
    block_tensor_2d_load_kernel[grid](data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_boundary_check_masks_oob():
    block_sanitizer.records.clear()
    # N=48 is not divisible by BLOCK_SIZE=32, so last block extends past end.
    # boundary_check=(0,) means hardware masks the OOB positions.
    N = 48
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (2,)  # 2 blocks: [0,32) and [32,64) -- second extends past N=48
    block_tensor_boundary_check_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records with boundary_check, got {len(block_sanitizer.records)}"


# ======== OOB Tests ===========


def test_block_tensor_1d_load_oob():
    block_sanitizer.records.clear()
    N = 48
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    grid = (2,)  # Second block [32,64) goes past N=48, no boundary_check
    block_tensor_1d_load_oob_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_1d_store_oob():
    block_sanitizer.records.clear()
    N = 48
    BLOCK_SIZE = 32
    data = torch.zeros(N, dtype=torch.float32)
    grid = (2,)  # Second block [32,64) goes past N=48, no boundary_check
    block_tensor_1d_store_oob_kernel[grid](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_2d_load_oob():
    block_sanitizer.records.clear()
    M, N = 4, 4
    BLOCK_M, BLOCK_N = 2, 2
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    # 3x2 grid but matrix is only 4x4 with BLOCK_M=2 -> pid_m=2 goes OOB
    grid = (3, 2)  # pid_m in [0,3), but M//BLOCK_M=2, so pid_m=2 is OOB
    block_tensor_2d_load_oob_kernel[grid](
        data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


# ======== Loop + Advance Tests ===========


def test_block_tensor_loop_advance_non_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    block_tensor_loop_advance_kernel[(1,)](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"


def test_block_tensor_loop_advance_oob():
    block_sanitizer.records.clear()
    N = 64
    BLOCK_SIZE = 32
    data = torch.randn(N, dtype=torch.float32)
    block_tensor_loop_advance_oob_kernel[(1,)](data, N=N, BLOCK_SIZE=BLOCK_SIZE)
    assert len(block_sanitizer.records) > 0, "Expected OOB to be detected"


def test_block_tensor_2d_loop_advance_non_oob():
    block_sanitizer.records.clear()
    M, N = 4, 8
    BLOCK_M, BLOCK_N = 4, 4
    data = torch.randn(M, N, dtype=torch.float32).contiguous()
    # 1 program, loops N//BLOCK_N=2 times
    block_tensor_2d_loop_advance_kernel[(1,)](
        data, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    assert (
        len(block_sanitizer.records) == 0
    ), f"Expected no OOB records, got {len(block_sanitizer.records)}"
