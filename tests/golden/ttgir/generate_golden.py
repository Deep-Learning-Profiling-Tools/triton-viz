"""Dump TTIR/TTGIR for matmul kernels (tl.dot, pipelined) and an elementwise kernel.

Host-only compilation via triton.compile with an explicit GPUTarget (no GPU
needed). Tries sm90 first, falls back to sm80.

Key detail: triton.compile via ASTSource skips the runtime JIT specialization,
so we must pass `attrs={(i,): [["tt.divisibility", 16]]}` for pointer/int args
ourselves, otherwise AxisInfo assumes alignment 1 and the software pipeliner
refuses to emit cp.async / TMA (s1 and s3 then come out identical).
Also: the innermost stride must be a compile-time 1 (don't multiply by a
runtime stride) or contiguity is unknown and vectorized async copies are
impossible.
"""
import os
import sys
import traceback

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,  # inner strides are 1 (row-major)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def matmul_blockptr_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(
        a_ptr, (M, K), (stride_am, 1), (pid_m * BLOCK_M, 0), (BLOCK_M, BLOCK_K), (1, 0)
    )
    b_bp = tl.make_block_ptr(
        b_ptr, (K, N), (stride_bk, 1), (0, pid_n * BLOCK_N), (BLOCK_K, BLOCK_N), (1, 0)
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_bp, boundary_check=(0, 1))
        b = tl.load(b_bp, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c_bp = tl.make_block_ptr(
        c_ptr,
        (M, N),
        (stride_cm, 1),
        (pid_m * BLOCK_M, pid_n * BLOCK_N),
        (BLOCK_M, BLOCK_N),
        (1, 0),
    )
    tl.store(c_bp, acc.to(tl.float16), boundary_check=(0, 1))


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# ----------------------------------------------------------------------------
# Compile helpers
# ----------------------------------------------------------------------------
MATMUL_SIG = {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_bk": "i32",
    "stride_cm": "i32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_K": "constexpr",
}
MATMUL_CONST = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}
# divisibility-16 on the three pointers + M,N,K + strides (mimics real JIT
# specialization of well-aligned tensors / sizes)
MATMUL_ATTRS = {(i,): [["tt.divisibility", 16]] for i in range(9)}

ADD_SIG = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "out_ptr": "*fp32",
    "n_elements": "i32",
    "BLOCK_SIZE": "constexpr",
}
ADD_CONST = {"BLOCK_SIZE": 1024}
ADD_ATTRS = {(i,): [["tt.divisibility", 16]] for i in range(4)}


def dump(tag, fn, sig, consts, attrs, num_stages, num_warps, caps=(90, 80)):
    last_err = None
    for cap in caps:
        target = GPUTarget("cuda", cap, 32)
        src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
        opts = {"num_warps": num_warps, "num_stages": num_stages}
        try:
            k = triton.compile(src, target=target, options=opts)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[{tag}] sm{cap} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue
        for ext in ("ttir", "ttgir"):
            if ext in k.asm:
                path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), f"{tag}_sm{cap}.{ext}"
                )
                with open(path, "w") as f:
                    f.write(k.asm[ext])
                print(f"wrote {path} ({len(k.asm[ext])} bytes)")
        print(f"[{tag}] sm{cap} OK; asm keys: {sorted(k.asm.keys())}")
    if last_err is not None:
        print(f"[{tag}] note: at least one target failed", file=sys.stderr)


if __name__ == "__main__":
    dump(
        "matmul_s3",
        matmul_kernel,
        MATMUL_SIG,
        MATMUL_CONST,
        MATMUL_ATTRS,
        num_stages=3,
        num_warps=4,
    )
    dump(
        "matmul_s1",
        matmul_kernel,
        MATMUL_SIG,
        MATMUL_CONST,
        MATMUL_ATTRS,
        num_stages=1,
        num_warps=4,
    )
    dump(
        "matmul_bp_s3",
        matmul_blockptr_kernel,
        MATMUL_SIG,
        MATMUL_CONST,
        MATMUL_ATTRS,
        num_stages=3,
        num_warps=4,
    )
    dump("add", add_kernel, ADD_SIG, ADD_CONST, ADD_ATTRS, num_stages=3, num_warps=4)
    print("done")
