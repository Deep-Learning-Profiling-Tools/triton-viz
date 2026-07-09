"""Phase B corpus: the triton 3.6 tutorials, vendored (plan S5).

Kernels are copied verbatim (comments trimmed) from
https://github.com/triton-lang/triton, branch ``release/3.6.x``,
``python/tutorials/{01,02,03,04,05,07}-*.py`` (MIT license). Deviations:
the ``@triton.autotune`` decorator on the matmul kernel is stripped — the
harness pins ONE config per LaunchSpec (the plan's autotune rule) — and
``tl.assume`` calls are kept as-is (result-free ops the reader ignores).

Every launch is labeled race-free: the tutorials are correct code, so the
interesting output is WHERE each kernel lands on the ladder — proofs vs
documented abstention boundaries (persistent grid-stride loops, multiple
sequential loops, the layer-norm lock) — and what the mutation mode does
to the proofs.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from evaluation.spec import Corpus, LaunchSpec

CORPUS = Corpus("tutorials")


# ── 01-vector-add ────────────────────────────────────────────────


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


CORPUS.add(
    LaunchSpec(
        name="tut01_vector_add",
        kernel_fn=add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128},
        make_args=lambda seed: (
            torch.randn(1000, generator=torch.Generator().manual_seed(seed)),
            torch.randn(1000, generator=torch.Generator().manual_seed(seed + 1)),
            torch.zeros(1000),
            1000,
        ),
        grid=(8,),
        expected="race-free",
        pattern="tutorial",
        params_note="01: masked elementwise add, n not a block multiple",
    )
)


# ── 02-fused-softmax (persistent grid-stride loop) ───────────────


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


CORPUS.add(
    LaunchSpec(
        name="tut02_softmax_persistent",
        kernel_fn=softmax_kernel,
        signature={
            "output_ptr": "*fp32",
            "input_ptr": "*fp32",
            "input_row_stride": "i32",
            "output_row_stride": "i32",
            "n_rows": "i32",
            "n_cols": "i32",
            "BLOCK_SIZE": "constexpr",
            "num_stages": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128, "num_stages": 2},
        make_args=lambda seed: (
            torch.zeros(64 * 100),
            torch.randn(64 * 100, generator=torch.Generator().manual_seed(seed)),
            100,
            100,
            64,
            100,
        ),
        grid=(4,),
        expected="race-free",
        pattern="tutorial",
        params_note="02: persistent kernel — the grid-stride loop's bounds "
        "are pid/num_programs, outside the concrete-bound loop model "
        "(expected abstention)",
    )
)


# ── 03-matrix-multiplication (autotune stripped, config pinned) ──


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_MATMUL_SIG = {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "c_ptr": "*fp16",
    "M": "i32",
    "N": "i32",
    "K": "i32",
    "stride_am": "i32",
    "stride_ak": "i32",
    "stride_bk": "i32",
    "stride_bn": "i32",
    "stride_cm": "i32",
    "stride_cn": "i32",
    "BLOCK_SIZE_M": "constexpr",
    "BLOCK_SIZE_N": "constexpr",
    "BLOCK_SIZE_K": "constexpr",
    "GROUP_SIZE_M": "constexpr",
    "ACTIVATION": "constexpr",
}


def _matmul_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    m = n = k = 64
    return (
        torch.randn(m * k, generator=g, dtype=torch.float16),
        torch.randn(k * n, generator=g, dtype=torch.float16),
        torch.zeros(m * n, dtype=torch.float16),
        m,
        n,
        k,
        k,
        1,
        n,
        1,
        n,
        1,
    )


for _name, _act in (
    ("tut03_matmul_grouped", ""),
    ("tut03_matmul_leaky_relu", "leaky_relu"),
):
    CORPUS.add(
        LaunchSpec(
            name=_name,
            kernel_fn=matmul_kernel,
            signature=_MATMUL_SIG,
            constexprs={
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 2,
                "ACTIVATION": _act,
            },
            make_args=_matmul_args,
            grid=(4,),
            expected="race-free",
            pattern="tutorial",
            params_note="03: grouped-swizzle matmul, one autotune config "
            f"pinned (ACTIVATION={_act or 'none'!r})",
        )
    )


# ── 04-low-memory-dropout ────────────────────────────────────────


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


CORPUS.add(
    LaunchSpec(
        name="tut04_seeded_dropout",
        kernel_fn=_seeded_dropout,
        signature={
            "x_ptr": "*fp32",
            "output_ptr": "*fp32",
            "n_elements": "i32",
            "p": "fp32",
            "seed": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128},
        make_args=lambda seed: (
            torch.randn(1000, generator=torch.Generator().manual_seed(seed)),
            torch.zeros(1000),
            1000,
            0.5,
            123,
        ),
        grid=(8,),
        expected="race-free",
        pattern="tutorial",
        params_note="04: philox tl.rand feeds only the VALUE (tl.where), "
        "not the footprint — the store stays provable",
    )
)


# ── 05-layer-norm ────────────────────────────────────────────────


@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


CORPUS.add(
    LaunchSpec(
        name="tut05_layernorm_fwd",
        kernel_fn=_layer_norm_fwd_fused,
        signature={
            "X": "*fp32",
            "Y": "*fp32",
            "W": "*fp32",
            "B": "*fp32",
            "Mean": "*fp32",
            "Rstd": "*fp32",
            "stride": "i32",
            "N": "i32",
            "eps": "fp32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128},
        make_args=lambda seed: (
            torch.randn(8 * 100, generator=torch.Generator().manual_seed(seed)),
            torch.zeros(8 * 100),
            torch.randn(100, generator=torch.Generator().manual_seed(seed + 1)),
            torch.randn(100, generator=torch.Generator().manual_seed(seed + 2)),
            torch.zeros(8),
            torch.zeros(8),
            100,
            100,
            1e-5,
        ),
        grid=(8,),
        expected="race-free",
        pattern="tutorial",
        params_note="05 fwd: three SEQUENTIAL loops over the row — outside "
        "the single-loop model (expected abstention)",
    )
)


@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,
    DY,
    DW,
    DB,
    X,
    W,
    Mean,
    Rstd,
    Lock,
    stride,
    N,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.debug_barrier()
    tl.atomic_xchg(Lock, 0)


CORPUS.add(
    LaunchSpec(
        name="tut05_layernorm_bwd_dx",
        kernel_fn=_layer_norm_bwd_dx_fused,
        signature={
            "DX": "*fp32",
            "DY": "*fp32",
            "DW": "*fp32",
            "DB": "*fp32",
            "X": "*fp32",
            "W": "*fp32",
            "Mean": "*fp32",
            "Rstd": "*fp32",
            "Lock": "*i32",
            "stride": "i32",
            "N": "i32",
            "GROUP_SIZE_M": "constexpr",
            "BLOCK_SIZE_N": "constexpr",
        },
        constexprs={"GROUP_SIZE_M": 4, "BLOCK_SIZE_N": 128},
        make_args=lambda seed: (
            torch.zeros(8 * 100),
            torch.randn(8 * 100, generator=torch.Generator().manual_seed(seed)),
            torch.zeros(4 * 100),
            torch.zeros(4 * 100),
            torch.randn(8 * 100, generator=torch.Generator().manual_seed(seed + 1)),
            torch.randn(100, generator=torch.Generator().manual_seed(seed + 2)),
            torch.randn(8, generator=torch.Generator().manual_seed(seed + 3)),
            torch.ones(8),
            torch.zeros(8, dtype=torch.int32),  # Lock[0:4] + Count[4:8]
            100,
            100,
        ),
        grid=(8,),
        expected="race-free",
        pattern="tutorial",
        params_note="05 bwd stage 1: the CAS spin-lock protects the "
        "grouped dw/db partial buffers — the await abstraction's real-world "
        "shape (Count branch is data-dependent → expected widening)",
    )
)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


CORPUS.add(
    LaunchSpec(
        name="tut05_layernorm_bwd_dwdb",
        kernel_fn=_layer_norm_bwd_dwdb,
        signature={
            "DW": "*fp32",
            "DB": "*fp32",
            "FINAL_DW": "*fp32",
            "FINAL_DB": "*fp32",
            "M": "i32",
            "N": "i32",
            "BLOCK_SIZE_M": "constexpr",
            "BLOCK_SIZE_N": "constexpr",
        },
        constexprs={"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 32},
        make_args=lambda seed: (
            torch.randn(4 * 100, generator=torch.Generator().manual_seed(seed)),
            torch.randn(4 * 100, generator=torch.Generator().manual_seed(seed + 1)),
            torch.zeros(100),
            torch.zeros(100),
            4,
            100,
        ),
        grid=(4,),
        expected="race-free",
        pattern="tutorial",
        params_note="05 bwd stage 2: 2-D tiled reduction loop, per-pid "
        "column stripes",
    )
)


# ── 07-extern-functions (libdevice) ──────────────────────────────


@triton.jit
def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


CORPUS.add(
    LaunchSpec(
        name="tut07_libdevice_asin",
        kernel_fn=asin_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": 128},
        make_args=lambda seed: (
            torch.rand(1000, generator=torch.Generator().manual_seed(seed)),
            torch.zeros(1000),
            1000,
        ),
        grid=(8,),
        expected="race-free",
        pattern="tutorial",
        params_note="07: extern libdevice call in value position",
    )
)
