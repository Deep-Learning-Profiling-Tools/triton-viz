"""RQ4 tier 2: the ORIGINAL kernel behind the aiter#3091 distillation.

``vpopc`` and ``_sum_bitmatrix_rows_fused`` are vendored verbatim from
ROCm/aiter (MIT license),
``aiter/ops/triton/_triton_kernels/moe/moe_routing/bitmatrix.py``.

The racy row launches the kernel the way ``_combined_routing_fused``
effectively executes it: on EVERY program instance, with no pid
partitioning of ``Ret`` (the fused caller inlines this body per pid and
only then reads ``ExpertHist + pid``). The control row is the
contract-respecting single-instance launch. Running the original (rather
than the distillation) is what exposed the unread-axis grid-pinning
soundness bug: the distillation's phase-2 ``tl.load(hist + pid)`` marks
the pid axis used and masked the class.
"""

import torch
import triton
import triton.language as tl

from evaluation.spec import Corpus, LaunchSpec

CORPUS = Corpus("aiter_originals")

N_BLKS = 2
BLOCK_M = 64


@triton.jit
def vpopc(x):
    """Vertical popcount (vendored from ROCm/aiter; credits: @apgoucher)."""
    tl.static_assert(
        x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers"
    )
    BLOCK_N: tl.constexpr = x.shape[-1]
    BATCHES: tl.constexpr = x.numel // BLOCK_N
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0F0F0F0F
    y = tl.sum(y, 2)
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000FF
    y = tl.sum(y, 2)
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_rows_fused(
    B,
    shape_bm,
    stride_bm,
    stride_bn,
    Ret,
    N_BLKS_BITMATRIX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    for i in tl.static_range(N_BLKS_BITMATRIX):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = i * 32 + tl.arange(0, 32)
        n_rows = shape_bm
        if EVEN_M:
            bits = tl.load(B + i * stride_bn + offs_m * stride_bm)
        else:
            bits = tl.load(
                B + i * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0
            )
        bits = tl.reshape(bits, [1, BLOCK_M])
        ret = vpopc(bits)
        ret = tl.reshape(ret, [32])
        tl.store(Ret + offs_n, ret)


_SIG = {
    "B": "*u32",
    "shape_bm": "i32",
    "stride_bm": "i32",
    "stride_bn": "i32",
    "Ret": "*u32",
    "N_BLKS_BITMATRIX": "constexpr",
    "BLOCK_M": "constexpr",
    "EVEN_M": "constexpr",
}
_CEXPR = {"N_BLKS_BITMATRIX": N_BLKS, "BLOCK_M": BLOCK_M, "EVEN_M": True}


def _args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    b = torch.randint(
        0, 2**31 - 1, (BLOCK_M, N_BLKS), dtype=torch.int32, generator=g
    ).view(torch.uint32)
    ret = torch.zeros(32 * N_BLKS, dtype=torch.int32).view(torch.uint32)
    return (b, BLOCK_M, b.stride(0), b.stride(1), ret)


CORPUS.add(
    LaunchSpec(
        name="aiter_sum_bitmatrix_fused_ctx_yes",
        kernel_fn=_sum_bitmatrix_rows_fused,
        signature=_SIG,
        constexprs=_CEXPR,
        make_args=_args,
        grid=(4,),
        expected="race",
        race_pair=("tl.store(Ret + offs_n, ret)",),
        pattern="aiter-3091-original",
        params_note="launched as the fused caller executes it: every pid",
    )
)
CORPUS.add(
    LaunchSpec(
        name="aiter_sum_bitmatrix_standalone_no",
        kernel_fn=_sum_bitmatrix_rows_fused,
        signature=_SIG,
        constexprs=_CEXPR,
        make_args=_args,
        grid=(1,),
        expected="race-free",
        pattern="aiter-3091-original",
        params_note="the contract-respecting single-instance launch",
    )
)
