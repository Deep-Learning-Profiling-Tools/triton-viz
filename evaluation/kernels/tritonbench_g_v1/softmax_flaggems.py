import logging

import torch
import triton
import triton.language as tl

# from ..utils import libentry

MAX_TILE_K = 8192
NUM_SMS = torch.cuda.get_device_properties(
    torch.cuda.current_device()
).multi_processor_count


def heur_tile_k(args):
    tile_k = 1
    upper_bound = min(args["K"], MAX_TILE_K)
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        num_waves = num_blocks / NUM_SMS
        if (num_waves > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def heur_tile_n_non_inner(args):
    return triton.cdiv(8192, args["TILE_K"])


def heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def heur_num_warps_non_inner(args):
    tile_size = args["TILE_N"] * args["TILE_K"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "TILE_K": heur_tile_k,
        "TILE_N": heur_tile_n_non_inner,
        "ONE_TILE_PER_CTA": heur_one_tile_per_cta,
        "num_warps": heur_num_warps_non_inner,
    }
)
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            alpha = tl.exp(m - m_new)
            z = z * alpha + tl.exp(inp - m_new)
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


def heur_tile_n_inner(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    else:
        return 4096


def heur_num_warps_inner(args):
    tile_size = args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


@triton.heuristics(
    {
        "TILE_N": heur_tile_n_inner,
        "ONE_TILE_PER_CTA": heur_one_tile_per_cta,
        "num_warps": heur_num_warps_inner,
    }
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            z = z * tl.exp(m - m_new) + tl.exp(inp - m_new)
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


def heur_tile_n_bwd_non_inner(args):
    return max(1, 1024 // args["TILE_K"])


# ------------------------  backward -------------------------------

@triton.autotune(
    configs=[
        triton.Config({"TILE_K": 32}),
        triton.Config({"TILE_K": 64}),
        triton.Config({"TILE_K": 128}),
        triton.Config({"TILE_K": 256}),
        triton.Config({"TILE_K": 512}),
        triton.Config({"TILE_K": 1024}),
    ],
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(
    {
        "TILE_N": heur_tile_n_bwd_non_inner,
        "ONE_TILE_PER_CTA": heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


def heru_tile_m(args):
    return max(1, 1024 // args["TILE_N"])



@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 32}),
        triton.Config({"TILE_N": 64}),
        triton.Config({"TILE_N": 128}),
        triton.Config({"TILE_N": 256}),
        triton.Config({"TILE_N": 512}),
        triton.Config({"TILE_N": 1024}),
    ],
    key=["M", "N"],
)
@triton.heuristics(
    values={
        "TILE_M": heru_tile_m,
        "ONE_TILE_PER_CTA": heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logging.debug("GEMS SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]  # pre_dim
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N  # post_dim

        with torch.cuda.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                softmax_kernel_non_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                    K,
                )
            else:
                grid = (M, 1, 1)
                softmax_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logging.debug("GEMS SOFTMAX VJP")
        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        with torch.cuda.device(in_grad.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                softmax_backward_kernel_non_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                    K,
                )
            else:
                grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
                softmax_backward_kernel_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                )
        return in_grad, None, None


def softmax(x, dim=-1, dtype=None):
    return Softmax.apply(x, dim, dtype)




##################################################################################################################################################


def test_softmax():
    # 创建一个字典用于保存每个分支的结果
    result = {}

    # Test case 1: 1D tensor, float32, default dim=-1
    x_1d = torch.rand((10,), device='cuda', dtype=torch.float32)
    out_1d = softmax(x_1d)
    result["test_case_1"] = out_1d

    # Test case 2: 2D tensor, float32, dim=1
    x_2d = torch.rand((4, 5), device='cuda', dtype=torch.float32)
    out_2d = softmax(x_2d, dim=1)
    result["test_case_2"] = out_2d

    # Test case 3: 2D tensor, float16, dim=0
    x_2d_fp16 = torch.rand((4, 5), device='cuda', dtype=torch.float16)
    out_2d_fp16 = softmax(x_2d_fp16, dim=0)
    result["test_case_3"] = out_2d_fp16

    # Test case 4: 3D tensor, float32, default dim=-1
    x_3d = torch.rand((2, 3, 4), device='cuda', dtype=torch.float32)
    out_3d = softmax(x_3d)
    result["test_case_4"] = out_3d

    # Test case 5: 3D tensor, float64, dim=1
    x_3d_fp64 = torch.rand((2, 3, 4), device='cuda', dtype=torch.float64)
    out_3d_fp64 = softmax(x_3d_fp64, dim=1)
    result["test_case_5"] = out_3d_fp64

    # Test case 6: 4D tensor, float32, with large K dimension
    x_4d_large_k = torch.rand((2, 3, 4, 1024), device='cuda', dtype=torch.float32)
    out_4d_large_k = softmax(x_4d_large_k, dim=-1)
    result["test_case_6"] = out_4d_large_k

    # Test case 7: Single-element tensor, float32
    x_single = torch.tensor([1.0], device='cuda', dtype=torch.float32)
    out_single = softmax(x_single)
    result["test_case_7"] = out_single

    # Test case 8: Large tensor, float32, with large N dimension
    x_large = torch.rand((1024, 1024), device='cuda', dtype=torch.float32)
    out_large = softmax(x_large, dim=1)
    result["test_case_8"] = out_large

    # Test case 9: Tensor with Inf and -Inf values, checking numerical stability
    x_inf = torch.tensor([float('inf'), -float('inf')], device='cuda', dtype=torch.float32)
    out_inf = softmax(x_inf)
    result["test_case_9"] = out_inf

    # Test case 10: Tensor with NaN values, checking if the output is NaN
    x_nan = torch.tensor([float('nan')], device='cuda', dtype=torch.float32)
    out_nan = softmax(x_nan)
    result["test_case_10"] = out_nan

    # Test case 11: Tensor with specific shape (non-square), float32, dim=-1
    x_shape1 = torch.rand((3, 7), device='cuda', dtype=torch.float32)
    out_shape1 = softmax(x_shape1)
    result["test_case_11"] = out_shape1

    # Test case 12: Tensor with small shape, float16, checking precision and sum
    x_small_fp16 = torch.rand((2, 2), device='cuda', dtype=torch.float16)
    out_small_fp16 = softmax(x_small_fp16)
    result["test_case_12"] = out_small_fp16

    # Test case 13: Large tensor with float16, checking performance and sum
    x_large_fp16 = torch.rand((512, 512), device='cuda', dtype=torch.float16)
    out_large_fp16 = softmax(x_large_fp16)
    result["test_case_13"] = out_large_fp16

    # Test case 14: Tensor with extreme values, checking overflow handling
    x_extreme = torch.tensor([1e5, -1e5], device='cuda', dtype=torch.float32)
    out_extreme = softmax(x_extreme)
    result["test_case_14"] = out_extreme

    # Test case 15: Very large tensor with float32, testing memory and performance
    x_very_large = torch.rand((2048, 2048), device='cuda', dtype=torch.float32)
    out_very_large = softmax(x_very_large, dim=1)
    result["test_case_15"] = out_very_large

    return result

# 执行测试
result_gold = test_softmax()
