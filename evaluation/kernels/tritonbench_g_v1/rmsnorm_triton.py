import torch

import triton
import triton.language as tl


torch.manual_seed(1234)


@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                )
    return out



##################################################################################################################################################


def test_rmsnorm_triton():
    results = {}
    
    # Case 1
    batch, M, K = 2, 4, 1024
    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')
    eps = 1e-6
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_1'] = out

    # Case 2: Different eps value
    eps = 1e-5
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_2'] = out

    # Case 3: Different batch size
    batch, M, K = 3, 4, 1024
    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')
    eps = 1e-6
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_3'] = out

    # Case 4: Different M size
    batch, M, K = 2, 5, 1024
    x = torch.randn((batch, M, K), dtype=torch.float16, device='cuda')
    rms_w = torch.randn((K,), dtype=torch.float16, device='cuda')
    eps = 1e-6
    out = rmsnorm_triton_wrapper(x, rms_w, eps)
    results['test_case_4'] = out

    return results

result_gold = test_rmsnorm_triton()
