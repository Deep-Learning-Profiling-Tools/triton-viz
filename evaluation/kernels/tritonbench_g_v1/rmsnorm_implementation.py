
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # parallel at m dimension
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    # parallel between blocks
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)  # careful stride_x_k
        xf = x.to(tl.float32)
        var += xf*xf
    var = tl.sum(var, axis=0) / N_SIZE  # reduce between wrap
    std = tl.sqrt(var + eps)

    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE

        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)

        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)


def rmsnorm_wrapper(x, rms_weights, eps=1e-6):
    batch, M, K = x.shape
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_weights, out,
                                *x.stride(),
                                *rms_weights.stride(),  # 1
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=4096,
                                num_warps=16
                                )
    return out




##################################################################################################################################################


def test_rmsnorm():
    # Define the input tensor x with shape (batch, M, K)
    batch = 2
    M = 3
    K = 4096
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")

    # Define the rms_weights tensor with shape (K,)
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")

    # Dictionary to store the results of different test cases
    results = {}

    # Test case 1
    out1 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_1'] = out1.cpu()

    # Additional test cases for branch coverage

    # Test case 2: Different batch size
    batch = 4
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out2 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_2'] = out2.cpu()

    # Test case 3: Different M size
    M = 5
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out3 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_3'] = out3.cpu()

    # Test case 4: Larger K size
    K = 8192
    rms_weights = torch.randn((K,), dtype=torch.float16, device="cuda")
    x = torch.randn((batch, M, K), dtype=torch.float16, device="cuda")
    out4 = rmsnorm_wrapper(x, rms_weights)
    results['test_case_4'] = out4.cpu()

    return results

# Execute the test function
result_gold = test_rmsnorm()
