
import math
import torch
import triton
import triton.language as tl


# TODO: autotune this better.
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = tl.extra.cuda.libdevice.llrint(127.0 * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise(x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs




##################################################################################################################################################


def test_quantize_rowwise():
    results = {}

    # Test case 1: Small 2D tensor
    x1 = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], device='cuda')
    output1, output_maxs1 = quantize_rowwise(x1)
    results['test_case_1'] = (output1, output_maxs1)

    # # Test case 2: Larger 2D tensor
    # x2 = torch.randn(4, 8, device='cuda')
    # output2, output_maxs2 = quantize_rowwise(x2)
    # results['test_case_2'] = (output2, output_maxs2)

    # Test case 3: Tensor with zeros
    x3 = torch.zeros(2, 5, device='cuda')
    output3, output_maxs3 = quantize_rowwise(x3)
    results['test_case_3'] = (output3, output_maxs3)

    return results

# Run the test function
result_gold = test_quantize_rowwise()
