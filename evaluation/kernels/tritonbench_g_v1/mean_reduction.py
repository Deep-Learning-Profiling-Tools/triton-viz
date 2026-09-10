import torch
import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] 
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)

def dim_compress(inp: torch.Tensor, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()

def mean_dim(x, dim, keepdim=False, *, dtype=None):
  if dtype is None:
    dtype = x.dtype
  
  shape = list(x.shape)
  if isinstance(dim, int):
     dim = [dim]
  dim = [d % x.ndim for d in dim]
  x = dim_compress(x, dim)
  N = 1
  for i in dim:
    N *= shape[i]
    shape[i] = 1
  M = x.numel() // N
  out = torch.empty(shape, dtype=dtype, device=x.device)
  grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

  with torch.cuda.device(x.device):
    mean_dim_kernel[grid](x, out, M, N, BLOCK_M=8, BLOCK_N=8)
  if not keepdim:
    out = out.squeeze(dim)
  return out



##################################################################################################################################################


import torch

def test_mean_dim():
    results = {}

    # Test case 1: Single reduction dimension
    b1 = torch.randn(2, 3, 4, 5, device="cuda")
    triton_result1 = mean_dim(b1, 1)
    results['test_case_1'] = triton_result1

    # Test case 2: Multiple reduction dimensions
    b2 = torch.randn(2, 3, 4, 5, device="cuda")
    triton_result2 = mean_dim(b2, [1, 2])
    results['test_case_2'] = triton_result2

    # Test case 3: Keep dimensions
    b3 = torch.randn(2, 3, 4, 5, device="cuda")
    triton_result3 = mean_dim(b3, [1, 2], keepdim=True)
    results['test_case_3'] = triton_result3

    # Test case 4: Different data type
    b4 = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float64)
    triton_result4 = mean_dim(b4, [1, 2], dtype=torch.float32)
    results['test_case_4'] = triton_result4

    return results

result_gold = test_mean_dim()
