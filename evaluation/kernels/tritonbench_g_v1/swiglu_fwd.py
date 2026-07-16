
import torch
import triton
import triton.language as tl

# Forward Triton kernel for Swish-Gated Linear Units (Swiglu)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128}),
        triton.Config({'BLOCK_N': 256}),
        triton.Config({'BLOCK_N': 512}),
        triton.Config({'BLOCK_N': 1024}),
    ],
    key=['ncols'],
)
@triton.jit
def _swiglu_fwd_kernel(
    X, Y, OUT, stride_x_row, stride_y_row, stride_out_row, ncols, BLOCK_N: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)

# Function to invoke the forward kernel
def _swiglu_fwd(xy, out=None):
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1, out.shape[-1])
        assert out.shape == x.shape
    assert out.stride(-1) == 1
    M, N = x.shape
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_N']))
    with torch.cuda.device(x.device.index):
        _swiglu_fwd_kernel[grid](x, y, out, x.stride(0), y.stride(0), out.stride(0), N)
    return out.reshape(*batch_shape, out.shape[-1])



##################################################################################################################################################


# Test the forward function with different configurations
def test_swiglu_fwd():
    results = {}
    # Test case 1
    batch_size = 4
    ncols = 128
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_1'] = out.detach().cpu()

    # Test case 2
    batch_size = 8
    ncols = 256
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_2'] = out.detach().cpu()

    # Test case 3
    batch_size = 16
    ncols = 512
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_3'] = out.detach().cpu()

    # Test case 4
    batch_size = 32
    ncols = 1024
    xy = torch.randn(batch_size, 2 * ncols, device='cuda', dtype=torch.float32)
    out = _swiglu_fwd(xy)
    results['test_case_4'] = out.detach().cpu()

    return results

# Run the tests
result_gold = test_swiglu_fwd()
