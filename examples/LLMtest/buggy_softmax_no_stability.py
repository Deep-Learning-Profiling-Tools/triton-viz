import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import Tracer

# Optional: visualizer LLM — set one or both before running.
_LL_CONFIG_PATH = ""
_LL_API_KEY = ""


@triton_viz.trace(client=Tracer())
@triton.jit
def softmax_no_stability_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_x = x_ptr + pid * stride_xm
    row_start_y = y_ptr + pid * stride_ym

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(row_start_x + offs * stride_xn, mask=mask, other=-float("inf"))

    # Bug: missing numerical stabilization (x - max(x)).
    ex = tl.exp(x)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    out = ex / denom
    tl.store(row_start_y + offs * stride_yn, out, mask=mask)


def run_demo():
    torch.manual_seed(0)
    rows, cols = 8, 64
    # Large positive values make overflow likely without max-subtraction.
    x = torch.randn((rows, cols), dtype=torch.float32) * 20.0 + 50.0
    y = torch.empty_like(x)

    grid = (rows,)
    block = 64
    softmax_no_stability_kernel[grid](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        cols,
        BLOCK_SIZE=block,
    )

    ref = torch.softmax(x, dim=1)
    finite_ratio = torch.isfinite(y).float().mean().item()
    max_diff = (torch.nan_to_num(y) - ref).abs().max().item()
    print(
        f"[buggy_softmax_no_stability] finite ratio: {finite_ratio:.3f}, max diff: {max_diff:.6f}"
    )

    if _LL_CONFIG_PATH:
        triton_viz.setup_llm(config_path=_LL_CONFIG_PATH)
    if _LL_API_KEY:
        triton_viz.setup_llm(api_key=_LL_API_KEY)
    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()
