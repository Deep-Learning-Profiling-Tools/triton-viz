import torch
import triton
import triton.language as tl
import triton_viz
from triton_viz.clients import Tracer


@triton_viz.trace(clients=Tracer())
@triton.jit
def softmax_kernel(
    x_ptr,
    lengths_ptr,
    y_ptr,
    stride_b,
    stride_l,
    B,
    BLOCK_SIZE: tl.constexpr,
    MASK_OFF_BY_ONE: tl.constexpr,
    SKIP_NUM_STABILITY: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * stride_b
    lengths = tl.load(lengths_ptr + pid, mask=pid < B, other=0).to(tl.int32)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < lengths
    if MASK_OFF_BY_ONE:
        mask = offs <= lengths

    x = tl.load(x_ptr + row_start + offs * stride_l, mask=mask, other=-float("inf"))

    if SKIP_NUM_STABILITY:
        shifted = x
    else:
        max_x = tl.max(x, axis=0)
        shifted = x - max_x

    num = tl.exp(shifted)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    denom = tl.where(denom == 0, 1.0, denom)
    out = num / denom
    out = tl.where(tl.math.isfinite(out), out, 0.0)

    tl.store(y_ptr + row_start + offs * stride_l, out, mask=mask)


def run_demo():
    case_name = "Softmax No-Max Bug"
    batch, l_max = 4, 64
    torch.manual_seed(0)
    device = torch.device("cpu")
    # 制造极大范围的数值，禁用 (x - max(x)) 会直接导致 exp 上溢
    base = torch.linspace(-1000.0, 1000.0, steps=l_max, device=device)
    x = torch.stack([base + i * 50 for i in range(batch)], dim=0)
    lengths_inclusive = torch.full((batch,), l_max - 1, dtype=torch.int32, device=device)
    valid_counts = lengths_inclusive + 1
    y = torch.empty_like(x)

    grid = (batch,)
    softmax_kernel[grid](
        x,
        lengths_inclusive,
        y,
        x.stride(0),
        x.stride(1),
        batch,
        BLOCK_SIZE=l_max,
        MASK_OFF_BY_ONE=True,
        SKIP_NUM_STABILITY=True,
    )

    idx = torch.arange(l_max)[None, :]
    mask = idx < valid_counts[:, None]
    ref = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=1).masked_fill(
        ~mask, 0.0
    )
    diff = (y - ref).abs()
    print(f"[{case_name}] max diff: {diff.max().item():.3e}")

    triton_viz.launch(share=True)


if __name__ == "__main__":
    run_demo()

