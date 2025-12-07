import argparse
import importlib
import time

import torch


def benchmark_flash_attn(variant: str = "slow",
                         B: int = 4,
                         H: int = 16,
                         S: int = 1024,
                         D: int = 128,
                         warmup: int = 20,
                         iters: int = 50,
                         device: str = "cuda"):
    assert variant in ("slow", "fixed")
    mod_name = f"flash_attn_triton_{variant}"
    fa = importlib.import_module(mod_name)

    # 简单 sanity check：模块里应该有 _flash_attn_forward / _flash_attn_backward
    assert hasattr(fa, "_flash_attn_forward")
    assert hasattr(fa, "_flash_attn_backward")

    torch.manual_seed(0)

    q = torch.randn(B, S, H, D, device=device, dtype=torch.float16, requires_grad=False)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    do = torch.randn_like(q)

    # forward 得到 o, lse, softmax_scale
    o, lse, softmax_scale = fa._flash_attn_forward(
        q, k, v, bias=None, causal=False, softmax_scale=None
    )

    # 预分配梯度张量（backward 会写入 dq/dk/dv）
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # warmup
    for _ in range(warmup):
        fa._flash_attn_backward(
            do, q, k, v, o, lse, dq, dk, dv,
            bias=None, causal=False, softmax_scale=softmax_scale
        )
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        fa._flash_attn_backward(
            do, q, k, v, o, lse, dq, dk, dv,
            bias=None, causal=False, softmax_scale=softmax_scale
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        times_ms.append((end - start) * 1e3)

    avg = sum(times_ms) / len(times_ms)
    p90 = sorted(times_ms)[int(0.9 * len(times_ms))]

    print(f"[variant={variant}] B={B}, H={H}, S={S}, D={D}")
    print(f"  mean bwd time: {avg:.3f} ms")
    print(f"  p90  bwd time: {p90:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["slow", "fixed"], default="slow")
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--S", type=int, default=1024)
    parser.add_argument("--D", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError("CUDA not available")

    benchmark_flash_attn(
        variant=args.variant,
        B=args.B,
        H=args.H,
        S=args.S,
        D=args.D,
    )


if __name__ == "__main__":
    main()
