"""Independent primitive controls and fused holdouts for the GPU pilot.

All launch choices are explicit. There is no autotuning or target-assembly read.
"""

import triton
import triton.language as tl


@triton.jit
def elementwise(
    X, Y, N: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, REPEAT: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0)
    for _ in range(REPEAT):
        if MODE == 1:
            x = x * x + 0.01
        elif MODE == 2:
            x = tl.exp(x * 0.01)
        elif MODE == 3:
            x = x / (1.0 + tl.exp(-x))
        elif MODE == 4:
            x = tl.maximum(x * 1.7 + 0.1, 0.0)
    tl.store(Y + offset, x, offset < N)


@triton.jit
def reduction(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * BLOCK + col)
    if MODE == 0:
        result = tl.sum(x, 0)
        tl.store(Y + row, result)
    elif MODE == 1:
        result = tl.max(x, 0)
        tl.store(Y + row, result)
    elif MODE == 2:
        z = tl.exp(x - tl.max(x, 0))
        result = z / tl.sum(z, 0)
        tl.store(Y + row * BLOCK + col, result)
    else:
        result = x * tl.rsqrt(tl.sum(x * x, 0) / BLOCK + 1e-5)
        tl.store(Y + row * BLOCK + col, result)


@triton.jit
def dot(X, Y, Z, BLOCK: tl.constexpr, REPEAT: tl.constexpr, BIAS: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    j = tl.arange(0, BLOCK)
    offsets = pid * BLOCK * BLOCK + i[:, None] * BLOCK + j[None, :]
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    acc = tl.full((BLOCK, BLOCK), 0, tl.float32)
    for _ in range(REPEAT):
        acc = tl.dot(x, y, acc)
    if BIAS:
        acc = tl.maximum(acc + 0.1, 0.0)
    tl.store(Z + offsets, acc)


def cases(role):
    if role == "control":
        result = []
        for n in (4096, 16384, 65536, 262144):
            for mode, repeat in ((0, 1), (1, 1), (1, 8), (2, 1), (2, 8)):
                result.append(
                    dict(kind="elementwise", n=n, block=512, mode=mode, repeat=repeat)
                )
            for mode in (0, 1):
                result.append(
                    dict(kind="reduction", n=n, block=512, mode=mode, repeat=1)
                )
            for repeat in (1, 4):
                result.append(dict(kind="dot", n=n, block=16, mode=0, repeat=repeat))
    elif role == "holdout":
        result = []
        for n in (8192, 32768, 131072):
            for mode in (3, 4):
                result.append(
                    dict(kind="elementwise", n=n, block=512, mode=mode, repeat=1)
                )
            for mode in (2, 3):
                result.append(
                    dict(kind="reduction", n=n, block=512, mode=mode, repeat=1)
                )
            result.append(dict(kind="dot", n=n, block=16, mode=1, repeat=2))
    else:
        raise ValueError("Unknown artifact role")
    for case in result:
        case["id"] = "_".join(
            str(case[key]) for key in ("kind", "n", "block", "mode", "repeat")
        )
        case["cv_group"] = str(case["n"])
    return result


def prepare(case, device):
    import torch

    n, block, mode, repeat = (case[k] for k in ("n", "block", "mode", "repeat"))
    # Deterministic bounded inputs avoid overflow in arithmetic-chain controls.
    x = torch.full((n,), 0.1, dtype=torch.float32, device=device)
    if case["kind"] == "dot":
        x = x.to(torch.float16)
        y = x.clone()
        out = torch.empty(n, dtype=torch.float32, device=device)
        return dot, (n // (block * block),), (x, y, out, block, repeat, bool(mode)), out
    out = torch.empty(n, dtype=torch.float32, device=device)
    if case["kind"] == "elementwise":
        return (
            elementwise,
            (triton.cdiv(n, block),),
            (x, out, n, block, mode, repeat),
            out,
        )
    return reduction, (n // block,), (x, out, n, block, mode), out


def check_output(case, output):
    import torch

    mode, repeat, block = (case[k] for k in ("mode", "repeat", "block"))
    if case["kind"] == "dot":
        value = float(torch.tensor(0.1, dtype=torch.float16)) ** 2 * block * repeat
        if mode:
            value = max(value + 0.1, 0)
    elif case["kind"] == "elementwise":
        value = torch.tensor(0.1)
        for _ in range(repeat):
            if mode == 1:
                value = value * value + 0.01
            elif mode == 2:
                value = torch.exp(value * 0.01)
            elif mode == 3:
                value = value / (1 + torch.exp(-value))
            elif mode == 4:
                value = torch.maximum(value * 1.7 + 0.1, torch.tensor(0.0))
    else:
        value = (0.1 * block, 0.1, 1 / block, 0.1 / (0.01 + 1e-5) ** 0.5)[mode]
        if mode < 2:
            output = output[: case["n"] // block]
    torch.testing.assert_close(
        output.cpu(), torch.full_like(output.cpu(), float(value)), rtol=2e-3, atol=2e-4
    )
