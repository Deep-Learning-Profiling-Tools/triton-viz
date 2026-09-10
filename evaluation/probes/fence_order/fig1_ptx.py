# Figure 1 (fig:aiter) of the TileRace paper, written as a real Triton kernel.
#   offs = range(0, N); store(hist + offs, offs); n = load(hist + p); store(out + p, n)
import os
import re
import torch
import triton
import triton.language as tl


@triton.jit
def fig1(hist_ptr, out_ptr, N: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)  # phase 1: full histogram
    n = tl.load(hist_ptr + p)  # phase 2: own slot
    tl.store(out_ptr + p, n)


@triton.jit
def fig1_bar(hist_ptr, out_ptr, N: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    tl.debug_barrier()
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


OUT = os.path.dirname(os.path.abspath(__file__))


def compile_and_dump(fn, name, N, num_warps):
    hist = torch.zeros(N, dtype=torch.int32, device="cuda")
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    k = fn[(N,)](hist, out, N=N, num_warps=num_warps)
    tag = f"{name}_N{N}_w{num_warps}"
    for key in ("ttir", "ttgir", "llir", "ptx"):
        with open(os.path.join(OUT, f"{tag}.{key}"), "w") as f:
            f.write(k.asm[key])
    ptx = k.asm["ptx"]
    # report the order of memory / barrier instructions in the PTX body
    body = ptx[ptx.index("{", ptx.index(".visible .entry")) :]
    lines = body.splitlines()
    interesting = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if re.match(
            r"(@%p\d+\s+)?(st\.|ld\.|bar\.|barrier|membar|fence|atom\.|red\.|ret)", s
        ):
            interesting.append((i, s))
    print(
        f"=== {tag}: sm_{k.metadata.target.arch if hasattr(k.metadata,'target') else '?'}; num_warps={k.metadata.num_warps}"
    )
    for i, s in interesting:
        print(f"  ptx:{i}: {s}")
    return k


print(
    "triton",
    triton.__version__,
    "torch",
    torch.__version__,
    "cap",
    torch.cuda.get_device_capability(),
)
for N in (32, 128, 1024):
    for w in (1, 4):
        compile_and_dump(fig1, "fig1", N, w)
compile_and_dump(fig1_bar, "fig1_bar", 128, 4)
compile_and_dump(fig1_bar, "fig1_bar", 1024, 4)
