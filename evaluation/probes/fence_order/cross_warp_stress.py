# Stress: each program stores a tile whose value encodes the launch iteration, then, WITHOUT a barrier,
# loads one element that a thread in a DIFFERENT warp stored, and records what it saw.
import torch
import triton
import triton.language as tl
import sys


@triton.jit
def cross_warp_nofence(buf_ptr, out_ptr, it, N: tl.constexpr, SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    base = pid * N
    tl.store(buf_ptr + base + offs, it + offs * 0)  # every element := it
    v = tl.load(
        buf_ptr + base + ((pid + SHIFT) % N)
    )  # element owned by another thread/warp
    tl.store(out_ptr + pid, v)


@triton.jit
def cross_warp_fence(buf_ptr, out_ptr, it, N: tl.constexpr, SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    base = pid * N
    tl.store(buf_ptr + base + offs, it + offs * 0)
    tl.debug_barrier()
    v = tl.load(buf_ptr + base + ((pid + SHIFT) % N))
    tl.store(out_ptr + pid, v)


def go(kernel, tag, iters, G, N, nw, shift):
    buf = torch.full((G * N,), -1, dtype=torch.int32, device="cuda")
    out = torch.full((G,), -1, dtype=torch.int32, device="cuda")
    stale = 0
    for it in range(1, iters + 1):
        kernel[(G,)](buf, out, it, N=N, SHIFT=shift, num_warps=nw)
        torch.cuda.synchronize()
        bad = (out != it).sum().item()
        stale += bad
    print(
        f"{tag}: N={N} num_warps={nw} grid={G} iters={iters} SHIFT={shift}: stale reads = {stale} / {iters*G}"
    )


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for N, nw, shift in [(64, 4, 33), (4096, 8, 2048), (8192, 32, 4097)]:
        go(cross_warp_nofence, "nofence", iters, 512, N, nw, shift)
    go(cross_warp_fence, "fence  ", iters, 512, 4096, 8, 2048)
