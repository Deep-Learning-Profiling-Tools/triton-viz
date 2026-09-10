# Empirical check: is the stale (pre-store) value ever observed by the phase-2 load?
import torch
import triton
import triton.language as tl
import sys


@triton.jit
def fig1(hist_ptr, out_ptr, N: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


@triton.jit
def fig1_bar(hist_ptr, out_ptr, N: tl.constexpr):
    p = tl.program_id(0)
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    tl.debug_barrier()
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


# single-CTA variant: p is a runtime scalar, grid=(1,), isolates the intra-CTA pair
@triton.jit
def fig1_one(hist_ptr, out_ptr, p, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


@triton.jit
def fig1_one_bar(hist_ptr, out_ptr, p, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(hist_ptr + offs, offs)
    tl.debug_barrier()
    n = tl.load(hist_ptr + p)
    tl.store(out_ptr + p, n)


def trial_grid(fn, N, w, iters):
    hist = torch.empty(N, dtype=torch.int32, device="cuda")
    out = torch.empty(N, dtype=torch.int32, device="cuda")
    ref = torch.arange(N, dtype=torch.int32, device="cuda")
    bad_launches = 0
    bad_slots = 0
    worst = None
    for _ in range(iters):
        hist.fill_(-1)
        out.fill_(-7)
        fn[(N,)](hist, out, N=N, num_warps=w)
        m = out != ref
        c = int(m.sum())
        if c:
            bad_launches += 1
            bad_slots += c
            if worst is None or c > worst[0]:
                worst = (
                    c,
                    torch.nonzero(m).flatten()[:8].tolist(),
                    out[m][:8].tolist(),
                )
    return bad_launches, bad_slots, worst


def trial_one(fn, N, w, p, iters):
    hist = torch.empty(N, dtype=torch.int32, device="cuda")
    out = torch.empty(N, dtype=torch.int32, device="cuda")
    bad = 0
    vals = set()
    for _ in range(iters):
        hist.fill_(-1)
        out.fill_(-7)
        fn[(1,)](hist, out, p, N=N, num_warps=w)
        v = int(out[p])
        if v != p:
            bad += 1
            vals.add(v)
    return bad, sorted(vals)


iters = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
torch.cuda.synchronize()
for N, w in ((128, 4), (1024, 4), (1024, 8)):
    for fn in (fig1, fig1_bar):
        bl, bs, worst = trial_grid(fn, N, w, iters)
        print(
            f"grid=({N},) N={N} warps={w} {fn.fn.__name__:9s}: launches_with_stale={bl}/{iters} stale_slots={bs} worst={worst}"
        )
for N, w, p in (
    (128, 4, 127),
    (128, 4, 100),
    (128, 4, 0),
    (1024, 4, 1000),
    (1024, 4, 0),
    (1024, 4, 513),
):
    for fn in (fig1_one, fig1_one_bar):
        bad, vals = trial_one(fn, N, w, p, iters)
        print(
            f"grid=(1,)  N={N} warps={w} p={p} {fn.fn.__name__:13s}: stale_reads={bad}/{iters} stale_values={vals}"
        )
