"""Smoke corpus: one representative per terminal state, to validate the
harness end to end. The real labeled corpus (Phase A, "TritonRaceBench")
follows the same shape at ~15 yes/no pairs."""

import torch
import triton
import triton.language as tl

from evaluation.spec import Corpus, LaunchSpec

CORPUS = Corpus("golden_smoke")


# ── proved@T0: folded-constant stride, disjoint per-pid footprints ──
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _add_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(4096, generator=g),
        torch.randn(4096, generator=g),
        torch.zeros(4096),
        4096,
    )


CORPUS.add(
    LaunchSpec(
        name="smoke_add_no",
        kernel_fn=add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },  # fmt: skip
        constexprs={"BLOCK": 1024},
        make_args=_add_args,
        grid=(4,),
        expected="race-free",
        pattern="elementwise-disjoint",
    )
)


# ── race-confirmed: every block stores the same fixed range ──
@triton.jit
def bcast_store_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v)


def _bcast_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(256, generator=g), torch.zeros(64))


CORPUS.add(
    LaunchSpec(
        name="smoke_bcast_store_yes",
        kernel_fn=bcast_store_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 64},
        make_args=_bcast_args,
        grid=(4,),
        expected="race",
        pattern="fixed-range-store",
    )
)


# ── data-dependent mask: SAME kernel, label flips with the flag data ──
@triton.jit
def dd_mask_kernel(flag_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    keep = tl.load(flag_ptr + offs) > 0
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v, mask=keep)


_DD_SIG = {
    "flag_ptr": "*i32",
    "x_ptr": "*fp32",
    "out_ptr": "*fp32",
    "BLOCK": "constexpr",
}


def _dd_args(flag: int):
    def make(seed: int) -> tuple:
        g = torch.Generator().manual_seed(seed)
        return (
            torch.full((64,), flag, dtype=torch.int32),
            torch.randn(256, generator=g),
            torch.zeros(64),
        )

    return make


CORPUS.add(
    LaunchSpec(
        name="smoke_dd_mask_live_yes",
        kernel_fn=dd_mask_kernel,
        signature=_DD_SIG,
        constexprs={"BLOCK": 64},
        make_args=_dd_args(1),
        grid=(4,),
        expected="race",
        pattern="data-dependent-mask",
        params_note="flags all ones: the dropped mask is really live",
    )
)
CORPUS.add(
    LaunchSpec(
        name="smoke_dd_mask_dead_no",
        kernel_fn=dd_mask_kernel,
        signature=_DD_SIG,
        constexprs={"BLOCK": 64},
        make_args=_dd_args(0),
        grid=(4,),
        expected="race-free",
        pattern="data-dependent-mask",
        params_note="flags all zero: the store never executes",
    )
)


# ── unsupported (indirect-address): gather ──
@triton.jit
def gather_kernel(idx_ptr, src_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    vals = tl.load(src_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, vals, mask=mask)


def _gather_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 256, (256,), dtype=torch.int32, generator=g),
        torch.randn(256, generator=g),
        torch.zeros(256),
        256,
    )


CORPUS.add(
    LaunchSpec(
        name="smoke_gather_no",
        kernel_fn=gather_kernel,
        signature={
            "idx_ptr": "*i32",
            "src_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
            "BLOCK": "constexpr",
        },  # fmt: skip
        constexprs={"BLOCK": 256},
        make_args=_gather_args,
        grid=(1,),
        expected="race-free",
        pattern="indirect-gather",
        params_note="static must abstain (indirect-address); dynamic may verdict",
    )
)


# ── proved@T1 only: input-dependent mask bound (T0 SAT falls to T1) ──
@triton.jit
def bounded_store_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    m = offs < n
    tl.store(out_ptr + offs, v, mask=m)


def _bounded_args(n: int):
    def make(seed: int) -> tuple:
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(4096, generator=g), torch.zeros(4096), n)

    return make


_BOUNDED_SIG = {
    "x_ptr": "*fp32",
    "out_ptr": "*fp32",
    "n": "i32",
    "BLOCK": "constexpr",
}
CORPUS.add(
    LaunchSpec(
        name="smoke_bounded_n0_no",
        kernel_fn=bounded_store_kernel,
        signature=_BOUNDED_SIG,
        constexprs={"BLOCK": 64},
        make_args=_bounded_args(0),
        grid=(4,),
        expected="race-free",
        pattern="input-dependent-bound",
        params_note="n=0 kills the store mask; provable only at T1",
    )
)
CORPUS.add(
    LaunchSpec(
        name="smoke_bounded_n5_yes",
        kernel_fn=bounded_store_kernel,
        signature=_BOUNDED_SIG,
        constexprs={"BLOCK": 64},
        make_args=_bounded_args(5),
        grid=(4,),
        expected="race",
        pattern="input-dependent-bound",
        params_note="n=5: blocks overlap on out[0:5]",
    )
)
