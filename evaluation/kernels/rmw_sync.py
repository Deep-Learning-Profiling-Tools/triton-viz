"""B.4 litmus corpus: RMW-return synchronization patterns (spec part B).

Four DRB-style pairs — last-block-done, single-fetch work queue, split-k
semaphore (non-spin), atomic-max-in-mask — each race-free version proved by
the RMW observation model (counting axiom + reads-through), each racy twin
obtained by breaking exactly the synchronization the proof depends on.

`work_queue_plain_fetch_yes` is the spec's plain-load twin: a LOADED head
value in the address is data-dependent indirection, outside the model on
both tracks — the honest outcome is abstention (unsupported), recorded as a
coverage miss rather than a verdict.
"""

import torch
import triton
import triton.language as tl

from evaluation.spec import Corpus, LaunchSpec

CORPUS = Corpus("rmw_sync")

NBLK = 4
BLOCK = 64


# ── last_block_done ──────────────────────────────────────────────


@triton.jit
def lbd_acq_rel_kernel(partial_ptr, counter_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.store(partial_ptr + pid, pid + 1)
    old = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    done = old == tl.num_programs(0) - 1
    p = tl.load(partial_ptr + 0, mask=done, other=0)
    tl.store(out_ptr, p, mask=done)


@triton.jit
def lbd_relaxed_kernel(partial_ptr, counter_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.store(partial_ptr + pid, pid + 1)
    old = tl.atomic_add(counter_ptr, 1, sem="relaxed")
    done = old == tl.num_programs(0) - 1
    p = tl.load(partial_ptr + 0, mask=done, other=0)
    tl.store(out_ptr, p, mask=done)


def _lbd_args(seed: int) -> tuple:
    return (
        torch.zeros(NBLK, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


_LBD_SIG = {"partial_ptr": "*i32", "counter_ptr": "*i32", "out_ptr": "*i32"}

CORPUS.add(
    LaunchSpec(
        name="lbd_no",
        kernel_fn=lbd_acq_rel_kernel,
        signature=_LBD_SIG,
        constexprs={},
        make_args=_lbd_args,
        grid=(NBLK,),
        expected="race-free",
        pattern="last-block-done",
        params_note="acq_rel counter + num_programs gate: proof holds for "
        "EVERY grid",
    )
)
CORPUS.add(
    LaunchSpec(
        name="lbd_relaxed_yes",
        kernel_fn=lbd_relaxed_kernel,
        signature=_LBD_SIG,
        constexprs={},
        make_args=_lbd_args,
        grid=(NBLK,),
        expected="race",
        race_pair=("tl.store(partial_ptr + pid", "p = tl.load(partial_ptr + 0"),
        pattern="last-block-done",
        params_note="dropped release/acquire: partial store vs last read",
    )
)


# ── work_queue_single_fetch ──────────────────────────────────────


@triton.jit
def wq_fetch_kernel(head_ptr, buf_ptr):
    pid = tl.program_id(0)
    idx = tl.atomic_add(head_ptr, 1, sem="relaxed")
    tl.store(buf_ptr + idx, pid)


@triton.jit
def wq_narrow_kernel(head_ptr, buf_ptr):
    pid = tl.program_id(0)
    idx = tl.atomic_add(head_ptr, 1, sem="relaxed")
    tl.store(buf_ptr + idx // 2, pid)


@triton.jit
def wq_plain_fetch_kernel(head_ptr, buf_ptr):
    pid = tl.program_id(0)
    idx = tl.load(head_ptr)
    tl.store(buf_ptr + idx, pid)


def _wq_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
    )


_WQ_SIG = {"head_ptr": "*i32", "buf_ptr": "*i32"}

CORPUS.add(
    LaunchSpec(
        name="wq_single_fetch_no",
        kernel_fn=wq_fetch_kernel,
        signature=_WQ_SIG,
        constexprs={},
        make_args=_wq_args,
        grid=(NBLK,),
        expected="race-free",
        pattern="work-queue-fetch",
        params_note="distinct observations → distinct slots (counting axiom "
        "pins the observation in the ADDRESS)",
    )
)
CORPUS.add(
    LaunchSpec(
        name="wq_narrow_slots_yes",
        kernel_fn=wq_narrow_kernel,
        signature=_WQ_SIG,
        constexprs={},
        make_args=_wq_args,
        grid=(NBLK,),
        expected="race",
        race_pair=("tl.store(buf_ptr + idx // 2, pid)",),
        pattern="work-queue-fetch",
        params_note="idx // 2: adjacent ranks share a slot",
    )
)
CORPUS.add(
    LaunchSpec(
        name="wq_plain_fetch_yes",
        kernel_fn=wq_plain_fetch_kernel,
        signature=_WQ_SIG,
        constexprs={},
        make_args=_wq_args,
        grid=(NBLK,),
        expected="race",
        pattern="work-queue-fetch",
        params_note="plain load of head: loaded-value address → both tracks "
        "abstain (indirect-address); honest coverage miss",
    )
)


# ── split_k_semaphore_nonspin ────────────────────────────────────


@triton.jit
def splitk_acq_rel_kernel(
    x_ptr, partial_ptr, sem_ptr, out_ptr, BLOCK: tl.constexpr, MAXB: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))
    old = tl.atomic_add(sem_ptr, 1, sem="acq_rel")
    done = old == tl.num_programs(0) - 1
    lanes = tl.arange(0, MAXB)
    lm = done & (lanes < tl.num_programs(0))
    p = tl.load(partial_ptr + lanes, mask=lm, other=0)
    tl.store(out_ptr, tl.sum(p, axis=0), mask=done)


@triton.jit
def splitk_relaxed_kernel(
    x_ptr, partial_ptr, sem_ptr, out_ptr, BLOCK: tl.constexpr, MAXB: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))
    old = tl.atomic_add(sem_ptr, 1, sem="relaxed")
    done = old == tl.num_programs(0) - 1
    lanes = tl.arange(0, MAXB)
    lm = done & (lanes < tl.num_programs(0))
    p = tl.load(partial_ptr + lanes, mask=lm, other=0)
    tl.store(out_ptr, tl.sum(p, axis=0), mask=done)


def _splitk_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (NBLK * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(NBLK, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


_SPLITK_SIG = {
    "x_ptr": "*i32",
    "partial_ptr": "*i32",
    "sem_ptr": "*i32",
    "out_ptr": "*i32",
    "BLOCK": "constexpr",
    "MAXB": "constexpr",
}

CORPUS.add(
    LaunchSpec(
        name="splitk_sem_no",
        kernel_fn=splitk_acq_rel_kernel,
        signature=_SPLITK_SIG,
        constexprs={"BLOCK": BLOCK, "MAXB": NBLK},
        make_args=_splitk_args,
        grid=(NBLK,),
        expected="race-free",
        pattern="split-k-semaphore",
        params_note="last arriver reduces the partials behind acquire",
    )
)
CORPUS.add(
    LaunchSpec(
        name="splitk_sem_relaxed_yes",
        kernel_fn=splitk_relaxed_kernel,
        signature=_SPLITK_SIG,
        constexprs={"BLOCK": BLOCK, "MAXB": NBLK},
        make_args=_splitk_args,
        grid=(NBLK,),
        expected="race",
        race_pair=("tl.store(partial_ptr + pid", "p = tl.load(partial_ptr + lanes"),
        pattern="split-k-semaphore",
        params_note="relaxed semaphore: partial store vs epilogue read",
    )
)


# ── atomic_max_scale ─────────────────────────────────────────────


@triton.jit
def amax_kernel(mx_ptr, out_ptr):
    pid = tl.program_id(0)
    m = tl.atomic_max(mx_ptr, pid + 1, sem="relaxed")
    keep = m >= 0
    tl.store(out_ptr + pid, m, mask=keep)


@triton.jit
def amax_torn_kernel(mx_ptr, out_ptr):
    pid = tl.program_id(0)
    v = tl.load(mx_ptr)
    m = tl.maximum(v, pid + 1)
    tl.store(mx_ptr, m)
    tl.store(out_ptr + pid, m)


def _amax_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(NBLK, dtype=torch.int32),
    )


_AMAX_SIG = {"mx_ptr": "*i32", "out_ptr": "*i32"}

CORPUS.add(
    LaunchSpec(
        name="amax_scale_no",
        kernel_fn=amax_kernel,
        signature=_AMAX_SIG,
        constexprs={},
        make_args=_amax_args,
        grid=(NBLK,),
        expected="race-free",
        pattern="atomic-max-mask",
        params_note="atomic_max return in mask position; per-pid stores "
        "disjoint for any observation",
    )
)
CORPUS.add(
    LaunchSpec(
        name="amax_torn_yes",
        kernel_fn=amax_torn_kernel,
        signature=_AMAX_SIG,
        constexprs={},
        make_args=_amax_args,
        grid=(NBLK,),
        expected="race",
        race_pair=("v = tl.load(mx_ptr)", "tl.store(mx_ptr, m)"),
        pattern="atomic-max-mask",
        params_note="plain read-modify-write of the max cell",
    )
)
