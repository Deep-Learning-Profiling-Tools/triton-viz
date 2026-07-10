"""TritonRaceBench — Phase A labeled micro corpus (plan S5).

DRB-style yes/no PAIRS per race pattern, named ``trbNNN_<pattern>_<yes|no>``.
Ground truth is scoped to the (kernel, launch) row; a kernel with any
yes-labeled launch derives the kernel-level "∃ racy input" truth that the
report's LADDER AUDIT checks proved@T0 claims against (an ALIASED
yes-launch is exempt — it violates the T0 non-aliasing premise).

trb001-trb011 are the new micro pairs; trb002/006/007/010 fold in the
golden_smoke rows; trb012-trb015 fold in the rmw_sync corpus (spec part B
litmus) and trb016-trb018 the await_sync corpus (spec C1 litmus), all
under stable trb names. The feature corpora remain runnable standalone.

Deliberate abstention rows (documented boundaries, scored as coverage
misses, never verdicts): trb010 (indirect scatter/gather), trb011 (nested
loops), trb013 plain-fetch (loaded-value address).
"""

from dataclasses import replace

import torch
import triton
import triton.language as tl

from evaluation.kernels import await_sync, golden_smoke, rmw_sync
from evaluation.spec import Corpus, LaunchSpec

CORPUS = Corpus("tritonracebench")

BLOCK = 64
GRID = (4,)


# ── trb001: pid-stride misalignment ──────────────────────────────


@triton.jit
def trb001_kernel(x_ptr, out_ptr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * STRIDE + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, v + 1)


def _trb001_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(4 * BLOCK, dtype=torch.int32),
    )


_TRB001_SIG = {
    "x_ptr": "*i32",
    "out_ptr": "*i32",
    "STRIDE": "constexpr",
    "BLOCK": "constexpr",
}

CORPUS.add(
    LaunchSpec(
        name="trb001_pid_stride_no",
        kernel_fn=trb001_kernel,
        signature=_TRB001_SIG,
        constexprs={"STRIDE": BLOCK, "BLOCK": BLOCK},
        make_args=_trb001_args,
        grid=GRID,
        expected="race-free",
        pattern="pid-stride",
        params_note="stride == BLOCK: per-pid tiles are disjoint",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb001_pid_stride_yes",
        kernel_fn=trb001_kernel,
        signature=_TRB001_SIG,
        constexprs={"STRIDE": BLOCK // 2, "BLOCK": BLOCK},
        make_args=_trb001_args,
        grid=GRID,
        expected="race",
        race_pair=("tl.store(out_ptr + offs, v + 1)",),
        pattern="pid-stride",
        params_note="stride BLOCK/2: adjacent tiles overlap by half a block",
    )
)


# ── trb002: fixed-range store (single writer vs every block) ─────


@triton.jit
def trb002_single_writer_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + pid * BLOCK + offs)
    tl.store(out_ptr + offs, v, mask=pid == 0)


CORPUS.add(
    LaunchSpec(
        name="trb002_fixed_range_no",
        kernel_fn=trb002_single_writer_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": BLOCK},
        make_args=lambda seed: (
            torch.randn(4 * BLOCK, generator=torch.Generator().manual_seed(seed)),
            torch.zeros(BLOCK),
        ),
        grid=GRID,
        expected="race-free",
        pattern="fixed-range-store",
        params_note="pid==0 mask makes the fixed range single-writer",
    )
)


# ── trb003: boundary handled by mask vs clamp ────────────────────


@triton.jit
def trb003_mask_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    v = tl.load(x_ptr + idx, mask=m, other=0)
    tl.store(out_ptr + idx, v, mask=m)


@triton.jit
def trb003_clamp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    cidx = tl.minimum(idx, n - 1)
    v = tl.load(x_ptr + cidx)
    tl.store(out_ptr + cidx, v)


def _trb003_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(4 * BLOCK, dtype=torch.int32),
        130,
    )


_TRB003_SIG = {"x_ptr": "*i32", "out_ptr": "*i32", "n": "i32", "BLOCK": "constexpr"}

CORPUS.add(
    LaunchSpec(
        name="trb003_tail_mask_no",
        kernel_fn=trb003_mask_kernel,
        signature=_TRB003_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb003_args,
        grid=GRID,
        expected="race-free",
        pattern="tail-boundary",
        params_note="tail handled by masking: out-of-range lanes inactive",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb003_tail_clamp_yes",
        kernel_fn=trb003_clamp_kernel,
        signature=_TRB003_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb003_args,
        grid=GRID,
        expected="race",
        race_pair=("tl.store(out_ptr + cidx, v)",),
        pattern="tail-boundary",
        params_note="tail handled by clamping to n-1: blocks 2 and 3 both "
        "write out[129] (n=130)",
    )
)


# ── trb004: atomic accumulate vs plain read-modify-write ─────────


@triton.jit
def trb004_atomic_kernel(x_ptr, acc_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    s = tl.sum(v, axis=0)
    tl.atomic_add(acc_ptr, s)


@triton.jit
def trb004_plain_kernel(x_ptr, acc_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    s = tl.sum(v, axis=0)
    a = tl.load(acc_ptr)
    tl.store(acc_ptr, a + s)


def _trb004_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(1, dtype=torch.int32),
    )


_TRB004_SIG = {"x_ptr": "*i32", "acc_ptr": "*i32", "BLOCK": "constexpr"}

CORPUS.add(
    LaunchSpec(
        name="trb004_atomic_accum_no",
        kernel_fn=trb004_atomic_kernel,
        signature=_TRB004_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb004_args,
        grid=GRID,
        expected="race-free",
        pattern="atomic-vs-plain-accum",
        params_note="cross-block accumulation through atomic_add",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb004_plain_accum_yes",
        kernel_fn=trb004_plain_kernel,
        signature=_TRB004_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb004_args,
        grid=GRID,
        expected="race",
        race_pair=("a = tl.load(acc_ptr)", "tl.store(acc_ptr, a + s)"),
        pattern="atomic-vs-plain-accum",
        params_note="the atomic dropped to a load/add/store",
    )
)


# ── trb005: pid branch ───────────────────────────────────────────


@triton.jit
def trb005_disjoint_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(out_ptr + offs, 1)
    else:
        tl.store(out_ptr + pid * BLOCK + offs, 2)


@triton.jit
def trb005_overlap_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(out_ptr + offs, 1)
    else:
        tl.store(out_ptr + offs, 2)


_TRB005_SIG = {"out_ptr": "*i32", "BLOCK": "constexpr"}

CORPUS.add(
    LaunchSpec(
        name="trb005_pid_branch_no",
        kernel_fn=trb005_disjoint_kernel,
        signature=_TRB005_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=lambda seed: (torch.zeros(4 * BLOCK, dtype=torch.int32),),
        grid=GRID,
        expected="race-free",
        pattern="pid-branch",
        params_note="both branches write pid-disjoint ranges (scf.if path "
        "conditions modeled)",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb005_pid_branch_yes",
        kernel_fn=trb005_overlap_kernel,
        signature=_TRB005_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=lambda seed: (torch.zeros(4 * BLOCK, dtype=torch.int32),),
        grid=GRID,
        expected="race",
        race_pair=("tl.store(out_ptr + offs, 1)", "tl.store(out_ptr + offs, 2)"),
        pattern="pid-branch",
        params_note="then/else branches of DIFFERENT blocks hit one range",
    )
)


# ── trb008: loop-carried overlap ─────────────────────────────────


@triton.jit
def trb008_disjoint_kernel(x_ptr, out_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * iters * BLOCK
    for k in range(iters):
        offs = base + k * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, v)


@triton.jit
def trb008_overlap_kernel(x_ptr, out_ptr, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    for k in range(iters):
        offs = base + k * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, v)


def _trb008_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (16 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(16 * BLOCK, dtype=torch.int32),
        4,
    )


_TRB008_SIG = {
    "x_ptr": "*i32",
    "out_ptr": "*i32",
    "iters": "i32",
    "BLOCK": "constexpr",
}

CORPUS.add(
    LaunchSpec(
        name="trb008_loop_stride_no",
        kernel_fn=trb008_disjoint_kernel,
        signature=_TRB008_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb008_args,
        grid=GRID,
        expected="race-free",
        pattern="loop-carried",
        params_note="each block walks its own iters*BLOCK segment",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb008_loop_stride_yes",
        kernel_fn=trb008_overlap_kernel,
        signature=_TRB008_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb008_args,
        grid=GRID,
        expected="race",
        race_pair=("tl.store(out_ptr + offs, v)",),
        pattern="loop-carried",
        params_note="block i's iteration k+1 aliases block i+1's iteration k",
    )
)


# ── trb009: aliased in-place launch ──────────────────────────────


@triton.jit
def trb009_shift_kernel(src_ptr, dst_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src_ptr + offs)
    tl.store(dst_ptr + offs + BLOCK, v)


_TRB009_SIG = {"src_ptr": "*i32", "dst_ptr": "*i32", "BLOCK": "constexpr"}


def _trb009_distinct_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(5 * BLOCK, dtype=torch.int32),
    )


def _trb009_aliased_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    buf = torch.randint(0, 100, (5 * BLOCK,), dtype=torch.int32, generator=g)
    return (buf, buf)  # the SAME storage as source and destination


CORPUS.add(
    LaunchSpec(
        name="trb009_shift_distinct_no",
        kernel_fn=trb009_shift_kernel,
        signature=_TRB009_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb009_distinct_args,
        grid=GRID,
        expected="race-free",
        pattern="aliased-inplace",
        params_note="distinct tensors: reads and shifted writes never meet",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb009_shift_inplace_yes",
        kernel_fn=trb009_shift_kernel,
        signature=_TRB009_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb009_aliased_args,
        grid=GRID,
        expected="race",
        race_pair=(
            "v = tl.load(src_ptr + offs)",
            "tl.store(dst_ptr + offs + BLOCK, v)",
        ),
        pattern="aliased-inplace",
        params_note="src IS dst: block i's shifted store hits block i+1's "
        "read range. Violates the T0 non-aliasing premise (aliased=True), "
        "so it must not count against a T0 proof in the ladder audit",
        aliased=True,
    )
)


# ── trb010: indirect scatter (abstention boundary, racy twin) ────


@triton.jit
def trb010_scatter_kernel(idx_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + i, v)


CORPUS.add(
    LaunchSpec(
        name="trb010_scatter_yes",
        kernel_fn=trb010_scatter_kernel,
        signature={
            "idx_ptr": "*i32",
            "x_ptr": "*i32",
            "out_ptr": "*i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": BLOCK},
        make_args=lambda seed: (
            torch.zeros(4 * BLOCK, dtype=torch.int32),  # every index is 0
            torch.ones(4 * BLOCK, dtype=torch.int32),
            torch.zeros(BLOCK, dtype=torch.int32),
        ),
        grid=GRID,
        expected="race",
        pattern="indirect-gather",
        params_note="all indices 0: every block stores out[0]; the static "
        "track must abstain (indirect-address) — the dynamic column is the "
        "comparison datum",
    )
)


# ── trb011: nested loops (abstention boundary pair) ──────────────


@triton.jit
def trb011_disjoint_kernel(x_ptr, out_ptr, ni, nj, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for i in range(ni):
        for j in range(nj):
            v = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, v + i + j)


@triton.jit
def trb011_overlap_kernel(x_ptr, out_ptr, ni, nj, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    for i in range(ni):
        for j in range(nj):
            v = tl.load(x_ptr + pid * BLOCK + offs)
            tl.store(out_ptr + offs, v + i + j)


def _trb011_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(4 * BLOCK, dtype=torch.int32),
        2,
        2,
    )


_TRB011_SIG = {
    "x_ptr": "*i32",
    "out_ptr": "*i32",
    "ni": "i32",
    "nj": "i32",
    "BLOCK": "constexpr",
}

CORPUS.add(
    LaunchSpec(
        name="trb011_nested_loop_no",
        kernel_fn=trb011_disjoint_kernel,
        signature=_TRB011_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb011_args,
        grid=GRID,
        expected="race-free",
        pattern="nested-loop",
        params_note="nested scf.for: static abstains (nested-loop kind)",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb011_nested_loop_yes",
        kernel_fn=trb011_overlap_kernel,
        signature=_TRB011_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb011_args,
        grid=GRID,
        expected="race",
        pattern="nested-loop",
        params_note="same fixed range from every block, still nested: both "
        "rows document the abstention boundary",
    )
)


# ── fold in golden_smoke / rmw_sync / await_sync under trb names ─

_FOLD = (
    (
        golden_smoke.CORPUS,
        {
            "smoke_bcast_store_yes": "trb002_fixed_range_yes",
            "smoke_dd_mask_dead_no": "trb006_dd_mask_dead_no",
            "smoke_dd_mask_live_yes": "trb006_dd_mask_live_yes",
            "smoke_bounded_n0_no": "trb007_bounded_n0_no",
            "smoke_bounded_n5_yes": "trb007_bounded_n5_yes",
            "smoke_gather_no": "trb010_gather_no",
        },
    ),
    (
        rmw_sync.CORPUS,
        {
            "lbd_no": "trb012_last_block_done_no",
            "lbd_relaxed_yes": "trb012_last_block_done_yes",
            "wq_single_fetch_no": "trb013_work_queue_no",
            "wq_narrow_slots_yes": "trb013_work_queue_narrow_yes",
            "wq_plain_fetch_yes": "trb013_work_queue_plain_yes",
            "splitk_sem_no": "trb014_splitk_sem_no",
            "splitk_sem_relaxed_yes": "trb014_splitk_sem_yes",
            "amax_scale_no": "trb015_atomic_max_no",
            "amax_torn_yes": "trb015_atomic_max_yes",
        },
    ),
    (
        await_sync.CORPUS,
        {
            "pc_wait_no": "trb016_pc_wait_no",
            "pc_wait_relaxed_writer_yes": "trb016_pc_wait_relaxed_writer_yes",
            "pc_wait_relaxed_spin_yes": "trb016_pc_wait_relaxed_spin_yes",
            "pc_wait_cta_scope_yes": "trb016_pc_wait_cta_scope_yes",
            "mutex_cas_no": "trb017_mutex_cas_no",
            "mutex_plain_unlock_yes": "trb017_mutex_plain_unlock_yes",
            "mutex_relaxed_cas_yes": "trb017_mutex_relaxed_cas_yes",
            "lookback_chain_no": "trb018_lookback_no",
            "lookback_cta_scope_yes": "trb018_lookback_cta_yes",
        },
    ),
)

for _corpus, _renames in _FOLD:
    _by_name = {s.name: s for s in _corpus.specs}
    for _orig, _trb in _renames.items():
        CORPUS.add(replace(_by_name[_orig], name=_trb))


# ── trb019: symbolic trip count (the T0 symbolic-loop-bounds stretch) ─


@triton.jit
def trb019_seg_walk_kernel(out_ptr, n, SEG: tl.constexpr, MASK: tl.constexpr):
    pid = tl.program_id(0)
    for k in range(0, n):
        tl.store(out_ptr + pid * SEG + k, 1, mask=k < MASK)


_TRB019_SIG = {"out_ptr": "*i32", "n": "i32", "SEG": "constexpr", "MASK": "constexpr"}


def _trb019_args(seed: int) -> tuple:
    return (torch.zeros(16 * BLOCK, dtype=torch.int32), 128)


CORPUS.add(
    LaunchSpec(
        name="trb019_symbolic_trip_no",
        kernel_fn=trb019_seg_walk_kernel,
        signature=_TRB019_SIG,
        constexprs={"SEG": BLOCK, "MASK": BLOCK},
        make_args=_trb019_args,
        grid=GRID,
        expected="race-free",
        pattern="symbolic-trip-count",
        params_note="mask k < SEG pins every iteration inside the pid's own "
        "segment: proved@T0 for ANY trip count n (the symbolic-loop-bounds "
        "stretch — the concrete-bounds encoder could only reach T1)",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb019_symbolic_trip_yes",
        kernel_fn=trb019_seg_walk_kernel,
        signature=_TRB019_SIG,
        constexprs={"SEG": BLOCK, "MASK": 2 * BLOCK},
        make_args=_trb019_args,
        grid=GRID,
        expected="race",
        race_pair=("tl.store(out_ptr + pid * SEG + k, 1, mask=k < MASK)",),
        pattern="symbolic-trip-count",
        params_note="mask k < 2*SEG: iterations SEG..n-1 spill into the "
        "next pid's segment (n=128)",
    )
)
