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


# ── trb020: partially overlapping masks (parameterized labels) ───


@triton.jit
def trb020_masked_halves_kernel(out_ptr, k1, k2, BLOCK: tl.constexpr):
    """Exactly ONE writer per branch on EVERY grid (the T1 claim covers
    all grids along the read axes): a parity split would put two
    same-branch blocks on one masked range for any grid >= 3 and the
    same-branch WAW would drown the mask-overlap question."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(out_ptr + offs, 1, mask=offs < k1)
    if pid == 1:
        tl.store(out_ptr + offs, 2, mask=offs >= k2)


_TRB020_SIG = {"out_ptr": "*i32", "k1": "i32", "k2": "i32", "BLOCK": "constexpr"}


def _trb020_args(k1: int, k2: int):
    def make(seed: int) -> tuple:
        return (torch.zeros(BLOCK, dtype=torch.int32), k1, k2)

    return make


CORPUS.add(
    LaunchSpec(
        name="trb020_mask_overlap_no",
        kernel_fn=trb020_masked_halves_kernel,
        signature=_TRB020_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb020_args(32, 32),
        grid=(2,),
        expected="race-free",
        pattern="partial-mask-overlap",
        params_note="k1=k2=32: even blocks own [0,32), odd blocks own "
        "[32,64) — the masks tile the range exactly",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb020_mask_overlap_yes",
        kernel_fn=trb020_masked_halves_kernel,
        signature=_TRB020_SIG,
        constexprs={"BLOCK": BLOCK},
        make_args=_trb020_args(40, 24),
        grid=(2,),
        expected="race",
        race_pair=(
            "tl.store(out_ptr + offs, 1, mask=offs < k1)",
            "tl.store(out_ptr + offs, 2, mask=offs >= k2)",
        ),
        pattern="partial-mask-overlap",
        params_note="k1=40, k2=24: the masks overlap on [24,40) — same "
        "kernel, labels flip with the scalar params",
    )
)


# ── trb021: one-sided synchronizes-with (CAS-guarded P/C) ────────


@triton.jit
def trb021_acq_rel_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def trb021_release_only_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="release", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def trb021_acquire_only_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 1)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acquire", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


_TRB021_SIG = {"flag_ptr": "*i32", "data_ptr": "*i32", "out_ptr": "*i32"}
_TRB021_PAIR = (
    "tl.store(data_ptr, 1, mask=is_prod)",
    "x = tl.load(data_ptr, mask=cons_mask, other=0)",
)


def _trb021_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )


for _name, _fn, _exp, _note in (
    (
        "trb021_guarded_acq_rel_no",
        trb021_acq_rel_kernel,
        "race-free",
        "both halves of the sw edge present (control row)",
    ),
    (
        "trb021_release_only_yes",
        trb021_release_only_kernel,
        "race",
        "release-only: the consumer's read side never acquires — half an "
        "sw edge is no sw edge",
    ),
    (
        "trb021_acquire_only_yes",
        trb021_acquire_only_kernel,
        "race",
        "acquire-only: the producer's write side never releases",
    ),
):
    CORPUS.add(
        LaunchSpec(
            name=_name,
            kernel_fn=_fn,
            signature=_TRB021_SIG,
            constexprs={},
            make_args=_trb021_args,
            grid=(2,),
            expected=_exp,  # type: ignore[arg-type]
            race_pair=None if _exp == "race-free" else _TRB021_PAIR,
            pattern="one-sided-sw",
            params_note=_note,
        )
    )


# ── trb022: acquire-on-failure (reader-success-independence) ─────


@triton.jit
def trb022_failed_cas_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    # Producer publishes via a SUCCESSFUL CAS 0->1. The consumer's cmp=7
    # can never match (flag stays in {0,1}), so its CAS always FAILS —
    # but a failed acquire-CAS still READS the location, and reading the
    # released value establishes the sw edge (rf-val is independent of
    # the reader's own success).
    cmp = tl.where(is_prod, 0, 7)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="acq_rel", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


@triton.jit
def trb022_failed_cas_relaxed_kernel(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    is_prod = pid == 0
    is_cons = pid == 1
    tl.store(data_ptr, 1, mask=is_prod)
    cmp = tl.where(is_prod, 0, 7)
    old = tl.atomic_cas(flag_ptr, cmp, 1, sem="relaxed", scope="gpu")
    cons_mask = is_cons & (old == 1)
    x = tl.load(data_ptr, mask=cons_mask, other=0)
    tl.store(out_ptr + pid, x, mask=cons_mask)


CORPUS.add(
    LaunchSpec(
        name="trb022_acquire_on_failure_no",
        kernel_fn=trb022_failed_cas_kernel,
        signature=_TRB021_SIG,
        constexprs={},
        make_args=_trb021_args,
        grid=(2,),
        expected="race-free",
        pattern="acquire-on-failure",
        params_note="the consumer's CAS always fails (cmp=7 never matches) "
        "yet its acquire read of the released value synchronizes — the "
        "positive case for rf-val's reader-success-independence",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb022_acquire_on_failure_relaxed_yes",
        kernel_fn=trb022_failed_cas_relaxed_kernel,
        signature=_TRB021_SIG,
        constexprs={},
        make_args=_trb021_args,
        grid=(2,),
        expected="race",
        race_pair=_TRB021_PAIR,
        pattern="acquire-on-failure",
        params_note="same failed-CAS gate, relaxed: no sw — the guard value "
        "arrives but nothing orders the data",
    )
)


# ── trb023: oversized flag (the over-report direction, on purpose) ─


def _trb023_args(seed: int) -> tuple:
    # flag lives in a 2048-element tensor: PAST the solver's rf-init cap
    # (_MAX_INITIAL_ATOMIC_ELEMENTS = 1024), so the CAS reader falls to
    # rf_unknown, which deliberately does NOT enable synchronizes-with.
    return (
        torch.zeros(2048, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )


CORPUS.add(
    LaunchSpec(
        name="trb023_oversized_flag_conservative",
        kernel_fn=trb021_acq_rel_kernel,
        signature=_TRB021_SIG,
        constexprs={},
        make_args=_trb023_args,
        grid=(2,),
        # GROUND TRUTH is race-free (identical to trb021_guarded_acq_rel_no
        # up to the flag allocation size), but the row is deliberately
        # UNLABELED: the solver cannot snapshot a >1024-element flag, the
        # closed world opens, and the guarded pair is reported
        # CONSERVATIVELY — the over-report direction of the monotonicity
        # lemma, demonstrated. Labeling it race-free would score the
        # designed behavior as a false positive.
        expected=None,
        pattern="oversized-flag-demo",
        params_note="monotonicity-lemma demo: rf-init cap exceeded → "
        "rf_unknown (no sw) → conservative race report on a race-free "
        "program; ground truth race-free, row excluded from P/R scoring",
    )
)


# ── trb024: cta-scope atomic pair (moral strength, with the ──────
# conflict-predicate refinement record in test_moral_strength_scopes)


@triton.jit
def trb024_cta_add_kernel(ctr_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.atomic_add(ctr_ptr, 1, sem="relaxed", scope="cta")
    tl.store(out_ptr + pid, 1)


@triton.jit
def trb024_gpu_add_kernel(ctr_ptr, out_ptr):
    pid = tl.program_id(0)
    tl.atomic_add(ctr_ptr, 1, sem="relaxed", scope="gpu")
    tl.store(out_ptr + pid, 1)


_TRB024_SIG = {"ctr_ptr": "*i32", "out_ptr": "*i32"}


def _trb024_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )


CORPUS.add(
    LaunchSpec(
        name="trb024_cta_scope_pair_yes",
        kernel_fn=trb024_cta_add_kernel,
        signature=_TRB024_SIG,
        constexprs={},
        make_args=_trb024_args,
        grid=GRID,
        expected="race",
        race_pair=('tl.atomic_add(ctr_ptr, 1, sem="relaxed", scope="cta")',),
        pattern="cta-scope-pair",
        params_note="PTX .cta scope covers one CTA only: cross-CTA adds at "
        "one cell are scope-mismatched (not morally strong) — torn, racy",
    )
)
CORPUS.add(
    LaunchSpec(
        name="trb024_gpu_scope_pair_no",
        kernel_fn=trb024_gpu_add_kernel,
        signature=_TRB024_SIG,
        constexprs={},
        make_args=_trb024_args,
        grid=GRID,
        expected="race-free",
        pattern="cta-scope-pair",
        params_note="gpu scope covers the peer CTA: the pair is mutually "
        "atomic (inclusive scopes, same width, same address)",
    )
)
