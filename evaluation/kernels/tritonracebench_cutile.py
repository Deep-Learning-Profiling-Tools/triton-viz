"""tritonracebench_cutile corpus: cuda.tile twin implementations of the
TritonRaceBench litmus rows — the benchmark's cuTile track (paper repo
TODO tag `bench-cutile`).

Twin pairing is by ROW NAME: every row here carries the same name as its
Triton twin in ``tritonracebench`` (same ground-truth label, same grid,
same argument contents from the same seeds), so cross-DSL comparison is
a name join. The kernels are semantics-preserving ports:

- ``tl.load(p + offs)``/``tl.store(p + offs, v)`` element addressing
  maps to ``ct.gather``/``ct.scatter`` (the compiler lowers them to
  ``pointer_offset`` + ``load_pointer``/``store_pointer``, the same
  affine shape the CuTile IR reader models); tile-aligned full-tile
  accesses use ``ct.load``/``ct.store`` where the twin is tile-aligned.
- Scalar guards that Triton writes as masks stay masks here (a mask
  conjunct ``... & (pid == k)``); guards the twin writes as ``if``
  branches stay branches (trb005 deliberately tests branch handling).
- Atomics map 1:1 (``tl.atomic_*`` → ``ct.atomic_*``); Triton sem/scope
  spellings map relaxed/acquire/release/acq_rel → MemoryOrder.*, and
  cta → MemoryScope.BLOCK, gpu → MemoryScope.DEVICE. Triton's defaults
  (acq_rel, gpu) are written out explicitly.
- Spin loops keep their shape: ``while ct.atomic_add(...).item() != v``.

Rows are consumed through the generic ``cutile`` LaunchSpec path (the
static CuTile-IR-reader track; cuda.tile has no interpreter). Rows whose
IR leaves the reader's fragment refuse with the construct named
(control-flow for branches/spins, indirect-address for loaded-value
addressing, atomic-cas for CAS) — the documented-boundary discipline the
Triton corpus already uses for trb010/011/013.

The specs JSON next to this module carries each row's CAPTURED CuTile IR
plus arg descriptors (see evaluation/tritonracebench_cutile_capture);
rebuild needs neither cuda-tile nor a GPU. Before the first capture the
JSON is absent and CORPUS is empty (capture-only mode).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cuda.tile as ct

from evaluation.spec import Corpus, LaunchSpec

ConstInt = ct.Constant[int]
MO = ct.MemoryOrder
MS = ct.MemoryScope

BLOCK = 64
NBLK = 4
GRID = (4,)

SPECS_PATH = Path(__file__).parent / "tritonracebench_cutile_specs.json"


def _i32(n: int) -> ct.Tile:
    raise RuntimeError("host-only helper")  # pragma: no cover


# ── trb001: pid-stride misalignment ──────────────────────────────


@ct.kernel
def trb001_kernel(x, out, STRIDE: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = pid * STRIDE + ct.arange(BLOCK, dtype=np.int32)
    v = ct.gather(x, offs)
    ct.scatter(out, offs, v + 1)


def _trb001_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(4 * BLOCK, dtype=torch.int32),
    )


# ── trb002: fixed-range store (single writer vs every block) ─────


@ct.kernel
def trb002_single_writer_kernel(x, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    v = ct.gather(x, pid * BLOCK + offs)
    ct.scatter(out, offs, v, mask=(offs >= 0) & (pid == 0))


@ct.kernel
def trb002_bcast_store_kernel(x, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    v = ct.gather(x, pid * BLOCK + offs)
    ct.scatter(out, offs, v)


# ── trb003: boundary handled by mask vs clamp ────────────────────


@ct.kernel
def trb003_mask_kernel(x, out, n, BLOCK: ConstInt):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    m = idx < n
    v = ct.gather(x, idx, mask=m, padding_value=0)
    ct.scatter(out, idx, v, mask=m)


@ct.kernel
def trb003_clamp_kernel(x, out, n, BLOCK: ConstInt):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    cidx = ct.minimum(idx, n - 1)
    v = ct.gather(x, cidx)
    ct.scatter(out, cidx, v)


def _trb003_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(4 * BLOCK, dtype=torch.int32),
        130,
    )


# ── trb004: atomic accumulate vs plain read-modify-write ─────────


@ct.kernel
def trb004_atomic_kernel(x, acc, BLOCK: ConstInt):
    pid = ct.bid(0)
    v = ct.load(x, index=(pid,), shape=(BLOCK,))
    s = ct.sum(v, axis=0)
    zero = ct.arange(1, dtype=np.int32)
    ct.atomic_add(acc, zero, s, memory_order=MO.ACQ_REL, memory_scope=MS.DEVICE)


@ct.kernel
def trb004_plain_kernel(x, acc, BLOCK: ConstInt):
    pid = ct.bid(0)
    v = ct.load(x, index=(pid,), shape=(BLOCK,))
    s = ct.sum(v, axis=0)
    zero = ct.arange(1, dtype=np.int32)
    a = ct.gather(acc, zero)
    ct.scatter(acc, zero, a + s)


def _trb004_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(1, dtype=torch.int32),
    )


# ── trb005: pid branch (deliberately tests branch handling) ──────


@ct.kernel
def trb005_disjoint_kernel(out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    if pid == 0:
        ct.scatter(out, offs, 1)
    else:
        ct.scatter(out, pid * BLOCK + offs, 2)


@ct.kernel
def trb005_overlap_kernel(out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    if pid == 0:
        ct.scatter(out, offs, 1)
    else:
        ct.scatter(out, offs, 2)


# ── trb006: data-dependent mask (golden_smoke dd_mask twin) ──────


@ct.kernel
def trb006_dd_mask_kernel(flag, x, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    keep = ct.gather(flag, offs) > 0
    v = ct.gather(x, pid * BLOCK + offs)
    ct.scatter(out, offs, v, mask=keep)


def _trb006_args(flagv: int):
    def make(seed: int) -> tuple:
        g = torch.Generator().manual_seed(seed)
        return (
            torch.full((64,), flagv, dtype=torch.int32),
            torch.randn(256, generator=g),
            torch.zeros(64),
        )

    return make


# ── trb007: input-dependent mask bound (golden_smoke bounded twin) ─


@ct.kernel
def trb007_bounded_store_kernel(x, out, n, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    v = ct.gather(x, pid * BLOCK + offs)
    ct.scatter(out, offs, v, mask=offs < n)


def _trb007_args(n: int):
    def make(seed: int) -> tuple:
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(4096, generator=g), torch.zeros(4096), n)

    return make


# ── trb008: loop-carried overlap ─────────────────────────────────


@ct.kernel
def trb008_disjoint_kernel(x, out, iters, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    base = pid * iters * BLOCK
    for k in range(0, iters):
        o = base + k * BLOCK + offs
        v = ct.gather(x, o)
        ct.scatter(out, o, v)


@ct.kernel
def trb008_overlap_kernel(x, out, iters, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    base = pid * BLOCK
    for k in range(0, iters):
        o = base + k * BLOCK + offs
        v = ct.gather(x, o)
        ct.scatter(out, o, v)


def _trb008_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (16 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(16 * BLOCK, dtype=torch.int32),
        4,
    )


# ── trb009: aliased in-place launch ──────────────────────────────


@ct.kernel
def trb009_shift_kernel(src, dst, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    v = ct.gather(src, offs)
    ct.scatter(dst, offs + BLOCK, v)


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


# ── trb010: indirect scatter / gather (abstention boundary) ──────


@ct.kernel
def trb010_scatter_kernel(idxp, x, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    i = ct.gather(idxp, offs)
    v = ct.gather(x, offs)
    ct.scatter(out, i, v)


@ct.kernel
def trb010_gather_kernel(idxp, src, out, n, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    m = offs < n
    idx = ct.gather(idxp, offs, mask=m, padding_value=0)
    vals = ct.gather(src, idx, mask=m, padding_value=0)
    ct.scatter(out, offs, vals, mask=m)


def _trb010_gather_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 256, (256,), dtype=torch.int32, generator=g),
        torch.randn(256, generator=g),
        torch.zeros(256),
        256,
    )


# ── trb011: nested loops (abstention boundary pair) ──────────────


@ct.kernel
def trb011_disjoint_kernel(x, out, ni, nj, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    for i in range(0, ni):
        for j in range(0, nj):
            v = ct.gather(x, offs)
            ct.scatter(out, offs, v + i + j)


@ct.kernel
def trb011_overlap_kernel(x, out, ni, nj, BLOCK: ConstInt):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    for i in range(0, ni):
        for j in range(0, nj):
            v = ct.gather(x, pid * BLOCK + offs)
            ct.scatter(out, offs, v + i + j)


def _trb011_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (4 * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(4 * BLOCK, dtype=torch.int32),
        2,
        2,
    )


# ── trb012: last_block_done (rmw_sync twin) ──────────────────────


@ct.kernel
def trb012_lbd_acq_rel_kernel(partial, counter, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    ct.scatter(partial, zero + pid, pid + 1)
    old = ct.atomic_add(
        counter, zero, 1, memory_order=MO.ACQ_REL, memory_scope=MS.DEVICE
    )
    done = old == (ct.num_blocks(0) - 1)
    p = ct.gather(partial, zero, mask=done, padding_value=0)
    ct.scatter(out, zero, p, mask=done)


@ct.kernel
def trb012_lbd_relaxed_kernel(partial, counter, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    ct.scatter(partial, zero + pid, pid + 1)
    old = ct.atomic_add(
        counter, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE
    )
    done = old == (ct.num_blocks(0) - 1)
    p = ct.gather(partial, zero, mask=done, padding_value=0)
    ct.scatter(out, zero, p, mask=done)


def _trb012_args(seed: int) -> tuple:
    return (
        torch.zeros(NBLK, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


# ── trb013: work-queue fetch (rmw_sync twin) ─────────────────────


@ct.kernel
def trb013_wq_fetch_kernel(head, buf):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    idx = ct.atomic_add(head, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
    ct.scatter(buf, idx, pid)


@ct.kernel
def trb013_wq_narrow_kernel(head, buf):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    idx = ct.atomic_add(head, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
    ct.scatter(buf, idx // 2, pid)


@ct.kernel
def trb013_wq_plain_fetch_kernel(head, buf):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    idx = ct.gather(head, zero)
    ct.scatter(buf, idx, pid)


def _trb013_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
    )


# ── trb014: split-k semaphore, non-spin (rmw_sync twin) ──────────


@ct.kernel
def trb014_splitk_acq_rel_kernel(x, partial, sem, out, BLOCK: ConstInt, MAXB: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    xv = ct.gather(x, offs)
    ct.scatter(partial, zero + pid, ct.sum(xv, axis=0))
    old = ct.atomic_add(sem, zero, 1, memory_order=MO.ACQ_REL, memory_scope=MS.DEVICE)
    done = old == (ct.num_blocks(0) - 1)
    lanes = ct.arange(MAXB, dtype=np.int32)
    lm = (lanes < ct.num_blocks(0)) & ct.broadcast_to(done, (MAXB,))
    p = ct.gather(partial, lanes, mask=lm, padding_value=0)
    ct.scatter(out, zero, ct.sum(p, axis=0), mask=done)


@ct.kernel
def trb014_splitk_relaxed_kernel(x, partial, sem, out, BLOCK: ConstInt, MAXB: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    xv = ct.gather(x, offs)
    ct.scatter(partial, zero + pid, ct.sum(xv, axis=0))
    old = ct.atomic_add(sem, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
    done = old == (ct.num_blocks(0) - 1)
    lanes = ct.arange(MAXB, dtype=np.int32)
    lm = (lanes < ct.num_blocks(0)) & ct.broadcast_to(done, (MAXB,))
    p = ct.gather(partial, lanes, mask=lm, padding_value=0)
    ct.scatter(out, zero, ct.sum(p, axis=0), mask=done)


def _trb014_args(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 100, (NBLK * BLOCK,), dtype=torch.int32, generator=g),
        torch.zeros(NBLK, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
    )


# ── trb015: atomic max in mask (rmw_sync twin) ───────────────────


@ct.kernel
def trb015_amax_kernel(mx, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    m = ct.atomic_max(
        mx, zero, pid + 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE
    )
    keep = m >= 0
    ct.scatter(out, zero + pid, m, mask=keep)


@ct.kernel
def trb015_amax_torn_kernel(mx, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    v = ct.gather(mx, zero)
    m = ct.maximum(v, pid + 1)
    ct.scatter(mx, zero, m)
    ct.scatter(out, zero + pid, m)


def _trb015_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(NBLK, dtype=torch.int32),
    )


# ── trb016: producer/consumer wait family (await_sync twin) ──────


@ct.kernel
def trb016_pc_wait_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_relaxed_writer_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_relaxed_spin_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.RELAXED, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_cta_scope_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.BLOCK)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.BLOCK
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_cta_reset_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 0, memory_order=MO.RELAXED, memory_scope=MS.BLOCK)
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_atomic_reset_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 0, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_flag_read_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        fv = ct.gather(flag, zero)
        ct.scatter(out, zero, fv)
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_or_poll_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_or(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


@ct.kernel
def trb016_pc_wait_xor_poll_kernel(flag, data, out, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid == 0:
        ct.store(data, index=(0,), tile=ct.arange(BLOCK, dtype=np.int32))
        ct.atomic_xchg(flag, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_xor(
                flag, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 1
        ):
            pass
        v = ct.load(data, index=(0,), shape=(BLOCK,))
        ct.store(out, index=(pid,), tile=v)


def _trb016_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(BLOCK, dtype=torch.int32),
        torch.zeros(2 * BLOCK, dtype=torch.int32),
    )


# ── trb017: mutex via CAS loop (await_sync twin) ─────────────────


@ct.kernel
def trb017_mutex_kernel(lock, x, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    while (
        ct.atomic_cas(
            lock, zero, 0, 1, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
        ).item()
        != 0
    ):
        pass
    v = ct.gather(x, zero)
    ct.scatter(x, zero, v + 1)
    ct.atomic_xchg(lock, zero, 0, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    ct.scatter(out, zero + pid, 1)


@ct.kernel
def trb017_mutex_plain_unlock_kernel(lock, x, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    while (
        ct.atomic_cas(
            lock, zero, 0, 1, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
        ).item()
        != 0
    ):
        pass
    v = ct.gather(x, zero)
    ct.scatter(x, zero, v + 1)
    ct.scatter(lock, zero, 0)
    ct.scatter(out, zero + pid, 1)


@ct.kernel
def trb017_mutex_relaxed_cas_kernel(lock, x, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    while (
        ct.atomic_cas(
            lock, zero, 0, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE
        ).item()
        != 0
    ):
        pass
    v = ct.gather(x, zero)
    ct.scatter(x, zero, v + 1)
    ct.atomic_xchg(lock, zero, 0, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    ct.scatter(out, zero + pid, 1)


def _trb017_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )


# ── trb018: decoupled look-back chain (await_sync twin) ──────────


@ct.kernel
def trb018_lookback_kernel(flag, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid > 0:
        while (
            ct.atomic_add(
                flag, zero + pid - 1, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            == 0
        ):
            pass
        prev = ct.gather(out, zero + pid - 1)
        ct.scatter(out, zero + pid, prev + 1)
    else:
        ct.scatter(out, zero + pid, 1)
    ct.atomic_xchg(flag, zero + pid, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)


@ct.kernel
def trb018_lookback_cta_scope_kernel(flag, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid > 0:
        while (
            ct.atomic_add(
                flag, zero + pid - 1, 0, memory_order=MO.ACQUIRE, memory_scope=MS.BLOCK
            ).item()
            == 0
        ):
            pass
        prev = ct.gather(out, zero + pid - 1)
        ct.scatter(out, zero + pid, prev + 1)
    else:
        ct.scatter(out, zero + pid, 1)
    ct.atomic_xchg(flag, zero + pid, 1, memory_order=MO.RELEASE, memory_scope=MS.BLOCK)


def _trb018_args(seed: int) -> tuple:
    return (
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )


# ── trb019: symbolic trip count ──────────────────────────────────


@ct.kernel
def trb019_seg_walk_kernel(out, n, SEG: ConstInt, MASK: ConstInt):
    pid = ct.bid(0)
    one = ct.arange(1, dtype=np.int32)
    for k in range(0, n):
        ct.scatter(out, pid * SEG + k + one, 1, mask=(one * 0 + k) < MASK)


def _trb019_args(seed: int) -> tuple:
    return (torch.zeros(16 * BLOCK, dtype=torch.int32), 128)


# ── trb020: partially overlapping masks ──────────────────────────


@ct.kernel
def trb020_masked_halves_kernel(out, k1, k2, BLOCK: ConstInt):
    # The twin's pid guards are branch-shaped in Triton; here they fold
    # into the store masks (identical footprint semantics) so the row
    # stays inside the reader's fragment and the mask-overlap question,
    # the row's point, is what the solver decides.
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=np.int32)
    ct.scatter(out, offs, 1, mask=(offs < k1) & (pid == 0))
    ct.scatter(out, offs, 2, mask=(offs >= k2) & (pid == 1))


def _trb020_args(k1: int, k2: int):
    def make(seed: int) -> tuple:
        return (torch.zeros(BLOCK, dtype=torch.int32), k1, k2)

    return make


# ── trb021: one-sided synchronizes-with (CAS-guarded P/C) ────────


@ct.kernel
def trb021_acq_rel_kernel(flag, data, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    prod = (zero >= 0) & (pid == 0)
    ct.scatter(data, zero, 1, mask=prod)
    cmpv = ct.where(prod, 0, 1)
    old = ct.atomic_cas(
        flag, zero, cmpv, 1, memory_order=MO.ACQ_REL, memory_scope=MS.DEVICE
    )
    cons = (pid == 1) & (old == 1)
    x = ct.gather(data, zero, mask=cons, padding_value=0)
    ct.scatter(out, zero + pid, x, mask=cons)


@ct.kernel
def trb021_release_only_kernel(flag, data, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    prod = (zero >= 0) & (pid == 0)
    ct.scatter(data, zero, 1, mask=prod)
    cmpv = ct.where(prod, 0, 1)
    old = ct.atomic_cas(
        flag, zero, cmpv, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE
    )
    cons = (pid == 1) & (old == 1)
    x = ct.gather(data, zero, mask=cons, padding_value=0)
    ct.scatter(out, zero + pid, x, mask=cons)


@ct.kernel
def trb021_acquire_only_kernel(flag, data, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    prod = (zero >= 0) & (pid == 0)
    ct.scatter(data, zero, 1, mask=prod)
    cmpv = ct.where(prod, 0, 1)
    old = ct.atomic_cas(
        flag, zero, cmpv, 1, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
    )
    cons = (pid == 1) & (old == 1)
    x = ct.gather(data, zero, mask=cons, padding_value=0)
    ct.scatter(out, zero + pid, x, mask=cons)


def _trb021_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )


# ── trb022: acquire-on-failure ───────────────────────────────────


@ct.kernel
def trb022_failed_cas_kernel(flag, data, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    prod = (zero >= 0) & (pid == 0)
    ct.scatter(data, zero, 1, mask=prod)
    # Consumer compares against 7, which the flag never holds: its CAS
    # always FAILS but still reads, and the acquire read of the released
    # value establishes the sw edge (same demonstration as the twin).
    cmpv = ct.where(prod, 0, 7)
    old = ct.atomic_cas(
        flag, zero, cmpv, 1, memory_order=MO.ACQ_REL, memory_scope=MS.DEVICE
    )
    cons = (pid == 1) & (old == 1)
    x = ct.gather(data, zero, mask=cons, padding_value=0)
    ct.scatter(out, zero + pid, x, mask=cons)


@ct.kernel
def trb022_failed_cas_relaxed_kernel(flag, data, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    prod = (zero >= 0) & (pid == 0)
    ct.scatter(data, zero, 1, mask=prod)
    cmpv = ct.where(prod, 0, 7)
    old = ct.atomic_cas(
        flag, zero, cmpv, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE
    )
    cons = (pid == 1) & (old == 1)
    x = ct.gather(data, zero, mask=cons, padding_value=0)
    ct.scatter(out, zero + pid, x, mask=cons)


def _trb023_args(seed: int) -> tuple:
    # flag lives in a 2048-element tensor: past the solver's rf-init cap,
    # the closed world opens, and the guarded pair reports conservatively
    # (the over-report direction of the monotonicity lemma) — same
    # demonstration as the Triton twin, ground truth race-free, unlabeled.
    return (
        torch.zeros(2048, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )


# ── trb024: cta-scope atomic pair (atomic compatibility) ─────────


@ct.kernel
def trb024_cta_add_kernel(ctr, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    ct.atomic_add(ctr, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.BLOCK)
    ct.scatter(out, zero + pid, 1)


@ct.kernel
def trb024_gpu_add_kernel(ctr, out):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    ct.atomic_add(ctr, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.DEVICE)
    ct.scatter(out, zero + pid, 1)


def _trb024_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )


# ── trb025: communication kernels, single-GPU half ───────────────


@ct.kernel
def trb025_comm_comp_kernel(sem, payload, out, N_COMM: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid < N_COMM:
        offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
        ct.scatter(payload, offs, ct.astype(offs + 1, np.float32))
        ct.atomic_xchg(sem, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        while (
            ct.atomic_add(
                sem, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != N_COMM
        ):
            pass
        offs = ct.arange(BLOCK, dtype=np.int32)
        v = ct.gather(payload, offs)
        ct.scatter(out, (pid - N_COMM) * BLOCK + ct.arange(BLOCK, dtype=np.int32), v)


@ct.kernel
def trb025_relaxed_poll_kernel(sem, payload, out, N_COMM: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid < N_COMM:
        offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
        ct.scatter(payload, offs, ct.astype(offs + 1, np.float32))
        ct.atomic_xchg(sem, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        # racy twin (a): the poll observes the arrival but at relaxed
        while (
            ct.atomic_add(
                sem, zero, 0, memory_order=MO.RELAXED, memory_scope=MS.DEVICE
            ).item()
            != N_COMM
        ):
            pass
        offs = ct.arange(BLOCK, dtype=np.int32)
        v = ct.gather(payload, offs)
        ct.scatter(out, (pid - N_COMM) * BLOCK + ct.arange(BLOCK, dtype=np.int32), v)


@ct.kernel
def trb025_poll_initial_kernel(sem, payload, out, N_COMM: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid < N_COMM:
        offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
        ct.scatter(payload, offs, ct.astype(offs + 1, np.float32))
        ct.atomic_xchg(sem, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        # racy twin (b): polls the WRONG counter value — the initial 0
        # exits immediately, so no acquire of the release arrival
        while (
            ct.atomic_add(
                sem, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
            ).item()
            != 0
        ):
            pass
        offs = ct.arange(BLOCK, dtype=np.int32)
        v = ct.gather(payload, offs)
        ct.scatter(out, (pid - N_COMM) * BLOCK + ct.arange(BLOCK, dtype=np.int32), v)


@ct.kernel
def trb025_role_skip_kernel(sem, payload, out, N_COMM: ConstInt, BLOCK: ConstInt):
    pid = ct.bid(0)
    zero = ct.arange(1, dtype=np.int32)
    if pid < N_COMM:
        offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
        ct.scatter(payload, offs, ct.astype(offs + 1, np.float32))
        ct.atomic_xchg(sem, zero, 1, memory_order=MO.RELEASE, memory_scope=MS.DEVICE)
    else:
        # racy twin (c): only the FIRST comp pid polls
        if pid == N_COMM:
            while (
                ct.atomic_add(
                    sem, zero, 0, memory_order=MO.ACQUIRE, memory_scope=MS.DEVICE
                ).item()
                != N_COMM
            ):
                pass
        offs = ct.arange(BLOCK, dtype=np.int32)
        v = ct.gather(payload, offs)
        ct.scatter(out, (pid - N_COMM) * BLOCK + ct.arange(BLOCK, dtype=np.int32), v)


def _trb025_args(seed: int) -> tuple:
    return (
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(16, dtype=torch.float32),
        torch.zeros(32, dtype=torch.float32),
    )


# ── the row table ────────────────────────────────────────────────
# name -> (kernel, make_args, extra positional args appended AFTER the
# tensor/scalar args (the ConstInt values, in parameter order), grid,
# expected, race_pair needles, pattern, params_note, aliased)

ROWS: dict[str, dict] = {}


def _row(
    name,
    kernel,
    make_args,
    consts,
    grid,
    expected,
    pattern,
    note,
    race_pair=None,
    aliased=False,
):
    assert name not in ROWS, name
    ROWS[name] = dict(
        kernel=kernel,
        make_args=make_args,
        consts=tuple(consts),
        grid=tuple(grid),
        expected=expected,
        race_pair=race_pair,
        pattern=pattern,
        note=note,
        aliased=aliased,
    )


_row(
    "trb001_pid_stride_no", trb001_kernel, _trb001_args, (BLOCK, BLOCK), GRID,
    "race-free", "pid-stride", "stride == BLOCK: per-pid tiles are disjoint",
)  # fmt: skip
_row(
    "trb001_pid_stride_yes", trb001_kernel, _trb001_args, (BLOCK // 2, BLOCK), GRID,
    "race", "pid-stride", "stride BLOCK/2: adjacent tiles overlap by half a block",
    race_pair=("ct.scatter(out, offs, v + 1)",),
)  # fmt: skip
_row(
    "trb002_fixed_range_no", trb002_single_writer_kernel,
    lambda seed: (
        torch.randn(4 * BLOCK, generator=torch.Generator().manual_seed(seed)),
        torch.zeros(BLOCK),
    ),
    (BLOCK,), GRID, "race-free", "fixed-range-store",
    "pid==0 mask makes the fixed range single-writer",
)  # fmt: skip
_row(
    "trb002_fixed_range_yes", trb002_bcast_store_kernel,
    lambda seed: (
        torch.randn(256, generator=torch.Generator().manual_seed(seed)),
        torch.zeros(64),
    ),
    (64,), GRID, "race", "fixed-range-store",
    "every block stores the same fixed range",
    race_pair=("ct.scatter(out, offs, v)",),
)  # fmt: skip
_row(
    "trb003_tail_mask_no", trb003_mask_kernel, _trb003_args, (BLOCK,), GRID,
    "race-free", "tail-boundary", "tail handled by masking: out-of-range lanes inactive",
)  # fmt: skip
_row(
    "trb003_tail_clamp_yes", trb003_clamp_kernel, _trb003_args, (BLOCK,), GRID,
    "race", "tail-boundary",
    "tail handled by clamping to n-1: blocks 2 and 3 both write out[129] (n=130)",
    race_pair=("ct.scatter(out, cidx, v)",),
)  # fmt: skip
_row(
    "trb004_atomic_accum_no", trb004_atomic_kernel, _trb004_args, (BLOCK,), GRID,
    "race-free", "atomic-vs-plain-accum", "cross-block accumulation through atomic_add",
)  # fmt: skip
_row(
    "trb004_plain_accum_yes", trb004_plain_kernel, _trb004_args, (BLOCK,), GRID,
    "race", "atomic-vs-plain-accum", "the atomic dropped to a load/add/store",
    race_pair=("a = ct.gather(acc, zero)", "ct.scatter(acc, zero, a + s)"),
)  # fmt: skip
_row(
    "trb005_pid_branch_no", trb005_disjoint_kernel,
    lambda seed: (torch.zeros(4 * BLOCK, dtype=torch.int32),),
    (BLOCK,), GRID, "race-free", "pid-branch",
    "both branches write pid-disjoint ranges; the cuTile reader's branch "
    "boundary makes this a documented control-flow refusal",
)  # fmt: skip
_row(
    "trb005_pid_branch_yes", trb005_overlap_kernel,
    lambda seed: (torch.zeros(4 * BLOCK, dtype=torch.int32),),
    (BLOCK,), GRID, "race", "pid-branch",
    "then/else branches of DIFFERENT blocks hit one range",
    race_pair=("ct.scatter(out, offs, 1)", "ct.scatter(out, offs, 2)"),
)  # fmt: skip
_row(
    "trb006_dd_mask_dead_no", trb006_dd_mask_kernel, _trb006_args(0), (64,), GRID,
    "race-free", "data-dependent-mask", "flags all zero: the store never executes",
)  # fmt: skip
_row(
    "trb006_dd_mask_live_yes", trb006_dd_mask_kernel, _trb006_args(1), (64,), GRID,
    "race", "data-dependent-mask", "flags all ones: the dropped mask is really live",
    race_pair=("ct.scatter(out, offs, v, mask=keep)",),
)  # fmt: skip
_row(
    "trb007_bounded_n0_no", trb007_bounded_store_kernel, _trb007_args(0), (64,), GRID,
    "race-free", "input-dependent-bound", "n=0 kills the store mask; provable only at T1",
)  # fmt: skip
_row(
    "trb007_bounded_n5_yes", trb007_bounded_store_kernel, _trb007_args(5), (64,), GRID,
    "race", "input-dependent-bound", "n=5: blocks overlap on out[0:5]",
    race_pair=("ct.scatter(out, offs, v, mask=offs < n)",),
)  # fmt: skip
_row(
    "trb008_loop_stride_no", trb008_disjoint_kernel, _trb008_args, (BLOCK,), GRID,
    "race-free", "loop-carried", "each block walks its own iters*BLOCK segment",
)  # fmt: skip
_row(
    "trb008_loop_stride_yes", trb008_overlap_kernel, _trb008_args, (BLOCK,), GRID,
    "race", "loop-carried", "block i's iteration k+1 aliases block i+1's iteration k",
    race_pair=("ct.scatter(out, o, v)",),
)  # fmt: skip
_row(
    "trb009_shift_distinct_no", trb009_shift_kernel, _trb009_distinct_args, (BLOCK,),
    GRID, "race-free", "aliased-inplace",
    "distinct tensors: reads and shifted writes never meet",
)  # fmt: skip
_row(
    "trb009_shift_inplace_yes", trb009_shift_kernel, _trb009_aliased_args, (BLOCK,),
    GRID, "race", "aliased-inplace",
    "src IS dst: block i's shifted store hits block i+1's read range; "
    "violates the T0 non-aliasing premise",
    race_pair=("v = ct.gather(src, offs)", "ct.scatter(dst, offs + BLOCK, v)"),
    aliased=True,
)  # fmt: skip
_row(
    "trb010_scatter_yes", trb010_scatter_kernel,
    lambda seed: (
        torch.zeros(4 * BLOCK, dtype=torch.int32),
        torch.ones(4 * BLOCK, dtype=torch.int32),
        torch.zeros(BLOCK, dtype=torch.int32),
    ),
    (BLOCK,), GRID, "race", "indirect-gather",
    "all indices 0: every block stores out[0]; the static track must "
    "abstain (indirect-address) — no dynamic column exists for cuTile",
)  # fmt: skip
_row(
    "trb010_gather_no", trb010_gather_kernel, _trb010_gather_args, (256,), (1,),
    "race-free", "indirect-gather",
    "static must abstain (indirect-address); documented boundary",
)  # fmt: skip
_row(
    "trb011_nested_loop_no", trb011_disjoint_kernel, _trb011_args, (BLOCK,), GRID,
    "race-free", "nested-loop", "nested loops: static abstains (documented boundary)",
)  # fmt: skip
_row(
    "trb011_nested_loop_yes", trb011_overlap_kernel, _trb011_args, (BLOCK,), GRID,
    "race", "nested-loop",
    "same fixed range from every block, still nested: both rows document "
    "the abstention boundary",
)  # fmt: skip
_row(
    "trb012_last_block_done_no", trb012_lbd_acq_rel_kernel, _trb012_args, (), (NBLK,),
    "race-free", "last-block-done", "acq_rel counter + num_blocks gate",
)  # fmt: skip
_row(
    "trb012_last_block_done_yes", trb012_lbd_relaxed_kernel, _trb012_args, (), (NBLK,),
    "race", "last-block-done", "dropped release/acquire: partial store vs last read",
    race_pair=("ct.scatter(partial, zero + pid, pid + 1)", "p = ct.gather(partial, zero"),
)  # fmt: skip
_row(
    "trb013_work_queue_no", trb013_wq_fetch_kernel, _trb013_args, (), (NBLK,),
    "race-free", "work-queue-fetch",
    "distinct observations, distinct slots; the RMW-return address leaves "
    "the static fragment (atomic result in an address)",
)  # fmt: skip
_row(
    "trb013_work_queue_narrow_yes", trb013_wq_narrow_kernel, _trb013_args, (), (NBLK,),
    "race", "work-queue-fetch", "idx // 2: adjacent ranks share a slot",
    race_pair=("ct.scatter(buf, idx // 2, pid)",),
)  # fmt: skip
_row(
    "trb013_work_queue_plain_yes", trb013_wq_plain_fetch_kernel, _trb013_args, (),
    (NBLK,), "race", "work-queue-fetch",
    "plain load of head: loaded-value address, honest coverage miss",
)  # fmt: skip
_row(
    "trb014_splitk_sem_no", trb014_splitk_acq_rel_kernel, _trb014_args, (BLOCK, NBLK),
    (NBLK,), "race-free", "split-k-semaphore",
    "last arriver reduces the partials behind acquire",
)  # fmt: skip
_row(
    "trb014_splitk_sem_yes", trb014_splitk_relaxed_kernel, _trb014_args, (BLOCK, NBLK),
    (NBLK,), "race", "split-k-semaphore",
    "relaxed semaphore: partial store vs epilogue read",
    race_pair=("ct.scatter(partial, zero + pid", "p = ct.gather(partial, lanes"),
)  # fmt: skip
_row(
    "trb015_atomic_max_no", trb015_amax_kernel, _trb015_args, (), (NBLK,),
    "race-free", "atomic-max-mask",
    "atomic_max return in mask position; per-pid stores disjoint for any observation",
)  # fmt: skip
_row(
    "trb015_atomic_max_yes", trb015_amax_torn_kernel, _trb015_args, (), (NBLK,),
    "race", "atomic-max-mask", "plain read-modify-write of the max cell",
    race_pair=("v = ct.gather(mx, zero)", "ct.scatter(mx, zero, m)"),
)  # fmt: skip

_PC_PAIR = ("ct.store(data, index=(0,), tile=", "v = ct.load(data, index=(0,)")
for _name, _kern, _exp, _note in (
    ("trb016_pc_wait_no", trb016_pc_wait_kernel, "race-free",
     "release publish + acquire spin: proof conditional on termination"),
    ("trb016_pc_wait_relaxed_writer_yes", trb016_pc_wait_relaxed_writer_kernel, "race",
     "relaxed publisher heads no release sequence"),
    ("trb016_pc_wait_relaxed_spin_yes", trb016_pc_wait_relaxed_spin_kernel, "race",
     "relaxed spinner acquires nothing"),
    ("trb016_pc_wait_cta_scope_yes", trb016_pc_wait_cta_scope_kernel, "race",
     "cta scope does not cover the peer CTA"),
    ("trb016_pc_wait_cta_reset_yes", trb016_pc_wait_cta_reset_kernel, "race",
     "cta-scoped relaxed reset po-before the gpu publish races the spin's "
     "failed iterations"),
    ("trb016_pc_wait_atomic_reset_no", trb016_pc_wait_atomic_reset_kernel, "race-free",
     "gpu-scoped relaxed reset is mutually atomic with the poll"),
    ("trb016_pc_wait_flag_read_yes", trb016_pc_wait_flag_read_kernel, "race",
     "plain read of the awaited flag po-before the publish races the "
     "failed iterations' write-backs"),
    ("trb016_pc_wait_or_poll_no", trb016_pc_wait_or_poll_kernel, "race-free",
     "identity atomic_or(0) poll republishes the observation"),
    ("trb016_pc_wait_xor_poll_no", trb016_pc_wait_xor_poll_kernel, "race-free",
     "identity atomic_xor(0) poll republishes the observation"),
):  # fmt: skip
    _row(
        _name, _kern, _trb016_args, (BLOCK,), (2,), _exp,
        "producer-consumer-wait", _note,
        race_pair=None if _exp == "race-free" else _PC_PAIR,
    )  # fmt: skip

_MUTEX_PAIR = ("v = ct.gather(x, zero)", "ct.scatter(x, zero, v + 1)")
for _name, _kern, _exp, _note in (
    ("trb017_mutex_cas_no", trb017_mutex_kernel, "race-free",
     "CAS lock (acquire) + xchg unlock (release)"),
    ("trb017_mutex_plain_unlock_yes", trb017_mutex_plain_unlock_kernel, "race",
     "plain-store unlock breaks the release chain (and the closed world)"),
    ("trb017_mutex_relaxed_cas_yes", trb017_mutex_relaxed_cas_kernel, "race",
     "relaxed CAS acquires nothing"),
):  # fmt: skip
    _row(
        _name, _kern, _trb017_args, (), (2,), _exp, "mutex-cas", _note,
        race_pair=None if _exp == "race-free" else _MUTEX_PAIR,
    )  # fmt: skip

_row(
    "trb018_lookback_no", trb018_lookback_kernel, _trb018_args, (), (4,),
    "race-free", "lookback-chain",
    "pid i spins on flag[i-1], publishes flag[i] with release",
)  # fmt: skip
_row(
    "trb018_lookback_cta_yes", trb018_lookback_cta_scope_kernel, _trb018_args, (), (4,),
    "race", "lookback-chain", "cta scope cannot order cross-CTA neighbors",
    race_pair=(
        "prev = ct.gather(out, zero + pid - 1)",
        "ct.scatter(out, zero + pid, prev + 1)",
        "ct.scatter(out, zero + pid, 1)",
    ),
)  # fmt: skip
_row(
    "trb019_symbolic_trip_no", trb019_seg_walk_kernel, _trb019_args, (BLOCK, BLOCK),
    GRID, "race-free", "symbolic-trip-count",
    "mask k < SEG pins every iteration inside the pid's own segment",
)  # fmt: skip
_row(
    "trb019_symbolic_trip_yes", trb019_seg_walk_kernel, _trb019_args,
    (BLOCK, 2 * BLOCK), GRID, "race", "symbolic-trip-count",
    "mask k < 2*SEG: iterations SEG..n-1 spill into the next pid's segment (n=128)",
    race_pair=("ct.scatter(out, pid * SEG + k + one, 1",),
)  # fmt: skip
_row(
    "trb020_mask_overlap_no", trb020_masked_halves_kernel, _trb020_args(32, 32),
    (BLOCK,), (2,), "race-free", "partial-mask-overlap",
    "k1=k2=32: the masks tile the range exactly",
)  # fmt: skip
_row(
    "trb020_mask_overlap_yes", trb020_masked_halves_kernel, _trb020_args(40, 24),
    (BLOCK,), (2,), "race", "partial-mask-overlap",
    "k1=40, k2=24: the masks overlap on [24,40)",
    race_pair=(
        "ct.scatter(out, offs, 1, mask=(offs < k1)",
        "ct.scatter(out, offs, 2, mask=(offs >= k2)",
    ),
)  # fmt: skip

_TRB021_PAIR = ("ct.scatter(data, zero, 1, mask=prod)", "x = ct.gather(data, zero")
for _name, _kern, _exp, _note in (
    ("trb021_guarded_acq_rel_no", trb021_acq_rel_kernel, "race-free",
     "both halves of the sw edge present (control row)"),
    ("trb021_release_only_yes", trb021_release_only_kernel, "race",
     "release-only: the consumer's read side never acquires"),
    ("trb021_acquire_only_yes", trb021_acquire_only_kernel, "race",
     "acquire-only: the producer's write side never releases"),
):  # fmt: skip
    _row(
        _name, _kern, _trb021_args, (), (2,), _exp, "one-sided-sw", _note,
        race_pair=None if _exp == "race-free" else _TRB021_PAIR,
    )  # fmt: skip

_row(
    "trb022_acquire_on_failure_no", trb022_failed_cas_kernel, _trb021_args, (), (2,),
    "race-free", "acquire-on-failure",
    "the consumer's CAS always fails yet its acquire read synchronizes",
)  # fmt: skip
_row(
    "trb022_acquire_on_failure_relaxed_yes", trb022_failed_cas_relaxed_kernel,
    _trb021_args, (), (2,), "race", "acquire-on-failure",
    "same failed-CAS gate, relaxed: no sw",
    race_pair=_TRB021_PAIR,
)  # fmt: skip
_row(
    "trb023_oversized_flag_conservative", trb021_acq_rel_kernel, _trb023_args, (),
    (2,), None, "oversized-flag-demo",
    "monotonicity-lemma demo: ground truth race-free, row excluded from "
    "P/R scoring (unlabeled)",
)  # fmt: skip
_row(
    "trb024_cta_scope_pair_yes", trb024_cta_add_kernel, _trb024_args, (), GRID,
    "race", "cta-scope-pair",
    "BLOCK scope covers one CTA only: cross-CTA adds at one cell are "
    "scope-mismatched, torn, racy",
    race_pair=("ct.atomic_add(ctr, zero, 1, memory_order=MO.RELAXED, memory_scope=MS.BLOCK)",),
)  # fmt: skip
_row(
    "trb024_gpu_scope_pair_no", trb024_gpu_add_kernel, _trb024_args, (), GRID,
    "race-free", "cta-scope-pair",
    "DEVICE scope covers the peer CTA: the pair is mutually atomic",
)  # fmt: skip

_TRB025_PAIR = (
    "ct.scatter(payload, offs, ct.astype(offs + 1",
    "v = ct.gather(payload, offs)",
)
for _name, _kern, _exp, _note, _pair in (
    ("trb025_comm_comp_no", trb025_comm_comp_kernel, "race-free",
     "release arrive + acquire poll orders every comp read after the publish", None),
    ("trb025_relaxed_poll_yes", trb025_relaxed_poll_kernel, "race",
     "relaxed poll: the arrival value carries, the ordering does not", _TRB025_PAIR),
    ("trb025_poll_initial_yes", trb025_poll_initial_kernel, "race",
     "polls the wrong counter value: no acquire of the release arrival", _TRB025_PAIR),
    ("trb025_role_skip_yes", trb025_role_skip_kernel, "race",
     "one branch of the role split skips the poll", _TRB025_PAIR),
):  # fmt: skip
    _row(
        _name, _kern, _trb025_args, (1, 16), (3,), _exp, "comm-comp", _note,
        race_pair=_pair,
    )  # fmt: skip

assert len(ROWS) == 61, len(ROWS)


# ── the corpus (from the captured specs JSON) ────────────────────

CORPUS = Corpus("tritonracebench_cutile")

if SPECS_PATH.exists():
    _payload = json.loads(SPECS_PATH.read_text())
    for _name, _rec in sorted(_payload["rows"].items()):
        _meta = ROWS[_name]
        _aliases = _rec.get("aliases", {})
        CORPUS.add(
            LaunchSpec(
                name=_name,
                kernel_fn=None,
                signature={},
                constexprs=dict(_rec.get("constexprs", {})),
                make_args=lambda seed: (),
                grid=tuple(_rec["grid"]),
                expected=_meta["expected"],
                race_pair=_meta["race_pair"],
                pattern=_meta["pattern"],
                params_note=_meta["note"],
                aliased=_meta["aliased"],
                frontend="cutile",
                cutile={
                    "ir": _rec["ir"],
                    "args": _rec["args"],
                    "kernel": _rec["kernel"],
                    "module": _rec["module"],
                },
            )
        )
    CORPUS.provenance = dict(_payload.get("meta", {}))
