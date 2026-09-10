"""Seven distinct race-free repairs for the unbalanced TritonRaceBench groups.

These add coverage rather than aliases of existing repaired rows. The paired
racy source, remaining conflict opportunity and validity argument are recorded
in evaluation/TRITONRACEBENCH_REPAIRS.md. Existing cases and pins are unchanged.
"""

from typing import Any, Callable

import torch
import triton
import triton.language as tl

from evaluation.spec import LaunchSpec


@triton.jit
def role_specific_order(flag_ptr, data_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(data_ptr, 1)
        tl.debug_barrier()
        tl.atomic_cas(flag_ptr, 0, 1, sem="release", scope="gpu")
    else:
        old = tl.atomic_cas(flag_ptr, 1, 1, sem="acquire", scope="gpu")
        tl.debug_barrier()
        value = tl.load(data_ptr, mask=old == 1, other=0)
        tl.store(out_ptr + pid, value, mask=old == 1)


@triton.jit
def batch_ticket_queue(head_ptr, buf_ptr):
    pid = tl.program_id(0)
    first = tl.atomic_add(head_ptr, 2, sem="relaxed", scope="gpu")
    lanes = tl.arange(0, 2)
    tl.store(buf_ptr + first + lanes, pid)


@triton.jit
def atomic_flag_observation(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(data_ptr + offs, offs)
        tl.debug_barrier()
        flag_value = tl.atomic_or(flag_ptr, 0, sem="relaxed", scope="gpu")
        tl.store(out_ptr, flag_value)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
    else:
        while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        value = tl.load(data_ptr + offs)
        tl.store(out_ptr + pid * BLOCK + offs, value)


@triton.jit
def cas_unlock_mutex(lock_ptr, x_ptr, out_ptr):
    pid = tl.program_id(0)
    while tl.atomic_cas(lock_ptr, 0, 1, sem="acquire", scope="gpu") != 0:
        pass
    tl.debug_barrier()
    value = tl.load(x_ptr)
    tl.store(x_ptr, value + 1)
    tl.debug_barrier()
    tl.atomic_cas(lock_ptr, 1, 0, sem="release", scope="gpu")
    tl.store(out_ptr + pid, 1)


@triton.jit
def failed_cas_arrival(sem_ptr, payload_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(payload_ptr + offs, (offs + 1).to(tl.float32))
        tl.debug_barrier()
        tl.atomic_xchg(sem_ptr, 1, sem="release", scope="gpu")
    else:
        # On arrival this CAS fails, but its read still acquires the release.
        while tl.atomic_cas(sem_ptr, 0, 0, sem="acquire", scope="gpu") != 1:
            pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        value = tl.load(payload_ptr + offs)
        tl.store(out_ptr + (pid - 1) * BLOCK + offs, value)


@triton.jit
def both_consumer_branches(sem_ptr, payload_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, BLOCK)
        tl.store(payload_ptr + offs, (offs + 1).to(tl.float32))
        tl.debug_barrier()
        tl.atomic_xchg(sem_ptr, 1, sem="release", scope="gpu")
    else:
        if pid == 1:
            while tl.atomic_add(sem_ptr, 0, sem="acquire", scope="gpu") != 1:
                pass
        else:
            while tl.atomic_or(sem_ptr, 0, sem="acquire", scope="gpu") != 1:
                pass
        tl.debug_barrier()
        offs = tl.arange(0, BLOCK)
        value = tl.load(payload_ptr + offs)
        tl.store(out_ptr + (pid - 1) * BLOCK + offs, value)


@triton.jit
def fenced_tile_handoff(flag_ptr, data_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tl.store(data_ptr + offs, offs + 1, mask=pid == 0)
    tl.debug_barrier()
    compare = tl.where(pid == 0, 0, 1)
    old = tl.atomic_cas(flag_ptr, compare, 1, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    value = tl.load(data_ptr + offs, mask=(pid == 1) & (old == 1), other=0)
    tl.store(out_ptr + offs, value, mask=(pid == 1) & (old == 1))


def _scalar_args(seed):
    return tuple(torch.zeros(n, dtype=torch.int32) for n in (1, 1, 2))


def _mutex_args(seed):
    return tuple(torch.zeros(n, dtype=torch.int32) for n in (1, 1, 4))


def _queue_args(seed):
    return tuple(torch.zeros(n, dtype=torch.int32) for n in (1, 64))


def _pc_args(seed):
    return tuple(torch.zeros(n, dtype=torch.int32) for n in (1, 64, 128))


def _comm_args(seed):
    return (torch.zeros(1, dtype=torch.int32), torch.zeros(16), torch.zeros(32))


def _fence_args(seed):
    return tuple(torch.zeros(n, dtype=torch.int32) for n in (1, 16, 16))


RepairRow = tuple[
    str,
    Any,
    tuple[str, ...],
    Callable[[int], tuple],
    tuple[int, ...],
    dict[str, int],
    str,
    str,
    str,
]

REPAIR_ROWS: tuple[RepairRow, ...] = (
    (
        "trb021_role_specific_order_no",
        role_specific_order,
        ("flag_ptr", "data_ptr", "out_ptr"),
        _scalar_args,
        (2,),
        {},
        "one-sided-sw",
        "trb021_acquire_only_yes",
        "Separate release-only producer and acquire-only consumer records restore both synchronization halves.",
    ),
    (
        "trb013_batch_ticket_no",
        batch_ticket_queue,
        ("head_ptr", "buf_ptr"),
        _queue_args,
        (4,),
        {},
        "work-queue-fetch",
        "trb013_work_queue_narrow_yes",
        "Reserve two adjacent slots per atomic ticket; increment-two ranks and lane offsets produce disjoint batches.",
    ),
    (
        "trb016_atomic_flag_observation_no",
        atomic_flag_observation,
        ("flag_ptr", "data_ptr", "out_ptr"),
        _pc_args,
        (2,),
        {"BLOCK": 64},
        "producer-consumer-wait",
        "trb016_pc_wait_flag_read_yes",
        "Replace the producer's plain flag read by a gpu-scoped identity atomic OR; preserve the observation and output.",
    ),
    (
        "trb017_cas_unlock_no",
        cas_unlock_mutex,
        ("lock_ptr", "x_ptr", "out_ptr"),
        _mutex_args,
        (2,),
        {},
        "mutex-cas",
        "trb017_mutex_plain_unlock_yes",
        "Release CAS unlock replaces the plain store, retaining successful unlock and acquire lock pairing.",
    ),
    (
        "trb025_failed_cas_arrival_no",
        failed_cas_arrival,
        ("sem_ptr", "payload_ptr", "out_ptr"),
        _comm_args,
        (3,),
        {"BLOCK": 16},
        "comm-comp",
        "trb025_poll_initial_yes",
        "Poll for the published value with acquire CAS; initial zero cannot exit and the failed CAS acquires arrival.",
    ),
    (
        "trb025_both_consumer_branches_no",
        both_consumer_branches,
        ("sem_ptr", "payload_ptr", "out_ptr"),
        _comm_args,
        (3,),
        {"BLOCK": 16},
        "comm-comp",
        "trb025_role_skip_yes",
        "Both consumer branches poll, using identity add and OR respectively; the second consumer remains active.",
    ),
    (
        "trb026_fenced_tile_handoff_no",
        fenced_tile_handoff,
        ("flag_ptr", "data_ptr", "out_ptr"),
        _fence_args,
        (2,),
        {"BLOCK": 16},
        "tile-level-fence",
        "trb026_guarded_no_producer_fence_yes",
        "Both tile fences order a vector payload through a scalar publication, covering every element rather than cloning the scalar control.",
    ),
)


def register(corpus):
    for (
        name,
        kernel,
        pointers,
        args,
        grid,
        constants,
        pattern,
        paired,
        reason,
    ) in REPAIR_ROWS:
        signature = {p: "*i32" for p in pointers}
        if pattern == "comm-comp":
            signature.update(payload_ptr="*fp32", out_ptr="*fp32")
        signature.update({key: "constexpr" for key in constants})
        corpus.add(
            LaunchSpec(
                name=name,
                kernel_fn=kernel,
                signature=signature,
                constexprs=constants,
                make_args=args,
                grid=grid,
                expected="race-free",
                pattern=pattern,
                params_note=f"Repair of {paired}. {reason}",
            )
        )
