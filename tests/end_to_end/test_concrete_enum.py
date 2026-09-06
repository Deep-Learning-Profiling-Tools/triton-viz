"""Acceptance tests for the L1 rung: per-instance concrete footprint
enumeration (``triton_viz/clients/race_detector/concrete_enum.py``, the
paper repo's ``design-route1-concrete-enumeration.md`` section 7.1).

Every kernel runs under the CPU interpreter on cloned tensors; each test
pins one clause of the design: the conflict predicate (plain vs atomic,
compatible atomic pairs, cta scope), the footprint representation (lane
multiplicity, byte intervals, masks, block pointers), the value-source
premise, the named disqualifiers, and the instance ceiling.
"""

import time

import pytest
import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceType
from triton_viz.clients.race_detector.concrete_enum import (
    ENUM_MAX_INSTANCES,
    enumerate_launch,
)


def _run(kernel, grid, *args, **kwargs):
    triton_viz.clear()
    return enumerate_launch(kernel, args, kwargs, grid)


def _line_no(kernel, needle: str) -> int:
    import inspect

    fn = getattr(kernel, "fn", kernel)
    lines, start = inspect.getsourcelines(fn)
    for i, line in enumerate(lines):
        if needle in line:
            return start + i
    raise AssertionError(f"needle {needle!r} not found")


# ── the scatter litmus pair ────────────────────────────────────────


@triton.jit
def _scatter_kernel(idx_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + i, v)


def test_scatter_duplicate_destinations_race_with_concrete_witness():
    idx = torch.zeros(8, dtype=torch.int32)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    o = _run(_scatter_kernel, (2,), idx, x, out, BLOCK=4)
    assert o.status == "races"
    assert o.n_instances == 2
    cross = [r for r in o.reports if r.witness_grid_a != r.witness_grid_b]
    assert cross, "the two instances collide at out[0]"
    rep = cross[0]
    assert rep.race_type == RaceType.WAW
    # witnesses are translated back to the CALLER's tensor
    assert rep.witness_addr == out.data_ptr()
    assert rep.byte_range == (out.data_ptr(), out.data_ptr() + 4)
    line = _line_no(_scatter_kernel, "tl.store(out_ptr + i, v)")
    assert rep.first_record.source_location[1] == line
    assert rep.second_record.source_location[1] == line
    # the index loads are value sources, the data loads are not
    assert o.n_value_source_loads == 2


def test_scatter_permutation_proves_clean_and_leaves_tensors_untouched():
    idx = torch.tensor([3, 1, 0, 2, 7, 5, 4, 6], dtype=torch.int32)
    x = torch.arange(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    o = _run(_scatter_kernel, (2,), idx, x, out, BLOCK=4)
    assert o.status == "ok"
    assert o.reason is None
    assert o.reports == []
    # the run executes on clones: the caller's contents are the launch's
    assert bool((out == 0).all())


# ── duplicate positions inside one operation (the A1 shape) ──────────


@triton.jit
def _dup_lane_store_kernel(out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs % 2, offs)


def test_duplicate_lanes_of_one_store_race_within_the_instance():
    o = _run(_dup_lane_store_kernel, (1,), torch.zeros(8, dtype=torch.int32), BLOCK=8)
    assert o.status == "races"
    rep = o.reports[0]
    assert rep.witness_grid_a == rep.witness_grid_b == (0, 0, 0)
    assert rep.race_type == RaceType.WAW


@triton.jit
def _two_stores_same_address_kernel(out_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, 1.0)
    tl.store(out_ptr + pid, 2.0)


def test_unfenced_same_instance_stores_across_operations_race(monkeypatch):
    from triton_viz.core.config import config as cfg

    monkeypatch.setattr(cfg, "race_detector_fence_order", True)
    o = _run(_two_stores_same_address_kernel, (4,), torch.zeros(4))
    assert o.status == "races"
    assert o.reports[0].race_type == RaceType.WAW


@triton.jit
def _atomic_scatter_kernel(idx_ptr, hist_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    tl.atomic_add(hist_ptr + i, 1, sem="relaxed", scope="gpu")


def test_duplicate_lanes_of_one_atomic_do_not_race():
    idx = torch.zeros(16, dtype=torch.int32)  # every lane hits hist[0]
    o = _run(
        _atomic_scatter_kernel, (2,), idx, torch.zeros(4, dtype=torch.int32), BLOCK=8
    )
    assert o.status == "ok"


# ── the conflict predicate ────────────────────────────────────────


@triton.jit
def _plain_vs_atomic_kernel(c_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(c_ptr, 5)
    else:
        tl.atomic_add(c_ptr, 1)


def test_plain_store_overlapping_another_instances_atomic_is_a_race():
    o = _run(_plain_vs_atomic_kernel, (2,), torch.zeros(1, dtype=torch.int32))
    assert o.status == "races"
    assert o.reports[0].race_type == RaceType.WAW
    assert {o.reports[0].witness_grid_a, o.reports[0].witness_grid_b} == {
        (0, 0, 0),
        (1, 0, 0),
    }


@triton.jit
def _histogram_kernel(x_ptr, hist_ptr, BLOCK: tl.constexpr, SCOPE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)
    tl.atomic_add(hist_ptr + v, 1, sem="relaxed", scope=SCOPE)


def test_gpu_scope_atomics_on_one_cell_are_a_compatible_pair():
    x = torch.randint(0, 4, (32,), dtype=torch.int32)
    o = _run(
        _histogram_kernel,
        (4,),
        x,
        torch.zeros(4, dtype=torch.int32),
        BLOCK=8,
        SCOPE="gpu",
    )
    assert o.status == "ok"


def test_cta_scope_atomics_across_instances_race():
    x = torch.zeros(32, dtype=torch.int32)
    o = _run(
        _histogram_kernel,
        (4,),
        x,
        torch.zeros(4, dtype=torch.int32),
        BLOCK=8,
        SCOPE="cta",
    )
    assert o.status == "races"
    assert o.reports[0].witness_grid_a != o.reports[0].witness_grid_b


@triton.jit
def _mixed_width_kernel(buf_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(buf_ptr, 7)  # int32: bytes [0, 4)
    else:
        narrow = buf_ptr.to(tl.pointer_type(tl.int8))
        tl.store(narrow + 2, 1)  # int8: bytes [2, 3)


def test_byte_intervals_catch_mixed_width_overlap():
    o = _run(_mixed_width_kernel, (2,), torch.zeros(1, dtype=torch.int32))
    assert o.status == "races"
    lo, hi = o.reports[0].byte_range
    assert hi - lo == 1


@triton.jit
def _torn_atomic_kernel(buf_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.atomic_add(buf_ptr, 1)  # int64 at bytes [0, 8)
    else:
        narrow = buf_ptr.to(tl.pointer_type(tl.int32))
        tl.atomic_add(narrow + 1, 1)  # int32 at bytes [4, 8): torn against the int64


def test_atomics_of_different_width_at_overlapping_bytes_race():
    o = _run(_torn_atomic_kernel, (2,), torch.zeros(1, dtype=torch.int64))
    assert o.status == "races"
    lo, hi = o.reports[0].byte_range
    assert hi - lo == 4


# ── plain-data reads are unrestricted, and their overlaps are reported ──


@triton.jit
def _plain_rw_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(x_ptr + offs, 1.0)
    else:
        v = tl.load(x_ptr + offs)
        tl.store(y_ptr + offs, v)


def test_plain_data_read_write_overlap_is_reported_as_a_race():
    o = _run(_plain_rw_kernel, (2,), torch.zeros(4), torch.zeros(4), BLOCK=4)
    assert o.status == "races"
    rep = o.reports[0]
    assert rep.race_type == RaceType.RAW
    assert rep.first_record.access_mode == "write"
    assert rep.second_record.access_mode == "read"
    assert o.n_value_source_loads == 0


# ── the value-source premise (A2) ──────────────────────────────────


@triton.jit
def _written_index_kernel(idx_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(idx_ptr + offs, 3)
    i = tl.load(idx_ptr + offs)
    tl.store(out_ptr + pid * BLOCK + i, 1.0)


def test_value_source_load_from_a_written_region_refuses_by_name():
    o = _run(
        _written_index_kernel,
        (2,),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(64),
        BLOCK=4,
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("value-source:")
    assert "premise" in o.reason


@triton.jit
def _mask_from_written_flag_kernel(flag_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    if pid == 0:
        tl.store(flag_ptr + offs, 1)
    f = tl.load(flag_ptr + offs)
    tl.store(out_ptr + offs, 1.0, mask=f == 1)


def test_value_source_through_a_mask_is_also_the_premise():
    o = _run(
        _mask_from_written_flag_kernel,
        (2,),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4),
        BLOCK=4,
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("value-source:")


# ── atomic return values at footprint positions ──────────────────────


@triton.jit
def _ticket_kernel(head_ptr, buf_ptr):
    pid = tl.program_id(0)
    idx = tl.atomic_add(head_ptr, 1, sem="relaxed")
    tl.store(buf_ptr + idx, pid)


def test_ticket_idiom_refuses_atomic_return_in_address():
    o = _run(
        _ticket_kernel,
        (4,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("atomic-return:")
    assert "address" in o.reason


@triton.jit
def _last_block_kernel(cnt_ptr, out_ptr, n):
    old = tl.atomic_add(cnt_ptr, 1)
    if old == n - 1:
        tl.store(out_ptr, 1.0)


def test_last_block_idiom_refuses_atomic_return_in_branch():
    o = _run(
        _last_block_kernel, (4,), torch.zeros(1, dtype=torch.int32), torch.zeros(1), 4
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("atomic-return:")
    assert "branch" in o.reason


@triton.jit
def _atomic_loop_bound_kernel(cnt_ptr, out_ptr):
    pid = tl.program_id(0)
    old = tl.atomic_add(cnt_ptr, 1)
    for i in range(old):
        tl.store(out_ptr + pid * 8 + i, 1.0)


def test_atomic_return_in_loop_bound_refuses():
    o = _run(
        _atomic_loop_bound_kernel,
        (2,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(64),
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("atomic-return:")
    assert "loop bound" in o.reason


@triton.jit
def _atomic_return_as_data_kernel(cnt_ptr, out_ptr):
    pid = tl.program_id(0)
    old = tl.atomic_add(cnt_ptr, 1)
    tl.store(out_ptr + pid, old)  # the return reaches only stored data


def test_atomic_return_reaching_only_stored_data_is_fine():
    o = _run(
        _atomic_return_as_data_kernel,
        (4,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )
    assert o.status == "ok"


@triton.jit
def _spin_kernel(flag_ptr, out_ptr):
    pid = tl.program_id(0)
    if pid == 1:
        while tl.atomic_add(flag_ptr, 0, sem="acquire") != 1:
            pass
        tl.store(out_ptr, 1.0)
    else:
        tl.store(out_ptr, 2.0)
        tl.atomic_xchg(flag_ptr, 1, sem="release")


def test_spin_on_an_atomic_poll_refuses_without_spinning():
    t0 = time.perf_counter()
    o = _run(_spin_kernel, (2,), torch.zeros(1, dtype=torch.int32), torch.zeros(1))
    assert o.status == "unsupported"
    assert o.reason.startswith("atomic-return:")
    assert time.perf_counter() - t0 < 5.0


# ── structure the symbolic frontends refuse, decided here ────────────


@triton.jit
def _pid_branch_nested_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        for i in range(n):
            for j in range(2):
                tl.store(out_ptr + i * 2 + j, 1.0)
    else:
        offs = tl.arange(0, BLOCK)
        tl.store(out_ptr + 64 + pid * BLOCK + offs, 2.0)


def test_pid_branch_with_nested_loops_is_decided_both_ways():
    clean = _run(_pid_branch_nested_kernel, (4,), torch.zeros(256), 8, BLOCK=4)
    assert clean.status == "ok"
    racy = _run(_pid_branch_nested_kernel, (4,), torch.zeros(256), 100, BLOCK=4)
    assert racy.status == "races"
    assert racy.reports[0].race_type == RaceType.WAW


@triton.jit
def _data_dependent_trip_kernel(n_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.load(n_ptr + pid)
    for i in range(n):
        tl.store(out_ptr + pid * BLOCK + i, 1.0)


def test_data_dependent_trip_count_is_decided_and_its_load_is_a_value_source():
    n = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    o = _run(_data_dependent_trip_kernel, (4,), n, torch.zeros(64), BLOCK=8)
    assert o.status == "ok"
    assert o.n_value_source_loads == 4
    racy = _run(
        _data_dependent_trip_kernel,
        (4,),
        torch.tensor([9, 2, 3, 4], dtype=torch.int32),
        torch.zeros(64),
        BLOCK=8,
    )
    assert racy.status == "races"


@triton.jit
def _masked_tail_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, 1.0, mask=offs < n)


def test_masked_off_lanes_are_not_footprints():
    # pointers of the masked-off lanes point past the tensor; a footprint
    # that ignored the mask would read as a same-instance overlap or worse
    o = _run(_masked_tail_kernel, (3,), torch.zeros(10), 10, BLOCK=4)
    assert o.status == "ok"
    assert o.n_ops == 3


@triton.jit
def _block_ptr_kernel(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    bp = tl.make_block_ptr(x_ptr, (M, N), (N, 1), (pid * BM, 0), (BM, BN), (1, 0))
    v = tl.load(bp, boundary_check=(0, 1))
    op = tl.make_block_ptr(y_ptr, (M, N), (N, 1), (0, 0), (BM, BN), (1, 0))
    tl.store(op, v, boundary_check=(0, 1))


def test_block_pointer_accesses_are_recorded():
    o = _run(
        _block_ptr_kernel, (2,), torch.zeros(8, 8), torch.zeros(8, 8), 8, 8, BM=4, BN=8
    )
    assert o.status == "races"  # both instances store the same 4x8 tile of y
    assert o.n_ops == 4
    lo, hi = o.reports[0].byte_range
    assert hi - lo == 4 * 8 * 4


@triton.jit
def _unmasked_copy_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2)


def test_unmasked_accesses_are_recorded_once():
    # tl.load/tl.store without a mask fire the raw AND the masked builder
    # hooks; the recorder must count each access exactly once
    o = _run(_unmasked_copy_kernel, (4,), torch.zeros(16), torch.zeros(16), BLOCK=4)
    assert o.status == "ok"
    assert o.n_ops == 8
    assert o.n_instances == 4
    assert o.instance_s is not None and o.max_instance_s is not None


# ── the instance ceiling and non-concrete grids ──────────────────────


def test_grid_above_the_ceiling_refuses_by_name_before_executing():
    t0 = time.perf_counter()
    o = _run(
        _unmasked_copy_kernel,
        (ENUM_MAX_INSTANCES + 1,),
        torch.zeros(16),
        torch.zeros(16),
        BLOCK=4,
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("instance-ceiling:")
    assert o.n_ops == 0
    assert time.perf_counter() - t0 < 1.0


def test_callable_grid_refuses_by_name():
    o = _run(
        _unmasked_copy_kernel,
        lambda meta: (4,),
        torch.zeros(16),
        torch.zeros(16),
        BLOCK=4,
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("no-grid:")


# ── the symbolic frontends stay untouched by a preceding enumeration ──


def test_enumeration_leaves_the_interpreter_patches_clean():
    """After a run (and after a refusal that aborts mid-kernel) the
    builder and the tensor dunders must be restored, so a subsequent
    symbolic launch behaves exactly as before."""
    from triton.runtime.interpreter import interpreter_builder

    from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector

    _run(
        _ticket_kernel,
        (4,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
    )
    leftovers = [
        name
        for name in dir(interpreter_builder)
        if getattr(
            getattr(interpreter_builder, name, None), "_tilerace_taint_wrapper", False
        )
    ]
    assert leftovers == []
    triton_viz.clear()
    det = SymbolicRaceDetector()
    traced = triton_viz.trace(client=det)(_scatter_kernel)
    idx = torch.zeros(8, dtype=torch.int32)
    traced[(2,)](idx, torch.zeros(8), torch.zeros(8), 4)
    assert det.last_status == "ok"
    assert det.last_reports


@pytest.mark.parametrize("n_instances", [1, 7, 64])
def test_every_instance_of_the_grid_is_evaluated(n_instances):
    o = _run(
        _unmasked_copy_kernel,
        (n_instances,),
        torch.zeros(4 * n_instances),
        torch.zeros(4 * n_instances),
        BLOCK=4,
    )
    assert o.status == "ok"
    assert o.n_instances == n_instances
    assert o.n_ops == 2 * n_instances


# ── aliased arguments keep aliasing on the clones ────────────────────


@triton.jit
def _shift_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs + 1, mask=offs + 1 < n, other=0.0)
    tl.store(out_ptr + offs, v, mask=offs < n)


def test_in_place_aliased_arguments_race_across_instances():
    """trb009's shape: the same tensor passed as input and output; each
    instance reads the element its neighbour writes. Cloning per argument
    would separate the aliases and prove a launch that never existed."""
    x = torch.zeros(16)
    o = _run(_shift_kernel, (4,), x, x, 16, BLOCK=4)
    assert o.status == "races"
    assert o.reports[0].race_type in (RaceType.RAW, RaceType.WAR)
    # the distinct-tensor twin is clean
    o = _run(_shift_kernel, (4,), torch.zeros(16), torch.zeros(16), 16, BLOCK=4)
    assert o.status == "ok"


def test_views_of_one_storage_keep_their_offsets_on_the_clone():
    base = torch.zeros(32)
    lo, hi = base[:16], base[16:]
    # instances write disjoint halves through the two views: clean
    o = _run(_shift_kernel, (4,), hi, lo, 16, BLOCK=4)
    assert o.status == "ok"
    # the same view twice: the in-place race again, and the witness lands
    # inside the CALLER's storage
    o = _run(_shift_kernel, (4,), hi, hi, 16, BLOCK=4)
    assert o.status == "races"
    assert hi.data_ptr() <= o.reports[0].witness_addr < hi.data_ptr() + 16 * 4


# ── taint through memory within an instance ─────────────────────────


@triton.jit
def _state_update_kernel(state_ptr, x_ptr, BLOCK: tl.constexpr):
    # the causal-conv shape: read the state, use it, write it back in place
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(state_ptr + offs)
    v = tl.load(x_ptr + offs + s, mask=offs + s < 64, other=0.0)
    tl.store(state_ptr + offs, s + 1)
    tl.store(x_ptr + offs, v)


def test_same_instance_in_place_state_update_is_decided():
    state = torch.zeros(16, dtype=torch.int32)
    o = _run(_state_update_kernel, (4,), state, torch.zeros(64), BLOCK=4)
    assert o.status == "ok"
    assert o.n_value_source_loads == 4


@triton.jit
def _relay_index_kernel(idx_ptr, scratch_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    tl.store(scratch_ptr + offs, i)  # relay through this instance's scratch
    tl.debug_barrier()
    j = tl.load(scratch_ptr + offs)
    tl.store(out_ptr + j, 1.0)


def test_relayed_index_makes_the_original_load_a_value_source():
    idx = torch.arange(8, dtype=torch.int32)
    o = _run(
        _relay_index_kernel,
        (2,),
        idx,
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(8),
        BLOCK=4,
    )
    assert o.status == "ok"
    assert o.n_value_source_loads == 4  # both the scratch loads and the idx loads


@triton.jit
def _relay_written_index_kernel(idx_ptr, scratch_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if pid == 1:
        tl.store(idx_ptr + offs - BLOCK, 0)  # instance 1 writes instance 0's indices
    i = tl.load(idx_ptr + offs)
    tl.store(scratch_ptr + offs, i)
    tl.debug_barrier()
    j = tl.load(scratch_ptr + offs)
    tl.store(out_ptr + j, 1.0)


def test_relayed_index_from_a_foreign_written_region_refuses():
    idx = torch.arange(8, dtype=torch.int32)
    o = _run(
        _relay_written_index_kernel,
        (2,),
        idx,
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(8),
        BLOCK=4,
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("value-source:")


@triton.jit
def _relay_ticket_kernel(head_ptr, scratch_ptr, buf_ptr):
    pid = tl.program_id(0)
    t = tl.atomic_add(head_ptr, 1)
    tl.store(scratch_ptr + pid, t)  # the ticket goes through memory...
    tl.debug_barrier()
    idx = tl.load(scratch_ptr + pid)
    tl.store(buf_ptr + idx, pid)  # ...and still reaches an address


def test_atomic_return_relayed_through_memory_refuses():
    o = _run(
        _relay_ticket_kernel,
        (4,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
    )
    assert o.status == "unsupported"
    assert o.reason.startswith("atomic-return:")
    assert "through memory" in o.reason


# ── the projected-cost refusal ─────────────────────────────────────


@triton.jit
def _slow_kernel(out_ptr, n_iter, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    for i in range(n_iter):
        tl.store(out_ptr + offs, i * 1.0)
        tl.debug_barrier()  # keep the projection control free of WAW conflicts


def test_projected_cost_refuses_early_by_name():
    """Instances that cost tens of milliseconds each on a grid that
    cannot finish in the budget: the refusal comes soon after the 5 s
    grace period, not at the watchdog."""
    t0 = time.perf_counter()
    o = enumerate_launch(
        _slow_kernel, (torch.zeros(4 * 4000), 400), {"BLOCK": 4}, (4000,), timeout_s=30
    )
    elapsed = time.perf_counter() - t0
    assert o.status == "unsupported"
    assert o.reason.startswith("projected-cost:")
    assert "per instance after the first" in o.reason
    assert 5.0 <= elapsed < 20.0
    assert o.n_instances > 1


def test_projection_leaves_a_launch_that_fits_alone():
    o = enumerate_launch(
        _slow_kernel, (torch.zeros(4 * 8), 50), {"BLOCK": 4}, (8,), timeout_s=30
    )
    assert o.status == "ok"
    assert o.n_instances == 8


# ── the in-bounds premise ──────────────────────────────────────────


@triton.jit
def _oob_store_kernel(out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, 1.0)  # no mask: the last instance runs past n


def test_out_of_bounds_store_refuses_by_name_before_executing():
    out = torch.zeros(6)
    o = _run(_oob_store_kernel, (2,), out, 6, BLOCK=4)
    assert o.status == "unsupported"
    assert o.reason.startswith("out-of-bounds:")
    assert "store" in o.reason and "instance (1, 0, 0)" in o.reason
    # instance 0 executed, instance 1 was refused at its first access
    assert o.n_instances == 2
    assert o.n_ops == 1


@triton.jit
def _oob_load_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs)  # unmasked read past x
    tl.store(out_ptr + offs, v, mask=offs < n)


def test_out_of_bounds_load_refuses_too():
    o = _run(_oob_load_kernel, (2,), torch.zeros(6), torch.zeros(8), 6, BLOCK=4)
    assert o.status == "unsupported"
    assert o.reason.startswith("out-of-bounds:")
    assert "load" in o.reason


def test_masked_off_lanes_may_point_anywhere():
    # the tail-guard idiom: lanes past n exist as pointers but are masked
    o = _run(_masked_tail_kernel, (3,), torch.zeros(10), 10, BLOCK=4)
    assert o.status == "ok"


@triton.jit
def _view_kernel(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, 1.0)


def test_bounds_are_the_storage_not_the_view():
    # a view into the middle of a storage: writing past the view's end but
    # inside the storage is in bounds for the premise (the model's bound is
    # the allocation); writing past the storage is not
    base = torch.zeros(16)
    o = _run(_view_kernel, (1,), base[4:8], BLOCK=8)  # [4, 12) of 16: inside
    assert o.status == "ok"
    o = _run(_view_kernel, (1,), base[12:16], BLOCK=8)  # [12, 20): past
    assert o.status == "unsupported"
    assert o.reason.startswith("out-of-bounds:")


def test_bounds_check_is_off_without_a_memory_map():
    from triton_viz.clients.race_detector.concrete_enum import (
        ConcreteFootprintRecorder,
    )

    assert ConcreteFootprintRecorder().check_bounds is False
    assert ConcreteFootprintRecorder(bounds=[]).check_bounds is True


def test_a_mid_kernel_refusal_leaves_the_language_state_clean():
    """Regression: after a refusal raised inside the kernel, the trace's
    own restore had already put the originals back and the recorder's
    cleanup re-installed the interpreter's reduce/scan and the builder's
    PatchOps it had captured, breaking the next real compile in the same
    process (seen under runner process reuse)."""
    import triton.language as tl_mod
    from triton.runtime.interpreter import interpreter_builder

    before = {
        n: getattr(tl_mod, n)
        for n in ("reduce", "associative_scan", "range", "static_range")
    }
    before_core = {n: getattr(tl_mod.core, n) for n in ("reduce", "associative_scan")}
    before_builder = {
        n: getattr(interpreter_builder, n)
        for n in ("create_addptr", "create_masked_load")
    }
    o = _run(_oob_store_kernel, (2,), torch.zeros(6), 6, BLOCK=4)
    assert o.reason.startswith("out-of-bounds:")
    for n, v in before.items():
        assert getattr(tl_mod, n) is v, n
    for n, v in before_core.items():
        assert getattr(tl_mod.core, n) is v, n
    for n, v in before_builder.items():
        assert getattr(interpreter_builder, n) is v, n
    # and the same after a ticket refusal (raised from a taint sink)
    _run(
        _ticket_kernel,
        (4,),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(64, dtype=torch.int32),
    )
    for n, v in before.items():
        assert getattr(tl_mod, n) is v, n
