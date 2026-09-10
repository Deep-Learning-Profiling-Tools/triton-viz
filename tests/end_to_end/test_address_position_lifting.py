"""Acceptance tests for the address-position lifting
(address_position_lifting_spec.md §4/§5).

A plain ``tl.load`` embedded in an event ADDRESS lowers to a ``Select``
over the read-only launch snapshot with its domain facts; verdicts on
such launches are per-launch + contents-snapshot scoped. The six
families below ARE the in-repo backing of the witness theorem's A1/A2
transport (the theorem lives in the paper): written-index fail-stop,
OOB-index domains, index/data aliasing, masked-gather defaults, the
co-admitted atomic-consumer surface, and the two latent-trap pins.
"""

import torch
import triton
import triton.language as tl

import triton_viz
from triton_viz.clients import RaceType
from triton_viz.clients.race_detector.race_detector import SymbolicRaceDetector


def _run(kernel, grid, *args):
    triton_viz.clear()
    det = SymbolicRaceDetector()
    traced = triton_viz.trace(client=det)(kernel)
    traced[grid](*args)
    return det


# ── the scatter litmus pair (spec §5.1, the driving tests) ─────────


@triton.jit
def _scatter_kernel(idx_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    v = tl.load(x_ptr + offs)
    tl.store(out_ptr + i, v)


def test_scatter_all_zero_index_races_with_concrete_witness():
    """All indices 0: every block stores out[0] — WAW across pids, and
    the witness byte must be CONCRETE (the snapshot array completes the
    model) and land on out's base."""
    idx = torch.zeros(8, dtype=torch.int32)
    x = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    det = _run(_scatter_kernel, (2,), idx, x, out, 4)
    assert det.last_status == "ok"
    assert det.last_reports
    assert det.last_premises == ("contents-snapshot",)
    rep = det.last_reports[0]
    assert rep.race_type == RaceType.WAW
    # alpha-renaming pin: the two copies must collide at DISTINCT pids
    assert rep.witness_grid_a != rep.witness_grid_b
    # witness concreteness pin: as_long() succeeded and hit out[0]
    assert rep.witness_addr == out.data_ptr()


def test_scatter_identity_index_is_clean():
    """The disjoint-index control: an identity permutation writes every
    slot once — proved clean within the per-launch + contents scope."""
    idx = torch.arange(8, dtype=torch.int32)
    x = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    det = _run(_scatter_kernel, (2,), idx, x, out, 4)
    assert det.last_status == "ok"
    assert det.last_reports == []
    assert det.last_premises == ("contents-snapshot",)


def test_gather_read_side_indirection_is_clean():
    """The classic gather (indices in a load ADDRESS, read side only)
    upgrades from abstention to a clean verdict."""

    @triton.jit
    def kernel(idx_ptr, src_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        idx = tl.load(idx_ptr + offs, mask=mask, other=0)
        vals = tl.load(src_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + offs, vals, mask=mask)

    idx = torch.randint(0, 8, (8,), dtype=torch.int32)
    src = torch.randn(8)
    out = torch.zeros(8)
    det = _run(kernel, (2,), idx, src, out, 8, 4)
    assert det.last_status == "ok"
    assert det.last_reports == []


# ── family 1: written-index fail-stop (spec §4.1) ──────────────────


def test_write_before_index_load_fail_stops():
    @triton.jit
    def kernel(idx_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(idx_ptr + offs, offs)  # write FIRST
        i = tl.load(idx_ptr + offs)
        tl.store(out_ptr + i, 1.0)

    idx = torch.zeros(8, dtype=torch.int32)
    out = torch.zeros(8, dtype=torch.float32)
    det = _run(kernel, (2,), idx, out, 4)
    assert det.last_status == "unsupported"
    assert "written by this kernel" in (det.unsupported_reason or "")


def test_write_after_index_load_fail_stops():
    @triton.jit
    def kernel(idx_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        i = tl.load(idx_ptr + offs)
        tl.store(out_ptr + i, 1.0)
        tl.store(idx_ptr + offs, offs)  # write AFTER the snapshot

    idx = torch.zeros(8, dtype=torch.int32)
    out = torch.zeros(8, dtype=torch.float32)
    det = _run(kernel, (2,), idx, out, 4)
    assert det.last_status == "unsupported"
    assert "previously read as a tl.load value source" in (det.unsupported_reason or "")


# ── family 2: OOB-index domains (spec §4.2) ────────────────────────


def test_oob_index_values_still_race_through_the_same_slot():
    """Index VALUES pointing outside out's extent are the TRUE addresses;
    two instances scattering through the same OOB slot must still race —
    domain facts pin the INNER address, never the outer one."""
    idx = torch.full((8,), 100, dtype=torch.int32)  # far outside out[0..8)
    x = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    det = _run(_scatter_kernel, (2,), idx, x, out, 4)
    assert det.last_status == "ok"
    assert det.last_reports
    assert det.last_reports[0].race_type == RaceType.WAW


# ── family 3: index/data tensor aliasing (spec §4.3) ───────────────


def test_index_tensor_aliasing_written_output_fail_stops():
    """The same storage passed as idx_ptr AND out_ptr: the store target
    overlaps the snapshot region — region tracking fail-stops."""
    buf = torch.zeros(8, dtype=torch.int32)
    x = torch.zeros(8, dtype=torch.float32)

    @triton.jit
    def kernel(idx_ptr, x_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        i = tl.load(idx_ptr + offs)
        v = tl.load(x_ptr + offs)
        tl.store(out_ptr + i, v.to(tl.int32))

    det = _run(kernel, (2,), buf, x, buf, 4)
    assert det.last_status == "unsupported"
    assert "previously read as a tl.load value source" in (det.unsupported_reason or "")


# ── family 4: masked-gather default interplay (spec §4.4) ──────────


@triton.jit
def _masked_scatter_kernel(
    idx_ptr, out_ptr, n, DEFAULT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    i = tl.load(idx_ptr + offs, mask=mask, other=DEFAULT)
    tl.store(out_ptr + i, 1.0)


def test_masked_index_default_is_the_real_address_and_races():
    """Masked-off lanes address out[DEFAULT] — the semantically true
    address. Two pids' masked-off lanes share DEFAULT=0 → WAW."""
    idx = torch.arange(8, dtype=torch.int32)  # in-bounds lanes disjoint
    out = torch.zeros(16, dtype=torch.float32)
    # n=2: lanes ≥ 2 masked off in BOTH pids → both store out[0]
    det = _run(_masked_scatter_kernel, (2,), idx, out, 2, 0, 4)
    assert det.last_status == "ok"
    assert det.last_reports
    assert det.last_reports[0].race_type == RaceType.WAW


def test_masked_index_disjoint_defaults_prove_clean():
    """Identity indices + a default that no active lane can produce:
    every address (active or defaulted) stays per-pid disjoint... the
    default 15 is shared across pids, so keep every lane ACTIVE instead:
    full-mask launch — clean."""
    idx = torch.arange(8, dtype=torch.int32)
    out = torch.zeros(16, dtype=torch.float32)
    det = _run(_masked_scatter_kernel, (2,), idx, out, 8, 15, 4)  # n=8: all active
    assert det.last_status == "ok"
    assert det.last_reports == []


def test_masked_index_load_without_other_is_unsupported():
    @triton.jit
    def kernel(idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = tl.load(idx_ptr + offs, mask=mask)  # no other=
        tl.store(out_ptr + i, 1.0)

    idx = torch.arange(8, dtype=torch.int32)
    out = torch.zeros(16, dtype=torch.float32)
    det = _run(kernel, (2,), idx, out, 2, 4)
    assert det.last_status == "unsupported"
    assert "without explicit `other`" in (det.unsupported_reason or "")


# ── family 5: the co-admitted atomic-consumer surface (spec §4.5) ──


@triton.jit
def _atomic_scatter_kernel(idx_ptr, dst_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    tl.atomic_add(dst_ptr + i, 1, sem="relaxed", scope="gpu")


@triton.jit
def _store_scatter_kernel(idx_ptr, dst_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(idx_ptr + offs)
    tl.store(dst_ptr + i, 1)


def test_atomic_add_through_loaded_index_all_equal_is_clean():
    """gpu-scope adds through an all-equal loaded index hit ONE cell —
    mutually atomic (inclusive scopes, same width, same address)."""
    idx = torch.zeros(8, dtype=torch.int32)
    dst = torch.zeros(8, dtype=torch.int32)
    det = _run(_atomic_scatter_kernel, (2,), idx, dst, 4)
    assert det.last_status == "ok"
    assert det.last_reports == []
    assert det.last_premises == ("contents-snapshot",)


def test_plain_store_twin_through_loaded_index_races():
    idx = torch.zeros(8, dtype=torch.int32)
    dst = torch.zeros(8, dtype=torch.int32)
    det = _run(_store_scatter_kernel, (2,), idx, dst, 4)
    assert det.last_status == "ok"
    assert det.last_reports
    assert det.last_reports[0].race_type == RaceType.WAW


def test_cas_through_loaded_index_keeps_domain_constraints():
    """The CAS-site constraint pin (spec §1 fragility): a CAS whose
    ADDRESS is a loaded index must carry the snapshot domain facts into
    the query — an out-of-range fabricated collision with the disjoint
    OUT tensor would otherwise be SAT. Identity indices → per-pid
    disjoint CAS cells → clean; and the premise is recorded."""

    @triton.jit
    def kernel(idx_ptr, lock_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = tl.load(idx_ptr + pid)
        tl.atomic_cas(lock_ptr + i, 0, 1, sem="acq_rel", scope="gpu")

    idx = torch.arange(2, dtype=torch.int32)
    lock = torch.zeros(2, dtype=torch.int32)
    det = _run(kernel, (2,), idx, lock, 1)
    assert det.last_status == "ok"
    assert det.last_reports == []
    assert det.last_premises == ("contents-snapshot",)


# ── family 6: latent-trap pins (spec §4.6) ─────────────────────────


def test_lifted_records_never_hit_the_finalize_force_eval_path():
    """rd:1324's defensive force-eval drops constraint conjunctions; a
    lifted-address record must arrive at finalize with its addr already
    a Z3 expr (pre-evaluated), never a raw SymbolicExpr — pinned through
    the report records (the recorded events are what the solver saw)."""
    from triton_viz.clients.symbolic_engine import SymbolicExpr

    idx = torch.zeros(8, dtype=torch.int32)
    x = torch.zeros(8, dtype=torch.float32)
    out = torch.zeros(8, dtype=torch.float32)
    det = _run(_scatter_kernel, (2,), idx, x, out, 4)
    assert det.last_reports
    for rep in det.last_reports:
        for rec in (rep.first_record, rep.second_record):
            assert not isinstance(rec.addr_expr, SymbolicExpr)


def test_snapshot_serves_are_cache_consistent_across_positions():
    """First-lowering-wins node caching: a load consumed FIRST as a
    value and THEN as an address must still produce the lifted address
    (the cached lowering was made under the provider, so both uses see
    the same select term). The all-equal index consumed both ways must
    therefore still race on the scatter side."""

    @triton.jit
    def kernel(idx_ptr, out_ptr, sum_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        i = tl.load(idx_ptr + offs)
        s = tl.sum(i, axis=0)  # VALUE use first
        tl.store(sum_ptr + pid, s)
        tl.store(out_ptr + i, 1.0)  # ADDRESS use second

    idx = torch.zeros(8, dtype=torch.int32)
    out = torch.zeros(8, dtype=torch.float32)
    ssum = torch.zeros(2, dtype=torch.int32)
    det = _run(kernel, (2,), idx, out, ssum, 4)
    assert det.last_status == "ok"
    assert any(r.race_type == RaceType.WAW for r in det.last_reports)
    assert det.last_premises == ("contents-snapshot",)
