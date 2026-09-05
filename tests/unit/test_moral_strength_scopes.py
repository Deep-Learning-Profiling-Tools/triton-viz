"""Moral-strength alignment of the conflict predicate (Tile IR / PTX).

Two conflicting atomics are exempt from the race predicate only when they
are morally strong w.r.t. each other: mutually INCLUSIVE scopes (for the
cross-CTA pairs the two-copy queries pose, both scopes in {gpu, sys}),
same access width, exact same address. Everything else — either side
cta-scoped, mixed widths, torn overlap — races like plain writes.

This is the implemented-semantics record for the paper's Def. conflict
(the divergence caveat can be dropped): each exempt cell has a mutation
twin proving the exemption is load-bearing, and each racy cell asserts
the scope-mismatch report.
"""

from __future__ import annotations

import pytest
from z3 import IntVal

from triton_viz.clients.race_detector.data import AccessEventRecord
from triton_viz.clients.race_detector.two_copy_symbolic_hb_solver import (
    TwoCopySymbolicHBSolver,
)
from triton_viz.core.data import AtomicRMW

FLAG = 1 << 20


def _atomic(scope: str, *, event_id: int, elem_size: int = 4, addr: int = FLAG):
    """A relaxed footprint-only RMW (no value modeling: old_value=None
    keeps the pair outside the rf machinery, isolating the CONFLICT
    predicate)."""
    return AccessEventRecord(
        op_type=AtomicRMW,
        access_mode="read",
        addr_expr=IntVal(addr),
        active=True,
        reads=True,
        writes=True,
        is_atomic=True,
        atomic_kind="rmw",
        sem="relaxed",
        scope=scope,
        event_id=event_id,
        program_seq=event_id,
        elem_size=elem_size,
    )


def _races(records):
    return TwoCopySymbolicHBSolver(records, grid=(4, 1, 1), arange_dict={}).find_races()


# ── inclusive scopes: exempt (mutually atomic) ──────────────────


@pytest.mark.parametrize(
    "scopes",
    [("gpu", "gpu"), ("gpu", "sys"), ("sys", "sys")],
    ids=["gpu-gpu", "gpu-sys", "sys-sys"],
)
def test_inclusive_scope_atomic_pair_is_exempt(scopes):
    """Both scopes cover the peer CTA: the pair is morally strong — no
    race between the two atomics themselves."""
    a, b = scopes
    assert _races([_atomic(a, event_id=0), _atomic(b, event_id=1)]) == []


# ── cta on either side: racy across CTAs ────────────────────────


@pytest.mark.parametrize(
    "scopes",
    [("cta", "cta"), ("cta", "gpu"), ("gpu", "cta"), ("cta", "sys")],
    ids=["cta-cta", "cta-gpu", "gpu-cta", "cta-sys"],
)
def test_cta_scope_atomic_pair_races_across_ctas(scopes):
    """PTX .cta scope guarantees atomicity within one CTA only: a
    cross-CTA pair with EITHER side cta-scoped is scope-mismatched (not
    mutually inclusive) and must be reported — the Tile IR moral-strength
    classification."""
    a, b = scopes
    reports = _races([_atomic(a, event_id=0), _atomic(b, event_id=1)])
    assert reports, f"scope-mismatched pair {scopes} must race"


# ── the exemption is exact-address, same-width only ─────────────


def test_mixed_width_inclusive_pair_still_races():
    """Torn overlap: same base address, 4-byte vs 8-byte gpu atomics —
    width mismatch voids moral strength."""
    reports = _races(
        [
            _atomic("gpu", event_id=0, elem_size=4),
            _atomic("gpu", event_id=1, elem_size=8),
        ]
    )
    assert reports


def test_partial_overlap_inclusive_pair_still_races():
    """Same width, overlapping-but-unequal addresses: not the same
    location — races like plain writes."""
    reports = _races(
        [
            _atomic("gpu", event_id=0, elem_size=8, addr=FLAG),
            _atomic("gpu", event_id=1, elem_size=8, addr=FLAG + 4),
        ]
    )
    assert reports
