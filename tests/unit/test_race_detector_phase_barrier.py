import numpy as np
import pytest

from triton_viz.clients.race_detector.data import AccessType, MemoryAccess
from triton_viz.clients.race_detector.race_detector import detect_races


def _store_access(block_idx: int, offset: int) -> MemoryAccess:
    return MemoryAccess(
        access_type=AccessType.STORE,
        ptr=0,
        offsets=np.array([offset], dtype=np.int64),
        masks=np.array([True], dtype=np.bool_),
        grid_idx=(block_idx, 0, 0),
    )


@pytest.mark.xfail(
    reason=(
        "Known limitation: detect_races has no phase/epoch modeling and may report "
        "cross-phase false positives after a global sync."
    ),
    strict=False,
)
def test_detect_races_ignores_cross_phase_overlaps_after_barrier():
    """
    Reproducer for a 4-CTA / 4-stream barrier pattern:
    - Phase 0: block i writes slot i (no race inside phase 0)
    - Phase 1: block i writes slot (i + 1) % 4 (no race inside phase 1)
    A global sync (e.g., CAS spin barrier) exists between phases, so overlaps
    across phases should not be reported as races.
    """

    n = 4
    phase0 = [_store_access(block_idx=i, offset=i) for i in range(n)]
    phase1 = [_store_access(block_idx=i, offset=(i + 1) % n) for i in range(n)]

    assert detect_races(phase0) == []
    assert detect_races(phase1) == []

    # Desired behavior with phase awareness: still no races.
    # Current detector is phase-unaware and may report WAW here.
    assert detect_races(phase0 + phase1) == []
