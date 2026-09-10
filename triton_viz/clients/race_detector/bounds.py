"""The in-bounds premise for the concrete channels.

The paper's model assumes every access lies inside a tensor argument
(the in-bounds premise); the symbolic frontends carry it as a constraint
of the encoding, but the two CONCRETE channels, the L1 enumeration rung
(``concrete_enum.py``) and the C2/C3 replay (``compiled/replay.py``),
let the Triton interpreter dereference raw host pointers, so an
out-of-bounds kernel would corrupt the process before any verdict.
``StorageBounds`` is the shared check both channels run in their
before-callbacks, BEFORE the interpreter executes the access: the
active lanes of an access must fall inside ONE storage span (the cloned
allocation, not the view). Cost: a min/max over the lanes and one
bisection, a few microseconds per access.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class StorageBounds:
    """Sorted, disjoint byte spans ``[lo, hi)`` of the storages a launch
    may touch; ``violation`` returns the first offending byte of an
    access (or None when every active lane is inside one span)."""

    def __init__(self, spans: Iterable[tuple[int, int]]) -> None:
        ordered = sorted((int(lo), int(hi)) for lo, hi in spans)
        self.lo = np.asarray([lo for lo, _ in ordered], dtype=np.int64)
        self.hi = np.asarray([hi for _, hi in ordered], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.lo.size)

    def violation(self, lanes: np.ndarray, elem: int) -> int | None:
        """``lanes``: the ACTIVE lanes' start addresses (masked-off lanes
        excluded by the caller); ``elem``: the access width in bytes."""
        if lanes.size == 0:
            return None
        lo = int(lanes.min())
        hi = int(lanes.max()) + elem
        i = int(np.searchsorted(self.lo, lo, side="right")) - 1
        if i >= 0 and hi <= self.hi[i]:
            return None
        if i < 0 or lo >= self.hi[i]:
            return lo  # the lowest lane is below or past every span
        past = lanes + elem > self.hi[i]
        return int(lanes[past].min()) if past.any() else lo
