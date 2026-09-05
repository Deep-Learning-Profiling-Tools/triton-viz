"""The ladder-depth switch: ONE configuration with three operating levels.

Decision (Hao, 2026-09-04; paper repo ``design-route3-multipath-capture.md``
section 4b and ``design-route1-concrete-enumeration.md`` section 6b): the
detector exposes a single ladder-depth setting instead of per-feature
flags, so that every result carries exactly one provenance stamp for how
deep the concretization ladder was allowed to go.

  L0  (default) the shipped behavior: rungs T0 through the analyzed
      launch (``@interp``); no concrete-enumeration rung. Rows that
      abstain today keep abstaining, and L0 owns the current wall-time
      distribution (the paper's numbers are L0's).
  L1  L0 plus Route 1, the per-instance concrete footprint enumeration
      rung (``concrete_enum.py``), reached only when every symbolic rung
      has refused.
  L2  L1 plus Route 3's forked capture (future). L2 implies L1: Route
      3's path-ceiling handoff hands a row to the L1 rung.

The level is NOT an environment variable. It follows the ``ablations``
precedent: a constructor parameter on the detector clients and a field
of the evaluation harness's run configuration, stamped into the results
JSONL header (``ladder_level``) and into every row's verdict attributes,
so no dataset can mix levels unnoticed. The single gate that consults it
is the harness's third-track invocation (``evaluation/harness.run_one``).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


class LadderLevel(IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2


DEFAULT_LADDER_LEVEL = LadderLevel.L0
LADDER_LEVEL_NAMES: tuple[str, ...] = tuple(level.name for level in LadderLevel)


def parse_ladder_level(value: Any) -> LadderLevel:
    """Accept a ``LadderLevel``, its name (``"L1"``, case-insensitive), or
    its integer value (``1`` / ``"1"``). Anything else is a ``ValueError``:
    the level is provenance, so a typo must not silently mean L0."""
    if isinstance(value, LadderLevel):
        return value
    if isinstance(value, bool):
        raise ValueError(f"ladder level must be L0/L1/L2, got {value!r}")
    if isinstance(value, int):
        return LadderLevel(value)
    if isinstance(value, str):
        text = value.strip().upper()
        if text in LADDER_LEVEL_NAMES:
            return LadderLevel[text]
        if text.isdigit():
            return LadderLevel(int(text))
    raise ValueError(f"ladder level must be one of {LADDER_LEVEL_NAMES}, got {value!r}")
