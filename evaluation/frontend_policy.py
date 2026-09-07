"""Execution policy for the evaluation harness's symbolic frontends.

L2 stops when a frontend decides. The explicit comparison override retains
the independent interpreter observation needed for frontend coverage studies.
L0/L1 retain their historical every-frontend protocol.
"""

from __future__ import annotations

import os

from triton_viz.clients.race_detector.ladder import LadderLevel, parse_ladder_level

ALL_FRONTENDS_ENV = "TRITON_VIZ_EVAL_ALL_FRONTENDS"


def frontend_policy(ladder_level: LadderLevel) -> str:
    """Resolve and validate the policy before starting any analysis work."""
    override = os.environ.get(ALL_FRONTENDS_ENV, "")
    if override not in ("", "0", "1"):
        raise ValueError(f"{ALL_FRONTENDS_ENV} must be 0 or 1, got {override!r}")
    if parse_ladder_level(ladder_level) == LadderLevel.L2 and override != "1":
        return "on-demand"
    return "all"
