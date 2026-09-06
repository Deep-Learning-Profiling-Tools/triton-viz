"""Smoke tests for examples/race_detector/*.py.

Each example is run as ``__main__`` via ``runpy`` so the print branch in the
``if __name__ == "__main__":`` block actually executes; this guards the
``RaceReport`` print path against attribute drift.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import triton_viz
from triton_viz.core.config import config as cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    REPO_ROOT / "examples/race_detector/inplace_neighbor.py",
    REPO_ROOT / "examples/race_detector/reduction.py",
    REPO_ROOT / "examples/race_detector/scatter.py",
    REPO_ROOT / "examples/race_detector/transpose.py",
]


@pytest.fixture
def _enable_race_detector():
    saved = cfg.enable_race_detector
    cfg.enable_race_detector = True
    triton_viz.clear()
    yield
    triton_viz.clear()
    cfg.enable_race_detector = saved


@pytest.mark.parametrize(
    "path",
    EXAMPLES,
    ids=[p.stem for p in EXAMPLES],
)
def test_race_detector_example_runs(path, _enable_race_detector):
    runpy.run_path(str(path), run_name="__main__")
