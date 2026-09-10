"""The evaluation runner's CLI must build: a duplicate option (the rebase
of Route 3's --ladder-level onto Route 1's) made argparse raise at
startup, which no detector test could see."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_runner_help_builds():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", "evaluation.runner", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    for flag in ("--ladder-level", "--only-file", "--out-suffix"):
        assert flag in proc.stdout
