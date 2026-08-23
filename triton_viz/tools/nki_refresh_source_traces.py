"""Refresh source-only NKI traces without compiling or reading target profiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from triton_viz.tools import nki_operator_experiments as experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-name", default="source_trace_v2.jsonl")
    args = parser.parse_args(argv)
    written = 0
    pattern = re.compile(
        r"^(?P<op>.+)__r(?P<rows>\d+)__c(?P<cols>\d+)__(?P<dtype>[^/]+)$"
    )
    for case in sorted(path for path in args.root.iterdir() if path.is_dir()):
        match = pattern.match(case.name)
        if match is None or not (case / "inputs.json").is_file():
            continue
        op = match.group("op")
        rows_count, cols = int(match.group("rows")), int(match.group("cols"))
        dtype = match.group("dtype")
        metadata = json.loads((case / "inputs.json").read_text(encoding="utf-8"))
        seed = int(metadata.get("seed", 0))
        previous = experiments._ACTIVE_RNG
        experiments._ACTIVE_RNG = np.random.default_rng(seed)
        try:
            inputs = experiments.OPERATORS[op]["inputs"](rows_count, cols, dtype)
        finally:
            experiments._ACTIVE_RNG = previous
        experiments._trace_events(op, inputs, case / args.output_name)
        written += 1
    print(f"Refreshed {written} source-only traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
