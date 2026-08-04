"""Expand a compact sweep spec into an explicit run_microbench config."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def expand_sweeps(config: dict[str, Any]) -> dict[str, Any]:
    out = {
        "suite": config.get("suite", "expanded_sweep"),
        "description": config.get("description", "expanded by generate_sweep_config"),
        "warmup": config.get("warmup", 10),
        "iters": config.get("iters", 100),
        "benchmarks": [],
    }
    for kind, sweep in config.get("sweeps", {}).items():
        keys = list(sweep.keys())
        value_lists = [v if isinstance(v, list) else [v] for v in (sweep[k] for k in keys)]
        for combo in itertools.product(*value_lists):
            row = {k: v for k, v in zip(keys, combo)}
            modes = row.pop("mode", None)
            if modes is None:
                modes = row.pop("modes", ["independent"])
            if not isinstance(modes, list):
                modes = [modes]
            row.update({"name": kind, "kind": kind, "modes": modes})
            out["benchmarks"].append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.input.read_text(encoding="utf-8"))
    expanded = expand_sweeps(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expanded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(expanded['benchmarks'])} benchmark specs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
