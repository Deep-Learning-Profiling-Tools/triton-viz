"""Fit busy-only payload lookup from timing-free compiler opcode counts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from triton_viz.tools.nki_instruction_source_mapping import _RUNTIME_OPCODES


FIELDS = ["case", "engine", "dtype", "opcode_counts_json", "payload_active_ns"]


def fit(control_roots: list[Path], output: Path, engines: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in control_roots:
        for case in sorted(path for path in root.iterdir() if path.is_dir()):
            mapping = case / "hardware/source_mapping/instruction_mapping.csv"
            audit_path = case / "hardware/source_mapping/audit.json"
            if not mapping.is_file() or not audit_path.is_file():
                continue
            counts: dict[str, Counter[str]] = {engine: Counter() for engine in engines}
            with mapping.open(encoding="utf-8", newline="") as file:
                for instruction in csv.DictReader(file):
                    engine = instruction["engine"]
                    opcode = instruction["opcode"]
                    if engine in counts and opcode not in _RUNTIME_OPCODES:
                        counts[engine][opcode] += 1
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            dtype = "bfloat16" if case.name.endswith("__bfloat16") else "float32"
            for engine in engines:
                payload = float(audit.get("engines", {}).get(engine, {}).get("payload_active_ns") or 0)
                if counts[engine] and payload > 0:
                    rows.append(
                        {
                            "case": case.name,
                            "engine": engine,
                            "dtype": dtype,
                            "opcode_counts_json": json.dumps(dict(sorted(counts[engine].items())), separators=(",", ":")),
                            "payload_active_ns": payload,
                        }
                    )
    if not rows:
        raise ValueError("No mapped static opcode payload controls")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", action="append", choices=["gpsimd"], default=[])
    args = parser.parse_args(argv)
    engines = tuple(args.engine or ["gpsimd"])
    rows = fit(args.control_roots, args.output, engines)
    print(f"Wrote {len(rows)} static opcode payload rows for {','.join(engines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
