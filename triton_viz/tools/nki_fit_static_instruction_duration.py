"""Fit Level-B instruction duration tables from independent control profiles."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from triton_viz.tools.nki_cost_model import StaticInstructionDurationCalibration
from triton_viz.tools.nki_instruction_source_mapping import _RUNTIME_OPCODES


def fit(roots: list[Path], output: Path, engines: tuple[str, ...]) -> int:
    exact = defaultdict(list); opcode = defaultdict(list); families = defaultdict(list)
    for root in roots:
        paths = list(root.glob("*/hardware/explorer_parquet/Instruction.parquet"))
        paths += list(root.glob("*/explorer_parquet/Instruction.parquet"))
        for path in paths:
            for row in pq.read_table(
                path,
                columns=["engine", "opcode", "operands", "scalar_activation_fn", "duration_ns"],
            ).to_pylist():
                engine = str(row["engine"] or "").lower()
                op = str(row["opcode"] or "")
                duration = float(row["duration_ns"] or 0)
                if engine not in engines or op in _RUNTIME_OPCODES or duration <= 0:
                    continue
                exact[(engine, StaticInstructionDurationCalibration.signature(row))].append(duration)
                opcode[(engine, op)].append(duration)
                family, free_dim = StaticInstructionDurationCalibration.family_key(row)
                if free_dim > 0:
                    families[(engine, family, free_dim)].append(duration)
    rows = [
        {"engine": engine, "signature": signature, "opcode": "", "family": "", "free_dim": "", "duration_ns": statistics.median(values)}
        for (engine, signature), values in sorted(exact.items())
    ]
    rows += [
        {"engine": engine, "signature": "", "opcode": op, "family": "", "free_dim": "", "duration_ns": statistics.median(values)}
        for (engine, op), values in sorted(opcode.items())
    ]
    rows += [
        {"engine": engine, "signature": "", "opcode": "", "family": family, "free_dim": free_dim, "duration_ns": statistics.median(values)}
        for (engine, family, free_dim), values in sorted(families.items())
    ]
    if not exact:
        raise ValueError("No static instruction duration controls")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["engine", "signature", "opcode", "family", "free_dim", "duration_ns"])
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", action="append", choices=["scalar", "vector", "tensor"], default=[])
    args = parser.parse_args(argv)
    count = fit(args.roots, args.output, tuple(args.engine or ["scalar"]))
    print(f"Wrote {count} rich instruction duration rows")
    return 0


if __name__ == "__main__": raise SystemExit(main())
