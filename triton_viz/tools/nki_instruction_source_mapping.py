"""Map Explorer instructions to auditable NKI source fusion regions.

Current Inf2 artifacts have empty source-location columns, but retain Penguin
operation identities.  The evidence chain used here is Instruction.penguin_id
to penguin.py operation to a transfer-bounded source fusion region.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature


@dataclass(frozen=True)
class PenguinOp:
    op_id: int
    op_class: str
    opcode: str
    tensor_name: str
    kernel: str
    line: int


_OP_LINE = re.compile(r"= m\d+\.(?P<class>\w+)\(")
_ID = re.compile(r"\bid=(\d+)")
_DEBUG = re.compile(r'DebugLocation\(tensor_op_name="(?P<name>[^"]*)".*?kernel="(?P<kernel>[^"]*)"')
_NUMERIC = re.compile(r"\d+")
_RUNTIME_OPCODES = {
    "DRAIN", "NOTIFY", "EVENT_SEMAPHORE", "EVENT_SEMAPHORE_RANGE_CLEAR",
    "SET_ORDERING_MODE",
}


def _penguin_opcode(line: str, op_class: str) -> str:
    if op_class == "TensorReduceOp":
        return "reduce_sum" if "op=np.add" in line else "reduce"
    if op_class == "TensorSelect":
        return "where"
    if op_class == "SimpleBroadcastPartition":
        return "broadcast"
    if op_class == "ActivationOp" and "rsqrt" in line:
        return "rsqrt"
    match = re.search(r"\bop(?:0)?=.*?op=(?:np|m\d+)\.(\w+)", line)
    return match.group(1).lower() if match else op_class.lower()


def parse_penguin(path: Path) -> dict[int, PenguinOp]:
    """Parse identities without importing compiler-generated Python."""
    result = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        op_match, id_match = _OP_LINE.search(line), _ID.search(line)
        if not op_match or not id_match or "parent=v1" not in line:
            continue
        debug = _DEBUG.search(line)
        op_class, op_id = op_match.group("class"), int(id_match.group(1))
        result[op_id] = PenguinOp(
            op_id, op_class, _penguin_opcode(line, op_class),
            debug.group("name") if debug else "",
            debug.group("kernel") if debug else "", number,
        )
    return result


def load_regions(trace_path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _annotate_fusion_signature(events)
    groups: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("fusion_group") is not None:
            groups.setdefault(int(event["fusion_group"]), []).append(event)
    return [{
        "fusion_group": group_id,
        "source_region_id": members[0]["source_region_id"],
        "fusion_signature": members[0]["fusion_signature"],
        "fusion_pattern": members[0]["fusion_pattern"],
        "tokens": [str(event.get("api_op") or event.get("op") or "unknown").lower() for event in members],
        "free_dim": max(int((event.get("output_shape") or event.get("input_shape") or [0])[-1]) for event in members),
        "dtype": next((str(event["output_dtype"]) for event in members if event.get("output_dtype")), "float32"),
    } for group_id, members in sorted(groups.items())]


def assign_penguin_regions(ops: dict[int, PenguinOp], regions: list[dict[str, Any]]) -> dict[int, int]:
    """Align compute chunks separated by Penguin loads/stores to trace regions."""
    non_compute = {"SBAtomLoad", "SBAtomStore", "MemsetOp", "IndexValueInst", "SimpleBroadcastPartition"}
    chunks: list[list[int]] = []
    current: list[int] = []
    for op in sorted(ops.values(), key=lambda item: item.op_id):
        if op.op_class in {"SBAtomLoad", "SBAtomStore"} and current:
            chunks.append(current)
            current = []
        if op.op_class not in non_compute:
            current.append(op.op_id)
    if current:
        chunks.append(current)

    def score(chunk: list[int], region: dict[str, Any]) -> int:
        wanted, actual = region["tokens"], [ops[item].opcode for item in chunk]
        return sum(min(wanted.count(token), actual.count(token)) for token in set(wanted))

    mapping, cursor = {}, 0
    for region_index, region in enumerate(regions):
        remaining = len(regions) - region_index - 1
        choices = range(cursor, max(cursor, len(chunks) - remaining))
        try:
            chosen = max(choices, key=lambda index: score(chunks[index], region))
        except ValueError:
            break
        mapping.update({op_id: region_index for op_id in chunks[chosen]})
        cursor = chosen + 1
    return mapping


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    total, end = 0, -1
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(end, stop)
    return total


def _clip_to_active(
    intervals: Iterable[tuple[int, int]], active: Iterable[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Intersect instruction spans with Explorer's engine-active intervals."""
    result = []
    active_list = list(active)
    for start, end in intervals:
        for active_start, active_end in active_list:
            lo, hi = max(start, active_start), min(end, active_end)
            if lo < hi:
                result.append((lo, hi))
    return result


FIELDS = [
    "case", "instruction_id", "engine", "opcode", "start_ns", "end_ns", "duration_ns",
    "penguin_id", "bir_id", "bir_instruction_name", "source_op_id", "source_op_class",
    "source_opcode", "source_tensor_name", "source_line", "source_region_id", "fusion_group",
    "fusion_signature", "match_method", "confidence",
]


def map_case(case_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("instruction mapping requires pyarrow") from exc
    hardware, parquet = case_dir / "hardware", case_dir / "hardware" / "explorer_parquet"
    instructions = pq.read_table(parquet / "Instruction.parquet").to_pylist()
    active_rows = pq.read_table(parquet / "ActiveTime.parquet").to_pylist()
    ops = parse_penguin(hardware / "compiler_artifacts" / "penguin.py")
    regions = load_regions(case_dir / "trace.jsonl")
    op_regions = assign_penguin_regions(ops, regions)
    rows = []
    for instruction in instructions:
        raw = str(instruction.get("penguin_id") or "")
        source_id = next((int(value) for value in _NUMERIC.findall(raw) if int(value) in ops), None)
        region_index = op_regions.get(source_id) if source_id is not None else None
        op = ops.get(source_id) if source_id is not None else None
        rows.append({
            "case": case_dir.name, "instruction_id": str(instruction.get("id") or ""),
            "engine": str(instruction.get("engine") or "").lower(), "opcode": str(instruction.get("opcode") or ""),
            "start_ns": int(instruction.get("start_ts") or 0), "end_ns": int(instruction.get("end_ts") or 0),
            "duration_ns": int(instruction.get("duration_ns") or 0), "penguin_id": raw,
            "bir_id": str(instruction.get("bir_id") or ""),
            "bir_instruction_name": str(instruction.get("bir_instruction_name") or ""),
            "source_op_id": source_id if source_id is not None else "", "source_op_class": op.op_class if op else "",
            "source_opcode": op.opcode if op else "", "source_tensor_name": op.tensor_name if op else "",
            "source_line": op.line if op else "",
            "source_region_id": regions[region_index]["source_region_id"] if region_index is not None else "",
            "fusion_group": region_index if region_index is not None else "",
            "fusion_signature": regions[region_index]["fusion_signature"] if region_index is not None else "",
            "match_method": "penguin_id" if region_index is not None else "unattributed",
            "confidence": 1.0 if region_index is not None else 0.0,
        })

    # Compiler-created instructions inherit a region only inside that region's
    # direct-source time envelope. Prologue/epilogue remain unattributed.
    envelopes: dict[int, tuple[int, int]] = {}
    for row in rows:
        if row["fusion_group"] == "":
            continue
        key, interval = int(row["fusion_group"]), (row["start_ns"], row["end_ns"])
        old = envelopes.get(key)
        envelopes[key] = interval if old is None else (min(old[0], interval[0]), max(old[1], interval[1]))
    for row in rows:
        if row["fusion_group"] != "":
            continue
        if row["opcode"] in _RUNTIME_OPCODES:
            continue
        candidates = [group for group, (start, end) in envelopes.items()
                      if start <= row["start_ns"] and row["end_ns"] <= end]
        if len(candidates) == 1:
            group = candidates[0]
            row.update(source_region_id=regions[group]["source_region_id"], fusion_group=group,
                       fusion_signature=regions[group]["fusion_signature"],
                       match_method="single_region_time_envelope", confidence=0.7)

    # Activation lookup tables are loaded in the prologue, outside the payload
    # envelope. Attribute them only when exactly one source region owns an
    # ActivationOp; otherwise preserve the ambiguity.
    activation_regions = {op_regions[op_id] for op_id, op in ops.items()
                          if op.op_class == "ActivationOp" and op_id in op_regions}
    if len(activation_regions) == 1:
        group = next(iter(activation_regions))
        for row in rows:
            if row["fusion_group"] == "" and row["opcode"] in {"ACT_TABLE_LOAD", "ACTIVATE"}:
                row.update(source_region_id=regions[group]["source_region_id"], fusion_group=group,
                           fusion_signature=regions[group]["fusion_signature"],
                           match_method="unique_activation_owner", confidence=0.9)
    reduction_regions = {index for index, region in enumerate(regions)
                         if "reduce_sum" in region["tokens"]}
    if len(reduction_regions) == 1:
        group = next(iter(reduction_regions))
        for row in rows:
            if row["fusion_group"] == "" and row["opcode"] == "MEMSET" and row["penguin_id"]:
                row.update(source_region_id=regions[group]["source_region_id"], fusion_group=group,
                           fusion_signature=regions[group]["fusion_signature"],
                           match_method="unique_reduction_setup", confidence=0.8)

    audit: dict[str, Any] = {
        "case": case_dir.name, "instruction_count": len(rows),
        "mapped_instruction_count": sum(row["fusion_group"] != "" for row in rows),
        "unattributed_instruction_count": sum(row["fusion_group"] == "" for row in rows),
        "regions": regions, "engines": {},
    }
    for engine in sorted({row["engine"] for row in rows}):
        engine_rows = [row for row in rows if row["engine"] == engine]
        active_intervals = [(int(item.get("start_ts") or 0), int(item.get("end_ts") or 0))
                            for item in active_rows
                            if str(item.get("engine") or "").lower() == engine]
        explorer = sum(int(item.get("duration_ns") or 0) for item in active_rows
                       if str(item.get("engine") or "").lower() == engine)
        all_union = _union_ns(active_intervals)
        mapped_union = _union_ns(_clip_to_active(
            ((row["start_ns"], row["end_ns"]) for row in engine_rows if row["fusion_group"] != ""),
            active_intervals,
        ))
        payload_rows = [row for row in engine_rows if row["opcode"] not in _RUNTIME_OPCODES]
        payload_union = _union_ns(_clip_to_active(
            ((row["start_ns"], row["end_ns"]) for row in payload_rows), active_intervals))
        mapped_payload_union = _union_ns(_clip_to_active(
            ((row["start_ns"], row["end_ns"]) for row in payload_rows if row["fusion_group"] != ""),
            active_intervals,
        ))
        audit["engines"][engine] = {
            "instruction_count": len(engine_rows),
            "mapped_instruction_count": sum(row["fusion_group"] != "" for row in engine_rows),
            "explorer_active_ns": explorer, "instruction_union_ns": all_union,
            "active_reconciliation_error_percent": abs(all_union - explorer) / explorer * 100 if explorer else 0.0,
            "mapped_active_ns": mapped_union,
            "mapped_active_coverage_percent": mapped_union / explorer * 100 if explorer else 0.0,
            "payload_active_ns": payload_union,
            "mapped_payload_active_ns": mapped_payload_union,
            "mapped_payload_coverage_percent": (
                mapped_payload_union / payload_union * 100 if payload_union else 0.0
            ),
            "regions": {str(group): _union_ns(_clip_to_active(
                                                   ((row["start_ns"], row["end_ns"]) for row in engine_rows
                                                    if row["fusion_group"] == group), active_intervals))
                        for group in range(len(regions))},
        }
    return rows, audit


def write_case(case_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    rows, audit = map_case(case_dir)
    output = output_dir or case_dir / "hardware" / "source_mapping"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "instruction_mapping.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path); parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(write_case(args.case_dir, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
