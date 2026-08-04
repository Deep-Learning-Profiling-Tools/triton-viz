"""Serialize NKI Triton-Viz trace records into a performance-model friendly JSONL.

This module intentionally stays lightweight: it consumes the normalized records that
Triton-Viz already records via the ``Tracer`` client and emits one JSON object per
operation.  The output is the first, stable seam for a future tile-level performance
simulator: later cost models can read this stream and map each operation to an
engine timeline event.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from triton_viz.tools.nki_region_ir import build_region_ir

from triton_viz.core.data import (
    BinaryOp,
    Dot,
    Grid,
    Load,
    NkiCompute,
    ReduceSum,
    Store,
    Transfer,
)


def _jsonable(value: Any) -> Any:
    """Convert NumPy/dataclass values into JSON-serializable primitives."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


def _shape(value: Any) -> list[int]:
    return [int(v) for v in tuple(value)]


def _dma_geometry(
    shape: list[int], nbytes: int, partition_axis: int | None = None
) -> dict[str, int]:
    """Expose the SBUF partition geometry needed by the DMA cost model."""
    if len(shape) < 2:
        return {}
    axis = 0 if partition_axis is None else int(partition_axis)
    if not 0 <= axis < len(shape) or shape[axis] <= 0:
        return {}
    partitions = int(shape[axis])
    return {
        "partition_count": partitions,
        "free_bytes_per_partition": int(nbytes // partitions),
        "partition_axis": axis,
    }


def _dot_flops(record: Dot) -> int | None:
    """Estimate FMA FLOPs for a normalized matmul record when shapes are 2-D."""
    if len(record.input_shape) != 2 or len(record.other_shape) != 2:
        return None
    m, k = (int(v) for v in record.input_shape)
    k_other, n = (int(v) for v in record.other_shape)
    if k != k_other:
        return None
    return 2 * m * n * k


def _transfer_engine(mem_src: str, mem_dst: str, shape: list[int]) -> str:
    """Return the likely NKI engine class for a transfer record."""
    memories = {mem_src.lower(), mem_dst.lower()}
    if "hbm" in memories:
        return "dma"
    if memories <= {"sbuf", "psum"}:
        # Fragmented on-chip transfers (e.g., free dimension <= 4) are typically
        # lowered to Static DMA by the compiler instead of VectorE, especially
        # during SBUF scatter/gather operations like transpose2d.
        free_dim = shape[-1] if len(shape) >= 2 else shape[0] if shape else 1
        if free_dim <= 4:
            return "static_dma"
        return "vector"
    return "unknown"


def _first_offset(offsets: Any) -> int | None:
    """Return the first logical element's byte offset without serializing a tile."""
    array = np.asarray(offsets)
    if not array.size:
        return None
    return int(array.reshape(-1)[0])


def _byte_span(offsets: Any, nbytes: int, sides: int = 1) -> list[int] | None:
    """Return an inclusive-exclusive ``[lo, hi)`` byte range for an access.

    ``offsets`` holds per-element byte offsets from the base tensor pointer; the
    span is ``[min_offset, max_offset + element_stride)``. When ``offsets`` is
    contiguous this collapses to ``[base_off, base_off + nbytes)``. We only need
    a conservative bounding range for hazard overlap, not exact element sets, so
    ``max_offset + bytes/count`` is a safe upper bound without dumping the tile.
    """
    array = np.asarray(offsets)
    if not array.size:
        return None
    flat = array.reshape(-1)
    lo = int(flat.min())
    hi = int(flat.max())
    count = int(flat.size)
    stride = max(1, int(nbytes // max(1, count)))
    return [lo, hi + stride]


def _fusion_op_name(event: dict[str, Any]) -> str:
    """Return a stable lowering-pattern token for a source compute event."""
    op = str(event.get("op") or "unknown")
    if op == "compute":
        return str(event.get("api_op") or "compute").lower()
    if op == "binary":
        return str(event.get("binary_op") or "binary").lower()
    return op.lower()


def _annotate_fusion_signature(events: list[dict[str, Any]]) -> None:
    """Annotate contiguous source-compute regions with a lowering signature.

    A signature is deliberately a source-level *lookup key*, not a claim about
    compiler fusion.  Explorer calibration later maps the key to the actual
    per-engine lowering.  Grid and memory events bound regions, so operations
    from different program instances or separated by an explicit transfer are
    never grouped together.
    """
    compute_ops = {"binary", "compute", "dot", "reduce_sum"}
    index = 0
    group_id = 0
    while index < len(events):
        if events[index].get("op") not in compute_ops:
            index += 1
            continue

        grid_idx = events[index].get("grid_idx")
        end = index + 1
        while (
            end < len(events)
            and events[end].get("op") in compute_ops
            and events[end].get("grid_idx") == grid_idx
        ):
            end += 1

        group = events[index:end]
        tokens = [_fusion_op_name(event) for event in group]
        signature = "_".join(tokens)
        reduction_tokens = {"reduce_sum", "max", "min", "mean"}
        pattern = (
            "reduce_broadcast_chain"
            if reduction_tokens.intersection(tokens)
            else "elementwise_chain"
        )
        identity = {
            "kernel": str(group[0].get("kernel_name") or "kernel"),
            "grid_idx": grid_idx,
            "ordinal": group_id,
            "signature": signature,
            "shapes": [event.get("output_shape") or event.get("input_shape") for event in group],
            "dtypes": [event.get("output_dtype") or event.get("dtype") for event in group],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        source_region_id = f"{identity['kernel']}:r{group_id}:{digest}"
        left = index - 1
        while left >= 0 and events[left].get("op") not in compute_ops:
            left -= 1
        right = end
        while right < len(events) and events[right].get("op") not in compute_ops:
            right += 1
        # Only the adjacent memory window belongs to this region. Passing the
        # whole kernel confuses a later 1-partition weight load with an earlier
        # reduction region's layout.
        region_ir = build_region_ir(group, events[left + 1:index] + events[end:right])
        for group_index, event in enumerate(group):
            event.update(
                {
                    "fusion_signature": signature,
                    "fusion_pattern": pattern,
                    "fusion_group": group_id,
                    "fusion_group_index": group_index,
                    "fusion_group_size": len(group),
                    "source_region_id": source_region_id,
                    "region_ir": region_ir,
                    "region_ir_key": region_ir["structural_key"],
                }
            )
        group_id += 1
        index = end

    # Compiler lowering can move setup work across an explicit transfer. Keep
    # the local grammar composable, but expose immediate region context so a
    # measured two-pass control can model that interaction without an
    # operator/full-signature key.
    leaders = [event for event in events if event.get("fusion_group_index") == 0 and event.get("region_ir")]
    from triton_viz.tools.nki_region_ir import structural_family
    for ordinal in range(1, len(leaders)):
        if leaders[ordinal - 1]["region_ir"].get("has_mask_or_tail"):
            leaders[ordinal]["region_ir"]["has_mask_or_tail"] = True
    bases = [structural_family(event["region_ir"]) for event in leaders]
    for ordinal, leader in enumerate(leaders):
        ir = leader["region_ir"]
        if ordinal: ir["previous_family"] = bases[ordinal - 1]
        if ordinal + 1 < len(bases): ir["next_family"] = bases[ordinal + 1]
        canonical = json.dumps({k: v for k, v in ir.items() if k != "structural_key"}, sort_keys=True, separators=(",", ":"))
        ir["structural_key"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        group = leader["fusion_group"]
        for event in events:
            if event.get("fusion_group") == group:
                event["region_ir"] = ir; event["region_ir_key"] = ir["structural_key"]

def _annotate_static_dma_groups(events: list[dict[str, Any]]) -> None:
    """Describe consecutive scalar SBUF scatter groups for calibrated costing.

    ``tensor_copy``-based transposes appear in the trace as one ``[p, 1]``
    transfer per scalar free-dimension element. The compiler batches these into
    Static DMA packets, so their cost is not the sum of independent VectorE
    operations. Keep the event stream intact, but attach one compact group
    geometry to every member so the cost model can charge the measured group
    latency exactly once across the group.
    """
    index = 0
    group_id = 0
    while index < len(events):
        event = events[index]
        if event.get("engine") != "static_dma":
            index += 1
            continue
        end = index + 1
        key = (
            event.get("src_ptr"),
            event.get("dst_ptr"),
            event.get("partition_count"),
            event.get("free_bytes_per_partition"),
        )
        while end < len(events):
            candidate = events[end]
            if candidate.get("engine") != "static_dma":
                break
            candidate_key = (
                candidate.get("src_ptr"),
                candidate.get("dst_ptr"),
                candidate.get("partition_count"),
                candidate.get("free_bytes_per_partition"),
            )
            if candidate_key != key:
                break
            end += 1

        group = events[index:end]
        copies = len(group)
        itemsize = int(event.get("free_bytes_per_partition") or 0)
        dst_offsets = [member.get("dst_offset_first") for member in group]
        x = y = 0
        if itemsize > 0 and copies > 0 and all(offset is not None for offset in dst_offsets):
            normalized = [int(offset) // itemsize for offset in dst_offsets]
            if copies == 1:
                x = y = 1
            else:
                initial_stride = normalized[1] - normalized[0]
                if initial_stride > 0:
                    y = next(
                        (
                            position
                            for position in range(1, copies)
                            if normalized[position] - normalized[position - 1]
                            != initial_stride
                        ),
                        copies,
                    )
                    if copies % y == 0:
                        x = copies // y
        for group_index, member in enumerate(group):
            member.update(
                {
                    "static_dma_group": group_id,
                    "static_dma_group_index": group_index,
                    "static_dma_group_copies": copies,
                    "static_dma_group_x": x,
                    "static_dma_group_y": y,
                }
            )
        group_id += 1
        index = end


def record_to_event(record: Any, sequence: int, grid_idx: tuple[int, ...] | None) -> dict[str, Any]:
    """Convert one Triton-Viz record into a compact JSON event dictionary."""
    base: dict[str, Any] = {
        "seq": int(sequence),
        "grid_idx": _jsonable(grid_idx),
        "record_type": type(record).__name__,
    }

    if isinstance(record, Grid):
        return {**base, "op": "grid", "grid_idx": _jsonable(record.idx)}

    if isinstance(record, Transfer):
        src_shape = _shape(record.src_offsets.shape)
        dst_shape = _shape(record.dst_offsets.shape)
        if record.mem_dst.lower() == "sbuf":
            shape = dst_shape
            partition_axis = record.dst_partition_axis
        else:
            shape = src_shape
            partition_axis = record.src_partition_axis
        dma_pattern = getattr(record, "dma_pattern", "copy")
        if (
            dma_pattern == "copy"
            and len(src_shape) == 2
            and src_shape == list(reversed(dst_shape))
            and src_shape != dst_shape
        ):
            dma_pattern = "transpose"
        return {
            **base,
            "op": "transfer",
            "engine": _transfer_engine(record.mem_src, record.mem_dst, shape),
            "mem_src": record.mem_src,
            "mem_dst": record.mem_dst,
            "bytes": int(record.bytes),
            "src_ptr": int(record.src_ptr),
            "dst_ptr": int(record.dst_ptr),
            "src_offsets_shape": src_shape,
            "dst_offsets_shape": dst_shape,
            "src_offset_first": _first_offset(record.src_offsets),
            "dst_offset_first": _first_offset(record.dst_offsets),
            "dma_pattern": dma_pattern,
            "src_storage": int(record.src_ptr),
            "dst_storage": int(record.dst_ptr),
            "src_range": _byte_span(record.src_offsets, int(record.bytes)),
            "dst_range": _byte_span(record.dst_offsets, int(record.bytes)),
            **_dma_geometry(shape, int(record.bytes), partition_axis),
        }

    if isinstance(record, Dot):
        return {
            **base,
            "op": "dot",
            "engine": "tensor",
            "input_shape": _shape(record.input_shape),
            "other_shape": _shape(record.other_shape),
            "output_shape": _shape(record.output_shape),
            "flops": _dot_flops(record),
            "input_ptrs": [int(ptr) for ptr in getattr(record, "input_ptrs", ())],
            "output_ptr": (
                int(record.output_ptr)
                if getattr(record, "output_ptr", None) is not None
                else None
            ),
        }

    if isinstance(record, BinaryOp):
        return {
            **base,
            "op": "binary",
            "binary_op": record.op,
            "engine": "vector",
            "input_shape": _shape(record.input_shape),
            "other_shape": _shape(record.other_shape),
            "output_shape": _shape(record.output_shape),
            "elements": int(np.prod(record.output_shape, dtype=np.int64)),
            "input_ptrs": [int(ptr) for ptr in record.input_ptrs],
            "output_ptr": (
                int(record.output_ptr) if record.output_ptr is not None else None
            ),
        }

    if isinstance(record, ReduceSum):
        return {
            **base,
            "op": "reduce_sum",
            "engine": "vector_or_scalar",
            "input_shape": _shape(record.input_shape),
            "axis": record.index,
            "keep_dims": bool(record.keep_dims),
            "output_shape": _shape(record.output_shape),
            "input_ptrs": [int(ptr) for ptr in getattr(record, "input_ptrs", ())],
            "output_ptr": (
                int(record.output_ptr)
                if getattr(record, "output_ptr", None) is not None
                else None
            ),
        }

    if isinstance(record, NkiCompute):
        out_shape = list(record.output_shapes[0]) if record.output_shapes else []
        in_shape = list(record.input_shapes[0]) if record.input_shapes else []
        free_dim = int(out_shape[-1]) if out_shape else (int(in_shape[-1]) if in_shape else 0)
        elements = int(np.prod(out_shape, dtype=np.int64)) if out_shape else 0
        return {
            **base,
            "op": "compute",
            "api_op": record.api_op,
            # ScalarE (activation) vs VectorE, as classified by the interpreter.
            "engine": record.engine,
            "input_shape": in_shape,
            "output_shape": out_shape,
            "free_dim": free_dim,
            "elements": elements,
            "input_ptrs": [int(ptr) for ptr in record.input_ptrs],
            "output_ptr": (int(record.output_ptrs[0]) if record.output_ptrs else None),
            "input_dtypes": list(record.input_dtypes),
            "output_dtype": record.output_dtype,
        }

    if isinstance(record, Load):
        active = int(np.count_nonzero(record.masks))
        shape = _shape(record.offsets.shape)
        return {
            **base,
            "op": "load",
            "engine": "dma_or_vector_load",
            "mem_src": record.mem_src,
            "mem_dst": record.mem_dst,
            "bytes": int(record.bytes),
            "active_lanes": active,
            "offsets_shape": shape,
            # HBM source pointer + range so HBM-side hazards resolve. The SBUF
            # destination tile pointer is not yet recorded by the nl.* frontend.
            "src_ptr": int(record.ptr),
            "src_storage": int(record.ptr),
            "src_range": _byte_span(record.offsets, int(record.bytes)),
            **_dma_geometry(shape, int(record.bytes)),
        }

    if isinstance(record, Store):
        active = int(np.count_nonzero(record.masks))
        shape = _shape(record.offsets.shape)
        return {
            **base,
            "op": "store",
            "engine": "dma_or_vector_store",
            "mem_src": record.mem_src,
            "mem_dst": record.mem_dst,
            "bytes": int(record.bytes),
            "active_lanes": active,
            "offsets_shape": shape,
            # HBM destination pointer + range so HBM-side hazards resolve.
            "dst_ptr": int(record.ptr),
            "dst_storage": int(record.ptr),
            "dst_range": _byte_span(record.offsets, int(record.bytes)),
            **_dma_geometry(shape, int(record.bytes)),
        }

    return {**base, "op": "unknown", "payload": _jsonable(record)}


def records_to_events(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert a launch's records into ordered performance trace events."""
    events: list[dict[str, Any]] = []
    grid_idx: tuple[int, ...] | None = None
    for sequence, record in enumerate(records):
        if isinstance(record, Grid):
            grid_idx = tuple(record.idx)
        events.append(record_to_event(record, sequence, grid_idx))
    _annotate_static_dma_groups(events)
    _annotate_fusion_signature(events)
    return events


def write_jsonl(records: Iterable[Any], path: str | Path) -> list[dict[str, Any]]:
    """Write records to ``path`` as JSONL and return the emitted event list."""
    events = records_to_events(records)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(_jsonable(event), sort_keys=True, ensure_ascii=False) + "\n")
    return events


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return small aggregate counters useful in tests and quick CLI checks."""
    summary: dict[str, Any] = {
        "num_events": 0,
        "op_counts": {},
        "bytes_by_edge": {},
        "flops": 0,
    }
    for event in events:
        summary["num_events"] += 1
        op = event.get("op", "unknown")
        summary["op_counts"][op] = summary["op_counts"].get(op, 0) + 1
        if "bytes" in event:
            key = f"{event.get('mem_src', '?')}->{event.get('mem_dst', '?')}"
            summary["bytes_by_edge"][key] = summary["bytes_by_edge"].get(key, 0) + int(event["bytes"])
        if event.get("flops") is not None:
            summary["flops"] += int(event["flops"])
    return summary
