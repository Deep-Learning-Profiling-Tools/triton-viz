"""Structured, composable lowering features for one contiguous NKI region."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any


_REDUCTIONS = {"reduce_sum", "max", "min", "mean"}
_TRANSCENDENTAL = {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid"}
_ONE_INPUT = {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid", "relu"}


def _token(event: dict[str, Any]) -> str:
    return str(event.get("api_op") or event.get("binary_op") or event.get("op") or "unknown").lower()


def _shape(event: dict[str, Any], key: str) -> list[int]:
    value = event.get(key) or []
    return [int(item) for item in value] if isinstance(value, (list, tuple)) else []


def build_region_ir(members: list[dict[str, Any]], context: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return stable structural features; never include allocation addresses."""
    tokens = [_token(event) for event in members]
    shapes = [_shape(event, "input_shape") or _shape(event, "output_shape") for event in members]
    free_dim = max((shape[-1] for shape in shapes if shape), default=1)
    partition_count = max((shape[0] for shape in shapes if len(shape) > 1), default=1)
    reductions = [token for token in tokens if token in _REDUCTIONS]
    input_arity = [len(event.get("input_ptrs") or event.get("input_dtypes") or []) for event in members]
    one_input = sum(token in _ONE_INPUT or arity == 1 for token, arity in zip(tokens, input_arity))
    two_input = sum(token not in _REDUCTIONS and arity >= 2 for token, arity in zip(tokens, input_arity))

    producer: dict[int, int] = {}
    producer_shapes: dict[int, list[int]] = {}
    edges: list[list[int]] = []
    broadcast_edges = 0
    for index, event in enumerate(members):
        for pointer in event.get("input_ptrs") or []:
            if pointer in producer:
                edges.append([producer[pointer], index])
                source_shape = producer_shapes.get(pointer, [])
                target_shape = _shape(event, "output_shape")
                if source_shape and target_shape and source_shape[-1] == 1 < target_shape[-1]:
                    broadcast_edges += 1
        output = event.get("output_ptr")
        if output is not None:
            producer[output] = index
            producer_shapes[output] = _shape(event, "output_shape")

    context = context or []
    loads = [event for event in context if event.get("op") == "load"]
    stores = [event for event in context if event.get("op") == "store"]
    logical_free = max(
        (int(event.get("active_lanes", 0)) // max(1, int(event.get("partition_count", 1))) for event in loads),
        default=free_dim,
    )
    logical_free = min(free_dim, logical_free) if logical_free else free_dim
    tail = logical_free < free_dim
    dtype = str(next((event.get("output_dtype") for event in members if event.get("output_dtype")), "float32"))
    item_bytes = 2 if dtype.lower() in {"float16", "fp16", "bfloat16", "bf16"} else 4
    block_elems = max(1, 8192 // item_bytes)
    result = {
        "schema_version": 1,
        "tokens": tokens,
        "op_histogram": dict(sorted(Counter(tokens).items())),
        "reduction_kind": reductions[0] if len(set(reductions)) == 1 and reductions else ("mixed" if reductions else "none"),
        "reduction_count": len(reductions),
        "broadcast_edge_count": broadcast_edges + sum(
            1 for event in members
            if _shape(event, "input_shape") and _shape(event, "output_shape")
            and _shape(event, "input_shape")[-1] == 1 < _shape(event, "output_shape")[-1]
        ),
        "partition_broadcast_input_count": sum(1 for event in loads if int(event.get("partition_count", 1)) == 1 and partition_count > 1),
        "one_input_elementwise_count": one_input,
        "two_input_elementwise_count": two_input,
        "transcendental_count": sum(token in _TRANSCENDENTAL for token in tokens),
        "dtype": dtype,
        "partition_count": partition_count,
        "free_dim": free_dim,
        "logical_free_dim": logical_free,
        "free_block_count": int(math.ceil(logical_free / block_elems)),
        "has_mask_or_tail": tail or "where" in tokens or any(int(event.get("active_lanes", 0)) < math.prod(event.get("offsets_shape") or [0]) for event in loads),
        "memory_spaces": sorted({str(event.get("mem_src")) for event in loads + stores} | {str(event.get("mem_dst")) for event in loads + stores}),
        "uses_sbuf": any("SBUF" in (str(event.get("mem_src")), str(event.get("mem_dst"))) for event in loads + stores),
        "uses_psum": any("PSUM" in (str(event.get("mem_src")), str(event.get("mem_dst"))) for event in loads + stores),
        "dag_edges": edges,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["structural_key"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return result


def compositional_features(region: dict[str, Any]) -> dict[str, float]:
    """Numeric features used by an interpretable additive Level-A model."""
    names = ["reduction_count", "broadcast_edge_count", "one_input_elementwise_count",
             "two_input_elementwise_count", "transcendental_count", "free_block_count"]
    features = {name: float(region.get(name, 0)) for name in names}
    features["intercept"] = 1.0
    features["log2_free_dim"] = math.log2(max(1, int(region.get("logical_free_dim", 1))))
    features["mask_or_tail"] = float(bool(region.get("has_mask_or_tail")))
    features["two_reduction_interaction"] = float(int(region.get("reduction_count", 0)) >= 2)
    features["rsqrt_newton_interaction"] = float(
        region.get("op_histogram", {}).get("rsqrt", 0) > 0
        and region.get("op_histogram", {}).get("multiply", 0) >= 3
    )
    features["two_reduction_rsqrt_interaction"] = (
        features["two_reduction_interaction"] * features["rsqrt_newton_interaction"]
    )
    log_f = features["log2_free_dim"]
    for name in ("reduction_count", "one_input_elementwise_count",
                 "two_input_elementwise_count", "transcendental_count"):
        features[f"{name}_x_log2_free"] = features[name] * log_f
    total_elementwise = features["one_input_elementwise_count"] + features["two_input_elementwise_count"]
    features["elementwise_only"] = float(features["reduction_count"] == 0 and total_elementwise > 0)
    features["single_elementwise"] = float(features["reduction_count"] == 0 and total_elementwise == 1)
    for token, count in region.get("op_histogram", {}).items():
        features[f"op_{token}"] = float(count)
    return features


def structural_family(region: dict[str, Any]) -> str:
    """Compiler-relevant grammar family, independent of operator/signature."""
    reductions = int(region.get("reduction_count", 0))
    histogram = region.get("op_histogram", {})
    has_rsqrt = int(histogram.get("rsqrt", 0)) > 0
    suffix = "_masked" if region.get("has_mask_or_tail") else ""
    context_suffix = ""
    if region.get("previous_family"):
        context_suffix += "__after_" + str(region["previous_family"])
    if region.get("next_family"):
        context_suffix += "__before_" + str(region["next_family"])
    suffix += context_suffix
    if reductions >= 2 and has_rsqrt:
        return "two_reduction_rsqrt" + suffix
    if reductions and has_rsqrt:
        return "reduction_rsqrt" + suffix
    if reductions >= 2:
        return "two_reduction" + suffix
    if reductions and int(region.get("transcendental_count", 0)):
        return "reduction_transcendental" + suffix
    if reductions:
        return "reduction_broadcast" + suffix
    ops = {name for name, count in histogram.items() if count and name not in {"where"}}
    count = int(region.get("one_input_elementwise_count", 0)) + int(region.get("two_input_elementwise_count", 0))
    if int(region.get("partition_broadcast_input_count", 0)):
        if ops <= {"broadcast_to", "multiply"}:
            return "elementwise_broadcast_multiply" + suffix
        return "elementwise_broadcast_affine" + suffix
    if ops == {"multiply"}:
        return f"elementwise_multiply_n{count}" + suffix
    if len(ops) >= 3:
        return "elementwise_mixed" + suffix
    arity = "two" if int(region.get("two_input_elementwise_count", 0)) else "one"
    return f"elementwise_{arity}_n{count}{suffix}"
