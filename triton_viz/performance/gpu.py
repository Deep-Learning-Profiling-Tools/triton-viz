"""Experimental GPU backend: source work expansion and calibrated service time.

This is a source-only, steady-cache model, not an instruction simulator. The
additive service-time combiner is intentionally separate from NKI scheduling.
It does not claim to predict spills, cache misses, or asynchronous pipelines.
"""

from __future__ import annotations

import math
from collections import Counter

from .calibration import price
from .grammar import GrammarRule, select_rule

FEATURES = (
    "launch",
    "global_sectors",
    "alu_warps",
    "sfu_warps",
    "shuffle_steps",
    "tensor_flops",
    "waves",
)
_LAYOUT = {
    "program_id",
    "make_range",
    "addptr",
    "expand_dims",
    "broadcast",
    "splat",
    "unsplat",
    "reshape",
    "trans",
    "cast_impl",
    "fp_to_fp",
    "bitcast",
    "join",
    "split",
}
_REDUCE = {"reduce_sum", "reduce_max", "reduce_min", "sum", "max", "min"}
_ALU = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "maximum",
    "minimum",
    "negative",
    "absolute",
}
_SFU = {"exp", "exp2", "log", "log2", "sqrt", "sin", "cos", "rsqrt"}

GPU_RULES = (
    GrammarRule(
        "gpu.memory.sectors",
        10,
        lambda f: f["op"] in {"load", "store", "raw_load", "raw_store"},
        "global",
        "global transfer",
        "Count unique 32-byte sectors per source transfer",
    ),
    GrammarRule(
        "gpu.reduction.tree",
        10,
        lambda f: f["op"] in _REDUCE,
        "reduction",
        "tile reduction",
        "Predict a warp tree and inter-warp exchange",
    ),
    GrammarRule(
        "gpu.dot.flops",
        10,
        lambda f: f["op"] == "dot",
        "tensor",
        "2-D dot",
        "Count source dot work; tensor instruction selection remains unmodeled",
    ),
    GrammarRule(
        "gpu.scalar.issue",
        10,
        lambda f: f["op"] in {"binary_op", "unary_op", "fma", "rsqrt", "fabs"},
        "scalar",
        "scalar arithmetic",
        "Issue tile arithmetic in groups of 32 logical lanes",
    ),
    GrammarRule(
        "gpu.layout.source",
        10,
        lambda f: f["op"] in _LAYOUT,
        "layout",
        "source layout/index operation",
        "Layout is retained for auditing; physical conversions are not priced",
    ),
)


def expand(source, *, sm_count):
    if source.get("schema") != "triton-viz.gpu-source.v1":
        raise ValueError("Unsupported GPU source schema")
    if sm_count <= 0 or source["num_warps"] not in {4, 8}:
        raise ValueError("GPU backend requires positive SM count and num_warps 4 or 8")
    features = dict.fromkeys(FEATURES, 0.0)
    features["launch"] = 1.0
    features["waves"] = math.ceil(source["program_count"] / sm_count)
    rules = Counter()
    reasons = []
    dtypes = set()
    tile_shapes = set()
    for event in source["events"]:
        try:
            rule = select_rule(event, GPU_RULES)
        except ValueError:
            reasons.append("unknown_op:" + event["op"])
            continue
        rules[rule.rule_id] += 1
        family = rule.render_family(event)
        dtype = str(event["dtype"])
        floating = "pointer" not in dtype and any(
            token in dtype for token in ("float", "fp16", "fp32", "fp64", "bf16")
        )
        if floating:
            dtypes.add(dtype)
        warps = math.ceil(event["elements"] / 32)
        if family == "global":
            features["global_sectors"] += event["sectors"]
            if event["op"] in {"load", "raw_load"}:
                tile_shapes.add(tuple(event.get("shape", ())))
        elif family == "layout" and event["op"] in {"trans", "join", "split"}:
            reasons.append("unmodeled_layout_conversion:" + event["op"])
        elif family == "reduction":
            shapes = event["input_shapes"]
            elements = math.prod(shapes[0]) if shapes else 0
            outputs = max(1, event["elements"])
            width = max(1, elements // outputs)
            features["alu_warps"] += math.ceil(elements / 32)
            features["shuffle_steps"] += outputs * math.ceil(math.log2(width))
        elif family == "tensor":
            shapes = event["input_shapes"]
            if (
                len(shapes) < 2
                or len(shapes[0]) != 2
                or len(shapes[1]) != 2
                or shapes[0][1] != shapes[1][0]
            ):
                reasons.append("unsupported_dot_geometry")
            else:
                features["tensor_flops"] += (
                    2 * shapes[0][0] * shapes[0][1] * shapes[1][1]
                )
        elif family == "scalar" and floating:
            primitive = event.get("primitive", event["op"])
            if primitive in _SFU:
                features["sfu_warps"] += warps
            elif primitive in _ALU or primitive in {"fma", "fabs"}:
                features["alu_warps"] += warps
            else:
                reasons.append("unknown_float_primitive:" + primitive)
    return {
        "features": features,
        "rules": dict(rules),
        "ood_reasons": sorted(set(reasons)),
        "dtypes": sorted(dtypes),
        "tile_shapes": [list(shape) for shape in sorted(tile_shapes)],
        "num_warps": source["num_warps"],
        "num_stages": source["num_stages"],
    }


def predict(source, calibration, *, fingerprint, sm_count, strict=True):
    work = expand(source, sm_count=sm_count)
    reasons = work["ood_reasons"][:]
    configuration = source_configuration(work)
    if configuration not in calibration.get("source_configurations", []):
        reasons.append("uncovered_launch_configuration_or_dtype")
    if strict and reasons:
        raise ValueError("Out-of-distribution: " + "; ".join(reasons))
    contributions, domain_reasons = price(
        work["features"], calibration, fingerprint=fingerprint, strict=strict
    )
    return {
        "latency_us": sum(contributions.values()),
        "contributions_us": contributions,
        "ood_reasons": reasons + domain_reasons,
        "work": work,
        "backend": "gpu-source-service-v1",
        "experimental": True,
        "metric": "cuda_graph_steady_cache_kernel_us",
    }


def source_configuration(work):
    """Keep launch and tile geometry in the calibrated domain, not only bytes."""
    return [work["num_warps"], work["num_stages"], work["dtypes"], work["tile_shapes"]]
