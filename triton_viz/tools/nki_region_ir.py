"""Structured, composable lowering features for one contiguous NKI region."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_REDUCTIONS = {"reduce_sum", "max", "min", "mean"}
_TRANSCENDENTAL = {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid"}
_ONE_INPUT = {"exp", "rsqrt", "sqrt", "log", "sin", "cos", "tanh", "sigmoid", "relu"}
_IGNORED_FAMILY_OPS = {"where"}
REGION_IR_SCHEMA_NAME = "triton-viz.nki-region-ir"
REGION_IR_SCHEMA_VERSION = 2
SUPPORTED_REGION_IR_SCHEMA_VERSIONS = frozenset({1, REGION_IR_SCHEMA_VERSION})
_KNOWN_FAMILY_OPS = (
    _REDUCTIONS
    | _TRANSCENDENTAL
    | {
        "add",
        "broadcast_to",
        "divide",
        "maximum",
        "minimum",
        "multiply",
        "relu",
        "subtract",
        "where",
    }
)


@dataclass(frozen=True)
class GrammarRule:
    """One auditable region-family classification rule.

    Predicates at the same priority must be mutually exclusive.  Keeping rule
    metadata beside the predicate makes the production classifier inspectable
    without changing the stable family strings used by calibration CSVs.
    """

    rule_id: str
    priority: int
    predicate: Callable[[dict[str, Any]], bool]
    family: str | Callable[[dict[str, Any]], str]
    condition: str
    rationale: str
    evidence: tuple[str, ...] = ()

    def render_family(self, facts: dict[str, Any]) -> str:
        return self.family(facts) if callable(self.family) else self.family


@dataclass(frozen=True)
class GrammarMatch:
    """Classification result with evidence and explicit OOD diagnostics."""

    family: str
    rule_id: str
    rationale: str
    evidence: tuple[str, ...]
    ood_reasons: tuple[str, ...]
    consumed_features: tuple[str, ...]


def _token(event: dict[str, Any]) -> str:
    return str(
        event.get("api_op") or event.get("binary_op") or event.get("op") or "unknown"
    ).lower()


def _shape(event: dict[str, Any], key: str) -> list[int]:
    value = event.get(key) or []
    return [int(item) for item in value] if isinstance(value, (list, tuple)) else []


def region_ir_structural_key(region: dict[str, Any]) -> str:
    """Hash only compiler-relevant structure, excluding run provenance."""
    canonical = json.dumps(
        {
            key: value
            for key, value in region.items()
            if key not in {"provenance", "structural_key"}
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_region_ir(
    members: list[dict[str, Any]], context: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return stable structural features; never include allocation addresses."""
    tokens = [_token(event) for event in members]
    shapes = [
        _shape(event, "input_shape") or _shape(event, "output_shape")
        for event in members
    ]
    free_dim = max((shape[-1] for shape in shapes if shape), default=1)
    partition_count = max((shape[0] for shape in shapes if len(shape) > 1), default=1)
    reductions = [token for token in tokens if token in _REDUCTIONS]
    input_arity = [
        len(event.get("input_ptrs") or event.get("input_dtypes") or [])
        for event in members
    ]
    one_input = sum(
        token in _ONE_INPUT or arity == 1 for token, arity in zip(tokens, input_arity)
    )
    two_input = sum(
        token not in _REDUCTIONS and arity >= 2
        for token, arity in zip(tokens, input_arity)
    )

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
                if (
                    source_shape
                    and target_shape
                    and source_shape[-1] == 1 < target_shape[-1]
                ):
                    broadcast_edges += 1
        output = event.get("output_ptr")
        if output is not None:
            producer[output] = index
            producer_shapes[output] = _shape(event, "output_shape")

    context = context or []
    loads = [event for event in context if event.get("op") == "load"]
    stores = [event for event in context if event.get("op") == "store"]
    logical_free = max(
        (
            int(event.get("active_lanes", 0))
            // max(1, int(event.get("partition_count", 1)))
            for event in loads
        ),
        default=free_dim,
    )
    logical_free = min(free_dim, logical_free) if logical_free else free_dim
    tail = logical_free < free_dim
    dtypes = [
        str(dtype)
        for event in members
        for dtype in ([event.get("output_dtype")] + list(event.get("input_dtypes") or []))
        if dtype and str(dtype).lower() not in {"bool", "boolean"}
    ]
    dtype = Counter(dtypes).most_common(1)[0][0] if dtypes else "float32"
    item_bytes = 2 if dtype.lower() in {"float16", "fp16", "bfloat16", "bf16"} else 4
    block_elems = max(1, 8192 // item_bytes)
    result = {
        "schema_name": REGION_IR_SCHEMA_NAME,
        "schema_version": REGION_IR_SCHEMA_VERSION,
        "provenance": {"extractor": "triton_viz.tools.nki_region_ir.build_region_ir"},
        "tokens": tokens,
        "op_histogram": dict(sorted(Counter(tokens).items())),
        "reduction_kind": reductions[0]
        if len(set(reductions)) == 1 and reductions
        else ("mixed" if reductions else "none"),
        "reduction_count": len(reductions),
        "broadcast_edge_count": broadcast_edges
        + sum(
            1
            for event in members
            if _shape(event, "input_shape")
            and _shape(event, "output_shape")
            and _shape(event, "input_shape")[-1]
            == 1
            < _shape(event, "output_shape")[-1]
        ),
        "partition_broadcast_input_count": sum(
            1
            for event in loads
            if int(event.get("partition_count", 1)) == 1 and partition_count > 1
        ),
        "one_input_elementwise_count": one_input,
        "two_input_elementwise_count": two_input,
        "transcendental_count": sum(token in _TRANSCENDENTAL for token in tokens),
        "dtype": dtype,
        "partition_count": partition_count,
        "free_dim": free_dim,
        "logical_free_dim": logical_free,
        "free_block_count": math.ceil(logical_free / block_elems),
        "has_mask_or_tail": tail
        or "where" in tokens
        or any(
            int(event.get("active_lanes", 0))
            < math.prod(event.get("offsets_shape") or [0])
            for event in loads
        ),
        "memory_spaces": sorted(
            {str(event.get("mem_src")) for event in loads + stores}
            | {str(event.get("mem_dst")) for event in loads + stores}
        ),
        "uses_sbuf": any(
            "SBUF" in (str(event.get("mem_src")), str(event.get("mem_dst")))
            for event in loads + stores
        ),
        "uses_psum": any(
            "PSUM" in (str(event.get("mem_src")), str(event.get("mem_dst")))
            for event in loads + stores
        ),
        "dag_edges": edges,
    }
    result["structural_key"] = region_ir_structural_key(result)
    return result


def compositional_features(region: dict[str, Any]) -> dict[str, float]:
    """Numeric features used by an interpretable additive Level-A model."""
    names = [
        "reduction_count",
        "broadcast_edge_count",
        "one_input_elementwise_count",
        "two_input_elementwise_count",
        "transcendental_count",
        "free_block_count",
    ]
    features = {name: float(region.get(name, 0)) for name in names}
    features["intercept"] = 1.0
    features["log2_free_dim"] = math.log2(
        max(1, int(region.get("logical_free_dim", 1)))
    )
    features["mask_or_tail"] = float(bool(region.get("has_mask_or_tail")))
    features["two_reduction_interaction"] = float(
        int(region.get("reduction_count", 0)) >= 2
    )
    features["rsqrt_newton_interaction"] = float(
        region.get("op_histogram", {}).get("rsqrt", 0) > 0
        and region.get("op_histogram", {}).get("multiply", 0) >= 3
    )
    features["two_reduction_rsqrt_interaction"] = (
        features["two_reduction_interaction"] * features["rsqrt_newton_interaction"]
    )
    log_f = features["log2_free_dim"]
    for name in (
        "reduction_count",
        "one_input_elementwise_count",
        "two_input_elementwise_count",
        "transcendental_count",
    ):
        features[f"{name}_x_log2_free"] = features[name] * log_f
    total_elementwise = (
        features["one_input_elementwise_count"]
        + features["two_input_elementwise_count"]
    )
    features["elementwise_only"] = float(
        features["reduction_count"] == 0 and total_elementwise > 0
    )
    features["single_elementwise"] = float(
        features["reduction_count"] == 0 and total_elementwise == 1
    )
    for token, count in region.get("op_histogram", {}).items():
        features[f"op_{token}"] = float(count)
    return features


def _family_facts(region: dict[str, Any]) -> dict[str, Any]:
    reductions = int(region.get("reduction_count", 0))
    histogram = dict(region.get("op_histogram", {}))
    has_rsqrt = int(histogram.get("rsqrt", 0)) > 0
    ops = {
        name
        for name, count in histogram.items()
        if count and name not in _IGNORED_FAMILY_OPS
    }
    count = int(region.get("one_input_elementwise_count", 0)) + int(
        region.get("two_input_elementwise_count", 0)
    )
    return {
        "reductions": reductions,
        "has_rsqrt": has_rsqrt,
        "has_transcendental": int(region.get("transcendental_count", 0)) > 0,
        "has_partition_broadcast": int(region.get("partition_broadcast_input_count", 0))
        > 0,
        "ops": ops,
        "elementwise_count": count,
        "has_two_input": int(region.get("two_input_elementwise_count", 0)) > 0,
    }


GRAMMAR_RULES: tuple[GrammarRule, ...] = (
    GrammarRule(
        "reduction.two_with_rsqrt",
        100,
        lambda f: f["reductions"] >= 2 and f["has_rsqrt"],
        "two_reduction_rsqrt",
        "reductions >= 2 and has_rsqrt",
        "Two-or-more reductions interact with an rsqrt/Newton lowering.",
        ("two_reduction_rsqrt", "two_pass_reduce_affine"),
    ),
    GrammarRule(
        "reduction.one_with_rsqrt",
        100,
        lambda f: f["reductions"] == 1 and f["has_rsqrt"],
        "reduction_rsqrt",
        "reductions == 1 and has_rsqrt",
        "A single reduction feeds an rsqrt lowering.",
        ("reduce_rsqrt",),
    ),
    GrammarRule(
        "reduction.two_or_more",
        100,
        lambda f: (
            f["reductions"] >= 2 and not f["has_rsqrt"] and not f["has_transcendental"]
        ),
        "two_reduction",
        "reductions >= 2 and not has_rsqrt and not has_transcendental",
        "Two-or-more reductions lower as a distinct reduction grammar.",
        ("two_reduction",),
    ),
    GrammarRule(
        "reduction.with_transcendental",
        100,
        lambda f: (
            f["reductions"] >= 1 and not f["has_rsqrt"] and f["has_transcendental"]
        ),
        "reduction_transcendental",
        "reductions >= 1 and not has_rsqrt and has_transcendental",
        "A reduction is composed with a non-rsqrt transcendental operation.",
        ("softmax",),
    ),
    GrammarRule(
        "reduction.broadcast",
        100,
        lambda f: f["reductions"] == 1 and not f["has_transcendental"],
        "reduction_broadcast",
        "reductions == 1 and not has_transcendental",
        "A single reduction without a transcendental uses the reduction/broadcast grammar.",
        ("reduce_broadcast",),
    ),
    GrammarRule(
        "elementwise.partition_broadcast_multiply",
        100,
        lambda f: (
            f["reductions"] == 0
            and f["has_partition_broadcast"]
            and f["ops"] <= {"broadcast_to", "multiply"}
        ),
        "elementwise_broadcast_multiply",
        "no reductions and partition broadcast and ops subset of {broadcast_to,multiply}",
        "A partition-broadcast input feeds a multiply-only epilogue.",
        ("partition_broadcast_multiply",),
    ),
    GrammarRule(
        "elementwise.partition_broadcast_affine",
        100,
        lambda f: (
            f["reductions"] == 0
            and f["has_partition_broadcast"]
            and not f["ops"] <= {"broadcast_to", "multiply"}
        ),
        "elementwise_broadcast_affine",
        "no reductions and partition broadcast and non-multiply-only ops",
        "A partition-broadcast input feeds a non-multiply-only epilogue.",
        ("partition_broadcast_affine",),
    ),
    GrammarRule(
        "elementwise.multiply_chain",
        100,
        lambda f: (
            f["reductions"] == 0
            and not f["has_partition_broadcast"]
            and f["ops"] == {"multiply"}
        ),
        lambda f: f"elementwise_multiply_n{f['elementwise_count']}",
        "no reductions, no partition broadcast, and ops == {multiply}",
        "A multiply-only elementwise chain is instruction-selected separately.",
        ("elementwise_two",),
    ),
    GrammarRule(
        "elementwise.mixed",
        100,
        lambda f: (
            f["reductions"] == 0
            and not f["has_partition_broadcast"]
            and len(f["ops"]) >= 3
        ),
        "elementwise_mixed",
        "no reductions, no partition broadcast, and at least three distinct ops",
        "Three-or-more distinct elementwise operations form the mixed grammar.",
        ("mixed_epilogue",),
    ),
    GrammarRule(
        "elementwise.arity_chain",
        100,
        lambda f: (
            f["reductions"] == 0
            and not f["has_partition_broadcast"]
            and f["ops"] != {"multiply"}
            and len(f["ops"]) < 3
        ),
        lambda f: (
            f"elementwise_{'two' if f['has_two_input'] else 'one'}_n{f['elementwise_count']}"
        ),
        "no reductions, no partition broadcast, non-multiply-only, and fewer than three ops",
        "A non-multiply elementwise chain is keyed by input arity and operation count.",
        ("elementwise_one", "elementwise_two"),
    ),
)


def match_structural_family(
    region: dict[str, Any],
    *,
    rules: tuple[GrammarRule, ...] = GRAMMAR_RULES,
    strict: bool = False,
) -> GrammarMatch:
    """Classify a region and return the rule explanation and OOD diagnostics."""
    facts = _family_facts(region)
    matches = [rule for rule in rules if rule.predicate(facts)]
    if not matches:
        raise ValueError(f"No grammar rule matched region facts: {facts}")
    priority = max(rule.priority for rule in matches)
    winners = [rule for rule in matches if rule.priority == priority]
    if len(winners) != 1:
        ids = ", ".join(rule.rule_id for rule in winners)
        raise ValueError(f"Ambiguous grammar rules at priority {priority}: {ids}")

    unknown_ops = tuple(sorted(facts["ops"] - _KNOWN_FAMILY_OPS))
    ood_reasons = []
    if unknown_ops:
        ood_reasons.append("unknown_ops:" + ",".join(unknown_ops))
    if not facts["ops"] and facts["reductions"] == 0:
        ood_reasons.append("empty_elementwise_grammar")
    schema_version = int(region.get("schema_version", 1))
    if schema_version not in SUPPORTED_REGION_IR_SCHEMA_VERSIONS:
        ood_reasons.append(f"unsupported_schema:{schema_version}")
    if strict and ood_reasons:
        raise ValueError("Out-of-distribution region: " + "; ".join(ood_reasons))

    rule = winners[0]
    suffix = "_masked" if region.get("has_mask_or_tail") else ""
    context_suffix = ""
    if region.get("previous_family"):
        context_suffix += "__after_" + str(region["previous_family"])
    if region.get("next_family"):
        context_suffix += "__before_" + str(region["next_family"])
    suffix += context_suffix
    return GrammarMatch(
        family=rule.render_family(facts) + suffix,
        rule_id=rule.rule_id,
        rationale=rule.rationale,
        evidence=rule.evidence,
        ood_reasons=tuple(ood_reasons),
        consumed_features=(
            "reduction_count",
            "op_histogram",
            "transcendental_count",
            "partition_broadcast_input_count",
            "one_input_elementwise_count",
            "two_input_elementwise_count",
            "has_mask_or_tail",
            "previous_family",
            "next_family",
        ),
    )


def structural_family(region: dict[str, Any]) -> str:
    """Return the stable family string used by existing calibration tables."""
    return match_structural_family(region).family


def structural_calibration_key(region: dict[str, Any]) -> str:
    """Return an operator-name-free key for compiler instruction selection.

    A grammar family describes topology, but it intentionally does not fully
    distinguish primitives that lower to different NeuronCore engines.  The
    normalized source-op multiset supplies that missing compiler-facing
    evidence without introducing benchmark/operator names into the model.
    """
    match = match_structural_family(region)
    histogram = region.get("op_histogram") or {}
    ops = ",".join(
        f"{str(op).lower()}:{int(count)}"
        for op, count in sorted(histogram.items())
        if str(op).lower() not in _IGNORED_FAMILY_OPS
    )
    one = int(region.get("one_input_elementwise_count", 0))
    two = int(region.get("two_input_elementwise_count", 0))
    return f"{match.rule_id}|ops={ops or 'none'}|arity={one}:{two}"


def grammar_catalog() -> dict[str, Any]:
    """Return a JSON-serializable catalog of the production lowering rules."""
    return {
        "catalog_schema_version": 1,
        "region_ir_schema": {
            "name": REGION_IR_SCHEMA_NAME,
            "current_version": REGION_IR_SCHEMA_VERSION,
            "supported_versions": sorted(SUPPORTED_REGION_IR_SCHEMA_VERSIONS),
        },
        "rules": [
            {
                "rule_id": rule.rule_id,
                "priority": rule.priority,
                "condition": rule.condition,
                "rationale": rule.rationale,
                "evidence": list(rule.evidence),
            }
            for rule in GRAMMAR_RULES
        ],
    }
