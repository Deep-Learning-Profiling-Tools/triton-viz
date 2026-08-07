import json

import pytest

from triton_viz.tools.nki_grammar_report import build_report, collect_region_coverage
from triton_viz.tools.nki_region_ir import (
    REGION_IR_SCHEMA_NAME,
    REGION_IR_SCHEMA_VERSION,
    GrammarRule,
    build_region_ir,
    compositional_features,
    grammar_catalog,
    match_structural_family,
    region_ir_structural_key,
    structural_calibration_key,
    structural_family,
)


def test_region_ir_encodes_structure_dag_shape_dtype_and_tail():
    members = [
        {
            "op": "compute",
            "api_op": "multiply",
            "input_ptrs": [1, 2],
            "output_ptr": 3,
            "input_shape": [128, 1000],
            "output_shape": [128, 1000],
            "output_dtype": "float32",
        },
        {
            "op": "reduce_sum",
            "input_ptrs": [3],
            "output_ptr": 4,
            "input_shape": [128, 1000],
            "output_shape": [128, 1],
        },
        {
            "op": "compute",
            "api_op": "rsqrt",
            "input_ptrs": [4],
            "output_ptr": 5,
            "input_shape": [128, 1],
            "output_shape": [128, 1],
            "output_dtype": "float32",
        },
    ]
    context = [
        {
            "op": "load",
            "active_lanes": 128 * 900,
            "partition_count": 128,
            "offsets_shape": [128, 1000],
            "mem_src": "HBM",
            "mem_dst": "SBUF",
        }
    ]
    ir = build_region_ir(members, context)
    assert ir["reduction_count"] == 1
    assert ir["two_input_elementwise_count"] == 1
    assert ir["transcendental_count"] == 1
    assert ir["dag_edges"] == [[0, 1], [1, 2]]
    assert ir["logical_free_dim"] == 900 and ir["has_mask_or_tail"]
    assert ir["uses_sbuf"] and not ir["uses_psum"]
    assert compositional_features(ir)["rsqrt_newton_interaction"] == 0


def test_region_ir_key_ignores_pointer_values():
    a = [
        {
            "op": "compute",
            "api_op": "add",
            "input_ptrs": [1, 2],
            "output_ptr": 3,
            "input_shape": [128, 64],
            "output_shape": [128, 64],
            "output_dtype": "float32",
        }
    ]
    b = [{**a[0], "input_ptrs": [100, 200], "output_ptr": 300}]
    assert build_region_ir(a)["structural_key"] == build_region_ir(b)["structural_key"]


def test_calibration_key_separates_primitives_with_same_grammar_family():
    def region(op):
        return build_region_ir(
            [
                {
                    "op": "compute",
                    "api_op": op,
                    "input_ptrs": [1],
                    "output_ptr": 2,
                    "input_shape": [128, 512],
                    "output_shape": [128, 512],
                    "output_dtype": "float32",
                }
            ]
        )

    add = region("add")
    maximum = region("maximum")
    assert structural_family(add) == structural_family(maximum)
    assert structural_calibration_key(add) != structural_calibration_key(maximum)
    assert "maximum:1" in structural_calibration_key(maximum)


def test_region_ir_v2_provenance_is_excluded_from_structural_key():
    region = build_region_ir(
        [{"op": "compute", "api_op": "relu", "input_ptrs": [1], "output_ptr": 2}]
    )
    assert region["schema_name"] == REGION_IR_SCHEMA_NAME
    assert region["schema_version"] == REGION_IR_SCHEMA_VERSION
    changed = {
        **region,
        "provenance": {"extractor": "other", "git_revision": "deadbeef"},
    }
    assert region_ir_structural_key(changed) == region["structural_key"]


def test_structural_family_distinguishes_multiply_chain_from_add_chain():
    def event(op, inputs, output):
        return {
            "op": "compute",
            "api_op": op,
            "input_ptrs": inputs,
            "output_ptr": output,
            "input_shape": [128, 64],
            "output_shape": [128, 64],
            "output_dtype": "float32",
        }

    multiply = build_region_ir(
        [event("multiply", [1, 2], 3), event("multiply", [3, 4], 5)]
    )
    add = build_region_ir([event("add", [1, 2], 3), event("add", [3, 4], 5)])
    assert structural_family(multiply) == "elementwise_multiply_n2"
    assert structural_family(add) == "elementwise_two_n2"


def test_grammar_match_explains_rule_evidence_mask_and_context():
    region = {
        "schema_version": 1,
        "reduction_count": 2,
        "transcendental_count": 1,
        "op_histogram": {"reduce_sum": 2, "rsqrt": 1},
        "has_mask_or_tail": True,
        "previous_family": "reduction_broadcast",
    }
    match = match_structural_family(region)
    assert match.rule_id == "reduction.two_with_rsqrt"
    assert match.evidence
    assert match.ood_reasons == ()
    assert match.family == "two_reduction_rsqrt_masked__after_reduction_broadcast"


def test_grammar_match_reports_unknown_operations_and_strict_mode_rejects_them():
    region = {
        "schema_version": 1,
        "reduction_count": 0,
        "one_input_elementwise_count": 1,
        "two_input_elementwise_count": 0,
        "op_histogram": {"future_activation": 1},
    }
    match = match_structural_family(region)
    assert match.family == "elementwise_one_n1"
    assert match.ood_reasons == ("unknown_ops:future_activation",)
    with pytest.raises(ValueError, match="Out-of-distribution"):
        match_structural_family(region, strict=True)


def test_grammar_match_rejects_same_priority_ambiguity():
    rules = (
        GrammarRule("test.a", 10, lambda facts: True, "a", "true", "test"),
        GrammarRule("test.b", 10, lambda facts: True, "b", "true", "test"),
    )
    with pytest.raises(ValueError, match="Ambiguous grammar rules"):
        match_structural_family({"schema_version": 1}, rules=rules)


def test_grammar_classification_is_independent_of_operator_case_and_pointer_metadata():
    base = {
        "schema_version": 1,
        "reduction_count": 0,
        "one_input_elementwise_count": 0,
        "two_input_elementwise_count": 2,
        "op_histogram": {"multiply": 2},
    }
    decorated = {
        **base,
        "operator_name": "rmsnorm",
        "case": "/tmp/run-17",
        "pointer": 0xDEADBEEF,
    }
    assert match_structural_family(base) == match_structural_family(decorated)


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        (
            {
                "reduction_count": 1,
                "transcendental_count": 0,
                "op_histogram": {"reduce_sum": 1},
            },
            "reduction_broadcast",
        ),
        (
            {
                "reduction_count": 1,
                "transcendental_count": 1,
                "op_histogram": {"reduce_sum": 1, "exp": 1},
            },
            "reduction_transcendental",
        ),
        (
            {
                "reduction_count": 1,
                "transcendental_count": 1,
                "op_histogram": {"reduce_sum": 1, "rsqrt": 1},
            },
            "reduction_rsqrt",
        ),
        (
            {
                "reduction_count": 2,
                "transcendental_count": 0,
                "op_histogram": {"reduce_sum": 2},
            },
            "two_reduction",
        ),
        (
            {
                "reduction_count": 2,
                "transcendental_count": 1,
                "op_histogram": {"reduce_max": 1, "reduce_sum": 1, "exp": 1},
            },
            "reduction_transcendental",
        ),
        (
            {
                "reduction_count": 0,
                "partition_broadcast_input_count": 1,
                "op_histogram": {"broadcast_to": 1, "multiply": 1},
            },
            "elementwise_broadcast_multiply",
        ),
        (
            {
                "reduction_count": 0,
                "partition_broadcast_input_count": 1,
                "op_histogram": {"broadcast_to": 1, "multiply": 1, "add": 1},
            },
            "elementwise_broadcast_affine",
        ),
        (
            {
                "reduction_count": 0,
                "two_input_elementwise_count": 3,
                "op_histogram": {"add": 1, "multiply": 1, "subtract": 1},
            },
            "elementwise_mixed",
        ),
    ],
)
def test_declarative_rules_preserve_established_family_names(region, expected):
    assert structural_family({"schema_version": 1, **region}) == expected


def test_grammar_catalog_and_coverage_report_keep_ood_in_denominator(tmp_path):
    catalog = grammar_catalog()
    assert catalog["region_ir_schema"]["current_version"] == REGION_IR_SCHEMA_VERSION
    assert {rule["rule_id"] for rule in catalog["rules"]} >= {
        "reduction.broadcast",
        "elementwise.arity_chain",
    }

    case = tmp_path / "case"
    case.mkdir()
    trace = case / "trace.jsonl"
    events = [
        {
            "op": "compute",
            "engine": "vector",
            "api_op": "future_activation",
            "grid_idx": [0],
            "input_shape": [128, 64],
            "output_shape": [128, 64],
            "output_dtype": "float32",
            "input_ptrs": [1],
            "output_ptr": 2,
        }
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    rows = collect_region_coverage([tmp_path])
    report = build_report(rows)
    assert len(rows) == 1
    assert rows[0]["rule_id"] == "elementwise.arity_chain"
    assert rows[0]["in_scope"] is False
    assert report["coverage"]["region_count"] == 1
    assert report["coverage"]["ood_region_count"] == 1
    assert report["coverage"]["in_scope_percent"] == 0.0
    matched_rule = next(
        rule for rule in report["rules"] if rule["rule_id"] == "elementwise.arity_chain"
    )
    assert matched_rule["has_observed_evidence"] is False
    assert matched_rule["observed_cases"] == []
