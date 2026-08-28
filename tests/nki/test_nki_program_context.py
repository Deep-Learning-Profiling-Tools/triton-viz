import json
import csv

from triton_viz.tools.nki_fit_source_sequence_lowering import _cases
from triton_viz.tools.nki_program_context import program_context_features
from triton_viz.tools.nki_program_context import source_routing_state
from triton_viz.tools.nki_program_context import (
    source_branch_orientation_state,
    source_complete_routing_regime,
    source_full_routing_regime,
    source_join_ownership_state,
    source_local_topology_state,
    source_routing_regime,
)
from triton_viz.tools.nki_region_control_experiments import _declared_trace


def test_wide_masked_declared_trace_preserves_physical_width(tmp_path):
    events = _declared_trace(
        "elementwise_maximum_wide_masked",
        128,
        192,
        1,
        "float32",
        tmp_path / "trace.jsonl",
    )

    memory_events = [
        event for event in events if event["op"] in {"load", "store"}
    ]
    compute = next(event for event in events if event["op"] == "compute")
    assert memory_events
    assert all(event["offsets_shape"] == [128, 16384] for event in memory_events)
    assert compute["input_shape"] == [128, 16384]
    assert compute["output_shape"] == [128, 16384]


def test_program_context_reports_liveness_without_storage_identity():
    events = [
        {
            "op": "compute",
            "fusion_group": 0,
            "region_ir": {"free_dim": 2048, "logical_free_dim": 512},
            "output_storage": 17,
            "output_version": 0,
            "output_range": [0, 2048],
        },
        {
            "op": "compute",
            "fusion_group": 0,
            "region_ir": {"free_dim": 2048, "logical_free_dim": 512},
            "input_storages": [17],
            "input_versions": [0],
            "input_ranges": [[0, 2048]],
        },
    ]

    features = program_context_features(events)

    assert features["program_peak_live_bytes"] == 2048
    assert features["program_dependency_edge_count"] == 1
    assert features["program_allocation_to_logical_ratio"] == 4
    assert not any("17" in name for name in features)


def test_program_context_reports_precision_and_materialization_roles():
    events = [
        {
            "op": "compute",
            "fusion_group": 0,
            "region_ir": {
                "free_dim": 2048,
                "logical_free_dim": 1792,
                "partition_count": 64,
                "input_dtype": "bfloat16",
                "accumulator_dtype": "float32",
            },
        },
        {
            "op": "store",
            "bytes": 64 * 1792 * 2,
            "mem_src": "SBUF",
            "mem_dst": "HBM",
        },
        {
            "op": "load",
            "bytes": 64 * 1792 * 2,
            "mem_src": "HBM",
            "mem_dst": "SBUF",
        },
        {
            "op": "compute",
            "fusion_group": 1,
            "region_ir": {
                "free_dim": 2048,
                "logical_free_dim": 1792,
                "partition_count": 64,
                "input_dtype": "bfloat16",
                "accumulator_dtype": "bfloat16",
            },
        },
    ]

    features = program_context_features(events)

    assert features["program_region_count"] == 2
    assert features["program_materialization_boundary_count"] == 2
    assert features["program_input_bf16_region_count"] == 2
    assert features["program_accumulator_fp32_region_count"] == 1
    assert features["program_mixed_precision_region_count"] == 1


def test_source_sequence_cases_use_declared_trace_semantics(tmp_path):
    case = tmp_path / "phase1" / (
        "control_sequence_factorialdag2k__p64__f1792__n3000__bfloat16"
    )
    case.mkdir(parents=True)
    declared_region = {
        "dtype": "bfloat16",
        "input_dtype": "bfloat16",
        "accumulator_dtype": "bfloat16",
        "partition_count": 64,
        "free_dim": 2048,
        "logical_free_dim": 1792,
        "op_histogram": {"multiply": 1},
        "reduction_count": 0,
        "one_input_elementwise_count": 0,
        "two_input_elementwise_count": 1,
    }
    dependency_region = {**declared_region, "dtype": "float32"}
    for name, region in (
        ("trace.jsonl", declared_region),
        ("dependency_trace.jsonl", dependency_region),
    ):
        (case / name).write_text(
            json.dumps(
                {
                    "op": "compute",
                    "api_op": "multiply",
                    "fusion_group": 0,
                    "fusion_group_index": 0,
                    "input_dtypes": [region["dtype"], region["dtype"]],
                    "region_ir": region,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    with (tmp_path / "phase1" / "control_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "case",
                "vector_active_ns",
                "scalar_active_ns",
                "gpsimd_active_ns",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": case.name,
                "vector_active_ns": 1_000,
                "scalar_active_ns": 2_000,
                "gpsimd_active_ns": 3_000,
            }
        )

    rows = _cases([tmp_path / "phase1"], {})
    assert len(rows) == 3
    assert {row["dtype"] for row in rows} == {"bfloat16"}


def test_program_context_summarizes_branch_join_topology():
    events = [
        {
            "op": "compute",
            "fusion_group": 0,
            "region_ir": {
                "tokens": [
                    "add",
                    "multiply",
                    "reduce_sum",
                    "subtract",
                    "multiply",
                    "add",
                    "exp",
                ],
                "dag_edges": [
                    [0, 2],
                    [1, 3],
                    [3, 4],
                    [2, 5],
                    [4, 5],
                    [5, 6],
                ],
            },
        }
    ]

    features = program_context_features(events)

    assert features["program_dag_join_count"] == 1
    assert features["program_dag_join_add_count"] == 1
    assert features["program_dag_join_multiply_count"] == 0
    assert features["program_dag_join_branch_depth_imbalance_max"] == 1
    assert features["program_dag_join_branch_reduction_imbalance_max"] == 1
    assert features["program_dag_reduction_before_join_count"] == 1
    assert features["program_dag_post_join_transcendental_count"] == 1
    assert features["program_dag_branch_source_interleave_count"] > 0
    assert features["program_dag_branch_run_count_max"] >= 2
    assert features["program_dag_branch0_root_add"] == 1
    assert features["program_dag_branch1_root_multiply"] == 1
    assert features["program_dag_branch_reduction_signed_difference"] == 1


def test_source_routing_state_uses_low_dimensional_topology():
    assert source_routing_state({}) == "other"
    assert source_routing_state(
        {
            "program_dag_join_count": 3,
            "program_dag_branch_source_interleave_count": 7,
        }
    ) == "interleaved"
    assert source_routing_state(
        {
            "program_dag_join_count": 5,
            "program_dag_branch_source_interleave_count": 2,
        }
    ) == "blocked"
    assert source_routing_state(
        {
            "program_dag_join_count": 3,
            "program_dag_branch_add_multiply_order_max": 0.03,
        }
    ) == "reversed"
    assert source_routing_state(
        {
            "program_dag_join_count": 3,
            "program_dag_branch_add_multiply_order_max": 0.25,
        }
    ) == "canonical"


def test_program_context_recovers_join_update_owner_from_reuse():
    canonical = [
        {
            "op": "compute",
            "fusion_group": 0,
            "region_ir": {
                "tokens": ["add", "multiply", "add", "add"],
                "dag_edges": [[0, 2], [1, 2], [2, 3], [1, 3]],
            },
        }
    ]
    swapped = [
        {
            "op": "compute",
            "fusion_group": 0,
            "region_ir": {
                "tokens": ["add", "multiply", "add", "add"],
                "dag_edges": [[1, 2], [0, 2], [2, 3], [0, 3]],
            },
        }
    ]

    canonical_features = program_context_features(canonical)
    swapped_features = program_context_features(swapped)

    assert canonical_features["program_dag_join_update_branch0_count"] == 1
    assert canonical_features["program_dag_join_update_branch1_count"] == 0
    assert swapped_features["program_dag_join_update_branch0_count"] == 0
    assert swapped_features["program_dag_join_update_branch1_count"] == 1
    assert source_join_ownership_state(canonical_features) == "branch0_only"
    assert source_join_ownership_state(swapped_features) == "branch1_only"
    assert source_routing_regime(canonical_features).endswith(":branch0_only")


def test_source_local_topology_state_distinguishes_oriented_fanout():
    assert source_local_topology_state({}) == "linear"
    assert source_local_topology_state(
        {"program_dag_branch0_local_fanout_count": 1}
    ) == "fanout_branch0"
    assert source_local_topology_state(
        {"program_dag_branch1_local_rejoin_count": 1}
    ) == "fanout_branch1"
    both = {
        "program_dag_join_count": 3,
        "program_dag_branch_add_multiply_order_max": 0.3,
        "program_dag_join_update_branch0_count": 1,
        "program_dag_branch0_local_fanout_count": 1,
        "program_dag_branch1_local_fanout_count": 1,
    }
    assert source_local_topology_state(both) == "fanout_both"
    assert source_full_routing_regime(both).endswith(":fanout_both")


def test_source_branch_orientation_is_explicit_and_ordered():
    add_mul = {
        "program_dag_join_count": 3,
        "program_dag_branch_add_multiply_order_max": 0.3,
        "program_dag_join_update_branch0_count": 1,
        "program_dag_branch0_root_add": 1,
        "program_dag_branch1_root_multiply": 1,
    }
    mul_add = {
        **add_mul,
        "program_dag_branch0_root_add": 0,
        "program_dag_branch1_root_multiply": 0,
        "program_dag_branch0_root_multiply": 1,
        "program_dag_branch1_root_add": 1,
    }
    assert source_branch_orientation_state(add_mul) == "add_mul"
    assert source_branch_orientation_state(mul_add) == "mul_add"
    assert source_branch_orientation_state({}) == "other"
    assert source_complete_routing_regime(add_mul).endswith(":add_mul")
