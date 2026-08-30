import json
import csv

from triton_viz.tools.nki_program_context import program_context_features
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
