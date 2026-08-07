import csv
import json

from triton_viz.tools.nki_compare_lowering import compare_lowering, load_lowering


def _point(count=2, opcodes=None, rule_id="elementwise.arity_chain"):
    return {
        "rule_id": rule_id,
        "structural_key": "abc",
        "instruction_count": count,
        "opcode_fingerprint": opcodes or {"TENSOR_TENSOR": count},
    }


def test_compare_lowering_distinguishes_same_drift_and_missing():
    reference = {
        ("same", 0, "vector"): _point(),
        ("drift", 0, "vector"): _point(),
        ("removed", 0, "scalar"): _point(1, {"ACTIVATION": 1}),
    }
    candidate = {
        ("same", 0, "vector"): _point(),
        ("drift", 0, "vector"): _point(3),
        ("added", 0, "scalar"): _point(1, {"ACTIVATION": 1}),
    }
    rows = compare_lowering(reference, candidate)
    statuses = {row["case"]: row["status"] for row in rows}
    assert statuses == {
        "added": "missing_reference",
        "drift": "structural_drift",
        "removed": "missing_candidate",
        "same": "same_lowering",
    }


def test_load_lowering_aggregates_payload_opcodes_and_excludes_runtime(tmp_path):
    case = tmp_path / "control_add"
    mapping_dir = case / "hardware/source_mapping"
    mapping_dir.mkdir(parents=True)
    event = {
        "op": "compute",
        "engine": "vector",
        "api_op": "add",
        "grid_idx": [0],
        "input_shape": [128, 64],
        "output_shape": [128, 64],
        "output_dtype": "float32",
        "input_ptrs": [1, 2],
        "output_ptr": 3,
    }
    (case / "trace.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    fields = ["fusion_group", "engine", "opcode"]
    with (mapping_dir / "instruction_mapping.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"fusion_group": "0", "engine": "vector", "opcode": "TENSOR_TENSOR"},
                {"fusion_group": "0", "engine": "vector", "opcode": "TENSOR_TENSOR"},
                {"fusion_group": "0", "engine": "vector", "opcode": "DRAIN"},
            ]
        )
    loaded = load_lowering(tmp_path)
    point = loaded[("control_add", 0, "vector")]
    assert point["instruction_count"] == 2
    assert point["opcode_fingerprint"] == {"TENSOR_TENSOR": 2}
    assert point["rule_id"] == "elementwise.arity_chain"
