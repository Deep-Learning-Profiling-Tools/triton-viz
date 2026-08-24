from triton_viz.tools.nki_structural_control_design import (
    CONTROL_DESIGN,
    validate_design,
)
from triton_viz.tools.nki_region_control_experiments import _declared_trace


def test_structural_control_design_is_disjoint_and_target_safe():
    validate_design(CONTROL_DESIGN)
    assert CONTROL_DESIGN["target_postcompile_allowed"] is False
    assert CONTROL_DESIGN["artifact_role"] == "control"
    assert set(CONTROL_DESIGN["structures"]) == {
        "no_join_linear",
        "no_join_memory_only",
        "multi_root_join",
    }


def test_memory_interleave_control_has_two_roots_and_no_compute(tmp_path):
    events = _declared_trace(
        "memory_interleave_offset", 3, 257, 1, "float32", tmp_path / "trace.jsonl"
    )
    assert len([event for event in events if event["op"] == "load"]) == 2
    assert len([event for event in events if event["op"] == "store"]) == 2
    assert not any(event["op"] in {"compute", "reduce_sum"} for event in events)
    assert {event["output_parity"] for event in events if event["op"] == "store"} == {0, 1}
