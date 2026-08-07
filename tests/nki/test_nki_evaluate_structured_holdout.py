from triton_viz.tools.nki_evaluate_structured_holdout import (
    mapping_coverage_percent,
)


def test_mapping_coverage_defaults_to_zero_instead_of_claiming_evidence():
    assert mapping_coverage_percent({}, "vector") == 0.0


def test_mapping_coverage_reads_payload_coverage_not_total_instruction_coverage():
    audit = {
        "engines": {
            "vector": {
                "mapped_active_coverage_percent": 42.0,
                "mapped_payload_coverage_percent": 100.0,
            }
        }
    }
    assert mapping_coverage_percent(audit, "vector") == 100.0
