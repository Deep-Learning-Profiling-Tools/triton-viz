import inspect

from triton_viz.tools import nki_replay_operator_predictions as replay
from triton_viz.tools import nki_refresh_source_traces as refresh


def test_prediction_phase_cannot_read_target_post_compile_artifacts():
    source = inspect.getsource(replay.main)
    prediction, _labels = source.split(
        'hardware_us = float(source["hardware_dma_active_us"])', 1
    )
    for forbidden in (
        "Instruction.parquet",
        "DmaPacket.parquet",
        "instruction_mapping.csv",
        "source_mapping/audit.json",
        "explorer_summary.json",
    ):
        assert forbidden not in prediction


def test_replay_cli_has_no_post_compile_prediction_calibrations():
    source = inspect.getsource(replay.main)
    for forbidden_flag in (
        "--tensor-instruction-calibration-csv",
        "--tensor-instruction-mix-json",
        "--static-opcode-payload-csv",
        "--static-instruction-duration-csv",
        "--static-dma-packet-calibration-json",
    ):
        assert forbidden_flag not in source


def test_payload_labels_do_not_read_target_instruction_mapping():
    source = inspect.getsource(replay.main)
    for forbidden in (
        "Instruction.parquet",
        "DmaPacket.parquet",
        "instruction_mapping.csv",
        "source_mapping/audit.json",
        "mapping_audit",
    ):
        assert forbidden not in source
    assert "saved_aggregate_active_minus_independent_runtime_control" in source
    assert "explorer_summary.json" not in source


def test_source_trace_refresh_never_opens_target_results_or_hardware():
    source = inspect.getsource(refresh.main)
    for forbidden in (
        "operator_results.csv", "hardware", "explorer_summary.json",
        "Instruction.parquet", "DmaPacket.parquet", "profile.ntff", "file.neff",
    ):
        assert forbidden not in source
    assert "inputs.json" in source
