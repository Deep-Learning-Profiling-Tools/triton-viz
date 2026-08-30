import json

import pytest

from triton_viz.tools.nki_cost_model_pipeline import (
    _microbench_source_manifests,
    _split_case_count,
    _validate_command_contract,
    main,
)


def test_incomplete_resume_does_not_become_a_calibration_source(tmp_path):
    run = tmp_path / "microbench" / "dma"
    good = run / "case_good" / "manifest.json"
    bad = run / "case_bad" / "manifest.json"
    good.parent.mkdir(parents=True)
    bad.parent.mkdir(parents=True)
    good.write_text(json.dumps({"status": "ok"}))
    bad.write_text(json.dumps({"status": "error"}))
    (run / "run_manifest.json").write_text(
        json.dumps({"num_ok": 1, "num_benchmarks": 2})
    )

    assert _microbench_source_manifests(tmp_path) == []


def test_pipeline_dry_run_keeps_controls_and_holdouts_separate(tmp_path, capsys):
    tilebench = tmp_path / "tilebench"
    tilebench.mkdir()
    assert (
        main(
            [
                "collect",
                "--root",
                str(tmp_path / "run"),
                "--tilebench-dir",
                str(tilebench),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "/controls" in output
    assert "/holdouts/formal_fp32_v1" in output
    assert "/holdouts/attention_fp32_v1" in output
    assert "dma_strided_store_surface.json" in output
    assert "dma_partition_surface.json" in output
    assert "dma_partition_large_free.json" in output
    assert "dma_transpose_surface.json" in output
    assert "tensor_matmul_tiled_surface.json" in output
    assert "tensor_geometry_disjoint_v1.json" in output
    assert "tensor_geometry_disjoint_v5.json" in output
    assert "tensor_dot_count_low_disjoint_v3.json" in output
    assert "tensor_attention_pipeline_disjoint_a_v1.json" in output
    assert "tensor_attention_pipeline_disjoint_b_v1.json" in output
    assert "norm_pipeline_disjoint_a_v1" not in output
    assert "runtime_overhead.json" not in output
    assert "--skip-existing" in output
    assert "--resume" in output


def test_pipeline_fit_and_evaluate_dry_run_use_dtype_specific_dma(tmp_path, capsys):
    root = tmp_path / "run"
    assert main(["fit", "--root", str(root), "--dry-run"]) == 0
    fit_output = capsys.readouterr().out
    assert "nki_fit_strided_dma" in fit_output
    assert "nki_fit_runtime_overhead" not in fit_output
    assert "nki_fit_tensor_source_geometry" in fit_output
    assert "nki_fit_attention_pipeline" in fit_output
    assert "nki_fit_global_completion" in fit_output
    assert "--artifact-role control" in fit_output
    assert "--max-mean-wape 20" in fit_output
    assert "nki_fit_tensor_instruction" not in fit_output
    assert "nki_fit_static_opcode_payload" not in fit_output
    assert "nki_fit_static_instruction_duration" not in fit_output
    assert "nki_fit_static_dma_packets" not in fit_output
    assert "nki_fit_tensor_instruction_mix" not in fit_output
    assert "tensor_instruction_mix_audit.json" not in fit_output
    assert "static_dma_packets.json" not in fit_output
    assert "--min-payload-coverage 99.9" not in fit_output
    assert "--audit-output" in fit_output
    assert "structured_compute_audit.csv" in fit_output
    assert "dma_read_surface.csv" in fit_output
    assert "dma_read_bf16_surface.csv" in fit_output
    assert "dma_read_large_free.csv" in fit_output
    assert "dma_transpose_surface.csv" in fit_output
    assert "dma-affine" not in fit_output
    assert "runtime_overhead_affine.csv" not in fit_output

    assert main(["evaluate", "--root", str(root), "--dry-run"]) == 0
    evaluate_output = capsys.readouterr().out
    assert "dma_write_fp32.csv" in evaluate_output
    assert "dma_write_bf16.csv" in evaluate_output
    assert "--dma-read-surface-csv" in evaluate_output
    assert "--dma-write-surface-csv" in evaluate_output
    assert "--dma-transpose-surface-csv" in evaluate_output
    assert "--strided-dma-csv" in evaluate_output
    assert "--tensor-instruction-calibration-csv" not in evaluate_output
    assert "--tensor-instruction-mix-json" not in evaluate_output
    assert "--attention-repeat-reference-root" not in evaluate_output
    assert "--tensor-source-geometry-csv" in evaluate_output
    assert "--attention-pipeline-calibration-csv" in evaluate_output
    assert "--global-completion-csv" in evaluate_output
    assert "--static-opcode-payload-csv" not in evaluate_output
    assert "--static-instruction-duration-csv" not in evaluate_output
    assert "--static-dma-packet-calibration-json" not in evaluate_output
    assert "--runtime-overhead-csv" not in evaluate_output
    assert "--dma-elapsed-csv" in evaluate_output
    assert "--onchip-transfer-csv" in evaluate_output
    assert "runtime_overhead_bf16.csv" not in evaluate_output
    assert evaluate_output.count("--strict-calibration") == 12
    assert "--dma-model" not in evaluate_output
    assert "dma-affine" not in evaluate_output


def test_formal_split_is_exactly_the_documented_35_cases():
    data = json.loads(
        open("microbench/inf2_nki/configs/formal_holdouts.json").read()
    )
    formal = data["splits"]["formal_fp32_v1"]
    assert _split_case_count(formal) == 35
    assert formal["operators"]["interleave"][-1] == 4096
    assert formal["operators"]["layernorm"][-1] == 4096
    assert formal["operators"]["rmsnorm"][-1] == 4096
    assert _split_case_count(data["splits"]["full_fp32_v1"]) == 120
    assert _split_case_count(data["splits"]["tensor_fp32_v1"]) == 5
    assert _split_case_count(data["splits"]["tensor_bf16_v1"]) == 5
    assert _split_case_count(data["splits"]["attention_fp32_v1"]) == 4


def test_pipeline_child_parser_contract_rejects_old_static_dma_arguments():
    command = [
        "python",
        "-m",
        "triton_viz.tools.nki_fit_structural_static_dma",
        "/tmp/controls",
        "--output",
        "/tmp/static.csv",
    ]
    _validate_command_contract(command)
    with pytest.raises(SystemExit):
        _validate_command_contract(
            command + ["--compute-calibration-csv", "/tmp/compute.csv"]
        )
