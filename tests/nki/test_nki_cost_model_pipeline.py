from triton_viz.tools.nki_cost_model_pipeline import main


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
    assert "/holdouts/elementwise_fp32" in output
    assert "dma_strided_store_surface.json" in output
    assert "runtime_overhead.json" in output


def test_pipeline_fit_and_evaluate_dry_run_use_dtype_specific_dma(tmp_path, capsys):
    root = tmp_path / "run"
    assert main(["fit", "--root", str(root), "--dry-run"]) == 0
    fit_output = capsys.readouterr().out
    assert "nki_fit_strided_dma" in fit_output
    assert "nki_fit_runtime_overhead" in fit_output

    assert main(["evaluate", "--root", str(root), "--dry-run"]) == 0
    evaluate_output = capsys.readouterr().out
    assert "dma_write_fp32.csv" in evaluate_output
    assert "dma_write_bf16.csv" in evaluate_output
    assert "--strided-dma-csv" in evaluate_output
    assert "--runtime-overhead-csv" in evaluate_output
