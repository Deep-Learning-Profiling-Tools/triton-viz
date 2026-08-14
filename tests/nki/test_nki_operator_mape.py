import csv

import pytest

from triton_viz.tools.nki_operator_mape import main

pytestmark = pytest.mark.nki


def test_operator_mape_combines_multiple_results_files(tmp_path, capsys):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    fields = [
        "status",
        "error_vs_nc_pct",
        "dma_busy_error_pct",
        "vector_busy_error_pct",
        "scalar_busy_error_pct",
        "tensor_busy_error_pct",
    ]
    for path, values in (
        (first, (10.0, 2.0, 20.0, 30.0, 4.0)),
        (second, (-20.0, 4.0, 10.0, 20.0, -8.0)),
    ):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(zip(fields, ("ok", *values))))

    assert main([str(first), str(second)]) == 0
    output = capsys.readouterr().out
    assert "points=2" in output
    assert "error_vs_nc_pct=15.0000" in output
    assert "tensor_busy_error_pct=6.0000" in output


def test_operator_mape_rejects_results_without_successful_rows(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "status,error_vs_nc_pct\n"
        "error,boom\n"
    )
    with pytest.raises(ValueError, match="No successful operator rows"):
        main([str(path)])
