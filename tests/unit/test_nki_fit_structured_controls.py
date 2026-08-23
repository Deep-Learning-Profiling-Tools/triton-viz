import csv

from triton_viz.tools.nki_fit_structured_controls import _load_completion_by_case


def test_load_completion_accepts_operator_results_schema(tmp_path):
    path = tmp_path / "operator_results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("op", "rows", "cols", "dtype", "hardware_nc_p50_us"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "op": "softmax",
                "rows": "128",
                "cols": "3584",
                "dtype": "float32",
                "hardware_nc_p50_us": "17.5",
            }
        )

    assert _load_completion_by_case([tmp_path]) == {
        "softmax__r128__c3584__float32": 17_500.0
    }
