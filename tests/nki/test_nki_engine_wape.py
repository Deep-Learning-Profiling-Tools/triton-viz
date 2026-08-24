import pytest

from triton_viz.tools.nki_operator_experiments import _wape


def test_engine_wape_retains_small_denominators_without_case_explosion():
    rows = [
        {"status": "ok", "predicted": 1.0, "actual": 0.001},
        {"status": "ok", "predicted": 10.0, "actual": 10.0},
    ]
    expected = (0.999 + 0.0) / (0.001 + 10.0) * 100.0
    assert _wape(rows, "predicted", "actual") == pytest.approx(expected)
