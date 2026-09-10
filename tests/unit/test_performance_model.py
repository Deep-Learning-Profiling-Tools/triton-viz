import copy

import pytest

from triton_viz.performance.calibration import fit_controls, price, stable_digest
from triton_viz.performance.grammar import GrammarRule, select_rule
from triton_viz.performance.gpu import expand
from triton_viz.performance.gpu_measure import BusyGPU, assert_available


def _controls():
    return [
        dict(
            role="control",
            contaminated=False,
            fingerprint="test",
            cv_group=str(n),
            features={"launch": 1, "bytes": n},
            latency_us=2 + 0.01 * n,
        )
        for n in (10, 20, 40, 80)
    ]


def test_control_fit_validates_unseen_geometry_and_nonnegative_costs():
    model = fit_controls(_controls(), ("launch", "bytes"), fingerprint="test")
    assert model["cv"]["passed"]
    costs, reasons = price({"launch": 1, "bytes": 30}, model, fingerprint="test")
    assert sum(costs.values()) == pytest.approx(2.3, abs=1e-6)
    assert not reasons
    assert all(value >= 0 for value in costs.values())


@pytest.mark.parametrize(
    "field,value",
    [("role", "holdout"), ("contaminated", True), ("fingerprint", "other")],
)
def test_fit_refuses_invalid_control_provenance(field, value):
    rows = _controls()
    rows[0][field] = value
    with pytest.raises(ValueError):
        fit_controls(rows, ("launch", "bytes"), fingerprint="test")


def test_price_reports_domain_and_enforces_gate_and_fingerprint():
    model = fit_controls(_controls(), ("launch", "bytes"), fingerprint="test")
    with pytest.raises(ValueError, match="Out-of-distribution"):
        price({"launch": 1, "bytes": 100}, model, fingerprint="test")
    _, reasons = price(
        {"launch": 1, "bytes": 100}, model, fingerprint="test", strict=False
    )
    assert reasons
    with pytest.raises(ValueError, match="fingerprint"):
        price({"launch": 1, "bytes": 30}, model, fingerprint="different")
    model["cv"]["passed"] = False
    with pytest.raises(ValueError, match="validation gate"):
        price({"launch": 1, "bytes": 30}, model, fingerprint="test")


def test_shared_rule_selection_rejects_priority_ties():
    first = GrammarRule("one", 1, lambda facts: True, "x", "always", "test")
    second = GrammarRule("two", 1, lambda facts: True, "y", "always", "test")
    with pytest.raises(ValueError, match="Ambiguous"):
        select_rule({}, (first, second))


def test_gpu_expansion_is_name_independent_and_unknown_ops_are_ood():
    source = dict(
        schema="triton-viz.gpu-source.v1",
        num_warps=4,
        num_stages=2,
        program_count=96,
        events=[
            dict(op="load", dtype="fp32", elements=64, sectors=8),
            dict(op="binary_op", primitive="add", dtype="fp32", elements=64),
        ],
    )
    work = expand(source, sm_count=48)
    assert work["features"]["waves"] == 2
    assert work["features"]["alu_warps"] == 2
    assert work["features"]["global_sectors"] == 8
    renamed = copy.deepcopy(source)
    renamed["kernel_name"] = "a_different_operator"
    assert expand(renamed, sm_count=48) == work
    source["events"].append(dict(op="atomic_rmw", dtype="fp32", elements=1))
    assert expand(source, sm_count=48)["ood_reasons"] == ["unknown_op:atomic_rmw"]


def test_shared_machine_audit_rejects_foreign_or_unattributed_load():
    sample = {"processes": [], "utilization_pct": "0"}
    assert_available(sample)
    with pytest.raises(BusyGPU):
        assert_available(
            {**sample, "processes": [{"pid": 12, "name": "other"}]}, own_pid=11
        )
    with pytest.raises(BusyGPU):
        assert_available({**sample, "utilization_pct": "80"})
    with pytest.raises(BusyGPU):
        assert_available({**sample, "utilization_pct": "N/A"})
    graphics = {**sample, "graphics_processes": [{"pid": 99}]}
    with pytest.raises(BusyGPU):
        assert_available(graphics)
    assert_available(graphics, allowed_graphics=(99,))
    with pytest.raises(BusyGPU):
        assert_available({**graphics, "utilization_pct": "80"}, allowed_graphics=(99,))


def test_digest_is_order_independent():
    assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})
