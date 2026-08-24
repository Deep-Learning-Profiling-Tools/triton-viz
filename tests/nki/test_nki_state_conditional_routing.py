import pytest

from triton_viz.tools.nki_evaluate_state_conditional_routing import (
    _evaluate,
    _feature_names,
    _fit_state_models,
)


def _sample(case, state_features, vector, scalar):
    return {
        "case": case,
        "features": state_features,
        "payloads": {"vector": vector, "scalar": scalar},
    }


def test_state_conditional_model_conserves_vector_scalar_payload():
    states = {
        "canonical": {
            "program_dag_join_count": 3,
            "program_dag_branch_add_multiply_order_max": 0.3,
        },
        "reversed": {
            "program_dag_join_count": 3,
            "program_dag_branch_add_multiply_order_max": 0.03,
        },
        "interleaved": {
            "program_dag_join_count": 3,
            "program_dag_branch_source_interleave_count": 7,
        },
        "blocked": {
            "program_dag_join_count": 5,
            "program_dag_branch_source_interleave_count": 2,
        },
    }
    train = []
    for state, features in states.items():
        train.extend(
            (
                _sample(f"{state}_0", {**features, "free_dim_linear": 1}, 80, 20),
                _sample(f"{state}_1", {**features, "free_dim_linear": 2}, 160, 40),
            )
        )
    feature_names = [
        "free_dim_linear",
        "program_dag_join_count",
        "program_dag_branch_add_multiply_order_max",
        "program_dag_branch_source_interleave_count",
    ]
    models = _fit_state_models(train, feature_names)
    audit = [
        _sample(
            f"{state}_audit",
            {**features, "free_dim_linear": 1.5},
            120,
            30,
        )
        for state, features in states.items()
    ]

    report = _evaluate(
        models,
        audit,
        feature_names,
        "audit",
    )

    assert report["coverage"] == 1
    assert report["ood_states"] == {}
    for row in report["rows"]:
        assert abs(sum(row["predicted"]) - sum(row["actual"])) < 1e-6


def test_state_conditional_audit_reports_untrained_state_as_ood():
    canonical = {
        "program_dag_join_count": 3,
        "program_dag_branch_add_multiply_order_max": 0.3,
    }
    models = _fit_state_models(
        [
            _sample("a", {**canonical, "free": 1}, 80, 20),
            _sample("b", {**canonical, "free": 2}, 160, 40),
        ],
        ["free"],
    )
    blocked = {
        "program_dag_join_count": 5,
        "program_dag_branch_source_interleave_count": 2,
        "free": 1.5,
    }

    report = _evaluate(
        models,
        [_sample("blocked", blocked, 100, 50)],
        ["free"],
        "audit",
    )

    assert report["coverage"] == 0
    assert report["ood_states"] == {"blocked": 1}
    assert report["engines"]["vector"]["mape_pct"] is None


def test_join_ownership_models_use_composite_regime_keys():
    canonical_branch0 = {
        "program_dag_join_count": 3,
        "program_dag_branch_add_multiply_order_max": 0.3,
        "program_dag_join_update_branch0_count": 2,
        "program_dag_first_join_update_branch0": 1,
    }
    samples = [
        _sample("a", {**canonical_branch0, "free": 1}, 80, 20),
        _sample("b", {**canonical_branch0, "free": 2}, 160, 40),
    ]

    models = _fit_state_models(
        samples,
        ["free"],
        include_join_ownership=True,
    )

    assert list(models) == ["canonical:branch0_only"]


def test_no_geometry_feature_mode_excludes_shape_terms():
    names = _feature_names("no_geometry")

    assert names
    assert not any(
        term in name
        for name in names
        for term in (
            "free",
            "partition",
            "allocation",
            "bytes",
            "mask",
            "physical",
            "logical",
            "hbm",
            "transfer",
        )
    )


def test_unknown_feature_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown feature mode"):
        _feature_names("future")
