import pytest

from microbench.inf2_nki.tests.region_controls.kernels import (
    factorial_dag_schedule,
    factorial_dag_audit_schedule,
    factorial_dag_interleave_schedule,
    random_dag_schedule,
    random_semantic_schedule,
)


def test_random_semantic_schedule_is_frozen_broad_and_target_independent():
    schedules = [random_semantic_schedule(seed) for seed in range(1000, 1064)]
    assert schedules == [random_semantic_schedule(seed) for seed in range(1000, 1064)]
    assert len(set(schedules)) == 64
    assert min(map(len, schedules)) >= 6
    assert max(map(len, schedules)) >= 20
    for schedule in schedules:
        assert "reduce" in schedule
        assert set(schedule) & {"exp", "log", "rsqrt"}
        assert set(schedule) <= {
            "add", "subtract", "multiply", "divide", "maximum",
            "exp", "log", "rsqrt", "reduce", "where",
        }


def test_random_dag_schedule_is_frozen_and_has_branch_join_motifs():
    schedules = [random_dag_schedule(seed) for seed in range(2000, 2064)]
    assert schedules == [random_dag_schedule(seed) for seed in range(2000, 2064)]
    assert len(set(schedules)) == 64
    for schedule in schedules:
        assert {"a_reduce", "b_reduce", "cross_add"} <= set(schedule)
        assert any(token.startswith("a_") for token in schedule)
        assert any(token.startswith("b_") for token in schedule)


def test_factorial_dag_schedule_covers_frozen_single_factor_grid():
    schedules = [factorial_dag_schedule(seed) for seed in range(3000, 3054)]
    assert len(set(schedules)) == 54
    assert {s[-3] for s in schedules} == {"cross_add", "cross_multiply"}
    assert {sum(x.startswith("a_") for x in s) for s in schedules} >= {4, 6, 10}
    assert {s.count("a_reduce") + s.count("b_reduce") for s in schedules} == {1, 2}
    with pytest.raises(ValueError):
        factorial_dag_schedule(3054)


def test_factorial_dag_audit_is_disjoint_and_order_shifted():
    development = [factorial_dag_schedule(seed) for seed in range(3000, 3054)]
    audit = [factorial_dag_audit_schedule(seed) for seed in range(4000, 4054)]
    assert len(set(audit)) == 54
    assert set(development).isdisjoint(audit)
    assert development[0][:2] == ("a_add", "a_multiply")
    assert audit[0][:2] == ("a_multiply", "a_add")


def test_factorial_dag_interleave_is_third_disjoint_family():
    first = {factorial_dag_schedule(seed) for seed in range(3000, 3054)}
    second = {factorial_dag_audit_schedule(seed) for seed in range(4000, 4054)}
    third = {factorial_dag_interleave_schedule(seed) for seed in range(5000, 5054)}
    assert len(third) == 54
    assert third.isdisjoint(first | second)
    assert factorial_dag_interleave_schedule(5000)[:4] == (
        "a_add", "b_maximum", "a_multiply", "b_subtract"
    )
