import pytest

from microbench.inf2_nki.tests.region_controls.kernels import (
    factorial_dag_schedule,
    factorial_dag_audit_schedule,
    factorial_dag_interleave_schedule,
    factorial_dag_blocked_audit_schedule,
    factorial_dag_role_swap_audit_schedule,
    factorial_dag_join_target_audit_schedule,
    factorial_dag_alternating_join_audit_schedule,
    factorial_dag_paired_ownership_audit_schedule,
    factorial_dag_fanout_audit_schedule,
    factorial_dag_fanout_variant_audit_schedule,
    factorial_dag_fanout_chain_audit_schedule,
    factorial_dag_fanout_branch0_extra_schedule,
    factorial_dag_fanout_branch1_extra_schedule,
    factorial_dag_fanout_deep_audit_schedule,
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


def test_factorial_dag_blocked_audit_is_fourth_disjoint_family():
    prior = {
        *[factorial_dag_schedule(seed) for seed in range(3000, 3054)],
        *[factorial_dag_audit_schedule(seed) for seed in range(4000, 4054)],
        *[factorial_dag_interleave_schedule(seed) for seed in range(5000, 5054)],
    }
    audit = {
        factorial_dag_blocked_audit_schedule(seed)
        for seed in range(6000, 6054)
    }
    assert len(audit) == 54
    assert audit.isdisjoint(prior)
    with pytest.raises(ValueError):
        factorial_dag_blocked_audit_schedule(6054)


def test_factorial_dag_role_swap_audit_balances_covered_states():
    schedules = [
        factorial_dag_role_swap_audit_schedule(seed)
        for seed in range(7000, 7054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_role_swap_audit_schedule(7054)


def test_factorial_dag_join_target_audit_balances_covered_states():
    schedules = [
        factorial_dag_join_target_audit_schedule(seed)
        for seed in range(8000, 8054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_join_target_audit_schedule(8054)


def test_factorial_dag_alternating_join_audit_balances_covered_states():
    schedules = [
        factorial_dag_alternating_join_audit_schedule(seed)
        for seed in range(9000, 9054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_alternating_join_audit_schedule(9054)


def test_factorial_dag_paired_ownership_audit_balances_covered_states():
    schedules = [
        factorial_dag_paired_ownership_audit_schedule(seed)
        for seed in range(10000, 10054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_paired_ownership_audit_schedule(10054)


def test_factorial_dag_fanout_audit_balances_covered_states():
    schedules = [
        factorial_dag_fanout_audit_schedule(seed)
        for seed in range(11000, 11054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_fanout_audit_schedule(11054)


def test_factorial_dag_fanout_variant_audit_balances_covered_states():
    schedules = [
        factorial_dag_fanout_variant_audit_schedule(seed)
        for seed in range(12000, 12054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_fanout_variant_audit_schedule(12054)


def test_factorial_dag_fanout_chain_audit_balances_covered_states():
    schedules = [
        factorial_dag_fanout_chain_audit_schedule(seed)
        for seed in range(13000, 13054)
    ]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        factorial_dag_fanout_chain_audit_schedule(13054)


@pytest.mark.parametrize(
    ("function", "start"),
    [
        (factorial_dag_fanout_branch0_extra_schedule, 14000),
        (factorial_dag_fanout_branch1_extra_schedule, 15000),
        (factorial_dag_fanout_deep_audit_schedule, 16000),
    ],
)
def test_factorial_dag_fanout_factorial_schedules_cover_states(function, start):
    schedules = [function(seed) for seed in range(start, start + 54)]
    assert len(schedules) == 54
    assert len(set(schedules)) == 54
    assert schedules[0] == factorial_dag_schedule(3000)
    assert schedules[1] == factorial_dag_audit_schedule(4001)
    assert schedules[2] == factorial_dag_interleave_schedule(5002)
    assert schedules[3] == factorial_dag_blocked_audit_schedule(6003)
    with pytest.raises(ValueError):
        function(start + 54)
