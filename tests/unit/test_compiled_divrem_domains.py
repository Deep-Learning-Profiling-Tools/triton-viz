"""Exact signed arithmetic under the compiled encoder's declared domains."""

import pytest
import z3

from triton_viz.clients.common.ttir_reader import (
    AccessGraph,
    Arange,
    Bin,
    Cmp,
    Const,
    Loaded,
    LoopVar,
    NumPrograms,
    Observed,
    Param,
    Pid,
    Select,
)
from triton_viz.clients.race_detector.compiled import global_records as gr


def _env(*, symbolic_params=False):
    return gr._RaceEnv(
        AccessGraph("divrem", [], [], None),
        {"size": 16, "negative": -7, "zero": 0},
        symbolic_params=symbolic_params,
    )


def _domains(env):
    return [
        *(pid >= 0 for pid in env._pids),
        *(z3.Int(f"grid_{axis}") >= 1 for axis in range(3)),
        *(
            z3.And(values[0] >= key[0], values[0] < key[1])
            for key, values in env.arange_dict.items()
        ),
    ]


def _legacy(env, term, monkeypatch):
    # Force the pre-existing signed lowering at EVERY subtree. This is
    # independent of the interval transfer rules being tested.
    with monkeypatch.context() as patch:
        patch.setattr(gr, "_integer_bounds", lambda *args: (None, None))
        return env.eval(term)


def _proved_nonnegative_quotient(env, divisor):
    # Prove the division identity separately before reusing it in the
    # remainder proof. Combining both identities makes the nonlinear SMT
    # search heuristic-sensitive, although either obligation is cheap.
    equality = gr._trunc_div(env._pids[0], divisor) == env._pids[0] / divisor
    solver = z3.Solver()
    solver.set(timeout=2000)
    solver.add(*_domains(env), z3.Not(equality))
    assert solver.check() == z3.unsat
    return equality


_LANE = Arange("%lane", 0, 16)
_NEG_LANE = Arange("%negative_lane", -16, 0)
_MIXED_LANE = Arange("%mixed_lane", -8, 8)
_INPUTS = [
    Pid(0),
    Bin("+", Bin("*", Pid(0), Const(16)), _LANE),
    Bin("-", Const(0), Pid(1)),
    _NEG_LANE,
    _MIXED_LANE,
    Bin("-", Pid(0), Const(7)),
    Bin("%", Pid(0), Const(16)),
    Bin("//", Pid(0), NumPrograms(0)),
    Bin("min", Pid(0), Const(8)),
    Bin("max", Param("negative"), Pid(0)),
    Select(Cmp("slt", Pid(0), Const(4)), _LANE, Const(3)),
]


@pytest.mark.parametrize("operation", ["//", "%"])
@pytest.mark.parametrize("numerator", _INPUTS)
@pytest.mark.parametrize("denominator", [Const(4), Const(-4)])
def test_lowered_divrem_equals_original_for_all_domain_values(
    numerator, denominator, operation, monkeypatch
):
    env = _env()
    term = Bin(operation, numerator, denominator)
    lowered = env.eval(term)
    original = _legacy(env, term, monkeypatch)
    solver = z3.Solver()
    solver.set(timeout=2000)
    solver.add(*_domains(env), lowered != original)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("operation", ["//", "%"])
@pytest.mark.parametrize("symbolic_params", [False, True])
@pytest.mark.parametrize(
    "denominator", [Param("zero"), Param("size"), Param("negative")]
)
def test_zero_and_t0_unknown_parameter_cases_preserve_original(
    operation, symbolic_params, denominator, monkeypatch
):
    env = _env(symbolic_params=symbolic_params)
    term = Bin(operation, Pid(0), denominator)
    lowered = env.eval(term)
    original = _legacy(env, term, monkeypatch)
    quotient_identity = _proved_nonnegative_quotient(env, env.eval(denominator))
    if symbolic_params:
        divisor = env.eval(denominator)
        cases = [divisor > 0, divisor < 0, divisor == 0]
    else:
        cases = [z3.BoolVal(True)]
    for case in cases:
        solver = z3.Solver()
        solver.set(timeout=2000)
        solver.add(*_domains(env), quotient_identity, case, lowered != original)
        assert solver.check() == z3.unsat


@pytest.mark.parametrize("operation", ["//", "%"])
def test_num_programs_positivity_is_only_the_existing_grid_domain(
    operation, monkeypatch
):
    env = _env(symbolic_params=True)
    term = Bin(operation, Pid(0), NumPrograms(0))
    result = env.eval(term)
    expected = (
        env._pids[0] / z3.Int("grid_0")
        if operation == "//"
        else env._pids[0] % z3.Int("grid_0")
    )
    assert z3.eq(result, expected)
    # Z3's nonlinear division is expensive to compare symbolically; exact
    # representative grids test both divisible and nondivisible values.
    original = _legacy(env, term, monkeypatch)
    for grid in (1, 2, 7, 64):
        solver = z3.Solver()
        solver.add(*_domains(env), z3.Int("grid_0") == grid, result != original)
        assert solver.check() == z3.unsat


def test_group_tail_denominator_that_can_be_zero_or_negative_keeps_signed_semantics(
    monkeypatch,
):
    env = _env()
    group_id = Bin("//", Pid(0), Const(32))
    group_size = Bin("min", Bin("-", Const(16), Bin("*", group_id, Const(8))), Const(8))
    assert gr._integer_bounds(group_size, env.params, False) == (None, 8)
    for op in ("//", "%"):
        term = Bin(op, Bin("%", Pid(0), Const(32)), group_size)
        result = env.eval(term)
        original = _legacy(env, term, monkeypatch)
        # Includes partial groups, denominator zero, and negative group
        # sizes outside the launched grid. No launch/mask facts are used.
        for pid in (0, 31, 32, 63, 64, 95, 96, 127, 1024):
            solver = z3.Solver()
            solver.add(*_domains(env), env._pids[0] == pid, result != original)
            assert solver.check() == z3.unsat


@pytest.mark.parametrize(
    "term",
    [
        Loaded(0, "offsets", Pid(0), None, None),
        LoopVar("%loop"),
        Observed(0),
        Param("missing"),
    ],
)
def test_loaded_loop_observation_and_unknown_param_do_not_get_bounds(term):
    assert gr._integer_bounds(term, _env().params, False) == (None, None)


def test_t0_does_not_take_positive_param_bound_from_the_captured_launch():
    assert gr._integer_bounds(Param("size"), {"size": 16}, True) == (None, None)


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        (Bin("*", _NEG_LANE, _MIXED_LANE), (-112, 128)),
        (Bin("*", Const(0), Observed(0)), (0, 0)),
        (Bin("*", Pid(0), Const(-3)), (None, 0)),
        (Bin("*", Bin("-", Const(0), Pid(0)), Const(-3)), (0, None)),
        (Bin("min", Observed(0), Const(3)), (None, 3)),
        (Bin("max", Observed(0), Const(3)), (3, None)),
        (Bin("%", _NEG_LANE, Const(4)), (-3, 0)),
        (Bin("%", _MIXED_LANE, Const(4)), (-3, 3)),
        (Bin("%", Pid(0), Const(4)), (0, 3)),
    ],
)
def test_interval_endpoints_include_sign_and_partial_information(term, expected):
    assert gr._integer_bounds(term, {}, False) == expected


def test_nested_positive_swizzle_uses_native_remainder_without_signed_fallback(
    monkeypatch,
):
    env = _env()

    def unexpected_signed_fallback(*args):
        raise AssertionError("all operands have unconditional nonnegative domains")

    monkeypatch.setattr(gr, "_trunc_div", unexpected_signed_fallback)
    term = Bin("//", Bin("%", Pid(0), Const(64)), Const(8))
    result = env.eval(term)
    assert z3.eq(result, (env._pids[0] % 64) / 8)


def test_bounds_cache_reuses_shared_terms_without_repeating_transfer(monkeypatch):
    env = _env()
    shared = Bin("%", Pid(0), Param("size"))
    original = gr._compute_integer_bounds
    calls = []

    def counted(term, *args):
        calls.append(term)
        return original(term, *args)

    monkeypatch.setattr(gr, "_compute_integer_bounds", counted)
    assert env._integer_bounds(Bin("+", shared, shared)) == (0, 30)
    first_count = len(calls)
    assert calls.count(shared) == 1
    assert env._integer_bounds(shared) == (0, 15)
    assert len(calls) == first_count


def test_bounds_cache_tracks_mutation_replacement_and_symbolic_mode():
    env = _env()
    term = Bin("%", Pid(0), Param("size"))
    assert env._integer_bounds(term) == (0, 15)
    env.params["size"] = 8
    assert env._integer_bounds(term) == (0, 7)
    env.params = {"size": -8}
    assert env._integer_bounds(term) == (0, None)
    env.params = {"size": 32}
    assert env._integer_bounds(term) == (0, 31)
    env.symbolic_params = True
    assert env._integer_bounds(term) == (0, None)


def test_environment_bounds_caches_do_not_share_launch_facts():
    left, right = _env(), _env()
    right.params["size"] = -4
    assert left._integer_bounds(Param("size")) == (16, 16)
    assert right._integer_bounds(Param("size")) == (-4, -4)


@pytest.mark.parametrize("op", ["//", "%"])
@pytest.mark.parametrize("divisor_sign", [-1, 0, 1])
def test_nonnegative_dividend_native_identity_for_unbounded_divisor(
    op, divisor_sign, monkeypatch
):
    env = _env(symbolic_params=True)
    term = Bin(op, Pid(0), Param("size"))
    result = env.eval(term)
    original = _legacy(env, term, monkeypatch)
    divisor = env.eval(Param("size"))
    sign = {-1: divisor < 0, 0: divisor == 0, 1: divisor > 0}[divisor_sign]
    solver = z3.Solver()
    solver.set(timeout=2000)
    quotient_identity = _proved_nonnegative_quotient(env, divisor)
    solver.add(*_domains(env), quotient_identity, sign, result != original)
    assert solver.check() == z3.unsat
    if op == "//":
        assert z3.eq(result, env._pids[0] / divisor)
    else:
        # A native mod-by-zero would be a different undefined term. The
        # guard must preserve the original expanded remainder's value a.
        zero = z3.simplify(z3.substitute(result, (divisor, z3.IntVal(0))))
        assert z3.eq(zero, env._pids[0])
