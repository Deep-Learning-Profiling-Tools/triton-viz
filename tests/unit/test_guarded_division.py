"""Original masks certify rewrites without restricting grid or input scope."""

import pytest
import z3

from triton_viz.clients.race_detector import guarded_division as gd


def _grouped(*, masked=True, context=None):
    pid, lane, grid = z3.Ints("pid lane grid", ctx=context)
    quotient = pid / 4
    size = z3.If(quotient >= 0, 2 - 2 * quotient, 2)
    remainder = z3.If(size == 0, pid % 4, (pid % 4) % size)
    row = 64 * quotient + 32 * remainder + lane
    conditions = [pid >= 0, pid < grid, grid >= 1, lane >= 0, lane < 32]
    if masked:
        conditions.append(row < 64)
    return z3.And(*conditions), (pid, lane, grid), quotient, size, row


def _check(formula, *extra):
    solver = z3.Solver(ctx=formula.ctx)
    solver.set(timeout=3000)
    solver.add(formula, *extra)
    result = solver.check()
    assert result != z3.unknown
    return result, solver


def test_original_access_mask_certifies_constants_without_grid_pin():
    original, (pid, lane, grid), quotient, size, _ = _grouped()
    rewritten = gd.guarded_division_normal_form(original)
    assert not z3.eq(original, rewritten)
    assert _check(z3.Xor(original, rewritten))[0] == z3.unsat
    for extent in (4, 1024, 1000000):
        answer, solver = _check(rewritten, grid == extent, pid == 1, lane == 0)
        assert answer == z3.sat
        model = solver.model()
        assert z3.is_true(model.eval(original, model_completion=True))
        assert model.eval(quotient).as_long() == 0
        assert model.eval(size).as_long() == 2


def test_missing_mask_keeps_zero_and_negative_group_size_models():
    original, (pid, lane, grid), _, size, _ = _grouped(masked=False)
    rewritten = gd.guarded_division_normal_form(original)
    for instance, divisor in ((4, 0), (8, -2)):
        result, solver = _check(rewritten, grid == 16, pid == instance, lane == 0)
        assert result == z3.sat
        assert solver.model().eval(size).as_long() == divisor
        assert z3.is_true(solver.model().eval(original, model_completion=True))


@pytest.mark.parametrize("connective", ["or", "implies"])
def test_conditional_mask_is_not_promoted_to_unconditional_premise(connective):
    base, (pid, lane, grid), _, _, row = _grouped(masked=False)
    flag = z3.Bool("flag")
    condition = (
        z3.Or(flag, row < 64) if connective == "or" else z3.Implies(flag, row < 64)
    )
    original = z3.And(base, condition)
    rewritten = gd.guarded_division_normal_form(original)
    extra = flag if connective == "or" else z3.Not(flag)
    result, solver = _check(rewritten, grid == 16, pid == 8, lane == 0, extra)
    assert result == z3.sat
    assert z3.is_true(solver.model().eval(original, model_completion=True))


@pytest.mark.parametrize("answer", [z3.unknown, z3.sat])
def test_failed_certificate_keeps_original_formula(monkeypatch, answer):
    original, *_ = _grouped()
    check = z3.Solver.check

    def inconclusive(self, *args, **kwargs):
        # Allow discovery of a candidate value, then refuse its uniqueness.
        if any(z3.is_not(atom) and z3.is_eq(atom.arg(0)) for atom in self.assertions()):
            return answer
        return check(self, *args, **kwargs)

    monkeypatch.setattr(z3.Solver, "check", inconclusive)
    assert z3.eq(gd.guarded_division_normal_form(original), original)


def test_no_candidate_and_disabled_path_do_not_invoke_solver(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("no arithmetic certificate should be requested")

    monkeypatch.setattr(z3.Solver, "check", fail)
    x = z3.Int("x")
    direct = z3.And(x >= 0, x / 4 < 8)
    assert z3.eq(gd.guarded_division_normal_form(direct), direct)
    monkeypatch.setattr(gd, "_ENABLE_GUARDED_DIVISION", False)
    original, *_ = _grouped()
    assert z3.eq(gd.guarded_division_normal_form(original), original)


def test_quantified_input_is_unchanged():
    original, (pid, _, _), *_ = _grouped()
    quantified = z3.Exists(pid, original)
    assert z3.eq(gd.guarded_division_normal_form(quantified), quantified)


def test_certificate_exception_retains_supported_original_query(monkeypatch):
    original, *_ = _grouped()

    def fail(*args, **kwargs):
        raise z3.Z3Exception("injected optional certificate failure")

    monkeypatch.setattr(z3.Solver, "check", fail)
    assert z3.eq(gd.guarded_division_normal_form(original), original)


def test_exhausted_certificate_budget_does_not_change_query(monkeypatch):
    original, *_ = _grouped()
    monkeypatch.setattr(gd, "_CERTIFY_TIMEOUT_MS", 0)
    assert z3.eq(gd.guarded_division_normal_form(original), original)


def test_custom_context_and_independent_copy_guards():
    ctx = z3.Context()
    first, (pid, lane, grid), *_ = _grouped(context=ctx)
    other = z3.substitute(
        first, (pid, z3.Int("other_pid", ctx)), (lane, z3.Int("other_lane", ctx))
    )
    original = z3.And(first, other, pid != z3.Int("other_pid", ctx))
    rewritten = gd.guarded_division_normal_form(original)
    assert not z3.eq(original, rewritten)
    assert _check(z3.Xor(original, rewritten))[0] == z3.unsat
    assert _check(rewritten, grid == 1000000)[0] == z3.sat


def test_array_constraints_and_feasibility_are_retained():
    base, (pid, _, _), *_ = _grouped()
    values = z3.Array("values", z3.IntSort(), z3.IntSort())
    original = z3.And(base, z3.Select(values, pid) == 7)
    rewritten = gd.guarded_division_normal_form(original)
    assert not z3.eq(original, rewritten)
    assert _check(z3.Xor(original, rewritten))[0] == z3.unsat
    assert _check(rewritten, z3.Select(values, pid) == 8)[0] == z3.unsat
