"""Exact satisfiability normalization for isolated reads of free arrays.

If an array appears only as the base of one distinct Select expression,
that read can take any scalar value. Replacing the read by a fresh scalar
and binding the array to the constant array with that value preserves
satisfiability, including every original non-array variable's valuation.
The binding also makes an ordinary Z3 model satisfy the original formula,
so callers need neither model conversion nor a weaker witness policy.

This deliberately small rule does not handle multiple reads, snapshots
with several cells, array equalities, stores, quantifiers, or other sorts.
It adds no grid, input-content, activity, or address-domain premise.
"""

from __future__ import annotations

import z3


def single_read_array_normal_form(expression: z3.BoolRef) -> z3.BoolRef | None:
    """Return an equisatisfiable formula with native original witnesses.

    None means that the complete expression does not match. Eligibility is
    checked over the entire assertion, not just its address subexpression:
    a second read or any other use of an array invalidates the rule.

    Soundness has two directions. A model of the original formula provides
    each fresh scalar's value by evaluating its Select; replacing the array
    by that constant array satisfies the result. Conversely, the explicit
    constant-array bindings make every replaced Select equal its scalar,
    including when its index contains another eliminated array's read.
    Thus a model of the result satisfies the complete original formula.
    Every occurrence of an identical Select shares one fresh variable.
    """
    if not isinstance(expression, z3.BoolRef):
        return None
    reads: dict[z3.ArrayRef, z3.ExprRef] = {}
    pending, seen = [expression], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        # Hoisting a read out of a quantifier would change its dependency
        # on bound variables. Lambdas are quantifiers in the Z3 AST too.
        if z3.is_quantifier(node):
            return None
        if z3.is_select(node):
            array = node.arg(0)
            if (
                node.num_args() != 2
                or not z3.is_const(array)
                or array.decl().kind() != z3.Z3_OP_UNINTERPRETED
                or array.domain().kind() != z3.Z3_INT_SORT
                or array.range().kind() != z3.Z3_INT_SORT
            ):
                return None
            prior = reads.get(array)
            if prior is not None and not prior.eq(node):
                return None
            reads[array] = node
        for child in node.children():
            if child.sort().kind() == z3.Z3_ARRAY_SORT and not (
                z3.is_select(node) and child.eq(node.arg(0))
            ):
                return None
            pending.append(child)
    if not reads:
        return None

    replacements, bindings = [], []
    for array, read in reads.items():
        value = z3.FreshInt("single_read_value", ctx=expression.ctx)
        replacements.append((read, value))
        bindings.append(array == z3.K(array.domain(), value))
    scalar_formula = z3.substitute(expression, *replacements)
    return z3.And(scalar_formula, *bindings)


def single_read_array_solver(expression: z3.BoolRef) -> z3.Solver | None:
    """Build an exact solver without changing the caller's query limits.

    The solve-eqs tactic eliminates the constant-array witness bindings
    before the general SMT backend runs. Its model converter restores the
    original arrays automatically. Keeping those bindings in an ordinary
    solver leaves the KDA query on its expensive array-solving path.

    Callers retain their existing timeout settings and unknown handling.
    Additional scalar constraints, such as the original enumeration PID
    pins, are safe. All array-bearing constraints must already be present
    in expression when eligibility is checked.
    """
    try:
        rewritten = single_read_array_normal_form(expression)
        if rewritten is None:
            return None
        solver = z3.Then(
            *(
                z3.Tactic(name, ctx=expression.ctx)
                for name in ("simplify", "solve-eqs", "smt")
            )
        ).solver()
        solver.add(rewritten)
    except z3.Z3Exception:
        # Failure to prepare an optimization is not a proof or a refusal:
        # the caller constructs its unchanged complete query instead.
        return None
    return solver
