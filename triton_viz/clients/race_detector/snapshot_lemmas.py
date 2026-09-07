"""Entailed range and injectivity facts from existing finite array equations.

This helper never loads a snapshot or adds launch assumptions. It recognizes
only equations already asserted as top-level conjuncts. A finite, contiguous
integer interval and its values certify the resulting guarded lemmas. Values
outside that interval remain unconstrained, including in any-grid queries.
The original formula must remain present when these lemmas are used.
"""

from __future__ import annotations

from itertools import combinations

import z3


# Avoid generating quadratically many optional facts for an unrelated large
# base query. Declining injectivity leaves both the original formula and the
# linear number of valid range facts intact.
_MAX_INJECTIVITY_READS = 16


def _conjuncts(expression):
    pending = [expression]
    while pending:
        node = pending.pop()
        if z3.is_and(node):
            pending.extend(reversed(node.children()))
        else:
            yield node


def _nodes(expression):
    pending, seen = [expression], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        # A read with a bound index cannot be lifted out of its binder.
        if z3.is_quantifier(node):
            continue
        yield node
        pending.extend(node.children())


def snapshot_table_lemmas(expression) -> tuple[z3.BoolRef, ...]:
    """Return independently entailed lemmas, without solving or new premises.

    For each original table, completeness is checked by comparing the number
    of distinct keys to the inclusive interval's size. Every recorded value
    is checked when deriving the range; injectivity additionally requires all
    those values to differ. Conflicting equations and holes decline the table.
    Duplicate values retain only the valid range facts.
    """
    if not isinstance(expression, z3.BoolRef):
        return ()
    tables: dict[z3.ArrayRef, dict[int, int]] = {}
    inconsistent: set[z3.ArrayRef] = set()
    for atom in _conjuncts(expression):
        if not z3.is_eq(atom):
            continue
        for read, value in (atom.children(), tuple(reversed(atom.children()))):
            if not (
                z3.is_select(read)
                and read.num_args() == 2
                and z3.is_int_value(read.arg(1))
                and z3.is_int_value(value)
            ):
                continue
            array = read.arg(0)
            if (
                array.sort().domain().kind() != z3.Z3_INT_SORT
                or array.sort().range().kind() != z3.Z3_INT_SORT
            ):
                continue
            index, number = read.arg(1).as_long(), value.as_long()
            table = tables.setdefault(array, {})
            if index in table and table[index] != number:
                inconsistent.add(array)
            table[index] = number

    certificates = {}
    for array, table in tables.items():
        lower, upper = min(table), max(table)
        if array in inconsistent or len(table) != upper - lower + 1:
            continue
        values = tuple(table.values())
        certificates[array] = (
            lower,
            upper,
            min(values),
            max(values),
            len(set(values)) == len(table),
        )
    if not certificates:
        return ()

    reads: dict[z3.ArrayRef, list[z3.ArithRef]] = {array: [] for array in certificates}
    for node in _nodes(expression):
        if (
            z3.is_select(node)
            and node.num_args() == 2
            and node.arg(0) in certificates
            and not z3.is_int_value(node.arg(1))
        ):
            reads[node.arg(0)].append(node)

    lemmas: list[z3.BoolRef] = []
    for array, actual_reads in reads.items():
        lower, upper, minimum, maximum, injective = certificates[array]
        guarded = [
            (read, z3.And(read.arg(1) >= lower, read.arg(1) <= upper))
            for read in actual_reads
        ]
        lemmas.extend(
            z3.Implies(guard, z3.And(read >= minimum, read <= maximum))
            for read, guard in guarded
        )
        if injective and len(guarded) <= _MAX_INJECTIVITY_READS:
            lemmas.extend(
                z3.Implies(
                    z3.And(guard_a, guard_b, read_a == read_b),
                    read_a.arg(1) == read_b.arg(1),
                )
                for (read_a, guard_a), (read_b, guard_b) in combinations(guarded, 2)
            )
    return tuple(lemmas)
