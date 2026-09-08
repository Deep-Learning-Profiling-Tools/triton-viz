"""Entailed range and injectivity facts from existing finite array equations.

This helper never loads a snapshot or adds launch assumptions. It recognizes
only equations already asserted as top-level conjuncts. A finite, contiguous
integer interval and its values certify the resulting guarded lemmas. Values
outside that interval remain unconstrained, including in any-grid queries.
The original formula must remain present when these lemmas are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import z3


# Avoid generating quadratically many optional facts for an unrelated large
# base query. Declining injectivity leaves both the original formula and the
# linear number of valid range facts intact.
_MAX_INJECTIVITY_READS = 16


@dataclass(eq=False, frozen=True, slots=True)
class _CellFacts:
    """A persistent DAG of original asserted cell equations."""

    children: tuple = ()
    cell: tuple | None = None


@dataclass(frozen=True, slots=True)
class _Summary:
    facts: _CellFacts | None = None
    reads: tuple = ()


class SnapshotLemmaCache:
    """One solver's reusable facts about immutable, original expressions.

    Node summaries do not depend on any assumptions. Only conjunctions lift
    cell equations into the asserted-fact summary; bound reads never leave a
    quantifier. Certificates are keyed by the exact current fact summaries,
    so pair-specific reads reuse a common table without reparsing its cells.
    Removing or changing an asserted equation requires another certificate.
    Neither mutable condition lists nor solver decisions are memoized.
    """

    def __init__(self):
        self._context = None
        self._summaries = {}
        self._fact_joins = {}
        self._certificates = {}
        self._guarded_reads = {}
        self._injectivity = {}

    @staticmethod
    def _cell_equation(node):
        if not z3.is_eq(node):
            return None
        for read, value in (node.children(), tuple(reversed(node.children()))):
            if not (
                z3.is_select(read)
                and read.num_args() == 2
                and z3.is_int_value(read.arg(1))
                and z3.is_int_value(value)
            ):
                continue
            array = read.arg(0)
            if (
                array.sort().domain().kind() == z3.Z3_INT_SORT
                and array.sort().range().kind() == z3.Z3_INT_SORT
            ):
                return array, read.arg(1).as_long(), value.as_long()
        return None

    def _summary(self, expression):
        # Iterative postorder handles deep expressions without Python recursion.
        pending = [(expression, False)]
        while pending:
            node, visited = pending.pop()
            if node in self._summaries:
                continue
            if z3.is_quantifier(node):
                self._summaries[node] = _Summary()
                continue
            children = node.children()
            if children and not visited:
                pending.append((node, True))
                pending.extend((child, False) for child in children)
                continue
            summaries = [self._summaries[child] for child in children]
            facts = None
            if z3.is_and(node):
                roots = tuple(
                    dict.fromkeys(s.facts for s in summaries if s.facts is not None)
                )
                if roots:
                    if len(roots) == 1:
                        facts = roots[0]
                    else:
                        if roots not in self._fact_joins:
                            self._fact_joins[roots] = _CellFacts(roots)
                        facts = self._fact_joins[roots]
            else:
                cell = self._cell_equation(node)
                if cell is not None:
                    facts = _CellFacts(cell=cell)
            # Match the standalone helper's reverse-child DFS read ordering.
            own_read = (
                (node,)
                if z3.is_select(node)
                and node.num_args() == 2
                and not z3.is_int_value(node.arg(1))
                else ()
            )
            reads = tuple(
                dict.fromkeys(
                    (*own_read, *(r for s in reversed(summaries) for r in s.reads))
                )
            )
            self._summaries[node] = _Summary(facts, reads)
        return self._summaries[expression]

    @staticmethod
    def _certify(roots):
        tables, inconsistent = {}, set()
        pending, seen = list(reversed(roots)), set()
        while pending:
            facts = pending.pop()
            if facts in seen:
                continue
            seen.add(facts)
            if facts.cell is None:
                pending.extend(reversed(facts.children))
                continue
            array, index, number = facts.cell
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
        return certificates

    def lemmas(self, conditions) -> tuple[z3.BoolRef, ...]:
        conditions = tuple(c for c in conditions if not isinstance(c, bool))
        if not conditions or not all(isinstance(c, z3.BoolRef) for c in conditions):
            return ()
        context = conditions[0].ctx
        if any(c.ctx != context for c in conditions):
            raise z3.Z3Exception("snapshot conditions have different contexts")
        if self._context != context:
            # A solver normally uses one context. A reused helper may switch,
            # but equal-looking ASTs from different contexts never share facts.
            self._summaries.clear()
            self._fact_joins.clear()
            self._certificates.clear()
            self._guarded_reads.clear()
            self._injectivity.clear()
            self._context = context
        summaries = [self._summary(condition) for condition in conditions]
        roots = tuple(dict.fromkeys(s.facts for s in summaries if s.facts is not None))
        if roots not in self._certificates:
            self._certificates[roots] = self._certify(roots)
        certificates = self._certificates[roots]
        if not certificates:
            return ()
        reads: dict[z3.ArrayRef, list[z3.ArithRef]] = {
            array: [] for array in certificates
        }
        actual_reads = dict.fromkeys(r for s in reversed(summaries) for r in s.reads)
        for read in actual_reads:
            if read.arg(0) in certificates:
                reads[read.arg(0)].append(read)
        lemmas = []
        for array, table_reads in reads.items():
            certificate = certificates[array]
            lower, upper, minimum, maximum, injective = certificate
            guarded = []
            for read in table_reads:
                key = read, certificate
                if key not in self._guarded_reads:
                    guard = z3.And(read.arg(1) >= lower, read.arg(1) <= upper)
                    lemma = z3.Implies(guard, z3.And(read >= minimum, read <= maximum))
                    self._guarded_reads[key] = guard, lemma
                guard, lemma = self._guarded_reads[key]
                guarded.append((read, guard))
                lemmas.append(lemma)
            if injective and len(guarded) <= _MAX_INJECTIVITY_READS:
                for (read_a, guard_a), (read_b, guard_b) in combinations(guarded, 2):
                    pair_key = read_a, read_b, certificate
                    if pair_key not in self._injectivity:
                        self._injectivity[pair_key] = z3.Implies(
                            z3.And(guard_a, guard_b, read_a == read_b),
                            read_a.arg(1) == read_b.arg(1),
                        )
                    lemmas.append(self._injectivity[pair_key])
        return tuple(lemmas)


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
