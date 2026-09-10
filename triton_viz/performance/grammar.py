"""Auditable lowering rule selection shared by hardware backends."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GrammarRule:
    rule_id: str
    priority: int
    predicate: Callable[[dict[str, Any]], bool]
    family: str | Callable[[dict[str, Any]], str]
    condition: str
    rationale: str
    evidence: tuple[str, ...] = ()

    def render_family(self, facts: dict[str, Any]) -> str:
        return self.family(facts) if callable(self.family) else self.family


@dataclass(frozen=True)
class GrammarMatch:
    family: str
    rule_id: str
    rationale: str
    evidence: tuple[str, ...]
    ood_reasons: tuple[str, ...]
    consumed_features: tuple[str, ...]


def select_rule(facts: dict[str, Any], rules: tuple[GrammarRule, ...]) -> GrammarRule:
    """Select a unique highest-priority rule; never break ties by table order."""
    matches = [rule for rule in rules if rule.predicate(facts)]
    if not matches:
        raise ValueError(f"No grammar rule matched region facts: {facts}")
    priority = max(rule.priority for rule in matches)
    winners = [rule for rule in matches if rule.priority == priority]
    if len(winners) != 1:
        ids = ", ".join(rule.rule_id for rule in winners)
        raise ValueError(f"Ambiguous grammar rules at priority {priority}: {ids}")
    return winners[0]
