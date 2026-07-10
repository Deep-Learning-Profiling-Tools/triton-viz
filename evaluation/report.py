"""JSONL results -> RESULTS.md (Phase A: full DRB-style scoring).

Beyond the per-row table and terminal-state distribution:
  * DRB-style TP/FP/TN/FN + precision/recall + coverage, with abstentions
    split LLOV-style (race-unconfirmed vs never-entered-the-pipeline);
  * WITNESS-LEVEL scoring (departure 1): a TP only counts as
    witness-matched when a reported witness pair lands exactly on the
    planted ``race_pair`` source lines;
  * a per-pattern table (DRB taxonomy buckets);
  * the LADDER AUDIT (departure 4): rows are grouped by SPECIALIZATION
    (kernel, constexprs) and the kernel-level "∃ racy input" truth is
    derived from the launch labels. ``ladder-unsound`` counts proved@T0
    rows whose specialization has a premise-compatible (non-aliased)
    yes-launch; ``replay-unsound`` counts race-confirmed rows on a
    no-launch. Both are REQUIRED ZERO — they rank above FP.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _load(path: Path) -> tuple[dict, list[dict]]:
    header: dict = {}
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d.get("header"):
            header = d
        else:
            rows.append(d)
    return header, rows


# ── row-level helpers ────────────────────────────────────────────


def _verdict_class(r: dict) -> str:
    """'TP'|'FP'|'TN'|'FN'|'abstain-unconfirmed'|'abstain-unsupported'|'error'"""
    exp, verdict = r.get("expected"), r.get("verdict")
    if verdict == "error":
        return "error"
    if verdict == "abstain":
        return (
            "abstain-unconfirmed"
            if r.get("terminal") == "race-unconfirmed"
            else "abstain-unsupported"
        )
    if exp == "race":
        return "TP" if verdict == "race" else "FN"
    if exp == "race-free":
        return "FP" if verdict == "race" else "TN"
    return "unlabeled"


def _witness_match(r: dict) -> str | None:
    """'match' | 'mismatch' | None (not applicable).

    Matched when SOME reported witness pair's source lines are a SUBSET of
    the planted pair's resolved lines. ``race_pair`` lists the ACCEPTABLE
    endpoints (usually two; more when the same race has several real
    endpoint variants, e.g. a predecessor store in either branch); subset
    semantics stays strict — a witness touching any unplanted line does
    not match."""
    expected = {ln for ln in (r.get("race_pair_lines") or []) if ln is not None}
    if not expected or r.get("verdict") != "race":
        return None
    witnesses = (r.get("static") or {}).get("witnesses") or []
    for w in witnesses:
        first, second = w.get("first"), w.get("second")
        got = {loc[1] for loc in (first, second) if loc}
        if got and got <= expected:
            return "match"
    return "mismatch"


def _spec_key(r: dict) -> tuple:
    """The SPECIALIZATION a T0 claim is scoped to."""
    return (
        r.get("kernel") or r.get("name"),
        json.dumps(r.get("constexprs") or {}, sort_keys=True),
    )


def ladder_audit(rows: list[dict]) -> dict:
    """Cross-row audit of the claim ladder (plan S5, departure 4)."""
    by_spec: dict[tuple, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(_spec_key(r), []).append(r)

    ladder_unsound: list[str] = []
    replay_unsound: list[str] = []
    for group in by_spec.values():
        # Premise-compatible derived truth: an ALIASED yes-launch violates
        # the T0 non-aliasing premise and cannot contradict a T0 proof;
        # likewise a yes-launch with a DIFFERENT grid geometry cannot,
        # because the T0 claim keeps the launch's extent along unread pid
        # axes (symbolic_grid enforces the launch contract instead of
        # assuming it), so grid extents are part of the T0 premises.
        def _racy_compatible(g: dict, proved_row: dict) -> bool:
            return (
                g.get("expected") == "race"
                and not g.get("aliased")
                and list(g.get("grid") or []) == list(proved_row.get("grid") or [])
            )

        for g in group:
            terminal = g.get("terminal") or ""
            if terminal.startswith("proved@T0") and any(
                _racy_compatible(other, g) for other in group
            ):
                ladder_unsound.append(g["name"])
            if terminal == "race-confirmed" and g.get("expected") == "race-free":
                replay_unsound.append(g["name"])
    return {
        "ladder_unsound": sorted(ladder_unsound),
        "replay_unsound": sorted(replay_unsound),
    }


def _score(rows: list[dict]) -> dict:
    c: Counter[str] = Counter(_verdict_class(r) for r in rows)
    witness_matched = sum(1 for r in rows if _witness_match(r) == "match")
    witness_applicable = sum(1 for r in rows if _witness_match(r) is not None)
    n = len(rows)
    decided = c["TP"] + c["FP"] + c["TN"] + c["FN"]
    out: dict[str, object] = {k: v for k, v in sorted(c.items())}
    out["coverage"] = f"{decided}/{n}"
    if c["TP"] + c["FP"]:
        out["precision"] = round(c["TP"] / (c["TP"] + c["FP"]), 3)
    if c["TP"] + c["FN"]:
        out["recall"] = round(c["TP"] / (c["TP"] + c["FN"]), 3)
    if witness_applicable:
        out["witness-matched"] = f"{witness_matched}/{witness_applicable}"
    return out


def _mutation_class(r: dict) -> str | None:
    """'flip' (some mutant detects races) | 'degraded' (no mutant races,
    but some breaks the proof into an abstention — the proof demonstrably
    hinged on the mutated ingredient) | 'SURV' (every applicable mutant
    still proves clean: a vacuity suspect, or a genuinely degenerate
    launch) | 'n/a' | None (mutation mode off or row not proved)."""
    m = r.get("mutation")
    if not m or "error" in m:
        return None
    if not m.get("applicable"):
        return "n/a"
    statuses = [s for s in (m.get("results") or {}).values() if s != "n/a"]
    if any(s == "races" for s in statuses):
        return "flip"
    if any(s != "ok" for s in statuses):
        return "degraded"
    return "SURV"


def _mutation_cell(r: dict) -> str:
    return _mutation_class(r) or "-"


def _mutation_summary(rows: list[dict]) -> str | None:
    classes = {r["name"]: _mutation_class(r) for r in rows}
    ran = {n: c for n, c in classes.items() if c is not None}
    if not ran:
        return None
    applicable = {n: c for n, c in ran.items() if c != "n/a"}
    flipped = [n for n, c in applicable.items() if c == "flip"]
    degraded = [n for n, c in applicable.items() if c == "degraded"]
    survivors = [n for n, c in applicable.items() if c == "SURV"]
    return (
        f"**Mutation sensitivity**: {len(flipped)}/{len(applicable)} proofs "
        f"flipped to a race by at least one mutant"
        + (
            f"; degraded to abstention (proof hinged on the ingredient): {degraded}"
            if degraded
            else ""
        )
        + (f"; SURVIVORS (vacuity suspects): {survivors}" if survivors else "")
    )


def _pattern_table(rows: list[dict]) -> list[str]:
    by_pattern: dict[str, list[dict]] = {}
    for r in rows:
        by_pattern.setdefault(r.get("pattern") or "?", []).append(r)
    lines = [
        "| pattern | rows | TP | FP | TN | FN | abstain | witness |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pattern in sorted(by_pattern):
        grp = by_pattern[pattern]
        c = Counter(_verdict_class(r) for r in grp)
        abstain = c["abstain-unconfirmed"] + c["abstain-unsupported"]
        wm = sum(1 for r in grp if _witness_match(r) == "match")
        wa = sum(1 for r in grp if _witness_match(r) is not None)
        witness = f"{wm}/{wa}" if wa else "-"
        lines.append(
            f"| {pattern} | {len(grp)} | {c['TP']} | {c['FP']} | {c['TN']} "
            f"| {c['FN']} | {abstain} | {witness} |"
        )
    return lines


def render(paths: list[Path]) -> str:
    lines: list[str] = ["# Evaluation results", ""]
    for path in paths:
        header, rows = _load(path)
        lines += [
            f"## {header.get('corpus', path.stem)}",
            "",
            f"versions: triton {header.get('triton')}, z3 {header.get('z3')}, "
            f"torch {header.get('torch')}, numpy {header.get('numpy')}, "
            f"commit {header.get('commit')}, seed {header.get('seed')}"
            + (
                f", liger-kernel {header['liger_kernel']}"
                f" (upstream {header.get('liger_kernel_commit') or 'unknown'})"
                if header.get("liger_kernel")
                else ""
            ),
            "",
            "| kernel | pattern | expected | terminal | witness | mut | "
            "dyn status | C3 | wall s |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            dyn = r.get("dynamic") or {}
            diff = (r.get("static") or {}).get("differential")
            c3 = (
                "-"
                if diff is None
                else ("agree" if diff == [] else f"{len(diff)} mismatch")
            )
            wm = _witness_match(r)
            witness = {"match": "✓", "mismatch": "≠", None: "-"}[wm]
            terminal = r.get("terminal", "?")
            lines.append(
                f"| {r['name']} | {r.get('pattern', '')} | {r.get('expected', '')} "
                f"| {terminal} | {witness} | {_mutation_cell(r)} "
                f"| {dyn.get('status', '-')}"
                f"({dyn.get('n_reports', 0)}) | {c3} | {r.get('wall_s', '')} |"
            )
        lines += ["", "**Terminal states**: "]
        lines.append(
            ", ".join(
                f"{k}={v}"
                for k, v in sorted(Counter(r.get("terminal") for r in rows).items())
            )
        )
        lines += ["", "**Scores**: " + json.dumps(_score(rows)), ""]
        mut = _mutation_summary(rows)
        if mut:
            lines += [mut, ""]
        lines += ["**Per-pattern**:", ""]
        lines += _pattern_table(rows)
        audit = ladder_audit(rows)
        ok = not audit["ladder_unsound"] and not audit["replay_unsound"]
        lines += [
            "",
            "**Ladder audit** (required both zero): "
            f"ladder-unsound={len(audit['ladder_unsound'])} "
            f"{audit['ladder_unsound'] or ''}, "
            f"replay-unsound={len(audit['replay_unsound'])} "
            f"{audit['replay_unsound'] or ''} "
            f"→ {'PASS' if ok else 'FAIL'}",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    results = [Path(p) for p in sys.argv[1:]] or sorted(
        (Path(__file__).parent / "results").glob("*.jsonl")
    )
    print(render(results))
