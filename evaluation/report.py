"""JSONL results -> RESULTS.md (minimal skeleton version).

Full DRB-style scoring (per-pattern table, witness-level matching, ladder
audit) lands with Phase A; this version renders the per-row table, the
terminal-state distribution, and the basic TP/FP/coverage counts that the
`expected` labels already allow.
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


def _score(rows: list[dict]) -> dict:
    c: Counter[str] = Counter()
    for r in rows:
        exp, verdict = r.get("expected"), r.get("verdict")
        if verdict == "error":
            c["error"] += 1
        elif verdict == "abstain":
            # abstentions split (LLOV-style): reported-but-uncertified vs
            # never-entered-the-pipeline
            c[
                "abstain-unconfirmed"
                if r.get("terminal") == "race-unconfirmed"
                else "abstain-unsupported"
            ] += 1
        elif exp == "race":
            c["TP" if verdict == "race" else "FN"] += 1
        elif exp == "race-free":
            c["FP" if verdict == "race" else "TN"] += 1
    n = len(rows)
    decided = c["TP"] + c["FP"] + c["TN"] + c["FN"]
    out: dict[str, object] = dict(c)
    out["coverage"] = f"{decided}/{n}"
    if c["TP"] + c["FP"]:
        out["precision"] = round(c["TP"] / (c["TP"] + c["FP"]), 3)
    if c["TP"] + c["FN"]:
        out["recall"] = round(c["TP"] / (c["TP"] + c["FN"]), 3)
    return out


def render(paths: list[Path]) -> str:
    lines: list[str] = ["# Evaluation results", ""]
    for path in paths:
        header, rows = _load(path)
        lines += [
            f"## {header.get('corpus', path.stem)}",
            "",
            f"versions: triton {header.get('triton')}, z3 {header.get('z3')}, "
            f"torch {header.get('torch')}, numpy {header.get('numpy')}, "
            f"commit {header.get('commit')}, seed {header.get('seed')}",
            "",
            "| kernel | pattern | expected | terminal | dyn status | C3 | wall s |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            dyn = r.get("dynamic") or {}
            diff = (r.get("static") or {}).get("differential")
            c3 = (
                "-"
                if diff is None
                else ("agree" if diff == [] else f"{len(diff)} mismatch")
            )
            lines.append(
                f"| {r['name']} | {r.get('pattern', '')} | {r.get('expected', '')} "
                f"| {r.get('terminal', '?')} | {dyn.get('status', '-')}"
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
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    results = [Path(p) for p in sys.argv[1:]] or sorted(
        (Path(__file__).parent / "results").glob("*.jsonl")
    )
    print(render(results))
