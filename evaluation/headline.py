"""RQ2 headline numbers (plan S5) — an aggregation over the results JSONLs.

Feeds the paper's fragment-coverage placeholder. The COVERAGE corpus is
tutorials + liger (unlabeled-by-construction real code); the litmus corpora
are listed separately for context. Three headline families:

  1. proof-strength distribution — kernels reaching proved@T0 (the "any
     scalar params" claim neither the dynamic mode nor T1 can make), T1,
     and T1+assumes-termination;
  2. the static-vs-dynamic delta — rows where the DYNAMIC mode abstains
     (unsupported/aborted/timeout) while the static track produces a
     verdict (S2's acceptance criterion, quantified), and the reverse
     direction (dynamic verdicts where static abstains);
  3. the unsupported-kind distribution — where the next modeling
     investment pays.

Usage:  uv run python -m evaluation.headline [results-dir]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

COVERAGE_CORPORA = ("tutorials", "liger")


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if not d.get("header"):
            d["_corpus"] = path.stem
            out.append(d)
    return out


def _kind(r: dict) -> str | None:
    """The stable machine-readable class of an unsupported verdict (the
    'kind: message' prefix the client records)."""
    if r.get("terminal") not in ("unsupported", "race-unconfirmed"):
        return None
    reason = (r.get("static") or {}).get("reason") or ""
    head = reason.split(":", 1)[0].strip()
    return head if head and " " not in head else "other"


def _dyn_abstains(r: dict) -> bool:
    return (r.get("dynamic") or {}).get("status") not in ("ok",)


def _static_verdicts(r: dict) -> bool:
    return (r.get("static") or {}).get("status") in ("ok", "races")


def headline(results_dir: Path) -> str:
    all_rows = [r for p in sorted(results_dir.glob("*.jsonl")) for r in _rows(p)]
    coverage = [r for r in all_rows if r["_corpus"] in COVERAGE_CORPORA]
    lines = ["# RQ2 headline numbers", ""]

    def block(title: str, rows: list[dict]) -> None:
        lines.append(f"## {title} ({len(rows)} rows)")
        lines.append("")
        terminals = Counter(r.get("terminal") for r in rows)
        lines.append(
            "- terminal states: "
            + ", ".join(f"{k}={v}" for k, v in sorted(terminals.items()))
        )
        t0 = [
            r["name"] for r in rows if (r.get("terminal") or "").startswith("proved@T0")
        ]
        t1 = [
            r["name"] for r in rows if (r.get("terminal") or "").startswith("proved@T1")
        ]
        cond = [
            r["name"]
            for r in rows
            if "assumes-termination" in (r.get("terminal") or "")
        ]
        lines.append(
            f"- **proved@T0** (any scalar params, any grid — beyond both the "
            f"dynamic mode and T1): {len(t0)} — {t0}"
        )
        lines.append(
            f"- proved@T1 (this input, any grid): {len(t1)}"
            + (
                f", of which conditional on termination: {len(cond)} — {cond}"
                if cond
                else ""
            )
        )
        # static-vs-dynamic delta
        s_not_d = [r["name"] for r in rows if _static_verdicts(r) and _dyn_abstains(r)]
        d_not_s = [
            r["name"]
            for r in rows
            if not _static_verdicts(r)
            and (r.get("dynamic") or {}).get("status") == "ok"
        ]
        lines.append(
            f"- **static verdict where the dynamic mode abstains**: "
            f"{len(s_not_d)} — {s_not_d}"
        )
        lines.append(
            f"- dynamic runs where the static track abstains (the reachable-"
            f"region asymmetry's other side): {len(d_not_s)} — {d_not_s}"
        )
        kinds = Counter(k for r in rows if (k := _kind(r)) is not None)
        lines.append(
            "- unsupported kinds: "
            + (", ".join(f"{k}={v}" for k, v in kinds.most_common()) or "none")
        )
        lines.append("")

    block("Coverage corpus (tutorials + liger)", coverage)
    block("All corpora", all_rows)

    litmus = [r for r in all_rows if r["_corpus"] not in COVERAGE_CORPORA]
    proofs_mut = [
        r
        for r in litmus + coverage
        if r.get("mutation") and "error" not in r["mutation"]
    ]
    if proofs_mut:
        flipped = sum(
            1
            for r in proofs_mut
            if any(s == "races" for s in (r["mutation"].get("results") or {}).values())
        )
        lines.append(
            f"**Mutation-validated proofs** (all corpora with --mutate): "
            f"{flipped}/{len(proofs_mut)} flip to a race under at least one "
            "mutant."
        )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
    print(headline(d))
