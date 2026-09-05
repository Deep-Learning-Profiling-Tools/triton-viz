"""Row-by-row comparison of two evaluation datasets, with attribution.

The pinned rerun's step 7 (paper repo ``pre-submission/pinned-rerun.md``):
every row whose verdict changed between the previous pin and the new one
must be explained by one of the effects recorded in advance, and the
fence-order share must be separable from the ladder's (section 4c). This
tool is that diff, committed so the step is reproducible from the repo:

  * rows align by (corpus, name); a merged pinned file (rows carry
    ``corpus``) and a per-corpus runner file (the header carries it) both
    load; rows only on one side are listed, never silently dropped;
  * each row's SIGNATURE is (verdict, terminal, refusal kind): the kind is
    the verdict attributes' ``unsupported_kind`` when present, else the
    head of the static reason ("indirect-address: ..." -> indirect-address);
  * every changed row is classified: ``flip`` (race <-> race-free, the
    class that must be empty or explained one by one), ``downgrade``
    (decided -> undecided), ``upgrade`` (undecided -> decided),
    ``terminal`` (same verdict, another rung or terminal), ``reason``
    (same verdict and terminal, another refusal kind);
  * ``--legacy`` takes a third dataset: the changed rows rerun at the NEW
    commit with the memory-model switch off (``TRITON_VIZ_FENCE_ORDER=0``,
    section 4c item 2). A changed row whose legacy signature equals the
    base's is the fence order's; one equal to the new's is not (the
    ladder's or a non-gated change's); anything else is ``mixed``;
  * ``--names-dir`` writes the changed rows per corpus in the runner's
    ``--only-file`` format, which is how that legacy pass is spawned.

Usage:
    python -m evaluation.compare_runs BASE.jsonl NEW.jsonl [--legacy L.jsonl]
        [--corpus C ...] [--names-dir DIR] [--out REPORT.md] [--show-unchanged]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Key = tuple[str, str]  # (corpus, name)

UNDECIDED = ("abstain", "error", None)


@dataclass(frozen=True)
class Sig:
    verdict: str | None
    terminal: str | None
    kind: str | None

    def short(self) -> str:
        t = self.terminal or "-"
        return f"{t}" + (f" [{self.kind}]" if self.kind else "")


@dataclass
class Dataset:
    path: Path
    header: dict[str, Any]
    rows: dict[Key, dict[str, Any]]

    @property
    def label(self) -> str:
        h = self.header
        commit = h.get("pinned_commit") or h.get("commit") or "?"
        return f"{self.path.name} @ {commit}"

    def stamps(self) -> dict[str, Any]:
        """The provenance stamps, as the header says them and as the rows
        agree (a row set is reported as mixed when the rows disagree)."""
        h = self.header
        out: dict[str, Any] = {
            "commit": h.get("pinned_commit") or h.get("commit"),
            "ladder_level": h.get("ladder_level"),
            "fence_order": h.get("fence_order"),
            "rows": len(self.rows),
        }
        for key in ("ladder_level", "fence_order"):
            vals = Counter(r.get(key) for r in self.rows.values())
            if len(vals) > 1:
                out[key + "_rows"] = "MIXED " + ", ".join(
                    f"{v}:{n}" for v, n in sorted(vals.items(), key=str)
                )
            elif vals and out.get(key) is None:
                out[key] = next(iter(vals))
        return out


def load_dataset(path: Path, corpora: set[str] | None = None) -> Dataset:
    header: dict[str, Any] = {}
    rows: dict[Key, dict[str, Any]] = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("header"):
                header = r
                continue
            corpus = r.get("corpus") or header.get("corpus")
            if corpus is None or "name" not in r:
                raise ValueError(f"{path}: a row without corpus/name: {ln[:80]}")
            if corpora and corpus not in corpora:
                continue
            rows[(corpus, r["name"])] = r
    return Dataset(path, header, rows)


def signature(row: dict[str, Any]) -> Sig:
    if row.get("pinned_error"):
        return Sig("error", "timeout", None)
    verdict = row.get("verdict")
    terminal = row.get("terminal")
    kind = None
    if verdict in ("abstain", "error"):
        st = row.get("static") or {}
        va = st.get("verdict_attrs") or {}
        kind = va.get("unsupported_kind")
        if not kind or kind == "other":
            reason = st.get("reason") or row.get("harness_error") or ""
            head = reason.split(":", 1)[0].strip()
            kind = head if head and " " not in head and len(head) < 40 else kind
        if verdict == "error" and not kind:
            kind = terminal
    return Sig(verdict, terminal, kind)


def classify(a: Sig, b: Sig) -> str:
    if a == b:
        return "same"
    decided_a = a.verdict in ("race", "race-free")
    decided_b = b.verdict in ("race", "race-free")
    if decided_a and decided_b and a.verdict != b.verdict:
        return "flip"
    if decided_a and not decided_b:
        return "downgrade"
    if not decided_a and decided_b:
        return "upgrade"
    if a.verdict == b.verdict and a.terminal != b.terminal:
        return "terminal"
    if a.verdict != b.verdict:
        # abstain <-> error and the like: undecided either way
        return "terminal"
    return "reason"


CLASS_ORDER = ("flip", "downgrade", "upgrade", "terminal", "reason")


@dataclass
class Change:
    key: Key
    base: Sig
    new: Sig
    cls: str
    cause: str | None = None  # from --legacy


@dataclass
class Comparison:
    base: Dataset
    new: Dataset
    changes: list[Change] = field(default_factory=list)
    unchanged: int = 0
    only_base: list[Key] = field(default_factory=list)
    only_new: list[Key] = field(default_factory=list)
    legacy: Dataset | None = None
    legacy_missing: list[Key] = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter(c.cls for c in self.changes)

    def per_corpus(self) -> dict[str, Counter]:
        out: dict[str, Counter] = defaultdict(Counter)
        for c in self.changes:
            out[c.key[0]][c.cls] += 1
        for k in self.only_new:
            out[k[0]]["new"] += 1
        for k in self.only_base:
            out[k[0]]["gone"] += 1
        return out

    def transitions(self) -> Counter:
        return Counter((c.base.terminal, c.new.terminal) for c in self.changes)

    def causes(self) -> Counter:
        return Counter(c.cause for c in self.changes if c.cause)


def compare(base: Dataset, new: Dataset, legacy: Dataset | None = None) -> Comparison:
    cmp = Comparison(base, new, legacy=legacy)
    for key in sorted(set(base.rows) | set(new.rows)):
        if key not in base.rows:
            cmp.only_new.append(key)
            continue
        if key not in new.rows:
            cmp.only_base.append(key)
            continue
        a, b = signature(base.rows[key]), signature(new.rows[key])
        cls = classify(a, b)
        if cls == "same":
            cmp.unchanged += 1
            continue
        ch = Change(key, a, b, cls)
        if legacy is not None:
            if key in legacy.rows:
                lg = signature(legacy.rows[key])
                if lg == a:
                    ch.cause = "fence-order"
                elif lg == b:
                    ch.cause = "not-fence-order"
                else:
                    ch.cause = "mixed"
            else:
                ch.cause = "not-rerun"
                cmp.legacy_missing.append(key)
        cmp.changes.append(ch)
    return cmp


def write_names(cmp: Comparison, out_dir: Path) -> list[Path]:
    """The changed rows per corpus, one name per line, for the runner's
    ``--only-file`` (the legacy-order attribution pass)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by: dict[str, list[str]] = defaultdict(list)
    for c in cmp.changes:
        by[c.key[0]].append(c.key[1])
    written = []
    for corpus, names in sorted(by.items()):
        p = out_dir / f"{corpus}.txt"
        p.write_text(
            f"# changed rows of {corpus}: {cmp.base.label} -> {cmp.new.label}\n"
            + "\n".join(sorted(names))
            + "\n"
        )
        written.append(p)
    return written


def _stamp_table(datasets: list[tuple[str, Dataset]]) -> list[str]:
    lines = [
        "| dataset | file | commit | ladder | fence order | rows |",
        "|---|---|---|---|---|---|",
    ]
    for role, ds in datasets:
        s = ds.stamps()
        fence = s.get("fence_order_rows") or s.get("fence_order")
        ll = s.get("ladder_level_rows") or s.get("ladder_level")
        lines.append(
            f"| {role} | `{ds.path.name}` | {s['commit']} | {ll} | {fence} | {s['rows']} |"
        )
    return lines


def render_markdown(cmp: Comparison, show_unchanged: bool = False) -> str:
    counts = cmp.counts()
    out: list[str] = [
        f"# Row-by-row comparison: {cmp.base.label} -> {cmp.new.label}",
        "",
    ]
    ds = [("base", cmp.base), ("new", cmp.new)]
    if cmp.legacy is not None:
        ds.append(("legacy (switch off)", cmp.legacy))
    out += _stamp_table(ds)
    out += [
        "",
        f"Rows aligned by (corpus, name): {cmp.unchanged + len(cmp.changes)} "
        f"matched ({cmp.unchanged} unchanged, {len(cmp.changes)} changed), "
        f"{len(cmp.only_new)} only in new, {len(cmp.only_base)} only in base.",
        "",
        "| class | rows | meaning |",
        "|---|---:|---|",
        f"| flip | {counts['flip']} | race <-> race-free: explain one by one or chase |",
        f"| downgrade | {counts['downgrade']} | decided -> undecided |",
        f"| upgrade | {counts['upgrade']} | undecided -> decided |",
        f"| terminal | {counts['terminal']} | same verdict, another terminal or rung |",
        f"| reason | {counts['reason']} | same abstention, another refusal kind |",
    ]
    if cmp.legacy is not None:
        cs = cmp.causes()
        out += [
            "",
            "Attribution against the legacy-order rerun of the changed rows "
            "(same commit, `TRITON_VIZ_FENCE_ORDER=0`):",
            "",
            "| cause | rows |",
            "|---|---:|",
            f"| fence-order (legacy = base) | {cs['fence-order']} |",
            f"| not-fence-order (legacy = new: ladder or non-gated change) | {cs['not-fence-order']} |",
            f"| mixed (legacy = neither) | {cs['mixed']} |",
            f"| not-rerun (absent from the legacy file) | {cs['not-rerun']} |",
        ]
    pc = cmp.per_corpus()
    if pc:
        out += [
            "",
            "| corpus | flip | downgrade | upgrade | terminal | reason | new | gone |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for corpus in sorted(pc):
            c = pc[corpus]
            out.append(
                f"| {corpus} | {c['flip']} | {c['downgrade']} | {c['upgrade']} | "
                f"{c['terminal']} | {c['reason']} | {c['new']} | {c['gone']} |"
            )
    tr = cmp.transitions()
    if tr:
        out += ["", "| base terminal | new terminal | rows |", "|---|---|---:|"]
        for (a, b), n in sorted(tr.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            out.append(f"| {a} | {b} | {n} |")
    if cmp.changes:
        hdr = "| class | corpus | name | base | new |"
        sep = "|---|---|---|---|---|"
        if cmp.legacy is not None:
            hdr += " cause |"
            sep += "---|"
        out += ["", "## Changed rows", "", hdr, sep]
        order = {c: i for i, c in enumerate(CLASS_ORDER)}
        for ch in sorted(cmp.changes, key=lambda c: (order[c.cls], c.key)):
            line = (
                f"| {ch.cls} | {ch.key[0]} | {ch.key[1]} | {ch.base.verdict} / "
                f"{ch.base.short()} | {ch.new.verdict} / {ch.new.short()} |"
            )
            if cmp.legacy is not None:
                line += f" {ch.cause} |"
            out.append(line)
    if cmp.only_new:
        out += ["", "## Rows only in new", ""]
        out += [f"- {c}/{n}" for c, n in cmp.only_new]
    if cmp.only_base:
        out += ["", "## Rows only in base", ""]
        out += [f"- {c}/{n}" for c, n in cmp.only_base]
    if show_unchanged:
        out += ["", "## Unchanged rows", ""]
        keys = sorted(set(cmp.base.rows) & set(cmp.new.rows))
        changed = {c.key for c in cmp.changes}
        out += [
            f"- {c}/{n}: {signature(cmp.new.rows[(c, n)]).short()}"
            for c, n in keys
            if (c, n) not in changed
        ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="row-by-row comparison of two evaluation datasets"
    )
    ap.add_argument("base", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument(
        "--legacy",
        type=Path,
        help="the changed rows rerun at the NEW commit with the memory-model "
        "switch off (fence-order attribution)",
    )
    ap.add_argument("--corpus", action="append", help="restrict to these corpora")
    ap.add_argument(
        "--names-dir",
        type=Path,
        help="write the changed rows per corpus as runner --only-file lists",
    )
    ap.add_argument("--out", type=Path, help="write the markdown report here")
    ap.add_argument("--show-unchanged", action="store_true")
    ns = ap.parse_args(argv)
    corpora = set(ns.corpus) if ns.corpus else None
    base = load_dataset(ns.base, corpora)
    new = load_dataset(ns.new, corpora)
    legacy = load_dataset(ns.legacy, corpora) if ns.legacy else None
    cmp = compare(base, new, legacy)
    if ns.names_dir:
        for p in write_names(cmp, ns.names_dir):
            print(f"[compare] wrote {p}", file=sys.stderr)
    md = render_markdown(cmp, ns.show_unchanged)
    if ns.out:
        ns.out.write_text(md)
        print(f"[compare] report -> {ns.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
