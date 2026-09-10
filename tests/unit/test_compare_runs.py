"""Pins for the dataset comparison tool (``evaluation/compare_runs.py``,
the pinned rerun's step 7 and its fence-order attribution): alignment by
(corpus, name) across merged and per-corpus files, the change classes,
the legacy-order attribution, the --only-file lists, and the report."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import compare_runs as cr  # noqa: E402


def _row(name, verdict, terminal, reason=None, kind=None, corpus=None, **extra):
    d = {"name": name, "verdict": verdict, "terminal": terminal, "fence_order": True}
    if corpus:
        d["corpus"] = corpus
    st = {"status": "ok"}
    if reason:
        st["reason"] = reason
    if kind:
        st["verdict_attrs"] = {"unsupported_kind": kind}
    d["static"] = st
    d.update(extra)
    return d


def _write(path, header, rows):
    path.write_text("\n".join(json.dumps(x) for x in [header, *rows]) + "\n")
    return path


def _base_and_new(tmp_path):
    base = _write(
        tmp_path / "PINNED_old.jsonl",
        {
            "header": True,
            "pinned_commit": "old1",
            "ladder_level": "L0",
            "fence_order": False,
        },
        [
            _row("a", "race-free", "proved@T1", corpus="c"),
            _row("b", "race-free", "proved@T0", corpus="c"),
            _row(
                "c",
                "abstain",
                "unsupported",
                reason="indirect-address: line 3",
                corpus="c",
            ),
            _row("d", "abstain", "unsupported", kind="nested-loop", corpus="c"),
            _row("e", "race", "race-confirmed", corpus="c"),
            _row("f", "abstain", "unsupported", kind="control-flow", corpus="d"),
            _row("gone", "race-free", "proved@T1", corpus="d"),
        ],
    )
    # a per-corpus runner file: rows carry no corpus, the header does
    new_c = _write(
        tmp_path / "c_L2_pinned.jsonl",
        {
            "header": True,
            "corpus": "c",
            "commit": "new1",
            "ladder_level": "L2",
            "fence_order": True,
        },
        [
            _row("a", "race", "races-unclassified"),  # flip
            _row("b", "race-free", "proved@T0"),  # same
            _row("c", "race-free", "proved@enum"),  # upgrade
            _row("d", "abstain", "unsupported", kind="interpreter-error"),  # reason
            _row("e", "race", "race@interp"),  # terminal
            _row("newrow", "race-free", "proved@T1"),  # only in new
        ],
    )
    return base, new_c


def test_alignment_classes_and_transitions(tmp_path):
    base_p, new_p = _base_and_new(tmp_path)
    base = cr.load_dataset(base_p)
    new = cr.load_dataset(new_p)
    assert ("c", "a") in base.rows and ("c", "a") in new.rows
    cmp = cr.compare(base, new)
    by = {c.key[1]: c for c in cmp.changes}
    assert by["a"].cls == "flip"
    assert by["c"].cls == "upgrade" and by["c"].base.kind == "indirect-address"
    assert by["d"].cls == "reason" and by["d"].new.kind == "interpreter-error"
    assert by["e"].cls == "terminal"
    assert cmp.unchanged == 1 and cmp.only_new == [("c", "newrow")]
    assert cmp.only_base == [("d", "f"), ("d", "gone")]
    assert cmp.transitions()[("proved@T1", "races-unclassified")] == 1
    assert cmp.per_corpus()["c"]["flip"] == 1 and cmp.per_corpus()["d"]["gone"] == 2


def test_downgrade_and_pinned_error_rows():
    a = cr.signature(_row("x", "race-free", "proved@T1"))
    b = cr.signature(
        {"name": "x", "pinned_error": True, "harness_error": "exceeded 320s"}
    )
    assert b == cr.Sig("error", "timeout", None)
    assert cr.classify(a, b) == "downgrade"
    assert cr.classify(b, a) == "upgrade"
    assert cr.classify(a, a) == "same"


def test_legacy_attribution_and_names_dir(tmp_path):
    base_p, new_p = _base_and_new(tmp_path)
    legacy_p = _write(
        tmp_path / "c_L2_legacy.jsonl",
        {
            "header": True,
            "corpus": "c",
            "commit": "new1",
            "ladder_level": "L2",
            "fence_order": False,
        },
        [
            _row(
                "a", "race-free", "proved@T1", fence_order=False
            ),  # = base: fence order
            _row(
                "c", "race-free", "proved@enum", fence_order=False
            ),  # = new: not fence order
            _row(
                "d", "abstain", "unsupported", kind="solver", fence_order=False
            ),  # neither
        ],
    )
    cmp = cr.compare(
        cr.load_dataset(base_p), cr.load_dataset(new_p), cr.load_dataset(legacy_p)
    )
    by = {c.key[1]: c for c in cmp.changes}
    assert by["a"].cause == "fence-order"
    assert by["c"].cause == "not-fence-order"
    assert by["d"].cause == "mixed"
    assert by["e"].cause == "not-rerun" and cmp.legacy_missing == [("c", "e")]
    files = cr.write_names(cmp, tmp_path / "names")
    assert [p.name for p in files] == ["c.txt"]
    names = [
        ln for ln in files[0].read_text().splitlines() if ln and not ln.startswith("#")
    ]
    assert names == ["a", "c", "d", "e"]
    md = cr.render_markdown(cmp)
    assert "| fence-order (legacy = base) | 1 |" in md
    assert (
        "| flip | c | a | race-free / proved@T1 | race / races-unclassified | fence-order |"
        in md
    )
    assert "| legacy (switch off) |" in md and "| L2 | False | 3 |" in md


def test_cli_writes_report_and_restricts_corpora(tmp_path):
    base_p, new_p = _base_and_new(tmp_path)
    out = tmp_path / "report.md"
    rc = cr.main([str(base_p), str(new_p), "--corpus", "c", "--out", str(out)])
    assert rc == 0
    md = out.read_text()
    assert "1 only in new, 0 only in base" in md  # corpus d filtered out
    assert "| flip | 1 |" in md and "| upgrade | 1 |" in md
    assert "## Rows only in new" in md and "- c/newrow" in md
