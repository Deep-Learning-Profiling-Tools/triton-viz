"""tilelens.ir._mlir_walk: the bindings + text alignment layer under the TTIR reader.

Goldens live in tests/golden/ir/ttir/ (printed by Triton 3.6, read under every
release) and tests/golden/ir/ttir_<release>/ (a later release's own printing,
which shadows the base golden of the same name under that release; see
_goldens.py; provenance: tests/golden/ir/generate_ttir.py). Their pinned counts
and text-only attribute census are pinned per printing release, in
tests/golden/ir/expected.json (3.6) and expected_<release>.json. Regenerate the
pins of the installed release's own goldens after an intended change with
``TILELENS_IR_REGEN=1 pytest tests/unit/ir/test_mlir_walk.py -k regen``.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from tilelens.ir import _mlir_walk as W

from . import _goldens as G

REPO = G.REPO
GOLDEN = G.GOLDEN
TTIR = GOLDEN / "ttir"
# fail closed by design: the generic op form the printer emits only for
# modules that fail to verify
MISALIGNED = {"crafted_generic_form.ttir"}
# name -> the golden the installed release reads (its own printing first)
GOLDENS = G.texts("ttir")
FILES = sorted(GOLDENS)
ALIGNED = [f for f in FILES if f not in MISALIGNED]
BASE_FILES = sorted(p.name for p in TTIR.glob("*.ttir"))
# Every pinned text the installed release parses, checked against the pins
# of the release that printed it: the base goldens (but those its parser
# rejects, BASE_UNPARSABLE) and its own; the base ones keep their names.
_REFUSED_HERE = G.BASE_UNPARSABLE.get(G.RELEASE, {})
PINNED: dict[str, Path] = {
    **{
        n: TTIR / n
        for n in BASE_FILES
        if n not in MISALIGNED and n not in _REFUSED_HERE
    },
    **{
        f"{p.parent.name}/{n}": p
        for n, p in GOLDENS.items()
        if G.printed_by(p) != G.BASE_RELEASE and n not in MISALIGNED
    },
}


def _text(name: str) -> str:
    return GOLDENS[name].read_text(encoding="utf-8")


def _printed_by(name: str) -> str:
    return G.printed_by(GOLDENS[name])


@pytest.fixture(autouse=True)
def _fresh_cache():
    W._CACHE.clear()
    yield
    W._CACHE.clear()


# ─────────────────────────── goldens ───────────────────────────


def _text_only(table: W.Printer) -> dict[str, tuple[str, ...]]:
    """The text-only attributes of ``table``'s release: the text is their
    only source, so golden pins guard their extraction (spike condition 5)."""
    bound = {(o, a) for o, g in table.bind_attrs.items() for _, a in g}
    out = {
        op: tuple(k for k in keys if (op, k) not in bound)
        for op, keys in table.needed.items()
        if op not in ("cf.br", "cf.cond_br", "tt.func", "tt.call")
    }
    out["arith.constant"] = ("value", "splat")
    out["tt.descriptor_reduce"] = ("kind",)
    return out


def _census(m: W.Module) -> list[list]:
    text_only = _text_only(W.PRINTERS[m.release])
    c: collections.Counter[str] = collections.Counter()
    for op in m.ops:
        for key in text_only.get(op.name, ()):
            if key in op.attrs:
                c[f"{op.name}.{key}={op.attrs[key]!r}"] += 1
        if not op.results and not op.implicit and op.loc is not None:
            c["zero-result op with a text loc"] += 1
        if "successors" in op.attrs:
            c[f"{op.name} -> {len(op.attrs['successors'])} successors"] += 1
    return sorted([k, n] for k, n in c.items())


def _pins(m: W.Module) -> dict:
    return {
        "stats": dict(m.stats),
        "census": _census(m),
        "funcs": [f.sym_name for f in m.funcs],
    }


@pytest.mark.skipif(
    not os.environ.get("TILELENS_IR_REGEN"),
    reason="set TILELENS_IR_REGEN=1 to rewrite pins",
)
def test_regen_pins():
    """The pins of the goldens the installed release printed (its own
    directory; the base directory under the base release)."""
    own = [f for f in ALIGNED if _printed_by(f) == G.RELEASE]
    assert own, f"Triton {G.RELEASE} printed no golden: run generate_ttir.py first"
    G.pins_path(G.RELEASE).write_text(
        json.dumps(
            {f: _pins(W._walk(_text(f))) for f in own}, indent=1, ensure_ascii=False
        )
        + "\n"
    )


def _generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_generate_ttir", GOLDEN / "generate_ttir.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _real_compiles_available() -> bool:
    # Triton imported under TRITON_INTERPRET=1 builds its own standard library
    # as InterpretedFunctions, so nothing can compile for real in-process.
    import triton.language.standard as tl_standard
    from triton.runtime.jit import JITFunction

    return isinstance(tl_standard.cdiv, JITFunction)


@pytest.mark.skipif(
    not _real_compiles_available(),
    reason="Triton was imported under TRITON_INTERPRET=1: nothing compiles in-process",
)
def test_the_kernel_goldens_regenerate_byte_for_byte(monkeypatch, tmp_path):
    """generate_ttir.py prints, under the installed release, exactly the
    goldens it wrote into that release's directory (its paths made
    portable): its locs name the kernels' lines, so a line added above them
    shows here, not as goldens that no longer regenerate."""
    # tests/unit/test_multithreading.py sets TRITON_INTERPRET=1 at import
    # time, under which @triton.jit builds InterpretedFunctions: pin the knob
    # off while the generator's kernels are built, as the compile tests do.
    from triton import knobs

    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    missing = object()
    previous = knobs.runtime.__dict__.get("interpret", missing)
    knobs.runtime.__dict__["interpret"] = False
    try:
        gen = _generator()
    finally:
        if previous is missing:
            knobs.runtime.__dict__.pop("interpret", None)
        else:
            knobs.runtime.__dict__["interpret"] = previous
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    out = Path(gen.out_dir(G.RELEASE))
    todo = gen.jobs(G.RELEASE)
    assert todo and all((out / f"{name}.ttir").is_file() for name in todo)
    # Its compile takes ~25 s under 3.8's pipeline; a line added above it
    # moves the locs of kernel_dot_scaled, below it, too.
    del todo["kernel_deep_chain"]
    for name, spec in todo.items():
        want = (out / f"{name}.ttir").read_text(encoding="utf-8")
        assert gen.portable(gen.ttir(spec)) == want, name


@pytest.mark.parametrize("label", PINNED)
def test_golden_aligns_with_pinned_counts(label):
    path = PINNED[label]
    name = path.name
    pinned = G.pins(G.printed_by(path))
    m = W._walk(path.read_text(encoding="utf-8"))
    assert name in pinned, "new golden: regenerate the pins"
    assert _pins(m) == pinned[name]
    # the counters agree with the records
    assert m.stats["ops"] == len(m.ops) and m.stats["values"] == len(m.values)
    assert m.stats["ssa_edges"] == sum(len(op.operands) for op in m.ops)
    assert m.stats["implicit_ops"] == sum(op.implicit for op in m.ops)


def test_every_golden_is_pinned():
    # every release's own goldens are pinned by that release (checked here
    # under any release: no walk), and each shadows a base golden
    for release in G.releases_with_goldens("ttir"):
        own = sorted(p.name for p in G.own_dir("ttir", release).glob("*.ttir"))
        assert sorted(G.pins(release)) == [f for f in own if f not in MISALIGNED]
        assert set(own) <= set(BASE_FILES), release
    # a base golden a release's parser rejects is one it prints itself
    for release, refused in G.BASE_UNPARSABLE.items():
        own = {p.name for p in G.own_dir("ttir", release).glob("*.ttir")}
        assert set(refused) <= own & set(BASE_FILES), release
    # the pinned corpus: #361 goldens, spike corpus, review corpora, kernels, crafted
    prefixes = collections.Counter(f.split("_", 1)[0] for f in BASE_FILES)
    assert prefixes == {
        "golden": 35,
        "spike": 15,
        "adv": 12,
        "nat": 7,
        "kernel": 5,
        "crafted": 12,
    }


def test_base_goldens_a_release_respells_do_not_parse_under_it():
    """A base golden in BASE_UNPARSABLE is refused by the installed
    release's parser (never misread), and the release reads its own
    printing instead."""
    for name, fragment in G.BASE_UNPARSABLE.get(G.RELEASE, {}).items():
        with pytest.raises(W.ModuleParseError) as e:
            W._walk((TTIR / name).read_text(encoding="utf-8"))
        assert fragment in e.value.diagnostic and e.value.line_no is not None
        assert _printed_by(name) == G.RELEASE, name


def test_corpus_totals():
    totals: collections.Counter[str] = collections.Counter()
    for name in ALIGNED:
        totals.update(W._walk(_text(name)).stats)
    assert totals["ops"] > 3500 and totals["ssa_edges"] > 5000
    assert (
        totals["implicit_ops"] >= 20
        and totals["cf_edges"] >= 30
        and totals["pred_checks"] >= 20
    )


@pytest.mark.parametrize("name", ["crafted_generic_form.ttir"])
def test_generic_form_fails_closed(name):
    with pytest.raises(W.MisalignedModule) as e:
        W._walk(_text(name))
    # the generic form's integer enums never satisfy the text-only extractors
    assert any("arith.cmpi" in p for p in e.value.problems)
    assert any("tt.atomic_rmw" in p for p in e.value.problems)
    assert e.value.line_no == 3


def test_structure_records_are_consistent():
    for name in ALIGNED:
        m = W._walk(_text(name))
        assert m.ops[0].name == "builtin.module" and m.ops[0].path == ()
        assert [op.index for op in m.ops] == list(range(len(m.ops)))
        assert [b.index for b in m.blocks] == list(range(len(m.blocks)))
        assert [v.index for v in m.values] == list(range(len(m.values)))
        for op in m.ops:
            for v in op.results:
                assert m.values[v].op == op.index
            for k, blocks in enumerate(op.regions):
                for pos, b in enumerate(blocks):
                    blk = m.blocks[b]
                    assert (blk.op, blk.region, blk.position) == (op.index, k, pos)
                    for i in blk.ops:
                        assert m.ops[i].path == op.path + (b,)
            if op.path:
                assert m.blocks[op.block].ops[op.position] == op.index
        for v in m.values:
            assert (v.op is None) != (v.block is None)
        # pre-order: an op's regions come after it, its uses are in-module values
        for op in m.ops:
            assert all(0 <= v < len(m.values) for v in op.operands)
            assert op.operand_types == tuple(m.values[v].type for v in op.operands)


# ─────────────────────────── an independent attribute extractor ───────────────────────────


def _must(fn, pattern: str, s: str) -> re.Match:
    m = fn(pattern, s)
    assert m is not None, (pattern, s)
    return m


def _indep(name: str, line: str) -> dict:
    """Deliberately naive per-op regexes over the raw header line (strings kept):
    a second extractor for the text-only attributes (the review's differential)."""
    body = line.split(" = ", 1)[1] if re.match(r"\s*%[^=]*= ", line) else line.strip()
    out: dict = {}
    if name == "arith.cmpi" or name == "arith.cmpf":
        out["predicate"] = _must(re.match, r"arith\.cmp[if] (\w+),", body).group(1)
    elif name in ("tt.get_program_id", "tt.get_num_programs"):
        out["axis"] = "xyz".index(
            _must(re.match, r"tt\.get_\w+ ([xyz]) ", body).group(1)
        )
    elif name == "tt.make_range":
        out["start"] = int(_must(re.search, r"start = (-?\d+) : i32", body).group(1))
        out["end"] = int(_must(re.search, r"end = (-?\d+) : i32", body).group(1))
    elif name == "tt.atomic_rmw":
        out.update(
            zip(
                ("rmw_op", "sem", "scope"),
                _must(
                    re.match, r"tt\.atomic_rmw (\w+), (\w+), (\w+), %", body
                ).groups(),
            )
        )
    elif name == "tt.atomic_cas":
        out.update(
            zip(
                ("sem", "scope"),
                _must(re.match, r"tt\.atomic_cas (\w+), (\w+), %", body).groups(),
            )
        )
    elif name in ("tt.expand_dims", "tt.reduce", "tt.scan"):
        out["axis"] = int(_must(re.search, r"axis = (-?\d+) : i32", body).group(1))
    elif name == "tt.trans":
        out["order"] = tuple(
            int(x)
            for x in _must(re.search, r"order = array<i32: ([\d, ]+)>", body)
            .group(1)
            .split(",")
        )
    elif name == "tt.dot":
        m = re.search(r", inputPrecision = (\w+) :", body)
        out["inputPrecision"] = m.group(1) if m else "ieee"
    elif name == "tt.reshape":
        out["allow_reorder"] = " allow_reorder " in body
    elif name == "scf.for":
        out["unsignedCmp"] = body.startswith("scf.for unsigned ")
    elif name == "arith.constant":
        c = re.match(
            r"arith\.constant (?:\{[^}]*\} )?(dense<)?(-?\d+|true|false)>? : ",
            body + " : ",
        )
        if c and c.group(2) in ("true", "false"):
            out["value"] = c.group(2) == "true"
        elif c:
            out["value"] = int(c.group(2))
    return out


def test_independent_extractor_agrees_on_every_golden():
    checked = 0
    for name in ALIGNED:
        lines = _text(name).splitlines()
        for op in W._walk(_text(name)).ops:
            if op.implicit or op.line_no is None:
                continue
            for k, v in _indep(op.name, lines[op.line_no - 1]).items():
                assert op.attrs.get(k) == v, (name, op.line_no, op.name, k)
                checked += 1
    assert checked >= 700


# ─────────────────────────── mutation sensitivity ───────────────────────────
# The bindings parse the original text while the text layer reads a mutated
# copy: a text layer that mis-reads the module must never align silently.

_OPLINE = re.compile(
    r"^\s+(?:%[-\w.$]+(?::\d+)?(?:, %[-\w.$]+)* = )?[a-z_]+\.[\w.]+ .*loc\(.*\)\s*$"
)
_TERMINATOR = re.compile(
    r"tt\.return|scf\.yield|cf\.br|cf\.cond_br|scf\.condition|reduce\.return|scan\.return"
)


def _mutants(text: str, rng: random.Random) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    idx = [
        i
        for i, ln in enumerate(lines)
        if _OPLINE.match(ln) and not ln.rstrip().endswith("{")
    ]
    out = []
    pairs = [
        (i, j)
        for i, j in zip(idx, idx[1:])
        if j == i + 1
        and not _TERMINATOR.search(lines[j])
        and lines[i].split(" loc(")[0] != lines[j].split(" loc(")[0]
    ]
    same = [
        (i, j)
        for i, j in pairs
        if lines[i].split("=")[-1].split()[0] == lines[j].split("=")[-1].split()[0]
    ]
    for label, pool in (("swap-same-name", same), ("swap-any", pairs)):
        for i, j in rng.sample(pool, min(2, len(pool))):
            m = lines[:]
            m[i], m[j] = m[j], m[i]
            out.append((label, f"lines {i + 1}<->{j + 1}", "\n".join(m)))
    for i in rng.sample(idx, min(2, len(idx))):
        out.append(
            ("drop-line", f"line {i + 1}", "\n".join(lines[:i] + lines[i + 1 :]))
        )
    defs = re.findall(r"^\s+(%[-\w.$]+) = ", text, re.M)
    use_lines = [i for i in idx if re.search(r"= [a-z_.]+ .*%", lines[i])]
    for i in rng.sample(use_lines, min(2, len(use_lines))):
        lhs, rhs = lines[i].split(" = ", 1)
        uses = re.findall(r"%[-\w.$]+", rhs)
        cands = [d for d in defs if uses and d != uses[0]]
        if not cands:
            continue
        m = lines[:]
        m[i] = (
            lhs
            + " = "
            + re.sub(
                re.escape(uses[0]) + r"(?![-\w.$])", rng.choice(cands), rhs, count=1
            )
        )
        out.append(("repoint-use", f"line {i + 1}", "\n".join(m)))
    edits = (
        (
            "typed-attr",
            r"(tt\.make_range \{end = )(\d+)",
            lambda g: f"{g.group(1)}{int(g.group(2)) * 2}",
        ),
        (
            "vocab-predicate",
            r"(arith\.cmpi )(\w+)(,)",
            lambda g: f"{g.group(1)}{g.group(2)}x{g.group(3)}",
        ),
        (
            "vocab-sem",
            r"(tt\.atomic_\w+ (?:\w+, )?)(relaxed|acquire|release|acq_rel)(,)",
            lambda g: f"{g.group(1)}seq_cst,",
        ),
        (
            "vocab-scope",
            r"(tt\.atomic_\w+ (?:\w+, )?\w+, )(gpu|cta|sys)(,)",
            lambda g: f"{g.group(1)}device,",
        ),
        ("vocab-rmw", r"(tt\.atomic_rmw )(\w+)(,)", lambda g: f"{g.group(1)}nand,"),
        (
            "vocab-axis",
            r"(tt\.get_program_id )([xyz])( :)",
            lambda g: f"{g.group(1)}w :",
        ),
        ("vocab-precision", r"(inputPrecision = )(\w+)", lambda g: f"{g.group(1)}fp8"),
        (
            "pred-comment",
            r"(// pred: \^bb)(\d+)",
            lambda g: f"{g.group(1)}{int(g.group(2)) + 7}",
        ),
        # a NEEDED attribute the text no longer prints
        (
            "drop-needed",
            r"(tt\.make_range \{end = \d+ : i32), start = \d+ : i32\}",
            lambda g: g.group(1) + "}",
        ),
        (
            "drop-needed",
            r"(tt\.trans %[-\w.$]+) \{order = array<i32: [\d, ]+>\}",
            lambda g: g.group(1),
        ),
    )
    for label, pat, rep in edits:
        for i, ln in enumerate(lines):
            if re.search(pat, ln):
                m = lines[:]
                m[i] = re.sub(pat, rep, ln, count=1)
                out.append((label, f"line {i + 1}", "\n".join(m)))
                break
    # a result-bearing op's loc: the bindings' result locs are its second
    # source (Op.loc / callers / loc_name, and Value.name of the results)
    aliases = dict(re.findall(r"^(#loc\d*) = loc\((.*)\)$", text, re.M))
    resolve = W._LocParser(aliases, "")
    trees = {a: resolve.parse(f"loc({a})") for a in aliases}
    sited: list[tuple[int, re.Match[str]]] = []
    for i in idx:
        lm = re.search(r" loc\((#loc\d*)\)$", lines[i].rstrip())
        if lm and re.match(r"\s+%", lines[i]) and lm.group(1) in trees:
            sited.append((i, lm))
    for i, lm in rng.sample(sited, min(2, len(sited))):
        others = sorted(a for a in trees if trees[a] != trees[lm.group(1)])
        ln = lines[i].rstrip()
        mt = lines[:]
        mt[i] = ln[: lm.start(1)] + rng.choice(others) + ln[lm.end(1) :]
        out.append(("repoint-loc", f"line {i + 1}", "\n".join(mt)))
    def_line = {
        a: k for k, ln in enumerate(lines) for a in re.findall(r"^(#loc\d*) =", ln)
    }
    for i, lm in rng.sample(sited, min(2, len(sited))):
        k = def_line[lm.group(1)]
        garbled = re.sub(
            r'(":)(\d+)(:\d+\)\s*)$',
            lambda g: f"{g.group(1)}{int(g.group(2)) + 1000}{g.group(3)}",
            lines[k],
        )
        if garbled == lines[k]:  # a name loc: rename it
            garbled = re.sub(r'^(#loc\d* = loc\(")', r"\1garbled_", lines[k])
        if garbled == lines[k]:  # loc(unknown), callsite(...), fused[...]
            continue
        mt = lines[:]
        mt[k] = garbled
        out.append(
            ("garble-loc-alias", f"line {k + 1} (used on line {i + 1})", "\n".join(mt))
        )
    return out


# the problem each targeted mutation class must be caught by (the structural
# classes may trip any of several checks)
_REASON = {
    "typed-attr": "make_range",
    "vocab-predicate": "closed vocabulary",
    "vocab-sem": "closed vocabulary",
    "vocab-scope": "closed vocabulary",
    "vocab-rmw": "closed vocabulary",
    "vocab-axis": "closed vocabulary",
    "vocab-precision": "closed vocabulary",
    "pred-comment": "printer preds",
    "drop-needed": "not recovered",
    "repoint-loc": "loc differs",
    "garble-loc-alias": "loc differs",
}


def test_mutations_are_detected():
    rng = random.Random(0)
    by_class: collections.Counter[str] = collections.Counter()
    missed = []
    for name in ALIGNED:
        text = _text(name)
        if name == "kernel_deep_chain.ttir":
            continue  # 1200-op chain: slow and adds nothing here
        for cls, desc, mt in _mutants(text, rng):
            by_class[cls] += 1
            try:
                W._walk(text, scan_text=mt)
            except W.MisalignedModule as e:
                assert e.problems and isinstance(e.line_no, (int, type(None)))
                reason = _REASON.get(cls)
                if reason is None or any(reason in p for p in e.problems):
                    continue
                missed.append(f"{name} {cls} {desc}: caught only by {e.problems[:2]}")
                continue
            missed.append(f"{name} {cls} {desc}")
    assert not missed
    for cls in (
        "swap-same-name",
        "swap-any",
        "drop-line",
        "repoint-use",
        "typed-attr",
        "vocab-predicate",
        "drop-needed",
        "repoint-loc",
        "garble-loc-alias",
    ):
        assert by_class[cls] >= 20, by_class
    for cls in (
        "vocab-sem",
        "vocab-scope",
        "vocab-rmw",
        "vocab-axis",
        "vocab-precision",
        "pred-comment",
    ):
        assert by_class[cls] >= 2, by_class


def test_mutation_reports_the_mutated_line():
    text = _text("golden_add_sm80.ttir")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if "arith.cmpi slt" in ln)
    lines[i] = lines[i].replace("arith.cmpi slt", "arith.cmpi lt")
    with pytest.raises(W.MisalignedModule) as e:
        W._walk(text, scan_text="\n".join(lines))
    assert e.value.line_no == i + 1
    assert "closed vocabulary" in e.value.problems[0]


def test_in_vocabulary_predicate_swap_is_a_single_source_attribute():
    # documented blind spot (spike: 0/32): an in-vocabulary spelling has no
    # second source; only the golden census above guards its extraction
    text = _text("golden_add_sm80.ttir")
    m = W._walk(text, scan_text=text.replace("arith.cmpi slt", "arith.cmpi sge", 1))
    assert "sge" in [op.attrs.get("predicate") for op in m.ops]


@pytest.mark.parametrize(
    "name, old, new, match",
    [
        # a zero-result op's loc has no second source: a garbled trailer must
        # not read as "no loc" while the rest of the module prints locs
        ("golden_add_sm80.ttir", "tt.return loc(", "tt.return oc(", "prints no loc"),
        ("golden_add_sm80.ttir", "} loc(#loc)", "} loc#(#loc)", "prints no loc"),
        # a garbled value must not pass as a recovered attribute
        ("golden_add_sm80.ttir", "start = 0 : i32", "start = 0 : im32", "is not a int"),
        (
            "adv_views.ttir",
            "order = array<i32: 1, 0>",
            "order = array<i32: 1, 0>x",
            "is not a tuple",
        ),
        # a garbled key must not fall back to the elided default (ieee)
        (
            "golden_matmul_s1_sm80.ttir",
            "inputPrecision = tf32",
            "inputPrecision4 = tf32",
            "tt.dot assignments",
        ),
    ],
)
def test_garbled_text_is_not_misread(name, old, new, match):
    text = _text(name)
    assert old in text
    with pytest.raises(W.MisalignedModule, match=match):
        W._walk(text, scan_text=text.replace(old, new, 1))


# Single-source fields: the text is their only source, so a text layer that
# misreads them still aligns (the golden census and the independent extractor
# guard their extraction instead). Every other field of an aligned reading
# must equal the baseline reading.
_SINGLE_SOURCE_ATTRS = frozenset(
    {
        # constant values (beyond the signed-range check)
        ("arith.constant", "value"),
        ("arith.constant", "literal"),
        ("arith.constant", "splat"),
        # in-vocabulary enum swaps
        ("arith.cmpi", "predicate"),
        ("arith.cmpf", "predicate"),
        ("tt.atomic_rmw", "rmw_op"),
        ("tt.atomic_rmw", "sem"),
        ("tt.atomic_rmw", "scope"),
        ("tt.atomic_cas", "sem"),
        ("tt.atomic_cas", "scope"),
        ("tt.get_program_id", "axis"),
        ("tt.get_num_programs", "axis"),
        ("tt.descriptor_reduce", "kind"),
        # a deleted clause reads as the printer-elided default
        ("tt.dot", "inputPrecision"),
        ("tt.dot", "maxNumImpreciseAcc"),
        ("tt.reshape", "allow_reorder"),
        ("scf.for", "unsignedCmp"),
        # no getter and no type relation
        ("tt.elementwise_inline_asm", "packed_element"),
    }
)
_NO_ATTRS = W._FrozenMap({})


def _op_shape(op: W.Op) -> W.Op:
    """The op without the fields _silent_diffs compares on its own."""
    return dataclasses.replace(
        op,
        attrs=_NO_ATTRS,
        line_no=None,
        end_line=None,
        loc=None,
        callers=(),
        loc_name=None,
    )


def _has_pred_comments(text: str) -> bool:
    return re.search(r"^\s*\^.*//", text, re.M) is not None


def _silent_diffs(base: W.Module, got: W.Module, pred_comments: bool) -> list[str]:
    """What an aligned reading ``got`` reads differently from ``base``, minus
    the single-source fields: line numbers, zero-result op locs, entry-block
    labels, func argument attrs, discardable (non-NEEDED) attrs, the
    attributes in _SINGLE_SOURCE_ATTRS, the order of a cf.cond_br's
    successors, and any cf successor when the text prints no pred comments."""
    shape = lambda m: (len(m.ops), len(m.blocks), len(m.values), len(m.funcs))  # noqa: E731
    if shape(base) != shape(got):
        return [f"record counts {shape(base)} -> {shape(got)}"]
    out = [f"value {a.index}" for a, b in zip(base.values, got.values) if a != b]
    for oa, ob in zip(base.ops, got.ops):
        where = f"op {oa.index} {oa.name} (line {oa.line_no})"
        # a zero-result op's loc has no second source
        site = (oa.loc, oa.callers, oa.loc_name)
        if (oa.results or oa.implicit) and site != (ob.loc, ob.callers, ob.loc_name):
            out.append(f"{where}: loc {oa.loc} -> {ob.loc}")
        if _op_shape(oa) != _op_shape(ob):
            out.append(f"{where}: structure")
        for k in sorted(set(oa.attrs) | set(ob.attrs)):
            va, vb = oa.attrs.get(k), ob.attrs.get(k)
            if va == vb and type(va) is type(vb):
                continue
            if (
                k not in W.PRINTERS[base.release].needed.get(oa.name, ())
                or (oa.name, k) in _SINGLE_SOURCE_ATTRS
            ):
                continue
            if k == "successors" and (
                not pred_comments
                or (oa.name == "cf.cond_br" and sorted(va or ()) == sorted(vb or ()))
            ):
                continue
            out.append(f"{where}: {k} {va!r} -> {vb!r}")
    for ba, bb in zip(base.blocks, got.blocks):
        if ba.position == 0:  # an entry block's label is never referenced
            ba, bb = (
                dataclasses.replace(ba, label=None),
                dataclasses.replace(bb, label=None),
            )
        if ba != bb:
            out.append(f"block {ba.index} ({ba.label})")
    for fa, fb in zip(base.funcs, got.funcs):
        strip = lambda f: dataclasses.replace(  # noqa: E731
            f, args=tuple(dataclasses.replace(x, attrs=_NO_ATTRS) for x in f.args)
        )
        if strip(fa) != strip(fb):
            out.append(f"func {fa.sym_name}")
    return out


def test_fuzzed_text_layer_fails_closed():
    """Random character / line edits to the text-layer input: every outcome is
    MisalignedModule, or an aligned Module that reads like the baseline in
    every field with a second source; never another exception."""
    rng = random.Random(1)
    pool = list('%^{}()<>[],:="#@ \\x0123456789abcdefgilnorstxyzE.-_/') + [
        "loc(",
        "dense<",
        "->",
        "\n",
        "//",
    ]
    names = [n for n in ALIGNED if n != "kernel_deep_chain.ttir"]
    outcomes: collections.Counter[str] = collections.Counter()
    baseline: dict[str, W.Module] = {}
    silent = []
    for _ in range(400):
        name = rng.choice(names)
        text = s = _text(name)
        for _ in range(rng.randint(1, 3)):
            k = rng.randrange(len(s))
            r = rng.random()
            if r < 0.4:
                s = s[:k] + s[k + 1 :]
            elif r < 0.8:
                s = s[:k] + rng.choice(pool) + s[k:]
            else:
                lines = s.split("\n")
                i = rng.randrange(len(lines))
                s = "\n".join(lines[:i] + [lines[i]] + lines[i:])
        try:
            got = W._walk(text, scan_text=s)
        except W.MisalignedModule:
            outcomes["misaligned"] += 1
            continue
        outcomes["aligned"] += 1
        if name not in baseline:
            baseline[name] = W._walk(text)
        diffs = _silent_diffs(baseline[name], got, _has_pred_comments(text))
        if diffs:
            silent.append((name, diffs[:3]))
    assert not silent
    assert outcomes["misaligned"] > 250 and outcomes["aligned"] > 50


def test_silent_diff_sees_what_the_cross_checks_guard():
    # the fuzz test's comparison flags a result op's loc, a NEEDED attribute
    # with a second source, and a cf.br retarget under pred comments
    text = _text("golden_nested_guard_merge_sm80.ttir")
    base = W._walk(text)
    load = next(op for op in base.ops if op.name == "tt.load")
    moved = dataclasses.replace(load, loc=W.SourceLoc("elsewhere.py", 1, 1))
    ops = list(base.ops)
    ops[load.index] = moved
    assert _silent_diffs(base, dataclasses.replace(base, ops=tuple(ops)), True)
    rng_op = next(op for op in base.ops if op.name == "tt.make_range")
    ops = list(base.ops)
    ops[rng_op.index] = dataclasses.replace(
        rng_op, attrs=W._FrozenMap({**rng_op.attrs, "start": 1})
    )
    assert _silent_diffs(base, dataclasses.replace(base, ops=tuple(ops)), True)
    (br,) = _ops(base, "cf.br")
    ops = list(base.ops)
    ops[br.index] = dataclasses.replace(
        br, attrs=W._FrozenMap({"successors": (br.attrs["successors"][0] - 1,)})
    )
    changed = dataclasses.replace(base, ops=tuple(ops))
    assert _silent_diffs(base, changed, True)
    assert not _silent_diffs(base, changed, False)


def test_pred_comments_are_checked_when_present():
    text = _text("golden_nested_guard_merge_sm80.ttir")
    assert W._walk(text).stats["pred_checks"] == 5
    stripped = re.sub(r"[ \t]*//[^\n]*", "", text)
    assert (
        W._walk(stripped).stats["pred_checks"] == 0
    )  # no comments: the check is skipped
    # one comment missing while the printer's others are present: misaligned
    one_gone = re.sub(r"(\^bb3:)\s*//[^\n]*", r"\1", text, count=1)
    with pytest.raises(W.MisalignedModule, match="no predecessor comment"):
        W._walk(text, scan_text=one_gone)
    with pytest.raises(W.MisalignedModule, match="printer preds"):
        W._walk(
            text,
            scan_text=text.replace("// 2 preds: ^bb3, ^bb4", "// 2 preds: ^bb3, ^bb3"),
        )


def test_successor_arity_is_checked():
    text = _text("golden_nested_guard_merge_sm80.ttir")
    # drop a successor's operand group in the text layer only
    bad = re.sub(r"(cf\.br \^bb5)\(%[-\w.$]+ : i32\)", r"\1", text, count=1)
    assert bad != text
    with pytest.raises(W.MisalignedModule):
        W._walk(text, scan_text=bad)


def test_successor_operand_groups_are_checked():
    # an operand moved from one successor group to the other keeps the total
    # count and the SSA order: only the per-successor arity sees it
    text = _text("crafted_same_dest.ttir")
    old = "cf.cond_br %c, ^bb1(%a : i32), ^bb1(%b : i32)"
    assert old in text
    moved = text.replace(old, "cf.cond_br %c, ^bb1(%a, %b : i32, i32), ^bb1")
    with pytest.raises(W.MisalignedModule, match="successor operands"):
        W._walk(text, scan_text=moved)


# ─────────────────────────── review fixes, one by one ───────────────────────────


def _ops(m: W.Module, name: str) -> list[W.Op]:
    return [op for op in m.ops if op.name == name]


def test_unicode_source_path():
    m = W._walk(_text("nat_k_uni.ttir"))
    files = {op.loc.file for op in m.ops if op.loc is not None}
    assert files == {"内核/k_uni.py"}


def test_unicode_parameter_names_come_from_namelocs():
    m = W._walk(_text("nat_uni_params.ttir"))
    (f,) = m.funcs
    # the printed SSA names are sanitised (%_CF80_ptr, %_E695B0_n)
    assert [(a.index, a.name, a.type) for a in f.args] == [
        (0, "π_ptr", "!tt.ptr<f32>"),
        (1, "数_n", "i32"),
    ]
    assert [m.values[a.value].name for a in f.args] == ["π_ptr", "数_n"]
    assert m.blocks[m.ops[f.op].regions[0][0]].arg_names == ("π_ptr", "数_n")


def test_unicode_strings_cross_checked_against_bindings():
    m = W._walk(_text("kernel_unicode_msgs.ttir"))
    (a,) = _ops(m, "tt.assert")
    assert a.attrs["message"] == "错误: π must be > 0"
    m = W._walk(_text("crafted_unicode_strings.ttir"))
    assert [f.sym_name for f in m.funcs] == ["核"]
    assert [a.name for a in m.funcs[0].args] == ["π_ptr", "数"]
    assert _ops(m, "tt.assert")[0].attrs["message"] == '错误 "quoted" \\ back'
    assert _ops(m, "tt.print")[0].attrs["prefix"] == "π="
    assert _ops(m, "tt.load")[0].loc_name == "值"
    assert {op.loc.file for op in m.ops if op.loc} == {"/tmp/内核/k.py"}


@pytest.mark.parametrize(
    "body, value",
    [
        ("\\E9\\94\\99", "错"),
        ("a\\22b\\\\c", 'a"b\\c'),
        ("tab\\09nl\\0A", "tab\tnl\n"),
        ("\\n\\t", "\n\t"),
        ("plain", "plain"),
    ],
)
def test_unescape_decodes_utf8_bytes(body, value):
    assert W._unescape(body) == value


@pytest.mark.parametrize("body", ["\\E9\\94", "\\q", "\\"])
def test_unescape_rejects_malformed(body):
    with pytest.raises(ValueError):
        W._unescape(body)


def test_uppercase_exponent_floats():
    m = W._walk(_text("kernel_eps_consts.ttir"))
    consts = {op.attrs["literal"]: dict(op.attrs) for op in _ops(m, "arith.constant")}
    assert consts["9.99999997E-7"] == {
        "literal": "9.99999997E-7",
        "value": ("float", "9.99999997E-7"),
    }
    assert consts["dense<9.99999996E-13>"]["value"] == ("float", "9.99999996E-13")
    assert consts["dense<9.99999996E-13>"]["splat"] is True
    m = W._walk(_text("adv_consts.ttir"))
    by_lit = {op.attrs["literal"]: op.attrs for op in _ops(m, "arith.constant")}
    assert by_lit["dense<-2.14748365E+9>"]["splat"] is True
    assert by_lit["dense<0x7FC00000>"]["value"] == (
        "float",
        "0x7FC00000",
    )  # NaN bit pattern
    assert by_lit["-9223372036854775807"]["value"] == -9223372036854775807


def test_non_splat_dense_constants():
    text = _text("crafted_attr_dicts.ttir").replace(
        "%r = tt.make_range",
        "%ds = arith.constant dense<[1, 2]> : tensor<2xi32>\n    %r = tt.make_range",
        1,
    )
    m = W._walk(text)
    (c,) = [
        op
        for op in _ops(m, "arith.constant")
        if op.attrs["literal"].startswith("dense")
    ]
    assert c.attrs["value"] == ("dense", "[1, 2]") and c.attrs["splat"] is False


def test_leading_attr_dict_on_arith_constant():
    m = W._walk(_text("crafted_attr_dicts.ttir"))
    values = [op.attrs["value"] for op in _ops(m, "arith.constant")]
    assert values == [16, -3]
    m = W._walk(_text("nat_hint_scalar_const.ttir"))
    assert 64 in [op.attrs["value"] for op in _ops(m, "arith.constant")]


def test_generic_closer_attr_dict_is_read():
    (red,) = _ops(W._walk(_text("crafted_attr_dicts.ttir")), "tt.reduce")
    assert red.attrs["axis"] == 0 and red.attrs["axis_note"] == 3


def test_tt_dot_default_precision():
    m = W._walk(_text("kernel_dot_precisions.ttir"))
    got = [
        (op.attrs["inputPrecision"], op.attrs["maxNumImpreciseAcc"])
        for op in _ops(m, "tt.dot")
    ]
    assert got == [("ieee", 0), ("tf32x3", 0)]
    (d,) = _ops(W._walk(_text("spike_dot.ttir")), "tt.dot")
    assert d.attrs["inputPrecision"] == "tf32"


def test_reshape_allow_reorder():
    m = W._walk(_text("adv_views.ttir"))
    assert [op.attrs["allow_reorder"] for op in _ops(m, "tt.reshape")] == [True, True]
    text = _text("adv_views.ttir").replace(" allow_reorder", "", 1)
    assert [op.attrs["allow_reorder"] for op in _ops(W._walk(text), "tt.reshape")] == [
        False,
        True,
    ]
    with pytest.raises(W.MisalignedModule, match="tt.reshape keyword"):
        W._walk(
            _text("adv_views.ttir"),
            scan_text=_text("adv_views.ttir").replace("allow_reorder", "reorder", 1),
        )


@pytest.mark.parametrize(
    "name, shapes",
    [
        ("crafted_empty_else.ttir", {"scf.if": [(1, 1)]}),
        ("crafted_empty_for.ttir", {"scf.for": [(1,)]}),
        ("nat_empty_then.ttir", {"scf.if": [(1, 1)]}),
        (
            "crafted_empty_bodies.ttir",
            {"scf.if": [(1, 1), (1, 1), (1, 0)], "scf.for": [(1,)]},
        ),
    ],
)
def test_empty_region_bodies(name, shapes):
    m = W._walk(_text(name))
    for opname, want in shapes.items():
        assert [tuple(len(r) for r in op.regions) for op in _ops(m, opname)] == want
    for blk in m.blocks:
        if m.ops[blk.op].name in ("scf.if", "scf.for"):
            last = m.ops[blk.ops[-1]]
            assert last.name == "scf.yield"
            if last.implicit:
                assert last.line_no is None and last.operands == () and last.loc is None


def test_empty_for_body_keeps_its_induction_variable():
    m = W._walk(_text("crafted_empty_for.ttir"))
    (loop,) = _ops(m, "scf.for")
    (body,) = loop.regions[0]
    assert len(m.blocks[body].args) == 1 and m.blocks[body].arg_types == ("i32",)
    assert [m.ops[i].implicit for i in m.blocks[body].ops] == [True]


# The tensordesc types of adv_descs as each release prints them.
_DESC_TYPES = {
    "3.6": ("!tt.tensordesc<tensor<32x32xf16>>", "!tt.tensordesc<tensor<1x32xf16>>"),
    "3.8": ("!tt.tensordesc<32x32xf16>", "!tt.tensordesc<1x32xf16>"),
}


def test_descriptor_operand_order():
    m = W._walk(_text("adv_descs.ttir"))
    tile, row = _DESC_TYPES[_printed_by("adv_descs.ttir")]
    (store,) = _ops(m, "tt.descriptor_store")
    (red,) = _ops(m, "tt.descriptor_reduce")
    (scatter,) = _ops(m, "tt.descriptor_scatter")
    types = lambda op: [m.values[v].type for v in op.operands]  # noqa: E731
    # ODS order (desc, src, indices...), printed `%desc[%i, %j], %src`
    assert types(store) == [tile, "tensor<32x32xf16>", "i32", "i32"]
    assert types(red) == types(store) and red.attrs["kind"] == "add"
    # descriptor_scatter prints in ODS order (desc, x_offsets, y_offset, src)
    assert types(scatter) == [row, "tensor<32xi32>", "i32", "tensor<32x32xf16>"]


def test_dot_scaled_operand_order():
    m = W._walk(_text("kernel_dot_scaled.ttir"))
    (d,) = _ops(m, "tt.dot_scaled")
    # ODS (a, b, c, a_scale, b_scale), printed `%a scale %as, %b scale %bs, %c`
    assert [m.values[v].type for v in d.operands] == [
        "tensor<128x64xf8E4M3FN>",
        "tensor<64x128xf8E4M3FN>",
        "tensor<128x128xf32>",
        "tensor<128x2xi8>",
        "tensor<128x2xi8>",
    ]
    assert [m.values[v].name for v in d.operands[3:]] == ["a_scale", "b_scale"]


def test_cf_successors_are_block_indices():
    m = W._walk(_text("golden_nested_guard_merge_sm80.ttir"))
    for op in m.ops:
        if op.name in ("cf.br", "cf.cond_br"):
            parent = m.blocks[op.block]
            for s in op.attrs["successors"]:
                dest = m.blocks[s]
                assert (dest.op, dest.region) == (
                    parent.op,
                    parent.region,
                ) and dest.position > 0
    (br,) = _ops(m, "cf.br")
    assert m.blocks[br.attrs["successors"][0]].label == "^bb5"
    same = W._walk(_text("crafted_same_dest.ttir"))
    first = _ops(same, "cf.cond_br")[0]
    assert len(set(first.attrs["successors"])) == 1  # both edges into ^bb1


def test_parser_invented_locs_are_absent():
    m = W._walk(_text("spike_spin_while.ttir"))
    whiles = _ops(m, "scf.while")
    before = m.blocks[whiles[1].regions[0][0]]
    assert before.arg_names == (None,)  # `scf.while (%v_1 = %v)` prints no arg loc
    for op in m.ops:
        for loc in (op.loc, *op.callers):
            assert loc is None or not loc.file.startswith(
                ("/proc/self/fd", tempfile.gettempdir())
            )


def test_callsite_and_fused_locs():
    m = W._walk(_text("crafted_locs.ttir"))
    b = [op for op in m.ops if op.name == "arith.addi"][1]
    assert b.loc == W.SourceLoc("y.py", 3, 4) and b.loc_name == "inner"
    assert b.callers == (W.SourceLoc("k.py", 2, 5), W.SourceLoc("k.py", 3, 6))
    (store,) = _ops(m, "tt.store")
    assert store.loc == W.SourceLoc("k.py", 2, 5) and store.callers == (
        W.SourceLoc("k.py", 2, 5),
    )
    (ret,) = _ops(m, "tt.return")
    assert ret.loc == W.SourceLoc("k.py", 12, 1)  # alias defined after its use


def test_quoted_symbols_and_strings_with_syntax_chars():
    m = W._walk(_text("crafted_symbols_strings.ttir"))
    assert [(f.sym_name, f.visibility) for f in m.funcs] == [
        ('f{%x} "q" (a)', "private"),
        ("k", "public"),
    ]
    (call,) = _ops(m, "tt.call")
    assert call.attrs["callee"] == 'f{%x} "q" (a)' and len(call.results) == 2
    (asm,) = _ops(m, "tt.elementwise_inline_asm")
    assert asm.attrs["asm_string"] == '{ mov.u32 $0, %tid.x; } // loc("x") }'
    assert asm.attrs["pure"] is True and asm.attrs["packed_element"] == 1


def test_atomic_enums():
    m = W._walk(_text("spike_atomics.ttir"))
    assert [
        (op.attrs["rmw_op"], op.attrs["sem"], op.attrs["scope"])
        for op in _ops(m, "tt.atomic_rmw")
    ] == [
        ("fadd", "relaxed", "cta"),
        ("max", "release", "sys"),
        ("min", "acquire", "gpu"),
        ("and", "acq_rel", "gpu"),
        ("or", "acq_rel", "gpu"),
        ("xor", "acq_rel", "gpu"),
        ("exch", "relaxed", "sys"),
        ("add", "acq_rel", "gpu"),
    ]
    assert [
        (op.attrs["sem"], op.attrs["scope"]) for op in _ops(m, "tt.atomic_cas")
    ] == [("acq_rel", "cta")]


def test_program_id_axes():
    m = W._walk(_text("golden_tile2d_sm80.ttir"))
    assert sorted(op.attrs["axis"] for op in _ops(m, "tt.get_program_id")) == [0, 1]


def test_deep_def_chain():
    m = W._walk(_text("kernel_deep_chain.ttir"))
    assert m.stats["ops"] > 1200 and m.stats["ssa_edges"] > 2400


def test_deep_region_nesting_is_iterative():
    depth = 400  # well past Python's default recursion limit per nested frame
    lines = [
        "module {",
        "  tt.func public @k(%c: i1, %p: !tt.ptr<i32>) attributes {noinline = false} {",
    ]
    lines.append("    %v = arith.constant 7 : i32")
    lines += ["    scf.if %c {"] * depth
    lines.append("    tt.store %p, %v : !tt.ptr<i32>")
    lines += ["    }"] * depth
    lines += ["    tt.return", "  }", "}"]
    m = W._walk("\n".join(lines))
    assert len(_ops(m, "scf.if")) == depth and m.stats["implicit_ops"] == depth
    assert max(len(op.path) for op in m.ops) == depth + 2


def test_deep_loc_alias_chain_fails_closed():
    n = 300
    aliases = ['#loc0 = loc("k.py":1:1)'] + [
        f'#loc{i} = loc("n{i}"(#loc{i - 1}))' for i in range(1, n)
    ]
    text = "\n".join(
        aliases
        + [
            "module {",
            "  tt.func public @k(%p: !tt.ptr<i32>) attributes {noinline = false} {",
            f"    %v = arith.constant 7 : i32 loc(#loc{n - 1})",
            "    tt.store %p, %v : !tt.ptr<i32>",
            "    tt.return",
            "  }",
            "}",
        ]
    )
    with pytest.raises(W.MisalignedModule, match="too deep"):
        W._walk(text)


def _const_module(lit: str, ty: str) -> str:
    return (
        "module {\n"
        "  tt.func public @k() attributes {noinline = false} {\n"
        f"    %c = arith.constant {lit} : {ty}\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


@pytest.mark.parametrize(
    "lit, ty, match",
    [
        # the parser accepts these, but MLIR holds the bits as a negative
        # value (the printer re-prints -1, -2147483648, -1, dense<-1>, -1)
        ("4294967295", "i32", "signed"),
        ("2147483648", "i32", "signed"),
        ("255", "i8", "signed"),
        ("dense<4294967295>", "tensor<4xi32>", "signed"),
        ("18446744073709551615", "i64", "signed"),
        ("dense<1>", "tensor<4xi1>", "true / false"),  # printed dense<true>
    ],
)
def test_constant_outside_the_printed_signed_range_is_refused(lit, ty, match):
    with pytest.raises(W.MisalignedModule, match=match):
        W._walk(_const_module(lit, ty))


@pytest.mark.parametrize(
    "lit, ty, value",
    [
        ("-1", "i32", -1),
        ("2147483647", "i32", 2**31 - 1),
        ("-2147483648", "i32", -(2**31)),
        ("-128", "i8", -128),
        ("dense<-1>", "tensor<4xi32>", -1),
        ("-9223372036854775808", "i64", -(2**63)),
        ("9223372036854775807", "index", 2**63 - 1),
    ],
)
def test_constant_in_the_printed_signed_range(lit, ty, value):
    (c,) = _ops(W._walk(_const_module(lit, ty)), "arith.constant")
    assert c.attrs["value"] == value


_FOR = """module {
  tt.func public @k(%p: !tt.ptr<i32>, %n: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for KW%i = %c0 to %n step %c1  : i32 {
      %q = tt.addptr %p, %i : !tt.ptr<i32>, i32
      tt.store %q, %i : !tt.ptr<i32>
    }
    tt.return
  }
}
"""


def test_scf_for_unsigned_compare_is_recorded():
    signed, unsigned = _FOR.replace("KW", ""), _FOR.replace("KW", "unsigned ")
    (loop,) = _ops(W._walk(signed), "scf.for")
    assert loop.attrs["unsignedCmp"] is False
    (loop,) = _ops(W._walk(unsigned), "scf.for")
    assert loop.attrs["unsignedCmp"] is True
    # an unknown header keyword is a misread, never an ignored word
    with pytest.raises(W.MisalignedModule, match="scf.for header"):
        W._walk(unsigned, scan_text=_FOR.replace("KW", "signless "))
    with pytest.raises(W.MisalignedModule, match="scf.for header"):
        W._walk(signed, scan_text=_FOR.replace("KW", "").replace(" step ", " stride "))


def test_ttgir_is_refused():
    text = "#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\n"
    text += _text("golden_add_sm80.ttir")
    with pytest.raises((W.MisalignedModule, W.ModuleParseError)):
        W._walk(text)


# ─────────────────────────── parse errors and the parse input ───────────────────────────


def test_parse_error_carries_the_diagnostic(capfd):
    text = _text("golden_add_sm80.ttir")
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if "tt.make_range {" in ln)
    lines[i] = lines[i].replace("tt.make_range {", "tt.make_range_v2 {")
    with pytest.raises(W.ModuleParseError) as e:
        W.walk_module("\n".join(lines))
    assert (
        "tt.make_range_v2" in e.value.diagnostic and "is unknown" in e.value.diagnostic
    )
    assert "<ttir>" in e.value.diagnostic and "/proc/self/fd" not in e.value.diagnostic
    assert e.value.line_no == i + 1
    out, err = capfd.readouterr()
    assert "make_range_v2" not in err  # captured, not leaked onto the process stderr


@pytest.mark.parametrize(
    "old, new",
    [
        ("{noinline = false}", "{noinline = }"),  # malformed attribute dict
        (
            "arith.addf %2, %cst : tensor<64xf32>",
            "arith.addf %2, %cst : tensor<32xf32>",
        ),  # operand type mismatch
    ],
)
def test_malformed_text_is_a_parse_error(old, new):
    text = _text("nat_k_uni.ttir")
    assert old in text
    with pytest.raises(W.ModuleParseError):
        W._walk(text.replace(old, new, 1))


def test_unwritable_tmpdir_is_a_parse_error(monkeypatch, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    if os.access(ro, os.W_OK):
        pytest.skip("directory permissions are not enforced (root?)")
    monkeypatch.delattr(os, "memfd_create", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(ro))
    with pytest.raises(W.ModuleParseError, match="temporary file"):
        W._walk(_text("nat_k_uni.ttir"))
    ro.chmod(0o700)


def test_tempfile_fallback(monkeypatch, tmp_path):
    monkeypatch.delattr(os, "memfd_create", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    m = W._walk(_text("spike_spin_while.ttir"))
    assert m.stats["ops"] == 19
    assert list(tmp_path.iterdir()) == []  # the parse input is removed
    # parser-invented locs name the temp path: filtered on both sides
    assert m.blocks[_ops(m, "scf.while")[1].regions[0][0]].arg_names == (None,)
    with pytest.raises(W.ModuleParseError) as e:
        W._walk(
            _text("nat_k_uni.ttir").replace("tt.make_range {", "tt.make_range_v2 {", 1)
        )
    assert str(tmp_path) not in e.value.diagnostic and "<ttir>" in e.value.diagnostic


def test_no_fd_leak():
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        W._walk(_text("nat_k_uni.ttir"))
    with pytest.raises(W.ModuleParseError):
        W._walk(
            _text("nat_k_uni.ttir").replace("tt.make_range {", "tt.make_range_v2 {", 1)
        )
    assert len(os.listdir("/proc/self/fd")) == before


# ─────────────────────────── per-release printer tables ───────────────────────────


def test_printer_tables_are_keyed_by_release():
    assert set(W.PRINTERS) == {"3.6", "3.8"}
    for release, table in W.PRINTERS.items():
        assert table.release == release
        # every leading keyword has a closed vocabulary, and every table
        # names only what its fields hold
        for op, keys in table.keywords.items():
            assert all((op, k) in table.vocab for k in keys), (release, op)
        for (op, key), ints in table.bind_ints.items():
            assert ("int", key) in table.bind_attrs.get(op, ()), (release, op)
            assert set(ints) == table.vocab[(op, key)], (release, op)
    assert W.printer("3.6") is W.PRINTERS["3.6"]
    assert W.printer().release == G.RELEASE


def test_the_3_8_table_is_3_6_plus_the_audited_changes():
    """3.8's printer is 3.6's (audited) but for ttg.barrier and the missing
    block-pointer types: any other difference must be a deliberate edit."""
    t36, t38 = W.PRINTERS["3.6"], W.PRINTERS["3.8"]
    barrier = {"ttg.barrier"}
    for field in ("needed", "keywords", "bind_attrs", "defaults", "printed_order"):
        a, b = dict(getattr(t36, field)), dict(getattr(t38, field))
        assert {k: v for k, v in b.items() if k not in barrier} == a, field
    for field in ("vocab", "bind_ints", "attr_types"):
        a, b = dict(getattr(t36, field)), dict(getattr(t38, field))
        assert {k: v for k, v in b.items() if k[0] not in barrier} == a, field
    assert t38.keywords["ttg.barrier"] == ("addrSpace",)
    assert t38.bind_attrs["ttg.barrier"] == (("int", "addrSpace"),)
    assert dict(t38.bind_ints[("ttg.barrier", "addrSpace")]) == {
        "none": 0,
        "local": 1,
        "global_read": 2,
        "global_write": 4,
        "tensor_read": 8,
        "tensor_write": 16,
        "all": 31,
    }
    assert t36.elides_yield == t38.elides_yield
    assert (t36.block_pointer_types, t38.block_pointer_types) == (True, False)


@pytest.mark.parametrize("version", ["3.7.0", "3.9.0rc1", "4.0.0", "3.60.1"])
def test_an_unknown_release_fails_closed_before_any_parse(monkeypatch, version):
    import triton

    def no_parse(_data, _table):
        raise AssertionError("parsed a text for an unknown release")

    monkeypatch.setattr(W, "_bind_walk", no_parse)
    monkeypatch.setattr(triton, "__version__", version)
    release = ".".join(version.split(".")[:2])
    for walk in (W.walk_module, W._walk):
        with pytest.raises(W.UnknownTritonRelease) as e:
            walk(_text("golden_add_sm80.ttir"))
        assert (e.value.release, e.value.version) == (release, version)
        assert f"no printer table for Triton {version}" in e.value.message
        assert "3.6.x, 3.8.x" in e.value.message
    assert not W._CACHE  # nothing cached for it
    with pytest.raises(W.UnknownTritonRelease):
        W.printer(release)


def test_the_cache_is_keyed_by_release(monkeypatch):
    import triton

    text = _text("golden_add_sm80.ttir")
    here = W.walk_module(text)
    assert here.release == G.RELEASE
    # a (pretended) other release with the same table: its own entry
    other = dataclasses.replace(W.printer(), release="9.9")
    monkeypatch.setattr(W, "PRINTERS", {**W.PRINTERS, "9.9": other})
    monkeypatch.setattr(triton, "__version__", "9.9.0")
    there = W.walk_module(text)
    assert there.release == "9.9" and there is not here and len(W._CACHE) == 2
    assert dataclasses.replace(there, release=here.release) == here


# The barrier tl.debug_barrier() prints, per release.
_BARRIER = {"3.6": "gpu.barrier", "3.8": "ttg.barrier all"}


def _barrier_module(barrier: str) -> str:
    return (
        "module {\n"
        "  tt.func public @k(%p: !tt.ptr<f32>) attributes {noinline = false} {\n"
        f"    {barrier}\n"
        "    %v = tt.load %p : !tt.ptr<f32>\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


def test_the_release_barrier_aligns():
    m = W._walk(_barrier_module(_BARRIER[G.RELEASE]))
    (barrier,) = [op for op in m.ops if op.name.endswith(".barrier")]
    assert barrier.name == _BARRIER[G.RELEASE].split()[0] and not barrier.results
    if G.RELEASE == "3.8":
        assert dict(barrier.attrs) == {"addrSpace": "all"}
        # addrSpace is needed, and read by the bindings too
        plain = W._walk(_barrier_module("")).stats
        assert m.stats["bind_attrs"] - plain["bind_attrs"] == 1
        assert m.stats["needed_attrs"] - plain["needed_attrs"] == 1
    # the other release's barrier: 3.6's parser has no ttg dialect, 3.8's
    # still has gpu.barrier (its reader refuses it, see test_ttir_reader)
    for release, spelling in _BARRIER.items():
        if release == G.RELEASE:
            continue
        if G.RELEASE == "3.6":
            with pytest.raises(W.ModuleParseError, match="is unknown"):
                W._walk(_barrier_module(spelling))
        else:
            assert _ops(W._walk(_barrier_module(spelling)), "gpu.barrier")


_ONLY_3_8 = pytest.mark.skipif(
    G.RELEASE != "3.8", reason="needs Triton 3.8's bindings (ttg.barrier)"
)


@_ONLY_3_8
@pytest.mark.parametrize(
    "word, bits",
    sorted(W.PRINTERS["3.8"].bind_ints[("ttg.barrier", "addrSpace")].items()),
)
def test_ttg_barrier_addr_space_is_cross_checked(word, bits):
    text = _barrier_module(f"ttg.barrier {word}")
    (barrier,) = _ops(W._walk(text), "ttg.barrier")
    assert barrier.attrs["addrSpace"] == word
    # an in-vocabulary swap in the text layer: the bindings' bits see it
    other = "none" if word != "none" else "all"
    with pytest.raises(
        W.MisalignedModule, match=f"attr addrSpace: text '{other}' vs bindings {bits}"
    ):
        W._walk(text, scan_text=_barrier_module(f"ttg.barrier {other}"))


@_ONLY_3_8
def test_ttg_barrier_spellings_outside_the_one_keyword_syntax_misalign():
    # a bit combination prints as `a|b`: not one keyword, so not read
    with pytest.raises(W.MisalignedModule, match="leading keyword"):
        W._walk(_barrier_module("ttg.barrier local|global_read"))
    text = _barrier_module("ttg.barrier all")
    with pytest.raises(W.MisalignedModule, match="closed vocabulary"):
        W._walk(text, scan_text=_barrier_module("ttg.barrier every"))


# two spellings that once reached (and aborted) 3.8's parser: the type split
# over two lines, and a comment between `ptr<` and `tensor`
_SPLIT_NEWLINE = "!tt.ptr<\n tensor<32xf32>>"
_SPLIT_COMMENT = "!tt.ptr< // c\n tensor<32xf32>>"

_BLOCK_PTR = """module {
  tt.func public @k(%p: !tt.ptr<f32>, %b: TYPE) attributes {noinline = false} {
    tt.return
  }
}
"""


@pytest.mark.parametrize(
    "text, line",
    [
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr<tensor<32x32xf32>>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr<tensor<32xf32>, 1>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr< tensor <4xf32>>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt<ptr<tensor<4xf32>>>"), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr<!tt.ptr<tensor<4xf32>>>"), 2),
        # split over lines, or around a comment: no single line holds it
        (_BLOCK_PTR.replace("TYPE", _SPLIT_NEWLINE), 2),
        (_BLOCK_PTR.replace("TYPE", _SPLIT_COMMENT), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt.ptr\n<tensor<4xf32>>"), 2),
        (_BLOCK_PTR.replace("TYPE", '!tt.ptr< // "q>\n // b\n tensor<4xf32>>'), 2),
        (_BLOCK_PTR.replace("TYPE", "!tt<ptr // c\n<tensor<4xf32>>>"), 2),
        # a type alias may hide the pointee
        ("!t = tensor<4xf32>\n" + _BLOCK_PTR.replace("TYPE", "!tt.ptr<!t>"), 1),
        ("!t // c\n = tensor<4xf32>\n" + _BLOCK_PTR.replace("TYPE", "!tt.ptr<!t>"), 1),
        (
            "// c\n\n  !t\n = tensor<4xf32>\n"
            + _BLOCK_PTR.replace("TYPE", "!tt.ptr<!t>"),
            3,
        ),
    ],
)
def test_block_pointer_types_are_refused_before_the_3_8_parser(monkeypatch, text, line):
    """3.8's parser aborts the process on a block-pointer type: the walk
    refuses such a text before the bindings see it (also under 3.6's
    bindings: the screen runs first), and 3.6's table does not screen."""

    def no_parse(_data, _table):
        raise AssertionError("the bindings got a block-pointer type")

    monkeypatch.setattr(W, "_bind_walk", no_parse)
    with pytest.raises(W.ModuleParseError, match="no block pointers") as e:
        W._walk(text, table=W.PRINTERS["3.8"])
    assert e.value.line_no == line and f"line {line}:" in e.value.diagnostic
    W._screen(text, W.PRINTERS["3.6"])  # 3.6 has block pointers: no screen


def test_the_block_pointer_screen_reads_types_not_strings():
    text = _module_with_print('"ptr<tensor<"')
    W._screen(text, W.PRINTERS["3.8"])
    # nor comments (the parser skips them), nor a `//` inside a string
    W._screen(
        text.replace("tt.return", "tt.return // !tt.ptr<tensor<4xf32>>"),
        W.PRINTERS["3.8"],
    )
    W._screen(_module_with_print('"ptr< // "'), W.PRINTERS["3.8"])
    in_string = _BLOCK_PTR.replace("TYPE", "i32").replace(
        "{noinline = false}",
        '{noinline = false, s = "a // b", t = !tt.ptr<\n tensor<4xf32>>}',
    )
    with pytest.raises(W.ModuleParseError, match="line 2: a block-pointer type"):
        W._screen(in_string, W.PRINTERS["3.8"])
    unterminated = text.replace('"ptr<tensor<"', '"ptr<tensor<')
    with pytest.raises(W.ModuleParseError, match="block-pointer type"):
        W._screen(unterminated, W.PRINTERS["3.8"])


def _module_with_print(prefix: str) -> str:
    return (
        "module {\n"
        "  tt.func public @k(%x: i32) attributes {noinline = false} {\n"
        f"    tt.print {prefix} {{hex = false, isSigned = array<i32: 1>}} : %x : i32\n"
        "    tt.return\n"
        "  }\n"
        "}\n"
    )


_ABORT = r"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from tilelens.ir import _mlir_walk as W
text = sys.argv[2]
try:
    m = W.walk_module(text)
    print("aligned", m.release)
except W.ModuleParseError as e:
    print("refused", e.line_no)
except W.MisalignedModule:
    print("misaligned")
"""


@pytest.mark.parametrize(
    "pointee, aligned",
    [
        ("!tt.ptr<tensor<32x32xf32>>", True),
        # the text layer reads a type within one line and no comments
        # outside block labels: these misalign where the parser takes them
        (_SPLIT_NEWLINE, False),
        (_SPLIT_COMMENT, False),
    ],
)
def test_a_block_pointer_text_never_kills_the_process(pointee, aligned):
    text = _BLOCK_PTR.replace("TYPE", pointee)
    r = subprocess.run(
        [sys.executable, "-c", _ABORT, str(REPO), text],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, (r.returncode, r.stderr[-2000:])
    if not W.printer().block_pointer_types:
        want = "refused 2"
    else:
        want = f"aligned {G.RELEASE}" if aligned else "misaligned"
    assert r.stdout.strip() == want


# ─────────────────────────── cache, immutability, lifetime ───────────────────────────


def test_cache_returns_the_same_module_and_caches_misalignment(monkeypatch):
    text = _text("golden_add_sm80.ttir")
    assert W.walk_module(text) is W.walk_module(text)
    bad = _text("crafted_generic_form.ttir")
    with pytest.raises(W.MisalignedModule) as first:
        W.walk_module(bad)

    def no_parse(_data, _table):
        raise AssertionError("parsed twice")

    monkeypatch.setattr(W, "_bind_walk", no_parse)
    with pytest.raises(W.MisalignedModule) as again:
        W.walk_module(bad)
    assert (
        again.value.problems == first.value.problems and again.value is not first.value
    )
    W.walk_module(text)


def test_cache_is_bounded():
    for k in range(W._CACHE_SIZE + 5):
        W.walk_module(_text("nat_k_uni.ttir") + f"\n// {k}\n")
    assert len(W._CACHE) == W._CACHE_SIZE


def test_module_is_immutable():
    m = W._walk(_text("spike_atomics.ttir"))
    op = next(op for op in m.ops if op.attrs)
    with pytest.raises(TypeError):
        op.attrs["x"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        op.name = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        m.stats["ops"] = 0  # type: ignore[index]


def test_records_hash_pickle_and_deep_copy():
    m = W._walk(_text("spike_atomics.ttir"))
    assert pickle.loads(pickle.dumps(m)) == m and copy.deepcopy(m) == m
    again = W._walk(_text("spike_atomics.ttir"))
    assert again == m and hash(again) == hash(m)
    memo = {op: op.index for op in m.ops}  # ops are usable as keys
    assert [memo[op] for op in again.ops] == list(range(len(m.ops)))
    assert {f.args[0]: 1 for f in m.funcs if f.args}


_PLAIN = (int, str, bool, float, type(None))


def _assert_plain(root) -> int:
    """Every object reachable from ``root`` is plain Python data."""
    n = 0
    stack = [root]
    while stack:
        x = stack.pop()
        n += 1
        if isinstance(x, _PLAIN):
            continue
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            assert type(x).__module__ == W.__name__, type(x)
            stack += [getattr(x, f.name) for f in dataclasses.fields(x)]
        elif isinstance(x, tuple):
            stack += list(x)
        elif isinstance(x, W._FrozenMap):
            stack += list(x.keys()) + list(x.values())
        else:
            raise AssertionError(
                f"non-plain object {type(x).__module__}.{type(x).__qualname__}"
            )
    return n


def test_no_binding_object_escapes():
    for name in (
        "golden_matmul_s3_sm80.ttir",
        "spike_reduce_scan.ttir",
        "adv_cf_blockargs.ttir",
    ):
        assert _assert_plain(W._walk(_text(name))) > 100


def test_threads_walk_consistently():
    names = ALIGNED[:12]
    ref = {n: _pins(W._walk(_text(n))) for n in names}
    errors: list[str] = []

    def work(k: int) -> None:
        for n in names[k % 3 :] + names[: k % 3]:
            if _pins(W._walk(_text(n))) != ref[n]:
                errors.append(n)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


_HAZARD = r"""
import gc, sys
sys.path.insert(0, sys.argv[1])
from tilelens.ir import _mlir_walk as W
texts = [open(p, encoding="utf-8").read() for p in sys.argv[2:]]
first = [W.walk_module(t) for t in texts]
sig = [[(op.name, op.operands, op.results, op.result_types) for op in m.ops] for m in first]
W._CACHE.clear()
del first
gc.collect()
# everything the walks created is gone; read types and locs again, walk again
again = [W._walk(t) for t in texts]
gc.collect()
assert sig == [[(op.name, op.operands, op.results, op.result_types) for op in m.ops] for m in again]
for m in again:
    for v in m.values:
        v.type.encode()
del again
gc.collect()
try:
    W._walk(texts[0].replace("tt.make_range {", "tt.make_range_v2 {", 1))
except W.ModuleParseError:
    pass
gc.collect()
print("ok", len(texts))
"""


def test_subprocess_walk_then_drop_everything_is_safe():
    paths = [
        str(TTIR / n)
        for n in (
            "golden_matmul_s3_sm80.ttir",
            "spike_spin_while.ttir",
            "nat_k_uni.ttir",
        )
    ]
    for _ in range(
        3
    ):  # the unsafe pattern crashed nondeterministically (SIGSEGV / SIGBUS / hang)
        r = subprocess.run(
            [sys.executable, "-c", _HAZARD, str(REPO), *paths],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        assert r.stdout.strip().endswith("ok 3")


_FORK = r"""
import os, signal, sys, threading, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from tilelens.ir import _mlir_walk as W
text = open(sys.argv[2], encoding="utf-8").read()
inside, release = threading.Event(), threading.Event()

def mid_walk():
    # another thread inside a walk: both locks held, fd 2 redirected
    with W._CACHE_LOCK, W._PARSE_LOCK, W._StderrCapture():
        inside.set()
        release.wait()

holder = threading.Thread(target=mid_walk)
holder.start()
inside.wait()
pid = os.fork()
if pid == 0:
    signal.alarm(60)  # a deadlocked walk dies (SIGALRM) instead of hanging
    os.write(2, b"child stderr\n")
    W.walk_module(text)
    os._exit(0)
_, status = os.waitpid(pid, 0)
release.set()
holder.join()
print("child exit", os.waitstatus_to_exitcode(status))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_fork_inside_a_parse_window_is_safe():
    r = subprocess.run(
        [sys.executable, "-c", _FORK, str(REPO), str(TTIR / "nat_k_uni.ttir")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "child exit 0"  # the child's walk took fresh locks
    assert "child stderr" in r.stderr  # and its fd 2 is the real stderr again


def test_other_fd2_output_during_a_parse_is_passed_on(capfd, monkeypatch):
    enter = W._StderrCapture.__enter__

    def enter_then_write(self):
        got = enter(self)
        os.write(2, b"another thread's line\n")  # lands in the capture buffer
        return got

    monkeypatch.setattr(W._StderrCapture, "__enter__", enter_then_write)
    W._walk(_text("nat_k_uni.ttir"))
    assert "another thread's line" in capfd.readouterr().err


_RSS = r"""
import gc, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from tilelens.ir import _mlir_walk as W
text = open(sys.argv[2], encoding="utf-8").read()

def rss_kib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])

for _ in range(100):
    W._walk(text)
gc.collect()
before = rss_kib()
for _ in range(300):
    W._walk(text)
gc.collect()
print((rss_kib() - before) / 300)
"""


@pytest.mark.skipif(not os.path.exists("/proc/self/status"), reason="needs /proc")
def test_parse_memory_is_reclaimed():
    # without the body-block erase each parse of this 12 KiB text leaked
    # 12-20 KiB; with it about 1-2 KiB (the empty module op, allocator noise)
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            _RSS,
            str(REPO),
            str(TTIR / "golden_matmul_s3_sm80.ttir"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    per_parse_kib = float(r.stdout.split()[-1])
    assert per_parse_kib < 8.0
