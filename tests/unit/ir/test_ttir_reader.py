"""tilelens.ir.ttir_reader: the TTIR -> AccessGraph reader on top of _mlir_walk.

Goldens: tests/golden/ir/ttir/ (the walk layer's corpus) and
tests/golden/ir/reader_ttir/ (the audit probes, the phase-2 review probes
and reader-specific shapes, host-compiled from
tests/golden/ir/reader_kernels.py by tests/golden/ir/generate_reader_ttir.py),
each read as the installed release prints it where it has its own printing
(ttir_<release>/, reader_ttir_<release>/: see _goldens.py). The differential
oracle is #361's regex reader, vendored verbatim as _oracle_ttir_reader_361.py.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from tilelens.ir import ParseCache, Refusal, SourceLocation
from tilelens.ir import _mlir_walk as W
from tilelens.ir import ttir_reader as R
from tilelens.ir.ttir_reader import (
    AccessGraph,
    Arange,
    Bin,
    Cmp,
    Const,
    DataDep,
    IntCast,
    IterArgOffset,
    Observed,
    Param,
    Pid,
    Select,
    TTIRKind,
    UnsupportedTTIR,
    mentions_observed,
    observed_indices,
    parse_ttir,
    width_obligations,
)

from . import _goldens as G
from . import _oracle_ttir_reader_361 as O

REPO = Path(__file__).resolve().parents[3]
GOLDEN = REPO / "tests" / "golden" / "ir"
KERNELS = GOLDEN / "reader_kernels.py"
DIRS = ("ttir", "reader_ttir")
# "<dir>/<name>" -> the golden the installed release reads
PATHS = {f"{d}/{n}": p for d in DIRS for n, p in G.texts(d).items()}
FILES = list(PATHS)


def _path(name: str) -> Path:
    return PATHS[name]


def _text(name: str) -> str:
    return _path(name).read_text(encoding="utf-8")


def _graph(name: str) -> AccessGraph:
    return parse_ttir(_text(name))


def _refusal(name_or_text: str) -> UnsupportedTTIR:
    text = _text(name_or_text) if name_or_text.endswith(".ttir") else name_or_text
    with pytest.raises(UnsupportedTTIR) as info:
        parse_ttir(text)
    return info.value


def _module(body: str, args: str = "%p: !tt.ptr<f32>, %n: i32", extra: str = "") -> str:
    """A minimal TTIR module (no locs) around ``body``'s op lines."""
    lines = "\n    ".join(line.strip() for line in body.strip().splitlines())
    return (
        f"module {{\n  tt.func public @k({args}) attributes {{noinline = false}} {{\n"
        f"    {lines}\n    tt.return\n  }}\n{extra}}}\n"
    )


# Where a refusal's loc points, per release: 3.8's frontend locates an op
# at its own AST node (scf.yield at the loop header, a call spanning lines
# at its first line), 3.6's at the last child it visited.
_REFUSAL_SITE = {
    "p1_variant_delta": {"3.6": "p += k", "3.8": "for k in range(-8, n):"},
    "loop_observed_advance": {"3.6": "atomic_add", "3.8": "for i in range(0, n):"},
    # the asm call's operand list / its first line
    "rv_inline_asm_store": {"3.6": "[p, v]", "3.8": "tl.inline_asm_elementwise("},
}


def _source_line(loc) -> str:
    """The reader_kernels.py source line a probe golden's loc points at."""
    assert loc is not None and loc.file.endswith("reader_kernels.py"), loc
    return KERNELS.read_text(encoding="utf-8").splitlines()[loc.line - 1]


@pytest.fixture(autouse=True)
def _fresh_cache():
    W._CACHE.clear()
    yield
    W._CACHE.clear()


# ─────────────────────────── refusal form (D8) ───────────────────────────


def test_kinds_are_the_representational_ones():
    assert {k.value for k in TTIRKind} == {
        "untested-triton-version",
        "indirect-address",
        "data-dependent-bound",
        "nested-loop",
        "control-flow",
        "block-pointer",
        "out-of-vocabulary",
        "call",
        "loop-variant-advance",
        "inline-asm",
        "reader-misalignment",
        "unparsable",
        "other",
    }
    # a kind is its string, also when formatted
    assert (
        TTIRKind.CALL == "call" and f"{TTIRKind.CALL}" == str(TTIRKind.CALL) == "call"
    )


def test_unsupported_ttir_carries_structured_fields():
    loc = W.SourceLoc("k.py", 3, 4)
    e = UnsupportedTTIR("control-flow", "scf.while", line_no=7, loc=loc)
    assert e.kind is TTIRKind.CONTROL_FLOW
    assert (e.message, e.line_no, e.loc, str(e)) == ("scf.while", 7, loc, "scf.while")
    back = pickle.loads(pickle.dumps(e))
    assert (back.kind, back.message, back.line_no, back.loc) == (
        e.kind,
        e.message,
        7,
        loc,
    )
    # client-owned kinds are not reader kinds
    for kind in ("spin-shape", "cas-value", "data-dependent-mask"):
        with pytest.raises(ValueError):
            UnsupportedTTIR(kind, "no")
    # the verdict record copies the fields, no string round-trip; the
    # private walk loc becomes the public SourceLocation (D20)
    r = Refusal.from_exception(e)
    assert (r.kind, r.message, r.line_no, r.loc) == (
        "control-flow",
        "scf.while",
        7,
        SourceLocation("k.py", 3, 4),
    )
    assert type(r.loc) is SourceLocation


def test_parse_cache_reads_with_this_reader():
    cache = ParseCache()
    parsed = cache.get(_text("ttir/golden_add_sm80.ttir"))
    assert isinstance(parsed.graph, AccessGraph) and parsed.error is None
    refused = cache.get(_text("reader_ttir/p2_swap.ttir"))
    assert isinstance(refused.refusal, UnsupportedTTIR) and refused.error is None
    assert refused.refusal.kind is TTIRKind.LOOP_VARIANT_ADVANCE


def test_walk_failures_map_to_kinds():
    e = _refusal("ttir/crafted_generic_form.ttir")
    assert e.kind is TTIRKind.READER_MISALIGNMENT
    assert e.line_no == 3 and "arith.cmpi" in e.message
    e = _refusal("module {\n  tt.func public @k( {\n}\n")
    assert e.kind is TTIRKind.UNPARSABLE
    assert e.line_no == 2 and "<ttir>" in e.message


def test_every_walk_table_release_has_a_reader_vocabulary():
    assert set(R._VOCABULARIES) == set(W.PRINTERS)


@pytest.mark.parametrize("version", ["3.7.0", "3.9.0", "4.0.0"])
def test_an_unknown_release_is_refused_as_untested(monkeypatch, version):
    import triton

    monkeypatch.setattr(triton, "__version__", version)
    e = _refusal("ttir/golden_add_sm80.ttir")
    assert e.kind is TTIRKind.UNTESTED_TRITON_VERSION
    assert f"no printer table for Triton {version}" in e.message
    assert (e.line_no, e.loc) == (None, None)


def test_a_release_without_a_reader_vocabulary_is_refused(monkeypatch):
    """A walk-layer table alone does not make the reader read a release."""
    vocabularies = {k: v for k, v in R._VOCABULARIES.items() if k != G.RELEASE}
    monkeypatch.setattr(R, "_VOCABULARIES", vocabularies)
    e = _refusal("ttir/golden_add_sm80.ttir")
    assert e.kind is TTIRKind.UNTESTED_TRITON_VERSION
    assert f"no op vocabulary for Triton {G.RELEASE}" in e.message


def test_refusals_carry_the_op_line_and_source_loc():
    e = _refusal("reader_ttir/p3_call_offset.ttir")
    assert e.kind is TTIRKind.CALL and "_p3_helper" in e.message
    assert (
        "tt.call"
        in _text("reader_ttir/p3_call_offset.ttir").splitlines()[e.line_no - 1]
    )
    assert "_p3_helper(" in _source_line(e.loc)


# ─────────────────────────── the differential oracle ───────────────────────────
# #361's regex reader (single-path) against the new reader on every golden.
# Where both accept, the access inventory must match after _normalize; the
# table lists every file where the two legitimately differ, with the reason
# and exactly what differs: the normalized fields (see _diff) when both
# accept, else which reader refuses.

_NEW = "new refuses"
_OLD = "#361 refuses"
_CALL = "fix 3: tt.call -> call (#361 reads the callee in the caller's env)"
_BITCAST = "atomics through a same-width tt.bitcast pointer (#361: refused)"
_NAMELOC = "a parameter without NameLoc is arg<index> (#361: its printed name)"
_EXPANDED = (
    "a loop-carried pointer tile expanded in the loop gets its own iter_args "
    "entry with re-placed lanes (#361 keeps the stale dims: a false in-bounds)"
)
EXPECTED_DIFF: dict[str, tuple[str, str | frozenset[str]]] = {
    # the audit's soundness fixes
    "reader_ttir/p1_variant_delta.ttir": (
        "fix 1: advance by the induction var -> loop-variant-advance "
        "(#361: delta=LoopVar)",
        _NEW,
    ),
    "reader_ttir/loop_observed_advance.ttir": (
        "fix 1: advance by an atomic observed in the loop -> loop-variant-advance",
        _NEW,
    ),
    "reader_ttir/p2_swap.ttir": (
        "fix 2: swapped pointer iter_args -> loop-variant-advance (#361: delta 0)",
        _NEW,
    ),
    "reader_ttir/p3_call_guarded.ttir": (_CALL, _NEW),
    "reader_ttir/p3_call_offset.ttir": (_CALL, _NEW),
    "reader_ttir/rv_inline_asm_store.ttir": (
        "fix 7: impure inline asm -> inline-asm (#361 misses its st.global)",
        _NEW,
    ),
    "ttir/spike_inline_asm.ttir": ("fix 7: impure inline asm -> inline-asm", _NEW),
    "reader_ttir/pure_asm_int_addr.ttir": (
        "fix 7: a pure asm handed the address as an integer -> inline-asm",
        _NEW,
    ),
    "reader_ttir/tile3d_shared_arange.ttir": (
        "expand_dims tracks each lane's position (#361 keeps the first placement, "
        "collapsing dims 1 and 2 of one make_range into one lane variable)",
        frozenset({"events[0].offset"}),
    ),
    "reader_ttir/expand_iterarg_3d.ttir": (
        _EXPANDED,
        frozenset({"events[0].offset", "events[1].offset", "n_iter_args"}),
    ),
    "reader_ttir/expand_iterarg_mask.ttir": (
        _EXPANDED,
        frozenset({"events[0].offset", "n_iter_args"}),
    ),
    "reader_ttir/observed_lanes.ttir": (
        "two lanes of one tensor atomic's observations in an address "
        "(#361: one symbol for both, a false in-bounds)",
        _NEW,
    ),
    # #361 weaknesses the walk-based reader does not share
    "reader_ttir/unsigned_index.ttir": ("divui is Bin('u//') (#361: unmodeled)", _OLD),
    "reader_ttir/loop_two_step_advance.ttir": (
        "two addptrs per iteration: delta is their sum (#361: refused)",
        _OLD,
    ),
    "reader_ttir/where_pointer.ttir": (
        "arith.select of same-base pointers selects offsets (#361: refused)",
        _OLD,
    ),
    "ttir/spike_if_yield.ttir": (
        "a same-base pointer yielded by scf.if selects offsets (#361: refused)",
        _OLD,
    ),
    "ttir/golden_atomic_fmax_sm80.ttir": (_BITCAST, _OLD),
    "ttir/golden_atomic_fmax_sm90.ttir": (_BITCAST, _OLD),
    "ttir/nat_hint_scalar_const.ttir": (
        "arith.constant with a leading attr dict (#361's regex: DataDep)",
        _OLD,
    ),
    "ttir/crafted_odd_names.ttir": (
        "values by identity (#361's regexes miss the names)",
        _OLD,
    ),
    "ttir/crafted_unicode_strings.ttir": (
        "quoted func symbol (#361: no tt.func found)",
        _OLD,
    ),
    "ttir/nat_k_uni.ttir": (
        "a non-ASCII path decodes as UTF-8 (#361: raw \\XX escapes)",
        frozenset({"events[0].loc", "events[1].loc"}),
    ),
    "ttir/nat_uni_params.ttir": (
        "parameters by NameLoc (π_ptr, 数_n), not printed names",
        frozenset({"args", "events[0].base", "events[1].base", "events[1].mask"}),
    ),
    "ttir/crafted_empty_else.ttir": (
        _NAMELOC,
        frozenset({"args", "events[0].base", "events[0].path"}),
    ),
    "ttir/crafted_empty_for.ttir": (
        _NAMELOC,
        frozenset({"args", "events[0].base", "loop"}),
    ),
    "ttir/crafted_locs.ttir": (_NAMELOC, frozenset({"args", "events[0].offset"})),
}

# The new reader's refusal kind for every refused golden (the rest parse).
REFUSED = {
    "ttir/adv_cf_blockargs.ttir": "control-flow",
    "ttir/adv_descs.ttir": "out-of-vocabulary",
    "ttir/adv_hinted.ttir": "other",  # an integer tt.reduce feeds an address
    "ttir/adv_multi_func.ttir": "call",
    "ttir/adv_multi_result.ttir": "other",  # a loop result feeds an address
    "ttir/adv_nest3.ttir": "control-flow",  # a loop under an scf.if
    "ttir/adv_views.ttir": "other",  # tt.reshape of a pointer tile
    "ttir/adv_while_nested.ttir": "control-flow",
    "ttir/crafted_attr_dicts.ttir": "other",
    "ttir/crafted_deep_nest.ttir": "control-flow",
    "ttir/crafted_fwd_ref_cf.ttir": "control-flow",
    "ttir/crafted_generic_form.ttir": "reader-misalignment",
    "ttir/crafted_same_dest.ttir": "control-flow",
    "ttir/crafted_symbols_strings.ttir": "call",
    "ttir/golden_early_return_loaded_sm80.ttir": "control-flow",
    "ttir/golden_early_return_pid_sm80.ttir": "control-flow",
    "ttir/golden_gather_sm80.ttir": "indirect-address",
    "ttir/golden_gather_sm90.ttir": "indirect-address",
    "ttir/golden_guard_then_loop_sm80.ttir": "control-flow",
    "ttir/golden_loop_under_if_sm80.ttir": "control-flow",
    # integer offsets carried by the loop (rewritten block pointers)
    "ttir/golden_matmul_bp_s3_sm80.ttir": "loop-variant-advance",
    "ttir/golden_matmul_bp_s3_sm90.ttir": "loop-variant-advance",
    "ttir/golden_matmul_tma_s1_sm90.ttir": "out-of-vocabulary",
    "ttir/golden_matmul_tma_s3_sm90.ttir": "out-of-vocabulary",
    "ttir/golden_matmul_tma_ws_s3_sm90.ttir": "out-of-vocabulary",
    "ttir/golden_nested_guard_merge_sm80.ttir": "control-flow",
    "ttir/golden_nested_loops_sm80.ttir": "nested-loop",
    "ttir/golden_sequential_loops_sm80.ttir": "nested-loop",
    "ttir/spike_early_return.ttir": "control-flow",
    "ttir/spike_early_return_loop.ttir": "control-flow",
    "ttir/spike_inline_asm.ttir": "inline-asm",
    "ttir/spike_nested_for.ttir": "nested-loop",
    "ttir/spike_noinline_call.ttir": "call",
    "ttir/spike_spin_while.ttir": "control-flow",
    "reader_ttir/p1_variant_delta.ttir": "loop-variant-advance",
    "reader_ttir/loop_observed_advance.ttir": "loop-variant-advance",
    "reader_ttir/p2_swap.ttir": "loop-variant-advance",
    "reader_ttir/p3_call_guarded.ttir": "call",
    "reader_ttir/p3_call_offset.ttir": "call",
    "reader_ttir/p3_call_formals.ttir": "call",
    "reader_ttir/rv_inline_asm_store.ttir": "inline-asm",
    "reader_ttir/int_iterarg_offset.ttir": "loop-variant-advance",
    "reader_ttir/pure_asm_int_addr.ttir": "inline-asm",
    "reader_ttir/observed_lanes.ttir": "indirect-address",
}


def _tokens(term, graph, canon: dict) -> tuple:
    """Flat pre-order tokens of a term from either reader: IntCast and the
    D9 widths dropped, make_range sites renamed by first appearance, the
    (single) loop's identity dropped, and an IterArgOffset replaced by its
    base, offset0 and delta. Iterative (kernel_deep_chain is deep)."""
    out: list[tuple] = []
    stack = [term]
    while stack:
        t = stack.pop()
        if t is None:
            out.append(("None",))
            continue
        name = type(t).__name__
        if name == "IntCast":
            stack.append(t.x)
        elif name in ("Bin", "BoolBin"):
            out.append((name, t.op))
            stack += [t.b, t.a]
        elif name == "Cmp":
            out.append((name, t.pred))
            stack += [t.b, t.a]
        elif name == "Select":
            out.append((name,))
            stack += [t.f, t.t, t.cond]
        elif name == "Not":
            out.append((name,))
            stack.append(t.a)
        elif name == "Const":
            out.append((name, t.value))
        elif name in ("Pid", "NumPrograms"):
            out.append((name, t.axis))
        elif name == "Arange":
            out.append(
                (name, canon.setdefault(t.ssa, len(canon)), t.start, t.end, t.dim)
            )
        elif name == "Param":
            out.append((name, t.name))
        elif name == "LoopVar":
            out.append((name,))
        elif name == "IterArgOffset":
            info = graph.iter_args[t.arg_id]
            out.append((name, info.base_param))
            stack += [info.delta, info.offset0]
        elif name == "Observed":
            out.append((name, t.access_index))
        elif name == "DataDep":
            out.append((name,))
        else:
            raise AssertionError(f"unexpected term {name}")
    return tuple(out)


def _normalize(graph) -> dict:
    """What both readers must agree on, as plain comparable data. Left out:
    FuncArg.int_bits (new), and elem_float of non-atomic accesses, which
    the new reader sets from the pointee on every access (#361: atomics
    only)."""
    canon: dict = {}

    def tok(t):
        return _tokens(t, graph, canon)

    events = [
        {
            "kind": a.kind,
            "base": a.base_param,
            "offset": tok(a.offset),
            "mask": tok(a.mask),
            "path": tok(a.path),
            "in_loop": a.in_loop,
            "atomic": None
            if a.atomic is None
            else (a.atomic.rmw_op, a.atomic.sem, a.atomic.scope),
            "elem_bits": a.elem_bits,
            "loc": None if a.loc is None else (a.loc.file, a.loc.line, a.loc.col),
            "line_no": a.line_no,
            "guarded": a.guarded,
            "mask_dropped": a.mask_dropped,
            "atomic_val": tok(a.atomic_val),
            "atomic_cmp": tok(a.atomic_cmp),
            "elem_float": a.elem_float if a.atomic is not None else None,
        }
        for a in graph.accesses
    ]
    loop = graph.loop
    return {
        "kernel": graph.kernel_name,
        "args": [
            (a.name, a.is_ptr, a.elem_bits, a.elem_float) for a in graph.func_args
        ],
        "pid_axes": sorted(graph.pid_axes),
        "loop": None
        if loop is None
        else [tok(loop.lower), tok(loop.upper), tok(loop.step)],
        "n_iter_args": len(graph.iter_args),
        "events": events,
    }


def _diff(mine: dict, theirs: dict) -> set[str]:
    """The normalized fields that differ: ``events[i].<field>`` per event
    when both have the same number of events, else top-level keys."""
    out = {k for k in mine if k != "events" and mine[k] != theirs[k]}
    if len(mine["events"]) != len(theirs["events"]):
        return out | {"events"}
    for i, (a, b) in enumerate(zip(mine["events"], theirs["events"])):
        out |= {f"events[{i}].{k}" for k in a if a[k] != b[k]}
    return out


def _run(reader, text: str):
    try:
        return reader.parse_ttir(text), None
    except reader.UnsupportedTTIR as e:
        return None, e


@pytest.mark.parametrize("name", FILES)
def test_differential_oracle(name):
    text = _text(name)
    mine, mine_refusal = _run(R, text)
    theirs, theirs_refusal = _run(O, text)
    # the new reader's outcome is pinned
    assert (None if mine_refusal is None else mine_refusal.kind) == REFUSED.get(name)
    reason, differs = EXPECTED_DIFF.get(name, ("", frozenset()))
    if mine is not None and theirs is not None:
        # exactly the listed fields differ; everything else still matches
        assert _diff(_normalize(mine), _normalize(theirs)) == differs, reason
    elif mine is None and theirs is None:
        assert not differs, reason
    else:
        outcome = _NEW if mine is None else _OLD
        assert (
            outcome == differs
        ), f"{name}: new {mine_refusal!r} vs #361 {theirs_refusal!r} ({reason})"


# How the reader of a later release reads a base golden it prints itself
# (the base text, printed by 3.6, shadowed by its own): name -> refusal kind,
# where it differs from the release's own printing.
BASE_UNDER: dict[str, dict[str, str]] = {
    "3.8": {
        # 3.6's gpu.barrier: 3.8's frontend emits ttg.barrier
        "ttir/adv_zero_result.ttir": "out-of-vocabulary",
        "ttir/spike_misc.ttir": "out-of-vocabulary",
        # the 3.6 !tt.tensordesc spelling
        "ttir/adv_descs.ttir": "unparsable",
        "ttir/golden_matmul_tma_s1_sm90.ttir": "unparsable",
        "ttir/golden_matmul_tma_s3_sm90.ttir": "unparsable",
        "ttir/golden_matmul_tma_ws_s3_sm90.ttir": "unparsable",
    },
}


def test_shadowed_base_goldens_read_as_pinned():
    """The installed release reads each base golden it prints itself as its
    own printing reads (REFUSED), but where BASE_UNDER pins the difference."""
    shadowed = [n for n in FILES if G.printed_by(_path(n)) != G.BASE_RELEASE]
    for name in shadowed:
        got = _run(R, (GOLDEN / name).read_text(encoding="utf-8"))[1]
        want = BASE_UNDER.get(G.RELEASE, {}).get(name, REFUSED.get(name))
        assert (None if got is None else got.kind) == want, (name, got)
        if want == "out-of-vocabulary" and name in BASE_UNDER.get("3.8", {}):
            assert "op gpu.barrier is not TTIR" in got.message
    # a release other than the base one reads its own printings
    assert G.RELEASE == G.BASE_RELEASE or shadowed


def test_base_under_names_shadowed_goldens():
    for release, table in BASE_UNDER.items():
        for name in table:
            d, n = name.split("/")
            assert (G.own_dir(d, release) / n).is_file(), (release, name)


def test_oracle_corpus_coverage():
    # the tables name real goldens, and most files are compared event by event
    assert set(EXPECTED_DIFF) <= set(FILES) and set(REFUSED) <= set(FILES)
    both = [
        f
        for f in FILES
        if f not in REFUSED and f not in EXPECTED_DIFF and _run(O, _text(f))[0]
    ]
    assert len(both) >= 45
    assert sum(len(_graph(f).accesses) for f in both) >= 120


def test_oracle_is_the_verbatim_361_reader():
    source = (Path(__file__).parent / "_oracle_ttir_reader_361.py").read_bytes()
    body = source.split(b"\n", 5)[5]
    assert hashlib.sha256(body).hexdigest() == (
        "3a82e07d3dc2aae16f2d3098894e411c04779ae61c2f787eb2cb15278ff7a4df"
    )


# ─────────────────────────── the audit probes ───────────────────────────


def test_p1_loop_variant_advance_refuses():
    e = _refusal("reader_ttir/p1_variant_delta.ttir")
    assert e.kind is TTIRKind.LOOP_VARIANT_ADVANCE and "loop-variant" in e.message
    assert _REFUSAL_SITE["p1_variant_delta"][G.RELEASE] in _source_line(e.loc)
    # #361 accepted it with a LoopVar advance (sanitizer: false 'ok')
    g = O.parse_ttir(_text("reader_ttir/p1_variant_delta.ttir"))
    assert isinstance(g.iter_args[0].delta, O.LoopVar)


def test_p2_swapped_iter_args_refuse():
    e = _refusal("reader_ttir/p2_swap.ttir")
    assert (
        e.kind is TTIRKind.LOOP_VARIANT_ADVANCE
        and "not advanced from itself" in e.message
    )
    g = O.parse_ttir(_text("reader_ttir/p2_swap.ttir"))
    assert [i.delta for i in g.iter_args.values()] == [O.Const(0), O.Const(0)]


def test_observation_inside_the_loop_is_a_variant_advance():
    e = _refusal("reader_ttir/loop_observed_advance.ttir")
    assert e.kind is TTIRKind.LOOP_VARIANT_ADVANCE
    assert _REFUSAL_SITE["loop_observed_advance"][G.RELEASE] in _source_line(e.loc)


@pytest.mark.parametrize(
    "name", ["p3_call_guarded", "p3_call_offset", "p3_call_formals"]
)
def test_p3_calls_refuse(name):
    e = _refusal(f"reader_ttir/{name}.ttir")
    assert e.kind is TTIRKind.CALL and "_p3_helper" in e.message


def test_quoted_callee_and_a_second_function_refuse_as_call():
    e = _refusal("ttir/crafted_symbols_strings.ttir")
    assert e.kind is TTIRKind.CALL and "'f{%x} \"q\" (a)'" in e.message
    # a second tt.func refuses even when nothing calls it
    extra = (
        "  tt.func private @h(%q: !tt.ptr<f32>) attributes {noinline = true} {\n"
        "    tt.return\n  }\n"
    )
    e = _refusal(_module("", extra=extra))
    assert e.kind is TTIRKind.CALL and "'h'" in e.message


def test_p4_walkers_resolve_loop_carried_pointers():
    direct = _graph("reader_ttir/p4_observed_direct.ttir")
    loop = _graph("reader_ttir/p4_observed_loop.ttir")
    delta = _graph("reader_ttir/p4_observed_delta.ttir")
    for g in (direct, loop, delta):
        load = next(a for a in g.accesses if a.kind == "load")
        assert mentions_observed(load.offset, g)
        assert observed_indices(load.offset, g) == {0}
    # the address reaches the observation only through the iter_arg (its
    # offset0, resp. its delta): #361's term-local walker misses it
    for g, part, name in ((loop, "offset0", "loop"), (delta, "delta", "delta")):
        load = next(a for a in g.accesses if a.kind == "load")
        assert isinstance(load.offset, IterArgOffset)
        assert mentions_observed(getattr(g.iter_args[0], part), g)
        theirs = O.parse_ttir(_text(f"reader_ttir/p4_observed_{name}.ttir"))
        theirs_load = next(a for a in theirs.accesses if a.kind == "load")
        assert not O.mentions_observed(theirs_load.offset)


def test_walkers_descend_datadep_keep():
    g = AccessGraph("k", (), (), None)
    kept = DataDep("bool op over loaded data", keep=Cmp("eq", Observed(3), Const(0)))
    assert mentions_observed(kept, g) and observed_indices(kept, g) == {3}
    assert not mentions_observed(DataDep("loaded value"), g)


def test_rv_inline_asm_refuses():
    e = _refusal("reader_ttir/rv_inline_asm_store.ttir")
    assert e.kind is TTIRKind.INLINE_ASM and "side effects" in e.message
    assert _REFUSAL_SITE["rv_inline_asm_store"][G.RELEASE] in _source_line(e.loc)
    # a pure asm with a pointer operand is still an address handed to asm
    text = _text("ttir/spike_inline_asm.ttir").replace("pure = false", "pure = true")
    e = _refusal(text)
    assert e.kind is TTIRKind.INLINE_ASM and "pointer operand" in e.message
    # a pure asm over data is plain data
    text = "\n".join(
        line
        for line in _text("ttir/spike_inline_asm.ttir").splitlines()
        if "pure = false" not in line
    )
    assert [a.kind for a in parse_ttir(text).accesses] == ["load"]


# ─────────────────────────── the phase-2 review probes ───────────────────────────
# ir_mode_audit/probes_phase2/ttir-reader/: compiled kernels (now goldens in
# reader_ttir/) and hand-written TTIR.


def _footprint(g, access, params: dict, iters: int) -> set[int]:
    """The modeled element offsets of ``access`` over iterations
    ``0 .. iters-1`` with every arange lane free, the lanes keyed by
    (make_range, dim) as a consumer keys them; mask applied."""
    lanes = sorted(
        {
            (n.ssa, n.dim, n.start, n.end)
            for n in R._nodes((access.offset, access.mask), g.iter_args)
            if isinstance(n, Arange)
        }
    )

    def ev(t, env):
        if isinstance(t, Const):
            return t.value
        if isinstance(t, Param):
            return env[t.name]
        if isinstance(t, Arange):
            return env[(t.ssa, t.dim)]
        if isinstance(t, IterArgOffset):
            info = g.iter_args[t.arg_id]
            return ev(info.offset0, env) + env["k"] * ev(info.delta, env)
        if isinstance(t, IntCast):
            return ev(t.x, env)
        a, b = ev(t.a, env), ev(t.b, env)
        if isinstance(t, Cmp):
            return int({"slt": a < b}[t.pred])
        return {"+": a + b, "-": a - b, "*": a * b}[t.op]

    out = set()
    for k in range(iters):
        for values in itertools.product(*(range(s, e) for _, _, s, e in lanes)):
            env = dict(params, k=k)
            env.update({(ssa, dim): v for (ssa, dim, _, _), v in zip(lanes, values)})
            if access.mask is None or ev(access.mask, env):
                out.add(ev(access.offset, env))
    return out


def test_expanded_loop_carried_tiles_keep_their_lanes():
    # k1: [N, N] pointer tile expanded to 3D in the loop, next to the same
    # make_range at dim 0 (#361 kept the tile's dims 0/1: a 4-offset footprint)
    g = _graph("reader_ttir/expand_iterarg_3d.ttir")
    load = g.accesses[0]
    assert load.kind == "load" and load.in_loop
    (tile,) = [
        n for n in R._nodes((load.offset,), None) if isinstance(n, IterArgOffset)
    ]
    expanded = g.iter_args[tile.arg_id]
    assert expanded.base_param == "x_ptr" and expanded.delta == Const(1)
    n = 4
    want = {
        j * n + m - i * n + k
        for i, j, m in itertools.product(range(n), repeat=3)
        for k in range(2)
    }
    assert _footprint(g, load, {}, 2) == want  # [-12, 16], every OOB offset kept
    # k2: 1D pointer expanded to 2D, masked at its own lane (dim 1)
    g = _graph("reader_ttir/expand_iterarg_mask.ttir")
    load = g.accesses[0]
    assert load.offset == IterArgOffset(1) and g.iter_args[1].delta == Const(4)
    assert _footprint(g, load, {"M": 2}, 2) == {0, 1, 4, 5}


def test_tensor_observations_do_not_meet_across_lanes():
    # k8: c[:, None] - c[None, :] over one tensor atomic's old values
    e = _refusal("reader_ttir/observed_lanes.ttir")
    assert e.kind is TTIRKind.INDIRECT_ADDRESS and "across lanes" in e.message
    assert "c[:, None] - c[None, :]" in _source_line(e.loc)
    # #361 wrote both lanes as one symbol, offset (0 + X) + (0 - X) with the
    # same X: every lane stores to offset 0, a false in-bounds
    off = O.parse_ttir(_text("reader_ttir/observed_lanes.ttir")).accesses[-1].offset
    assert off.b.op == "-" and off.a.b == off.b.b
    assert O.mentions_observed(off.a.b)
    # an expanded tile of a pointer that advances by per-lane observations
    body = """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
        %ps = tt.splat %p : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
        %q = tt.addptr %ps, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
        %one = arith.constant dense<1> : tensor<4xi32>
        %old = tt.atomic_rmw add, acq_rel, gpu, %q, %one : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> tensor<4xi32>
        %res = scf.for %k = %c0 to %n step %c1 iter_args(%a = %q) -> (tensor<4x!tt.ptr<i32>>)  : i32 {
          %e = tt.expand_dims %a {axis = 0 : i32} : tensor<4x!tt.ptr<i32>> -> tensor<1x4x!tt.ptr<i32>>
          %v = tt.load %e : tensor<1x4x!tt.ptr<i32>>
          %a2 = tt.addptr %a, %old : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
          scf.yield %a2 : tensor<4x!tt.ptr<i32>>
        }"""
    e = _refusal(_module(body, args="%p: !tt.ptr<i32>, %n: i32"))
    assert e.kind is TTIRKind.OTHER and "per-lane atomic result" in e.message


def test_loop_carried_integers_make_addresses_loop_variant():
    # k3: offs += B carried by the loop
    e = _refusal("reader_ttir/int_iterarg_offset.ttir")
    assert (
        e.kind is TTIRKind.LOOP_VARIANT_ADVANCE and "carried by the loop" in e.message
    )
    # H4: a pointer advanced by a loop-carried integer
    body = """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%a = %p, %s = %c0) -> (!tt.ptr<f32>, i32)  : i32 {
          %v = tt.load %a : !tt.ptr<f32>
          %a2 = tt.addptr %a, %s : !tt.ptr<f32>, i32
          %s2 = arith.addi %s, %c1 : i32
          scf.yield %a2, %s2 : !tt.ptr<f32>, i32
        }"""
    assert _refusal(_module(body)).kind is TTIRKind.LOOP_VARIANT_ADVANCE


def test_an_address_handed_to_opaque_ops_as_an_integer_refuses():
    # k6: tl.inline_asm_elementwise(..., is_pure=True) given x_ptr.to(tl.int64)
    e = _refusal("reader_ttir/pure_asm_int_addr.ttir")
    assert e.kind is TTIRKind.INLINE_ASM and "tt.ptr_to_int" in e.message
    # through arithmetic with loaded data, and through a loop-carried value
    args = "%p: !tt.ptr<i64>, %n: i32"
    head = """
        %x = tt.load %p : !tt.ptr<i64>
        %i = tt.ptr_to_int %p : !tt.ptr<i64> -> i64
        %a = arith.addi %i, %x : i64"""
    asm = (
        '%y = tt.elementwise_inline_asm "mov.b64 $0, $1;" {constraints = "=l,l", '
        "packed_element = 1 : i32, pure = true} %a : i64 -> i64"
    )
    assert _refusal(_module(head + "\n" + asm, args=args)).kind is TTIRKind.INLINE_ASM
    loop = """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %z = arith.constant 0 : i64
        %i = tt.ptr_to_int %p : !tt.ptr<i64> -> i64
        %a = scf.for %k = %c0 to %n step %c1 iter_args(%s = %z) -> (i64)  : i32 {
          %s2 = arith.addi %s, %i : i64
          scf.yield %s2 : i64
        }"""
    assert _refusal(_module(loop + "\n" + asm, args=args)).kind is TTIRKind.INLINE_ASM
    extern = (
        '%y = tt.extern_elementwise %a {libname = "", libpath = "", pure = true, '
        'symbol = "f"} : (i64) -> i64'
    )
    e = _refusal(_module(head + "\n" + extern, args=args))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "tt.ptr_to_int" in e.message
    # loaded data carries no address: an asm over it stays plain data
    data = head.replace("%i, %x", "%x, %x") + "\n" + asm
    assert [a.kind for a in parse_ttir(_module(data, args=args)).accesses] == ["load"]


def test_llvm_and_gpu_ops_outside_the_inert_ones_refuse():
    # H1a-c: memory effects through ops of the llvm dialect
    cases = {
        "llvm.inttoptr": """
            %i = tt.ptr_to_int %p : !tt.ptr<i32> -> i64
            %lp = llvm.inttoptr %i : i64 to !llvm.ptr<1>
            %c = arith.constant 7 : i32
            %old = llvm.atomicrmw add %lp, %c monotonic : !llvm.ptr<1>, i32""",
        "llvm.inline_asm": """
            %i = tt.ptr_to_int %p : !tt.ptr<i32> -> i64
            %c = arith.constant 7 : i32
            %r = llvm.inline_asm has_side_effects "st.global.b32 [$1], $2; mov.b32 $0, 0;", "=r,l,r" %i, %c : (i64, i32) -> i32""",
    }
    for name, body in cases.items():
        e = _refusal(_module(body, args="%p: !tt.ptr<i32>, %n: i32"))
        assert e.kind is TTIRKind.OUT_OF_VOCABULARY and name in e.message
    # inside a combine region too
    body = """
        %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
        %s = "tt.reduce"(%r) <{axis = 0 : i32}> ({
        ^bb0(%a: i32, %b: i32):
          %i = tt.ptr_to_int %p : !tt.ptr<i32> -> i64
          %lp = llvm.inttoptr %i : i64 to !llvm.ptr<1>
          %y = arith.addi %a, %b : i32
          tt.reduce.return %y : i32
        }) : (tensor<4xi32>) -> i32"""
    e = _refusal(_module(body, args="%p: !tt.ptr<i32>, %n: i32"))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "llvm.inttoptr" in e.message
    # the inert ones stay inert: the release's barrier (3.8's ttg.barrier is
    # the one ttg op accepted); 3.8 still parses gpu.barrier, which its
    # frontend never emits: refused
    barrier, other = _BARRIERS[G.RELEASE]
    g = parse_ttir(_module(f"{barrier}\n%v = tt.load %p : !tt.ptr<f32>"))
    assert [a.kind for a in g.accesses] == ["load"]
    e = _refusal(_module(f"{other}\n%v = tt.load %p : !tt.ptr<f32>"))
    assert e.kind is _OTHER_BARRIER[G.RELEASE], e.message
    if G.RELEASE == "3.8":
        assert "op gpu.barrier is not TTIR" in e.message
        e = _refusal(_module("ttg.local_barrier\n%v = tt.load %p : !tt.ptr<f32>"))
        assert e.kind in (TTIRKind.UNPARSABLE, TTIRKind.OUT_OF_VOCABULARY)


# release -> (the barrier tl.debug_barrier() prints, the other release's)
_BARRIERS = {
    "3.6": ("gpu.barrier", "ttg.barrier all"),
    "3.8": ("ttg.barrier all", "gpu.barrier"),
}
# what the other release's barrier gets: 3.6 has no ttg dialect
_OTHER_BARRIER = {"3.6": TTIRKind.UNPARSABLE, "3.8": TTIRKind.OUT_OF_VOCABULARY}


def test_pointers_outside_global_memory_refuse():
    # H8: a shared-memory (address space 3) pointer argument
    e = _refusal(
        _module(
            "%c1 = arith.constant 1 : i32\ntt.store %p, %c1 : !tt.ptr<i32, 3>",
            args="%p: !tt.ptr<i32, 3>, %n: i32",
        )
    )
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "address space" in e.message
    # a pointer made into another address space
    body = """
        %c = arith.constant 0 : i64
        %q = tt.int_to_ptr %c : i64 -> !tt.ptr<f32, 3>
        %v = tt.load %q : !tt.ptr<f32, 3>"""
    e = _refusal(_module(body))
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "address space" in e.message
    assert e.line_no == 4  # the tt.int_to_ptr


def test_generic_load_reads_its_mask_by_operand_segment():
    # G1: generic tt.load with `other` and no mask; an i1 `other` is no mask
    load = (
        '%v = "tt.load"(%p, %f) <{boundaryCheck = array<i32>, cache = 1 : i32, '
        "evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: SEG>}> "
        ": (!tt.ptr<i1>, i1) -> i1"
    )
    for segments, mask in (("1, 0, 1", None), ("1, 1, 0", Const(0))):
        g = parse_ttir(
            _module(
                "%f = arith.constant false\n" + load.replace("SEG", segments),
                args="%p: !tt.ptr<i1>, %n: i32",
            )
        )
        (access,) = g.accesses
        assert (access.mask, access.mask_dropped) == (mask, False), segments


# ─────────────────────────── modeled shapes ───────────────────────────


def test_two_step_advance_is_the_sum():
    g = _graph("reader_ttir/loop_two_step_advance.ttir")
    (info,) = g.iter_args
    assert (info.base_param, info.offset0, info.delta) == (
        "x_ptr",
        Const(0),
        Bin("+", Param("s"), Const(2)),
    )
    assert g.accesses[0].offset == IterArgOffset(0) and g.accesses[0].in_loop


def test_same_base_pointer_selects():
    g = _graph("reader_ttir/where_pointer.ttir")
    (store,) = g.accesses
    assert store.base_param == "x_ptr" and isinstance(store.offset, Select)
    g = _graph("ttir/spike_if_yield.ttir")
    store = g.accesses[-1]
    assert store.kind == "store" and store.base_param == "out_ptr"
    assert isinstance(store.offset, Select) and isinstance(store.offset.cond, Cmp)


def test_three_dims_of_one_make_range_are_three_lanes():
    (store,) = _graph("reader_ttir/tile3d_shared_arange.ttir").accesses
    ranges = [n for n in R._nodes((store.offset,), None) if isinstance(n, Arange)]
    assert len({r.ssa for r in ranges}) == 1
    assert sorted(r.dim for r in ranges) == [0, 1, 2]


def test_same_width_pointer_bitcast_keeps_the_base():
    g = _graph("ttir/golden_atomic_fmax_sm80.ttir")
    atomics = [a for a in g.accesses if a.kind == "atomic_rmw"]
    assert [a.atomic.rmw_op for a in atomics] == ["max", "umin"]
    assert all(
        a.base_param == "out_ptr" and a.elem_bits == 32 and not a.elem_float
        for a in atomics
    )
    # the atomics' mask reads loaded data
    assert all(a.mask_dropped and a.mask is None for a in atomics)
    # a bitcast to another element width changes what an element offset means
    body = """
        %q = tt.bitcast %p : !tt.ptr<f32> -> !tt.ptr<f16>
        %v = tt.load %q : !tt.ptr<f16>"""
    e = _refusal(_module(body))
    assert e.kind is TTIRKind.OTHER and "element width" in e.message


# The block-pointer case per release: 3.6 has the ops and types; 3.8 has
# neither (its parser aborts on the type), so the walk refuses the text
# before parsing.
_BLOCK_POINTER_KIND = {"3.6": "block-pointer", "3.8": "unparsable"}


def test_other_refusals():
    cases = {
        _BLOCK_POINTER_KIND[G.RELEASE]: """
            %c0 = arith.constant 0 : i32
            %c1 = arith.constant 1 : i64
            %c32 = arith.constant 32 : i64
            %bp = tt.make_tensor_ptr %p, [%c32, %c32], [%c32, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<32x32xf32>>
            %v = tt.load %bp : !tt.ptr<tensor<32x32xf32>>""",
        "out-of-vocabulary": """
            %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
            %v = arith.sitofp %r : tensor<64xi32> to tensor<64xf32>
            %s = "tt.reduce"(%v) <{axis = 0 : i32}> ({
            ^bb0(%a: f32, %b: f32):
              %x = tt.load %p : !tt.ptr<f32>
              %y = arith.addf %a, %x : f32
              tt.reduce.return %y : f32
            }) : (tensor<64xf32>) -> f32""",
        "control-flow": "%c0 = arith.constant 0 : i32\n"
        "%b = arith.cmpi slt, %n, %c0 : i32\n"
        'cf.assert %b, "neg"',
    }
    for kind, body in cases.items():
        assert _refusal(_module(body)).kind == kind, kind
    if G.RELEASE == "3.8":
        e = _refusal(_module(cases["unparsable"]))
        assert "no block pointers" in e.message and e.line_no == 7
    e = _refusal(
        _module(
            "%x = tt.load %p : !tt.ptr<f32>\n%y = tt.extern_elementwise %x "
            '{libname = "", libpath = "", pure = false, symbol = "foo"} : (f32) -> f32'
        )
    )
    assert e.kind is TTIRKind.OUT_OF_VOCABULARY and "'foo'" in e.message
    # deep structured nesting refuses instead of exhausting the recursion limit
    depth = 250
    nest = (
        "".join("scf.if %c {\n" for _ in range(depth))
        + "tt.store %p, %f : !tt.ptr<f32>\n"
        + "}\n" * depth
    )
    e = _refusal(
        _module(
            "%f = arith.constant 1.0 : f32\n" + nest, args="%p: !tt.ptr<f32>, %c: i1"
        )
    )
    assert e.kind is TTIRKind.OTHER and "nested deeper" in e.message


def test_pure_extern_and_signed_i1_compare_are_data():
    g = parse_ttir(
        _module(
            "%x = tt.load %p : !tt.ptr<f32>\n%y = tt.extern_elementwise %x "
            '{libname = "", libpath = "", pure = true, symbol = "__nv_expf"} : (f32) -> f32\n'
            "tt.store %p, %y : !tt.ptr<f32>"
        )
    )
    assert [a.kind for a in g.accesses] == ["load", "store"]
    # a signed compare of i1 reads true as -1: not the boolean model, so the
    # mask is dropped (widened), never misread
    g = parse_ttir(
        _module(
            """
        %c0 = arith.constant 0 : i32
        %b = arith.cmpi slt, %n, %c0 : i32
        %t = arith.constant true
        %c = arith.cmpi slt, %b, %t : i1
        %v = tt.load %p, %c : !tt.ptr<f32>"""
        )
    )
    (load,) = g.accesses
    assert load.mask is None and load.mask_dropped


def test_matmul_loop_iter_args_and_params():
    g = _graph("ttir/golden_matmul_s3_sm80.ttir")
    assert g.kernel_name == "matmul_kernel" and g.loop is not None
    assert [i.base_param for i in g.iter_args] == ["a_ptr", "b_ptr"]
    assert all(
        i.arg_id == k and i.loop_ssa == g.loop.loop_ssa
        for k, i in enumerate(g.iter_args)
    )
    assert Param("K") in list(R._nodes((g.loop.upper,), None))
    loads = [a for a in g.accesses if a.kind == "load"]
    assert [a.offset for a in loads] == [IterArgOffset(0), IterArgOffset(1)]
    assert all(a.in_loop for a in loads) and not g.accesses[-1].in_loop
    assert g.loop.bits == 32 and not g.loop.unsigned and g.loop.line_no is not None
    assert g.arg("K").int_bits == 32 and g.arg("a_ptr").elem_bits == 16


# ─────────────────────────── integer widths (D9) ───────────────────────────


def _value(t, env: dict) -> int:
    """The unbounded-integer reading of a (shallow) term: casts as identity."""
    if isinstance(t, Const):
        return t.value
    if isinstance(t, Pid):
        return env["pid"]
    if isinstance(t, Param):
        return env[t.name]
    if isinstance(t, IntCast):
        return _value(t.x, env)
    if isinstance(t, Cmp):
        a, b = _value(t.a, env), _value(t.b, env)
        return int({"slt": a < b, "ult": a < b, "eq": a == b}[t.pred])
    if isinstance(t, Bin):
        a, b = _value(t.a, env), _value(t.b, env)
        ops = {"+": a + b, "*": a * b, "-": a - b}
        if t.op in ("u//", "//"):
            return int(a / b)
        if t.op == "%":
            return a - b * int(a / b)
        return ops[t.op]
    raise AssertionError(type(t).__name__)


def _holds(ob, env: dict) -> bool:
    v = _value(ob.term, env)
    if ob.signed:
        return -(1 << (ob.bits - 1)) <= v < (1 << (ob.bits - 1))
    return 0 <= v < (1 << ob.bits)


def _integer_nodes(g):
    for a in g.accesses:
        yield from R._nodes((a.offset, a.mask, a.path), g.iter_args)
    if g.loop is not None:
        yield from R._nodes((g.loop.lower, g.loop.upper, g.loop.step), None)


@pytest.mark.parametrize("name", [f for f in FILES if f not in REFUSED])
def test_every_integer_op_carries_its_width(name):
    for n in _integer_nodes(_graph(name)):
        if isinstance(n, Bin):
            # bits=None only for the element-offset sums of tt.addptr
            assert isinstance(n.bits, int) or (n.bits is None and n.op == "+")
        elif isinstance(n, Cmp):
            assert isinstance(n.bits, int) and n.bits >= 1
        elif isinstance(n, IntCast):
            assert n.kind in ("trunci", "extsi", "extui") and n.src_bits != n.dst_bits


def test_i32_wrap_obligations():
    g = _graph("reader_ttir/rv_i32_wrap.ttir")
    (store,) = g.accesses
    inner = Bin("*", Pid(0), Param("S"), 32)
    outer = Bin("*", inner, Param("S"), 32)
    assert store.offset == Bin("+", Const(0), outer)
    obs = width_obligations(g, store)
    assert [(o.term, o.bits, o.signed) for o in obs] == [
        (outer, 32, True),
        (inner, 32, True),
    ]
    assert all("pid * S" in _source_line(o.loc) for o in obs)
    text = _text("reader_ttir/rv_i32_wrap.ttir").splitlines()
    assert all("arith.muli" in text[o.line_no - 1] for o in obs)
    # (pid * 65536) * 65536 is 0 in i32: the unbounded model is exact only for pid 0
    assert all(_holds(o, {"pid": 0, "S": 65536}) for o in obs)
    assert [_holds(o, {"pid": 1, "S": 65536}) for o in obs] == [False, True]


def test_trunci_obligation():
    g = _graph("reader_ttir/rv_trunci_alias.ttir")
    (store,) = g.accesses
    wide = Bin("*", IntCast("extsi", 32, 64, Pid(0)), Const(1 << 32), 64)
    assert store.offset == Bin("+", Const(0), IntCast("trunci", 64, 32, wide))
    obs = width_obligations(g, store)
    assert [(o.term, o.bits, o.signed) for o in obs] == [
        (wide, 32, True),
        (wide, 64, True),
    ]
    trunc = obs[0]
    assert (
        "arith.trunci"
        in _text("reader_ttir/rv_trunci_alias.ttir").splitlines()[trunc.line_no - 1]
    )
    assert ".to(tl.int32)" in _source_line(trunc.loc)
    # trunc_i32(pid * 2**32) is 0 for every pid: exact only for pid 0
    assert _holds(trunc, {"pid": 0}) and not _holds(trunc, {"pid": 1})
    assert _holds(obs[1], {"pid": 1})


def test_unsigned_reads_need_non_negative_operands():
    g = _graph("reader_ttir/unsigned_index.ttir")
    (store,) = g.accesses
    quotient = Bin("u//", Pid(0), Const(3), 32)
    assert store.offset == Bin("+", Const(0), IntCast("extui", 32, 64, quotient))
    assert store.mask == Cmp("ult", Pid(0), Param("n"), 32)
    obs = {(o.term, o.bits, o.signed) for o in width_obligations(g, store)}
    assert obs == {
        (quotient, 32, False),  # the extui operand
        (quotient, 32, True),  # the divui result
        (Pid(0), 32, False),
        (Const(3), 32, False),
        (Param("n"), 32, False),  # cmpi ult reads n unsigned
    }
    obs = width_obligations(g, store)
    assert all(_holds(o, {"pid": 5, "n": 8}) for o in obs)
    # n = -1 reads as 2**32 - 1 under ult: the unbounded model is not exact
    assert not all(_holds(o, {"pid": 5, "n": -1}) for o in obs)


def test_narrowing_casts_and_extui_in_spike_casts():
    g = _graph("ttir/spike_casts.ttir")
    load = g.accesses[0]
    casts = [n for n in R._nodes((load.offset,), None) if isinstance(n, IntCast)]
    assert {(c.kind, c.src_bits, c.dst_bits) for c in casts} == {
        ("trunci", 32, 16),
        ("extsi", 16, 64),
        ("trunci", 32, 8),
        ("extui", 8, 64),
    }
    obs = [
        (o.bits, o.signed) for o in width_obligations(g, load) if o.line_no is not None
    ]
    assert (16, True) in obs and (8, True) in obs and (8, False) in obs


def test_i1_casts_and_unsigned_loops():
    # extsi of an i1 maps true to -1: read as 0 - extui(b)
    g = parse_ttir(
        _module(
            """
        %c0 = arith.constant 0 : i32
        %b = arith.cmpi slt, %n, %c0 : i32
        %e = arith.extsi %b : i1 to i32
        %q = tt.addptr %p, %e : !tt.ptr<f32>, i32
        %v = tt.load %q : !tt.ptr<f32>"""
        )
    )
    b = Cmp("slt", Param("arg1"), Const(0), 32)
    assert g.accesses[0].offset == Bin(
        "+", Const(0), Bin("-", Const(0), IntCast("extui", 1, 32, b), 32)
    )
    # trunci to i1 keeps only bit 0: exact for 0 / 1
    g = parse_ttir(
        _module("%b = arith.trunci %n : i32 to i1\n%v = tt.load %p, %b : !tt.ptr<f32>")
    )
    ((ob,),) = [width_obligations(g, a) for a in g.accesses]
    assert (ob.term, ob.bits, ob.signed) == (Param("arg1"), 1, False)
    # an unsigned loop compare needs non-negative bounds
    g = parse_ttir(
        _module(
            """
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        scf.for unsigned %i = %c0 to %n step %c1  : i32 {
          %q = tt.addptr %p, %i : !tt.ptr<f32>, i32
          %v = tt.load %q : !tt.ptr<f32>
        }"""
        )
    )
    assert g.loop.unsigned and g.loop.bits == 32
    obs = [
        (o.term, o.bits, o.signed, o.line_no)
        for o in width_obligations(g, g.accesses[0])
    ]
    latch = Bin("+", Bin("-", Param("arg1"), Const(1), 32), Const(1), 32)
    assert obs == [
        (Const(0), 32, False, 5),
        (Param("arg1"), 32, False, 5),
        (Const(1), 32, False, 5),
        (latch, 32, False, 5),  # the increment does not wrap, read unsigned
    ]


def test_loop_increment_obligation():
    # k4: for k in range(lo, n, 1 << 20) wraps its induction variable when
    # n is near INT32_MAX (the GPU then runs iterations at negative k)
    g = _graph("reader_ttir/iv_wrap.ttir")
    (store,) = g.accesses
    latch = Bin("+", Bin("-", Param("n"), Const(1), 32), Const(1 << 20), 32)
    obs = width_obligations(g, store)
    assert [(o.term, o.bits, o.signed, o.role) for o in obs] == [
        (latch, 32, True, "loop")
    ]
    assert "range(lo, n, 1 << 20)" in _source_line(obs[0].loc)
    assert (
        "scf.for" in _text("reader_ttir/iv_wrap.ttir").splitlines()[obs[0].line_no - 1]
    )
    assert _holds(obs[0], {"n": 1000})
    assert not _holds(obs[0], {"n": (1 << 31) - (1 << 19)})
    # an access outside the loop gets no loop obligations
    g = _graph("ttir/golden_matmul_s3_sm80.ttir")
    assert not any(o.role == "loop" for o in width_obligations(g, g.accesses[-1]))
    assert any(o.role == "loop" for o in width_obligations(g, g.accesses[0]))


def test_signed_remainder_needs_its_quotient_to_fit():
    # H9: remsi(INT_MIN, -1) is undefined, though its value 0 fits
    body = """
        %cm = arith.constant -2147483648 : i32
        %cn = arith.constant -1 : i32
        %r = arith.remsi %cm, %cn : i32
        %q = tt.addptr %p, %r : !tt.ptr<f32>, i32
        %v = tt.load %q : !tt.ptr<f32>"""
    g = parse_ttir(_module(body))
    (load,) = g.accesses
    lo = Const(-(1 << 31))
    obs = width_obligations(g, load)
    assert [(o.term, o.bits, o.signed, o.line_no) for o in obs] == [
        (Bin("%", lo, Const(-1), 32), 32, True, 5),
        (Bin("//", lo, Const(-1), 32), 32, True, 5),
    ]
    assert _holds(obs[0], {}) and not _holds(obs[1], {})


def test_obligation_roles():
    # a node the path, mask and offset share is listed once, under the most
    # restrictive role (path, then mask, then offset)
    body = """
        %c4 = arith.constant 4 : i32
        %pid = tt.get_program_id x : i32
        %o = arith.muli %pid, %c4 : i32
        %s = arith.subi %n, %c4 : i32
        %b = arith.cmpi slt, %o, %s : i32
        scf.if %b {
          %t = arith.addi %o, %c4 : i32
          %m = arith.cmpi slt, %t, %n : i32
          %o2 = arith.addi %t, %c4 : i32
          %q = tt.addptr %p, %o2 : !tt.ptr<f32>, i32
          %v = tt.load %q, %m : !tt.ptr<f32>
        }"""
    g = parse_ttir(_module(body))
    (load,) = g.accesses
    o = Bin("*", Pid(0), Const(4), 32)
    s = Bin("-", Param("arg1"), Const(4), 32)
    t = Bin("+", o, Const(4), 32)
    o2 = Bin("+", t, Const(4), 32)
    assert [(ob.term, ob.role) for ob in width_obligations(g, load)] == [
        (o, "path"),
        (s, "path"),
        (t, "mask"),
        (o2, "offset"),
    ]


def test_obligation_sites_do_not_affect_term_equality():
    a = Bin("+", Pid(0), Const(1), 32, line_no=3, loc=W.SourceLoc("a.py", 1, 1))
    b = Bin("+", Pid(0), Const(1), 32, line_no=9, loc=None)
    assert a == b and hash(a) == hash(b) and repr(a) == repr(b)
    assert Bin("+", Pid(0), Const(1), 64) != a  # the width is part of the value


def test_deep_terms_stay_iterative():
    g = _graph("ttir/kernel_deep_chain.ttir")
    (store,) = g.accesses
    assert sum(1 for n in R._nodes((store.offset,), None) if isinstance(n, Bin)) > 1000
    obs = width_obligations(g, store)
    assert len(obs) > 1000 and all(o.signed and o.bits == 32 for o in obs)
    assert not mentions_observed(store.offset, g)
    # pickle round-trips such a graph; the generated ==, hash (and repr,
    # deepcopy) recurse, as the module docstring says
    assert _fingerprint(pickle.loads(pickle.dumps(g))) == _fingerprint(g)
    again = _graph("ttir/kernel_deep_chain.ttir").accesses[0].offset
    # at Python's default limit: importing tilelens.visualizer.draw (e.g. from
    # tests/unit/test_trace_io.py at collection) raises it process-wide
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        with pytest.raises(RecursionError):
            hash(store.offset)
        with pytest.raises(RecursionError):
            _ = store.offset == again
    finally:
        sys.setrecursionlimit(limit)


# ─────────────────────────── frozen, deterministic graphs ───────────────────────────


_FROZEN_LEAVES = (int, str, float, bool, type(None), TTIRKind)


def _assert_deeply_frozen(obj) -> None:
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, _FROZEN_LEAVES):
            continue
        if isinstance(x, (tuple, frozenset)):
            stack.extend(x)
        elif dataclasses.is_dataclass(x):
            assert type(x).__dataclass_params__.frozen, type(x).__name__
            stack.extend(getattr(x, f.name) for f in dataclasses.fields(x))
        else:
            raise AssertionError(f"mutable {type(x).__name__} in the graph")


def _fingerprint(obj) -> str:
    """Every field (the compare=False sites included), iteratively."""
    out: list[str] = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if dataclasses.is_dataclass(x):
            out.append(type(x).__name__)
            stack.extend(reversed([getattr(x, f.name) for f in dataclasses.fields(x)]))
        elif isinstance(x, (tuple, frozenset)):
            items = sorted(x) if isinstance(x, frozenset) else list(x)
            out.append(f"{type(x).__name__}{len(items)}")
            stack.extend(reversed(items))
        else:
            out.append(repr(x))
    return hashlib.sha256("\x00".join(out).encode()).hexdigest()


ACCEPTED = [f for f in FILES if f not in REFUSED]


def test_graphs_are_frozen():
    g = _graph("ttir/golden_matmul_s3_sm80.ttir")
    for obj, attr in (
        (g, "accesses"),
        (g.accesses[0], "offset"),
        (g.accesses[0].offset, "arg_id"),
        (g.iter_args[0], "delta"),
        (g.loop, "upper"),
        (g.func_args[0], "name"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, None)
    for name in ACCEPTED:
        _assert_deeply_frozen(_graph(name))
    # hand-built graphs are coerced to immutable containers too
    built = AccessGraph("k", [], [], None, iter_args=[], pid_axes={0})
    assert (built.func_args, built.accesses, built.iter_args, built.pid_axes) == (
        (),
        (),
        (),
        frozenset({0}),
    )


def test_parse_is_deterministic():
    for name in ACCEPTED:
        first = _fingerprint(_graph(name))
        W._CACHE.clear()  # a fresh bindings parse
        assert _fingerprint(_graph(name)) == first, name


SMALL = [
    "ttir/golden_matmul_s3_sm80.ttir",
    "ttir/spike_atomics.ttir",
    "reader_ttir/p4_observed_loop.ttir",
]


def test_graphs_hash_pickle_and_copy():
    for name in SMALL:
        g = _graph(name)
        W._CACHE.clear()
        again = _graph(name)
        assert g == again and hash(g) == hash(again)
        assert pickle.loads(pickle.dumps(g)) == g
        assert copy.deepcopy(g) == g
        assert _fingerprint(pickle.loads(pickle.dumps(g))) == _fingerprint(g)


def test_parse_is_deterministic_across_processes():
    script = (
        "import hashlib, pickle, sys\n"
        "from tilelens.ir.ttir_reader import parse_ttir\n"
        "for p in sys.argv[1:]:\n"
        "    g = parse_ttir(open(p, encoding='utf-8').read())\n"
        "    print(hashlib.sha256(pickle.dumps(g, protocol=4)).hexdigest())\n"
    )
    paths = [str(_path(n)) for n in SMALL]
    runs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", script, *paths],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        runs.append(out.stdout.split())
    here = [
        hashlib.sha256(pickle.dumps(_graph(n), protocol=4)).hexdigest() for n in SMALL
    ]
    assert runs[0] == runs[1] == here
