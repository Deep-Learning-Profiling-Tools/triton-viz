"""Walk a printed TTIR module: MLIR bindings for structure, the text for attributes.

Private to ``tilelens.ir``; only ``ttir_reader.py`` imports it. This is the one
module that touches Triton's private MLIR bindings (``triton._C.libtriton.ir``),
and every lifetime hazard of those bindings stays here (D10a, amendment D7+).

``walk_module(text)`` reads the same bytes twice:

1. The bindings parse the text (``ir.parse_mlir_module``, TTIR dialects only):
   op names, operand / result values, full type strings, region / block
   nesting, block arguments, value locs, and the few attributes the 3.6
   getters can read (``get_str_attr`` / ``get_bool_attr`` /
   ``get_flat_symbol_ref_attr``).
2. A line scan of the text rebuilds the same op tree and recovers what the
   bindings keep opaque: integer and enum attributes, constant values, cf
   successors, and the locs of zero-result ops.

The two trees are zipped in pre-order and must agree on every op: name,
result / operand / region / block / block-argument counts, every SSA edge
(each printed use resolves, through region-scoped name tables, to the value
the bindings report for that operand slot), every value loc, every
bindings-readable attribute, the type-constrained integer attributes, the cf
successor arities and the printer's ``// pred:`` comments. Text-only enums
must lie in a closed per-op vocabulary, other recovered attributes must have
their expected Python type, and a module that prints locs must print one on
every op (zero-result op locs have no second source). The only tolerated
differences are the printer's own elisions: a trailing zero-operand
``scf.yield`` of an ``scf.for`` / ``scf.if`` block (also when that yield is
the block's only op and the region prints as ``{ }``), unprinted empty
trailing regions, and the default-valued attributes listed in the
release's ``Printer.defaults``. Anything else raises
``MisalignedModule``; a text the MLIR parser rejects raises
``ModuleParseError`` with the parser's own diagnostic.

The result is frozen pure-Python data: ``Module`` holds ``Op`` / ``Block`` /
``Value`` / ``Func`` records, which compare by value, hash, pickle and
deep-copy. Values are dense local ints (indices into ``Module.values``),
never the bindings' ``value.id()`` pointers. No binding object ever leaves
``_bind_walk``.

Binding lifetime (spike condition 2): the context is pinned on the module
(``mod.context = ctx``), everything is extracted inside one function, the
module's body block is erased, the module is dropped before the context, and
nothing derived from either is returned. No binding erases the module op
itself, so each parse still leaks that (empty) op; results are cached by
sha256 of the text (condition 4). The parse window redirects the process's
fd 2 under a lock; a forked child gets both back.

The text vocabulary is per Triton minor release: ``PRINTERS`` maps a
release ("3.6", "3.8") to its ``Printer`` table (what the reader needs,
the leading keywords and their closed vocabularies, the elided defaults,
the printed operand orders, the attributes the bindings can read, what the
dialect's types allow), each audited against that release's printer. The
installed Triton's release selects the table (``printer()``); a release
without one fails closed (``UnknownTritonRelease``), never borrowing
another release's table. The syntax recognizers below are shared by every
table's release (audited per release too); a release whose printer spells
a construct differently needs its own row. ``Module.release`` records the
table a module was read with.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import os
import re
import sys
import tempfile
import threading
import types
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

# ─────────────────────────── records ───────────────────────────


class _FrozenMap(Mapping[str, Any]):
    """The read-only mapping behind the records' mapping fields. Unlike
    ``MappingProxyType`` it hashes (its values are plain hashable data) and
    pickles, so the frozen records hash, pickle and deep-copy too."""

    __slots__ = ("_d",)

    def __init__(self, items: Mapping[str, Any]) -> None:
        self._d = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        return hash(frozenset(self._d.items()))

    def __repr__(self) -> str:
        return repr(self._d)

    def __reduce__(self) -> tuple[Any, ...]:
        return (_FrozenMap, (self._d,))


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int


@dataclass(frozen=True)
class Value:
    index: int  # position in Module.values
    type: str  # full type text: "tensor<64x!tt.ptr<f32>>", "i32", ...
    op: int | None  # defining op index (an op result), else None
    block: int | None  # owning block index (a block argument), else None
    position: int  # result number, or argument number
    # NameLoc name of the value's own loc (the Python variable / parameter
    # name), None when unnamed. Printed SSA names are never exposed: the
    # printer sanitises and uniquifies them.
    name: str | None


@dataclass(frozen=True)
class Block:
    index: int  # position in Module.blocks
    op: int  # owning op index
    region: int  # region number within the owning op
    position: int  # block number within the region
    label: str | None  # printed label ("^bb1"); None for an unlabeled entry block
    args: tuple[int, ...]  # value indices
    arg_types: tuple[str, ...]
    arg_names: tuple[str | None, ...]  # NameLoc names (see Value.name)
    ops: tuple[int, ...]  # op indices in program order


@dataclass(frozen=True)
class Op:
    index: int  # pre-order position in Module.ops (the text order)
    name: str  # "tt.load", "scf.for", "builtin.module", ...
    operands: tuple[int, ...]  # value indices, in ODS operand order
    operand_types: tuple[str, ...]
    results: tuple[int, ...]  # value indices
    result_types: tuple[str, ...]
    # Recovered attributes (read-only). Integer attrs are ints, enums their
    # printed keyword ("slt", "acq_rel", "ieee"); program-id axes are ints;
    # arith.constant "value" is an int (the signed value, as the printer
    # prints signless integers) / bool, ("float", literal) for a float, or
    # ("dense", literal) for a non-splat dense (then "splat" is False; a
    # dense splat has "splat" True and a scalar "value"); cf ops carry
    # "successors" as destination block indices.
    attrs: Mapping[str, Any]
    regions: tuple[tuple[int, ...], ...]  # block indices, per walked region
    # block indices from the module body down to the parent block; () for
    # the module op itself
    path: tuple[int, ...]
    position: int  # position within the parent block
    line_no: int | None  # header line (1-based); None for an elided terminator
    end_line: int | None  # closing line of a region op, else == line_no
    loc: SourceLoc | None  # the op's own site (callee frame of a callsite loc)
    callers: tuple[SourceLoc, ...]  # callsite chain, innermost caller first
    loc_name: str | None  # NameLoc label of the op's loc, if any
    implicit: bool = False  # a terminator the printer elided

    @property
    def block(self) -> int | None:
        return self.path[-1] if self.path else None


@dataclass(frozen=True)
class FuncArg:
    index: int
    value: int  # value index of the entry-block argument
    type: str
    name: str | None  # NameLoc name: the Python parameter name
    attrs: Mapping[str, Any]  # printed argument attributes (tt.divisibility, ...)


@dataclass(frozen=True)
class Func:
    op: int  # the tt.func op index
    sym_name: str
    visibility: str
    args: tuple[FuncArg, ...]  # () for a body-less declaration


@dataclass(frozen=True)
class Module:
    ops: tuple[Op, ...]
    blocks: tuple[Block, ...]
    values: tuple[Value, ...]
    funcs: tuple[Func, ...]  # tt.func ops in text order
    stats: Mapping[str, int]  # what the alignment checked (tests / bulk tool)
    release: str  # the Triton release whose Printer table read the text ("3.6")


class MisalignedModule(Exception):
    """The text scan and the bindings disagree (or the text holds a construct
    the text layer cannot read faithfully). ``problems`` lists every mismatch
    found, ``line_no`` is the first text line involved (None if unknown)."""

    def __init__(self, problems: Sequence[str], line_no: int | None = None) -> None:
        self.problems = tuple(problems) or ("misaligned module",)
        self.line_no = line_no
        super().__init__(self.problems[0])


class ModuleParseError(Exception):
    """The MLIR parser rejected the text (``diagnostic`` is its own message,
    with the temporary parse path replaced by ``<ttir>``), the parse input
    could not be created, or the text holds a construct the release's parser
    cannot be handed (``Printer.block_pointer_types``)."""

    def __init__(self, diagnostic: str, line_no: int | None = None) -> None:
        self.diagnostic = diagnostic
        self.line_no = line_no
        super().__init__(diagnostic)


class UnknownTritonRelease(Exception):
    """The installed Triton's minor release has no ``Printer`` table: the
    text layer does not know that release's printer, so nothing is read
    (``release`` is the minor release, ``version`` the full version)."""

    def __init__(self, release: str, version: str) -> None:
        self.release = release
        self.version = version
        known = ", ".join(f"{r}.x" for r in PRINTERS)
        self.message = (
            f"the TTIR walk layer has no printer table for Triton {version} "
            f"(it reads the printers of Triton {known}); a release is added "
            "to tilelens.ir._mlir_walk.PRINTERS after auditing its printer"
        )
        super().__init__(self.message)


# ─────────────────────────── types ───────────────────────────


@dataclass(frozen=True)
class TypeInfo:
    """A printed TTIR type split into shape and element (D9 widths)."""

    text: str
    shape: tuple[int, ...]  # () for scalars
    elem: str  # element type text: "i32", "f16", "!tt.ptr<f32>", ...
    int_bits: int | None  # iN element -> N (signless); index -> 64
    float_bits: int | None
    pointee: str | None  # element pointer -> pointee type text
    pointee_bits: int | None
    block_ptr: bool  # !tt.ptr<tensor<...>>


_FLOAT_BITS = {
    "f64": 64, "f32": 32, "f16": 16, "bf16": 16, "tf32": 32,
    "f8E4M3FN": 8, "f8E5M2": 8, "f8E4M3FNUZ": 8, "f8E5M2FNUZ": 8,
    "f8E4M3B11FNUZ": 8, "f8E8M0FNU": 8, "f4E2M1FN": 4,
}  # fmt: skip
_RE_TENSOR = re.compile(r"^tensor<((?:\d+x)*)(.*)>$")
_RE_INT = re.compile(r"^i(\d+)$")
_RE_PTR = re.compile(r"^!tt\.ptr<(.*?)(?:, \d+)?>$")


def _scalar_bits(t: str) -> int | None:
    m = _RE_INT.match(t)
    if m:
        return int(m.group(1))
    return _FLOAT_BITS.get(t)


def parse_type(text: str) -> TypeInfo:
    shape: tuple[int, ...] = ()
    elem = text
    m = _RE_TENSOR.match(text)
    if m:
        shape = tuple(int(d) for d in m.group(1).split("x") if d)
        elem = m.group(2)
    im = _RE_INT.match(elem)
    pm = _RE_PTR.match(elem)
    pointee = pm.group(1) if pm else None
    int_bits = int(im.group(1)) if im else (64 if elem == "index" else None)
    return TypeInfo(
        text=text,
        shape=shape,
        elem=elem,
        int_bits=int_bits,
        float_bits=_FLOAT_BITS.get(elem),
        pointee=pointee,
        pointee_bits=_scalar_bits(pointee) if pointee else None,
        block_ptr=bool(pointee and pointee.startswith("tensor<")),
    )


# ─────────────────────────── strings and brackets ───────────────────────────


def _string_end(s: str, start: int) -> int:
    """Index of the quote closing the string literal that opens at ``start``."""
    i = start + 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i
        i += 1
    raise ValueError("unterminated string literal")


_HEX = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09}


def _unescape(body: str) -> str:
    """Decode an MLIR string literal body. The printer escapes every
    non-printable or non-ASCII byte as ``\\XX``; the escapes of one character
    are the bytes of its UTF-8 encoding, so collect bytes and decode once."""
    out = bytearray()
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out += c.encode("utf-8")
            i += 1
            continue
        nxt = body[i + 1 : i + 2]
        if nxt in _SIMPLE_ESCAPES:
            out.append(_SIMPLE_ESCAPES[nxt])
            i += 2
        elif len(body) >= i + 3 and body[i + 1] in _HEX and body[i + 2] in _HEX:
            out.append(int(body[i + 1 : i + 3], 16))
            i += 3
        else:
            raise ValueError(f"bad string escape {body[i : i + 3]!r}")
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"string literal is not UTF-8: {e}") from None


def _mask(line: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Replace every string literal's body by '_' (same length, so indices into
    the masked line index the raw line too). Returns the masked line and the
    literals as (open quote index, close quote index, decoded text)."""
    out: list[str] = []
    lits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            j = _string_end(line, i)
            lits.append((i, j, _unescape(line[i + 1 : j])))
            out.append('"' + "_" * (j - i - 1) + '"')
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out), lits


_OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _match_close(s: str, i: int) -> int:
    """Index of the bracket closing ``s[i]`` in masked text ('->' is no '>')."""
    stack = [_OPEN[s[i]]]
    j = i + 1
    while j < len(s):
        c = s[j]
        if c in _OPEN:
            stack.append(_OPEN[c])
        elif c == ">" and s[j - 1] == "-":
            pass
        elif c in ")]}>":
            if c != stack[-1]:
                raise ValueError(f"bracket mismatch at column {j + 1}")
            stack.pop()
            if not stack:
                return j
        j += 1
    raise ValueError("unclosed bracket")


def _split_top(s: str, sep: str = ",") -> list[tuple[int, str]]:
    """Split masked text on ``sep`` at bracket depth 0. Returns (start index of
    the stripped part in ``s``, stripped part) for every non-empty part."""
    parts: list[tuple[int, str]] = []
    depth = 0
    begin = 0
    for i, c in enumerate(s):
        if c in "([{<":
            depth += 1
        elif c in ")]}" or (c == ">" and (i == 0 or s[i - 1] != "-")):
            depth -= 1
        elif c == sep and depth == 0:
            parts.append((begin, s[begin:i]))
            begin = i + 1
    parts.append((begin, s[begin:]))
    out = []
    for start, p in parts:
        if p.strip():
            out.append((start + len(p) - len(p.lstrip()), p.strip()))
    return out


def _trailing_loc(masked: str) -> tuple[int, int] | None:
    """(start, end) of the ``loc(...)`` that ends the masked text, if any."""
    s = masked.rstrip()
    if not s.endswith(")"):
        return None
    depth = 0
    for j in range(len(s) - 1, -1, -1):
        c = s[j]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                if s[max(0, j - 3) : j] == "loc" and (
                    j < 4 or not (s[j - 4].isalnum() or s[j - 4] in "_.$")
                ):
                    return j - 3, len(s)
                return None
    return None


def _blank_dicts(masked: str) -> str:
    """Masked text with every top-level ``{...}`` replaced by spaces."""
    out = list(masked)
    i = 0
    while i < len(masked):
        if masked[i] == "{":
            j = _match_close(masked, i)
            out[i : j + 1] = " " * (j + 1 - i)
            i = j
        i += 1
    return "".join(out)


# ─────────────────────────── locs ───────────────────────────
# One grammar for both sources: the text's `loc(#locN)` trailers resolved
# through the `#locN = loc(...)` table, and the bindings' fully inlined
# `str(value.get_loc())`. Both normalise to nested tuples compared with ==.

_RE_LOC_ALIAS_DEF = re.compile(r"^(#loc\d*)\s*=\s*loc\((.*)\)\s*$")
_RE_LOC_ALIAS = re.compile(r"#loc\d*")
_RE_FILE_POS = re.compile(r":(\d+):(\d+)")
_LOC_DEPTH_LIMIT = 128


class _LocParser:
    def __init__(self, aliases: Mapping[str, str], parse_path: str) -> None:
        self._aliases = aliases  # "#loc12" -> inner text of loc(...)
        self._parse_path = parse_path
        self._alias_memo: dict[str, tuple] = {}
        self._text_memo: dict[str, tuple | None] = {}

    def parse(self, loc_text: str) -> tuple | None:
        """Normalised tree of a ``loc(...)`` text; None for a loc the parser
        invented (it names the temporary parse path: the text printed none)."""
        got = self._text_memo.get(loc_text, _MISSING)
        if got is not _MISSING:
            return got  # type: ignore[return-value]
        s = loc_text.strip()
        if not (s.startswith("loc(") and s.endswith(")")):
            raise ValueError(f"not a loc: {loc_text[:80]!r}")
        tree: tuple | None = self._full(s[4:-1], 0)
        if _names_path(tree, self._parse_path):  # type: ignore[arg-type]
            tree = None
        self._text_memo[loc_text] = tree
        return tree

    def _full(self, s: str, depth: int) -> tuple:
        tree, rest = self._expr(s.strip(), depth)
        if rest.strip():
            raise ValueError(f"trailing loc text: {rest[:60]!r}")
        return tree

    def _expr(self, s: str, depth: int) -> tuple[tuple, str]:
        if depth > _LOC_DEPTH_LIMIT:
            raise ValueError("loc nesting too deep")
        if s.startswith("#loc"):
            m = _RE_LOC_ALIAS.match(s)
            assert m is not None
            name = m.group(0)
            if name not in self._alias_memo:
                if name not in self._aliases:
                    raise ValueError(f"undefined loc alias {name}")
                self._alias_memo[name] = ("pending",)
                self._alias_memo[name] = self._full(self._aliases[name], depth + 1)
            elif self._alias_memo[name] == ("pending",):
                raise ValueError(f"cyclic loc alias {name}")
            return self._alias_memo[name], s[m.end() :]
        if s.startswith("unknown"):
            return ("unknown",), s[len("unknown") :]
        if s.startswith("callsite("):
            callee, rest = self._expr(s[len("callsite(") :].lstrip(), depth + 1)
            rest = rest.lstrip()
            if not rest.startswith("at "):
                raise ValueError("callsite loc without 'at'")
            caller, rest = self._expr(rest[3:].lstrip(), depth + 1)
            rest = rest.lstrip()
            if not rest.startswith(")"):
                raise ValueError("unclosed callsite loc")
            return ("callsite", callee, caller), rest[1:]
        if s.startswith("fused"):
            rest = s[len("fused") :]
            meta: str | None = None
            if rest.startswith("<"):
                close = _match_close(_mask(rest)[0], 0)
                meta = rest[1:close]
                rest = rest[close + 1 :]
            if not rest.startswith("["):
                raise ValueError("bad fused loc")
            parts = []
            rest = rest[1:].lstrip()
            while not rest.startswith("]"):
                p, rest = self._expr(rest, depth + 1)
                parts.append(p)
                rest = rest.lstrip()
                if rest.startswith(","):
                    rest = rest[1:].lstrip()
                elif not rest.startswith("]"):
                    raise ValueError("bad fused loc list")
            return ("fused", meta, tuple(parts)), rest[1:]
        if s.startswith('"'):
            end = _string_end(s, 0)
            text = _unescape(s[1:end])
            rest = s[end + 1 :]
            m = _RE_FILE_POS.match(rest)
            if m:
                rest = rest[m.end() :]
                if rest.lstrip().startswith("to"):
                    raise ValueError("file range locs are not supported")
                return ("file", text, int(m.group(1)), int(m.group(2))), rest
            if rest.startswith("("):
                child, rest = self._expr(rest[1:], depth + 1)
                rest = rest.lstrip()
                if not rest.startswith(")"):
                    raise ValueError("unclosed name loc")
                return ("name", text, child), rest[1:]
            return ("name", text, None), rest
        raise ValueError(f"unrecognized loc: {s[:60]!r}")


_MISSING = object()


def _names_path(tree: tuple, path: str) -> bool:
    """Does any file loc inside ``tree`` name ``path``? (iterative)"""
    stack = [tree]
    while stack:
        t = stack.pop()
        if t is None:
            continue
        kind = t[0]
        if kind == "file":
            if t[1] == path:
                return True
        elif kind == "name":
            stack.append(t[2])
        elif kind == "callsite":
            stack += [t[1], t[2]]
        elif kind == "fused":
            stack += list(t[2])
    return False


def _loc_site(
    tree: tuple | None,
) -> tuple[SourceLoc | None, tuple[SourceLoc, ...], str | None]:
    """(site, callers, name) of a normalised loc. The callee frame of a
    callsite is the op's site (a memory op belongs to the callee, #361
    _LocTable); the caller chain follows, innermost first. Recursion depth is
    bounded by the parser's ``_LOC_DEPTH_LIMIT``."""
    if tree is None:
        return None, (), None
    kind = tree[0]
    if kind == "file":
        return SourceLoc(tree[1], tree[2], tree[3]), (), None
    if kind == "name":
        site, callers, _ = _loc_site(tree[2])
        return site, callers, tree[1]
    if kind == "callsite":
        site, callers, name = _loc_site(tree[1])
        csite, ccallers, _ = _loc_site(tree[2])
        return site, callers + ((csite,) if csite else ()) + ccallers, name
    if kind == "fused":
        for part in tree[2]:
            got = _loc_site(part)
            if got[0] is not None:
                return got
    return None, (), None


# ─────────────────────────── text scan ───────────────────────────

_RE_RESULTS = re.compile(
    r"^((?:%[-\w.$]+(?::\d+)?)(?:\s*,\s*%[-\w.$]+(?::\d+)?)*)\s*=\s*"
)
_RE_USE = re.compile(r"%[-\w.$]+(?:#\d+)?")
_RE_OPNAME = re.compile(r"^[A-Za-z_][\w$.]*")
_RE_LABEL = re.compile(r"^\^[-\w.$]+")
_RE_SUCC = re.compile(r"\^[-\w.$]+")
_RE_WORD_BEFORE = re.compile(r"([A-Za-z_]\w*)\s*$")
_RE_DEF_EQ = re.compile(r"\s*=(?!=)")
_RE_BARE_ASSIGN = re.compile(
    r"(?<![\w.$%])([A-Za-z_][\w.$]*)\s*=(?!=)\s*([A-Za-z_]\w*|-?\d+)?"
)
_RE_PREDS = re.compile(
    r"^(?:pred:\s*(\^[-\w.$]+)|(\d+)\s+preds:\s*(.*)|no predecessors)$"
)


class _TextError(Exception):
    def __init__(self, line_no: int | None, msg: str) -> None:
        super().__init__(msg)
        self.line_no = line_no
        self.msg = msg


@dataclass
class _ArgDef:
    """A block argument printed in a label, a func header or an op header."""

    name: str  # "%x"
    type: str | None  # printed type text (labels / func headers)
    loc: str | None  # printed "loc(...)" text
    attrs: dict[str, Any]  # printed argument attrs (func headers)


@dataclass
class _TBlock:
    label: str | None
    args: list[_ArgDef]
    ops: list["_TOp"]
    line_no: int
    # the printer's predecessor comment: labels (a multiset); None if absent
    preds: tuple[str, ...] | None = None


@dataclass
class _TOp:
    name: str
    line_no: int
    raw: str  # stripped raw header line (comment removed)
    masked: str  # masked header line (same length)
    lits: list[tuple[int, int, str]]
    name_end: int  # index just past the op name
    hdr_end: int  # end of the header operands/attrs (before loc / region opener)
    result_names: list[str]
    n_results: int
    opens_region: bool
    regions: list[list[_TBlock]]
    close_line: int | None = None
    close_raw: str = ""
    close_masked: str = ""
    close_lits: list[tuple[int, int, str]] | None = None


@dataclass
class _TextTree:
    root: _TOp
    aliases: dict[str, str]
    pred_comments: bool  # any block label carries a predecessor comment


def _parse_results(prefix: str) -> tuple[list[str], int]:
    names: list[str] = []
    n = 0
    for tok in prefix.split(","):
        tok = tok.strip()
        if ":" in tok:
            base, k = tok.split(":")
            names += [f"{base}#{i}" for i in range(int(k))]
            n += int(k)
        else:
            names.append(tok)
            n += 1
    return names, n


def _arg_def(
    part_raw: str, part_masked: str, lits_rel: list[tuple[int, int, str]]
) -> _ArgDef:
    """``%x: type {attrs} loc(...)`` (masked part + raw part, same indices)."""
    colon = part_masked.find(":")
    if not part_masked.startswith("%") or colon < 0:
        raise ValueError(f"bad block argument {part_raw[:60]!r}")
    name = part_masked[:colon].strip()
    rest_m = part_masked[colon + 1 :]
    rest_r = part_raw[colon + 1 :]
    loc = None
    span = _trailing_loc(rest_m)
    if span is not None:
        loc = rest_r[span[0] : span[1]]
        rest_m, rest_r = rest_m[: span[0]], rest_r[: span[0]]
    attrs: dict[str, Any] = {}
    brace = rest_m.find("{")
    if brace >= 0:
        close = _match_close(rest_m, brace)
        attrs = _parse_dict(
            rest_m,
            brace,
            close,
            [(a - colon - 1, b - colon - 1, t) for a, b, t in lits_rel],
        )
        if rest_m[close + 1 :].strip():
            raise ValueError(f"text after argument attrs: {part_raw[:60]!r}")
        rest_r = rest_r[:brace]
    return _ArgDef(name, rest_r.strip(), loc, attrs)


def _arg_list(
    raw: str, masked: str, lits: list[tuple[int, int, str]], open_idx: int
) -> tuple[list[_ArgDef], int]:
    """Parse the parenthesised argument list opening at ``open_idx``; returns
    the defs and the index of the closing paren."""
    close = _match_close(masked, open_idx)
    inner_m = masked[open_idx + 1 : close]
    defs = []
    for start, part in _split_top(inner_m):
        a = open_idx + 1 + start
        rel = [(x - a, y - a, t) for x, y, t in lits if a <= x < a + len(part)]
        defs.append(_arg_def(raw[a : a + len(part)], part, rel))
    return defs, close


def _pred_comment(comment: str, line_no: int) -> tuple[str, ...]:
    m = _RE_PREDS.match(comment.strip())
    if m is None:
        raise _TextError(line_no, f"unrecognized block comment {comment[:60]!r}")
    if m.group(1):
        return (m.group(1),)
    if m.group(2):
        preds = tuple(p.strip() for p in m.group(3).split(","))
        if len(preds) != int(m.group(2)) or not all(
            _RE_LABEL.fullmatch(p) for p in preds
        ):
            raise _TextError(line_no, f"malformed predecessor comment {comment[:60]!r}")
        return preds
    return ()


def _parse_op_line(
    ln: int, raw: str, masked: str, lits: list[tuple[int, int, str]]
) -> _TOp:
    rm = _RE_RESULTS.match(masked)
    result_names: list[str] = []
    n_results = 0
    start = 0
    if rm:
        result_names, n_results = _parse_results(rm.group(1))
        start = rm.end()
    body = masked[start:]
    if body.startswith('"'):  # generic form: "tt.reduce"(...)
        end = _string_end(raw, start)
        name = next((t for a, _b, t in lits if a == start), "")
        name_end = end + 1
    else:
        nm = _RE_OPNAME.match(body)
        if nm is None:
            raise _TextError(ln, f"cannot read an op name: {raw[:80]!r}")
        name = nm.group(0)
        name_end = start + nm.end()
        if "." not in name:
            name = f"builtin.{name}"
    opens = masked.endswith("{")
    hdr_end = len(masked)
    if opens:
        hdr_end = masked.rfind("{")
    else:
        span = _trailing_loc(masked)
        if span is not None:
            hdr_end = span[0]
    return _TOp(
        name,
        ln,
        raw,
        masked,
        lits,
        name_end,
        hdr_end,
        result_names,
        n_results,
        opens,
        [],
    )


def _scan_text(text: str) -> _TextTree:
    """Rebuild the op tree from printed lines. Returns a synthetic file-level
    op whose single region holds the top-level ops, and the ``#loc`` table."""
    root = _TOp("<file>", 0, "", "", [], 0, 0, [], 0, True, [[]])
    stack = [root]
    aliases: dict[str, str] = {}
    pred_comments = False
    for ln, raw_line in enumerate(text.splitlines(), start=1):
        raw = raw_line.strip()
        if not raw or raw.startswith("//"):
            continue
        try:
            masked, lits = _mask(raw)
        except ValueError as e:
            raise _TextError(ln, str(e)) from None
        comment = None
        cut = masked.find("//")
        if cut >= 0:
            comment = raw[cut + 2 :].strip()
            raw, masked = raw[:cut].rstrip(), masked[:cut].rstrip()
            lits = [x for x in lits if x[1] < cut]
        top = stack[-1]
        if masked.startswith("#"):
            m = _RE_LOC_ALIAS_DEF.match(raw)
            if len(stack) != 1 or m is None:
                raise _TextError(
                    ln,
                    f"attribute alias {raw[:60]!r}: only #loc aliases are TTIR (TTGIR input?)",
                )
            aliases[m.group(1)] = m.group(2)
            continue
        if comment is not None and not masked.startswith("^"):
            raise _TextError(ln, f"unexpected comment {comment[:60]!r}")
        try:
            if masked.startswith("^"):
                if top is root:
                    raise _TextError(ln, "block label outside any region")
                lm = _RE_LABEL.match(masked)
                if lm is None:
                    raise _TextError(ln, f"bad block label {raw[:60]!r}")
                after = masked[lm.end() :]
                args: list[_ArgDef] = []
                if after.startswith("("):
                    args, close = _arg_list(raw, masked, lits, lm.end())
                    after = masked[close + 1 :]
                if after.strip() != ":":
                    raise _TextError(ln, f"bad block label {raw[:60]!r}")
                preds = None
                if comment is not None:
                    preds = _pred_comment(comment, ln)
                    pred_comments = True
                top.regions[-1].append(_TBlock(lm.group(0), args, [], ln, preds))
                continue
            if masked.startswith("}"):
                if top is root:
                    raise _TextError(ln, "unbalanced '}'")
                rest = masked[1:].lstrip()
                if rest.endswith("{"):  # "} else {", "} do {", "}, {"
                    top.regions.append([])
                    continue
                if rest.startswith(")"):  # generic op closer "}) ..."
                    rest = rest[1:].lstrip()
                off = len(masked) - len(rest)
                top.close_line = ln
                top.close_raw = raw[off:]
                top.close_masked = rest
                top.close_lits = [(a - off, b - off, t) for a, b, t in lits if a >= off]
                stack.pop()
                continue
            op = _parse_op_line(ln, raw, masked, lits)
        except ValueError as e:
            raise _TextError(ln, str(e)) from None
        region = top.regions[-1]
        if not region:
            region.append(_TBlock(None, [], [], ln))
        region[-1].ops.append(op)
        if op.opens_region:
            op.regions = [[]]
            stack.append(op)
    if len(stack) != 1:
        raise _TextError(
            stack[-1].line_no,
            f"region opened at line {stack[-1].line_no} is never closed",
        )
    return _TextTree(root, aliases, pred_comments)


# ─────────────────────────── attribute recovery (text) ───────────────────────────


def _attr_value(v: str, v_start: int, lits: list[tuple[int, int, str]]) -> Any:
    if v.startswith('"'):
        return next((t for a, _b, t in lits if a == v_start), None)
    m = re.fullmatch(r"(-?\d+)(?:\s*:\s*(?:i\d+|index))?", v)
    if m:
        return int(m.group(1))
    if v in ("true", "false"):
        return v == "true"
    m = re.fullmatch(r"array<i\d+(?::\s*(.*))?>", v)
    if m:
        return tuple(int(x) for x in (m.group(1) or "").split(",") if x.strip())
    m = re.fullmatch(r"dense<(-?\d+)>\s*:\s*tensor<.*>", v)
    if m:
        return ("splat", int(m.group(1)))
    return v  # anything else stays raw text


def _parse_dict(
    masked: str, open_idx: int, close_idx: int, lits: list[tuple[int, int, str]]
) -> dict[str, Any]:
    d: dict[str, Any] = {}
    inner = masked[open_idx + 1 : close_idx]
    for start, part in _split_top(inner):
        p_start = open_idx + 1 + start
        eq = _split_top(part, "=")
        if len(eq) == 2:
            (_k_off, k), (v_off, v) = eq
            key = k
            if key.startswith('"'):
                key = next((t for a, _b, t in lits if a == p_start), key.strip('"'))
            d[key] = _attr_value(v, p_start + v_off, lits)
        elif len(eq) == 1:
            d[part] = True  # unit attribute
        else:
            raise ValueError(f"bad attribute {part[:60]!r}")
    return d


def _dicts(
    masked: str, lits, start: int, stop: int
) -> list[tuple[int, dict[str, Any]]]:
    """Every ``{...}`` / ``<{...}>`` attribute dict in ``masked[start:stop]``
    with its paren depth (0 = op attributes)."""
    out = []
    depth = 0
    i = start
    while i < stop:
        c = masked[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "{":
            close = _match_close(masked, i)
            out.append((depth, _parse_dict(masked, i, close, lits)))
            i = close
        i += 1
    return out


def _leading_keywords(masked: str, start: int, stop: int) -> list[str]:
    """Bare tokens between the op name and the first operand / attr / type:
    ``arith.cmpi slt, %a`` -> ['slt'];  ``tt.atomic_rmw fadd, relaxed, gpu, %p``
    -> ['fadd', 'relaxed', 'gpu'];  ``tt.get_program_id x : i32`` -> ['x']."""
    s = masked[start:stop]
    m = re.match(r"\s*((?:[A-Za-z_]\w*\s*,\s*)*[A-Za-z_]\w*)(?=\s*(?:,|:|$))", s)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",")]


_RE_INT_LIT = re.compile(r"-?\d+")
_RE_HEX_LIT = re.compile(r"0x[0-9A-Fa-f]+")
_RE_FLOAT_LIT = re.compile(
    r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|inf|nan)"
)


def _scalar_literal(lit: str, elem: TypeInfo) -> Any:
    """A scalar constant literal read against its element type."""
    if lit in ("true", "false"):
        if elem.int_bits != 1:
            raise ValueError(f"bool literal {lit} of type {elem.elem}")
        return lit == "true"
    if elem.int_bits is not None:
        if _RE_INT_LIT.fullmatch(lit):
            return int(lit)
        if _RE_HEX_LIT.fullmatch(lit):
            return int(lit, 16)
        raise ValueError(f"integer constant {lit!r}")
    if elem.float_bits is not None:
        if _RE_FLOAT_LIT.fullmatch(lit) or _RE_HEX_LIT.fullmatch(lit):
            return ("float", lit)
        raise ValueError(f"float constant {lit!r}")
    raise ValueError(f"constant of unsupported element type {elem.elem}")


def _constant_attrs(t: _TOp, result_type: str | None) -> dict[str, Any]:
    """``arith.constant [{attrs}] <value> [: <type>]`` (raw text keeps a
    ``dense<"0x...">`` blob intact)."""
    s = t.masked[t.name_end : t.hdr_end]
    off = t.name_end
    lead = len(s) - len(s.lstrip())
    if s.lstrip().startswith("{"):  # a leading attr-dict precedes the value
        close = _match_close(s, lead)
        off += close + 1
        s = s[close + 1 :]
    parts = _split_top(s, ":")
    if not parts:
        raise ValueError("arith.constant without a value")
    p_off, p_masked = parts[0]
    lit = t.raw[off + p_off : off + p_off + len(p_masked)]
    if result_type is None:
        raise ValueError("arith.constant without a result")
    ty = parse_type(result_type)
    elem = parse_type(ty.elem)
    a: dict[str, Any] = {"literal": lit}
    if lit.startswith("dense<") and lit.endswith(">"):
        inner = lit[6:-1].strip()
        if not ty.shape and ty.text == ty.elem:
            raise ValueError(f"dense constant of scalar type {ty.text}")
        if inner.startswith(("[", '"')) or not inner:
            a["value"] = ("dense", inner)
            a["splat"] = False
        else:
            a["value"] = _scalar_literal(inner, elem)
            a["splat"] = True
    else:
        if ty.shape:
            raise ValueError(f"scalar literal {lit!r} of tensor type {ty.text}")
        a["value"] = _scalar_literal(lit, elem)
    return a


def _symbol(t: _TOp) -> str:
    m = re.search(r"@", t.masked[t.name_end : t.hdr_end])
    if m is None:
        raise ValueError("no symbol")
    at = t.name_end + m.start()
    if t.masked[at + 1 : at + 2] == '"':
        return next(x for a, _b, x in t.lits if a == at + 1)
    sm = re.match(r"@([\w$.-]+)", t.raw[at:])
    if sm is None:
        raise ValueError("bad symbol")
    return sm.group(1)


def _first_string(t: _TOp) -> str | None:
    return next((x for a, _b, x in t.lits if t.name_end <= a < t.hdr_end), None)


_AXES = {"x": 0, "y": 1, "z": 2}
_RESHAPE_KEYWORDS = frozenset({"allow_reorder", "efficient_layout"})
_V = r"%[-\w.$]+"  # a printed value name
_U = _V + r"(?:#\d+)?"  # a use (result #k of a multi-result op)
# `scf.for [unsigned] %iv = %lb to %ub step %s
#          [iter_args(%a = %init, ...) -> (T, ...)] [: T]`
_RE_SCF_FOR = re.compile(
    rf"\s*(unsigned\s+)?{_V}\s*=\s*{_U}\s+to\s+{_U}\s+step\s+{_U}"
    rf"(?:\s+iter_args\(\s*{_V}\s*=\s*{_U}(?:\s*,\s*{_V}\s*=\s*{_U})*\s*\)"
    r"\s*->\s*\(.+\))?(?:\s*:\s*[A-Za-z_]\w*)?\s*"
)


def _text_attrs(
    t: _TOp, result_types: Sequence[str], printer: "Printer"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(attrs, func header args info) recovered from the op's own text:
    header and, for a region op, its closing line."""
    name = t.name
    a: dict[str, Any] = {}
    for depth, d in _dicts(t.masked, t.lits, t.name_end, t.hdr_end):
        if depth == 0:
            a.update(d)
    if t.close_masked:
        stop = len(t.close_masked)
        span = _trailing_loc(t.close_masked)
        if span is not None:
            stop = span[0]
        for depth, d in _dicts(t.close_masked, t.close_lits or [], 0, stop):
            if depth == 0:
                a.update(d)
    extra: dict[str, Any] = {}
    keys = printer.keywords.get(name)
    if keys is not None:
        kw = _leading_keywords(t.masked, t.name_end, t.hdr_end)
        if len(kw) != len(keys):
            raise ValueError(
                f"expected {len(keys)} leading keyword(s) {keys}, got {kw}"
            )
        a.update(zip(keys, kw))
    if name == "arith.constant":
        a.update(_constant_attrs(t, result_types[0] if result_types else None))
    elif name == "tt.dot":
        # `%a, %b, %c(, inputPrecision = X)?`: any other bare assignment is a
        # misread, not an elided default
        bare = _blank_dicts(t.masked[t.name_end : t.hdr_end])
        assigned = _RE_BARE_ASSIGN.findall(bare)
        if assigned not in (
            [],
            [("inputPrecision", assigned[0][1] if assigned else "")],
        ):
            raise ValueError(f"unrecognized tt.dot assignments {assigned}")
        if assigned:
            a["inputPrecision"] = assigned[0][1]
    elif name == "tt.reshape":
        bare = _blank_dicts(t.masked[t.name_end : t.hdr_end])
        m = re.match(r"\s*%[-\w.$]+(?:#\d+)?((?:\s+[A-Za-z_]\w*)*)\s*(?::|$)", bare)
        if m is None:
            raise ValueError("unrecognized tt.reshape syntax")
        for word in m.group(1).split():
            if word not in _RESHAPE_KEYWORDS:
                raise ValueError(f"unknown tt.reshape keyword {word!r}")
            a[word] = True
    elif name == "scf.for":
        # the one header keyword is `unsigned` (unsignedCmp: the loop compares
        # its bounds unsigned, which changes the trip count); any other
        # header shape is a misread
        m = _RE_SCF_FOR.fullmatch(t.masked, t.name_end, t.hdr_end)
        if m is None:
            raise ValueError("unrecognized scf.for header")
        if "unsignedCmp" in a:
            raise ValueError("scf.for prints unsignedCmp in its attribute dict")
        a["unsignedCmp"] = m.group(1) is not None
    elif name == "tt.elementwise_inline_asm":
        s = _first_string(t)
        if s is not None:
            a["asm_string"] = s
    elif name == "tt.call":
        a["callee"] = _symbol(t)
    elif name == "tt.func":
        a["sym_name"] = _symbol(t)
        vm = re.match(r"\s*(public|private|nested)\b", t.masked[t.name_end :])
        if vm:
            a["visibility"] = vm.group(1)
        # the parameter list: `@name(%a: T {attrs} loc(..), ...)`
        m = re.search(r"@", t.masked[t.name_end : t.hdr_end])
        assert m is not None
        at = t.name_end + m.start()
        sym_end = (
            _string_end(t.masked, at + 1) + 1
            if t.masked[at + 1 : at + 2] == '"'
            else at + 1
        )
        while sym_end < t.hdr_end and (
            t.masked[sym_end].isalnum() or t.masked[sym_end] in "_$.-"
        ):
            sym_end += 1
        if t.masked[sym_end : sym_end + 1] != "(":
            raise ValueError("tt.func without a parameter list")
        close = _match_close(t.masked, sym_end)
        if "%" in t.masked[sym_end:close]:
            extra["args"], _ = _arg_list(t.raw, t.masked, t.lits, sym_end)
        else:
            extra["args"] = []
    elif name == "tt.print":
        s = _first_string(t)
        if s is not None:
            a["prefix"] = s
    elif name == "tt.assert":
        s = _first_string(t)
        if s is not None:
            a["message"] = s
    elif name in ("cf.br", "cf.cond_br"):
        extra["successors"] = _successor_groups(t)
    if name not in ("cf.br", "cf.cond_br") and "^" in t.masked[t.name_end : t.hdr_end]:
        raise ValueError(f"successor syntax on {name} is not supported")
    return a, extra


def _successor_groups(t: _TOp) -> list[tuple[str, int]]:
    """(label, number of printed operands) per successor, in order."""
    s = t.masked
    out = []
    for m in _RE_SUCC.finditer(s, t.name_end, t.hdr_end):
        n = 0
        j = m.end()
        if s[j : j + 1] == "(":
            close = _match_close(s, j)
            n = len(_RE_USE.findall(s, j, close))
        out.append((m.group(0), n))
    return out


def _header_uses_and_defs(t: _TOp) -> tuple[list[str], list[str], list[str]]:
    """(uses, the word printed right before each use, defs) in the header.
    Defs are block arguments printed in the header: ``%x: type`` (func args)
    and ``%iv = ...`` / ``iter_args(%a = %init)`` / ``scf.while (%a = %init)``."""
    s = t.masked
    uses: list[str] = []
    before: list[str] = []
    defs: list[str] = []
    for m in _RE_USE.finditer(s, t.name_end, t.hdr_end):
        if s.startswith(":", m.end(), t.hdr_end) or _RE_DEF_EQ.match(
            s, m.end(), t.hdr_end
        ):
            defs.append(m.group(0))
        else:
            uses.append(m.group(0))
            w = _RE_WORD_BEFORE.search(s, max(t.name_end, m.start() - 32), m.start())
            before.append(w.group(1) if w else "")
    return uses, before, defs


# ─────────────────────────── per-version tables ───────────────────────────
# One Printer per Triton minor release, keyed by release in PRINTERS. A
# table states what its release's printer prints; it is audited against
# that release (the goldens, the conformance corpus, and a bulk walk of
# real compiled kernels, tools/ir_bulk_conformance.py), never inferred from
# another release.


def _order_descriptor_store(uses: list[str], _before: list[str]) -> list[str]:
    # `%desc[%i, %j], %src`  ->  ODS (desc, src, indices...)
    return uses[:1] + uses[-1:] + uses[1:-1] if len(uses) >= 2 else uses


def _order_dot_scaled(uses: list[str], before: list[str]) -> list[str]:
    # `%a scale %as, %b scale %bs, %c`  ->  ODS (a, b, c, a_scale?, b_scale?)
    main = [u for u, w in zip(uses, before) if w != "scale"]
    scales = [u for u, w in zip(uses, before) if w == "scale"]
    return main + scales


def _frozen(d: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return types.MappingProxyType(dict(d))


@dataclass(frozen=True, eq=False)
class Printer:
    """What the text layer knows of one Triton minor release's TTIR printer.

    ``needed``: what the reader consumes, per op; every key must be
    recovered. ``keywords``: the leading enum keywords each custom syntax
    prints, in order. ``vocab``: the closed vocabularies of the text-only
    enums (any other spelling, or a generic-form integer, is a
    misalignment). ``defaults``: attributes the custom printer omits while
    they hold their default value. ``attr_types``: value types of the
    recovered non-enum attributes (a garbled value that falls back to raw
    text must not pass as recovered). ``bind_attrs``: attributes the
    release's getters can read, per op, as (getter, name), read on the
    bindings side and cross-checked against the text; ``bind_ints`` maps a
    keyword the text prints to the integer the ``int`` getter reads for it.
    ``printed_order``: custom syntaxes whose printed operand order is not
    the ODS operand order (the SSA-edge check fails closed on the others).
    ``elides_yield``: ops whose custom printer drops a trailing
    zero-operand scf.yield. ``block_pointer_types``: the dialect has
    ``!tt.ptr<tensor<...>>``; where it has not, a text holding one is
    refused before the bindings parse it."""

    release: str
    needed: Mapping[str, tuple[str, ...]]
    keywords: Mapping[str, tuple[str, ...]]
    vocab: Mapping[tuple[str, str], frozenset[str]]
    defaults: Mapping[str, Mapping[str, Any]]
    attr_types: Mapping[tuple[str, str], type]
    bind_attrs: Mapping[str, tuple[tuple[str, str], ...]]
    bind_ints: Mapping[tuple[str, str], Mapping[str, int]]
    printed_order: Mapping[str, Callable[[list[str], list[str]], list[str]]]
    elides_yield: frozenset[str]
    block_pointer_types: bool

    def extend(self, release: str, **changes: Any) -> "Printer":
        """This table with ``changes`` merged in: a mapping field's entries
        are added to (or replace) this table's, any other field replaced."""
        merged: dict[str, Any] = {}
        for key, value in changes.items():
            base = getattr(self, key)
            merged[key] = (
                _frozen({**base, **value}) if isinstance(base, Mapping) else value
            )
        return dataclasses.replace(self, release=release, **merged)


_SEM = frozenset({"relaxed", "acquire", "release", "acq_rel"})
_SCOPE = frozenset({"gpu", "cta", "sys"})
_AXIS_WORDS = frozenset(_AXES)

# Triton 3.6 (the D10a spike, its review corpora and the #361 goldens).
_PRINTER_3_6 = Printer(
    release="3.6",
    needed=_frozen(
        {
            "arith.constant": ("value",),
            "tt.make_range": ("start", "end"),
            "tt.get_program_id": ("axis",),
            "tt.get_num_programs": ("axis",),
            "arith.cmpi": ("predicate",),
            "arith.cmpf": ("predicate",),
            "tt.expand_dims": ("axis",),
            "tt.reduce": ("axis",),
            "tt.scan": ("axis", "reverse"),
            "tt.atomic_rmw": ("rmw_op", "sem", "scope"),
            "tt.atomic_cas": ("sem", "scope"),
            "tt.elementwise_inline_asm": (
                "asm_string",
                "constraints",
                "pure",
                "packed_element",
            ),
            "tt.call": ("callee",),
            "tt.func": ("sym_name", "visibility", "noinline"),
            "tt.load": ("isVolatile",),
            "tt.trans": ("order",),
            "tt.reshape": ("allow_reorder",),
            "tt.dot": ("inputPrecision", "maxNumImpreciseAcc"),
            "tt.print": ("prefix",),
            "scf.for": ("unsignedCmp",),
            "cf.br": ("successors",),
            "cf.cond_br": ("successors",),
        }
    ),
    keywords=_frozen(
        {
            "arith.cmpi": ("predicate",),
            "arith.cmpf": ("predicate",),
            "tt.atomic_rmw": ("rmw_op", "sem", "scope"),
            "tt.atomic_cas": ("sem", "scope"),
            "tt.get_program_id": ("axis",),
            "tt.get_num_programs": ("axis",),
            "tt.descriptor_reduce": ("kind",),
        }
    ),
    vocab=_frozen(
        {
            ("arith.cmpi", "predicate"): frozenset(
                {"eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"}
            ),
            ("arith.cmpf", "predicate"): frozenset(
                {
                    "false",
                    "oeq",
                    "ogt",
                    "oge",
                    "olt",
                    "ole",
                    "one",
                    "ord",
                    "ueq",
                    "ugt",
                    "uge",
                    "ult",
                    "ule",
                    "une",
                    "uno",
                    "true",
                }
            ),  # fmt: skip
            ("tt.atomic_rmw", "rmw_op"): frozenset(
                {
                    "and",
                    "or",
                    "xor",
                    "add",
                    "fadd",
                    "max",
                    "min",
                    "umax",
                    "umin",
                    "exch",
                }
            ),
            ("tt.atomic_rmw", "sem"): _SEM,
            ("tt.atomic_rmw", "scope"): _SCOPE,
            ("tt.atomic_cas", "sem"): _SEM,
            ("tt.atomic_cas", "scope"): _SCOPE,
            ("tt.get_program_id", "axis"): _AXIS_WORDS,
            ("tt.get_num_programs", "axis"): _AXIS_WORDS,
            ("tt.dot", "inputPrecision"): frozenset(
                {"tf32", "tf32x3", "ieee", "bf16x3", "bf16x6"}
            ),
            ("tt.descriptor_reduce", "kind"): frozenset(
                {"add", "min", "max", "inc", "dec", "and", "or", "xor"}
            ),
            ("tt.func", "visibility"): frozenset({"public", "private", "nested"}),
        }
    ),
    defaults=_frozen(
        {
            "tt.dot": _frozen({"inputPrecision": "ieee", "maxNumImpreciseAcc": 0}),
            "tt.load": _frozen({"isVolatile": False}),
            "tt.reshape": _frozen({"allow_reorder": False, "efficient_layout": False}),
            "tt.func": _frozen({"visibility": "public"}),
        }
    ),
    attr_types=_frozen(
        {
            ("tt.make_range", "start"): int,
            ("tt.make_range", "end"): int,
            ("tt.expand_dims", "axis"): int,
            ("tt.reduce", "axis"): int,
            ("tt.scan", "axis"): int,
            ("tt.scan", "reverse"): bool,
            ("tt.elementwise_inline_asm", "asm_string"): str,
            ("tt.elementwise_inline_asm", "constraints"): str,
            ("tt.elementwise_inline_asm", "pure"): bool,
            ("tt.elementwise_inline_asm", "packed_element"): int,
            ("tt.call", "callee"): str,
            ("tt.func", "sym_name"): str,
            ("tt.func", "noinline"): bool,
            ("tt.load", "isVolatile"): bool,
            ("tt.trans", "order"): tuple,
            ("tt.reshape", "allow_reorder"): bool,
            ("scf.for", "unsignedCmp"): bool,
            ("tt.dot", "maxNumImpreciseAcc"): int,
            ("tt.print", "prefix"): str,
            ("tt.print", "hex"): bool,
            ("tt.print", "isSigned"): tuple,
            ("tt.assert", "message"): str,
        }
    ),
    # the 3.6 getters: get_str_attr / get_bool_attr / get_flat_symbol_ref_attr
    # (integer and enum attributes are opaque to them)
    bind_attrs=_frozen(
        {
            "tt.func": (
                ("str", "sym_name"),
                ("str", "sym_visibility"),
                ("bool", "noinline"),
            ),
            "tt.call": (("sym", "callee"),),
            "tt.elementwise_inline_asm": (
                ("str", "asm_string"),
                ("str", "constraints"),
                ("bool", "pure"),
            ),
            "tt.load": (("bool", "isVolatile"),),
            "arith.constant": (("bool", "value"),),
            "tt.scan": (("bool", "reverse"),),
            "tt.print": (("str", "prefix"), ("bool", "hex")),
            "tt.assert": (("str", "message"),),
        }
    ),
    bind_ints=_frozen({}),
    printed_order=_frozen(
        {
            # audited against TritonOps.td
            "tt.descriptor_store": _order_descriptor_store,
            "tt.descriptor_reduce": _order_descriptor_store,
            "tt.dot_scaled": _order_dot_scaled,
        }
    ),
    elides_yield=frozenset({"scf.for", "scf.if"}),
    block_pointer_types=True,
)

# Triton 3.8 (audited on 3.8.0: 365 host-compiled kernels of the
# conformance and soundness corpora and the golden generators, extra and
# descriptor kernels for sm89 / sm90 / sm100, the goldens regenerated under
# 3.8 and 1127 Triton-cache texts, all walked with 0 misalignments; printed
# attributes, elided defaults, enum keywords and operand orders are 3.6's,
# the goldens pin identically). tl.debug_barrier() prints `ttg.barrier
# all` instead of `gpu.barrier`: its one attribute, addrSpace, is a bit
# enum printed as one keyword per set of bits (`all`, or single flags;
# a combination prints `local|global_read`, which the one-keyword syntax
# does not read: misaligned), and the 3.8 get_int_attr reads its bits. The
# dialect has no block-pointer types (tt.make_tensor_ptr / tt.advance are
# gone, tl.make_block_ptr lowers to pointer arithmetic), and its parser
# aborts the process on a `!tt.ptr<tensor<...>>` instead of reporting an
# error. The tensordesc type prints `!tt.tensordesc<32x32xf16>` (3.6:
# `!tt.tensordesc<tensor<32x32xf16>>`); types are compared as printed, so
# that takes no table entry.
_ADDR_SPACE = _frozen(
    {
        "none": 0,
        "local": 1,
        "global_read": 2,
        "global_write": 4,
        "tensor_read": 8,
        "tensor_write": 16,
        "all": 31,
    }
)
_PRINTER_3_8 = _PRINTER_3_6.extend(
    "3.8",
    # addrSpace is not read by the TTIR reader (the barrier is inert there);
    # recovered for the consumers that order memory (a race detector)
    needed={"ttg.barrier": ("addrSpace",)},
    keywords={"ttg.barrier": ("addrSpace",)},
    vocab={("ttg.barrier", "addrSpace"): frozenset(_ADDR_SPACE)},
    bind_attrs={"ttg.barrier": (("int", "addrSpace"),)},
    bind_ints={("ttg.barrier", "addrSpace"): _ADDR_SPACE},
    block_pointer_types=False,
)

PRINTERS: Mapping[str, Printer] = _frozen(
    {p.release: p for p in (_PRINTER_3_6, _PRINTER_3_8)}
)


def triton_release() -> tuple[str, str]:
    """(minor release, full version) of the installed Triton, e.g.
    ("3.6", "3.6.0"): the release is the version's first two components,
    as the D10b gate reads it."""
    import triton

    version = str(triton.__version__)
    return ".".join(version.split(".")[:2]), version


def printer(release: str | None = None) -> Printer:
    """The Printer table of ``release`` (default: the installed Triton's);
    raises ``UnknownTritonRelease`` for a release without one."""
    version = release
    if release is None:
        release, version = triton_release()
    table = PRINTERS.get(release)
    if table is None:
        raise UnknownTritonRelease(release, version or release)
    return table


_GETTERS = {
    "str": "get_str_attr",
    "bool": "get_bool_attr",
    "sym": "get_flat_symbol_ref_attr",
    "int": "get_int_attr",
}
_GETTER_TYPES: dict[str, type] = {"str": str, "bool": bool, "sym": str, "int": int}
_BIND_KEY = {"sym_visibility": "visibility"}


def _is_exactly(value: Any, want: type) -> bool:
    """``isinstance`` without bool passing as int; a tuple holds ints."""
    if isinstance(value, bool) != (want is bool) or not isinstance(value, want):
        return False
    return want is not tuple or all(
        isinstance(x, int) and not isinstance(x, bool)
        for x in value  # type: ignore[attr-defined]
    )


# A block-pointer type anywhere in a text: `ptr<tensor` in any spelling the
# parser accepts (`!tt.ptr<tensor<..>>`, `!tt<ptr<tensor<..>>>`, whitespace,
# newlines or `//` comments in between), and any type alias definition, which
# could hide the pointee. The _GAP forms read the raw text (a comment runs to
# the end of its line), the others the code view of _screen_view.
_GAP = r"(?:\s|//[^\n]*(?![^\n]))*"
_RE_BLOCK_PTR_TYPE = re.compile(r"\bptr\s*<\s*tensor\b")
_RE_BLOCK_PTR_TYPE_GAP = re.compile(rf"\bptr{_GAP}<{_GAP}tensor\b")
_RE_TYPE_ALIAS_DEF = re.compile(r"^\s*(![-\w.$]+)\s*=", re.M)
_RE_TYPE_ALIAS_DEF_GAP = re.compile(rf"^\s*![-\w.$]+{_GAP}=", re.M)


def _code_line(line: str) -> str:
    """``line`` as the parser's tokens see it, same length: string literal
    bodies masked ('_'), a ``//`` comment outside strings blanked to the end
    of the line. After an unterminated string the rest stays raw (the
    parser stops there; reading it can only add a refusal)."""
    out: list[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            try:
                j = _string_end(line, i)
            except ValueError:
                out.append(line[i:])
                break
            out.append('"' + "_" * (j - i - 1) + '"')
            i = j + 1
            continue
        if line.startswith("//", i):
            out.append(" " * (len(line) - i))
            break
        out.append(c)
        i += 1
    return "".join(out)


def _screen_view(text: str) -> str:
    """The code view of ``text`` (``_code_line`` per line), offsets kept, so
    a match's line is its count of newlines before it plus one."""
    return "\n".join(_code_line(line) for line in text.split("\n"))


def _screen(text: str, table: Printer) -> None:
    """Refuse, before the bindings see it, a text the release's parser
    cannot be handed: without block-pointer types (3.8) the parser aborts
    the whole process on one (an assertion in PointerType::get), so such a
    text never reaches it. The whole text is searched, so a type split over
    lines or around a comment is found too; string literals are not read as
    types, comments are skipped."""
    if table.block_pointer_types or not (
        _RE_BLOCK_PTR_TYPE_GAP.search(text) or _RE_TYPE_ALIAS_DEF_GAP.search(text)
    ):
        return
    view = _screen_view(text)
    found = []
    m = _RE_BLOCK_PTR_TYPE.search(view)
    if m is not None:
        found.append((m.start(), "a block-pointer type (!tt.ptr<tensor<...>>)"))
    m = _RE_TYPE_ALIAS_DEF.search(view)
    if m is not None:
        found.append(
            (m.start(1), "a type alias definition (it may name a block-pointer type)")
        )
    if not found:
        return
    at, what = min(found)
    ln = view.count("\n", 0, at) + 1
    raise ModuleParseError(
        f"line {ln}: {what}: the TTIR of Triton {table.release} has no "
        "block pointers, and its parser aborts the process on one; the "
        "text is refused before parsing",
        ln,
    )


def _type_checks(
    name: str, attrs: Mapping[str, Any], opnd_types, res_types
) -> list[str] | None:
    """Cross-check text-only integer attributes against the bindings' types,
    the one independent channel for them. None = no check applies."""
    if (
        name == "tt.make_range"
        and isinstance(attrs.get("start"), int)
        and isinstance(attrs.get("end"), int)
    ):
        shape = parse_type(res_types[0]).shape
        if shape != (attrs["end"] - attrs["start"],):
            return [
                f"make_range [{attrs['start']}, {attrs['end']}) vs result shape {shape}"
            ]
        return []
    if name == "tt.expand_dims" and isinstance(attrs.get("axis"), int):
        src, dst = parse_type(opnd_types[0]).shape, parse_type(res_types[0]).shape
        ax = attrs["axis"]
        if not 0 <= ax <= len(src) or dst != src[:ax] + (1,) + src[ax:]:
            return [f"expand_dims axis {ax}: {src} -> {dst}"]
        return []
    if name in ("tt.reduce", "tt.scan") and isinstance(attrs.get("axis"), int):
        src = parse_type(opnd_types[0]).shape
        ax = attrs["axis"]
        want = src if name == "tt.scan" else src[:ax] + src[ax + 1 :]
        got = parse_type(res_types[0]).shape
        if not (0 <= ax < len(src)) or got != want:
            return [f"{name} axis {ax}: {src} -> {got}"]
        return []
    if name == "tt.trans" and isinstance(attrs.get("order"), tuple):
        src, dst = parse_type(opnd_types[0]).shape, parse_type(res_types[0]).shape
        order = attrs["order"]
        if sorted(order) != list(range(len(src))) or dst != tuple(
            src[i] for i in order
        ):
            return [f"trans order {order}: {src} -> {dst}"]
        return []
    if name == "arith.constant" and "value" in attrs:
        t = parse_type(res_types[0])
        v = attrs["value"]
        if isinstance(v, bool) or not isinstance(v, int):
            return []  # kind vs element type is checked by _scalar_literal
        bits = parse_type(t.elem).int_bits
        assert bits is not None
        # the printer prints i1 as true / false and every wider signless
        # integer as signed: `4294967295 : i32` parses, but MLIR holds -1
        if bits == 1:
            return [f"constant {v}: the printer prints {t.text} as true / false"]
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if lo <= v <= hi:
            return []
        return [f"constant {v} is outside the printed (signed) range of {t.text}"]
    return None


# ─────────────────────────── bindings side ───────────────────────────


@dataclass
class _BOp:
    name: str
    operands: tuple[int, ...]  # raw value ids (valid within one parse only)
    operand_types: tuple[str, ...]
    results: tuple[int, ...]
    result_types: tuple[str, ...]
    result_locs: tuple[str, ...]
    region_ids: tuple[int, ...]
    region_sizes: tuple[int, ...]
    block_id: int | None
    attrs: dict[str, Any]


@dataclass
class _BBlock:
    region_id: int
    args: tuple[int, ...]
    arg_types: tuple[str, ...]
    arg_locs: tuple[str, ...]
    ops: list[int]  # indices into the walk list


@dataclass
class _BindWalk:
    ops: list[_BOp]
    blocks: dict[int, _BBlock]
    region_blocks: dict[int, list[int]]  # raw region id -> raw block ids in order
    path: str  # the parse path (parser-invented locs name it)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


class _ParseInput:
    """The text as a file path for ``parse_mlir_module``: an anonymous memfd
    via ``/proc/self/fd`` when available, else a temporary file."""

    def __init__(self, data: bytes) -> None:
        self.path: str | None = None
        self._fd: int | None = None
        self._tmp: str | None = None
        memfd_create = getattr(os, "memfd_create", None)
        if memfd_create is not None:
            try:
                fd = memfd_create("tilelens-ttir", getattr(os, "MFD_CLOEXEC", 0))
            except OSError:
                fd = None
            if fd is not None:
                path = f"/proc/self/fd/{fd}"
                try:
                    _write_all(fd, data)
                    ok = os.path.exists(path)
                except OSError:
                    ok = False
                if ok:
                    self._fd, self.path = fd, path
                    return
                os.close(fd)
        try:
            fd, tmp = tempfile.mkstemp(prefix="tilelens-", suffix=".ttir")
        except OSError as e:
            raise ModuleParseError(
                f"cannot create a temporary file for the MLIR parser: {e}"
            ) from None
        try:
            try:
                _write_all(fd, data)
            finally:
                os.close(fd)
        except OSError as e:
            os.unlink(tmp)
            raise ModuleParseError(f"cannot write the MLIR parser input: {e}") from None
        self._tmp = self.path = tmp

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._tmp is not None:
            try:
                os.unlink(self._tmp)
            except OSError:
                pass
            self._tmp = None


# (saved fd 2, capture buffer) while a capture redirects fd 2; set and
# cleared under _PARSE_LOCK, read by the fork handler (_after_fork_in_child)
_REDIRECT: tuple[int, int] | None = None


class _StderrCapture:
    """Redirect fd 2 (where the C++ parser prints its diagnostic) into an
    anonymous file for the duration of a ``with`` block; only used under
    ``_PARSE_LOCK``. Everything written to fd 2 in that window lands in
    ``data``: on a failed parse that is the diagnostic, on a successful one
    it belongs to someone else (parser warnings, other threads), so
    ``replay()`` passes it on to the real fd 2."""

    def __init__(self) -> None:
        self._buf: int | None = None
        self._saved: int | None = None
        self.data = b""
        self.text = ""

    def __enter__(self) -> "_StderrCapture":
        global _REDIRECT
        try:
            memfd_create = getattr(os, "memfd_create", None)
            if memfd_create is not None:
                buf = memfd_create("tilelens-diag", getattr(os, "MFD_CLOEXEC", 0))
            else:
                with tempfile.TemporaryFile() as f:
                    buf = os.dup(f.fileno())
        except OSError:
            return self  # no capture: the diagnostic stays on stderr
        try:
            sys.stderr.flush()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            saved = os.dup(2)
        except OSError:
            os.close(buf)
            return self
        # published before fd 2 moves (and cleared after it is back), so a
        # fork at any point of the window can restore fd 2 in the child
        _REDIRECT = (saved, buf)
        try:
            os.dup2(buf, 2)
        except OSError:
            _REDIRECT = None
            os.close(saved)
            os.close(buf)
            return self
        self._buf, self._saved = buf, saved
        return self

    def __exit__(self, *exc: object) -> None:
        global _REDIRECT
        if self._saved is not None:
            os.dup2(self._saved, 2)
            _REDIRECT = None
            os.close(self._saved)
            self._saved = None
        if self._buf is not None:
            try:
                os.lseek(self._buf, 0, os.SEEK_SET)
                chunks = []
                while chunk := os.read(self._buf, 1 << 16):
                    chunks.append(chunk)
                self.data = b"".join(chunks)
                self.text = self.data.decode("utf-8", "replace")
            finally:
                os.close(self._buf)
                self._buf = None

    def replay(self) -> None:
        if self.data:
            try:
                _write_all(2, self.data)
            except OSError:
                pass


_PARSE_LOCK = threading.Lock()  # fd-2 redirection is process-wide


def _bind_walk(data: bytes, table: Printer) -> _BindWalk:
    """Parse ``data`` with the bindings and flatten the module into
    pure-Python records, reading the attributes ``table`` names. The context
    is pinned on the module for the module's whole life, the module is
    dropped before the context, and no binding object survives the call
    (dropping the context first segfaults). After a successful parse,
    whatever else reached fd 2 in the window is passed on."""
    from triton._C.libtriton import ir  # TTIR dialects only: no backend (TTGIR) loading

    source = _ParseInput(data)
    try:
        path = source.path
        assert path is not None
        ctx = ir.context()
        try:
            ir.load_dialects(ctx)
            capture = _StderrCapture()
            mod = None
            with _PARSE_LOCK, capture:
                try:
                    mod = ir.parse_mlir_module(path, ctx)
                except RuntimeError:
                    pass
            if mod is None:
                raise _parse_error(capture.text, path)
            capture.replay()
            try:
                walk, failure = _pinned_extract(mod, ctx, table)
            finally:
                del mod  # the module before its context
        finally:
            del ctx
    finally:
        source.close()
    if walk is None:
        raise MisalignedModule([f"bindings walk failed: {failure}"])
    ops, blocks, region_blocks = walk
    return _BindWalk(ops, blocks, region_blocks, path)


def _pinned_extract(mod, ctx, table: Printer) -> tuple[Any, str | None]:
    """Pin ``ctx`` on ``mod`` (proton's pattern: the module keeps its context
    alive), then extract. Returns (records, None) or (None, reason); an
    exception's traceback, whose frames hold binding objects, dies here while
    the module and its context are still alive."""
    try:
        mod.context = ctx
    except Exception as e:  # noqa: BLE001
        return (
            None,
            f"cannot pin the MLIR context on the module: {type(e).__name__}: {e}",
        )
    body: list[Any] = []
    try:
        walk = _extract(mod, body, table)
    except Exception as e:  # noqa: BLE001  (bindings drift: an unexpected getter result)
        return None, f"{type(e).__name__}: {e}"
    # No binding erases a module, so each parse would leak its whole op tree
    # (~20 KiB for a 12 KiB text); erasing the body block frees all but the
    # empty module op. Safe here: extraction is over, ctx is pinned, and the
    # body block is the one binding object still alive (dropped at once).
    if body:
        try:
            body.pop().erase()
        except Exception:  # noqa: BLE001  (bindings drift: keep the bounded leak)
            pass
    return walk, None


_RE_DIAG_POS = re.compile(r'loc\("<ttir>":(\d+):\d+\)')


def _parse_error(diag: str, path: str) -> ModuleParseError:
    diag = diag.replace(path, "<ttir>").strip() or "the MLIR parser rejected the text"
    m = _RE_DIAG_POS.search(diag)
    return ModuleParseError(diag, int(m.group(1)) if m else None)


def _extract(
    mod, body: list[Any], table: Printer
) -> tuple[list[_BOp], dict[int, _BBlock], dict[int, list[int]]]:
    """Copy the walked module into pure-Python records (post-order walk:
    ops of a block arrive in program order, blocks of a region in order).
    The module's body block (the one binding object kept) goes to ``body``,
    for ``_pinned_extract`` to erase."""
    ops: list[_BOp] = []
    blocks: dict[int, _BBlock] = {}
    region_blocks: dict[int, list[int]] = {}

    def cb(op) -> None:
        blk = op.get_block()
        bid = None
        if blk is not None:
            bid = blk.id()
            rec = blocks.get(bid)
            if rec is None:
                args = [blk.get_argument(i) for i in range(blk.get_num_arguments())]
                parent = blk.get_parent()
                rid = parent.id()
                if parent.get_parent_region() is None:
                    body.append(blk)  # the module's body block
                rec = blocks[bid] = _BBlock(
                    rid,
                    tuple(x.id() for x in args),
                    tuple(str(x.get_type()) for x in args),
                    tuple(str(x.get_loc()) for x in args),
                    [],
                )
                region_blocks.setdefault(rid, []).append(bid)
            rec.ops.append(len(ops))
        name = op.get_name()
        opnds = [op.get_operand(i) for i in range(op.get_num_operands())]
        res = [op.get_result(i) for i in range(op.get_num_results())]
        regs = [op.get_region(i) for i in range(op.get_num_regions())]
        attrs: dict[str, Any] = {}
        for kind, aname in table.bind_attrs.get(name, ()):
            getter = getattr(op, _GETTERS[kind], None)
            if getter is None:  # the table names a getter these bindings lack
                raise TypeError(
                    f"{name}.{aname}: the bindings have no {_GETTERS[kind]} "
                    f"(the Triton {table.release} table reads it)"
                )
            v = getter(aname)
            if v is not None:
                if not _is_exactly(v, _GETTER_TYPES[kind]):
                    raise TypeError(
                        f"{name}.{aname}: getter returned {type(v).__name__}"
                    )
                attrs[aname] = v
        ops.append(
            _BOp(
                name,
                tuple(x.id() for x in opnds),
                tuple(str(x.get_type()) for x in opnds),
                tuple(x.id() for x in res),
                tuple(str(x.get_type()) for x in res),
                tuple(str(x.get_loc()) for x in res),
                tuple(x.id() for x in regs),
                tuple(x.size() for x in regs),
                bid,
                attrs,
            )
        )

    mod.walk(cb)
    return ops, blocks, region_blocks


# ─────────────────────────── alignment ───────────────────────────


@dataclass
class _OpB:  # mutable op builder
    name: str
    operands_raw: tuple[int, ...]
    operand_types: tuple[str, ...]
    results: tuple[int, ...]
    result_types: tuple[str, ...]
    attrs: dict[str, Any]
    path: tuple[int, ...]
    position: int
    line_no: int | None
    end_line: int | None
    loc: SourceLoc | None
    callers: tuple[SourceLoc, ...]
    loc_name: str | None
    implicit: bool
    regions: list[tuple[int, ...]]
    successor_labels: list[tuple[str, int]] | None = None
    func_args: list[_ArgDef] | None = None


@dataclass
class _BlockB:
    op: int
    region: int
    position: int
    label: str | None
    name: str  # the printer's name: label, or ^bb0 for an unlabeled entry
    args: tuple[int, ...]
    arg_types: tuple[str, ...]
    arg_names: tuple[str | None, ...]
    ops: list[int]
    path: tuple[int, ...]  # path of the ops inside this block
    preds: tuple[str, ...] | None
    line_no: int


# what Module.stats counts
_STATS = (
    "ops", "implicit_ops", "blocks", "values", "funcs", "ssa_edges", "result_locs", "arg_locs",
    "bind_attrs", "type_checks", "needed_attrs", "cf_edges", "pred_checks",
)  # fmt: skip


class _Aligner:
    def __init__(self, tree: _TextTree, bw: _BindWalk, table: Printer) -> None:
        self.tree = tree
        self.bw = bw
        self.table = table
        self.locp = _LocParser(tree.aliases, bw.path)
        self.problems: list[tuple[int | None, str]] = []
        self.ops: list[_OpB] = []
        self.blocks: list[_BlockB] = []
        self.values: list[list[Any]] = []  # [type, op, block, position, name]
        self.vmap: dict[int, int] = {}  # raw value id -> value index
        self.uses: list[
            tuple[int, list[str], list[str], tuple[dict[str, int], ...]]
        ] = []
        self.region_labels: dict[tuple[int, int], dict[str, int]] = {}
        # op lines with / without a printed trailing loc: the printer (debug
        # info on) gives every op one, so a module mixing both is misread
        self.loc_lines: list[int] = []
        self.no_loc_lines: list[int] = []
        self.stats: collections.Counter[str] = collections.Counter(
            dict.fromkeys(_STATS, 0)
        )

    def bad(self, line: int | None, msg: str) -> None:
        self.problems.append((line, f"line {line}: {msg}" if line is not None else msg))

    # ── values ──
    def new_value(
        self,
        raw: int,
        type_: str,
        op: int | None,
        block: int | None,
        pos: int,
        loc: str,
    ) -> int:
        if raw in self.vmap:
            raise ValueError("a value appears twice in the walk")
        idx = len(self.values)
        name = None
        tree = self.locp.parse(loc)
        if tree is not None:
            name = _loc_site(tree)[2]
        self.values.append([type_, op, block, pos, name])
        self.vmap[raw] = idx
        return idx

    def text_loc(self, t: _TOp) -> tuple | None:
        masked, raw = (
            (t.close_masked, t.close_raw) if t.opens_region else (t.masked, t.raw)
        )
        span = _trailing_loc(masked)
        (self.loc_lines if span is not None else self.no_loc_lines).append(t.line_no)
        if span is None:
            return None
        return self.locp.parse(raw[span[0] : span[1]])

    # ── traversal ──
    def run(self) -> Module:
        root_w = [i for i, b in enumerate(self.bw.ops) if b.block_id is None]
        top = self.tree.root.regions[0][0].ops if self.tree.root.regions[0] else []
        if len(root_w) != 1 or len(top) != 1:
            self.bad(
                None,
                f"expected one top-level op: text has {len(top)}, bindings {len(root_w)}",
            )
            raise self.failure()
        # task: ("op", text op, walk index, parent block index | None, position, scopes)
        #    or ("implicit", walk index, parent block index, position)
        stack: list[tuple] = [("op", top[0], root_w[0], None, 0, ({},))]
        while stack:
            task = stack.pop()
            if task[0] == "op":
                children = self.visit(*task[1:])
            else:
                children = []
                self.implicit(*task[1:])
            stack.extend(reversed(children))
        if len(self.ops) != len(self.bw.ops):
            self.bad(
                None, f"{len(self.ops)} aligned ops != {len(self.bw.ops)} walked ops"
            )
        if self.loc_lines:
            for line in self.no_loc_lines:
                self.bad(line, "op prints no loc while the module prints locs")
        if not self.problems:
            self.check_uses()
            self.check_successors()
        if self.problems:
            raise self.failure()
        return self.freeze()

    def failure(self) -> MisalignedModule:
        first = next((ln for ln, _ in self.problems if ln is not None), None)
        return MisalignedModule([m for _, m in self.problems], first)

    def implicit(self, wi: int, block: int, pos: int) -> None:
        b = self.bw.ops[wi]
        idx = len(self.ops)
        self.stats["implicit_ops"] += 1
        self.ops.append(
            _OpB(
                b.name,
                (),
                (),
                (),
                (),
                {},
                self.blocks[block].path,
                pos,
                None,
                None,
                None,
                (),
                None,
                True,
                [],
            )
        )
        self.blocks[block].ops.append(idx)

    def visit(
        self,
        t: _TOp,
        wi: int,
        block: int | None,
        pos: int,
        scopes: tuple[dict[str, int], ...],
    ) -> list:
        b = self.bw.ops[wi]
        line = t.line_no
        idx = len(self.ops)
        path = self.blocks[block].path if block is not None else ()
        if block is not None:
            self.blocks[block].ops.append(idx)
        structural_ok = True
        if t.name != b.name:
            self.bad(line, f"text op {t.name!r} != walked op {b.name!r}")
            structural_ok = False
        if t.n_results != len(b.results):
            self.bad(
                line,
                f"{t.name}: {t.n_results} printed results != {len(b.results)} walked",
            )
            structural_ok = False
        uses, before, hdefs = _header_uses_and_defs(t)
        if len(uses) != len(b.operands):
            self.bad(
                line,
                f"{t.name}: {len(uses)} printed operands != {len(b.operands)} walked",
            )
        # printed regions: trailing empty regions may be omitted
        if len(t.regions) > len(b.region_ids) or any(
            b.region_sizes[i] for i in range(len(t.regions), len(b.region_ids))
        ):
            self.bad(
                line,
                f"{t.name}: {len(t.regions)} printed regions vs walked sizes {list(b.region_sizes)}",
            )
            structural_ok = False
        # locs
        try:
            tloc = self.text_loc(t)
            for k, s in enumerate(b.result_locs):
                tree = self.locp.parse(s)
                if tree is not None:
                    self.stats["result_locs"] += 1
                    if tree != tloc:
                        self.bad(
                            line,
                            f"{t.name}: result {k} loc differs: text {tloc} vs bindings {tree}",
                        )
        except ValueError as e:
            self.bad(line, f"{t.name}: unreadable loc: {e}")
            tloc = None
        site, callers, lname = _loc_site(tloc)
        # attributes
        attrs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        if structural_ok:
            try:
                attrs, extra = _text_attrs(t, b.result_types, self.table)
            except ValueError as e:
                self.bad(line, f"{t.name}: {e}")
            self.check_attrs(b, attrs, line)
        rec = _OpB(
            b.name,
            b.operands,
            b.operand_types,
            (),
            b.result_types,
            attrs,
            path,
            pos,
            line,
            t.close_line if t.opens_region else line,
            site,
            callers,
            lname,
            False,
            [],
            successor_labels=extra.get("successors"),
            func_args=extra.get("args"),
        )
        self.ops.append(rec)
        rec.results = tuple(
            self.new_value(raw, b.result_types[k], idx, None, k, b.result_locs[k])
            for k, raw in enumerate(b.results)
        )
        if len(t.result_names) == len(b.results):
            for name, raw in zip(t.result_names, b.results):
                scopes[-1][name] = raw
        self.uses.append((idx, uses, before, scopes))
        if (
            b.name == "tt.func"
            and rec.func_args is not None
            and [d.name for d in rec.func_args] != hdefs
        ):
            self.bad(
                line,
                f"tt.func: parameter list {[d.name for d in rec.func_args]} vs header defs {hdefs}",
            )
        if not structural_ok:
            rec.regions = [() for _ in b.region_ids]
            return []
        if hdefs and not b.region_ids:
            self.bad(line, f"{t.name}: header defines {hdefs} but the op has no region")
        children: list[tuple] = []
        for ri, rid in enumerate(b.region_ids):
            children += self.region(t, b, idx, ri, rid, hdefs, rec, scopes)
        return children

    def region(
        self, t: _TOp, b: _BOp, idx: int, ri: int, rid: int, hdefs, rec: _OpB, scopes
    ) -> list:
        line = t.line_no
        wblocks = self.bw.region_blocks.get(rid, [])
        if len(wblocks) != b.region_sizes[ri]:
            self.bad(line, f"{t.name}: region {ri} has a block without ops")
        tblocks = t.regions[ri] if ri < len(t.regions) else []
        if not tblocks and len(wblocks) == 1 and b.name in self.table.elides_yield:
            only = self.bw.blocks[wblocks[0]].ops
            tail = self.bw.ops[only[-1]] if len(only) == 1 else None
            if tail is not None and tail.name == "scf.yield" and not tail.operands:
                # `{ }`: the region's one block held only the elided yield
                tblocks = [_TBlock(None, [], [], line)]
        if len(tblocks) != len(wblocks):
            self.bad(
                line,
                f"{t.name}: region {ri}: {len(tblocks)} printed blocks != {len(wblocks)} walked",
            )
            rec.regions.append(())
            return []
        region_scope: dict[str, int] = {}
        inner = scopes + (region_scope,)
        labels: dict[str, int] = {}
        self.region_labels[(idx, ri)] = labels
        keys = []
        children: list[tuple] = []
        for bi, (tb, rbid) in enumerate(zip(tblocks, wblocks)):
            bb = self.bw.blocks[rbid]
            bidx = len(self.blocks)
            keys.append(bidx)
            name = tb.label if tb.label is not None else "^bb0"
            if tb.label is None and bi != 0:
                self.bad(tb.line_no, f"{t.name}: unlabeled block {ri}.{bi}")
            if name in labels:
                self.bad(tb.line_no, f"{t.name}: duplicate block label {name}")
            labels[name] = bidx
            blk = _BlockB(
                idx,
                ri,
                bi,
                tb.label,
                name,
                (),
                bb.arg_types,
                (),
                [],
                rec.path + (bidx,),
                tb.preds,
                tb.line_no,
            )
            self.blocks.append(blk)
            blk.args = tuple(
                self.new_value(raw, bb.arg_types[ai], None, bidx, ai, bb.arg_locs[ai])
                for ai, raw in enumerate(bb.args)
            )
            blk.arg_names = tuple(self.values[v][4] for v in blk.args)
            # printed arguments: the label's, or the op header's for the
            # implicit entry block of region 0 (func args, iv + iter_args,
            # scf.while inits)
            if tb.label is not None:
                printed = tb.args
            elif bi == 0 and ri == 0:
                printed = (
                    rec.func_args
                    if rec.func_args is not None
                    else [_ArgDef(n, None, None, {}) for n in hdefs]
                )
            else:
                printed = []
            if len(printed) != len(bb.args):
                self.bad(
                    tb.line_no,
                    f"{t.name}: block {ri}.{bi}: {len(printed)} printed args != {len(bb.args)} walked",
                )
            else:
                for ai, (d, raw) in enumerate(zip(printed, bb.args)):
                    region_scope[d.name] = raw
                    self.check_arg(d, bb, ai, tb.line_no, t.name)
            # ops, with the one tolerated gap: an elided trailing scf.yield
            n_text, n_walk = len(tb.ops), len(bb.ops)
            elided = False
            if n_text != n_walk:
                tail = self.bw.ops[bb.ops[-1]] if bb.ops else None
                elided = (
                    n_text == n_walk - 1
                    and tail is not None
                    and tail.name == "scf.yield"
                    and not tail.operands
                    and b.name in self.table.elides_yield
                )
                if not elided:
                    self.bad(
                        tb.line_no,
                        f"{t.name}: block {ri}.{bi}: {n_text} printed ops != {n_walk} walked",
                    )
                    continue
            for p, (tchild, wchild) in enumerate(zip(tb.ops, bb.ops)):
                children.append(("op", tchild, wchild, bidx, p, inner))
            if elided:
                children.append(("implicit", bb.ops[-1], bidx, n_text))
        rec.regions.append(tuple(keys))
        return children

    def check_arg(
        self, d: _ArgDef, bb: _BBlock, ai: int, line: int, owner: str
    ) -> None:
        if d.type is not None and d.type != bb.arg_types[ai]:
            self.bad(
                line,
                f"{owner}: argument {d.name}: printed type {d.type!r} != {bb.arg_types[ai]!r}",
            )
        try:
            bind = self.locp.parse(bb.arg_locs[ai])
            text = self.locp.parse(d.loc) if d.loc is not None else None
        except ValueError as e:
            self.bad(line, f"{owner}: argument {d.name}: unreadable loc: {e}")
            return
        if bind is not None or text is not None:
            self.stats["arg_locs"] += 1
            if bind != text:
                self.bad(
                    line,
                    f"{owner}: argument {d.name}: loc differs: text {text} vs bindings {bind}",
                )

    def check_attrs(self, b: _BOp, attrs: dict[str, Any], line: int) -> None:
        name = b.name
        table = self.table
        for key, value in list(attrs.items()):
            vocab = table.vocab.get((name, key))
            if vocab is not None and not (isinstance(value, str) and value in vocab):
                self.bad(line, f"{name}: {key} {value!r} outside the closed vocabulary")
        for key, default in table.defaults.get(name, {}).items():
            attrs.setdefault(key, default)
        for key, value in attrs.items():
            want = table.attr_types.get((name, key))
            if want is not None and not _is_exactly(value, want):
                self.bad(line, f"{name}: {key} {value!r} is not a {want.__name__}")
        if (
            name in ("tt.get_program_id", "tt.get_num_programs")
            and attrs.get("axis") in _AXES
        ):
            attrs["axis"] = _AXES[attrs["axis"]]
        for key, bv in b.attrs.items():
            self.stats["bind_attrs"] += 1
            akey = _BIND_KEY.get(key, key)
            tv = attrs.get(akey)
            ints = table.bind_ints.get((name, key))
            if ints is not None:  # a keyword the bindings read as its integer
                tv = ints.get(tv) if isinstance(tv, str) else None
            if tv != bv or not _is_exactly(tv, type(bv)):
                self.bad(
                    line,
                    f"{name}: attr {key}: text {attrs.get(akey)!r} vs bindings {bv!r}",
                )
        try:
            tc = _type_checks(name, attrs, b.operand_types, b.result_types)
        except (IndexError, ValueError) as e:
            tc = [f"type check failed: {e}"]
        if tc is not None:
            self.stats["type_checks"] += 1
            for msg in tc:
                self.bad(line, f"{name}: {msg}")
        if name in ("cf.br", "cf.cond_br"):
            return  # successors are checked once every block is known
        for key in table.needed.get(name, ()):
            self.stats["needed_attrs"] += 1
            if attrs.get(key) is None:
                self.bad(line, f"{name}: attribute {key!r} not recovered")

    def check_uses(self) -> None:
        for idx, uses, before, scopes in self.uses:
            rec = self.ops[idx]
            order = self.table.printed_order.get(rec.name)
            if order is not None:
                uses = order(uses, before)
            got = []
            for u in uses:
                got.append(next((d[u] for d in reversed(scopes) if u in d), None))
            self.stats["ssa_edges"] += len(uses)
            if tuple(got) != rec.operands_raw:
                unresolved = [u for u, v in zip(uses, got) if v is None]
                what = (
                    f"unresolved {unresolved}"
                    if unresolved
                    else "operand values differ"
                )
                if not unresolved and sorted(got) == sorted(rec.operands_raw):  # type: ignore[type-var]
                    what = "operand ORDER differs (same multiset)"
                self.bad(rec.line_no, f"{rec.name}: SSA edges: {what} ({uses})")

    def check_successors(self) -> None:
        got: dict[int, list[str]] = collections.defaultdict(list)
        for rec in self.ops:
            groups = rec.successor_labels
            if groups is None:
                continue
            parent = self.blocks[rec.path[-1]]
            labels = self.region_labels[(parent.op, parent.region)]
            succ = []
            n_dest_args = 0
            for label, n_printed in groups:
                dest = labels.get(label)
                if dest is None:
                    self.bad(rec.line_no, f"{rec.name}: unknown successor {label}")
                    return
                n = len(self.blocks[dest].args)
                if n_printed != n:
                    self.bad(
                        rec.line_no,
                        f"{rec.name}: {label}: {n_printed} successor operands != {n} block args",
                    )
                n_dest_args += n
                succ.append(dest)
                got[dest].append(parent.name)
            own = 1 if rec.name == "cf.cond_br" else 0
            if len(succ) != (2 if rec.name == "cf.cond_br" else 1):
                self.bad(rec.line_no, f"{rec.name}: {len(succ)} successors")
            if len(rec.operands_raw) != own + n_dest_args:
                self.bad(
                    rec.line_no,
                    f"{rec.name}: {len(rec.operands_raw)} operands != {own} + {n_dest_args} successor args",
                )
            self.stats["cf_edges"] += len(succ)
            rec.attrs["successors"] = tuple(succ)
            self.stats["needed_attrs"] += 1
        # the printer's predecessor comments: a second, printer-computed CFG
        for bidx, blk in enumerate(self.blocks):
            want = blk.preds
            if want is None:
                if (
                    self.tree.pred_comments
                    and blk.label is not None
                    and blk.position != 0
                ):
                    self.bad(blk.line_no, f"block {blk.name}: no predecessor comment")
                if blk.position == 0 and got.get(bidx):
                    self.bad(
                        blk.line_no,
                        f"entry block {blk.name} has predecessors {got[bidx]}",
                    )
                continue
            self.stats["pred_checks"] += 1
            if collections.Counter(want) != collections.Counter(got.get(bidx, [])):
                self.bad(
                    blk.line_no,
                    f"block {blk.name}: printer preds {sorted(want)} != {sorted(got.get(bidx, []))}",
                )

    def freeze(self) -> Module:
        values = tuple(Value(i, *v) for i, v in enumerate(self.values))
        blocks = tuple(
            Block(
                i,
                b.op,
                b.region,
                b.position,
                b.label,
                b.args,
                b.arg_types,
                b.arg_names,
                tuple(b.ops),
            )
            for i, b in enumerate(self.blocks)
        )
        ops = []
        funcs = []
        for i, r in enumerate(self.ops):
            operands = tuple(self.vmap[v] for v in r.operands_raw)
            ops.append(
                Op(
                    i,
                    r.name,
                    operands,
                    r.operand_types,
                    r.results,
                    r.result_types,
                    _FrozenMap(r.attrs),
                    tuple(r.regions),
                    r.path,
                    r.position,
                    r.line_no,
                    r.end_line,
                    r.loc,
                    r.callers,
                    r.loc_name,
                    r.implicit,
                )
            )
            if r.name == "tt.func":
                args: tuple[FuncArg, ...] = ()
                if r.regions and r.regions[0]:
                    entry = blocks[r.regions[0][0]]
                    printed = r.func_args or []
                    args = tuple(
                        FuncArg(
                            k,
                            v,
                            entry.arg_types[k],
                            entry.arg_names[k],
                            _FrozenMap(printed[k].attrs if k < len(printed) else {}),
                        )
                        for k, v in enumerate(entry.args)
                    )
                funcs.append(Func(i, r.attrs["sym_name"], r.attrs["visibility"], args))
        self.stats.update(
            ops=len(ops), blocks=len(blocks), values=len(values), funcs=len(funcs)
        )
        return Module(
            tuple(ops),
            blocks,
            values,
            tuple(funcs),
            _FrozenMap(self.stats),
            self.table.release,
        )


# ─────────────────────────── entry point ───────────────────────────


def _walk(
    text: str, scan_text: str | None = None, *, table: Printer | None = None
) -> Module:
    """Uncached walk with ``table`` (default: the installed Triton's
    ``printer()``). ``scan_text`` (tests only) feeds the text layer a
    different string than the bindings parse, to prove the checks catch a
    text layer that mis-reads the module."""
    if table is None:
        table = printer()
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ModuleParseError(f"the text is not encodable as UTF-8: {e}") from None
    _screen(text, table)
    bw = _bind_walk(data, table)
    try:
        tree = _scan_text(text if scan_text is None else scan_text)
    except _TextError as e:
        raise MisalignedModule(
            [f"line {e.line_no}: {e.msg}" if e.line_no else e.msg], e.line_no
        ) from None
    try:
        return _Aligner(tree, bw, table).run()
    except MisalignedModule:
        raise
    except (ValueError, KeyError, IndexError, AssertionError, TypeError) as e:
        # a text shape the aligner does not model: fail closed
        raise MisalignedModule([f"aligner: {type(e).__name__}: {e}"]) from None


_CACHE_SIZE = 32
_CACHE: collections.OrderedDict[
    tuple[bytes, str], Module | tuple[tuple[str, ...], int | None]
] = collections.OrderedDict()
_CACHE_LOCK = threading.Lock()


def walk_module(text: str) -> Module:
    """Walk one printed TTIR module (see the module docstring) with the
    installed Triton's ``Printer`` table.

    Raises ``UnknownTritonRelease`` when that release has no table,
    ``MisalignedModule`` when the text layer and the bindings disagree and
    ``ModuleParseError`` when the MLIR parser rejects the text (or the text
    holds a construct the release's parser cannot be handed). Results (and
    misalignments) are cached by sha256 of the text and the release, so a
    text is parsed once while it stays among the last ``_CACHE_SIZE``
    distinct texts.
    """
    table = printer()
    key = (
        hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest(),
        table.release,
    )
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
    if hit is None:
        try:
            hit = _walk(text, table=table)
        except MisalignedModule as e:
            hit = (e.problems, e.line_no)
        with _CACHE_LOCK:
            _CACHE[key] = hit
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
    if isinstance(hit, Module):
        return hit
    raise MisalignedModule(*hit)


def _after_fork_in_child() -> None:
    """A fork while another thread holds ``_PARSE_LOCK`` / ``_CACHE_LOCK``
    leaves the child a lock nobody releases (its next walk would hang), and a
    fork inside a parse window leaves the child's fd 2 in the capture buffer.
    The child gets fresh locks and its fd 2 back; the capture's two fds stay
    open (their owner is the parent's thread, which does not run here)."""
    global _PARSE_LOCK, _CACHE_LOCK, _REDIRECT
    _PARSE_LOCK = threading.Lock()
    _CACHE_LOCK = threading.Lock()
    redirect, _REDIRECT = _REDIRECT, None
    if redirect is not None:
        try:
            os.dup2(redirect[0], 2)
        except OSError:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
