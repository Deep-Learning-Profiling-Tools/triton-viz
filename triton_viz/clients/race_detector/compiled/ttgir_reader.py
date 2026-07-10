"""Textual TTGIR reader for the compiled-mode race detector.

Parses the printed TritonGPU IR of one kernel specialization into an
``EventGraph``: shared-memory access events, the async synchronization
structure, and enough SSA context (constants, loop iter_args, the
``addi``/``cmpi``/``select`` rotation chains) for the happens-before model.

Why text instead of the pybind walk: the bindings expose structure but not
attribute literals (``{num = 2 : i32}``, constants, ``make_range`` bounds),
and ``GluonOpBuilder.to_linear_layout`` SIGABRTs on shared encodings in the
3.6.0 wheel. The v1 op vocabulary is closed and the printed form is regular
(one op per line, SSA names unique per function, ``loc(#locN)`` trailers),
so a line parser over the closed vocabulary is the most robust option;
golden-file tests pin the printer format per triton version.

Anything that touches a memdesc outside the vocabulary marks the kernel
``unsupported`` (never silently wrong) — same policy as the dynamic mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class UnsupportedTTGIR(Exception):
    """Raised when the IR contains constructs the v1 model cannot analyze."""


# ───────────────────────────── data model ─────────────────────────────


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int
    var_name: str | None = None

    def render(self) -> str:
        base = f"{self.file}:{self.line}:{self.col}"
        return f"{base} ({self.var_name})" if self.var_name else base


@dataclass(frozen=True)
class MemDescType:
    dims: tuple[int, ...]
    elem_bits: int
    layout_alias: str
    mutable: bool


@dataclass
class Allocation:
    name: str  # SSA name of the ttg.local_alloc result
    memdesc: MemDescType
    loc: SourceLoc | None
    # Stage count: leading dim when views are taken via memdesc_index;
    # resolved by the reader from observed memdesc_index result ranks.
    stages: int = 1
    # True once a ttg.memdesc_index views this buffer: the leading memdesc dim
    # is then the stage axis, even when the depth is 1 (e.g. a single-buffered
    # ``memdesc<1x64x32>`` indexed by ``memdesc_index %buf[%idx]``). Keyed off
    # the indexing op, not ``stages > 1``, so a depth-1 staged buffer is not
    # mistaken for an un-staged one.
    has_stage_dim: bool = False

    @property
    def buffer_dims(self) -> tuple[int, ...]:
        return self.memdesc.dims[1:] if self.has_stage_dim else self.memdesc.dims

    @property
    def stage_bytes(self) -> int:
        n = 1
        for d in self.buffer_dims:
            n *= d
        return n * self.memdesc.elem_bits // 8


@dataclass
class SsaDef:
    """One parsed SSA definition (only ops the model cares about)."""

    kind: str
    operands: tuple[str, ...]
    attrs: dict[str, Any]
    line_no: int
    segment: str  # "prologue" | "loop" | "epilogue"


@dataclass
class CopyEvent:
    """``ttg.async_copy_global_to_local`` — an async-proxy smem write."""

    alloc: str
    index_ssa: str  # operand of the memdesc_index that produced the dst view
    token: str  # result token of the copy
    src_layout: str  # layout alias of the global pointer tensor
    segment: str
    body_pos: int  # program-order position within its segment
    loc: SourceLoc | None
    line_no: int


@dataclass
class LoadEvent:
    """``ttg.local_load`` — a generic-proxy smem read."""

    alloc: str
    index_ssa: str | None  # None when loading a whole (un-staged) memdesc
    token: str | None  # async token operand, if any
    result_layout: str  # layout alias/attr of the loaded tensor
    segment: str
    body_pos: int
    loc: SourceLoc | None
    line_no: int


@dataclass
class StoreEvent:
    """``ttg.local_store`` / ``ttg.local_alloc``-with-operand — generic write."""

    alloc: str
    index_ssa: str | None
    segment: str
    body_pos: int
    loc: SourceLoc | None
    line_no: int


@dataclass
class CommitEvent:
    """``ttg.async_commit_group`` — closes a commit group."""

    token: str  # result
    copy_tokens: tuple[str, ...]
    segment: str
    body_pos: int
    line_no: int


@dataclass
class DotEvent:
    """``ttng.warp_group_dot`` (sm90 wgmma) — smem reads of its memdesc
    operands.

    ``reads`` lists the (allocation, index_ssa) of every memdesc operand
    (a and/or b may instead be register-resident and then do not appear).
    ``is_async`` mirrors the printed ``isAsync = true`` attribute: an async
    wgmma's reads stay pending until a ``warp_group_dot_wait`` retires it;
    a sync one completes before the op returns.
    """

    reads: tuple[tuple[str, str | None], ...]
    is_async: bool
    segment: str
    body_pos: int
    loc: SourceLoc | None
    line_no: int


@dataclass
class DotWaitEvent:
    """``ttng.warp_group_dot_wait {pendings=N}`` — wgmma-agent counting wait:
    at most N warp-group MMAs remain pending after it returns."""

    pendings: int
    segment: str
    body_pos: int
    line_no: int


@dataclass
class WaitEvent:
    """``ttg.async_wait %tok0, %tok1, ... {num=N}``.

    ``operand_tokens`` are the async-token SSA operands the wait synchronizes
    on (loop-carried commit-group tokens in the pipelined loop). They identify
    *which* allocations the wait actually awaits; an empty tuple is the
    operandless ``async_wait {num=0}`` "wait for all groups" form. The HB
    model uses these to gate coverage per allocation (see hb.py) so that
    dropping a token — a wait that no longer awaits an allocation it is
    assumed to guard — degrades to a report rather than a silent ``ok``.
    """

    result: str | None
    num: int
    operand_tokens: tuple[str, ...]
    segment: str
    body_pos: int
    loc: SourceLoc | None
    line_no: int


@dataclass
class LoopInfo:
    induction_var: str
    lower: str
    upper: str
    step: str
    # iter_args in order: (block_arg_name, init_operand_name)
    iter_args: tuple[tuple[str, str], ...] = ()
    yields: tuple[str, ...] = ()
    line_no: int = 0


@dataclass
class EventGraph:
    num_warps: int
    threads_per_warp: int
    target: str
    layouts: dict[str, str]  # alias -> attribute text
    allocations: dict[str, Allocation]
    defs: dict[str, SsaDef]
    constants: dict[str, int]
    loop: LoopInfo | None
    copies: list[CopyEvent] = field(default_factory=list)
    loads: list[LoadEvent] = field(default_factory=list)
    stores: list[StoreEvent] = field(default_factory=list)
    commits: list[CommitEvent] = field(default_factory=list)
    waits: list[WaitEvent] = field(default_factory=list)
    dots: list[DotEvent] = field(default_factory=list)
    dot_waits: list[DotWaitEvent] = field(default_factory=list)
    kernel_name: str = ""

    def iter_arg_init(self, arg_name: str) -> str | None:
        if self.loop is None:
            return None
        for arg, init in self.loop.iter_args:
            if arg == arg_name:
                return init
        return None

    def yielded_for_arg(self, arg_name: str) -> str | None:
        """SSA name yielded into ``arg_name`` for the next iteration."""
        if self.loop is None:
            return None
        for pos, (arg, _init) in enumerate(self.loop.iter_args):
            if arg == arg_name:
                if pos < len(self.loop.yields):
                    return self.loop.yields[pos]
        return None


# ───────────────────────────── regexes ─────────────────────────────

_SSA = r"%[\w.\-#]+"

_RE_ALIAS = re.compile(r"^(#\w+) = (#ttg\.\w+<\{.*\}>|#ttg\.\w+)\s*$")
_RE_LOC_FILE = re.compile(r'^(#loc\d*) = loc\("([^"]+)":(\d+):(\d+)\)')
_RE_LOC_NAME = re.compile(r'^(#loc\d*) = loc\("([^"]+)"\((#loc\d*)\)\)')
_RE_LOC_CALLSITE = re.compile(r"^(#loc\d*) = loc\(callsite\((#loc\d*) at (#loc\d*)\)\)")
_RE_LOC_TRAILER = re.compile(r"loc\((#loc\d*|#loc)\)\s*$")
_RE_MODULE_ATTRS = re.compile(r"^module attributes \{(.*)\} \{")
_RE_FUNC = re.compile(r"tt\.func\s+\w+\s+@(\w+)\(")
_RE_RESULT = re.compile(rf"^({_SSA})(?::(\d+))?\s*=\s*(.*)$")
_RE_CONST_INT = re.compile(r"^arith\.constant (-?\d+) : i\d+")
_RE_ADDI = re.compile(rf"^arith\.addi ({_SSA}), ({_SSA}) : i32")
_RE_CMPI = re.compile(rf"^arith\.cmpi (\w+), ({_SSA}), ({_SSA}) : i32")
_RE_SELECT = re.compile(rf"^arith\.select ({_SSA}), ({_SSA}), ({_SSA}) : i32")
_RE_LOCAL_ALLOC = re.compile(
    rf"^ttg\.local_alloc\s*({_SSA})?\s*:.*!ttg\.memdesc<([^>]+)>"
)
_RE_MEMDESC_INDEX = re.compile(rf"^ttg\.memdesc_index ({_SSA})\[({_SSA})\]")
_RE_ASYNC_COPY = re.compile(
    rf"^ttg\.async_copy_global_to_local ({_SSA}), ({_SSA})"
    rf"(?: mask ({_SSA}))?(?: other ({_SSA}))?"
    rf".*?tensor<.*?,\s*(#\w+)>"
)
_RE_COMMIT = re.compile(r"^ttg\.async_commit_group(?:\s+tokens\s+(.+?))?\s*(?:loc|$)")
_RE_ASYNC_WAIT = re.compile(
    rf"^ttg\.async_wait((?:\s+{_SSA},?)*)\s*\{{num = (\d+) : i32\}}"
)
_RE_LOCAL_LOAD = re.compile(
    rf"^ttg\.local_load ({_SSA})(?: token ({_SSA}))?\s*:.*->\s*tensor<[^,>]+,\s*(.+?)>\s*(?:loc|$)"
)
_RE_LOCAL_STORE = re.compile(rf"^ttg\.local_store ({_SSA}), ({_SSA})")
_RE_WARP_GROUP_DOT = re.compile(rf"^ttng\.warp_group_dot ((?:{_SSA},?\s*)+)\{{")
_RE_WARP_GROUP_DOT_WAIT = re.compile(
    rf"^ttng\.warp_group_dot_wait ((?:{_SSA},?\s*)*)\{{pendings = (\d+) : i32\}}"
)
_RE_SCF_FOR = re.compile(
    rf"^(?:({_SSA})(?::\d+)?\s*=\s*)?scf\.for ({_SSA}) = ({_SSA}) to ({_SSA}) "
    rf"step ({_SSA})(?: iter_args\((.*?)\))?\s*->"
)
_RE_SCF_FOR_NOARGS = re.compile(
    rf"^(?:({_SSA})(?::\d+)?\s*=\s*)?scf\.for ({_SSA}) = ({_SSA}) to ({_SSA}) step ({_SSA})\s*(?::| \{{)"
)
_RE_SCF_YIELD = re.compile(r"^scf\.yield (.*?) : ")
_RE_MEMDESC_TYPE = re.compile(
    r"^([\dx]+)x([a-z]+\d+),\s*(#\w+),\s*(#\w+)(,\s*mutable)?"
)

# ttg/ttng/gpu ops that the v1 model understands or can safely ignore.
_KNOWN_TTG_OPS = {
    "ttg.local_alloc",
    "ttg.local_load",
    "ttg.local_store",
    "ttg.local_dealloc",
    "ttg.memdesc_index",
    "ttg.async_copy_global_to_local",
    "ttg.async_commit_group",
    "ttg.async_wait",
    "ttg.convert_layout",  # smem scratch is internal; ordered by Membar
}

# The sm90 subset (M4 tranche 1): the wgmma agent and its counting wait.
# fence_async_shared only ADDS ordering (generic→async proxy); the model
# never relies on it for a proof, and the shapes where ignoring it could
# hide one behind a report (generic stores mixed with async ops on one
# allocation) are already gated unsupported in analyze_graph.
_KNOWN_TTNG_OPS = {
    "ttng.warp_group_dot",
    "ttng.warp_group_dot_wait",
    "ttng.fence_async_shared",
}

_DTYPE_BITS = {
    "f64": 64,
    "f32": 32,
    "f16": 16,
    "bf16": 16,
    "f8": 8,
    "i64": 64,
    "i32": 32,
    "i16": 16,
    "i8": 8,
    "i1": 1,
}


def _parse_memdesc(body: str) -> MemDescType:
    m = _RE_MEMDESC_TYPE.match(body)
    if not m:
        raise UnsupportedTTGIR(f"unparsable memdesc type: {body!r}")
    dims_s, dtype, layout, _space, mutable = m.groups()
    dims = tuple(int(d) for d in dims_s.split("x") if d)
    bits = _DTYPE_BITS.get(dtype)
    if bits is None:
        raise UnsupportedTTGIR(f"unknown memdesc element type {dtype!r}")
    return MemDescType(
        dims=dims, elem_bits=bits, layout_alias=layout, mutable=bool(mutable)
    )


def _split_ssa_list(text: str) -> tuple[str, ...]:
    return tuple(t.strip() for t in text.split(",") if t.strip().startswith("%"))


class _LocTable:
    def __init__(self) -> None:
        self.raw: dict[str, tuple[str, ...]] = {}

    def add_line(self, line: str) -> bool:
        m = _RE_LOC_FILE.match(line)
        if m:
            self.raw[m.group(1)] = ("file", m.group(2), m.group(3), m.group(4))
            return True
        m = _RE_LOC_NAME.match(line)
        if m:
            self.raw[m.group(1)] = ("name", m.group(2), m.group(3))
            return True
        m = _RE_LOC_CALLSITE.match(line)
        if m:
            self.raw[m.group(1)] = ("callsite", m.group(3))
            return True
        if line.startswith("#loc") and "= loc(" in line:
            self.raw[line.split(" =")[0]] = ("unknown",)
            return True
        return False

    def resolve(self, loc_id: str | None, _depth: int = 0) -> SourceLoc | None:
        if loc_id is None or _depth > 8:
            return None
        entry = self.raw.get(loc_id)
        if entry is None:
            return None
        kind = entry[0]
        if kind == "file":
            return SourceLoc(entry[1], int(entry[2]), int(entry[3]))
        if kind == "name":
            inner = self.resolve(entry[2], _depth + 1)
            if inner is None:
                return None
            return SourceLoc(inner.file, inner.line, inner.col, entry[1])
        if kind == "callsite":
            return self.resolve(entry[1], _depth + 1)
        return None


def parse_ttgir(text: str) -> EventGraph:
    """Parse one TTGIR module into an EventGraph.

    Raises :class:`UnsupportedTTGIR` for constructs outside the v1 model
    (ttng ops, nested loops with smem events, unparsable memdescs, ...).
    """
    layouts: dict[str, str] = {}
    locs = _LocTable()
    num_warps = 4
    threads_per_warp = 32
    target = ""
    kernel_name = ""

    defs: dict[str, SsaDef] = {}
    constants: dict[str, int] = {}
    allocations: dict[str, Allocation] = {}
    # memdesc_index result -> (alloc name, index ssa)
    views: dict[str, tuple[str, str]] = {}
    loop: LoopInfo | None = None

    copies: list[CopyEvent] = []
    loads: list[LoadEvent] = []
    stores: list[StoreEvent] = []
    commits: list[CommitEvent] = []
    waits: list[WaitEvent] = []
    dots: list[DotEvent] = []
    dot_waits: list[DotWaitEvent] = []

    segment = "prologue"
    loop_depth = 0
    seen_loop_with_events = False
    # Set once a ttg.local_dealloc frees a buffer. v1 does not model the
    # allocation base / live ranges, so a buffer that is freed and then
    # re-allocated (storage aliasing) is outside the model — a local_alloc
    # after a dealloc must degrade to unsupported. A *terminal* dealloc (the
    # stock epilogue cleanup, with no later alloc) is harmless and ignored.
    seen_dealloc = False
    body_pos = {"prologue": 0, "loop": 0, "epilogue": 0}

    def next_pos() -> int:
        body_pos[segment] += 1
        return body_pos[segment]

    def resolve_view(name: str, line_no: int) -> tuple[str, str | None]:
        if name in views:
            alloc, idx = views[name]
            return alloc, idx
        if name in allocations:
            return name, None
        raise UnsupportedTTGIR(
            f"line {line_no}: memdesc operand {name} does not resolve to a "
            "local_alloc (unsupported producer)"
        )

    lines = text.splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if locs.add_line(line):
            continue
        m = _RE_ALIAS.match(line)
        if m:
            layouts[m.group(1)] = m.group(2)
            continue
        m = _RE_MODULE_ATTRS.match(line)
        if m:
            attrs = m.group(1)
            mw = re.search(r'"ttg\.num-warps" = (\d+)', attrs)
            if mw:
                num_warps = int(mw.group(1))
            mt = re.search(r'"ttg\.threads-per-warp" = (\d+)', attrs)
            if mt:
                threads_per_warp = int(mt.group(1))
            mg = re.search(r'ttg\.target = "([^"]+)"', attrs)
            if mg:
                target = mg.group(1)
            continue
        m = _RE_FUNC.search(line)
        if m and not kernel_name:
            kernel_name = m.group(1)
            continue

        loc_m = _RE_LOC_TRAILER.search(line)
        loc_id = loc_m.group(1) if loc_m else None
        # Loc aliases are defined at the BOTTOM of the printed module; store
        # the id now and resolve after the full text has been scanned.
        loc: Any = loc_id

        # Region tracking: scf.for opens a region; its closing "}" returns
        # to the parent segment. Other ops with regions (scf.if) are
        # unsupported when they contain smem events — detected by vocabulary
        # check below (their body ops still get scanned).
        results: list[str] = []
        body = line
        rm = _RE_RESULT.match(line)
        if rm:
            results = [rm.group(1)]
            body = rm.group(3)

        if body.startswith("scf.for"):
            fm = _RE_SCF_FOR.match(body) or _RE_SCF_FOR_NOARGS.match(body)
            if not fm:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable scf.for")
            if loop is not None and seen_loop_with_events:
                raise UnsupportedTTGIR(
                    f"line {line_no}: multiple loops with shared-memory "
                    "events are not supported in v1"
                )
            groups = fm.groups()
            iter_args: list[tuple[str, str]] = []
            if len(groups) >= 6 and groups[5]:
                for pair in re.findall(rf"({_SSA}) = ({_SSA})", groups[5]):
                    iter_args.append((pair[0], pair[1]))
            loop = LoopInfo(
                induction_var=groups[1],
                lower=groups[2],
                upper=groups[3],
                step=groups[4],
                iter_args=tuple(iter_args),
                line_no=line_no,
            )
            segment = "loop"
            loop_depth = 1
            continue

        if segment == "loop":
            if body.startswith("scf.yield"):
                ym = _RE_SCF_YIELD.match(body)
                if ym and loop is not None:
                    loop.yields = _split_ssa_list(ym.group(1))
                continue
            if line == "}" or line.startswith("} loc"):
                loop_depth -= 1
                if loop_depth == 0:
                    segment = "epilogue"
                continue
            if body.endswith("{"):
                # A line opening a region inside the loop body — an scf.if /
                # scf.while opener, or a "} else {" continuation that both
                # closes and reopens a region. v1 does not model conditional
                # or nested regions inside the pipelined loop: their smem
                # events are not unconditional, and naive brace counting
                # mis-tracks the loop/epilogue boundary (a "} else {" nets +1,
                # not 0, leaving every epilogue event mislabeled "loop").
                # Fail fast rather than silently mis-model.
                raise UnsupportedTTGIR(
                    f"line {line_no}: nested/conditional region inside a "
                    "pipelined loop is not modeled in v1"
                )

        op_kind = body.split(" ")[0].rstrip(",")

        # Vocabulary guard: any ttng op outside the modeled sm90 subset, or
        # a ttg/gpu op outside the known set, is outside the model.
        if op_kind.startswith("ttng.") and op_kind not in _KNOWN_TTNG_OPS:
            raise UnsupportedTTGIR(
                f"line {line_no}: {op_kind} is not modeled "
                "(Hopper TMA/mbarrier / Blackwell path — see plan M4)"
            )
        if op_kind == "gpu.barrier":
            raise UnsupportedTTGIR(
                f"line {line_no}: explicit gpu.barrier at TTGIR is outside "
                "the v1 happens-before model"
            )
        if op_kind.startswith("ttg.") and op_kind not in _KNOWN_TTG_OPS:
            raise UnsupportedTTGIR(f"line {line_no}: unmodeled op {op_kind}")

        cm = _RE_CONST_INT.match(body)
        if cm and results:
            constants[results[0]] = int(cm.group(1))
            continue
        am = _RE_ADDI.match(body)
        if am and results:
            defs[results[0]] = SsaDef("addi", am.groups(), {}, line_no, segment)
            continue
        am = _RE_CMPI.match(body)
        if am and results:
            defs[results[0]] = SsaDef(
                "cmpi",
                (am.group(2), am.group(3)),
                {"pred": am.group(1)},
                line_no,
                segment,  # type: ignore[dict-item]
            )
            continue
        am = _RE_SELECT.match(body)
        if am and results:
            defs[results[0]] = SsaDef("select", am.groups(), {}, line_no, segment)
            continue

        if op_kind == "ttg.local_dealloc":
            # Buffer freed. Harmless on its own (terminal cleanup); only a
            # subsequent local_alloc — potential storage reuse — is a problem.
            seen_dealloc = True
            continue

        if op_kind == "ttg.local_alloc":
            if seen_dealloc:
                raise UnsupportedTTGIR(
                    f"line {line_no}: local_alloc after local_dealloc — buffer "
                    "reuse / allocation aliasing is not modeled in v1"
                )
            lm = _RE_LOCAL_ALLOC.match(body)
            if not lm or not results:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable local_alloc")
            operand, memdesc_body = lm.group(1), lm.group(2)
            memdesc = _parse_memdesc(memdesc_body)
            allocations[results[0]] = Allocation(results[0], memdesc, loc)
            if operand is not None:
                stores.append(
                    StoreEvent(results[0], None, segment, next_pos(), loc, line_no)
                )
                if segment == "loop":
                    seen_loop_with_events = True
            continue

        if op_kind == "ttg.memdesc_index":
            im = _RE_MEMDESC_INDEX.match(body)
            if not im or not results:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable memdesc_index")
            base, idx = im.group(1), im.group(2)
            alloc, parent_idx = resolve_view(base, line_no)
            if parent_idx is not None:
                raise UnsupportedTTGIR(
                    f"line {line_no}: nested memdesc_index is unsupported"
                )
            views[results[0]] = (alloc, idx)
            allocations[alloc].stages = allocations[alloc].memdesc.dims[0]
            allocations[alloc].has_stage_dim = True
            continue

        if op_kind == "ttg.async_copy_global_to_local":
            am2 = _RE_ASYNC_COPY.match(body)
            if not am2 or not results:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable async_copy")
            dst = am2.group(2)
            alloc, idx = resolve_view(dst, line_no)
            copies.append(
                CopyEvent(
                    alloc=alloc,
                    index_ssa=idx if idx is not None else "",
                    token=results[0],
                    src_layout=am2.group(5),
                    segment=segment,
                    body_pos=next_pos(),
                    loc=loc,
                    line_no=line_no,
                )
            )
            if segment == "loop":
                seen_loop_with_events = True
            continue

        if op_kind == "ttg.async_commit_group":
            cm2 = _RE_COMMIT.match(body)
            if cm2 is None:
                # A genuinely unparsable commit group (e.g. operand-style
                # ``async_commit_group %tok`` from a printer/version we have
                # not modeled). The bare ``async_commit_group`` (no tokens)
                # form DOES match — that is a real, intentionally-empty group.
                # Silently treating a non-match as an empty group would corrupt
                # commit-rank accounting, so fail fast like every other smem op.
                raise UnsupportedTTGIR(f"line {line_no}: unparsable async_commit_group")
            if not results:
                # The op always prints its !ttg.async.token result; a
                # result-less commit group is malformed. Its token is what a
                # wait names (commit_by_result in hb.py keys on it), so a group
                # with no token is unreachable by any wait — fail closed rather
                # than fabricate an empty token.
                raise UnsupportedTTGIR(
                    f"line {line_no}: async_commit_group without an SSA result token"
                )
            tokens = _split_ssa_list(cm2.group(1)) if cm2.group(1) else ()
            commits.append(
                CommitEvent(
                    token=results[0],
                    copy_tokens=tokens,
                    segment=segment,
                    body_pos=next_pos(),
                    line_no=line_no,
                )
            )
            continue

        if op_kind == "ttg.async_wait":
            wm = _RE_ASYNC_WAIT.match(body)
            if not wm:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable async_wait")
            waits.append(
                WaitEvent(
                    result=results[0] if results else None,
                    num=int(wm.group(2)),
                    operand_tokens=_split_ssa_list(wm.group(1) or ""),
                    segment=segment,
                    body_pos=next_pos(),
                    loc=loc,
                    line_no=line_no,
                )
            )
            continue

        if op_kind == "ttg.local_load":
            lm2 = _RE_LOCAL_LOAD.match(body)
            if not lm2 or not results:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable local_load")
            alloc, idx = resolve_view(lm2.group(1), line_no)
            loads.append(
                LoadEvent(
                    alloc=alloc,
                    index_ssa=idx,
                    token=lm2.group(2),
                    result_layout=lm2.group(3),
                    segment=segment,
                    body_pos=next_pos(),
                    loc=loc,
                    line_no=line_no,
                )
            )
            if segment == "loop":
                seen_loop_with_events = True
            continue

        if op_kind == "ttg.local_store":
            sm = _RE_LOCAL_STORE.match(body)
            if not sm:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable local_store")
            alloc, idx = resolve_view(sm.group(2), line_no)
            stores.append(StoreEvent(alloc, idx, segment, next_pos(), loc, line_no))
            if segment == "loop":
                seen_loop_with_events = True
            continue

        if op_kind == "ttng.warp_group_dot":
            dm = _RE_WARP_GROUP_DOT.match(body)
            if not dm:
                raise UnsupportedTTGIR(f"line {line_no}: unparsable warp_group_dot")
            operands = _split_ssa_list(dm.group(1))
            reads: list[tuple[str, str | None]] = []
            for name in operands:
                if name in views or name in allocations:
                    reads.append(resolve_view(name, line_no))
            # Every memdesc operand must have resolved to a local_alloc view;
            # one produced some other way (memdesc_trans, subslice, ...) would
            # otherwise silently drop a real smem read from the model. The
            # operand types sit between ':' and '->' in the printed op.
            type_sig = body.split(" : ", 1)[1] if " : " in body else ""
            n_memdesc = type_sig.split("->", 1)[0].count("memdesc")
            if n_memdesc != len(reads):
                raise UnsupportedTTGIR(
                    f"line {line_no}: warp_group_dot has {n_memdesc} memdesc "
                    f"operands but only {len(reads)} resolve to local_allocs"
                )
            dots.append(
                DotEvent(
                    reads=tuple(reads),
                    is_async="isAsync = true" in body,
                    segment=segment,
                    body_pos=next_pos(),
                    loc=loc,
                    line_no=line_no,
                )
            )
            if segment == "loop" and reads:
                seen_loop_with_events = True
            continue

        if op_kind == "ttng.warp_group_dot_wait":
            wm2 = _RE_WARP_GROUP_DOT_WAIT.match(body)
            if not wm2:
                raise UnsupportedTTGIR(
                    f"line {line_no}: unparsable warp_group_dot_wait"
                )
            dot_waits.append(
                DotWaitEvent(
                    pendings=int(wm2.group(2)),
                    segment=segment,
                    body_pos=next_pos(),
                    line_no=line_no,
                )
            )
            continue

        if op_kind == "ttng.fence_async_shared":
            # Proxy fence: adds generic→async ordering. Never load-bearing
            # for the modeled proofs (see _KNOWN_TTNG_OPS note); no event.
            continue

        # ttg.convert_layout / tt.* / arith.* on tensors: not events in the v1
        # model. (ttg.local_dealloc is handled above.)

    # Resolve loc ids (aliases live at the bottom of the file).
    event_lists: list[list[Any]] = [copies, loads, stores, waits, dots]
    for ev_list in event_lists:
        for ev in ev_list:
            if isinstance(ev.loc, str):
                ev.loc = locs.resolve(ev.loc)
    for alloc_obj in allocations.values():
        if isinstance(alloc_obj.loc, str):
            alloc_obj.loc = locs.resolve(alloc_obj.loc)

    return EventGraph(
        num_warps=num_warps,
        threads_per_warp=threads_per_warp,
        target=target,
        layouts=layouts,
        allocations=allocations,
        defs=defs,
        constants=constants,
        loop=loop,
        copies=copies,
        loads=loads,
        stores=stores,
        commits=commits,
        waits=waits,
        dots=dots,
        dot_waits=dot_waits,
        kernel_name=kernel_name,
    )
