"""CuTile IR reader — the cuda.tile front-end of the shared access-graph
model.

Parses the FINAL CuTile IR text (``cuda.tile._compile.compile_tile(...,
return_final_ir=True)``, captured at launch time by
evaluation/tilebench_cutile_capture) into the SAME
:class:`~.ttir_reader.AccessGraph` the TTIR reader produces, so the
compiled race-detector track (``encode_graph`` →
``TwoCopySymbolicHBSolver``, tier selector, launch-scoped rung) runs
unchanged on cuTile kernels.

Semantic mapping (why this is a thin front-end, not a new model):

- ``tile_bid(axis)`` ≡ ``tt.get_program_id`` → :class:`Pid`.
- Tile-space addressing lowers to the SAME affine algebra Triton kernels
  hand-write: ``tile_load(view, index=(i,))`` over a partition view with
  ``tile_shape=(T,)`` has footprint ``i*T + arange(0,T)`` per axis,
  scaled by the array view's strides and CLIPPED to its logical shape —
  cuTile has no explicit masks; the reader materializes the implicit
  OOB-drop semantics as ordinary mask terms ``0 <= off < shape_axis``.
- ``pointer_offset`` + ``tile_atomic_rmw(pointer, update, mask, ...)``
  is exactly the TTIR ``addptr`` + ``tt.atomic_rmw`` shape (the cuTile
  compiler routes per-element atomics through raw pointers and emits the
  bounds-check mask itself); it lowers to the same atomic events.
- Structured ``for $i in range(a, b, c)`` loops map to the single
  :class:`LoopInfo` slot with Term bounds; loop-carried non-token values
  bind to :class:`DataDep` (they are tile VALUES — cuTile advances
  addresses by index arithmetic, not carried pointers).
- Scalar params keep their python names; an array param ``p`` arrives
  flattened as ``p_0`` (base pointer), ``p_1..p_r`` (shape dims) and
  ``p_{r+1}..p_{2r}`` (strides). Metadata slots become :class:`Param`
  terms under their FLATTENED names — the harness binds their values
  from the captured descriptors.

Uncertainty discipline is inherited verbatim: an unmodeled op binds its
results to :class:`DataDep` (never an exception); DataDep reaching an
address raises :class:`UnsupportedTTIR` (kind="indirect-address"),
reaching a mask drops it and flags ``mask_dropped`` (widened, proof-only),
reaching an atomic update clears ``atomic_val``. Unknown BLOCK structure
fails closed (kind="control-flow").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ttir_reader import (
    AccessEvent,
    AccessGraph,
    Arange,
    AtomicInfo,
    Bin,
    BoolBin,
    Cmp,
    Const,
    DataDep,
    FuncArg,
    LoopInfo,
    LoopVar,
    Not,
    Param,
    Pid,
    PtrValue,
    Select,
    Term,
    UnsupportedTTIR,
    _set_arange_dim,
)

_DTYPE_BITS = {
    "float64": 64, "float32": 32, "float16": 16, "bfloat16": 16,
    "int64": 64, "int32": 32, "int16": 16, "int8": 8,
    "uint64": 64, "uint32": 32, "uint16": 16, "uint8": 8,
    "bool_": 1,
    "float8_e4m3fn": 8, "float8_e5m2": 8, "float8_e8m0fnu": 8,
}  # fmt: skip
_FLOAT_DTYPES = {d for d in _DTYPE_BITS if d.startswith(("float", "bfloat"))}

_CMP_FN = {"lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge", "eq": "eq", "ne": "ne"}
# c_mod is C-style truncation-toward-zero — exactly the remsi semantics
# Bin("%") already carries (python floor-mod is SYNTHESIZED from it by a
# sign-fix select the reader models faithfully via boolean xor below)
_ARITH_FN = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//", "mod": "%",
             "c_mod": "%", "min": "min", "max": "max"}  # fmt: skip
_RMW_MODE = {
    "ADD_INT": "add", "ADD_FLOAT": "fadd", "MIN_INT": "min", "MAX_INT": "max",
    "MIN_FLOAT": "fmin", "MAX_FLOAT": "fmax", "AND": "and", "OR": "or",
    "XOR": "xor", "EXCHANGE": "exch",
}  # fmt: skip
_SCOPE = {"DEVICE": "gpu", "BLOCK": "cta", "SYSTEM": "sys", "NONE": "gpu"}


class _Token:
    """Memory-ordering token — opaque to footprints."""


_TOKEN = _Token()


@dataclass
class _ArrayView:
    base: str  # python param name
    shape: list[Any]  # int | Term per axis
    strides: list[Any]  # int | Term per axis
    dtype: str


@dataclass
class _PartView:
    array: _ArrayView
    tile_shape: list[int]
    padding: str


# ───────────────────────── text utilities ─────────────────────────

_NAME = r"[$\w.]+"
_RE_TYPED_NAME = re.compile(rf"^\s*({_NAME})(?:\{{[^}}]*\}})?\s*:\s*(.*)$")
_RE_OP = re.compile(r"^(\w+)\((.*)\)$")
_RE_FOR = re.compile(rf"^for ({_NAME}) in range\((.*?)\)(?:\s*\(with (.*)\))?\s*$")
_RE_TILE_TYPE = re.compile(r"^(?:const )?Tile\[(\w+),\(([^)]*)\)\]")
_RE_ARRAY_TYPE = re.compile(r"^Array\[(\w+),\(([^)]*)\):\(([^)]*)\)\]")
_RE_PARTVIEW_TYPE = re.compile(
    r"^PartitionView\[.*tile_shape=\(([^)]*)\),order=\(([^)]*)\),"
    r"padding_mode=PaddingMode\.(\w+)\]"
)


def _split_top(s: str, sep: str = ",") -> list[str]:
    """Split at top level, respecting (), [] and {} nesting."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_prov(name: str) -> str:
    """``$73{input_ptr_0, $0}`` → ``$73`` (provenance braces are display
    metadata, not part of the SSA name)."""
    return name.split("{", 1)[0].strip()


def _parse_kwargs(argstr: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in _split_top(argstr):
        if not item:
            continue
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


def _ints(csv: str) -> list[int]:
    return [int(x) for x in _split_top(csv) if x != ""]


# ───────────────────────── the walker ─────────────────────────


@dataclass
class _State:
    kernel_name: str
    env: dict[str, Any] = field(default_factory=dict)
    accesses: list[AccessEvent] = field(default_factory=list)
    pid_axes: set[int] = field(default_factory=set)
    loop: LoopInfo | None = None
    in_loop: bool = False
    arange_n: int = 0
    unknown_ops: dict[str, int] = field(default_factory=dict)
    func_args: list[FuncArg] = field(default_factory=list)
    ptr_meta: dict[str, tuple[int, bool]] = field(default_factory=dict)


def _as_term(v: Any, ctx: str) -> Term:
    if isinstance(v, (int, bool)):
        return Const(int(v))
    if isinstance(
        v, (Const, Pid, Param, Arange, LoopVar, Bin, Cmp, BoolBin, Select, DataDep)
    ):
        return v  # type: ignore[return-value]
    return DataDep(f"{ctx}: unmodeled value {type(v).__name__}")


def _datadep_whys(t: Any, out: set[str] | None = None) -> set[str]:
    """Every DataDep reason inside ``t`` — the abstention message names the
    actual polluter (an unmodeled op vs genuinely loaded data)."""
    if out is None:
        out = set()
    if isinstance(t, DataDep):
        out.add(t.why)
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(t, attr, None)
        if sub is not None:
            _datadep_whys(sub, out)
    return out


def _has_datadep(t: Any) -> bool:
    return bool(_datadep_whys(t))


def parse_cutile_ir(text: str, kernel_name: str = "cutile_kernel") -> AccessGraph:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise UnsupportedTTIR("empty CuTile IR", kind="parse")
    st = _State(kernel_name=kernel_name)
    _parse_header(lines[0], st)
    _walk(lines, 1, 0, st)
    return AccessGraph(
        kernel_name=kernel_name,
        func_args=st.func_args,
        accesses=st.accesses,
        loop=st.loop,
        iter_args={},
        pid_axes=st.pid_axes,
    )


def _parse_header(line: str, st: _State) -> None:
    """``(a_0: Tile[pointer[float32],()], ..., N: Tile[int32,()]):``"""
    inner = line.strip()
    if not (inner.startswith("(") and inner.endswith("):")):
        raise UnsupportedTTIR(f"unrecognized IR header: {line[:80]}", kind="parse")
    ptr_bases: dict[str, tuple[int, bool]] = {}
    scalars: list[str] = []
    for item in _split_top(inner[1:-2]):
        m = _RE_TYPED_NAME.match(item)
        if not m:
            raise UnsupportedTTIR(f"unparsable param {item!r}", kind="parse")
        name, typ = m.group(1), m.group(2)
        pm = re.match(r"Tile\[pointer\[(\w+)\],", typ)
        if pm:
            dt = pm.group(1)
            bits = _DTYPE_BITS.get(dt)
            if bits is None:
                raise UnsupportedTTIR(f"unknown pointee dtype {dt}", kind="parse")
            if not name.endswith("_0"):
                raise UnsupportedTTIR(
                    f"pointer param {name!r} outside the p_0 flattening " "convention",
                    kind="parse",
                )
            base = name[:-2]
            ptr_bases[base] = (bits, dt in _FLOAT_DTYPES)
            st.env[name] = PtrValue(base, Const(0))
        else:
            scalars.append(name)
            st.env[name] = Param(name)
    st.ptr_meta = ptr_bases
    for base, (bits, is_f) in ptr_bases.items():
        st.func_args.append(FuncArg(base, True, bits, is_f))
    for name in scalars:
        # p_1/p_2/... metadata slots of a flattened array param are not
        # python-level scalars; everything else is
        m = re.match(r"^(.*)_(\d+)$", name)
        if m and m.group(1) in ptr_bases:
            continue
        st.func_args.append(FuncArg(name, False, 0))


def _skip_block(lines: list[str], i: int, indent: int) -> int:
    """Skip a ``do (...)``-introduced nested block (reduce/scan combiner
    lambdas): the ``do`` line, then everything indented deeper, through
    the block's own terminator."""
    i += 1  # the `do (...)` line
    n = len(lines)
    while i < n:
        raw = lines[i]
        cur = len(raw) - len(raw.lstrip())
        if cur <= indent and not raw.strip().startswith(("(", "continue", "end")):
            return i
        if cur <= indent and raw.strip().startswith(("continue", "end")):
            return i + 1
        i += 1
    return i


def _walk(lines: list[str], i: int, indent: int, st: _State) -> int:
    """Process ops at ``indent`` until dedent / return / continue."""
    n = len(lines)
    while i < n:
        raw = lines[i]
        cur_indent = len(raw) - len(raw.lstrip())
        if cur_indent < indent:
            return i
        line = raw.strip()
        if line == "return":
            return i + 1
        if line.startswith("continue"):
            return i + 1
        if line.startswith("do ("):
            # combiner lambda of a value-level op (tile_reduce/scan): its
            # results were already bound DataDep by the op line — the
            # block body is pure value math, skip it wholesale
            i = _skip_block(lines, i, cur_indent)
            continue
        i = _handle_line(lines, i, indent, line, st)
    return i


def _handle_line(lines: list[str], i: int, indent: int, line: str, st: _State) -> int:
    lhs, eq, rhs = line.partition(" = ")
    if not eq:
        # statement forms: a result-free `for` (no carried tokens) and
        # if-block keywords
        fm = _RE_FOR.match(line)
        if fm:
            return _handle_for(lines, i, indent, [], fm, st)
        if line in ("then", "else") or line.startswith(("then", "else", "if ")):
            raise UnsupportedTTIR(
                f"line {i + 1}: `if` block structure is not modeled",
                kind="control-flow",
            )
        raise UnsupportedTTIR(
            f"line {i + 1}: unrecognized statement {line[:60]!r}",
            kind="control-flow",
        )
    results = []
    for item in _split_top(lhs):
        m = _RE_TYPED_NAME.match(item)
        if not m:
            raise UnsupportedTTIR(
                f"line {i + 1}: unparsable result {item!r}", kind="parse"
            )
        results.append((_strip_prov(m.group(1)), m.group(2)))

    fm = _RE_FOR.match(rhs)
    if fm:
        return _handle_for(lines, i, indent, results, fm, st)
    if rhs.startswith("loop ") or rhs.startswith("loop("):
        raise UnsupportedTTIR(
            f"line {i + 1}: while-form `loop` construct (carried values, "
            "data-dependent trip) is not modeled",
            kind="control-flow",
        )
    if rhs.startswith("if ") or rhs == "if":
        raise UnsupportedTTIR(
            f"line {i + 1}: `if` block structure is not modeled",
            kind="control-flow",
        )

    om = _RE_OP.match(rhs)
    if not om:
        raise UnsupportedTTIR(
            f"line {i + 1}: unrecognized op form {rhs[:60]!r}", kind="parse"
        )
    op, kwargs = om.group(1), _parse_kwargs(om.group(2))
    _handle_op(op, results, kwargs, i + 1, st)
    return i + 1


# ───────────────────────── op handlers ─────────────────────────


def _val(st: _State, token: str) -> Any:
    token = _strip_prov(token)
    if token in st.env:
        return st.env[token]
    if re.fullmatch(r"-?\d+", token):
        return Const(int(token))
    if token in ("None", "True", "False"):
        return {"None": None, "True": Const(1), "False": Const(0)}[token]
    return DataDep(f"unresolved SSA {token}")


def _tuple_vals(st: _State, s: str) -> list[Any]:
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return [_val(st, t) for t in _split_top(s) if t]


def _tile_shape_of(typ: str) -> list[int] | None:
    m = _RE_TILE_TYPE.match(typ)
    if not m:
        return None
    dims = m.group(2)
    if not dims.strip():
        return []
    try:
        return _ints(dims)
    except ValueError:
        return None


def _new_arange(st: _State, size: int, dim: int) -> Term:
    st.arange_n += 1
    return Arange(f"ct_ar{st.arange_n}", 0, size, dim)


def _handle_for(
    lines: list[str],
    i: int,
    indent: int,
    results: list[tuple[str, str]],
    fm: re.Match,
    st: _State,
) -> int:
    if st.in_loop or st.loop is not None:
        raise UnsupportedTTIR(
            f"line {i + 1}: multiple/nested loops", kind="nested-loop"
        )
    iv = _strip_prov(fm.group(1))
    bounds = [_as_term(_val(st, b), "loop bound") for b in _split_top(fm.group(2))]
    if len(bounds) == 2:
        bounds = [Const(0), *bounds, Const(1)][:3]
    if len(bounds) != 3:
        raise UnsupportedTTIR(
            f"line {i + 1}: range() with {len(bounds)} bounds", kind="parse"
        )
    st.loop = LoopInfo(
        loop_ssa=iv, induction_var=iv, lower=bounds[0], upper=bounds[1], step=bounds[2]
    )
    # `do (params)` line, then the body header `(params):` one level in
    j = i + 1
    if j < len(lines) and lines[j].strip().startswith("do ("):
        j += 1
    body_indent = indent + 4
    if j < len(lines):
        hdr = lines[j].strip()
        if hdr.startswith("(") and hdr.endswith("):"):
            for item in _split_top(hdr[1:-2]):
                m = _RE_TYPED_NAME.match(item)
                if not m:
                    continue
                pname, ptyp = _strip_prov(m.group(1)), m.group(2)
                if pname == iv:
                    st.env[pname] = LoopVar(iv)
                elif ptyp.strip() == "Token":
                    st.env[pname] = _TOKEN
                else:
                    st.env[pname] = DataDep("loop-carried value")
            j += 1
    st.in_loop = True
    j = _walk(lines, j, body_indent, st)
    st.in_loop = False
    for rname, rtyp in results:
        st.env[rname] = _TOKEN if rtyp.strip() == "Token" else DataDep("loop result")
    return j


def _record_view_access(
    kind: str,
    pv: Any,
    index_vals: list[Any],
    line_no: int,
    st: _State,
) -> None:
    if not isinstance(pv, _PartView):
        raise UnsupportedTTIR(
            f"line {line_no}: {kind} view is not a partition view",
            kind="parse",
        )
    arr = pv.array
    rank = len(pv.tile_shape)
    if len(index_vals) != rank or len(arr.shape) != rank or len(arr.strides) != rank:
        raise UnsupportedTTIR(
            f"line {line_no}: {kind} rank mismatch (index {len(index_vals)}, "
            f"tile {rank}, array {len(arr.shape)})",
            kind="parse",
        )
    offset: Term | None = None
    mask: Term | None = None
    mask_dropped = False
    for ax in range(rank):
        idx = _as_term(index_vals[ax], f"{kind} index")
        whys = _datadep_whys(idx)
        if whys:
            raise UnsupportedTTIR(
                f"line {line_no}: {kind} tile index: data-dependent "
                f"({'; '.join(sorted(whys))})",
                kind="indirect-address",
            )
        ts = pv.tile_shape[ax]
        ar = _new_arange(st, ts, ax if rank > 1 else -1)
        off_ax = Bin("+", Bin("*", idx, Const(ts)), ar)
        shape_ax = arr.shape[ax]
        shape_t = Const(shape_ax) if isinstance(shape_ax, int) else shape_ax
        if _has_datadep(shape_t):
            mask_dropped = True
        else:
            clip = BoolBin(
                "and",
                Cmp("sge", off_ax, Const(0)),
                Cmp("slt", off_ax, shape_t),
            )
            mask = clip if mask is None else BoolBin("and", mask, clip)
        stride_ax = arr.strides[ax]
        stride_t = Const(stride_ax) if isinstance(stride_ax, int) else stride_ax
        if _has_datadep(stride_t):
            raise UnsupportedTTIR(
                f"line {line_no}: {kind} stride: data-dependent",
                kind="indirect-address",
            )
        contrib = Bin("*", off_ax, stride_t)
        offset = contrib if offset is None else Bin("+", offset, contrib)
    bits = _DTYPE_BITS.get(arr.dtype, 0)
    st.accesses.append(
        AccessEvent(
            kind=kind,
            base_param=arr.base,
            offset=offset if offset is not None else Const(0),
            mask=mask,
            elem_bits=bits,
            loc=None,
            line_no=line_no,
            in_loop=st.in_loop,
            mask_dropped=mask_dropped,
            elem_float=arr.dtype in _FLOAT_DTYPES,
        )
    )


def _handle_op(
    op: str,
    results: list[tuple[str, str]],
    kw: dict[str, str],
    line_no: int,
    st: _State,
) -> None:
    env = st.env

    def bind(value: Any) -> None:
        env[results[0][0]] = value

    if op in ("make_token", "join_tokens"):
        for rname, _ in results:
            env[rname] = _TOKEN
        return
    if op in ("assume_div_by", "assume_bounded"):
        bind(_val(st, kw["x"]))
        return
    if op == "typed_const":
        v = kw["value"]
        if v in ("True", "False"):
            bind(Const(1 if v == "True" else 0))
            return
        try:
            bind(Const(int(v)))
        except ValueError:
            bind(DataDep(f"non-integer constant {v}"))
        return
    if op == "tile_bid":
        axis = int(kw["axis"])
        st.pid_axes.add(axis)
        bind(Pid(axis))
        return
    if op == "tile_arange":
        shape = _tile_shape_of(results[0][1])
        if shape is None or len(shape) != 1:
            bind(DataDep("arange with unparsable shape"))
        else:
            bind(_new_arange(st, shape[0], -1))
        return
    if op in ("tile_reshape", "tile_broadcast", "tile_astype", "tile_expand_dims"):
        src = _val(st, kw["x"])
        if isinstance(src, PtrValue):
            bind(src)
            return
        tshape = _tile_shape_of(results[0][1])
        if (
            op in ("tile_reshape", "tile_expand_dims")
            and tshape is not None
            and len(tshape) > 1
            and not isinstance(src, (DataDep, _Token, _ArrayView, _PartView))
        ):
            sized = [ax for ax, s in enumerate(tshape) if s > 1]
            if len(sized) == 1:
                src = _set_arange_dim(src, sized[0])
        bind(src)
        return
    if op == "raw_binary_arith":
        fn = kw["fn"].strip('"')
        a, b = _val(st, kw["lhs"]), _val(st, kw["rhs"])
        if fn == "cdiv":
            at, bt = _as_term(a, "cdiv"), _as_term(b, "cdiv")
            bind(Bin("//", Bin("+", at, Bin("-", bt, Const(1))), bt))
        elif fn in _ARITH_FN:
            bind(Bin(_ARITH_FN[fn], _as_term(a, fn), _as_term(b, fn)))
        else:
            bind(DataDep(f"arith fn {fn}"))
        return
    if op == "raw_cmp":
        fn = kw["fn"].strip('"')
        if fn in _CMP_FN:
            bind(Cmp(_CMP_FN[fn], _as_term(_val(st, kw["lhs"]), "cmp"),
                     _as_term(_val(st, kw["rhs"]), "cmp")))  # fmt: skip
        else:
            bind(DataDep(f"cmp fn {fn}"))
        return
    if op == "raw_binary_bitwise":
        fn = kw["fn"].strip('"')
        a = _as_term(_val(st, kw["lhs"]), "boolbin")
        b = _as_term(_val(st, kw["rhs"]), "boolbin")
        if fn in ("and_", "or_"):
            bind(BoolBin("and" if fn == "and_" else "or", a, b))
        elif fn == "xor" and results[0][1].startswith("Tile[bool_"):
            # boolean xor — the sign-disagreement test of the python
            # floor-div/mod lowering (c_mod + fix). (a ∧ ¬b) ∨ (¬a ∧ b)
            # keeps it fully modeled; INTEGER xor (bitonic partner
            # indexing) stays DataDep below.
            bind(
                BoolBin(
                    "or",
                    BoolBin("and", a, Not(b)),
                    BoolBin("and", Not(a), b),
                )
            )
        else:
            bind(DataDep(f"bitwise fn {fn}"))
        return
    if op == "fma":
        # lhs*rhs + acc — an arithmetic identity, modelable for any dtype
        bind(
            Bin(
                "+",
                Bin(
                    "*",
                    _as_term(_val(st, kw["lhs"]), "fma"),
                    _as_term(_val(st, kw["rhs"]), "fma"),
                ),
                _as_term(_val(st, kw["acc"]), "fma"),
            )
        )
        return
    if op == "unaryop":
        fn = kw.get("fn", "").strip('"')
        if fn == "neg":
            bind(Bin("-", Const(0), _as_term(_val(st, kw["operand"]), "neg")))
        else:
            bind(DataDep(f"unary fn {fn}"))
        return
    if op == "raw_where":
        c = _as_term(_val(st, kw["cond"]), "where")
        x = _as_term(_val(st, kw["x"]), "where")
        y = _as_term(_val(st, kw["y"]), "where")
        bind(Select(c, x, y))
        return
    if op == "make_tensor_view":
        base = _val(st, kw["base_ptr"])
        if not isinstance(base, PtrValue):
            raise UnsupportedTTIR(
                f"line {line_no}: tensor view base is not a pointer",
                kind="parse",
            )
        am = _RE_ARRAY_TYPE.match(results[0][1])
        if not am:
            raise UnsupportedTTIR(
                f"line {line_no}: unparsable Array type {results[0][1][:60]!r}",
                kind="parse",
            )
        dtype = am.group(1)
        shape_spec = _split_top(am.group(2))
        stride_spec = _split_top(am.group(3))
        dyn_shapes = _tuple_vals(st, kw.get("shape", "()"))
        dyn_strides = _tuple_vals(st, kw.get("dynamic_strides", "()"))
        vshape: list[Any] = []
        di = 0
        for spec_s in shape_spec:
            if spec_s == "?":
                vshape.append(_as_term(dyn_shapes[di], "view shape"))
                di += 1
            else:
                vshape.append(int(spec_s))
        vstrides: list[Any] = []
        si = 0
        for spec_s in stride_spec:
            if spec_s == "?":
                vstrides.append(_as_term(dyn_strides[si], "view stride"))
                si += 1
            else:
                vstrides.append(int(spec_s))
        env[results[0][0]] = _ArrayView(base.base_param, vshape, vstrides, dtype)
        return
    if op == "make_partition_view":
        arr = _val(st, kw["array"])
        if not isinstance(arr, _ArrayView):
            raise UnsupportedTTIR(
                f"line {line_no}: partition view over non-array", kind="parse"
            )
        pm = _RE_PARTVIEW_TYPE.match(results[0][1])
        if not pm:
            raise UnsupportedTTIR(
                f"line {line_no}: unparsable PartitionView type "
                f"{results[0][1][:80]!r}",
                kind="parse",
            )
        env[results[0][0]] = _PartView(arr, _ints(pm.group(1)), pm.group(3))
        return
    if op == "tile_load":
        pv = _val(st, kw["view"])
        _record_view_access("load", pv, _tuple_vals(st, kw["index"]), line_no, st)
        for rname, rtyp in results:
            env[rname] = _TOKEN if rtyp.strip() == "Token" else DataDep("loaded value")
        return
    if op == "tile_store":
        pv = _val(st, kw["view"])
        _record_view_access("store", pv, _tuple_vals(st, kw["index"]), line_no, st)
        for rname, _ in results:
            env[rname] = _TOKEN
        return
    if op in ("load_pointer", "store_pointer"):
        # the raw-pointer gather/scatter path — semantically TTIR's
        # tt.load/tt.store over addptr chains: per-element offsets, an
        # explicit mask (compiler-emitted bounds + user routing)
        ptr = _val(st, kw["pointer"])
        if not isinstance(ptr, PtrValue):
            raise UnsupportedTTIR(
                f"line {line_no}: {op} base is not a pointer",
                kind="indirect-address",
            )
        whys = _datadep_whys(ptr.offset)
        if whys:
            raise UnsupportedTTIR(
                f"line {line_no}: pointer offset: data-dependent "
                f"({'; '.join(sorted(whys))})",
                kind="indirect-address",
            )
        mask_raw = kw.get("mask", "None")
        if mask_raw == "None":
            mask_v: Term | None = None
            mask_dropped = False
        else:
            mv = _as_term(_val(st, mask_raw), f"{op} mask")
            mask_dropped = _has_datadep(mv)
            mask_v = None if mask_dropped else mv
        bits, is_f = st.ptr_meta.get(ptr.base_param, (0, False))
        st.accesses.append(
            AccessEvent(
                kind="load" if op == "load_pointer" else "store",
                base_param=ptr.base_param,
                offset=ptr.offset,
                mask=mask_v,
                elem_bits=bits,
                loc=None,
                line_no=line_no,
                in_loop=st.in_loop,
                mask_dropped=mask_dropped,
                elem_float=is_f,
            )
        )
        for rname, rtyp in results:
            env[rname] = _TOKEN if rtyp.strip() == "Token" else DataDep("loaded value")
        return
    if op == "pointer_offset":
        ptr = _val(st, kw["pointer"])
        if not isinstance(ptr, PtrValue):
            raise UnsupportedTTIR(
                f"line {line_no}: pointer_offset base is not a pointer",
                kind="indirect-address",
            )
        off = _as_term(_val(st, kw["offset"]), "pointer offset")
        whys = _datadep_whys(off)
        if whys:
            raise UnsupportedTTIR(
                f"line {line_no}: pointer offset: data-dependent "
                f"({'; '.join(sorted(whys))})",
                kind="indirect-address",
            )
        bind(PtrValue(ptr.base_param, Bin("+", ptr.offset, off)))
        return
    if op == "tile_atomic_rmw":
        ptr = _val(st, kw["pointer"])
        if not isinstance(ptr, PtrValue):
            raise UnsupportedTTIR(
                f"line {line_no}: atomic_rmw of a non-pointer value",
                kind="indirect-address",
            )
        mode = kw.get("mode", "").split(".")[-1]
        rmw = _RMW_MODE.get(mode)
        if rmw is None:
            raise UnsupportedTTIR(
                f"line {line_no}: unknown atomic mode {mode}", kind="parse"
            )
        mask_v = _as_term(_val(st, kw["mask"]), "atomic mask")
        mask_dropped = _has_datadep(mask_v)
        upd = _as_term(_val(st, kw["update"]), "atomic update")
        bits, is_f = st.ptr_meta.get(ptr.base_param, (0, False))
        sem = kw.get("memory_order", "").split(".")[-1].lower() or "acq_rel"
        scope = _SCOPE.get(kw.get("memory_scope", "").split(".")[-1], "gpu")
        st.accesses.append(
            AccessEvent(
                kind="atomic_rmw",
                base_param=ptr.base_param,
                offset=ptr.offset,
                mask=None if mask_dropped else mask_v,
                elem_bits=bits,
                loc=None,
                line_no=line_no,
                in_loop=st.in_loop,
                atomic=AtomicInfo(rmw_op=rmw, sem=sem, scope=scope),
                mask_dropped=mask_dropped,
                atomic_val=None if (_has_datadep(upd) or is_f) else upd,
                elem_float=is_f,
            )
        )
        for rname, rtyp in results:
            env[rname] = _TOKEN if rtyp.strip() == "Token" else DataDep("atomic result")
        return

    # every other op: value-level over-approximation, never an exception
    st.unknown_ops[op] = st.unknown_ops.get(op, 0) + 1
    for rname, rtyp in results:
        env[rname] = _TOKEN if rtyp.strip() == "Token" else DataDep(f"cutile op {op}")
