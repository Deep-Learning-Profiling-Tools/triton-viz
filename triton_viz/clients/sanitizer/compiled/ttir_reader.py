"""Textual TTIR reader for the compiled-mode sanitizer.

Parses the pre-optimization Triton IR (TTIR) of one kernel specialization
into an ``AccessGraph``: the kernel's function arguments, every global
memory access (``tt.load`` / ``tt.store``) as an *element offset* expression
relative to a base pointer argument, the mask guarding it, and the loop
structure. Scalar arguments (``n_elements``, ``M``, strides, ...) stay
symbolic (``Param`` nodes) and are substituted with concrete launch values
later; ``tl.constexpr`` values are already folded into TTIR constants.

Why TTIR (not TTGIR): out-of-bounds is cleanest in the element address
space, before layouts/pipelining add noise, and TTIR has no indirect loads
unless the kernel itself gathers — which is exactly the data-dependent case
we report as ``unsupported``. v1 does NOT fall back to interpretation
automatically; to check an unsupported kernel, run the eager
``Sanitizer()`` on it.

Address model: ``tt.addptr(base, off)`` accumulates an ELEMENT offset; the
byte address is ``base.data_ptr() + offset * elem_size``. An access is OOB
iff, for some program id / arange lane / loop iteration with its mask true,
the element offset escapes ``[0, numel)`` of its base tensor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class UnsupportedTTIR(Exception):
    """Raised for constructs outside the compiled sanitizer's v1 model
    (indirect/data-dependent addressing, block pointers, nested loops, ...).
    The client converts this into a dynamic-mode fallback or unsupported
    status — never a silent wrong verdict."""


# ─────────────────────────── address-expression terms ───────────────────────────
# A small lazily-evaluated tree. Leaves that are only known at launch time
# (scalar kernel args) are Param nodes; pid / arange / loop variables become
# free Z3 variables with range constraints in the OOB query.


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Pid:
    axis: int  # 0=x, 1=y, 2=z


@dataclass(frozen=True)
class Arange:
    ssa: str  # unique per make_range site
    start: int
    end: int
    # Which tensor dimension this lane index varies along. -1 = 1D / not yet
    # placed; 0/1 set by expand_dims. A single make_range reused for both the
    # row and column of a 2D tile (triton does this) must become TWO
    # independent variables — keyed by (ssa, dim) — or the modeled footprint
    # collapses to the diagonal (the same collapse bug fixed in dynamic mode).
    dim: int = -1


@dataclass(frozen=True)
class Param:
    name: str  # scalar kernel argument, substituted per launch


@dataclass(frozen=True)
class IterArgOffset:
    """The element-offset contribution of a loop-carried pointer at the
    current iteration: ``offset0 + k * delta`` (resolved from the graph's
    loop info at eval time)."""

    arg_id: int


@dataclass(frozen=True)
class LoopVar:
    """The scf.for induction variable; a free variable in [lower, upper)
    in the OOB query (e.g. it appears in masks like ``K - k*BLOCK_K``)."""

    loop_ssa: str


@dataclass(frozen=True)
class Bin:
    op: str  # + - * // (// = signed divide, matching arith.divsi)
    a: "Term"
    b: "Term"


@dataclass(frozen=True)
class Cmp:
    pred: str  # slt/sle/sgt/sge/eq/ne
    a: "Term"
    b: "Term"


@dataclass(frozen=True)
class BoolBin:
    op: str  # and / or
    a: "Term"
    b: "Term"


@dataclass(frozen=True)
class Select:
    cond: "Term"
    t: "Term"
    f: "Term"


# Sentinel for a value loaded from memory (tt.load result) or computed from
# loaded data (arith.*f, tt.dot, ...). If one ever reaches an address or mask
# it means data-dependent addressing → unsupported.
@dataclass(frozen=True)
class DataDep:
    why: str = "value derived from loaded data"


Term = (
    Const
    | Pid
    | Arange
    | Param
    | IterArgOffset
    | LoopVar
    | Bin
    | Cmp
    | BoolBin
    | Select
    | DataDep
)


@dataclass(frozen=True)
class PtrValue:
    """A pointer-typed SSA value: base argument + accumulated element
    offset (a single lane's offset; arange/loop free vars cover all lanes
    and iterations in the query)."""

    base_param: str
    offset: Term


# ─────────────────────────── graph structures ───────────────────────────


@dataclass(frozen=True)
class FuncArg:
    name: str
    is_ptr: bool
    elem_bits: int  # for ptr args: pointee width; 0 for scalars


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int


@dataclass(frozen=True)
class AccessEvent:
    kind: str  # "load" | "store"
    base_param: str
    offset: Term
    mask: Term | None  # None = unconditional access
    elem_bits: int
    loc: SourceLoc | None
    line_no: int


@dataclass(frozen=True)
class IterArgInfo:
    arg_id: int
    base_param: str
    offset0: Term
    delta: Term  # per-iteration element advance


@dataclass(frozen=True)
class LoopInfo:
    loop_ssa: str
    induction_var: str
    lower: Term
    upper: Term
    step: Term


@dataclass
class AccessGraph:
    kernel_name: str
    func_args: list[FuncArg]
    accesses: list[AccessEvent]
    loop: LoopInfo | None
    iter_args: dict[int, IterArgInfo] = field(default_factory=dict)

    def arg(self, name: str) -> FuncArg | None:
        for a in self.func_args:
            if a.name == name:
                return a
        return None


# ─────────────────────────── regexes ───────────────────────────

_SSA = r"%[\w.]+"
_DTYPE_BITS = {
    "f64": 64, "f32": 32, "f16": 16, "bf16": 16, "f8": 8,
    "i64": 64, "i32": 32, "i16": 16, "i8": 8, "i1": 1,
    "u64": 64, "u32": 32,
}  # fmt: skip

_RE_LOC_FILE = re.compile(r'^(#loc\d*) = loc\("([^"]+)":(\d+):(\d+)\)')
_RE_LOC_NAME = re.compile(r'^(#loc\d*) = loc\("[^"]+"\((#loc\d*)\)\)')
_RE_LOC_TRAILER = re.compile(r"loc\((#loc\d*|#loc)\)\s*$")
_RE_FUNC = re.compile(r"tt\.func\s+\w+\s+@(\w+)\((.*)\)\s*attributes")
_RE_RESULT = re.compile(rf"^({_SSA})(?::\d+)?\s*=\s*(.*)$")
_RE_GET_PID = re.compile(r"^tt\.get_program_id (\w+)")
_RE_MAKE_RANGE = re.compile(
    r"^tt\.make_range \{end = (-?\d+) : i32, start = (-?\d+) : i32\}"
)
_RE_CONST_INT = re.compile(r"^arith\.constant (-?\d+) : i\d+")
_RE_CONST_DENSE = re.compile(r"^arith\.constant dense<(-?\d+)> : tensor")
_RE_SPLAT = re.compile(rf"^tt\.splat ({_SSA}) : ([^-]+)->")
_RE_EXPAND = re.compile(rf"^tt\.expand_dims ({_SSA}) \{{axis = (\d+)")
_RE_BROADCAST = re.compile(rf"^tt\.broadcast ({_SSA})")
_RE_ADDPTR = re.compile(rf"^tt\.addptr ({_SSA}), ({_SSA})")
_RE_BIN = re.compile(rf"^arith\.(muli|addi|subi|divsi) ({_SSA}), ({_SSA})")
_RE_CMPI = re.compile(rf"^arith\.cmpi (\w+), ({_SSA}), ({_SSA})")
_RE_BOOLBIN = re.compile(rf"^arith\.(andi|ori) ({_SSA}), ({_SSA})")
_RE_SELECT = re.compile(rf"^arith\.select ({_SSA}), ({_SSA}), ({_SSA})")
_RE_EXT = re.compile(rf"^arith\.(extsi|trunci|extui) ({_SSA})")
_RE_LOAD = re.compile(rf"^tt\.load ({_SSA})((?:, {_SSA})*)\s*(?::|loc|$)")
_RE_STORE = re.compile(rf"^tt\.store ({_SSA}), ({_SSA})((?:, {_SSA})*)\s*(?::|loc|$)")
_RE_PTR_ELEM = re.compile(r"!tt\.ptr<(\w+)>")
_RE_SCF_FOR = re.compile(
    rf"^scf\.for ({_SSA}) = ({_SSA}) to ({_SSA}) step ({_SSA})"
    # iter_args + "-> (types)" appear only when the loop yields values; a
    # pure-side-effect loop (e.g. a store loop, no accumulator) ends at the
    # ": i32 {" type annotation with no arrow. Match both, or the loop is
    # missed and its induction var leaks as an unbound (data-dependent) SSA.
    rf"(?: iter_args\((.*?)\))?\s*(?:->|:)"
)
_RE_SCF_YIELD = re.compile(r"^scf\.yield (.*?)\s*:")


def _elem_bits(type_str: str) -> int:
    m = _RE_PTR_ELEM.search(type_str)
    if m:
        return _DTYPE_BITS.get(m.group(1), 0)
    return 0


def _split_ssa(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip().startswith("%")]


class _LocTable:
    def __init__(self) -> None:
        self._file: dict[str, tuple[str, int, int]] = {}
        self._alias: dict[str, str] = {}

    def add(self, line: str) -> bool:
        m = _RE_LOC_FILE.match(line)
        if m:
            self._file[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)))
            return True
        m = _RE_LOC_NAME.match(line)
        if m:
            self._alias[m.group(1)] = m.group(2)
            return True
        if line.startswith("#loc") and "= loc(" in line:
            return True
        return False

    def resolve(self, loc_id: str | None, _d: int = 0) -> SourceLoc | None:
        if loc_id is None or _d > 8:
            return None
        if loc_id in self._file:
            f, ln, col = self._file[loc_id]
            return SourceLoc(f, ln, col)
        if loc_id in self._alias:
            return self.resolve(self._alias[loc_id], _d + 1)
        return None


def parse_ttir(text: str) -> AccessGraph:
    """Parse one TTIR module into an AccessGraph.

    Raises :class:`UnsupportedTTIR` for indirect addressing, block pointers,
    nested/while loops, or any op outside the v1 address vocabulary that
    feeds a pointer.
    """
    locs = _LocTable()
    kernel_name = ""
    func_args: list[FuncArg] = []
    # SSA name -> value: Term (int/bool), PtrValue, or DataDep
    env: dict[str, object] = {}
    accesses: list[AccessEvent] = []
    loop: LoopInfo | None = None
    iter_args: dict[int, IterArgInfo] = {}

    lines = text.splitlines()
    # Pre-scan loc table (aliases live at the bottom).
    for line in lines:
        locs.add(line.strip())

    def val(name: str) -> object:
        v = env.get(name)
        if v is None:
            # Unknown SSA reaching an address/mask: be conservative.
            return DataDep(f"unresolved SSA {name}")
        return v

    def as_term(v: object, ctx: str) -> Term:
        if isinstance(v, DataDep):
            raise UnsupportedTTIR(f"{ctx}: data-dependent ({v.why})")
        if isinstance(v, PtrValue):
            raise UnsupportedTTIR(f"{ctx}: pointer used as integer")
        return v  # type: ignore[return-value]

    def parse_func_args(arg_text: str) -> None:
        for m in re.finditer(r"(%[\w.]+): (!tt\.ptr<\w+>|i\d+|f\d+)", arg_text):
            name, ty = m.group(1)[1:], m.group(2)
            is_ptr = ty.startswith("!tt.ptr")
            bits = _elem_bits(ty) if is_ptr else 0
            fa = FuncArg(name=name, is_ptr=is_ptr, elem_bits=bits)
            func_args.append(fa)
            # Pointer args seed addptr chains; scalar args are Param leaves.
            env[f"%{name}"] = PtrValue(name, Const(0)) if is_ptr else Param(name)

    def base_elem_bits(param: str) -> int:
        fa = next((a for a in func_args if a.name == param), None)
        return fa.elem_bits if fa else 0

    # ── body parse (single function; loop handled inline) ──
    in_loop = False
    loop_body_yields: list[str] = []
    loop_iter_arg_ssa: list[tuple[str, str]] = []  # (arg_ssa, init_ssa)
    loop_meta: dict[str, object] = {}

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _RE_FUNC.search(line)
        if m and not kernel_name:
            kernel_name = m.group(1)
            parse_func_args(m.group(2))
            continue
        if not kernel_name:
            continue

        loc_m = _RE_LOC_TRAILER.search(line)
        loc = locs.resolve(loc_m.group(1)) if loc_m else None

        rm = _RE_RESULT.match(line)
        res = rm.group(1) if rm else None
        body = rm.group(2) if rm else line

        # ---- scf.for ----
        fm = _RE_SCF_FOR.match(body)
        if fm:
            # ``loop`` is only set at the closing brace, so a second
            # SEQUENTIAL loop is caught by it — but a NESTED loop opens while
            # the outer one is still in flight (loop is still None), so guard
            # on in_loop too. Nested loops carry independent induction
            # variables the single-loop model cannot represent; reject rather
            # than silently mis-bound the outer var to the inner's range.
            if loop is not None or in_loop:
                raise UnsupportedTTIR(f"line {line_no}: multiple/nested loops")
            ind, lo, up, st, iters = fm.groups()
            pairs: list[tuple[str, str]] = []
            if iters:
                pairs = list(re.findall(rf"({_SSA}) = ({_SSA})", iters))
            loop_meta = {
                "ssa": res or "%loop",
                "ind": ind,
                "lower": as_term(val(lo), "loop lower"),
                "upper": as_term(val(up), "loop upper"),
                "step": as_term(val(st), "loop step"),
            }
            # Bind induction var as a loop free variable.
            env[ind] = LoopVar(res or "%loop")
            # Bind ptr iter_args to IterArgOffset; ignore non-ptr (accumulators).
            arg_id = 0
            for arg_ssa, init_ssa in pairs:
                iv = val(init_ssa)
                if isinstance(iv, PtrValue):
                    iter_args[arg_id] = IterArgInfo(
                        arg_id=arg_id,
                        base_param=iv.base_param,
                        offset0=iv.offset,
                        delta=Const(0),  # filled at yield
                    )
                    env[arg_ssa] = PtrValue(iv.base_param, IterArgOffset(arg_id))
                    loop_iter_arg_ssa.append((arg_ssa, init_ssa))
                    arg_id += 1
                else:
                    env[arg_ssa] = DataDep("loop accumulator")
                    loop_iter_arg_ssa.append((arg_ssa, init_ssa))
            in_loop = True
            continue

        if in_loop and (line == "}" or line.startswith("} loc")):
            in_loop = False
            # Resolve deltas from the yields, positionally.
            ptr_idx = 0
            for pos, (arg_ssa, _init) in enumerate(loop_iter_arg_ssa):
                if not isinstance(env.get(arg_ssa), PtrValue):
                    continue
                if pos >= len(loop_body_yields):
                    raise UnsupportedTTIR("loop yield/iter_arg count mismatch")
                yssa = loop_body_yields[pos]
                yv = env.get(yssa)
                if not isinstance(yv, PtrValue):
                    raise UnsupportedTTIR("loop yields a non-pointer for a ptr arg")
                delta = _extract_loop_delta(yv.offset, ptr_idx)
                if delta is None:
                    raise UnsupportedTTIR(
                        f"loop pointer advance for arg {ptr_idx} is not a "
                        "simple monotonic addptr"
                    )
                info = iter_args[ptr_idx]
                iter_args[ptr_idx] = IterArgInfo(
                    info.arg_id, info.base_param, info.offset0, delta
                )
                ptr_idx += 1
            loop = LoopInfo(
                loop_ssa=str(loop_meta["ssa"]),
                induction_var=str(loop_meta["ind"]),
                lower=loop_meta["lower"],  # type: ignore[arg-type]
                upper=loop_meta["upper"],  # type: ignore[arg-type]
                step=loop_meta["step"],  # type: ignore[arg-type]
            )
            continue

        ym = _RE_SCF_YIELD.match(body)
        if ym and in_loop:
            loop_body_yields = _split_ssa(ym.group(1))
            continue

        # ---- value-producing ops ----
        handled = _parse_value_op(body, res, env, val, as_term, base_elem_bits)
        if handled:
            continue

        # ---- accesses ----
        lm = _RE_LOAD.match(body)
        if lm:
            _record_access(
                "load",
                lm.group(1),
                lm.group(2),
                None,
                env,
                val,
                accesses,
                base_elem_bits,
                loc,
                line_no,
            )
            if res is not None:
                env[res] = DataDep("loaded value")
            continue
        sm = _RE_STORE.match(body)
        if sm:
            _record_access(
                "store",
                sm.group(1),
                sm.group(3),
                None,
                env,
                val,
                accesses,
                base_elem_bits,
                loc,
                line_no,
            )
            continue

        # ---- ops whose result is just data (ignored) ----
        if res is not None and (
            body.startswith(
                (
                    "arith.addf",
                    "arith.mulf",
                    "arith.subf",
                    "arith.divf",
                    "arith.cmpf",
                    "tt.dot",
                    "arith.truncf",
                    "arith.extf",
                    "arith.sitofp",
                    "tt.reduce",
                    "math.",
                )
            )
        ):
            env[res] = DataDep("float/reduction value")
            continue
        if body.startswith(("tt.return", "tt.reduce.return")):
            continue
        if body.startswith("tt.make_block_ptr") or body.startswith("tt.advance"):
            raise UnsupportedTTIR(f"line {line_no}: block pointers are unsupported")
        # Unknown op producing a value used downstream → conservative DataDep.
        if res is not None:
            env[res] = DataDep(f"unmodeled op at line {line_no}")

    if not kernel_name:
        raise UnsupportedTTIR("no tt.func found (not TTIR?)")

    return AccessGraph(
        kernel_name=kernel_name,
        func_args=func_args,
        accesses=accesses,
        loop=loop,
        iter_args=iter_args,
    )


def _set_arange_dim(v: object, dim: int) -> object:
    """Tag every Arange in an integer expression with the tensor dimension
    it varies along (set by expand_dims). Non-Arange leaves pass through."""
    if isinstance(v, Arange):
        return Arange(v.ssa, v.start, v.end, dim if v.dim < 0 else v.dim)
    if isinstance(v, Bin):
        return Bin(v.op, _set_arange_dim(v.a, dim), _set_arange_dim(v.b, dim))  # type: ignore[arg-type]
    if isinstance(v, Cmp):
        return Cmp(v.pred, _set_arange_dim(v.a, dim), _set_arange_dim(v.b, dim))  # type: ignore[arg-type]
    if isinstance(v, BoolBin):
        return BoolBin(v.op, _set_arange_dim(v.a, dim), _set_arange_dim(v.b, dim))  # type: ignore[arg-type]
    if isinstance(v, Select):
        return Select(
            _set_arange_dim(v.cond, dim),  # type: ignore[arg-type]
            _set_arange_dim(v.t, dim),  # type: ignore[arg-type]
            _set_arange_dim(v.f, dim),  # type: ignore[arg-type]
        )
    return v


def _extract_loop_delta(offset: Term, arg_id: int) -> Term | None:
    """From a yielded pointer offset of the shape
    ``IterArgOffset(arg_id) + delta`` (any association), pull out ``delta``."""
    if isinstance(offset, IterArgOffset):
        return Const(0)
    if isinstance(offset, Bin) and offset.op == "+":
        if isinstance(offset.a, IterArgOffset) and offset.a.arg_id == arg_id:
            return offset.b
        if isinstance(offset.b, IterArgOffset) and offset.b.arg_id == arg_id:
            return offset.a
    return None


def _parse_value_op(body, res, env, val, as_term, base_elem_bits) -> bool:
    """Parse one address-structure value op into env. Returns True if handled."""
    if res is None:
        return False

    m = _RE_GET_PID.match(body)
    if m:
        env[res] = Pid({"x": 0, "y": 1, "z": 2}[m.group(1)])
        return True
    m = _RE_MAKE_RANGE.match(body)
    if m:
        env[res] = Arange(res, int(m.group(2)), int(m.group(1)))
        return True
    m = _RE_CONST_INT.match(body)
    if m:
        env[res] = Const(int(m.group(1)))
        return True
    m = _RE_CONST_DENSE.match(body)
    if m:
        env[res] = Const(int(m.group(1)))
        return True
    if body.startswith("arith.constant"):
        env[res] = DataDep("float/array constant")
        return True
    m = _RE_SPLAT.match(body)
    if m:
        env[res] = val(m.group(1))  # replicate scalar / seed ptr
        return True
    m = _RE_EXPAND.match(body)
    if m and body.startswith("tt.expand_dims"):
        # axis is the inserted size-1 dim; the lane index varies along the
        # OTHER dim (1 - axis for a 1D->2D expand). Tag every Arange inside.
        axis = int(m.group(2))
        env[res] = _set_arange_dim(val(m.group(1)), 1 - axis)
        return True
    m = _RE_BROADCAST.match(body)
    if m and body.startswith("tt.broadcast"):
        env[res] = val(m.group(1))  # shape change, value passthrough
        return True
    m = _RE_EXT.match(body)
    if m:
        env[res] = val(m.group(2))  # width change, value passthrough
        return True
    m = _RE_ADDPTR.match(body)
    if m:
        base, off = val(m.group(1)), val(m.group(2))
        if not isinstance(base, PtrValue):
            raise UnsupportedTTIR("addptr base is not a pointer")
        off_t = as_term(off, "addptr offset")  # DataDep here → indirect → unsupported
        env[res] = PtrValue(base.base_param, Bin("+", base.offset, off_t))
        return True
    m = _RE_BIN.match(body)
    if m:
        op = {"muli": "*", "addi": "+", "subi": "-", "divsi": "//"}[m.group(1)]
        a, b = val(m.group(2)), val(m.group(3))
        if isinstance(a, DataDep) or isinstance(b, DataDep):
            env[res] = DataDep("arith over loaded data")
        else:
            env[res] = Bin(op, as_term(a, "arith"), as_term(b, "arith"))
        return True
    m = _RE_CMPI.match(body)
    if m:
        a, b = val(m.group(2)), val(m.group(3))
        if isinstance(a, DataDep) or isinstance(b, DataDep):
            env[res] = DataDep("cmpi over loaded data")
        else:
            env[res] = Cmp(m.group(1), as_term(a, "cmpi"), as_term(b, "cmpi"))
        return True
    m = _RE_BOOLBIN.match(body)
    if m:
        a, b = val(m.group(2)), val(m.group(3))
        if isinstance(a, DataDep) or isinstance(b, DataDep):
            env[res] = DataDep("bool op over loaded data")
        else:
            env[res] = BoolBin(
                "and" if m.group(1) == "andi" else "or",
                as_term(a, "bool"),
                as_term(b, "bool"),
            )
        return True
    m = _RE_SELECT.match(body)
    if m:
        c, t, f = val(m.group(1)), val(m.group(2)), val(m.group(3))
        if any(isinstance(x, DataDep) for x in (c, t, f)):
            env[res] = DataDep("select over loaded data")
        else:
            env[res] = Select(
                as_term(c, "select"), as_term(t, "select"), as_term(f, "select")
            )
        return True
    return False


def _record_access(
    kind,
    ptr_ssa,
    extra_ops,
    _unused,
    env,
    val,
    accesses,
    base_elem_bits,
    loc,
    line_no,
) -> None:
    ptr = val(ptr_ssa)
    if not isinstance(ptr, PtrValue):
        raise UnsupportedTTIR(f"line {line_no}: {kind} of a non-pointer value")
    # Mask: for load it's the first trailing operand; for store the operand
    # after value. _RE_LOAD captures trailing ", %x" groups; for store the
    # caller passed the post-value trailing operands.
    mask: Term | None = None
    trailing = _split_ssa(extra_ops) if extra_ops else []
    if trailing:
        mv = val(trailing[0])
        if isinstance(mv, DataDep):
            # Mask derived from loaded data — can't reason statically.
            raise UnsupportedTTIR(f"line {line_no}: data-dependent mask")
        if isinstance(mv, PtrValue):
            raise UnsupportedTTIR(f"line {line_no}: pointer as mask")
        mask = mv  # type: ignore[assignment]
    accesses.append(
        AccessEvent(
            kind=kind,
            base_param=ptr.base_param,
            offset=ptr.offset,
            mask=mask,
            elem_bits=base_elem_bits(ptr.base_param),
            loc=loc,
            line_no=line_no,
        )
    )
