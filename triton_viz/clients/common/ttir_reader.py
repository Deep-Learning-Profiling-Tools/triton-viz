"""Textual TTIR reader shared by the compiled-mode clients.

Parses the pre-optimization Triton IR (TTIR) of one kernel specialization
into an ``AccessGraph``: the kernel's function arguments, every global
memory access (``tt.load`` / ``tt.store`` / ``tt.atomic_rmw`` /
``tt.atomic_cas``) as an *element offset* expression
relative to a base pointer argument, the mask guarding it, and the loop
structure. Scalar arguments (``n_elements``, ``M``, strides, ...) stay
symbolic (``Param`` nodes) and are substituted with concrete launch values
later; ``tl.constexpr`` values are already folded into TTIR constants.

Why TTIR (not TTGIR): element addressing is cleanest here, before
layouts/pipelining add noise, and TTIR has no indirect loads unless the
kernel itself gathers — the data-dependent case, marked with ``DataDep``.

This module is mechanism-only: it parses and flags (``DataDep`` markers,
``guarded`` accesses, ``UnsupportedTTIR``); what to do about a flagged or
unsupported kernel — report it, fall back to the interpreter, ... — is the
policy of each client that consumes the graph (sanitizer OOB checking,
race-detector global-memory front-end).

Address model: ``tt.addptr(base, off)`` accumulates an ELEMENT offset; the
byte address is ``base.data_ptr() + offset * elem_size``. An access is OOB
iff, for some program id / arange lane / loop iteration with its mask true,
the element offset escapes ``[0, numel)`` of its base tensor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace


class UnsupportedTTIR(Exception):
    """Raised for constructs outside the compiled-mode v1 model
    (indirect/data-dependent addressing, block pointers, nested loops, ...).
    The client converts this into an ``unsupported`` status (empty records) —
    never a silent wrong verdict. v1 does not auto-fall back to interpreted
    checking; run the eager ``Sanitizer()`` to check an unsupported kernel.

    ``kind`` is a stable, machine-readable class of the limitation — the
    hybrid tier selector routes on it (an "indirect-address" kernel goes to
    the interpreter front-end) and the evaluation reports its distribution:
    "indirect-address" | "data-dependent-bound" | "nested-loop" |
    "out-of-vocabulary" | "control-flow" | "block-pointer" |
    "unmodelable-condition" | "data-dependent-mask" |
    "cas-synchronization" | "spin-shape" | "other".

    "spin-shape" (spec C1.1): an ``scf.while`` that is not the recognized
    await form — the reason string names exactly which clause broke
    (carried values, extra memory ops, non-comparison condition, ...).
    """

    def __init__(self, msg: str, kind: str = "other") -> None:
        super().__init__(msg)
        self.kind = kind


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
class NumPrograms:
    """``tt.get_num_programs axis`` — the launch grid size along ``axis``.
    Uniform across program instances, but it PARAMETERIZES the kernel's
    behavior by the grid (last-block gates compare an atomic observation
    against it), so parsing one records the axis in ``pid_axes``: the
    verdict must stay symbolic along that dim. The race encoder lowers it
    to the SAME ``grid_<axis>`` variable the solver's symbolic grid uses."""

    axis: int


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
    op: str  # + - * // % min max (// and % truncate toward zero: divsi/remsi)
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


@dataclass(frozen=True)
class Not:
    """Boolean negation — the path condition of an scf.if else-region."""

    a: "Term"


# Sentinel for a value loaded from memory (tt.load result) or computed from
# loaded data (arith.*f, tt.dot, ...). If one ever reaches an address or mask
# it means data-dependent addressing → unsupported.
@dataclass(frozen=True)
class DataDep:
    why: str = "value derived from loaded data"


@dataclass(frozen=True)
class Observed:
    """The OLD value observed by the atomic at ``graph.accesses[access_index]``
    (spec part B): a fresh per-program-instance symbol, NOT a function of
    other leaves. The reader binds an INTEGER-typed ``tt.atomic_rmw`` /
    ``tt.atomic_cas`` result to this instead of ``DataDep`` so downstream
    masks and branch conditions stay modelable; float-typed atomic results
    keep the DataDep fallback (the value model is Int-sort only).

    Consumer policy (mechanism lives here, policy with each client):
      * race-detector global encoder: interns one Z3 var per index, ties it
        to the record's ``old_value`` (rf-justified) when the observation is
        modelable, and fails closed on address uses of unmodeled ones;
      * sanitizer OOB: a free variable — sound widening for proofs, with
        the mask_dropped-style witness abstention;
      * differential (C3): no concrete value exists — the access is
        excluded SYMMETRICALLY from both sides of the diff.
    """

    access_index: int


def mentions_observed(term: object) -> bool:
    """True when ``term`` contains an :class:`Observed` leaf."""
    if isinstance(term, Observed):
        return True
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None and mentions_observed(sub):
            return True
    return False


def observed_indices(term: object) -> set[int]:
    """Access indices of every :class:`Observed` leaf in ``term``."""
    out: set[int] = set()
    if isinstance(term, Observed):
        out.add(term.access_index)
        return out
    for attr in ("a", "b", "cond", "t", "f"):
        sub = getattr(term, attr, None)
        if sub is not None:
            out |= observed_indices(sub)
    return out


# DataDep is also the generic unknown-value top (unresolved SSA, loop
# accumulators, unmodeled ops, ...). Only these ``why`` prefixes mean the
# value truly derives from MEMORY CONTENTS — the per-term policy classifies
# just those as indirection (the interpreter-front-end route); the rest are
# modeling gaps and keep the default kind.
_MEMORY_WHYS = (
    "loaded value",
    "atomic result",
    "arith over loaded data",
    "cmpi over loaded data",
    "select over loaded data",
    "bool op over loaded data",
    "float/reduction value",
)


def _from_memory(v: object) -> bool:
    return isinstance(v, DataDep) and v.why.startswith(_MEMORY_WHYS)


Term = (
    Const
    | Pid
    | NumPrograms
    | Arange
    | Param
    | IterArgOffset
    | LoopVar
    | Bin
    | Cmp
    | BoolBin
    | Select
    | Not
    | DataDep
    | Observed
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
    # Float-typed pointee (f*/bf*): atomic results on it stay DataDep — the
    # Int-sort observation model must not carry float values (spec B.5).
    elem_float: bool = False


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    col: int


@dataclass(frozen=True)
class AtomicInfo:
    """Atomicity metadata for ``tt.atomic_rmw`` / ``tt.atomic_cas`` accesses."""

    rmw_op: str | None  # "fadd", "max", "exch", ... ; None for CAS
    sem: str  # memory semantic: "acq_rel", "relaxed", ...
    scope: str  # sync scope: "gpu", "cta", "sys"


@dataclass(frozen=True)
class AccessEvent:
    kind: str  # "load" | "store" | "atomic_rmw" | "atomic_cas"
    base_param: str
    offset: Term
    mask: Term | None  # None = unconditional access
    elem_bits: int
    loc: SourceLoc | None
    line_no: int
    # True when some enclosing scf.if condition could NOT be modeled (it
    # derives from loaded data). The access is then checked as if
    # unconditional: UNSAT stays a sound proof, but a SAT model may sit in a
    # branch the launch never takes, so it must not be reported as a witness
    # (check_graph turns it into ``unsupported``). Modeled conditions ride
    # in ``path`` instead and do not set this flag.
    guarded: bool = False
    # Conjunction of the MODELED enclosing branch conditions, with
    # else-regions negated (Not). The access executes iff path ∧ mask, so a
    # SAT model under both constraints is a real, reachable witness.
    path: Term | None = None
    # True when the access sits inside the scf.for body: it executes once
    # per iteration — and NOT AT ALL when the launch's trip count is zero,
    # which consumers must model (a zero-trip loop has no footprint).
    in_loop: bool = False
    # Present iff kind is atomic_*: an atomic is a read AND a write of its
    # footprint (RMW), which is what is_read/is_write encode for consumers
    # that build read/write event pairs (the race detector front-end).
    atomic: AtomicInfo | None = None
    # True when the printed mask operand derived from loaded data and was
    # over-approximated as FREE (mask=None): dropping a constraint only
    # widens the modeled footprint, so UNSAT stays a sound proof — but a SAT
    # model may pick a lane the real mask disables, so it follows the same
    # uncertainty discipline as ``guarded`` (never reported as a witness).
    mask_dropped: bool = False
    # For atomics: the printed VALUE operand (tt.atomic_rmw val /
    # tt.atomic_cas val) as a Term, or None when it is not modelable
    # (loaded data). The race encoder models the RMW write part from it.
    atomic_val: "Term | None" = None
    # For tt.atomic_cas only: the compare operand.
    atomic_cmp: "Term | None" = None
    # Float-typed pointee: the observation is never modeled (spec B.5).
    elem_float: bool = False
    # The await abstraction (spec C1): True when this access is the single
    # kept read of a recognized scf.while spin loop. ``exit_pred`` is the
    # loop's EXIT predicate over Observed(this access) — asserted on the
    # event, justified by termination (in any terminating execution the
    # final iteration's read observed the exit value); dropped iterations
    # lose no conflict pairs because every dropped event is a read of the
    # same location with the same footprint as this one. Verdicts over
    # await-bearing kernels are therefore conditional on termination
    # (surfaced as ``assumes_termination``).
    awaited: bool = False
    exit_pred: "Term | None" = None

    @property
    def is_read(self) -> bool:
        return self.kind != "store"

    @property
    def is_write(self) -> bool:
        return self.kind != "load"


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
    # Every pid axis with a parsed tt.get_program_id — recorded at PARSE
    # time, before any DataDep swallowing. Consumers deciding grid coverage
    # must use THIS set, not the axes that happen to survive into modeled
    # address/mask terms: a pid read into a stored value, a dropped mask, or
    # an unmodeled branch condition still distinguishes the blocks' behavior.
    pid_axes: set[int] = field(default_factory=set)

    def arg(self, name: str) -> FuncArg | None:
        for a in self.func_args:
            if a.name == name:
                return a
        return None


# ─────────────────────────── regexes ───────────────────────────

# `#N` is a result index into a multi-result op (`%acc#2` = third result of
# `%acc:3 = scf.for ...`). It must be part of the operand token or lines like
# `tt.store %ptrs, %acc#2, %mask` fail to match the store regex and fail
# closed even though the stored VALUE plays no part in address math. The env
# never defines `%x#N` names, so val() resolves them to DataDep("unresolved
# SSA") — sound in every consuming position (mask → dropped and flagged
# ``mask_dropped``, i.e. proof-only; addptr/ptr → unsupported).
# `-` is part of the token class: negative constants print as `%c-1_i32`,
# and truncating at the hyphen made every kernel with one fail closed.
_SSA = r"%[-\w.]+(?:#\d+)?"
_DTYPE_BITS = {
    "f64": 64, "f32": 32, "f16": 16, "bf16": 16, "f8": 8,
    "i64": 64, "i32": 32, "i16": 16, "i8": 8, "i1": 1,
    "u64": 64, "u32": 32,
    # MLIR spells the fp8 families out (torchao's quant kernels take
    # fp8 pointers); all are one byte wide
    "f8E4M3FN": 8, "f8E5M2": 8, "f8E4M3FNUZ": 8, "f8E5M2FNUZ": 8,
    "f8E4M3B11FNUZ": 8, "f8E8M0FNU": 8,
}  # fmt: skip

_RE_LOC_FILE = re.compile(r'^(#loc\d*) = loc\("([^"]+)":(\d+):(\d+)\)')
_RE_LOC_NAME = re.compile(r'^(#loc\d*) = loc\("[^"]+"\((#loc\d*)\)\)')
_RE_LOC_TRAILER = re.compile(r"loc\((#loc\d*|#loc)\)\s*$")
_RE_FUNC = re.compile(r"tt\.func\s+\w+\s+@(\w+)\((.*)\)\s*attributes")
_RE_RESULT = re.compile(rf"^({_SSA})(?::\d+)?\s*=\s*(.*)$")
_RE_GET_PID = re.compile(r"^tt\.get_program_id (\w+)")
_RE_GET_NPROG = re.compile(r"^tt\.get_num_programs (\w+)")
_RE_MAKE_RANGE = re.compile(
    r"^tt\.make_range \{end = (-?\d+) : i32, start = (-?\d+) : i32\}"
)
_RE_CONST_INT = re.compile(r"^arith\.constant (-?\d+) : i\d+")
_RE_CONST_DENSE = re.compile(r"^arith\.constant dense<(-?\d+)> : tensor")
_RE_CONST_DENSE_BOOL = re.compile(r"^arith\.constant dense<(true|false)> : tensor")
_RE_CONST_BOOL = re.compile(r"^arith\.constant (true|false)\b")
_RE_SPLAT = re.compile(rf"^tt\.splat ({_SSA}) : ([^-]+)->")
_RE_EXPAND = re.compile(rf"^tt\.expand_dims ({_SSA}) \{{axis = (\d+)")
_RE_BROADCAST = re.compile(rf"^tt\.broadcast ({_SSA})")
_RE_ADDPTR = re.compile(rf"^tt\.addptr ({_SSA}), ({_SSA})")
_RE_BIN = re.compile(
    rf"^arith\.(muli|addi|subi|divsi|remsi|minsi|maxsi) ({_SSA}), ({_SSA})"
)
_RE_CMPI = re.compile(rf"^arith\.cmpi (\w+), ({_SSA}), ({_SSA})")
# andi/ori operate on any integer width; only the i1 form is boolean logic.
# The printed result type distinguishes them (": tensor<..xi1>" / ": i1").
_RE_BOOLBIN = re.compile(rf"^arith\.(andi|ori) ({_SSA}), ({_SSA})\s*:\s*(\S+)")
_RE_SELECT = re.compile(rf"^arith\.select ({_SSA}), ({_SSA}), ({_SSA})")
_RE_EXT = re.compile(rf"^arith\.(extsi|trunci|extui) ({_SSA})")
# Trailing attributes print in TWO spellings: a dict (`{isVolatile =
# true}` for volatile spin reads) or bare assignments (`cacheModifier =
# ca` — liger's cache-hinted loads); both are irrelevant to the footprint.
_RE_LOAD = re.compile(
    rf"^tt\.load ({_SSA})((?:, {_SSA})*)\s*"
    rf"(?:\{{[^}}]*\}})?(?:\s+\w+\s*=\s*\w+)*\s*(?::|loc|$)"
)
_RE_STORE = re.compile(
    rf"^tt\.store ({_SSA}), ({_SSA})((?:, {_SSA})*)\s*"
    rf"(?:\{{[^}}]*\}})?(?:\s+\w+\s*=\s*\w+)*\s*(?::|loc|$)"
)
# Atomic RMW prints (op, sem, scope, ptr, val, mask); an unmasked tl.atomic_*
# still carries a mask operand (a dense<true> constant), so the group is
# always present. CAS prints (sem, scope, ptr, cmp, val) — no mask exists.
_RE_ATOMIC_RMW = re.compile(
    rf"^tt\.atomic_rmw (\w+), (\w+), (\w+), ({_SSA}), ({_SSA}), ({_SSA})\s*(?::|loc|$)"
)
_RE_ATOMIC_CAS = re.compile(
    rf"^tt\.atomic_cas (\w+), (\w+), ({_SSA}), ({_SSA}), ({_SSA})\s*(?::|loc|$)"
)
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
_RE_SCF_IF = re.compile(rf"^scf\.if ({_SSA})")
# The await shape (C1.1): only the argument-free, result-free spin form is
# accepted; anything carrying values is refused as "spin-shape".
_RE_SCF_WHILE_SPIN = re.compile(r"^scf\.while\s*:\s*\(\)\s*->\s*\(\)\s*\{")
_RE_SCF_CONDITION = re.compile(rf"^scf\.condition\(({_SSA})\)")


@dataclass
class _IfFrame:
    """Walker state for one open scf.if region."""

    cond: "Term | None"  # modeled condition; None → accesses stay `guarded`
    res: str | None  # single-result SSA name ("%r"), if the if yields
    branch: str = "then"
    # Yield VALUES resolved at the yield line — then/else regions legally
    # reuse the same SSA names, so resolving at close time would read the
    # else-region's overwrites.
    then_vals: "list[object] | None" = None
    else_vals: "list[object] | None" = None


@dataclass
class _WhileFrame:
    """Walker state for one open scf.while spin candidate (C1.1).

    The CONDITION region ("before") holds the awaited re-read plus its
    address bookkeeping and ends at scf.condition; the BODY region ("do")
    must be pure bookkeeping (scf.yield only). Any clause violation refuses
    the kernel with kind="spin-shape" naming the clause."""

    open_line: int
    stage: str = "cond"  # "cond" → "body"
    n_accesses_before: int = 0
    cond_val: object | None = None  # resolved AT the scf.condition line


def _branch_state(frames: list) -> "tuple[bool, Term | None, bool]":
    """(guarded, path, in_loop) for an access under the open frames:
    ``guarded`` if any enclosing condition is unmodeled; ``path`` is the
    conjunction of the modeled ones (else-regions negated); ``in_loop`` when
    an scf.for body encloses the access."""
    guarded = False
    path: Term | None = None
    in_loop = False
    for f in frames:
        if f == "for":
            in_loop = True
            continue
        if not isinstance(f, _IfFrame):
            continue
        if f.cond is None:
            guarded = True
            continue
        c: Term = f.cond if f.branch == "then" else Not(f.cond)
        path = c if path is None else BoolBin("and", path, c)
    return guarded, path, in_loop


def _elem_bits(type_str: str) -> int:
    m = _RE_PTR_ELEM.search(type_str)
    if m:
        return _DTYPE_BITS.get(m.group(1), 0)
    return 0


def _elem_is_float(type_str: str) -> bool:
    m = _RE_PTR_ELEM.search(type_str)
    return m is not None and m.group(1).startswith(("f", "bf"))


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
            fa = FuncArg(
                name=name,
                is_ptr=is_ptr,
                elem_bits=bits,
                elem_float=_elem_is_float(ty) if is_ptr else False,
            )
            func_args.append(fa)
            # Pointer args seed addptr chains; scalar args are Param leaves.
            env[f"%{name}"] = PtrValue(name, Const(0)) if is_ptr else Param(name)

    def base_elem_bits(param: str) -> int:
        fa = next((a for a in func_args if a.name == param), None)
        return fa.elem_bits if fa else 0

    def base_elem_float(param: str) -> bool:
        fa = next((a for a in func_args if a.name == param), None)
        return fa.elem_float if fa else True  # unknown pointee: fail closed

    def operand_term(v: object) -> "Term | None":
        """An atomic cmp/val operand as a Term, or None when unmodelable."""
        return None if isinstance(v, (DataDep, PtrValue)) else v  # type: ignore[return-value]

    def observed_result_binding() -> object:
        """The env value for the just-recorded access's result: Observed
        for an integer-typed access (spec part B / the await re-read),
        DataDep otherwise (float pointees stay outside the Int model)."""
        if accesses and not accesses[-1].elem_float:
            return Observed(len(accesses) - 1)
        return DataDep("atomic result")

    # ── body parse (single function; loop handled inline) ──
    # Region stack: "for" | _IfFrame. Tracking scf.if frames keeps the
    # walker's brace accounting honest (an if's closing brace inside a loop
    # must not be mistaken for the loop's close, nor its scf.yield for the
    # loop's yield), carries the modeled branch condition for the accesses
    # inside (``path``), and marks accesses under an UNMODELED condition as
    # ``guarded``.
    frames: list = []
    loop_body_yields: list[str] = []
    loop_iter_arg_ssa: list[tuple[str, str]] = []  # (arg_ssa, init_ssa)
    loop_meta: dict[str, object] = {}
    pid_axes: set[int] = set()

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

        # ---- scf.while body region: pure bookkeeping only (C1.1) ----
        # Placed FIRST so stray ops in the "do" region are refused before
        # any other handler could record them; brace lines fall through to
        # the region-close logic below.
        if (
            frames
            and isinstance(frames[-1], _WhileFrame)
            and frames[-1].stage == "body"
            and not line.startswith("}")
        ):
            if body.startswith("scf.yield"):
                continue
            raise UnsupportedTTIR(
                f"line {line_no}: spin-loop body must be pure bookkeeping "
                f"(scf.yield), found: {body.split(' ', 1)[0]}",
                kind="spin-shape",
            )

        # ---- scf.while (the await abstraction, C1) ----
        if body.startswith("scf.while"):
            if res is not None or not _RE_SCF_WHILE_SPIN.match(body):
                raise UnsupportedTTIR(
                    f"line {line_no}: scf.while carries values (iter args or "
                    "results) — only the argument-free spin form is the "
                    "await shape",
                    kind="spin-shape",
                )
            if any(isinstance(f, _WhileFrame) for f in frames):
                raise UnsupportedTTIR(
                    f"line {line_no}: nested spin loops are not the await " "shape",
                    kind="spin-shape",
                )
            frames.append(
                _WhileFrame(open_line=line_no, n_accesses_before=len(accesses))
            )
            continue

        cm = _RE_SCF_CONDITION.match(body)
        if cm:
            top = frames[-1] if frames else None
            if not (isinstance(top, _WhileFrame) and top.stage == "cond"):
                raise UnsupportedTTIR(
                    f"line {line_no}: scf.condition outside a spin loop",
                    kind="control-flow",
                )
            # Resolve NOW: region SSA names must not be re-read at close.
            top.cond_val = val(cm.group(1))
            continue

        if line.startswith("} do") and frames and isinstance(frames[-1], _WhileFrame):
            top = frames[-1]
            if top.cond_val is None:
                raise UnsupportedTTIR(
                    f"line {line_no}: spin loop without scf.condition",
                    kind="spin-shape",
                )
            top.stage = "body"
            continue

        # ---- scf.for ----
        fm = _RE_SCF_FOR.match(body)
        if fm:
            # ``loop`` is only set at the closing brace, so a second
            # SEQUENTIAL loop is caught by it — but a NESTED loop opens while
            # the outer one is still in flight (loop is still None), so guard
            # on open frames too. Nested loops carry independent induction
            # variables the single-loop model cannot represent, and a loop
            # under an scf.if runs a condition-dependent iteration count;
            # reject rather than silently mis-bound the induction var.
            if loop is not None or frames:
                raise UnsupportedTTIR(
                    f"line {line_no}: multiple/nested loops",
                    # A loop under an scf.if runs a branch-dependent
                    # iteration count — a control-flow limitation, not one
                    # more induction variable.
                    kind=(
                        "control-flow"
                        if any(isinstance(f, _IfFrame) for f in frames)
                        else "nested-loop"
                    ),
                )
            ind, lo, up, st, iters = fm.groups()
            pairs: list[tuple[str, str]] = []
            if iters:
                pairs = list(re.findall(rf"({_SSA}) = ({_SSA})", iters))
            bound_terms: dict[str, Term] = {}
            for label, ssa in (("lower", lo), ("upper", up), ("step", st)):
                bv = val(ssa)
                if isinstance(bv, DataDep):
                    # The CSR shape: for k in range(loaded_start, loaded_end).
                    raise UnsupportedTTIR(
                        f"loop {label} bound: data-dependent ({bv.why})",
                        kind="data-dependent-bound" if _from_memory(bv) else "other",
                    )
                if mentions_observed(bv):
                    # A trip count driven by an atomic observation is a
                    # dynamic work-fetch loop — outside the single-loop
                    # model (looped RMW fetch is a B+C1 stretch item).
                    raise UnsupportedTTIR(
                        f"loop {label} bound depends on an atomic observation",
                        kind="data-dependent-bound",
                    )
                bound_terms[label] = as_term(bv, f"loop {label}")
            loop_meta = {
                "ssa": res or "%loop",
                "ind": ind,
                "lower": bound_terms["lower"],
                "upper": bound_terms["upper"],
                "step": bound_terms["step"],
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
            frames.append("for")
            continue

        # ---- scf.if: track the region and model its condition ----
        if body.startswith("scf.if"):
            im = _RE_SCF_IF.match(body)
            cond_t: Term | None = None
            if im:
                cv = val(im.group(1))
                # A pointer can't be a condition; loaded data (DataDep)
                # can't be modeled → the region stays pessimistically
                # ``guarded`` exactly as before this feature.
                if not isinstance(cv, (DataDep, PtrValue)):
                    cond_t = cv  # type: ignore[assignment]
            frames.append(_IfFrame(cond=cond_t, res=res))
            # Fallback binding; upgraded to Select at the closing brace when
            # the condition and both branches' single yield are modelable.
            if res is not None:
                env[res] = DataDep("scf.if result")
            continue

        if frames and (
            line == "}" or line.startswith("} loc") or line.startswith("} else")
        ):
            if line.startswith("} else"):
                # The then-region closes and the else-region opens: the same
                # if frame stays on the stack with its condition negated for
                # the accesses that follow.
                top = frames[-1]
                if not isinstance(top, _IfFrame):
                    raise UnsupportedTTIR(f"line {line_no}: unexpected `else`")
                top.branch = "else"
                continue
            popped = frames.pop()
            if isinstance(popped, _WhileFrame):
                _finalize_await(popped, accesses, line_no)
                continue
            if isinstance(popped, _IfFrame):
                if (
                    popped.res is not None
                    and popped.cond is not None
                    and popped.then_vals is not None
                    and popped.else_vals is not None
                    and len(popped.then_vals) == 1
                    and len(popped.else_vals) == 1
                ):
                    tv, ev = popped.then_vals[0], popped.else_vals[0]
                    # Yielded pointers or loaded data keep the DataDep
                    # fallback (a stored VALUE never enters address math;
                    # an address use of the result then fails closed).
                    if not any(isinstance(x, (DataDep, PtrValue)) for x in (tv, ev)):
                        env[popped.res] = Select(
                            popped.cond,
                            as_term(tv, "scf.if yield"),
                            as_term(ev, "scf.if yield"),
                        )
                continue
            # A "for" frame closed: resolve deltas from the yields, positionally.
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
        if ym and frames and frames[-1] == "for":
            # Only the loop's own yield resolves iter-arg deltas; an scf.if's
            # yield inside the loop body must not clobber it.
            loop_body_yields = _split_ssa(ym.group(1))
            continue
        if ym and frames and isinstance(frames[-1], _IfFrame):
            # Resolve yield VALUES here, not at the closing brace: then/else
            # regions legally reuse the same SSA names, so a close-time
            # lookup would read the else-region's overwrites.
            fr = frames[-1]
            vals = [val(s) for s in _split_ssa(ym.group(1))]
            if fr.branch == "then":
                fr.then_vals = vals
            else:
                fr.else_vals = vals
            continue

        # ---- other control flow: fail closed ----
        # scf.for, scf.if and the scf.while await shape are region-tracked
        # above. Anything else that steers control flow (unstructured cf.*)
        # would be flat-scanned as if it executed unconditionally — reject
        # the kernel instead.
        if body.startswith(("scf.", "cf.")) and not body.startswith(
            ("scf.for", "scf.if", "scf.yield")
        ):
            raise UnsupportedTTIR(
                f"line {line_no}: control flow {body.split(' ', 1)[0]} is unsupported",
                kind="control-flow",
            )

        # ---- value-producing ops ----
        handled = _parse_value_op(
            body, res, env, val, as_term, base_elem_bits, pid_axes
        )
        if handled:
            continue

        # ---- accesses ----
        lm = _RE_LOAD.match(body)
        if lm:
            guarded, path, in_loop = _branch_state(frames)
            _record_access(
                "load",
                lm.group(1),
                lm.group(2),
                guarded,
                env,
                val,
                accesses,
                base_elem_bits,
                loc,
                line_no,
                path=path,
                in_loop=in_loop,
                base_elem_float=base_elem_float,
            )
            if res is not None:
                in_while_cond = any(
                    isinstance(f, _WhileFrame) and f.stage == "cond" for f in frames
                )
                # A spin re-read's value IS an observation (the await's
                # exit predicate is asserted over it, C1.2); everywhere
                # else a loaded value stays DataDep.
                env[res] = (
                    observed_result_binding()
                    if in_while_cond
                    else DataDep("loaded value")
                )
            continue
        sm = _RE_STORE.match(body)
        if sm:
            if any(isinstance(f, _WhileFrame) for f in frames):
                raise UnsupportedTTIR(
                    f"line {line_no}: store inside a spin loop is not the "
                    "await shape",
                    kind="spin-shape",
                )
            guarded, path, in_loop = _branch_state(frames)
            _record_access(
                "store",
                sm.group(1),
                sm.group(3),
                guarded,
                env,
                val,
                accesses,
                base_elem_bits,
                loc,
                line_no,
                path=path,
                in_loop=in_loop,
            )
            continue
        am = _RE_ATOMIC_RMW.match(body)
        if am:
            guarded, path, in_loop = _branch_state(frames)
            _record_access(
                "atomic_rmw",
                am.group(4),
                am.group(6),  # the mask operand
                guarded,
                env,
                val,
                accesses,
                base_elem_bits,
                loc,
                line_no,
                atomic=AtomicInfo(am.group(1), am.group(2), am.group(3)),
                path=path,
                in_loop=in_loop,
                atomic_val=operand_term(val(am.group(5))),
                base_elem_float=base_elem_float,
            )
            if res is not None:
                env[res] = observed_result_binding()
            continue
        am = _RE_ATOMIC_CAS.match(body)
        if am:
            guarded, path, in_loop = _branch_state(frames)
            _record_access(
                "atomic_cas",
                am.group(3),
                "",  # CAS has no mask operand: unconditional footprint
                guarded,
                env,
                val,
                accesses,
                base_elem_bits,
                loc,
                line_no,
                atomic=AtomicInfo(None, am.group(1), am.group(2)),
                path=path,
                in_loop=in_loop,
                atomic_val=operand_term(val(am.group(5))),
                atomic_cmp=operand_term(val(am.group(4))),
                base_elem_float=base_elem_float,
            )
            if res is not None:
                env[res] = observed_result_binding()
            continue

        # ---- fail closed on unrecognized memory ops ----
        # A tt.load/tt.store/tt.atomic_* syntax variant the regexes above did
        # not match must NOT fall through to the value/DataDep handling below:
        # a store has no result so it would be silently dropped, and an
        # atomic's access would go unchecked while its result becomes a
        # harmless-looking DataDep. Either way check_graph would then prove
        # "ok" without having checked a real access. Bail to unsupported
        # instead so the proof stays sound.
        if body.startswith(
            (
                "tt.load",
                "tt.store",
                "tt.atomic_",
                "tt.descriptor_",
                "tt.experimental_descriptor_",
            )
        ):
            raise UnsupportedTTIR(
                f"line {line_no}: unsupported memory op syntax: {body[:60]}",
                kind="out-of-vocabulary",
            )

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
            raise UnsupportedTTIR(
                f"line {line_no}: block pointers are unsupported",
                kind="block-pointer",
            )
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
        pid_axes=pid_axes,
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
    if isinstance(v, Not):
        return Not(_set_arange_dim(v.a, dim))  # type: ignore[arg-type]
    return v


def _finalize_await(frame: _WhileFrame, accesses: list, line_no: int) -> None:
    """Validate the C1.1 shape contract at the spin loop's closing brace and
    stamp the kept read with ``awaited`` + the EXIT predicate.

    ``scf.condition(c)`` continues WHILE c holds, so the exit predicate is
    ``Not(c)`` — for ``while load(flag) != 1`` that is ``flag == 1``; for
    the CAS form ``while cas(lock,0,1) != 0`` it is ``old == 0`` (success).
    Memory order/scope stay exactly as the op was written: a relaxed spin
    must yield no synchronizes-with edge — that IS the missing-acquire bug
    the detector exists to find."""
    where = f"line {frame.open_line} (scf.while)"
    n_new = len(accesses) - frame.n_accesses_before
    if n_new != 1:
        raise UnsupportedTTIR(
            f"{where}: the spin condition must re-read exactly one location "
            f"(found {n_new} memory accesses)",
            kind="spin-shape",
        )
    idx = len(accesses) - 1
    acc = accesses[idx]
    if acc.elem_float:
        raise UnsupportedTTIR(
            f"{where}: the awaited location is float-typed (the observation "
            "model is Int-sort only)",
            kind="spin-shape",
        )
    # The await encoding keeps ONE read and drops every earlier iteration —
    # sound only when the re-read is side-effect-free on the awaited
    # location. A plain load never writes; a CAS writes exactly once (on
    # success — the single modeled write). A mutating RMW re-read
    # (atomic_add(flag, 1) spins) writes on EVERY dropped iteration: the
    # loop can terminate by observing its OWN increments, and modeling the
    # exit value as read-from a release writer fabricates a
    # synchronizes-with edge (adversarial finding: self-satisfying spin
    # proved a real data race away). Accept an RMW only when its written
    # value provably equals the observation: add/or/xor with a constant 0.
    if acc.kind == "atomic_rmw":
        op = ((acc.atomic.rmw_op if acc.atomic else None) or "").lower()
        identity = op in ("add", "or", "xor") and acc.atomic_val == Const(0)
        if not identity:
            raise UnsupportedTTIR(
                f"{where}: the spin re-read MUTATES the awaited location "
                f"(atomic {op or '?'} with a non-identity operand); dropped "
                "iterations would lose real writes",
                kind="spin-shape",
            )
    cv = frame.cond_val
    if not isinstance(cv, Cmp):
        raise UnsupportedTTIR(
            f"{where}: the spin condition is not a comparison over the " "awaited read",
            kind="spin-shape",
        )
    a_is_obs = isinstance(cv.a, Observed) and cv.a.access_index == idx
    b_is_obs = isinstance(cv.b, Observed) and cv.b.access_index == idx
    expected = cv.b if a_is_obs else cv.a
    if a_is_obs == b_is_obs or idx in observed_indices(expected):
        raise UnsupportedTTIR(
            f"{where}: the spin condition must compare the awaited read "
            "against a loop-invariant expected value",
            kind="spin-shape",
        )
    accesses[idx] = replace(acc, awaited=True, exit_pred=Not(cv))


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


def _parse_value_op(body, res, env, val, as_term, base_elem_bits, pid_axes) -> bool:
    """Parse one address-structure value op into env. Returns True if handled."""
    if res is None:
        return False

    m = _RE_GET_PID.match(body)
    if m:
        axis = {"x": 0, "y": 1, "z": 2}.get(m.group(1))
        if axis is None:
            # Printer drift must surface as the designed error, not a bare
            # KeyError escaping into the client's launch teardown.
            raise UnsupportedTTIR(
                f"unknown program-id axis {m.group(1)!r}",
                kind="out-of-vocabulary",
            )
        # Parse-time record (see AccessGraph.pid_axes): the read counts even
        # if this value never survives into a modeled term.
        pid_axes.add(axis)
        env[res] = Pid(axis)
        return True
    m = _RE_GET_NPROG.match(body)
    if m:
        axis = {"x": 0, "y": 1, "z": 2}.get(m.group(1))
        if axis is None:
            raise UnsupportedTTIR(
                f"unknown num-programs axis {m.group(1)!r}",
                kind="out-of-vocabulary",
            )
        # The verdict depends on this grid dim (see NumPrograms): keep the
        # axis symbolic even when no pid read distinguishes blocks along it.
        pid_axes.add(axis)
        env[res] = NumPrograms(axis)
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
    m = _RE_CONST_DENSE_BOOL.match(body) or _RE_CONST_BOOL.match(body)
    if m:
        # i1 constants (e.g. the dense<true> mask of an unmasked atomic).
        # Const(0/1) in a boolean position is coerced by the evaluator.
        env[res] = Const(1 if m.group(1) == "true" else 0)
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
            raise UnsupportedTTIR(
                "addptr base is not a pointer",
                kind="indirect-address" if _from_memory(base) else "other",
            )
        if isinstance(off, DataDep):
            # A value in an address chain that cannot be modeled: a free
            # address makes the query meaningless, so this stays
            # whole-kernel unsupported. Only offsets truly derived from
            # MEMORY CONTENTS classify as indirection (the interpreter
            # front-end route); modeling gaps (loop accumulators, unmodeled
            # ops, ...) keep the default kind so the buckets stay honest.
            raise UnsupportedTTIR(
                f"addptr offset: data-dependent ({off.why})",
                kind="indirect-address" if _from_memory(off) else "other",
            )
        off_t = as_term(off, "addptr offset")
        env[res] = PtrValue(base.base_param, Bin("+", base.offset, off_t))
        return True
    m = _RE_BIN.match(body)
    if m:
        op = {
            "muli": "*",
            "addi": "+",
            "subi": "-",
            "divsi": "//",
            "remsi": "%",
            "minsi": "min",
            "maxsi": "max",
        }[m.group(1)]
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
        ty = m.group(4)
        if not (ty == "i1" or ty.endswith("i1>")):
            # Wide-int andi/ori is BITWISE arithmetic, not boolean logic;
            # modeling it as And/Or would silently corrupt address math
            # (e.g. ``offs & 8`` collapsing to a {0,1} truth value). Degrade
            # to DataDep so an address use fails closed as unsupported.
            env[res] = DataDep(f"bitwise arith.{m.group(1)} on non-i1 type {ty}")
            return True
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
    guarded,
    env,
    val,
    accesses,
    base_elem_bits,
    loc,
    line_no,
    atomic=None,
    path=None,
    in_loop=False,
    atomic_val=None,
    atomic_cmp=None,
    base_elem_float=None,
) -> None:
    ptr = val(ptr_ssa)
    if not isinstance(ptr, PtrValue):
        raise UnsupportedTTIR(
            f"line {line_no}: {kind} of a non-pointer value",
            kind="indirect-address" if _from_memory(ptr) else "other",
        )
    # Mask: for load it's the first trailing operand; for store the operand
    # after value. _RE_LOAD captures trailing ", %x" groups; for store the
    # caller passed the post-value trailing operands.
    mask: Term | None = None
    mask_dropped = False
    trailing = _split_ssa(extra_ops) if extra_ops else []
    if trailing:
        mv = val(trailing[0])
        if isinstance(mv, DataDep):
            # Mask derived from loaded data: over-approximate it as free
            # (any lane may be active) instead of failing the whole kernel.
            # See AccessEvent.mask_dropped for the soundness discipline.
            mask_dropped = True
        elif isinstance(mv, PtrValue):
            raise UnsupportedTTIR(f"line {line_no}: pointer as mask")
        else:
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
            guarded=guarded,
            atomic=atomic,
            path=path,
            mask_dropped=mask_dropped,
            in_loop=in_loop,
            atomic_val=atomic_val,
            atomic_cmp=atomic_cmp,
            elem_float=(base_elem_float(ptr.base_param) if base_elem_float else False),
        )
    )
