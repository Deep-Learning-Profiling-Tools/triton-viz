"""Atomic-ordering barrier-coverage gate over post-Membar PTX (the A2 gate).

Checks that the TTGIR-to-LLVM lowering emitted the CTA-level ordering
barriers that non-relaxed atomic memory semantics require: the rule fixed
by triton-lang/triton PR #10816. Before that fix, a ``release`` atomic was
lowered with only its per-instruction qualifier, ordering the issuing
thread but synchronizing nobody else in the CTA; the fix inserts a CTA
barrier immediately before release/acq_rel atomics and immediately after
acquire/acq_rel ones (dischargeable by the result-staging barrier).

Obligation side: a lightweight scan of the TTIR text for
``tt.atomic_rmw`` / ``tt.atomic_cas`` sites with their (sem, scope) and
user source location. Deliberately independent of ``parse_ttir``: the
full reader refuses whole kernels outside its fragment, while the gate
needs only the atomic sites, so it stays applicable where the graph
reader abstains. The regexes mirror ``common/ttir_reader.py``'s.

Discharge side: a linear parse of the PTX into basic blocks (branch
target labels ``$L__BB*`` and branch/return instructions bound blocks;
debug labels ``$L__tmp*`` / ``$L__func_*`` do not), atomic instructions
(``atom.*`` / ``red.*``; sem and scope read from the qualifier token set,
since RMW prints scope-before-sem while CAS prints sem-before-scope),
``bar.sync`` barriers, and a barrier / memory / staging-store / other
classification per instruction. Obligations match PTX sites by user
source line (the ``.loc`` table) plus op kind.

The rule (numCTAs == 1, NVIDIA):
  * sem in {release, acq_rel}: scanning backwards from the atomic inside
    its block, skipping non-memory instructions, the first significant
    instruction must be an unpredicated ``bar.sync``.
  * sem in {acquire, acq_rel}: the forward mirror, additionally skipping
    ``st.shared`` (the result-staging store: the st.shared / bar.sync /
    ld.shared reload sequence discharges the post-side barrier, per the
    #10816 rule).
  * relaxed: no obligation.

Fail-closed: anything outside this vocabulary (cluster barriers,
``tt.atomic_poll``, unmatched or location-less obligations, sem
mismatches) refuses with a named kind. A refusal is never a pass and
never a violation — the unsupported-not-race discipline.

Spec: ``impl-spec-a2-gate.md`` in the paper repo, to be ported into
``race_detector_static_hybrid_plan.md`` Part II §8.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_SEMS = ("relaxed", "acquire", "release", "acq_rel")
_SCOPES = ("cta", "gpu", "sys")

# ── obligations from TTIR ─────────────────────────────────────────────

_RE_LOC_DEF = re.compile(r'^#loc(\d*) = loc\("([^"]+)":(\d+):(\d+)\)')
# Named alias defs, e.g. `#loc6 = loc("old"(#loc1))` (an SSA value name
# wrapping the real location); resolved transitively.
_RE_LOC_ALIAS = re.compile(r'^#loc(\d*) = loc\("[^"]*"\(#loc(\d*)\)\)')
_RE_LOC_REF = re.compile(r"loc\(#loc(\d*)\)\s*$")
# Mirrors _RE_ATOMIC_RMW / _RE_ATOMIC_CAS in common/ttir_reader.py, kept
# local so the gate needs no graph parse: (rmw_op, sem, scope) / (sem,
# scope).
_RE_RMW = re.compile(r"\btt\.atomic_rmw (\w+), (\w+), (\w+),")
_RE_CAS = re.compile(r"\btt\.atomic_cas (\w+), (\w+),")
_RE_POLL = re.compile(r"\btt\.atomic_poll\b")


class GateUnsupported(Exception):
    """Named refusal; ``kind`` is the stable routing/bucketing prefix."""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind


@dataclass
class Obligation:
    kind: str  # "rmw" | "cas"
    sem: str  # relaxed | acquire | release | acq_rel
    scope: str
    loc: tuple[str, int] | None  # (user file path, line)
    ttir_line: int


def ttir_obligations(text: str) -> list[Obligation]:
    """Extract the atomic ordering obligations from TTIR text.

    Raises :class:`GateUnsupported` on vocabulary it must not guess
    about (``tt.atomic_poll``, unknown sems).
    """
    lines = text.splitlines()
    loc_table: dict[str, tuple[str, int]] = {}
    aliases: dict[str, str] = {}
    for raw in lines:
        s = raw.strip()
        m = _RE_LOC_DEF.match(s)
        if m:
            loc_table[m.group(1)] = (m.group(2), int(m.group(3)))
            continue
        a = _RE_LOC_ALIAS.match(s)
        if a:
            aliases[a.group(1)] = a.group(2)
    for key, target in aliases.items():
        seen = {key}
        while target in aliases and target not in seen:
            seen.add(target)
            target = aliases[target]
        if target in loc_table:
            loc_table[key] = loc_table[target]
    obs: list[Obligation] = []
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if _RE_POLL.search(s):
            raise GateUnsupported(
                "atomic-poll",
                f"ttir line {i}: tt.atomic_poll is outside the gate's v1 vocabulary",
            )
        m = _RE_RMW.search(s)
        if m:
            kind, sem, scope = "rmw", m.group(2), m.group(3)
        else:
            mc = _RE_CAS.search(s)
            if mc is None:
                continue
            kind, sem, scope = "cas", mc.group(1), mc.group(2)
        if sem not in _SEMS:
            raise GateUnsupported("unknown-sem", f"ttir line {i}: sem {sem!r}")
        lm = _RE_LOC_REF.search(s)
        loc = loc_table.get(lm.group(1)) if lm else None
        obs.append(Obligation(kind, sem, scope, loc, i))
    return obs


# ── PTX parse ─────────────────────────────────────────────────────────

_RE_FILE = re.compile(r'^\.file\s+(\d+)\s+"([^"]+)"')
_RE_LOC = re.compile(r"^\.loc\s+(\d+)\s+(\d+)\s+(\d+)")
_RE_BB_LABEL = re.compile(r"^\$L__BB\S*:")
_RE_ANY_LABEL = re.compile(r"^[$\w][\w$.]*:\s*$")
_RE_PRED = re.compile(r"^@!?%p\d+\s+")

# Memory-space-touching instruction prefixes for the adjacency rule.
# ld.param/ld.const and st.param read/write the parameter space and are
# irrelevant to CTA data ordering, so they classify as "other".
_MEM_PREFIXES = (
    "ld.global",
    "ld.shared",
    "ld.local",
    "ld.volatile",
    "st.global",
    "st.local",
    "st.volatile",
    "cp.async",
)


@dataclass
class PtxInstr:
    idx: int
    text: str  # predicate stripped
    cls: str  # "barrier" | "memory" | "st_shared" | "other"
    predicated: bool
    block: int
    loc: tuple[str, int] | None


@dataclass
class PtxSite:
    instr: PtxInstr
    kind: str  # "rmw" | "cas"
    sem: str
    scope: str | None
    space: str | None  # "global" | "shared" | None (generic)


@dataclass
class PtxProgram:
    instrs: list[PtxInstr] = field(default_factory=list)
    sites: list[PtxSite] = field(default_factory=list)


def _classify_atomic(op_token: str) -> tuple[str, str, str | None, str | None]:
    """(kind, sem, scope, space) from an atom./red. mnemonic token."""
    toks = op_token.split(".")
    kind = "cas" if "cas" in toks else "rmw"
    sem = next((t for t in toks if t in _SEMS), "relaxed")
    scope = next((t for t in toks if t in _SCOPES), None)
    space = next((t for t in toks if t in ("global", "shared", "local")), None)
    return kind, sem, scope, space


def parse_ptx(text: str) -> PtxProgram:
    """Linear parse: instructions, blocks, atomic sites, barriers."""
    prog = PtxProgram()
    # The .file table sits at the END of the PTX module, after the code
    # it annotates, so it must be collected in a pre-pass.
    files: dict[str, str] = {}
    for raw in text.splitlines():
        fm = _RE_FILE.match(raw.split("//", 1)[0].strip())
        if fm:
            files[fm.group(1)] = fm.group(2)
    cur_loc: tuple[str, int] | None = None
    block = 0
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        fm = _RE_FILE.match(line)
        if fm:
            files[fm.group(1)] = fm.group(2)
            continue
        lm = _RE_LOC.match(line)
        if lm:
            f = files.get(lm.group(1))
            cur_loc = (f, int(lm.group(2))) if f else None
            continue
        if line.startswith("."):
            continue  # other directives
        if _RE_BB_LABEL.match(line):
            block += 1
            continue
        if _RE_ANY_LABEL.match(line):
            continue  # debug/function labels bound nothing
        if line in ("{", "}") or line.endswith("{") or line.startswith(")"):
            continue
        predicated = bool(_RE_PRED.match(line))
        body = _RE_PRED.sub("", line)
        op_token = body.split(None, 1)[0] if body else ""
        if op_token.startswith("barrier.cluster"):
            raise GateUnsupported(
                "cluster-barrier",
                "cluster barrier flavor is outside the gate's v1 scope",
            )
        if op_token.startswith(("bar.sync", "barrier.sync")):
            cls = "barrier"
        elif op_token.startswith("st.shared"):
            cls = "st_shared"
        elif op_token.startswith(("atom.", "red.")) or op_token.startswith(
            _MEM_PREFIXES
        ):
            cls = "memory"
        elif op_token.startswith(("ld.param", "ld.const", "st.param")):
            cls = "other"
        elif op_token.startswith(("ld.", "st.")):
            cls = "memory"  # generic-address ld/st
        else:
            cls = "other"
        instr = PtxInstr(len(prog.instrs), body, cls, predicated, block, cur_loc)
        prog.instrs.append(instr)
        if op_token.startswith(("atom.", "red.")):
            kind, sem, scope, space = _classify_atomic(op_token)
            prog.sites.append(PtxSite(instr, kind, sem, scope, space))
        elif op_token.startswith(("ld.", "st.")):
            # The zero-RMW peephole: triton lowers e.g.
            # tl.atomic_add(p, 0, sem="acquire") to a sem-qualified plain
            # load (ld.global.gpu.acquire, often inline-asm) with a
            # broadcast staging sequence, not an atom instruction. Such
            # sem-carrying ld/st are obligation sites too ("ldst" kind,
            # matchable against rmw obligations).
            toks = op_token.split(".")
            lsem = next((t for t in toks if t in _SEMS), None)
            if lsem is not None:
                lscope = next((t for t in toks if t in _SCOPES), None)
                lspace = next(
                    (t for t in toks if t in ("global", "shared", "local")), None
                )
                prog.sites.append(PtxSite(instr, "ldst", lsem, lscope, lspace))
        if op_token.startswith("bra") or op_token.startswith("ret"):
            block += 1
    return prog


# ── the coverage check ────────────────────────────────────────────────


@dataclass
class GateResult:
    status: str  # "verified" | "violation" | "unsupported"
    reason: str | None
    reports: list[str]
    obligations: int = 0


def _covered_before(prog: PtxProgram, site: PtxSite) -> bool:
    i = site.instr.idx - 1
    while i >= 0 and prog.instrs[i].block == site.instr.block:
        ins = prog.instrs[i]
        if ins.cls == "barrier":
            return not ins.predicated
        if ins.cls in ("memory", "st_shared"):
            return False
        i -= 1
    return False


def _covered_after(prog: PtxProgram, site: PtxSite) -> bool:
    i = site.instr.idx + 1
    while i < len(prog.instrs) and prog.instrs[i].block == site.instr.block:
        ins = prog.instrs[i]
        if ins.cls == "barrier":
            return not ins.predicated
        if ins.cls == "st_shared":
            i += 1  # result-staging store; its bar.sync discharges the post side
            continue
        if ins.cls == "memory":
            return False
        i += 1
    return False


def _match_sites(
    obs: list[Obligation], prog: PtxProgram
) -> list[tuple[Obligation, list[PtxSite]]]:
    matched: list[tuple[Obligation, list[PtxSite]]] = []
    for ob in obs:
        if ob.loc is None:
            raise GateUnsupported(
                "no-loc",
                f"ttir line {ob.ttir_line}: atomic without a resolvable "
                "source location (line info disabled, or a callsite loc)",
            )
        base = os.path.basename(ob.loc[0])
        cands = [
            s
            for s in prog.sites
            if (s.kind == ob.kind or (s.kind == "ldst" and ob.kind == "rmw"))
            and s.instr.loc is not None
            and s.instr.loc[1] == ob.loc[1]
            and os.path.basename(s.instr.loc[0]) == base
        ]
        if not cands:
            raise GateUnsupported(
                "obligation-unmatched",
                f"no PTX atomic found for the {ob.sem} {ob.kind} at "
                f"{base}:{ob.loc[1]} (ttir line {ob.ttir_line})",
            )
        sems = {s.sem for s in cands}
        if sems != {ob.sem}:
            raise GateUnsupported(
                "sem-mismatch",
                f"{base}:{ob.loc[1]}: ttir sem {ob.sem!r} vs ptx sems "
                f"{sorted(sems)}",
            )
        matched.append((ob, cands))
    return matched


def check_gate(ttir_text: str, ptx_text: str, kernel: str = "<kernel>") -> GateResult:
    """Run the barrier-coverage gate for one compiled specialization."""
    try:
        obs = ttir_obligations(ttir_text)
        prog = parse_ptx(ptx_text)
        reports: list[str] = []
        for ob, sites in _match_sites(obs, prog):
            if ob.sem == "relaxed":
                continue
            base = os.path.basename(ob.loc[0]) if ob.loc else "?"
            where = f"{base}:{ob.loc[1]}" if ob.loc else f"ttir:{ob.ttir_line}"
            for s in sites:
                if ob.sem in ("release", "acq_rel") and not _covered_before(prog, s):
                    reports.append(
                        f"{kernel}: {ob.sem} {ob.kind} at {where}: no CTA "
                        f"barrier before the atomic ({s.instr.text.split()[0]})"
                    )
                if ob.sem in ("acquire", "acq_rel") and not _covered_after(prog, s):
                    reports.append(
                        f"{kernel}: {ob.sem} {ob.kind} at {where}: no CTA "
                        f"barrier after the atomic ({s.instr.text.split()[0]})"
                    )
    except GateUnsupported as e:
        return GateResult("unsupported", f"{e.kind}: {e}", [], 0)
    if reports:
        return GateResult("violation", None, reports, len(obs))
    reason = "no atomic ordering obligations" if not obs else None
    return GateResult("verified", reason, [], len(obs))
