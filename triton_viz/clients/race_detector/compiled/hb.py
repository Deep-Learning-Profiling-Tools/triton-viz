"""Happens-before model for the cp.async pipeline at TTGIR level.

TTGIR carries NO CTA barriers (the backend Membar pass inserts ``bar.sync``
during lowering — verified: zero barrier ops in any dump, five in the PTX).
Ordering at this level comes from:

  * program order within a thread,
  * commit-group counting: ``ttg.async_wait {num=N}`` blocks until at most N
    commit groups are outstanding, i.e. every group except the N most recent
    ones is complete,
  * the multibuffer rotation: ``memdesc_index`` indices are loop-carried
    ``addi``/``cmpi``/``select`` chains implementing ``(k + c) mod S``.

Model boundary (documented in the plan §1/§5): threads are abstracted as
advancing through loop iterations in lockstep (the Membar-inserted barriers
bound warp drift), so the checkable contract is the RAW direction — an async
copy must be covered by the wait that guards the load reading its slot.
This catches the real mutation classes: wrong/deleted wait nums, shrunk
stage dims, rotation off-by-one, dropped commit groups. WAR (a far-ahead
copy overwriting a slot mid-read) is barrier-protected under the same
assumption and is not checked for generic-proxy reads (local_load).

sm90 adds a second async agent: ``ttng.warp_group_dot`` reads its smem
operands asynchronously and stays PENDING until a
``ttng.warp_group_dot_wait {pendings=N}`` retires it (all but the N most
recent wgmma complete) — Membar barriers do NOT retire it, so here the WAR
direction IS checkable and checked: an async copy must not overwrite a slot
while a wgmma read of it can still be pending. RAW for wgmma reads reuses
the load machinery (the read starts after the guarding async_wait).

Everything extracted here is *checked, not trusted*: rotation closed forms
are validated by exhaustive simulation of the parsed select chain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .ttgir_reader import (
    EventGraph,
    LoadEvent,
    SourceLoc,
    UnsupportedTTGIR,
)


# ───────────────────────────── slot expressions ─────────────────────────────


@dataclass(frozen=True)
class ConstSlot:
    value: int


@dataclass(frozen=True)
class RotatingSlot:
    """Slot used at loop iteration k is ``(base + k) mod modulus``."""

    base: int
    modulus: int


SlotExpr = ConstSlot | RotatingSlot


def _eval_chain(
    graph: EventGraph,
    name: str,
    env: dict[str, int],
    _stack: frozenset[str] = frozenset(),
) -> int:
    """Concretely evaluate an SSA scalar given loop-carried values in env.

    A name reappearing on the active evaluation stack means the printed SSA
    references form a cycle (malformed/adversarial input) — unsupported, not
    a RecursionError.
    """
    if name in env:
        return env[name]
    if name in graph.constants:
        return graph.constants[name]
    if name in _stack:
        raise UnsupportedTTGIR(f"cyclic scalar chain through {name}")
    d = graph.defs.get(name)
    if d is None:
        raise UnsupportedTTGIR(f"cannot evaluate {name} (unknown producer)")
    stack = _stack | {name}
    if d.kind == "addi":
        return _eval_chain(graph, d.operands[0], env, stack) + _eval_chain(
            graph, d.operands[1], env, stack
        )
    if d.kind == "subi":
        return _eval_chain(graph, d.operands[0], env, stack) - _eval_chain(
            graph, d.operands[1], env, stack
        )
    if d.kind == "xori":
        return _eval_chain(graph, d.operands[0], env, stack) ^ _eval_chain(
            graph, d.operands[1], env, stack
        )
    if d.kind == "cmpi":
        a = _eval_chain(graph, d.operands[0], env, stack)
        b = _eval_chain(graph, d.operands[1], env, stack)
        pred = d.attrs["pred"]
        table = {
            "sge": a >= b,
            "sgt": a > b,
            "sle": a <= b,
            "slt": a < b,
            "eq": a == b,
            "ne": a != b,
        }
        if pred not in table:
            raise UnsupportedTTGIR(f"cmpi predicate {pred} unsupported")
        return int(table[pred])
    if d.kind == "select":
        c = _eval_chain(graph, d.operands[0], env, stack)
        return _eval_chain(graph, d.operands[1] if c else d.operands[2], env, stack)
    raise UnsupportedTTGIR(f"cannot evaluate {name} (op {d.kind})")


def _chain_iter_arg(graph: EventGraph, name: str, seen: set[str]) -> str | None:
    """Find the single loop iter_arg the scalar chain ``name`` depends on."""
    if name in seen:
        return None
    seen.add(name)
    if graph.loop and any(arg == name for arg, _ in graph.loop.iter_args):
        return name
    if name in graph.constants:
        return None
    d = graph.defs.get(name)
    if d is None:
        return None
    found: str | None = None
    for op in d.operands:
        sub = _chain_iter_arg(graph, op, seen)
        if sub is not None:
            if found is not None and found != sub:
                raise UnsupportedTTGIR(
                    f"index chain {name} depends on multiple iter_args"
                )
            found = sub
    return found


def resolve_slot(graph: EventGraph, index_ssa: str | None) -> SlotExpr:
    """Resolve a memdesc_index operand to a slot expression.

    Constants resolve directly. Loop-carried chains are CHECKED against the
    rotation closed form ``(base + k) mod S`` by simulating the parsed
    ``addi``/``cmpi``/``select`` chain for enough iterations to cover two
    full periods; any mismatch (or an unrecognizable chain) is unsupported.
    """
    if index_ssa is None or index_ssa == "":
        return ConstSlot(0)
    if index_ssa in graph.constants:
        return ConstSlot(graph.constants[index_ssa])
    if graph.loop is None:
        raise UnsupportedTTGIR(f"non-constant slot index {index_ssa} outside a loop")

    arg = _chain_iter_arg(graph, index_ssa, set())
    if arg is None:
        raise UnsupportedTTGIR(
            f"slot index {index_ssa} is not a constant or iter_arg chain"
        )
    init_name = graph.iter_arg_init(arg)
    yielded = graph.yielded_for_arg(arg)
    if init_name is None or init_name not in graph.constants:
        raise UnsupportedTTGIR(f"iter_arg {arg} has non-constant init")
    init = graph.constants[init_name]

    # The value USED at iteration k is index_ssa evaluated with arg = its
    # k-th value; arg advances via the yielded chain. Simulate.
    # Derive the modulus from the cmpi bound in the chain (any cmpi against
    # a constant); fall back to allocation stage count at the caller.
    modulus = _find_modulus(graph, index_ssa, set())
    if modulus is None or modulus <= 0:
        raise UnsupportedTTGIR(f"cannot derive rotation modulus for {index_ssa}")
    if not _chain_constants_bounded(graph, index_ssa, 2 * modulus + 4):
        raise UnsupportedTTGIR(
            f"slot index {index_ssa} carries a constant beyond the "
            "simulation window — periodicity cannot be validated"
        )

    sim_values = []
    arg_val = init
    steps = 2 * modulus + 4
    for _k in range(steps):
        used = _eval_chain(graph, index_ssa, {arg: arg_val})
        sim_values.append(used)
        if yielded is None:
            raise UnsupportedTTGIR(f"iter_arg {arg} is never advanced by scf.yield")
        arg_val = _eval_chain(graph, yielded, {arg: arg_val})

    base = sim_values[0] % modulus
    for k, v in enumerate(sim_values):
        if v != (base + k) % modulus:
            raise UnsupportedTTGIR(
                f"slot index {index_ssa} does not follow (base + k) mod "
                f"{modulus}: simulated {sim_values}"
            )
    return RotatingSlot(base=base, modulus=modulus)


def _validate_slot(slot: SlotExpr, stages: int, alloc: str, line_no: int) -> None:
    """A resolved slot must fit the allocation's stage geometry, else the IR
    violates the model's assumptions — fail closed (unsupported) rather than
    feed an out-of-range slot to the solver and emit a proof/report computed
    under a broken buffer model. A rotating slot must wrap at the stage count;
    a constant slot must index an existing stage ``[0, stages)``.
    """
    if isinstance(slot, RotatingSlot):
        if slot.modulus != stages:
            raise UnsupportedTTGIR(
                f"line {line_no}: rotation modulus {slot.modulus} != "
                f"stage count {stages} of {alloc}"
            )
    else:  # ConstSlot
        if not 0 <= slot.value < stages:
            raise UnsupportedTTGIR(
                f"line {line_no}: constant slot {slot.value} is out of range "
                f"[0, {stages}) for {alloc}"
            )


def _chain_constants_bounded(
    graph: EventGraph, ssa: str, bound: int, _seen: set[str] | None = None
) -> bool:
    """Periodicity guard for finite-window chain validation: every
    constant reachable from ``ssa`` (cmpi bounds, add/sub/xor immediates,
    iter_arg inits along the chain) must satisfy |c| ≤ bound. A large
    constant is exactly what lets a chain match the canonical pattern
    inside the simulation window and diverge beyond it (e.g. a counter
    compared against 50, or an init of -10^6); with the parsed op set
    {addi, subi, xori, cmpi, select} and all constants in-window, any
    behavioral change crosses inside the window and the simulation
    catches it."""
    seen = _seen if _seen is not None else set()
    if ssa in seen:
        return True
    seen.add(ssa)
    if ssa in graph.constants:
        return abs(graph.constants[ssa]) <= bound
    if graph.loop is not None:
        for arg, init in graph.loop.iter_args:
            if arg == ssa:
                ok = True
                if init in graph.constants:
                    ok &= abs(graph.constants[init]) <= bound
                yielded = graph.yielded_for_arg(arg)
                if yielded is not None:
                    ok &= _chain_constants_bounded(graph, yielded, bound, seen)
                return ok
    d = graph.defs.get(ssa)
    if d is None:
        return True  # unknown producers fail later in _eval_chain
    return all(_chain_constants_bounded(graph, op, bound, seen) for op in d.operands)


def _chain_dependent_args(graph: EventGraph, ssa: str) -> set[str]:
    """Iter_args the chain transitively depends on (through defs AND the
    yield chains of the args it reaches)."""
    deps: set[str] = set()
    frontier = [ssa]
    seen: set[str] = set()
    arg_names = (
        {arg for arg, _ in graph.loop.iter_args} if graph.loop is not None else set()
    )
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in arg_names:
            if name not in deps:
                deps.add(name)
                yielded = graph.yielded_for_arg(name)
                if yielded is not None:
                    frontier.append(yielded)
            continue
        d = graph.defs.get(name)
        if d is not None:
            frontier.extend(d.operands)
    return deps


def _simulate_chain(graph: EventGraph, ssa: str, steps: int) -> list[int]:
    """Simulate a loop-carried scalar chain for ``steps`` iterations,
    advancing ALL constant-init iter_args in lockstep (a chain may depend
    on several — e.g. the mbarrier phase flips when the slot counter
    wraps). Chains touching a non-constant-init arg fail as unsupported
    via _eval_chain."""
    if graph.loop is None:
        raise UnsupportedTTGIR(f"scalar chain {ssa} simulated outside a loop")
    deps = _chain_dependent_args(graph, ssa)
    env: dict[str, int] = {}
    for arg, init in graph.loop.iter_args:
        if arg in deps and init in graph.constants:
            env[arg] = graph.constants[init]
    out: list[int] = []
    for _ in range(steps):
        out.append(_eval_chain(graph, ssa, env))
        new_env: dict[str, int] = {}
        for arg in env:
            yielded = graph.yielded_for_arg(arg)
            if yielded is None:
                raise UnsupportedTTGIR(f"iter_arg {arg} is never advanced")
            new_env[arg] = _eval_chain(graph, yielded, env)
        env = new_env
    return out


def _find_modulus(graph: EventGraph, name: str, seen: set[str]) -> int | None:
    if name in seen:
        return None
    seen.add(name)
    d = graph.defs.get(name)
    if d is None:
        return None
    if d.kind == "cmpi":
        bound = d.operands[1]
        if bound in graph.constants:
            return graph.constants[bound]
    for op in d.operands:
        sub = _find_modulus(graph, op, seen)
        if sub is not None:
            return sub
    return None


# ───────────────────────────── pipeline model ─────────────────────────────


@dataclass(frozen=True)
class ModelCopy:
    """One async copy with its slot and commit rank.

    Commit rank: prologue copies have constant ranks 1..P in program order.
    A loop-body copy at iteration k has rank ``P + g*k + pos`` where g is
    the number of commit groups per iteration and pos its 1-based position
    among the body's commits. ``rank is None`` means the copy is never
    committed — no wait can ever cover it.
    """

    alloc: str
    slot: SlotExpr
    const_rank: int | None  # for prologue copies
    loop_pos: int | None  # 1-based commit position within the loop body
    loc: SourceLoc | None
    line_no: int
    committed: bool
    # Program-order position within its segment — the WAR direction needs
    # to know whether a wgmma / dot-wait precedes the copy in the body.
    segment: str = "prologue"
    body_pos: int = 0


@dataclass(frozen=True)
class ModelLoad:
    """One local_load with its slot and the wait that guards it.

    ``wait_num is None`` means no wait guards the load (uncovered).
    ``issued_before_wait`` counts the commit groups issued in the loop body
    BEFORE the guarding wait (0 in the observed dumps: the wait leads the
    body). Total groups issued when the wait at iteration k returns is
    ``P + g*k + issued_before_wait``.
    """

    alloc: str
    slot: SlotExpr
    wait_num: int | None
    issued_before_wait: int
    loc: SourceLoc | None
    line_no: int
    # True when this "load" is really a warp_group_dot smem read joined to
    # the RAW machinery (reports should name the wgmma, not a local_load).
    via_dot: bool = False
    # ALL preceding wait_barriers in the same segment — the guards that
    # order TMA copies (commit-group counting orders cp.async ones). A
    # read may be protected by several barriers (one per input buffer);
    # a copy is covered when ANY matching guard orders it.
    barrier_guards: tuple["ModelBarrierWait", ...] = ()
    # Program-order position (TMA issued-before is positional, not ranked).
    body_pos: int = 0


@dataclass(frozen=True)
class ModelDotRead:
    """One smem operand of an async ``warp_group_dot``, with its slot and
    wgmma rank.

    Rank counts async wgmma issues: prologue dots have constant ranks
    1..D in program order; a loop-body dot at iteration k has rank
    ``D + w*k + pos`` (w async dots per iteration, pos 1-based). A
    ``warp_group_dot_wait {pendings=N}`` that has seen ``issued`` wgmma
    guarantees ranks ≤ issued - N are complete.
    """

    alloc: str
    slot: SlotExpr
    const_rank: int | None  # for prologue dots
    loop_pos: int | None  # 1-based async-dot position within the loop body
    body_pos: int
    loc: SourceLoc | None
    line_no: int


@dataclass(frozen=True)
class ModelDotWait:
    """One ``warp_group_dot_wait {pendings=N}`` with its counting context.

    ``issued_before`` counts async wgmma issued earlier in the same segment
    body; total issued when the wait at loop iteration k returns is
    ``D + w*k + issued_before`` (for a prologue wait, ``issued_before``
    alone).
    """

    pendings: int
    segment: str
    body_pos: int
    issued_before: int


@dataclass(frozen=True)
class ModelBarrierWait:
    """A validated ``wait_barrier`` guard.

    Under the validated protocol the wait at loop iteration k targets
    arming index ``(k + slot.base) div S`` of barrier slot
    ``(slot.base + k) mod S``. ``phase_valid`` records whether the parity
    chain simulation matched that arming's parity — a mismatched chain is
    a REAL coverage hole (the hardware wait returns against a phase that
    already completed), so the wait then covers nothing.
    """

    barrier_alloc: str
    slot: SlotExpr
    segment: str
    body_pos: int
    line_no: int
    phase_valid: bool
    # ONE-SHOT protocol (barrier initialized inside the loop): each
    # iteration arms a fresh phase-0 barrier and waits it in the same
    # body, so the wait covers exactly the same-or-earlier-iteration
    # copies issued before it in the body.
    one_shot: bool = False


@dataclass(frozen=True)
class ModelTmaCopy:
    """One TMA global→local copy with its data slot and its arming.

    ``arming_valid`` is False when the arming's ``barrier_expect`` byte
    count is SMALLER than the copies it covers — the phase then completes
    with bytes still in flight, so the wait orders nothing for this copy.
    (An expect LARGER than its arrivals deadlocks and is unsupported.)
    """

    alloc: str
    slot: SlotExpr
    barrier_alloc: str
    barrier_slot: SlotExpr
    segment: str
    body_pos: int
    loc: SourceLoc | None
    line_no: int
    arming_valid: bool


@dataclass(frozen=True)
class AsyncWaitBound:
    """One ``ttg.async_wait`` with everything the WAW query needs to
    anchor its counting bound at an arbitrary program point: at loop
    iteration k it guarantees commit ranks ≤ P + g*k + issued_before -
    num are complete (prologue waits: issued_before - num), for the
    allocations it actually awaits (``guarded`` None = operandless
    wait-all)."""

    segment: str
    body_pos: int
    num: int
    issued_before: int
    guarded: frozenset[str] | None


@dataclass
class PipelineModel:
    prologue_commits: int  # P
    commits_per_iter: int  # g
    copies: list[ModelCopy]
    loads: list[ModelLoad]
    generic_only: bool  # no async machinery at all
    prologue_dots: int = 0  # D — async wgmma issued in the prologue
    dots_per_iter: int = 0  # w
    dot_reads: list[ModelDotRead] = None  # type: ignore[assignment]
    dot_waits: list[ModelDotWait] = None  # type: ignore[assignment]
    tma_copies: list[ModelTmaCopy] = None  # type: ignore[assignment]
    barrier_waits: list[ModelBarrierWait] = None  # type: ignore[assignment]
    async_wait_bounds: list[AsyncWaitBound] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dot_reads is None:
            self.dot_reads = []
        if self.dot_waits is None:
            self.dot_waits = []
        if self.tma_copies is None:
            self.tma_copies = []
        if self.barrier_waits is None:
            self.barrier_waits = []
        if self.async_wait_bounds is None:
            self.async_wait_bounds = []


def _token_allocs(
    graph: EventGraph,
    token: str,
    commit_by_result: dict[str, tuple[str, ...]],
    copy_alloc_by_token: dict[str, str],
    wait_operands_by_result: dict[str, tuple[str, ...]],
    seen: set[str],
) -> set[str]:
    """Allocations that an async token transitively awaits.

    A commit-group result token awaits the allocations of its copies. A
    loop-carried iter_arg token awaits whatever its init and the value yielded
    into it await — the pipeline rotation threads a fresh commit token into the
    same arg each iteration, all of one allocation by construction. A prior
    async_wait's result token awaits whatever that wait's own operands await
    (wait-chaining). Cycles (malformed SSA) terminate via ``seen``.
    """
    if token in seen:
        return set()
    seen.add(token)
    allocs: set[str] = set()
    copies = commit_by_result.get(token)
    if copies is not None:
        for ct in copies:
            a = copy_alloc_by_token.get(ct)
            if a is not None:
                allocs.add(a)
        return allocs

    def recurse(nxt: str) -> None:
        allocs.update(
            _token_allocs(
                graph,
                nxt,
                commit_by_result,
                copy_alloc_by_token,
                wait_operands_by_result,
                seen,
            )
        )

    if graph.loop is not None and any(arg == token for arg, _ in graph.loop.iter_args):
        init = graph.iter_arg_init(token)
        if init is not None:
            recurse(init)
        yielded = graph.yielded_for_arg(token)
        if yielded is not None:
            recurse(yielded)
    chained = wait_operands_by_result.get(token)
    if chained is not None:
        for tok in chained:
            recurse(tok)
    return allocs


def _wait_guarded_allocs(
    graph: EventGraph,
    operand_tokens: tuple[str, ...],
    commit_by_result: dict[str, tuple[str, ...]],
    copy_alloc_by_token: dict[str, str],
    wait_operands_by_result: dict[str, tuple[str, ...]],
) -> set[str]:
    """Allocations a wait actually awaits, via its operand tokens."""
    allocs: set[str] = set()
    for tok in operand_tokens:
        allocs |= _token_allocs(
            graph,
            tok,
            commit_by_result,
            copy_alloc_by_token,
            wait_operands_by_result,
            set(),
        )
    return allocs


# ───────────────────────── mbarrier / TMA protocol ─────────────────────────


def _validate_barrier_waits(graph: EventGraph) -> list[ModelBarrierWait]:
    """Resolve and validate every wait_barrier.

    A loop wait's slot must rotate over the barrier stages; its phase
    chain is SIMULATED (all constant-init iter_args advanced in lockstep)
    and compared against the canonical parity ``((k + base) div S) mod 2``
    — the parity of the arming the wait targets. A recognizable-but-
    mismatched chain marks the wait ``phase_valid=False`` (it covers
    nothing: the hardware wait returns against an already-completed
    phase); an unsimulatable chain is unsupported. Waits outside the loop
    are outside the modeled protocol shapes.
    """
    out: list[ModelBarrierWait] = []
    for bw in graph.barrier_waits:
        if bw.segment != "loop":
            raise UnsupportedTTGIR(
                f"line {bw.line_no}: wait_barrier outside the pipelined loop "
                "is not modeled"
            )
        stages = graph.allocations[bw.barrier_alloc].stages
        slot = resolve_slot(graph, bw.index_ssa)
        _validate_slot(slot, stages, bw.barrier_alloc, bw.line_no)
        init = graph.barrier_init.get(bw.barrier_alloc)
        if init is None:
            raise UnsupportedTTGIR(
                f"line {bw.line_no}: wait_barrier on a never-initialized "
                "barrier — mbarrier ops on uninitialized storage are UB"
            )
        one_shot = init[0] == "loop"
        if one_shot:
            # Fresh barrier every iteration: its phase counter restarts at
            # 0, so the wait must target parity 0 constantly (a literal 0
            # or a chain that provably stays 0). Staged in-loop barriers
            # have no modeled protocol.
            if stages != 1 or not isinstance(slot, ConstSlot):
                raise UnsupportedTTGIR(
                    f"line {bw.line_no}: staged in-loop (one-shot) barrier "
                    "is not modeled"
                )
            if bw.phase_ssa in graph.constants:
                phase_valid = graph.constants[bw.phase_ssa] == 0
            else:
                try:
                    window = 8
                    phase_valid = (
                        _chain_constants_bounded(graph, bw.phase_ssa, window)
                        and _simulate_chain(graph, bw.phase_ssa, window) == [0] * window
                    )
                except UnsupportedTTGIR:
                    phase_valid = False
            out.append(
                ModelBarrierWait(
                    barrier_alloc=bw.barrier_alloc,
                    slot=RotatingSlot(base=0, modulus=1),
                    segment=bw.segment,
                    body_pos=bw.body_pos,
                    line_no=bw.line_no,
                    phase_valid=phase_valid,
                    one_shot=True,
                )
            )
            continue
        if isinstance(slot, ConstSlot):
            if stages != 1:
                raise UnsupportedTTGIR(
                    f"line {bw.line_no}: constant-slot wait_barrier over a "
                    f"{stages}-stage barrier is not modeled"
                )
            slot = RotatingSlot(base=0, modulus=1)
        base, s = slot.base, slot.modulus
        steps = 4 * s + 4
        if bw.phase_ssa in graph.constants:
            sim = [graph.constants[bw.phase_ssa]] * steps
        else:
            if not _chain_constants_bounded(graph, bw.phase_ssa, steps):
                raise UnsupportedTTGIR(
                    f"line {bw.line_no}: phase chain carries a constant "
                    "beyond the simulation window — periodicity cannot be "
                    "validated"
                )
            sim = _simulate_chain(graph, bw.phase_ssa, steps)
        expected = [((k + base) // s) % 2 for k in range(steps)]
        out.append(
            ModelBarrierWait(
                barrier_alloc=bw.barrier_alloc,
                slot=slot,
                segment=bw.segment,
                body_pos=bw.body_pos,
                line_no=bw.line_no,
                phase_valid=sim == expected,
            )
        )
    return out


def _build_tma_copies(graph: EventGraph) -> list[ModelTmaCopy]:
    """Resolve TMA copies, pair each with its arming ``barrier_expect``,
    and validate the arming: one loop expect chain per barrier allocation,
    prologue expects covering exactly the slots the loop chain first
    reaches one period late, predicate equality between an expect and its
    copies, and the expected byte count against the paired copies' sizes
    (larger ⇒ the phase never completes ⇒ deadlock, unsupported; smaller
    ⇒ early completion ⇒ the copies are uncovered, ``arming_valid=False``).
    """
    # Every protocol op must target an INITIALIZED barrier, and the init
    # must precede the first protocol event on it (one-shot: within the
    # body; persistent: within the prologue) — mbarrier ops on
    # uninitialized storage are UB, and a wait on garbage state can
    # return immediately, voiding all coverage.
    protocol_uses: dict[str, list[tuple[str, int, int]]] = {}
    for e in graph.expects:
        protocol_uses.setdefault(e.barrier_alloc, []).append(
            (e.segment, e.body_pos, e.line_no)
        )
    for ce in graph.tma_copies:
        protocol_uses.setdefault(ce.barrier_alloc, []).append(
            (ce.segment, ce.body_pos, ce.line_no)
        )
    for bw in graph.barrier_waits:
        protocol_uses.setdefault(bw.barrier_alloc, []).append(
            (bw.segment, bw.body_pos, bw.line_no)
        )
    seg_rank = {"prologue": 0, "loop": 1, "epilogue": 2}
    for bar, uses in protocol_uses.items():
        init = graph.barrier_init.get(bar)
        if init is None:
            raise UnsupportedTTGIR(
                f"line {uses[0][2]}: mbarrier protocol op on {bar}, which "
                "is never init_barrier'd — UB on uninitialized storage"
            )
        init_seg, init_pos = init
        for use_seg, use_pos, use_line in uses:
            if (seg_rank[use_seg], use_pos) < (seg_rank[init_seg], init_pos):
                raise UnsupportedTTGIR(
                    f"line {use_line}: mbarrier protocol op on {bar} before "
                    "its init_barrier — UB on uninitialized storage"
                )

    # expects resolved and grouped
    loop_expect: dict[str, Any] = {}  # barrier alloc -> (expect, RotatingSlot)
    prologue_expects: dict[tuple[str, int], Any] = {}  # (alloc, slot) -> expect
    for e in graph.expects:
        stages = graph.allocations[e.barrier_alloc].stages
        slot = resolve_slot(graph, e.index_ssa)
        _validate_slot(slot, stages, e.barrier_alloc, e.line_no)
        if e.segment == "loop":
            if isinstance(slot, ConstSlot):
                if stages != 1:
                    raise UnsupportedTTGIR(
                        f"line {e.line_no}: constant-slot loop barrier_expect "
                        f"over a {stages}-stage barrier is not modeled"
                    )
                slot = RotatingSlot(base=0, modulus=1)
            if e.barrier_alloc in loop_expect:
                raise UnsupportedTTGIR(
                    f"line {e.line_no}: multiple loop barrier_expect chains "
                    "on one barrier allocation are not modeled"
                )
            loop_expect[e.barrier_alloc] = (e, slot)
        elif e.segment == "prologue":
            if not isinstance(slot, ConstSlot):
                raise UnsupportedTTGIR(
                    f"line {e.line_no}: rotating prologue barrier_expect"
                )
            key = (e.barrier_alloc, slot.value)
            if key in prologue_expects:
                raise UnsupportedTTGIR(
                    f"line {e.line_no}: double-armed prologue barrier slot"
                )
            prologue_expects[key] = e
        else:
            raise UnsupportedTTGIR(
                f"line {e.line_no}: epilogue barrier_expect is not modeled"
            )

    # Prologue coverage: the loop chain (base b, S) reaches slots < b only
    # in its second period (arming index 1) — those slots need exactly the
    # prologue armings as index 0. Slots ≥ b get loop arming index 0.
    for bar, (e, slot) in loop_expect.items():
        assert isinstance(slot, RotatingSlot)
        want = set(range(slot.base))
        have = {s for (a, s) in prologue_expects if a == bar}
        if want != have:
            raise UnsupportedTTGIR(
                f"line {e.line_no}: prologue armings {sorted(have)} do not "
                f"cover slots {sorted(want)} of the loop arming chain"
            )

    # copies paired to armings
    copies: list[ModelTmaCopy] = []
    arming_bytes: dict[Any, list[int]] = {}
    copy_records: list[tuple[Any, Any]] = []  # (arming key, record index)
    for ce in graph.tma_copies:
        d_stages = graph.allocations[ce.alloc].stages
        d_slot = resolve_slot(graph, ce.index_ssa)
        _validate_slot(d_slot, d_stages, ce.alloc, ce.line_no)
        b_stages = graph.allocations[ce.barrier_alloc].stages
        b_slot = resolve_slot(graph, ce.barrier_index_ssa)
        _validate_slot(b_slot, b_stages, ce.barrier_alloc, ce.line_no)
        if ce.segment == "loop":
            if isinstance(b_slot, ConstSlot):
                if b_stages != 1:
                    raise UnsupportedTTGIR(
                        f"line {ce.line_no}: constant-slot loop TMA arming "
                        f"over a {b_stages}-stage barrier is not modeled"
                    )
                b_slot = RotatingSlot(base=0, modulus=1)
            pair = loop_expect.get(ce.barrier_alloc)
            if pair is None or pair[1] != b_slot:
                raise UnsupportedTTGIR(
                    f"line {ce.line_no}: loop TMA copy signals a barrier slot "
                    "chain with no matching barrier_expect (would deadlock)"
                )
            expect = pair[0]
            arming_key: Any = ("loop", ce.barrier_alloc)
        elif ce.segment == "prologue":
            if not isinstance(b_slot, ConstSlot):
                raise UnsupportedTTGIR(
                    f"line {ce.line_no}: rotating prologue TMA arming"
                )
            expect = prologue_expects.get((ce.barrier_alloc, b_slot.value))
            if expect is None:
                raise UnsupportedTTGIR(
                    f"line {ce.line_no}: prologue TMA copy signals an unarmed "
                    "barrier slot (would deadlock)"
                )
            arming_key = ("prologue", ce.barrier_alloc, b_slot.value)
        else:
            raise UnsupportedTTGIR(
                f"line {ce.line_no}: epilogue TMA global→local copy is not " "modeled"
            )
        if ce.pred_ssa != expect.pred_ssa:
            raise UnsupportedTTGIR(
                f"line {ce.line_no}: TMA copy predicate differs from its "
                "arming barrier_expect predicate — partial arrivals are not "
                "modeled"
            )
        arming_bytes.setdefault(arming_key, []).append(
            graph.allocations[ce.alloc].stage_bytes
        )
        copy_records.append(
            (
                arming_key,
                ModelTmaCopy(
                    alloc=ce.alloc,
                    slot=d_slot,
                    barrier_alloc=ce.barrier_alloc,
                    barrier_slot=b_slot,
                    segment=ce.segment,
                    body_pos=ce.body_pos,
                    loc=ce.loc,
                    line_no=ce.line_no,
                    arming_valid=True,
                ),
            )
        )

    expect_by_key: dict[Any, Any] = {
        ("loop", bar): e for bar, (e, _slot) in loop_expect.items()
    }
    expect_by_key.update(
        {("prologue", a, s): e for (a, s), e in prologue_expects.items()}
    )
    for key, e in expect_by_key.items():
        total = sum(arming_bytes.get(key, []))
        if e.bytes > total:
            raise UnsupportedTTGIR(
                f"line {e.line_no}: barrier_expect awaits {e.bytes} bytes but "
                f"its TMA copies arrive only {total} — the phase never "
                "completes (deadlock)"
            )
    for key, rec in copy_records:
        e = expect_by_key[key]
        total = sum(arming_bytes[key])
        copies.append(rec if e.bytes == total else replace(rec, arming_valid=False))
    return copies


def _pred_loop_bound_distance(graph: EventGraph, pred_ssa: str | None) -> int | None:
    """``pred == (iv < upper - d)`` → d, else None."""
    if pred_ssa is None or graph.loop is None:
        return None
    d = graph.defs.get(pred_ssa)
    if d is None or d.kind != "cmpi" or d.attrs.get("pred") != "slt":
        return None
    lhs, rhs = d.operands
    if lhs != graph.loop.induction_var:
        return None
    if rhs == graph.loop.upper:
        return 0
    x = graph.defs.get(rhs)
    if x is not None and x.kind == "subi":
        a, b = x.operands
        if a == graph.loop.upper and b in graph.constants:
            return graph.constants[b]
    return None


def _pred_prologue_min_trip(graph: EventGraph, pred_ssa: str | None) -> int | None:
    """``pred == (upper > c)`` → c, else None."""
    if pred_ssa is None or graph.loop is None:
        return None
    d = graph.defs.get(pred_ssa)
    if d is None or d.kind != "cmpi" or d.attrs.get("pred") != "sgt":
        return None
    lhs, rhs = d.operands
    if lhs != graph.loop.upper:
        return None
    if rhs in graph.constants:
        return graph.constants[rhs]
    return None


def validate_reuse_drain(graph: EventGraph, model: "PipelineModel") -> str | None:
    """Storage may be reused (epilogue local_alloc after local_dealloc)
    only when every async agent is provably drained first:

      * wgmma — an epilogue ``warp_group_dot_wait {pendings=0}`` before
        the first reuse;
      * cp.async — an epilogue ``async_wait {num=0}`` before it;
      * TMA arrivals — every loop arming is consumed by a phase-valid
        wait before the loop exits: with the arming chain base b_e, wait
        base b_w and copy predicate ``iv < upper - d``, issued armings
        stay ≤ waited armings iff d ≥ b_e - b_w; a prologue arming of
        slot s is waited iff trip > s, so its predicate ``upper > c``
        needs c ≥ s.

    Returns the failure reason, or None when the drain is proven (or no
    reuse exists). The caller abstains on a failed drain only when no
    race reports were produced — a racy pipeline is reported as racy, not
    hidden behind the reuse abstention.
    """
    reuse = [a for a in graph.allocations.values() if a.post_dealloc]
    if not reuse:
        return None
    first = min((a.body_pos for a in reuse), default=0)
    if model.dot_reads and not any(
        w.segment == "epilogue" and w.pendings == 0 and w.body_pos <= first
        for w in model.dot_waits
    ):
        return (
            "storage reuse after dealloc without an epilogue "
            "warp_group_dot_wait {pendings=0} drain"
        )
    if graph.copies and not any(
        w.segment == "epilogue" and w.num == 0 and w.body_pos <= first
        for w in graph.waits
    ):
        return (
            "storage reuse after dealloc without an epilogue "
            "async_wait {num=0} drain"
        )
    # TMA local→global stores are async smem READS: one preceding a reuse
    # alloc must be drained by an async_tma_store_wait {pendings=0} before
    # the reuse.
    seg_rank = {"prologue": 0, "loop": 1, "epilogue": 2}
    reuse_keys = [(seg_rank[a.segment], a.body_pos) for a in reuse]
    for ts in graph.tma_stores:
        t_key = (seg_rank[ts.segment], ts.body_pos)
        for r_key in reuse_keys:
            # alloc body_pos is a counter SNAPSHOT: an event with pos ≤ the
            # snapshot precedes the alloc (equality = immediately before).
            if t_key > r_key:
                continue
            if not any(
                sw.pendings == 0
                and t_key < (seg_rank[sw.segment], sw.body_pos) <= r_key
                for sw in graph.tma_store_waits
            ):
                return (
                    f"line {ts.line_no}: storage reuse after dealloc without "
                    "an async_tma_store_wait {pendings=0} draining the TMA "
                    "store first"
                )

    valid_wait_bases: dict[str, list[int]] = {}
    one_shot_wait_pos: dict[str, int] = {}
    for w in model.barrier_waits:
        if not w.phase_valid:
            continue
        if w.one_shot:
            one_shot_wait_pos[w.barrier_alloc] = w.body_pos
        elif isinstance(w.slot, RotatingSlot):
            valid_wait_bases.setdefault(w.barrier_alloc, []).append(w.slot.base)

    # The predicate arithmetic below reads the induction variable as the
    # iteration index — only valid for the canonical lower=0, step=1 loop.
    persistent_copies = [
        (ce, raw)
        for ce, raw in zip(model.tma_copies, graph.tma_copies)
        if not (ce.segment == "loop" and ce.barrier_alloc in one_shot_wait_pos)
    ]
    if persistent_copies and graph.loop is not None:
        lower_c = graph.constants.get(graph.loop.lower)
        step_c = graph.constants.get(graph.loop.step)
        if lower_c != 0 or step_c != 1:
            return (
                "storage reuse after dealloc: the TMA drain predicate "
                "arithmetic requires the canonical lower=0/step=1 loop "
                f"(found lower={lower_c}, step={step_c})"
            )

    for ce, raw in zip(model.tma_copies, graph.tma_copies):
        if ce.segment == "loop":
            os_pos = one_shot_wait_pos.get(ce.barrier_alloc)
            if os_pos is not None:
                # One-shot: the same iteration's wait forces the arrival
                # to land before the body ends — drained at loop exit.
                if ce.body_pos < os_pos:
                    continue
                return (
                    f"line {ce.line_no}: storage reuse after dealloc without "
                    "a proven TMA drain (one-shot copy issued after its "
                    "wait_barrier)"
                )
            bases = valid_wait_bases.get(ce.barrier_alloc, [])
            d = _pred_loop_bound_distance(graph, raw.pred_ssa)
            assert isinstance(ce.barrier_slot, RotatingSlot)
            if not bases or d is None or d < ce.barrier_slot.base - max(bases):
                return (
                    f"line {ce.line_no}: storage reuse after dealloc without "
                    "a proven TMA drain (loop copy predicate must stop the "
                    "prefetch at least the arming distance before the trip "
                    "end)"
                )
        else:  # prologue
            c = _pred_prologue_min_trip(graph, raw.pred_ssa)
            assert isinstance(ce.barrier_slot, ConstSlot)
            bases = valid_wait_bases.get(ce.barrier_alloc, [])
            if not bases:
                return (
                    f"line {ce.line_no}: storage reuse after dealloc with no "
                    "phase-valid wait chain covering the prologue arming"
                )
            s_p = ce.barrier_slot.value
            mod = graph.allocations[ce.barrier_alloc].stages
            first_wait_iter = min((s_p - b) % mod for b in bases)
            if c is None or c < first_wait_iter:
                return (
                    f"line {ce.line_no}: storage reuse after dealloc without "
                    "a proven TMA drain (prologue copy must be predicated on "
                    "the trip count reaching its first covering wait)"
                )
    return None


def build_pipeline_model(graph: EventGraph) -> PipelineModel:
    """Derive the counting model from the event graph.

    Raises UnsupportedTTGIR when the async structure falls outside the
    shapes the model can describe soundly.
    """
    if not graph.copies and not graph.dots and not graph.tma_copies:
        return PipelineModel(0, 0, [], [], generic_only=True)

    # mbarrier / TMA protocol (sm90 tranche 2)
    barrier_waits = _validate_barrier_waits(graph)
    tma_copies = _build_tma_copies(graph)

    # Commit ranks. Token -> commit mapping first.
    copy_to_commit: dict[str, tuple[str, int]] = {}  # copy token -> (segment, idx)
    prologue_rank = 0
    loop_pos = 0
    commit_rank_by_token: dict[str, tuple[int | None, int | None]] = {}
    for c in graph.commits:
        if c.segment == "prologue":
            prologue_rank += 1
            rank: tuple[int | None, int | None] = (prologue_rank, None)
        elif c.segment == "loop":
            loop_pos += 1
            rank = (None, loop_pos)
        else:
            # commits in the epilogue do not protect anything we model
            rank = (None, None)
        for tok in c.copy_tokens:
            commit_rank_by_token[tok] = rank
            copy_to_commit[tok] = (c.segment, c.body_pos)

    P = prologue_rank
    g = loop_pos

    copies: list[ModelCopy] = []
    for ce in graph.copies:
        slot = resolve_slot(graph, ce.index_ssa)
        _validate_slot(slot, graph.allocations[ce.alloc].stages, ce.alloc, ce.line_no)
        committed = ce.token in commit_rank_by_token
        const_rank, lpos = commit_rank_by_token.get(ce.token, (None, None))
        if ce.segment == "loop" and committed and lpos is None:
            raise UnsupportedTTGIR(
                f"line {ce.line_no}: loop copy committed outside the loop"
            )
        if ce.segment == "prologue" and committed and const_rank is None:
            raise UnsupportedTTGIR(
                f"line {ce.line_no}: prologue copy committed inside the loop"
            )
        copies.append(
            ModelCopy(
                alloc=ce.alloc,
                slot=slot,
                const_rank=const_rank,
                loop_pos=lpos,
                loc=ce.loc,
                line_no=ce.line_no,
                committed=committed,
                segment=ce.segment,
                body_pos=ce.body_pos,
            )
        )

    # ── wgmma agent (sm90): async-dot ranks and dot-wait counting ──
    # A SYNC warp_group_dot completes before it returns — its reads join the
    # RAW load machinery below but never stay pending, so it takes no rank.
    prologue_dot_rank = 0
    dot_loop_pos = 0
    dot_reads: list[ModelDotRead] = []
    dot_waits: list[ModelDotWait] = []
    for de in graph.dots:
        if not de.is_async:
            continue
        if de.segment == "prologue":
            prologue_dot_rank += 1
            drank: tuple[int | None, int | None] = (prologue_dot_rank, None)
        elif de.segment == "loop":
            dot_loop_pos += 1
            drank = (None, dot_loop_pos)
        else:
            raise UnsupportedTTGIR(
                f"line {de.line_no}: async warp_group_dot in the epilogue is "
                "outside the modeled pipeline shapes"
            )
        for alloc, idx in de.reads:
            slot = resolve_slot(graph, idx)
            _validate_slot(slot, graph.allocations[alloc].stages, alloc, de.line_no)
            dot_reads.append(
                ModelDotRead(
                    alloc=alloc,
                    slot=slot,
                    const_rank=drank[0],
                    loop_pos=drank[1],
                    body_pos=de.body_pos,
                    loc=de.loc,
                    line_no=de.line_no,
                )
            )
    D = prologue_dot_rank
    w = dot_loop_pos

    for dw in graph.dot_waits:
        issued_before = sum(
            1
            for de in graph.dots
            if de.is_async and de.segment == dw.segment and de.body_pos < dw.body_pos
        )
        dot_waits.append(
            ModelDotWait(
                pendings=dw.pendings,
                segment=dw.segment,
                body_pos=dw.body_pos,
                issued_before=issued_before,
            )
        )

    # Per-allocation coverage gate: which allocations each wait actually
    # awaits, derived from its operand tokens (not just its num count). A wait
    # that names explicit tokens but omits a load's allocation cannot order
    # that load after any copy to it, so the load is uncovered even if the num
    # count would otherwise "cover" it. On stock pipeliner IR every wait names
    # the tokens consistent with its num, so this never downgrades a real
    # proof; it closes the blind spot where a dropped/weakened wait operand
    # silently read as a proof instead of a report.
    commit_by_result: dict[str, tuple[str, ...]] = {
        c.token: c.copy_tokens for c in graph.commits if c.token
    }
    copy_alloc_by_token: dict[str, str] = {c.token: c.alloc for c in graph.copies}
    wait_operands_by_result: dict[str, tuple[str, ...]] = {
        w.result: w.operand_tokens for w in graph.waits if w.result
    }

    # Wait guarding each load: prefer the token edge; otherwise the nearest
    # preceding wait in the same segment; otherwise uncovered.
    # wgmma smem reads (sync or async) join as pseudo-loads for the RAW
    # direction: the read starts after the guarding async_wait, so the
    # counting contract is identical to a local_load's. The WAR direction
    # (the read possibly still pending when a later copy lands) is handled
    # separately via dot_reads/dot_waits.
    raw_read_events: list[tuple[LoadEvent, bool]] = [(le, False) for le in graph.loads]
    for de in graph.dots:
        for alloc, idx in de.reads:
            raw_read_events.append(
                (
                    LoadEvent(
                        alloc=alloc,
                        index_ssa=idx,
                        token=None,
                        result_layout="",
                        segment=de.segment,
                        body_pos=de.body_pos,
                        loc=de.loc,
                        line_no=de.line_no,
                    ),
                    True,
                )
            )
    wait_by_result = {w.result: w for w in graph.waits if w.result}

    async_wait_bounds: list[AsyncWaitBound] = []
    for wv in graph.waits:
        if wv.segment == "epilogue":
            continue  # runs after every modeled writer instance
        issued_before = sum(
            1
            for c in graph.commits
            if c.segment == wv.segment and c.body_pos < wv.body_pos
        )
        wb_guarded: frozenset[str] | None = None
        if wv.operand_tokens:
            wb_guarded = frozenset(
                _wait_guarded_allocs(
                    graph,
                    wv.operand_tokens,
                    commit_by_result,
                    copy_alloc_by_token,
                    wait_operands_by_result,
                )
            )
        async_wait_bounds.append(
            AsyncWaitBound(
                segment=wv.segment,
                body_pos=wv.body_pos,
                num=wv.num,
                issued_before=issued_before,
                guarded=wb_guarded,
            )
        )

    loads: list[ModelLoad] = []
    for le, via_dot in raw_read_events:
        slot = resolve_slot(graph, le.index_ssa)
        _validate_slot(slot, graph.allocations[le.alloc].stages, le.alloc, le.line_no)
        wait = None
        if le.token is not None and le.token in wait_by_result:
            wait = wait_by_result[le.token]
        else:
            candidates = [
                w
                for w in graph.waits
                if w.segment == le.segment and w.body_pos < le.body_pos
            ]
            wait = candidates[-1] if candidates else None
        if wait is not None and wait.segment != le.segment:
            raise UnsupportedTTGIR(
                f"line {le.line_no}: load guarded by a wait in another segment"
            )
        wait_num = wait.num if wait is not None else None
        if wait is not None and wait.operand_tokens:
            guarded = _wait_guarded_allocs(
                graph,
                wait.operand_tokens,
                commit_by_result,
                copy_alloc_by_token,
                wait_operands_by_result,
            )
            if le.alloc not in guarded:
                wait_num = None  # this wait does not await the load's alloc
        issued_before_wait = 0
        if wait is not None and wait.segment == "loop":
            issued_before_wait = sum(
                1
                for c in graph.commits
                if c.segment == "loop" and c.body_pos < wait.body_pos
            )
        loads.append(
            ModelLoad(
                alloc=le.alloc,
                slot=slot,
                wait_num=wait_num,
                issued_before_wait=issued_before_wait,
                loc=le.loc,
                line_no=le.line_no,
                via_dot=via_dot,
                barrier_guards=tuple(
                    w
                    for w in barrier_waits
                    if w.segment == le.segment and w.body_pos < le.body_pos
                ),
                body_pos=le.body_pos,
            )
        )

    return PipelineModel(
        prologue_commits=P,
        commits_per_iter=g,
        copies=copies,
        loads=loads,
        generic_only=False,
        prologue_dots=D,
        dots_per_iter=w,
        dot_reads=dot_reads,
        dot_waits=dot_waits,
        tma_copies=tma_copies,
        barrier_waits=barrier_waits,
        async_wait_bounds=async_wait_bounds,
    )
