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

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if self.dot_reads is None:
            self.dot_reads = []
        if self.dot_waits is None:
            self.dot_waits = []


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


def build_pipeline_model(graph: EventGraph) -> PipelineModel:
    """Derive the counting model from the event graph.

    Raises UnsupportedTTGIR when the async structure falls outside the
    shapes the model can describe soundly.
    """
    if not graph.copies and not graph.dots:
        return PipelineModel(0, 0, [], [], generic_only=True)

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
    )
