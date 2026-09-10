"""Two-copy SMT query for the cp.async pipeline model.

For every (async-copy, local_load) pair on the same allocation, ask Z3
whether there exist loop iterations ``k_copy``, ``k_load`` (with a symbolic
trip count T — no unrolling, valid for every launch of the specialization)
such that

  * both touch the same multibuffer slot,
  * the copy was issued before the load's guarding wait returned, and
  * the wait does NOT guarantee the copy's commit group has completed
    (counting semantics: ``async_wait {num=N}`` leaves at most the N most
    recent groups outstanding).

SAT means a witness execution where the load reads a slot whose async copy
the wait does not cover — a RAW race on shared memory. UNSAT over all pairs
proves the specialization has no such wait-coverage violation, within the
model boundary in hb.py / the plan.

sm90 adds the wgmma agent and with it a WAR query: for every
(async-copy, warp_group_dot smem read) pair on the same allocation, ask
whether the copy can overwrite a slot while a wgmma read of it is still
pending — i.e. no ``warp_group_dot_wait {pendings=N}`` executed before the
copy retires that wgmma under the pendings counting. The wgmma reads also
join the RAW query as pseudo-loads (they start after the guarding
async_wait, so the commit-group counting contract is a load's).

Scope of the query (deliberately): it solves over symbolic iterations, slots
and trip count, with commit-group/wait-coverage counting as the race
predicate. It does NOT encode the copy/load active masks, sub-tile byte
overlap, or per-thread/register footprint — sound here because the modeled
cp.async shape is whole-tile copy + whole-tile load per slot (same slot ⇒ full
byte overlap) and masking does not change which commit group a wait covers.
The ``byte_offset`` in a report is a representative witness byte computed from
the layout closed forms after the solve, not a solved quantity.

A side artifact per query can be exported as SMT-LIB2 — the "SMT-IR"
interchange format from the design plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from z3 import And, BoolVal, Int, Not, Or, Solver, sat

from ..data import RaceType
from .hb import (
    ConstSlot,
    ModelCopy,
    ModelDotRead,
    ModelLoad,
    ModelTmaCopy,
    PipelineModel,
    RotatingSlot,
    build_pipeline_model,
    validate_reuse_drain,
)
from .layouts import BlockedLayout, parse_shared_layout
from .ttgir_reader import EventGraph, StoreEvent, UnsupportedTTGIR, parse_ttgir


@dataclass(frozen=True)
class CompiledRaceReport:
    """A shared-memory race found by the compiled-mode analysis.

    ``race_type`` reuses the dynamic detector's :class:`RaceType` enum so
    consumers can branch on it uniformly across both modes: ``RAW`` for a
    load/wgmma read of an uncovered async copy, ``WAR`` for a copy
    overwriting a slot a wgmma read may still be pending on (sm90).
    """

    race_type: RaceType
    alloc: str
    alloc_var: str | None  # user variable name from loc, if any
    writer_loc: str
    reader_loc: str
    writer_line: int  # TTGIR line of the async copy
    reader_line: int  # TTGIR line of the local_load
    witness: dict[str, int] = field(default_factory=dict)
    message: str = ""

    def render(self) -> str:
        return self.message


def _slot_term(slot: ConstSlot | RotatingSlot, k: Any) -> Any:
    if isinstance(slot, ConstSlot):
        return slot.value
    return (slot.base + k) % slot.modulus


def _witness_byte(
    graph: EventGraph, copy: ModelCopy | ModelTmaCopy, slot_value: int
) -> int | None:
    """Representative byte the copy writes in its slot, via the layout
    closed forms (thread 0, register 0): blocked owner coords mapped through
    the swizzled shared offset. Best-effort witness enrichment."""
    try:
        alloc = graph.allocations[copy.alloc]
        shape = alloc.buffer_dims
        src_attr = graph.layouts.get("", "")
        # CopyEvent carries the src layout alias; ModelCopy doesn't — find it
        # back through any copy event on the same alloc.
        for ce in graph.copies:
            if ce.line_no == copy.line_no:
                src_attr = graph.layouts.get(ce.src_layout, "")
                break
        blocked = BlockedLayout.parse(src_attr)
        shared = parse_shared_layout(graph.layouts.get(alloc.memdesc.layout_alias, ""))
        coords = blocked.owner_coords(0, 0, shape)
        elem_off = shared.element_offset(coords, shape)
        elem_bytes = alloc.memdesc.elem_bits // 8
        return slot_value * alloc.stage_bytes + elem_off * elem_bytes
    except Exception:
        return None


@dataclass
class AnalysisResult:
    status: str  # "ok" | "unsupported"
    reports: list[CompiledRaceReport]
    unsupported_reason: str | None = None
    smtlib: list[str] = field(default_factory=list)  # one entry per SAT query


def _check_pair(
    graph: EventGraph,
    model: PipelineModel,
    copy: ModelCopy,
    load: ModelLoad,
    collect_smtlib: bool,
) -> tuple[CompiledRaceReport | None, str | None]:
    k_load = Int("k_load")
    k_copy = Int("k_copy")
    trip = Int("trip_count")

    cons = [trip >= 1, k_load >= 0, k_load < trip]

    slot_l = _slot_term(load.slot, k_load)
    if copy.const_rank is not None or copy.loop_pos is None:
        # Prologue copy (or uncommitted prologue copy): fixed slot/rank.
        slot_c = _slot_term(copy.slot, 0)
        rank_c = copy.const_rank
        issued_ok = True  # prologue copies are issued before any loop wait
    else:
        cons += [k_copy >= 0, k_copy < trip]
        slot_c = _slot_term(copy.slot, k_copy)
        rank_c = (
            model.prologue_commits + model.commits_per_iter * k_copy + copy.loop_pos
        )
        issued_ok = None  # encoded below

    cons.append(slot_c == slot_l)

    issued_at_wait = (
        model.prologue_commits
        + model.commits_per_iter * k_load
        + load.issued_before_wait
    )

    if issued_ok is None:
        # The copy must already have been issued when the load's wait runs:
        # its commit group is among those counted at the wait.
        cons.append(rank_c <= issued_at_wait)  # type: ignore[operator]

    if load.wait_num is None:
        pass  # no wait guards the load: any issued same-slot copy races
    elif not copy.committed or rank_c is None:
        pass  # uncommitted copy: no wait can ever cover it
    else:
        # NOT covered: the group is among the wait_num most recent ones.
        cons.append(rank_c > issued_at_wait - load.wait_num)  # type: ignore[operator]

    solver = Solver()
    solver.add(And(*cons))
    if solver.check() != sat:
        return None, None

    m = solver.model()

    def val(v: Any) -> int:
        r = m.eval(v, model_completion=True)
        return r.as_long()

    kl = val(k_load)
    kc = val(k_copy) if (copy.const_rank is None and copy.loop_pos is not None) else -1
    slot_value = (
        val(slot_l) if not isinstance(load.slot, ConstSlot) else load.slot.value
    )
    byte = _witness_byte(graph, copy, slot_value)

    alloc = graph.allocations[copy.alloc]
    alloc_var = alloc.loc.var_name if alloc.loc else None
    writer_loc = copy.loc.render() if copy.loc else f"ttgir:{copy.line_no}"
    reader_loc = load.loc.render() if load.loc else f"ttgir:{load.line_no}"
    witness = {
        "k_load": kl,
        "k_copy": kc,
        "slot": slot_value,
        "trip_count": val(trip),
    }
    if byte is not None:
        witness["byte_offset"] = byte
    reader_op = "ttng.warp_group_dot read" if load.via_dot else "ttg.local_load"
    msg = (
        f"shared-memory RAW race on {alloc_var or copy.alloc}: "
        f"{reader_op} at {reader_loc} (iteration {kl}) can read slot "
        f"{slot_value} while ttg.async_copy_global_to_local at {writer_loc}"
        f"{f' (iteration {kc})' if kc >= 0 else ' (prologue)'} is still "
        f"in flight — the guarding async_wait"
        f"{f' (num={load.wait_num})' if load.wait_num is not None else ''} "
        f"does not cover its commit group"
    )
    report = CompiledRaceReport(
        race_type=RaceType.RAW,
        alloc=copy.alloc,
        alloc_var=alloc_var,
        writer_loc=writer_loc,
        reader_loc=reader_loc,
        writer_line=copy.line_no,
        reader_line=load.line_no,
        witness=witness,
        message=msg,
    )
    smtlib = solver.to_smt2() if collect_smtlib else None
    return report, smtlib


def _check_war_pair(
    graph: EventGraph,
    model: PipelineModel,
    copy: ModelCopy | ModelTmaCopy,
    dot: ModelDotRead,
    collect_smtlib: bool,
    writer_op: str = "ttg.async_copy_global_to_local",
) -> tuple[CompiledRaceReport | None, str | None]:
    """WAR on the wgmma agent: can the writer (a cp.async copy, a TMA
    copy, or a re-executed in-loop local_alloc store over reused storage)
    land in a slot while a wgmma read of that slot is still pending?

    Pending means: issued before the writer in program order, and NOT
    retired by any ``warp_group_dot_wait`` executed before it — each wait
    that ran with ``issued`` wgmma seen guarantees only ranks ≤ issued -
    pendings are complete.
    """
    if copy.segment == "prologue":
        # A prologue copy precedes every wgmma issue; write-then-read is the
        # RAW query's territory.
        return None, None

    k_c = Int("k_copy")
    k_r = Int("k_dot")
    trip = Int("trip_count")
    cons = [trip >= 1, k_c >= 0, k_c < trip]

    if dot.const_rank is not None:
        # Prologue dot: fixed rank, precedes every loop copy.
        rank_r: Any = dot.const_rank
        slot_r = _slot_term(dot.slot, 0)
    else:
        cons += [k_r >= 0, k_r < trip]
        rank_r = model.prologue_dots + model.dots_per_iter * k_r + dot.loop_pos
        slot_r = _slot_term(dot.slot, k_r)
        # Issued before the copy executes.
        if dot.body_pos < copy.body_pos:
            cons.append(k_r <= k_c)
        else:
            cons.append(k_r < k_c)

    cons.append(_slot_term(copy.slot, k_c) == slot_r)

    # Not retired by any dot-wait in effect when the copy issues. Loop waits
    # before the copy in the body last executed at k_c, later ones at k_c-1;
    # epilogue waits run after every loop copy and never help.
    for dw in model.dot_waits:
        if dw.segment == "prologue":
            cons.append(rank_r > dw.issued_before - dw.pendings)
        elif dw.segment == "loop":
            k_eff = k_c if dw.body_pos < copy.body_pos else k_c - 1
            bound = (
                model.prologue_dots
                + model.dots_per_iter * k_eff
                + dw.issued_before
                - dw.pendings
            )
            cons.append(Or(k_eff < 0, rank_r > bound))

    solver = Solver()
    solver.add(And(*cons))
    if solver.check() != sat:
        return None, None

    m = solver.model()

    def val(v: Any) -> int:
        r = m.eval(v, model_completion=True)
        return r.as_long()

    kc = val(k_c)
    kr = val(k_r) if dot.const_rank is None else -1
    slot_value = (
        val(_slot_term(copy.slot, k_c))
        if not isinstance(copy.slot, ConstSlot)
        else copy.slot.value
    )
    byte = _witness_byte(graph, copy, slot_value)

    alloc = graph.allocations[copy.alloc]
    alloc_var = alloc.loc.var_name if alloc.loc else None
    writer_loc = copy.loc.render() if copy.loc else f"ttgir:{copy.line_no}"
    reader_loc = dot.loc.render() if dot.loc else f"ttgir:{dot.line_no}"
    witness = {
        "k_copy": kc,
        "k_dot": kr,
        "slot": slot_value,
        "trip_count": val(trip),
    }
    if byte is not None:
        witness["byte_offset"] = byte
    msg = (
        f"shared-memory WAR race on {alloc_var or copy.alloc}: "
        f"{writer_op} at {writer_loc} (iteration {kc}) "
        f"can overwrite slot {slot_value} while the ttng.warp_group_dot "
        f"read at {reader_loc}"
        f"{f' (iteration {kr})' if kr >= 0 else ' (prologue)'} is still "
        "pending — no warp_group_dot_wait retires it before the write"
    )
    report = CompiledRaceReport(
        race_type=RaceType.WAR,
        alloc=copy.alloc,
        alloc_var=alloc_var,
        writer_loc=writer_loc,
        reader_loc=reader_loc,
        writer_line=copy.line_no,
        reader_line=dot.line_no,
        witness=witness,
        message=msg,
    )
    smtlib = solver.to_smt2() if collect_smtlib else None
    return report, smtlib


def _check_tma_pair(
    graph: EventGraph,
    model: PipelineModel,
    copy: ModelTmaCopy,
    load: ModelLoad,
    collect_smtlib: bool,
) -> tuple[CompiledRaceReport | None, str | None]:
    """RAW on the TMA/mbarrier protocol: can the read see a slot whose TMA
    copy is not ordered by its guarding ``wait_barrier``?

    Under the validated protocol the wait at iteration k targets arming
    ``(k + b_w) div S`` of barrier slot ``(b_w + k) mod S``; the copy at
    iteration k' belongs to arming ``(k' + b_e) div S`` of slot
    ``(b_e + k') mod S``. The copy is covered iff the guard exists, is
    phase-valid, its arming is byte-exact, the barrier slots coincide and
    the copy's arming is at or before the wait's target — which, given
    slot equality, is the linear ``k' + b_e ≤ k + b_w``.
    """
    k_load = Int("k_load")
    k_copy = Int("k_copy")
    trip = Int("trip_count")
    cons = [trip >= 1, k_load >= 0, k_load < trip]

    slot_l = _slot_term(load.slot, k_load)
    if copy.segment == "prologue":
        slot_c = _slot_term(copy.slot, 0)
        bslot_c: Any = _slot_term(copy.barrier_slot, 0)
    else:
        cons += [k_copy >= 0, k_copy < trip]
        slot_c = _slot_term(copy.slot, k_copy)
        bslot_c = _slot_term(copy.barrier_slot, k_copy)
        # Issued before the read executes (program order).
        if copy.body_pos < load.body_pos:
            cons.append(k_copy <= k_load)
        else:
            cons.append(k_copy < k_load)

    cons.append(slot_c == slot_l)

    covered_terms = []
    matching_guards = [
        g
        for g in load.barrier_guards
        if g.barrier_alloc == copy.barrier_alloc and g.phase_valid and copy.arming_valid
    ]
    for guard in matching_guards:
        if guard.one_shot:
            # A one-shot copy issued before its same-body wait is forced
            # complete by that wait in ITS OWN iteration — which precedes
            # (or is) the reading iteration entirely.
            if copy.segment != "prologue" and copy.body_pos < guard.body_pos:
                covered_terms.append(BoolVal(True))
            continue
        assert isinstance(guard.slot, RotatingSlot)
        gslot = _slot_term(guard.slot, k_load)
        if copy.segment == "prologue":
            # Arming index 0: covered whenever the waited slot matches.
            covered_terms.append(gslot == bslot_c)
        else:
            assert isinstance(copy.barrier_slot, RotatingSlot)
            covered_terms.append(
                And(
                    gslot == bslot_c,
                    k_copy + copy.barrier_slot.base <= k_load + guard.slot.base,
                )
            )
    if covered_terms:
        cons.append(Not(Or(*covered_terms)))

    solver = Solver()
    solver.add(And(*cons))
    if solver.check() != sat:
        return None, None

    m = solver.model()

    def val(v: Any) -> int:
        r = m.eval(v, model_completion=True)
        return r.as_long()

    kl = val(k_load)
    kc = val(k_copy) if copy.segment != "prologue" else -1
    slot_value = (
        val(slot_l) if not isinstance(load.slot, ConstSlot) else load.slot.value
    )
    alloc = graph.allocations[copy.alloc]
    alloc_var = alloc.loc.var_name if alloc.loc else None
    writer_loc = copy.loc.render() if copy.loc else f"ttgir:{copy.line_no}"
    reader_loc = load.loc.render() if load.loc else f"ttgir:{load.line_no}"
    reader_op = "ttng.warp_group_dot read" if load.via_dot else "ttg.local_load"
    witness = {"k_load": kl, "k_copy": kc, "slot": slot_value, "trip_count": val(trip)}
    same_barrier = [
        g for g in load.barrier_guards if g.barrier_alloc == copy.barrier_alloc
    ]
    if not load.barrier_guards:
        hole = "no wait_barrier guards the read"
    elif not same_barrier:
        hole = "no guard waits on the copy's barrier"
    elif not copy.arming_valid:
        hole = "the arming's barrier_expect undercounts its arrivals"
    elif not any(g.phase_valid for g in same_barrier):
        hole = "the guard's phase chain does not match its arming parity"
    else:
        hole = "the wait's target arming does not cover the copy"
    msg = (
        f"shared-memory RAW race on {alloc_var or copy.alloc}: "
        f"{reader_op} at {reader_loc} (iteration {kl}) can read slot "
        f"{slot_value} while ttng.async_tma_copy_global_to_local at "
        f"{writer_loc}"
        f"{f' (iteration {kc})' if kc >= 0 else ' (prologue)'} is still in "
        f"flight — {hole}"
    )
    report = CompiledRaceReport(
        race_type=RaceType.RAW,
        alloc=copy.alloc,
        alloc_var=alloc_var,
        writer_loc=writer_loc,
        reader_loc=reader_loc,
        writer_line=copy.line_no,
        reader_line=load.line_no,
        witness=witness,
        message=msg,
    )
    smtlib = solver.to_smt2() if collect_smtlib else None
    return report, smtlib


def _writer_slot_terms(writer: Any, k: Any) -> Any:
    return _slot_term(writer.slot, 0 if writer.segment == "prologue" else k)


def _not_retired_terms(
    graph: EventGraph,
    model: PipelineModel,
    w1: Any,
    k1: Any,
    k2: Any,
    w2_body_pos: int,
) -> list[Any]:
    """Constraints asserting writer w1 (at iteration k1 unless prologue)
    is NOT forced complete by any synchronization executed before writer
    w2's issue point (iteration k2, position w2_body_pos).

    cp.async: every async_wait instance in effect bounds the completed
    commit ranks; TMA persistent: the latest wait instance on w1's
    barrier slot bounds the completed armings; TMA one-shot: w1's own
    same-body wait retires it for any later iteration."""
    cons: list[Any] = []
    if isinstance(w1, ModelCopy):
        if not w1.committed:
            return []  # no commit group — nothing ever retires it
        rank1 = (
            w1.const_rank
            if w1.const_rank is not None
            else model.prologue_commits + model.commits_per_iter * k1 + w1.loop_pos
        )
        for wb in model.async_wait_bounds:
            if wb.guarded is not None and w1.alloc not in wb.guarded:
                continue
            if wb.segment == "prologue":
                cons.append(rank1 > wb.issued_before - wb.num)
            else:  # loop
                k_eff = k2 if wb.body_pos < w2_body_pos else k2 - 1
                bound = (
                    model.prologue_commits
                    + model.commits_per_iter * k_eff
                    + wb.issued_before
                    - wb.num
                )
                cons.append(Or(k_eff < 0, rank1 > bound))
        return cons
    # ModelTmaCopy
    if not w1.arming_valid:
        return []  # under-armed: the wait never certifies completion
    for g in model.barrier_waits:
        if g.barrier_alloc != w1.barrier_alloc or not g.phase_valid:
            continue
        if g.one_shot:
            if w1.segment != "prologue" and w1.body_pos < g.body_pos:
                # Retired by its own iteration's wait for any later point;
                # not retired only for a same-iteration writer issued
                # before that wait.
                if w2_body_pos < g.body_pos:
                    cons.append(k2 == k1)
                else:
                    cons.append(BoolVal(False))
            continue
        assert isinstance(g.slot, RotatingSlot)
        S = g.slot.modulus
        b_w = g.slot.base
        if w1.segment == "prologue":
            assert isinstance(w1.barrier_slot, ConstSlot)
            arm_key = w1.barrier_slot.value  # a1*S + s1v with a1 = 0
            s1v: Any = w1.barrier_slot.value
        else:
            assert isinstance(w1.barrier_slot, RotatingSlot)
            arm_key = k1 + w1.barrier_slot.base
            s1v = (w1.barrier_slot.base + k1) % S
        k_eff = k2 if g.body_pos < w2_body_pos else k2 - 1
        k_star = k_eff - ((b_w + k_eff - s1v) % S)
        cons.append(Or(k_star < 0, k_star + b_w < arm_key))
    return cons


def _check_waw_pair(
    graph: EventGraph,
    model: PipelineModel,
    w1: Any,
    w2: Any,
    w1_op: str,
    w2_op: str,
    collect_smtlib: bool,
) -> tuple[CompiledRaceReport | None, str | None]:
    """WAW: can writer w2 target a slot while writer w1's write of the
    same slot is still in flight (not retired by any synchronization
    executed before w2's issue)? Whole-tile writes ⇒ same slot is full
    byte overlap."""
    if w2.segment == "prologue":
        # Both prologue: back-to-back unretired writes to one slot would
        # need same const slot; the pair (w1 prologue, w2 prologue)
        # matters only when slots collide.
        if w1.segment != "prologue":
            return None, None  # w1 in loop cannot precede a prologue w2
    k1 = Int("k_w1")
    k2 = Int("k_w2")
    trip = Int("trip_count")
    cons = [trip >= 1]

    if w1.segment != "prologue":
        cons += [k1 >= 0, k1 < trip]
    if w2.segment != "prologue":
        cons += [k2 >= 0, k2 < trip]

    # program order: w1 issued strictly before w2
    if w1.segment == "prologue" and w2.segment == "prologue":
        if not (w1.body_pos < w2.body_pos):
            return None, None
    elif w1.segment == "prologue":
        pass  # prologue precedes every loop iteration
    else:
        if w1.body_pos < w2.body_pos:
            cons.append(k1 <= k2)
        else:
            cons.append(k1 < k2)
        if w1 is w2:
            cons.append(k1 < k2)

    cons.append(_writer_slot_terms(w1, k1) == _writer_slot_terms(w2, k2))
    cons += _not_retired_terms(graph, model, w1, k1, k2, w2.body_pos)

    solver = Solver()
    solver.add(And(*cons))
    if solver.check() != sat:
        return None, None

    m = solver.model()

    def val(v: Any) -> int:
        r = m.eval(v, model_completion=True)
        return r.as_long()

    def val_term(t: Any) -> int:
        return t if isinstance(t, int) else val(t)

    kv1 = val(k1) if w1.segment != "prologue" else -1
    kv2 = val(k2) if w2.segment != "prologue" else -1
    alloc = graph.allocations[w1.alloc]
    alloc_var = alloc.loc.var_name if alloc.loc else None
    loc1 = w1.loc.render() if w1.loc else f"ttgir:{w1.line_no}"
    loc2 = w2.loc.render() if w2.loc else f"ttgir:{w2.line_no}"
    witness = {
        "k_w1": kv1,
        "k_w2": kv2,
        "slot": val_term(_writer_slot_terms(w2, k2)),
        "trip_count": val(trip),
    }
    msg = (
        f"shared-memory WAW race on {alloc_var or w1.alloc}: "
        f"{w2_op} at {loc2}"
        f"{f' (iteration {kv2})' if kv2 >= 0 else ' (prologue)'} can "
        f"overwrite slot {witness['slot']} while {w1_op} at {loc1}"
        f"{f' (iteration {kv1})' if kv1 >= 0 else ' (prologue)'} is still "
        "in flight — no synchronization retires the first write before "
        "the second"
    )
    report = CompiledRaceReport(
        race_type=RaceType.WAW,
        alloc=w1.alloc,
        alloc_var=alloc_var,
        writer_loc=loc2,
        reader_loc=loc1,
        writer_line=w2.line_no,
        reader_line=w1.line_no,
        witness=witness,
        message=msg,
    )
    smtlib = solver.to_smt2() if collect_smtlib else None
    return report, smtlib


_SEG_RANK = {"prologue": 0, "loop": 1, "epilogue": 2}


def _fence_between(
    graph: EventGraph, store: StoreEvent, read_seg: str, read_pos: int
) -> bool:
    """Does a fence execute between the store and the read on EVERY
    path? A loop fence only runs if the loop runs, so across segments it
    counts only when anchored to the loop-side endpoint's own iteration
    (after a loop store / before a loop read); for prologue→epilogue
    pairs a loop fence proves nothing at trip count 0."""
    sseg, spos = store.segment, store.body_pos
    for f in graph.fences:
        fseg, fpos = f.segment, f.body_pos
        if sseg == read_seg:
            if fseg == sseg and spos < fpos < read_pos:
                return True
        elif sseg == "prologue" and read_seg == "loop":
            if (fseg == "prologue" and fpos > spos) or (
                fseg == "loop" and fpos < read_pos
            ):
                return True
        elif sseg == "prologue" and read_seg == "epilogue":
            if (fseg == "prologue" and fpos > spos) or (
                fseg == "epilogue" and fpos < read_pos
            ):
                return True
        elif sseg == "loop" and read_seg == "epilogue":
            if (fseg == "loop" and fpos > spos) or (
                fseg == "epilogue" and fpos < read_pos
            ):
                return True
    return False


def _missing_fence_report(
    graph: EventGraph,
    store: StoreEvent,
    reader_op: str,
    read_loc: Any,
    read_line: int,
) -> CompiledRaceReport:
    alloc = graph.allocations[store.alloc]
    alloc_var = alloc.loc.var_name if alloc.loc else None
    writer_loc = store.loc.render() if store.loc else f"ttgir:{store.line_no}"
    reader_loc = read_loc.render() if read_loc else f"ttgir:{read_line}"
    return CompiledRaceReport(
        race_type=RaceType.RAW,
        alloc=store.alloc,
        alloc_var=alloc_var,
        writer_loc=writer_loc,
        reader_loc=reader_loc,
        writer_line=store.line_no,
        reader_line=read_line,
        witness={},
        message=(
            f"shared-memory RAW race on {alloc_var or store.alloc}: the "
            f"generic-proxy store at {writer_loc} is not ordered before the "
            f"async-proxy {reader_op} at {reader_loc} — no "
            "ttng.fence_async_shared between them"
        ),
    )


def analyze_graph(graph: EventGraph, collect_smtlib: bool = False) -> AnalysisResult:
    # ── structural gates independent of the pipeline model ──
    # mbarrier storage is a sync object: any DATA access on it is outside
    # the model.
    data_touches = (
        [(le.alloc, le.line_no) for le in graph.loads]
        + [(st.alloc, st.line_no) for st in graph.stores]
        + [(ce.alloc, ce.line_no) for ce in graph.copies]
        + [(ce.alloc, ce.line_no) for ce in graph.tma_copies]
        + [(ts.alloc, ts.line_no) for ts in graph.tma_stores]
        + [(alloc, de.line_no) for de in graph.dots for alloc, _idx in de.reads]
    )
    for alloc_name, line_no in data_touches:
        if alloc_name in graph.barrier_allocs:
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {line_no}: data access on an mbarrier " "allocation"
                ),
            )

    for ts in graph.tma_stores:
        if ts.segment == "loop":
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {ts.line_no}: TMA local→global store inside "
                    "the pipelined loop is not modeled (store-wait counting)"
                ),
            )
        if graph.allocations[ts.alloc].memdesc.mutable:
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {ts.line_no}: TMA store from a mutable "
                    "allocation is not modeled"
                ),
            )

    # Generic stores meeting async machinery. A MUTABLE generically-stored
    # allocation with async readers/writers is outside the model. An
    # IMMUTABLE one (the single-assignment local_alloc-with-operand — IR
    # typing permits no other writes) read by an async-proxy consumer
    # (wgmma / TMA store) is ordered by an intervening fence_async_shared;
    # a missing fence is the real stale-read bug the op exists for.
    async_write_allocs = {c.alloc for c in graph.copies} | {
        c.alloc for c in graph.tma_copies
    }
    dot_read_allocs = {alloc for de in graph.dots for alloc, _idx in de.reads}
    tma_read_allocs = {ts.alloc for ts in graph.tma_stores}
    fence_reports: list[CompiledRaceReport] = []
    for st in graph.stores:
        if st.alloc in async_write_allocs:
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {st.line_no}: generic store to an "
                    "async-copied allocation is outside the v1 pipeline model"
                ),
            )
        if st.alloc not in dot_read_allocs and st.alloc not in tma_read_allocs:
            continue
        if graph.allocations[st.alloc].memdesc.mutable:
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {st.line_no}: generic store to a mutable "
                    "async-read allocation crosses the generic→async proxy "
                    "boundary, which the model does not order"
                ),
            )
        for de in graph.dots:
            if any(a == st.alloc for a, _ in de.reads) and not _fence_between(
                graph, st, de.segment, de.body_pos
            ):
                fence_reports.append(
                    _missing_fence_report(
                        graph, st, "warp_group_dot read", de.loc, de.line_no
                    )
                )
        for ts in graph.tma_stores:
            if ts.alloc == st.alloc and not _fence_between(
                graph, st, ts.segment, ts.body_pos
            ):
                fence_reports.append(
                    _missing_fence_report(
                        graph,
                        st,
                        "TMA local→global copy",
                        ts.loc,
                        ts.line_no,
                    )
                )

    model = build_pipeline_model(graph)
    if model.generic_only:
        # Generic-proxy-only smem use (e.g. num_stages=1 local_alloc +
        # local_load): ordering is inserted by the backend Membar pass —
        # nothing for the async model to check beyond the fence gate above.
        status = "ok"
        return AnalysisResult(status=status, reports=fence_reports)

    for ce_any in list(graph.copies) + list(graph.tma_copies):
        if ce_any.segment == "epilogue":
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {ce_any.line_no}: epilogue async copy is "
                    "outside the modeled pipeline shapes"
                ),
            )

    reports: list[CompiledRaceReport] = list(fence_reports)
    smtlib: list[str] = []

    # WAR writers: cp.async copies, TMA copies, and the re-executed
    # in-loop single-assignment stores (their storage is reused across
    # iterations) on allocations with async wgmma readers.
    war_writers: list[tuple[Any, str]] = [
        (c, "ttg.async_copy_global_to_local") for c in model.copies
    ]
    war_writers += [
        (c, "ttng.async_tma_copy_global_to_local") for c in model.tma_copies
    ]
    for st in graph.stores:
        if st.segment == "loop" and st.alloc in dot_read_allocs:
            war_writers.append(
                (
                    ModelCopy(
                        alloc=st.alloc,
                        slot=ConstSlot(0),
                        const_rank=None,
                        loop_pos=None,
                        loc=st.loc,
                        line_no=st.line_no,
                        committed=False,
                        segment=st.segment,
                        body_pos=st.body_pos,
                    ),
                    "ttg.local_alloc (re-executed store)",
                )
            )

    for copy in model.copies:
        for load in model.loads:
            if copy.alloc != load.alloc:
                continue
            report, smt = _check_pair(graph, model, copy, load, collect_smtlib)
            if report is not None:
                reports.append(report)
                if smt:
                    smtlib.append(smt)
    for tcopy in model.tma_copies:
        for load in model.loads:
            if tcopy.alloc != load.alloc:
                continue
            report, smt = _check_tma_pair(graph, model, tcopy, load, collect_smtlib)
            if report is not None:
                reports.append(report)
                if smt:
                    smtlib.append(smt)
    # WAW: ordered pairs of async writers on one allocation (incl. the
    # same event at two iterations). Generic stores are Membar-ordered
    # among themselves and gated against async writers, so async×async
    # is the complete unordered-writer surface.
    async_writers: list[tuple[Any, str]] = [
        (c, "ttg.async_copy_global_to_local") for c in model.copies
    ] + [(c, "ttng.async_tma_copy_global_to_local") for c in model.tma_copies]
    for w1, w1_op in async_writers:
        for w2, w2_op in async_writers:
            if w1.alloc != w2.alloc:
                continue
            report, smt = _check_waw_pair(
                graph, model, w1, w2, w1_op, w2_op, collect_smtlib
            )
            if report is not None:
                reports.append(report)
                if smt:
                    smtlib.append(smt)

    for writer, writer_op in war_writers:
        for dot in model.dot_reads:
            if writer.alloc != dot.alloc:
                continue
            report, smt = _check_war_pair(
                graph, model, writer, dot, collect_smtlib, writer_op=writer_op
            )
            if report is not None:
                reports.append(report)
                if smt:
                    smtlib.append(smt)

    # Storage reuse (epilogue alloc after dealloc) demands a proven drain
    # of every async agent before a PROOF can be claimed. A racy pipeline
    # is reported as racy either way — the reuse abstention must not hide
    # the race verdict.
    if not reports:
        drain_reason = validate_reuse_drain(graph, model)
        if drain_reason is not None:
            return AnalysisResult(
                status="unsupported", reports=[], unsupported_reason=drain_reason
            )
    return AnalysisResult(status="ok", reports=reports, smtlib=smtlib)


def analyze_ttgir(text: str, collect_smtlib: bool = False) -> AnalysisResult:
    """Top-level entry: TTGIR text → analysis result.

    Unsupported constructs yield ``status="unsupported"`` with a reason —
    never a silent wrong verdict (same contract as the dynamic mode).
    """
    try:
        graph = parse_ttgir(text)
        if not graph.kernel_name:
            # No tt.func parsed at all: the input is not TTGIR (empty
            # string, PTX, ...). "ok" here would read as a proof.
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason="input contains no tt.func (not TTGIR?)",
            )
        return analyze_graph(graph, collect_smtlib=collect_smtlib)
    except UnsupportedTTGIR as exc:
        return AnalysisResult(
            status="unsupported", reports=[], unsupported_reason=str(exc)
        )
    except RecursionError:
        # Defense in depth: pathological SSA shapes must degrade to an
        # honest unsupported, never crash the host run.
        return AnalysisResult(
            status="unsupported",
            reports=[],
            unsupported_reason="pathological SSA structure (recursion limit)",
        )
