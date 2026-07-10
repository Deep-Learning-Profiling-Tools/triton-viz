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

from z3 import And, Int, Or, Solver, sat

from ..data import RaceType
from .hb import (
    ConstSlot,
    ModelCopy,
    ModelDotRead,
    ModelLoad,
    PipelineModel,
    RotatingSlot,
    build_pipeline_model,
)
from .layouts import BlockedLayout, parse_shared_layout
from .ttgir_reader import EventGraph, UnsupportedTTGIR, parse_ttgir


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


def _witness_byte(graph: EventGraph, copy: ModelCopy, slot_value: int) -> int | None:
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
    copy: ModelCopy,
    dot: ModelDotRead,
    collect_smtlib: bool,
) -> tuple[CompiledRaceReport | None, str | None]:
    """WAR on the wgmma agent: can the copy (writer) land in a slot while
    the wgmma read of that slot is still pending?

    Pending means: issued before the copy in program order, and NOT retired
    by any ``warp_group_dot_wait`` executed before the copy — each wait that
    ran with ``issued`` wgmma seen guarantees only ranks ≤ issued - pendings
    are complete.
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
        f"ttg.async_copy_global_to_local at {writer_loc} (iteration {kc}) "
        f"can overwrite slot {slot_value} while the ttng.warp_group_dot "
        f"read at {reader_loc}"
        f"{f' (iteration {kr})' if kr >= 0 else ' (prologue)'} is still "
        "pending — no warp_group_dot_wait retires it before the copy"
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


def analyze_graph(graph: EventGraph, collect_smtlib: bool = False) -> AnalysisResult:
    model = build_pipeline_model(graph)
    if model.generic_only:
        # Generic-proxy-only smem use (e.g. num_stages=1 local_alloc +
        # local_load): ordering is inserted by the backend Membar pass —
        # nothing for the v1 async model to check (plan §1 non-goals).
        return AnalysisResult(status="ok", reports=[])

    # Async machinery present. Generic stores on an alloc that also has
    # async copies fall outside the observed pipeline shapes; likewise a
    # generic store feeding a wgmma read crosses the generic→async proxy
    # boundary (fence_async_shared territory), which the model does not
    # order.
    async_allocs = {c.alloc for c in model.copies}
    dot_read_allocs = {alloc for de in graph.dots for alloc, _idx in de.reads}
    for st in graph.stores:
        if st.alloc in async_allocs:
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {st.line_no}: generic store to an "
                    "async-copied allocation is outside the v1 pipeline model"
                ),
            )
        if st.alloc in dot_read_allocs:
            return AnalysisResult(
                status="unsupported",
                reports=[],
                unsupported_reason=(
                    f"ttgir line {st.line_no}: generic store to a "
                    "warp_group_dot-read allocation crosses the generic→async "
                    "proxy boundary, which the model does not order"
                ),
            )

    reports: list[CompiledRaceReport] = []
    smtlib: list[str] = []
    for copy in model.copies:
        for load in model.loads:
            if copy.alloc != load.alloc:
                continue
            report, smt = _check_pair(graph, model, copy, load, collect_smtlib)
            if report is not None:
                reports.append(report)
                if smt:
                    smtlib.append(smt)
        for dot in model.dot_reads:
            if copy.alloc != dot.alloc:
                continue
            report, smt = _check_war_pair(graph, model, copy, dot, collect_smtlib)
            if report is not None:
                reports.append(report)
                if smt:
                    smtlib.append(smt)
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
