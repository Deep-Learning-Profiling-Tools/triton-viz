"""C2 (witness replay) and C3 (differential cross-check) — plan §I.4.

Both channels share one primitive: run the kernel under the INTERPRETER
with a :class:`FootprintRecorder` client that logs the concrete byte
addresses each selected grid block touches. The interpreter executes real
load semantics, so data-dependent masks — exactly what the static encoding
over-approximates — evaluate concretely here.

C2 — ``replay_witness``: re-run only the two witness program ids of a SAT
report on CLONED launch tensors and intersect their footprints. A
confirmed overlap upgrades the report to ``race-confirmed``; no overlap
demotes it to ``race-unconfirmed`` (a potential over-approximation
artifact, reported as *potential*, never definite). This is the soundness
patch for widened records (dropped masks), not an optional extra.

C3 — ``cross_check``: for chosen pids, compare the interpreter footprint
against the static model's concrete enumeration (``differential``). The
two sides share nothing but the kernel text — a mismatch exposes either a
compiler lowering the TTIR reader misread or an interpreter semantics
deviation.

Model notes:
  * Replay happens under the interpreter INSIDE a process that also does
    real compiles (the warmup-only client). The trace machinery
    context-manages its patches, but triton's interpreter patches
    ``tl.core.tensor`` dunders in place (see trace.py's warmup-only
    comment); the compiled client therefore replays ONLY when there is a
    SAT report to classify — proofs never engage the interpreter.
  * Tensors must be CLONES of the pre-launch state: finalize runs after
    the real kernel already mutated the originals.
  * Footprint granularity is element-start byte addresses per
    (tensor data_ptr, kind); mutual atomicity is honored at intersection
    time (rmw∩rmw at the same addresses is not a conflict).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ....core.callbacks import ForLoopCallbacks, OpCallbacks
from ....core.client import Client
from ....core.data import (
    AtomicCas,
    AtomicRMW,
    Load,
    Op,
    RawLoad,
    RawStore,
    Store,
)

# footprint kinds
_READS = ("load",)
_WRITES = ("store",)
_RMW = ("atomic_rmw", "atomic_cas")


class FootprintRecorder(Client):
    """Interpreter client recording per-block concrete byte footprints.

    ``target_pids=None`` records every block; otherwise only the given
    pids execute (other blocks are skipped via ``pre_run_callback``,
    which is safe because this client runs in its OWN trace).
    """

    NAME = "footprint_recorder"

    def __init__(self, target_pids: set[tuple[int, int, int]] | None = None) -> None:
        super().__init__()
        self.target_pids = target_pids
        self._current_pid: tuple[int, int, int] = (0, 0, 0)
        self._active = True
        # pid -> (base data_ptr, kind) -> set of byte addresses
        self.footprints: dict[
            tuple[int, int, int], dict[tuple[int, str], set[int]]
        ] = {}
        # tensor bases seen via arg_callback, sorted for base resolution
        self._bases: list[int] = []

    # ── lifecycle ────────────────────────────────────────────────────
    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        if hasattr(arg, "data_ptr"):
            self._bases.append(int(arg.data_ptr()))
            self._bases.sort()

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        pass

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        pid = tuple(grid_idx) + (0,) * (3 - len(grid_idx))
        self._current_pid = pid  # type: ignore[assignment]
        self._active = self.target_pids is None or pid in self.target_pids

    def pre_run_callback(self, fn: Any) -> bool:
        return self._active

    def post_run_callback(self, fn: Any) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn: Any, *args: Any, **kwargs: Any) -> bool:
        return False  # interpreter only; no real compile needed

    def post_warmup_callback(self, jit_fn: Any, ret: Any) -> None:
        pass

    def finalize(self) -> list:
        return []

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks()

    # ── recording ────────────────────────────────────────────────────
    def _base_of(self, addr: int) -> int:
        """Map an address to the greatest captured tensor base <= addr —
        the same convention as the Tracer's _get_tensor."""
        base = self._bases[0] if self._bases else 0
        for b in self._bases:
            if b > addr:
                break
            base = b
        return base

    def _record(self, kind: str, addrs: np.ndarray, mask: np.ndarray | None) -> None:
        if not self._active:
            return
        flat = np.asarray(addrs).reshape(-1)
        if mask is not None:
            m = np.broadcast_to(np.asarray(mask), np.asarray(addrs).shape).reshape(-1)
            flat = flat[m.astype(bool)]
        if flat.size == 0:
            return
        per_pid = self.footprints.setdefault(self._current_pid, {})
        base = self._base_of(int(flat[0]))
        per_pid.setdefault((base, kind), set()).update(int(a) for a in flat)

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def pre_load(ptr, mask, keys):
            if keys is None:  # triton path: ptr.data = absolute addresses
                self._record("load", ptr.data, mask.data if mask is not None else None)

        def pre_store(ptr, mask, keys):
            if keys is None:
                self._record("store", ptr.data, mask.data if mask is not None else None)

        def pre_raw_load(ptr):
            self._record("load", ptr.data, None)

        def pre_raw_store(ptr, value):
            self._record("store", ptr.data, None)

        def pre_atomic_rmw(rmw_op, ptr, val, mask, sem=None, scope=None, *a, **k):
            m = getattr(mask, "data", mask) if mask is not None else None
            self._record("atomic_rmw", ptr.data, m)

        def pre_atomic_cas(ptr, cmp, val, sem=None, scope=None, *a, **k):
            self._record("atomic_cas", ptr.data, None)

        table = {
            Load: OpCallbacks(before_callback=pre_load),
            Store: OpCallbacks(before_callback=pre_store),
            RawLoad: OpCallbacks(before_callback=pre_raw_load),
            RawStore: OpCallbacks(before_callback=pre_raw_store),
            AtomicRMW: OpCallbacks(before_callback=pre_atomic_rmw),
            AtomicCas: OpCallbacks(before_callback=pre_atomic_cas),
        }
        return table.get(op_type, OpCallbacks())


# ─────────────────────── the replay primitive ───────────────────────


@dataclass
class ReplayResult:
    footprints: dict[tuple[int, int, int], dict[tuple[int, str], set[int]]]
    # original tensor data_ptr -> clone data_ptr (footprints use CLONE bases)
    base_map: dict[int, int] = field(default_factory=dict)
    error: str | None = None


def run_replay(
    jit_fn: Any,
    args: tuple,
    kwargs: dict,
    grid: tuple[int, ...],
    target_pids: set[tuple[int, int, int]] | None,
) -> ReplayResult:
    """Run ``jit_fn`` under the interpreter, executing only ``target_pids``,
    with every tensor argument CLONED (originals are never touched).

    Returns clone-based footprints plus the original→clone base mapping.
    Never raises: replay is a best-effort classifier; on failure the caller
    keeps the unconfirmed classification.
    """
    # NOTE: `from ....core import trace` resolves to the trace() FUNCTION
    # (the package re-exports shadow the submodule); import the module.
    import importlib

    trace_mod = importlib.import_module("triton_viz.core.trace")

    recorder = FootprintRecorder(target_pids)
    base_map: dict[int, int] = {}
    try:
        cloned_args = []
        for a in args:
            if hasattr(a, "data_ptr") and hasattr(a, "clone"):
                c = a.detach().clone()
                base_map[int(a.data_ptr())] = int(c.data_ptr())
                cloned_args.append(c)
            else:
                cloned_args.append(a)
        cloned_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "data_ptr") and hasattr(v, "clone"):
                c = v.detach().clone()
                base_map[int(v.data_ptr())] = int(c.data_ptr())
                cloned_kwargs[k] = c
            else:
                cloned_kwargs[k] = v

        traced = trace_mod.TritonTrace(jit_fn, recorder)
        n_before = len(trace_mod.launches)
        try:
            traced[grid](*cloned_args, **cloned_kwargs)
        finally:
            # The replay is internal bookkeeping, not a user launch.
            del trace_mod.launches[n_before:]
        return ReplayResult(footprints=recorder.footprints, base_map=base_map)
    except Exception as e:  # noqa: BLE001
        return ReplayResult(
            footprints={}, base_map=base_map, error=f"{type(e).__name__}: {e}"
        )


# ─────────────────────── C2: witness confirmation ───────────────────────


def _kinds_conflict(kind_a: str, kind_b: str) -> bool:
    """write∩(read|write|rmw) or rmw∩(read|write) conflict. rmw∩rmw pairs
    never reach this check — whether two atomics conflict depends on
    scope/width, which a footprint cannot express, so confirm_witness
    classifies them unavailable up front."""
    a_writes = kind_a in _WRITES or kind_a in _RMW
    b_writes = kind_b in _WRITES or kind_b in _RMW
    return a_writes or b_writes


def _focused_overlap(
    fp_a: dict[tuple[int, str], set[int]],
    fp_b: dict[tuple[int, str], set[int]],
    focus_a: tuple[int, str],
    focus_b: tuple[int, str],
) -> bool:
    """Conflicting overlap RESTRICTED to the report's own access pair
    (either direction). A whole-block intersection is not sound for
    classification: two blocks racing on tensor X would 'confirm' an
    unrelated widened report on tensor Y whose accesses never execute."""
    for fa, fb in ((focus_a, focus_b), (focus_b, focus_a)):
        if fa[0] != fb[0]:
            continue  # distinct clone bases cannot overlap
        if not _kinds_conflict(fa[1], fb[1]):
            continue
        if fp_a.get(fa, set()) & fp_b.get(fb, set()):
            return True
    return False


# Replaying a launch grid with more blocks than this is declined (skipped
# blocks still cost the grid-loop iteration, ~µs each).
REPLAY_MAX_BLOCKS = 1_000_000


def _concrete_grid(launch_grid: Any) -> tuple[int, int, int] | None:
    """The captured launch grid as a concrete 3-tuple, or None (callable
    grids / missing capture cannot parameterize a faithful replay)."""
    if not isinstance(launch_grid, (tuple, list)) or not launch_grid:
        return None
    try:
        dims = [int(d) for d in launch_grid]
    except Exception:  # noqa: BLE001
        return None
    dims += [1] * (3 - len(dims))
    return (dims[0], dims[1], dims[2])


def confirm_witness(
    jit_fn: Any,
    args: tuple,
    kwargs: dict,
    pid_a: tuple[int, int, int],
    pid_b: tuple[int, int, int],
    launch_grid: Any,
    focus_a: tuple[int, str] | None = None,
    focus_b: tuple[int, str] | None = None,
) -> tuple[str, str | None]:
    """C2: replay the two witness blocks concretely and classify the
    report. The replay runs under the REAL launch grid — a synthetic
    max(pid)+1 grid changes the meaning of every grid-observing construct
    (``tl.num_programs`` in a dropped mask flips its value and fabricates
    confirmations) — so witness pids outside the launch grid, a
    non-concrete grid, or an oversized grid classify as unavailable.
    ``focus_x`` = (SNAPSHOT tensor base, kind bucket) of the report's two
    accesses; the overlap check is restricted to that pair. Also
    unavailable: missing foci, rmw∩rmw pairs (scope/width live outside the
    footprint), and intra-instance reports (same pid twice — duplicate
    lanes collapse in an address SET). Returns ``("confirmed", None)``,
    ``("unconfirmed", why)``, or ``("unavailable", why)``."""
    if pid_a == pid_b:
        return (
            "unavailable",
            "intra-instance reports are not classifiable at footprint " "granularity",
        )
    if focus_a is None or focus_b is None:
        return ("unavailable", "report accesses could not be resolved to tensors")
    if focus_a[1] in _RMW and focus_b[1] in _RMW:
        return (
            "unavailable",
            "atomic-atomic conflicts depend on scope/width, which footprints "
            "cannot express",
        )
    grid = _concrete_grid(launch_grid)
    if grid is None:
        return ("unavailable", "the launch grid is not concretely known")
    if any(p >= g or p < 0 for pid in (pid_a, pid_b) for p, g in zip(pid, grid)):
        return (
            "unavailable",
            "the witness blocks do not exist on this launch's grid",
        )
    if grid[0] * grid[1] * grid[2] > REPLAY_MAX_BLOCKS:
        return ("unavailable", "launch grid too large to replay")
    result = run_replay(jit_fn, args, kwargs, grid, {pid_a, pid_b})
    if result.error is not None:
        return ("unavailable", f"replay failed: {result.error}")
    clone_a = result.base_map.get(focus_a[0])
    clone_b = result.base_map.get(focus_b[0])
    if clone_a is None or clone_b is None:
        return ("unavailable", "witness tensors were not cloned for replay")
    fp_a = result.footprints.get(pid_a, {})
    fp_b = result.footprints.get(pid_b, {})
    if _focused_overlap(fp_a, fp_b, (clone_a, focus_a[1]), (clone_b, focus_b[1])):
        return ("confirmed", None)
    return (
        "unconfirmed",
        "the witness accesses' concrete footprints do not conflict on this "
        "launch's data (likely an over-approximation artifact)",
    )


# ─────────────────────── C3: differential cross-check ───────────────────────


def cross_check(
    graph: Any,
    params: dict[str, int],
    tensors: dict[str, Any],  # name -> GlobalTensor (T1 capture)
    jit_fn: Any,
    args: tuple,
    kwargs: dict,
    pids: list[tuple[int, int, int]],
    grid: tuple[int, ...],
) -> list[str]:
    """C3: for each pid, enumerate the STATIC model's concrete footprint
    and diff it against the INTERPRETER's. Returns human-readable
    mismatches (empty = the lowering and the interpreter agree). Static
    over-approximated accesses are excluded from both sides' comparison
    scope (they have no exact static footprint)."""
    from .differential import KIND_BUCKET, diff_footprints, static_footprints

    result = run_replay(jit_fn, args, kwargs, grid, set(pids))
    if result.error is not None:
        return [f"replay failed: {result.error}"]

    bases = {name: (meta.data_ptr, meta.elem_size) for name, meta in tensors.items()}
    base_to_name = {meta.data_ptr: name for name, meta in tensors.items()}

    issues: list[str] = []
    for pid in pids:
        static = static_footprints(graph, params, bases, pid)
        # rebase the interpreter footprint from clone bases to names
        dyn: dict[tuple[str, str], set[int]] = {}
        clone_to_orig = {c: o for o, c in result.base_map.items()}
        for (clone_base, kind), addrs in result.footprints.get(pid, {}).items():
            orig_base = clone_to_orig.get(clone_base)
            if orig_base is None or orig_base not in base_to_name:
                issues.append(f"pid {pid}: unknown tensor base {clone_base:#x}")
                continue
            name = base_to_name[orig_base]
            delta = orig_base - clone_base
            dyn.setdefault((name, kind), set()).update(a + delta for a in addrs)
        # Drop buckets containing ANY over-approximated access from BOTH
        # sides: the static side has no exact footprint for the widened
        # access, but its exact SIBLINGS in the same (tensor, kind) bucket
        # are still enumerated — a one-sided deletion fabricates a
        # static-only divergence for the common exact+widened-same-tensor
        # pattern. Symmetric exclusion means these buckets are simply
        # outside the diff's scope (reported via `skipped`).
        skipped_kinds = {
            (a.base_param, KIND_BUCKET[a.kind])
            for a in graph.accesses
            if a.mask_dropped or a.guarded
        }
        for key in list(dyn):
            if key in skipped_kinds:
                del dyn[key]
        for key in list(static.footprints):
            if key in skipped_kinds:
                del static.footprints[key]
        for m in diff_footprints(static.footprints, dyn):
            issues.append(f"pid {pid}: {m}")
    return issues
