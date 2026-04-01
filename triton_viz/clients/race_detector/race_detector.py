from collections.abc import Callable
from typing import Any, Optional
from collections import defaultdict

import numpy as np
from z3 import (
    Solver,
    Int,
    And,
    Or,
    sat,
    substitute,
    is_bool,
    is_true,
)
from z3.z3 import ArithRef

from ...core.callbacks import OpCallbacks
from ...core.config import config as cfg
from ...core.data import (
    Op,
    RawLoad,
    Load,
    RawStore,
    Store,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
    AtomicCas,
    AtomicRMW,
)
from ..symbolic_engine import (
    SymbolicExpr,
    SymbolicClient,
)
from .data import AccessType, MemoryAccess, SymbolicMemoryAccess, RaceRecord, RaceType
from ...utils.traceback_utils import extract_user_frames


def _flatten_offsets(ptr) -> np.ndarray:
    return np.asarray(ptr.data).flatten().astype(np.int64)


def _flatten_mask(mask, n: int) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    return np.asarray(mask.data).flatten().astype(bool)


def _is_inter_block_scope(scope: str | None) -> bool:
    if scope is None:
        return True
    return "cta" not in scope.lower()


def _get_elem_size(handle) -> int:
    """Extract element byte size from a tensor handle or SymbolicExpr."""
    try:
        dtype = handle.dtype
        while hasattr(dtype, "element_ty"):
            dtype = dtype.element_ty
        return max(1, dtype.primitive_bitwidth // 8)
    except (AttributeError, TypeError):
        return 1


class RaceDetector(SymbolicClient):
    NAME = "race_detector"

    def __init__(self):
        super().__init__()
        self._symbolic_accesses: list[SymbolicMemoryAccess] = []
        self._concrete_accesses: list[MemoryAccess] = []
        self._need_concrete_fallback: bool = False
        self._force_concrete: bool = False
        self._grid: tuple[int, ...] = (1, 1, 1)
        self._phase: str = "init"
        # Each entry: (OpCallbacks, original_overrider_fn)
        self._op_callbacks: list[tuple[OpCallbacks, Callable]] = []
        self._races: list[RaceRecord] = []
        self._symbolic_cas_counts: dict[tuple[str, ...], int] = {}
        self._symbolic_barrier_candidates: dict[
            tuple[str, ...], set[str]
        ] = defaultdict(set)
        self._symbolic_race_results: list[RaceRecord] | None = None
        self._event_counter: int = 0

    # ── Client interface ─────────────────────────────────────────

    def pre_run_callback(self, fn: Callable) -> bool:
        if self._phase == "z3_done":
            return False
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        if self._phase != "symbolic":
            return True
        # End of symbolic phase (block 0 finished or bailed out)
        if self._need_concrete_fallback:
            self._symbolic_accesses.clear()
            self._phase = "concrete"
            self._force_concrete = True
            self._disable_overriders()
            return True
        else:
            self._races = self._check_symbolic_races()
            if self._need_concrete_fallback:
                # _check_symbolic_races hit an unhandled mask shape
                self._symbolic_accesses.clear()
                self._races.clear()
                self._phase = "concrete"
                self._force_concrete = True
                self._disable_overriders()
                from ...core.patch import SymbolicBailout

                raise SymbolicBailout()
            # Symbolic analysis succeeded — replay grid concretely for side effects
            self._symbolic_race_results = list(self._races)
            self._need_concrete_fallback = True
            self._force_concrete = True
            self._disable_overriders()
            from ...core.patch import SymbolicBailout

            raise SymbolicBailout()

    def pre_warmup_callback(self, jit_fn: Callable, *args, **kwargs) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid: tuple[int, ...]):
        self._grid = tuple(int(g) for g in grid)
        self._need_concrete_fallback = False
        self._symbolic_cas_counts.clear()
        self._symbolic_barrier_candidates.clear()
        # Preserve symbolic results across replay restart; clear on fresh invocation
        if not self._force_concrete:
            self._symbolic_race_results = None
        self._symbolic_accesses.clear()
        self._concrete_accesses.clear()
        self._races.clear()
        self._event_counter = 0
        # Re-enable overriders (may have been disabled by a previous concrete fallback)
        for cb, orig_overrider in self._op_callbacks:
            cb.op_overrider = orig_overrider
        if self._force_concrete or cfg.num_sms > 1:
            # Skip symbolic phase: either a previous SymbolicBailout forced
            # concrete mode, or multiple SMs make symbolic execution unsafe.
            self._phase = "concrete"
            self._disable_overriders()
            self._force_concrete = False
        else:
            self._phase = "symbolic"
        SymbolicExpr.ARANGE_DICT.clear()

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        self.grid_idx = grid_idx

    # ── Race-detector-specific operation overriders ─────────────

    def _op_load_overrider(self, ptr, mask, other, *args):
        if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
            self._bail_to_concrete("load: data-dependent pointer")
        if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
            self._bail_to_concrete("load: data-dependent mask")
        ptr_sym = SymbolicExpr.from_value(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        other_sym = SymbolicExpr.from_value(other) if other is not None else None
        ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)
        elem_size = _get_elem_size(ptr)
        self._record_symbolic_access(
            AccessType.LOAD, ptr_sym, mask_sym, elem_size=elem_size
        )
        return ret

    def _op_store_overrider(self, ptr, value, mask, *args):
        if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
            self._bail_to_concrete("store: data-dependent pointer")
        if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
            self._bail_to_concrete("store: data-dependent mask")
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        ret = SymbolicExpr.create("store", ptr_sym, value_sym, mask_sym)
        elem_size = _get_elem_size(ptr)
        self._record_symbolic_access(
            AccessType.STORE, ptr_sym, mask_sym, elem_size=elem_size
        )
        return ret

    def _op_make_block_ptr_overrider(
        self, base, shape, strides, offsets, tensor_shape, order
    ):
        self._bail_to_concrete("MakeBlockPtr not supported")

    def _op_tensor_pointer_load_overrider(
        self,
        ptr,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        is_volatile,
    ):
        self._bail_to_concrete("TensorPointerLoad not supported")

    def _op_tensor_pointer_store_overrider(
        self, ptr, value, boundary_check, cache_modifier, eviction_policy
    ):
        self._bail_to_concrete("TensorPointerStore not supported")

    def _op_atomic_cas_overrider(self, ptr, cmp, val, sem, scope):
        ptr_expr = SymbolicExpr.from_value(ptr)
        if self._phase == "symbolic":
            ptr_key = self._symbolic_ptr_signature(ptr_expr)
            self._symbolic_cas_counts[ptr_key] = (
                self._symbolic_cas_counts.get(ptr_key, 0) + 1
            )
            if self._symbolic_cas_counts[ptr_key] > 1:
                # Heuristic: a second CAS to the *same address* during
                # symbolic execution of block 0 likely indicates a
                # while-loop spin barrier.  Symbolic execution cannot
                # terminate while loops (tensor comparisons are always
                # truthy), so bail out and restart in concrete mode.
                self._need_concrete_fallback = True
                from ...core.patch import SymbolicBailout

                raise SymbolicBailout()
            self._note_symbolic_atomic(ptr_expr, "cas", scope)
        ret = super()._op_atomic_cas_overrider(ptr, cmp, val, sem, scope)
        elem_size = _get_elem_size(ptr)
        self._record_symbolic_access(
            AccessType.ATOMIC,
            ptr_expr,
            None,
            atomic_op="cas",
            elem_size=elem_size,
            atomic_scope=str(scope).lower() if scope else None,
            atomic_sem=str(sem).lower() if sem else None,
        )
        return ret

    def _op_atomic_rmw_overrider(self, rmwOp, ptr, val, mask, sem, scope):
        ret = super()._op_atomic_rmw_overrider(rmwOp, ptr, val, mask, sem, scope)
        if self._phase == "symbolic" and "add" in str(rmwOp).lower():
            self._note_symbolic_atomic(SymbolicExpr.from_value(ptr), "add", scope)
        # The interpreter always supplies a TensorHandle mask (never None),
        # even when the user omits it. Treat trivially-all-True masks as
        # None to avoid unnecessary symbolic wrapping.  Guard against
        # object-dtype arrays (containing SymbolicExpr) which are
        # spuriously truthy for block 0 but may be PID-dependent.
        mask_expr = None
        if mask is not None:
            is_trivially_true = False
            if hasattr(mask, "data"):
                data = np.asarray(mask.data)
                if data.dtype != object and data.all():
                    is_trivially_true = True
            if not is_trivially_true:
                mask_expr = SymbolicExpr.from_value(mask)
        elem_size = _get_elem_size(ptr)
        self._record_symbolic_access(
            AccessType.ATOMIC,
            SymbolicExpr.from_value(ptr),
            mask_expr,
            atomic_op=f"rmw:{str(rmwOp).lower()}",
            elem_size=elem_size,
            atomic_scope=str(scope).lower() if scope else None,
            atomic_sem=str(sem).lower() if sem else None,
        )
        return ret

    # ── For-loop hook overrides ──────────────────────────────────

    def _on_data_dependent_value(self) -> None:
        self._bail_to_concrete("data-dependent loop value")

    def _should_skip_loop_hooks(self) -> bool:
        return self._phase != "symbolic"

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        overrider_map = self._build_op_overrider_map()
        overrider_map.update(
            {
                Load: self._op_load_overrider,
                Store: self._op_store_overrider,
                MakeBlockPointer: self._op_make_block_ptr_overrider,
                TensorPointerLoad: self._op_tensor_pointer_load_overrider,
                TensorPointerStore: self._op_tensor_pointer_store_overrider,
            }
        )

        # Concrete before-callbacks (active during concrete phase)

        @self.lock_fn
        def before_load(ptr, mask, keys):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = _flatten_offsets(ptr)
            masks = _flatten_mask(mask, len(offsets))
            elem_size = _get_elem_size(ptr)
            event_id = self._next_event_id()
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.LOAD,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                    event_id=event_id,
                    elem_size=elem_size,
                )
            )

        @self.lock_fn
        def before_raw_load(ptr):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = _flatten_offsets(ptr)
            masks = _flatten_mask(None, len(offsets))
            elem_size = _get_elem_size(ptr)
            event_id = self._next_event_id()
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.LOAD,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                    event_id=event_id,
                    elem_size=elem_size,
                )
            )

        @self.lock_fn
        def before_store(ptr, mask, keys):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = _flatten_offsets(ptr)
            masks = _flatten_mask(mask, len(offsets))
            elem_size = _get_elem_size(ptr)
            event_id = self._next_event_id()
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.STORE,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                    event_id=event_id,
                    elem_size=elem_size,
                )
            )

        @self.lock_fn
        def before_raw_store(ptr, value):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = _flatten_offsets(ptr)
            masks = _flatten_mask(None, len(offsets))
            elem_size = _get_elem_size(ptr)
            event_id = self._next_event_id()
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.STORE,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                    event_id=event_id,
                    elem_size=elem_size,
                )
            )

        @self.lock_fn
        def before_atomic_rmw(rmwOp, ptr, val, mask, sem, scope):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = _flatten_offsets(ptr)
            masks = _flatten_mask(mask, len(offsets))
            elem_size = _get_elem_size(ptr)
            event_id = self._next_event_id()
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.ATOMIC,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                    event_id=event_id,
                    atomic_op=f"rmw:{str(rmwOp).lower()}",
                    elem_size=elem_size,
                    atomic_scope=str(scope).lower() if scope else None,
                    atomic_sem=str(sem).lower() if sem else None,
                )
            )

        @self.lock_fn
        def before_atomic_cas(ptr, cmp, val, sem, scope):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = _flatten_offsets(ptr)
            masks = _flatten_mask(None, len(offsets))
            elem_size = _get_elem_size(ptr)
            event_id = self._next_event_id()
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.ATOMIC,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                    event_id=event_id,
                    atomic_op="cas",
                    elem_size=elem_size,
                    atomic_scope=str(scope).lower() if scope else None,
                    atomic_sem=str(sem).lower() if sem else None,
                )
            )

        BEFORE_MAP: dict[type[Op], Callable] = {
            Load: before_load,
            RawLoad: before_raw_load,
            Store: before_store,
            RawStore: before_raw_store,
            AtomicRMW: before_atomic_rmw,
            AtomicCas: before_atomic_cas,
        }

        overrider = overrider_map.get(op_type)
        before = BEFORE_MAP.get(op_type)

        if overrider is not None:
            locked_overrider = self.lock_fn(overrider)
            callbacks = OpCallbacks(
                op_overrider=locked_overrider,
                before_callback=before,
            )
            self._op_callbacks.append((callbacks, locked_overrider))
            return callbacks

        return OpCallbacks()

    def finalize(self) -> list:
        if self._symbolic_race_results is not None:
            races = self._symbolic_race_results
            self._symbolic_race_results = None
        elif self._phase == "concrete":
            races = detect_races(self._concrete_accesses)
        else:
            races = []
        self._symbolic_accesses.clear()
        self._concrete_accesses.clear()
        self._races.clear()
        self._phase = "init"
        self._event_counter = 0
        return races

    # ── Internal helpers ─────────────────────────────────────────

    def _record_symbolic_access(
        self,
        access_type: AccessType,
        ptr_expr: SymbolicExpr,
        mask_expr: Optional[SymbolicExpr],
        atomic_op: Optional[str] = None,
        elem_size: int = 1,
        atomic_scope: str | None = None,
        atomic_sem: str | None = None,
    ) -> None:
        is_data_dependent = ptr_expr.has_op("load")
        if mask_expr is not None and mask_expr.has_op("load"):
            is_data_dependent = True
        if is_data_dependent:
            self._bail_to_concrete("data-dependent access")
        event_id = self._next_event_id()
        self._symbolic_accesses.append(
            SymbolicMemoryAccess(
                access_type=access_type,
                ptr_expr=ptr_expr,
                mask_expr=mask_expr,
                is_data_dependent=is_data_dependent,
                call_path=extract_user_frames(),
                event_id=event_id,
                atomic_op=atomic_op,
                elem_size=elem_size,
                atomic_scope=atomic_scope,
                atomic_sem=atomic_sem,
            )
        )

    def _disable_overriders(self) -> None:
        for cb, _orig in self._op_callbacks:
            cb.op_overrider = None

    def _next_event_id(self) -> int:
        event_id = self._event_counter
        self._event_counter += 1
        return event_id

    def _bail_to_concrete(self, reason: str = "") -> None:
        self._need_concrete_fallback = True
        from ...core.patch import SymbolicBailout

        raise SymbolicBailout()

    def _symbolic_ptr_signature(self, ptr_expr: SymbolicExpr) -> tuple[str, ...]:
        ptr_z3, _ = ptr_expr.eval()
        if isinstance(ptr_z3, list):
            return tuple(str(e) for e in ptr_z3)
        return (str(ptr_z3),)

    def _note_symbolic_atomic(
        self, ptr_expr: SymbolicExpr, atomic_op: str, scope
    ) -> None:
        scope_str = str(scope).lower() if scope is not None else None
        if not _is_inter_block_scope(scope_str):
            return
        ptr_key = self._symbolic_ptr_signature(ptr_expr)
        self._symbolic_barrier_candidates[ptr_key].add(atomic_op)
        if {"add", "cas"} <= self._symbolic_barrier_candidates[ptr_key]:
            self._need_concrete_fallback = True
            from ...core.patch import SymbolicBailout

            raise SymbolicBailout()

    def _check_symbolic_races(self) -> list[RaceRecord]:
        if not self._symbolic_accesses:
            return []

        grid = self._grid

        # Pre-evaluate all expressions to populate ARANGE_DICT
        for acc in self._symbolic_accesses:
            acc.ptr_expr.eval()
            if acc.mask_expr is not None:
                acc.mask_expr.eval()

        # Create block-specific PID variables
        pid_a = [Int(f"pid_a_{i}") for i in range(3)]
        pid_b = [Int(f"pid_b_{i}") for i in range(3)]

        # Build substitution pairs
        orig_pids = [SymbolicExpr.PID0, SymbolicExpr.PID1, SymbolicExpr.PID2]
        sub_a: list[tuple[ArithRef, ArithRef]] = list(zip(orig_pids, pid_a))
        sub_b: list[tuple[ArithRef, ArithRef]] = list(zip(orig_pids, pid_b))

        # Create block-specific arange variables (ARANGE_DICT now populated)
        arange_constraints_a: list = []
        arange_constraints_b: list = []
        for (start, end), (orig_var, _) in SymbolicExpr.ARANGE_DICT.items():
            var_a = Int(f"arange_a_{start}_{end}")
            var_b = Int(f"arange_b_{start}_{end}")
            sub_a.append((orig_var, var_a))
            sub_b.append((orig_var, var_b))
            arange_constraints_a.append(And(var_a >= start, var_a < end))
            arange_constraints_b.append(And(var_b >= start, var_b < end))

        # Common constraints
        grid_constraints = And(
            pid_a[0] >= 0,
            pid_a[0] < grid[0],
            pid_a[1] >= 0,
            pid_a[1] < grid[1],
            pid_a[2] >= 0,
            pid_a[2] < grid[2],
            pid_b[0] >= 0,
            pid_b[0] < grid[0],
            pid_b[1] >= 0,
            pid_b[1] < grid[1],
            pid_b[2] >= 0,
            pid_b[2] < grid[2],
        )
        different_blocks = Or(
            pid_a[0] != pid_b[0],
            pid_a[1] != pid_b[1],
            pid_a[2] != pid_b[2],
        )

        def _apply_sub(expr, pairs):
            if isinstance(expr, list):
                return [substitute(e, *pairs) for e in expr]
            return substitute(expr, *pairs)

        def _broadcast_mask(mask_z3, sub_pairs, n_lanes):
            """Return a list of n_lanes Z3 booleans, or None if unhandled.

            Supported forms:
              None mask        → [True] * n_lanes  (all active)
              Scalar z3 bool   → [val] * n_lanes   (uniform mask)
              List len == n    → per-lane list      (exact match)

            Any other shape returns None → caller should bail to concrete.
            """
            if mask_z3 is None:
                return [True] * n_lanes
            val = _apply_sub(mask_z3, sub_pairs)
            if isinstance(val, list):
                if len(val) != n_lanes:
                    return None  # shape mismatch — bail
                if not all(is_bool(v) for v in val):
                    return None  # non-bool lane — bail
                return val
            if is_bool(val):
                return [val] * n_lanes
            return None  # unrecognised form — bail

        races: list[RaceRecord] = []
        accesses = self._symbolic_accesses

        for i in range(len(accesses)):
            for j in range(i, len(accesses)):
                acc_a = accesses[i]
                acc_b = accesses[j]
                if acc_a.epoch != acc_b.epoch:
                    continue

                race_type = _classify_race_type(acc_a.access_type, acc_b.access_type)
                if race_type is None:
                    continue

                # Get Z3 expressions for pointers
                ptr_z3_a, ptr_constr_a = acc_a.ptr_expr.eval()
                ptr_z3_b, ptr_constr_b = acc_b.ptr_expr.eval()

                # Apply block-specific substitutions
                addr_a = _apply_sub(ptr_z3_a, sub_a)
                addr_b = _apply_sub(ptr_z3_b, sub_b)

                # Evaluate masks
                mask_z3_a, mask_constr_a = (None, None)
                if acc_a.mask_expr is not None:
                    mask_z3_a, mask_constr_a = acc_a.mask_expr.eval()

                mask_z3_b, mask_constr_b = (None, None)
                if acc_b.mask_expr is not None:
                    mask_z3_b, mask_constr_b = acc_b.mask_expr.eval()

                # Normalize addresses to lists
                addrs_a = addr_a if isinstance(addr_a, list) else [addr_a]
                addrs_b = addr_b if isinstance(addr_b, list) else [addr_b]

                # Broadcast masks to match address lengths
                masks_a = _broadcast_mask(mask_z3_a, sub_a, len(addrs_a))
                if masks_a is None:
                    self._need_concrete_fallback = True
                    return []
                masks_b = _broadcast_mask(mask_z3_b, sub_b, len(addrs_b))
                if masks_b is None:
                    self._need_concrete_fallback = True
                    return []

                # Byte-range element sizes
                elem_size_a = acc_a.elem_size
                elem_size_b = acc_b.elem_size

                # Build lane-pair candidate predicates
                candidate_preds = []
                for ia, a_expr in enumerate(addrs_a):
                    ma = masks_a[ia]
                    for ib, b_expr in enumerate(addrs_b):
                        mb = masks_b[ib]
                        if elem_size_a > 1 or elem_size_b > 1:
                            addr_overlap = And(
                                a_expr < b_expr + elem_size_b,
                                b_expr < a_expr + elem_size_a,
                            )
                        else:
                            addr_overlap = a_expr == b_expr
                        terms = [t for t in (ma, mb, addr_overlap) if t is not True]
                        if not terms:
                            pred = addr_overlap
                        elif len(terms) == 1:
                            pred = terms[0]
                        else:
                            pred = And(*terms)
                        candidate_preds.append((ia, ib, pred))

                if not candidate_preds:
                    continue

                overlap = (
                    candidate_preds[0][2]
                    if len(candidate_preds) == 1
                    else Or(*(pred for _, _, pred in candidate_preds))
                )

                # Build solver
                solver = Solver()
                solver.add(grid_constraints)
                solver.add(different_blocks)
                for c in arange_constraints_a:
                    solver.add(c)
                for c in arange_constraints_b:
                    solver.add(c)

                # Add ptr constraints (substituted)
                if ptr_constr_a is not None:
                    solver.add(_apply_sub(ptr_constr_a, sub_a))
                if ptr_constr_b is not None:
                    solver.add(_apply_sub(ptr_constr_b, sub_b))

                # Add mask side constraints (range/shape, not lane-active semantics)
                if mask_constr_a is not None:
                    solver.add(_apply_sub(mask_constr_a, sub_a))
                if mask_constr_b is not None:
                    solver.add(_apply_sub(mask_constr_b, sub_b))

                # Check for overlap
                solver.add(overlap)
                if solver.check() == sat:
                    model = solver.model()
                    block_a_idx = tuple(
                        model.evaluate(p, model_completion=True).as_long()
                        for p in pid_a
                    )
                    block_b_idx = tuple(
                        model.evaluate(p, model_completion=True).as_long()
                        for p in pid_b
                    )

                    # Find witness from the actual lane pair whose predicate is satisfied.
                    # Use max(a, b) so the witness falls in the actual overlap region.
                    witness_addr = None
                    for ia, ib, pred in candidate_preds:
                        val = model.evaluate(pred, model_completion=True)
                        if is_true(val):
                            a_val = model.evaluate(
                                addrs_a[ia], model_completion=True
                            ).as_long()
                            b_val = model.evaluate(
                                addrs_b[ib], model_completion=True
                            ).as_long()
                            witness_addr = max(a_val, b_val)
                            break
                    if witness_addr is None:
                        a_val = model.evaluate(
                            addrs_a[0], model_completion=True
                        ).as_long()
                        b_val = model.evaluate(
                            addrs_b[0], model_completion=True
                        ).as_long()
                        witness_addr = max(a_val, b_val)

                    mem_a = MemoryAccess(
                        access_type=acc_a.access_type,
                        ptr=0,
                        offsets=np.array([witness_addr]),
                        masks=np.array([True]),
                        grid_idx=block_a_idx,
                        call_path=acc_a.call_path,
                        epoch=acc_a.epoch,
                    )
                    mem_b = MemoryAccess(
                        access_type=acc_b.access_type,
                        ptr=0,
                        offsets=np.array([witness_addr]),
                        masks=np.array([True]),
                        grid_idx=block_b_idx,
                        call_path=acc_b.call_path,
                        epoch=acc_b.epoch,
                    )
                    races.append(
                        RaceRecord(
                            race_type=race_type,
                            address_offset=witness_addr,
                            access_a=mem_a,
                            access_b=mem_b,
                        )
                    )

        return races


def _classify_race_type(
    ta: AccessType,
    tb: AccessType,
) -> Optional[RaceType]:
    """Classify race type from two access types."""
    if ta == AccessType.LOAD and tb == AccessType.LOAD:
        return None
    if ta == AccessType.ATOMIC and tb == AccessType.ATOMIC:
        return None
    is_write_a = ta in (AccessType.STORE, AccessType.ATOMIC)
    is_write_b = tb in (AccessType.STORE, AccessType.ATOMIC)
    if is_write_a and is_write_b:
        return RaceType.WAW
    return RaceType.RW


def _active_offsets(access: MemoryAccess) -> np.ndarray:
    active = access.offsets[access.masks]
    if access.elem_size > 1:
        byte_offsets = np.concatenate([active + b for b in range(access.elem_size)])
        return np.unique(byte_offsets).astype(np.int64)
    return np.unique(active).astype(np.int64)


def _matched_barrier_addrs(acc: MemoryAccess, barrier_addrs: set[int]) -> list[int]:
    if acc.access_type != AccessType.ATOMIC:
        return []
    if not _is_inter_block_scope(acc.atomic_scope):
        return []
    atomic_op = (acc.atomic_op or "").lower()
    if "add" not in atomic_op and "cas" not in atomic_op:
        return []
    return [int(off) for off in _active_offsets(acc) if int(off) in barrier_addrs]


def _detect_global_barrier_addresses(accesses: list[MemoryAccess]) -> set[int]:
    blocks = {acc.grid_idx for acc in accesses}
    if len(blocks) < 2:
        return set()

    stats: dict[int, dict[str, set[tuple[int, ...]]]] = defaultdict(
        lambda: {"add": set(), "cas": set()}
    )

    for acc in accesses:
        if acc.access_type != AccessType.ATOMIC:
            continue
        atomic_op = (acc.atomic_op or "").lower()
        is_add = "add" in atomic_op
        is_cas = "cas" in atomic_op
        if not is_add and not is_cas:
            continue
        if not _is_inter_block_scope(acc.atomic_scope):
            continue
        for off in _active_offsets(acc):
            addr = int(off)
            if is_add:
                stats[addr]["add"].add(acc.grid_idx)
            if is_cas:
                stats[addr]["cas"].add(acc.grid_idx)

    barrier_addrs: set[int] = set()
    for addr, state in stats.items():
        if state["add"] >= blocks and state["cas"] >= blocks:
            barrier_addrs.add(addr)
    return barrier_addrs


def _annotate_block_epochs(
    block_accesses: list[MemoryAccess], barrier_addrs: set[int]
) -> int:
    block_accesses.sort(key=lambda a: a.event_id)
    epoch = 0
    pending: dict[int, set[str]] = {}  # barrier_addr → {"add", "cas"}

    for acc in block_accesses:
        matched = _matched_barrier_addrs(acc, barrier_addrs)
        if matched:
            atomic_op = (acc.atomic_op or "").lower()
            op_kind = "add" if "add" in atomic_op else "cas"
            for addr in matched:
                pending.setdefault(addr, set()).add(op_kind)
            acc.epoch = epoch
            continue

        # Non-barrier access: advance epoch if any pending barrier completed
        if any({"add", "cas"} <= ops for ops in pending.values()):
            epoch += 1
            pending.clear()

        acc.epoch = epoch

    return epoch


def _apply_epoch_annotations(accesses: list[MemoryAccess]) -> None:
    for acc in accesses:
        acc.epoch = 0

    barrier_addrs = _detect_global_barrier_addresses(accesses)
    if not barrier_addrs:
        return

    per_block: dict[tuple[int, ...], list[MemoryAccess]] = defaultdict(list)
    for acc in accesses:
        per_block[acc.grid_idx].append(acc)

    completed_rounds_per_block: dict[tuple[int, ...], int] = {}
    for grid_idx, block_accesses in per_block.items():
        completed_rounds_per_block[grid_idx] = _annotate_block_epochs(
            block_accesses, barrier_addrs
        )

    globally_completed_rounds = min(completed_rounds_per_block.values(), default=0)
    if globally_completed_rounds <= 0:
        return

    for acc in accesses:
        if acc.epoch > globally_completed_rounds:
            acc.epoch = globally_completed_rounds


def detect_races(accesses: list[MemoryAccess]) -> list[RaceRecord]:
    """Detect data races using an inverted index on byte addresses.

    A race occurs when:
    1. Two different blocks access the same byte address
    2. At least one access is a write (Store)
    3. The accesses are not both atomic
    """
    _apply_epoch_annotations(accesses)

    addr_to_accesses: dict[int, list[MemoryAccess]] = defaultdict(list)

    for access in accesses:
        unique_offsets = _active_offsets(access)
        for off in unique_offsets:
            addr_to_accesses[int(off)].append(access)

    races: list[RaceRecord] = []
    seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...], RaceType, int, int]] = set()

    for addr, access_list in addr_to_accesses.items():
        if len(access_list) < 2:
            continue

        # Fast path: if every access at this address is ATOMIC, no race is
        # possible (ATOMIC-ATOMIC pairs are always safe).  This avoids an
        # O(n²) inner loop when a CAS-spin barrier produces thousands of
        # ATOMIC accesses at the same sync address.
        if all(acc.access_type == AccessType.ATOMIC for acc in access_list):
            continue

        block_accesses: dict[tuple[int, ...], list[MemoryAccess]] = defaultdict(list)
        for acc in access_list:
            block_accesses[acc.grid_idx].append(acc)

        blocks = list(block_accesses.keys())
        if len(blocks) < 2:
            continue

        for bi in range(len(blocks)):
            for bj in range(bi + 1, len(blocks)):
                block_a, block_b = blocks[bi], blocks[bj]
                pair_key = (min(block_a, block_b), max(block_a, block_b))
                accesses_a = block_accesses[block_a]
                accesses_b = block_accesses[block_b]

                for acc_a in accesses_a:
                    for acc_b in accesses_b:
                        if acc_a.epoch != acc_b.epoch:
                            continue
                        if (
                            acc_a.access_type == AccessType.ATOMIC
                            and acc_b.access_type == AccessType.ATOMIC
                        ):
                            continue
                        if (
                            acc_a.access_type == AccessType.LOAD
                            and acc_b.access_type == AccessType.LOAD
                        ):
                            continue
                        race_type = _classify_race(acc_a, acc_b)
                        if race_type is None:
                            continue
                        dedup_key = (
                            pair_key[0],
                            pair_key[1],
                            race_type,
                            addr,
                            acc_a.epoch,
                        )
                        if dedup_key in seen_pairs:
                            continue
                        seen_pairs.add(dedup_key)
                        races.append(
                            RaceRecord(
                                race_type=race_type,
                                address_offset=addr,
                                access_a=acc_a,
                                access_b=acc_b,
                            )
                        )

    return races


def _classify_race(a: MemoryAccess, b: MemoryAccess) -> Optional[RaceType]:
    """Classify the race type between two accesses from different blocks."""
    ta = a.access_type
    tb = b.access_type
    if ta == AccessType.LOAD and tb == AccessType.LOAD:
        return None
    if ta == AccessType.ATOMIC and tb == AccessType.ATOMIC:
        return None
    is_write_a = ta in (AccessType.STORE, AccessType.ATOMIC)
    is_write_b = tb in (AccessType.STORE, AccessType.ATOMIC)
    if is_write_a and is_write_b:
        return RaceType.WAW
    return RaceType.RW
