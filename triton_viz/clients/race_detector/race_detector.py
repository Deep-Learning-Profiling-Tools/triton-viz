from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    Literal,
    NoReturn,
    TypeVar,
    cast,
)

from torch import Tensor
from z3 import (
    Solver,
    Int,
    And,
    Or,
    BoolVal,
)
from z3.z3 import BoolRef, ArithRef

import triton.language as tl

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    Load,
    Store,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
)
from ..utils import (
    check_storage_contiguous,
    get_physical_addr_from_tensor_slice,
    check_inner_stride_equal_to_one,
)
from ..symbolic_engine import (
    SymbolicExpr,
    SymbolicClient,
    RangeWrapper,
    PendingCheck,
    Z3Expr,
    ConstraintConjunction,
)
from .data import AccessEventRecord
from ...utils.traceback_utils import extract_user_frames
from ...core.config import config as cfg

RaceDetectorT = TypeVar("RaceDetectorT", bound="RaceDetector")

AccessMode = Literal["read", "write"]


def _range_to_iterator_constraint(
    var: ArithRef, *, start: int, stop: int, step: int
) -> BoolRef:
    """Return a Z3 constraint describing values produced by ``range(start, stop, step)``."""
    if step == 0:
        raise ValueError("range() step cannot be 0")

    if step > 0:
        bounds = And(var >= start, var < stop)
    else:
        bounds = And(var <= start, var > stop)

    abs_step = abs(step)
    if abs_step == 1:
        return bounds

    return And(bounds, (var - start) % abs_step == 0)


def _make_event_signature(
    access_mode: AccessMode,
    source_location: tuple[str, int, str] | None,
    addr_expr: Z3Expr,
    local_constraints: ConstraintConjunction,
) -> int:
    """Signature used to dedupe repeated access events within a single loop.

    Distinct from sanitizer's ``_make_signature``: ``access_mode`` and
    ``source_location`` are part of the key so a ``load`` and a ``store`` at
    the same address inside the same loop stay as separate events (different
    program-order nodes for future HB analysis).
    """
    if isinstance(addr_expr, list):
        if len(addr_expr) == 1:
            addr_hash = hash(addr_expr[0])
        else:
            addr_hash = hash(tuple(hash(e) for e in addr_expr))
    else:
        addr_hash = hash(addr_expr)

    constr_hash = 0 if local_constraints is None else hash(local_constraints)
    return hash((access_mode, source_location, addr_hash, constr_hash))


@dataclass
class PendingEvent(PendingCheck):
    """Extends ``PendingCheck`` with access-mode and op_type metadata.

    ``LoopContext.pending_checks`` is typed as ``list[PendingCheck]`` but is
    fine accepting subclass instances at runtime. Keeping the extension local
    to this module avoids widening the shared schema used by sanitizer.
    """

    access_mode: AccessMode = "read"
    op_type: type[Op] = Load


@dataclass(frozen=True)
class _FnSymbolicCache:
    fn: Callable
    grid: tuple[int, ...]
    args: tuple

    @cached_property
    def hash_value(self) -> int:
        return hash((self.fn, self.grid, self.args))

    def __hash__(self) -> int:
        return self.hash_value


# Independent from sanitizer's equivalent set so the two detectors don't
# pollute each other's grid-enumeration cache.
_fn_symbolic_cache_set: set[_FnSymbolicCache] = set()


class RaceDetector(Client):
    """Factory class that returns the concrete race-detector implementation
    based on the value of ``cfg.enable_race_detector``.
    """

    NAME = "race_detector"

    def __new__(cls: type[RaceDetectorT], *args: Any, **kwargs: Any) -> RaceDetectorT:
        if cls is RaceDetector:
            target_cls = cast(
                type["RaceDetector"],
                SymbolicRaceDetector if cfg.enable_race_detector else NullRaceDetector,
            )
            obj = object.__new__(target_cls)
            cast(Any, target_cls).__init__(obj, *args, **kwargs)
            return cast(RaceDetectorT, obj)
        return cast(RaceDetectorT, object.__new__(cls))

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.abort_on_error: bool = abort_on_error

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn: Callable, *args: Any, **kwargs: Any) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        raise NotImplementedError

    def finalize(self) -> list:
        raise NotImplementedError

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        raise NotImplementedError

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        raise NotImplementedError

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        raise NotImplementedError

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        raise NotImplementedError


class SymbolicRaceDetector(RaceDetector, SymbolicClient):
    def __init__(self, abort_on_error: bool = False):
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[AccessEventRecord] = []
        self.grid: tuple[int, ...] | None = None
        self.grid_idx = None
        self.tensors: list[Tensor] = []
        self.tensor_addrs: list[tuple[int, int, Tensor]] = []
        self.tensor_names: dict[int, set[str]] = {}
        self.need_full_grid: bool | None = None
        self.last_grid: tuple[int, int, int] | None = None
        self.cache_args: list[Any] = []
        self.cache_grid: tuple[int, ...] | None = None
        self.addr_ok: BoolRef | None = None
        self.pid_ok: BoolRef | None = None
        self.solver: Solver | None = None
        self.addr_sym: ArithRef | None = None

    # ── Tensor resolution helpers (copied from SymbolicSanitizer) ─────────

    def _collect_tensor_base(self, expr: SymbolicExpr) -> int | None:
        def walk(node: SymbolicExpr) -> int | None:
            if (
                node.op == "const"
                and isinstance(node.dtype, tl.pointer_type)
                and not node.shape
            ):
                return node.to_py()
            for child in node.children.values():
                if child is not None:
                    base = walk(child)
                    if base is not None:
                        return base
            return None

        return walk(expr)

    def _record_tensor_name(self, tensor: Tensor, name: str) -> None:
        if not name:
            return
        names = self.tensor_names.setdefault(id(tensor), set())
        names.add(name)

    def _get_tensor_name(self, tensor: Tensor) -> str | None:
        names = self.tensor_names.get(id(tensor))
        if not names:
            return None
        return ", ".join(sorted(names))

    def _resolve_tensor(self, symbolic_expr: SymbolicExpr) -> Tensor | None:
        """Best-effort base-tensor lookup. Returns None on failure.

        Deliberately does NOT use a nearest-segment fallback (that's only
        meaningful with a concrete witness address from the Z3 model, which
        doesn't exist at access-capture time).
        """
        base = self._collect_tensor_base(symbolic_expr)
        if base is None:
            return None
        for tensor in self.tensors:
            if tensor.data_ptr() == base:
                return tensor
        for start, end, tensor in self.tensor_addrs:
            if start <= base <= end:
                return tensor
        return None

    # ── Event recording ───────────────────────────────────────────────────

    def _record_access_event(
        self,
        access_mode: AccessMode,
        op_type: type[Op],
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        tensor = self._resolve_tensor(symbolic_expr)
        tensor_name = self._get_tensor_name(tensor) if tensor is not None else None

        # Full solver-assertion snapshot at the moment this event is recorded.
        # When called from _loop_hook_after it runs between solver.push() and
        # solver.pop(), so the snapshot captures:
        #   addr_ok, pid_ok, innermost loop iterator, all outer loop iterators.
        # plus the expression's own local constraints — giving Step 2 enough
        # context to run alias queries without replaying callbacks.
        solver_snapshot: tuple[Any, ...] = (
            tuple(self.solver.assertions()) if self.solver is not None else ()
        )
        if expr_constraints is None:
            local: tuple[Any, ...] = ()
        elif isinstance(expr_constraints, (list, tuple)):
            local = tuple(expr_constraints)
        else:
            local = (expr_constraints,)

        self.records.append(
            AccessEventRecord(
                op_type=op_type,
                access_mode=access_mode,
                tensor=tensor,
                tensor_name=tensor_name,
                symbolic_expr=symbolic_expr,
                addr_expr=access_addr,
                premises=solver_snapshot + local,
                local_constraints=local,
                source_location=source_location,
                grid_idx=self.grid_idx,
            )
        )

    def _handle_access_check(
        self,
        expr: SymbolicExpr,
        op_type: type[Op],
        access_mode: AccessMode,
    ) -> None:
        """Capture a memory access expression.

        Outside any loop: recorded immediately. Inside a loop: deferred to the
        enclosing loop's flush point, with ``_make_event_signature`` used to
        dedupe events that repeat across iterations of the same loop.
        """
        z3_addr, z3_constraints = expr.eval()
        frames = extract_user_frames(num_frames=1)
        frame = frames[-1] if frames else None
        source_location = (
            (frame.filename, frame.lineno, frame.func_name) if frame else None
        )

        if not self.loop_stack:
            self._record_access_event(
                access_mode,
                op_type,
                z3_addr,
                z3_constraints,
                expr,
                source_location,
            )
            return

        ctx = self.loop_stack[-1]
        signature = _make_event_signature(
            access_mode, source_location, z3_addr, z3_constraints
        )
        pending_idx = ctx.signature_cache.get(signature)
        if pending_idx is None:
            ctx.signature_cache[signature] = len(ctx.pending_checks)
            ctx.pending_checks.append(
                PendingEvent(
                    symbolic_expr=expr,
                    addr_expr=z3_addr,
                    constraints=z3_constraints,
                    source_location=source_location,
                    access_mode=access_mode,
                    op_type=op_type,
                )
            )
        else:
            if cfg.verbose:
                print("[RaceDetector]  ↪ skip duplicated addr in loop")

    def _clear_cache(self) -> None:
        self.cache_args.clear()
        self.cache_grid = None

    # ── Client callbacks ──────────────────────────────────────────────────

    def pre_run_callback(self, fn: Callable) -> bool:
        if self.cache_grid:
            fn_cache = _FnSymbolicCache(fn, self.cache_grid, tuple(self.cache_args))
            self._clear_cache()
            if fn_cache not in _fn_symbolic_cache_set:
                _fn_symbolic_cache_set.add(fn_cache)
                return True
            return False
        if self.need_full_grid is None:
            return True
        return self.need_full_grid

    def post_run_callback(self, fn: Callable) -> bool:
        if self.need_full_grid is None:
            self.need_full_grid = False
        if self.grid_idx == self.last_grid or not self.need_full_grid:
            self._clear_cache()
            self.tensors.clear()
            self.tensor_addrs.clear()
            self.tensor_names.clear()
        ret = self.need_full_grid
        self.need_full_grid = None
        return ret

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        if not hasattr(arg, "data_ptr"):
            if name not in ["num_warps", "num_stages", "maxnreg", "num_ctas"]:
                self.cache_args.append(arg)
            return
        from triton.runtime.jit import TensorWrapper

        if isinstance(arg, TensorWrapper):
            arg = arg.base
        if arg.is_contiguous() or check_storage_contiguous(arg):
            start = arg.data_ptr()
            end = arg.data_ptr() + (arg.numel() - 1) * arg.element_size()
            tensor_physical_addresses = [(start, end, arg)]
        elif check_inner_stride_equal_to_one(arg):
            tensor_physical_addresses = [
                (start, end, arg)
                for start, end in get_physical_addr_from_tensor_slice(arg)
            ]
        else:
            raise ValueError(
                "The race detector only supports contiguously stored tensors for now!"
            )
        self.cache_args.append((arg.shape, arg.stride(), arg.dtype))
        self._record_tensor_name(arg, name)
        self.tensors.append(arg)
        self.tensor_addrs.extend(tensor_physical_addresses)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        grid = tuple(int(g) for g in grid)
        self.cache_grid = grid
        self.last_grid = (grid[0] - 1, grid[1] - 1, grid[2] - 1)
        self.grid = grid
        addr = Int("addr")

        addr_ok_expr = (
            Or(*[And(addr >= s, addr <= e) for s, e, _ in self.tensor_addrs])
            if self.tensor_addrs
            else BoolVal(False)
        )
        self.addr_ok = cast(BoolRef, addr_ok_expr)

        pid_ok_expr = And(
            SymbolicExpr.PID0 < self.grid[0],
            SymbolicExpr.PID1 < self.grid[1],
            SymbolicExpr.PID2 < self.grid[2],
            SymbolicExpr.PID0 >= 0,
            SymbolicExpr.PID1 >= 0,
            SymbolicExpr.PID2 >= 0,
        )
        self.pid_ok = cast(BoolRef, pid_ok_expr)

        self.solver = Solver()
        # Race detector keeps addr_ok and pid_ok as positive premises — they're
        # the common preconditions every captured access event carries forward
        # for Step 2's alias query. (Sanitizer adds Not(addr_ok) here instead.)
        self.solver.add(self.addr_ok)
        self.solver.add(self.pid_ok)
        self.addr_sym = addr

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self.grid_idx = grid_idx

    def _on_data_dependent_value(self) -> None:
        self.need_full_grid = True

    # ── Op overriders ─────────────────────────────────────────────────────

    def _op_load_overrider(self, ptr, mask, other, *args):
        if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
            self.need_full_grid = True
            ptr = ptr.replace_subtree("load")
        if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
            self.need_full_grid = True
            mask = mask.replace_subtree("load")
        ptr_sym = SymbolicExpr.from_value(ptr)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        other_sym = SymbolicExpr.from_value(other) if other is not None else None
        ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)
        self._handle_access_check(ret, Load, "read")
        return ret

    def _op_store_overrider(self, ptr, value, mask, *args):
        if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
            self.need_full_grid = True
            ptr = ptr.replace_subtree("load")
        if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
            self.need_full_grid = True
            mask = mask.replace_subtree("load")
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
        ret = SymbolicExpr.create("store", ptr_sym, value_sym, mask_sym)
        self._handle_access_check(ret, Store, "write")
        return ret

    def _op_make_block_ptr_overrider(
        self, base, shape, strides, offsets, tensor_shape, order
    ):
        base_sym = SymbolicExpr.from_value(base)
        shape_syms = [SymbolicExpr.from_value(s) for s in shape]
        stride_syms = [SymbolicExpr.from_value(s) for s in strides]
        offset_syms = [SymbolicExpr.from_value(o) for o in offsets]
        block_shape_vals = [int(b) for b in tensor_shape]
        order_vals = [int(o) for o in order]
        return SymbolicExpr.create(
            "make_block_ptr",
            base_sym,
            shape_syms,
            stride_syms,
            offset_syms,
            block_shape_vals,
            order_vals,
        )

    def _op_tensor_pointer_load_overrider(
        self,
        ptr,
        boundary_check,
        padding_option,
        cache_modifier,
        eviction_policy,
        is_volatile,
    ):
        ptr_sym = SymbolicExpr.from_value(ptr)
        ret = SymbolicExpr.create("tensor_pointer_load", ptr_sym, boundary_check)
        self._handle_access_check(ret, TensorPointerLoad, "read")
        return ret

    def _op_tensor_pointer_store_overrider(
        self, ptr, value, boundary_check, cache_modifier, eviction_policy
    ):
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        ret = SymbolicExpr.create(
            "tensor_pointer_store", ptr_sym, value_sym, boundary_check
        )
        self._handle_access_check(ret, TensorPointerStore, "write")
        return ret

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
        overrider = overrider_map.get(op_type)
        if overrider is not None:
            return OpCallbacks(op_overrider=self.lock_fn(overrider))
        return OpCallbacks()

    # ── For-loop hooks ────────────────────────────────────────────────────

    def _loop_hook_before(self, lineno, iterable):
        if not isinstance(iterable, RangeWrapper):
            if cfg.verbose:
                print("not a range wrapper, skipping for-loop iterator association.")
            return
        super()._loop_hook_before(lineno, iterable)
        if cfg.verbose:
            print(f"[RaceDetector] ▶ enter loop@{lineno}, len={iterable.length}")

    def _loop_hook_after(self, lineno: int) -> None:
        if not self.loop_stack or self.loop_stack[-1].lineno != lineno:
            return
        ctx = self.loop_stack.pop()

        solver = self.solver
        addr_sym = self.addr_sym
        assert solver is not None
        assert addr_sym is not None

        all_iterator_constraints: list[BoolRef] = []
        if ctx.pending_checks:
            solver.push()
            iterator_constraint = _range_to_iterator_constraint(
                ctx.idx_z3, start=ctx.start, stop=ctx.stop, step=ctx.step
            )
            solver.add(iterator_constraint)
            all_iterator_constraints.append(iterator_constraint)

            # Also add constraints for all outer loops that are still active.
            for outer_ctx in self.loop_stack:
                outer_constraint = _range_to_iterator_constraint(
                    outer_ctx.idx_z3,
                    start=outer_ctx.start,
                    stop=outer_ctx.stop,
                    step=outer_ctx.step,
                )
                solver.add(outer_constraint)
                all_iterator_constraints.append(outer_constraint)

        for pending in ctx.pending_checks:
            # Items enqueued by _handle_access_check are PendingEvent instances
            # (subclass of PendingCheck) — narrow here so the attribute accesses
            # are type-safe under Literal["read", "write"].
            assert isinstance(pending, PendingEvent)
            if cfg.verbose:
                print(
                    "[RaceDetector] ▶ recording:",
                    pending.addr_expr,
                    f" with iterator constraints: {all_iterator_constraints} ",
                    f" and expression-related constraints: {pending.constraints} ",
                )
            self._record_access_event(
                pending.access_mode,
                pending.op_type,
                pending.addr_expr,
                pending.constraints,
                pending.symbolic_expr,
                pending.source_location,
            )

        if ctx.pending_checks:
            solver.pop()

        if cfg.verbose:
            print(
                f"[RaceDetector] ▶ leave loop@{lineno} end. "
                f"(recorded {len(ctx.pending_checks)} unique addr patterns)"
            )

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicClient.register_for_loop_callback(self)

    def finalize(self) -> list:
        return []


class NullRaceDetector(RaceDetector):
    """A do-nothing object returned when the race-detector backend is 'off'.
    Any attribute access raises an explicit error so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = False, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)

    def _disabled(self, method: str) -> NoReturn:
        raise RuntimeError(
            f"[NullRaceDetector] '{method}' was called, "
            "but race-detector backend is off; no functionality is available."
        )

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        self._disabled("arg_callback")

    def finalize(self) -> list:
        self._disabled("finalize")

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        self._disabled("grid_callback")

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self._disabled("grid_idx_callback")

    def register_op_callback(
        self, op_type: type[Op], *args: Any, **kwargs: Any
    ) -> OpCallbacks:
        self._disabled("register_op_callback")

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        self._disabled("register_for_loop_callback")

    def __getattr__(self, name: str) -> Any:
        self._disabled(name)
