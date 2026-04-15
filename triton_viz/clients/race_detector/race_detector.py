from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    NoReturn,
    TypeVar,
    cast,
)

from z3 import (
    Solver,
    Int,
    And,
    Or,
    BoolVal,
)
from z3.z3 import BoolRef

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
    SymbolicMemoryClient,
    RangeWrapper,
    PendingCheck,
    Z3Expr,
    ConstraintConjunction,
    _range_to_iterator_constraint,
)
from .data import AccessEventRecord
from ...utils.traceback_utils import extract_user_frames
from ...core.config import config as cfg

RaceDetectorT = TypeVar("RaceDetectorT", bound="RaceDetector")

AccessMode = Literal["read", "write"]


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


class SymbolicRaceDetector(RaceDetector, SymbolicMemoryClient):
    def __init__(self, abort_on_error: bool = False):
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[AccessEventRecord] = []

    # Explicit forwarders to SymbolicMemoryClient: the factory class has
    # NotImplementedError stubs (for Client's @abstractmethod contract) that
    # would otherwise shadow SymbolicMemoryClient's impls in MRO lookup.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicMemoryClient.grid_idx_callback(self, grid_idx)

    def finalize(self) -> list:
        return SymbolicMemoryClient.finalize(self)

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicMemoryClient.register_for_loop_callback(self)

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

    # ── Client callbacks ──────────────────────────────────────────────────

    def pre_run_callback(self, fn: Callable) -> bool:
        # No cross-launch dedup cache: skipping individual blocks via a cache
        # hit would leave the kernel in a partial-grid state (patch.py uses
        # `continue` on False and bypasses post_run_callback, so it can't
        # abort the whole launch). Instead, we run a single representative
        # block and abort via post_run_callback — same number of blocks on
        # every launch, no non-deterministic side effects.
        if self.need_full_grid is None:
            return True
        return self.need_full_grid

    def post_run_callback(self, fn: Callable) -> bool:
        if self.need_full_grid is None:
            self.need_full_grid = False
        if self.grid_idx == self.last_grid or not self.need_full_grid:
            self.tensors.clear()
            self.tensor_addrs.clear()
            self.tensor_names.clear()
        ret = self.need_full_grid
        self.need_full_grid = None
        return ret

    def arg_callback(self, name: str, arg: Any, arg_cvt: Any) -> None:
        if not hasattr(arg, "data_ptr"):
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
        self._record_tensor_name(arg, name)
        self.tensors.append(arg)
        self.tensor_addrs.extend(tensor_physical_addresses)

    def grid_callback(self, grid: tuple[int, ...]) -> None:
        grid = tuple(int(g) for g in grid)
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
