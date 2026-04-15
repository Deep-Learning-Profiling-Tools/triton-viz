from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    NoReturn,
    TypeVar,
    cast,
)
import sys

from torch import Tensor
from z3 import (
    Solver,
    Int,
    And,
    Or,
    Not,
    sat,
    BoolVal,
)
from z3.z3 import BoolRef, IntNumRef

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
from .data import OutOfBoundsRecordZ3
from ...utils.traceback_utils import (
    extract_user_frames,
    location_to_traceback_info,
)
from .report import (
    print_oob_record,
    print_oob_record_pdb_style,
)
from ...core.config import config as cfg

SanitizerT = TypeVar("SanitizerT", bound="Sanitizer")


class Sanitizer(Client):
    """
    Factory class that returns the concrete sanitizer implementation
    based on the value of ``cfg.enable_sanitizer``.
    """

    NAME = "sanitizer"

    def __new__(cls: type[SanitizerT], *args: Any, **kwargs: Any) -> SanitizerT:
        if cls is Sanitizer:
            target_cls = cast(
                type["Sanitizer"],
                SymbolicSanitizer if cfg.enable_sanitizer else NullSanitizer,
            )
            obj = object.__new__(target_cls)
            cast(Any, target_cls).__init__(obj, *args, **kwargs)
            return cast(SanitizerT, obj)
        return cast(SanitizerT, object.__new__(cls))

    def __init__(self, abort_on_error: bool = True, *args, **kwargs):
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


def _make_signature(
    addr_expr: Z3Expr,
    constraints: ConstraintConjunction,
) -> int:
    """
    Convert (addr, constraints) into a stable string signature.
    • addr_expr can be a single z3 expr or list[expr]
    • constraints is a conjunction of expr
    """
    if isinstance(addr_expr, list):
        if len(addr_expr) == 1:
            addr_hash = hash(addr_expr[0])
        else:
            # Order-stable hashing avoids O(n log n) sorting in hot paths.
            addr_hash = hash(tuple(hash(e) for e in addr_expr))
    else:
        addr_hash = hash(addr_expr)

    constr_hash = 0 if constraints is None else hash(constraints)

    return hash((addr_hash, constr_hash))


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


_fn_symbolic_cache_set: set[_FnSymbolicCache] = set()


class SymbolicSanitizer(Sanitizer, SymbolicMemoryClient):
    def __init__(self, abort_on_error: bool = True):
        super().__init__(abort_on_error=abort_on_error)
        self.records: list[OutOfBoundsRecordZ3] = []
        self.cache_args: list[Any] = []
        self.cache_grid: tuple[int, ...] | None = None

    # Explicit forwarders to SymbolicMemoryClient: the factory class has
    # NotImplementedError stubs (for Client's @abstractmethod contract) that
    # would otherwise shadow SymbolicMemoryClient's impls in MRO lookup.
    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        SymbolicMemoryClient.grid_idx_callback(self, grid_idx)

    def finalize(self) -> list:
        return SymbolicMemoryClient.finalize(self)

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return SymbolicMemoryClient.register_for_loop_callback(self)

    def _find_tensor_for_expr(
        self, symbolic_expr: SymbolicExpr, violation_addr: int
    ) -> Tensor:
        # Prefer mapping from pointer base addresses present in the expression.
        resolved = self._resolve_tensor(symbolic_expr)
        if resolved is not None:
            return resolved

        # Fall back to the closest registered segment.
        if self.tensor_addrs:

            def _distance(seg: tuple[int, int, Tensor]) -> int:
                start, end, _tensor = seg
                if violation_addr < start:
                    return start - violation_addr
                if violation_addr > end:
                    return violation_addr - end
                return 0

            return min(self.tensor_addrs, key=_distance)[2]

        # Fall back to the first registered tensor.
        if self.tensors:
            return self.tensors[0]

        raise RuntimeError("No tensor registered in SymbolicSanitizer!")

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        # Use push/pop on persistent solver
        solver = self.solver
        addr_sym = self.addr_sym
        assert solver is not None
        assert addr_sym is not None

        def _check_single_addr(addr_expr: Z3Expr) -> None:
            solver.push()
            solver.add(addr_sym == addr_expr)
            if expr_constraints is not None:
                solver.add(expr_constraints)
            if solver.check() == sat:
                # Get the model to find the violation address
                model = solver.model()
                violation_val = model.evaluate(addr_sym, model_completion=True)
                if isinstance(violation_val, IntNumRef):
                    violation_addr = violation_val.as_long()
                else:
                    raise RuntimeError(
                        "Unexpected violation address type from Z3 model!"
                    )

                # Find the tensor that this address belongs to
                tensor = self._find_tensor_for_expr(symbolic_expr, violation_addr)

                # Determine operation type from symbolic expression
                if symbolic_expr.op in ("store", "tensor_pointer_store"):
                    op_type: type[Load] | type[Store] = Store
                else:
                    op_type = Load

                # Report with symbolic expression and source location
                self._report(
                    op_type, tensor, violation_addr, symbolic_expr, source_location
                )
            solver.pop()

        if isinstance(access_addr, list):
            for addr in access_addr:
                _check_single_addr(addr)
            return

        _check_single_addr(access_addr)

    def _handle_access_check(self, expr: SymbolicExpr) -> None:
        """
        Evaluate a memory access expression and either defer it (inside a loop)
        or check it immediately (outside a loop).

        Returns nothing; duplicate addresses inside a loop are skipped.
        """
        # check memory access using z3
        z3_addr, z3_constraints = expr.eval()
        if not self.loop_stack:
            self._check_range_satisfiable(z3_addr, z3_constraints, expr)
            return

        ctx = self.loop_stack[-1]
        signature = _make_signature(z3_addr, z3_constraints)
        pending_idx = ctx.signature_cache.get(signature)
        if pending_idx is None:
            # Capture source location now while we're still in the user's tl.load/tl.store call.
            # This is a lightweight operation that only traverses frame objects.
            # The actual source line will be read later only if an error is detected.
            frames = extract_user_frames(num_frames=1)
            frame = frames[-1] if frames else None
            source_location = (
                (frame.filename, frame.lineno, frame.func_name) if frame else None
            )
            ctx.signature_cache[signature] = len(ctx.pending_checks)
            pending_check = PendingCheck(
                symbolic_expr=expr,
                addr_expr=z3_addr,
                constraints=z3_constraints,
                source_location=source_location,
            )
            ctx.pending_checks.append(pending_check)
        else:
            if cfg.verbose:
                print("[Sanitizer]  ↪ skip duplicated addr in loop")

    def _report(
        self,
        op_type: type[Load] | type[Store],
        tensor: Tensor,
        violation_address: int,
        symbolic_expr: SymbolicExpr | None = None,
        source_location: tuple[str, int, str] | None = None,
    ) -> None:
        # Use pre-captured location if available (for deferred checks in loops),
        # otherwise capture it now (for immediate checks outside loops)
        if source_location is not None:
            traceback_info = [location_to_traceback_info(source_location)]
        else:
            traceback_info = extract_user_frames()

        tensor_name = self._get_tensor_name(tensor)
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=None,
            symbolic_expr=symbolic_expr,
            tensor_name=tensor_name,
        )
        if self.abort_on_error:
            # Use the new PDB-style print function if available
            if symbolic_expr is not None or cfg.verbose:
                print_oob_record_pdb_style(oob_record, symbolic_expr)
            else:
                print_oob_record(oob_record)
            sys.exit(1)
        else:
            self.records.append(oob_record)

    def _clear_cache(self) -> None:
        self.cache_args.clear()
        self.cache_grid = None

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
            # TODO: We should init a reserved_args field per backend to filter out these args
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
                "The address sanitizer only supports contiguously stored tensors for now!"
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
        self.solver.add(Not(self.addr_ok))
        self.solver.add(self.pid_ok)
        self.addr_sym = addr

    # ── Sanitizer-specific operation overriders ─────────────────

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
        self._handle_access_check(ret)
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
        self._handle_access_check(ret)
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
        self._handle_access_check(ret)
        return ret

    def _op_tensor_pointer_store_overrider(
        self, ptr, value, boundary_check, cache_modifier, eviction_policy
    ):
        ptr_sym = SymbolicExpr.from_value(ptr)
        value_sym = SymbolicExpr.from_value(value)
        ret = SymbolicExpr.create(
            "tensor_pointer_store", ptr_sym, value_sym, boundary_check
        )
        self._handle_access_check(ret)
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

    # ── For-loop hook overrides ──────────────────────────────────

    def _loop_hook_before(self, lineno, iterable):
        if not isinstance(iterable, RangeWrapper):
            if cfg.verbose:
                print("not a range wrapper, skipping for-loop iterator association.")
            return
        super()._loop_hook_before(lineno, iterable)
        if cfg.verbose:
            print(f"[Sanitizer] ▶ enter loop@{lineno}, len={iterable.length}")

    def _loop_hook_after(self, lineno: int) -> None:
        if not self.loop_stack or self.loop_stack[-1].lineno != lineno:
            return
        ctx = self.loop_stack.pop()
        # Execute pending checks that were deferred during loop execution
        solver = self.solver
        addr_sym = self.addr_sym
        assert solver is not None
        assert addr_sym is not None
        all_iterator_constraints: list[BoolRef] = []
        if ctx.pending_checks:
            solver.push()
            # Add constraint for the current (innermost) loop
            iterator_constraint = _range_to_iterator_constraint(
                ctx.idx_z3, start=ctx.start, stop=ctx.stop, step=ctx.step
            )
            solver.add(iterator_constraint)
            all_iterator_constraints.append(iterator_constraint)

            # Also add constraints for all outer loops that are still active.
            # This is critical for nested loops where the address expression
            # depends on outer loop variables.
            for outer_ctx in self.loop_stack:
                outer_constraint = _range_to_iterator_constraint(
                    outer_ctx.idx_z3,
                    start=outer_ctx.start,
                    stop=outer_ctx.stop,
                    step=outer_ctx.step,
                )
                solver.add(outer_constraint)
                all_iterator_constraints.append(outer_constraint)

        for pending_check in ctx.pending_checks:
            addr_expr = pending_check.addr_expr
            expr_constraints = pending_check.constraints
            symbolic_expr = pending_check.symbolic_expr
            # Use the source location captured when the check was created,
            # not the current location (which would be the loop exit point)
            source_location = pending_check.source_location

            if cfg.verbose:
                print(
                    "[Sanitizer] ▶ checking:",
                    addr_expr,
                    f" with iterator constraints: {all_iterator_constraints} ",
                    f" and expression-related constraints: {expr_constraints} ",
                )

            self._check_range_satisfiable(
                addr_expr,
                expr_constraints,
                symbolic_expr,
                source_location,
            )
        if ctx.pending_checks:
            solver.pop()

        if cfg.verbose:
            print(
                f"[Sanitizer] ▶ leave loop@{lineno} end. "
                f"(checked {len(ctx.pending_checks)} unique addr patterns)"
            )


class NullSanitizer(Sanitizer):
    """
    A do-nothing object returned when the sanitizer backend is 'off'.
    Any attribute access raises an explicit error so misuse is obvious.
    """

    def __init__(self, abort_on_error: bool = True, *args: Any, **kwargs: Any):
        super().__init__(abort_on_error=abort_on_error)

    def _disabled(self, method: str) -> NoReturn:
        raise RuntimeError(
            f"[NullSanitizer] '{method}' was called, "
            "but sanitizer backend is off; no functionality is available."
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
