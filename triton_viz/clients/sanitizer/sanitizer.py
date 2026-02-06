from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    NoReturn,
    Optional,
    TypeVar,
    cast,
)
import sys

import numpy as np
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
from z3.z3 import BoolRef, ArithRef, IntNumRef

import triton.language as tl

from ...core.client import Client
from ...core.callbacks import OpCallbacks, ForLoopCallbacks
from ...core.data import (
    Op,
    RawLoad,
    Load,
    RawStore,
    Store,
    UnaryOp,
    BinaryOp,
    TernaryOp,
    ProgramId,
    Dot,
    MakeRange,
    AddPtr,
    ExpandDims,
    Broadcast,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    Splat,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
    Idiv,
    Rsqrt,
    CastImpl,
    Reshape,
    Trans,
    Join,
    Fabs,
    Ashr,
    Advance,
    FpToFp,
    Umulhi,
    CumSum,
    Bitcast,
    AtomicCas,
    AtomicRMW,
)
from ..utils import (
    check_storage_contiguous,
    get_physical_addr_from_tensor_slice,
    check_inner_stride_equal_to_one,
)
from ..symbolic_engine import (
    SymbolicExpr,
    SymbolicExprDataWrapper,
    PendingCheck,
    LoopContext,
    Z3Expr,
    ConstraintConjunction,
)
from .data import OutOfBoundsRecordZ3
from .report import (
    _get_traceback_info,
    _get_user_code_location,
    _location_to_traceback_info,
    print_oob_record,
    print_oob_record_pdb_style,
)
from ...core.config import config as cfg

SanitizerT = TypeVar("SanitizerT", bound="Sanitizer")


def _range_to_iterator_constraint(
    var: ArithRef, *, start: int, stop: int, step: int
) -> BoolRef:
    """
    Return a constraint describing values produced by `range(start, stop, step)`.

    This is used to bound a loop iterator Z3 variable for satisfiability checks.
    """
    if step == 0:
        raise ValueError("range() step cannot be 0")

    if step > 0:
        bounds = And(var >= start, var < stop)
    else:
        # Python range with negative step iterates while value > stop.
        bounds = And(var <= start, var > stop)

    abs_step = abs(step)
    if abs_step == 1:
        return bounds

    # `i in range(start, stop, step)` iff bounds hold and (i-start) is a multiple
    # of abs(step). Use a positive modulus to avoid negative-divisor semantics.
    return And(bounds, (var - start) % abs_step == 0)


@dataclass
class RangeWrapper:
    iterable: Any
    length: int
    start: int
    stop: int
    step: int

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterable)

    def __len__(self) -> int:
        return self.length


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


_UNARY_NUMPY_TO_SYM_OP: dict[Callable[..., Any], str] = {
    np.cos: "cos",
    np.exp: "exp",
    np.exp2: "exp2",
    np.abs: "abs",
    np.floor: "floor",
    np.ceil: "ceil",
    np.log: "log",
    np.log2: "log2",
    np.sqrt: "sqrt",
    np.sin: "sin",
}

_BINARY_NUMPY_TO_SYM_OP: dict[Callable[..., Any], str] = {
    np.add: "add",
    np.subtract: "sub",
    np.multiply: "mul",
    np.divide: "div",
    np.less: "less",
    np.less_equal: "less_equal",
    np.greater: "greater",
    np.greater_equal: "greater_equal",
    np.not_equal: "not_equal",
    np.equal: "equal",
    np.fmod: "mod",
    np.maximum: "maximum",
    np.minimum: "minimum",
    np.bitwise_and: "bitwise_and",
    np.bitwise_or: "bitwise_or",
    np.bitwise_xor: "bitwise_xor",
    np.right_shift: "right_shift",
    np.left_shift: "left_shift",
}


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


class SymbolicSanitizer(Sanitizer):
    def __init__(self, abort_on_error: bool = True):
        super().__init__(abort_on_error=abort_on_error)  # Initialize parent class
        self.records: list[OutOfBoundsRecordZ3] = []
        self.grid: Optional[tuple[int, ...]] = None
        self.grid_idx: Optional[tuple[int, ...]] = None
        self.tensors: list[Tensor] = []
        self.tensor_addrs: list[tuple[int, int, Tensor]] = []
        self.tensor_names: dict[int, set[str]] = {}
        self.need_full_grid: Optional[bool] = None
        self.loop_stack: list[LoopContext] = []
        self.last_grid: Optional[tuple[int, int, int]] = None
        self.cache_args: list[Any] = []
        self.cache_grid: Optional[tuple[int, ...]] = None
        self.addr_ok: Optional[BoolRef] = None
        self.pid_ok: Optional[BoolRef] = None
        self.solver: Optional[Solver] = None
        self.addr_sym: Optional[ArithRef] = None
        SymbolicExpr.set_loop_ctx_provider(
            lambda *_args, **_kwargs: self.loop_stack[-1] if self.loop_stack else None
        )

    def _collect_tensor_base(self, expr: SymbolicExpr) -> Optional[int]:
        def walk(node: SymbolicExpr) -> Optional[int]:
            if node.op == "const" and isinstance(node.dtype, tl.pointer_type):
                return node.to_py()
            for child in node.children.values():
                if child is not None:
                    base = walk(child)
                    if base is not None:
                        return base
            return None

        return walk(expr)

    def _find_tensor_for_expr(
        self, symbolic_expr: SymbolicExpr, violation_addr: int
    ) -> Tensor:
        # Prefer mapping from pointer base addresses present in the expression.
        base_candidate = self._collect_tensor_base(symbolic_expr)
        if base_candidate is not None:
            base = base_candidate
            for tensor in self.tensors:
                if tensor.data_ptr() == base:
                    return tensor
            for start, end, tensor in self.tensor_addrs:
                if start <= base <= end:
                    return tensor

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

    def _record_tensor_name(self, tensor: Tensor, name: str) -> None:
        if not name:
            return
        names = self.tensor_names.setdefault(id(tensor), set())
        names.add(name)

    def _get_tensor_name(self, tensor: Tensor) -> Optional[str]:
        names = self.tensor_names.get(id(tensor))
        if not names:
            return None
        return ", ".join(sorted(names))

    def _check_range_satisfiable(
        self,
        access_addr: Z3Expr,
        expr_constraints: ConstraintConjunction,
        symbolic_expr: SymbolicExpr,
        source_location: Optional[tuple[str, int, str]] = None,
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
                if symbolic_expr.op == "store":
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
            source_location = _get_user_code_location()
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
        symbolic_expr: Optional[SymbolicExpr] = None,
        source_location: Optional[tuple[str, int, str]] = None,
    ) -> None:
        # Use pre-captured location if available (for deferred checks in loops),
        # otherwise capture it now (for immediate checks outside loops)
        if source_location is not None:
            traceback_info = [_location_to_traceback_info(source_location)]
        else:
            traceback_info = _get_traceback_info()

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
        self.cache_grid = grid
        self.last_grid = (grid[0] - 1, grid[1] - 1, grid[2] - 1)
        self.grid = tuple(int(g) for g in grid)
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

    def grid_idx_callback(self, grid_idx: tuple[int, ...]) -> None:
        self.grid_idx = grid_idx

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        def op_program_id_overrider(axis):
            return SymbolicExpr.create("pid", axis)

        def op_raw_load_overrider(ptr, cache_modifier, eviction_policy, is_volatile):
            return op_load_overrider(
                ptr, None, None, cache_modifier, eviction_policy, is_volatile
            )

        def op_load_overrider(ptr, mask, other, *args):
            # deal with indirect loads
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                self.need_full_grid = True
                ptr = ptr.replace_subtree("load")

            if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
                self.need_full_grid = True
                mask = mask.replace_subtree("load")

            ptr_sym = SymbolicExpr.from_value(ptr)
            mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
            other_sym = SymbolicExpr.from_value(other) if other is not None else None

            ret = SymbolicExpr.create(
                "load",
                ptr_sym,
                mask_sym,
                other_sym,
            )

            # check memory access using z3 (defer in loops or check immediately)
            self._handle_access_check(ret)
            return ret

        def op_raw_store_overrider(ptr, value, cache_modifier, eviction_policy):
            return op_store_overrider(ptr, value, None, cache_modifier, eviction_policy)

        def op_store_overrider(ptr, value, mask, *args):
            # deal with indirect loads
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                self.need_full_grid = True
                ptr = ptr.replace_subtree("load")

            if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
                self.need_full_grid = True
                mask = mask.replace_subtree("load")
            ptr_sym = SymbolicExpr.from_value(ptr)
            value_sym = SymbolicExpr.from_value(value)
            mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
            ret = SymbolicExpr.create(
                "store",
                ptr_sym,
                value_sym,
                mask_sym,
            )

            # check memory access using z3 (defer in loops or check immediately)
            self._handle_access_check(ret)
            return ret

        def op_unary_op_overrider(arg, op):
            arg_sym = SymbolicExpr.from_value(arg)
            try:
                name = _UNARY_NUMPY_TO_SYM_OP[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported unary operation: {op} on {arg_sym}"
                )
            return SymbolicExpr.create(name, arg_sym)

        def op_binary_op_overrider(lhs, rhs, op):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            try:
                op_name = _BINARY_NUMPY_TO_SYM_OP[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported binary operation: {op} between {lhs_sym} and {rhs_sym}"
                )
            return SymbolicExpr.create(op_name, lhs_sym, rhs_sym)

        def op_ternary_op_overrider(lhs, rhs, other, op):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            other = SymbolicExpr.from_value(other)
            if op is np.where:
                return SymbolicExpr.create("where", lhs_sym, rhs_sym, other)
            else:
                raise NotImplementedError(
                    f"Unsupported ternary operation: {op} between {lhs_sym}, {rhs_sym} and {other}"
                )

        def op_addptr_overrider(ptr, offset):
            ptr_sym = SymbolicExpr.from_value(ptr)
            offset_sym = SymbolicExpr.from_value(offset)
            return SymbolicExpr.create("addptr", ptr_sym, offset_sym)

        def op_dot_overrider(a, b, d, input_precision, max_num_imprecise_acc):
            a_sym = SymbolicExpr.from_value(a)
            b_sym = SymbolicExpr.from_value(b)
            d_sym = SymbolicExpr.from_value(d) if d is not None else None
            return SymbolicExpr.create("dot", a_sym, b_sym, d_sym)

        def op_make_range_overrider(ret_ty, start, end):
            return SymbolicExpr.create(
                "arange",
                SymbolicExpr.from_value(ret_ty),
                SymbolicExpr.from_value(start),
                SymbolicExpr.from_value(end),
            )

        def op_expand_dims_overrider(arg, axis):
            return SymbolicExpr.create(
                "expand_dims", SymbolicExpr.from_value(arg), axis
            )

        def op_broadcast_overrider(arg, shape):
            return SymbolicExpr.create("broadcast", SymbolicExpr.from_value(arg), shape)

        def op_reduce_sum_overrider(input, axis=None, keep_dims=False, **kwargs):
            return SymbolicExpr.create(
                "sum", SymbolicExpr.from_value(input), axis, keep_dims
            )

        def op_reduce_max_overrider(input, axis=None, keep_dims=False, **kwargs):
            return SymbolicExpr.create(
                "max", SymbolicExpr.from_value(input), axis, keep_dims
            )

        def op_reduce_min_overrider(input, axis=None, keep_dims=False, **kwargs):
            return SymbolicExpr.create(
                "min", SymbolicExpr.from_value(input), axis, keep_dims
            )

        def op_splat_overrider(shape, arg):
            return SymbolicExpr.create("splat", shape, SymbolicExpr.from_value(arg))

        def op_make_block_ptr_overrider(
            base, shape, strides, offsets, tensor_shape, order
        ):
            raise NotImplementedError("MakeBlockPtr is not supported yet.")

        def op_tensor_pointer_load_overrider(
            ptr,
            boundary_check,
            padding_option,
            cache_modifier,
            eviction_policy,
            is_volatile,
        ):
            raise NotImplementedError("TensorPointerLoad is not supported yet.")

        def op_tensor_pointer_store_overrider(
            ptr, value, boundary_check, cache_modifier, eviction_policy
        ):
            raise NotImplementedError("TensorPointerStore is not supported yet.")

        def op_idiv_overrider(lhs, rhs):
            result = SymbolicExpr.from_value(lhs) // SymbolicExpr.from_value(rhs)
            return result

        def op_rsqrt_overrider(arg):
            return SymbolicExpr.create("rsqrt", SymbolicExpr.from_value(arg))

        def op_cast_impl_overrider(src, dst_type):
            return SymbolicExpr.create("cast_impl", src, dst_type)

        def op_reshape_overrider(arg, shape, allow_reorder):
            # For symbolic execution, we track the reshape operation
            # arg is the input tensor, shape is the new shape, allow_reorder is a flag
            arg_sym = SymbolicExpr.from_value(arg)
            shape_sym = SymbolicExpr.from_value(shape)
            # Create a reshape symbolic expression
            return SymbolicExpr.create("reshape", arg_sym, shape_sym)

        def op_join_overrider(lhs, rhs):
            # Join operation combines two tensors along the last axis
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            # Create a join symbolic expression
            return SymbolicExpr.create("join", lhs_sym, rhs_sym)

        def op_fabs_overrider(arg):
            arg_sym = SymbolicExpr.from_value(arg)
            return SymbolicExpr.create("fabs", arg_sym)

        def op_ashr_overrider(lhs, rhs):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            return SymbolicExpr.create("ashr", lhs_sym, rhs_sym)

        def op_advance_overrider(ptr, offsets):
            # Advance operation for block pointers
            ptr_sym = SymbolicExpr.from_value(ptr)
            offsets_sym = SymbolicExpr.from_value(offsets)
            return SymbolicExpr.create("advance", ptr_sym, offsets_sym)

        def op_umulhi_overrider(lhs, rhs):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            return SymbolicExpr.create("umulhi", lhs_sym, rhs_sym)

        def op_trans_overrider(arg, perm=[1, 0]):
            return SymbolicExpr.create("trans", SymbolicExpr.from_value(arg), perm)

        def op_cumsum_overrider(input, axis, reverse=False, dtype=None):
            return SymbolicExpr.create(
                "cumsum", SymbolicExpr.from_value(input), axis, reverse, dtype
            )

        def op_fp_to_fp_overrider(src, dst_type, rounding_mode):
            return SymbolicExpr.create(
                "fp_to_fp", SymbolicExpr.from_value(src), dst_type, rounding_mode
            )

        def op_bitcast_overrider(src, dst_type):
            src_sym = SymbolicExpr.from_value(src)
            return SymbolicExpr.create("bitcast", src_sym, dst_type)

        def op_atomic_cas_overrider(ptr, cmp, val, sem, scope):
            ptr_sym = SymbolicExpr.from_value(ptr)
            cmp_sym = SymbolicExpr.from_value(cmp)
            val_sym = SymbolicExpr.from_value(val)
            return SymbolicExpr.create("atomic_cas", ptr_sym, cmp_sym, val_sym)

        def op_atomic_rmw_overrider(rmwOp, ptr, val, mask, sem, scope):
            ptr_sym = SymbolicExpr.from_value(ptr)
            val_sym = SymbolicExpr.from_value(val)
            mask_sym = SymbolicExpr.from_value(mask)
            return SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)

        OP_TYPE_TO_OVERRIDER: dict[type[Op], Callable] = {
            ProgramId: op_program_id_overrider,
            RawLoad: op_raw_load_overrider,
            Load: op_load_overrider,
            RawStore: op_raw_store_overrider,
            Store: op_store_overrider,
            UnaryOp: op_unary_op_overrider,
            BinaryOp: op_binary_op_overrider,
            TernaryOp: op_ternary_op_overrider,
            Dot: op_dot_overrider,
            MakeRange: op_make_range_overrider,
            AddPtr: op_addptr_overrider,
            ExpandDims: op_expand_dims_overrider,
            Broadcast: op_broadcast_overrider,
            ReduceSum: op_reduce_sum_overrider,
            ReduceMax: op_reduce_max_overrider,
            ReduceMin: op_reduce_min_overrider,
            Splat: op_splat_overrider,
            MakeBlockPointer: op_make_block_ptr_overrider,
            TensorPointerLoad: op_tensor_pointer_load_overrider,
            TensorPointerStore: op_tensor_pointer_store_overrider,
            Idiv: op_idiv_overrider,
            Rsqrt: op_rsqrt_overrider,
            CastImpl: op_cast_impl_overrider,
            Reshape: op_reshape_overrider,
            Trans: op_trans_overrider,
            Join: op_join_overrider,
            Fabs: op_fabs_overrider,
            Ashr: op_ashr_overrider,
            Advance: op_advance_overrider,
            FpToFp: op_fp_to_fp_overrider,
            Umulhi: op_umulhi_overrider,
            CumSum: op_cumsum_overrider,
            Bitcast: op_bitcast_overrider,
            AtomicCas: op_atomic_cas_overrider,
            AtomicRMW: op_atomic_rmw_overrider,
        }

        if op_type in OP_TYPE_TO_OVERRIDER:
            return OpCallbacks(op_overrider=self.lock_fn(OP_TYPE_TO_OVERRIDER[op_type]))
        else:
            return OpCallbacks()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        def _materialize_loop_value(expr: Any) -> int:
            if isinstance(expr, SymbolicExpr):
                if expr.op == "const":
                    return SymbolicExprDataWrapper.coerce_int(expr.to_py())
                elif expr.has_op("load"):
                    self.need_full_grid = True
                    expr = expr.replace_subtree("load")
                    return SymbolicExprDataWrapper.coerce_int(expr.to_py())
                else:
                    z3_expr, _ = expr.eval()
                    if isinstance(z3_expr, IntNumRef):
                        return z3_expr.as_long()
                    self.need_full_grid = True
                    expr = expr.replace_subtree()
                    return SymbolicExprDataWrapper.coerce_int(expr.to_py())

            return int(expr)

        @self.lock_fn
        def _wrap_range(
            iterable,
            _lineno,
            _range_type,
            iter_args=None,
            iter_kwargs=None,
            _iter_callable=None,
        ):
            """
            Wrap range-like iterables so we can evaluate bounds once.
            The caller can pass the original range call arguments to avoid
            evaluating Python's built-in range on symbolic values.
            """
            iter_args = tuple(iter_args or ())
            iter_kwargs = iter_kwargs or {}

            if isinstance(iterable, RangeWrapper):
                return iterable

            args = tuple(SymbolicExpr.from_value(v) for v in iter_args)
            # Prefer explicit args; fall back to kwargs when args are empty.
            if not args and iter_kwargs:
                start_expr = SymbolicExpr.from_value(iter_kwargs.get("start", 0))
                stop_expr = SymbolicExpr.from_value(
                    iter_kwargs.get("stop", iter_kwargs.get("end"))
                )
                step_expr = SymbolicExpr.from_value(iter_kwargs.get("step", 1))
                if stop_expr is not None:
                    args = (start_expr, stop_expr, step_expr)

            if not args and (
                isinstance(iterable, range)
                or (
                    hasattr(iterable, "start")
                    and hasattr(iterable, "stop")
                    and hasattr(iterable, "step")
                )
            ):
                args = tuple(
                    SymbolicExpr.from_value(getattr(iterable, attr, default))
                    for attr, default in (("start", 0), ("stop", 0), ("step", 1))
                )

            if not args:
                return None

            start_expr, stop_expr, step_expr = 0, None, 1
            if len(args) == 1:
                stop_expr = args[0]
            elif len(args) == 2:
                start_expr, stop_expr = args[0], args[1]
            else:
                start_expr, stop_expr, step_expr = args[0], args[1], args[2]

            step = _materialize_loop_value(step_expr)
            start = _materialize_loop_value(start_expr)
            stop = _materialize_loop_value(stop_expr)

            concrete_range = range(start, stop, step)
            length = len(concrete_range)

            return RangeWrapper(
                concrete_range, length=length, start=start, stop=stop, step=step
            )

        @self.lock_fn
        def loop_hook_before(lineno, iterable):
            if not isinstance(iterable, RangeWrapper):
                if cfg.verbose:
                    print(
                        "not a range wrapper, skipping for-loop iterator association."
                    )
                return

            idx_z3 = Int(f"loop_i_{lineno}")
            sym = SymbolicExpr.create("const", idx_z3, tl.int32)
            idx = tl.tensor(sym, tl.int32)
            ctx = LoopContext(
                lineno,
                iterable.length,
                idx,
                idx_z3,
                start=iterable.start,
                stop=iterable.stop,
                step=iterable.step,
            )
            sym.loop_ctx = ctx
            self.loop_stack.append(ctx)
            if cfg.verbose:
                print(f"[Sanitizer] ▶ enter loop@{lineno}, len={iterable.length}")

        @self.lock_fn
        def loop_hook_iter_overrider(lineno, idx):
            if self.loop_stack and self.loop_stack[-1].lineno == lineno:
                return self.loop_stack[-1].idx
            return idx

        @self.lock_fn
        def loop_hook_after(lineno: int) -> None:
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

        return ForLoopCallbacks(
            range_wrapper_factory=_wrap_range,
            before_loop_callback=loop_hook_before,
            loop_iter_overrider=loop_hook_iter_overrider,
            after_loop_callback=loop_hook_after,
        )

    def finalize(self) -> list:
        return []


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
