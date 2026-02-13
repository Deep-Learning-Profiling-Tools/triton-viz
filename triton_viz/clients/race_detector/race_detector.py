from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Optional
from collections import defaultdict

import numpy as np
from z3 import Solver, Int, And, Or, sat, substitute
from z3.z3 import ArithRef, IntNumRef

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
from ..symbolic_engine import (
    SymbolicExpr,
    SymbolicExprDataWrapper,
    LoopContext,
)
from .data import AccessType, MemoryAccess, SymbolicMemoryAccess, RaceRecord, RaceType
from ...utils.traceback_utils import extract_user_frames


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


@dataclass
class _RangeWrapper:
    iterable: Any
    length: int
    start: int
    stop: int
    step: int

    def __iter__(self) -> Iterator[Any]:
        return iter(self.iterable)

    def __len__(self) -> int:
        return self.length


class RaceDetector(Client):
    NAME = "race_detector"

    def __init__(self):
        super().__init__()
        self._symbolic_accesses: list[SymbolicMemoryAccess] = []
        self._concrete_accesses: list[MemoryAccess] = []
        self._need_concrete_fallback: bool = False
        self._grid: tuple[int, ...] = (1, 1, 1)
        self._phase: str = "init"
        # Each entry: (OpCallbacks, original_overrider_fn)
        self._op_callbacks: list[tuple[OpCallbacks, Callable]] = []
        self._races: list[RaceRecord] = []
        self._loop_stack: list[LoopContext] = []
        SymbolicExpr.set_loop_ctx_provider(
            lambda *_a, **_kw: self._loop_stack[-1] if self._loop_stack else None
        )

    # ── Client interface ─────────────────────────────────────────

    def pre_run_callback(self, fn: Callable) -> bool:
        if self._phase == "z3_done":
            return False
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        if self._phase != "symbolic":
            return True
        # End of symbolic phase (block 0 finished)
        if self._need_concrete_fallback:
            self._concretize_block0()
            self._phase = "concrete"
            self._disable_overriders()
            return True
        else:
            self._races = self._check_symbolic_races()
            self._phase = "z3_done"
            return False

    def pre_warmup_callback(self, jit_fn: Callable, *args, **kwargs) -> bool:
        return False

    def post_warmup_callback(self, jit_fn: Callable, ret: Any) -> None:
        pass

    def arg_callback(self, name, arg, arg_cvt):
        pass

    def grid_callback(self, grid: tuple[int, ...]):
        self._grid = tuple(int(g) for g in grid)
        self._phase = "symbolic"
        self._need_concrete_fallback = False
        self._symbolic_accesses.clear()
        self._concrete_accesses.clear()
        self._races.clear()
        # Re-enable overriders (may have been disabled by a previous concrete fallback)
        for cb, orig_overrider in self._op_callbacks:
            cb.op_overrider = orig_overrider
        SymbolicExpr.ARANGE_DICT.clear()

    def grid_idx_callback(self, grid_idx: tuple[int, ...]):
        self.grid_idx = grid_idx

    def register_op_callback(self, op_type: type[Op]) -> OpCallbacks:
        # ── Symbolic overriders (active during symbolic phase) ────

        def op_program_id_overrider(axis):
            return SymbolicExpr.create("pid", axis)

        def op_raw_load_overrider(ptr, cache_modifier, eviction_policy, is_volatile):
            return op_load_overrider(
                ptr, None, None, cache_modifier, eviction_policy, is_volatile
            )

        def op_load_overrider(ptr, mask, other, *args):
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                self._need_concrete_fallback = True
                ptr = ptr.replace_subtree("load")
            if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
                self._need_concrete_fallback = True
                mask = mask.replace_subtree("load")
            ptr_sym = SymbolicExpr.from_value(ptr)
            mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
            other_sym = SymbolicExpr.from_value(other) if other is not None else None
            ret = SymbolicExpr.create("load", ptr_sym, mask_sym, other_sym)
            self._record_symbolic_access(AccessType.LOAD, ptr_sym, mask_sym)
            return ret

        def op_raw_store_overrider(ptr, value, cache_modifier, eviction_policy):
            return op_store_overrider(ptr, value, None, cache_modifier, eviction_policy)

        def op_store_overrider(ptr, value, mask, *args):
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                self._need_concrete_fallback = True
                ptr = ptr.replace_subtree("load")
            if isinstance(mask, SymbolicExpr) and mask.has_op("load"):
                self._need_concrete_fallback = True
                mask = mask.replace_subtree("load")
            ptr_sym = SymbolicExpr.from_value(ptr)
            value_sym = SymbolicExpr.from_value(value)
            mask_sym = SymbolicExpr.from_value(mask) if mask is not None else None
            ret = SymbolicExpr.create("store", ptr_sym, value_sym, mask_sym)
            self._record_symbolic_access(AccessType.STORE, ptr_sym, mask_sym)
            return ret

        def op_atomic_rmw_overrider(rmwOp, ptr, val, mask, sem, scope):
            ptr_sym = SymbolicExpr.from_value(ptr)
            val_sym = SymbolicExpr.from_value(val)
            mask_sym = SymbolicExpr.from_value(mask)
            ret = SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)
            # Record ptr_expr (not the full atomic node) to avoid Z3 NotImplementedError
            self._record_symbolic_access(AccessType.ATOMIC, ptr_sym, mask_sym)
            return ret

        def op_atomic_cas_overrider(ptr, cmp, val, sem, scope):
            ptr_sym = SymbolicExpr.from_value(ptr)
            cmp_sym = SymbolicExpr.from_value(cmp)
            val_sym = SymbolicExpr.from_value(val)
            ret = SymbolicExpr.create("atomic_cas", ptr_sym, cmp_sym, val_sym)
            self._record_symbolic_access(AccessType.ATOMIC, ptr_sym, None)
            return ret

        def op_unary_op_overrider(arg, op):
            arg_sym = SymbolicExpr.from_value(arg)
            name = _UNARY_NUMPY_TO_SYM_OP.get(op)
            if name is None:
                raise NotImplementedError(f"Unsupported unary operation: {op}")
            return SymbolicExpr.create(name, arg_sym)

        def op_binary_op_overrider(lhs, rhs, op):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            op_name = _BINARY_NUMPY_TO_SYM_OP.get(op)
            if op_name is None:
                raise NotImplementedError(f"Unsupported binary operation: {op}")
            return SymbolicExpr.create(op_name, lhs_sym, rhs_sym)

        def op_ternary_op_overrider(lhs, rhs, other, op):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            other_sym = SymbolicExpr.from_value(other)
            if op is np.where:
                return SymbolicExpr.create("where", lhs_sym, rhs_sym, other_sym)
            raise NotImplementedError(f"Unsupported ternary operation: {op}")

        def op_addptr_overrider(ptr, offset):
            return SymbolicExpr.create(
                "addptr",
                SymbolicExpr.from_value(ptr),
                SymbolicExpr.from_value(offset),
            )

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

        def op_idiv_overrider(lhs, rhs):
            return SymbolicExpr.from_value(lhs) // SymbolicExpr.from_value(rhs)

        def op_rsqrt_overrider(arg):
            return SymbolicExpr.create("rsqrt", SymbolicExpr.from_value(arg))

        def op_cast_impl_overrider(src, dst_type):
            return SymbolicExpr.create("cast_impl", src, dst_type)

        def op_reshape_overrider(arg, shape, allow_reorder):
            return SymbolicExpr.create(
                "reshape",
                SymbolicExpr.from_value(arg),
                SymbolicExpr.from_value(shape),
            )

        def op_trans_overrider(arg, perm=[1, 0]):
            return SymbolicExpr.create("trans", SymbolicExpr.from_value(arg), perm)

        def op_join_overrider(lhs, rhs):
            return SymbolicExpr.create(
                "join",
                SymbolicExpr.from_value(lhs),
                SymbolicExpr.from_value(rhs),
            )

        def op_fabs_overrider(arg):
            return SymbolicExpr.create("fabs", SymbolicExpr.from_value(arg))

        def op_ashr_overrider(lhs, rhs):
            return SymbolicExpr.create(
                "ashr",
                SymbolicExpr.from_value(lhs),
                SymbolicExpr.from_value(rhs),
            )

        def op_advance_overrider(ptr, offsets):
            return SymbolicExpr.create(
                "advance",
                SymbolicExpr.from_value(ptr),
                SymbolicExpr.from_value(offsets),
            )

        def op_fp_to_fp_overrider(src, dst_type, rounding_mode):
            return SymbolicExpr.create(
                "fp_to_fp",
                SymbolicExpr.from_value(src),
                dst_type,
                rounding_mode,
            )

        def op_umulhi_overrider(lhs, rhs):
            return SymbolicExpr.create(
                "umulhi",
                SymbolicExpr.from_value(lhs),
                SymbolicExpr.from_value(rhs),
            )

        def op_cumsum_overrider(input, axis, reverse=False, dtype=None):
            return SymbolicExpr.create(
                "cumsum",
                SymbolicExpr.from_value(input),
                axis,
                reverse,
                dtype,
            )

        def op_bitcast_overrider(src, dst_type):
            return SymbolicExpr.create(
                "bitcast", SymbolicExpr.from_value(src), dst_type
            )

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

        # ── Concrete before-callbacks (active during concrete phase) ──

        @self.lock_fn
        def before_load(ptr, mask, keys):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = mask.data.flatten().astype(bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.LOAD,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_raw_load(ptr):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = np.ones(offsets.shape, dtype=bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.LOAD,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_store(ptr, mask, keys):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = mask.data.flatten().astype(bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.STORE,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_raw_store(ptr, value):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = np.ones(offsets.shape, dtype=bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.STORE,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_atomic_rmw(rmwOp, ptr, val, mask, sem, scope):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = mask.data.flatten().astype(bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.ATOMIC,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        @self.lock_fn
        def before_atomic_cas(ptr, cmp, val, sem, scope):
            if self._phase != "concrete":
                return
            grid_idx = self.grid_idx
            if grid_idx is None:
                return
            offsets = ptr.data.flatten().astype(np.int64)
            masks = np.ones(offsets.shape, dtype=bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=AccessType.ATOMIC,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=grid_idx,
                    call_path=extract_user_frames(),
                )
            )

        # ── Map op types to callbacks ─────────────────────────────

        OVERRIDER_MAP: dict[type[Op], Callable] = {
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

        BEFORE_MAP: dict[type[Op], Callable] = {
            Load: before_load,
            RawLoad: before_raw_load,
            Store: before_store,
            RawStore: before_raw_store,
            AtomicRMW: before_atomic_rmw,
            AtomicCas: before_atomic_cas,
        }

        overrider = OVERRIDER_MAP.get(op_type)
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

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        def _materialize_loop_value(expr: Any) -> int:
            if isinstance(expr, SymbolicExpr):
                if expr.op == "const":
                    return SymbolicExprDataWrapper.coerce_int(expr.to_py())
                elif expr.has_op("load"):
                    self._need_concrete_fallback = True
                    expr = expr.replace_subtree("load")
                    return SymbolicExprDataWrapper.coerce_int(expr.to_py())
                else:
                    z3_expr, _ = expr.eval()
                    if isinstance(z3_expr, IntNumRef):
                        return z3_expr.as_long()
                    self._need_concrete_fallback = True
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
            if self._phase != "symbolic":
                return None
            iter_args = tuple(iter_args or ())
            iter_kwargs = iter_kwargs or {}
            if isinstance(iterable, _RangeWrapper):
                return iterable

            args = tuple(SymbolicExpr.from_value(v) for v in iter_args)
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
            return _RangeWrapper(
                concrete_range,
                length=len(concrete_range),
                start=start,
                stop=stop,
                step=step,
            )

        @self.lock_fn
        def loop_hook_before(lineno, iterable):
            if self._phase != "symbolic":
                return
            if not isinstance(iterable, _RangeWrapper):
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
            self._loop_stack.append(ctx)

        @self.lock_fn
        def loop_hook_iter_overrider(lineno, idx):
            if self._phase != "symbolic":
                return idx
            if self._loop_stack and self._loop_stack[-1].lineno == lineno:
                return self._loop_stack[-1].idx
            return idx

        @self.lock_fn
        def loop_hook_after(lineno: int) -> None:
            if self._phase != "symbolic":
                return
            if not self._loop_stack or self._loop_stack[-1].lineno != lineno:
                return
            self._loop_stack.pop()

        return ForLoopCallbacks(
            range_wrapper_factory=_wrap_range,
            before_loop_callback=loop_hook_before,
            loop_iter_overrider=loop_hook_iter_overrider,
            after_loop_callback=loop_hook_after,
        )

    def finalize(self) -> list:
        if self._phase == "z3_done":
            races = list(self._races)
        elif self._phase == "concrete":
            races = detect_races(self._concrete_accesses)
        else:
            races = []
        self._symbolic_accesses.clear()
        self._concrete_accesses.clear()
        self._races.clear()
        self._phase = "init"
        return races

    # ── Internal helpers ─────────────────────────────────────────

    def _record_symbolic_access(
        self,
        access_type: AccessType,
        ptr_expr: SymbolicExpr,
        mask_expr: Optional[SymbolicExpr],
    ) -> None:
        is_data_dependent = ptr_expr.has_op("load")
        if mask_expr is not None and mask_expr.has_op("load"):
            is_data_dependent = True
        if is_data_dependent:
            self._need_concrete_fallback = True
        self._symbolic_accesses.append(
            SymbolicMemoryAccess(
                access_type=access_type,
                ptr_expr=ptr_expr,
                mask_expr=mask_expr,
                is_data_dependent=is_data_dependent,
                call_path=extract_user_frames(),
            )
        )

    def _disable_overriders(self) -> None:
        for cb, _orig in self._op_callbacks:
            cb.op_overrider = None

    def _concretize_block0(self) -> None:
        for sym_access in self._symbolic_accesses:
            ptr_expr = sym_access.ptr_expr
            mask_expr = sym_access.mask_expr
            if sym_access.is_data_dependent:
                ptr_expr = ptr_expr.replace_subtree("load")
                if mask_expr is not None and mask_expr.has_op("load"):
                    mask_expr = mask_expr.replace_subtree("load")
            ptr_concrete = ptr_expr.concretize()
            offsets = ptr_concrete.data.flatten().astype(np.int64)
            if mask_expr is not None:
                mask_concrete = mask_expr.concretize()
                masks = mask_concrete.data.flatten().astype(bool)
            else:
                masks = np.ones(offsets.shape, dtype=bool)
            self._concrete_accesses.append(
                MemoryAccess(
                    access_type=sym_access.access_type,
                    ptr=0,
                    offsets=offsets,
                    masks=masks,
                    grid_idx=(0, 0, 0),
                    call_path=sym_access.call_path,
                )
            )

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

        races: list[RaceRecord] = []
        accesses = self._symbolic_accesses

        for i in range(len(accesses)):
            for j in range(i, len(accesses)):
                acc_a = accesses[i]
                acc_b = accesses[j]

                race_type = _classify_race_type(acc_a.access_type, acc_b.access_type)
                if race_type is None:
                    continue

                # Get Z3 expressions for pointers
                ptr_z3_a, ptr_constr_a = acc_a.ptr_expr.eval()
                ptr_z3_b, ptr_constr_b = acc_b.ptr_expr.eval()

                # Apply block-specific substitutions
                addr_a = _apply_sub(ptr_z3_a, sub_a)
                addr_b = _apply_sub(ptr_z3_b, sub_b)

                # Build address overlap constraint
                if isinstance(addr_a, list) and isinstance(addr_b, list):
                    overlap = Or(*[a == b for a in addr_a for b in addr_b])
                elif isinstance(addr_a, list):
                    overlap = Or(*[a == addr_b for a in addr_a])
                elif isinstance(addr_b, list):
                    overlap = Or(*[addr_a == b for b in addr_b])
                else:
                    overlap = addr_a == addr_b

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

                # Add mask constraints
                if acc_a.mask_expr is not None:
                    mask_z3_a, mask_constr_a = acc_a.mask_expr.eval()
                    mask_a_sub = _apply_sub(mask_z3_a, sub_a)
                    if isinstance(mask_a_sub, list):
                        solver.add(Or(*mask_a_sub))
                    else:
                        solver.add(mask_a_sub)
                    if mask_constr_a is not None:
                        solver.add(_apply_sub(mask_constr_a, sub_a))

                if acc_b.mask_expr is not None:
                    mask_z3_b, mask_constr_b = acc_b.mask_expr.eval()
                    mask_b_sub = _apply_sub(mask_z3_b, sub_b)
                    if isinstance(mask_b_sub, list):
                        solver.add(Or(*mask_b_sub))
                    else:
                        solver.add(mask_b_sub)
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
                    if isinstance(addr_a, list):
                        witness_addr = model.evaluate(
                            addr_a[0], model_completion=True
                        ).as_long()
                    else:
                        witness_addr = model.evaluate(
                            addr_a, model_completion=True
                        ).as_long()

                    mem_a = MemoryAccess(
                        access_type=acc_a.access_type,
                        ptr=0,
                        offsets=np.array([witness_addr]),
                        masks=np.array([True]),
                        grid_idx=block_a_idx,
                        call_path=acc_a.call_path,
                    )
                    mem_b = MemoryAccess(
                        access_type=acc_b.access_type,
                        ptr=0,
                        offsets=np.array([witness_addr]),
                        masks=np.array([True]),
                        grid_idx=block_b_idx,
                        call_path=acc_b.call_path,
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
    if is_write_a and tb == AccessType.LOAD:
        return RaceType.RAW
    if ta == AccessType.LOAD and is_write_b:
        return RaceType.RAW
    return None


def detect_races(accesses: list[MemoryAccess]) -> list[RaceRecord]:
    """Detect data races using an inverted index on byte addresses.

    A race occurs when:
    1. Two different blocks access the same byte address
    2. At least one access is a write (Store)
    3. The accesses are not both atomic
    """
    addr_to_accesses: dict[int, list[MemoryAccess]] = defaultdict(list)

    for access in accesses:
        active_offsets = access.offsets[access.masks]
        unique_offsets = np.unique(active_offsets)
        for off in unique_offsets:
            addr_to_accesses[int(off)].append(access)

    races: list[RaceRecord] = []
    seen_pairs: set[tuple[tuple[int, ...], tuple[int, ...], RaceType, int]] = set()

    for addr, access_list in addr_to_accesses.items():
        if len(access_list) < 2:
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
                        dedup_key = (pair_key[0], pair_key[1], race_type, addr)
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
    elif is_write_a and tb == AccessType.LOAD:
        return RaceType.RAW
    elif ta == AccessType.LOAD and is_write_b:
        return RaceType.RAW
    return None
