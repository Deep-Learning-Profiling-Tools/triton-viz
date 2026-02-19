from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    Any,
    ClassVar,
    Optional,
    Union,
    TypeAlias,
    cast,
)

import numpy as np
from anytree import Node, RenderTree
from z3 import (
    Int,
    IntVal,
    If,
    Sum,
    And,
    Or,
    simplify,
    Int2BV,
    BV2Int,
    BitVecRef,
    BoolVal,
)
from z3.z3 import BoolRef, ArithRef, IntNumRef, ExprRef, Tactic, Probe

import triton.language as tl
from triton.runtime.interpreter import TensorHandle, _get_np_dtype

from ..core.client import Client
from ..core.callbacks import OpCallbacks, ForLoopCallbacks
from ..core.data import (
    Op,
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
    RawLoad,
    RawStore,
)


Z3Expr: TypeAlias = Union[ExprRef, list[ExprRef], Tactic, Probe]
ConstraintExpr: TypeAlias = Union[ExprRef, bool, int, float]
ConstraintConjunction: TypeAlias = Optional[BoolRef]


def _constraint_to_bool(expr: ConstraintExpr) -> BoolRef:
    if isinstance(expr, BoolRef):
        return expr
    if isinstance(expr, bool):
        return BoolVal(expr)
    if isinstance(expr, IntNumRef):
        return BoolVal(expr.as_long() != 0)
    if isinstance(expr, (int, float)):
        return BoolVal(int(expr) != 0)
    return cast(BoolRef, expr != 0)


def _iter_constraints_to_bool(
    expr_constraints: Sequence[ConstraintExpr],
) -> Iterator[BoolRef]:
    for constraint in expr_constraints:
        yield _constraint_to_bool(constraint)


def _and_constraints(
    *constraints: Optional[ConstraintExpr | Sequence[ConstraintExpr]],
) -> ConstraintConjunction:
    parts: list[BoolRef] = []
    for constraint in constraints:
        if constraint is None:
            continue
        if isinstance(constraint, (list, tuple)):
            parts.extend(_iter_constraints_to_bool(constraint))
            continue
        parts.append(_constraint_to_bool(constraint))

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return And(*parts)


@dataclass
class PendingCheck:
    symbolic_expr: SymbolicExpr
    addr_expr: Z3Expr
    constraints: ConstraintConjunction
    # Lightweight source location: (filename, lineno, func_name)
    # Captured immediately to preserve accurate line info for deferred checks
    source_location: Optional[tuple[str, int, str]] = None


@dataclass
class LoopContext:
    lineno: int
    length: int
    idx: tl.tensor
    idx_z3: ArithRef
    start: int = 0
    stop: int = 0
    step: int = 1
    signature_cache: dict[int, int] = field(default_factory=dict)
    pending_checks: list[PendingCheck] = field(default_factory=list)


class SymbolicExprDataWrapper:
    """
    This wrapper is used as a workaround of triton interpreter legacy code.
    In def _get_bool(self) of class tensor,
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Since we replaced TensorHandle with SymbolicExpr,
    we need to wrap SymbolicExpr with a class that has size attribute, and data.size != 1.
    """

    def __init__(self, symbolic_expr: SymbolicExpr, value: Optional[str] = None):
        self._value = value
        self.symbolic_expr = symbolic_expr

    def invalidate(self) -> None:
        self._value = None

    def _ensure_value(self) -> str:
        if self._value is None:
            self._value = str(self.symbolic_expr)
        return self._value

    @property
    def value(self) -> str:
        return self._ensure_value()

    @value.setter
    def value(self, v: str) -> None:
        self._value = v

    @property
    def size(self) -> int:
        shape = self.symbolic_expr.shape
        if len(shape) == 0:
            return 1
        return math.prod(shape)

    @staticmethod
    def coerce_int(val: Any) -> int:
        if isinstance(val, IntNumRef):
            return val.as_long()
        if isinstance(val, (int, np.integer, bool)):
            return int(val)
        if isinstance(val, float):
            return int(val)
        if isinstance(val, TensorHandle):
            if val.data.size != 1:
                raise ValueError(
                    "Expected scalar TensorHandle, got size " f"{val.data.size}"
                )
            return int(val.data.item())
        if isinstance(val, np.ndarray):
            if val.size != 1:
                raise ValueError(f"Expected scalar ndarray, got size {val.size}")
            return int(val.item())
        raise ValueError(
            f"SymbolicExprDataWrapper cannot coerce type {type(val)} to int"
        )

    def __int__(self) -> int:
        int_val, _ = self.symbolic_expr.eval()
        if isinstance(int_val, list):
            int_val = int_val[0]
        return self.coerce_int(int_val)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


class SymbolicExpr:
    BASIC_OPS: ClassVar[tuple[str, ...]] = ("const", "pid", "arange")
    INDIRECT_OPS: ClassVar[tuple[str, ...]] = (
        "load",
        "store",
        "tensor_pointer_load",
        "tensor_pointer_store",
    )
    UNARY_OPS: ClassVar[tuple[str, ...]] = (
        "cos",
        "exp",
        "exp2",
        "abs",
        "fabs",
        "floor",
        "ceil",
        "log",
        "log2",
        "sqrt",
        "sin",
        "rsqrt",
    )
    BINARY_OP_SYMBOL_TABLE: ClassVar[dict[str, str]] = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "idiv": "//",
        "mod": "%",
        "less": "<",
        "less_equal": "<=",
        "greater": ">",
        "greater_equal": ">=",
        "not_equal": "!=",
        "equal": "==",
        "maximum": "max",
        "minimum": "min",
        "bitwise_and": "&",
        "bitwise_or": "|",
        "bitwise_xor": "^",
        "right_shift": ">>",
        "left_shift": "<<",
        "ashr": ">>>",
        "umulhi": "umulhi",
    }
    BINARY_OPS: ClassVar[tuple[str, ...]] = tuple(BINARY_OP_SYMBOL_TABLE.keys())
    TERNARY_OPS: ClassVar[tuple[str, ...]] = ("where",)
    REDUCE_OPS: ClassVar[tuple[str, ...]] = ("sum", "max", "min", "dot")
    SCAN_OPS: ClassVar[tuple[str, ...]] = ("cumsum",)
    POINTER_OPS: ClassVar[tuple[str, ...]] = ("make_block_ptr", "addptr", "advance")
    RESHAPE_OPS: ClassVar[tuple[str, ...]] = (
        "splat",
        "expand_dims",
        "broadcast",
        "reshape",
        "join",
        "trans",
    )
    CAST_OPS: ClassVar[tuple[str, ...]] = ("cast_impl", "bitcast", "fp_to_fp")
    ATOMIC_OPS: ClassVar[tuple[str, ...]] = ("atomic_cas", "atomic_rmw")
    SUPPORTED_OPS: ClassVar[tuple[str, ...]] = (
        BASIC_OPS
        + INDIRECT_OPS
        + UNARY_OPS
        + BINARY_OPS
        + TERNARY_OPS
        + REDUCE_OPS
        + POINTER_OPS
        + RESHAPE_OPS
        + CAST_OPS
        + SCAN_OPS
        + ATOMIC_OPS
    )

    PID0: ClassVar[ArithRef] = Int("pid_0")
    PID1: ClassVar[ArithRef] = Int("pid_1")
    PID2: ClassVar[ArithRef] = Int("pid_2")

    ARANGE_DICT: ClassVar[
        dict[tuple[int, int], tuple[ArithRef, ConstraintConjunction]]
    ] = {}
    _OP_CLASS_MAP: ClassVar[dict[str, type[SymbolicExpr]]] = {}

    @classmethod
    def register_op_class(
        cls, op_cls: type[SymbolicExpr], op_types: tuple[str, ...]
    ) -> None:
        for op_type in op_types:
            cls._OP_CLASS_MAP[op_type] = op_cls

    @classmethod
    def create(cls, op: str, *args: Any) -> SymbolicExpr:
        op_cls = cls._OP_CLASS_MAP.get(op)
        if op_cls is None:
            raise NotImplementedError(f"No operator class registered for {op}")
        return op_cls(op, *args)

    def __init__(self, op: str):
        """
        :param op: Operation type, e.g. "const", "add", "sub", "mul", "div", "pid", "arange"
        """
        assert op in self.SUPPORTED_OPS, f"Unsupported op: {op}"
        self.op = op
        # Tensor handle attributes, including `attr`, `dtype`, and `data`
        self.attr: dict[str, Any] = {}
        self.dtype: Optional[tl.core.dtype] = None

        # Functions and arguments for concretization
        self.concrete_fn: Optional[Callable[..., Any]] = None

        # deal with args
        self.children: dict[str, Optional[SymbolicExpr]] = {}

        # for-loop iterator association
        self.loop_ctx: Optional[LoopContext] = None

        # z3
        self.z3: Optional[Z3Expr] = None

        self.constraints: ConstraintConjunction = None
        self._simplified_z3: Optional[Z3Expr] = None
        self._simplified_constraints: Optional[ConstraintConjunction] = None
        self._has_op_cache: dict[str, bool] = {}
        self._data_wrapper: Optional[SymbolicExprDataWrapper] = None

    @property
    def shape(self) -> tuple[int, ...]:
        if not self.dtype:
            return ()
        if isinstance(self.dtype, tl.block_type):
            return tuple(int(x) for x in self.dtype.shape)
        elif isinstance(self.dtype, tl.pointer_type):
            return tuple(int(x) for x in self.dtype.element_ty.shape)
        return ()

    def add_child(self, name: str, value: Any) -> None:
        child = SymbolicExpr.from_value(value) if value is not None else None
        self.children[name] = child
        setattr(self, name, child)
        self._has_op_cache.clear()
        self._simplified_z3 = None
        self._simplified_constraints = None
        if self._data_wrapper is not None:
            self._data_wrapper.invalidate()

    def __add__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("add", self, other)

    def __sub__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("sub", self, other)

    def __mul__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("mul", self, other)

    def __truediv__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("div", self, other)

    def __floordiv__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("idiv", self, other)

    def __mod__(self, other: SymbolicExpr) -> SymbolicExpr:
        return SymbolicExpr.create("mod", self, other)

    def __lt__(self, other: object) -> Any:
        return SymbolicExpr.create("less", self, other)

    def __le__(self, other: object) -> Any:
        return SymbolicExpr.create("less_equal", self, other)

    def __ne__(self, other: object) -> Any:
        return SymbolicExpr.create("not_equal", self, other)

    def __eq__(self, other: object) -> Any:
        return SymbolicExpr.create("equal", self, other)

    def _to_anytree(self) -> Node:
        """Convert this SymbolicExpr into an anytree Node."""
        # Generate the node label
        label = self._node_label()
        root = Node(label)
        # Recursively add child nodes
        for child_key, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                Node(f"{child_key}: None", parent=root)
                continue
            child_node = child_symbolic_expr._to_anytree()
            child_node.name = f"{child_key}: {child_node.name}"
            child_node.parent = root

        return root

    def _node_label(self) -> str:
        """Generate a short label for this node."""
        label = self._node_label_core()
        label = f"{label} [dtype={self.dtype}]"

        return label

    def _node_label_core(self) -> str:
        return self.op

    triton_scala_dtypes: ClassVar[tuple[tl.core.dtype, ...]] = (
        tl.int1,
        tl.int8,
        tl.int16,
        tl.int32,
        tl.int64,
        tl.uint8,
        tl.uint16,
        tl.uint32,
        tl.uint64,
        tl.float16,
        tl.float32,
        tl.float64,
    )
    builtin_scala_types: ClassVar[tuple[type, ...]] = (int, float)
    tuple_types: ClassVar[tuple[type, ...]] = (tl.core.tuple, tuple, list)

    @staticmethod
    def _infer_literal_dtype(var: Any) -> tl.core.dtype | tl.pointer_type:
        if isinstance(var, tl.core.tensor):
            var = var.handle
        if isinstance(var, TensorHandle):
            if len(var.data) != 1:
                raise ValueError(
                    f"Unsupported var.data: {var.data} with length more than one!"
                )
            if var.dtype in SymbolicExpr.triton_scala_dtypes:  # if an immediate
                return var.dtype
            if isinstance(var.dtype, tl.pointer_type):  # if a pointer
                return var.dtype
        if isinstance(var, tl.core.dtype):
            return var
        if isinstance(var, SymbolicExpr.tuple_types):
            seq = cast(Sequence[Any], var)
            if len(seq) == 0:
                raise ValueError("Cannot infer dtype from an empty tuple/list.")
            first_dtype = SymbolicExpr._infer_literal_dtype(seq[0])
            for v in seq[1:]:  # assume only one consistent dtype in the tuple
                if SymbolicExpr._infer_literal_dtype(v) != first_dtype:
                    raise ValueError(
                        f"All elements in the tuple must have the same dtype, but found {first_dtype} and {SymbolicExpr.from_value(v).dtype}"
                    )
            return first_dtype
        if isinstance(var, SymbolicExpr.builtin_scala_types):
            return tl.int32 if isinstance(var, int) else tl.float32
        raise ValueError(f"Unsupported type: {type(var)}")

    # Stored on the class and may be accessed through either the class or an instance;
    # therefore the callable must tolerate an extra bound argument (self/cls).
    loop_ctx_provider: ClassVar[Optional[Callable[..., Optional[LoopContext]]]] = None

    @classmethod
    def set_loop_ctx_provider(cls, fn: Callable[..., Optional[LoopContext]]) -> None:
        """Register a function that, on each call,
        returns the current active `LoopContext` (or `None` if none exists)."""
        cls.loop_ctx_provider = fn

    @classmethod
    def from_value(cls, var: Any) -> SymbolicExpr:
        """Create a SymbolicExpr from a Python value."""
        if isinstance(var, tl.core.tensor):  # if a triton tensor
            var = var.handle  # get its handle

        if isinstance(var, cls):  # if already SymbolicExpr
            return var

        dtype = SymbolicExpr._infer_literal_dtype(var)  # get the triton dtype

        if isinstance(var, SymbolicExpr.tuple_types):  # if a tuple
            seq = cast(Sequence[Any], var)
            return cls.create("const", tuple(seq), dtype)
        if isinstance(var, TensorHandle):  # if a TensorHandle
            if len(var.data) != 1:
                raise ValueError(
                    "SymbolicExpr.from_value only supports scalar TensorHandle!"
                )
            return cls.create("const", var.data.item(), dtype)
        if isinstance(
            var, SymbolicExpr.builtin_scala_types
        ):  # if a python builtin type
            return cls.create("const", var, dtype)
        if isinstance(var, tl.core.dtype):
            # If it's a triton dtype, we can create a const node with it
            return cls.create("const", var, dtype)

        raise ValueError("Unknown type:", type(var))

    def eval(
        self, simplify_constraints: bool = True
    ) -> tuple[Z3Expr, ConstraintConjunction]:
        """
        Returns a tuple (expr, constraints):
        - expr: Z3 expression (or list of expressions) corresponding to the root node
        - constraints: conjunction of constraint expressions (Z3 Bool or scalars),
          or None when there are no constraints,
          recording all range constraints created by program_id and arange
        """
        expr, constraints = self._to_z3()

        if self._simplified_z3 is None:
            if isinstance(expr, list):
                self._simplified_z3 = [simplify(e) for e in expr]
            else:
                self._simplified_z3 = simplify(expr)

        if constraints is not None and simplify_constraints:
            if self._simplified_constraints is None:
                # Z3 stubs type simplify(...) as ExprRef even when input is BoolRef.
                self._simplified_constraints = cast(BoolRef, simplify(constraints))
            return self._simplified_z3, self._simplified_constraints

        return self._simplified_z3, constraints

    def to_py(self) -> Any:
        """Return a Python value for this expression."""
        raise NotImplementedError("to_py must be implemented by subclasses")

    def concretize(self) -> Any:
        """Return a concrete TensorHandle for this expression."""
        raise NotImplementedError(f"Concretize for op {self.op} is not implemented")

    def replace_subtree(self, anchor_op: Optional[str] = None) -> SymbolicExpr:
        """
        Post-order traversal that replaces *all* sub-trees with constant nodes
        produced by `concretize`.
        """
        for name, child in list(self.children.items()):
            if child is None:
                continue
            new_child = child.replace_subtree(anchor_op)
            self.add_child(name, new_child)

        # inplace replace to "const" node
        if anchor_op is None or (
            self.op == anchor_op
            and all(
                (child is None) or (not child.has_op(anchor_op))
                for child in self.children.values()
            )
        ):
            concrete = self.concretize()
            if not isinstance(concrete, TensorHandle):
                raise TypeError(f"Unexpected dtype: {type(concrete)}!")

            self = SymbolicExpr.create("const", concrete, concrete.dtype)

        return self

    def has_op(self, op_name: str) -> bool:
        """Return True when the subtree contains an op with the given name."""
        cached = self._has_op_cache.get(op_name)
        if cached is not None:
            return cached
        if self.op == op_name:
            self._has_op_cache[op_name] = True
            return True
        for _, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                continue
            if child_symbolic_expr.has_op(op_name):
                self._has_op_cache[op_name] = True
                return True
        self._has_op_cache[op_name] = False
        return False

    def to_tree_str(self) -> str:
        """
        Render the AST as an ASCII tree using anytree.RenderTree.
        """
        root = self._to_anytree()
        lines = []
        for prefix, _, node in RenderTree(root):
            lines.append(f"{prefix}{node.name}")
        return "\n" + "\n".join(lines)

    # Tensor handle methods, not used in sanitizer
    # TODO: inherits a tensor handle protocol?
    def set_attr(self, key, value):
        self.attr[key] = value

    def __str__(self) -> str:
        return self.to_tree_str()

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def data(self) -> SymbolicExprDataWrapper:
        """Return a wrapper suitable for external consumers."""
        if self._data_wrapper is None:
            self._data_wrapper = SymbolicExprDataWrapper(self)
        return self._data_wrapper

    def _to_z3(self) -> tuple[Z3Expr, ConstraintConjunction]:
        if self.z3 is not None:
            return self.z3, self.constraints

        self.z3, self.constraints = self._to_z3_impl()
        if self.z3 is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return self.z3, self.constraints

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class ConstSymbolicExpr(SymbolicExpr):
    value: Any

    def __init__(self, op: str, value: Any, dtype: tl.core.dtype | tl.pointer_type):
        super().__init__(op)
        self.value = value
        self.dtype = dtype

    def _node_label_core(self) -> str:
        return f"const={self.value}"

    def to_py(self) -> Any:
        """
        Valid only for nodes with op == 'const':
        - If `value` is a TensorHandle:
            • Scalar  -> return int/float
            • Multi-element -> return a Python list
        - Otherwise, return the original Python object
          (e.g., int, float, tuple, list, etc.).
        """
        v = self.value
        if isinstance(v, TensorHandle):
            if len(v.data) == 1:
                return v.data.item()  # scalar case
            else:
                return v.data.tolist()  # multi-element case
        return v

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        value = self.value
        if isinstance(value, TensorHandle):
            value = self.value.data

        if self.loop_ctx:  # if the self is a loop iterator
            z3_expr: Z3Expr = self.loop_ctx.idx_z3
        elif isinstance(
            value, np.ndarray
        ):  # only const nodes can be created with ndarray
            z3_expr = [IntVal(int(v)) for v in value.flat]
        elif isinstance(value, tuple):
            z3_expr = [IntVal(int(v)) for v in value]
        elif isinstance(value, (int, float)):
            # Convert to int for Z3 - Z3's IntVal cannot parse float strings
            z3_expr = IntVal(int(value))
        else:
            # For other values, use 0 as a placeholder (e.g., for optional mask/other)
            z3_expr = IntVal(0)
        return z3_expr, None

    def concretize(self) -> Any:
        dtype = self.dtype
        if dtype is None:
            raise RuntimeError("const node is missing dtype information")

        if isinstance(self.value, (SymbolicExpr.builtin_scala_types, tl.pointer_type)):
            return TensorHandle(
                np.array([self.value], dtype=_get_np_dtype(dtype)),
                dtype,
            )
        elif isinstance(self.value, SymbolicExpr.tuple_types):
            seq = cast(Sequence[Any], self.value)
            np_array = np.array(seq, dtype=_get_np_dtype(dtype))
            return TensorHandle(np_array, dtype)
        elif isinstance(self.value, TensorHandle):
            return self.value

        raise RuntimeError(f"Unsupported const value type: {type(self.value)}")


class PidSymbolicExpr(SymbolicExpr):
    axis: SymbolicExpr

    def __init__(self, op: str, axis: Any):
        super().__init__(op)
        self.add_child("axis", axis)
        self.dtype = tl.int32

    def _node_label_core(self) -> str:
        axis_val = self.axis.to_py()
        return f"pid_{axis_val}"

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        axis_val = self.axis.to_py()
        if axis_val == 0:
            return SymbolicExpr.PID0, None
        if axis_val == 1:
            return SymbolicExpr.PID1, None
        return SymbolicExpr.PID2, None

    def concretize(self) -> Any:
        return self.concrete_fn(self.axis.to_py())  # type: ignore


class ArangeSymbolicExpr(SymbolicExpr):
    ret_ty: ConstSymbolicExpr
    start: ConstSymbolicExpr
    end: ConstSymbolicExpr

    def __init__(self, op: str, ret_ty: Any, start: Any, end: Any):
        super().__init__(op)
        self.add_child("ret_ty", ret_ty)
        self.add_child("start", start)
        self.add_child("end", end)
        # Program ID / arange are always int32
        start_const = cast(ConstSymbolicExpr, self.start)
        end_const = cast(ConstSymbolicExpr, self.end)
        self.dtype = tl.block_type(tl.int32, [end_const.value - start_const.value])

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        start = self.start.to_py()
        end = self.end.to_py()
        key = (start, end)
        if key in SymbolicExpr.ARANGE_DICT:
            return SymbolicExpr.ARANGE_DICT[key]
        name = f"arange_{start}_{end}"
        v = Int(name)
        constraints = _and_constraints(v >= start, v < end)
        SymbolicExpr.ARANGE_DICT[key] = (v, constraints)
        return v, constraints

    def concretize(self) -> Any:
        return self.concrete_fn(
            self.ret_ty.to_py(), self.start.to_py(), self.end.to_py()
        )  # type: ignore


class IndirectSymbolicExprBase(SymbolicExpr):
    ptr: SymbolicExpr
    mask: Optional[SymbolicExpr]
    other: Optional[SymbolicExpr]

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        ptr_expr = self.ptr
        ptr, constraints_ptr = ptr_expr._to_z3()
        mask_expr = self.mask
        mask_constraint: Optional[ConstraintExpr | Sequence[ConstraintExpr]] = None
        if mask_expr is not None:
            mask, _ = mask_expr._to_z3()
            mask_constraint = mask
        return ptr, _and_constraints(constraints_ptr, mask_constraint)

    def concretize(self) -> Any:
        ptr_concrete = self.ptr.concretize()
        if self.mask is None:
            mask_concrete = TensorHandle(
                np.ones_like(ptr_concrete.data, dtype=bool), tl.int1
            )
        else:
            mask_concrete = self.mask.concretize()

        if self.other is None:
            other_concrete = None
        else:
            other_concrete = self.other.concretize()

        return self.concrete_fn(
            ptr_concrete,
            mask_concrete,
            other_concrete,
            None,  # cache_modifier
            None,  # eviction_policy
            None,  # is_volatile
        )  # type: ignore


class LoadSymbolicExpr(IndirectSymbolicExprBase):
    def __init__(self, op: str, ptr: Any, mask: Any = None, other: Any = None):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("mask", mask)
        self.add_child("other", other)
        self.dtype = self.ptr.dtype.element_ty  # type: ignore


class StoreSymbolicExpr(IndirectSymbolicExprBase):
    value: SymbolicExpr

    def __init__(
        self,
        op: str,
        ptr: Any,
        value: Any,
        mask: Any = None,
        other: Any = None,
    ):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("value", value)
        self.add_child("mask", mask)
        self.add_child("other", other)


class UnarySymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr

    def __init__(self, op: str, arg: Any):
        if op not in SymbolicExpr.UNARY_OPS:
            raise NotImplementedError(f"Unsupported unary op: {op}")
        super().__init__(op)
        self.add_child("arg", arg)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        val, constraints = self.arg._to_z3()
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Unary op {self.op} is not implemented")
        return handler(val), constraints

    @staticmethod
    def _abs(val) -> Z3Expr:
        return If(val >= 0, val, -val)

    _Z3_BUILDERS: ClassVar[dict[str, Callable[[Z3Expr], Z3Expr]]] = {
        "abs": _abs,
        "fabs": _abs,
    }


class BinarySymbolicExpr(SymbolicExpr):
    lhs: SymbolicExpr
    rhs: SymbolicExpr

    def __init__(self, op: str, lhs: Any, rhs: Any):
        if op not in SymbolicExpr.BINARY_OPS:
            raise NotImplementedError(f"Unsupported binary op: {op}")
        super().__init__(op)
        self.add_child("lhs", lhs)
        self.add_child("rhs", rhs)
        self.dtype = self.lhs.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        lhs, constraints_lhs = self.lhs._to_z3()
        rhs, constraints_rhs = self.rhs._to_z3()
        constraints = _and_constraints(constraints_lhs, constraints_rhs)

        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return handler(self, lhs, rhs), constraints

    def concretize(self) -> Any:
        lhs_concrete = self.lhs.concretize()
        rhs_concrete = self.rhs.concretize()
        np_op = self._NUMPY_OPS.get(self.op, None)
        if np_op is None:
            raise NotImplementedError(f"Concretize for op {self.op} is not implemented")
        return self.concrete_fn(lhs_concrete, rhs_concrete, np_op)  # type: ignore

    @staticmethod
    def _apply_binop(op_func, left, right):
        lhs_is_list = isinstance(left, list)
        rhs_is_list = isinstance(right, list)
        if lhs_is_list and rhs_is_list:
            if len(left) != len(right):
                raise ValueError(
                    f"List operands must have same length: {len(left)} vs {len(right)}"
                )
            return [op_func(li, ri) for li, ri in zip(left, right)]
        if lhs_is_list:
            return [op_func(li, right) for li in left]
        if rhs_is_list:
            return [op_func(left, ri) for ri in right]
        return op_func(left, right)

    @staticmethod
    def _infer_bitwidth(expr):
        dtype = getattr(expr, "dtype", None)
        if dtype is None:
            return None
        if hasattr(dtype, "primitive_bitwidth"):
            return dtype.primitive_bitwidth
        if hasattr(dtype, "element_ty") and hasattr(
            dtype.element_ty, "primitive_bitwidth"
        ):
            return dtype.element_ty.primitive_bitwidth
        return None

    @classmethod
    def _to_bv(cls, v, bitwidth):
        if isinstance(v, list):
            return [cls._to_bv(x, bitwidth) for x in v]
        if isinstance(v, BoolRef):
            v = If(v, IntVal(1), IntVal(0))
        if isinstance(v, BitVecRef):
            return v
        return Int2BV(v, bitwidth)

    @classmethod
    def _from_bv(cls, v):
        if isinstance(v, list):
            return [cls._from_bv(x) for x in v]
        return BV2Int(v, is_signed=False)

    def _op_add(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a + b, lhs, rhs)

    def _op_sub(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a - b, lhs, rhs)

    def _op_mul(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a * b, lhs, rhs)

    def _op_idiv(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a / b, lhs, rhs)

    def _op_mod(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a % b, lhs, rhs)

    def _op_less(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a < b, lhs, rhs)

    def _op_less_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a <= b, lhs, rhs)

    def _op_greater(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a > b, lhs, rhs)

    def _op_greater_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a >= b, lhs, rhs)

    def _op_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a == b, lhs, rhs)

    def _op_not_equal(self, lhs, rhs):
        return self._apply_binop(lambda a, b: a != b, lhs, rhs)

    def _op_maximum(self, lhs, rhs):
        return self._apply_binop(lambda a, b: If(a >= b, a, b), lhs, rhs)

    def _op_minimum(self, lhs, rhs):
        return self._apply_binop(lambda a, b: If(a <= b, a, b), lhs, rhs)

    def _op_bitwise_and(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _bit_and(a, b):
            if isinstance(a, BoolRef) and isinstance(b, BoolRef):
                return And(a, b)
            return self._from_bv(self._to_bv(a, bitwidth) & self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_and, lhs, rhs)

    def _op_bitwise_or(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _bit_or(a, b):
            if isinstance(a, BoolRef) and isinstance(b, BoolRef):
                return Or(a, b)
            return self._from_bv(self._to_bv(a, bitwidth) | self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_or, lhs, rhs)

    def _op_bitwise_xor(self, lhs, rhs):
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(lhs_expr)
            or self._infer_bitwidth(rhs_expr)
            or 64
        )

        def _bit_xor(a, b):
            return self._from_bv(self._to_bv(a, bitwidth) ^ self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_xor, lhs, rhs)

    def _op_ashr(self, lhs, rhs):
        raise NotImplementedError("Arithmetic shift right is not implemented in Z3")

    _NUMPY_OPS: ClassVar[dict[str, Callable[[Any, Any], Any]]] = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "mod": np.fmod,
        "less": np.less,
        "less_equal": np.less_equal,
        "greater": np.greater,
        "greater_equal": np.greater_equal,
        "equal": np.equal,
        "not_equal": np.not_equal,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "bitwise_and": np.bitwise_and,
        "bitwise_or": np.bitwise_or,
        "bitwise_xor": np.bitwise_xor,
        "right_shift": np.right_shift,
        "left_shift": np.left_shift,
    }

    _Z3_BUILDERS: ClassVar[
        dict[str, Callable[[BinarySymbolicExpr, Z3Expr, Z3Expr], Z3Expr]]
    ] = {
        "add": _op_add,
        "sub": _op_sub,
        "mul": _op_mul,
        "idiv": _op_idiv,
        "mod": _op_mod,
        "less": _op_less,
        "less_equal": _op_less_equal,
        "greater": _op_greater,
        "greater_equal": _op_greater_equal,
        "equal": _op_equal,
        "not_equal": _op_not_equal,
        "maximum": _op_maximum,
        "minimum": _op_minimum,
        "bitwise_and": _op_bitwise_and,
        "bitwise_or": _op_bitwise_or,
        "bitwise_xor": _op_bitwise_xor,
        "ashr": _op_ashr,
    }


class WhereSymbolicExpr(SymbolicExpr):
    cond: SymbolicExpr
    lhs: SymbolicExpr
    rhs: SymbolicExpr

    def __init__(self, op: str, cond: Any, lhs: Any, rhs: Any):
        super().__init__(op)
        self.add_child("cond", cond)
        self.add_child("lhs", lhs)
        self.add_child("rhs", rhs)
        self.dtype = self.lhs.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return handler(self)

    def _where(self) -> tuple[Z3Expr, ConstraintConjunction]:
        def _normalize(expr):
            if not isinstance(expr, list):
                return [expr]
            return expr

        def _broadcast(*lists):
            max_len = max(len(lst) for lst in lists)
            broadcasted = []
            for lst in lists:
                if len(lst) == 1:
                    broadcasted.append(lst * max_len)
                else:
                    assert len(lst) == max_len, "Incompatible lengths for broadcasting"
                    broadcasted.append(lst)
            return tuple(broadcasted)

        cond_expr = self.cond
        lhs_expr = self.lhs
        rhs_expr = self.rhs
        cond, constraints_cond = cond_expr._to_z3()
        lhs, constraints_lhs = lhs_expr._to_z3()
        rhs, constraints_rhs = rhs_expr._to_z3()

        cond = _normalize(cond)
        lhs = _normalize(lhs)
        rhs = _normalize(rhs)
        cond, lhs, rhs = _broadcast(cond, lhs, rhs)

        z3_expr = [If(cond[i], lhs[i], rhs[i]) for i in range(len(cond))]
        constraints = _and_constraints(
            constraints_cond, constraints_lhs, constraints_rhs
        )
        return z3_expr, constraints

    _Z3_BUILDERS: ClassVar[
        dict[str, Callable[[WhereSymbolicExpr], tuple[Z3Expr, ConstraintConjunction]]]
    ] = {
        "where": _where,
    }


class ReduceSymbolicExpr(SymbolicExpr):
    _SUPPORTED_OPS: ClassVar[tuple[str, ...]] = ("sum", "max", "min")
    input: SymbolicExpr
    keepdims: SymbolicExpr
    axis: Optional[SymbolicExpr]

    def __init__(self, op: str, input: Any, axis: Any = None, keepdims: bool = False):
        if op not in self._SUPPORTED_OPS:
            raise NotImplementedError(f"Unsupported reduce op: {op}")
        super().__init__(op)
        self.add_child("input", input)
        self.add_child("keepdims", keepdims)
        self.add_child("axis", axis)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return handler(self)

    def _reduce_sum(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        return Sum(arr), constraints

    def _reduce_max(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        return reduce(lambda a, b: If(a >= b, a, b), arr), constraints

    def _reduce_min(self) -> tuple[Z3Expr, ConstraintConjunction]:
        arr, constraints = self.input._to_z3()
        return reduce(lambda a, b: If(a <= b, a, b), arr), constraints

    _Z3_BUILDERS: ClassVar[
        dict[str, Callable[[ReduceSymbolicExpr], tuple[Z3Expr, ConstraintConjunction]]]
    ] = {
        "sum": _reduce_sum,
        "max": _reduce_max,
        "min": _reduce_min,
    }


class DotSymbolicExpr(SymbolicExpr):
    a: SymbolicExpr
    b: SymbolicExpr
    d: Optional[SymbolicExpr]

    def __init__(self, op: str, a: Any, b: Any, d: Any = None):
        super().__init__(op)
        self.add_child("a", a)
        self.add_child("b", b)
        self.add_child("d", d)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class CumsumSymbolicExpr(SymbolicExpr):
    input: SymbolicExpr
    axis: SymbolicExpr
    reverse: SymbolicExpr

    def __init__(self, op: str, input: Any, axis: Any, reverse: Any, dtype: Any):
        super().__init__(op)
        self.add_child("input", input)
        self.add_child("axis", axis)
        self.add_child("reverse", reverse)
        self.dtype = dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class MakeBlockPtrSymbolicExpr(SymbolicExpr):
    base: SymbolicExpr
    ndim: int
    block_shape_values: list[int]
    order_values: list[int]
    shape_keys: list[str]
    stride_keys: list[str]
    offset_keys: list[str]

    def __init__(
        self,
        op: str,
        base: Any,
        shape_list: Sequence[Any],
        stride_list: Sequence[Any],
        offset_list: Sequence[Any],
        block_shape_vals: Sequence[int],
        order_vals: Sequence[int],
    ):
        super().__init__(op)
        self.add_child("base", base)
        self.ndim = len(block_shape_vals)
        self.block_shape_values = list(block_shape_vals)
        self.order_values = list(order_vals)
        self.shape_keys: list[str] = []
        self.stride_keys: list[str] = []
        self.offset_keys: list[str] = []
        for i in range(self.ndim):
            shape_key = f"shape_{i}"
            stride_key = f"stride_{i}"
            offset_key = f"offset_{i}"
            self.add_child(shape_key, shape_list[i])
            self.add_child(stride_key, stride_list[i])
            self.add_child(offset_key, offset_list[i])
            self.shape_keys.append(shape_key)
            self.stride_keys.append(stride_key)
            self.offset_keys.append(offset_key)
        self.dtype = base.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Use TensorPointerLoad/Store to access block pointers"
        )


class AddPtrSymbolicExpr(SymbolicExpr):
    _INT_DTYPES: ClassVar[tuple[type, ...]] = (int, np.integer, bool)
    ptr: SymbolicExpr
    offset: SymbolicExpr

    def __init__(self, op: str, ptr: Any, offset: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("offset", offset)
        self.dtype = self.ptr.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        ptr_expr = self.ptr
        offset_expr = self.offset
        ptr_z3, constraints_ptr = ptr_expr._to_z3()
        offset_z3, constraints_offset = offset_expr._to_z3()
        constraints = _and_constraints(constraints_ptr, constraints_offset)
        ptr_dtype = cast(Any, ptr_expr.dtype)
        element_bytewidth = max(1, ptr_dtype.scalar.element_ty.primitive_bitwidth // 8)
        if not isinstance(ptr_z3, list) and not isinstance(offset_z3, list):  # hot path
            z3_expr = ptr_z3 + offset_z3 * element_bytewidth
        elif isinstance(ptr_z3, list) and isinstance(offset_z3, list):
            if len(ptr_z3) != len(offset_z3):
                raise ValueError(
                    f"ptr {ptr_z3} and offset {offset_z3} don't have the same length!"
                )
            z3_expr = [p + o * element_bytewidth for p, o in zip(ptr_z3, offset_z3)]
        elif isinstance(ptr_z3, list):
            z3_expr = [p + offset_z3 * element_bytewidth for p in ptr_z3]
        else:  # isinstance(offset_z3, list):
            z3_expr = [ptr_z3 + o * element_bytewidth for o in offset_z3]
        return z3_expr, constraints

    def concretize(self) -> Any:
        return self.concrete_fn(self.ptr.concretize(), self.offset.concretize())  # type: ignore


class AdvanceSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    ndim: int
    delta_keys: list[str]

    def __init__(self, op: str, ptr: Any, offset_list: Sequence[Any]):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.ndim = len(offset_list)
        self.delta_keys: list[str] = []
        for i in range(self.ndim):
            dk = f"delta_{i}"
            self.add_child(dk, offset_list[i])
            self.delta_keys.append(dk)
        self.dtype = ptr.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Use TensorPointerLoad/Store to access block pointers"
        )


class SplatSymbolicExpr(SymbolicExpr):
    block_type: SymbolicExpr
    arg: SymbolicExpr

    def __init__(self, op: str, block_type: Any, arg: Any):
        super().__init__(op)
        self.add_child("block_type", block_type)
        self.add_child("arg", arg)
        self.dtype = self.block_type.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()

    def concretize(self) -> Any:
        return self.concrete_fn(self.block_type.to_py(), self.arg.concretize())  # type: ignore


class ExpandDimsSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr
    axis: SymbolicExpr

    def __init__(self, op: str, arg: Any, axis: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("axis", axis)
        # Update dtype to reflect the new shape with an inserted dimension of size 1
        arg_shape = list(self.arg.shape) if self.arg.shape else []
        axis_val = axis if isinstance(axis, int) else axis.to_py()
        # Handle negative axis
        if axis_val < 0:
            axis_val = len(arg_shape) + 1 + axis_val
        # Insert dimension of size 1 at the specified axis
        new_shape = arg_shape[:axis_val] + [1] + arg_shape[axis_val:]
        self.dtype = tl.block_type(cast(Any, self.arg.dtype).scalar, new_shape)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class BroadcastSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr
    target_shape: tuple[int, ...]

    def __init__(self, op: str, arg: Any, shape: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        # Store the target shape for broadcasting
        if isinstance(shape, (list, tuple)):
            self.target_shape = tuple(shape)
        elif hasattr(shape, "to_py"):
            self.target_shape = tuple(shape.to_py())
        else:
            self.target_shape = ()
        # Update dtype to reflect the broadcast shape
        arg_dtype = self.arg.dtype
        if arg_dtype is not None and hasattr(arg_dtype, "element_ty"):
            elem_ty = arg_dtype.element_ty
        else:
            elem_ty = arg_dtype if arg_dtype else tl.int32
        self.dtype = (
            tl.block_type(elem_ty, list(self.target_shape))
            if self.target_shape
            else arg_dtype
        )

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class ReshapeSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr

    def __init__(self, op: str, arg: Any, shape: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.dtype = self.arg.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class TransSymbolicExpr(SymbolicExpr):
    arg: SymbolicExpr
    permutation: SymbolicExpr

    def __init__(self, op: str, arg: Any, permutation: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("permutation", permutation)
        self.dtype = self.arg.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class JoinSymbolicExpr(SymbolicExpr):
    lhs: SymbolicExpr
    rhs: SymbolicExpr

    def __init__(self, op: str, lhs: Any, rhs: Any):
        super().__init__(op)
        self.add_child("lhs", lhs)
        self.add_child("rhs", rhs)
        self.dtype = self.lhs.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(
            "Join operation is not implemented in Z3 evaluation yet"
        )


class CastSymbolicExpr(SymbolicExpr):
    _SUPPORTED_OPS: ClassVar[tuple[str, ...]] = ("cast_impl", "bitcast")
    src: SymbolicExpr
    dst_type: SymbolicExpr

    def __init__(self, op: str, src: Any, dst_type: Any):
        if op not in self._SUPPORTED_OPS:
            raise NotImplementedError(f"Unsupported cast op: {op}")
        super().__init__(op)
        self.add_child("src", src)
        self.add_child("dst_type", dst_type)
        self.dtype = self.dst_type.to_py()

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.src._to_z3()

    def concretize(self) -> Any:
        src_concrete = self.src.concretize()
        return self.concrete_fn(src_concrete, self.dtype)  # type: ignore


class FpToFpSymbolicExpr(SymbolicExpr):
    src: SymbolicExpr
    dst_type: SymbolicExpr
    rounding_mode: SymbolicExpr

    def __init__(self, op: str, src: Any, dst_type: Any, rounding_mode: Any):
        super().__init__(op)
        self.add_child("src", src)
        self.add_child("dst_type", dst_type)
        self.add_child("rounding_mode", rounding_mode)
        self.dtype = self.dst_type.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class AtomicCasSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    cmp: SymbolicExpr
    val: SymbolicExpr

    def __init__(self, op: str, ptr: Any, cmp: Any, val: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("cmp", cmp)
        self.add_child("val", val)
        self.dtype = self.val.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError("atomic_cas operation is not implemented yet")


class AtomicRmwSymbolicExpr(SymbolicExpr):
    ptr: SymbolicExpr
    val: SymbolicExpr
    mask: Optional[SymbolicExpr]

    def __init__(self, op: str, ptr: Any, val: Any, mask: Any = None):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("val", val)
        self.add_child("mask", mask)
        self.dtype = self.val.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class TensorPointerSymbolicExpr(SymbolicExpr):
    """Common base for block-pointer load/store expressions."""

    ptr: SymbolicExpr
    boundary_check: tuple[int, ...]

    @staticmethod
    def _resolve_element_dtype(
        ptr: SymbolicExpr,
    ) -> Optional[tl.core.dtype]:
        dt = ptr.dtype
        if dt is None:
            return None
        if isinstance(dt, tl.pointer_type):
            return dt.element_ty
        return dt

    def _resolve_block_ptr_components(
        self, ptr: SymbolicExpr
    ) -> tuple[
        SymbolicExpr,
        list[SymbolicExpr],
        list[SymbolicExpr],
        list[SymbolicExpr],
        list[int],
    ]:
        """Walk advance chain -> (base, shapes[], strides[], offsets[], block_shape[])"""
        if isinstance(ptr, MakeBlockPtrSymbolicExpr):
            shapes = [getattr(ptr, k) for k in ptr.shape_keys]
            strides = [getattr(ptr, k) for k in ptr.stride_keys]
            offsets = [getattr(ptr, k) for k in ptr.offset_keys]
            return ptr.base, shapes, strides, offsets, ptr.block_shape_values
        elif isinstance(ptr, AdvanceSymbolicExpr):
            base, shapes, strides, offsets, bs = self._resolve_block_ptr_components(
                ptr.ptr
            )
            deltas = [getattr(ptr, k) for k in ptr.delta_keys]
            new_offsets = [
                SymbolicExpr.create("add", off, d) for off, d in zip(offsets, deltas)
            ]
            return base, shapes, strides, new_offsets, bs
        raise TypeError(f"Expected block pointer, got {type(ptr)}")

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        (
            base,
            shapes,
            strides,
            offsets,
            block_shape,
        ) = self._resolve_block_ptr_components(self.ptr)

        base_dtype = base.dtype
        if isinstance(base_dtype, tl.pointer_type) and hasattr(
            base_dtype.element_ty, "primitive_bitwidth"
        ):
            elem_size = max(1, base_dtype.element_ty.primitive_bitwidth // 8)
        else:
            elem_size = 1

        base_z3, c_base = base._to_z3()
        addr = base_z3
        parts: list[ConstraintExpr | Sequence[ConstraintExpr]] = []
        if c_base:
            parts.append(c_base)

        for d in range(len(block_shape)):
            k_d = Int(f"blk_k_{d}")
            off_z3, c_off = offsets[d]._to_z3()
            stride_z3, c_stride = strides[d]._to_z3()

            addr = addr + (off_z3 + k_d) * stride_z3 * elem_size
            parts.append(And(k_d >= 0, k_d < block_shape[d]))
            if c_off:
                parts.append(c_off)
            if c_stride:
                parts.append(c_stride)

            if d in self.boundary_check:
                shape_z3, c_shape = shapes[d]._to_z3()
                parts.append(And(off_z3 + k_d >= 0, off_z3 + k_d < shape_z3))
                if c_shape:
                    parts.append(c_shape)

        return addr, _and_constraints(*parts)


class TensorPointerLoadSymbolicExpr(TensorPointerSymbolicExpr):
    def __init__(self, op: str, ptr: Any, boundary_check: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.boundary_check = tuple(boundary_check) if boundary_check else ()
        self.dtype = self._resolve_element_dtype(ptr)


class TensorPointerStoreSymbolicExpr(TensorPointerSymbolicExpr):
    value: SymbolicExpr

    def __init__(self, op: str, ptr: Any, value: Any, boundary_check: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("value", value)
        self.boundary_check = tuple(boundary_check) if boundary_check else ()
        self.dtype = self._resolve_element_dtype(ptr)


SymbolicExpr.register_op_class(ConstSymbolicExpr, ("const",))
SymbolicExpr.register_op_class(PidSymbolicExpr, ("pid",))
SymbolicExpr.register_op_class(ArangeSymbolicExpr, ("arange",))
SymbolicExpr.register_op_class(LoadSymbolicExpr, ("load",))
SymbolicExpr.register_op_class(StoreSymbolicExpr, ("store",))
SymbolicExpr.register_op_class(UnarySymbolicExpr, SymbolicExpr.UNARY_OPS)
SymbolicExpr.register_op_class(BinarySymbolicExpr, SymbolicExpr.BINARY_OPS)
SymbolicExpr.register_op_class(WhereSymbolicExpr, ("where",))
SymbolicExpr.register_op_class(ReduceSymbolicExpr, ("sum", "max", "min"))
SymbolicExpr.register_op_class(DotSymbolicExpr, ("dot",))
SymbolicExpr.register_op_class(CumsumSymbolicExpr, ("cumsum",))
SymbolicExpr.register_op_class(MakeBlockPtrSymbolicExpr, ("make_block_ptr",))
SymbolicExpr.register_op_class(AddPtrSymbolicExpr, ("addptr",))
SymbolicExpr.register_op_class(AdvanceSymbolicExpr, ("advance",))
SymbolicExpr.register_op_class(SplatSymbolicExpr, ("splat",))
SymbolicExpr.register_op_class(ExpandDimsSymbolicExpr, ("expand_dims",))
SymbolicExpr.register_op_class(BroadcastSymbolicExpr, ("broadcast",))
SymbolicExpr.register_op_class(ReshapeSymbolicExpr, ("reshape",))
SymbolicExpr.register_op_class(TransSymbolicExpr, ("trans",))
SymbolicExpr.register_op_class(JoinSymbolicExpr, ("join",))
SymbolicExpr.register_op_class(CastSymbolicExpr, ("cast_impl", "bitcast"))
SymbolicExpr.register_op_class(FpToFpSymbolicExpr, ("fp_to_fp",))
SymbolicExpr.register_op_class(AtomicCasSymbolicExpr, ("atomic_cas",))
SymbolicExpr.register_op_class(AtomicRmwSymbolicExpr, ("atomic_rmw",))
SymbolicExpr.register_op_class(TensorPointerLoadSymbolicExpr, ("tensor_pointer_load",))
SymbolicExpr.register_op_class(
    TensorPointerStoreSymbolicExpr, ("tensor_pointer_store",)
)


# ── Shared constants and utilities for symbolic clients ──────────

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


class SymbolicClient(Client):
    """Base class for clients that use the symbolic engine.

    Provides shared operation overriders and for-loop infrastructure.
    Subclasses must implement domain-specific memory operation handlers.
    """

    def __init__(self):
        super().__init__()
        self.loop_stack: list[LoopContext] = []
        SymbolicExpr.set_loop_ctx_provider(
            lambda *_args, **_kwargs: (self.loop_stack[-1] if self.loop_stack else None)
        )

    # ── Shared operation overriders ───────────────────────────────

    def _op_program_id_overrider(self, axis):
        return SymbolicExpr.create("pid", axis)

    def _op_unary_op_overrider(self, arg, op):
        arg_sym = SymbolicExpr.from_value(arg)
        name = _UNARY_NUMPY_TO_SYM_OP.get(op)
        if name is None:
            raise NotImplementedError(f"Unsupported unary operation: {op}")
        return SymbolicExpr.create(name, arg_sym)

    def _op_binary_op_overrider(self, lhs, rhs, op):
        lhs_sym = SymbolicExpr.from_value(lhs)
        rhs_sym = SymbolicExpr.from_value(rhs)
        op_name = _BINARY_NUMPY_TO_SYM_OP.get(op)
        if op_name is None:
            raise NotImplementedError(f"Unsupported binary operation: {op}")
        return SymbolicExpr.create(op_name, lhs_sym, rhs_sym)

    def _op_ternary_op_overrider(self, lhs, rhs, other, op):
        lhs_sym = SymbolicExpr.from_value(lhs)
        rhs_sym = SymbolicExpr.from_value(rhs)
        other_sym = SymbolicExpr.from_value(other)
        if op is np.where:
            return SymbolicExpr.create("where", lhs_sym, rhs_sym, other_sym)
        raise NotImplementedError(f"Unsupported ternary operation: {op}")

    def _op_addptr_overrider(self, ptr, offset):
        return SymbolicExpr.create(
            "addptr",
            SymbolicExpr.from_value(ptr),
            SymbolicExpr.from_value(offset),
        )

    def _op_dot_overrider(self, a, b, d, input_precision, max_num_imprecise_acc):
        a_sym = SymbolicExpr.from_value(a)
        b_sym = SymbolicExpr.from_value(b)
        d_sym = SymbolicExpr.from_value(d) if d is not None else None
        return SymbolicExpr.create("dot", a_sym, b_sym, d_sym)

    def _op_make_range_overrider(self, ret_ty, start, end):
        return SymbolicExpr.create(
            "arange",
            SymbolicExpr.from_value(ret_ty),
            SymbolicExpr.from_value(start),
            SymbolicExpr.from_value(end),
        )

    def _op_expand_dims_overrider(self, arg, axis):
        return SymbolicExpr.create("expand_dims", SymbolicExpr.from_value(arg), axis)

    def _op_broadcast_overrider(self, arg, shape):
        return SymbolicExpr.create("broadcast", SymbolicExpr.from_value(arg), shape)

    def _op_reduce_sum_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        return SymbolicExpr.create(
            "sum", SymbolicExpr.from_value(input), axis, keep_dims
        )

    def _op_reduce_max_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        return SymbolicExpr.create(
            "max", SymbolicExpr.from_value(input), axis, keep_dims
        )

    def _op_reduce_min_overrider(self, input, axis=None, keep_dims=False, **kwargs):
        return SymbolicExpr.create(
            "min", SymbolicExpr.from_value(input), axis, keep_dims
        )

    def _op_splat_overrider(self, shape, arg):
        return SymbolicExpr.create("splat", shape, SymbolicExpr.from_value(arg))

    def _op_idiv_overrider(self, lhs, rhs):
        return SymbolicExpr.from_value(lhs) // SymbolicExpr.from_value(rhs)

    def _op_rsqrt_overrider(self, arg):
        return SymbolicExpr.create("rsqrt", SymbolicExpr.from_value(arg))

    def _op_cast_impl_overrider(self, src, dst_type):
        return SymbolicExpr.create("cast_impl", src, dst_type)

    def _op_reshape_overrider(self, arg, shape, allow_reorder):
        return SymbolicExpr.create(
            "reshape",
            SymbolicExpr.from_value(arg),
            SymbolicExpr.from_value(shape),
        )

    def _op_trans_overrider(self, arg, perm=[1, 0]):
        return SymbolicExpr.create("trans", SymbolicExpr.from_value(arg), perm)

    def _op_join_overrider(self, lhs, rhs):
        return SymbolicExpr.create(
            "join",
            SymbolicExpr.from_value(lhs),
            SymbolicExpr.from_value(rhs),
        )

    def _op_fabs_overrider(self, arg):
        return SymbolicExpr.create("fabs", SymbolicExpr.from_value(arg))

    def _op_ashr_overrider(self, lhs, rhs):
        return SymbolicExpr.create(
            "ashr",
            SymbolicExpr.from_value(lhs),
            SymbolicExpr.from_value(rhs),
        )

    def _op_advance_overrider(self, ptr, offsets):
        ptr_sym = SymbolicExpr.from_value(ptr)
        offset_syms = [SymbolicExpr.from_value(o) for o in offsets]
        return SymbolicExpr.create("advance", ptr_sym, offset_syms)

    def _op_fp_to_fp_overrider(self, src, dst_type, rounding_mode):
        return SymbolicExpr.create(
            "fp_to_fp",
            SymbolicExpr.from_value(src),
            dst_type,
            rounding_mode,
        )

    def _op_umulhi_overrider(self, lhs, rhs):
        return SymbolicExpr.create(
            "umulhi",
            SymbolicExpr.from_value(lhs),
            SymbolicExpr.from_value(rhs),
        )

    def _op_cumsum_overrider(self, input, axis, reverse=False, dtype=None):
        return SymbolicExpr.create(
            "cumsum",
            SymbolicExpr.from_value(input),
            axis,
            reverse,
            dtype,
        )

    def _op_bitcast_overrider(self, src, dst_type):
        return SymbolicExpr.create("bitcast", SymbolicExpr.from_value(src), dst_type)

    def _op_raw_load_overrider(self, ptr, cache_modifier, eviction_policy, is_volatile):
        return self._op_load_overrider(
            ptr, None, None, cache_modifier, eviction_policy, is_volatile
        )

    def _op_raw_store_overrider(self, ptr, value, cache_modifier, eviction_policy):
        return self._op_store_overrider(
            ptr, value, None, cache_modifier, eviction_policy
        )

    def _op_atomic_cas_overrider(self, ptr, cmp, val, sem, scope):
        ptr_sym = SymbolicExpr.from_value(ptr)
        cmp_sym = SymbolicExpr.from_value(cmp)
        val_sym = SymbolicExpr.from_value(val)
        return SymbolicExpr.create("atomic_cas", ptr_sym, cmp_sym, val_sym)

    def _op_atomic_rmw_overrider(self, rmwOp, ptr, val, mask, sem, scope):
        ptr_sym = SymbolicExpr.from_value(ptr)
        val_sym = SymbolicExpr.from_value(val)
        mask_sym = SymbolicExpr.from_value(mask)
        return SymbolicExpr.create("atomic_rmw", ptr_sym, val_sym, mask_sym)

    def _build_op_overrider_map(self) -> dict[type[Op], Callable]:
        """Return a mapping of shared Op types to their overrider methods."""
        return {
            ProgramId: self._op_program_id_overrider,
            UnaryOp: self._op_unary_op_overrider,
            BinaryOp: self._op_binary_op_overrider,
            TernaryOp: self._op_ternary_op_overrider,
            AddPtr: self._op_addptr_overrider,
            Dot: self._op_dot_overrider,
            MakeRange: self._op_make_range_overrider,
            ExpandDims: self._op_expand_dims_overrider,
            Broadcast: self._op_broadcast_overrider,
            ReduceSum: self._op_reduce_sum_overrider,
            ReduceMax: self._op_reduce_max_overrider,
            ReduceMin: self._op_reduce_min_overrider,
            Splat: self._op_splat_overrider,
            Idiv: self._op_idiv_overrider,
            Rsqrt: self._op_rsqrt_overrider,
            CastImpl: self._op_cast_impl_overrider,
            Reshape: self._op_reshape_overrider,
            Trans: self._op_trans_overrider,
            Join: self._op_join_overrider,
            Fabs: self._op_fabs_overrider,
            Ashr: self._op_ashr_overrider,
            Advance: self._op_advance_overrider,
            FpToFp: self._op_fp_to_fp_overrider,
            Umulhi: self._op_umulhi_overrider,
            CumSum: self._op_cumsum_overrider,
            Bitcast: self._op_bitcast_overrider,
            AtomicCas: self._op_atomic_cas_overrider,
            AtomicRMW: self._op_atomic_rmw_overrider,
            RawLoad: self._op_raw_load_overrider,
            RawStore: self._op_raw_store_overrider,
        }

    def register_op_callback(self, op_type: type[Op], *args, **kwargs) -> OpCallbacks:
        overrider_map = self._build_op_overrider_map()
        overrider = overrider_map.get(op_type)
        if overrider is not None:
            return OpCallbacks(op_overrider=self.lock_fn(overrider))
        return OpCallbacks()

    # ── For-loop infrastructure ───────────────────────────────────

    def _on_data_dependent_value(self) -> None:
        """Hook called when a data-dependent value forces concretization.

        Subclasses must override to set their fallback flag.
        """
        raise NotImplementedError

    def _should_skip_loop_hooks(self) -> bool:
        """Return True to skip loop hook processing."""
        return False

    def _materialize_loop_value(self, expr: Any) -> int:
        if isinstance(expr, SymbolicExpr):
            if expr.op == "const":
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())
            elif expr.has_op("load"):
                self._on_data_dependent_value()
                expr = expr.replace_subtree("load")
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())
            else:
                z3_expr, _ = expr.eval()
                if isinstance(z3_expr, IntNumRef):
                    return z3_expr.as_long()
                self._on_data_dependent_value()
                expr = expr.replace_subtree()
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())
        return int(expr)

    def _wrap_range(
        self,
        iterable,
        _lineno,
        _range_type,
        iter_args=None,
        iter_kwargs=None,
        _iter_callable=None,
    ):
        if self._should_skip_loop_hooks():
            return None
        iter_args = tuple(iter_args or ())
        iter_kwargs = iter_kwargs or {}

        if isinstance(iterable, RangeWrapper):
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

        step = self._materialize_loop_value(step_expr)
        start = self._materialize_loop_value(start_expr)
        stop = self._materialize_loop_value(stop_expr)

        concrete_range = range(start, stop, step)
        length = len(concrete_range)

        return RangeWrapper(
            concrete_range, length=length, start=start, stop=stop, step=step
        )

    def _loop_hook_before(self, lineno, iterable):
        if self._should_skip_loop_hooks():
            return
        if not isinstance(iterable, RangeWrapper):
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

    def _loop_hook_iter_overrider(self, lineno, idx):
        if self._should_skip_loop_hooks():
            return idx
        if self.loop_stack and self.loop_stack[-1].lineno == lineno:
            return self.loop_stack[-1].idx
        return idx

    def _loop_hook_after(self, lineno: int) -> None:
        if self._should_skip_loop_hooks():
            return
        if not self.loop_stack or self.loop_stack[-1].lineno != lineno:
            return
        self.loop_stack.pop()

    def register_for_loop_callback(self) -> ForLoopCallbacks:
        return ForLoopCallbacks(
            range_wrapper_factory=self.lock_fn(self._wrap_range),
            before_loop_callback=self.lock_fn(self._loop_hook_before),
            loop_iter_overrider=self.lock_fn(self._loop_hook_iter_overrider),
            after_loop_callback=self.lock_fn(self._loop_hook_after),
        )
