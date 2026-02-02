from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property, reduce
import math
from typing import (
    Any,
    ClassVar,
    NoReturn,
    Optional,
    Union,
    TypeAlias,
    TypeVar,
    cast,
)
import sys

import numpy as np
from torch import Tensor
from anytree import Node, RenderTree
from z3 import (
    Solver,
    Int,
    IntVal,
    If,
    Sum,
    And,
    Or,
    Not,
    sat,
    simplify,
    Int2BV,
    BV2Int,
    BitVecRef,
    BoolVal,
)
from z3.z3 import BoolRef, ArithRef, IntNumRef, ExprRef, Tactic, Probe

import triton.language as tl
from triton.runtime.interpreter import TensorHandle, _get_np_dtype

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
from .data import OutOfBoundsRecordZ3
from .report import (
    _get_traceback_info,
    _get_user_code_location,
    _location_to_traceback_info,
    print_oob_record,
    print_oob_record_pdb_style,
)
from ...core.config import config as cfg


Z3Expr: TypeAlias = Union[ExprRef, list[ExprRef], Tactic, Probe]
ConstraintExpr: TypeAlias = Union[ExprRef, bool, int, float]
ConstraintConjunction: TypeAlias = Optional[BoolRef]
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
    symbolic_expr: "SymbolicExpr"
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


class SymbolicExprDataWrapper:
    """
    This wrapper is used as a workaround of triton interpreter legacy code.
    In def _get_bool(self) of class tensor,
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Since we replaced TensorHandle with SymbolicExpr,
    we need to wrap SymbolicExpr with a class that has size attribute, and data.size != 1.
    """

    def __init__(self, symbolic_expr: "SymbolicExpr", value: Optional[str] = None):
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
    INDIRECT_OPS: ClassVar[tuple[str, ...]] = ("load", "store")
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
    _OP_CLASS_MAP: ClassVar[dict[str, type["SymbolicExpr"]]] = {}

    @classmethod
    def register_op_class(
        cls, op_cls: type["SymbolicExpr"], op_types: tuple[str, ...]
    ) -> None:
        for op_type in op_types:
            cls._OP_CLASS_MAP[op_type] = op_cls

    @classmethod
    def create(cls, op: str, *args: Any) -> "SymbolicExpr":
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
        self.children: dict[str, Optional["SymbolicExpr"]] = {}

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

    def __add__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr.create("add", self, other)

    def __sub__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr.create("sub", self, other)

    def __mul__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr.create("mul", self, other)

    def __truediv__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr.create("div", self, other)

    def __floordiv__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        return SymbolicExpr.create("idiv", self, other)

    def __mod__(self, other: "SymbolicExpr") -> "SymbolicExpr":
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
    def from_value(cls, var: Any) -> "SymbolicExpr":
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

    def replace_subtree(self, anchor_op: Optional[str] = None) -> "SymbolicExpr":
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
            if self.value.data.size != 1:
                raise RuntimeError("Only scalar TensorHandle is supported in const!")
            return self.value

        raise RuntimeError(f"Unsupported const value type: {type(self.value)}")


class PidSymbolicExpr(SymbolicExpr):
    axis: "SymbolicExpr"

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
    # Class-level counter for generating unique variable names
    _arange_counter: ClassVar[int] = 0

    def __init__(self, op: str, ret_ty: Any, start: Any, end: Any):
        super().__init__(op)
        self.add_child("ret_ty", ret_ty)
        self.add_child("start", start)
        self.add_child("end", end)
        # Program ID / arange are always int32
        start_const = cast(ConstSymbolicExpr, self.start)
        end_const = cast(ConstSymbolicExpr, self.end)
        self.dtype = tl.block_type(tl.int32, [end_const.value - start_const.value])
        # Assign a unique ID for this arange instance
        self._unique_id = ArangeSymbolicExpr._arange_counter
        ArangeSymbolicExpr._arange_counter += 1

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        start = self.start.to_py()
        end = self.end.to_py()
        # Use unique ID to ensure each arange instance has its own Z3 variable.
        # This is critical for multi-dimensional tensors where different dimensions
        # may use arange with the same (start, end) but should be independent.
        # The _to_z3() method in the base class caches the result per instance,
        # so the same ArangeSymbolicExpr object will always return the same variable.
        name = f"arange_{self._unique_id}_{start}_{end}"
        v = Int(name)
        constraints = _and_constraints(v >= start, v < end)
        return v, constraints

    def concretize(self) -> Any:
        return self.concrete_fn(
            self.ret_ty.to_py(), self.start.to_py(), self.end.to_py()
        )  # type: ignore


class IndirectSymbolicExprBase(SymbolicExpr):
    ptr: "SymbolicExpr"
    mask: Optional["SymbolicExpr"]
    other: Optional["SymbolicExpr"]

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
    value: "SymbolicExpr"

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
    arg: "SymbolicExpr"

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
    lhs: "SymbolicExpr"
    rhs: "SymbolicExpr"

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
        dict[str, Callable[["BinarySymbolicExpr", Z3Expr, Z3Expr], Z3Expr]]
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
    cond: "SymbolicExpr"
    lhs: "SymbolicExpr"
    rhs: "SymbolicExpr"

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
        dict[str, Callable[["WhereSymbolicExpr"], tuple[Z3Expr, ConstraintConjunction]]]
    ] = {
        "where": _where,
    }


class ReduceSymbolicExpr(SymbolicExpr):
    _SUPPORTED_OPS: ClassVar[tuple[str, ...]] = ("sum", "max", "min")
    input: "SymbolicExpr"
    keepdims: "SymbolicExpr"
    axis: Optional["SymbolicExpr"]

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
        dict[
            str, Callable[["ReduceSymbolicExpr"], tuple[Z3Expr, ConstraintConjunction]]
        ]
    ] = {
        "sum": _reduce_sum,
        "max": _reduce_max,
        "min": _reduce_min,
    }


class DotSymbolicExpr(SymbolicExpr):
    a: "SymbolicExpr"
    b: "SymbolicExpr"
    d: Optional["SymbolicExpr"]

    def __init__(self, op: str, a: Any, b: Any, d: Any = None):
        super().__init__(op)
        self.add_child("a", a)
        self.add_child("b", b)
        self.add_child("d", d)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class CumsumSymbolicExpr(SymbolicExpr):
    input: "SymbolicExpr"
    axis: "SymbolicExpr"
    reverse: "SymbolicExpr"

    def __init__(self, op: str, input: Any, axis: Any, reverse: Any, dtype: Any):
        super().__init__(op)
        self.add_child("input", input)
        self.add_child("axis", axis)
        self.add_child("reverse", reverse)
        self.dtype = dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class MakeBlockPtrSymbolicExpr(SymbolicExpr):
    base: "SymbolicExpr"
    strides: "SymbolicExpr"
    offsets: "SymbolicExpr"
    block_shape: "SymbolicExpr"
    order: "SymbolicExpr"

    def __init__(
        self,
        op: str,
        base: Any,
        shape: Any,
        strides: Any,
        offsets: Any,
        block_shape: Any,
        order: Any,
    ):
        super().__init__(op)
        self.add_child("base", base)
        self.add_child("shape", shape)
        self.add_child("strides", strides)
        self.add_child("offsets", offsets)
        self.add_child("block_shape", block_shape)
        self.add_child("order", order)
        self.dtype = base.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class AddPtrSymbolicExpr(SymbolicExpr):
    _INT_DTYPES: ClassVar[tuple[type, ...]] = (int, np.integer, bool)
    ptr: "SymbolicExpr"
    offset: "SymbolicExpr"

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
    ptr: "SymbolicExpr"
    offsets: "SymbolicExpr"

    def __init__(self, op: str, ptr: Any, offsets: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("offsets", offsets)

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError("Advance operation is not implemented yet")


class SplatSymbolicExpr(SymbolicExpr):
    block_type: "SymbolicExpr"
    arg: "SymbolicExpr"

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
    arg: "SymbolicExpr"
    axis: "SymbolicExpr"

    def __init__(self, op: str, arg: Any, axis: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("axis", axis)
        self.dtype = self.arg.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class BroadcastSymbolicExpr(SymbolicExpr):
    arg: "SymbolicExpr"

    def __init__(self, op: str, arg: Any, shape: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.dtype = self.arg.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class ReshapeSymbolicExpr(SymbolicExpr):
    arg: "SymbolicExpr"

    def __init__(self, op: str, arg: Any, shape: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.dtype = self.arg.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class TransSymbolicExpr(SymbolicExpr):
    arg: "SymbolicExpr"
    permutation: "SymbolicExpr"

    def __init__(self, op: str, arg: Any, permutation: Any):
        super().__init__(op)
        self.add_child("arg", arg)
        self.add_child("permutation", permutation)
        self.dtype = self.arg.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        return self.arg._to_z3()


class JoinSymbolicExpr(SymbolicExpr):
    lhs: "SymbolicExpr"
    rhs: "SymbolicExpr"

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
    src: "SymbolicExpr"
    dst_type: "SymbolicExpr"

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
    src: "SymbolicExpr"
    dst_type: "SymbolicExpr"
    rounding_mode: "SymbolicExpr"

    def __init__(self, op: str, src: Any, dst_type: Any, rounding_mode: Any):
        super().__init__(op)
        self.add_child("src", src)
        self.add_child("dst_type", dst_type)
        self.add_child("rounding_mode", rounding_mode)
        self.dtype = self.dst_type.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class AtomicCasSymbolicExpr(SymbolicExpr):
    ptr: "SymbolicExpr"
    cmp: "SymbolicExpr"
    val: "SymbolicExpr"

    def __init__(self, op: str, ptr: Any, cmp: Any, val: Any):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("cmp", cmp)
        self.add_child("val", val)
        self.dtype = self.val.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError("atomic_cas operation is not implemented yet")


class AtomicRmwSymbolicExpr(SymbolicExpr):
    ptr: "SymbolicExpr"
    val: "SymbolicExpr"
    mask: Optional["SymbolicExpr"]

    def __init__(self, op: str, ptr: Any, val: Any, mask: Any = None):
        super().__init__(op)
        self.add_child("ptr", ptr)
        self.add_child("val", val)
        self.add_child("mask", mask)
        self.dtype = self.val.dtype

    def _to_z3_impl(self) -> tuple[Z3Expr, ConstraintConjunction]:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


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
