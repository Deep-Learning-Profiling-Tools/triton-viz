from collections import namedtuple
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property, reduce
from typing import Any, ClassVar, Optional, Union, TypeAlias, cast
import sys
import re

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
from z3.z3util import get_vars

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
from .report import _get_traceback_info, print_oob_record, print_oob_record_pdb_style
from ...core.config import config as cfg


Z3Expr: TypeAlias = Union[ExprRef, int, list[ExprRef], list[int], Tactic, Probe]
ConstraintExpr: TypeAlias = Union[ExprRef, int, float]


@dataclass
class LoopContext:
    lineno: int
    length: int
    idx_z3: ArithRef
    start: int = 0
    stop: int = 0
    step: int = 1
    values: list[int] = field(default_factory=list)
    signature_cache: set[int] = field(default_factory=set)
    pending_checks: list[tuple[Z3Expr, list[BoolRef], "SymbolicExpr"]] = field(
        default_factory=list
    )
    # Clean up variable names by removing suffixes like _81, _144
    # Cache compiled regex pattern for better performance
    re_pattern = re.compile(r"(loop_i|arange)_\d+")


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

    def __new__(cls, *args, **kwargs) -> "Sanitizer":
        if cls is Sanitizer:
            target_cls = cast(
                type["Sanitizer"],
                SymbolicSanitizer if cfg.enable_sanitizer else NullSanitizer,
            )
            obj = object.__new__(target_cls)
            cast(Any, target_cls).__init__(obj, *args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, abort_on_error: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abort_on_error: bool = abort_on_error

    def pre_run_callback(self, fn: Callable) -> bool:
        return True

    def post_run_callback(self, fn: Callable) -> bool:
        return True

    def pre_warmup_callback(self, jit_fn, *args, **kwargs) -> bool:
        return False

    def post_warmup_callback(self, jit_fn, ret) -> None:
        pass

    def arg_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def finalize(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def grid_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def grid_idx_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def register_op_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def register_for_loop_callback(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError


class SymbolicExprDataWrapper:
    """
    This wrapper is used as a workaround of triton interpreter legacy code.
    In def _get_bool(self) of class tensor,
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Since we replaced TensorHandle with SymbolicExpr,
    we need to wrap SymbolicExpr with a class that has size attribute, and data.size != 1.
    """

    def __init__(self, value: str, symbolic_expr: "SymbolicExpr"):
        self.value = value
        self.symbolic_expr = symbolic_expr

    @property
    def size(self) -> int:
        shape = self.symbolic_expr.shape
        if len(shape) == 0:
            return 1
        return reduce(lambda x, y: x * y, shape)

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


Spec = namedtuple(
    "Spec",
    [
        "req",  # Required Arguments (tuple)
        "opt",  # Optional Arguments (tuple)
    ],
    defaults=((), ()),
)


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

    OP_SPEC: ClassVar[dict[str, Spec]] = {}

    PID0: ClassVar[ArithRef] = Int("pid_0")
    PID1: ClassVar[ArithRef] = Int("pid_1")
    PID2: ClassVar[ArithRef] = Int("pid_2")

    ARANGE_DICT: ClassVar[dict[tuple[int, int], tuple[ArithRef, list[BoolRef]]]] = {}
    _OP_CLASS_MAP: ClassVar[dict[str, type["SymbolicExpr"]]] = {}

    @classmethod
    def register_op_class(
        cls, op_cls: type["SymbolicExpr"], op_types: tuple[str, ...]
    ) -> None:
        for op_type in op_types:
            cls._OP_CLASS_MAP[op_type] = op_cls

    @classmethod
    def create(cls, op: str, *args: Any) -> "SymbolicExpr":
        op_cls = cls._OP_CLASS_MAP.get(op, cls)
        return op_cls(op, *args)

    def __init__(self, op: str, *args: Any):
        """
        :param op: Operation type, e.g. "const", "add", "sub", "mul", "div", "pid", "arange"
        :param args: Sub-expressions (for compound operations)
        """
        assert op in self.SUPPORTED_OPS, f"Unsupported op: {op}"
        self.op = op
        self.attrs: dict[str, Any] = {}
        self.dtype: Optional[tl.core.dtype] = None

        # Functions and arguments for concretization
        self.concrete_fn: Optional[Callable[..., Any]] = None

        # deal with args
        self.children: dict[str, Optional["SymbolicExpr"]] = {}
        self._init_from_spec(*args)

        # for-loop iterator association
        self.loop_ctx: Optional[LoopContext] = None

        # z3
        self.z3: Optional[Z3Expr] = None

        self.constraints: list[BoolRef] = []

    @property
    def shape(self) -> tuple[int, ...]:
        if not self.dtype:
            return ()
        if self.dtype is tl.block_type:
            return self.dtype.shape
        elif self.dtype is tl.pointer_type:
            return self.dtype.element_ty.shape
        return ()

    def _init_from_spec(self, *args: Any) -> None:
        if self.op not in self.OP_SPEC:
            raise NotImplementedError(f"Unsupported op: {self.op}")
        spec = self.OP_SPEC[self.op]

        if self.op == "const":  # leaf node
            self.value = args[0]
            self.dtype = args[1]
        else:
            # store in self.children
            names = list(spec.req) + list(spec.opt)
            for name, val in zip(names, args):
                val = SymbolicExpr.from_value(val) if val is not None else None
                self.children[name] = val
            for name in names[len(args) :]:
                self.children[name] = None

            self._post_init()

    def _post_init(self) -> None:
        return

    def __getattr__(self, name: str) -> Any:
        if name in self.children:
            return self.children[name]
        raise AttributeError(name)

    def set_attr(self, name: str, values: Any) -> None:
        self.attrs[name] = values

    def __add__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("add", self, other)

    def __sub__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("sub", self, other)

    def __mul__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("mul", self, other)

    def __truediv__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("div", self, other)

    def __floordiv__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("idiv", self, other)

    def __mod__(self, other: "SymbolicExpr") -> "SymbolicExpr":
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("mod", self, other)

    def __lt__(self, other: object) -> Any:
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("less", self, other)

    def __le__(self, other: object) -> Any:
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("less_equal", self, other)

    def __ne__(self, other: object) -> Any:
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr.create("not_equal", self, other)

    def __eq__(self, other: object) -> Any:
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
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

    def to_tree_str(self) -> str:
        """
        Render the AST as an ASCII tree using anytree.RenderTree.
        """
        root = self._to_anytree()
        lines = []
        for prefix, _, node in RenderTree(root):
            lines.append(f"{prefix}{node.name}")
        return "\n" + "\n".join(lines)

    def __str__(self) -> str:
        return self.to_tree_str()

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def data(self) -> SymbolicExprDataWrapper:
        return SymbolicExprDataWrapper(self.__str__(), self)

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
        if isinstance(var, tl.core.dtype):
            return var
        if isinstance(var, tl.core.tensor):
            var = var.handle
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
        if isinstance(var, TensorHandle):
            if len(var.data) != 1:
                raise ValueError(
                    f"Unsupported var.data: {var.data} with length more than one!"
                )
            if var.dtype in SymbolicExpr.triton_scala_dtypes:  # if an immediate
                return var.dtype
            if isinstance(var.dtype, tl.pointer_type):  # if a pointer
                return var.dtype
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
        if isinstance(var, tl.core.tensor):  # if a triton tensor
            var = var.handle  # get its handle

        if isinstance(var, cls):  # if already SymbolicExpr
            return var

        dtype = SymbolicExpr._infer_literal_dtype(var)  # get the triton dtype

        if isinstance(var, SymbolicExpr.tuple_types):  # if a tuple
            seq = cast(Sequence[Any], var)
            return cls.create("const", tuple(seq), dtype)
        if isinstance(var, TensorHandle):  # if a TensorHandle
            return cls.create("const", var.data.item(), dtype)
        if isinstance(
            var, SymbolicExpr.builtin_scala_types
        ):  # if a python builtin type
            return cls.create("const", var, dtype)
        if isinstance(var, tl.core.dtype):
            # If it's a triton dtype, we can create a const node with it
            return cls.create("const", var, dtype)

        raise ValueError("Unknown type:", type(var))

    def eval(self) -> tuple[Z3Expr, list[BoolRef]]:
        """
        Returns a tuple (expr, constraints):
        - expr: Z3 expression (or list of expressions) corresponding to the root node
        - constraints: list of Z3 BoolExpr objects, recording all range constraints created by program_id and arange
        """
        expr, constraints = self._to_z3()

        if isinstance(expr, list):
            expr = [simplify(e) for e in expr]
        else:
            expr = simplify(expr)

        return expr, constraints

    def _to_z3(self) -> tuple[Z3Expr, list[BoolRef]]:
        if self.z3 is not None:
            return self.z3, self.constraints

        self._to_z3_impl()
        if self.z3 is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        return self.z3, self.constraints

    def _to_z3_impl(self) -> None:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")

    def has_op(self, op_name: str) -> bool:
        if self.op == op_name:
            return True
        for _, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                continue
            if child_symbolic_expr.has_op(op_name):
                return True
        return False

    def to_py(self) -> Any:
        """
        Valid only for nodes with op == 'const':
        - If `value` is a TensorHandle:
            • Scalar  -> return int/float
            • Multi-element -> return a Python list
        - Otherwise, return the original Python object
          (e.g., int, float, tuple, list, etc.).
        """
        if self.op != "const":
            raise TypeError("SymbolicExpr.to_py() can be used only on 'const' nodes")

        v = self.value
        if isinstance(v, TensorHandle):
            if len(v.data) == 1:
                return v.data.item()  # scalar case
            else:
                return v.data.tolist()  # multi-element case
        return v

    def concretize(self) -> Any:
        return self._concretize_impl()

    def _concretize_impl(self) -> Any:
        raise NotImplementedError(f"Concretize for op {self.op} is not implemented")

    def replace_subtree(self, anchor_op: Optional[str] = None) -> "SymbolicExpr":
        """
        Post-order traversal that replaces *all* sub-trees with constant nodes
        produced by `concretize`.
        """
        for name, child in list(self.children.items()):
            if child is None:
                continue
            self.children[name] = child.replace_subtree(anchor_op)

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


class BasicSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "const": Spec(req=("value", "dtype")),
        "pid": Spec(req=("axis",)),
        "arange": Spec(req=("ret_ty", "start", "end")),
    }

    def _post_init(self) -> None:
        if self.op == "pid":
            self.dtype = tl.int32
        elif self.op == "arange":  # Program ID / arange are always int32
            self.dtype = tl.block_type(tl.int32, [self.end.value - self.start.value])

    def _node_label_core(self) -> str:
        if self.op == "const":
            return f"const={self.value}"
        if self.op == "pid":
            axis_node = self.children.get("axis")
            if axis_node is None:
                raise ValueError("pid node is missing required children: axis")
            axis_val = axis_node.to_py()
            return f"pid_{axis_val}"
        return super()._node_label_core()

    def _to_z3_impl(self) -> None:
        builder = self._Z3_BUILDERS.get(self.op)
        if builder is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        builder(self)

    def _build_const(self) -> None:
        value = self.value
        if isinstance(value, TensorHandle):
            value = self.value.data

        if self.loop_ctx:  # if the self is a loop iterator
            self.z3 = self.loop_ctx.idx_z3
        elif isinstance(value, np.ndarray):
            self.z3 = [IntVal(int(v)) for v in value.flat]
        elif isinstance(value, tuple):
            self.z3 = [IntVal(int(v)) for v in value]
        elif isinstance(value, (int, float)):
            # Convert to int for Z3 - Z3's IntVal cannot parse float strings
            self.z3 = IntVal(int(value))
        elif value is None:
            # For None values, use 0 as a placeholder (e.g., for optional mask/other)
            self.z3 = IntVal(0)
        else:
            # For other types (e.g., tl.core.dtype), try converting to int
            self.z3 = IntVal(int(value))
        self.constraints = []

    def _build_pid(self) -> None:
        axis_node = self.children.get("axis")
        if axis_node is None:
            raise ValueError("pid node is missing required child: axis")

        axis_val = axis_node.to_py()
        if axis_val == 0:
            self.z3 = SymbolicExpr.PID0
        elif axis_val == 1:
            self.z3 = SymbolicExpr.PID1
        else:
            self.z3 = SymbolicExpr.PID2
        self.constraints = []

    def _build_arange(self) -> None:
        start = self.start.value
        end = self.end.value
        name = f"arange_{start}_{end}"
        key = (start, end)
        if key in SymbolicExpr.ARANGE_DICT:
            self.z3, self.constraints = SymbolicExpr.ARANGE_DICT[key]
            return
        v = Int(name)
        self.z3 = v
        self.constraints = [v >= start, v < end]
        SymbolicExpr.ARANGE_DICT[key] = (self.z3, self.constraints)

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["BasicSymbolicExpr"], None]]] = {
        "const": _build_const,
        "pid": _build_pid,
        "arange": _build_arange,
    }

    def _concretize_impl(self) -> Any:
        handler = self._CONCRETIZE_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Concretize for op {self.op} is not implemented")
        return handler(self)

    def _concretize_const(self) -> Any:
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

    def _concretize_pid(self) -> Any:
        fn = self.concrete_fn
        if fn is None:
            raise RuntimeError(f"{self.op}'s concrete function is not set!")
        return fn(self.axis.to_py())

    def _concretize_arange(self) -> Any:
        fn = self.concrete_fn
        if fn is None:
            raise RuntimeError(f"{self.op}'s concrete function is not set!")
        return fn(self.ret_ty.to_py(), self.start.to_py(), self.end.to_py())

    _CONCRETIZE_BUILDERS: ClassVar[dict[str, Callable[["BasicSymbolicExpr"], Any]]] = {
        "const": _concretize_const,
        "pid": _concretize_pid,
        "arange": _concretize_arange,
    }


class IndirectSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "load": Spec(req=("ptr",), opt=("mask", "other")),
        "store": Spec(req=("ptr", "value"), opt=("mask", "other")),
    }

    def _post_init(self) -> None:
        if self.op == "load":
            assert self.ptr is not None
            self.dtype = self.ptr.dtype.element_ty

    def _to_z3_impl(self) -> None:
        ptr, constraints_ptr = self.ptr._to_z3()
        constraints = list(constraints_ptr)
        if self.mask is not None:
            mask, _ = self.mask._to_z3()
            if isinstance(mask, list):
                constraints.extend(mask)
            else:
                constraints.append(mask)
        self.z3 = ptr
        self.constraints = constraints

    def _concretize_impl(self) -> Any:
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

        fn = self.concrete_fn
        if fn is None:
            raise RuntimeError(f"{self.op}'s concrete function is not set!")

        # load
        return fn(
            ptr_concrete,
            mask_concrete,
            other_concrete,
            None,  # cache_modifier
            None,  # eviction_policy
            None,  # is_volatile
        )


class UnarySymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        op: Spec(req=("arg",)) for op in SymbolicExpr.UNARY_OPS
    }

    def _to_z3_impl(self) -> None:
        val, constraints = self.arg._to_z3()
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Unary op {self.op} is not implemented")
        self.z3 = handler(val)
        self.constraints = constraints

    @staticmethod
    def _abs(val) -> Z3Expr:
        return If(val >= 0, val, -val)

    _Z3_BUILDERS: ClassVar[dict[str, Callable[[Z3Expr], Z3Expr]]] = {
        "abs": _abs,
        "fabs": _abs,
    }


class BinarySymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        op: Spec(req=("lhs", "rhs")) for op in SymbolicExpr.BINARY_OPS
    }

    def _post_init(self) -> None:
        lhs = self.children.get("lhs")
        rhs = self.children.get("rhs")
        if lhs is None or rhs is None:
            raise ValueError("Binary op requires both lhs and rhs operands")
        self.dtype = lhs.dtype

    def _to_z3_impl(self) -> None:
        lhs, constraints_lhs = self.lhs._to_z3()
        rhs, constraints_rhs = self.rhs._to_z3()
        self.constraints = constraints_lhs + constraints_rhs

        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        self.z3 = handler(self, lhs, rhs)

    def _concretize_impl(self) -> Any:
        fn = self.concrete_fn
        if fn is None:
            raise RuntimeError(f"{self.op}'s concrete function is not set!")
        lhs_concrete = self.lhs.concretize()
        rhs_concrete = self.rhs.concretize()
        np_op = self._NUMPY_OPS.get(self.op, None)
        if np_op is None:
            raise NotImplementedError(f"Concretize for op {self.op} is not implemented")
        return fn(lhs_concrete, rhs_concrete, np_op)

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
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(self.lhs)
            or self._infer_bitwidth(self.rhs)
            or 64
        )

        def _bit_and(a, b):
            return self._from_bv(self._to_bv(a, bitwidth) & self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_and, lhs, rhs)

    def _op_bitwise_or(self, lhs, rhs):
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(self.lhs)
            or self._infer_bitwidth(self.rhs)
            or 64
        )

        def _bit_or(a, b):
            return self._from_bv(self._to_bv(a, bitwidth) | self._to_bv(b, bitwidth))

        return self._apply_binop(_bit_or, lhs, rhs)

    def _op_bitwise_xor(self, lhs, rhs):
        bitwidth = (
            self._infer_bitwidth(self)
            or self._infer_bitwidth(self.lhs)
            or self._infer_bitwidth(self.rhs)
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


class TernarySymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "where": Spec(req=("cond", "lhs", "rhs")),
    }

    def _post_init(self) -> None:
        self.dtype = self.lhs.dtype

    def _to_z3_impl(self) -> None:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        handler(self)

    def _where(self) -> None:
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

        cond, constraints_cond = self.cond._to_z3()
        lhs, constraints_lhs = self.lhs._to_z3()
        rhs, constraints_rhs = self.rhs._to_z3()

        cond = _normalize(cond)
        lhs = _normalize(lhs)
        rhs = _normalize(rhs)
        cond, lhs, rhs = _broadcast(cond, lhs, rhs)

        if not (len(cond) == len(lhs) == len(rhs)):
            raise ValueError(
                f"where op requires cond, lhs, rhs to have the same length, got {len(cond)}, {len(lhs)}, {len(rhs)}"
            )
        self.z3 = [If(cond[i], lhs[i], rhs[i]) for i in range(len(cond))]
        self.constraints = constraints_cond + constraints_lhs + constraints_rhs

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["TernarySymbolicExpr"], None]]] = {
        "where": _where,
    }


class ReduceSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "sum": Spec(req=("input", "axis", "keepdims")),
        "max": Spec(req=("input", "axis", "keepdims")),
        "min": Spec(req=("input", "axis", "keepdims")),
        "dot": Spec(req=("a", "b"), opt=("d",)),
    }

    def _to_z3_impl(self) -> None:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        handler(self)

    def _reduce_sum(self) -> None:
        arr, self.constraints = self.input._to_z3()
        self.z3 = Sum(arr)

    def _reduce_max(self) -> None:
        arr, self.constraints = self.input._to_z3()
        self.z3 = reduce(lambda a, b: If(a >= b, a, b), arr)

    def _reduce_min(self) -> None:
        arr, self.constraints = self.input._to_z3()
        self.z3 = reduce(lambda a, b: If(a <= b, a, b), arr)

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["ReduceSymbolicExpr"], None]]] = {
        "sum": _reduce_sum,
        "max": _reduce_max,
        "min": _reduce_min,
    }


class ScanSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "cumsum": Spec(req=("input", "axis", "reverse", "dtype")),
    }

    def _to_z3_impl(self) -> None:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")


class PointerSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "make_block_ptr": Spec(
            req=("base", "shape", "strides", "offsets", "block_shape", "order")
        ),
        "addptr": Spec(req=("ptr", "offset")),
        "advance": Spec(req=("ptr", "offsets")),
    }
    _INT_DTYPES: ClassVar[tuple[type, ...]] = (int, np.integer, bool)

    def _post_init(self) -> None:
        if self.op == "addptr":
            assert self.ptr is not None
            self.dtype = self.ptr.dtype.element_ty

    def _to_z3_impl(self) -> None:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        handler(self)

    def _concretize_impl(self) -> Any:
        fn = self.concrete_fn
        if fn is None:
            raise RuntimeError(f"{self.op}'s concrete function is not set!")
        if self.op == "addptr":
            return fn(self.ptr.concretize(), self.offset.concretize())
        raise NotImplementedError(f"Concretize for op {self.op} is not implemented")

    def _addptr(self) -> None:
        ptr_z3, constraints_ptr = self.ptr._to_z3()
        offset_z3, constraints_offset = self.offset._to_z3()
        self.constraints = constraints_ptr + constraints_offset
        element_bytewidth = max(
            1, self.ptr.dtype.scalar.element_ty.primitive_bitwidth // 8
        )
        if isinstance(ptr_z3, list) and isinstance(offset_z3, list):
            if len(ptr_z3) != len(offset_z3):
                raise ValueError(
                    f"ptr {ptr_z3} and offset {offset_z3} don't have the same length!"
                )
            self.z3 = [p + o * element_bytewidth for p, o in zip(ptr_z3, offset_z3)]
        elif isinstance(ptr_z3, list):
            self.z3 = [p + offset_z3 * element_bytewidth for p in ptr_z3]
        elif isinstance(offset_z3, list):
            self.z3 = [ptr_z3 + o * element_bytewidth for o in offset_z3]
        else:
            self.z3 = ptr_z3 + offset_z3 * element_bytewidth

    def _advance(self) -> None:
        raise NotImplementedError("Advance operation is not implemented yet")

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["PointerSymbolicExpr"], None]]] = {
        "addptr": _addptr,
        "advance": _advance,
    }


class ReshapeSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "splat": Spec(req=("shape", "arg")),
        "expand_dims": Spec(req=("arg", "axis")),
        "broadcast": Spec(req=("arg", "shape")),
        "reshape": Spec(req=("arg", "shape")),
        "trans": Spec(req=("arg", "permutation")),
        "join": Spec(req=("lhs", "rhs")),
    }

    def _post_init(self) -> None:
        if self.op == "join":
            lhs = self.children.get("lhs")
            if lhs is None:
                raise ValueError("Join op requires lhs")
            self.dtype = lhs.dtype
        elif self.op == "splat":
            shape_expr = self.children.get("shape")
            arg_expr = self.children.get("arg")
            if shape_expr is None or arg_expr is None or shape_expr.dtype is None:
                raise ValueError("Splat op requires shape and arg")
            shape = shape_expr.dtype.shape
            self.dtype = tl.block_type(arg_expr.dtype, shape)
        elif self.op in ("expand_dims", "broadcast", "reshape", "trans"):
            arg_expr = self.children.get("arg")
            if arg_expr is None:
                raise ValueError(f"{self.op} op requires arg")
            self.dtype = arg_expr.dtype

    def _to_z3_impl(self) -> None:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        handler(self)

    def _concretize_impl(self) -> Any:
        if self.op == "splat":
            fn = self.concrete_fn
            if fn is None:
                raise RuntimeError(f"{self.op}'s concrete function is not set!")
            shape_expr = self.children.get("shape")
            arg_expr = self.children.get("arg")
            if shape_expr is None or arg_expr is None:
                raise ValueError("Splat op requires shape and arg")
            return fn(shape_expr.to_py(), arg_expr.concretize())
        return super()._concretize_impl()

    def _reshape_passthrough(self) -> None:
        self.z3, self.constraints = self.arg._to_z3()

    def _join(self) -> None:
        raise NotImplementedError(
            "Join operation is not implemented in Z3 evaluation yet"
        )

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["ReshapeSymbolicExpr"], None]]] = {
        "splat": _reshape_passthrough,
        "expand_dims": _reshape_passthrough,
        "broadcast": _reshape_passthrough,
        "reshape": _reshape_passthrough,
        "trans": _reshape_passthrough,
        "join": _join,
    }


class CastSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "cast_impl": Spec(req=("src", "dst_type")),
        "bitcast": Spec(req=("src", "dst_type")),
        "fp_to_fp": Spec(req=("src", "dst_type", "rounding_mode")),
    }

    def _post_init(self) -> None:
        if self.op in ("cast_impl", "bitcast", "fp_to_fp"):
            assert self.dst_type is not None
            if self.dst_type.op == "const":
                self.dtype = self.dst_type.value

    def _to_z3_impl(self) -> None:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        handler(self)

    def _concretize_impl(self) -> Any:
        if self.op in ("cast_impl", "bitcast"):
            fn = self.concrete_fn
            if fn is None:
                raise RuntimeError(f"{self.op}'s concrete function is not set!")
            src_concrete = self.src.concretize()
            dst_type_value = (
                self.dst_type.value
                if hasattr(self.dst_type, "value")
                else self.dst_type
            )
            return fn(src_concrete, dst_type_value)
        return super()._concretize_impl()

    def _cast_passthrough(self) -> None:
        self.z3, self.constraints = self.src._to_z3()

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["CastSymbolicExpr"], None]]] = {
        "cast_impl": _cast_passthrough,
        "bitcast": _cast_passthrough,
    }


class AtomicSymbolicExpr(SymbolicExpr):
    OP_SPEC: ClassVar[dict[str, Spec]] = {
        "atomic_cas": Spec(req=("ptr", "cmp", "val")),
        "atomic_rmw": Spec(req=("ptr", "val", "mask")),
    }

    def _to_z3_impl(self) -> None:
        handler = self._Z3_BUILDERS.get(self.op)
        if handler is None:
            raise NotImplementedError(f"Eval for op {self.op} is not implemented")
        handler(self)

    def _atomic_cas(self) -> None:
        raise NotImplementedError("atomic_cas operation is not implemented yet")

    def _atomic_rmw(self) -> None:
        raise NotImplementedError(f"Eval for op {self.op} is not implemented")

    _Z3_BUILDERS: ClassVar[dict[str, Callable[["AtomicSymbolicExpr"], None]]] = {
        "atomic_cas": _atomic_cas,
        "atomic_rmw": _atomic_rmw,
    }


SymbolicExpr.register_op_class(BasicSymbolicExpr, SymbolicExpr.BASIC_OPS)
SymbolicExpr.register_op_class(IndirectSymbolicExpr, SymbolicExpr.INDIRECT_OPS)
SymbolicExpr.register_op_class(UnarySymbolicExpr, SymbolicExpr.UNARY_OPS)
SymbolicExpr.register_op_class(BinarySymbolicExpr, SymbolicExpr.BINARY_OPS)
SymbolicExpr.register_op_class(TernarySymbolicExpr, SymbolicExpr.TERNARY_OPS)
SymbolicExpr.register_op_class(ReduceSymbolicExpr, SymbolicExpr.REDUCE_OPS)
SymbolicExpr.register_op_class(ScanSymbolicExpr, SymbolicExpr.SCAN_OPS)
SymbolicExpr.register_op_class(PointerSymbolicExpr, SymbolicExpr.POINTER_OPS)
SymbolicExpr.register_op_class(ReshapeSymbolicExpr, SymbolicExpr.RESHAPE_OPS)
SymbolicExpr.register_op_class(CastSymbolicExpr, SymbolicExpr.CAST_OPS)
SymbolicExpr.register_op_class(AtomicSymbolicExpr, SymbolicExpr.ATOMIC_OPS)


def _sexpr_or_str(expr: Any) -> str:
    if isinstance(expr, (int, np.integer)):
        return str(int(expr))
    sexpr = getattr(expr, "sexpr", None)
    if callable(sexpr):
        return sexpr()
    return str(expr)


def _make_signature(
    addr_expr: Z3Expr, constraints: list[BoolRef], re_pattern: re.Pattern[str]
) -> int:
    """
    Convert (addr, constraints) into a stable string signature.
    • addr_expr can be a single z3 expr or list[expr]
    • constraints is list[expr]
    """
    if isinstance(addr_expr, list):
        addr_repr = "|".join(sorted(_sexpr_or_str(e) for e in addr_expr))
    else:
        addr_repr = _sexpr_or_str(addr_expr)

    constr_repr = "|".join(sorted(c.sexpr() for c in constraints))

    addr_repr = re_pattern.sub(r"\1", addr_repr)
    constr_repr = re_pattern.sub(r"\1", constr_repr)
    return hash(addr_repr + "##" + constr_repr)


@dataclass(frozen=True)
class _FnSymbolicCache:
    fn: Callable
    grid: tuple[int, ...]
    args: tuple

    @cached_property
    def hash_value(self):
        return hash((self.fn, self.grid, self.args))

    def __hash__(self):
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

    def _collect_pointer_bases(self, expr: Optional[SymbolicExpr]) -> list[int]:
        if expr is None:
            return []

        bases: list[int] = []
        seen: set[int] = set()

        def walk(node: SymbolicExpr) -> None:
            if node.op == "const" and isinstance(node.dtype, tl.pointer_type):
                try:
                    val = node.to_py()
                except Exception:
                    val = None
                if isinstance(val, (int, np.integer)):
                    base = int(val)
                    if base not in seen:
                        seen.add(base)
                        bases.append(base)
            for child in node.children.values():
                if child is not None:
                    walk(child)

        walk(expr)
        return bases

    def _find_tensor_for_expr(
        self, symbolic_expr: Optional[SymbolicExpr], violation_addr: int
    ) -> Optional[Tensor]:
        # Prefer mapping from pointer base addresses present in the expression.
        base_candidates = self._collect_pointer_bases(symbolic_expr)
        if base_candidates:
            for base in base_candidates:
                for tensor in self.tensors:
                    if tensor.data_ptr() == base:
                        return tensor
            for base in base_candidates:
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

        return None

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
        expr_constraints: list[BoolRef],
        symbolic_expr: SymbolicExpr,
    ) -> None:
        if isinstance(access_addr, list):
            for addr in access_addr:
                self._check_range_satisfiable(addr, expr_constraints, symbolic_expr)
            return

        # Use push/pop on persistent solver
        assert self.solver is not None
        assert self.addr_sym is not None
        self.solver.push()
        self.solver.add(self.addr_sym == access_addr)
        bool_constraints: list[BoolRef] = []
        for c in expr_constraints:
            if isinstance(c, BoolRef):
                bool_constraints.append(c)
            elif isinstance(c, bool):
                bool_constraints.append(BoolVal(c))
            elif isinstance(c, IntNumRef):
                bool_constraints.append(BoolVal(c.as_long() != 0))
            elif isinstance(c, (int, float)):
                bool_constraints.append(BoolVal(int(c) != 0))
            else:
                bool_constraints.append(cast(BoolRef, c != 0))
        self.solver.add(And(*bool_constraints))
        if self.solver.check() == sat:
            # Get the model to find the violation address
            model = self.solver.model()
            violation_val = model.evaluate(self.addr_sym, model_completion=True)
            if isinstance(violation_val, IntNumRef):
                violation_addr = violation_val.as_long()
            else:
                raise RuntimeError("Unexpected violation address type from Z3 model!")

            # Find the tensor that this address belongs to
            tensor = self._find_tensor_for_expr(symbolic_expr, violation_addr)

            # Determine operation type from symbolic expression
            if symbolic_expr.op == "store":
                op_type: type[Load] | type[Store] = Store
            else:
                op_type = Load

            # Report with symbolic expression
            self._report(op_type, tensor, violation_addr, symbolic_expr)
        self.solver.pop()

    def _handle_access_check(self, expr: SymbolicExpr) -> None:
        """
        Evaluate a memory access expression and either defer it (inside a loop)
        or check it immediately (outside a loop).

        Returns nothing; duplicate addresses inside a loop are skipped.
        """
        # check memory access using z3
        z3_addr, z3_constraints = expr.eval()
        if self.loop_stack:  # for-loop iterator association
            ctx = self.loop_stack[-1]

            # check if addr already appeared before in the for-loop
            signature = _make_signature(z3_addr, z3_constraints, ctx.re_pattern)
            if signature in ctx.signature_cache:  # if appeared before
                if cfg.verbose:
                    print("[Sanitizer]  ↪ skip duplicated addr in loop")
            else:  # new addr expr
                if cfg.verbose:
                    print(
                        "[Sanitizer]  ↪ new addr in for-loop, will check later",
                        z3_addr,
                        z3_constraints,
                    )
                ctx.signature_cache.add(signature)
                # Store the expression along with the z3 data for later checking
                ctx.pending_checks.append((z3_addr, z3_constraints, expr))
        else:  # non-loop case
            self._check_range_satisfiable(z3_addr, z3_constraints, expr)

    def _report(
        self,
        op_type: type[Load] | type[Store],
        tensor: Tensor,
        violation_address: int,
        symbolic_expr: Optional[SymbolicExpr] = None,
    ) -> None:
        traceback_info = _get_traceback_info()
        tensor_name = self._get_tensor_name(tensor)
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=[],
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

        assert self.addr_ok is not None
        assert self.pid_ok is not None

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
            _unary_map = {
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
            arg_sym = SymbolicExpr.from_value(arg)
            try:
                name = _unary_map[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported unary operation: {op} on {arg_sym}"
                )
            return SymbolicExpr.create(name, arg_sym)

        def op_binary_op_overrider(lhs, rhs, op):
            _binary_map = {
                np.add: lambda lhs, rhs: lhs + rhs,
                np.subtract: lambda lhs, rhs: lhs - rhs,
                np.multiply: lambda lhs, rhs: lhs * rhs,
                np.divide: lambda lhs, rhs: lhs / rhs,
                np.less: lambda lhs, rhs: lhs < rhs,
                np.less_equal: lambda lhs, rhs: lhs <= rhs,
                np.greater: lambda lhs, rhs: lhs > rhs,
                np.greater_equal: lambda lhs, rhs: lhs >= rhs,
                np.not_equal: lambda lhs, rhs: lhs != rhs,
                np.equal: lambda lhs, rhs: lhs == rhs,
                np.fmod: lambda lhs, rhs: lhs % rhs,
                np.maximum: lambda lhs, rhs: SymbolicExpr.create("maximum", lhs, rhs),
                np.minimum: lambda lhs, rhs: SymbolicExpr.create("minimum", lhs, rhs),
                np.bitwise_and: lambda lhs, rhs: SymbolicExpr.create(
                    "bitwise_and", lhs, rhs
                ),
                np.bitwise_or: lambda lhs, rhs: SymbolicExpr.create(
                    "bitwise_or", lhs, rhs
                ),
                np.bitwise_xor: lambda lhs, rhs: SymbolicExpr.create(
                    "bitwise_xor", lhs, rhs
                ),
                np.right_shift: lambda lhs, rhs: SymbolicExpr.create(
                    "right_shift", lhs, rhs
                ),
                np.left_shift: lambda lhs, rhs: SymbolicExpr.create(
                    "left_shift", lhs, rhs
                ),
            }
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            try:
                func = _binary_map[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported binary operation: {op} between {lhs_sym} and {rhs_sym}"
                )
            result = func(lhs_sym, rhs_sym)
            return result

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
            if isinstance(expr, ArithRef):
                expr = SymbolicExpr.create(
                    "const", SymbolicExprDataWrapper.coerce_int(expr), tl.int32
                )

            if isinstance(expr, SymbolicExpr):
                if expr.op == "const":
                    return SymbolicExprDataWrapper.coerce_int(expr.to_py())
                self.need_full_grid = True
                expr = expr.replace_subtree()
                return SymbolicExprDataWrapper.coerce_int(expr.to_py())

            return int(expr)

        def _get_constant_step(step_expr: Any) -> int:
            step = _materialize_loop_value(step_expr)
            if step == 0:
                raise ValueError("Loop step cannot be zero.")
            return step

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

            step = _get_constant_step(step_expr)
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
            self.loop_stack.append(
                LoopContext(
                    lineno,
                    iterable.length,
                    idx_z3,
                    start=iterable.start,
                    stop=iterable.stop,
                    step=iterable.step,
                )
            )
            if cfg.verbose:
                print(f"[Sanitizer] ▶ enter loop@{lineno}, len={iterable.length}")

        @self.lock_fn
        def loop_hook_iter_overrider(lineno, idx):
            if self.loop_stack and self.loop_stack[-1].lineno == lineno:
                self.loop_stack[-1].values.append(idx)
                sym = SymbolicExpr.create("const", idx, tl.int32)
                sym.loop_ctx = self.loop_stack[-1]
                return tl.core.tensor(sym, tl.int32)
            return idx

        @self.lock_fn
        def loop_hook_iter_listener(lineno, idx):
            if cfg.verbose:
                print(f"[Sanitizer] ▶ loop@{lineno} idx={idx}")

        @self.lock_fn
        def loop_hook_after(lineno: int) -> None:
            if not self.loop_stack or self.loop_stack[-1].lineno != lineno:
                return
            ctx = self.loop_stack.pop()
            # add constraints for loop_i
            iterator_constraints: list[Z3Expr] = []
            if ctx.values:
                iterator_constraints.append(
                    Or(*[ctx.idx_z3 == v for v in set(ctx.values)])
                )

            def _filter_constraints(addr_expr, constraints):
                addr_eq = self.addr_sym == addr_expr
                relevant_vars = set(get_vars(addr_eq))
                filtered = []
                for constraint in constraints:
                    constraint_vars = set(get_vars(constraint))
                    if constraint_vars and not constraint_vars.issubset(relevant_vars):
                        continue
                    filtered.append(constraint)
                return filtered

            # execute pending checks
            for check_tuple in ctx.pending_checks:
                # Handle both old format (2-tuple) and new format (3-tuple)
                addr_expr, expr_constraints, symbolic_expr = check_tuple

                combined_constraints = expr_constraints + iterator_constraints

                if cfg.verbose:
                    print(
                        "[Sanitizer] ▶ checking:",
                        addr_expr,
                        f" with iterator constraints: {iterator_constraints} ",
                        f" and expression-related constraints: {expr_constraints} ",
                    )

                if isinstance(addr_expr, list):
                    for single_addr_expr in addr_expr:
                        filtered_constraints = _filter_constraints(
                            single_addr_expr, combined_constraints
                        )
                        self._check_range_satisfiable(
                            single_addr_expr,
                            filtered_constraints,
                            symbolic_expr,
                        )
                else:
                    filtered_constraints = _filter_constraints(
                        addr_expr, combined_constraints
                    )
                    self._check_range_satisfiable(
                        addr_expr,
                        filtered_constraints,
                        symbolic_expr,
                    )

            if cfg.verbose:
                print(
                    f"[Sanitizer] ▶ leave loop@{lineno} end. "
                    f"(checked {len(ctx.pending_checks)} unique addr patterns)"
                )

        return ForLoopCallbacks(
            range_wrapper_factory=_wrap_range,
            before_loop_callback=loop_hook_before,
            loop_iter_overrider=loop_hook_iter_overrider,
            loop_iter_listener=loop_hook_iter_listener,
            after_loop_callback=loop_hook_after,
        )

    def finalize(self) -> list:
        return []


class NullSanitizer(Sanitizer):
    """
    A do-nothing object returned when the sanitizer backend is 'off'.
    Any attribute access raises an explicit error so misuse is obvious.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize parent class

    def _disabled(self, method: str):
        raise RuntimeError(
            f"[NullSanitizer] '{method}' was called, "
            "but sanitizer backend is off; no functionality is available."
        )

    def arg_callback(self, *args, **kwargs):
        self._disabled("arg_callback")

    def finalize(self, *args, **kwargs):
        self._disabled("finalize")

    def grid_callback(self, *args, **kwargs):
        self._disabled("grid_callback")

    def grid_idx_callback(self, *args, **kwargs):
        self._disabled("grid_idx_callback")

    def register_op_callback(self, *args, **kwargs):
        self._disabled("register_op_callback")

    def register_for_loop_callback(self, *args, **kwargs):
        self._disabled("register_for_loop_callback")

    def __getattr__(self, name):
        self._disabled(name)
