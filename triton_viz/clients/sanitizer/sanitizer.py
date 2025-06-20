import traceback
from abc import ABC
from typing import Tuple, Callable, Optional, Type
from collections import namedtuple

import numpy as np
from anytree import Node, RenderTree
from z3 import Solver, Int, IntVal, If, Sum, And, Or, Not, sat, simplify

import triton.language as tl
from triton.runtime.interpreter import TensorHandle

from ...core.client import Client
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
    Splat,
    MakeBlockPointer,
    TensorPointerLoad,
    TensorPointerStore,
    Idiv,
    Rsqrt,
    CastImpl,
)
from ..utils import (
    check_out_of_bounds_access,
    check_storage_contiguous,
    get_physical_addr_from_tensor_slice,
    check_inner_stride_equal_to_one,
)
from .data import (
    TracebackInfo,
    OutOfBoundsRecord,
    OutOfBoundsRecordBruteForce,
    OutOfBoundsRecordZ3,
)
from ...core import config as cfg


def print_oob_record(oob_record: OutOfBoundsRecord, max_display=10):
    """
    Print detailed logs for a given OOB record.

    Parameters
    ----------
    oob_record : OutOfBoundsRecord
        The record containing information about out-of-bounds accesses.
    max_display : int
        Maximum number of invalid accesses to display in detail.
    """
    if issubclass(oob_record.op_type, Store):
        op_type = "Store"
    elif issubclass(oob_record.op_type, Load):
        op_type = "Load"
    else:
        assert False, "Not supported op type: " + str(oob_record.op_type)

    # Read the tensor from the record
    tensor = oob_record.tensor

    # Basic info about the OOB event
    print("============================================================")
    print("                 Out-Of-Bounds Access Detected              ")
    print("============================================================")
    print(f"Operation: {op_type}")
    print(
        f"Tensor Info: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}"
    )
    print(f"Tensor base memory address: {tensor.data_ptr()}")
    print(
        "Valid Access Range: [0, %d)" % (np.prod(tensor.shape) * tensor.element_size())
    )
    for traceback_info in oob_record.user_code_tracebacks:
        print(
            f"File: {traceback_info.filename}, Line: {traceback_info.lineno}, in {traceback_info.func_name}"
        )
        print(f"  Code: {traceback_info.line_of_code}")
    print("------------------------------------------------------------")

    if isinstance(oob_record, OutOfBoundsRecordBruteForce):
        # Convert memoryviews to NumPy arrays
        offsets_arr = np.array(oob_record.offsets)
        invalid_access_masks_arr = np.array(oob_record.invalid_access_masks)

        # Determine all invalid indices
        invalid_indices = np.where(invalid_access_masks_arr.flatten())[0]
        assert len(invalid_indices) != 0, "No invalid accesses found in this record."

        # Print OOB details
        print(f"Total invalid accesses: {len(invalid_indices)}")
        invalid_offsets = offsets_arr.flatten()[invalid_indices]
        print("Invalid offsets:")
        print(invalid_offsets)

    elif isinstance(oob_record, OutOfBoundsRecordZ3):
        # Read the violation index and constraints
        violation_address = oob_record.violation_address
        constraints = oob_record.constraints

        # Print OOB details
        print(f"Invalid access detected at address: {violation_address}")
        print("Constraints:")
        for constraint in constraints:
            print(constraint)

    else:
        raise NotImplementedError(
            "Invalid OutOfBoundsRecord type: " + str(type(oob_record))
        )

    print("============================================================")
    print("            End of Out-Of-Bounds Record Details             ")
    print("============================================================")


def _get_traceback_info():
    """
    Why do both _grid_executor_call and _jit_function_call appear in the call stacks?

    1) Main kernel dispatch (kernel[grid](...)) triggers _grid_executor_call.
    2) Inlined @triton.jit functions trigger _jit_function_call.
    3) Some code sees only _grid_executor_call if no separate JIT function is present or patched.
    4) Complex kernels (e.g., fused_attention) may show both: outer dispatch and inner JIT calls.
    """
    oob_filename, oob_lineno, oob_func_name, oob_line_of_code = "", -1, "", ""

    stack_summary = traceback.extract_stack()

    user_code_tracebacks = []
    # scan the call stack
    for i, frame in enumerate(stack_summary):
        user_code_index = None
        if (
            "_jit_function_call" in frame.name
            and "triton_viz/core/patch.py" in frame.filename
        ):
            user_code_index = (
                i + 2
            )  # _grid_executor_call -> run_grid_loops -> user code
        elif (
            "_grid_executor_call" in frame.name
            and "triton_viz/core/patch.py" in frame.filename
        ):
            user_code_index = i + 2  # the same as above

        if user_code_index is not None:
            frame = stack_summary[user_code_index]
            oob_filename = frame.filename
            oob_lineno = frame.lineno
            oob_func_name = frame.name
            oob_line_of_code = frame.line
            traceback_info = TracebackInfo(
                filename=oob_filename,
                lineno=oob_lineno,
                func_name=oob_func_name,
                line_of_code=oob_line_of_code,
            )
            user_code_tracebacks.append(traceback_info)

    return user_code_tracebacks


def _get_tensor(tensor_list, data_ptr):
    # From a give ptr, get where the original tensor is stored
    # Tensors have been sorted by ptr
    ret_idx = 0
    for i in range(len(tensor_list)):
        if data_ptr < tensor_list[i].data_ptr():
            break
        ret_idx = i
    return tensor_list[ret_idx]


class SanitizerBruteForce(Client):
    def __init__(
        self, callpath: Optional[bool] = True, abort_on_error: Optional[bool] = True
    ):
        self.callpath = callpath
        self.abort_on_error = abort_on_error
        self.tensors: list = []
        self.records: list = []

    def _report(self, op_type, record):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordBruteForce(
            op_type=op_type, user_code_tracebacks=traceback_info, **record
        )
        if self.abort_on_error:
            if np.any(oob_record.invalid_access_masks):
                print_oob_record(oob_record)
                assert (
                    False
                ), "Out-of-bounds access detected. See detailed report above."
        else:
            self.records.append(oob_record)

    def arg_callback(self, arg, arg_cvt):
        if hasattr(arg, "data_ptr"):
            assert check_storage_contiguous(
                arg
            ), "The address sanitizer only supports contiguouly stored tensors for now"
            self.tensors.append(arg)

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    def grid_callback(self, grid: Tuple[int]):
        self.tensors = sorted(self.tensors, key=lambda x: x.data_ptr())

    def register_op_callback(
        self, op_type: Type[Op]
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        def pre_load_callback(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(op_type, oob)
            ptr.data = tensor.data_ptr() + oob["corrected_offsets"]

        def pre_store_callback(ptr, value, mask, cache_modifier, eviction_policy):
            first_loc = np.unravel_index(np.argmax(mask, axis=None), mask.data.shape)
            first_ptr = ptr.data[first_loc]
            tensor = _get_tensor(self.tensors, first_ptr)
            oob = check_out_of_bounds_access(ptr.data, mask.data, tensor)
            self._report(
                op_type, check_out_of_bounds_access(ptr.data, mask.data, tensor)
            )
            ptr.data = tensor.data_ptr() + oob["corrected_offsets"]

        if op_type is Load:
            return pre_load_callback, None, None
        elif op_type is Store:
            return pre_store_callback, None, None

        return None, None, None

    def finalize(self) -> list:
        return self.records


class SymbolicExprDataWrapper:
    """
    This wrapper is used as a workaround of triton interpreter legacy code.
    In def _get_bool(self) of class tensor,
        "data = self.handle.data
        return bool(data) if data.size == 1 else True"
    Since we replaced TensorHandle with SymbolicExpr,
    we need to wrap SymbolicExpr with a class that has size attribute, and data.size != 1.
    """

    def __init__(self, value, symbolic_expr):
        self.value = value
        self.symbolic_expr = symbolic_expr

    @property
    def size(self):
        return 2

    def __int__(self):
        int_val = self.symbolic_expr.eval()
        if not isinstance(int_val, int):
            raise ValueError(
                f"SymbolicExprDataWrapper is type: {type(int_val)}, value: {int_val} and cannot be converted to int"
            )
        return int_val

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


Spec = namedtuple(
    "Spec",
    [
        "req",  # Required Arguments (tuple)
        "opt",  # Optional Arguments (tuple)
        "post",  # post-hook: def(self) -> None
    ],
    defaults=((), (), None),
)


def _load_dtype(self):
    self.set_element_ty(self.children["ptr"].get_element_ty())


def _store_dtype(self):
    self.set_element_ty(self.children["ptr"].get_element_ty())


def _broadcast_dtype(self):
    self.dtype_tt = self.children["arg"].dtype_tt


def _binary_dtype(expr):
    expr.dtype_tt = expr.lhs.dtype_tt


def _binary_shape(expr):
    lhs, rhs = expr.children["lhs"], expr.children["rhs"]
    if not lhs.shape:
        expr.shape = rhs.shape
    elif not rhs.shape:
        expr.shape = lhs.shape
    else:
        assert lhs.shape == rhs.shape, f"lhs shape {lhs.shape} != rhs shape {rhs.shape}"
        expr.shape = lhs.shape


def _binary_post(expr):
    _binary_shape(expr)
    _binary_dtype(expr)


def _pid_post(expr):
    expr.dtype_tt = tl.int32  # Program ID is always int32


def _arange_post(expr):
    expr.dtype_tt = tl.int32  # tl.arange is always int32


def _cast_impl_post(expr):
    expr.dtype_tt = expr.dst_type


class SymbolicExpr:
    BASIC_OPS = ("const", "pid", "arange")
    INDIRECT_OPS = ("load", "store")
    UNARY_OPS = (
        "cos",
        "exp",
        "exp2",
        "abs",
        "floor",
        "ceil",
        "log",
        "log2",
        "sqrt",
        "sin",
        "rsqrt",
    )
    BINARY_OP_SYMBOL_TABLE = {
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
        "bitwise_and": "&",
    }
    BINARY_OPS = tuple(BINARY_OP_SYMBOL_TABLE.keys())
    TERNARY_OPS = ("where",)
    REDUCE_OPS = ("sum", "dot")
    POINTER_OPS = ("make_block_ptr",)
    BROADCAST_OPS = ("splat", "expand_dims", "broadcast")
    CAST_OPS = ("cast_impl",)
    SUPPORTED_OPS = (
        BASIC_OPS
        + INDIRECT_OPS
        + UNARY_OPS
        + BINARY_OPS
        + TERNARY_OPS
        + REDUCE_OPS
        + POINTER_OPS
        + BROADCAST_OPS
        + CAST_OPS
    )

    OP_SPEC = {
        # Core / primitive ops
        "arange": Spec(req=("start", "end"), post=_arange_post),
        "pid": Spec(req=("grid", "axis"), post=_pid_post),
        # Memory access ops
        "load": Spec(req=("ptr",), opt=("mask", "other"), post=_load_dtype),
        "store": Spec(req=("ptr", "value"), opt=("mask", "other"), post=_store_dtype),
        # Unary ops
        **{
            op: Spec(req=("arg",))
            for op in (
                "cos",
                "exp",
                "exp2",
                "abs",
                "floor",
                "ceil",
                "log",
                "log2",
                "sqrt",
                "sin",
                "rsqrt",
            )
        },
        #  Binary ops
        **{
            op: Spec(req=("lhs", "rhs"), post=_binary_post)
            for op in (
                "add",
                "sub",
                "mul",
                "div",
                "idiv",
                "mod",
                "less",
                "less_equal",
                "greater",
                "greater_equal",
                "not_equal",
                "equal",
                "maximum",
                "bitwise_and",
            )
        },
        # Ternary ops
        "where": Spec(req=("cond", "lhs", "rhs")),
        # Reduction ops
        "sum": Spec(req=("input", "axis", "keepdims")),
        "dot": Spec(req=("a", "b"), opt=("d",)),
        # Pointer utilities
        "make_block_ptr": Spec(
            req=("base", "shape", "strides", "offsets", "block_shape", "order")
        ),
        # Broadcasting / shape manipulation
        "splat": Spec(req=("arg", "shape"), post=_broadcast_dtype),
        "expand_dims": Spec(req=("arg", "axis"), post=_broadcast_dtype),
        "broadcast": Spec(req=("arg", "shape"), post=_broadcast_dtype),
        # Casting
        "cast_impl": Spec(req=("src", "dst_type"), post=_cast_impl_post),
    }

    def __init__(self, op, *args):
        """
        :param op: Operation type, e.g. "const", "add", "sub", "mul", "div", "pid", "arange"
        :param args: Sub-expressions (for compound operations)
        :param value: For "const" op, the constant value
        :param grid, axis: For "pid" op, the grid and axis
        """
        assert op in self.SUPPORTED_OPS, f"Unsupported op: {op}"
        self.op = op
        self.attrs = {}
        self.dtype_tt = None
        self.shape = []

        # Functions and arguments for concretization
        self._binary_numpy_op = None
        self._concrete_fn = None
        self._concrete_args = ()
        self._concrete_kwargs = {}

        # deal with args
        self.children = {}  # Used for storing child expressions
        if self.op == "const":  # leaf nodes
            self.value = args[0]
            if len(args) >= 2:
                self.dtype_tt = args[1]
        else:
            self._init_from_spec(*args)

    def _init_from_spec(self, *args):
        if self.op not in self.OP_SPEC:
            raise NotImplementedError(f"Unsupported op: {self.op}")
        spec = self.OP_SPEC[self.op]

        # check if args number match the spec
        min_n = len(spec.req)
        max_n = min_n + len(spec.opt)
        if not (min_n <= len(args) <= max_n):
            raise ValueError(
                f"{self.op} expects {min_n} - {max_n} args, got {len(args)}"
            )

        # store in self.children
        names = list(spec.req) + list(spec.opt)
        for name, val in zip(names, args):
            val = SymbolicExpr.from_value(val)
            self.children[name] = val
        for name in names[len(args) :]:
            self.children[name] = None

        # post-hook
        if spec.post:
            spec.post(self)

    def __getattr__(self, name):
        if name in self.children:
            return self.children[name]
        raise AttributeError(name)

    def set_attr(self, name, values):
        self.attrs[name] = values

    def get_element_ty(self):
        return self.dtype_tt

    def set_element_ty(self, dtype_tt):
        self.dtype_tt = dtype_tt

    def get_dtype_bytewidth(self, dtype_tt):
        element_bitwidth = dtype_tt.primitive_bitwidth
        element_bytewidth = max(1, element_bitwidth // 8)
        return element_bytewidth

    def __add__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("add", self, other)

    def __sub__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("sub", self, other)

    def __mul__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("mul", self, other)

    def __truediv__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("div", self, other)

    def __floordiv__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("idiv", self, other)

    def __mod__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("mod", self, other)

    def __lt__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("less", self, other)

    def __le__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("less_equal", self, other)

    def __ne__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("not_equal", self, other)

    def __eq__(self, other):
        assert isinstance(other, SymbolicExpr), "Operand must be a SymbolicExpr!"
        return SymbolicExpr("equal", self, other)

    def to_anytree(self):
        """Convert this SymbolicExpr into an anytree Node."""
        # Generate the node label
        label = self._node_label()
        root = Node(label)

        # Recursively add child nodes
        for child_key, child_symbolic_expr in self.children.items():
            if child_symbolic_expr is None:
                Node(f"{child_key}: None", parent=root)
                continue
            child_node = child_symbolic_expr.to_anytree()
            child_node.name = f"{child_key}: {child_node.name}"
            child_node.parent = root

        return root

    def _node_label(self):
        """Generate a short label for this node."""
        if self.op == "const":
            label = f"const={self.value}"
        elif self.op == "pid":
            axis_val = self.axis.to_py()
            grid_val = self.grid.to_py()
            label = f"pid_{axis_val}={grid_val[axis_val]}"
        else:
            label = self.op

        # Add the dtype to the label if available
        label = f"{label} [dtype={self.get_element_ty()}]"

        return label

    def to_tree_str(self) -> str:
        """
        Render the AST as an ASCII tree using anytree.RenderTree.
        """
        root = self.to_anytree()
        lines = []
        for prefix, _, node in RenderTree(root):
            lines.append(f"{prefix}{node.name}")
        return "\n" + "\n".join(lines)

    def __str__(self):
        return self.to_tree_str()

    def __repr__(self):
        return self.__str__()

    @property
    def data(self):
        return SymbolicExprDataWrapper(self.__str__(), self)

    triton_scala_dtypes = (
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
    builtin_scala_types = (int, float)
    tuple_types = (tl.core.tuple, tuple)

    @staticmethod
    def _infer_literal_dtype(var):
        if isinstance(var, tl.core.dtype):
            return var
        if isinstance(var, SymbolicExpr.tuple_types):
            first_dtype = SymbolicExpr._infer_literal_dtype(var[0])
            for v in var[1:]:  # assume only one consistent dtype in the tuple
                if SymbolicExpr._infer_literal_dtype(v) != first_dtype:
                    raise ValueError(
                        f"All elements in the tuple must have the same dtype, but found {first_dtype} and {SymbolicExpr.from_value(v).dtype_tt}"
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
                return var.get_element_ty()
        if isinstance(var, SymbolicExpr.builtin_scala_types):
            return tl.int32 if isinstance(var, int) else tl.float32
        raise ValueError(f"Unsupported type: {type(var)}")

    @classmethod
    def from_value(cls, var):
        if isinstance(var, cls):  # if already SymbolicExpr
            return var

        dtype_tt = SymbolicExpr._infer_literal_dtype(var)  # get the triton dtype

        if isinstance(var, (tl.core.tuple, tuple)):  # if a tuple
            return cls("const", tuple(var), dtype_tt)
        if isinstance(var, TensorHandle):  # if a TensorHandle
            return cls("const", var.data.item(), dtype_tt)
        if isinstance(
            var, SymbolicExpr.builtin_scala_types
        ):  # if a python builtin type
            return cls("const", var, dtype_tt)
        if isinstance(var, tl.core.dtype):
            # If it's a triton dtype, we can create a const node with it
            return cls("const", var, dtype_tt)

        raise ValueError("Unknown type:", type(var))

    def eval(self):
        """
        Returns a tuple (expr, constraints):
        - expr: Z3 expression corresponding to the root node
        - constraints: list of Z3 BoolExpr objects, recording all range constraints created by program_id and arange
        """
        self._arange_counter = 0  # Used to name arange variables
        self._arange_dict = {}  # make sure each arange only has one name
        self._vars = {}
        self._constraints = []
        expr = self._to_z3(self)
        if not self._constraints:
            assert expr.is_int(), "The address expression should be an integer"
            return simplify(expr).as_long()
        return expr, self._constraints

    def _to_z3(self, node):
        # Recursively convert the current node to a Z3 expression
        if node.op == "const":
            return IntVal(node.value)

        if node.op == "pid":
            axis_val = node.axis.to_py()
            grid_val = node.grid.to_py()
            name = f"pid_{axis_val}"
            if name not in self._vars:
                v = Int(name)
                self._vars[name] = v
                # Add constraint: 0 ≤ pid < grid[axis]
                self._constraints.append(v >= 0)
                self._constraints.append(v < grid_val[axis_val])
            return self._vars[name]

        if node.op == "arange":
            if id(node) in self._arange_dict:
                return self._arange_dict[id(node)]
            idx = self._arange_counter
            self._arange_counter += 1
            name = f"arange_{idx}"
            v = Int(name)
            self._vars[name] = v
            start = node.start.value
            end = node.end.value
            self._constraints.append(v >= start)
            self._constraints.append(v < end)
            self._arange_dict[id(node)] = v
            return v

        # Unary operations (only abs is demonstrated here; others can be added using z3.Function as needed)
        if node.op in self.UNARY_OPS:
            val = self._to_z3(node.arg)
            if node.op == "abs":
                return If(val >= 0, val, -val)
            raise NotImplementedError(f"Unary op {node.op} is not implemented")

        # Binary arithmetic, comparison, etc.
        if node.op in self.BINARY_OPS:
            lhs = self._to_z3(node.lhs)
            rhs = self._to_z3(node.rhs)
            if node.op == "add":
                return lhs + rhs
            if node.op == "sub":
                return lhs - rhs
            if node.op == "mul":
                return lhs * rhs
            if node.op in ("idiv"):
                return lhs / rhs
            if node.op == "mod":
                return lhs % rhs
            if node.op == "less":
                return lhs < rhs
            if node.op == "less_equal":
                return lhs <= rhs
            if node.op == "greater":
                return lhs > rhs
            if node.op == "greater_equal":
                return lhs >= rhs
            if node.op == "equal":
                return lhs == rhs
            if node.op == "not_equal":
                return lhs != rhs
            if node.op == "maximum":
                return If(lhs >= rhs, lhs, rhs)
            if node.op == "bitwise_and":
                return And(lhs, rhs)

        # where(cond, lhs, rhs)
        if node.op == "where":
            cond = self._to_z3(node.cond)
            lhs = self._to_z3(node.lhs)
            rhs = self._to_z3(node.rhs)
            return If(cond, lhs, rhs)

        # sum(input, axis, keepdims)
        if node.op == "sum":
            arr = self._to_z3(node.input)
            return Sum(arr)

        if node.op == "load" or node.op == "store":
            # Load and store operations
            ptr = self._to_z3(node.ptr)
            if node.mask is not None:
                mask = self._to_z3(node.mask)
                self._constraints.append(mask)
            return ptr

        if node.op in ("splat", "expand_dims", "broadcast"):
            return self._to_z3(node.arg)

        # Other operations can be implemented as needed
        raise NotImplementedError(f"Eval for op {node.op} is not implemented")

    def has_op(self, op_name: str) -> bool:
        if self.op == op_name:
            return True
        for _, child_symbolic_expr in self.children.items():
            if child_symbolic_expr.has_op(op_name):
                return True
        return False

    def to_py(self):
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
                raise NotImplementedError(
                    "SymbolicExpr.to_py() for multi-element tensors is not implemented yet!"
                )
        return v

    _concrete_fn_cache = {}
    _binary_numpy_op_cache = {}

    @property
    def concrete_fn(self):
        """Return the concrete evaluation function bound to this node."""
        if self._concrete_fn is None and self.op in SymbolicExpr._concrete_fn_cache:
            self._concrete_fn = SymbolicExpr._concrete_fn_cache[self.op]
        return self._concrete_fn

    @concrete_fn.setter
    def concrete_fn(self, fn):
        """Bind / override the concrete evaluation function."""
        self._concrete_fn = fn
        SymbolicExpr._concrete_fn_cache[self.op] = fn

    @property
    def binary_numpy_op(self):
        """Return the numpy operation corresponding to this binary op."""
        if (
            self._binary_numpy_op is None
            and self.op in SymbolicExpr._binary_numpy_op_cache
        ):
            self._binary_numpy_op = SymbolicExpr._binary_numpy_op_cache[self.op]
        return self._binary_numpy_op

    @binary_numpy_op.setter
    def binary_numpy_op(self, op):
        """Bind / override the numpy operation for this binary op."""
        self._binary_numpy_op = op
        SymbolicExpr._binary_numpy_op_cache[self.op] = op

    def concretize(self_or_cls, obj=None):
        """
        Usage:
        1. expr.concretize()               — Evaluate this SymbolicExpr instance to its concrete value.
        2. SymbolicExpr.concretize(x)      — Recursively concretize any object x (which may contain nested SymbolicExprs).
        """
        if obj is None:  # expr.concretize()
            obj = self_or_cls
        elif isinstance(
            self_or_cls, SymbolicExpr
        ):  # expr.concretize(x), which is erroneous
            raise TypeError(
                "Either use expr.concretize() or SymbolicExpr.concretize(x)!"
            )

        # Non-SymbolicExpr -> return as-is
        if not isinstance(obj, SymbolicExpr):
            return obj

        # concretize logic
        if obj.concrete_fn is None:
            if obj.op == "const":
                result = TensorHandle(obj.value, obj.dtype_tt)
            else:
                raise RuntimeError(f"{obj.op}'s concrete function is not set!")
        elif obj.op == "pid":
            result = obj.concrete_fn(obj.axis.to_py())
        elif obj.op == "arange":
            result = obj.concrete_fn(obj.start.to_py(), obj.end.to_py() + 1)
        elif obj.op == "splat":
            result = obj.concrete_fn(
                obj.arg.concretize(), obj.children["shape"].to_py()
            )
        elif obj.op in SymbolicExpr.BINARY_OPS:
            result = obj.concrete_fn(
                obj.lhs.concretize(), obj.rhs.concretize(), obj.binary_numpy_op
            )
        else:
            concrete_args = [
                SymbolicExpr.concretize(v) for k, v in obj.children.items()
            ]
            result = obj.concrete_fn(*concrete_args)

        return result


class ConstTupleExpr(SymbolicExpr):
    def __init__(self, value):
        super().__init__("const", tuple(value))


class SanitizerSymbolicExecution(Client):
    def __init__(self, abort_on_error):
        self.abort_on_error = abort_on_error
        self.grid = None
        self.tensors = []
        self.constraints = []
        self.tensor_addrs = []
        self.unique_load_store_id = 0

    def _check_range_satisfiable(self, access_addr, expr_constraints):
        out_of_bound_constraint = Not(
            Or(
                *(
                    And(start <= access_addr, access_addr <= end)
                    for start, end in self.tensor_addrs
                )
            )
        )
        s = Solver()
        s.add(out_of_bound_constraint)
        s.add(And(*expr_constraints))
        if s.check() == sat:
            print("out of bound access detected!")

    def _report(self, op_type, tensor, violation_address):
        traceback_info = _get_traceback_info()
        oob_record = OutOfBoundsRecordZ3(
            op_type=op_type,
            user_code_tracebacks=traceback_info,
            tensor=tensor,
            violation_address=violation_address,
            constraints=self.constraints,
        )
        if self.abort_on_error:
            print_oob_record(oob_record)
            raise ValueError(
                "Out-of-bounds access detected. See detailed report above."
            )
        else:
            self.records.append(oob_record)

    def arg_callback(self, arg, arg_cvt):
        if not hasattr(arg, "data_ptr"):
            return
        if arg.is_contiguous or check_storage_contiguous(arg):
            start = arg.data_ptr()
            end = arg.data_ptr() + (arg.numel() - 1) * arg.element_size()
            tensor_physical_addresses = [(start, end)]
        elif check_inner_stride_equal_to_one(arg):
            tensor_physical_addresses = get_physical_addr_from_tensor_slice(arg)
        else:
            raise ValueError(
                "The address sanitizer only supports contiguouly stored tensors for now!"
            )

        self.tensors.append(arg)
        self.tensor_addrs.extend(tensor_physical_addresses)

    def grid_callback(self, grid: Tuple[int]):
        self.grid = tuple(int(g) for g in grid)

    def grid_idx_callback(self, grid_idx: Tuple[int]):
        pass

    def register_op_callback(
        self, op_type: Type[Op]
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        def op_program_id_overrider(axis):
            assert self.grid, "Grid not initialized!"
            return SymbolicExpr("pid", self.grid, axis)

        def op_raw_load_overrider(ptr, cache_modifier, eviction_policy, is_volatile):
            return op_load_overrider(
                ptr, None, None, cache_modifier, eviction_policy, is_volatile
            )

        def op_load_overrider(
            ptr, mask, other, cache_modifier, eviction_policy, is_volatile
        ):
            # deal with indirect loads
            if isinstance(ptr, SymbolicExpr) and ptr.has_op("load"):
                print("indirect loading:", ptr)
                # ptr = ptr.concretize()
                # print('concretized ptr:', ptr)

            # make sure ptr is a SymbolicExpr
            if isinstance(ptr, TensorHandle) and isinstance(ptr.dtype, tl.pointer_type):
                ptr_sym = SymbolicExpr.from_value(ptr)
            elif isinstance(ptr, SymbolicExpr):
                ptr_sym = ptr
            else:
                raise ValueError(f"Unsupported ptr type: {type(ptr)}")

            if mask is None:
                ret = SymbolicExpr("load", ptr_sym)
            elif other is None:
                ret = SymbolicExpr("load", ptr_sym, mask)
            else:
                ret = SymbolicExpr("load", ptr_sym, mask, other)

            # check memory access using z3
            ret_eval = ret.eval()
            if isinstance(ret_eval, int):
                self._check_range_satisfiable(ret_eval, [])
            elif isinstance(ret_eval, tuple):
                self._check_range_satisfiable(*ret_eval)
            else:
                raise ValueError(f"Unsupported ret_eval type: {type(ret_eval)}")
            return ret

        def op_raw_store_overrider(ptr, value, cache_modifier, eviction_policy):
            return op_store_overrider(ptr, value, None, cache_modifier, eviction_policy)

        def op_store_overrider(ptr, value, mask, cache_modifier, eviction_policy):
            # make sure ptr is a SymbolicExpr
            if isinstance(ptr, TensorHandle) and isinstance(ptr.dtype, tl.pointer_type):
                ptr_sym = SymbolicExpr.from_value(ptr)
            elif isinstance(ptr, SymbolicExpr):
                ptr_sym = ptr
            else:
                raise ValueError(f"Unsupported ptr type: {type(ptr)}")

            value = SymbolicExpr.from_value(value)
            if mask is None:
                ret = SymbolicExpr("store", ptr_sym, value)
            else:
                ret = SymbolicExpr("store", ptr_sym, value, mask)

            # check memory access using z3
            ret_eval = ret.eval()
            if isinstance(ret_eval, int):
                self._check_range_satisfiable(ret_eval, [])
            elif isinstance(ret_eval, tuple):
                self._check_range_satisfiable(*ret_eval)
            else:
                raise ValueError(f"Unsupported ret_eval type: {type(ret_eval)}")
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
            return SymbolicExpr(name, arg_sym)

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
                np.maximum: lambda lhs, rhs: SymbolicExpr("maximum", lhs, rhs),
                np.bitwise_and: lambda lhs, rhs: SymbolicExpr("bitwise_and", lhs, rhs),
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
            result.binary_numpy_op = op  # Store the numpy operation for later use
            return result

        def op_ternary_op_overrider(lhs, rhs, other, op):
            lhs_sym = SymbolicExpr.from_value(lhs)
            rhs_sym = SymbolicExpr.from_value(rhs)
            other = SymbolicExpr.from_value(other)
            if op is np.where:
                return SymbolicExpr("where", lhs_sym, rhs_sym, other)
            else:
                raise NotImplementedError(
                    f"Unsupported ternary operation: {op} between {lhs_sym}, {rhs_sym} and {other}"
                )

        def op_addptr_overrider(ptr, offset):
            """
            In addptr operator, ptr is a pointer address with dtype_tt, and offset is a scalar.
            """
            # Read dtype_tt from ptr.
            # Here, ptr is either a TensorHandle or a SymbolicExpr.
            dtype_tt = ptr.get_element_ty()
            assert dtype_tt, f"dtype_tt of ptr not found! ptr content: {ptr}"

            # Read bitwidth from dtype_tt, and then convert it to bytewidth.
            element_bitwidth = dtype_tt.primitive_bitwidth
            element_bytewidth = max(1, element_bitwidth // 8)

            # convert ptr to SymbolicExpr
            ptr_sym = SymbolicExpr.from_value(ptr)

            # convert offset to SymbolicExpr
            offset_sym = SymbolicExpr.from_value(offset)
            element_bytewidth_sym = SymbolicExpr.from_value(element_bytewidth)

            # calculate the new address, and store the dtype_tt information in its SymbolicExpr.
            ret = ptr_sym + offset_sym * element_bytewidth_sym
            ret.set_element_ty(dtype_tt)
            return ret

        def op_dot_overrider(a, b, d, input_precision, max_num_imprecise_acc):
            a_sym = SymbolicExpr.from_value(a)
            b_sym = SymbolicExpr.from_value(b)
            d_sym = SymbolicExpr.from_value(d)
            return SymbolicExpr("dot", a_sym, b_sym, d_sym)

        def op_make_range_overrider(start, end):
            return SymbolicExpr(
                "arange",
                SymbolicExpr.from_value(start),
                SymbolicExpr.from_value(end - 1),
            )

        def op_expand_dims_overrider(arg, axis):
            return SymbolicExpr("expand_dims", SymbolicExpr.from_value(arg), axis)

        def op_broadcast_overrider(arg, shape):
            return SymbolicExpr("broadcast", SymbolicExpr.from_value(arg), shape)

        def op_reduce_sum_overrider(input, axis=None, keep_dims=False, **kwargs):
            ret = SymbolicExpr(
                "sum", SymbolicExpr.from_value(input), axis, keep_dims, kwargs
            )
            return ret

        def op_splat_overrider(arg, shape):
            return SymbolicExpr("splat", SymbolicExpr.from_value(arg), shape)

        def op_make_block_ptr_overrider(
            base, shape, strides, offsets, tensor_shape, order
        ):
            base = SymbolicExpr.from_value(base)
            assert (
                len(shape)
                == len(strides)
                == len(offsets)
                == len(tensor_shape)
                == len(order)
            ), f"Length of shape ({len(shape)}), strides ({len(strides)}), offsets ({len(offsets)}), tensor_shape ({len(tensor_shape)}) and order ({len(order)}) must be the same!"
            shape = [SymbolicExpr.from_value(shape_i) for shape_i in shape]
            strides = [SymbolicExpr.from_value(strides_i) for strides_i in strides]
            offsets = [SymbolicExpr.from_value(offset_i) for offset_i in offsets]
            tensor_shape = [
                SymbolicExpr.from_value(tensor_shape_i)
                for tensor_shape_i in tensor_shape
            ]
            order = [SymbolicExpr.from_value(order_i) for order_i in order]

            ret = SymbolicExpr(
                "make_block_ptr", base, shape, strides, offsets, tensor_shape, order
            )

            ret.set_element_ty(base.get_element_ty())

            return ret

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
            return SymbolicExpr.from_value(lhs) // SymbolicExpr.from_value(rhs)

        def op_rsqrt_overrider(arg):
            return SymbolicExpr("rsqrt", SymbolicExpr.from_value(arg))

        def op_cast_impl_overrider(src, dst_type):
            return SymbolicExpr("cast_impl", src, dst_type)

        OP_TYPE_TO_OVERRIDER = {
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
            Splat: op_splat_overrider,
            MakeBlockPointer: op_make_block_ptr_overrider,
            TensorPointerLoad: op_tensor_pointer_load_overrider,
            TensorPointerStore: op_tensor_pointer_store_overrider,
            Idiv: op_idiv_overrider,
            Rsqrt: op_rsqrt_overrider,
            CastImpl: op_cast_impl_overrider,
        }

        if op_type in OP_TYPE_TO_OVERRIDER:
            return None, None, OP_TYPE_TO_OVERRIDER[op_type]
        else:
            return None, None, None

    def finalize(self) -> list:
        return []


class NullSanitizer(Client):
    """
    A do-nothing object returned when the sanitizer backend is 'off'.
    Any attribute access raises an explicit error so misuse is obvious.
    """

    def __init__(self, abort_on_error):
        pass

    def _disabled(self, method: str):
        raise RuntimeError(
            f"[NullSanitizer] '{method}' was called, "
            "but sanitizer backend is off; no functionality is available."
        )

    def arg_callback(self, *a, **k):
        self._disabled("arg_callback")

    def finalize(self, *a, **k):
        self._disabled("finalize")

    def grid_callback(self, *a, **k):
        self._disabled("grid_callback")

    def grid_idx_callback(self, *a, **k):
        self._disabled("grid_idx_callback")

    def register_op_callback(self, *a, **k):
        self._disabled("register_op_callback")

    def __getattr__(self, name):
        self._disabled(name)


class Sanitizer(ABC):
    """
    Factory class that returns the concrete sanitizer implementation
    based on the value of ``cfg.sanitizer_backend``.
    """

    def __new__(cls, abort_on_error: bool = False):
        backend = cfg.sanitizer_backend

        if backend == "brute_force":
            return SanitizerBruteForce(abort_on_error)

        if backend == "symexec":
            return SanitizerSymbolicExecution(abort_on_error)

        if backend == "off":
            return NullSanitizer(abort_on_error)

        raise ValueError(f"Invalid TRITON_SANITIZER_BACKEND: {backend!r} ")


Sanitizer.register(SanitizerBruteForce)
Sanitizer.register(SanitizerSymbolicExecution)
Sanitizer.register(NullSanitizer)
