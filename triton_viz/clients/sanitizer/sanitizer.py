import traceback
from abc import ABC
from typing import Tuple, Callable, Optional, Type
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

    # record the index of two key functions in core/patch.py
    jit_index = None
    grid_index = None

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
    SUPPORTED_OPS = (
        BASIC_OPS
        + INDIRECT_OPS
        + UNARY_OPS
        + BINARY_OPS
        + TERNARY_OPS
        + REDUCE_OPS
        + POINTER_OPS
        + BROADCAST_OPS
    )

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
        self.children = {}  # Used for storing child expressions
        # Functions and arguments for concretization
        self._concrete_fn = None
        self._concrete_args = ()
        self._concrete_kwargs = {}
        # leaf nodes
        if self.op == "const":
            self.value = args[0]
            if len(args) >= 2:
                self.dtype_tt = args[1]
        elif self.op == "pid":
            assert len(args) == 2, "pid op expects two arguments!"
            self.grid = args[0]
            self.axis = args[1]
        elif self.op == "arange":
            assert len(args) == 2, "arange op expects two arguments!"
            self.start = args[0]
            self.end = args[1]
            # TODO: setup self.shape here
        elif self.op == "load":
            assert len(args) in (1, 2, 3), "load op expects 1, 2 or 3 arguments!"
            self.ptr = args[0]
            self.mask = args[1] if len(args) >= 2 else None
            self.other = args[2] if len(args) >= 3 else None
            self.set_element_ty(self.ptr.get_element_ty())
        elif self.op == "store":
            assert len(args) in (2, 3, 4), "store op expects 2, 3 or 4 arguments!"
            self.ptr = args[0]
            self.value = args[1]
            self.mask = args[2] if len(args) >= 3 else None
            self.other = args[3] if len(args) >= 4 else None
            self.set_element_ty(self.ptr.get_element_ty())
        elif self.op in self.UNARY_OPS:
            assert len(args) == 1, f"{self.op} op expects one argument!"
            self.arg = args[0]
        elif self.op in self.BINARY_OPS:
            assert len(args) == 2, f"{self.op} op expects two arguments!"
            self.lhs = args[0]
            self.rhs = args[1]
            if not self.lhs.shape:  # lhs is a scalar
                ret_shape = self.rhs.shape
            elif not self.rhs.shape:  # rhs is a scalar
                ret_shape = self.lhs.shape
            else:  # both are blocks
                assert (
                    self.lhs.shape == self.rhs.shape
                ), f"lhs shape {self.lhs.shape} should be equal to rhs shape {self.rhs.shape}"
                ret_shape = self.lhs.shape
            self.shape = ret_shape
        elif self.op in self.TERNARY_OPS:
            assert len(args) == 3, f"{self.op} op expects three arguments!"
            if self.op == "where":
                self.cond = args[0]
                self.lhs = args[1]
                self.rhs = args[2]
            else:
                raise NotImplementedError(f"Unsupported ternary op: {self.op}")
        elif self.op in self.REDUCE_OPS:
            if self.op == "sum":
                self.input = args[0]
                self.axis = args[1]
                self.keepdims = args[2]
            elif self.op == "dot":
                self.a = args[0]
                self.b = args[1]
                if len(args) >= 3:
                    self.d = args[2]
            else:
                raise NotImplementedError(f"Unsupported reduce op: {self.op}")
        elif self.op == "make_block_ptr":
            assert len(args) == 6, "make_block_ptr op expects six arguments!"
            self.base = args[0]
            self.shape = args[1]
            self.strides = args[2]
            self.offsets = args[3]
            self.block_shape = args[4]
            self.order = args[5]
        elif self.op in self.BROADCAST_OPS:
            self.arg = args[0]
            self.dtype_tt = self.arg.dtype_tt
            if self.op == "splat":
                assert len(args) == 2, "splat op expects two arguments!"
                self.shape = args[1]
            elif self.op == "expand_dims":
                assert len(args) == 2, "expand_dims op expects two arguments!"
                self.axis = args[1]
                self.arg.shape.insert(self.axis, 1)
            elif self.op == "broadcast":
                assert len(args) == 2, "broadcast op expects two arguments!"
                self.shape = args[1]
        else:
            raise NotImplementedError(f"Unsupported op: {self.op}")

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
        for child_name, child_expr in self._children():
            # Build the child subtree
            child_node = child_expr.to_anytree()
            # Prefix the child node's name with the field name
            child_node.name = f"{child_name}: {child_node.name}"
            child_node.parent = root

        return root

    def _node_label(self):
        """Generate a short label for this node."""
        if self.op == "const":
            label = f"const={self.value}"
        elif self.op == "pid":
            label = f"pid_{self.axis}={self.grid[self.axis]}"
        else:
            label = self.op

        # Add the dtype to the label if available
        label = f"{label} [dtype={self.get_element_ty()}]"

        return label

    def _children(self):
        """
        Return a list of (field_name, SymbolicExpr) pairs for each child,
        depending on the operation type.
        """
        children = []
        if self.op == "const" or self.op == "pid":
            pass
        elif self.op == "arange":
            children.append(("start", self.start))
            children.append(("end", self.end))
        elif self.op == "load":
            children.append(("ptr", self.ptr))
            if self.mask is not None:
                children.append(("mask", self.mask))
            if self.other is not None:
                children.append(("other", self.other))
        elif self.op == "store":
            children.append(("ptr", self.ptr))
            children.append(("value", self.value))
            if self.mask is not None:
                children.append(("mask", self.mask))
            if self.other is not None:
                children.append(("other", self.other))
        elif self.op in self.UNARY_OPS:
            children.append(("arg", self.arg))
        elif self.op in self.BINARY_OPS:
            children.append(("lhs", self.lhs))
            children.append(("rhs", self.rhs))
        elif self.op == "where":
            children.append(("cond", self.cond))
            children.append(("lhs", self.lhs))
            children.append(("rhs", self.rhs))
        elif self.op == "sum":
            children.append(("input", self.input))
            # Axis and keepdims are included in the label, not as separate nodes
        elif self.op in ("splat", "expand_dims", "broadcast"):
            children.append(("arg", self.arg))
        else:
            raise NotImplementedError(f"Unsupported operation: {self.op}")
        return children

    def to_tree_str(self) -> str:
        """
        Render the AST as an ASCII tree using anytree.RenderTree.
        """
        root = self.to_anytree()
        lines = []
        for prefix, _, node in RenderTree(root):
            lines.append(f"{prefix}{node.name}")
        return "\n".join(lines)

    def __str__(self):
        return self.to_tree_str()

    def __repr__(self):
        return self.__str__()

    @property
    def data(self):
        return SymbolicExprDataWrapper(self.__str__(), self)

    @classmethod
    def from_value(cls, var):
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
        # if already SymbolicExpr
        if isinstance(var, cls):
            return var

        # if construct from a TensorHandle
        if isinstance(var, TensorHandle):
            # if an immediate
            if var.dtype in triton_scala_dtypes:
                if len(var.data) == 1:
                    return cls("const", var.data.item())
                else:
                    return cls("const", var.data)
            # if a pointer
            elif isinstance(var.dtype, tl.pointer_type):
                if len(var.data) != 1:
                    raise ValueError(
                        "Unsupported tl.pointer_type with length more than one!"
                    )
                return cls("const", var.data.item(), var.get_element_ty())
            else:
                raise ValueError("Unsupported TensorHandle dtype", var.dtype)

        if isinstance(var, builtin_scala_types):
            return cls("const", var)

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
            name = f"pid_{node.axis}"
            if name not in self._vars:
                v = Int(name)
                self._vars[name] = v
                # Add constraint: 0 ≤ pid < grid[axis]
                self._constraints.append(v >= 0)
                self._constraints.append(v < node.grid[node.axis])
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
        if node.op == "abs":
            c = self._to_z3(node.arg)
            return If(c >= 0, c, -c)
        if node.op in self.UNARY_OPS:
            c = self._to_z3(node.arg)
            if node.op == "abs":
                return If(c >= 0, c, -c)
            raise NotImplementedError(f"Unary op {node.op} is not implemented")

        # Binary arithmetic, comparison, etc.
        if node.op in self.BINARY_OPS:
            l = self._to_z3(node.lhs)
            r = self._to_z3(node.rhs)
            if node.op == "add":
                return l + r
            if node.op == "sub":
                return l - r
            if node.op == "mul":
                return l * r
            if node.op in ("idiv"):
                return l / r
            if node.op == "mod":
                return l % r
            if node.op == "less":
                return l < r
            if node.op == "less_equal":
                return l <= r
            if node.op == "greater":
                return l > r
            if node.op == "greater_equal":
                return l >= r
            if node.op == "equal":
                return l == r
            if node.op == "not_equal":
                return l != r
            if node.op == "maximum":
                return If(l >= r, l, r)
            if node.op == "bitwise_and":
                return And(l, r)

        # where(cond, lhs, rhs)
        if node.op == "where":
            c = self._to_z3(node.cond)
            l = self._to_z3(node.lhs)
            r = self._to_z3(node.rhs)
            return If(c, l, r)

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
        for child_key, child_symbolic_expr in self.children.items():
            if child_symbolic_expr.has_op(op_name):
                return True
        return False

    @staticmethod
    def _concretize_item(obj):
        return obj.concretize() if isinstance(obj, SymbolicExpr) else obj

    def concretize(self):
        """
        Concretize the symbolic expression into a concrete value.
        This is used to evaluate the symbolic expression and return a concrete value.
        """
        if self.op == "splat":
            print("op:", self.op)
            print("arg:", self._concrete_args)
            print("kwargs:", self._concrete_kwargs)
        if self.op == "const":
            return self.value
        if self._concrete_fn is None:
            raise RuntimeError("Concrete function is not set for this SymbolicExpr.")
        new_args = [self._concretize_item(a) for a in self._concrete_args]
        new_kw = {k: self._concretize_item(v) for k, v in self._concrete_kwargs.items()}
        return self._concrete_fn(*new_args, **new_kw)

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
                ptr = SymbolicExpr("load", SymbolicExpr.from_value(ptr))
            elif isinstance(ptr, SymbolicExpr):
                ptr = SymbolicExpr("load", ptr)
            else:
                raise ValueError(f"Unsupported ptr type: {type(ptr)}")

            if mask is None:
                ret = SymbolicExpr("load", ptr)
            elif other is None:
                ret = SymbolicExpr("load", ptr, mask)
            else:
                ret = SymbolicExpr("load", ptr, mask, other)

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
                ptr = SymbolicExpr("load", SymbolicExpr.from_value(ptr))
            elif isinstance(ptr, SymbolicExpr):
                ptr = SymbolicExpr("load", ptr)
            else:
                raise ValueError(f"Unsupported ptr type: {type(ptr)}")

            value = SymbolicExpr.from_value(value)
            if mask is None:
                ret = SymbolicExpr("store", ptr, value)
            else:
                ret = SymbolicExpr("store", ptr, value, mask)

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
            arg = SymbolicExpr.from_value(arg)
            try:
                name = _unary_map[op]
            except KeyError:
                raise NotImplementedError(f"Unsupported unary operation: {op} on {arg}")
            return SymbolicExpr(name, arg)

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
            lhs = SymbolicExpr.from_value(lhs)
            rhs = SymbolicExpr.from_value(rhs)
            try:
                func = _binary_map[op]
            except KeyError:
                raise NotImplementedError(
                    f"Unsupported binary operation: {op} between {lhs} and {rhs}"
                )
            return func(lhs, rhs)

        def op_ternary_op_overrider(lhs, rhs, other, op):
            lhs = SymbolicExpr.from_value(lhs)
            rhs = SymbolicExpr.from_value(rhs)
            other = SymbolicExpr.from_value(other)
            if op is np.where:
                return SymbolicExpr("where", lhs, rhs, other)
            else:
                raise NotImplementedError(
                    f"Unsupported ternary operation: {op} between {lhs}, {rhs} and {other}"
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
            ptr = SymbolicExpr.from_value(ptr)

            # convert offset to SymbolicExpr
            offset = SymbolicExpr.from_value(offset)
            element_bytewidth = SymbolicExpr.from_value(element_bytewidth)

            # calculate the new address, and store the dtype_tt information in its SymbolicExpr.
            ret = ptr + offset * element_bytewidth
            ret.set_element_ty(dtype_tt)
            return ret

        def op_dot_overrider(a, b, d, input_precision, max_num_imprecise_acc):
            a = SymbolicExpr.from_value(a)
            b = SymbolicExpr.from_value(b)
            d = SymbolicExpr.from_value(d)
            return SymbolicExpr("dot", a, b, d)

        def op_make_range_overrider(start, end):
            start = SymbolicExpr.from_value(start)
            end = SymbolicExpr.from_value(end - 1)
            return SymbolicExpr("arange", start, end)

        def op_expand_dims_overrider(arg, axis):
            arg = SymbolicExpr.from_value(arg)
            return SymbolicExpr("expand_dims", arg, axis)

        def op_broadcast_overrider(arg, shape):
            arg = SymbolicExpr.from_value(arg)
            return SymbolicExpr("broadcast", arg, shape)

        def op_reduce_sum_overrider(input, axis=None, keep_dims=False, **kwargs):
            input = SymbolicExpr.from_value(input)
            ret = SymbolicExpr("sum", input, axis, keep_dims, kwargs)
            return ret

        def op_splat_overrider(arg, shape):
            arg = SymbolicExpr.from_value(arg)
            return SymbolicExpr("splat", arg, shape)

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
            print(ret)

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
            lhs = SymbolicExpr.from_value(lhs)
            rhs = SymbolicExpr.from_value(rhs)
            return lhs // rhs

        def op_rsqrt_overrider(arg):
            arg = SymbolicExpr.from_value(arg)
            return SymbolicExpr("rsqrt", arg)

        def op_cast_impl_overrider(src, dst_type):
            src = SymbolicExpr.from_value(src)
            return src

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


class NullSanitizer:
    """
    A do-nothing object returned when the sanitizer backend is 'off'.
    Any attribute access raises an explicit error so misuse is obvious.
    """

    def __getattr__(self, name):
        raise RuntimeError(
            "Sanitizer backend is off; no sanitizer functionality is available."
        )


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
            return NullSanitizer()

        raise ValueError(f"Invalid TRITON_SANITIZER_BACKEND: {backend!r} ")


Sanitizer.register(SanitizerBruteForce)
Sanitizer.register(SanitizerSymbolicExecution)
Sanitizer.register(NullSanitizer)
