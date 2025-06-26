import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle
from triton_viz.clients.sanitizer.sanitizer import (
    SymbolicExpr,
    SanitizerSymbolicExecution,
)
from triton_viz.core.data import AddPtr


def test_addptr_expr_eval():
    base = SymbolicExpr.from_value(1000)  # Synthetic base address
    base.dtype_tt = tl.pointer_type(
        tl.int32
    )  # Simulate a pointer type, int32 = 4 bytes
    offset = SymbolicExpr.from_value(3)
    expr = SymbolicExpr("addptr", base, offset)
    assert expr.eval()[0] == 1000 + 3 * 4


def test_addptr_overrider():
    # Run through sanitizer's overrider
    ptr_dtype = tl.pointer_type(tl.int32)
    ptr_th = TensorHandle(np.array([1000]), ptr_dtype)
    offset_th = TensorHandle(np.array([3]), tl.int32)
    sanitizer = SanitizerSymbolicExecution(abort_on_error=False)
    _, _, addptr_fn = sanitizer.register_op_callback(AddPtr)
    assert addptr_fn is not None
    expr = addptr_fn(ptr_th, offset_th)  # offset = 3
    assert expr.op == "addptr"
    assert expr.eval()[0] == 1000 + 3 * 4  # element_bytewidth = 4
